#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
'''
fileserver_space_check() is a periodic task that gets executed by celery daemon
on a repeating schedule.  If there are any actions to take, it will execute a
celery task called manage_data() which will archive or delete one fileset category
for one Result object.  This function will in turn call a function to do the
actual work in the filesystem.
'''
from __future__ import absolute_import
import sys
import os
import re
import pytz
import traceback
import socket
import time
from datetime import timedelta, datetime
from celery.task import task
from celery.task import periodic_task
from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger
from django.db.models import Q
from django.utils import timezone
from django.core import mail
from django.core.exceptions import ObjectDoesNotExist

from iondb.rundb.models import GlobalConfig, \
    DMFileSet, DMFileStat, User, Experiment, Backup, Results, EventLog, Message, PluginResultJob
from iondb.utils.TaskLock import TaskLock
import iondb.settings as settings
from iondb.rundb.data.tasks import delete_action, archive_action, export_action
from iondb.rundb.data import dmactions_types
from iondb.rundb.data import exceptions as DMExceptions
from iondb.rundb.data import dmactions
from iondb.rundb.data.dm_utils import slugify
from iondb.rundb.data import dm_utils
from iondb.anaserve import client
logger = get_task_logger('data_management')
logid = {'logid': "%s" % ('DM')}

# at celeryd start-up, dump an entry into log file.
from celery.signals import celeryd_after_setup


@celeryd_after_setup.connect
def configure_workers(sender=None, conf=None, **kwargs):
    if 'periodic' in sender:
        logger.info("Restarted: %s" % sender, extra={'logid': "%s" % ('DM')})


@task(queue="dmmanage", expires=30, ignore_result=True)
def manage_manual_action():
    logid = {'logid': "%s" % ('manual_action')}
    logger.debug("manage_manual_action", extra=logid)

    def getfirstmanualaction():
        manual_selects = DMFileStat.objects.filter(action_state__in=['SA', 'SE', 'SD']).order_by('created')
        for item in manual_selects:
            try:
                dmactions.action_validation(item, item.dmfileset.auto_action)
                return item
            except(DMExceptions.FilesInUse, DMExceptions.FilesMarkedKeep):
                logger.debug("%s Failed action_validation.  Try next fileset" %
                             item.result.resultsName, extra=logid)
            except DMExceptions.BaseInputLinked:
                # want to allow Basecalling Input delete if all results are expired
                related_objs = DMFileStat.objects.filter(
                    result__experiment=item.result.experiment, dmfileset__type=dmactions_types.BASE)
                if related_objs.count() == related_objs.filter(created__lt=threshdate).count():
                    item.allow_delete = True
                    return item
                else:
                    logger.debug("%s Failed action_validation.  Try next fileset" %
                                 item.result.resultsName, extra=logid)
            except:
                logger.error(traceback.format_exc(), extra=logid)
        raise DMExceptions.NoDMFileStat("NONE FOUND")

    try:
        # Create lock file to prevent more than one celery task for manual action (export or archive)
        lock_id = 'manual_action_lock_id'
        logid = {'logid': "%s" % (lock_id)}
        applock = TaskLock(lock_id)

        if not applock.lock():
            logger.info("Currently processing: %s(%s)" % (lock_id, applock.get()), extra=logid)
            return

        logger.debug("lock file created: %s(%s)" % (lock_id, applock.get()), extra=logid)

    except Exception as e:
        logger.error(traceback.format_exc(), extra=logid)

    try:
        #
        # Check for manually selected Archive and Export actions - action_state == 'SA' or 'SE' or 'SD'
        # Manual Delete actions do not get processed here.  Only suspended Delete actions get processed here.
        # These jobs should execute even when auto action is disabled.
        # Note: manual actions will not be executed in the order they are selected, but by age.
        #
        actiondmfilestat = getfirstmanualaction()

        user = actiondmfilestat.user_comment.get('user', 'dm_agent')
        user_comment = actiondmfilestat.user_comment.get('user_comment', 'Manual Action')
        logger.info("Picked: %s" % (actiondmfilestat.result.resultsName), extra=logid)

        dmfileset = actiondmfilestat.dmfileset
        applock.update(actiondmfilestat.result.resultsName)
        if actiondmfilestat.action_state == 'SA':
            logger.info("Manual Archive Action: %s from %s" %
                        (dmfileset.type, actiondmfilestat.result.resultsName), extra=logid)
            archive_action(user, user_comment, actiondmfilestat, lock_id, msg_banner=True)
        elif actiondmfilestat.action_state == 'SE':
            logger.info("Manual Export Action: %s from %s" %
                        (dmfileset.type, actiondmfilestat.result.resultsName), extra=logid)
            export_action(user, user_comment, actiondmfilestat, lock_id, msg_banner=True)
        elif actiondmfilestat.action_state == 'SD':
            logger.info("Delete Action: %s from %s" %
                        (dmfileset.type, actiondmfilestat.result.resultsName), extra=logid)
            delete_action(user, "Continuing delete action after being suspended",
                          actiondmfilestat, lock_id, msg_banner=True)
        else:
            logger.warn("Dev Error: we don't handle this '%s' here" %
                        actiondmfilestat.action_state, extra=logid)

    except DMExceptions.NoDMFileStat:
        applock.unlock()
        logger.debug("Worker PID %d lock_id destroyed on exit %s" % (os.getpid(), lock_id), extra=logid)
    # NOTE: all these exceptions are also handled in manage_data() below.  beaucoup de duplication de code
    except (DMExceptions.FilePermission,
            DMExceptions.InsufficientDiskSpace,
            DMExceptions.MediaNotSet,
            DMExceptions.MediaNotAvailable,
            DMExceptions.FilesInUse) as e:
        applock.unlock()
        message = Message.objects.filter(tags__contains=e.tag)
        if not message:
            Message.error(e.message, tags=e.tag)
            # TODO: TS-6525: This logentry will repeat for an Archive action every 30 seconds until the cause of the exception is fixed.
            # at least, while the message banner is raised, suppress additional Log Entries.
            dmactions.add_eventlog(actiondmfilestat, "%s - %s" %
                                   (dmfileset.type, e.message), username='dm_agent')
        # State needs to be Error since we do not know previous state (ie, do NOT
        # set to Local, it might get deleted automatically)
        actiondmfilestat.setactionstate('E')
    except DMExceptions.SrcDirDoesNotExist as e:
        applock.unlock()
        msg = "Src Dir not found: %s. Setting action_state to Deleted" % e.message
        EventLog.objects.add_entry(actiondmfilestat.result, msg, username='dm_agent')
        actiondmfilestat.setactionstate('DD')
        logger.info(msg, extra=logid)
    except Exception as e:
        applock.unlock()
        msg = "action error on %s " % actiondmfilestat.result.resultsName
        msg += " Error: %s" % str(e)
        logger.error("%s - %s" % (dmfileset.type, msg), extra=logid)
        logger.error(traceback.format_exc(), extra=logid)
        dmactions.add_eventlog(actiondmfilestat, "%s - %s" % (dmfileset.type, msg), username='dm_agent')
        # State needs to be Error since we do not know previous state (ie, do NOT
        # set to Local, it might get deleted automatically)
        actiondmfilestat.setactionstate('E')

    return


@task(queue="dmmanage", expires=30, ignore_result=True)
def manage_data(deviceid, dmfileset, pathlist, auto_acknowledge_enabled, auto_action_enabled):
    logid = {'logid': "%s" % ('manage_data')}
    logger.debug("manage_data: %s %s" % (dmfileset['auto_action'], dmfileset['type']), extra=logid)

    def getfirstnotpreserved(dmfilestats, action, threshdate):
        '''QuerySet of DMFileStat objects.  Returns first instance of an object
        with preserve_data set to False and passes the action_validation test'''
        logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
        logger.debug("Looking at %d dmfilestat objects" % dmfilestats.count(), extra=logid)
        for archiveme in dmfilestats:
            if not archiveme.getpreserved():
                try:
                    dmactions.action_validation(archiveme, action)
                    return archiveme
                except(DMExceptions.FilesInUse, DMExceptions.FilesMarkedKeep):
                    logger.debug("%s Failed action_validation.  Try next fileset" %
                                 archiveme.result.resultsName, extra=logid)
                except DMExceptions.BaseInputLinked:
                    # want to allow Basecalling Input delete if all results are expired
                    related_objs = DMFileStat.objects.filter(
                        result__experiment=archiveme.result.experiment, dmfileset__type=dmactions_types.BASE)
                    if related_objs.count() == related_objs.filter(created__lt=threshdate).count():
                        archiveme.allow_delete = True
                        return archiveme
                    else:
                        logger.debug("%s Failed action_validation.  Try next fileset" %
                                     archiveme.result.resultsName, extra=logid)
                except:
                    logger.error(traceback.format_exc(), extra=logid)
            else:
                logger.debug("Skipped a preserved fileset: %s" % archiveme.result.resultsName, extra=logid)

        logger.info("%d filestat objects are preserved." % dmfilestats.count(), extra=logid)
        raise DMExceptions.NoDMFileStat("NONE FOUND")

    try:
        # logger.debug("manage_data lock for %s (%d)" % (dmfileset['type'], os.getpid()), extra = logid)
        # Create lock file to prevent more than one celery task for each process_type and partition
        lock_id = "%s_%s" % (hex(deviceid), slugify(dmfileset['type']))
        logid = {'logid': "%s" % (lock_id)}
        applock = TaskLock(lock_id)

        if not applock.lock():
            logger.info("Currently processing: %s(%s)" % (lock_id, applock.get()), extra=logid)
            return

        logger.debug("lock file created: %s(%s)" % (lock_id, applock.get()), extra=logid)

    except Exception as e:
        logger.error(traceback.format_exc(), extra=logid)

    try:
        #---------------------------------------------------------------------------
        # Database object filtering
        #---------------------------------------------------------------------------
        actiondmfilestat = None
        user_comment = "Auto Action"
        # Order by incrementing pk.  This puts them in chronological order.
        # Select DMFileStat objects of category DMFileSet.type (1/4th of all objects)
        dmfilestats = DMFileStat.objects.filter(dmfileset__type=dmfileset['type']).order_by('created')
        tot_obj = dmfilestats.count()

        # Select objects not yet processed
        dmfilestats = dmfilestats.filter(action_state__in=['L', 'S', 'N', 'A'])
        tot_act = dmfilestats.count()

        # Select objects that are old enough
        threshdate = datetime.now(pytz.UTC) - timedelta(days=dmfileset['auto_trigger_age'])
        dmfilestats = dmfilestats.filter(created__lt=threshdate)
        tot_exp = dmfilestats.count()

        # Select objects stored on the deviceid
        query = Q()
        for path in pathlist:
            if dmfileset['type'] == dmactions_types.SIG:
                query |= Q(result__experiment__expDir__startswith=path)
            # These categories have files in both directory paths.
            elif dmfileset['type'] in [dmactions_types.INTR, dmactions_types.BASE]:
                query |= Q(result__experiment__expDir__startswith=path)
                query |= Q(result__reportstorage__dirPath__startswith=path)
            else:
                query |= Q(result__reportstorage__dirPath__startswith=path)

        dmfilestats = dmfilestats.filter(query)
        tot_onpath = dmfilestats.count()

        # Exclude objects marked 'Keep' upfront to optimize db access
        if dmfileset['type'] == dmactions_types.SIG:
            dmfilestats = dmfilestats.exclude(result__experiment__storage_options="KI")
        else:
            dmfilestats = dmfilestats.exclude(preserve_data=True)
        tot_notpreserved = dmfilestats.count()

        # Compress to single log entry
        logger.info("Total: %d Active: %d Expired: %d On Path: %d Not Preserved: %d" %
                    (tot_obj, tot_act, tot_exp, tot_onpath, tot_notpreserved), extra=logid)

        #---------------------------------------------------------------------------
        # Archive
        #---------------------------------------------------------------------------
        if dmfileset['auto_action'] == 'ARC':
            '''
            Rules:
            1) archive a fileset as soon as age threshold reached, regardless of disk threshold
            2) select oldest fileset
            3) select unprocessed fileset
            4) Archiving does not require 'S' -> 'N' -> 'A' progression before action
            5) Do not include filesets marked 'E' in auto-action - can get stuck on that fileset forever
            '''
            logger.info("Exceed %d days. Eligible to archive: %d" %
                        (dmfileset['auto_trigger_age'], dmfilestats.count()), extra=logid)
            actiondmfilestat = None
            # Bail out if disabled
            if auto_action_enabled != True:
                logger.info("Data management auto-action is disabled.", extra=logid)
                applock.unlock()
                logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(), lock_id), extra=logid)
                return

            # Select first object stored on the deviceid
            try:
                actiondmfilestat = getfirstnotpreserved(dmfilestats, dmactions.ARCHIVE, threshdate)
                logger.info("Picked: %s" % (actiondmfilestat.result.resultsName), extra=logid)
            except DMExceptions.NoDMFileStat:
                logger.debug("No filesets to archive on this device", extra=logid)
                applock.unlock()
                logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(), lock_id), extra=logid)
            except:
                logger.error(traceback.format_exc(), extra=logid)
                applock.unlock()
                logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(), lock_id), extra=logid)
            else:
                applock.update(actiondmfilestat.result.resultsName)
                archive_action('dm_agent', user_comment, actiondmfilestat, lock_id)

        #---------------------------------------------------------------------------
        # Delete
        #---------------------------------------------------------------------------
        elif dmfileset['auto_action'] == 'DEL':
            '''
            Rules: If auto-acknowledge is True:
            promote to Acknowledged and delete when age and disk threshold reached.
            If auto-acknowledge is False:
                If fileset type is SIG:
                    'S' -> 'N' -> 'A' progression
                Else:
                    promote to 'A'
                delete an 'A' fileset
            '''
            logger.info("Exceed %d days. Eligible to delete: %d" %
                        (dmfileset['auto_trigger_age'], dmfilestats.count()), extra=logid)
            if auto_acknowledge_enabled:
                if dmfileset['type'] == dmactions_types.SIG:
                    logger.debug("Sig Proc Input Files auto acknowledge enabled", extra=logid)
                '''
                Do not require 'S' -> 'N' -> 'A' progression before action; mark 'A' and process
                '''
                a_list = dmfilestats.filter(action_state='A')
                if a_list.count() > 0:
                    # there are filesets acknowledged and ready to be deleted.  Covers situation where user has
                    # already acknowledged but recently enabled auto-acknowledge as well.
                    # deleteme = a_list[0]
                    try:
                        actiondmfilestat = getfirstnotpreserved(a_list, dmactions.DELETE, threshdate)
                        logger.info("Picked: %s" % (actiondmfilestat.result.resultsName), extra=logid)
                    except DMExceptions.NoDMFileStat:
                        logger.info("No filesets to delete on this device", extra=logid)
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" %
                                     (os.getpid(), lock_id), extra=logid)
                    except:
                        logger.error(traceback.format_exc(), extra=logid)
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" %
                                     (os.getpid(), lock_id), extra=logid)

                if actiondmfilestat == None:
                    # Select oldest fileset regardless if its 'L','S','N','A'.  This covers situation where user
                    # recently enabled auto-acknowledge
                    try:
                        actiondmfilestat = getfirstnotpreserved(dmfilestats, dmactions.DELETE, threshdate)
                        logger.info("Picked: %s" % (actiondmfilestat.result.resultsName), extra=logid)
                    except DMExceptions.NoDMFileStat:
                        logger.info("No filesets to delete on this device", extra=logid)
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" %
                                     (os.getpid(), lock_id), extra=logid)
                    except:
                        logger.error(traceback.format_exc(), extra=logid)
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" %
                                     (os.getpid(), lock_id), extra=logid)
            else:
                if dmfileset['type'] == dmactions_types.SIG:
                    logger.debug("Sig Proc Input Files auto acknowledge disabled", extra=logid)
                    '''
                    Need to select # of filesets and mark 'S'
                    Need to find 'S' filesets and notify (set to 'N')
                    Set queue length to 5
                    sum(S,N,A) == queue length
                    '''
                    # Get email recipient
                    try:
                        recipient = User.objects.get(username='dm_contact').email
                        recipient = recipient.replace(',', ' ').replace(';', ' ').split()
                    except:
                        recipient = None

                    # Need to select Experiments to process with appropriate dmfilestats action states.
                    exps = Experiment.objects.exclude(
                        storage_options='KI').exclude(expDir='').order_by('date')
                    l_list = exps.filter(
                        results_set__dmfilestat__in=dmfilestats.filter(action_state='L')).distinct()
                    s_list = exps.filter(
                        results_set__dmfilestat__in=dmfilestats.filter(action_state='S')).distinct()
                    n_list = exps.filter(
                        results_set__dmfilestat__in=dmfilestats.filter(action_state='N')).distinct()
                    a_list = exps.filter(
                        results_set__dmfilestat__in=dmfilestats.filter(action_state='A')).distinct()

                    logger.info("Experiments (Keep=False) L:%d S:%d N:%d A:%d" %
                                (l_list.count(), s_list.count(), n_list.count(), a_list.count()), extra=logid)
                    queue_length = 5
                    to_select = queue_length - (a_list.count() + n_list.count() + s_list.count())
                    if to_select > 0:
                        # Mark to_select number of oldest Experiments, from 'L', to 'S'
                        promoted = dmfilestats.filter(
                            result__experiment__id__in=list(l_list[:to_select].values_list('id', flat=True)))
                        if auto_action_enabled:
                            promoted.update(action_state='S')
                            for dmfilestat in promoted:
                                EventLog.objects.add_entry(
                                    dmfilestat.result, "Signal Processing Input Selected for Deletion", username='dm_agent')

                    # Get updated list of Selected items
                    selected = dmfilestats.filter(action_state='S')

                    # Send Selected items to be notified
                    if selected.count() > 0 and auto_action_enabled:
                        logger.debug("notify recipient %s" % (recipient), extra=logid)
                        if notify([dmfilestat.result.resultsName for dmfilestat in selected], recipient):
                            selected.update(action_state='N')
                            for dmfilestat in selected:
                                EventLog.objects.add_entry(
                                    dmfilestat.result, "Notification for Deletion Sent", username='dm_agent')

                    try:
                        actiondmfilestat = getfirstnotpreserved(
                            dmfilestats.filter(action_state='A'), dmactions.DELETE, threshdate)
                        logger.info("Picked: %s" % (actiondmfilestat.result.resultsName), extra=logid)
                    except DMExceptions.NoDMFileStat:
                        logger.info("No filesets to delete on this device", extra=logid)
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" %
                                     (os.getpid(), lock_id), extra=logid)
                    except:
                        logger.error(traceback.format_exc(), extra=logid)
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" %
                                     (os.getpid(), lock_id), extra=logid)

                else:
                    try:
                        actiondmfilestat = getfirstnotpreserved(dmfilestats, dmactions.DELETE, threshdate)
                        logger.info("Picked: %s" % (actiondmfilestat.result.resultsName), extra=logid)
                    except DMExceptions.NoDMFileStat:
                        logger.info("No filesets to delete on this device", extra=logid)
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" %
                                     (os.getpid(), lock_id), extra=logid)
                    except:
                        logger.error(traceback.format_exc(), extra=logid)
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" %
                                     (os.getpid(), lock_id), extra=logid)

            # Bail out if disabled
            if auto_action_enabled != True:
                logger.info("Data management auto-action is disabled.", extra=logid)
                applock.unlock()
                logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(), lock_id), extra=logid)
                return

            if actiondmfilestat is not None:
                applock.update(actiondmfilestat.result.resultsName)
                delete_action('dm_agent', user_comment, actiondmfilestat, lock_id,
                              confirmed=getattr(actiondmfilestat, 'allow_delete', False))

        else:
            logger.error("Unknown or unhandled action: %s" % dmfileset['auto_action'], extra=logid)
            applock.unlock()

    except (DMExceptions.FilePermission,
            DMExceptions.InsufficientDiskSpace,
            DMExceptions.MediaNotSet,
            DMExceptions.MediaNotAvailable,
            DMExceptions.FilesInUse) as e:
        applock.unlock()
        message = Message.objects.filter(tags__contains=e.tag)
        if not message:
            Message.error(e.message, tags=e.tag)
            # TODO: TS-6525: This logentry will repeat for an Archive action every 30 seconds until the cause of the exception is fixed.
            # at least, while the message banner is raised, suppress additional Log Entries.
            EventLog.objects.add_entry(actiondmfilestat.result, "%s - %s" %
                                       (dmfileset['type'], e.message), username='dm_agent')
    except DMExceptions.SrcDirDoesNotExist as e:
        applock.unlock()
        msg = "Src Dir not found: %s. Setting action_state to Deleted" % e.message
        EventLog.objects.add_entry(actiondmfilestat.result, msg, username='dm_agent')
        actiondmfilestat.setactionstate('DD')
        logger.info(msg, extra=logid)
    except Exception as inst:
        applock.unlock()
        msg = ''
        if actiondmfilestat:
            msg = "Auto-action error on %s " % actiondmfilestat.result.resultsName
        msg += " Error: %s" % str(inst)
        logger.error("%s - %s" % (dmfileset['type'], msg), extra=logid)
        logger.error(traceback.format_exc(), extra=logid)

        if actiondmfilestat:
            EventLog.objects.add_entry(actiondmfilestat.result, "%s - %s" %
                                       (dmfileset['type'], msg), username='dm_agent')

    return


@periodic_task(run_every=timedelta(seconds=60), expires=15, soft_time_limit=55, queue="periodic", ignore_result=True)
#@task
def fileserver_space_check():
    '''For each file server, compare disk usage to backup threshold value.
    If disk usage exceeds threshold, launch celery task to delete/archive
    raw data directories.
    python -c "from iondb.bin import djangoinit; from iondb.rundb.data import data_management as dm; dm.fileserver_space_check()"
    '''
    logid = {'logid': "%s" % ('DM')}
    try:
        starttime = time.time()
        # Get GlobalConfig object in order to access auto-acknowledge bit
        global_config = GlobalConfig.get()
        auto_acknowledge = global_config.auto_archive_ack
        auto_action_enabled = global_config.auto_archive_enable

        category_list = dm_utils.dm_category_list()

        #-------------------------------------------------------------
        # Action loop - for each category, launch action per deviceid
        #-------------------------------------------------------------
        if not auto_action_enabled:
            logger.info("Data management auto-action is disabled.", extra=logid)

        # update any DMFileStats that are in use by active analysis jobs
        try:
            update_files_in_use()
        except:
            logger.error('Unable to update active DMFileStats', extra=logid)
            logger.error(traceback.format_exc(), extra=logid)

        # Checks for manual export and archive requests.
        manage_manual_action.delay()

        for category, dict in category_list.iteritems():
            for deviceid in dict['devlist']:
                pathlist = [item['path'] for item in dict['partitions'] if item['devid'] == deviceid]
                manage_data.delay(
                    deviceid, dict['dmfileset'], pathlist, auto_acknowledge, auto_action_enabled)
        endtime = time.time()
        logger.info("Disk Check: %f s" % (endtime - starttime), extra=logid)
    except SoftTimeLimitExceeded:
        logger.error("fileserver_space_check exceeded execution time limit", extra=logid)
    return


def backfill_create_dmfilestat():
    '''
    Examines every result object for related DMFileStat objects and creates if missing
    Migrates pre-TS3.6 servers to the new data management schema.  Called from postinst.in
    '''
    from datetime import date
    logid = {'logid': "%s" % ('DM')}

    def get_result_version(result):
        # get the major version the report was generated with
        simple_version = re.compile(r"^(\d+\.?\d*)")
        try:
            versions = dict(v.split(':') for v in result.analysisVersion.split(",") if v)
            version = float(simple_version.match(versions['db']).group(1))
        except Exception:
            # version = float(settings.RELVERSION)
            version = 2.2
        return version

    # Create a file to capture the current status of each dataset
    try:
        track = open("/var/log/ion/dataset_track.csv", "ab")
        track.write("\n====================================================\n")
        track.write("Generated on: %s\n" % date.today())
        track.write("====================================================\n")
    except:
        logger.error(traceback.format_exc(), extra=logid)

    # set of all dmfileset objects
    DMFileSets = DMFileSet.objects.filter(version=settings.RELVERSION)
    DMFileSets_2_2 = DMFileSet.objects.filter(version='2.2')

    # set of results objects with no associated dmfilestat objects
    # for result in Results.objects.select_related().filter(dmfilestat=None).order_by('-pk'):
    from django.db.models import Count
    for result in Results.objects.annotate(num=Count('dmfilestat')).filter(num__lt=4):

        # Create DMFileStat objects, one for each dmfileset
        if get_result_version(result) < 2.3:
            file_sets = DMFileSets_2_2
        else:
            file_sets = DMFileSets

        for dmfileset in file_sets:
            try:
                dmfilestat = result.get_filestat(dmfileset.type)
            except:
                pass
            else:
                if dmfilestat:
                    continue

            kwargs = {
                'result': result,
                'dmfileset': dmfileset,
            }
            dmfilestat = DMFileStat(**kwargs)
            dmfilestat.save()

            if dmfileset.type not in dmactions_types.SIG:
                dmfilestat.preserve_data = result.autoExempt
                # Override the created timestamp to creation of result.timeStamp
                if timezone.is_naive(result.timeStamp):
                    result.timeStamp = result.timeStamp.replace(tzinfo=pytz.utc)
                    # result.save()
                dmfilestat.created = result.timeStamp
            else:
                # Override the created timestamp to creation of experiment.date
                if timezone.is_naive(result.experiment.date):
                    result.experiment.date = result.experiment.date.replace(tzinfo=pytz.utc)
                dmfilestat.created = result.experiment.date
                # Migrate experiment.storage_options == 'KI' to dmfilestat.preservedata = True
                # dmfilestat.setpreserved(True if result.experiment.storage_options == 'KI' else False)
                # Migrate experiment.storage_options setting to dmfilestat.preservedata
                # If 'KI' or 'A', then set preservedata to True (TS-6516).  If 'D', then False
                dmfilestat.setpreserved(False if result.experiment.storage_options == 'D' else True)
                # Log the storage_option for this raw dataset.
                if result.experiment.storage_options == 'D':
                    storage_option = 'Delete'
                elif result.experiment.storage_options == 'A':
                    storage_option = 'Archive'
                elif result.experiment.storage_options == 'KI':
                    storage_option = 'Keep'
                try:
                    track.write("%s,%s,%s\n" %
                                (result.experiment.date, storage_option, result.experiment.expName))
                except:
                    pass

            dmfilestat.save()

            # This will fill in diskspace field for each dmfilestat object.
            # Intense I/O activity
            # dm_utils.update_diskspace(DMFileStat)

            # For Signal Processing Input DMFileSet, modify object if
            # 1) Experiment has a BackupObj
            # 2) No BackupObj, but expDir is a symlink
            # 3) No BackupObj, but expDir does not exist
            if dmfileset.type in dmactions_types.SIG:
                origbackupdate = None
                try:
                    # Some people have more than one backup object here; assume the last is correct
                    # because some error prevented the previous backups.
                    backup = Backup.objects.filter(experiment=result.experiment).order_by('-pk')
                    backup = backup[0]  # will not get here unless query is successful
                    if backup.isBackedUp:
                        dmfilestat.action_state = 'AD'
                        dmfilestat.archivepath = backup.backupPath
                    else:
                        dmfilestat.action_state = 'DD'
                    origbackupdate = backup.backupDate

                except ObjectDoesNotExist:

                    # find those experiments that are missing BackupObj
                    if os.path.islink(result.experiment.expDir):
                        # raw data has been archived
                        dmfilestat.action_state = 'AD'
                        dmfilestat.archivepath = os.path.realpath(result.experiment.expDir)
                        origbackupdate = os.path.getctime(dmfilestat.archivepath)
                    elif not os.path.exists(result.experiment.expDir):
                        # raw data has been deleted
                        dmfilestat.action_state = 'DD'
                        origbackupdate = datetime.now(pytz.UTC)
                    else:
                        # raw data directory exists
                        origbackupdate = None

                except IndexError:
                    pass
                except:
                    logger.warn(traceback.format_exc(), extra=logid)
                finally:
                    if origbackupdate:
                        dmfilestat.save()
                        # Create an EventLog that records the date of this Backup event
                        msg = 'archived to %s' % dmfilestat.archivepath if dmfilestat.action_state == 'AD' else 'deleted'
                        event_log = EventLog.objects.add_entry(
                            result, "Raw data has been %s" % (msg), username='dm_agent')
                        event_log.created = backup.backupDate
                        event_log.save()

            # For Base, Output, Intermediate, modify if report files have been archived/deleted
            # Limit to a single dmfilestat category else we repeat EventLog entries
            elif dmfileset.type in dmactions_types.OUT:
                if result.reportStatus in ["Archiving", "Archived"] or result.is_archived():
                    dmfilestat.action_state = 'AD'   # files have been archived

                    # create EventLog entry for each Log entry
                    # result.metadata contains log of previous prune/archive/export actions
                    # metadata dictionary can contain:
                    # {'Status','Info','Log'}
                    # result.metaData['Status'] is most recent action status
                    # result.metaData['Info'] is most recent action information
                    # result.metaData['Log'] is a list of actions taken in the past: (Last item in list should be same as metaData['Status'] and ['Info'])
                    # Each item in ['Log'] is a dictionary containing
                    # {'Comment','Date','Info','Status'}
                    for item in result.metaData.get('Log', None):
                        event_log = EventLog.objects.add_entry(
                            result, "%s. Comment %s" % (item['Info'], item['Comment']), username='dm_agent')
                        event_log.created = datetime.strptime(item['Date'], "%Y-%m-%d %H:%M:%S.%f")
                        event_log.save()

                        # Search for string indicating archive path
                        if "Archiving report " in item['Info']:
                            arch_dir = item['Info'].split()[-1]
                            # print ("we are guessing this is an archive path: '%s'" % arch_dir)
                            dmfilestat.archivepath = arch_dir

                        dmfilestat.save()

        else:
            pass
    try:
        track.close()
    except:
        pass

    # Update default dmfileset to v2.2 for results that fail when parsing analysisVersion
    # as this is likely to happen for very old results which would be better fit by 2.2 filters
    to_fix = []
    for pk, analysis_version, set_version in DMFileStat.objects.values_list('pk', 'result__analysisVersion', 'dmfileset__version'):
        simple_version = re.compile(r"^(\d+\.?\d*)")
        try:
            versions = dict(v.split(':') for v in analysis_version.split(",") if v)
            version = float(simple_version.match(versions['db']).group(1))
        except Exception:
            if set_version == '3.6':
                to_fix.append(pk)

    dmfilestats = DMFileStat.objects.filter(pk__in=to_fix)
    for dmfileset in DMFileSets_2_2:
        dmfilestats.filter(dmfileset__type=dmfileset.type).update(dmfileset=dmfileset)


def notify(name_list, recipient):
    '''sends an email with list of experiments slated for removal'''

    # Check for blank email
    # TODO: check for valid email address
    if recipient is None or recipient == "":
        return False

    # Needed to send email
    settings.EMAIL_HOST = 'localhost'
    settings.EMAIL_PORT = 25
    settings.EMAIL_USE_TLS = False

    try:
        site_name = GlobalConfig.get().site_name
    except:
        site_name = "Torrent Server"

    hname = socket.gethostname()

    subject_line = 'Torrent Server Data Management Action Request'
    reply_to = 'donotreply@iontorrent.com'
    message = 'From: %s (%s)\n' % (site_name, hname)
    message += '\n'
    message += 'Results drive capacity threshold has been reached.\n'
    message += 'Signal Processing files have been identified for removal.\n'
    message += 'Please go to Services Page and acknowledge so that removal can proceed.\n'
    message += 'Removal will not occur without this acknowledgement.\n'
    message += '\n'
    message += 'The following Reports have Signal Processing files selected for Deletion:'
    message += "\n"
    count = 0
    for e in name_list:
        message += "- %s\n" % e
        count += 1

    # Send the email only if there are runs that have not triggered a notification
    if count > 0:
        try:
            mail.send_mail(subject_line, message, reply_to, recipient)
        except:
            logger.warning(traceback.format_exc(), extra=logid)
            return False
        else:
            logger.info("Notification email sent for user acknowledgement", extra=logid)
            return True


def update_files_in_use():
    """
    Updates DMFileStats files_in_use (used in dmactions.action_validation to block file removal)
    Contacts jobserver to retrieve currently active analysis jobs.
    Also adds any results with active plugins.
    """
    old_in_use = DMFileStat.objects.exclude(files_in_use='')

    conn = client.connect(settings.JOBSERVER_HOST, settings.JOBSERVER_PORT)
    running = conn.running()
    active = []
    for item in running:
        try:
            result = Results.objects.get(pk=item[2])
        except ObjectDoesNotExist:
            logger.warn("Results object does not exist: %d" % (item[2]), extra=logid)
            continue
        # check the status - completed results still show up on the running list for a short time
        if result.status == 'Completed' or result.status == 'TERMINATED':
            continue
        # set files_in_use for current result plus any related sigproc dmfilestats
        dmfilestats = result.dmfilestat_set.all() | DMFileStat.objects.filter(
            dmfileset__type=dmactions_types.SIG, result__experiment_id=result.experiment_id)
        msg = "%s (%s), status = %s" % (result.resultsName, result.id, result.status)
        dmfilestats.update(files_in_use=msg)
        active.extend(dmfilestats.values_list('pk', flat=True))

    # Add reports that have plugins currently running
    # consider only plugins with state='Started' and starttime within a day
    timerange = datetime.now(pytz.UTC) - timedelta(days=1)
    pluginresults = PluginResultJob.objects.filter(state='Started', starttime__gt=timerange)
    for pk in set(pluginresults.values_list('plugin_result__result__pk', flat=True)):
        result = Results.objects.get(pk=pk)
        dmfilestats = result.dmfilestat_set.all()
        msg = "plugins running on %s (%s)" % (result.resultsName, result.id)
        dmfilestats.update(files_in_use=msg)
        active.extend(dmfilestats.values_list('pk', flat=True))

    # reset files_in_use to blank for any non-active dmfilestats
    num = old_in_use.exclude(pk__in=active).update(files_in_use='')
    logger.debug('update_files_in_use(): %s dmilestats in use by active jobs, %s released for dmactions' %
                 (len(active), num), extra=logid)


def test_tracker():
    '''Debugging tool'''
    from datetime import date
    # Create a file to capture the current status of each dataset
    try:
        track = open("/var/log/ion/dataset_track.csv", "ab")
        track.write("\n====================================================\n")
        track.write("Generated on: %s\n" % date.today())
        track.write("====================================================\n")
    except:
        print traceback.format_exc()

    for result in Results.objects.all():
        # Log the storage_option for this raw dataset.
        if result.experiment.storage_options == 'D':
            storage_option = 'Delete'
        elif result.experiment.storage_options == 'A':
            storage_option = 'Archive'
        elif result.experiment.storage_options == 'KI':
            storage_option = 'Keep'
        track.write("%s,%s,%s\n" % (result.experiment.date, storage_option, result.experiment.expName))
    track.close()
