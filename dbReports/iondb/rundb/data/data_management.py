# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
#!/usr/bin/env python
'''
fileserver_space_check() is a periodic task that gets executed by celery daemon
on a repeating schedule.  If there are any actions to take, it will execute a
celery task called manage_data() which will archive or delete one fileset category
for one Result object.  This function will in turn call a function to do the
actual work in the filesystem.
'''
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
from celery.utils.log import get_task_logger
from django.db.models import Q
from django.utils import timezone
from django.core import mail
from django.core.exceptions import ObjectDoesNotExist

from iondb.rundb.models import FileServer, ReportStorage, BackupConfig, GlobalConfig, \
    DMFileSet, DMFileStat, User, Experiment, Backup, Results, EventLog, Message, PluginResult
from iondb.utils.files import percent_full, getdeviceid
from iondb.utils.TaskLock import TaskLock
import iondb.settings as settings
from iondb.rundb.data.tasks import delete_action, archive_action, export_action
from iondb.rundb.data import dmactions_types
from iondb.rundb.data import exceptions as DMExceptions
from iondb.rundb.data import dmactions
from iondb.rundb.data.dmactions import slugify
from iondb.anaserve import client
from iondb.rundb.data import tasks as datatasks
logger = get_task_logger('data_management')


#at celeryd start-up, dump an entry into log file.
from celery.signals import celeryd_after_setup
@celeryd_after_setup.connect
def configure_workers(sender=None, conf=None, **kwargs):
    if 'periodic' in sender:
        logger.info("Restarted: %s" % sender)


@task(queue="periodic", expires=15)
def manage_manual_action():
    logger.debug("manage_manual_action")
    try:
        #Create lock file to prevent more than one celery task for manual action (export or archive)
        lock_id = 'manual_action_lock_id'
        applock = TaskLock(lock_id)
        if not(applock.lock()):
            logger.debug("failed to acquire lock file: %d" % os.getpid())
            logger.info("manage_manual_action task still executing")
            return

        logger.debug("Worker PID %d lock_id created %s" % (os.getpid(),lock_id))

    except Exception as e:
        logger.exception(e)

    try:
        #
        # Check for manually selected Archive and Export actions - action_state == 'SA' or 'SE' or 'SD'
        # Manual Delete actions do not get processed here.  Only suspended Delete actions get processed here.
        # These jobs should execute even when auto action is disabled.
        # Note: manual actions will not be executed in the order they are selected, but by age.
        #
        manualSelects = DMFileStat.objects.filter(action_state__in=['SA','SE','SD']).order_by('created')
        if manualSelects.exists():
            actiondmfilestat = manualSelects[0]
            user = actiondmfilestat.user_comment.get('user', 'dm_agent')
            user_comment = actiondmfilestat.user_comment.get('user_comment', 'Manual Action')

            dmfileset = actiondmfilestat.dmfileset
            if actiondmfilestat.action_state == 'SA':
                logger.info("Manual Archive Action: %s from %s" % (dmfileset.type,actiondmfilestat.result.resultsName))
                archive_action(user, user_comment, actiondmfilestat, lock_id, msg_banner = True)
            elif actiondmfilestat.action_state == 'SE':
                logger.info("Manual Export Action: %s from %s" % (dmfileset.type,actiondmfilestat.result.resultsName))
                export_action(user, user_comment, actiondmfilestat, lock_id, msg_banner = True)
            elif actiondmfilestat.action_state == 'SD':
                logger.info("Delete Action: %s from %s" % (dmfileset.type,actiondmfilestat.result.resultsName))
                delete_action(user, "Continuing delete action after being suspended", actiondmfilestat, lock_id, msg_banner = True)
            else:
                logger.warn("Dev Error: we don't handle this '%s' here" % actiondmfilestat.action_state)

        else:
            applock.unlock()
            logger.debug("Worker PID %d lock_id destroyed on exit %s" % (os.getpid(),lock_id))
#NOTE: all these exceptions are also handled in manage_data() below.  beaucoup de duplication de code
    except (DMExceptions.FilePermission,
            DMExceptions.InsufficientDiskSpace,
            DMExceptions.MediaNotSet,
            DMExceptions.MediaNotAvailable,
            DMExceptions.FilesInUse) as e:
        applock.unlock()
        message  = Message.objects.filter(tags__contains=e.tag)
        if not message:
            Message.error(e.message,tags=e.tag)
            #TODO: TS-6525: This logentry will repeat for an Archive action every 30 seconds until the cause of the exception is fixed.
            #at least, while the message banner is raised, suppress additional Log Entries.
            dmactions.add_eventlog(actiondmfilestat, "%s - %s" % (dmfileset.type, e.message), username='dm_agent')
        # Revert this dmfilestat object action-state to Local
        actiondmfilestat.setactionstate('L')
    except DMExceptions.SrcDirDoesNotExist as e:
        applock.unlock()
        msg = "Src Dir not found: %s. Setting action_state to Deleted" % e.message
        EventLog.objects.add_entry(actiondmfilestat.result,msg,username='dm_agent')
        actiondmfilestat.setactionstate('DD')
        logger.info(msg)
    except Exception as e:
        applock.unlock()
        msg = "action error on %s " % actiondmfilestat.result.resultsName
        msg += " Error: %s" % str(e)
        logger.exception("%s - %s" % (dmfileset.type, msg))
        dmactions.add_eventlog(actiondmfilestat,"%s - %s" % (dmfileset.type, msg),username='dm_agent')
        # Revert this dmfilestat object action-state to Local
        actiondmfilestat.setactionstate('L')

    return


@task(queue="periodic", expires=15)
def manage_data(deviceid, dmfileset, pathlist, auto_acknowledge_enabled, auto_action_enabled):
    logger.debug("manage_data: %s %s %s" % (dmfileset['auto_action'], hex(deviceid),dmfileset['type']))

    def getfirstnotpreserved(dmfilestats, action, threshdate):
        '''QuerySet of DMFileStat objects.  Returns first instance of an object
        with preserve_data set to False and passes the action_validation test'''
        logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
        logger.debug("Looking at %d dmfilestat objects" % dmfilestats.count())
        for archiveme in dmfilestats:
            if not archiveme.getpreserved():
                try:
                    dmactions.action_validation(archiveme,action)
                    return archiveme
                except(DMExceptions.FilesInUse,DMExceptions.FilesMarkedKeep):
                    logger.debug("%s Failed action_validation.  Try next fileset" % archiveme.result.resultsName)
                except DMExceptions.BaseInputLinked:
                    # want to allow Basecalling Input delete if all results are expired
                    relatedObjs = DMFileStat.objects.filter(result__experiment=archiveme.result.experiment, dmfileset__type=dmactions_types.BASE)
                    if relatedObjs.count() == relatedObjs.filter(created__lt=threshdate).count():
                        archiveme.allow_delete = True
                        return archiveme
                    else:
                        logger.debug("%s Failed action_validation.  Try next fileset" % archiveme.result.resultsName)
                except:
                    logger.error(traceback.format_exc())
            else:
                logger.debug("Skipped a preserved fileset: %s" % archiveme.result.resultsName)

        logger.info("%d filestat objects are preserved." % dmfilestats.count())
        raise DMExceptions.NoDMFileStat("NONE FOUND")

    try:
        #logger.debug("manage_data lock for %s (%d)" % (dmfileset['type'], os.getpid()))
        #Create lock file to prevent more than one celery task for each process_type and partition
        lock_id = "%s_%s" % (hex(deviceid),slugify(dmfileset['type']))
        applock = TaskLock(lock_id)

        if not(applock.lock()):
            logger.info("Did not acquire lock file: %s" % lock_id)
            return

        logger.debug("Worker PID %d lock_id created %s" % (os.getpid(),lock_id))

    except Exception as e:
        logger.exception(e)

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
        dmfilestats = dmfilestats.filter(action_state__in=['L','S','N','A'])
        tot_act = dmfilestats.count()
        logger.info("(%s)Total %s: %d Active: %d" %(hex(deviceid),dmfileset['type'],tot_obj,tot_act))

        # Select objects that are old enough
        threshdate = datetime.now(pytz.UTC) - timedelta(days=dmfileset['auto_trigger_age'])
        dmfilestats = dmfilestats.filter(created__lt=threshdate)
        logger.info("(%s)Total %s: %d Expired: %d" %(hex(deviceid),dmfileset['type'],tot_obj,dmfilestats.count()))

        # Select objects stored on the deviceid
        query = Q()
        for path in pathlist:
            if dmfileset['type'] == dmactions_types.SIG:
                query |= Q(result__experiment__expDir__startswith=path)
            # These categories have files in both directory paths.
            elif dmfileset['type'] in [dmactions_types.INTR,dmactions_types.BASE]:
                query |= Q(result__experiment__expDir__startswith=path)
                query |= Q(result__reportstorage__dirPath__startswith=path)
            else:
                query |= Q(result__reportstorage__dirPath__startswith=path)

        dmfilestats = dmfilestats.filter(query)
        logger.info("(%s)Total %s: %d On '%s' path: %d" %(hex(deviceid),dmfileset['type'],tot_obj,path,dmfilestats.count()))

        # Exclude objects marked 'Keep' upfront to optimize db access
        if dmfileset['type'] == dmactions_types.SIG:
            dmfilestats = dmfilestats.exclude(result__experiment__storage_options="KI")
        else:
            dmfilestats = dmfilestats.exclude(preserve_data=True)
        logger.info("(%s)Total %s: %d Not Preserved: %d" %(hex(deviceid),dmfileset['type'],tot_obj,dmfilestats.count()))


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
            logger.info("(%s)Exceed %d days. Eligible to archive: %d" %(hex(deviceid),dmfileset['auto_trigger_age'],dmfilestats.count()))
            actiondmfilestat = None
            #Bail out if disabled
            if auto_action_enabled != True:
                logger.info("Data management auto-action is disabled.")
                applock.unlock()
                logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))
                return

            # Select first object stored on the deviceid
            try:
                actiondmfilestat = getfirstnotpreserved(dmfilestats, dmactions.ARCHIVE, threshdate)
                logger.info("(%s)Picked: %s" % (hex(deviceid),actiondmfilestat.result.resultsName))
            except DMExceptions.NoDMFileStat:
                logger.debug("No filesets to archive on this device: %s" % hex(deviceid))
                applock.unlock()
                logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))
            except:
                logger.error(traceback.format_exc())
                applock.unlock()
                logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))
            else:
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
            logger.info("(%s)Exceed %d days. Eligible to delete: %d" %(hex(deviceid),dmfileset['auto_trigger_age'],dmfilestats.count()))
            if auto_acknowledge_enabled:
                if dmfileset['type'] == dmactions_types.SIG:
                    logger.debug("Sig Proc Input Files auto acknowledge enabled")
                '''
                Do not require 'S' -> 'N' -> 'A' progression before action; mark 'A' and process
                '''
                A_list = dmfilestats.filter(action_state='A')
                if A_list.count() > 0:
                    # there are filesets acknowledged and ready to be deleted.  Covers situation where user has
                    #already acknowledged but recently enabled auto-acknowledge as well.
                    #deleteme = A_list[0]
                    try:
                        actiondmfilestat = getfirstnotpreserved(A_list, dmactions.DELETE, threshdate)
                        logger.info("(%s)Picked: %s" % (hex(deviceid),actiondmfilestat.result.resultsName))
                    except DMExceptions.NoDMFileStat:
                        logger.info("(%s) No filesets to delete on this device" % hex(deviceid))
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))
                    except:
                        logger.error(traceback.format_exc())
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))

                if actiondmfilestat == None:
                    # Select oldest fileset regardless if its 'L','S','N','A'.  This covers situation where user
                    # recently enabled auto-acknowledge
                    try:
                        actiondmfilestat = getfirstnotpreserved(dmfilestats, dmactions.DELETE, threshdate)
                        logger.info("(%s)Picked: %s" % (hex(deviceid),actiondmfilestat.result.resultsName))
                    except DMExceptions.NoDMFileStat:
                        logger.info("(%s) No filesets to delete on this device" % hex(deviceid))
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))
                    except:
                        logger.error(traceback.format_exc())
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))
            else:
                if dmfileset['type'] == dmactions_types.SIG:
                    logger.debug("Sig Proc Input Files auto acknowledge disabled")
                    '''
                    Need to select # of filesets and mark 'S'
                    Need to find 'S' filesets and notify (set to 'N')
                    Set queue length to 5
                    sum(S,N,A) == queue length
                    '''
                    #Get email recipient
                    try:
                        recipient = User.objects.get(username='dm_contact').email
                        recipient = recipient.replace(',',' ').replace(';',' ').split()
                    except:
                        recipient = None

                    # Need to select Experiments to process with appropriate dmfilestats action states.
                    exps = Experiment.objects.exclude(storage_options='KI').exclude(expDir='').order_by('date')
                    L_list = exps.filter(results_set__dmfilestat__in = dmfilestats.filter(action_state='L')).distinct()
                    S_list = exps.filter(results_set__dmfilestat__in = dmfilestats.filter(action_state='S')).distinct()
                    N_list = exps.filter(results_set__dmfilestat__in = dmfilestats.filter(action_state='N')).distinct()
                    A_list = exps.filter(results_set__dmfilestat__in = dmfilestats.filter(action_state='A')).distinct()

                    logger.info("Experiments (Keep=False) L:%d S:%d N:%d A:%d" % (L_list.count(),S_list.count(),N_list.count(),A_list.count()))
                    queue_length = 5
                    to_select = queue_length - (A_list.count() + N_list.count() + S_list.count())
                    if to_select > 0:
                        # Mark to_select number of oldest Experiments, from 'L', to 'S'
                        promoted = dmfilestats.filter(result__experiment__id__in=list(L_list[:to_select].values_list('id',flat=True)))
                        if auto_action_enabled:
                            promoted.update(action_state = 'S')
                            for dmfilestat in promoted:
                                EventLog.objects.add_entry(dmfilestat.result, "Signal Processing Input Selected for Deletion", username='dm_agent')

                    # Get updated list of Selected items
                    selected = dmfilestats.filter(action_state='S')

                    # Send Selected items to be notified
                    if selected.count() > 0 and auto_action_enabled:
                        logger.debug("notify recipient %s" % (recipient))
                        if notify([dmfilestat.result.resultsName for dmfilestat in selected],recipient):
                            selected.update(action_state = 'N')
                            for dmfilestat in selected:
                                EventLog.objects.add_entry(dmfilestat.result, "Notification for Deletion Sent", username='dm_agent')

                    try:
                        actiondmfilestat = getfirstnotpreserved(dmfilestats.filter(action_state='A'), dmactions.DELETE, threshdate)
                        logger.info("(%s)Picked: %s" % (hex(deviceid),actiondmfilestat.result.resultsName))
                    except DMExceptions.NoDMFileStat:
                        logger.info("(%s) No filesets to delete on this device" % hex(deviceid))
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))
                    except:
                        logger.error(traceback.format_exc())
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))

                else:
                    try:
                        actiondmfilestat = getfirstnotpreserved(dmfilestats, dmactions.DELETE, threshdate)
                        logger.info("(%s)Picked: %s" % (hex(deviceid),actiondmfilestat.result.resultsName))
                    except DMExceptions.NoDMFileStat:
                        logger.info("(%s) No filesets to delete on this device" % hex(deviceid))
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))
                    except:
                        logger.error(traceback.format_exc())
                        applock.unlock()
                        logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))

            #Bail out if disabled
            if auto_action_enabled != True:
                logger.info("Data management auto-action is disabled.")
                applock.unlock()
                logger.debug("Worker PID %d lock_id destroyed %s" % (os.getpid(),lock_id))
                return

            if actiondmfilestat is not None:
                delete_action('dm_agent', user_comment, actiondmfilestat, lock_id, confirmed=getattr(actiondmfilestat,'allow_delete',False) )

        else:
            logger.error("Unknown or unhandled action: %s" % dmfileset['auto_action'])
            applock.unlock()

    except (DMExceptions.FilePermission,
            DMExceptions.InsufficientDiskSpace,
            DMExceptions.MediaNotSet,
            DMExceptions.MediaNotAvailable,
            DMExceptions.FilesInUse) as e:
        applock.unlock()
        message  = Message.objects.filter(tags__contains=e.tag)
        if not message:
            Message.error(e.message,tags=e.tag)
            #TODO: TS-6525: This logentry will repeat for an Archive action every 30 seconds until the cause of the exception is fixed.
            #at least, while the message banner is raised, suppress additional Log Entries.
            EventLog.objects.add_entry(actiondmfilestat.result,"%s - %s" % (dmfileset['type'], e.message),username='dm_agent')
    except DMExceptions.SrcDirDoesNotExist as e:
        applock.unlock()
        msg = "Src Dir not found: %s. Setting action_state to Deleted" % e.message
        EventLog.objects.add_entry(actiondmfilestat.result,msg,username='dm_agent')
        actiondmfilestat.setactionstate('DD')
        logger.info(msg)
    except Exception as inst:
        applock.unlock()
        msg = ''
        if actiondmfilestat:
            msg = "Auto-action error on %s " % actiondmfilestat.result.resultsName
        msg += " Error: %s" % str(inst)
        logger.exception("%s - %s" % (dmfileset['type'], msg))

        if actiondmfilestat:
            EventLog.objects.add_entry(actiondmfilestat.result,"%s - %s" % (dmfileset['type'], msg),username='dm_agent')

    return


@periodic_task(run_every=timedelta(seconds=60), expires=15, queue="periodic")
#@task
def fileserver_space_check():
    '''For each file server, compare disk usage to backup threshold value.
    If disk usage exceeds threshold, launch celery task to delete/archive
    raw data directories.
    '''
    # Get GlobalConfig object in order to access auto-acknowledge bit
    gc = GlobalConfig.get()
    auto_acknowledge = gc.auto_archive_ack
    auto_action_enabled = gc.auto_archive_enable

    # Get list of File Server objects
    file_servers = FileServer.objects.all().order_by('pk').values()

    # Get list of Report Storage objects
    report_storages = ReportStorage.objects.all().order_by('pk').values()

    # dict of fileset cateogries each with list of partition ids that can be acted upon.
    category_list = {}
    #-------------------------------------------------
    # DELETE action only happens if threshold reached
    #-------------------------------------------------
    for dmfileset in DMFileSet.objects.filter(version=settings.RELVERSION).filter(auto_action='DEL').values():

        cat_name = slugify(dmfileset['type'])
        category_list[cat_name] = {
            'dmfileset':dmfileset,
            'devlist':[],
            'partitions':[],
        }

        for partition in _partitions(file_servers,report_storages):

            if partition['diskusage'] >= dmfileset['auto_trigger_usage']:

                logger.info("%s %s %.2f%% exceeds %s %.0f%%" % (
                    hex(partition['devid']),
                    partition['path'],
                    partition['diskusage'],
                    dmfileset['type'],
                    dmfileset['auto_trigger_usage']))

                category_list[cat_name]['devlist'].append(partition['devid'])
                category_list[cat_name]['partitions'].append(partition)

            else:

                logger.info("%s %s %.2f%% below %s %.0f%%" % (
                    hex(partition['devid']),
                    partition['path'],
                    partition['diskusage'],
                    dmfileset['type'],
                    dmfileset['auto_trigger_usage']))

        # uniquify the deviceid list
        category_list[cat_name]['devlist'] = list(set(category_list[cat_name]['devlist']))

    #-------------------------------------------------------------------------------
    #ARCHIVE action happens as soon as grace period has expired (no threshold check)
    #-------------------------------------------------------------------------------
    for dmfileset in DMFileSet.objects.filter(version=settings.RELVERSION).filter(auto_action='ARC').values():

        cat_name = slugify(dmfileset['type'])
        category_list[cat_name] = {
            'dmfileset':dmfileset,
            'devlist':[],
            'partitions':[],
        }

        for partition in _partitions(file_servers,report_storages):
            logger.debug("%s %s" %( partition['path'],hex(partition['devid'])))
            category_list[cat_name]['devlist'].append(partition['devid'])
            category_list[cat_name]['partitions'].append(partition)

        # uniquify the deviceid list
        category_list[cat_name]['devlist'] = list(set(category_list[cat_name]['devlist']))


    #-------------------------------------------------------------
    # Action loop - for each category, launch action per deviceid
    #-------------------------------------------------------------
    if not auto_action_enabled:
        logger.info("Data management auto-action is disabled.")

    # update any DMFileStats that are in use by active analysis jobs
    try:
        update_files_in_use()
    except:
        logger.error('Unable to update active DMFileStats')
        logger.error(traceback.format_exc())

    # Checks for manual export and archive requests.
    manage_manual_action.delay()

    for category,dict in category_list.iteritems():
        for deviceid in dict['devlist']:
            pathlist = [item['path'] for item in dict['partitions'] if item['devid'] == deviceid]
            async_task_result = manage_data.delay(deviceid, dict['dmfileset'], pathlist, auto_acknowledge, auto_action_enabled)

    return


def _partitions(file_servers,report_storages):
    partitions = []
    for fs in file_servers:
        if os.path.exists(fs['filesPrefix']):
            partitions.append(
                {
                    'path':fs['filesPrefix'],
                    'diskusage':fs['percentfull'],
                    'devid':getdeviceid(fs['filesPrefix']),
                }
            )
    for rs in report_storages:
        if os.path.exists(rs['dirPath']):
            partitions.append(
                {
                    'path':rs['dirPath'],
                    'diskusage':rs.get('percentfull',percent_full(rs['dirPath'])),
                    'devid':getdeviceid(rs['dirPath']),
                }
            )
    return partitions


def backfill_create_dmfilestat():
    '''
    Examines every result object for related DMFileStat objects and creates if missing
    Migrates pre-TS3.6 servers to the new data management schema.  Called from postinst.in
    '''
    from datetime import date

    def get_result_version(result):
        #get the major version the report was generated with
        simple_version = re.compile(r"^(\d+\.?\d*)")
        try:
            versions = dict(v.split(':') for v in result.analysisVersion.split(",") if v)
            version = float(simple_version.match(versions['db']).group(1))
        except Exception:
            #version = float(settings.RELVERSION)
            version = 2.2
        return version

    # Create a file to capture the current status of each dataset
    try:
        track = open("/var/log/ion/dataset_track.csv", "ab")
        track.write("\n====================================================\n")
        track.write("Generated on: %s\n" % date.today())
        track.write("====================================================\n")
    except:
        logger.error(traceback.format_exc())

    # set of all dmfileset objects
    DMFileSets = DMFileSet.objects.filter(version=settings.RELVERSION)
    DMFileSets_2_2 = DMFileSet.objects.filter(version='2.2')

    # set of results objects with no associated dmfilestat objects
    #for result in Results.objects.select_related().filter(dmfilestat=None).order_by('-pk'):
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
                continue

            kwargs = {
                'result':result,
                'dmfileset':dmfileset,
            }
            dmfilestat = DMFileStat(**kwargs)
            dmfilestat.save()

            if dmfileset.type not in dmactions_types.SIG:
                dmfilestat.preserve_data = result.autoExempt
                #Override the created timestamp to creation of result.timeStamp
                if timezone.is_naive(result.timeStamp):
                    result.timeStamp = result.timeStamp.replace(tzinfo = pytz.utc)
                    #result.save()
                dmfilestat.created = result.timeStamp
            else:
                #Override the created timestamp to creation of experiment.date
                if timezone.is_naive(result.experiment.date):
                    result.experiment.date = result.experiment.date.replace(tzinfo = pytz.utc)
                dmfilestat.created = result.experiment.date
                #Migrate experiment.storage_options == 'KI' to dmfilestat.preservedata = True
                #dmfilestat.setpreserved(True if result.experiment.storage_options == 'KI' else False)
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
                    track.write("%s,%s,%s\n" % (result.experiment.date,storage_option,result.experiment.expName))
                except:
                    pass

            dmfilestat.save()

            #This will fill in diskspace field for each dmfilestat object.
            #Intense I/O activity
            #dmactions.update_diskspace(DMFileStat)

            # For Signal Processing Input DMFileSet, modify object if
            #1) Experiment has a BackupObj
            #2) No BackupObj, but expDir is a symlink
            #3) No BackupObj, but expDir does not exist
            if dmfileset.type in dmactions_types.SIG:
                origbackupdate = None
                try:
                    #Some people have more than one backup object here; assume the last is correct
                    #because some error prevented the previous backups.
                    backup = Backup.objects.filter(experiment=result.experiment).order_by('-pk')
                    backup = backup[0]  # will not get here unless query is successful
                    if backup.isBackedUp:
                        dmfilestat.action_state = 'AD'
                        dmfilestat.archivepath = backup.backupPath
                    else:
                        dmfilestat.action_state = 'DD'
                    origbackupdate = backup.backupDate

                except ObjectDoesNotExist:

                    #find those experiments that are missing BackupObj
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
                    logger.warn(traceback.format_exc())
                finally:
                    if origbackupdate:
                        dmfilestat.save()
                        #Create an EventLog that records the date of this Backup event
                        msg = 'archived to %s' % dmfilestat.archivepath if dmfilestat.action_state == 'AD' else 'deleted'
                        el = EventLog.objects.add_entry(result, "Raw data has been %s" % (msg), username='dm_agent')
                        el.created = backup.backupDate
                        el.save()

            # For Base, Output, Intermediate, modify if report files have been archived/deleted
            # Limit to a single dmfilestat category else we repeat EventLog entries
            elif dmfileset.type in dmactions_types.OUT:
                if result.reportStatus in ["Archiving","Archived"] or result.is_archived():
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
                        el = EventLog.objects.add_entry(result,"%s. Comment %s" %(item['Info'], item['Comment']), username='dm_agent')
                        el.created = dt_obj = datetime.strptime(item['Date'], "%Y-%m-%d %H:%M:%S.%f")
                        el.save()

                        # Search for string indicating archive path
                        if "Archiving report " in item['Info']:
                            arch_dir = item['Info'].split()[-1]
                            #print ("we are guessing this is an archive path: '%s'" % arch_dir)
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
    for pk,analysisVersion,setVersion in DMFileStat.objects.values_list('pk','result__analysisVersion','dmfileset__version'):
        simple_version = re.compile(r"^(\d+\.?\d*)")
        try:
            versions = dict(v.split(':') for v in analysisVersion.split(",") if v)
            version = float(simple_version.match(versions['db']).group(1))
        except Exception:
            if not setVersion == '2.2':
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

    #Needed to send email
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
            logger.warning(traceback.format_exc())
            return False
        else:
            logger.info("Notification email sent for user acknowledgement")
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
            logger.warn("Results object does not exist: %d" % (item[2]))
            continue
        # check the status - completed results still show up on the running list for a short time
        if result.status == 'Completed' or result.status == 'TERMINATED':
            continue
        # set files_in_use for current result plus any related sigproc dmfilestats
        dmfilestats = result.dmfilestat_set.all() | DMFileStat.objects.filter(
                dmfileset__type=dmactions_types.SIG, result__experiment_id=result.experiment_id)
        msg = "%s (%s), status = %s" % (result.resultsName, result.id, result.status)
        dmfilestats.update(files_in_use = msg)
        active.extend(dmfilestats.values_list('pk',flat=True))

    # Add reports that have plugins currently running
    # consider only plugins with state='Started' and starttime within a day
    timerange = datetime.now(pytz.UTC) - timedelta(days=1)
    pluginresults = PluginResult.objects.filter(state='Started', starttime__gt=timerange)
    for pk in set(pluginresults.values_list('result__pk', flat=True)):
        result = Results.objects.get(pk=pk)
        dmfilestats = result.dmfilestat_set.all()
        msg = "plugins running on %s (%s)" % (result.resultsName, result.id)
        dmfilestats.update(files_in_use = msg)
        active.extend(dmfilestats.values_list('pk',flat=True))

    # reset files_in_use to blank for any non-active dmfilestats
    num = old_in_use.exclude(pk__in=active).update(files_in_use='')
    logger.debug('update_files_in_use(): %s dmilestats in use by active jobs, %s released for dmactions' % (len(active), num))


def test_tracker():
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
        track.write("%s,%s,%s\n" % (result.experiment.date,storage_option,result.experiment.expName))
    track.close()
