# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from celery.task import task, periodic_task
from celery.utils.log import get_task_logger
from iondb.celery import app

import sys
import traceback
import logging

from django.core import urlresolvers
from django import shortcuts
from iondb.rundb.models import Message, Results, EventLog, DMFileStat
from iondb.rundb.data import dmactions
from iondb.rundb.data import dm_utils
from iondb.rundb.data import dmfilestat_utils
from iondb.rundb.data import dmactions_types
from iondb.rundb.data.dmactions_types import FILESET_TYPES
from iondb.rundb.data.project_msg_banner import project_msg_banner
from celery.exceptions import SoftTimeLimitExceeded

logger = get_task_logger('data_management')
logid = {'logid':"%s" % ('tasks')}


@app.task(queue = "dmmanage", ignore_result = True)
def action_group(user, categories, action, dmfilestat_dict, user_comment, backup_directory=None, confirmed=False):
    '''Single task to group multiple results' actions status.
    Cycle through the dmfilestat objects per result_id, then per category.
    DELETE action is performed immediately
    ARCHIVE/EXPORT change action state to Pending, actual action will be launched by data_management.py periodic task.
    TEST action prints selected files to log.
    '''
    project_msg = {}
    for result_pk, DMFileStats in dmfilestat_dict.iteritems():
        logger.debug("result_pk: %s" % result_pk, extra = logid)
        logger.debug("%s contains %d" % (type(DMFileStats),DMFileStats.count()), extra = logid)

        msg_dict = {}
        for selection_id in categories:
            logger.debug("category: %s" % selection_id, extra = logid)

            for dmfilestat in DMFileStats.filter(dmfileset__type=selection_id):
                try:
                    if action == dmactions.TEST:
                        test_action(user, user_comment, dmfilestat)
                        status = "success"
                    elif action == dmactions.DELETE:
                        delete_action(user, user_comment, dmfilestat, confirmed=confirmed)
                        status = "success"
                    elif action == dmactions.ARCHIVE or action == dmactions.EXPORT:
                        status = dmactions.set_action_pending(user, user_comment, action, dmfilestat, backup_directory)
                    else:
                        status = "error, unknown action POSTed: '%s'" % action
                        logger.error(status, extra = logid)
                except Exception as inst:
                    msg_dict[selection_id] = "Error: %s" % str(inst)
                    logger.error("%s - %s" % (selection_id, msg_dict[selection_id]), extra = logid)
                    logger.error(traceback.format_exc(), extra = logid)
                    EventLog.objects.add_entry(dmfilestat.result,"%s - %s. User Comment: %s" % (selection_id, msg_dict[selection_id],user_comment),username=user)
                else:
                    msg_dict[selection_id] = status
                    logger.debug("%s - %s" % (selection_id, msg_dict[selection_id]), extra = logid)

        # Generates message per result
        logger.debug("%s" % msg_dict, extra = logid)
        project_msg[result_pk] = msg_dict

    logger.debug(project_msg, extra = logid)
    #Generate a status message per group of results?
    project_msg_banner(user, project_msg, action)


def delete_action(user, user_comment, dmfilestat, lockfile=None, msg_banner = False, confirmed=False):
    ''' Delete Action by wrapping invocation with celery task for sync / async execution'''
    try:
        dmactions.delete(user, user_comment, dmfilestat, lockfile, msg_banner, confirmed)
    except:
        raise

def archive_action(user, user_comment, dmfilestat, lockfile=None, msg_banner = False):
    ''' Archive Action by wrapping invocation with celery task for sync / async execution'''
    try:
        backup_directory = dmfilestat.archivepath if dmfilestat.action_state == 'SA' else None
        dmactions.archive(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory)
    except:
        raise


def export_action(user, user_comment, dmfilestat, lockfile=None, msg_banner = False):
    ''' Export Action by wrapping invocation with celery task for sync / async execution'''
    try:
        backup_directory = dmfilestat.archivepath if dmfilestat.action_state == 'SE' else None
        dmactions.export(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory)
    except:
        raise


def test_action(user, user_comment, dmfilestat, lockfile=None, msg_banner = False):
    ''' Test Action by wrapping invocation with celery task for sync / async execution'''
    try:
        backup_directory = dmfilestat.archivepath
        dmactions.test(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory)
    except:
        raise


@app.task(queue="diskutil", ignore_result = True)
def update_dmfilestats_diskspace(dmfilestat):
    ''' Task to update DMFileStat.diskspace '''
    search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]
    try:
        cached_file_list = dm_utils.get_walk_filelist(search_dirs, list_dir=dmfilestat.result.get_report_dir())
        dmfilestat_utils.update_diskspace(dmfilestat, cached=cached_file_list)
    except:
        logger.exception(traceback.format_exc(), extra = logid)


@app.task(queue="diskutil", time_limit=600, ignore_result = True)
def update_dmfilestat_diskusage(resultpk):
    '''
    Task to update DMFileStat.diskspace for all associated with this resultpk
    This task is launched at the end of pipeline execution.
    NOTE: This can be a long-lived task
    '''
    logid = {'logid':"%s" % ('tasks')}
    try:
        result = Results.objects.get(pk=resultpk)
        search_dirs = [result.get_report_dir(), result.experiment.expDir]
        cached_file_list = dm_utils.get_walk_filelist(search_dirs, list_dir=result.get_report_dir(), save_list=True)
        for dmtype in FILESET_TYPES:
            dmfilestat = result.get_filestat(dmtype)
            dmfilestat_utils.update_diskspace(dmfilestat, cached=cached_file_list)
    except SoftTimeLimitExceeded:
        logger.warn("Time exceeded update_diskusage for (%d) %s" % (resultpk,result.resultsName), extra = logid)
    except:
        raise
    
    try:
        disk_total = 0
        for dmfilestat in [ result.get_filestat(dmtype) for dmtype in FILESET_TYPES]:
            if dmfilestat.dmfileset.type == dmactions_types.SIG:
                dmfilestat.result.experiment.diskusage = dmfilestat.diskspace if dmfilestat.diskspace != None else 0
                dmfilestat.result.experiment.save()
            else:
                partial = dmfilestat.diskspace
                disk_total += int(partial) if partial != None else 0
                result.diskusage = disk_total
                result.save()
        # See dmaction._update_related_objects() which also updates Exp & Results diskusage fields
    except:
        logger.error(traceback.format_exc(), extra = logid)
        raise    


@periodic_task(run_every=300, expires=60, queue="diskutil")
def backfill_dmfilestats_diskspace():
    ''' Backfill records with DMFileStat.diskspace = None, one at a time
        These could be older data sets or new ones where update_diskusage task failed
    '''
    dmfilestats = DMFileStat.objects.filter(diskspace=None, action_state='L', files_in_use='').order_by('-created')
    if dmfilestats.count() > 0:
        dmfilestat = dmfilestats[0]
        search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]
        try:
            cached_file_list = dm_utils.get_walk_filelist(search_dirs, list_dir=dmfilestat.result.get_report_dir(), save_list=True)
            dmfilestat_utils.update_diskspace(dmfilestat, cached=cached_file_list)
        except:
            logger.error(traceback.format_exc(), extra = logid)
            raise


@app.task
def save_serialized_json(resultpk):
    ''' Quick task to serialize and save in a json file all result-related dbase objects '''
    try:
        result = Results.objects.get(pk=resultpk)
        dmactions.write_serialized_json(result, result.get_report_dir())
    except:
        raise

