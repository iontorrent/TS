# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from celery.task import task
from celery.utils.log import get_task_logger

import sys
import traceback
import logging

from django.core import urlresolvers
from django import shortcuts
from iondb.rundb.models import Message, Results, EventLog
from iondb.rundb.data import dmactions
from iondb.rundb.data.project_msg_banner import project_msg_banner

logger = get_task_logger('data_management')


def _store_message(user, pk, status, action):
    url = urlresolvers.reverse('dm_log', args=(pk,))
    report = Results.objects.get(id=pk)
    token = 'complete' if status else 'failed'
    msg = "Report (%s) %s %s. <a href='%s'  data-toggle='modal' data-target='#modal_report_log'>View Report Log</a>" % (report.resultsName, action, token, url)
    func = Message.success if status else Message.error
    return func(msg, route=user)


def _store_messages(user, pk, status_list, action):
    url = urlresolvers.reverse('dm_log', args=(pk,))
    report = Results.objects.get(id=pk)
    msg = "Report (%s) %s " % (report.resultsName, action.title())
    for category,status in status_list.iteritems():
        msg += " %s, " % str(category)
        msg += 'success' if status == "Success" else status
    msg += " <a href='%s'  data-toggle='modal' data-target='#modal_report_log'>View Report Log</a>" % (url)
    func = Message.success if status else Message.error
    return func(msg, route=user)


@task(queue="periodic")
def action_group(user, categories, action, dmfilestat_dict, user_comment, backup_directory=None, confirmed=False):
    '''Single task to group multiple results' actions status.
    Cycle through the dmfilestat objects per result_id, then per category.
    DELETE action is performed immediately
    ARCHIVE/EXPORT change action state to Pending, actual action will be launched by data_management.py periodic task.
    TEST action prints selected files to log.
    '''
    project_msg = {}
    for result_pk, DMFileStats in dmfilestat_dict.iteritems():
        logger.debug("result_pk: %s" % result_pk)
        logger.debug("%s contains %d" % (type(DMFileStats),DMFileStats.count()))

        msg_dict = {}
        for selection_id in categories:
            logger.debug("category: %s" % selection_id)

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
                        logger.error(status)
                except Exception as inst:
                    msg_dict[selection_id] = "Error: %s" % str(inst)
                    logger.exception("%s - %s" % (selection_id, msg_dict[selection_id]))
                    EventLog.objects.add_entry(dmfilestat.result,"%s - %s. User Comment: %s" % (selection_id, msg_dict[selection_id],user_comment),username=user)
                else:
                    msg_dict[selection_id] = status
                    logger.debug("%s - %s" % (selection_id, msg_dict[selection_id]))

        # Generates message per result
        logger.debug("%s" % msg_dict)
        #message = _store_messages(user, result_pk, msg_dict, action)
        project_msg[result_pk] = msg_dict

    logger.debug(project_msg)
    #Generate a status message per group of results?
    project_msg_banner(user, project_msg, action)

@task(queue="periodic")
def delete_action(user, user_comment, dmfilestat, lockfile=None, msg_banner = False, confirmed=False):
    ''' Delete Action by wrapping invocation with celery task for sync / async execution'''
    try:
        dmactions.delete(user, user_comment, dmfilestat, lockfile, msg_banner, confirmed)
    except:
        raise


@task(queue="periodic")
def archive_action(user, user_comment, dmfilestat, lockfile=None, msg_banner = False):
    ''' Archive Action by wrapping invocation with celery task for sync / async execution'''
    try:
        backup_directory = dmfilestat.archivepath if dmfilestat.action_state == 'SA' else None
        dmactions.archive(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory)
    except:
        raise


@task(queue="periodic")
def export_action(user, user_comment, dmfilestat, lockfile=None, msg_banner = False):
    ''' Export Action by wrapping invocation with celery task for sync / async execution'''
    try:
        backup_directory = dmfilestat.archivepath if dmfilestat.action_state == 'SE' else None
        dmactions.export(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory)
    except:
        raise


@task(queue="periodic")
def test_action(user, user_comment, dmfilestat, lockfile=None, msg_banner = False):
    ''' Test Action by wrapping invocation with celery task for sync / async execution'''
    try:
        backup_directory = dmfilestat.archivepath
        dmactions.test(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory)
    except:
        raise

@task(queue="periodic")
def update_dmfilestats_diskspace(dmfilestat):
    ''' Task to update DMFileStat.diskspace '''
    dmactions.update_diskspace(dmfilestat)
