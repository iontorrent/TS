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
from iondb.rundb.data.dmactions import slugify

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


def project_msg_banner(user, project_msg, action):
    try:
        msg = ''
        thistag = ''
        logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
        for pk,status_list in project_msg.iteritems():
            url = urlresolvers.reverse('dm_log', args=(pk,))
            report = Results.objects.get(id=pk)
            msg += "(%s) %s " % (report.resultsName, action.title())
            for category,status in status_list.iteritems():
                msg += " %s, " % str(category)
                msg += status
                grpstatus = status
                thistag = "%s_%s_%s" % (str(pk),action,slugify(category))
            msg += " <a href='%s'  data-toggle='modal' data-target='#modal_report_log'>View Report Log</a></br>" % (url)
            logger.debug("MESSAGE: %s" % msg)

        #If message already exists (ie, scheduled task) delete it.
        Message.objects.filter(tags=thistag).delete()

        if len(project_msg) > 1:
            thistag = "%s_%s_%s" % ('project',action,slugify(category))

        if grpstatus == 'scheduled':
            func = Message.info
        elif grpstatus == 'success':
            func = Message.success
        else:
            func = Message.error
    except:
        func = Message.error
        logger.exception(traceback.format_exc())
    return func(msg, route=user, tags=thistag)


@task
def action_group(user, categories, action, dmfilestat_dict, user_comment, backup_directory=None):
    '''Single task to group multiple results' actions status.
    Cycle through the dmfilestat objects per result_id, then per category.
    DELETE action is performed immediately
    ARCHIVE/EXPORT change action state to Pending, actual action will be launched by data_management.py periodic task.
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
                    if action == dmactions.DELETE:
                        delete_action(user, user_comment, dmfilestat)
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

@task
def delete_action(user, user_comment, dmfilestat):
    ''' Delete Action by wrapping invocation with celery task for sync / async execution'''
    try:
        dmactions.delete(user, user_comment, dmfilestat)
    except:
        raise


@task
def archive_action(user, user_comment, dmfilestat):
    ''' Archive Action by wrapping invocation with celery task for sync / async execution'''
    try:
        backup_directory = dmfilestat.archivepath if dmfilestat.action_state == 'SA' else None
        dmactions.archive(user, user_comment, dmfilestat, backup_directory)
    except:
        raise


@task
def export_action(user, user_comment, dmfilestat):
    ''' Export Action by wrapping invocation with celery task for sync / async execution'''
    try:
        backup_directory = dmfilestat.archivepath if dmfilestat.action_state == 'SE' else None
        dmactions.export(user, user_comment, dmfilestat, backup_directory)
    except:
        raise

@task
def update_dmfilestats_diskspace(dmfilestat):
    ''' Task to update DMFileStat.diskspace '''
    dmactions.update_diskspace(dmfilestat)
