#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import sys
import traceback
from celery.utils.log import get_task_logger
from django.core import urlresolvers
from iondb.rundb.models import Message, Results

logger = get_task_logger('data_management')

def project_msg_banner(user, project_msg, action):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
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

def slugify(something):
    '''convert whitespace to hyphen and lower case everything'''
    import re
    return re.sub(r'\W+','-',something.lower())