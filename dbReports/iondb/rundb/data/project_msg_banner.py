#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from __future__ import absolute_import
import sys
import traceback
from celery.utils.log import get_task_logger
from django.core.urlresolvers import reverse
from iondb.rundb.models import Message, Results
from iondb.rundb.data.dm_utils import slugify

logger = get_task_logger('data_management')
d = {'logid': "%s" % ('proj_msg_banner')}


def project_msg_banner(user, project_msg, action):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=d)
    try:
        msg = ''
        thistag = ''
        grpstatus = ''
        logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=d)
        for pk, status_list in project_msg.iteritems():
            report = Results.objects.get(id=pk)
            msg += "(%s) %s " % (report.resultsName, action.title())
            for category, status in status_list.iteritems():
                msg += " %s, " % str(category)
                msg += status
                grpstatus = status
                thistag = "%s_%s_%s" % (str(pk), action, slugify(category))
            try:
                # urlresolvers is currently throwing an exception, unknown why.  TS-
                # url = reverse('dm_log', args=(pk,))
                url = '/data/datamanagement/log/%s/' % pk
                msg += " <a href='%s'  data-toggle='modal' data-target='#modal_report_log'>View Report Log</a></br>" % (
                    url)
            except:
                logger.error(traceback.format_exc(), extra=d)
            logger.debug("MESSAGE: %s" % msg, extra=d)

        # If message already exists (ie, scheduled task) delete it.
        Message.objects.filter(tags=thistag).delete()

        if len(project_msg) > 1:
            thistag = "%s_%s_%s" % ('project', action, slugify(category))

        if grpstatus.lower() == 'scheduled':
            func = Message.info
        elif grpstatus.lower() == 'success':
            func = Message.success
        else:
            func = Message.error
    except:
        func = Message.error
        logger.error(traceback.format_exc(), extra=d)
    return func(msg, route=user, tags=thistag)
