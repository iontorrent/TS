# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.conf import settings
from celery.task import task
import xmlrpclib
import traceback

from django.core import urlresolvers
from iondb.rundb.models import Message, Results
from iondb.backup import ion_archiveResult


def _store_message(user, pk, status, action):
    url = urlresolvers.reverse('report_metadata_log', kwargs={'pkR': pk})
    report = Results.objects.get(id=pk)
    token = 'complete' if status else 'failed'
    msg = "Report (%s) %s %s. <a href='%s'  data-toggle='modal' data-target='#modal_report_log'>View Report Log</a>" % (report.resultsName, action, token, url)
    func = Message.success if status else Message.error
    return func(msg, route=user)


def _invoke_rpc(rpc, logger, pk, comment):
    status = False
    try:
        proxy = xmlrpclib.ServerProxy('http://127.0.0.1:%d' % settings.IARCHIVE_PORT, allow_none=True)
        methodToCall = getattr(proxy, rpc)
        status = methodToCall(pk, comment)
    except:
        logger.exception(traceback.format_exc())
    return status


@task
def archive_report(user, pk, comment):
    ''' Archive Report by wrapping invocation with celerytask for sync / async execution'''  
    logger = archive_report.get_logger()
    status = _invoke_rpc('archive_report', logger, pk, comment)
    message = _store_message(user, pk, status, 'archival')
    return status, message


@task
def export_report(user, pk, comment):
    ''' Export Report by wrapping invocation with celery task for sync / async execution'''
    logger = export_report.get_logger()
    status = _invoke_rpc('export_report', logger, pk, comment)
    message = _store_message(user, pk, status, 'export')
    return status, message


@task
def prune_report(user, pk, comment):
    ''' Prune Report by wrapping invocation with celery task for sync / async execution'''
    logger = prune_report.get_logger()
    status = _invoke_rpc('prune_report', logger, pk, comment)
    message = _store_message(user, pk, status, 'prune')
    return status, message

@task
def sync_filesystem_and_db_report_state():
    logger = sync_filesystem_and_db_report_state.get_logger()
    logger.info("Scanning reports on filesystem to ensure database state is consistent (bug TS-5486).")
    reports = Results.objects.exclude(reportStatus__in=ion_archiveResult.STATUS)
    for report in reports:
        logger.info('examining report %s ' % report)
        if report.is_archived():
            report.reportStatus = ion_archiveResult.ARCHIVED
            report.save()
            logger.info('marking report %s as ARCHIVED' % report)

