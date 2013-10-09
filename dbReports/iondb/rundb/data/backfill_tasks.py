#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import logging
from celery import task
from celery.utils.log import get_task_logger
from iondb.utils.files import getdiskusage


# Setup log file logging for backfill_exp_disusage
filename = '/var/log/ion/%s.log' % 'backfill_exp_diskusage'
log = logging.getLogger('backfill_exp_diskusage')
log.propagate = False
log.setLevel(logging.DEBUG)
handler = logging.handlers.RotatingFileHandler(
    filename, maxBytes=1024 * 1024 * 10, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)

'''
Main entry point to the celery task to backfill diskusage field of Experiment objects.
'''
@task(queue="periodic")
def backfill_exp_diskusage():
    '''
    For every Experiment object in database, scan filesystem and determine disk usage.
    Intended to be run at package installation to populate existing databases.
    '''
    from django.db.models import Q
    from iondb.rundb import models

    log.info("")
    log.info("===== New Run =====")

    log.info("EXPERIMENTS:")
    #query = Q(diskusage=None) | Q(diskusage=0)
    query = Q(diskusage=None)
    obj_list = models.Experiment.objects.filter(query).exclude(status="planned").values('pk','expName')
    #obj_list = models.Experiment.objects.values('pk','expName')
    old_way = False
    if old_way:
        for obj in obj_list:
            log.info(obj['expName'])
            try:
                setRunDiskspace(obj['pk'])
            except:
                log.exception(traceback.format_exc())
    else:
        setRunDiskspace_task.delay(list(obj_list))

'''
Recursive celery task to process a list of directories
'''
@task(queue="periodic")
def setRunDiskspace_task(obj_list):
    logger = get_task_logger('backfill_exp_diskusage')
    if len(obj_list) > 0:
        obj = obj_list.pop(0)
        logger.info(obj['expName'])
        try:
            setRunDiskspace(obj['pk'])
        except:
            logger.exception(traceback.format_exc())
        # recursion
        setRunDiskspace_task.delay(obj_list)
    else:
        # terminus recursivus
        logger.info("===== terminus recursivus =====")
        pass


@task(queue="periodic")
def setRunDiskspace(experimentpk):
    '''Sets diskusage field in Experiment record with data returned from du command'''
    logger = logging.getLogger(__name__)
    try:
        from iondb.rundb import models
        from django.core.exceptions import ObjectDoesNotExist
        # Get Experiment record
        exp = models.Experiment.objects.get(pk=experimentpk)
    except ObjectDoesNotExist:
        pass
    except:
        raise
    else:
        # Get filesystem location of given Experiment record
        directory = exp.expDir

        used = getdiskusage(directory)
        logger.info("%d" % used)

        # Update database entry for the given Experiment record
        try:
            used = used if used != None else "0"
            exp.diskusage = int(used)
            exp.save()
        except:
            # field does not exist, cannot update
            pass


@task(queue="periodic")
def backfill_result_diskusage():
    '''Due to error in initial code that filled in the diskusage field, this function
    updates every Result object's diskusage value.
    '''
    import traceback
    from django.db.models import Q
    from iondb.rundb import models

    # Setup log file logging
    filename = '/var/log/ion/%s.log' % 'backfill_result_diskusage'
    log = logging.getLogger('backfill_result_diskusage')
    log.propagate = False
    log.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(
        filename, maxBytes=1024 * 1024 * 10, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)

    log.info("")
    log.info("===== New Run =====")

    log.info("RESULTS:")
    #query = Q(diskusage=None) | Q(diskusage=0)
    query = Q(diskusage=None)
    obj_list = models.Results.objects.filter(query).values('pk','resultsName')
    #obj_list = models.Results.objects.values('pk','resultsName')
    for obj in obj_list:
        log.debug(obj['resultsName'])
        try:
            setResultDiskspace(obj['pk'])
        except:
            log.exception(traceback.format_exc())


@task(queue="periodic")
def setResultDiskspace(resultpk):
    '''Sets diskusage field in Results record with data returned from du command'''
    logger = logging.getLogger(__name__)
    try:
        from iondb.rundb import models
        from django.core.exceptions import ObjectDoesNotExist
        # Get Results record
        result = models.Results.objects.get(pk=resultpk)
    except ObjectDoesNotExist:
        logger.error("Object does not exist error")
        pass
    except:
        raise
    else:
        # Get filesystem location of given Results record
        directory = result.get_report_dir()

        used = getdiskusage(directory)
        logger.debug("%d" % used)
        # Update database entry for the given Experiment record
        try:
            used = used if used != None else "0"
            result.diskusage = int(used)
            result.save()
        except:
            # field does not exist, cannot update
            pass
