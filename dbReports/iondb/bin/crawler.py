#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Crawler
=======

The crawler is a service that monitors directories containing PGM experiment
data and creates database records for each new experiment it finds.

The crawler runs in a loop, and in each iteration performs the following tasks:

#. Generate a list of folders.
#. In each folder, look for a file named ``explog.txt``. If it exists, parse
   the file to obtain metadata about the experiment.
#. Compare this metadata to existing database records. If a match is found,
   then the experiment has already been added to the database. Otherwise,
   the experiment does not exist in the database, and a new record is created.

Throughout the loop, the crawler generates logging information as well as
status information that can be accessed over XMLRPC. The ``Status`` class
provides the XMLRPC access.

This module uses the Twisted XMLRPC server, which on Ubuntu can be installed
with ``sudo apt-get install python-twisted``.
"""
import datetime
import glob
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib
import copy

from iondb.bin.djangoinit import *
from django import db
from django.db import connection
from django.core.exceptions import ObjectDoesNotExist

from twisted.internet import reactor
from twisted.internet import task
from twisted.web import xmlrpc, server

from iondb.rundb import models
from iondb.rundb.data import backfill_tasks as tasks
from ion.utils.explogparser import load_log
from ion.utils.explogparser import parse_log
from iondb.utils.crawler_utils import getFlowOrder
from iondb.utils.crawler_utils import folder_mtime, explog_time
from iondb.utils.crawler_utils import tdelt2secs

__version__ = filter(str.isdigit, "$Revision: 75064 $")

LOG_BASENAME = "explog.txt"
LOG_FINAL_BASENAME = "explog_final.txt"


# TODO: these are defined in backup.py as well.  Needs to be consolidated.
# These strings will be displayed in Runs Page under FTP Status field
RUN_STATUS_COMPLETE = "Complete"
RUN_STATUS_MISSING = "Missing File(s)"
RUN_STATUS_ABORT = "User Aborted"
RUN_STATUS_SYS_CRIT = "Lost Chip Connection"

DO_THUMBNAIL = True


class CrawlLog(object):
    """``CrawlLog`` objects store and log metadata about the main crawl loop.
    Data logged by the ``CrawlLog`` includes the ten most recently saved runs
    to the database, and the crawler service's uptime.
    """
    MAX_EXPRS = 10
    BASE_LOG_NAME = "crawl.log"
    PRIVILEGED_BASE = "/var/log/ion"

    def __init__(self):
        self.expr_deque = []
        self.lock = threading.Lock()
        self.expr_count = 0
        self.start_time = None
        self.current_folder = '(none)'
        self.state = '(none)'
        self.state_time = datetime.datetime.now()
        # set up debug logging
        self.errors = logging.getLogger('crawler')
        self.errors.propagate = False
        self.errors.setLevel(logging.INFO)
        try:
            fname = os.path.join(self.PRIVILEGED_BASE, self.BASE_LOG_NAME)
            infile = open(fname, 'a')
            infile.close()
        except IOError:
            fname = self.BASE_LOG_NAME
        #rothandle = logging.handlers.RotatingFileHandler(fname, 'a', 65535)
        rothandle = logging.handlers.RotatingFileHandler(fname, maxBytes=1024 * 1024 * 10, backupCount=5)
        cachehandle = logging.handlers.MemoryHandler(1024, logging.ERROR, rothandle)
        #fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(lineno)d] ""%(message)s")
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        rothandle.setFormatter(fmt)
        self.errors.addHandler(rothandle)
        self.errors.addHandler(cachehandle)

    def start(self):
        """Register the start of the crawl service, in order to measure
        crawler uptime.
        """
        self.start_time = datetime.datetime.now()

    def time_elapsed(self):
        """Return the crawler's total uptime."""
        if self.start_time is not None:
            return datetime.datetime.now() - self.start_time
        else:
            return datetime.timedelta(0)

    def add_experiment(self, exp):
        """Add an experiment to the queue of experiments recently inserted
        into the database."""
        if isinstance(exp, models.Experiment):
            self.lock.acquire()
            self.expr_count += 1
            if len(self.expr_deque) >= self.MAX_EXPRS:
                self.expr_deque.pop(0)
            self.expr_deque.append(exp)
            self.lock.release()

    def prev_exprs(self):
        """Return a list of ten strings, which are the names of the last
        ten experiments inserted into the database."""
        self.lock.acquire()
        ret = list(self.expr_deque)
        self.lock.release()
        return ret

    def set_state(self, state_msg):
        """Set the crawler service's state message, for example,
        "working" or "sleeping."""
        self.lock.acquire()
        self.state = state_msg
        self.state_time = datetime.datetime.now()
        self.lock.release()

    def get_state(self):
        """Return a string containing the service's state message
        (see `set_state`)."""
        self.lock.acquire()
        ret = (self.state, self.state_time)
        self.lock.release()
        return ret



class Status(xmlrpc.XMLRPC):
    """The ``Status`` class provides access to a ``CrawlLog`` through
    the XMLRPC protocol."""
    def __init__(self, logger):
        xmlrpc.XMLRPC.__init__(self)
        self.logger = logger

    def xmlrpc_current_folder(self):
        return self.logger.current_folder

    def xmlrpc_time_elapsed(self):
        return tdelt2secs(self.logger.time_elapsed())

    def xmlrpc_prev_experiments(self):
        return map(str, self.logger.prev_exprs())

    def xmlrpc_experiments_found(self):
        return self.logger.expr_count

    def xmlrpc_state(self):
        msg, dt = self.logger.get_state()
        return (msg, tdelt2secs(datetime.datetime.now() - dt))

    def xmlrpc_hostname(self):
        return socket.gethostname()


def extract_rig(folder):
    """Given the name of a folder storing experiment data, return the name
    of the PGM from which the date came."""
    return os.path.basename(os.path.dirname(folder))


def extract_prefix(folder):
    """Given the name of a folder storing experiment data, return the
    name of the directory under which all PGMs at a given location
    store their data."""
    return os.path.dirname(os.path.dirname(folder))


def get_planned_exp_objects(d, folder, logobj):
    '''
    Find pre-created Plan, Experiment and ExperimentAnalysisSettings objects.
    If not found, create from system default templates.
    '''
    planObj = None
    expObj = None
    easObj = None

    expName = d.get("experiment_name", '')

    # Check if User selected Plan on instrument: explog contains guid or shortId
    selectedPlanGUId = d.get("planned_run_guid", '')

    planShortId = d.get("planned_run_short_id", '')
    if (planShortId is None or len(planShortId) == 0):
        planShortId = d.get("pending_run_short_id", '')

    logobj.debug("...planShortId=%s; selectedPlanGUId=%s" % (planShortId, selectedPlanGUId))

    # Find selected Plan
    if selectedPlanGUId:
        try:
            planObj = models.PlannedExperiment.objects.get(planGUID=selectedPlanGUId)
        except models.PlannedExperiment.DoesNotExist:
            logobj.warn("No plan with GUId %s found in database " % selectedPlanGUId)
        except models.PlannedExperiment.MultipleObjectsReturned:
            logobj.warn("Multiple plan with GUId %s found in database " % selectedPlanGUId)
    elif planShortId:
        # Warning: this is NOT guaranteed to be unique in db!
        try:
            planObj = models.PlannedExperiment.objects.filter(planShortID=planShortId, planExecuted=True).order_by("-date")[0]
        except IndexError:
            logobj.warn("No plan with short id %s and planExecuted=True found in database " % planShortId)

    if planObj:
        try:
            expObj = planObj.experiment
            easObj = expObj.get_EAS()

            # Make sure this plan is not already used by an existing experiment!
            if expObj.status == 'run':
                logobj.info('WARNING: Plan %s is already associated to an experiment, a copy will be created' % planObj.pk)
                planObj.pk = None
                planObj.save()

                expObj.pk = None
                expObj.unique = folder
                expObj.plan = planObj
                expObj.repResult = None
                expObj.save()

                easObj.pk = None
                easObj.experiment = expObj
                easObj.save()

            sampleSetObj = planObj.sampleSet
            if sampleSetObj:
                logobj.debug("crawler going to mark planObj.name=%s; sampleSet.id=%d as run" %(planObj.planDisplayedName, sampleSetObj.id))

                sampleSetObj.status = "run"
                sampleSetObj.save()


        except:
            logobj.warn(traceback.format_exc())
            logobj.warn("Error in trying to retrieve experiment and eas for planName=%s, pk=%s" % (planObj.planName, planObj.pk))
    else:
    #if user does not use a plan for the run, fetch the system default plan template, and clone it for this run
        logobj.warn("expName: %s not yet in database and needs a sys default plan" % expName)
        try:
            chipversion = d.get('chipversion','')
            if chipversion:
                explogChipType = chipversion
                if explogChipType.startswith('1.10'):
                    explogChipType = 'P1.1.17'
                elif explogChipType.startswith('1.20'):
                    explogChipType = 'P1.2.18'
            else:
                explogChipType = d.get('chiptype','')

            systemDefaultPlanTemplate = None

            if explogChipType:
                systemDefaultPlanTemplate = models.PlannedExperiment.get_latest_plan_or_template_by_chipType(explogChipType)

            if not systemDefaultPlanTemplate:
                logobj.debug("Chip-specific system default plan template not found in database for chip=%s; experiment=%s" % (explogChipType, expName))
                systemDefaultPlanTemplate = models.PlannedExperiment.get_latest_plan_or_template_by_chipType()

            logobj.debug("Use system default plan template=%s for chipType=%s in experiment=%s" % (systemDefaultPlanTemplate.planDisplayedName, explogChipType, expName))

            # copy Plan
            currentTime = datetime.datetime.now()

            planObj = copy.copy(systemDefaultPlanTemplate)
            planObj.pk = None
            planObj.planGUID = None
            planObj.planShortID = None
            planObj.isReusable = False
            planObj.isSystem = False
            planObj.isSystemDefault = False
            planObj.planName = "CopyOfSystemDefault_" + expName
            planObj.planDisplayedName = planObj.planName
            planObj.planStatus = 'run'
            planObj.planExecuted = True
            planObj.date = currentTime
            planObj.save()

            # copy Experiment
            expObj = copy.copy(systemDefaultPlanTemplate.experiment)
            expObj.pk = None
            expObj.unique = folder
            expObj.plan = planObj
            expObj.chipType = explogChipType
            expObj.date = currentTime
            expObj.save()

            # copy EAS
            easObj = systemDefaultPlanTemplate.experiment.get_EAS()
            easObj.pk = None
            easObj.experiment = expObj
            easObj.isEditable = True
            easObj.date = currentTime
            easObj.save()

            logobj.debug("cloned systemDefaultPlanTemplate: planObj.pk=%s; easObj.pk=%s; expObj.pk=%s" % (planObj.pk, easObj.pk, expObj.pk))

            #clone the qc thresholds as well
            qcValues = systemDefaultPlanTemplate.plannedexperimentqc_set.all()

            for qcValue in qcValues:
                qcObj = copy.copy(qcValue)

                qcObj.pk = None
                qcObj.plannedExperiment = planObj
                qcObj.save()

            logobj.info("crawler (using template=%s) AFTER SAVING SYSTEM DEFAULT CLONE %s for experiment=%s;" % (systemDefaultPlanTemplate.planDisplayedName, planObj.planName, expName))

        except:
            logobj.warn(traceback.format_exc())
            logobj.warn("Error in trying to clone system default plan template for experiment=%s" % (expName))

    return planObj, expObj, easObj

def exp_kwargs(d, folder, logobj):
    """Converts the output of `parse_log` to the a dictionary of
    keyword arguments needed to create an ``Experiment`` database object.
    """
    identical_fields = ("sample", "cycles", "flows", "project",)
    simple_maps = (
        ("experiment_name", "expName"),
        ("chipbarcode", "chipBarcode"),
        ("user_notes", "notes"),
        ##("seqbarcode", "seqKitBarcode"),
        ("autoanalyze", "autoAnalyze"),
        ("prebeadfind", "usePreBeadfind"),
        ("librarykeysequence", "libraryKey"),
        ("barcodeid", "barcodeKitName"),
        ("isReverseRun", "isReverseRun"),
        ("library", "reference"),
    )

    chiptype = d.get('chiptype','')
    chipversion = d.get('chipversion','')
    if chipversion:
        chiptype = chipversion
    if chiptype.startswith('1.10'):
        chiptype = 'P1.1.17'
    elif chiptype.startswith('1.20'):
        chiptype = 'P1.2.18'

    full_maps = (
        ("chipType", chiptype),
        ("pgmName", d.get('devicename', extract_rig(folder))),
        ("log", json.dumps(d, indent=4)),
        ("expDir", folder),
        ("unique", folder),
        ("baselineRun", d.get("runtype") == "STD" or d.get("runtype") == "Standard"),
        ("date", explog_time(d.get("start_time", ""), folder)),
        ("storage_options", models.GlobalConfig.objects.all()[0].default_storage_options),
        ("flowsInOrder", getFlowOrder(d.get("image_map", ""))),
        ("reverse_primer", d.get('reverse_primer', 'Ion Kit')),
    )

    derive_attribute_list = ["sequencekitname", "seqKitBarcode", "sequencekitbarcode",
                             "libraryKitName", "libraryKitBarcode"]

    ret = {}
    for f in identical_fields:
        ret[f] = d.get(f, '')
    for k1, k2 in simple_maps:
        ret[k2] = d.get(k1, '')
    for k, v in full_maps:
        ret[k] = v

    for attribute in derive_attribute_list:
        ret[attribute] = ''

    #N.B. this field is not used
    ret['storageHost'] = 'localhost'

    # If Flows keyword is defined in explog.txt...
    if ret['flows'] != "":
        # Cycles should be based on number of flows, not cycles published in log file
        # (Use of Cycles is deprecated in any case! We should be able to enter a random number here)
        ret['cycles'] = int(int(ret['flows']) / len(ret['flowsInOrder']))
    else:
        # ...if Flows is not defined in explog.txt:  (Very-old-dataset support)
        ret['flows'] = len(ret['flowsInOrder']) * int(ret['cycles'])
        logobj.warn("Flows keyword missing: Calculated Flows is %d" % int(ret['flows']))

    if ret['barcodeKitName'].lower() == 'none':
        ret['barcodeKitName'] = ''

    if len(d.get('blocks', [])) > 0:
        ret['rawdatastyle'] = 'tiled'
        ret['autoAnalyze'] = False
        for bs in d['blocks']:
            # Hack alert.  Watch how explogparser.parse_log munges these strings when detecting which one is the thumbnail entry
            # Only thumbnail will have 0,0 as first and second element of the string.
            if '0' in bs.split(',')[0] and '0' in bs.split(',')[1]:
                continue
            if auto_analyze_block(bs, logobj):
                ret['autoAnalyze'] = True
                logobj.debug("Block Run. Detected at least one block to auto-run analysis")
                break
        if ret['autoAnalyze'] is False:
            logobj.debug("Block Run. auto-run whole chip has not been specified")
    else:
        ret['rawdatastyle'] = 'single'

    sequencingKitName = d.get("seqkitname", '')
    #do not replace plan's seqKit info if explog has blank seqkitname
    if sequencingKitName and sequencingKitName != "NOT_SCANNED":
        ret['sequencekitname'] = sequencingKitName

    #in rundb_experiment, there are 2 attributes for sequencingKitBarcode!!
    sequencingKitBarcode = d.get("seqkitpart", '')
    if sequencingKitBarcode and sequencingKitBarcode != "NOT_SCANNED":
        ret['seqKitBarcode'] = sequencingKitBarcode
        ret['sequencekitbarcode'] = sequencingKitBarcode

    libraryKitBarcode = d.get("libbarcode", '')
    if libraryKitBarcode and libraryKitBarcode != "NOT_SCANNED":
        ret['libraryKitBarcode'] = libraryKitBarcode

    libraryKitName = d.get('libkit', '')
    if libraryKitName and libraryKitName != "NOT_SCANNED":
        ret['libraryKitName'] = libraryKitName

    ##note: if PGM is running the old version, there is no isReverseRun in explog.txt.
    isReverseRun = d.get("isreverserun", '')
    if isReverseRun == "Yes":
        ret['isReverseRun'] = True
    else:
        ret['isReverseRun'] = False

    #instrument could have blank runType or be absent all together in explog
    runType = d.get('runtype', "")
    if not runType:
        runType = "GENS"
    ret['runType'] = runType

    # Limit input sizes to defined field widths in models.py
    ret['notes'] = ret['notes'][:1024]
    ret['expDir'] = ret['expDir'][:512]
    ret['expName'] = ret['expName'][:128]
    ret['pgmName'] = ret['pgmName'][:64]
    ret['unique'] = ret['unique'][:512]
    ret['storage_options'] = ret['storage_options'][:200]
    ret['project'] = ret['project'][:64]
    ret['sample'] = ret['sample'][:64]
    ret['reference'] = ret['reference'][:64]
    ret['chipBarcode'] = ret['chipBarcode'][:64]
    ret['seqKitBarcode'] = ret['seqKitBarcode'][:64]
    ret['chipType'] = ret['chipType'][:32]
    ret['flowsInOrder'] = ret['flowsInOrder'][:512]
    ret['libraryKey'] = ret['libraryKey'][:64]
    ret['barcodeKitName'] = ret['barcodeKitName'][:128]
    ret['reverse_primer'] = ret['reverse_primer'][:128]
    ret['sequencekitname'] = ret['sequencekitname'][:512]
    ret['sequencekitbarcode'] = ret['sequencekitbarcode'][:512]
    ret['libraryKitName'] = ret['libraryKitName'][:512]
    ret['libraryKitBarcode'] = ret['libraryKitBarcode'][:512]
    ret['runType'] = ret['runType'][:512]

    return ret


def update_exp_objects_from_log(d, folder, planObj, expObj, easObj, logobj):
    """ Update plan, experiment, sample and experimentAnalysisSettings
        Returns the experiment object
    """

    if planObj:
        logobj.info("Going to update db objects for plan=%s" % planObj.planName)

    kwargs = exp_kwargs(d, folder, logobj)

    # PairedEnd runs no longer supported
    if kwargs['isReverseRun']:
        logobj.warn("PAIRED-END is NO LONGER SUPPORTED. Skipping experiment %s" % expObj.expName)
        return None

    # get valid libraryKey (explog or entered during planning), if failed use default value
    libraryKey = kwargs['libraryKey']
    if libraryKey and models.LibraryKey.objects.filter(sequence=libraryKey, direction='Forward').exists():
        pass
    elif easObj.libraryKey and models.LibraryKey.objects.filter(sequence=easObj.libraryKey, direction='Forward').exists():
        kwargs['libraryKey'] = easObj.libraryKey
    else:
        try:
            defaultForwardLibraryKey = models.LibraryKey.objects.get(direction='Forward', isDefault=True)
            kwargs['libraryKey'] = defaultForwardLibraryKey.sequence
        except models.LibraryKey.DoesNotExist:
            logobj.warn("No default forward library key in database for experiment %s" % expObj.expName)
            return None
        except models.LibraryKey.MultipleObjectsReturned:
            logobj.warn("Multiple default forward library keys found in database for experiment %s" % expObj.expName)
            return None

    #20121104-PDD-TODO: change .info to .debug or remove the print line all together

    # *** Update Experiment ***
    expObj.status = 'run'
    expObj.runMode = planObj.runMode
    for key,value in kwargs.items():
        if key == "sequencekitname":
            if value:
                setattr(expObj, key, value)
            else:
                logobj.debug("crawler.update_exp_objects_from_log() SKIPPED key=%s; value=%s" %(key, value))
        else:
            setattr(expObj, key, value)

    expObj.save()
    logobj.info("Updated experiment=%s, pk=%s, expDir=%s" % (expObj.expName, expObj.pk, expObj.expDir) )


    # *** Update Plan ***
    plan_keys = ['expName', 'runType', 'isReverseRun']
    for key in plan_keys:
        setattr(planObj, key, kwargs[key])

    planObj.planStatus = 'run'
    planObj.planExecuted = True # this should've been done by instrument already

    # add project from instrument, if any. Note: this will not create a project if doesn't exist
    projectName = kwargs['project']
    if projectName and projectName not in planObj.projects.values_list('name', flat=True):
        try:
            planObj.projects.add(models.Project.objects.get(name=projectName))
        except:
            logobj.warn("Couldn't add project %s to %s plan: project does not exist" % (projectName, planObj.planName))

    planObj.save()
    logobj.info("Updated plan=%s, pk=%s" % (planObj.planName, planObj.pk))


    # *** Update ExperimentAnalysisSettings ***
    eas_keys = ['barcodeKitName', 'reference', 'libraryKey']

    for key in eas_keys:
        setattr(easObj, key, kwargs[key])

    #do not replace plan's EAS value if explog does not have a value for it
    eas_keys = ['libraryKitName', 'libraryKitBarcode']
    for key in eas_keys:
        if (key in kwargs) and kwargs[key]:
            setattr(easObj, key, kwargs[key])

    easObj.status = 'run'
    easObj.save()
    logobj.info("Updated EAS=%s" % easObj)

    # Refresh default cmdline args - this is needed in case chip type or kits changed from their planned values
    default_args = planObj.get_default_cmdline_args()
    for key,value in default_args.items():
        setattr(easObj, key, value)
    easObj.save()

    # *** Update samples associated with experiment***
    sampleCount = expObj.samples.all().count()
    sampleName = kwargs['sample']

    #if this is a barcoded run, user can't change any sample names on the instrument, only sample status update is needed
    if sampleName and not easObj.barcodeKitName:
        need_new_sample = False
        if sampleCount == 1:
            sample = expObj.samples.all()[0]
            # change existing sample name only if this sample has association to 1 experiment,
            # otherwise need to dissociate the original sample and add a new one
            if sample.name != sampleName:
                sample_found = models.Sample.objects.filter(name = sampleName)
                if sample_found:
                   sample.experiments.remove(expObj)
                   sample_found[0].experiments.add(expObj)
                   logobj.info("Replaced sample=%s; sample.id=%s" %(sampleName, sample_found[0].pk))
                elif sample.experiments.count() == 1:
                   sample.name = sampleName
                   sample.displayedName = sampleName
                   sample.save()
                   logobj.info("Updated sample=%s; sample.id=%s" %(sampleName, sample.pk))
                else:
                    sample.experiments.remove(expObj)
                    need_new_sample = True

        if sampleCount == 0 or need_new_sample:
            sample_kwargs = {
                            'name' : sampleName,
                            'displayedName' : sampleName,
                            'date' : expObj.date,
                            }
            try:
                sample, created = models.Sample.objects.get_or_create(name=sampleName, defaults=sample_kwargs)
                sample.experiments.add(expObj)
                sample.save()
                logobj.info("Added sample=%s; sample.id=%s" %(sampleName, sample.pk))
            except:
                logobj.debug("Failed to add sample=%s to experiment=%s" %(sampleName, expObj.expName))
                logobj.debug(traceback.format_exc())

    # update status for all samples
    for sample in expObj.samples.all():
        sample.status = expObj.status
        sample.save()

    return expObj


def construct_crawl_directories(logger):
    """Query the database and build a list of directories to crawl.
    Returns an array.
    For every Rig in the database, construct a filesystem path and
    get all subdirectories in that path."""
    def dbase_isComplete(name, explist):
        '''
        explist consists of tuples containing experiment name and ftpstatus
        '''
        for item in explist:
            if name == item[0]:
                if item[1] in [RUN_STATUS_COMPLETE,RUN_STATUS_ABORT,RUN_STATUS_MISSING,RUN_STATUS_SYS_CRIT]:
                    return True
        return False

    fserves = models.FileServer.objects.all()
    ret = []
    for fs in fserves:
        l = fs.location
        rigs = models.Rig.objects.filter(location=l)
        for r in rigs:
            rig_folder = os.path.join(fs.filesPrefix, r.name)
            if os.path.exists(rig_folder):
                logger.errors.debug("Checking %s" % rig_folder)
                exp_val_list = models.Experiment.objects.filter(expDir__startswith=rig_folder).values_list('expDir', 'ftpStatus')
                try:
                    subdir_bases = os.listdir(rig_folder)
                    # create array of paths for all directories in Rig's directory
                    s1 = [os.path.join(rig_folder, subd) for subd in subdir_bases]
                    s2 = [subd for subd in s1 if os.path.isdir(subd)]
                    # create array of paths of not complete ftp transfer only
                    s3 = [subd for subd in s2 if dbase_isComplete(subd, exp_val_list) is False]
                    ret.extend(s3)
                except:
                    logger.errors.error(traceback.format_exc())
                    logger.set_state('error')
    return ret


def get_filecount(exp):
    if 'tiled' in exp.rawdatastyle:
        #N.B. Hack - we check ftp status of thumbnail data only
        expDir = os.path.join(exp.expDir, 'thumbnail')
    else:
        expDir = exp.expDir

    file_count = len(glob.glob(os.path.join(expDir, "acq*.dat")))
    return file_count


def sleep_delay(start, current, delay):
    """Sleep until delay seconds past start, based on the current time
    as specified by ``current``."""
    diff = current - start
    s = delay - tdelt2secs(diff)
    if s > 0:
        time.sleep(s)


def get_name_from_json(exp, key, thumbnail_analysis):
    data = exp.log
    name = data.get(key, False)
    twig = ''
    if thumbnail_analysis:
        twig = '_tn'
    # also ignore name if it has the string value "None"
    if not name or name == "None":
        return 'Auto_%s_%s%s' % (exp.pretty_print(), exp.pk, twig)
    else:
        return '%s_%s%s' % (str(name), exp.pk, twig)


def generate_http_post(exp, logger, thumbnail_analysis=False):
    try:
        GC = models.GlobalConfig.objects.all().order_by('pk')[0]
        base_recalibrate = GC.base_recalibrate
        mark_duplicates = GC.mark_duplicates
        realign = GC.realign
    except models.GlobalConfig.DoesNotExist:
        base_recalibrate = False
        mark_duplicates = False
        realign = False

    #instead of relying solely on globalConfig, user can now set isDuplicateReads for the experiment
    eas = exp.get_EAS()
    if (eas):
        ##logger.errors.info("crawler.generate_http_post() exp.name=%s; id=%s; isDuplicateReads=%s" %(exp.expName, str(exp.pk), str(eas.isDuplicateReads)))
        mark_duplicates = eas.isDuplicateReads

    report_name = get_name_from_json(exp, 'autoanalysisname', thumbnail_analysis)

    params = urllib.urlencode({'report_name': report_name,
                               'tf_config': '',
                               'path': exp.expDir,
                               'submit': ['Start Analysis'],
                               'do_thumbnail': "%r" % thumbnail_analysis,
                               'do_base_recal': base_recalibrate,
                               'realign': realign,
                               'mark_duplicates': mark_duplicates
    })

    status_msg = "Generated POST"
    try:
        connection_url = 'http://127.0.0.1/report/analyze/%s/0/' % (exp.pk)
        f = urllib.urlopen(connection_url, params)
    except IOError:
        logger.errors.debug('could not make connection %s' % connection_url)
        try:
            connection_url = 'https://127.0.0.1/report/analyze/%s/0/' % (exp.pk)
            f = urllib.urlopen(connection_url, params)
        except IOError:
            logger.errors.error(" !! Failed to start analysis.  could not connect to %s" % connection_url)
            status_msg = "Failure to generate POST"
            f = None

    if f:
        error_code = f.getcode()
        if error_code is not 200:
            logger.errors.error(" !! Failed to start analysis. URL failed with error code %d for %s" % (error_code, f.geturl()))
            status_msg = "Failure to generate POST"

    return status_msg


def check_for_autoanalyze(exp):
    if 'tiled' in exp.rawdatastyle:
        return True
    else:
        return False


def usedPostBeadfind(exp):
    def convert_to_bool(value):
        '''We must convert the string to a python bool value'''
        v = value.lower()
        if v == 'yes':
            return True
        else:
            return False

    data = exp.log
    ret = False
    if 'postbeadfind' in data:
        ret = convert_to_bool(data['postbeadfind'])
    return ret


def check_for_file(expDir, filelookup):
    """Check if ``filelookup`` has been FTP'd before adding the
    experiment to the database."""
    return os.path.exists(os.path.join(expDir, filelookup))


def check_for_abort(expDir, filelookup):
    """Check if run has been aborted by the user"""
    # parse explog_final.txt
    try:
        f = open(os.path.join(expDir, filelookup), 'r')
    except:
        print "Error opening file: ", os.path.join(expDir, filelookup)
        return False

    #search for the line we need
    line = f.readline()
    while line:
        if "WARNINGS:" in line:
            if "Critical: Aborted" in line:
                f.close()
                return True
        line = f.readline()

    f.close()
    return False


def check_for_critical(exp, filelookup):
    """Check if run has been aborted by the PGM software"""
    # parse explog_final.txt
    try:
        f = open(os.path.join(exp.expDir, filelookup), 'r')
    except:
        print "Error opening file: ", os.path.join(exp.expDir, filelookup)
        return False

    #search for the line we need
    line = f.readline()
    while line:
        if "WARNINGS:" in line:
            if "Critical: Lost Chip Connection" in line:
                exp.ftpStatus = RUN_STATUS_SYS_CRIT
                exp.save()
                f.close()
                return True
            elif "Critical: Aborted" in line:
                exp.ftpStatus = RUN_STATUS_ABORT
                exp.save()
                f.close()
                return True
        line = f.readline()

    f.close()
    return False


def get_flow_per_cycle(exp):
    data = exp.log
    im = 'image_map'
    if im in data:
        # If there are no space characters in the string, its the 'new' format
        if data[im].strip().find(' ') == -1:
            # when Image Map is "Image Map: tacgtacgtctgagcatcgatcgatgtacagc"
            flows = len(data[im])
        else:
            # when Image Map is "Image Map: 4 0 r4 1 r3 2 r2 3 r1" or "4 0 T 1 A 2 C 3 G"
            flows = int(data[im].split(' ')[0])
        return flows
    else:
        return 4  # default to 4 if we have no other information


def get_last_file(exp):
    last = exp.flows - 1
    pre = '0000'
    final = pre + str(last)
    file_num = final[-4::]
    filename = 'acq_%s.dat' % file_num
    return filename


def check_for_completion(exp):
    """Check if a the ftp is really complete or if we need to wait"""

    file_list = []
    file_list.append(get_last_file(exp))
    if exp.usePreBeadfind:
        file_list.append('beadfind_pre_0003.dat')
    if usedPostBeadfind(exp):
        file_list.append('beadfind_post_0003.dat')

    if check_for_file(exp.expDir, 'explog_final.txt'):
        #check for critical errors
        if check_for_critical(exp, 'explog_final.txt'):
            return True  # run is complete; true

        if 'tiled' in exp.rawdatastyle:
            #N.B. Hack - we check status of thumbnail data only
            expDir = os.path.join(exp.expDir, 'thumbnail')
        else:
            expDir = exp.expDir

        #check for required files
        for filename in file_list:
            if not check_for_file(expDir, filename):
                exp.ftpStatus = RUN_STATUS_MISSING
                exp.save()
                return False

        return True
    else:
        return False


def analyze_early_block(blockstatus, logger):
    '''blockstatus is the string from explog.txt starting with keyword BlockStatus.
    Evaluates the AnalyzeEarly flag.
    Returns boolean True when argument is 1'''
    try:
        arg = blockstatus.split(',')[5].strip()
        flag = int(arg.split(':')[1]) == 1
    except:
        logger.errors.error(traceback.format_exc())
        flag = False
    return flag


def auto_analyze_block(blockstatus, logobj):
    '''blockstatus is the string from explog.txt starting with keyword BlockStatus.
    Evaluates the AutoAnalyze flag.
    Returns boolean True when argument is 1'''
    try:
        arg = blockstatus.split(',')[4].strip()
        flag = int(arg.split(':')[1]) == 1
    except:
        logobj.error(traceback.format_exc())
        flag = False
    return flag


def get_block_subdir(blockdata, logger):
    '''blockstatus is the string from explog.txt starting with keyword BlockStatus.
    Evaluates the X and Y arguments.
    Returns string containing rawdata subdirectory'''
    try:
        blockx = blockdata.split(',')[0].strip()
        blocky = blockdata.split(',')[1].strip()
        subdir = "%s_%s" % (blockx, blocky)
    except:
        logger.errors.error(traceback.format_exc())
        subdir = ''
    return subdir


def ready_to_process(exp):
    # Rules
    # analyzeearly flag true
    # acq_0000.dat file exists for every analysis job
    # -OR-
    # oninst flag is true (and ignore analyzeearly)

    readyToStart = False
    data = exp.log

    # Process block raw data
    if 'tiled' in exp.rawdatastyle:
        readyToStart = True

    # Process single image dataset
    else:
        if 'analyzeearly' in data:
            readyToStart = data.get('analyzeearly')

    return readyToStart


def ready_to_process_thumbnail(exp):

    if 'tiled' in exp.rawdatastyle:
        if os.path.exists(os.path.join(exp.expDir, 'thumbnail')):
            return True

    return False


def autorun_thumbnail(exp, logger):

    if 'tiled' in exp.rawdatastyle:
        #support for old format
        for bs in exp.log['blocks']:
            if bs.find('thumbnail') > 0:
                arg = bs.split(',')[4].strip()
                if int(arg.split(':')[1]) == 1:
                    #logger.errors.info ("Request to auto-run thumbnail analysis")
                    return True
        #support new format
        try:
            thumb = exp.log['thumbnail_000']
            logger.errors.debug("THUMB: %s" % thumb)
            if thumb.find('AutoAnalyze:1') > 0:
                return True
        except:
            pass
    return False


def crawl(folders, logger):
    """Crawl over ``folders``, reporting information to the ``CrawlLog``
    ``logger``."""
    if folders:
        logger.errors.info("checking %d directories" % len(folders))

    for f in folders:
        try:
            text = load_log(f, LOG_BASENAME)
            logger.errors.debug("checking directory %s" % f)
            #------------------------------------
            # Check for a valid explog.txt file
            #------------------------------------
            if text is not None:

                logger.current_folder = f
                # Catch any errors parsing this log file and continue to the next experiment.
                try:
                        d = parse_log(text)
                except:
                	logger.errors.info("Error parsing explog, skipping %s" % f)
                	logger.errors.exception(traceback.format_exc())
                	continue

                exp = None
                exp_set = models.Experiment.objects.filter(unique=f)
                if exp_set:
                    # Experiment already in database
                    exp = exp_set[0]
                    if exp.ftpStatus == RUN_STATUS_COMPLETE:
                        continue
                else:
                    # Need to create experiment for this folder
                    try:
                        planObj, expObj, easObj = get_planned_exp_objects(d, f, logger.errors)
                        exp = update_exp_objects_from_log(d, f, planObj, expObj, easObj, logger.errors)
                    except:
                        logger.errors.info("Error updating experiment objects, skipping %s" % f)
                        logger.errors.exception(traceback.format_exc())
                        continue

                if exp is not None:

                    logger.errors.info("%s" % f)

                    #--------------------------------
                    # Update FTP Transfer Status
                    #--------------------------------
                    if check_for_completion(exp):
                        exptxt = load_log(f, LOG_FINAL_BASENAME)
                        if exptxt is not None:
                            exp.log = json.dumps(parse_log(exptxt), indent=4)
                        else:
                            # explog_final exists, but is not readable yet.
                            continue

                        # FTP transfer is complete
                        if exp.ftpStatus == RUN_STATUS_ABORT or exp.ftpStatus == RUN_STATUS_SYS_CRIT:
                            logger.errors.info("FTP status: Aborted")
                        else:
                            exp.ftpStatus = RUN_STATUS_COMPLETE
                        exp.save()

                        # Measure and record disk space usage of this dataset
                        # This is celery task which will execute in separate worker thread
                        try:
                            tasks.setRunDiskspace.delay(exp.pk)
                        except:
                            logger.errors.exception(traceback.format_exc())

                    else:
                        if exp.ftpStatus != RUN_STATUS_MISSING:
                            exp.ftpStatus = get_filecount(exp)
                            exp.save()
                            logger.errors.info("FTP status: Transferring")

                    #--------------------------------
                    # Handle auto-run analysis
                    #--------------------------------
                    composite_exists, thumbnail_exists = reports_exist(exp)
                    logger.errors.debug("Reports Exist? %s %s" % (composite_exists, thumbnail_exists))
                    if not composite_exists:
                        logger.errors.debug("No auto-run analysis has been started")
                        # Check if auto-run for whole chip has been requested
                        if exp.autoAnalyze:
                            logger.errors.debug("  auto-run whole chip analysis has been requested")
                            #if check_analyze_early(exp) or check_for_completion(exp):
                            if ready_to_process(exp) or check_for_completion(exp):
                                logger.errors.info("  Start a whole chip auto-run analysis job")
                                generate_http_post(exp, logger)
                            else:
                                logger.errors.info("  Do not start a whole chip auto-run job yet")
                        else:
                            logger.errors.debug("  auto-run whole chip analysis has not been requested")
                    else:
                        logger.errors.debug("composite report already exists")

                    if not thumbnail_exists:
                        # Check if auto-run for thumbnail has been requested
                        # But only if its a block dataset
                        if 'tiled' in exp.rawdatastyle:
                            if autorun_thumbnail(exp, logger):
                                logger.errors.debug("auto-run thumbnail analysis has been requested")
                                if ready_to_process_thumbnail(exp) or check_for_completion(exp):
                                    logger.errors.info("  Start a thumbnail auto-run analysis job")
                                    generate_http_post(exp, logger, DO_THUMBNAIL)
                                else:
                                    logger.errors.info("  Do not start a thumbnail auto-run job yet")
                            else:
                                logger.errors.debug("auto-run thumbnail analysis has not been requested")
                        else:
                            logger.errors.debug("This is not a block dataset; no thumbnail to process")
                    else:
                        logger.errors.debug("thumbnail report already exists")
                else:
                    logger.errors.debug("exp is empty")
            else:
                # No explog.txt exists; probably an engineering test run
                logger.errors.debug("no %s was found in %s" % (LOG_BASENAME, f))
        except:
            logger.errors.exception(traceback.format_exc())

    logger.current_folder = '(none)'


def loop(logger, end_event, delay):
    """Outer loop of the crawl thread, calls ``crawl`` every minute."""
    logger.start()
    while not end_event.isSet():
        connection.close()  # Close any db connection to force new one.
        try:
            logger.set_state('working')
            start = datetime.datetime.now()
            folders = construct_crawl_directories(logger)
            crawl(folders, logger)
            logger.set_state('sleeping')

        except KeyboardInterrupt:
            end_event.set()
        except:
            logger.errors.error(traceback.format_exc())
            logger.set_state('error')
        sleep_delay(start, datetime.datetime.now(), delay)
        db.reset_queries()
    sys.exit(0)


def reports_exist(exp):
    '''Determine whether an auto-analysis report has been created'''
    composite = False
    thumbnail = False
    reports = exp.sorted_results()
    if reports:
        # thumbnail report will have thumb=1 in meta data
        for report in reports:
            val = report.metaData.get('thumb', 0)
            if val == 1:
                thumbnail = True
            if val == 0:
                composite = True

    return composite, thumbnail


def checkThread(thread, log, reactor):
    '''Checks thread for aliveness.
    If a valid reactor object is passed, the reactor will be stopped
    thus stopping the daemon entirely.'''
    thread.join(1)
    if not thread.is_alive():
        log.errors.critical("loop thread is dead")
        if reactor.running:
            log.errors.critical("ionCrawler daemon is exiting")
            reactor.stop()


def main(argv):
    logger = CrawlLog()
    logger.errors.info("Crawler Initializing")

    exit_event = threading.Event()
    loopfunc = lambda: loop(logger, exit_event, settings.CRAWLER_PERIOD)
    lthread = threading.Thread(target=loopfunc)
    lthread.setDaemon(True)
    lthread.start()
    logger.errors.info("ionCrawler Started Ver: %s" % __version__)

    # check thread health periodically and exit if thread is dead
    # pass in reactor object below to kill process when thread dies
    l = task.LoopingCall(checkThread, lthread, logger, reactor)
    l.start(30.0)  # call every 30 second

    # start the xml-rpc server
    r = Status(logger)
    reactor.listenTCP(settings.CRAWLER_PORT, server.Site(r))
    reactor.run()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
