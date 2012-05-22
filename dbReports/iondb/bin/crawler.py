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
from logging import handlers as loghandlers
import os
from os import path
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from djangoinit import *

sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django import db

from twisted.internet import reactor
from twisted.internet import task
from twisted.web import xmlrpc,server

from iondb.rundb import models
from django.db import connection
    
import logregexp as lre
import urllib
import string

__version__ = filter(str.isdigit, "$Revision: 30068 $")

LOG_BASENAME = "explog.txt"
LOG_FINAL_BASENAME = "explog_final.txt"
ENTRY_RE = re.compile(r'^(?P<name>[^:]+)[:](?P<value>.*)$')
CLEAN_RE = re.compile(r'\s|[/]+')
TIMESTAMP_RE = models.Experiment.PRETTY_PRINT_RE


# TODO: these are defined in backup.py as well.  Needs to be consolidated.
# These strings will be displayed in Runs Page under FTP Status field
RUN_STATUS_COMPLETE = "Complete"
RUN_STATUS_MISSING  = "Missing File(s)"
RUN_STATUS_ABORT    = "User Aborted"
RUN_STATUS_SYS_CRIT = "Lost Chip Connection"

DO_THUMBNAIL=True

def tdelt2secs(td):
    """Convert a ``datetime.timedelta`` object into a floating point
    number of seconds."""
    day_seconds = float(td.days*24*3600)
    ms_seconds = float(td.microseconds)/1000000.0
    return  day_seconds + float(td.seconds) + ms_seconds

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
            fname = path.join(self.PRIVILEGED_BASE,self.BASE_LOG_NAME)
            infile = open(fname, 'a')
            infile.close()
        except IOError:
            fname = self.BASE_LOG_NAME
        #rothandle = logging.handlers.RotatingFileHandler(fname, 'a', 65535)
        rothandle = logging.handlers.RotatingFileHandler(fname, maxBytes=1024*1024*10, backupCount=5)
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
        if isinstance(exp,models.Experiment):
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
        ret = (self.state,self.state_time)
        self.lock.release()
        return ret

logger = CrawlLog()


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
        return map(str,self.logger.prev_exprs())
    def xmlrpc_experiments_found(self):
        return self.logger.expr_count
    def xmlrpc_state(self):
        msg,dt = self.logger.get_state()
        return (msg,tdelt2secs(datetime.datetime.now() - dt))
    def xmlrpc_hostname(self):
        return socket.gethostname()
    def xmlrpc_startPE(self,experiment_name,pe_forward,pe_reverse):
        '''Sends POST URL to start a Paired-End report'''
        self.logger.errors.info("xmlrpc_startPE for fwd:%s and rev:%s" % (pe_forward,pe_reverse))
        try:
            exp = models.Experiment.objects.get(expName=experiment_name)
        except:
            self.logger.errors.error("expName: %s not found in database" % experiment_name)
            return "Failed to find experiment in database"
        try:
            fwd_name = os.path.basename(pe_forward)     # Get just the directory name
            strip_str = fwd_name.split("_")[-1] # Strip off the final "_nnn" which is not in the Report Name but is in the directory name
            fwd_name = fwd_name.replace('_'+strip_str, '')
            
            rev_name = os.path.basename(pe_reverse)     # Get just the directory name
            strip_str = rev_name.split("_")[-1] # Strip off the final "_nnn" which is not in the Report Name but is in the directory name
            rev_name = rev_name.replace('_'+strip_str, '')
            
            autoReportName = "PE_%s_%s" % (fwd_name,rev_name)
            self.logger.errors.info("Paired-End Report autogenerated name: %s" % autoReportName)
        except:
            self.logger.errors.error(traceback.format_exc())
            return "Failed to auto-generate Paired-End Report name"
        
        return generate_http_post_PE(exp,pe_forward,pe_reverse,autoReportName)

ENTRY_MAP = {
    "gain": lre.float_parse,
    "datacollect_version": lre.int_parse,
    "liveview_version": lre.int_parse,
    "firmware_version": lre.int_parse,
    "fpga_version": lre.int_parse,
    "driver_version": lre.int_parse,
    "script_version": lre.dot_separated_parse,
    "board_version": lre.int_parse,
    "kernel_build": lre.kernel_parse,
    "prerun": lre.yn_parse,
    # TODO: Enumerate things that will break if this is changed
    # It is used to parse job files produced by the PGM, yes?
    "cycles": lre.int_parse,
    "livechip": lre.yn_parse,
    "continuous_clocking": lre.yn_parse,
    "auto_cal": lre.yn_parse,
    "frequency": lre.int_parse,
    "oversample": lre.oversample_parse,
    "frame_time": lre.float_parse,
    "num_frames": lre.int_parse,
    "autoanalyze": lre.yn_parse,
    "dac": lre.dac_parse,
    "cal_chip_hist": lre.space_separated_parse,
    "cal_chip_high_low_inrange": lre.cal_range_parse,
    "prebeadfind": lre.yn_parse,
    "flows": lre.int_parse,
    "analyzeearly": lre.yn_parse,
#    "chiptype": lre.chip_parse,    
}

def load_log(folder,logName):
    """Retrieve the contents of the experiment log found in ``folder``,
    or return ``None`` if no log can be found."""
    fname = path.join(folder, logName)
    if path.isfile(fname):
        infile = None
        try:
            infile = open(fname)
            text = infile.read()
        except IOError:
            text = None
        finally:
            if infile is not None:
                infile.close()
    else:
        text = None
    return text

def folder_mtime(folder):
    """Determine the time at which the experiment was performed. In order
    to do this reliably, ``folder_mtime`` tries the following approaches,
    and returns the result of the first successful approach:
    
    #. Parse the name of the folder, looking for a YYYY_MM_DD type
       timestamp
    #. ``stat()`` the folder and return the ``mtime``.
    """
    match = TIMESTAMP_RE.match(path.basename(folder))
    if match is not None:
        dt = datetime.datetime(*map(int,match.groups(1)))
        return dt
    else:
        acqfnames = glob.glob(path.join(folder, "*.acq"))
        if len(acqfnames) > 0:
            fname = acqfnames[0]
        else:
            fname = folder
        stat = os.stat(fname)
        return datetime.datetime.fromtimestamp(stat.st_mtime)

def parse_log(text):
    """Take the raw text of the experiment log, and return a dictionary
    of the entries contained in the log, parsed into the appropriate
    datatype.
    """
    def filter_non_printable(str):
        if isinstance(str,basestring):
            return ''.join([c for c in str if ord(c) > 31 or ord(c) == 9])
        else:
            return str
    def clean_name(name):
        no_ws = CLEAN_RE.sub("_", name)
        return no_ws.lower()
    def extract_entries(text):
        ret = []
        for line in text.split('\n'):
            match = ENTRY_RE.match(line)
            if match is not None:
                d = match.groupdict()
                ret.append((clean_name(d['name']),d['value'].strip()))
        return ret
    ret = {}
    entries = extract_entries(text)
    for name,value in entries:
        # utf-8 replace code to ensure we don't crash on invalid characters
        #ret[name] = ENTRY_MAP.get(name, lre.text_parse)(value.decode("ascii","ignore"))
        ret[name] = filter_non_printable(ENTRY_MAP.get(name, lre.text_parse)(value.decode("ascii","ignore")))
    #For the oddball repeating keyword: BlockStatus
    #create an array of them.
    ret['blocks'] = []
    for line in text.split('\n'):
        if line.startswith('BlockStatus') or line.startswith('RegionStatus') or line.startswith('TileStatus'):
            ret['blocks'].append (line)
    return ret

def extract_rig(folder):
    """Given the name of a folder storing experiment data, return the name
    of the PGM from which the date came."""
    return path.basename(path.dirname(folder))

def extract_prefix(folder):
    """Given the name of a folder storing experiment data, return the
    name of the directory under which all PGMs at a given location
    store their data."""
    return path.dirname(path.dirname(folder))

def getFlowOrder(rawString):

    '''Parses out a nuke flow order string from entry in explog.txt of the form:
    rawString format:  "4 0 r4 1 r3 2 r2 3 r1" or "4 0 T 1 A 2 C 3 G".'''

    #Initialize return value
    flowOrder = ''
    
    #Capitalize all lowercase; strip leading and trailing whitespace
    rawString = string.upper(rawString).strip()
    
    # If there are no space characters, it is 'new' format
    if rawString.find(' ') == -1:
        flowOrder = rawString
    else:
        #Define translation table
        table = {
                "R1":'G',
                "R2":'C',
                "R3":'A',
                "R4":'T',
                "T":'T',
                "A":'A',
                "C":'C',
                "G":'G'}
        #Loop thru the tokenized rawString extracting the nukes in order and append to return string
        for c in rawString.split(" "):
            try:
                flowOrder += table[c]
            except KeyError:
                pass

    # Add a safeguard.
    if len(flowOrder) < 4:
        flowOrder = 'TACG'
        
    return flowOrder

  
def exp_kwargs(d,folder):
    """Converts the output of `parse_log` to the a dictionary of
    keyword arguments needed to create an ``Experiment`` database object. """
    identical_fields = ("project","sample","library","cycles","flows",)
    simple_maps = (
        ("experiment_name","expName"),
        ("chiptype", "chipType"),
        ("chipbarcode", "chipBarcode"),
        ("user_notes", "notes"),
        ##("seqbarcode", "seqKitBarcode"),
        ("autoanalyze", "autoAnalyze"),
        ("prebeadfind", "usePreBeadfind"),
        ##("librarykeysequence", "libraryKey"),
        ("barcodeid", "barcodeId"),
        ("isReverseRun", "isReverseRun"),
        )
    full_maps = (
        ("pgmName",d.get('devicename',extract_rig(folder))),
        ("log", json.dumps(d, indent=4)),
        ("expDir", folder),
        ("unique", folder),
        ("baselineRun", d.get("runtype") == "STD" or d.get("runtype") == "Standard"),
        ("date", folder_mtime(folder)),
        ("storage_options",models.GlobalConfig.objects.all()[0].default_storage_options),
        ("flowsInOrder",getFlowOrder(d.get("image_map", ""))),
        ("reverse_primer",d.get('reverse_primer', 'Ion Kit')),
        )

    derive_attribute_list = [ "libraryKey", "reverselibrarykey", "forward3primeadapter", 
                             "reverse3primeadapter", "sequencekitname", "seqKitBarcode", 
                             "sequencekitbarcode", "librarykitname", "librarykitbarcode"]
    
    ret = {}
    for f in identical_fields:
        ret[f] = d.get(f,'')
    for k1,k2 in simple_maps:
        ret[k2] = d.get(k1,'')
    for k,v in full_maps:
        ret[k] = v

    for attribute in derive_attribute_list:
        ret[attribute] = ''

    #N.B. this field is not used
    ret['storageHost'] = 'localhost'

    # If Flows keyword is defined in explog.txt...
    if ret['flows'] != "":
        # Cycles should be based on number of flows, not cycles published in log file
        # (Use of Cycles is deprecated in any case! We should be able to enter a random number here)
        ret['cycles'] = int( int(ret['flows']) / len(ret['flowsInOrder']) )
    else:
        # ...if Flows is not defined in explog.txt:  (Very-old-dataset support)
        ret['flows'] = len(ret['flowsInOrder']) * int(ret['cycles'])
        logger.errors.warn ("Flows keyword missing: Calculated Flows is %d" % int(ret['flows']))

    if ret['barcodeId'].lower() == 'none':
        ret['barcodeId'] = ''
        
    # blocked datasets are indicated by presence of BlockStatus keywords
    if 'blockstatus' in d:
        ret['rawdatastyle'] = 'tiled'
        ret['autoAnalyze'] = False
        for bs in d['blocks']:
            if bs.find('thumbnail') > 0:
                continue
            if auto_analyze_block(bs) or analyze_on_instrument_block(bs):
                ret['autoAnalyze'] = True
                logger.errors.debug ("Block Run. Detected at least one block to auto-run analysis")
                break
        if ret['autoAnalyze'] == False:
            logger.errors.debug ("Block Run. auto-run whole chip has not been specified")
    else:
        ret['rawdatastyle'] = 'single'

    planShortId = d.get("planned_run_short_id", '')
    
    #fix [TS-3064] for PGM backward compatibility
    if (planShortId == None or len(planShortId) == 0):
        planShortId = d.get("pending_run_short_id", '')
    
    logger.errors.debug ("...planShortId=%s" % planShortId)
    
    print 'crawler: plannedRunShortId=', planShortId

    sequencingKitName = d.get("seqkitname", '')
    if sequencingKitName != "NOT_SCANNED":
        ret['sequencekitname'] = sequencingKitName
        
    #in rundb_experiment, there are 2 attributes for sequencingKitBarcode!!
    sequencingKitBarcode = d.get("seqkitpart", '')
    if sequencingKitBarcode != "NOT_SCANNED":
        ret['seqKitBarcode'] = sequencingKitBarcode
        ret['sequencekitbarcode'] = sequencingKitBarcode

    libraryKitName = d.get('libkit', '')
    if libraryKitName != "NOT_SCANNED":
        ret['librarykitname'] = libraryKitName

    libraryKitBarcode = d.get("libbarcode", '')
    if libraryKitBarcode != "NOT_SCANNED":
        ret['librarykitbarcode'] = libraryKitBarcode
        

    #Rules for applying the library key overrides: 
    #1) If plan is used and library key is specified, use that value
    #2) Otherwise, if user has specified one on PGM's advanced page 
    #   Validation required:
    #   Why: It could be left-over from a previous run and is not compatible with current run)
    #   How: It has to be pre-defined in db and is in the direction of the the new run.
    #   What: If it passes validation, use it
    #3) Otherwise, use system default for that direction
    #4) If plan is NOT used, and user has specified one in PGM's advanced page, do validation as above
    #5) If it passes validation, use it
    #6) Otherwise, use system default for that direction as defined in db 
    #7) If the system default somehow has no value, we'll use the library key from 
    #   PGM's advanced setup page 

    isReverseRun = d.get("isreverserun", '')
    
    #if PGM is running the old version, there is no isReverseRun in explog.txt. Check the plan if used
    if not isReverseRun:
        if (planShortId and len(planShortId) > 0):  
            try:
                planObj = models.PlannedExperiment.objects.get(planShortID=planShortId)
                if planObj.isReverseRun:
                    isReverseRun = "Yes"
                else:
                    isReverseRun = "No"

                logger.errors.info ("...planShortId=%s is for reverse=%s" % (planShortId, isReverseRun))
            except models.PlannedExperiment.DoesNotExist:
                logger.errors.warn("No plan with short id %s found in database " % planShortId )
            except models.PlannedExperiment.MultipleObjectsReturned:
                logger.errors.warn("Multiple plan with short id %s found in database " % planShortId)
                                
    if isReverseRun == "Yes":
        ret['isReverseRun'] = True
        ret['reverselibrarykey'] = ''
        
        try:
            defaultReverseLibraryKey = models.LibraryKey.objects.get(direction='Reverse', isDefault=True)
            defaultReverse3primeAdapter = models.ThreePrimeadapter.objects.get(direction='Reverse', isDefault=True)
            
            validatedPgmLibraryKey = None
            dbPgmLibraryKey = None
            pgmLibraryKey = d.get("librarykeysequence", '')
            
            hasPassed = False
            if pgmLibraryKey == None or len(pgmLibraryKey) == 0:
                #logger.errors.debug("...pgmLibraryKey not specified. ")
                hasPassed = False
            else:
                dbPgmLibraryKeys = models.LibraryKey.objects.filter(sequence=pgmLibraryKey)
        
                if dbPgmLibraryKeys:
                    for dbKey in dbPgmLibraryKeys:
                        if dbKey.direction == "Reverse":
                            #logger.errors.debug("...pgmLibraryKey %s has been validated for reverse run" % pgmLibraryKey)
                            validatedPgmLibraryKey = dbKey
                            hasPassed = True
                            break
                else:
                    hasPassed = False
            
            #set default in case plan is not used or not found in db
            if hasPassed:
                #logger.errors.debug("...Default for reverse run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)

                ret['reverselibrarykey'] = validatedPgmLibraryKey.sequence
            else:
                #logger.errors.debug("...Default for reverse run. Use default library key=%s " % defaultReverseLibraryKey.sequence)               
                ret['reverselibrarykey'] = defaultReverseLibraryKey.sequence
                
            ret['reverse3primeadapter'] = defaultReverse3primeAdapter.sequence
            
            if (planShortId and len(planShortId) > 0):  
                planObj = models.PlannedExperiment.objects.get(planShortID=planShortId)
                if planObj is not None:
                    #logger.errors.debug("...REVERSE plan is FOUND for planShortId=%s " % planShortId)
                    
                    if planObj.reverselibrarykey:
                        #logger.errors.debug("...Plan used for reverse run. Use plan library key=%s " % planObj.reverselibrarykey)
    
                        ret['reverselibrarykey'] = planObj.reverselibrarykey
                    else:
                        if hasPassed:
                            #logger.errors.debug("...Plan used for reverse run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)
    
                            ret['reverselibrarykey'] = validatedPgmLibraryKey.sequence
                        else:
                            #logger.errors.debug("...Plan used for reverse run. Use default library key=%s " % defaultReverseLibraryKey.sequence)
    
                            ret['reverselibrarykey'] = defaultReverseLibraryKey.sequence
                        
                    if planObj.reverse3primeadapter:
                        ret['reverse3primeadapter'] = planObj.reverse3primeadapter
                    else:
                        ret['reverse3primeadapter'] = defaultReverse3primeAdapter.sequence
                        
                else:
                    if hasPassed:
                        #logger.errors.debug("...Plan used but not on db for reverse run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)
    
                        ret['reverselibrarykey'] = validatedPgmLibraryKey.sequence
                    else:
                        logger.errors.debug("...Plan used but not on db for reverse run. Use default library key=%s " % defaultReverseLibraryKey.sequence)
                        
                        ret['reverselibrarykey'] = defaultReverseLibraryKey.sequence
    
                    ret['reverse3primeadapter'] = defaultReverse3primeAdapter.sequence
                 
            #this should never happen                 
            if ret['reverselibrarykey']== None or len(ret['reverselibrarykey']) == 0:
                #logger.errors.debug("...A library key cannot be determined for this REVERSE run  Use PGM default. ")
                ret['reverselibraryKey'] = d.get("librarykeysequence", '')

        except models.LibraryKey.DoesNotExist:
            logger.errors.warn("No default reverse library key in database for experiment %s" % ret['expName'])
            return ret, False
        except models.Experiment.MultipleObjectsReturned:
            logger.errors.warn("Multiple default reverse library keys found in database for experiment %s" % ret['expName'])
            return ret, False
        except models.ThreePrimeadapter.DoesNotExist:
            logger.errors.warn("No default reverse 3' adapter in database for experiment %s" % ret['expName'])
            return ret, False
        except models.ThreePrimeadapter.MultipleObjectsReturned:
            logger.errors.warn("Multiple default reverse 3' adapters found in database for experiment %s" % ret['expName'])
            return ret, False
        except models.PlannedExperiment.DoesNotExist:
            logger.errors.warn("No plan with short id %s found in database " % planShortId )
            #use default if plan is not found
        except models.PlannedExperiment.MultipleObjectsReturned:
            logger.errors.warn("Multiple plan with short id %s found in database " % planShortId)
            return ret, False                    
        except:
            logger.errors.warn("Experiment %s" % ret['expName'])
            logger.errors.warn(traceback.format_exc())
            return ret, False
    else:
        ret['isReverseRun'] = False
        ret['libraryKey'] = ''
            
        try:
            #NOTE: In v2.2, there is no way to tell if a run (aka an experiment) is part of a paired-end forward run or not
            defaultForwardLibraryKey = models.LibraryKey.objects.get(direction='Forward', isDefault=True)
            defaultForward3primeAdapter = models.ThreePrimeadapter.objects.get(direction='Forward', isDefault=True)
                          
            validatedPgmLibraryKey = None
            dbPgmLibraryKey = None
            pgmLibraryKey = d.get("librarykeysequence", '')
            #logger.errors.debug("...pgmLibraryKey is %s " % pgmLibraryKey)
                        
            hasPassed = False
            if pgmLibraryKey == None or len(pgmLibraryKey) == 0:
                #logger.errors.debug("...pgmLibraryKey not specified. ")
                hasPassed = False
            else:
                dbPgmLibraryKeys = models.LibraryKey.objects.filter(sequence=pgmLibraryKey)
        
                if dbPgmLibraryKeys:
                    for dbKey in dbPgmLibraryKeys:
                        if dbKey.direction == "Forward":
                            #logger.errors.debug("...pgmLibraryKey %s has been validated for forward run" % pgmLibraryKey)
                            validatedPgmLibraryKey = dbKey
                            hasPassed = True
                            break
                else:
                    hasPassed = False

            #set default in case plan is not used or not found in db           
            if hasPassed:
                #logger.errors.debug("...Default for forward run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)

                ret['libraryKey'] = validatedPgmLibraryKey.sequence
            else:
                #logger.errors.debug("...Default for forward run. Use default library key=%s " % defaultForwardLibraryKey.sequence)

                ret['libraryKey'] = defaultForwardLibraryKey.sequence
                
            ret['forward3primeadapter'] = defaultForward3primeAdapter.sequence     
                    
            if (planShortId and len(planShortId) > 0):  
                planObj = models.PlannedExperiment.objects.get(planShortID=planShortId)
                if planObj is not None:
    
                    #logger.errors.debug("...FORWARD plan is FOUND for planShortId=%s " % planShortId)
                                    
                    if planObj.libraryKey:
                        #logger.errors.debug("...Plan used for forward run. Use plan library key=%s " % planObj.libraryKey)
    
                        ret['libraryKey'] = planObj.libraryKey
                    else:
                        if hasPassed:
                            #logger.errors.debug("...Plan used for forward run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)
    
                            ret['libraryKey'] = validatedPgmLibraryKey.sequence
                        else:
                            #logger.errors.debug("...Plan used for forward run. Use default library key=%s " % defaultForwardLibraryKey.sequence)
    
                            ret['libraryKey'] = defaultForwardLibraryKey.sequence
                        
                    if planObj.forward3primeadapter:
                        ret['forward3primeadapter'] = planObj.forward3primeadapter
                    else:
                        ret['forward3primeadapter'] = defaultForward3primeAdapter.sequence
    
                else:
                    if hasPassed:
                        #logger.errors.debug("...Plan used but not on db for forward run. Use PGM library key=%s " % validatedPgmLibraryKey.sequence)
    
                        ret['libraryKey'] = validatedPgmLibraryKey.sequence
                    else:
                        #logger.errors.debug("...Plan used but not on db for forward run. Use default library key=%s " % defaultForwardLibraryKey.sequence)
                        
                        ret['libraryKey'] = defaultForwardLibraryKey.sequence
    
                    ret['forward3primeadapter'] = defaultForward3primeAdapter.sequence
                        
            if ret['libraryKey'] == None or ret['libraryKey'] == "":
                #logger.errors.debug("...A library key cannot be determined for this FORWARD run  Use PGM default. ")
                ret['libraryKey'] = d.get("librarykeysequence", '')

        except models.LibraryKey.DoesNotExist:
            logger.errors.warn("No default forward library key in database for experiment %s" % ret['expName'])
            return ret, False
        except models.Experiment.MultipleObjectsReturned:
            logger.errors.warn("Multiple default forward library keys found in database for experiment %s" % ret['expName'])
            return ret, False
        except models.ThreePrimeadapter.DoesNotExist:
            logger.errors.warn("No default forward 3' adapter in database for experiment %s" % ret['expName'])
            return ret, False
        except models.ThreePrimeadapter.MultipleObjectsReturned:
            logger.errors.warn("Multiple default forward 3' adapters found in database for experiment %s" % ret['expName'])
            return ret, False 
        except models.PlannedExperiment.DoesNotExist:
            logger.errors.warn("No plan with short id %s found in database " % planShortId)
            #use default if plan is not found
        except models.PlannedExperiment.MultipleObjectsReturned:
            logger.errors.warn("Multiple plan with short id %s found in database " % planShortId)
            return ret, False                      
        except:
            logger.errors.warn("Experiment %s" % ret['expName'])
            logger.errors.warn(traceback.format_exc())
            return ret, False
                 
    # Limit input sizes to defined field widths in models.py
    ret['notes'] = ret['notes'][:128]
    ret['expDir'] = ret['expDir'][:512]
    ret['expName'] = ret['expName'][:128]
    ret['pgmName'] = ret['pgmName'][:64]
    ret['unique'] = ret['unique'][:512]
    ret['storage_options'] = ret['storage_options'][:200]
    ret['project'] = ret['project'][:64]
    ret['sample'] = ret['sample'][:64]
    ret['library'] = ret['library'][:64]
    ret['chipBarcode'] = ret['chipBarcode'][:64]
    ret['seqKitBarcode'] = ret['seqKitBarcode'][:64]
    ret['chipType'] = ret['chipType'][:32]
    ret['flowsInOrder'] = ret['flowsInOrder'][:512]
    ret['libraryKey'] = ret['libraryKey'][:64]
    ret['barcodeId'] = ret['barcodeId'][:128]
    ret['reverse_primer'] = ret['reverse_primer'][:128]
    ret['reverselibrarykey'] = ret['reverselibrarykey'][:64]    
    ret['reverse3primeadapter'] = ret['reverse3primeadapter'][:512]
    ret['forward3primeadapter'] = ret['forward3primeadapter'][:512]
    ret['sequencekitbarcode'] = ret['sequencekitbarcode'][:512]
    ret['librarykitbarcode'] = ret['librarykitbarcode'][:512]
    ret['sequencekitname'] = ret['sequencekitname'][:512]    
    ret['sequencekitbarcode'] = ret['sequencekitbarcode'][:512]
    ret['librarykitname'] = ret['librarykitname'][:512]    
    ret['librarykitbarcode'] = ret['librarykitbarcode'][:512]

    logger.errors.debug("For experiment %s" % ret['expName'])        
    logger.errors.debug("...Ready to save run: isReverseRun=%s;" % ret['isReverseRun'])
    logger.errors.debug("...Ready to save run: libraryKey=%s;" % ret['libraryKey']) 
    logger.errors.debug("...Ready to save run: forward3primeadapter=%s;" % ret['forward3primeadapter'])     
    logger.errors.debug("...Ready to save run: reverselibrarykey=%s;" % ret['reverselibrarykey'])     
    logger.errors.debug("...Ready to save run: reverse3primeadapter=%s;" % ret['reverse3primeadapter'])  
        
    return ret, True

def exp_from_kwargs(kwargs,logger, save=True):
    """Create an experiment from the given keyword arguments. 
    """
    ret = models.Experiment(**kwargs)
    matches = models.Experiment.objects.filter(unique=ret.unique)
    if matches:
        e = matches[0]
        if e.ftpStatus != RUN_STATUS_COMPLETE:
            return e
        else:
            return None
        
    if save:
        ret.save()
    logger.add_experiment(ret)
    return ret

def construct_crawl_directories(logger):
    """Query the database and build a list of directories to crawl.
    Returns an array.
    For every Rig in the database, construct a filesystem path and
    get all subdirectories in that path."""
    def dbase_isComplete(name):
        try:
            run = models.Experiment.objects.get(expDir=name)
            if run.ftpStatus.strip() == RUN_STATUS_COMPLETE or \
               run.ftpStatus.strip() == RUN_STATUS_ABORT or \
               run.ftpStatus.strip() == RUN_STATUS_SYS_CRIT:
                return True
            else:
                return False
        except models.Experiment.DoesNotExist:
            logger.errors.debug("Experiment %s does not exist in database" % name)
            return False
        except models.Experiment.MultipleObjectsReturned:
            logger.errors.debug("Multiple Experiments %s exist " % name)
            return False
        except:
            logger.errors.debug("Experiment %s" % name)
            logger.errors.debug(traceback.format_exc())
            return False
    
    fserves = models.FileServer.objects.all()      
    ret = []
    for fs in fserves:
        l = fs.location
        rigs = models.Rig.objects.filter(location=l)
        for r in rigs:
            rig_folder = path.join(fs.filesPrefix,r.name)
            if path.exists(rig_folder):
                try:
                    subdir_bases = os.listdir(rig_folder)
                    # create array of paths for all directories in Rig's directory
                    s1 = [path.join(rig_folder,subd) for subd in subdir_bases]
                    s2 = [subd for subd in s1 if path.isdir(subd)]
                    # create array of paths of not complete ftp transfer only
                    s3 = [subd for subd in s2 if dbase_isComplete(subd) == False]
                    ret.extend(s3)
                except:
                    logger.errors.error(traceback.format_exc())
                    logger.set_state('error')
    return ret

def get_percent(exp):
    """get the percent complete to make a progress bar on the web interface"""
    
    def filecount(dir_name):
        return len(glob.glob(path.join(dir_name, "acq*.dat")))
        
    if 'tiled' in exp.rawdatastyle:
        #N.B. Hack - we check ftp status of thumbnail data only
        expDir = path.join(exp.expDir,'thumbnail')
    else:
        expDir = exp.expDir
        
    flowspercycle = get_flow_per_cycle(exp) #if we exclude the wash flow which is no longer output
    expectedFiles = float(exp.flows)
    file_count = float(filecount(expDir))
    try: # don't want zero division to kill us
        percent_complete = int((file_count/expectedFiles)*100)
    except:
        percent_complete = 0 
    if percent_complete >= 99:  # make the bar never quite reach 100% since we don't count beadfind files
        percent_complete = 99
            
    return percent_complete 

def sleep_delay(start,current,delay):
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
    if not name:
        return 'Auto_%s_%s%s' % (exp.pretty_print(),exp.pk,twig)
    else:
        return '%s_%s%s' % (str(name),exp.pk,twig)

def getForwardReport(experiment):
    '''Return Report name from an existing Experiment with the same chipBarcode'''
    #Default value is the string "None"
    name = "None"
    
    # If this is not a reverse run, then nothing to do
    if not experiment.isReverseRun:
        logger.errors.debug("This is not a reverse Run")
        return name
    
    #Get this experiment's chipBarcode
    chipBarcode = experiment.chipBarcode
    
    # Degnerate case where no chipBarcode is defined
    if chipBarcode == "":
        logger.errors.debug("This Run has no chipbarcode defined")
        return name
    
    logger.errors.debug("This experiment's chipBarcode: %s" % chipBarcode)
    
    #Get any Experiments, other than this one, with identical chipBarcode
    #Order by PK if there are multiple returns
    pe_experiments = models.Experiment.objects.filter(chipBarcode=chipBarcode).order_by('pk').exclude(expName__exact=experiment.expName)
    num_experiments = len(pe_experiments)
    logger.errors.debug("Num other experiments with same chipBarcode: %d" % num_experiments)
    
    if num_experiments > 0:
        #In case more than one Experiment is returned, we select the last (most recent)
        fwd_experiment = pe_experiments[num_experiments - 1]
        logger.errors.debug("Assume this is FWD experiment: %s" % fwd_experiment.expName)
        logger.errors.info("Paired-End Report will be triggered for:")
        logger.errors.info("Forward Exp - '%s'" % fwd_experiment.expName)
        logger.errors.info("Reverse Exp - '%s'" % experiment.expName)
    
        #Get Reports for Experiment, in inverse time order
        Reports = fwd_experiment.sorted_results_with_reports()
        num_reports = len(Reports)
        logger.errors.debug("Num Reports for FWD experiment: %d" % num_reports)
    
        if num_reports > 0:
            #In case more than one Report is returned, we select the most recent
            report = Reports[0]
            name = report.get_report_dir()
            logger.errors.debug("Use this Report as FWD Run: %s" % name)
        else:
            logger.errors.warn("There are no Reports for the Forward Experiment!")
    
    return name

def generate_http_post_PE(exp,f,r,autoReportName):
    '''Wrapper function which will trigger Paired-End Analysis pipeline execution.
    exp is Experiment object, f is forward run Report name, r is reverse run Report name'''
    
    return generate_http_post(exp,thumbnail_analysis=False,pe_forward=f,pe_reverse=r,report_name=autoReportName)

def generate_http_post(exp,thumbnail_analysis=False,pe_forward="",pe_reverse="",report_name=""):
    def get_default_command(chipName):
        gc = models.GlobalConfig.objects.all()
        ret = None
        if len(gc)>0:
            for i in gc:
                ret = i.default_command_line
        else:
            ret = 'Analysis'
        # Need to also get chip specific arguments from dbase
        #print "chipType is %s" % chipName
        chips = models.Chip.objects.all()
        for chip in chips:
            if chip.name in chipName:
                ret = ret + " " + chip.args
        return ret

    try:
        basecallerArgs = models.GlobalConfig.objects.all().order_by('pk')[0]
        basecallerArgs = basecallerArgs.basecallerargs
    except models.GlobalConfig.DoesNotExist:
        basecallerArgs = "BaseCaller"

    if pe_forward == "":
        pe_forward = getForwardReport(exp)
    if pe_reverse == "":
        pe_reverse = "None"
    if report_name == "":
        report_name = get_name_from_json(exp,'autoanalysisname',thumbnail_analysis)
        
    params = urllib.urlencode({'report_name':report_name,
                            'tf_config':'',
                            'path':exp.expDir,
                            'args':get_default_command(exp.chipType),
                            'basecallerArgs':basecallerArgs,
                            'submit': ['Start Analysis'],
                            'do_thumbnail':"%r" % thumbnail_analysis,
                            'forward_list':pe_forward,
                            'reverse_list':pe_reverse})
    #logger.errors.debug (params)
    headers = {"Content-type": "text/html",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    
    status_msg = "Generated POST"
    try:
        f = urllib.urlopen('http://%s/rundb/newanalysis/%s/0' % (settings.QMASTERHOST, exp.pk), params)
        response = f.read()
        #logger.errors.debug(response)
    except:
        logger.errors.error('could not autostart %s' % exp.expName)
        logger.errors.error(traceback.format_exc())
        try:
            f = urllib.urlopen('https://%s/rundb/newanalysis/%s/0' % (settings.QMASTERHOST, exp.pk), params)
            response = f.read()
            #logger.errors.debug(response)
        except:
            logger.errors.error('could not autostart %s' % exp.expName)
            logger.errors.error(traceback.format_exc())
            status_msg = "Failure to generate POST"
    return status_msg

def check_for_autoanalyze(exp):
    data = exp.log
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
    return path.exists(path.join(expDir,filelookup))

def check_for_abort(expDir, filelookup):
    """Check if run has been aborted by the user"""
    # parse explog_final.txt
    try:
        f = open(path.join(expDir,filelookup), 'r')
    except:
        print "Error opening file: ", path.join(expDir,filelookup)
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
        f = open(path.join(exp.expDir,filelookup), 'r')
    except:
        print "Error opening file: ", path.join(exp.expDir,filelookup)
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
            flows=len(data[im])
        else:
            # when Image Map is "Image Map: 4 0 r4 1 r3 2 r2 3 r1" or "4 0 T 1 A 2 C 3 G"
            flows=int(data[im].split(' ')[0])
        return flows
    else:
        return 4 #default to 4 if we have no other information

def get_last_file(exp):
    fpc = get_flow_per_cycle(exp)
    last = exp.flows - 1
    pre = '0000'
    final = pre+str(last)
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
        
    if check_for_file(exp.expDir,'explog_final.txt'):            
        #check for critical errors
        if check_for_critical(exp,'explog_final.txt'):
            return True # run is complete; true
        
        if 'tiled' in exp.rawdatastyle:
            #N.B. Hack - we check status of thumbnail data only
            expDir = path.join(exp.expDir,'thumbnail')
        else:
            expDir = exp.expDir
            
        #check for required files
        for file in file_list:
            if not check_for_file(expDir,file):
                exp.ftpStatus = RUN_STATUS_MISSING
                exp.save()
                return False

        return True
    else:
        return False
    
def analyze_early_block(blockstatus):
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
    
def auto_analyze_block(blockstatus):
    '''blockstatus is the string from explog.txt starting with keyword BlockStatus.
    Evaluates the AutoAnalyze flag.
    Returns boolean True when argument is 1'''
    try:
        arg = blockstatus.split(',')[4].strip()
        flag = int(arg.split(':')[1]) == 1
    except:
        logger.errors.error(traceback.format_exc())
        flag = False
    return flag
    
def analyze_on_instrument_block(blockstatus):
    '''blockstatus is the string from explog.txt starting with keyword BlockStatus.
    Evaluates the oninst flag.
    Returns boolean True when argument is 1'''
    try:
        arg = blockstatus.split(',')[8].strip()
        flag = int(arg.split(':')[1]) == 1
    except:
        # TODO: change this to warn or error once instrumente code is writing the oninst argument.
        #logger.errors.debug(traceback.format_exc())
        flag = False
    return flag

def get_block_subdir(blockstatus):
    '''blockstatus is the string from explog.txt starting with keyword BlockStatus.
    Evaluates the X and Y arguments.
    Returns string containing rawdata subdirectory'''
    try:
        blockdata = blockstatus.split(':')[1].strip()
        blockx = blockdata.split(',')[0].strip()
        blocky = blockdata.split(',')[1].strip()
        subdir = "%s_%s" % (blockx,blocky)
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
        for bs in data['blocks']:
            if bs.find('thumbnail') > 0:
                continue
            if analyze_early_block(bs):
                # This block wants early analysis;
                if analyze_on_instrument_block(bs):
                    # This block will analyze on-instrument
                    readyToStart = True
                else:
                    # This block will analyze off-instrument
                    # test for valid data
                    subdir = os.path.join(exp.expDir,get_block_subdir(bs))
                    if raw_data_exists(subdir):
                        # If all subdirs are ready, this will remain True at exit
                        logger.errors.debug("Ready on %s" % subdir)
                        readyToStart = True
                    else:
                        # cannot start analysis - no valid data here
                        logger.errors.debug("Waiting on %s" % subdir)
                        readyToStart = False
                        break
            else:
                # This block is not designated for early analysis
                pass
    
    # Process single image dataset    
    else:
        if 'analyzeearly' in data:
            readyToStart = data.get('analyzeearly')
            
    return readyToStart

def raw_data_exists(dir):
    '''Checks if raw data files exist yet or subdirectories exist yet (Block data runs)'''
    
    if not os.path.isdir(dir):
        logger.errors.warn("Invalid directory: %s" % dir)
        return False
    
    # Method: check for the first flow file: acq_0000.dat
    # Case of 31x series chips where all data is in one directory
    # try/catch prtoects against permission errors
    try:
        if 'acq_0000.dat' in os.listdir(dir):
            logger.errors.debug("We found acq_0000.dat file")
            return True
    except:
        logger.errors.error(traceback.format_exc())
        return False
    
    # Method: check for any subdirectory, if so, then true
    try:
        subdir = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
        if len(subdir) > 0:
            logger.errors.debug("We found a subdirectory")
            return True
    except:
        logger.errors.error(traceback.format_exc())
        return False
    
    return False

def ready_to_process_thumbnail(exp):
    
    if 'tiled' in exp.rawdatastyle:
        for bs in exp.log['blocks']:
            if bs.find('thumbnail') > 0:
                if analyze_early_block(bs):
                    #logger.errors.info ("Request to analyze-early thumbnail analysis")
                    # Determine if thumbnail directory exists - sufficient to start analysis
                    if os.path.exists(os.path.join(exp.expDir,'thumbnail')):
                        return True
    
    return False

def autorun_thumbnail(exp):

    if 'tiled' in exp.rawdatastyle:
        for bs in exp.log['blocks']:
            if bs.find('thumbnail') > 0:
                arg = bs.split(',')[4].strip()
                if int(arg.split(':')[1]) == 1:
                    #logger.errors.info ("Request to auto-run thumbnail analysis")
                    return True
        
    return False

def crawl(folders,logger):
    """Crawl over ``folders``, reporting information to the ``CrawlLog``
    ``logger``."""
    logger.errors.info("checking %d directories" % len(folders))
    
    for f in folders:
        text = load_log(f,LOG_BASENAME)
        logger.errors.debug("checking directory %s" % f)
        #------------------------------------
        # Check for a valid explog.txt file
        #------------------------------------
        if text is not None:
            
            #-----------------------------------
            # Check for valid raw data
            #-----------------------------------
            if raw_data_exists(f) is False:
                logger.errors.debug ("Waiting for data...Experiment not added")
                continue
            
            logger.current_folder = f
            d = parse_log(text)
            kwargs, status = exp_kwargs(d,f)
            
            if (status == True):
                exp = exp_from_kwargs(kwargs, logger, False)
                if exp is not None:
                    
                    logger.errors.debug ("%s" % f)
                    
                    #--------------------------------
                    # Update FTP Transfer Status
                    #--------------------------------
                    if check_for_completion(exp):
                        # FTP transfer is complete
                        if exp.ftpStatus == RUN_STATUS_ABORT or exp.ftpStatus == RUN_STATUS_SYS_CRIT:
                            logger.errors.info("FTP status: Aborted")
                        else:
                            exp.ftpStatus = RUN_STATUS_COMPLETE
                            
                        exp.log = json.dumps(parse_log(load_log(f,LOG_FINAL_BASENAME)), indent=4)
                        exp.save()
                    else:
                        if exp.ftpStatus != RUN_STATUS_MISSING:
                            exp.ftpStatus = get_percent(exp)
                            exp.save()
                            logger.errors.info("FTP status: Transferring")
                    
                    #--------------------------------
                    # Handle auto-run analysis
                    #--------------------------------
                    composite_exists,thumbnail_exists = reports_exist(exp)
                    
                    if not composite_exists:
                        logger.errors.debug ("No auto-run analysis has been started")
                        # Check if auto-run for whole chip has been requested
                        if exp.autoAnalyze:
                            logger.errors.debug ("  auto-run whole chip analysis has been requested")
                            #if check_analyze_early(exp) or check_for_completion(exp):
                            if ready_to_process(exp) or check_for_completion(exp):
                                logger.errors.info ("  Start a whole chip auto-run analysis job")
                                generate_http_post(exp)
                            else:
                                logger.errors.info ("  Do not start a whole chip auto-run job yet")
                        else:
                            logger.errors.debug ("  auto-run whole chip analysis has not been requested")
                    else:
                        logger.errors.debug ("composite report already exists")
                        
                    if not thumbnail_exists:
                        # Check if auto-run for thumbnail has been requested
                        # But only if its a block dataset
                        if 'tiled' in exp.rawdatastyle:
                            if autorun_thumbnail(exp):
                                logger.errors.debug ("auto-run thumbnail analysis has been requested") 
                                if ready_to_process_thumbnail(exp) or check_for_completion(exp):
                                    logger.errors.info ("  Start a thumbnail auto-run analysis job")
                                    generate_http_post(exp,DO_THUMBNAIL)
                                else:
                                    logger.errors.info ("  Do not start a thumbnail auto-run job yet")
                            else:
                               logger.errors.debug ("auto-run thumbnail analysis has not been requested")
                        else:
                            logger.errors.debug ("This is not a block dataset; no thumbnail to process")
                    else:
                        logger.errors.debug ("thumbnail report already exists")
                else:
                    logger.errors.debug ("exp is empty")
            else:
                logger.errors.info("Experiment is not added due to invalid experiment status from exp_kwargs. " )
        else:
            # No explog.txt exists; probably an engineering test run
            logger.errors.debug("no %s was found in %s" % (LOG_BASENAME,f))
            pass
    logger.current_folder = '(none)'

def loop(logger,end_event,delay):
    """Outer loop of the crawl thread, calls ``crawl`` every minute."""
    logger.start()
    while not end_event.isSet():
        connection.close()  # Close any db connection to force new one.
        try:
            logger.set_state('working')
            start = datetime.datetime.now()
            folders = construct_crawl_directories(logger)
            try:
                crawl(folders,logger)
            except:
                exc = traceback.format_exc()
                logger.errors.error(exc)
                reactor.stop()
                end_event.set()
                sys.exit(1)
            logger.set_state('sleeping')
            
        except KeyboardInterrupt:
            end_event.set()
        except:
            logger.errors.error(traceback.format_exc())
            logger.set_state('error')   
        sleep_delay(start,datetime.datetime.now(),delay)     
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
            val = report.metaData.get('thumb',0)
            if val == 1:
                thumbnail = True
            if val == 0:
                composite = True
    
    return composite,thumbnail

def checkThread (thread,log,reactor):
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
    logger.errors.info("Crawler Initializing")

    exit_event = threading.Event()
    loopfunc = lambda: loop(logger,exit_event,settings.CRAWLER_PERIOD)
    lthread = threading.Thread(target=loopfunc)
    lthread.setDaemon(True)
    lthread.start()
    logger.errors.info("ionCrawler Started Ver: %s" % __version__)
    
    # check thread health periodically and exit if thread is dead
    # pass in reactor object below to kill process when thread dies
    l = task.LoopingCall(checkThread,lthread,logger,reactor)
    l.start(30.0) # call every 30 second
    
    # start the xml-rpc server
    r = Status(logger)
    reactor.listenTCP(settings.CRAWLER_PORT, server.Site(r))
    reactor.run()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))
