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
try:
    import json
except ImportError:
    import simplejson as json
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
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django import db

from twisted.internet import reactor
from twisted.internet import task
from twisted.web import xmlrpc,server

from djangoinit import *
from iondb.rundb import models, views
from django.db import connection
    
import logregexp as lre
import urllib
import string

__version__ = filter(str.isdigit, "$Revision: 21186 $")

LOG_BASENAME = "explog.txt"
LOG_FINAL_BASENAME = "explog_final.txt"
ENTRY_RE = re.compile(r'^(?P<name>[^:]+)[:](?P<value>.*)$')
CLEAN_RE = re.compile(r'\s|[/]+')
TIMESTAMP_RE = models.Experiment.PRETTY_PRINT_RE

DEBUG = False

# These strings will be displayed in Runs Page under FTP Status field
RUN_STATUS_COMPLETE = "Complete"
RUN_STATUS_MISSING  = "Missing File(s)"
RUN_STATUS_ABORT    = "User Aborted"
RUN_STATUS_SYS_CRIT = "Lost Chip Connection"


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
    "analyzeearly": lre.yn_parse
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
    
    #Capitalize all lowercase
    rawString = string.upper(rawString)
    
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
        ("autoanalyze", "autoAnalyze"),
        ("seqbarcode", "seqKitBarcode"),
        ("autoanalyze", "autoAnalyze"),
        ("prebeadfind", "usePreBeadfind"),
        ("librarykeysequence", "libraryKey"),
        ("barcodeid", "barcodeId"),
        )
    full_maps = (
        ("pgmName",extract_rig(folder)),
        ("log", json.dumps(d, indent=4)),
        ("expDir", folder),
        ("unique", folder),
        ("baselineRun", d.get("runtype") == "STD" or d.get("runtype") == "Standard"),
        ("date", folder_mtime(folder)),
        ("storage_options",models.GlobalConfig.objects.all()[0].default_storage_options),
        ("flowsInOrder",getFlowOrder(d.get("image_map", ""))),
        ("reverse_primer",d.get('reverse_primer', 'Ion Kit')),
        )
    ret = {}
    for f in identical_fields:
        ret[f] = d.get(f,'')
    for k1,k2 in simple_maps:
        ret[k2] = d.get(k1,'')
    for k,v in full_maps:
        ret[k] = v
    if 'localhost' in settings.QMASTERHOST:
        ret['storageHost'] = 'localhost'    # Data is on primary server
    else:
        ret['storageHost'] = os.uname()[1]  # Data is on secondary server
    
    # If Flows keyword is defined in explog.txt...
    if ret['flows'] != "":
        # Cycles should be based on number of flows, not cycles published in log file
        # (Use of Cycles is deprecated in any case! We should be able to enter a random number here)
        ret['cycles'] = int(ret['flows'] / len(ret['flowsInOrder']))
    else:
        # ...if Flows is not defined in explog.txt:  (Very-old-dataset support)
        ret['flows'] = len(ret['flowsInOrder']) * ret['cycles']
        logger.errors.debug ("Flows keyword missing: Calculated Flows is %d" % ret['flows'])

    if ret['barcodeId'].lower() == 'none':
        ret['barcodeId'] = ''
        
    # blocked datasets are indicated by presence of RegionStatus keywords
    if 'regionstatus' in d or 'tilestatus' in d or 'blockstatus' in d:
        ret['rawdatastyle'] = 'tiled'
        ret['autoAnalyze'] = True
    else:
        ret['rawdatastyle'] = 'single'

    return ret

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

#def emit_crawl_dirs():
#    """Print to standard out the JSON containing the names of all
#    directories potentially containing experiment data."""
#    import StringIO
#    oldout = sys.stdout
#    buf = StringIO.StringIO()
#    sys.stdout = buf
#    dirs = construct_crawl_directories()
#    serial = json.dumps(dirs)
#    sys.stdout = oldout
#    buf.close()
#    print serial
#
#def get_crawl_dirs_mp():
#    """Spawn another process to query the database and generate the list
#    of directories to crawl. The reason for spawning this process is to
#    avoid a memory leak due to Django queries.
#    """
#    args = (sys.executable, "-c", "from crawler import emit_crawl_dirs;"
#            + " emit_crawl_dirs()")
#    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
#                            stderr=subprocess.STDOUT)
#    out,er = proc.communicate()
#    if proc.returncode == 0:
#        return json.loads(out)
#    else:
#        raise ValueError("Query process failed")    

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
    flowspercycle = get_flow_per_cycle(exp) #if we exclude the wash flow which is no longer output
    expectedFiles = float(exp.flows)
    file_count = float(filecount(exp.expDir))
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

def get_name_from_json(exp, key):
    data = exp.log
    name = data.get(key, False)
    twig = ''
    if '/thumbnail' in exp.expDir:
        twig = '_tn'
    if not name:
        return 'Auto_%s_%s%s' % (exp.pretty_print(),exp.pk,twig)
    else:
        return '%s_%s%s' % (str(name),exp.pk,twig)

def thumbnail_report_post(exp,logger):
    
    # Proceed only if this is a block dataset
    if 'blockstatus' in exp.log:
    
        # modify the raw data path
        exp.expDir = os.path.join(exp.expDir,'thumbnail')
        
        # trigger the analysis job
        generate_http_post(exp,logger)
        
        #debug
        logger.errors.debug("Launching a thumbnail job")
    
def generate_http_post(exp,logger):
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
    params = urllib.urlencode({'report_name':get_name_from_json(exp,'autoanalysisname'),
                               'tf_config':'',
                               'path':exp.expDir,
                               'args':get_default_command(exp.chipType),
                               'submit': ['Start Analysis'],
                               'qname':settings.SGEQUEUENAME,
                               })
    headers = {"Content-type": "text/html",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    try:
        f = urllib.urlopen('http://%s/rundb/newanalysis/%s/0' % (settings.QMASTERHOST, exp.pk), params)
        f.read()
        f.close()
    except:
        logger.errors.error('could not autostart %s' % exp.expName)
        logger.errors.debug(traceback.format_exc())
        try:
            f = urllib.urlopen('https://%s/rundb/newanalysis/%s/0' % (settings.QMASTERHOST, exp.pk), params)
            f.read()
            f.close()
        except:
            logger.errors.error('could not autostart %s' % exp.expName)
            logger.errors.debug(traceback.format_exc())
    return 

def check_for_autoanalyze(exp):
    data = exp.log
    if 'autoanalyze' in data or 'tilestatus' in data or 'blockstatus' in data:
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
        try:
            # when Image Map is "Image Map: 4 0 r4 1 r3 2 r2 3 r1" or "4 0 T 1 A 2 C 3 G"
            flows=int(data[im].split(' ')[0])
        except:
            # when Image Map is "Image Map: tacgtacgtctgagcatcgatcgatgtacagc"
            flows=len(data[im])
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
        
        #check for required files
        for file in file_list:
            if not check_for_file(exp.expDir,file):
                exp.ftpStatus = RUN_STATUS_MISSING
                exp.save()
                return False

        return True
    else:
        return False

def check_analyze_early(exp):
    data = exp.log
    # default is not to analyize early
    wantAnalyzeEarly = False

    # but we are allowed to analyze early if user requests it
    if 'analyzeearly' in data:
        wantAnalyzeEarly = data.get('analyzeearly')

    # for blocked datasets, the global setting for analyzeearly needs to be true
    if 'regionstatus' in data or 'tilestatus' in data or 'blockstatus' in data:
        wantAnalyzeEarly = True
        
    return wantAnalyzeEarly

def thumbnailMerge(exp,logger):
    '''Trigger creation of a thumbnail image set from the two individual thumbnails
    provided currently by blocked datasets.
    Make system call to merge command line tool.
    '''
    # Is this a blocked dataset that should have a thumbnail created.
    if 'blockstatus' in exp.log:
    
        topDir = os.path.join(exp.expDir,'thumbnail_top')
        bottomDir = os.path.join(exp.expDir,'thumbnail_bottom')
        outDir = os.path.join(exp.expDir,'thumbnail')
        
        if os.path.exists(outDir):
            logger.errors.debug ("Thumbnail directory %s does exist already" % outDir)
            return False
        
        # Check for valid directory paths
        if not os.path.isdir(topDir):
            topDir = os.path.join(exp.expDir,'X-1_Y1')
            if not os.path.isdir(topDir):
                logger.errors.debug ("Required subdir is missing (X-1_Y1 or thumbnail_top")
                return True
            
        if not os.path.isdir(bottomDir):
            bottomDir = os.path.join(exp.expDir,'X-1_Y0')
            if not os.path.isdir(bottomDir):
                logger.errors.debug ("Required subdir is missing (X-1_Y0 or thumbnail_bottom")
                return True
        
        # Create the output directory so we can open a log file
        os.mkdir(outDir)
        # Open a log file
        logfh = open(os.path.join(outDir,'MergeDatserrout.log'),'wb')
        # Launch a process into the wind.  Hope it finishes.  maybe you should submit a job instead?
        args = ['MergeDats','-s',topDir,'-t',bottomDir,'-o',outDir]
        proc = subprocess.Popen(args,stderr=subprocess.STDOUT,stdout=logfh)
        logger.errors.debug ("MergeDats launched: %s" % outDir)

    return False

def raw_data_exists(dir):
    '''Checks if raw data files exist yet or subdirectories exist yet (Block data runs)'''
    # Method: check for any subdirectory, if so, then true
    subdir = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
    if len(subdir) > 0:
        logger.errors.debug("We found a subdirectory")
        return True
    # Method: check for a .dat file
    datfile = [name for name in os.listdir(dir) if '.dat' in name]
    if len(datfile) > 0:
        logger.errors.debug("We found a dat file")
        return True
    return False

def crawl(folders,logger):
    """Crawl over ``folders``, reporting information to the ``CrawlLog``
    ``logger``."""
    logger.errors.debug("checking %d directories" % len(folders))
    #TODO: Refactor this section, bearing in mind that the construct_crawl_directories()
    #TODO: no longer returns ALL directories, but only the directories of those runs
    #TODO: which are not yet finished transferring.
    for f in folders:
        text = load_log(f,LOG_BASENAME)
        logger.errors.debug("checking directory %s" % f)
        if text is not None:
            
            if raw_data_exists(f) is False:
                logger.errors.info ("Waiting for data...Experiment not added")
                continue
            
            logger.current_folder = f
            d = parse_log(text)
            kwargs = exp_kwargs(d,f)
            exp = exp_from_kwargs(kwargs, logger, False)
            if exp is not None:
                # Identify 'alpha' machine by lack of 'autoanalyze' parameter in explog.txt
                if not DEBUG and not check_for_autoanalyze(exp):
                    if check_for_file(exp.expDir,'beadfind_post_0003.dat'):
                        exp.ftpStatus = RUN_STATUS_COMPLETE
                        exp.save()
                # If we are using a 'beta' we have more options open for checking ftp completion
                elif not DEBUG and check_for_completion(exp):
                    if exp.ftpStatus == RUN_STATUS_ABORT or exp.ftpStatus == RUN_STATUS_SYS_CRIT:
                        exp.log = json.dumps(parse_log(load_log(f,LOG_FINAL_BASENAME)), indent=4)
                        exp.save()
                        logger.errors.info("Aborted: %s" % f)
                    else:
                        exp.ftpStatus = RUN_STATUS_COMPLETE
                        exp.log = json.dumps(parse_log(load_log(f,LOG_FINAL_BASENAME)), indent=4)
                        exp.save()
                        if exp.autoAnalyze and not check_analyze_early(exp):
                            if thumbnailMerge(exp,logger): # Create thumbnail from dataset
                                continue
                            generate_http_post(exp,logger) #auto start analysis
                            thumbnail_report_post(exp,logger)
                            logger.errors.info("Starting auto analysis: %s" % f)
                # if we fall through, it is still transferring...
                else:
                    if exp.ftpStatus != RUN_STATUS_MISSING:
                        #exp.ftpStatus = "Transferring"
                        exp.ftpStatus = get_percent(exp)
                        exp.save()
                        logger.errors.info("Transferring: %s" % f)
                # check separately that we want to start the run before the ftp is complete
                # the check in this case is that autoanalysis=True and AnalyzeEarly=True
                if exp.autoAnalyze and check_analyze_early(exp):
                    # check that there isn't already one started
                    try:
                        res = exp.results_set.all()                        
                        if not res:
                            if thumbnailMerge(exp,logger): # Create thumbnail from dataset
                                continue
                            generate_http_post(exp,logger) #auto start analysis
                            thumbnail_report_post(exp,logger)
                            logger.errors.info("Starting auto analysis: %s" % f)
                    except:
                        logger.errors.error(traceback.format_exc())
                else:
                    logger.errors.debug ("autoAnalyze or analyzeEarly are not set")
            else:
                logger.errors.debug ("exp is empty for %s" % f)
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
