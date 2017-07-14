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
import sys
import threading
import time
import traceback
import urllib
import argparse

#from iondb.bin.djangoinit import *
from iondb.bin import djangoinit
from django import db
from django.db import connection
from django.conf import settings

from twisted.internet import reactor
from twisted.internet import task
from twisted.web import xmlrpc, server

from iondb.rundb import models
from ion.utils.explogparser import load_log
from ion.utils.explogparser import parse_log
try:
    import iondb.version as version  # @UnresolvedImport
    GITHASH = version.IonVersionGetGitHash()
except:
    GITHASH = ""

LOG_BASENAME = "explog.txt"
LOG_FINAL_BASENAME = "explog_final.txt"


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

    def __init__(self, _disableautoanalysis):
        '''Initialization function'''
        self.disableautoanalysis = _disableautoanalysis
        self.expr_deque = []
        self.lock = threading.Lock()
        self.expr_count = 0
        self.start_time = None
        self.current_folder = '(none)'
        self.state = '(none)'
        self.state_time = datetime.datetime.now()
        self.exp_errors = {}
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
        # rothandle = logging.handlers.RotatingFileHandler(fname, 'a', 65535)
        rothandle = logging.handlers.RotatingFileHandler(fname, maxBytes=1024 * 1024 * 10, backupCount=5)
        cachehandle = logging.handlers.MemoryHandler(1024, logging.ERROR, rothandle)
        # fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(lineno)d] ""%(message)s")
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

    def add_exp_error(self, unique, msg):
        self.lock.acquire()
        self.exp_errors[str(unique)] = (datetime.datetime.now(), msg)
        self.lock.release()

    def get_exp_errors(self):
        self.lock.acquire()
        ret = self.exp_errors
        self.lock.release()
        return ret


class Status(xmlrpc.XMLRPC):

    """The ``Status`` class provides access to a ``CrawlLog`` through
    the XMLRPC protocol."""

    def __init__(self, logger):
        xmlrpc.XMLRPC.__init__(self)
        self.logger = logger

    def xmlrpc_current_folder(self):
        '''Return current folder'''
        return self.logger.current_folder

    def xmlrpc_time_elapsed(self):
        '''Return time_elapsed'''
        return tdelt2secs(self.logger.time_elapsed())

    def xmlrpc_prev_experiments(self):
        '''Return previous folder list'''
        return map(str, self.logger.prev_exprs())

    def xmlrpc_experiments_found(self):
        '''Return number of folders found'''
        return self.logger.expr_count

    def xmlrpc_state(self):
        '''Return state'''
        msg, dt = self.logger.get_state()
        return (msg, tdelt2secs(datetime.datetime.now() - dt))

    def xmlrpc_hostname(self):
        '''Return hostname'''
        return socket.gethostname()

    def xmlrpc_exp_errors(self):
        '''Return list of errors: (date, exp folder, error msg)'''
        exp_errors = self.logger.get_exp_errors()
        ret = [(v[0], k, v[1]) for k, v in exp_errors.iteritems()]
        return sorted(ret, key=lambda l: l[0], reverse=True)


def extract_prefix(folder):
    """Given the name of a folder storing experiment data, return the
    name of the directory under which all PGMs at a given location
    store their data."""
    return os.path.dirname(os.path.dirname(folder))


def tdelt2secs(td):
    """Convert a ``datetime.timedelta`` object into a floating point
    number of seconds."""
    day_seconds = float(td.days * 24 * 3600)
    ms_seconds = float(td.microseconds) / 1000000.0
    return day_seconds + float(td.seconds) + ms_seconds


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
                if item[1] in [RUN_STATUS_COMPLETE, RUN_STATUS_ABORT, RUN_STATUS_MISSING, RUN_STATUS_SYS_CRIT]:
                    return True
        return False

    fserves = models.FileServer.objects.all()
    ret = []
    for fs in fserves:
        l = fs.location
        rigs = models.Rig.objects.filter(location=l)
        if rigs.count() == 0:
            logger.errors.info("No rigs at this location: %s" % l.name)
        for r in rigs:
            rig_folder = os.path.join(fs.filesPrefix, r.name)
            if os.path.exists(rig_folder):
                logger.errors.debug("Checking %s" % rig_folder)
                exp_val_list = models.Experiment.objects.filter(
                    expDir__startswith=rig_folder).values_list('expDir', 'ftpStatus')
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
    '''Return number of acq files'''
    if 'tiled' in exp.rawdatastyle:
        # N.B. Hack - we check ftp status of thumbnail data only
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
    '''Return directory name'''
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
    '''Send POST to start analysis process'''
    report_name = get_name_from_json(exp, 'autoanalysisname', thumbnail_analysis)

    blockArgs = 'fromRaw'
    if (exp.getPlatform in ['s5', 'proton']) and not thumbnail_analysis:
        blockArgs = 'fromWells'  # default pipeline setting for fullchip on-instrument analysis

    params = urllib.urlencode({'report_name': report_name,
                               'do_thumbnail': "%r" % thumbnail_analysis,
                               'blockArgs': blockArgs
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
            msg = " !! Failed to start analysis.  could not connect to %s" % connection_url
            logger.errors.error(msg)
            logger.add_exp_error(exp.unique, msg)
            status_msg = "Failure to generate POST"
            f = None

    if f:
        error_code = f.getcode()
        if error_code is not 200:
            msg = " !! Failed to start analysis. URL failed with error code %d for %s" % (
                error_code, f.geturl())
            msg2 = f.read()
            logger.errors.error(msg)
            logger.errors.error(msg2)
            logger.add_exp_error(exp.unique, '%s\n Error: %s' % (msg, msg2))
            status_msg = "Failure to generate POST"

    return status_msg


def generate_updateruninfo_post(_folder, logger):
    '''Generates a POST event to update the database objects with explog data'''
    params = urllib.urlencode(
        {'datapath': _folder}
    )
    fhandle = None
    try:
        status_msg = "Generated POST"
        connection_url = 'http://127.0.0.1/rundb/updateruninfo/'
        fhandle = urllib.urlopen(connection_url, params)
    except IOError:
        logger.errors.warn('could not make connection %s' % connection_url)
        status_msg = "Failure to generate POST"
    if fhandle:
        error_code = fhandle.getcode()
        if error_code is not 200:
            msg = " !! Failed to update run info. URL failed with error code %d for %s" % (
                error_code, fhandle.geturl())
            msg2 = "%s" % "".join(fhandle.readlines())
            logger.errors.error(msg)
            logger.errors.error(msg2)
            logger.add_exp_error(_folder, '%s\n Error: %s' % (msg, msg2))
            status_msg = "Failure to generate POST"
    return status_msg


def usedPostBeadfind(exp):
    '''Return whether to use postbeadfind files in analysis'''
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

    # search for the line we need
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

    # search for the line we need
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


def get_last_file(exp):
    '''Returns name of latest acq file'''
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

    if check_for_file(exp.expDir, LOG_FINAL_BASENAME):
        # check for critical errors
        if check_for_critical(exp, LOG_FINAL_BASENAME):
            return True  # run is complete; true

        if 'tiled' in exp.rawdatastyle:
            # N.B. Hack - we check status of thumbnail data only
            expDir = os.path.join(exp.expDir, 'thumbnail')
        else:
            expDir = exp.expDir

        # check for required files
        for filename in file_list:
            if not check_for_file(expDir, filename):
                exp.ftpStatus = RUN_STATUS_MISSING
                exp.save()
                return False

        return True
    else:
        return False


def ready_to_process(exp):
    '''Returns if composite is ready to start analyzing'''
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
    '''Returns if thumbnail is ready to start analyzing'''
    if 'tiled' in exp.rawdatastyle:
        if os.path.exists(os.path.join(exp.expDir, 'thumbnail')):
            return True

    return False


def autorun_thumbnail(exp, logger):
    '''Returns whether thumbnail autorun should be executed'''
    if 'tiled' in exp.rawdatastyle:
        # support for old format
        for bs in exp.log['blocks']:
            if bs.find('thumbnail') > 0:
                arg = bs.split(',')[4].strip()
                if int(arg.split(':')[1]) == 1:
                    # logger.errors.info ("Request to auto-run thumbnail analysis")
                    return True
        # support new format
        try:
            thumb = exp.log['thumbnail_000']
            logger.errors.debug("THUMB: %s" % thumb)
            if thumb.find('AutoAnalyze:1') > 0:
                return True
        except:
            pass
    return False


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


def crawl(folders, logger):
    """Crawl over ``folders``, reporting information to the ``CrawlLog``
    ``logger``."""

    def get_expobj(_folder):
        '''Returns Experiment object associated with given folder'''
        exp_set = models.Experiment.objects.filter(unique=_folder)
        if exp_set:
            # Experiment object exists in database
            exp = exp_set[0]
            logger.errors.info("Experiment in database: %s" % (_folder))
        else:
            exp = None
        return exp

    def update_expobj_ftpcompleted(_expobj, folder):
        '''Update Experiment object with completed ftp status'''
        # Store final explog in database
        exptxt = load_log(folder, LOG_FINAL_BASENAME)
        if exptxt is not None:
            _expobj.log = json.dumps(parse_log(exptxt), indent=4)
        else:
            # explog_final exists, but is not readable yet.
            return

        # Set FTP transfer status to complete
        if _expobj.ftpStatus == RUN_STATUS_ABORT or _expobj.ftpStatus == RUN_STATUS_SYS_CRIT:
            logger.errors.info("FTP status: Aborted")
        else:
            _expobj.ftpStatus = RUN_STATUS_COMPLETE
            logger.errors.info("FTP status: Complete")

        # Save experiment object
        _expobj.save()

        return

    def update_expobj_ftptransfer(_expobj):
        '''Update Experiment object with in-transfer ftp status'''
        if _expobj.ftpStatus != RUN_STATUS_MISSING:
            _expobj.ftpStatus = get_filecount(_expobj)
            _expobj.save()
            logger.errors.info("FTP status: Transferring")
        return

    def handle_composite_report(folder, composite_exists, exp):
        '''Start composite report analysis'''
        if composite_exists:
            logger.errors.debug("composite report already exists")
            return

        # Check if auto-run for whole chip has been requested
        if exp.autoAnalyze:
            logger.errors.debug("  auto-run whole chip analysis has been requested")
            if ready_to_process(exp) or check_for_completion(exp):
                logger.errors.info("  Start a whole chip auto-run analysis job")
                generate_http_post(exp, logger)
            else:
                logger.errors.info("  Do not start a whole chip auto-run job yet")
        else:
            logger.errors.debug("  auto-run whole chip analysis has not been requested")

    def handle_thumbnail_report(folder, thumbnail_exists, exp):
        '''Start thumbnail report analysis'''
        if not 'tiled' in exp.rawdatastyle:
            logger.errors.debug("This is not a block dataset; no thumbnail to process")
            return

        if thumbnail_exists:
            logger.errors.debug("thumbnail report already exists")
            return

        # Check if auto-run for thumbnail has been requested
        if autorun_thumbnail(exp, logger):
            logger.errors.debug("auto-run thumbnail analysis has been requested")
            if ready_to_process_thumbnail(exp) or check_for_completion(exp):
                logger.errors.info("  Start a thumbnail auto-run analysis job")
                generate_http_post(exp, logger, DO_THUMBNAIL)
            else:
                logger.errors.info("  Do not start a thumbnail auto-run job yet")
        else:
            logger.errors.debug("auto-run thumbnail analysis has not been requested")
        return

    def explog_exists(folder):
        return os.path.isfile(os.path.join(folder, LOG_BASENAME))

    #-----------------------------------------------------------
    # Main
    #-----------------------------------------------------------
    if folders:
        logger.errors.info("checking %d directories" % len(folders))

    for folder in folders:
        try:
            logger.errors.info("checking directory %s" % folder)
            logger.current_folder = folder

            exp = get_expobj(folder)

            if exp is None:
                if explog_exists(folder):
                    logger.errors.info("Updating the database records: %s" % (folder))
                    generate_updateruninfo_post(folder, logger)
                else:
                    logger.errors.debug("Missing %s" % os.path.join(folder, LOG_BASENAME))
            else:
                #--------------------------------
                # Update FTP Transfer Status
                #--------------------------------
                if check_for_completion(exp):
                    update_expobj_ftpcompleted(exp, folder)
                else:
                    update_expobj_ftptransfer(exp)

                #--------------------------------
                # Handle auto-run analysis
                # Conditions for starting auto-analysis:
                # . ftpStatus is not one of the complete states
                # . no report(s) exist.
                # i.e. - if a user deletes all reports after ftp transfer is complete, do not start auto-analysis.
                #--------------------------------
                if not logger.disableautoanalysis:
                    composite_exists, thumbnail_exists = reports_exist(exp)
                    handle_composite_report(folder, composite_exists, exp)
                    handle_thumbnail_report(folder, thumbnail_exists, exp)
                else:
                    logger.errors.info("auto-analysis start has been disabled")

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


def main(args):
    '''Main function'''
    logger = CrawlLog(args.disableautoanalysis)
    logger.errors.info("Crawler Initializing")

    if logger.disableautoanalysis:
        logger.errors.info("Auto-Analysis has been disabled")

    exit_event = threading.Event()
    loopfunc = lambda: loop(logger, exit_event, settings.CRAWLER_PERIOD)
    lthread = threading.Thread(target=loopfunc)
    lthread.setDaemon(True)
    lthread.start()
    logger.errors.info("ionCrawler Started Ver: %s" % GITHASH)

    # check thread health periodically and exit if thread is dead
    # pass in reactor object below to kill process when thread dies
    l = task.LoopingCall(checkThread, lthread, logger, reactor)
    l.start(30.0)  # call every 30 second

    # start the xml-rpc server
    r = Status(logger)
    reactor.listenTCP(settings.CRAWLER_PORT, server.Site(r))
    reactor.run()


if __name__ == '__main__':
    '''Command line'''
    parser = argparse.ArgumentParser(description="ionCrawler daemon")
    parser.add_argument('--disableautoanalysis',
                        action="store_true",
                        default=False,
                        help='Disable launching analysis when new experiment data is detected')

    args = parser.parse_args()
    sys.exit(main(args))
