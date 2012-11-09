#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Ion Job Server
==============

The Job Server connects the Torrent PC Analysis frontend to the 
compute infrastructure that performs actual data analysis. It is a
tool for monitoring and managing the computational tasks required
for Ion data analysis.

The job server can either submit jobs to a DRMAA-compatible grid
resource management system (currently only Sun Grid Engine is supported),
or it can execute jobs locally by spawning analysis processes itself. The
job server's behavior is determined first by the ``SGE_ENABLED`` setting
in `settings.py`. If ``SGE_ENABLED`` is ``True``, then the job server will try
the following:

#. Check for environment variables. If any of these environment variables
   are not set, the job server will attempt to extract them from `settings.py`.

   * ``SGE_ROOT``
   * ``SGE_CELL``
   * ``SGE_CLUSTER_NAME``
   * ``SGE_QMASTER_PORT``
   * ``SGE_EXECD_PORT``

#. Import the python-drmaa package. This package can be installed using
   `setuptools`. It also requires that libdrmaa be installed. On Ubuntu,
   this can be installed with ``sudo apt-get install libdrmaa1.0``.
#. Contact the SGE Master.

If either of the first two steps fail, they will fail silently, and the
job server will revert to local-only mode. If the job server fails to
contact the SGE Master (for example, because the ``SGE_QMASTER_PORT`` is
blocked), the job server will raise an exception and terminate.

This module requires Twisted's XMLRPC server. On Ubuntu, this can be installed
with ``sudo apt-get install python-twisted``.
"""
import datetime
import json
import os
from os import path 
import re
import signal
import subprocess
import sys
import threading
import tempfile
import traceback
import logging

from twisted.web import xmlrpc,server

#for tmap queue
import urllib
import urllib2
from twisted.internet import utils, reactor
from twisted.web import xmlrpc, server
from twisted.application import service
from twisted.internet import defer, utils
from twisted.python import log
import shutil
from random import choice
import string

#import libs to zip with
import zipfile

LOG_FILENAME = '/var/log/ion/jobserver.log'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    propagate=False,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    )

__version__ = filter(str.isdigit, "$Revision: 42967 $")

class ProcessExecutionService(service.Service):
    """a queue to do one thing at a time"""
    maxActive = 1
    def __init__(self, maxActive=None):
        if maxActive is not None:
            self.maxActive = maxActive
        self.queue = defer.DeferredQueue()

    def getProcessOutput(self, *a, **k):
        return self.enqueue(utils.getProcessOutput, a, k)

    def getProcessValue(self, *a, **k):
        return self.enqueue(utils.getProcessValue, a, k)

    def getProcessOutputAndValue(self, *a, **k):
        return self.enqueue(utils.getProcessOutputAndValue, a, k)

    def startService(self):
        for i in range(self.maxActive):
            self.queue.get().addCallback(self.execute)

    def enqueue(self, callable, args, kwargs):
        d = defer.Deferred()
        self.queue.put((d, callable, args, kwargs))
        return d

    def execute(self, queueItem):
        def processNext(result):
            self.queue.get().addCallback(self.execute)
            return result

        d, callable, args, kwargs = queueItem
        callable(*args, **kwargs).addBoth(processNext).chainDeferred(d)

# local settings
try:
    sys.path.append(path.dirname(path.dirname(__file__)))
    import settings
except ImportError:
    sys.path.pop()
    sys.path.append("../")
    try:
        import settings
    except ImportError:
        sys.path.pop()


# distributed resource management

try:
    # We want to determine if we have the ability to make calls to the
    # drmaa interface. In order to do so, we first need to set environment
    # variables.
    if not settings.SGE_ENABLED:
        # bail if SGE (currently the only supported drmaa target) is
        # disabled from the site settings file.
        raise ImportError
    # Set env. vars necessary to talk to SGE. The SGE root contains files
    # describing where the grid master lives. The other variables determine
    # which of several possible grid masters to talk to.
    for k in ("SGE_ROOT", "SGE_CELL", "SGE_CLUSTER_NAME",
              "SGE_QMASTER_PORT", "SGE_EXECD_PORT", "DRMAA_LIBRARY_PATH"):
        print "DEBUG: " + k
        if not k in os.environ:
            print "DEBUG: " + str(getattr(settings,k))
            os.environ[k] = str(getattr(settings,k))
    try:
        import drmaa
    except RuntimeError:
        # drmaa will sometimes raise RuntimeError if libdrmaa1.0 is not
        # installed.
        logging.info("Unexpected error: %s" % str(sys.exc_info()))
        raise ImportError
    import atexit # provides cleanup of the session object
    try:
        HAVE_DRMAA = True
        # create a single drmaa session
        _session = drmaa.Session()
        try:
            _session.initialize()
            logging.info("DRMAA session initialized")
        except:
            logging.info("Unexpected error: %s" % str(sys.exc_info()))
        atexit.register(_session.exit)
        djs = drmaa.JobState
        # globally define some status messages
        _decodestatus = {
            djs.UNDETERMINED: 'process status cannot be determined',
            djs.QUEUED_ACTIVE: 'job is queued and active',
            djs.SYSTEM_ON_HOLD: 'job is queued and in system hold',
            djs.USER_ON_HOLD: 'job is queued and in user hold',
            djs.USER_SYSTEM_ON_HOLD: ('job is queued and in user '
                                      'and system hold'),
            djs.RUNNING: 'job is running',
            djs.SYSTEM_SUSPENDED: 'job is system suspended',
            djs.USER_SUSPENDED: 'job is user suspended',
            djs.DONE: 'job finished normally',
            djs.FAILED: 'job finished, but failed',
            }
        InvalidJob = drmaa.errors.InvalidJobException
    except drmaa.errors.InternalException:
        # If we successfully import drmaa, but it somehow wasn't configured
        # properly, we will gracefully bail by raising ImportError
        raise ImportError
except (ImportError, AttributeError):
    # drmaa import failed
    HAVE_DRMAA = False
    InvalidJob = ValueError

# regexps
SCRIPTNAME_RE = re.compile(r'^ion_analysis_(\d+)\.py$')

# utility functions
def index2scriptname(index):
    return "ion_analysis_%02d.py" % index
def index2paramsname(index):
    return "ion_params_%02d.json" % index
def safewrite(fname,s):
    outfile = None
    try:
        outfile = open(fname, 'w')
        outfile.write(s)
    finally:
        if outfile is not None:
            outfile.close()
def writeint(fname,n):
    safewrite(fname, "%d" % n)
def have_drmaa(host):
    return HAVE_DRMAA

class Analysis(object):
    """``Analysis`` objects capture the properties of a single running
    analysis job. While ``Analysis`` is an abstract base class,
    ``DRMAnalysis`` and ``LocalAnalysis`` are the implementations for
    grid mode and local mode, respectively.

    Each analysis writes out a script file (passed in with the ``script``
    argument), a parameters file (``params``),
    and a list of miscellaneous files (``files``) needed for the analysis job.
    It then spawns the analysis and
    generates a proxy object which an ``AnalysisQueue`` object can use
    to wait for a job to finish. Once the job has finished, the ``Analysis``
    object handles cleanup.

    The ``pk`` argument specifies a unique identifier for the job.

    The ``savePath`` argument specifices the working directory for the
    analysis job.

    ``Analysis`` objects are also responsible for job control. Using the
    ``suspend()``, ``resume()``, and ``terminate()`` methods, a job can be
    paused (suspended), resumed, or terminated.
    """
    ANALYSIS_TYPE = ""
    def __init__(self, name, script, params, files, savePath, pk, chipType, chips, job_type):
        """Initialize by storing essential parameters."""
        super(Analysis,self).__init__()
        self.name = name
        self.script = script
        self.params = params
        self.savePath = savePath
        self.chipType = chipType
        self.pk = pk
        self.chips = chips
        for pair in files:
            assert len(pair) == 2
            for ele in pair:
                assert isinstance(ele, (str,unicode))
        self.files = files
        self.job_type = job_type
    def get_id(self):
        """Returns the running job's ID number given by the underlying
        execution system.

        If the job server is running in grid mode,
        then the ID returned is a grid "job id," whereas if the job
        server is running in local mode, it will be a process ID
        number.
        """
        return None
    def status_string(self):
        """Return a message describing the state of the analysis. By
        default, this will be an empty string."""
        return ""
    def initiate(self, rootdir):
        """Begin an analysis, and return a proxy object on which
        the ``AnalysisQueue`` will wait."""
        return None
    def conclude(self, comm_result):
        """Clean up after an analysis has completed."""
        return False
    def suspend(self):
        """Suspend a job in progess. Returns ``True`` if the analysis
        was successfully suspended, and otherwise it returns ``False``."""
        return False
    def resume(self):
        """Resume a suspended job. Returns ``True`` if the analysis was
        successfully resumed, other it returns ``False``."""
        return False
    def terminate(self):
        """Terminate a job in progress. Returns ``True`` if the analysis
        was successfully terminated, otherwise it returns ``False``."""
        return False
    def _get_script_index(self, adir):
        matches = [SCRIPTNAME_RE.match(fn) for fn in os.listdir(adir)]
        extract_valid = lambda acc,x: x and acc.append(int(x.groups(1)[0]))
        inds = reduce(extract_valid, matches, [])
        return (inds and max(inds) + 1) or 0
    def _write_out(self,adir):
        """Dump out files needed for the analysis into directory 'adir'."""
        # make sure we have params as JSON text
        os.umask(0002)
        if isinstance(self.params,dict):
            self.params = json.dumps(self.params)
        if not path.isdir(adir):
            try:
                os.makedirs(adir)
            except:
                logging.error("Analysis cannot start. Failed to create directory: %s." % adir)
                logging.debug(traceback.format_exc())
                return None, None
        # find the appropriate script index, in case we are re-running
        script_index = self._get_script_index(adir)
        script_fname = path.join(adir,index2scriptname(script_index))
        params_fname = path.join(adir,index2paramsname(script_index))
        # dump out script and parameters
        safewrite(script_fname,self.script)
        os.chmod(script_fname, 0775)
        safewrite(params_fname,self.params)
        for name,content in self.files:
            safewrite(path.join(adir,name),content)
        manifest = "\n".join(name for name,content in self.files)
        safewrite(path.join(adir,"manifest.txt"), manifest)
        return script_fname,params_fname

def tolerate_invalid_job(fn):
    """Decorator to catch invalid job references and handle them silently."""
    def ret(*args):
        try:
            result = fn(*args)
        except InvalidJob:
            result = False
        return result
    ret.func_name = fn.func_name
    ret.__doc__ = fn.__doc__
    return ret

class DRMAnalysis(Analysis):
    """``DRMAnalysis`` implements analysis on Sun Grid Engine."""
    ANALYSIS_TYPE = "grid"
    class DRMWaiter(object):
        """Wrapper around a job id to allow the AnalysisQueue to .communicate()
        with a grid job as if it were a process."""
        def __init__(self, jobid, parent):
            self.jobid = jobid
            self.parent = parent
        def communicate(self):
            timeout = drmaa.Session.TIMEOUT_WAIT_FOREVER
            try:
                self.parent.retval = _session.wait(self.jobid,timeout)
            except:
                self.parent.terminated = True
    def __init__(self, name, script, params, files, savePath, pk, chipType, chips, job_type):
        super(DRMAnalysis,self).__init__(name,script,params,files,savePath,pk,chipType,chips,job_type)
        self.retval = None
        self.jobid = None
        self.terminated = False
    def get_sge_params(self, chip_to_slots,chipType):
        ret = '-pe ion_pe 1'
#       ret = '-pe ion_pe 1 -l h_vmem=10000M'
        for chip,args in chip_to_slots.iteritems():
            if chip in chipType:
                ret = args.strip()
                return ret
        return ret
    def initiate(self, rootdir):
        """Spawn an analysis on the grid.

        Instructs the grid to capture the analysis script's standard output
        and standard error in two files called ``drmaa_stdout.txt`` and
        ``drmaa_stderr.txt``, respectively.
        """
        adir = path.join(self.savePath)
        script_fname,params_fname = self._write_out(adir)
        if script_fname == None:
            return None
        jt = _session.createJobTemplate()
        qname = 'tl.q'
        if self.job_type == 'PairedEnd':
            qname = 'all.q'
        #SGE
        jt.nativeSpecification = "%s -w w -q %s" % (self.get_sge_params(self.chips,self.chipType),qname)
        #TORQUE
        #jt.nativeSpecification = ""
        jt.remoteCommand = "python"
        jt.workingDirectory = adir
        jt.outputPath = ":" + path.join(adir, "drmaa_stdout.txt")
        #jt.errorPath = ":" + path.join(adir, "drmaa_stderr.txt")
        jt.args = (script_fname, params_fname)
        jt.joinFiles = True # Merge stdout and stderr
        self.jobid = _session.runJob(jt)
        ret = self.DRMWaiter(self.jobid, self)
        _session.deleteJobTemplate(jt)
        return ret
    def conclude(self, comm_result):
        """Clean up once a grid job finishes.
        
        If the job completed successfully, writes out "1" into
        a file ``status.txt`` in the job's working directory. THIS IS A BUG,
        and instead should write out the job's exit status.

        If the job was terminated, then the method writes "-1" instead.
        """
        outpath = path.join(self.savePath,"status.txt")
        if self.terminated:
            writeint(outpath, -1)
            return -1
        else:
            assert self.retval is not None
            retcode = int(self.retval.hasExited)
            writeint(outpath, retcode)
            return retcode == 0
    def _running(self):
        return (self.jobid is not None) and (self.retval is None)
    @tolerate_invalid_job
    def suspend(self):
        """Suspends the job by issuing a command to the grid."""
        if not self._running():
            return False
        _session.control(self.jobid,drmaa.JobControlAction.SUSPEND)
        return True
    @tolerate_invalid_job
    def resume(self):
        """Resumes the job by issuing a command to the grid."""
        if not self._running():
            return False
        _session.control(self.jobid,drmaa.JobControlAction.RESUME)
        return True
    @tolerate_invalid_job
    def terminate(self):
        """Terminates the job by issuing a command to the grid."""
        logging.debug("DRMAA terminate job")
        if not self._running():
            return False

        # todo, read joblist from database instead of a file
        joblistfile = os.path.join(self.savePath, 'job_list.txt')
        logging.debug("DRMA def terminate(self): %s " % joblistfile)
        if os.path.exists(joblistfile):
            fd = open(joblistfile)
            for line in fd:
                blockjobid = line.rstrip('\n')
                try:
                    logging.debug("terminate job %s in status %s" % (blockjobid,_session.jobStatus(blockjobid)))
                    _session.control(blockjobid,drmaa.JobControlAction.TERMINATE)
                except Exception,e:
                    logging.error(str(e))
                    pass
            fd.close()

        _session.control(self.jobid,drmaa.JobControlAction.TERMINATE)
        return True
    def get_id(self):
        return self.jobid
    def status_string(self):  
        unknown = "(unknown)"
        try:
            jid = _session.jobStatus(self.jobid)
        except InvalidJob:
            return unknown
        return _decodestatus.get(jid, unknown)

class LocalAnalysis(Analysis):
    """Describes a local, non-grid analysis. Runs by spawning a process."""
    ANALYSIS_TYPE = "local"
    def __init__(self, name, script, params, files, savePath, pk, chipType,chips,job_type):
        super(LocalAnalysis,self).__init__(name,script,params,files,savePath,pk,chipType,chips,job_type)
        self.proc = None  
    
    def initiate(self, rootdir):
        """Initiates a local job by spawning a process directly.

        Writes the process's PID to a file named ``pid`` in the
        job's working directory. The returned proxy object is a
        `subprocess.Popen` object.
        """
        if self.proc is not None:
            # we shouldn't run one analysis object more than once
            return None
        # determine where we will write out analysis files, and create it
        adir = path.join(self.savePath)
        script_fname,params_fname = self._write_out(adir)
        args = (sys.executable, script_fname, params_fname)
        # create process
        self.proc = subprocess.Popen(args, cwd=adir)
        # save PID
        writeint(path.join(self.savePath,"pid"), self.proc.pid)
        return self.proc
    def conclude(self, comm_result):
        """
        Concludes by writing the process's return code to ``status.txt`` in
        the job's working directory.
        """
        retcode = self.proc.returncode
        assert retcode is not None
        # save the return code to file in order to determine if the analysis
        # completed successfully or not
        writeint(path.join(self.savePath,"status.txt"), retcode)
        return retcode == 0
    def _running(self):
        """Determine if the analysis process is running."""
        return self.proc and self.proc.returncode is None
    def suspend(self):
        """Suspends the process by sending it ``SIGSTOP``."""
        if not self._running():
            return False
        try:
            self.proc.send_signal(signal.SIGSTOP)
        except:
            return False
        return self._running()
    def resume(self):
        """Resumes the process by sending it ``SIGCONT``."""
        if not self._running():
            return False
        try:
            self.proc.send_signal(signal.SIGCONT)
        except:
            return False
        return True       
    def terminate(self):
        """Terminates the process by calling ``subprocess.Popen``'s
        ``terminate()`` method.
        """
        if not self._running():
            return False
        self.proc.terminate()
        return True
    def get_id(self):
        if self.proc is None:
            return None
        return self.proc.pid

class AnalysisQueue(object):
    """
    The ``AnalysisQueue`` is responsible for managing and monitoring
    all jobs the job server is currently running. It is intended to be a
    singleton object.

    It maintains a queue
    of analyses (literally ``Analysis`` objects) waiting to be run. It
    operates a thread that sequentially pops analyses from the queue and
    runs them each in a separate thread.

    The process of running each analysis consists of three parts. It is
    implemented in the ``run_analysis`` method.
    
    #. First, the ``AnalysisQueue`` acquires a lock, and calls the
       ``Analysis`` object's ``initiate()`` method.
    #. If the call to ``initiate()`` returns a valid proxy object,
       the ``AnalysisQueue`` releases the lock and calls the proxy's
       ``wait()`` method.
    #. Once the proxy's ``wait()`` returns, the ``AnalysisQueue`` again
       acquires a lock and calls
       the analysis object's ``conclude()`` method to clean up.

    The reason for acquiring a lock is to allow the ``AnalysisQueue`` to keep
    track of which analyses are running, and to make it easier to implement
    queuing in the future. For example, it would be straightforward to
    have the ``AnalysisQueue``'s main thread wait until there are less than
    N analyses running before initiating another.
    """
    def __init__(self, rootdir):
        if rootdir.startswith("../"):
            rootdir = path.join(os.getcwd(), rootdir)
        self.cv = threading.Condition()
        self.exit_event = threading.Event()
        self.q = []
        self.monitors = []
        self.running = {}
        self.rootdir = rootdir
        self.start_time = None
    def is_running(self,pk):
        """Determine if an analysis identified by ``pk`` is in progress."""
        return pk in self.running
    def run_analysis(self,a):
        """Spawn a thread which attempts to start an analysis."""
        def go():
            # acquire a lock while initiating
            self.cv.acquire()
            try:
                waiter = a.initiate(self.rootdir)
            finally:
                self.cv.release()
            if waiter is not None:
                # analysis was successfully initiated
                logging.info("%s successfully started" % str(a.name))
                assert a.pk not in self.running
                self.running[a.pk] = a
                # wait for analysis to conclude
                comm_result = waiter.communicate()
                # acquire lock before terminating
                self.cv.acquire()
                try:
                    a.conclude(comm_result)
                finally:
                    if a.pk in self.running:
                        del self.running[a.pk]
                    self.cv.release()
                logging.info("%s completed" % str(a.name))
            else:
                # bail, initiation failed
                logging.error("%s failed to start" % str(a.name))
                return
        tr = threading.Thread(target=go)
        tr.setDaemon(True)
        self.monitors.append(tr)
        tr.start()
        return tr
    def loop(self):
        """Remove un-initiated analyses from the analysis queue, and
        run them."""
        self.start_time = datetime.datetime.now()
        def _loop():
            while not self.exit_event.isSet():
                self.cv.acquire()
                while len(self.q) == 0:
                    self.cv.wait()
                    if self.exit_event.is_set():
                        logging.info ("Main loop exiting")
                        return # leave loop if we're done
                a = self.q.pop(0)
                self.cv.release()
                self.run_analysis(a)
        tr = threading.Thread(target=_loop)
        tr.setDaemon(True)
        tr.start()
        return tr
    def add_analysis(self, a):
        """Add an analysis to the queue."""
        self.cv.acquire()
        self.q.append(a)
        self.cv.notify()
        self.cv.release()
        logging.info("Added analysis %s" % a.name)
    def stop(self):
        """Terminate the main loop."""
        self.exit_event.set()
        self.cv.notify()
    def status(self, save_path, pk):
        """Determine the status of an analysis identified by 'pk' running
        at 'save_path'."""
        self.cv.acquire()
        try:
            if pk in self.running:
                ret = (True, self.running[pk].status_string())
            else:
                fname = path.join(save_path, "status.txt")
                if not path.exists(fname):
                    ret = (False, "Unknown")
                else:
                    infile = open(fname)
                    retcode = int(infile.read())
                    infile.close()
                    if retcode == 0:
                        ret = (True,"Completed Successfully")
                    else:
                        ret = (False,"Failed")
        finally:
            self.cv.release()
        return ret
    def all_jobs(self):
        """Return a list of (pk,proxy) for all currently running jobs."""
        return self.running.items()
    def n_jobs(self):
        """Return the number of jobs currently running."""
        return len(self.running)
    def uptime(self):
        """Return the amount of time the ``AnalysisQueue`` has been running."""
        if self.start_time is None:
            return 0
        else:
            diff = datetime.datetime.now() - self.start_time
            seconds = float(diff.days*24*3600)
            seconds += diff.seconds
            seconds += float(diff.microseconds)/1000000.0
            return seconds
    def control_job(self,pk,signal):
        logging.debug("Analysis queue control_job: %s %s" % (pk , signal))
        """Terminate, suspend, or resume a job."""
        self.cv.acquire()
        try:
            if not self.is_running(pk):
                ret = (False, "not running")
            else:
                a = self.running[pk]
                fn = {"term":a.terminate,
                      "stop":a.suspend,
                      "cont":a.resume,}.get(signal.lower())
                if fn is None:
                    ret = (False, "invalid signal")
                else:
                    ret = (fn(), "executed")
        finally:
            self.cv.release()
        return ret
    def best_analysis_class(self):
        if have_drmaa(""):
            return DRMAnalysis
        else:
            return LocalAnalysis

class AnalysisServer(xmlrpc.XMLRPC):
    """Remote procedure call server that links the database with the
    analysis queue.

    Built on top of Twisted's XMLRPC server.
    """
    def __init__(self, analysis_queue, execService):
        xmlrpc.XMLRPC.__init__(self)
        self.q = analysis_queue
        self.execService = execService

    def xmlrpc_updatestatus(self,
            primarykeyPath,
            status,
            reportLink):
        
        from ion.reports import uploadMetrics
        try:
            uploadMetrics.updateStatus(primarykeyPath,status,reportLink)
        except:
            logging.error(traceback.format_exc())
            return traceback.format_exc()
        return 0
    
    def xmlrpc_uploadmetrics(self,
            tfmapperstats_outputfile,
            procPath,
            beadPath,      
            filterPath, 
            alignmentSummaryPath,
            peakOut,
            QualityPath,
            BaseCallerJsonPath,
            pePath,
            primarykeyPath,
            uploadStatusPath,
            STATUS,
            reportLink):
        """Upload Metrics to the database"""

        from ion.reports import uploadMetrics
        try:
            return_message = uploadMetrics.writeDbFromFiles(
                tfmapperstats_outputfile,
                procPath,
                beadPath,
                filterPath,
                alignmentSummaryPath,
                STATUS,
                peakOut,
                QualityPath,
                BaseCallerJsonPath,
                pePath,
                primarykeyPath,
                uploadStatusPath)

            # this will replace the five progress squares with a re-analysis button
            uploadMetrics.updateStatus(primarykeyPath, STATUS, reportLink)
        except:
            logging.error(traceback.format_exc())
            return traceback.format_exc()

        return return_message


    def xmlrpc_uploadanalysismetrics(self, beadPath, primarykeyPath):
        logging.info("Testing Upload Analysis Metrics")
        print(beadPath)
        from ion.reports import uploadMetrics
        try:
            message = uploadMetrics.updateAnalysisMetrics(beadPath, primarykeyPath)
        except Exception as err:
            logging.error("Upload Analysis Metrics failed: %s", err)
        logging.info("Completed Upload Analysis Metrics")
        return "What an intriguing result: " + message


    def xmlrpc_submitjob(self,jt_nativeSpecification, jt_remoteCommand,
                           jt_workingDirectory, jt_outputPath,
                           jt_errorPath, jt_args, jt_joinFiles):
        jt = _session.createJobTemplate()
        jt.nativeSpecification = jt_nativeSpecification
        jt.remoteCommand = jt_remoteCommand
        jt.workingDirectory = jt_workingDirectory
        jt.outputPath = jt_outputPath
        jt.errorPath = jt_errorPath
        jt.args = jt_args
        jt.joinFiles = jt_joinFiles
        jobid = _session.runJob(jt)
        _session.deleteJobTemplate(jt)
        return jobid
    def xmlrpc_jobstatus(self, jobid):
        """Get the status of the job"""
        try:
            logging.debug("xmlrpc jobstatus for %s" % jobid)
            status = _session.jobStatus(jobid)
        except:
            logging.error(traceback.format_exc())
            status = "DRMAA BUG"
        return status
    def xmlrpc_startanalysis(self,name,script,parameters,files,savePath,pk,chipType,chips,job_type):
        """Add an analysis to the ``AnalysisQueue``'s queue of waiting
        analyses."""
        logging.debug("Analysis request received: %s" % name)
        ACls = self.q.best_analysis_class()
        la = ACls(name,script,parameters,files,savePath,pk,chipType,chips,job_type)
        self.q.add_analysis(la)
        return name
    def xmlrpc_status(self, save_path, pk):
        """Get the status of the job specified by ``pk`` from the 
        ``AnalysisQueue``."""
        return self.q.status(save_path, pk)
    def xmlrpc_n_running(self):
        """Return the number of jobs the ``AnalysisQueue`` is currently
        running."""
        return self.q.n_jobs()
    def xmlrpc_uptime(self):
        """Return the ``AnalysisQueue``'s uptime."""
        logging.debug("uptime checked")
        return self.q.uptime()
    def xmlrpc_running(self):
        """Return status information about all jobs currently running."""
        items = self.q.all_jobs()
        ret = []
        for pk,a in items:
            ret.append((a.name, a.get_id(), a.pk, a.ANALYSIS_TYPE,a.status_string()))
            logging.debug("Name:%s JobId:%s PK:%s State:'%s'" % (a.name,a.get_id(),a.pk,a.status_string()))
        return ret       
    def xmlrpc_control_job(self,pk,signal):
        """Send the given signal to the job specified by ``pk``."""
        logging.debug("xmlrpc_control_job: %s %s" % (pk , signal))
        return self.q.control_job(pk,signal)
    def xmlrpc_test_path(self,path):
        """Determine if ``path`` is readable and writeable by the job
        server."""
        return os.access(path, os.R_OK | os.W_OK)
    
    def xmlrpc_tmap(self, id, fasta, short, long , version, read_sample_size, read_exclude_length, index_version):
        """ Provides a way to kick off the tmap index generation
            this should spawn a process that calls the build_genome_index.pl script
            it may take up to 3 hours. 
            The django server should contacts this method from a view function
            When the index creation processes has exited, cleanly or other wise
            a callback will post to a url that will update the record for the library data
            letting the genome manager know that this now exists
            until then this genome will be listed in a unfinished state.
        """
        
        build_genome_index = '/usr/local/bin/build_genome_index.pl'
        logging.debug("tmap xml-rpc established")

        tmap_version = settings.TMAP_VERSION
        logging.debug("tmap version:" + tmap_version)
        
        args = [
            "--auto-fix",
            "--fasta", fasta,
            "--genome-name-short", short,
            "--genome-name-long", long,
            "--genome-version", version
        ]
        if read_sample_size:
            args.append("--read-sample-size")
            args.append(read_sample_size)

        uuid_path = ''.join([choice(string.letters + string.digits) for i in range(10)])

        #move the fasta to a guaranteed uniq path
        temp_path = "/results/referenceLibrary/temp/"
        temp_path_uniq = tempfile.mkdtemp(suffix=short,dir=temp_path)
        os.chmod(temp_path_uniq, 0777)
        logging.debug("Temp path created " + temp_path_uniq)

        try:
            os.rename(os.path.join(temp_path, fasta) , os.path.join(temp_path_uniq,fasta))
        except IOError:
            logging.debug("There was an error moving the temp files to a more unique directory")


        #TODO: break this out into another function

        #TODO: check for gzip as well
        
        #If the uploaded  file was a zip file tell build genome index to uncompress it
        if zipfile.is_zipfile(os.path.join(temp_path_uniq, fasta)):
            #here let us open the zip file to find the fasta name
            zf = zipfile.ZipFile(os.path.join(temp_path_uniq, fasta), mode='r', allowZip64=True)
            file_list = zf.infolist()

            #removed hidden os x metadata
            file_list = [file_in_zip.filename for file_in_zip in file_list if not file_in_zip.filename.startswith("__")]

            if len(file_list) != 1:
                logging.debug("Error with zip file:" + str(fasta) + " zip contains " + str(len(file_list)) + " files. Should have only 1 ")
                return False, "Error with zip file, zip must contain only one fasta file."

            #if the zip has only 1 file, excluding metadata feed that name to the index builder
            zipped_fasta_name = file_list[0]
            args.append('-c')
            args.append(zipped_fasta_name)
            
            logging.debug("Unzipped File" + str(fasta) + " Found zipped fasta " + str(zipped_fasta_name) )

        logging.debug(args)
        query_args = { 'status':'started', 'index_version' : tmap_version }
        logging.debug(query_args)
        encoded_args = urllib.urlencode(query_args)
        url = 'http://localhost/rundb/genomestatus/' + id
        logging.debug("Attempting to GET the url: " + url)
        try:
            url_read = urllib2.urlopen(url, encoded_args).read()
            logging.debug(url_read)
        except:
            logging.debug("failed the update genome status")

        d = execService.getProcessOutputAndValue(build_genome_index, args, path=temp_path_uniq, env=os.environ)
        d.addCallback(finished, id, fasta, short, temp_path_uniq, tmap_version, read_sample_size)
        logging.debug("genome building was started")

        return True, "Genome Building started"

def finished(val, id, temp_fasta, short, temp_path, tmap_version, read_sample_size):
    """This is a callback function that will be called when the build_genome_index.pl
    script returns"""
    
    if val[2] == 0:

        logging.debug("genome building returned successful with 0")
        #open a url that will set the status of the genome index as done.

        query_args = { 'status':'created', 'enabled': True, 'index_version' : tmap_version }

        encoded_args = urllib.urlencode(query_args)
        url = 'http://localhost/rundb/genomestatus/' + id
        url_read = urllib2.urlopen(url, encoded_args).read()
        logging.debug(url_read)

        os.umask(0002)
        shutil.move(os.path.join(temp_path, short) , settings.TMAP_DIR)
        shutil.rmtree(temp_path)
        logging.debug("removing : " + str(temp_path))

        #TODO: if it worked then add the genome sample size length
        
    else:
        logging.debug("genome building returned with an error: " +  str(val) )
        #don't provide enabled which will implicitly disable the genome
        query_args = { 'status':'error', 'verbose_error' : json.dumps(val), 'index_version' : tmap_version }
        encoded_args = urllib.urlencode(query_args)
        url = 'http://localhost/rundb/genomestatus/' + id
        url_read = urllib2.urlopen(url, encoded_args).read()
        logging.debug(url_read)
        
        shutil.rmtree(temp_path)

if __name__ == '__main__':
    
    logging.info("ionJobServer Started Ver: %s" % __version__)

    execService = ProcessExecutionService()

    aq = AnalysisQueue(settings.ANALYSIS_ROOT)
    aq.loop()
    
    r = AnalysisServer(aq, execService)
    reactor.callLater(0, r.execService.startService)
    reactor.listenTCP(settings.JOBSERVER_PORT,server.Site(r))
    reactor.run()

