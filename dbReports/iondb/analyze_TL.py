#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

#####################################################################
#
# Analysis implementation details
#
#####################################################################

 
def initreports():
    """Performs initialization for both report writing and background
    report generation."""
    print "INIT REPORTS"
    #
    # Begin report writing
    #
    prefix = os.getcwd()
    if not path.isdir(prefix):
        raise ValueError, "'%s' directory does not exist." % prefix
    os.mkdir(SCRATCH_DIR)
    writers.init_writers(SCRATCH_DIR, not HAVE_MP)
    #if HAVE_MP:
        # I want to have reporting on a single, separate process instead of on
        # another thread that spawns processes repeatedly.  I catch reporting
        # exceptions so that spurious problems don't bring down the final report.
    #    def reportPoller(killSignal):
    #        while killSignal.value == 0:
    #            start = time.time()
    #            try:
    #                livereports.write_report(False)
    #            except:
    #                print >> sys.stderr, "Caught %s" % str(sys.exc_info())
    #                
    #            #print "\nPID is %d... that took %.2f sec, sleeping now\n" % (os.getpid(), time.time() - start)
    #            time.sleep(10)
    #        #print "\nexiting report writing process, goodbye!\n"
    #    killSignal = multiprocessing.Value("i", 0)
    #    p = multiprocessing.Process(target = reportPoller, args=(killSignal,))
    #    p.daemon = True
    #    p.start()
    #    return p, killSignal
    return None,None

def killReportWriter(*args):
    global killSignal
    global reportPoller
    print "sending termination signal to reporting process"
    killSignal.value = 1 # We could use other methods for signaling, like a pipe or a file.
    reportPoller.join()

    print "writing report one last time for good luck..."
    start = time.time()
    #livereports.write_report()
    print "it took %.2f sec" % (time.time() - start)

def makeReport(env):
    outfile = open('report.html', 'w')
    if env['blast'] or env['whole_chip']:
        outfile.write(("<html><head><title>Please Redirect</title></head><body>"
        "<a href=Default_Report.php>Click Here</a></body></html>"))
    else:
        outfile.write(("<html><head><title>Please Redirect</title></head><body>"
        "<a href=Default_Report.php>Click Here</a></body></html>"))
    outfile.close()

def makeHistogram():
    pylab.clf()
    if os.path.exists("blastn.histo.dat"):
        run = True
        try:
            h = pylab.load("blastn.histo.dat")
        except Exception:
            run = False
        if len(h) > 0 and run:
            xaxis = [i for i in range(len(h))]   
            ymax = max(h) + 10
            xlen = len(h) + 10
            pylab.gcf().set_size_inches((8,4))
            pylab.bar(xaxis,h, facecolor = "blue", align = 'center')
            pylab.xlabel("Correct Bases - Incorrect Bases")
            pylab.title("Match-Mismatch Reads Relaxed")
            pylab.axis([0,xlen,0,ymax])
            pylab.savefig("alignment_histogram.png")
            pylab.clf()

            if len(h) > 20:
                ymax = max(h[20:]) + 10
                xlen = len(h[20:]) + 20 + 10  #add 20 to get to start and then add 10 more at the end
                pylab.gcf().set_size_inches((8,4))
                pylab.bar(xaxis[20:],h[20:], facecolor = "blue", align = 'center')
                pylab.xlabel("Correct Bases - Incorrect Bases")
                pylab.title("Match-Mismatch Reads Relaxed Filtered")
                pylab.axis([0,xlen,0,ymax])
                pylab.savefig("alignment_histogram_truncated.png")
                pylab.clf()

    if os.path.exists("megablast.histo.dat"):
        run=True
        try:
            h = pylab.load("megablast.histo.dat")
        except Exception:
            run = False
        if len(h) > 0 and run:
            xaxis = [i for i in range(len(h))]  
            ymax = max(h) + 10
            xlen = len(h) + 10
            pylab.gcf().set_size_inches((8,4))
            pylab.bar(xaxis,h, facecolor = "blue", align = 'center')
            pylab.xlabel("Correct Bases - Incorrect Bases")
            pylab.title("Match-Mismatch Reads Strict")
            pylab.axis([0,xlen,0,ymax])
            pylab.savefig("alignment_histogram_mega.png")
            pylab.clf()

            if len(h) > 20:
                ymax = max(h[20:]) + 10
                xlen = len(h[20:]) + 20 + 10
                pylab.gcf().set_size_inches((8,4))
                pylab.bar(xaxis[20:],h[20:], facecolor = "blue", align = 'center')
                pylab.xlabel("Correct Bases - Incorrect Bases")
                pylab.title("Match-Mismatch Reads Strict Filtered")
                pylab.axis([0,xlen,0,ymax])
                pylab.savefig("alignment_histogram_mega_truncated.png")
                pylab.clf()

    #hooks for q17 plot filename is set to change and expected 
    if os.path.exists("q10.dat"):
        q = pylab.load("q10.dat")
	qplot = QPlot2(q, 10, expected='TCAGGTGAGTCGAGCGAGT')
	qplot.render()
	pylab.savefig("Q10.png")
	pylab.clf()
    #hooks for q17 plot
    if os.path.exists("q17.dat"):
        q = pylab.load("q17.dat")
	qplot = QPlot2(q, 17, expected='TCAGGTGAGTCGAGCGAGT')
	qplot.render()
	pylab.savefig("Q17.png")
	pylab.clf()

def runFullChip(env):
    pathprefix = env["prefix"]
    print "RUNNING FULL CHIP ANALYSIS"   
    command = "Analysis -r1 %s -c %s" % (pathprefix, env["whole_chip_cycles"])
    fin = Popen(command, shell=True, stdout=PIPE)    
    folder = fin.stdout.readlines()[0]
    folder = folder.strip()
    folder = folder[1:]
    folder = folder.split("/")
    folder = folder[1]
    #make keypass.fasq file -c(cut key) -k(key flows)
    com = "SFFRead -q keypass.fastq -c -k TCAG %s/rawlib.sff > keypass.summary" % folder
    os.system(com)
    #call whole chip metrics script (as os.system because all output to file)
    com = "analyzeReads %s" % env["libraryName"]
    os.system(com)
    shutil.copy("%s/bfmask.stats" % folder, os.getcwd())
    shutil.copy("%s/rawlib.sff" % folder, os.getcwd())
    rawlibpath = "%s/rawlib.sff" % folder
    try:
        blast_to_ionogram.CreateIonograms("blastn.output.xml", "TopBeads", "rawlib.sff")
    except Exception:
        pass
    try:
        blast_to_ionogram.CreateIonograms("megablast.output.xml","TopMBeads", "rawlib.sff")
    except Exception:
        pass
    makeHistogram()


def start():

    import  os
    import tempfile

# matplotlib/numpy compatibility
    os.environ['HOME'] = tempfile.mkdtemp()
    from matplotlib import use
    use("Agg")

    import atexit
    import datetime
    from    os import path
    import  shutil
    import socket
    import subprocess
    import sys
    import time
    import numpy
    import pickle
    from subprocess import *

#if __name__ == "__main__":
    # The module wasn't imported, so we're not running through SCons.
    # Attempt to run through SCons, or print a usage message and exit.
#    if len(sys.argv) < 2:
#        print "usage: python analyze_N.py params_N.json OR scons -f analyze_N.py params=params_N.json -j 4"
#    else:
#        if os.name == 'nt':
#            imgname = 'scons.bat'
#        else:
#            imgname = 'scons'
#        # os.system seems to work better here than subprocess.call
#        sys.exit(os.system("%s -f %s params=%s --debug=stacktrace --debug=time -k" % (imgname, sys.argv[0], sys.argv[1])))
#    sys.exit(0)


# Print useful info to the console. This is the stuff that shows up in the
# asterisk-bounded box at the top of the console as the analysis starts.
# We want to print this info before importing any analysis specific modules
# in order to use the output for debugging should one of the imports break
# the analysis.
    python_data = [sys.executable, sys.version, sys.platform, socket.gethostname(),
               str(os.getpid()), os.getcwd(),
               os.environ.get("JOB_ID", '[Stand-alone]'),
               os.environ.get("JOB_NAME", '[Stand-alone]'),
               datetime.datetime.now().strftime("%H:%M:%S %b/%d/%Y")
               ]
    python_data_labels = ["Python Executable", "Python Version", "Platform",
                      "Hostname", "PID", "Working Directory", "SGE Job ID",
                      "SGE Job Name", "Analysis Start Time"]
    _MARGINS = 4
    _TABSIZE = 4
    _max_sum = max(map(lambda (a,b): len(a) + len(b), zip(python_data,python_data_labels)))
    _info_width = _max_sum + _MARGINS + _TABSIZE
    print '*'*_info_width
    for d,l in zip(python_data, python_data_labels):
        spacer = ' '*(_max_sum - (len(l) + len(d)) + _TABSIZE)
        print '* %s%s%s *' % (str(l),spacer,str(d).replace('\n',' '))
    print '*'*_info_width

# Import analysis-specific packages. We try to follow the from A import B
# convention when B is a module in package A, or import A when
# A is a module.
    from ion.analysis import cafie,sigproc
    from ion.dirty import weka
    from ion.fileformats import sff
    from ion.reports import id2xy, blast_to_ionogram
    from ion.utils import template as Template, beadmask as Bm
    from ion.web.db import shortcuts,writers
    from ion.web.db.writers import livereports
    from ion.web.db.writers import uploaders, transmit
    from ion.reports import fasta
    from ion.logging import exnhandler
    from ion.utils.filenames import escape_name, unescape_name
    from ion.reports.plotters import *


    import csv
    import datetime
    import pylab
    import re
    import math
    import random
    import simplejson as json
    from scipy import cluster, signal, linalg
    import traceback
    import threading


# This is a dirty hack to work around a race condition in Popen.wait.
# See: http://bugs.python.org/issue1236.
#
    origWaitPID = os.waitpid
    def newWaitPID(*args):
        try:
            return origWaitPID(*args)
        except OSError:        
            print >> sys.stderr, "Silenced %s" % str(sys.exc_info())
            return (0,0)

    os.waitpid = newWaitPID

#####################################################################
#
# Configuration code for SCons integration and parameter retrieval.
#
#####################################################################

# Create a SCons environment. This is a datastructure that can't be
# modified once SCons actually starts executing the build tasks. It
# gets passed to all analysis functions, so it is the best way to store
# critical metadata for the analysis.
#
# Also notice how it's Environment(), not scons.Environment(). SCons magically
# imports itself when running the analysis. So all symbols defined at the module
# level in SCons are also defined in the analysis.
#env = Environment()
 
    EXTERNAL_PARAMS = {}
    env = {}

# Load the analysis parameters and metadata from a json file passed in on the
# command line with --params=<json file name>
# we expect this loop to iterate only once. This is more elegant than
# trying to index into ARGUMENTS.
    file = open(sys.argv[1], 'r')
    EXTERNAL_PARAMS = json.loads(file.read())
    file.close()
    for k,v in EXTERNAL_PARAMS.iteritems():
        if isinstance(v, unicode):
            EXTERNAL_PARAMS[k] = str(v)
    env["params_file"] = 'params_adeno.json'
 
 
# Assemble some basic quantities and strings from the external parameters.
# These will be added to the SCons environment below.

# Where the raw data lives (generally some path on the network)
    pathprefix = str( str(EXTERNAL_PARAMS['path_prefix']) + str(EXTERNAL_PARAMS['data_directory']))
# The order in which reagents flow across the chip, generally 'TACG'
    flowOrder = EXTERNAL_PARAMS["floworder"]
# no longer used? -jdh
    start = int(EXTERNAL_PARAMS['start_file'])
# The number of times all four reagents are flown over the chip
    flows = int(EXTERNAL_PARAMS['cycles'])
# The numbers that appear in the names of files we're interested in. If
# acq_0003.dat is a file we want to analyze, the integer 3 will appear in this
# list.
#note that experiments may or may not include a wash image step per cycle, the var 'hasWashFlow', when set to 1, will allow us to ignore those images
    hasWashFlow = int(EXTERNAL_PARAMS['hasWashFlow'])


# We now fill the environment with all the metadata we'll need for the analysis

# Absolute network path to the file we use for beadfind. May be nonsense if
# a bead mask path is provided by the user
    env["beadfindfile"] = str(str(EXTERNAL_PARAMS['path_prefix']) + EXTERNAL_PARAMS['beadfind_file'])
# Absolute network path to the directory containing raw data files
    env["prefix"] = pathprefix
# not used

# the fraction of carry forward for which to correct, in [0,1)
    env["cf"] = float(EXTERNAL_PARAMS.get("cf", 0.0))
# the fraction of incomplete extension for which to correct, in [0,1)
    env["ie"] = float(EXTERNAL_PARAMS.get("ie", 0.0))
#droop, pol loss
    env["droop"] = float(EXTERNAL_PARAMS.get("droop", 0.0))
# number of cycles, four flows per cycle
    env["cycles"] = flows
# the last frame to load, temporarily not in use. We can decrease the time
# the analysis takes to run by loading less data, so this will be important
    env["lastframe"] = 0
# URL to which to upload reporting data
    env["return"] = EXTERNAL_PARAMS["return"]
# A unique (with high probability) string identifying the analysis to the
# database
    env["idhash"] = EXTERNAL_PARAMS["idhash"]



#library name for blasting
    env["libraryName"] = EXTERNAL_PARAMS.get("libraryName", "adenovirus")
#Do we want to blast after the run?
    env["blast"] = EXTERNAL_PARAMS.get("blast", False)
#kick off whole chip analysis from commercial pipeline
    env["whole_chip"] = EXTERNAL_PARAMS.get("whole_chip", False)
#whole chip cycles
    env["whole_chip_cycles"] = EXTERNAL_PARAMS.get("whole_chip_cycles", 5)

    dtnow = datetime.datetime.now()
# the time at which the analysis was started, mostly for debugging purposes
    env["report_start_time"] = dtnow.strftime("%c")

#
# Setup parallel processing
#
    HAVE_MP = os.name == 'posix' and env["parallel"]
    if HAVE_MP:
        try:
            import multiprocessing
            SetOption('num_jobs', env["nprocs"])
        except ImportError: # for versions before Python 2.6
            HAVE_MP = False

# We could protect the process spawning and joining with
# a lock to prevent the waitpid() OSErrors, but the overhead
# is a bit troublesome... since the errors don't hurt us (they
# happen at the end of the process's run), we ignore them and
# choose not to lock the upstream calls.
#
# from threading import Lock
# startLock = Lock()

    SCRATCH_DIR = "./locks"
    EXN_LOG_FNAME = "./exceptionlog.txt"
    EXN_FILENAME_INDEX = 0
    
    global env
    initreports()
    runFullChip(env)
    makeReport(env)
#killSignal = 1
#killReportWriter()
    sys.exit(0)

    
