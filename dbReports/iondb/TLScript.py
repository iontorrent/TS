#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Loads Fullchip analysis pipeline
Uses output from there to generate charts and graphs and dumps to current directory
Adds key metrics to database(not implemented yet)
"""

__version__ = filter(str.isdigit, "$Revision: 21944 $")

# First we need to bootstrap the analysis to start correctly from the command
# line or as the child process of a web server. We import a few things we
# need to make that happen before we import the modules needed for the actual
# analysis.
import  os
import tempfile

# matplotlib/numpy compatibility
os.environ['HOME'] = tempfile.mkdtemp()
from matplotlib import use
use("Agg")

import atexit
import datetime
from os import path
import shutil
import socket
import subprocess
import sys
import time
import numpy
import pickle
from subprocess import *
import StringIO
from collections import deque
import shlex
import multiprocessing
from urlparse import urlunsplit
import glob

# Import analysis-specific packages. We try to follow the from A import B
# convention when B is a module in package A, or import A when
# A is a module.
#from ion.analysis import cafie,sigproc
#from ion.fileformats import sff
from ion.reports import blast_to_ionogram, tfGraphs, plotKey, \
    parseCafie, uploadMetrics, parseBeadfind, \
    parseProcessParams, beadDensityPlot, parseCafie, parseCafieRegions,\
    libGraphs, beadHistogram, plotRawRegions, trimmedReadLenHisto, MaskMerge,\
    StatsMerge, processParametersMerge

from ion.reports.plotters import *
from ion.utils.aggregate_alignment import *
from ion.utils.align_full_chip import *
from iondb.plugin_json import *

from iondb.anaserve import client

sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
import iondb.settings
from iondb.plugins import *

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
import json
import urllib

import dateutil.parser

#import libs to zip with
import zipfile
try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

modes = { zipfile.ZIP_DEFLATED: 'deflated',
          zipfile.ZIP_STORED:   'stored',
          }
#########################################################################
# Sun Grid Code                                                         #
#########################################################################
try:
    # We want to determine if we have the ability to make calls to the
    # drmaa interface. In order to do so, we first need to set environment
    # variables.
    if not iondb.settings.SGE_ENABLED:
        # bail if SGE (currently the only supported drmaa target) is
        # disabled from the site settings file.
        raise ImportError
    # Set env. vars necessary to talk to SGE. The SGE root contains files
    # describing where the grid master lives. The other variables determine
    # which of several possible grid masters to talk to.
    for k in ("SGE_ROOT", "SGE_CELL", "SGE_CLUSTER_NAME",
              "SGE_QMASTER_PORT", "SGE_EXECD_PORT", "DRMAA_LIBRARY_PATH"):
        if not k in os.environ:
            os.environ[k] = str(getattr(iondb.settings,k))
    try:
        import drmaa
    except RuntimeError:
        # drmaa will sometimes raise RuntimeError if libdrmaa1.0 is not
        # installed.
        raise ImportError
    import atexit # provides cleanup of the session object
    try:
        HAVE_DRMAA = True
        # create a single drmaa session
        _session = drmaa.Session()
        try:
            _session.initialize()
            print 'session initialized'
        except:
            print "Session already open"
        #atexit.register(_session.exit)
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
#####################################################################
#
# Analysis implementation details
#
#####################################################################

def printtime(message, *args):
    if args:
        message = message % args
    print "[ " + time.strftime('%X') + " ] " + message

def write_version():
    a = subprocess.Popen('/opt/ion/iondb/bin/lversionChk.sh --ion', shell=True, stdout=subprocess.PIPE)
    ret = a.stdout.readlines()
    f = open('version.txt','w')
    for i in ret:
        f.write(i)
    f.close()

def initreports():
    """Performs initialization for both report writing and background
    report generation."""
    printtime("INIT REPORTS")
    #
    # Begin report writing
    #
    os.umask(0002)
    TMPL_DIR = '/usr/lib/python2.6/dist-packages/ion/web/db/writers'
    shutil.copy("%s/report_layout.json" % TMPL_DIR, "report_layout.json")
    shutil.copy("%s/parsefiles.php" % TMPL_DIR, "parsefiles.php")
    shutil.copy("%s/log.html" % TMPL_DIR, 'log.html')
    shutil.copy("%s/alignment_summary.html" % TMPL_DIR, "alignment_summary.html")
    shutil.copy("%s/format_whole.php" % TMPL_DIR, "Default_Report.php")
    shutil.copy("%s/csa.php" % TMPL_DIR, "csa.php")
    #shutil.copy("%s/ziparc.php" % TMPL_DIR, "ziparc.php")
    #shutil.copy("%s/format_whole_debug.php" % TMPL_DIR, "Detailed_Report.php")
    # create analysis progress bar file
    f = open('progress.txt','w')
    f.write('wellfinding = yellow\n')
    f.write('signalprocessing = grey\n')
    f.write('basecalling = grey\n')
    f.write('sffread = grey\n')
    f.write('alignment = grey')
    f.close()

def gimme_crop_regions(chiptype, nXtiles, nYtiles):
    dimensions={"314":[1280,1152],
                "324":[1280,1152],
                "316":[2736,2640],
                "318":[3392,3792]}
    xlen = dimensions[chiptype][0]
    ylen = dimensions[chiptype][1]
    xdim = xlen/nXtiles
    ydim = ylen/nYtiles
    cropList = []
    for y in range(nYtiles):
        for x in range(nXtiles):
            xorig=x * (xdim)
            yorig=y * (ydim)
            xdiml = (xdim if (xorig + xdim) <= xlen else xdim - ((xorig + xdim) - xlen))
            ydiml = (ydim if (yorig + ydim) <= ylen else ydim - ((yorig + ydim) - ylen))
            #For debug testing, use small regions
            xdiml = 200
            ydiml = 200
            cropList.append([xorig,yorig,xdiml,ydiml])
    return cropList

def GetBlocksToAnalyze(env):
    '''Returns array of block dictionary objects defined in explog.txt'''
    blocks = []
    # expLog.txt contents from Experiment.log field
    exp_json = json.loads(env['exp_json'])
    log = json.loads(exp_json['log'])
    blockstatus = log['blocks']

    for line in blockstatus:
        # Remove keyword; divide argument by comma delimiter into an array
        args = line.strip().replace('BlockStatus:','').split(',')
        #autoanalyze
        autoanalyze = int(args[4].split(':')[1].strip()) == 1
        #analyzeearly
        analyzeearly = int(args[5].split(':')[1].strip()) == 1
        if autoanalyze and analyzeearly:
            # Define Block dictionary object
            #   id_str contains a unique id string
            #   datasubdir contains name of block directory (i.e. 'X0_Y128')
            #   jobcmd contains array of Analysis command line arguments for SGE
            #   jobid contains SGE job id returned when job is queued
            #   status contains SGE job status string
            block = {'id_str':'',
                    'datasubdir':'',
                    'jobcmd':[],
                    'jobid':None,
                    'status':None}
            
            block['datasubdir'] = "%s_%s" % (args[0].strip(),args[1].strip())
            block['id_str'] = block['datasubdir']
            
            resultDir = './%s%s' % ('block_', block['datasubdir'])
            base_args = env['analysisArgs'].strip().split(' ')
            # hack alert
            # raw data dir is second element in analysisArgs and needs the block subdirectory appended
            base_args[1] = '%s/%s' % (base_args[1],block['datasubdir'])
            base_args.append("--libraryKey=%s" % env["libraryKey"])
            base_args.append("--no-subdir")
            base_args.append("--output-dir=%s" % resultDir)
            base_args.append(" >> %s/ReportLog.html 2>&1" % resultDir)
            block['jobcmd'] = base_args
            
            blocks.append(block)
    
    return blocks

def make318BlocksToAnalyze(env,blockList):
    '''Returns array of block dictionary objects created from'''
    blocks = []
    for region in blockList:
        
        # Define Block dictionary object
        #   id_str contains a unique id string
        #   datasubdir contains name of block directory (i.e. 'X0_Y128')
        #   jobcmd contains array of Analysis command line arguments for SGE
        #   jobid contains SGE job id returned when job is queued
        #   status contains SGE job status string
        block = {'id_str':'',
                'datasubdir':'',
                'jobcmd':[],
                'jobid':None,
                'status':None}
        
        block['datasubdir'] = ''
        block['id_str'] = 'X%d_Y%d' % (region[0],region[1])
        
        resultDir = './%sX%d_Y%d' % ('block_',region[0],region[1])
        base_args = env['analysisArgs'].strip().split(' ')
        # for debugging, quick Analysis
        #base_args.append("--flowlimit=60")
        base_args.append("--libraryKey=%s" % env["libraryKey"])
        base_args.append("--no-subdir")
        base_args.append("--output-dir=%s" % resultDir)
        base_args.append("--analysis-region=%d,%d,%d,%d" % (region[0],region[1],region[2],region[3]))
        base_args.append(" >> %s/ReportLog.html 2>&1" % resultDir)
        block['jobcmd'] = base_args
        
        blocks.append(block)
        
    return blocks

def spawn_sge_job(args, rpath):
    adir = path.join(os.getcwd())
    jt = _session.createJobTemplate()
    jt.nativeSpecification = "-pe ion_pe 1 -q all.q"
    jt.remoteCommand = args[0]
    jt.workingDirectory = adir
    out_path = "%s/drmaa_stdout.html" % rpath
    err_path = "%s/drmaa_stderr.txt" % rpath
    logout = open(path.join(out_path), "w")
    logout.write("<html><head><meta http-equiv=refresh content='5'; URL=''></head><pre> \n")
    logout.close()
    jt.outputPath = ":" + path.join(adir, out_path)
    jt.errorPath = ":" + path.join(adir, err_path)
    jt.args = (args[1:])
    jt.joinFiles = False
    jobid = _session.runJob(jt)
    _session.deleteJobTemplate(jt)
    return jobid

def submitBlockJob(blockObj,env):
    '''Direct submission to SGE, skipping ionJobServer, database Results entry'''
    resultDir = './%s%s' % ('block_', blockObj['id_str'])
    # Remove existing directory if it exists
    if path.isdir(resultDir):
        for root, dirs, files in os.walk(resultDir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir (resultDir)
    # we create output dir here so SGE has a place to write stderr and stdout files
    os.mkdir(resultDir)
    # return SGE jobid
    jobid = spawn_sge_job(blockObj['jobcmd'],resultDir)
    return jobid

def runFullChip(env, skipAnalysis):
    STATUS = None
    basefolder = 'plugin_out'
    if not path.isdir(basefolder):
        os.umask(0000)   #grant write permission to plugin user
        os.mkdir(basefolder)
        os.umask(0002)
    pathprefix = env["prefix"]
    libsff = "%s_%s.sff" % (env['expName'], env['resultsName'])
    tfsff = "%s_%s.tf.sff" % (env['expName'], env['resultsName'])
    
    libKeyArg = "--libraryKey=%s" % env["libraryKey"]
    
    write_version()

    #-------------------------------------------------------------
    # Gridded data processing
    #-------------------------------------------------------------
    if GRID_MODE and ('tiled' in env['rawdatastyle'] or '318' in env['chipType']):
        printtime("RUNNING FULL CHIP MULTI-BLOCK ANALYSIS")
        
        if 'tiled' in env['rawdatastyle']:
            # List of block objects to analyze
            blocks = GetBlocksToAnalyze(env)
            offset_str = "use_blocks"
        else:
            printtime("RUNNING 318 ANALYSIS IN SUB-BLOCK MODE")
            # Divide into quadrants (2 by 2 regions or blocks)
            blockList = gimme_crop_regions(env['chipType'].strip('"')[:3],2,2)
            # List of block objects to analyze
            blocks = make318BlocksToAnalyze(env,blockList)
            offset_str = "use_analysis_regions"
        
        if not skipAnalysis:
            # Launch multiple SGE Block analysis jobs
            for block in blocks:
                printtime("Submit block (%s) analysis job to SGE" % block['id_str'])
                block['jobid'] = submitBlockJob(block,env)
                printtime("Job ID: %s" % str(block['jobid']))
                #TODO: error handling
        
        job_list=[block['jobid'] for block in blocks]
        
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Watch status of jobs.  As they finish, run the merge script and remove
        # the job from the list.
        if not skipAnalysis:
            while len(job_list) > 0:
                for job in job_list:
                    block = [block for block in blocks if block['jobid'] == job][0]
                    #check status of SGE jobid
                    block['status'] = _session.jobStatus(block['jobid'])
                    if block['status'] is djs.DONE:
                        block['status'] = 'complete'
                    if block['status'] is 'complete' or block['status'] is djs.FAILED:
                        printtime("Job %s has ended with status %s" % (str(block['jobid']),block['status']))
                        job_list.remove(block['jobid'])
                time.sleep (5)
                    
            printtime("All jobs processed")
        
        #--------------------------------------------------------
        # Start merging results files
        #--------------------------------------------------------
        dirs = ['block_%s' % block['id_str'] for block in blocks]
        if len(dirs) > 0:
            cwd = os.getcwd()
            #####################################################
            # Create block reports                              #
            #####################################################
            for dir in dirs:
                r = subprocess.call(["ln", "-s", os.path.join(cwd,"Default_Report.php"), os.path.join(dir,"Default_Report.php")])
                if r:
                    printtime("couldn't create symbolic link")
            #####################################################
            # Grab one of the processParameters.txt files       #
            #####################################################
            for dir in dirs:
                ppfile = os.path.join(dir,'processParameters.txt')
                if os.path.isfile(ppfile):
                    processParametersMerge.processParametersMerge(ppfile,True)
                    break
                
            ########################################
            # Merge individual block SFF files     #
            ########################################
            printtime("Merging Library SFF files")
            if len(dirs) > 0:
                cmd = 'SFFMerge'
                cmd = cmd + ' -i rawlib.sff'
                cmd = cmd + ' -o %s ' % libsff
                cmd = cmd + " ".join(dirs)
                printtime("DEBUG: Calling '%s'" % cmd)
                call(cmd,shell=True)
            else:
                printtime("No library SFF files were found to merge")

            printtime("Merging Test Fragment SFF files")
            if len(dirs) > 0:
                cmd = 'SFFMerge'
                cmd = cmd + ' -i rawtf.sff'
                cmd = cmd + ' -o %s ' % tfsff
                cmd = cmd + " ".join(dirs)
                printtime("DEBUG: Calling '%s'" % cmd)
                call(cmd,shell=True)
            else:
                printtime("No test fragment SFF files were found to merge")

            ########################################
            # Merge individual block bead metrics files     #
            ########################################
            printtime("Plot Heatmaps for each block")
            for dir in dirs:
                os.chdir(os.path.join(cwd,dir))

                bfmaskPath = "bfmask.bin"
                if os.path.isfile(bfmaskPath):
                    com = "BeadmaskParse -m MaskBead %s" % bfmaskPath
                    os.system(com)
                else:
                    printtime("Warning: no bfmask.bin file exists.")

                maskpath = "MaskBead.mask"
                if os.path.exists(maskpath):
                    try:
                        # Makes Bead_density_contour.png
                        beadDensityPlot.genHeatmap(maskpath)
                    except:
                        traceback.print_exc()
                else:
                    printtime("Warning: no MaskBead.mask file exists.")

            os.chdir(cwd)

            printtime("Merging MaskBead.mask files")
            bfmaskfiles = [dir for dir in dirs if path.isfile(os.path.join(dir,'bfmask.bin'))]
            if len(bfmaskfiles) > 0:
                MaskMerge.main_merge('MaskBead',bfmaskfiles, True, offset_str)
                for dir in dirs:
                    if os.path.exists(os.path.join(dir,maskpath)):
                        os.remove(os.path.join(dir,maskpath))
                    else:
                        printtime("Warning: no MaskBead.mask file exists.")
            else:
                printtime("No bfmask.bin files were found to merge")

            ########################################
            # Merge individual block bead stats files     #
            ########################################
            printtime("Merging bfmask.stats files")
            bfmaskfiles = [dir for dir in dirs if path.isfile(os.path.join(dir,'bfmask.stats'))]
            if len(bfmaskfiles) > 0:
                StatsMerge.main_merge(bfmaskfiles, True)
            else:
                printtime("No bfmask.stats files were found to merge")
        else:
            printtime("Warning: no block results directories found. Nothing merged.")
            
        printtime("Finished Multi Block processing")
    #-------------------------------------------------------------
    # Single Block data processing
    #-------------------------------------------------------------
    else:
        if not skipAnalysis:
            printtime("RUNNING FULL CHIP ANALYSIS")
            command = "%s %s %s >> ReportLog.html 2>&1" %  (env['analysisArgs'], libKeyArg, "--no-subdir")
            printtime("Analysis command: %s" % command)
            status = call(command,shell=True)
            if int(status) == 2:
                STATUS = 'Checksum Error'
                uploadMetrics.updateStatus(STATUS)
            elif int(status) == 3:
                STATUS = 'PGM Operation Error'
                uploadMetrics.updateStatus(STATUS)
            elif int(status) != 0:
                STATUS = 'ERROR'
                uploadMetrics.updateStatus(STATUS)
            csp = path.join(env['pathToRaw'],'checksum_status.txt')
            if not path.exists(csp) and not env['skipchecksum'] and STATUS==None:
                try:
                    os.umask(0002)
                    f = open(csp, 'w')
                    f.write(str(status))
                    f.close()
                except:
                    traceback.print_exc()
        else:
            printtime('Skipping Full Chip Analysis')
        printtime("Finished Single Block processing")
        
    sys.stdout.flush()
    sys.stderr.flush()
    #end tiled versus single processing
    #at this point, there should be SFF files for further processing
    
    tfKey = "ATCG"
    libKey = env['libraryKey']
    floworder = env['flowOrder']
    printtime("Using flow order: %s" % floworder)
    printtime("Using library key: %s" % libKey)

    ##################################################
    # change sff names to have analysis and          #
    # experiment names attached                      #
    ##################################################
    if path.exists('rawlib.sff'):
        os.rename("rawlib.sff", libsff)
    if path.exists('rawtf.sff'):
        os.rename("rawtf.sff", tfsff)

    ##################################################
    # Unfiltered SFF
    ##################################################

    unfiltered_dir = "unfiltered"
    if path.exists(unfiltered_dir):

        top_dir = os.getcwd()

        #change to the unfiltered dir
        os.chdir(os.path.join(top_dir,unfiltered_dir))

        #grab the first file named untrimmed.sff
        try:
            untrimmed_sff = glob.glob("*.untrimmed.sff")[0]
        except IndexError:
            printtime("Error, unable to find the untrimmed sff file")

        #rename untrimmed to trimmed
        trimmed_sff = untrimmed_sff.replace("untrimmed.sff","trimmed.sff")

        # 3' adapter details
        qual_cutoff = env['reverse_primer_dict']['qual_cutoff']
        qual_window = env['reverse_primer_dict']['qual_window']
        adapter_cutoff = env['reverse_primer_dict']['adapter_cutoff']
        adapter = env['reverse_primer_dict']['sequence']

        # If flow order is missing, assume classic flow order:
        if floworder == "0":
            floworder = "TACG"
            printtime("warning: floworder redefine required.  set to TACG")

        #we will always need the input and output files
        trimArgs = "--in-sff %s --out-sff %s" % (untrimmed_sff,trimmed_sff)
        trimArgs += " --flow-order %s" % (floworder)
        trimArgs += " --key %s" % (libKey)
        trimArgs += " --qual-cutoff %s" % (qual_cutoff)
        trimArgs += " --qual-window-size %s" % (qual_window)
        trimArgs += " --adapter-cutoff %s" % (adapter_cutoff)
        trimArgs += " --adapter %s" % (adapter)
        trimArgs += " --min-read-len 5"

        try:
            com = "SFFTrim %s " % (trimArgs)
            printtime("Unfiltered SFFTrim Command:\n%s" % com)
            ret = call(com,shell=True)
            if int(ret)!=0 and STATUS==None:
                STATUS='ERROR'
        except:
            printtime('Failed Unfiltered SFFTrim')

        sffs = glob.glob("*.sff")
        for sff in sffs:
            try:
                com = "SFFRead -q %s %s" % (sff.replace(".sff",".fastq"), sff)

                ret = call(com,shell=True)
                if int(ret)!=0 and STATUS==None:
                    STATUS='ERROR'
            except:
                printtime('Failed to convert SFF' + str(sff) + ' to fastq')

        #trim status
        for status in ["untrimmed","trimmed"]:
            os.chdir(os.path.join(top_dir,unfiltered_dir))
            if not os.path.exists(status):
                os.makedirs(status)
            os.chdir(os.path.join(top_dir,unfiltered_dir,status))

            try:
                printtime("Trim Status",)
                align_full_chip_core("../*." + status + ".sff", libKey, tfKey, floworder, env['fastqpath'], env['align_full'], -1, False, False, True, DIR_BC_FILES, env)
            except OSError:
                printtime('Trim Status Alignment Failed to start')
                alignError = open("alignment.error", "w")
                alignError.write(str(traceback.format_exc()))
                alignError.close()
                traceback.print_exc()

        os.chdir(top_dir)
    else:
        printtime("Directory unfiltered does not exist")

    sys.stdout.flush()
    sys.stderr.flush()
    
    ##################################################
    # Trim the SFF file if it has been requested     #
    ##################################################

    #only trim if SFF is false
    if not env['sfftrim']:
        printtime("Attempting to trim the SFF file")

        TrimLibSFF = libsff[:4] + "trimmed.sff"

        #we will always need the input and output files
        trimArgs = "--in-sff %s --out-sff %s" % (libsff,TrimLibSFF)

        qual_cutoff = env['reverse_primer_dict']['qual_cutoff']
        qual_window = env['reverse_primer_dict']['qual_window']
        adapter_cutoff = env['reverse_primer_dict']['adapter_cutoff']
        adapter = env['reverse_primer_dict']['sequence']

        if not env['sfftrim_args']:
            printtime("no args found, using default args")
            trimArgs = trimArgs + " --flow-order %s --key %s" % (floworder, libKey)
            trimArgs = trimArgs + " --qual-cutoff %d --qual-window-size %d --adapter-cutoff %d --adapter %s" % (qual_cutoff,qual_window,adapter_cutoff,adapter)
            trimArgs = trimArgs + " --min-read-len 5 "
        else:
            printtime("using non default args %s" % env['sfftrim_args'])
            trimArgs = trimArgs + " " + env['sfftrim_args']

        try:
            com = "SFFTrim %s " % (trimArgs)
            printtime("SFFTrim command:")
            printtime(com)
            ret = call(com,shell=True)
            if int(ret)!=0 and STATUS==None:
                STATUS='ERROR'
        except:
            printtime('Failed SFFTrim')

        #if the trim did not fail then move the untrimmed file to untrimmed.expname.sff
        #and move trimmed to expname.sff to ensure backwards compatability
        if path.exists(libsff):
            os.rename(libsff, "untrimmed." + libsff )
        if path.exists(TrimLibSFF):
            os.rename(TrimLibSFF, libsff)
    else:
        printtime("Not attempting to trim the SFF")


    #####################################################
    # Barcode trim SFF if barcodes have been specified  #
    # Creates one fastq per barcode, plus unknown reads #
    #####################################################

    if env['barcodeId'] is not '':
        try:
            com = 'barcodeSplit -s -i %s -b barcodeList.txt -c barcodeMask.bin -f %s' % (libsff,floworder)
            printtime(com)
            ret = call(com,shell=True)
            if int(ret) != 0 and STATUS==None:
                STATUS='ERROR'
            else:
                # Rename bc trimmed sff
                if path.exists("bctrimmed_"+libsff):
                    os.rename("bctrimmed_"+libsff, libsff)
        except:
            printtime("Failed barcodeSplit")


    ##################################################
    # Once we have the new SFF, run SFFSummary
    # to get the predicted quality scores
    ##################################################

    try:
        com = "SFFSummary -o quality.summary --sff-file %s --read-length 50,100,150 --min-length 0,0,0 --qual 0,17,20 -d readLen.txt" % (libsff)
        printtime("sffsummary: %s" % com)
        ret = call(com,shell=True)
        if int(ret)!=0 and STATUS==None:
            STATUS='ERROR'
    except:
        printtime('Failed SFFSummary')

    ##################################################
    #make keypass.fastq file -c(cut key) -k(key flows)#
    ##################################################
    # create analysis progress bar file
    f = open('progress.txt','w')
    f.write('wellfinding = green\n')
    f.write('signalprocessing = green\n')
    f.write('basecalling = green\n')
    f.write('sffread = yellow\n')
    f.write('alignment = grey')
    f.close()

    try:
        com = "SFFRead -q %s %s > keypass.summary" % (env['fastqpath'],libsff)
        ret = call(com,shell=True)
        if int(ret)!=0 and STATUS==None:
            STATUS='ERROR'
    except:
        printtime('Failed SFFRead')


    ##################################################
    #generate TF Metrics                             #
    #look for both keys and append same file         #
    ##################################################
    printtime("Calling TFMapper")

    output = "TFMapper.stats"
    block_tfsff = "rawtf.sff"
    block_libsff = "rawlib.sff"
    if GRID_MODE and ('tiled' in env['rawdatastyle'] or '318' in env['chipType']):
        #Iterative TFMapper
        try:
            cmd = "TFMapperIterative -m 0 --flow-order=%s --tfkey=%s --libkey=%s %s " % (floworder, tfKey, libKey, block_tfsff)
            cmd = cmd + " ".join(dirs)
            cmd = cmd + " > %s" % output
            printtime("DEBUG: Calling '%s'" % cmd)
            os.system(cmd)
        except:
            printtime("TFMapper failed")
    
        try:
            cmd = "TFMapperIterative -m 1 --flow-order=%s --tfkey=%s --libkey=%s  %s " % (floworder, tfKey, libKey, block_libsff)
            cmd = cmd + " ".join(dirs)
            cmd = cmd + " >> %s" % output
            printtime("DEBUG: Calling '%s'" % cmd)
            os.system(cmd)
        except:
            printtime("TFMapper failed")
    else:
        try:
            com = "TFMapper -m 0 --flow-order=%s --tfkey=%s --libkey=%s %s > %s" % (floworder, tfKey, libKey, tfsff, output)
            os.system(com)
        except:
            printtime("TFMapper failed")
    
        try:
            #com = "TFMapper %s >> %s" % (libsff, output)
            com = "TFMapper -m 1 --flow-order=%s --tfkey=%s --libkey=%s %s >> %s" % (floworder, tfKey, libKey, libsff, output)
            os.system(com)
        except:
            printtime("TFMapper failed")

    ########################################################
    #generate the TF Metrics including plots               #
    ########################################################
    tfMetrics = None
    if os.path.exists("TFMapper.stats"):
        mF = open("TFMapper.stats", 'r')
        metricsData = mF.readlines()
        mF.close()
        try:
            # Q17 TF Read Length Plot
            tfMetrics = tfGraphs.parseLog(metricsData)
            tfGraphs.Q17(tfMetrics)
        except Exception:
            printtime("Metrics Gen Failed")
            traceback.print_exc()

    ########################################################
    #Generate Raw Data Traces for lib and TF keys          #
    ########################################################
    tfRawPath = 'avgNukeTrace_%s.txt' % tfKey
    libRawPath = 'avgNukeTrace_%s.txt' % libKey
    peakOut = 'raw_peak_signal'

    if GRID_MODE and ('tiled' in env['rawdatastyle'] or '318' in env['chipType']):
        shutil.copy("%s/%s" % (dirs[0], tfRawPath), tfRawPath)
        shutil.copy("%s/%s" % (dirs[0], libRawPath), libRawPath)

    if os.path.exists(tfRawPath):
        try:
            kp = plotKey.KeyPlot(tfKey, floworder, 'Test Fragment')
            kp.parse(tfRawPath)
            kp.dump_max(peakOut)
            kp.plot()
        except:
            printtime("TF key graph didn't render")
            traceback.print_exc()

    if os.path.exists(libRawPath):
        try:
            kp = plotKey.KeyPlot(libKey, floworder, 'Library')
            kp.parse(libRawPath)
            kp.dump_max(peakOut)
            kp.plot()
        except:
            printtime("Lib key graph didn't render")
            traceback.print_exc()

    ########################################################
    #Generate Ionograms                                    #
    ########################################################
    printtime("Calling parseCafie")

    cafiePath = 'cafieMetrics.txt'

    if GRID_MODE and ('tiled' in env['rawdatastyle'] or '318' in env['chipType']):
        shutil.copy("%s/%s" % (dirs[0], cafiePath), cafiePath)
        
    if os.path.exists(cafiePath):
        try:
            cafieMetrics = parseCafie.generateMetrics(cafiePath,floworder)
        except:
            printtime("Cafie Metrics is Empty")
            cafieMetrics = None
            traceback.print_exc()
    else:
        printtime("cafieMetrics.txt doesn't exist")
        cafieMetrics = None

    ########################################################
    #Make Bead Density Plots                               #
    ########################################################
    printtime("Make Bead Density Plots")
    bfmaskPath = "bfmask.bin"
    if os.path.isfile(bfmaskPath):
        com = "BeadmaskParse -m MaskBead %s" % bfmaskPath
        os.system(com)
    else:
        # this is ok in GRID_MODE
        printtime("Warning: no bfmask.bin file exists.")

    maskpath = "MaskBead.mask"
    if os.path.exists(maskpath):
        try:
            # Makes Bead_density_contour.png
            beadDensityPlot.genHeatmap(maskpath)
            os.remove(maskpath)
        except:
            traceback.print_exc()
    else:
        printtime("Warning: no MaskBead.mask file exists.")

    ########################################################
    # Make lib_cafie.txt                                   #
    ########################################################
    regionsPath = "cafieRegions.txt"
    if os.path.exists(regionsPath):
        try:
            hm = parseCafieRegions.HeatMap()
            hm.savePath = os.getcwd()
            hm.loadData(regionsPath)
            hm.parseRegion()
            hm.writeMetricsFile()
        except:
            printtime('CafieRegions failed')
            traceback.print_exc()

    ########################################################
    # Make beadfind histogram for every region             #
    ########################################################
    procPath = 'processParameters.txt'
    try:
        processMetrics = parseProcessParams.generateMetrics(procPath)
    except:
        printtime("processMetrics failed")
        processMetrics = None

    """
    ########################################################
    # Make per region key incorporation traces             #
    ########################################################
    perRegionTF = "averagedKeyTraces_TF.txt"
    perRegionLib = "averagedKeyTraces_Lib.txt"
    if os.path.exists(perRegionTF):
        pr = plotRawRegions.PerRegionKey(tfKey, floworder,'TFTracePerRegion.png')
        pr.parse(perRegionTF)
        pr.plot()

    if os.path.exists(perRegionLib):
        pr = plotRawRegions.PerRegionKey(libKey, floworder,'LibTracePerRegion.png')
        pr.parse(perRegionLib)
        pr.plot()
    """

    sys.stdout.flush()
    sys.stderr.flush()
    
    ########################################################
    #Attempt to align                                      #
    ########################################################

    # create analysis progress bar file
    f = open('progress.txt','w')
    f.write('wellfinding = green\n')
    f.write('signalprocessing = green\n')
    f.write('basecalling = green\n')
    f.write('sffread = green\n')
    f.write('alignment = yellow')
    f.close()

    # False is to not generate Default.sam.parsed file.
    sam_parsed = True
    
    try:
        align_full_chip(libsff, libKey, tfKey, floworder, env['fastqpath'], env['align_full'], DIR_BC_FILES, env, sam_parsed)
    except Exception:
        printtime("Alignment Failed")
        traceback.print_exc()

    #make the read length historgram
    try:
        trimmedReadLenHisto.trimmedReadLenHisto('readLen.txt','readLenHisto.png')
    except:
        printtime("Failed to create trimmedReadLenHisto")

    ##################################################
    # Create zip of files
    ##################################################



    #sampled sff
    make_zip(libsff.replace(".sff",".sampled.sff")+'.zip', libsff.replace(".sff",".sampled.sff"))
    #sampled fastq
    make_zip(env['fastqpath'].replace(".fastq",".sampled.fastq")+'.zip', env['fastqpath'].replace(".fastq",".sampled.fastq"))

    #library sff
    make_zip(libsff + '.zip', libsff )

    #fastq zip
    make_zip(env['fastqpath'] + '.zip', env['fastqpath'])

    #tf sff
    make_zip(tfsff + '.zip', tfsff)

    ########################################################
    # Zip up and move sff, fastq, bam, bai files           #
    # Move zip files to results directory                  #
    ########################################################
    if os.path.exists(DIR_BC_FILES):
        os.chdir(DIR_BC_FILES)
        filestem = "%s_%s" % (env['expName'], env['resultsName'])
        list = os.listdir('./')
        extlist = ['sff','bam','bai','fastq']
        for ext in extlist:
            filelist = fnmatch.filter(list, "*." + ext)
            zipname = filestem + '.barcode.' + ext + '.zip'
            for file in filelist:
                make_zip(zipname, file)
                os.rename(file,os.path.join("../",file))

        # Move zip files up one directory to results directory
        for ext in extlist:
            zipname = filestem + '.barcode.' + ext + '.zip'
            if os.path.isfile(zipname):
                os.rename(zipname,os.path.join("../",zipname))

        os.chdir('../')


    ########################################################
    #ParseFiles and Upload Metrics                         #
    ########################################################
    beadPath = 'bfmask.stats'
    lib_cafiePath = 'lib_cafie.txt'
    libPath = 'alignment.summary'
    QualityPath = 'quality.summary'
    printtime("Attempting to Upload to Database")
    filterMetrics = None

    try:
        beadMetrics = parseBeadfind.generateMetrics(beadPath)
    except:
        printtime('beadmetrics Failed')
        beadMetrics = None
        traceback.print_exc()

    #attempt to upload the metrics to the Django database
    try:
        uploadMetrics.populateDb(tfMetrics, cafieMetrics, processMetrics,
                                 beadMetrics, filterMetrics, libPath, STATUS, peakOut, lib_cafiePath, QualityPath)
    except:
        traceback.print_exc()

    # create analysis progress bar file
    f = open('progress.txt','w')
    f.write('wellfinding = green\n')
    f.write('signalprocessing = green\n')
    f.write('basecalling = green\n')
    f.write('sffread = green\n')
    f.write('alignment = green')
    f.close()

    try:
        # Call script which creates and populates a file with
        # experiment metrics.  RSMAgent_TS then forwards this file
        # to the Axeda remote system monitoring server.
        primary_key = open("primary.key").readline()
        primary_key = primary_key.split(" = ")
        primary_key = primary_key[1]
        cmd = "/opt/ion/RSM/createExperimentMetrics.py " + str(primary_key)
        printtime(str(cmd))
        os.system(cmd)
    except:
        printtime("RSM createExperimentMetrics.py failed")

    ########################################################
    # Run experimental script                              #
    ########################################################

    printtime('Running Plugins')

    launcher = PluginRunner()

    #Plugins will run in reverse alphabetical order order.
    for plugin in sorted(env['plugins'], key=lambda plugin: plugin["name"],reverse=True):
        if plugin != '':
            runPlug = True
            printtime("Plugin %s is enabled" % plugin['name'])

            if plugin['autorun'] == False:
                printtime("     Auto Run is disabled for this plugin.  Skipping")
                continue    #skip to next plugin

            # Blank fields indicate execute in all cases.
            # Exclude this plugin if non-blank entry does not match run info:
            for label in ['project','sample','libraryName','chipType']:
                if plugin[label] != None and plugin[label] != "":
                    runPlug = False
                    #print "     needs to match something in: %s" % plugin[label]
                    for i in plugin[label].split(','):
                        i = i.strip()
                        #print "env[%s] = %s" % (label,env[label])
                        if i in env[label]:
                            # Okay to run plugin
                            runPlug = True
                            #print "     match found %s equals %s" % (i,env[label])
                            break

            if runPlug == False:
                printtime("     did not run after all.")
                continue    #skip to next plugin

            try:
                primary_key = open("primary.key").readline()
                primary_key = primary_key.split(" = ")
                primary_key = primary_key[1]
                printtime(primary_key)
            except:
                printtime("Error, unable to get the primary key")

            try:
                #https://iontorrent.jira.com/wiki/display/TS/Plugin+json+file+format
                env['analysis_dir'] = os.getcwd()
                env['testfrag_key'] = 'ATCG'
                start_json = make_plugin_json(env,plugin,primary_key,basefolder,url_root)
                ret = launcher.callPluginXMLRPC(start_json, env['master_node'], iondb.settings.IPLUGIN_PORT)
            except:
                printtime('plugin %s failed...' % plugin['name'])
                traceback.print_exc()
                
def make_zip(zip_file, to_zip, arcname=None):
    """Try to make a zip of a file if it exists"""
    if os.path.exists(to_zip):
        zf = zipfile.ZipFile(zip_file, mode='a', allowZip64=True)
        try:
            #adding file with compression
            if arcname == None:
                zf.write(to_zip, compress_type=compression)
            else:
                zf.write(to_zip, arcname, compress_type=compression)
            print "Created ", zip_file, " of", to_zip
        except OSError:
            print 'OSError with - :', to_zip
        except zipfile.LargeZipFile:
            printtime("The zip file was too large, ZIP64 extensions could not be enabled")
        except:
            printtime("Unexpected error creating zip")
            traceback.print_exc()
        finally:
            zf.close()
    else:
        printtime("Unable to make zip because the file " + str(to_zip) + " did not exist!")
        
def getExpLogMsgs(env):
    """
    Parses explog_final.txt for warning messages and dumps them to
    ReportLog.html.
    This only works if the raw data files have not been deleted.
    For a from-wells analysis, you may not have raw data.
    """
    inputFile = path.join(env['pathToRaw'],'explog_final.txt')
    outputFile = path.join('./','ReportLog.html')
    try:
        f = open (inputFile, 'r')
    except:
        printtime("Cannot open file: %s " % inputFile)
        return True

    line = f.readline()
    while line:
        if "WARNINGS:" in line:
            if len("WARNINGS: ") < len(line):
                # print to output file
                try:
                    g = open (outputFile, 'a')
                    g.write("From PGM explog_final.txt:\n")
                    g.write (line)
                    g.close()
                except:
                    printtime("Cannot open file: %s " % outputFile)
        line = f.readline()

    f.close()
    return False

def get_pgm_log_files(rawdatadir):
    # Create a tarball of the pgm raw data log files for inclusion into CSA.
    # tarball it now before the raw data gets deleted.
    files=['explog_final.txt',
    'explog.txt',
    'InitLog.txt',
    'RawInit.txt',
    'RawInit.jpg',
    'InitValsW3.txt',
    'InitValsW2.txt']
    for file in files:
        make_zip ('pgm_logs.zip',os.path.join(rawdatadir,file), file)
    return
    
if __name__=="__main__":
    ########################################################
    # Print nice header information                        #
    ########################################################
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
    print('*'*_info_width)
    for d,l in zip(python_data, python_data_labels):
        spacer = ' '*(_max_sum - (len(l) + len(d)) + _TABSIZE)
        print('* %s%s%s *' % (str(l),spacer,str(d).replace('\n',' ')))
    print('*'*_info_width)

    # Sub directory to contain fastq files for barcode enabled runs
    DIR_BC_FILES = 'bc_files'
    
    #####################################################################
    # Load the analysis parameters and metadata from a json file passed in on the
    # command line with --params=<json file name>
    # we expect this loop to iterate only once. This is more elegant than
    # trying to index into ARGUMENTS.
    ##################################################################### 
    EXTERNAL_PARAMS = {}
    env = {}
    if len(sys.argv) > 1:
        env["params_file"] = sys.argv[1]
    else:
        env["params_file"] = 'ion_params_00.json'
    file = open(env["params_file"], 'r')
    EXTERNAL_PARAMS = json.loads(file.read())
    file.close()
    for k,v in EXTERNAL_PARAMS.iteritems():
        if isinstance(v, unicode):
            EXTERNAL_PARAMS[k] = str(v)

    # Where the raw data lives (generally some path on the network)
    pathprefix = str(EXTERNAL_PARAMS['pathToData'])
    env['prefix'] = pathprefix
    # this is the library name for the run taken from the library field in the database
    env["libraryName"] = EXTERNAL_PARAMS.get("libraryName", "none")
    if env["libraryName"] == "":
        env["libraryName"] = "none"
    dtnow = datetime.datetime.now()
    # the time at which the analysis was started, mostly for debugging purposes
    env["report_start_time"] = dtnow.strftime("%c")
    # get command line args
    env['analysisArgs'] = EXTERNAL_PARAMS.get("analysisArgs")
    # name of current analysis
    env['resultsName'] = EXTERNAL_PARAMS.get("resultsName")
    # name of current experiment
    env['expName'] = EXTERNAL_PARAMS.get("expName")
    #library key input
    env['libraryKey'] = EXTERNAL_PARAMS.get("libraryKey")
    #path to the raw data
    env['pathToRaw'] = EXTERNAL_PARAMS.get("pathToData")
    #plugins
    env['plugins'] = EXTERNAL_PARAMS.get("plugins")
    # path to fastqfile
    env['fastqpath'] = EXTERNAL_PARAMS.get('fastqpath')
    # skipChecksum?
    env['skipchecksum'] = EXTERNAL_PARAMS.get('skipchecksum',False)
    # Do Full Align?
    env['align_full'] = EXTERNAL_PARAMS.get('align_full')
    # Check to see if a SFFTrim should be done
    env['sfftrim'] = EXTERNAL_PARAMS.get('sfftrim')
    # Get SFFTrim args
    env['sfftrim_args'] = EXTERNAL_PARAMS.get('sfftrim_args')
    env['flowOrder'] = EXTERNAL_PARAMS.get('flowOrder').strip()
    env['project'] = EXTERNAL_PARAMS.get('project')
    env['sample'] = EXTERNAL_PARAMS.get('sample')
    env['chipType'] = EXTERNAL_PARAMS.get('chiptype')
    env['barcodeId'] = EXTERNAL_PARAMS.get('barcodeId','')
    env['reverse_primer_dict'] = EXTERNAL_PARAMS.get('reverse_primer_dict')
    env['rawdatastyle'] = EXTERNAL_PARAMS.get('rawdatastyle', 'single')

    #extra JSON
    env['extra'] = EXTERNAL_PARAMS.get('extra', '{}')
    # Aligner options
    env['aligner_opts_extra'] = EXTERNAL_PARAMS.get('aligner_opts_extra', '{}')

    #Get plan info
    env['plan'] = EXTERNAL_PARAMS.get('plan', '{}')

    #get the name of the master node
    try:
        # SHOULD BE iondb.settings.QMASTERHOST, but this always seems to be "localhost"
        act_qmaster = os.path.join(iondb.settings.SGE_ROOT, iondb.settings.SGE_CELL, 'common', 'act_qmaster')
        master_node = open(act_qmaster).readline().strip()
    except IOError:
        master_node = "localhost"

    env['master_node'] = master_node

    env['net_location'] = EXTERNAL_PARAMS.get('net_location','http://' + master_node )
    env['url_path'] = EXTERNAL_PARAMS.get('url_path','output/Home')

    def to_bool(string):
        if string.lower() == 'true':
            return True
        else:
            return False
    try:
        skipAnalysis = to_bool(sys.argv[2])
    except:
        skipAnalysis = False

    # assemble the URL path for this analysis result - this is passed to plugins
    # url_root is just absolute path, no hostname or scheme components
    url_root = os.path.join(env['url_path'],os.path.basename(os.getcwd()))

    printtime("URL string %s" % url_root)

    #get the experiment json data
    env['exp_json'] = EXTERNAL_PARAMS.get('exp_json')
    #get the name of the site
    env['site_name'] = EXTERNAL_PARAMS.get('site_name')

    #drops a zip file of the pgm log files
    get_pgm_log_files(env['pathToRaw'])
    
    # Generate a system information file for diagnostics porpoises.
    try:
        com="/opt/ion/iondb/bin/collectInfo.sh 2>&1 >> ./sysinfo.txt"
        os.system(com)
    except:
        print traceback.format_exc()
    
    #############################################################
    # Code to start full chip analysis                           #
    #############################################################
    os.umask(0002)
    initreports()
    logout = open("ReportLog.html", "w")
    logout.close()
    
    sys.stdout.flush()
    sys.stderr.flush()
    
    GRID_MODE = False
    runFullChip(env, skipAnalysis)
    getExpLogMsgs(env)
    make_zip ('pgm_logs.zip',os.path.join(env['pathToRaw'],'explog_final.txt'), 'explog_final.txt')
    printtime("Run Complete")
    sys.exit(0)

