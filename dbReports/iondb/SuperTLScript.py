#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Loads Fullchip analysis pipeline
Uses output from there to generate charts and graphs and dumps to current directory
Adds key metrics to database(not implemented yet)
"""

__version__ = filter(str.isdigit, "$Revision: 17459 $")

# First we need to bootstrap the analysis to start correctly from the command
# line or as the child process of a web server. We import a few things we
# need to make that happen before we import the modules needed for the actual
# analysis.
import os
import tempfile

# matplotlib/numpy compatibility
os.environ['HOME'] = tempfile.mkdtemp()
from matplotlib import use
use("Agg")

import ConfigParser
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
from iondb.plugin_json import *

from django.db import models
from iondb.rundb import models

from iondb.anaserve import client
from django.conf import settings

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

try:
    import drmaa
except RuntimeError:
    # drmaa will sometimes raise RuntimeError if libdrmaa1.0 is not
    # installed.
    print "Unexpected error:", sys.exc_info()
    raise ImportError

def write_version():
    a = subprocess.Popen('/opt/ion/iondb/bin/lversionChk.sh', shell=True, stdout=subprocess.PIPE)
    ret = a.stdout.readlines()
    f = open('version.txt','w')
    for i in ret[:len(ret)-1]:
        f.write(i)
    f.close()

#####################################################################
#
# Analysis implementation details
#
#####################################################################
class MyConfigParser(ConfigParser.RawConfigParser):
    def read(self, filename):
        try:
            text = open(filename).read()
        except IOError:
            pass
        else:
            file = StringIO.StringIO("[global]\n" + text)
            self.readfp(file, filename)

def printtime(message):
    print "[ " + time.strftime('%X') + " ] " + message

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

def getThumbnailSize():
    # expLog.txt
    exp_json = json.loads(env['exp_json'])
    log = json.loads(exp_json['log'])
    blockstatus = log['blocks']
    W = 0
    H = 0
    for line in blockstatus:
        # Remove keyword; divide argument by comma delimiter into an array
        args = line.strip().replace('BlockStatus:','').split(',')
        datasubdir = "%s_%s" % (args[0].strip(),args[1].strip())
        if datasubdir == 'thumbnail_top':
            W = int(args[2].strip('W '))
            H = int(args[3].strip('H '))
            break

    return [W,2*H]

def GetBlocksToAnalyze(env):
    '''Returns array of block dictionary objects defined in explog.txt'''
    blocks = []

    if is_thumbnail or is_wholechip:

        block = {'id_str':'',
                'datasubdir':'',
                'jobcmd':[],
                'jobid':None,
                'status':None}

        base_args = env['analysisArgs'].strip().split(' ')
        base_args.append("--libraryKey=%s" % env["libraryKey"])
        base_args.append("--no-subdir")

        if is_thumbnail:
            thumbnailsize = getThumbnailSize()
            block['id_str'] = 'thumbnail'
            base_args.append("--cfiedr-regions-size=50x50")
            base_args.append("--block-size=%sx%s" % (thumbnailsize[0], thumbnailsize[1]))
            base_args.append("--beadfind-thumbnail 1")

        if is_wholechip:
            block['id_str'] = 'wholechip'

        block['jobcmd'] = base_args
        print base_args
        print block
        blocks.append(block)

    else:
        # expLog.txt contents from Experiment.log field
        exp_json = json.loads(env['exp_json'])
        log = json.loads(exp_json['log'])
        blockstatus = log['blocks']
        # contains regular blocks and a thumbnail block
        for line in blockstatus:
            # Remove keyword; divide argument by comma delimiter into an array
            args = line.strip().replace('BlockStatus:','').split(',')
            #autoanalyze
            autoanalyze = int(args[4].split(':')[1].strip()) == 1
            #analyzeearly
            analyzeearly = int(args[5].split(':')[1].strip()) == 1
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

            #ignore thumbnails
            if block['datasubdir'] =='X-1_Y0' or block['datasubdir'] =='thumbnail_bottom':
                continue
            if block['datasubdir'] =='X-1_Y1' or block['datasubdir'] =='thumbnail_top':
                continue

            base_args = env['analysisArgs'].strip().split(' ')
            # raw data dir is last element in analysisArgs and needs the block subdirectory appended
            base_args[-1] = path.join(base_args[-1], block['datasubdir'])
            rawDirectory = base_args[-1]
            base_args.append("--libraryKey=%s" % env["libraryKey"])
            base_args.append("--no-subdir")

            block['jobcmd'] = base_args

            #if autoanalyze and analyzeearly:
            if os.path.isdir(rawDirectory):
                print base_args
                print block
                blocks.append(block)

    return blocks

def spawn_sge_job(rpath):
    out_path = "%s/drmaa_stdout_block.html" % rpath
    err_path = "%s/drmaa_stderr_block.txt" % rpath
    logout = open(path.join(out_path), "w")
    logout.write("<html><head><meta http-equiv=refresh content='5'; URL=''></head><pre> \n")
    logout.close()
    cwd = os.getcwd()

    jt = _session.createJobTemplate()
    #SGE
    jt.nativeSpecification = "-pe ion_pe 1 -l qname=all.q"
#    jt.nativeSpecification = "-pe ion_pe 1 -l h_vmem=10000M,qname=all.q"
    #TORQUE
    #jt.nativeSpecification = ""
    jt.remoteCommand = "python"
    jt.workingDirectory = path.join(cwd, rpath)
    jt.outputPath = ":" + path.join(cwd, out_path)
    jt.errorPath = ":" + path.join(cwd, err_path)
    jt.args = ['BlockTLScript.py', 'ion_params_00.json']
    jt.joinFiles = False
    jobid = _session.runJob(jt)
    _session.deleteJobTemplate(jt)

    return jobid

def submitBlockJob(blockObj, env):

    file_in = open("ion_params_00.json", 'r')
    TMP_PARAMS = json.loads(file_in.read())
    file_in.close()

    '''Direct submission to SGE, skipping ionJobServer, database Results entry'''
    if is_wholechip:
        resultDir = "./"
    elif is_thumbnail:
        resultDir = "./"
        if not "thumbnail" in TMP_PARAMS["pathToData"]:
            TMP_PARAMS["pathToData"] = path.join(TMP_PARAMS["pathToData"], 'thumbnail')
            env['pathToRaw'] = TMP_PARAMS["pathToData"]
    else:
        resultDir = './%s%s' % ('block_', blockObj['id_str'])
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
        TMP_PARAMS["pathToData"] = path.join(TMP_PARAMS["pathToData"], blockObj['id_str'])

    TMP_PARAMS["analysisArgs"] = ' '.join(blockObj['jobcmd'])
    file_out = open("%s/ion_params_00.json" % resultDir, 'w')
    json.dump(TMP_PARAMS, file_out)
    file_out.close()

    shutil.copy("/opt/ion/iondb/BlockTLScript.py", "%s/BlockTLScript.py" % resultDir)

    jobid = spawn_sge_job(resultDir)

    return jobid

def runFullChip(env):
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

    #-------------------------------------------------------------
    # Gridded data processing
    #-------------------------------------------------------------
    write_version()

    printtime("RUNNING FULL CHIP MULTI-BLOCK ANALYSIS")
    # List of block objects to analyze
    blocks = GetBlocksToAnalyze(env)
    dirs = ['block_%s' % block['id_str'] for block in blocks]

    #####################################################
    # Create block reports                              #
    #####################################################
    if is_blockprocessing:
        cwd = os.getcwd()
        for rdir in dirs:
            if not os.path.exists(rdir):
                os.mkdir(rdir)
            r = subprocess.call(["ln", "-s", os.path.join(cwd,"Default_Report.php"), os.path.join(rdir,"Default_Report.php")])
            if r:
                printtime("couldn't create symbolic link")
            r = subprocess.call(["ln", "-s", os.path.join(cwd,"parsefiles.php"), os.path.join(rdir,"parsefiles.php")])
            if r:
                printtime("couldn't create symbolic link")
            r = subprocess.call(["ln", "-s", os.path.join(cwd,"DefaultTFs.conf"), os.path.join(rdir,"DefaultTFs.conf")])
            if r:
                printtime("couldn't create symbolic link")

    #TODO
    doblocks = 1
    if doblocks:

        # Launch multiple SGE Block analysis jobs
        for block in blocks:
            block['jobid'] = submitBlockJob(block,env)
            printtime("Submitted block (%s) analysis job to SGE with job ID (%s)" % (block['id_str'], str(block['jobid'])))

        job_list=[block['jobid'] for block in blocks]

        # Watch status of jobs.  As they finish, run the merge script and remove
        # the job from the list.

        while len(job_list) > 0:
            for job in job_list:
                block = [block for block in blocks if block['jobid'] == job][0]
                #check status of SGE jobid
                block['status'] = _session.jobStatus(block['jobid'])
                if block['status']==drmaa.JobState.DONE or block['status']==drmaa.JobState.FAILED:
                    printtime("Job %s has ended with status %s" % (str(block['jobid']),block['status']))
                    job_list.remove(block['jobid'])
#                else:
#                    printtime("Job %s has status %s" % (str(block['jobid']),block['status']))

                sys.stdout.flush()
                sys.stderr.flush()
            time.sleep (5)
            printtime("waiting for %d blocks to be finished" % len(job_list))

    printtime("All jobs processed")

    tfKey = "ATCG"
    libKey = env['libraryKey']
    floworder = env['flowOrder']
    printtime("Using flow order: %s" % floworder)
    printtime("Using library key: %s" % libKey)

    #--------------------------------------------------------
    # Start merging results files
    #--------------------------------------------------------
    if is_thumbnail or is_wholechip:
        printtime("MERGING: THUMBNAIL OR 31X - skipping merge process.")

        res = models.Results.objects.get(pk=env['primary_key'])
        res.metaData["thumb"] = 1
        #res.timeStamp = datetime.datetime.now()
        res.save()
        printtime("thumbnail: "+str(res.metaData["thumb"]))

    else:
        printtime("MERGING: start merging ")
        #####################################################
        # Grab one of the processParameters.txt files       #
        #####################################################
        printtime("Merging processParameters.txt")

        for dir in dirs:
            ppfile = os.path.join(dir,'processParameters.txt')
            if os.path.isfile(ppfile):
                processParametersMerge.processParametersMerge(ppfile,True)
                break

        ############################################
        # Merge individual quality.summary files #
        ############################################
        printtime("Merging individual quality.summary files")

        config_out = ConfigParser.RawConfigParser()
        config_out.optionxform = str # don't convert to lowercase
        config_out.add_section('global')

        numberkeys = ['Number of 50BP Reads at Q0', 'Number of 150BP Reads at Q0', 'Number of 150BP Reads',
                'Number of 100BP Reads', 'Number of Bases at Q17', 'Number of 50BP Reads',
                'Number of 150BP Reads at Q17', 'Number of Bases at Q20', 'Number of 50BP Reads at Q20',
                'Number of Bases at Q0', 'Number of 100BP Reads at Q20', 'Number of Reads at Q0',
                'Number of 100BP Reads at Q17', 'Number of 50BP Reads at Q17', 'Number of Reads at Q20',
                'Number of 150BP Reads at Q20', 'Number of 100BP Reads at Q0', 'Number of Reads at Q17']

        maxkeys = ['Max Read Length at Q17','Max Read Length at Q20', 'Max Read Length at Q0']

        meankeys = ['Mean Read Length at Q17', 'Mean Read Length at Q20', 'Mean Read Length at Q0']

        config_in = MyConfigParser()
        for i,dir in enumerate(dirs):
            summaryfile=os.path.join(dir, 'quality.summary')
            if not path.exists(summaryfile):
                printtime("ERROR: skipped %s" % summaryfile)

            config_in.optionxform = str # don't convert to lowercase
            config_in.read(summaryfile)
            for key in numberkeys:
                value_in = config_in.get('global',key)
                if i==0:
                    value_out = 0
                else:
                    value_out = config_out.get('global', key)
                config_out.set('global', key, int(value_in) + int(value_out))
            for key in maxkeys:
                value_in = config_in.get('global',key)
                if i==0:
                    value_out = 0
                else:
                    value_out = config_out.get('global', key)
                config_out.set('global', key, max(int(value_in),int(value_out)))
            for key in meankeys:
                value_in = config_in.get('global',key)
                if i==0:
                    value_out = 0
                else:
                    value_out = config_out.get('global', key)
                config_out.set('global', key, float(value_out)+float(value_in)/len(dirs))

        with open('quality.summary', 'wb') as configfile:
            config_out.write(configfile)

        ############################################
        # Merge individual alignment.summary files #
        ############################################
        printtime("Merging individual alignment.summary files")

        config_out = ConfigParser.RawConfigParser()
        config_out.optionxform = str # don't convert to lowercase
        config_out.add_section('global')

        fixedkeys = [ 'Genome', 'Genome Version', 'Index Version', 'Genomesize' ]

        numberkeys = ['Total number of Reads',
                      'Filtered Mapped Bases in Q7 Alignments', 'Filtered Mapped Bases in Q10 Alignments',
                      'Filtered Mapped Bases in Q17 Alignments', 'Filtered Mapped Bases in Q20 Alignments',
                      'Filtered Mapped Bases in Q47 Alignments',
                      'Filtered Q7 Alignments', 'Filtered Q10 Alignments', 'Filtered Q17 Alignments',
                      'Filtered Q20 Alignments', 'Filtered Q47 Alignments']

        maxkeys = ['Filtered Q7 Longest Alignment','Filtered Q10 Longest Alignment','Filtered Q17 Longest Alignment',
                   'Filtered Q20 Longest Alignment','Filtered Q47 Longest Alignment']

        meankeys = ['Filtered Q7 Mean Alignment Length','Filtered Q10 Mean Alignment Length',
                    'Filtered Q17 Mean Alignment Length', 'Filtered Q20 Mean Alignment Length', 'Filtered Q47 Mean Alignment Length',
                    'Filtered Q7 Coverage Percentage', 'Filtered Q10 Coverage Percentage', 'Filtered Q17 Coverage Percentage',
                    'Filtered Q20 Coverage Percentage', 'Filtered Q47 Coverage Percentage', 'Filtered Q7 Mean Coverage Depth',
                    'Filtered Q10 Mean Coverage Depth', 'Filtered Q17 Mean Coverage Depth', 'Filtered Q20 Mean Coverage Depth',
                    'Filtered Q47 Mean Coverage Depth']

        config_in = MyConfigParser()
        for i,dir in enumerate(dirs):
            config_in.optionxform = str # don't convert to lowercase
            config_in.read(os.path.join(dir, 'alignment.summary'))
            for key in numberkeys:
                value_in = config_in.get('global',key)
                if i==0:
                    value_out = 0
                else:
                    value_out = config_out.get('global', key)
                config_out.set('global', key, int(value_in) + int(value_out))
            for key in maxkeys:
                value_in = config_in.get('global',key)
                if i==0:
                    value_out = 0
                else:
                    value_out = config_out.get('global', key)
                config_out.set('global', key, max(int(value_in),int(value_out)))
            for key in fixedkeys:
                value_in = config_in.get('global',key)
                if i==0:
                    config_out.set('global', key, value_in)
            for key in meankeys:
                value_in = config_in.get('global',key)
                if i==0:
                    value_out = 0
                else:
                    value_out = config_out.get('global', key)
                config_out.set('global', key, float(value_out)+float(value_in)/len(dirs))

        with open('alignment.summary', 'wb') as configfile:
            config_out.write(configfile)



        #########################################
        # Merge individual alignTable.txt files #
        #########################################
        printtime("Merging individual alignTable.txt files")

        table = 0
        header = None
        for dir in dirs:
            alignTableFile = os.path.join(dir,'alignTable.txt')
            if os.path.exists(alignTableFile):
                if header is None:
                    header = numpy.loadtxt(alignTableFile, dtype='string', comments='#')
                table += numpy.loadtxt(alignTableFile, dtype='int', comments='#',skiprows=1)
            else:
                printtime("ERROR: skipped %s" % alignTableFile)
        #fix first column
        table[:,0] = (header[1:,0])
        f_handle = open('alignTable.txt', 'w')
        numpy.savetxt(f_handle, header[0][None], fmt='%s', delimiter='\t')
        numpy.savetxt(f_handle, table, fmt='%i', delimiter='\t')
        f_handle.close()

        ########################################
        # Merge individual block SFF files     #
        ########################################
        printtime("Merging Library SFF files")
        try:
            cmd = 'SFFMerge'
            cmd = cmd + ' -i rawlib.sff'
            cmd = cmd + ' -o %s ' % libsff
            for dir in dirs:
                rawlibsff = os.path.join(dir,'rawlib.sff')
                if path.exists(rawlibsff):
                    cmd = cmd + ' %s' % dir
                else:
                    printtime("ERROR: skipped %s" % rawlibsff)
            printtime("DEBUG: Calling '%s'" % cmd)
            call(cmd,shell=True)
        except:
            printtime("SFFMerge failed (library)")

        printtime("Merging Test Fragment SFF files")
        try:
            cmd = 'SFFMerge'
            cmd = cmd + ' -i rawtf.sff'
            cmd = cmd + ' -o %s ' % tfsff
            for dir in dirs:
                rawtfsff = os.path.join(dir,'rawtf.sff')
                if path.exists(rawtfsff):
                    cmd = cmd + ' %s' % dir
                else:
                    printtime("ERROR: skipped %s" % rawtfsff)
            printtime("DEBUG: Calling '%s'" % cmd)
            call(cmd,shell=True)
        except:
            printtime("SFFMerge failed (test fragments)")

        ########################################
        # Merge individual block bead metrics files     #
        ########################################
        printtime("Merging individual block bead metrics files")

        try:
            cmd = 'BeadmaskMerge -i bfmask.bin -o bfmask.bin '
            for dir in dirs:
                bfmaskbin = os.path.join(dir,'bfmask.bin')
                if path.exists(bfmaskbin):
                    cmd = cmd + ' %s' % dir
                else:
                    printtime("ERROR: skipped %s" % bfmaskbin)
            printtime("DEBUG: Calling '%s'" % cmd)
            call(cmd,shell=True)
        except:
            printtime("BeadmaskMerge failed (test fragments)")

        maskpath = "MaskBead.mask"
        printtime("Merging MaskBead.mask files")
        bfmaskfiles = [dir for dir in dirs if path.isfile(os.path.join(dir,'bfmask.bin'))]

        if len(bfmaskfiles) > 0:
            offset_str = "use_blocks"
            MaskMerge.main_merge('MaskBead',bfmaskfiles, True, offset_str)
    #            for dir in dirs:
    #                if os.path.exists(os.path.join(dir,maskpath)):
    #                    os.remove(os.path.join(dir,maskpath))
    #                else:
    #                    printtime("Warning: no MaskBead.mask file exists.")
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

        #############################################
        # Merge individual block bam files   #
        #############################################
        printtime("Merging bam files")
        try:
    #        cmd = 'picard-tools MergeSamFiles'
            cmd = 'java -Xmx2g -jar /opt/picard/picard-tools-current/MergeSamFiles.jar'
            for dir in dirs:
    #            bamfile = "%s/%s_%s.bam" % (dir, env['expName'], env['resultsName'])
                bamfile = "%s/rawlib.bam" % (dir)
                if path.exists(bamfile):
                    cmd = cmd + ' I=%s' % bamfile
                else:
                    printtime("ERROR: skipped %s" % bamfile)
            cmd = cmd + ' O=%s_%s.bam' % (env['expName'], env['resultsName'])
            cmd = cmd + ' ASSUME_SORTED=true'
            cmd = cmd + ' USE_THREADING=true'
            cmd = cmd + ' VALIDATION_STRINGENCY=LENIENT'
            printtime("DEBUG: Calling '%s'" % cmd)
            call(cmd,shell=True)
        except:
            printtime("bam file merge failed")

        sys.stdout.flush()
        sys.stderr.flush()

        ##################################################
        #generate TF Metrics                             #
        #look for both keys and append same file         #
        ##################################################
        printtime("Calling TFMapper")

        output = "TFMapper.stats"
        block_tfsff = "rawtf.sff"
        block_libsff = "rawlib.sff"

        #Iterative TFMapper
        try:
            cmd = "TFMapperIterative -m 0 --flow-order=%s --tfkey=%s --libkey=%s %s " % (floworder, tfKey, libKey, block_tfsff)
            for dir in dirs:
                rawtfsff = os.path.join(dir,block_tfsff)
                if path.exists(rawtfsff):
                    cmd = cmd + ' %s' % dir
                else:
                    printtime("ERROR: skipped %s" % rawtfsff)
            cmd = cmd + " > %s" % output
            printtime("DEBUG: Calling '%s'" % cmd)
            os.system(cmd)
        except:
            printtime("TFMapper failed")

        try:
            cmd = "TFMapperIterative -m 1 --flow-order=%s --tfkey=%s --libkey=%s %s " % (floworder, tfKey, libKey, block_libsff)
            for dir in dirs:
                rawlibsff = os.path.join(dir,block_libsff)
                if path.exists(rawtfsff):
                    cmd = cmd + ' %s' % dir
                else:
                    printtime("ERROR: skipped %s" % rawlibsff)
            cmd = cmd + " >> %s" % output
            printtime("DEBUG: Calling '%s'" % cmd)
            os.system(cmd)
        except:
            printtime("TFMapper failed")

        # These settings do not reflect reality of block processing but are intended
        # to 'clean up' the progress indicators only.
        # create analysis progress bar file
        f = open('progress.txt','w')
        f.write('wellfinding = green\n')
        f.write('signalprocessing = green\n')
        f.write('basecalling = green\n')
        f.write('sffread = green\n')
        f.write('alignment = green')
        f.close()

    ## END MERGING

    ########################################################
    #generate the TF Metrics including plots               #
    ########################################################
    printtime("generate the TF Metrics including plots")

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
    #ParseFiles and Upload Metrics                         #
    ########################################################
    printtime("Attempting to Upload to Database")
    sys.stdout.flush()
    sys.stderr.flush()

    beadPath = 'bfmask.stats'
    lib_cafiePath = 'lib_cafie.txt'
    libPath = 'alignment.summary'
    QualityPath = 'quality.summary'
    filterMetrics = None

    #TODO, need to merge cafieMetrics.txt files
    printtime("TODO: need to merge cafieMetrics.txt files")
    cafieMetrics = None
    processMetrics = None
    #libMetrics ... copy from alignment.summary
    peakOut = 'raw_peak_signal'

    if os.path.isfile(beadPath):
        try:
            beadMetrics = parseBeadfind.generateMetrics(beadPath)
        except:
            printtime('generating beadMetrics failed')
            beadMetrics = None
            traceback.print_exc()
    else:
        printtime('generating beadMetrics failed - file %s is missing' % beadPath)
        beadMetrics = None

    #attempt to upload the metrics to the Django database
    try:
        uploadMetrics.populateDb(tfMetrics, cafieMetrics, processMetrics,
                                 beadMetrics, filterMetrics, libPath, STATUS,
                                 peakOut, lib_cafiePath, QualityPath)
    except:
        traceback.print_exc()

    # this will replace the five progress squares with a re-analysis button
    uploadMetrics.updateStatus("Completed", True)

    sys.stdout.flush()
    sys.stderr.flush()

    ########################################################
    #Make Bead Density Plots                               #
    ########################################################
    printtime("Make Bead Density Plots")
    sys.stdout.flush()
    sys.stderr.flush()

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
#            os.remove(maskpath)
        except:
            traceback.print_exc()
    else:
        printtime("Warning: no MaskBead.mask file exists.")

    sys.stdout.flush()
    sys.stderr.flush()

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
                #https://iontorrent.jira.com/wiki/display/TS/Plugin+json+file+format
                env['analysis_dir'] = os.getcwd()
                env['testfrag_key'] = 'ATCG'
                printtime("RAWDATA: %s" % env['pathToRaw'])
                start_json = make_plugin_json(env,plugin,env['primary_key'],basefolder,url_root)
                ret = launcher.callPluginXMLRPC(start_json, env['master_node'], iondb.settings.IPLUGIN_PORT)
            except:
                printtime('plugin %s failed...' % plugin['name'])
                traceback.print_exc()

    sys.stdout.flush()
    sys.stderr.flush()
    
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
        printtime("Cannot open file %s" % inputFile)
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
                    printtime("Cannot open file %s" % outputFile)
        line = f.readline()

    f.close()
    return False

def get_pgm_log_files(rawdatadir):
    # Create a tarball of the pgm raw data log files for inclusion into CSA.
    # tarball it now before the raw data gets deleted.
    files=['explog_final.txt',
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
        printtime('* %s%s%s *' % (str(l),spacer,str(d).replace('\n',' ')))
    print('*'*_info_width)

    #TODO, does it belong to blockTLScript?
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
    file = open(sys.argv[1], 'r')
    EXTERNAL_PARAMS = json.loads(file.read())
    file.close()
    for k,v in EXTERNAL_PARAMS.iteritems():
        if isinstance(v, unicode):
            EXTERNAL_PARAMS[k] = str(v)
    env["params_file"] = 'ion_params_00.json'

    # Where the raw data lives (generally some path on the network)
    pathprefix = str(EXTERNAL_PARAMS['pathToData'])
    env['prefix'] = pathprefix
    # this is the library name for the run taken from the library field in the database
    env["libraryName"] = EXTERNAL_PARAMS.get("libraryName", "none")
    if env["libraryName"]=="":
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

    try:
        primary_key = open("primary.key").readline()
        primary_key = primary_key.split(" = ")
        env['primary_key'] = primary_key[1]
        printtime(env['primary_key'])
    except:
        printtime("Error, unable to get the primary key")

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

    # assemble the URL path for this analysis result
    # http://<servername>/<output dir>/<Location name>/<analysis dir>
    url_scheme = 'http'
    url_location = env['net_location'].replace('http://','')
    url_path = os.path.join(env['url_path'],os.path.basename(os.getcwd()))
    url_root = urlunsplit([url_scheme,url_location,url_path,"",""])
    printtime("URL string %s" % url_root)

    #get the experiment json data
    env['exp_json'] = EXTERNAL_PARAMS.get('exp_json')
    #get the name of the site
    env['site_name'] = EXTERNAL_PARAMS.get('site_name')

    is_thumbnail = False
    is_wholechip = False
    is_blockprocessing = False

    if env['rawdatastyle'] == 'single':
        is_wholechip = True
    else:
        #TODO
        if "_tn" in env['resultsName']:
           is_thumbnail = True
        else:            
           is_blockprocessing = True

    #drops a zip file of the pgm log files
    get_pgm_log_files(env['pathToRaw'])
    
    # Generate a system information file for diagnostics porpoises.
    try:
        com="/opt/ion/iondb/bin/collectInfo.sh >> ./sysinfo.txt"
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

    try:
        for k in ("SGE_ROOT", "SGE_CELL", "SGE_CLUSTER_NAME",
                  "SGE_QMASTER_PORT", "SGE_EXECD_PORT", "DRMAA_LIBRARY_PATH"):
            if not k in os.environ:
                os.environ[k] = str(getattr(iondb.settings,k))
        _session = drmaa.Session()
        _session.initialize()
        print 'session initialized'
    except:
        print "Unexpected error:", sys.exc_info()
        sys.exit(1)

    runFullChip(env)

    _session.exit

    getExpLogMsgs(env)
    printtime("Run Complete")
    sys.exit(0)
