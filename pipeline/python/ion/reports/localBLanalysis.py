#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Loads OneBlock analysis pipeline
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

import datetime
import shutil
import socket
import subprocess
from subprocess import *
import sys
import time
import pickle
from collections import deque
from urlparse import urlunsplit
import glob

# Import analysis-specific packages. We try to follow the from A import B
# convention when B is a module in package A, or import A when
# A is a module.
#from ion.analysis import cafie,sigproc
#from ion.fileformats import sff
#from ion.reports import blast_to_ionogram, tfGraphs, plotKey, \
#    parseBeadfind, parseProcessParams, beadDensityPlot, \
#    libGraphs, beadHistogram, plotRawRegions, trimmedReadLenHisto

#from ion.reports.plotters import *
#from ion.utils.aggregate_alignment import *
#from ion.utils.align_full_chip import *
#from iondb.plugin_json import *

#from iondb.anaserve import client

#sys.path.append('/opt/ion/')
#os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
#import iondb.settings
#from iondb.plugins import *

import math
from scipy import cluster, signal, linalg
import traceback
import json

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

#####################################################################
#
# Analysis implementation details
#
#####################################################################

def printtime(message, *args):
    if args:
        message = message % args
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
    #TODO, temporary commented out
    #shutil.copy("%s/parsefiles.php" % TMPL_DIR, "parsefiles.php")
    shutil.copy("%s/log.html" % TMPL_DIR, 'log.html')
    shutil.copy("%s/alignment_summary.html" % TMPL_DIR, os.path.join(ALIGNMENT_RESULTS,"alignment_summary.html"))
    #shutil.copy("%s/format_whole.php" % TMPL_DIR, "Default_Report.php")
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


def makeAlignGraphs():
    ###################################################
    # Make histograms graphs from tmap output         #
    ###################################################
    qualdict = {}
    qualdict['Q10'] = 'red'
    qualdict['Q17'] = 'yellow'
    qualdict['Q20'] = 'green'
    qualdict['Q47'] = 'blue'
    for qual in qualdict.keys():
        dpath = qual + ".histo.dat"
        if os.path.exists(dpath):
            try:
                data = libGraphs.parseFile(dpath)
                it = libGraphs.QXXAlign(data, "Filtered "+qual+" Read Length", 'Count',
                                        "Filtered "+qual+" Read Length", qualdict[qual], "Filtered_Alignments_"+qual+".png")
                it.plot()
            except:
                traceback.print_exc()
                
def align_full_chip(libsff, libKey, tfKey, floworder, fastqName, align_full, DIR_BC_FILES, env, outputdir):
    #call whole chip metrics script (as os.system because all output to file)
    if os.path.exists('alignment.summary'):
        try:
            os.rename('alignment.summary', 'alignment.summary_%s' % datetime.datetime.now())
        except:
            printtime('error renaming')
            traceback.print_exc()
    align_full_chip_core(libsff, libKey, tfKey, floworder, fastqName, align_full, 1, True, False, False, DIR_BC_FILES, env, outputdir)

def align_full_chip_core(libsff, libKey, tfKey, floworder, fastqName, align_full, graph_max_x, do_barcode, make_align_graphs, sam_parsed, DIR_BC_FILES, env, outputdir):
    #collect all the meta data for the SAM file
    SAM_META = {}

    # id - this hash comes from the fastq file
    try:
        #read the first line of the fastq file
        fastqFile = open(fastqName,'r')
        id_hash = fastqFile.readline()
        fastqFile.close()

        #now just pull out the hash
        id_hash = id_hash[1:6]

        SAM_META['ID'] = id_hash

    except IOError:
        printtime("Could not read fastq file.  The ID for the SAM file could not be found.")

    # sm - name for reads - project name
    SAM_META['SM'] = env['project']

    # lb - library name
    SAM_META['LB'] = env['libraryName']

    # pu - the platform unit
    SAM_META['PU'] = "PGM/" + env['chipType'].replace('"',"")

    SAM_META['PL'] = "IONTORRENT"

    #TODO: do not assume localhost.  Find the name of the masternode
    try:
        #this will get the exp data from the database
        #exp_json = json.loads(env['exp_json'])

        # ds - the "notes", only the alphanumeric and space characters.
        #SAM_META['DS'] = ''.join(ch for ch in exp_json['notes'] if ch.isalnum() or ch == " ")

        # dt - the run date
        #exp_log_json = json.loads(exp_json['log'])
        #iso_exp_time = exp_log_json['start_time']

        #convert to ISO time
        #iso_exp_time = dateutil.parser.parse(iso_exp_time)

        #SAM_META['DT'] = iso_exp_time.isoformat()

        #the site name should be here, also remove spaces
        site_name = env['site_name']
        site_name = ''.join(ch for ch in site_name if ch.isalnum() )
        SAM_META['CN'] = site_name

        #env['flows'] = exp_json['flows']

    except:
        printtime("There was an error getting the site name, because the Torrent Browser could not be contacted")
        traceback.print_exc()

    #Now build the SAM meta data arg string
    aligner_opts_rg= '--aligner-opts-rg "'
    aligner_opts_extra = ''
    if sam_parsed:
        aligner_opts_extra += ' -p 1'
    if env['aligner_opts_extra']:
        print '  found extra alignment options: "%s"' % env['aligner_opts_extra']
        aligner_opts_extra = ' --aligner-opts-extra "'
        aligner_opts_extra += env['aligner_opts_extra'] + '"'
    first = True
    for key, value in SAM_META.items():
        if value:
            sam_arg =  r'-R \"'
            end =  r'\"'

            sam_arg = sam_arg + key + ":" + value + end

            if first:
                aligner_opts_rg = aligner_opts_rg + sam_arg
                first = False
            else:
                aligner_opts_rg = aligner_opts_rg + " " + sam_arg

    #add the trailing quote
    aligner_opts_rg = aligner_opts_rg + '"'

    if 0 < graph_max_x:
        # establish the read-length histogram range by using the simple rule: 0.6 * num-flows
        flowsUsed = 0
        try:
            flowsUsed = int(env['flows'])
        except:
            flowsUsed = 400
        graph_max_x = 100 * math.trunc((0.6 * flowsUsed + 99)/100.0)
    if graph_max_x < 400:
        graph_max_x = 400

    #-----------------------------------
    # DEFAULT SINGLE SFF/FASTQ BEHAVIOR - (Runs for barcoded runs too)
    #-----------------------------------
    if (align_full):
        #If a full align is forced add a '--align-all-reads' flag
        com = "alignmentQC.pl"
        com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
        com += " --output-dir %s" % outputdir
        com += " --input %s" % libsff
        com += " --genome %s" % env["libraryName"]
        com += " --max-plot-read-len %s" % graph_max_x
        com += " --align-all-reads"
        com += " %s %s" % (aligner_opts_rg,aligner_opts_extra)
        com += " >> ReportLog.html 2>&1"
    else:
        # Add -p 1 to enable default.sam file generation
        com = "alignmentQC.pl"
        com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
        com += " --output-dir %s" % outputdir
        com += " --input %s" % libsff
        com += " --genome %s" % env["libraryName"]
        com += " --max-plot-read-len %s" % graph_max_x
        com += " %s %s" % (aligner_opts_rg,aligner_opts_extra)
        com += " >> ReportLog.html 2>&1"

    try:
        printtime("Alignment QC command line:\n%s" % com)
        retcode = subprocess.call(com, shell=True)
        if retcode != 0:
            printtime("alignmentQC failed, return code: %d" % retcode)
            alignError = open("alignment.error", "w")
            alignError.write('alignmentQC returned with error code: ')
            alignError.write(str(retcode))
            alignError.close()
    except OSError:
        printtime('Alignment Failed to start')
        alignError = open("alignment.error", "w")
        alignError.write(str(traceback.format_exc()))
        alignError.close()
        traceback.print_exc()
    if make_align_graphs:
        makeAlignGraphs()

    #--------------------------------------------
    # BARCODE HANDLING BEHAVIOR (Multiple FASTQ)
    #--------------------------------------------
    if env['barcodeId'] and True == do_barcode:
        printtime("Renaming non-barcoded alignment results to 'comprehensive'")
        files = [ 'alignment.summary',
                  'alignmentQC_out.txt',
                  'alignTable.txt',
                ]
        for fname in files:
            try:
                if os.path.exists(fname):
                    os.rename(fname, fname + ".comprehensive")
            except:
                printtime('error renaming')
                traceback.print_exc()
        # Only make the graphs from the alignment of comprehensive fastq file
	if make_align_graphs:
            makeAlignGraphs()

        printtime("STARTING BARCODE ALIGNMENTS")
        if not os.path.exists(DIR_BC_FILES):
            os.mkdir(DIR_BC_FILES)

        barcodeList = parse_bcfile('barcodeList.txt')

        align_full = True
        for bcid in (x['id_str'] for x in barcodeList):
            sffName = "%s_%s_%s.sff" % (bcid, env['expName'], env['resultsName'])
            if not os.path.exists(sffName):
                printtime("No barcode SFF file found for '%s'" % bcid)
                continue
            if (align_full):
                printtime("Align All Reads")
                #If a full align is forced add a '--align-all-reads' flag
                com = "alignmentQC.pl" 
                com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
                com += " --output-dir %s" % outputdir
                com += " --input %s" % sffName
                com += " --genome %s" % env["libraryName"]
                com += " --max-plot-read-len %s" % graph_max_x
                com += " --align-all-reads"
                com += " %s %s" % (aligner_opts_rg, aligner_opts_extra)
                com += " >> ReportLog.html 2>&1" 
            else:
                printtime("Align Subset of Reads")
                # Add -p 1 to enable default.sam file generation
                com = "alignmentQC.pl" 
                com += " --logfile %s" % os.path.join(outputdir,"alignmentQC_out.txt")
                com += " --output-dir %s" % outputdir
                com += " --input %s" % sffName
                com += " --genome %s" % env["libraryName"]
                com += " --max-plot-read-len %s" % graph_max_x
                com += " %s %s" % (aligner_opts_rg, aligner_opts_extra)
                com += " >> ReportLog.html 2>&1"
            try:
                printtime("Alignment QC command line:\n%s" % com)
                retcode = subprocess.call(com, shell=True)
                if retcode != 0:
                    printtime("alignmentQC failed, return code: %d" % retcode)
                    alignError = open("alignment.error", "a")
                    alignError.write(com)
                    alignError.write(': \nalignmentQC returned with error code: ')
                    alignError.write(str(retcode))
                    alignError.close()
            except OSError:
                printtime('Alignment Failed to start')
                alignError = open("alignment.error", "a")
                alignError.write(str(traceback.format_exc()))
                alignError.close()
                traceback.print_exc()

            #rename each output file based on barcode found in fastq filename
            #but ignore the comprehensive fastq output files
            if os.path.exists('alignment.summary'):
                try:
                    fname='alignment_%s.summary' % bcid
                    os.rename('alignment.summary', fname)
                    os.rename(fname,os.path.join(DIR_BC_FILES,fname))
                    fname='alignmentQC_out_%s.txt' % bcid
                    os.rename('alignmentQC_out.txt', fname)
                    os.rename(fname,os.path.join(DIR_BC_FILES,fname))
                    fname='alignTable_%s.txt' % bcid
                    os.rename('alignTable.txt', fname)
                    os.rename(fname,os.path.join(DIR_BC_FILES,fname))

                    #move fastq, sff, bam, bai files
                    extlist = ['fastq','sff','bam','bam.bai']
                    for ext in extlist:
                        bcfile = "%s_%s_%s.%s" % (bcid,env['expName'], env['resultsName'],ext)
                        if os.path.isfile(bcfile):
                            os.rename(bcfile,os.path.join(DIR_BC_FILES,bcfile))
                except:
                    printtime('error renaming')
                    traceback.print_exc()
        #rename comprehensive results back to default names
        files = [ 'alignment.summary',
                  'alignmentQC_out.txt',
                  'alignTable.txt',
                ]
        for fname in files:
            if os.path.exists(fname + '.comprehensive'):
                os.rename(fname + '.comprehensive', fname)

        aggregate_alignment (DIR_BC_FILES,'barcodeList.txt')
        
def runBlock(env):
    STATUS = None
    basefolder = 'plugin_out'
    if not os.path.isdir(basefolder):
        os.umask(0000)   #grant write permission to plugin user
        os.mkdir(basefolder)
        os.umask(0002)
    pathprefix = env["prefix"]
    libsff_filename = "rawlib.sff"
    tfsff_filename = "rawtf.sff"
    fastq_filename = "raw.fastq"
    bctrimmed_libsff_filename = "bctrimmed_rawlib.sff"

    fastq_path = os.path.join(BASECALLER_RESULTS, fastq_filename)
    libsff_path = os.path.join(BASECALLER_RESULTS, libsff_filename)
    tfsff_path = os.path.join(BASECALLER_RESULTS, tfsff_filename)
    bctrimmed_libsff_path =  os.path.join(BASECALLER_RESULTS,bctrimmed_libsff_filename)
    tfmapperstats_path = os.path.join(BASECALLER_RESULTS,"TFMapper.stats")

    libKeyArg = "--libraryKey=%s" % env["libraryKey"]

    #-------------------------------------------------------------
    # Single Block data processing
    #-------------------------------------------------------------
    if runFromRaw:
        printtime("RUNNING SINGLE BLOCK ANALYSIS")
        command = "%s >> ReportLog.html 2>&1" % (env['analysisArgs'])
        printtime("Analysis command: " + command)
        sys.stdout.flush()
        sys.stderr.flush()
        status = call(command,shell=True)
        #status = 2
        STATUS = None
        if int(status) == 2:
            STATUS = 'Checksum Error'
        elif int(status) == 3:
            STATUS = 'No Live Beads'
        elif int(status) != 0:
            STATUS = 'ERROR'

        if STATUS != None:
            printtime("Analysis finished with status '%s'" % STATUS)
            #TODO - maybe create file
            # uploadMetrics.updateStatus(STATUS)

        #TODO
        '''
        csp = os.path.join(env['pathToRaw'],'checksum_status.txt')
        if not os.path.exists(csp) and not env['skipchecksum'] and STATUS==None:
            try:
                os.umask(0002)
                f = open(csp, 'w')
                f.write(str(status))
                f.close()
            except:
                traceback.print_exc()
        '''
        printtime("Finished single block analysis")
    else:
        printtime('Skipping single block analysis')

    if runFromWells:
        tfKey = "ATCG"
        libKey = env['libraryKey']
        floworder = env['flowOrder']
        printtime("Using flow order: %s" % floworder)
        printtime("Using library key: %s" % libKey)

        if "block_" in mycwd:

            # Fix SFFTrim
            basecallerjson = os.path.join(BASECALLER_RESULTS, 'BaseCaller.json')
            r = subprocess.call(["ln", "-s", basecallerjson])
            if r:
                printtime("couldn't create symbolic link")

            # Fix SFFMerge
            r = subprocess.call(["ln", "-s", os.path.join('..', SIGPROC_RESULTS, 'processParameters.txt'), os.path.join(BASECALLER_RESULTS, 'processParameters.txt')])
            if r:
                printtime("couldn't create symbolic link")


        sys.stdout.flush()
        sys.stderr.flush()

        if not os.path.exists(libsff_path):
            printtime("ERROR: %s does not exist" % libsff_path)
            open('badblock.txt', 'w').close() 

        ##################################################
        # Unfiltered SFF
        ##################################################

        unfiltered_dir = "unfiltered"
        if os.path.exists(unfiltered_dir):

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
                printtime("Unfiltered SFFTrim")
                printtime("DEBUG: Calling '%s'" % com)
                ret = call(com,shell=True)
                if int(ret)!=0 and STATUS==None:
                    STATUS='ERROR'
            except:
                printtime('Failed Unfiltered SFFTrim')

            sffs = glob.glob("*.sff")
            for sff in sffs:
                try:
                    com = "SFFRead -q %s %s" % (sff.replace(".sff",".fastq"), sff)
                    printtime("DEBUG: Calling '%s'" % com)
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
                    align_full_chip_core("../*." + status + ".sff", libKey, tfKey, floworder, fastq_path, env['align_full'], -1, False, False, True, DIR_BC_FILES, env, ALIGNMENT_RESULTS)
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

            if not os.path.exists(libsff_path):
                printtime("ERROR: %s does not exist" % libsff_path)

            (head,tail) = os.path.split(libsff_path)
            libsff_trimmed_path = os.path.join(head,tail[:4] + "trimmed.sff")

            #we will always need the input and output files
            trimArgs = "--in-sff %s --out-sff %s" % (libsff_path,libsff_trimmed_path)

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
                printtime("using non default args" , env['sfftrim_args'])
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

# don't rename, result will be useless for --fromsff runs

#            if os.path.exists(libsff_path):
#                try:
#                    os.rename(libsff_path, "untrimmed." + libsff_path) #todo
#                except:
#                    printtime("ERROR: renaming %s" % libsff_path)

#            if os.path.exists(libsff_trimmed_path):
#                try:
#                    os.rename(libsff_trimmed_path, libsff_path)
#                except:
#                    printtime("ERROR: renaming %s" % libsff_trimmed_path)
        else:
            printtime("Not attempting to trim the SFF")


        #####################################################
        # Barcode trim SFF if barcodes have been specified  #
        # Creates one fastq per barcode, plus unknown reads #
        #####################################################

        if env['barcodeId'] is not '':
            try:
                com = 'barcodeSplit -s -i %s -b barcodeList.txt -c barcodeMask.bin -f %s' % (libsff_path,floworder)
                printtime("barcodeSplit", com)
                ret = call(com,shell=True)
                if int(ret) != 0 and STATUS==None:
                    STATUS='ERROR'
                else:
                    # Rename bc trimmed sff
                    if os.path.exists(bctrimmed_libsff):
                        os.rename(bctrimmed_libsff, libsff_path)
            except:
                printtime("Failed barcodeSplit")


        ##################################################
        # Once we have the new SFF, run SFFSummary
        # to get the predicted quality scores
        ##################################################

        try:
            com = "SFFSummary"
            com += " -o %s" % os.path.join(BASECALLER_RESULTS, 'quality.summary')
            com += " --sff-file %s" % libsff_path
            com += " --read-length 50,100,150"
            com += " --min-length 0,0,0"
            com += " --qual 0,17,20"
            com += " -d %s" % os.path.join(BASECALLER_RESULTS, 'readLen.txt')

            printtime("DEBUG: Calling '%s'" % com)
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
            com = "SFFRead -q %s" % fastq_path
            com += " %s" % libsff_path
            com += " > %s" % os.path.join(BASECALLER_RESULTS, 'keypass.summary')

            printtime("DEBUG: Calling '%s'" % com)
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

        try:
            com = "TFMapper -m 0 --logfile TFMapper.log --cafie-metrics"
            com += " --output-dir=%s" % (BASECALLER_RESULTS)
            com += " --wells-dir=%s" % (SIGPROC_RESULTS)
            com += " --sff-dir=%s" % (BASECALLER_RESULTS)
            com += " --tfkey=%s" % (tfKey)
            com += " --libkey=%s" % (libKey)
            com += " %s" % (tfsff_filename)
            com += " ./"
            com += " > %s" % (tfmapperstats_path)
            printtime("DEBUG: Calling '%s'" % com)
            os.system(com)
        except:
            printtime("ERROR: TFMapper failed")

        ########################################################
        #generate the TF Metrics including plots               #
        ########################################################
        printtime("generate the TF Metrics including plots")

        tfMetrics = None
#        if os.path.exists(tfmapperstats_path):
#            try:
#                # Q17 TF Read Length Plot
#                tfMetrics = parseTFstats.generateMetricsData(tfmapperstats_path)
#                tfGraphs.Q17(tfMetrics)
#                tfGraphs.genCafieIonograms(tfMetrics,floworder)
#            except Exception:
#                printtime("ERROR: Metrics Gen Failed")
#                traceback.print_exc()
#        else:
#            printtime("ERROR: %s doesn't exist" % tfmapperstats_path)
#            tfMetrics = None

        ########################################################
        #Generate Raw Data Traces for lib and TF keys          #
        ########################################################
#        printtime("Generate Raw Data Traces for lib and TF keys(iontrace_Test_Fragment.png, iontrace_Library.png)")
#
#        tfRawPath = 'avgNukeTrace_%s.txt' % tfKey
#        libRawPath = 'avgNukeTrace_%s.txt' % libKey
#        peakOut = 'raw_peak_signal'
#
#        if os.path.exists(tfRawPath):
#            try:
#                kp = plotKey.KeyPlot(tfKey, floworder, 'Test Fragment')
#                kp.parse(tfRawPath)
#                kp.dump_max(peakOut)
#                kp.plot()
#            except:
#                printtime("TF key graph didn't render")
#                traceback.print_exc()
#
#        if os.path.exists(libRawPath):
#            try:
#                kp = plotKey.KeyPlot(libKey, floworder, 'Library')
#                kp.parse(libRawPath)
#                kp.dump_max(peakOut)
#                kp.plot()
#            except:
#                printtime("Lib key graph didn't render")
#                traceback.print_exc()

        ########################################################
        #Make Bead Density Plots                               #
        ########################################################
#        printtime("Make Bead Density Plots")
#        bfmaskPath = os.path.join(SIGPROC_RESULTS,"bfmask.bin")
#        maskpath = os.path.join(SIGPROC_RESULTS,"MaskBead.mask")
#
#        if os.path.isfile(bfmaskPath):
#            com = "BeadmaskParse -m MaskBead %s" % bfmaskPath
#            os.system(com)
#            #TODO
#            try:
#                shutil.move('MaskBead.mask', maskpath)
#            except:
#                printtime("ERROR: MaskBead.mask already moved")
#        else:
#            printtime("Warning: no bfmask.bin file exists.")
#
#        if os.path.exists(maskpath):
#            try:
#                # Makes Bead_density_contour.png
#                beadDensityPlot.genHeatmap(maskpath, BASECALLER_RESULTS)
#    #            os.remove(maskpath)
#            except:
#                traceback.print_exc()
#        else:
#            printtime("Warning: no MaskBead.mask file exists.")

        sys.stdout.flush()
        sys.stderr.flush()

        ########################################################
        # Make beadfind histogram for every region             #
        ########################################################
#        printtime("Make beadfind histogram for every region")
#        procPath = 'processParameters.txt'
#        try:
#            processMetrics = parseProcessParams.generateMetrics(procPath)
#        except:
#            printtime("processMetrics failed")
#            processMetrics = None


        ########################################################
        # Make per region key incorporation traces             #
        ########################################################
#        printtime("Make per region key incorporation traces")
#        perRegionTF = "averagedKeyTraces_TF.txt"
#        perRegionLib = "averagedKeyTraces_Lib.txt"
#        if os.path.exists(perRegionTF):
#            pr = plotRawRegions.PerRegionKey(tfKey, floworder,'TFTracePerRegion.png')
#            pr.parse(perRegionTF)
#            pr.plot()
#
#        if os.path.exists(perRegionLib):
#            pr = plotRawRegions.PerRegionKey(libKey, floworder,'LibTracePerRegion.png')
#            pr.parse(perRegionLib)
#            pr.plot()


        sys.stdout.flush()
        sys.stderr.flush()
    else:
        printtime('Skipping SFF Processing')

    if runFromSFF:
        ########################################################
        #Attempt to align                                      #
        ########################################################
        printtime("Attempt to align")

        # create analysis progress bar file
        f = open('progress.txt','w')
        f.write('wellfinding = green\n')
        f.write('signalprocessing = green\n')
        f.write('basecalling = green\n')
        f.write('sffread = green\n')
        f.write('alignment = yellow')
        f.close()

        try:
            align_full_chip(libsff_path, libKey, tfKey, floworder, fastq_path, env['align_full'], DIR_BC_FILES, env, ALIGNMENT_RESULTS)
        except Exception:
            printtime("ERROR: Alignment Failed")
            traceback.print_exc()

#        printtime("make the read length histogram")
#        try:
#            filepath_readLenHistogram = os.path.join(ALIGNMENT_RESULTS,'readLenHisto.png')
#            trimmedReadLenHisto.trimmedReadLenHisto('readLen.txt',filepath_readLenHistogram)
#        except:
#            printtime("Failed to create %s" % filepath_readLenHistogram)

        ########################################################
        #ParseFiles                                            #
        ########################################################
        printtime('ParseFiles')

        # create analysis progress bar file
        f = open('progress.txt','w')
        f.write('wellfinding = green\n')
        f.write('signalprocessing = green\n')
        f.write('basecalling = green\n')
        f.write('sffread = green\n')
        f.write('alignment = green')
        f.close()

    else:
        printtime('Skipping TMAP Processing')

if __name__=="__main__":
    ########################################################
    # Print nice header information                        #
    ########################################################
    printtime("localBLanalysis.py: " + os.getcwd())
    sys.stdout.flush()
    sys.stderr.flush()

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

    sys.stdout.flush()
    sys.stderr.flush()

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
    afile = open(env["params_file"], 'r')
    EXTERNAL_PARAMS = json.loads(afile.read())
    afile.close()
    for k,v in EXTERNAL_PARAMS.iteritems():
        if isinstance(v, unicode):
            EXTERNAL_PARAMS[k] = str(v)

    # Where the raw data lives (generally some path on the network)
    pathprefix = str(EXTERNAL_PARAMS['pathToRaw'])
    env['prefix'] = pathprefix
    # this is the library name for the run taken from the library field in the database
    env["libraryName"] = EXTERNAL_PARAMS.get("libraryName", "none")
    if env["libraryName"] == "":
        env["libraryName"] = "none"
    #todo, workaround for thumbnails, to test tmap
    if env["libraryName"] == "none":
        env["libraryName"] = "e_coli_dh10b"
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
    env['libraryKey'] = EXTERNAL_PARAMS.get("libraryKey", 'TCAG')
    #path to the raw data
    #env['pathToRaw'] = EXTERNAL_PARAMS.get("pathToData")
    env['pathToRaw'] = EXTERNAL_PARAMS.get("pathToRaw")
    #plugins
    #env['plugins'] = EXTERNAL_PARAMS.get("plugins")
    # skipChecksum?
    #env['skipchecksum'] = EXTERNAL_PARAMS.get('skipchecksum',False)
    # Do Full Align?
    env['align_full'] = EXTERNAL_PARAMS.get('align_full')
    # Check to see if a SFFTrim should be done
    env['sfftrim'] = EXTERNAL_PARAMS.get('sfftrim')
    # Get SFFTrim args
    #env['sfftrim_args'] = EXTERNAL_PARAMS.get('sfftrim_args')
    env['flowOrder'] = EXTERNAL_PARAMS.get('flowOrder').strip()
    env['project'] = EXTERNAL_PARAMS.get('project')
    env['sample'] = EXTERNAL_PARAMS.get('sample')
    env['chipType'] = EXTERNAL_PARAMS.get('chiptype')
    env['barcodeId'] = EXTERNAL_PARAMS.get('barcodeId','')
    env['reverse_primer_dict'] = EXTERNAL_PARAMS.get('reverse_primer_dict')
    #env['rawdatastyle'] = EXTERNAL_PARAMS.get('rawdatastyle', 'single')
    env['blockArgs'] = EXTERNAL_PARAMS.get('blockArgs')
    env['flows'] = EXTERNAL_PARAMS.get('flows')
    env['site_name'] = EXTERNAL_PARAMS.get('site_name','')
    
    #extra JSON
    env['extra'] = EXTERNAL_PARAMS.get('extra', '')
    # Aligner options
    env['aligner_opts_extra'] = EXTERNAL_PARAMS.get('aligner_opts_extra', '')

    # define entry point
    if env['blockArgs'] == "fromRaw":
        runFromRaw = True
        runFromWells = True
        runFromSFF = True
    elif env['blockArgs'] == "fromWells":
        runFromRaw = False
        runFromWells = True
        runFromSFF = True
    elif env['blockArgs'] == "fromSFF":
        runFromRaw = False
        runFromWells = False
        runFromSFF = True
    else:
        runFromRaw = True
        runFromWells = True
        runFromSFF = True

    #get the experiment json data
    env['exp_json'] = EXTERNAL_PARAMS.get('exp_json')

    #############################################################
    # Code to start one block analysis                           #
    #############################################################

    mycwd=os.path.basename(os.getcwd())

    SIGPROC_RESULTS = "sigproc_results"
    BASECALLER_RESULTS = "basecaller_results"
    ALIGNMENT_RESULTS = "alignment_results"
    SIGPROC_RESULTS = "./"
    BASECALLER_RESULTS = "./"
    ALIGNMENT_RESULTS = "./"

    try:
        os.mkdir(SIGPROC_RESULTS)
    except:
        traceback.print_exc()

    try:
        os.mkdir(BASECALLER_RESULTS)
    except:
        traceback.print_exc()

    try:
        os.mkdir(ALIGNMENT_RESULTS)
    except:
        traceback.print_exc()

    # create symbolic links for merge process
    if "block_" in mycwd:

        _SIGPROC_RESULTS = os.path.join('..', SIGPROC_RESULTS, mycwd)
        _BASECALLER_RESULTS = os.path.join('..', BASECALLER_RESULTS, mycwd)
        _ALIGNMENT_RESULTS = os.path.join('..', ALIGNMENT_RESULTS, mycwd)

        r = subprocess.call(["ln", "-s", os.path.join('..', mycwd, SIGPROC_RESULTS), _SIGPROC_RESULTS])
        r = subprocess.call(["ln", "-s", os.path.join('..', mycwd, BASECALLER_RESULTS), _BASECALLER_RESULTS])
        r = subprocess.call(["ln", "-s", os.path.join('..', mycwd, ALIGNMENT_RESULTS), _ALIGNMENT_RESULTS])

    os.umask(0002)
#    initreports()
    logout = open("ReportLog.html", "w")
    logout.close()

    sys.stdout.flush()
    sys.stderr.flush()
	
    runBlock(env)
    printtime("Run Complete")
    sys.exit(0)
