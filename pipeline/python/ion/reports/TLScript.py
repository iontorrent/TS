#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Loads Fullchip analysis pipeline
Uses output from there to generate charts and graphs and dumps to current directory
Adds key metrics to database(not implemented yet)
"""

__version__ = filter(str.isdigit, "$Revision: 30539 $")

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

# import /etc/torrentserver/cluster_settings.py, provides JOBSERVER_HOST, JOBSERVER_PORT
import sys
sys.path.append('/etc')
from torrentserver.cluster_settings import *

import ConfigParser
import shutil
import socket
import subprocess
import time
import traceback
import fnmatch
import json
import xmlrpclib
from ion.utils import blockprocessing
from ion.utils import sigproc
from ion.utils import basecaller
from ion.utils import alignment
from ion.utils.compress import make_zip

from ion.utils.blockprocessing import printtime

from collections import deque
from urlparse import urlunsplit

# Import analysis-specific packages. We try to follow the from A import B
# convention when B is a module in package A, or import A when
# A is a module.
#from ion.analysis import cafie,sigproc
#from ion.fileformats import sff
from ion.reports import blast_to_ionogram, \
    parseBeadfind, parseProcessParams, beadDensityPlot, \
    libGraphs, beadHistogram

from ion.reports.plotters import *
from ion.utils.aggregate_alignment import *
sys.path.append('/opt/ion/')
import re

#####################################################################
#
# Analysis implementation details
#
#####################################################################

def getExpLogMsgs(env):
    """
    Parses explog_final.txt for warning messages and dumps them to
    ReportLog.html.
    This only works if the raw data files have not been deleted.
    For a from-wells analysis, you may not have raw data.
    """
    inputFile = os.path.join(env['pathToRaw'],'explog_final.txt')
    outputFile = os.path.join('./','ReportLog.html')
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
    
    # internal hack for easy access to experiment log file
    try:
        if os.path.isfile("/opt/ion/.ion-internal-server"):
            os.symlink(inputFile,'./explog_final.txt')
    except:
        printtime(traceback.format_exc())
        
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
    'InitValsW2.txt',
    'Controller',
    'debug']
    for afile in files:
        make_zip ('pgm_logs.zip',os.path.join(rawdatadir,afile), afile)

    return


def GetBlocksToAnalyze(env):
    blocks = []

    if is_thumbnail or is_wholechip:

        block = {'id_str':'',
                'datasubdir':'',
                'jobcmd':[],
                'jobid':None,
                'status':None}

        base_args = env['analysisArgs'].strip().split(' ')
        base_args.append("--libraryKey=%s" % env["libraryKey"])
        base_args.append("--tfKey=%s" % env["tfKey"])
        base_args.append("--no-subdir --wellsfileonly")

        if is_thumbnail:
            thumbnailsize = blockprocessing.getThumbnailSize(env['exp_json'])
            block['id_str'] = 'thumbnail'
            base_args.append("--cfiedr-regions-size=50x50")
            base_args.append("--block-size=%sx%s" % (thumbnailsize[0], thumbnailsize[1]))
            base_args.append("--beadfind-thumbnail 1")
            base_args.append("--bkg-debug-param 1")

        elif is_wholechip:
            block['id_str'] = 'wholechip'

        base_args.append("--output-dir=%s" % env['SIGPROC_RESULTS'])

        block['jobcmd'] = base_args
        print base_args
        print block
        blocks.append(block)

    else:
        explogblocks = blockprocessing.getBlocksFromExpLog(env['exp_json'])
        for block in explogblocks:
            #ignore thumbnail
            if block['id_str'] =='thumbnail':
                continue

            base_args = env['analysisArgs'].strip().split(' ')
            # raw data dir is last element in analysisArgs and needs the block subdirectory appended
            base_args[-1] = os.path.join(base_args[-1], block['datasubdir'])
            rawDirectory = base_args[-1]
            base_args.append("--libraryKey=%s" % env["libraryKey"])
            base_args.append("--no-subdir --wellsfileonly")
            base_args.append("--output-dir=%s" % env['SIGPROC_RESULTS'])

            block['jobcmd'] = base_args
            print "expblock: " + str(block)
            if (block['autoanalyze'] and block['analyzeearly']) or os.path.isdir(rawDirectory):
                print "base_args: " + str(base_args)
                print "block: " + str(block)
                blocks.append(block)

    return blocks

def initBlockReport(blockObj):

    file_in = open("ion_params_00.json", 'r')
    TMP_PARAMS = json.loads(file_in.read())
    file_in.close()

    if is_wholechip:
        resultDir = "./"
    elif is_thumbnail:
        resultDir = "./"
    else:
        resultDir = './%s%s' % ('block_', blockObj['id_str'])

        if not os.path.exists(resultDir):
            os.mkdir(resultDir)

        # update path to data
        TMP_PARAMS["pathToData"] = os.path.join(TMP_PARAMS["pathToData"], blockObj['id_str'])

        cwd = os.getcwd()
        r = subprocess.call(["ln", "-s", os.path.join(cwd,"Default_Report.php"), os.path.join(resultDir,"Default_Report.php")])
        if r:
            printtime("couldn't create symbolic link")
        r = subprocess.call(["ln", "-s", os.path.join(cwd,"parsefiles.php"), os.path.join(resultDir,"parsefiles.php")])
        if r:
            printtime("couldn't create symbolic link")
        r = subprocess.call(["ln", "-s", os.path.join(cwd,"DefaultTFs.conf"), os.path.join(resultDir,"DefaultTFs.conf")])
        if r:
            printtime("couldn't create symbolic link")

    # update analysisArgs section
    TMP_PARAMS["analysisArgs"] = ' '.join(blockObj['jobcmd'])

    if os.path.isfile("/opt/ion/.ion-internal-server"):
        TMP_PARAMS['sam_parsed'] = True
        printtime("HEY! Found .ion-internal-server")
    else:
        TMP_PARAMS['sam_parsed'] = False
        printtime("No file named .ion-internal-server")

    #write block specific ion_params_00.json
    file_out = open("%s/ion_params_00.json" % resultDir, 'w')
    json.dump(TMP_PARAMS, file_out)
    file_out.close()

    return resultDir

def spawn_cluster_job(rpath, scriptname, args, holds=None):
    out_path = "%s/drmaa_stdout_block.html" % rpath
    err_path = "%s/drmaa_stderr_block.txt" % rpath
    logout = open(os.path.join(out_path), "w")
    logout.write("<html><pre> \n")
    logout.close()
    cwd = os.getcwd()

    #SGE
    sge_queue = 'all.q'
    if is_thumbnail:
        sge_queue = 'thumbnail.q'
    jt_nativeSpecification = "-pe ion_pe 1 -q " + sge_queue

    # TODO experiment
    if is_blockprocessing and ("X1" in rpath): # process some blocks on instrument
        if env['pgmName'] == "Mustang": # != ""
            sge_queue = "proton_" + env['pgmName'].lower() + ".q"
            jt_nativeSpecification = "-q " + sge_queue

    printtime("Use "+ sge_queue)

    #TORQUE
    #jt_nativeSpecification = ""

    jt_remoteCommand = "python"
    jt_workingDirectory = os.path.join(cwd, rpath)
    jt_outputPath = ":" + os.path.join(cwd, out_path)
    jt_errorPath = ":" + os.path.join(cwd, err_path)
    jt_args = [os.path.join('/usr/bin',scriptname),args]
    jt_joinFiles = False

    if holds != None and len(holds) > 0:
        jt_nativeSpecification += " -hold_jid "
        for holdjobid in holds:
            jt_nativeSpecification += "%s," % holdjobid

    # TODO remove debug output
    print jt_nativeSpecification
    print jt_remoteCommand
    print jt_workingDirectory
    print jt_outputPath
    print jt_errorPath
    print jt_args

    try:
        jobid = jobserver.submitjob(
            jt_nativeSpecification,
            jt_remoteCommand,
            jt_workingDirectory,
            jt_outputPath,
            jt_errorPath,
            jt_args,
            jt_joinFiles)
    except:
        traceback.print_exc()
        jobid = -1

    return jobid


if __name__=="__main__":

    blockprocessing.printheader()

    env = blockprocessing.getparameter()

    try:
        primary_key = open("primary.key").readline()
        primary_key = primary_key.split(" = ")
        env['primary_key'] = primary_key[1]
        printtime(env['primary_key'])
    except:
        printtime("Error, unable to get the primary key")
    
    # assemble the URL path for this analysis result, relative to the webroot directory: (/var/www/)
    # <output dir>/<Location name>/<analysis dir>
    url_root = os.path.join(env['url_path'],os.path.basename(os.getcwd()))
    
    printtime("DEBUG url_root string %s" % url_root)
    printtime("DEBUG net_location string %s" % env['net_location'])
    printtime("DEBUG master_node string %s" % env['master_node'])
    
    report_config = ConfigParser.RawConfigParser()
    report_config.optionxform = str # don't convert to lowercase
    report_config.add_section('global')

    is_thumbnail = False
    is_wholechip = False
    is_blockprocessing = False

    if env['rawdatastyle'] == 'single':
        is_wholechip = True
        report_config.set('global', 'Type', '31x')
    else:
        if "thumbnail" in env['pathToRaw']:
           is_thumbnail = True
           report_config.set('global', 'Type', 'Thumbnail')
        else:
           is_blockprocessing = True
           report_config.set('global', 'Type', 'Composite')

    with open('ReportConfiguration.txt', 'wb') as reportconfigfile:
        report_config.write(reportconfigfile)

    #drops a zip file of the pgm log files
    get_pgm_log_files(env['pathToRaw'])

    #copy explog.txt into report directory
    try:
        explogfilepath = os.path.join(env['pathToRaw'],'explog.txt')
        if os.path.exists(explogfilepath):
            shutil.copy(explogfilepath, ".")
        else:
            printtime("ERROR: %s doesn't exist" % explogfilepath)
    except:
        printtime(traceback.format_exc())

    # Generate a system information file for diagnostics purposes.
    try:
        com="/usr/bin/ion_sysinfo 2>&1 >> ./sysinfo.txt"
        os.system(com)
    except:
        print traceback.format_exc()

    #############################################################
    # Code to start full chip analysis                           #
    #############################################################
    os.umask(0002)

    # define entry point
    print "blockArgs '"+str(env['blockArgs'])+"'"
    print "previousReport: '"+str(env['previousReport'])+"'"

    fromwellsfiles = []
    fromwellsfiles.append("iontrace_Library.png")
    fromwellsfiles.append("1.wells")
    fromwellsfiles.append("bfmask.stats")
    fromwellsfiles.append("analysis.bfmask.bin")
    fromwellsfiles.append("Bead_density_raw.png")
    fromwellsfiles.append("Bead_density_contour.png")
    fromwellsfiles.append("processParameters.txt")
    fromwellsfiles.append("avgNukeTrace_%s.txt" % env['tfKey'])
    fromwellsfiles.append("avgNukeTrace_%s.txt" % env['libraryKey'])
    fromsfffiles = []
    fromsfffiles.append("rawlib.sff")
    fromsfffiles.append("rawlib.fastq")
    fromsfffiles.append("quality.summary")
    fromsfffiles.append("bfmask.bin")
    fromsfffiles.append("readLen.txt")
    fromsfffiles.append("raw_peak_signal")
    fromsfffiles.append("beadSummary.filtered.txt")

    if env['blockArgs'] == "fromRaw":
        runFromRaw = True
        runFromWells = True
        runFromSFF = True
    elif env['blockArgs'] == "fromWells":
        runFromRaw = False
        runFromWells = True
        runFromSFF = True
        #r = subprocess.call(["ln", "-s", os.path.join(env['previousReport'], env['SIGPROC_RESULTS'])])
        for afilename in fromwellsfiles:
            try:
                shutil.copy( os.path.join(env['previousReport'], env['SIGPROC_RESULTS'], afilename) ,  os.path.join(env['SIGPROC_RESULTS'], afilename) )
            except:
                traceback.print_exc()
    elif env['blockArgs'] == "fromSFF":
        runFromRaw = False
        runFromWells = False
        runFromSFF = True
        #r = subprocess.call(["ln", "-s", os.path.join(env['previousReport'], env['SIGPROC_RESULTS'])])
        for afilename in fromwellsfiles:
            try:
                shutil.copy( os.path.join(env['previousReport'], env['SIGPROC_RESULTS'], afilename) ,  os.path.join(env['SIGPROC_RESULTS'], afilename) )
            except:
                traceback.print_exc()
        #r = subprocess.call(["ln", "-s", os.path.join(env['previousReport'], env['BASECALLER_RESULTS'])])
        for afilename in fromsfffiles:
            try:
                shutil.copy( os.path.join(env['previousReport'], env['BASECALLER_RESULTS'], afilename) ,  os.path.join(env['BASECALLER_RESULTS'], afilename) )
            except:
                traceback.print_exc()
        # barcode processing: copy barcodeSplit SFF/FASTQ files
        bcpath_old = os.path.join(env['previousReport'],env['DIR_BC_FILES'])        
        if os.path.exists(bcpath_old):
            printtime("Found barcode folder")
            try:
                if not os.path.exists(env['DIR_BC_FILES']):
                    os.mkdir(env['DIR_BC_FILES'])
                alist = os.listdir(bcpath_old)
                extlist = ['sff','fastq']
                for ext in extlist:
                    filelist = fnmatch.filter(alist, "*." + ext)
                    for bfile in filelist:
                        shutil.copy( os.path.join(bcpath_old, bfile), os.path.join(env['DIR_BC_FILES'], bfile) )
            except:
                traceback.print_exc()
    else:
        printtime("WARNING: start point not defined, create new report from raw data")
        runFromRaw = True
        runFromWells = True
        runFromSFF = True

    blockprocessing.initreports(env['SIGPROC_RESULTS'], env['BASECALLER_RESULTS'], env['ALIGNMENT_RESULTS'])
    logout = open("ReportLog.html", "w")
    logout.close()

    sys.stdout.flush()
    sys.stderr.flush()

    try:
        jobserver = xmlrpclib.ServerProxy("http://%s:%d" % (JOBSERVER_HOST, JOBSERVER_PORT), verbose=False, allow_none=True)
    except (socket.error, xmlrpclib.Fault):
        traceback.print_exc()

    basefolder = 'plugin_out'
    if not os.path.isdir(basefolder):
        os.umask(0000)   #grant write permission to plugin user
        os.mkdir(basefolder)
        os.umask(0002)

    #-------------------------------------------------------------
    # Update Report Status to 'Started'
    #-------------------------------------------------------------
    try:
        jobserver.updatestatus(os.path.join(os.getcwd(),'primary.key'),'Started',True)
    except:
        traceback.print_exc()
        
    #-------------------------------------------------------------
    # Gridded data processing
    #-------------------------------------------------------------
    blockprocessing.write_version()

    printtime("RUNNING FULL CHIP MULTI-BLOCK ANALYSIS")
    # List of block objects to analyze
    blocks = GetBlocksToAnalyze(env)
    dirs = ['block_%s' % block['id_str'] for block in blocks]
#    dirs = ['%s/block_%s' % (SIGPROC_RESULTS, block['id_str']) for block in blocks]

    #####################################################
    # Create block reports                              #
    #####################################################

    #TODO
    doblocks = 1
    if doblocks:

        sigproc_job_list = []
        basecaller_job_list = []
        alignment_job_list = []

        result_dirs = []
        for idx, block in enumerate(blocks):
            result_dirs.append(initBlockReport(block))

        if runFromRaw:
            printtime("PROCESSING FROM RAW")

            for idx, block in enumerate(blocks):
                block['jobid'] = spawn_cluster_job(result_dirs[idx],'BlockTLScript.py', '--do-sigproc')
                sigproc_job_list.append( str(block['jobid']) )
                printtime("Submitted block (%s) analysis job with job ID (%s)" % (block['id_str'], str(block['jobid'])))
            if not is_thumbnail and not is_wholechip:
                jid = spawn_cluster_job('.','MergeTLScript.py','--do-sigproc',sigproc_job_list)

        if runFromWells:
            printtime("PROCESSING FROM WELLS")

            for idx, block in enumerate(blocks):
                if env['blockArgs'] == "fromWells":
                    wait_list = []
                else:
                    wait_list = [ sigproc_job_list[idx] ]
                block['jobid'] = spawn_cluster_job(result_dirs[idx],'BlockTLScript.py','--do-basecalling',wait_list)
                basecaller_job_list.append( str(block['jobid']) )
                printtime("Submitted block (%s) analysis job with job ID (%s)" % (block['id_str'], str(block['jobid'])))
            if not is_thumbnail and not is_wholechip:
                jid = spawn_cluster_job('.','MergeTLScript.py','--do-basecalling',basecaller_job_list)

        if runFromSFF:
            printtime("PROCESSING FROM SFF")

            for idx, block in enumerate(blocks):
                if env['blockArgs'] == "fromSFF":
                    wait_list = []
                else:
                    wait_list = [ basecaller_job_list[idx] ]
                block['jobid'] = spawn_cluster_job(result_dirs[idx],'BlockTLScript.py','--do-alignment',wait_list)
                alignment_job_list.append( str(block['jobid']) )
                printtime("Submitted block (%s) analysis job with job ID (%s)" % (block['id_str'], str(block['jobid'])))
            if not is_thumbnail and not is_wholechip:
                jid = spawn_cluster_job('.','MergeTLScript.py','--do-alignment',alignment_job_list)
            else:
                jid = spawn_cluster_job('.','MergeTLScript.py','--do-zipping',alignment_job_list)

        # write job id's to file
        f = open('job_list.txt','w')
        for jobid in sigproc_job_list:
            f.write(jobid+'\n')
        for jobid in basecaller_job_list:
            f.write(jobid+'\n')
        for jobid in alignment_job_list:
            f.write(jobid+'\n')
        f.close()

        # Watch status of jobs.  As they finish remove the job from the list.

        pl_started = False
        while len(alignment_job_list) > 0:
            for job in alignment_job_list:
                block = [block for block in blocks if block['jobid'] == job][0]
                #check status of jobid
                try:
                    block['status'] = jobserver.jobstatus(block['jobid'])
                except:
                    traceback.print_exc()
                    continue

                if block['status']=='done' or block['status']=='failed' or block['status']=="DRMAA BUG":
                    printtime("Job %s has ended with status %s" % (str(block['jobid']),block['status']))
                    alignment_job_list.remove(block['jobid'])
#                else:
#                    printtime("Job %s has status %s" % (str(block['jobid']),block['status']))

                # Hack
                if is_thumbnail and not pl_started:
                    if os.path.exists('progress.txt'):
                        f = open('progress.txt')
                        text = f.read()
                        f.close()
                    else:
                        printtime("progress.txt not found (waiting for Analysis)")
                        continue
                    try:
                        matches = re.findall(r"wellfinding = green", text)
                        if len(matches) != 0:
                            pl_started = True
                            plugin_set = set()
                            plugin_set.add('rawPlots')
                            plugin_set.add('separator')
                            plugin_set.add('chipNoise')
                            blockprocessing.run_selective_plugins(plugin_set, env, basefolder, url_root)
                    except:
                        traceback.print_exc()
                        continue


            printtime("waiting for %d blocks to be finished" % len(alignment_job_list))
            time.sleep (5)

        if not is_thumbnail and not is_wholechip:
            while True:
                #check status of jobid
                try:
                    merge_status = jobserver.jobstatus(jid)
                except:
                    traceback.print_exc()
                    continue

                if merge_status=='done' or merge_status=='failed' or merge_status=="DRMAA BUG":
                    printtime("Job %s has ended with status %s" % (jid,merge_status))
                    break

            printtime("waiting for merge block")
            time.sleep (5)


    printtime("All jobs processed")

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

    ########################################################
    #ParseFiles and Upload Metrics                         #
    ########################################################
    printtime("Attempting to Upload to Database")

    alignmentSummaryPath = os.path.join(env['ALIGNMENT_RESULTS'],'alignment.summary')

    if is_thumbnail:
        # uploadMetrics will set align_sample=2 for thumbnails
        try:
            f = open(alignmentSummaryPath)
            text = f.read()
            f.close()
            matches = re.findall(r"Thumbnail = 1", text)
            if len(matches) == 0:
                printtime("Adding Thumbnail = 1 to file %s" % alignmentSummaryPath)
                f = open (alignmentSummaryPath, "a")
                f.write("Thumbnail = 1\n")
                f.close()
        except:
            printtime("Cannot open file %s" % alignmentSummaryPath)


    #attempt to upload the metrics to the Django database
    try:
        mycwd = os.getcwd()

        BaseCallerJsonPath = os.path.join(env['BASECALLER_RESULTS'],'BaseCaller.json')
        tfmapperstats_outputfile = os.path.join(env['BASECALLER_RESULTS'],"TFStats.json")
        merged_bead_mask_path = os.path.join(env['SIGPROC_RESULTS'], 'MaskBead.mask')
        QualityPath = os.path.join(env['BASECALLER_RESULTS'],'quality.summary')
        peakOut = 'raw_peak_signal'
        beadPath = os.path.join(env['SIGPROC_RESULTS'],'bfmask.stats')
        procPath = os.path.join(env['SIGPROC_RESULTS'],'processParameters.txt')
        filterPath = None
        reportLink = True

        if is_thumbnail or is_wholechip:
            if os.path.exists('blockstatus.txt'):
                f = open('blockstatus.txt')
                text = f.readlines()
                f.close()
                STATUS = "Completed"
                for line in text:
                    [component, status] = line.split('=')
                    print component, status
                    if int(status) != 0:
                        if component == 'Analysis':
                            if int(status) == 2:
                                STATUS = 'Checksum Error'
                            elif int(status) == 3:
                                STATUS = 'PGM Operation Error'#'No Live Beads'
                            else:
                                STATUS = "Error in %s" % component
                        elif component == 'alignmentQC.pl':
                            # TS-2992 alignment failure will still mark the analysis report complete
                            # but in the Default Report page, the alignment section will display the failure.
                            continue
                        else:
                            STATUS = "Error in %s" % component
                            break
            else:
                STATUS = "Error"
        else:
            #TODO Report Proton Full Chip Reports always as 'Completed'
            STATUS = "Completed"

        ret_message = jobserver.uploadmetrics(
            os.path.join(mycwd,tfmapperstats_outputfile),
            os.path.join(mycwd,procPath),
            os.path.join(mycwd,beadPath),
            filterPath,
            os.path.join(mycwd,alignmentSummaryPath),
            os.path.join(mycwd,peakOut),
            os.path.join(mycwd,QualityPath),
            os.path.join(mycwd,BaseCallerJsonPath),
            os.path.join(mycwd,'primary.key'),
            os.path.join(mycwd,'uploadStatus'),
            STATUS,
            reportLink)
        # this will replace the five progress squares with a re-analysis button
        print "jobserver.uploadmetrics retured: "+str(ret_message)
    except:
        traceback.print_exc()

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
    # Write checksum_status.txt to raw data directory      #
    ########################################################
    if is_wholechip:
        try:
            if os.path.isfile("analysis_return_code.txt"):
                shutil.copyfile("analysis_return_code.txt",os.path.join(env['pathToRaw'],"checksum_status.txt"))
        except:
            traceback.print_exc()

    if is_thumbnail or is_wholechip:
        blockprocessing.runplugins(env, basefolder, url_root)
    else:
        plugin_set = set()
        plugin_set.add('torrentscout')
        plugin_set.add('rawPlots')
        blockprocessing.run_selective_plugins(plugin_set, env, basefolder, url_root)

    if env['isReverseRun'] and env['pe_forward'] != "None":
        try:
            crawler = xmlrpclib.ServerProxy("http://%s:%d" % (CRAWLER_HOST, CRAWLER_PORT), verbose=False, allow_none=True)
        except (socket.error, xmlrpclib.Fault):
            traceback.print_exc()

        printtime("crawler hostname: "+crawler.hostname())
        printtime("PE Report status: "+crawler.startPE(env['expName'],env['pe_forward'],os.getcwd()))


    getExpLogMsgs(env)
    get_pgm_log_files(env['pathToRaw'])
    printtime("Run Complete")
    sys.exit(0)
