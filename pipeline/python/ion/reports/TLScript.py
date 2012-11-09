#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Loads Fullchip analysis pipeline
Uses output from there to generate charts and graphs and dumps to current directory
Adds key metrics to database
"""

__version__ = filter(str.isdigit, "$Revision: 43376 $")

# First we need to bootstrap the analysis to start correctly from the command
# line or as the child process of a web server. We import a few things we
# need to make that happen before we import the modules needed for the actual
# analysis.
import os
import tempfile

# matplotlib/numpy compatibility
os.environ['HOME'] = tempfile.mkdtemp()
#from matplotlib import use
#use("Agg")

# import /etc/torrentserver/cluster_settings.py, provides JOBSERVER_HOST, JOBSERVER_PORT
import sys
sys.path.append('/etc')
from torrentserver.cluster_settings import *

import hashlib
import ConfigParser
import shutil
import socket
import time
import traceback
import json
import xmlrpclib
from ion.utils import blockprocessing
from ion.utils import explogparser
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
    parseBeadfind, parseProcessParams, \
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
        make_zip ('pgm_logs.zip',os.path.join(rawdatadir,afile),arcname=afile)

    return


def GetBlocksToAnalyze(env):
    blocks = []

    if is_thumbnail or is_wholechip:

        block = {'id_str':'',
                'datasubdir':'',
                'jobid':None,
                'status':None}

        if is_thumbnail:
            block['id_str'] = 'thumbnail'

        elif is_wholechip:
            block['id_str'] = 'wholechip'

        print block
        blocks.append(block)

    else:
        explogblocks = explogparser.getBlocksFromExpLogJson(env['exp_json'],excludeThumbnail=True)
        for block in explogblocks:
            rawDirectory = block['datasubdir']
            print "expblock: " + str(block)
            if (block['autoanalyze'] and block['analyzeearly']) or os.path.isdir(os.path.join(env['pathToRaw'],rawDirectory)) or env['blockArgs'] == "fromWells":
                print "block: " + str(block)
                blocks.append(block)

    return blocks


def hash_matches(full_filename):
    ret = False
    try:
        with open(full_filename,'rb') as f:
            binary_content = f.read()
            md5sum = hashlib.md5(binary_content).hexdigest()

        head, tail = os.path.split(full_filename)
        with open(os.path.join(head,'MD5SUMS'),'r') as f:
            lines = f.readlines()
        expected_md5sums = {}
        for line in lines:
            ahash,filename = line.split()
            expected_md5sums[filename]=ahash
        ret = (expected_md5sums[tail]==md5sum)
    except:
        traceback.print_exc()
        pass
    return ret


def spawn_cluster_job(rpath, scriptname, args, holds=None):
    out_path = "%s/drmaa_stdout_block.txt" % rpath
    err_path = "%s/drmaa_stderr_block.txt" % rpath
    cwd = os.getcwd()

    #SGE
    sge_queue = 'all.q'
    if is_thumbnail:
        sge_queue = 'thumbnail.q'
    jt_nativeSpecification = "-pe ion_pe 1 -q " + sge_queue

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

    env,warn = explogparser.getparameter()
    print warn    

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

    # Generate a system information file for diagnostics purposes.
    try:
        com="/usr/bin/ion_sysinfo 2>&1 >> ./sysinfo.txt"
        os.system(com)
    except:
        print traceback.format_exc()

    #############################################################
    # Code to start full chip analysis                          #
    #############################################################
    os.umask(0002)

    '''
    fromwellsfiles = []
    fromwellsfiles.append("iontrace_Library.png")
    fromwellsfiles.append("1.wells")
    fromwellsfiles.append("bfmask.stats")
    fromwellsfiles.append("analysis.bfmask.stats")
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
    fromsfffiles.append("BaseCaller.json")
    '''
    print "oninstranalysis:(%s)" % env['oninstranalysis']
    if env['oninstranalysis'] and is_blockprocessing:
        env['blockArgs'] = "fromRaw" # because heatmaps have to be generated on TS

    # define entry point
    print "blockArgs '"+str(env['blockArgs'])+"'"
    print "previousReport: '"+str(env['previousReport'])+"'"

    explogfilepath = os.path.join(env['pathToRaw'],'explog.txt')
    explogfinalfilepath = os.path.join(env['pathToRaw'],'explog_final.txt')

    if env['blockArgs'] == "fromRaw":
        runFromRaw = True
        runFromWells = True
        runFromSFF = True

        if env['oninstranalysis'] and is_blockprocessing:
            os.symlink(os.path.join(env['pathToRaw'], 'onboard_results', env['SIGPROC_RESULTS']), env['SIGPROC_RESULTS'])
        else:
            if not os.path.isdir(env['SIGPROC_RESULTS']):
                try:
                    os.mkdir(env['SIGPROC_RESULTS'])
                except:
                    traceback.print_exc()

        if not os.path.isdir(env['BASECALLER_RESULTS']):
            try:
                os.mkdir(env['BASECALLER_RESULTS'])
            except:
                traceback.print_exc()

    elif env['blockArgs'] == "fromWells":
        runFromRaw = False
        runFromWells = True
        runFromSFF = True
        explogfilepath = os.path.join(env['previousReport'],'explog.txt')
        explogfinalfilepath = os.path.join(env['previousReport'],'explog_final.txt')

        sigproc_target = os.path.join(env['previousReport'], env['SIGPROC_RESULTS'])

        # fix, try to prepare old reports
        if not os.path.exists(sigproc_target):
            os.symlink(env['previousReport'], sigproc_target)

        os.symlink(sigproc_target,env['SIGPROC_RESULTS'])

        if not os.path.isdir(env['BASECALLER_RESULTS']):
            try:
                os.mkdir(env['BASECALLER_RESULTS'])
            except:
                traceback.print_exc()
    elif env['blockArgs'] == "fromSFF":
        runFromRaw = False
        runFromWells = False
        runFromSFF = True
        explogfilepath = os.path.join(env['previousReport'],'explog.txt')
        explogfinalfilepath = os.path.join(env['previousReport'],'explog_final.txt')
        previous_raw_peak_signal = os.path.join(env['previousReport'], 'raw_peak_signal')

        sigproc_target = os.path.join(env['previousReport'], env['SIGPROC_RESULTS'])
        basecaller_target = os.path.join(env['previousReport'], env['BASECALLER_RESULTS'])

        # fix, try to prepare old reports
        if not os.path.exists(sigproc_target):
            os.symlink(env['previousReport'], sigproc_target)
        if not os.path.exists(basecaller_target):
            os.symlink(env['previousReport'], basecaller_target)

        os.symlink(sigproc_target,env['SIGPROC_RESULTS'])
        os.symlink(basecaller_target, env['BASECALLER_RESULTS'])
        os.symlink(previous_raw_peak_signal, 'raw_peak_signal')
    else:
        printtime("WARNING: start point not defined, create new report from raw data")
        runFromRaw = True
        runFromWells = True
        runFromSFF = True

        if not os.path.isdir(env['SIGPROC_RESULTS']):
            try:
                os.mkdir(env['SIGPROC_RESULTS'])
            except:
                traceback.print_exc()
        if not os.path.isdir(env['BASECALLER_RESULTS']):
            try:
                os.mkdir(env['BASECALLER_RESULTS'])
            except:
                traceback.print_exc()

    #copy explog.txt into report directory
    try:
        if os.path.exists(explogfilepath):
            shutil.copy(explogfilepath, ".")
        else:
            printtime("ERROR: %s doesn't exist" % explogfilepath)
    except:
        printtime(traceback.format_exc())

    pluginbasefolder = 'plugin_out'
    blockprocessing.initTLReport(pluginbasefolder)
    
    env['report_root_dir'] = os.getcwd()
    
    sys.stdout.flush()
    sys.stderr.flush()

    #-------------------------------------------------------------
    # Update Report Status to 'Started'
    #-------------------------------------------------------------
    try:
        jobserver = xmlrpclib.ServerProxy("http://%s:%d" % (JOBSERVER_HOST, JOBSERVER_PORT), verbose=False, allow_none=True)
        debugging_cwd = os.getcwd()
    except:
        traceback.print_exc()
    
    def set_result_status(status):
        try:
            primary_key_file = os.path.join(os.getcwd(),'primary.key')
            jobserver.updatestatus(primary_key_file, status, True)
            printtime("TLStatus %s\tpid %d\tpk file %s started in %s" % 
                (status, os.getpid(), primary_key_file, debugging_cwd))
        except:
            traceback.print_exc()

    set_result_status('Started')
        
    
    #-------------------------------------------------------------
    # Initialize plugins
    #-------------------------------------------------------------
    plugins = blockprocessing.get_plugins_to_run(env['plugins'], env['report_type']) 
    blocklevel_plugins = [] 
    if is_blockprocessing:
      blocklevel_plugins = [p for p in plugins if ('runlevel' in p.keys() and 'block' in p['runlevel'])]    
        
    #-------------------------------------------------------------
    # Gridded data processing
    #-------------------------------------------------------------
    blockprocessing.write_version()

    printtime("RUNNING FULL CHIP MULTI-BLOCK ANALYSIS")
    # List of block objects to analyze
    blocks = GetBlocksToAnalyze(env)
    dirs = ['block_%s' % block['id_str'] for block in blocks]
    # dirs = ['%s/block_%s' % (SIGPROC_RESULTS, block['id_str']) for block in blocks]

    #####################################################
    # Create block reports                              #
    #####################################################

    #TODO
    doblocks = 1
    if doblocks:

        sigproc_job_dict = {}
        basecaller_job_dict = {}
        alignment_job_dict = {}
        merge_job_dict = {}

        result_dirs = {}
        for block in blocks:
            result_dirs[block['id_str']] = blockprocessing.initBlockReport(block, env['SIGPROC_RESULTS'], env['BASECALLER_RESULTS'], env['ALIGNMENT_RESULTS'], env['oninstranalysis'])

        # create a list of blocks
        blocks_to_process = []
        blocks_to_process.extend(blocks)
        timeout = 3*60*60
        if env['oninstranalysis'] and is_blockprocessing:
            timeout = 36*60*60
        
        env['block_dirs'] = [os.path.join(env['report_root_dir'],result_dirs[block['id_str']]) for block in blocks_to_process]

        while len(blocks_to_process) > 0 and timeout > 0:
            for block in blocks_to_process:
                printtime('waiting for %s block(s) to start' % str(len(blocks_to_process)))
                sys.stdout.flush()
                sys.stderr.flush()

                if runFromRaw:
                    if is_thumbnail or is_wholechip:
                        data_file = os.path.join(env['pathToRaw'],'acq_0000.dat')
                    else:
                        data_file = os.path.join(env['pathToRaw'],block['id_str'],'acq_0000.dat')

                    if env['oninstranalysis'] and is_blockprocessing:
                        data_file = os.path.join(env['SIGPROC_RESULTS'],'block_'+block['id_str'])

                    if os.path.exists(data_file):

                        if env['oninstranalysis'] and is_blockprocessing:
                            # wait until transfer seems to be finished
                            mod_time = os.stat(data_file).st_mtime
                            cur_time = time.time()
                            if cur_time - mod_time < 120:
                                printtime("mtime %s" % mod_time)
                                printtime("ctime %s" % cur_time)
                                timeout -= 10
                                time.sleep (10)
                                continue

                            wells_file = os.path.join(data_file, '1.wells')
                            if not hash_matches(wells_file):
                                printtime("WARNING: %s might be corrupt" % wells_file)
                                #blocks_to_process.remove(block)
                                #continue
                            
                        block['jobid'] = spawn_cluster_job(result_dirs[block['id_str']],'BlockTLScript.py', '--do-sigproc')
                        sigproc_job_dict[block['id_str']] = str(block['jobid'])
                        printtime("Submitted block (%s) analysis job with job ID (%s)" % (block['id_str'], str(block['jobid'])))
                    else:
                        printtime("missing %s" % data_file)
                        timeout -= 10
                        time.sleep (10)
                        continue


                if runFromWells:
                    try:
                        if env['blockArgs'] == "fromWells":
                            wait_list = []
                        else:
                            wait_list = [ sigproc_job_dict[block['id_str']] ]
                        block['jobid'] = spawn_cluster_job(result_dirs[block['id_str']],'BlockTLScript.py','--do-basecalling',wait_list)
                        basecaller_job_dict[block['id_str']] = str(block['jobid'])
                        printtime("Submitted block (%s) analysis job with job ID (%s)" % (block['id_str'], str(block['jobid'])))
                    except:
                        printtime("submitting basecaller job for block (%s) failed" % block['id_str'])


                if runFromSFF:
                    try:
                        if env['blockArgs'] == "fromSFF":
                            wait_list = []
                        else:
                            wait_list = [ basecaller_job_dict[block['id_str']] ]
                        block['jobid'] = spawn_cluster_job(result_dirs[block['id_str']],'BlockTLScript.py','--do-alignment',wait_list)
                        alignment_job_dict[block['id_str']] = str(block['jobid'])
                        printtime("Submitted block (%s) alignment job with job ID (%s)" % (block['id_str'], str(block['jobid'])))
                    except:
                        printtime("submitting alignment job for block (%s) failed" % block['id_str'])

                blocks_to_process.remove(block)


        if not is_thumbnail and not is_wholechip:
            if runFromRaw:
                merge_job_dict['sigproc'] = spawn_cluster_job('.','MergeTLScript.py','--do-sigproc',sigproc_job_dict.values())
            if runFromWells:
                merge_job_dict['basecaller'] = spawn_cluster_job('.','MergeTLScript.py','--do-basecalling',basecaller_job_dict.values())
            if runFromSFF:
                merge_job_dict['alignment'] = spawn_cluster_job('.','MergeTLScript.py','--do-alignment',alignment_job_dict.values()+[merge_job_dict.get('basecaller','')])
        else:
            merge_job_dict['merge/zipping'] = spawn_cluster_job('.','MergeTLScript.py','--do-zipping',alignment_job_dict.values())


        # write job id's to file
        f = open('job_list.txt','w')
        job_list = {}
        job_list['merge']= merge_job_dict
        for block, jobid in sigproc_job_dict.items():
            f.write(jobid+'\n')
            job_list[block] = {'sigproc': jobid}
        for block, jobid in basecaller_job_dict.items():
            f.write(jobid+'\n')
            if job_list.has_key(block):            
                job_list[block]['basecaller'] = jobid
            else:
                job_list[block] = {'basecaller': jobid}    
        for block, jobid in alignment_job_dict.items():
            f.write(jobid+'\n')
            if job_list.has_key(block):            
                job_list[block]['alignment'] = jobid
            else:
                job_list[block] = {'alignment': jobid}    
        for jobname, jobid in merge_job_dict.items():
            f.write(jobid+'\n')            
        f.close()
        # write more descriptive json list of jobs
        with open('job_list.json','w') as f:
            f.write(json.dumps(job_list,indent=2))

        # multilevel plugins preprocessing level
        blockprocessing.runplugins(plugins, env, pluginbasefolder, url_root, 'pre')

        # Watch status of jobs.  As they finish remove the job from the list.

        pl_started = False
        alignment_job_list = alignment_job_dict.values()
        while len(alignment_job_list) > 0:
            for job in alignment_job_list:
                block = [block for block in blocks if block['jobid'] == job][0]
                #check status of jobid
                try:
                    block['status'] = jobserver.jobstatus(block['jobid'])
                except:
                    traceback.print_exc()
                    continue

                if (len(blocklevel_plugins) > 0) and (block['status']=='done'):
                    block_pluginbasefolder = os.path.join(result_dirs[block['id_str']],pluginbasefolder)
                    env['blockId'] = block['id_str']
                    if not os.path.isdir(block_pluginbasefolder):
                        oldmask = os.umask(0000)   #grant write permission to plugin user
                        os.mkdir(block_pluginbasefolder)
                        os.umask(oldmask)  
                    plugins = blockprocessing.runplugins(plugins, env, block_pluginbasefolder, url_root, 'block')

                if block['status']=='done' or block['status']=='failed' or block['status']=="DRMAA BUG":
                    printtime("Job %s has ended with status %s" % (str(block['jobid']),block['status']))
                    alignment_job_list.remove(block['jobid'])
#                else:
#                    printtime("Job %s has status %s" % (str(block['jobid']),block['status']))
                
            if os.path.exists(os.path.join(env['SIGPROC_RESULTS'],'separator.mask.bin')) and not pl_started:
                plugins = blockprocessing.runplugins(plugins, env, pluginbasefolder, url_root, 'separator')
                pl_started = True


            printtime("waiting for %d blocks to be finished" % len(alignment_job_list))
            time.sleep (10)

        merge_job_list = merge_job_dict.keys()
        while len(merge_job_list) > 0:
            for key in merge_job_list:

                #check status of jobid
                try:
                    jid = merge_job_dict[key]
                    merge_status = jobserver.jobstatus(jid)
                except:
                    traceback.print_exc()
                    continue

                if merge_status=='done' or merge_status=='failed' or merge_status=="DRMAA BUG":
                    printtime("Job %s, %s has ended with status %s" % (key,jid,merge_status))
                    merge_job_list.remove(key)
                    break

            printtime("waiting for %d merge jobs to be finished" % len(merge_job_list))
            time.sleep (10)


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
        peakOut = os.path.join('.','raw_peak_signal')
        beadPath = os.path.join(env['SIGPROC_RESULTS'],'analysis.bfmask.stats')
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
                                STATUS = 'No Live Beads'
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
            "", #pe.json
            os.path.join(mycwd,'primary.key'),
            os.path.join(mycwd,'uploadStatus'),
            STATUS,
            reportLink)
        # this will replace the five progress squares with a re-analysis button
        print "jobserver.uploadmetrics returned: "+str(ret_message)
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


    #copy explog_final.txt into report directory
    try:
        if os.path.exists(explogfinalfilepath):
            shutil.copy(explogfinalfilepath, ".")
        else:
            printtime("ERROR: %s doesn't exist" % explogfinalfilepath)
    except:
        printtime(traceback.format_exc())



    ########################################################
    # Write checksum_status.txt to raw data directory      #
    ########################################################
    if is_wholechip:
        raw_return_code_file = os.path.join(env['SIGPROC_RESULTS'],"analysis_return_code.txt")
    elif is_blockprocessing:
        raw_return_code_file = os.path.join(env['BASECALLER_RESULTS'],"composite_return_code.txt")

    try:
        if (is_wholechip or is_blockprocessing) and os.path.isfile(raw_return_code_file):
            shutil.copyfile(raw_return_code_file,os.path.join(env['pathToRaw'],"checksum_status.txt"))
    except:
        traceback.print_exc()

    # default plugin level
    plugins = blockprocessing.runplugins(plugins, env, pluginbasefolder, url_root)    

    if env['isReverseRun'] and env['pe_forward'] != "None":
        try:
            crawler = xmlrpclib.ServerProxy("http://%s:%d" % (CRAWLER_HOST, CRAWLER_PORT), verbose=False, allow_none=True)
        except (socket.error, xmlrpclib.Fault):
            traceback.print_exc()

        printtime("crawler hostname: "+crawler.hostname())
        printtime("PE Report status: "+crawler.startPE(env['expName'],env['pe_forward'],os.getcwd()))


    getExpLogMsgs(env)
    get_pgm_log_files(env['pathToRaw'])
    
    # multilevel plugins postprocessing
    blockprocessing.runplugins(plugins, env, pluginbasefolder, url_root, 'post')
    # plugins last level - plugins in this level will wait for all previously launched plugins to finish
    blockprocessing.runplugins(plugins, env, pluginbasefolder, url_root, 'last')
    
    printtime("Run Complete")
    sys.exit(0)
