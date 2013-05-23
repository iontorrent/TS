#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Loads Fullchip analysis pipeline
Uses output from there to generate charts and graphs and dumps to current directory
Adds key metrics to database
"""

__version__ = filter(str.isdigit, "$Revision: 57236 $")

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

from ion.reports.plotters import *
from ion.utils.aggregate_alignment import *
sys.path.append('/opt/ion/')
import re

#####################################################################
#
# Analysis implementation details
#
#####################################################################

def getExpLogMsgs(explogfinalfilepath):
    """
    Parses explog_final.txt for warning messages and dumps them to stdout.
    This only works if the raw data files have not been deleted.
    For a from-wells analysis, you may not have raw data.
    """
    printtime("Check file '%s' for warnings" % explogfinalfilepath)
    try:
        with open(explogfinalfilepath, 'r') as f:
            try:
                text = f.readlines()
                for line in text:
                    if "WARNINGS:" in line and len("WARNINGS: ") < len(line):
                        printtime("WARNINGS from explog_final.txt:")
                        printtime(line)
            except:
                traceback.print_exc()
    except:
        printtime("Cannot open file %s" % explogfinalfilepath)


def get_pgm_log_files(rawdatadir):
    # Create a tarball of the pgm raw data log files for inclusion into CSA.
    # tarball it now before the raw data gets deleted.
    # inst diagnostic files are always in toplevel raw data dir:
    if 'thumbnail' in rawdatadir:
        rawdatadir = rawdatadir.replace('thumbnail', '')
    files=['explog_final.txt',
    'explog.txt',
    'InitLog.txt',
    'InitLog1.txt',
    'InitLog2.txt',
    'RawInit.txt',
    'RawInit.jpg',
    'InitValsW3.txt',
    'InitValsW2.txt',
    'Controller',
    'debug']
    for afile in files:
        if os.path.exists(os.path.join(rawdatadir,afile)):
            make_zip ('pgm_logs.zip',os.path.join(rawdatadir,afile),arcname=afile)

    return


def GetBlocksToAnalyze(env):
    blocks = []

    if is_thumbnail or is_wholechip:

        if is_thumbnail:
            blocks.append({'id_str':'thumbnail', 'datasubdir':'thumbnail', 'jobid':None, 'status':None})

        elif is_wholechip:
            blocks.append({'id_str':'wholechip', 'datasubdir':'', 'jobid':None, 'status':None})

    else:
        explogblocks = explogparser.getBlocksFromExpLogJson(env['exp_json'],excludeThumbnail=True)
        for block in explogblocks:
            rawDirectory = block['datasubdir']
            print "expblock: " + str(block)
            if (block['autoanalyze'] and block['analyzeearly']) or os.path.isdir(os.path.join(env['pathToRaw'],rawDirectory)) or env['blockArgs'] == "fromWells":
                print "block: " + str(block)
                blocks.append(block)

    print blocks
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
    jt_args = [os.path.join('/usr/bin',scriptname)] + args
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
    from_sigproc_files = []
    from_sigproc_files.append("iontrace_Library.png")
    from_sigproc_files.append("1.wells")
    from_sigproc_files.append("bfmask.stats")
    from_sigproc_files.append("analysis.bfmask.stats")
    from_sigproc_files.append("analysis.bfmask.bin")
    from_sigproc_files.append("Bead_density_raw.png")
    from_sigproc_files.append("Bead_density_contour.png")
    from_sigproc_files.append("processParameters.txt")
    from_sigproc_files.append("avgNukeTrace_%s.txt" % env['tfKey'])
    from_sigproc_files.append("avgNukeTrace_%s.txt" % env['libraryKey'])
    from_basecaller_files = []
    from_basecaller_files.append("rawlib.basecaller.bam")
    from_basecaller_files.append("rawlib.ionstats_basecaller.json")
    from_basecaller_files.append("bfmask.bin")
    from_basecaller_files.append("readLen.txt")
    from_basecaller_files.append("raw_peak_signal")
    from_basecaller_files.append("BaseCaller.json")
    '''
    print "oninstranalysis:(%s)" % env['oninstranalysis']
    if env['oninstranalysis'] and is_blockprocessing:
        env['blockArgs'] = "fromRaw" # because heatmaps have to be generated on TS

    # define entry point
    print "blockArgs '"+str(env['blockArgs'])+"'"
    print "previousReport: '"+str(env['previousReport'])+"'"

    if is_thumbnail:
        initlogfilepath = os.path.join( env['pathToRaw'],'..','InitLog.txt')
        initlog1filepath = os.path.join( env['pathToRaw'],'..','InitLog1.txt')
        initlog2filepath = os.path.join( env['pathToRaw'],'..','InitLog2.txt')
        explogfilepath = os.path.join( env['pathToRaw'],'..','explog.txt')
        explogfinalfilepath = os.path.join( env['pathToRaw'],'..','explog_final.txt')
    else:
        initlogfilepath = os.path.join( env['pathToRaw'],'InitLog.txt')
        initlog1filepath = os.path.join( env['pathToRaw'],'InitLog1.txt')
        initlog2filepath = os.path.join( env['pathToRaw'],'InitLog2.txt')
        explogfilepath = os.path.join( env['pathToRaw'],'explog.txt')
        explogfinalfilepath = os.path.join( env['pathToRaw'],'explog_final.txt')

    if env['blockArgs'] == "fromRaw":
        runFromDC = True
        runFromSigproc = True
        runFromBasecaller = True

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
        runFromDC = False
        runFromSigproc = True
        runFromBasecaller = True
        initlogfilepath = os.path.join(env['previousReport'],'InitLog.txt')
        initlog1filepath = os.path.join(env['previousReport'],'InitLog1.txt')
        initlog2filepath = os.path.join(env['previousReport'],'InitLog2.txt')
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
    else:
        printtime("WARNING: start point not defined, create new report from raw data")
        runFromDC = True
        runFromSigproc = True
        runFromBasecaller = True

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

    #copy InitLog*.txt and explog.txt into report directory
    for filepath in [initlogfilepath, initlog1filepath, initlog2filepath, explogfilepath]:
        try:
            if os.path.exists(filepath):
                shutil.copy(filepath, ".")
            else:
                printtime("WARN: Cannot copy %s: doesn't exist" % filepath)
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
    plugins, blocklevel_plugins = blockprocessing.get_plugins_to_run(env['plugins'], env['report_type'])

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
            result_dirs[block['id_str']] = blockprocessing.initBlockReport(block, env['SIGPROC_RESULTS'], env['BASECALLER_RESULTS'], env['ALIGNMENT_RESULTS'], pluginbasefolder, env['oninstranalysis'])

        # create a list of blocks
        blocks_to_process = []
        blocks_to_process.extend(blocks)
        timeout = 3*60*60
        if env['oninstranalysis'] and is_blockprocessing:
            timeout = 20*60*60

        env['block_dirs'] = [os.path.join(env['report_root_dir'],result_dirs[block['id_str']]) for block in blocks_to_process]

        while len(blocks_to_process) > 0 and timeout > 0:
            for block in blocks_to_process:
                printtime('waiting for %s block(s) to start' % str(len(blocks_to_process)))
                sys.stdout.flush()
                sys.stderr.flush()

                if runFromDC:
                    if is_thumbnail or is_wholechip:
                        data_file = os.path.join(env['pathToRaw'],'acq_0000.dat')
                    else:
                        data_file = os.path.join(env['pathToRaw'],block['id_str'],'acq_0000.dat')

                    if env['oninstranalysis'] and is_blockprocessing:
                        data_file = os.path.join(env['SIGPROC_RESULTS'],'block_'+block['id_str'])

                    if os.path.exists(data_file):

                        if env['oninstranalysis'] and is_blockprocessing:
                            # check activity on directory
                            mod_time = os.stat(data_file).st_mtime
                            cur_time = time.time()
                            if cur_time - mod_time < 180:
                                printtime("mtime %s" % mod_time)
                                printtime("ctime %s" % cur_time)
                                timeout -= 10
                                time.sleep (10)
                                continue

                            wait_list = [
                                         os.path.join(data_file, 'analysis_return_code.txt'),
                                         os.path.join(data_file, 'analysis.bfmask.bin'),
                                         os.path.join(data_file, 'analysis.bfmask.stats'),
                                         os.path.join(data_file, 'processParameters.txt'),
                                         os.path.join(data_file, 'avgNukeTrace_%s.txt' % env['tfKey']),
                                         os.path.join(data_file, 'avgNukeTrace_%s.txt' % env['libraryKey']),
                                         os.path.join(data_file, '1.wells'),
                                        ]

                            analysis_return_code_file = os.path.join(data_file, 'analysis_return_code.txt')
                            if os.path.exists(analysis_return_code_file):
                                try:
                                    with open(analysis_return_code_file, 'r') as f:
                                        analysis_return_code = int(f.read())
                                        if analysis_return_code != 0:
                                            printtime("WARNING: Analysis returned %s" % analysis_return_code)

                                        for transferred_file in wait_list:
                                            if not hash_matches(transferred_file):
                                                printtime("WARNING: %s might be corrupt" % transferred_file)
                                                #blocks_to_process.remove(block)
                                                #continue
                                except:
                                    traceback.print_exc()
                                    # continue with job submission
                            else:
                                # delayed (for example back-to-back run) or serious issue
                                printtime("WARNING: %s file transfer delayed" % analysis_return_code_file)
                                for transferred_file in wait_list:
                                    if not os.path.exists(transferred_file):
                                        printtime("WARNING: %s file transfer delayed" % transferred_file)
                                timeout -= 10
                                time.sleep (10)
                                continue

                        block['jobid'] = spawn_cluster_job(result_dirs[block['id_str']],'BlockTLScript.py', ['--do-sigproc'])
                        sigproc_job_dict[block['id_str']] = str(block['jobid'])
                        printtime("Submitted block (%s) analysis job with job ID (%s)" % (block['id_str'], str(block['jobid'])))
                    else:
                        printtime("missing %s" % data_file)
                        timeout -= 10
                        time.sleep (10)
                        continue


                if runFromSigproc:
                    try:
                        if env['blockArgs'] == "fromWells":
                            wait_list = []
                        else:
                            wait_list = [ sigproc_job_dict[block['id_str']] ]
                        block['jobid'] = spawn_cluster_job(result_dirs[block['id_str']],'BlockTLScript.py',['--do-basecalling'],wait_list)
                        basecaller_job_dict[block['id_str']] = str(block['jobid'])
                        printtime("Submitted block (%s) basecaller job with job ID (%s)" % (block['id_str'], str(block['jobid'])))
                    except:
                        printtime("submitting basecaller job for block (%s) failed" % block['id_str'])


                if runFromBasecaller:
                    try:
                        wait_list = [ basecaller_job_dict[block['id_str']] ]
                        block['jobid'] = spawn_cluster_job(result_dirs[block['id_str']],'BlockTLScript.py',['--do-alignment'],wait_list)
                        alignment_job_dict[block['id_str']] = str(block['jobid'])
                        printtime("Submitted block (%s) alignment job with job ID (%s)" % (block['id_str'], str(block['jobid'])))
                    except:
                        printtime("submitting alignment job for block (%s) failed" % block['id_str'])

                blocks_to_process.remove(block)


        if not is_thumbnail and not is_wholechip:
            if runFromDC:
                merge_job_dict['sigproc'] = spawn_cluster_job('.','MergeTLScript.py',['--do-sigproc'],sigproc_job_dict.values())
            if runFromSigproc:
                merge_job_dict['basecaller'] = spawn_cluster_job('.','MergeTLScript.py',['--do-basecalling'],basecaller_job_dict.values())
            if runFromBasecaller:
                merge_job_dict['alignment'] = spawn_cluster_job('.','MergeTLScript.py',['--do-alignment'],alignment_job_dict.values()+[merge_job_dict.get('basecaller','')])

        # always create links to downloadable files
        merge_job_dict['merge/zipping'] = spawn_cluster_job('.','MergeTLScript.py',['--do-zipping'],alignment_job_dict.values()+[merge_job_dict.get('alignment','')])

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

                if blocklevel_plugins and (block['status']=='done'):
                    block_pluginbasefolder = os.path.join(result_dirs[block['id_str']],pluginbasefolder)
                    env['blockId'] = block['id_str']
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

    ########################################################
    #ParseFiles and Upload Metrics                         #
    ########################################################
    printtime("Attempting to Upload to Database")

    #attempt to upload the metrics to the Django database
    try:
        mycwd = os.getcwd()

        ionParamsPath = os.path.join('.','ion_params_00.json')
        BaseCallerJsonPath = os.path.join(env['BASECALLER_RESULTS'],'BaseCaller.json')
        tfmapperstats_outputfile = os.path.join(env['BASECALLER_RESULTS'],"TFStats.json")
        merged_bead_mask_path = os.path.join(env['SIGPROC_RESULTS'], 'MaskBead.mask')
        ionstats_basecaller_json_path = os.path.join(env['BASECALLER_RESULTS'],'ionstats_basecaller.json')
        peakOut = os.path.join('.','raw_peak_signal')
        beadPath = os.path.join(env['SIGPROC_RESULTS'],'analysis.bfmask.stats')
        procPath = os.path.join(env['SIGPROC_RESULTS'],'processParameters.txt')
        ionstats_alignment_json_path = os.path.join(env['ALIGNMENT_RESULTS'],'ionstats_alignment.json')
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
                        if component == 'Beadfind':
                            if int(status) == 2:
                                STATUS = 'Checksum Error'
                            elif int(status) == 3:
                                STATUS = 'No Live Beads'
                            else:
                                STATUS = "Error in Beadfind"
                        elif component == 'Analysis':
                            if int(status) == 2:
                                STATUS = 'Checksum Error'
                            elif int(status) == 3:
                                STATUS = 'No Live Beads'
                            else:
                                STATUS = "Error in Analysis"
                        else:
                            STATUS = "Error in %s" % component
                        break
            else:
                STATUS = "Error"
        elif is_blockprocessing:
            try:
                raw_return_code_file = os.path.join(env['BASECALLER_RESULTS'],"composite_return_code.txt")
                f = open(raw_return_code_file)
                return_code = f.readline()
                f.close()
                if int(return_code) == 0:
                    STATUS = "Completed"
                    # remove unneeded block files
                    printtime("Remove unneeded block files %s" % len(dirs))
                    blockprocessing.remove_unneeded_block_files(dirs)
                else:
                    STATUS = "Completed with %s error(s)" % return_code
            except:
                STATUS = "Error"
                printtime(traceback.format_exc())
        else:
            STATUS = "Error"

        ret_message = jobserver.uploadmetrics(
            os.path.join(mycwd,tfmapperstats_outputfile),
            os.path.join(mycwd,procPath),
            os.path.join(mycwd,beadPath),
            os.path.join(mycwd,ionstats_alignment_json_path),
            os.path.join(mycwd,ionParamsPath),
            os.path.join(mycwd,peakOut),
            os.path.join(mycwd,ionstats_basecaller_json_path),
            os.path.join(mycwd,BaseCallerJsonPath),
            os.path.join(mycwd,'primary.key'),
            os.path.join(mycwd,'uploadStatus'),
            STATUS,
            reportLink,
            mycwd)
        # this will replace the five progress squares with a re-analysis button
        printtime("jobserver.uploadmetrics returned:\n"+str(ret_message))
    except:
        traceback.print_exc()

    try:
        # Call script which creates and populates a file with
        # experiment metrics.  RSMAgent_TS then forwards this file
        # to the Axeda remote system monitoring server.
        primary_key = open("primary.key").readline()
        primary_key = primary_key.split(" = ")
        primary_key = primary_key[1]
        rsm_message = jobserver.createRSMExperimentMetrics(primary_key)
        printtime("jobserver.createRSMExperimentMetrics returned: "+str(rsm_message))
    except:
        printtime("RSM createExperimentMetrics.py failed")


    #copy explog_final.txt into report directory
    try:
        if os.path.exists(explogfinalfilepath):
            shutil.copy(explogfinalfilepath, ".")
        else:
            printtime("WARN: Cannot copy %s: doesn't exist" % explogfinalfilepath)
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
        #never overwrite existing checksum_status.txt file, needed in R&D to keep important data
        if not os.path.exists(os.path.join(env['pathToRaw'],"checksum_status.txt")):

            keep_raw_data = False
            try:
                if env['libraryName']!='none' and len(env['libraryName'])>0:
                    if os.path.exists(ionstats_alignment_json_path):
                        afile = open(ionstats_alignment_json_path, 'r')
                        ionstats_alignment = json.load(afile)
                        afile.close()
                        if int(ionstats_alignment['AQ17']['num_bases']) > 13000000000:
                            keep_raw_data = True
                            f = open(raw_return_code_file+".keep", 'w')
                            f.write(str(99))
                            f.close()
                            printtime("INFO: keep raw data")
                        else:
                            printtime("INFO: delete raw data if all blocks have been processed successfully")
                    else:
                        printtime("ERROR: keep_raw_data check failed: ionstats_alignment.json is missing")
                else:
                    printtime("INFO: reference library not set, skip AQ17 check")
            except:
               printtime("ERROR: keep_raw_data check failed")
               traceback.print_exc()

            if (is_wholechip or is_blockprocessing) and os.path.isfile(raw_return_code_file):
                if keep_raw_data:
                    shutil.copyfile(raw_return_code_file+".keep",os.path.join(env['pathToRaw'],"checksum_status.txt"))
                else:
                    shutil.copyfile(raw_return_code_file,os.path.join(env['pathToRaw'],"checksum_status.txt"))

    except:
        traceback.print_exc()

    # default plugin level
    plugins = blockprocessing.runplugins(plugins, env, pluginbasefolder, url_root)

    getExpLogMsgs(explogfinalfilepath)
    get_pgm_log_files(env['pathToRaw'])

    # multilevel plugins postprocessing
    blockprocessing.runplugins(plugins, env, pluginbasefolder, url_root, 'post')
    # plugins last level - plugins in this level will wait for all previously launched plugins to finish
    blockprocessing.runplugins(plugins, env, pluginbasefolder, url_root, 'last')

    ####################################################
    # Record disk space usage for the Result directory #
    ####################################################
    try:
        jobserver.resultdiskspace(env['primary_key'])
    except:
        traceback.print_exc()

    printtime("Run Complete")
    sys.exit(0)
