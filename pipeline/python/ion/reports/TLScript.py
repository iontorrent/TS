#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Loads Fullchip analysis pipeline
Uses output from there to generate charts and graphs and dumps to current directory
Adds key metrics to database(not implemented yet)
"""

__version__ = filter(str.isdigit, "$Revision: 23670 $")

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
import datetime
import shutil
import socket
import xmlrpclib
import subprocess
import sys
import time
import numpy
import StringIO
import traceback
import json
import xmlrpclib
from ion.utils.plugin_json import *

from collections import deque
from urlparse import urlunsplit

# Import analysis-specific packages. We try to follow the from A import B
# convention when B is a module in package A, or import A when
# A is a module.
#from ion.analysis import cafie,sigproc
#from ion.fileformats import sff
from ion.reports import blast_to_ionogram, tfGraphs, plotKey, \
    parseBeadfind, parseTFstats, \
    parseProcessParams, beadDensityPlot, \
    libGraphs, beadHistogram, plotRawRegions, trimmedReadLenHisto, MaskMerge,\
    StatsMerge, processParametersMerge, mergeBaseCallerJson
from ion.reports.plotters import *
from ion.utils.aggregate_alignment import *
from ion.utils import TFPipeline
sys.path.append('/opt/ion/')
import re
from scipy import cluster, signal, linalg

import zipfile
try:
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

modes = { zipfile.ZIP_DEFLATED: 'deflated',
          zipfile.ZIP_STORED:   'stored',
          }

def write_version():
    a = subprocess.Popen('/opt/ion/iondb/bin/lversionChk.sh --ion', shell=True, stdout=subprocess.PIPE)
    ret = a.stdout.readlines()
    f = open('version.txt','w')
    for i in ret[:len(ret)-1]:
#    for i in ret:

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
            afile = StringIO.StringIO("[global]\n" + text)
            self.readfp(afile, filename)

def printtime(message, *args):
    if args:
        message = message % args
    print "[ " + time.strftime('%X') + " ] " + message
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
    'InitLog.txt',
    'RawInit.txt',
    'RawInit.jpg',
    'InitValsW3.txt',
    'InitValsW2.txt']
    for afile in files:
        make_zip ('pgm_logs.zip',os.path.join(rawdatadir,afile), afile)
        
    return

def isbadblock(blockdir, message):
    if os.path.exists(os.path.join(blockdir,'badblock.txt')):
        printtime("WARNING: %s: skipped %s" % (message,blockdir))
        return True
    return False

def initreports():
    """Performs initialization for both report writing and background
    report generation."""
    printtime("INIT REPORTS")

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

    #
    # Begin report writing
    #
    os.umask(0002)
    TMPL_DIR = '/usr/lib/python2.6/dist-packages/ion/web/db/writers'
    shutil.copy("%s/report_layout.json" % TMPL_DIR, "report_layout.json")
    shutil.copy("%s/parsefiles.php" % TMPL_DIR, "parsefiles.php")
    shutil.copy("%s/log.html" % TMPL_DIR, 'log.html')
    shutil.copy("%s/alignment_summary.html" % TMPL_DIR, os.path.join(ALIGNMENT_RESULTS,"alignment_summary.html"))
    shutil.copy("%s/format_whole.php" % TMPL_DIR, "Default_Report.php")
    shutil.copy("%s/csa.php" % TMPL_DIR, "csa.php")
    shutil.copy("/usr/lib/python2.6/dist-packages/ion/reports/BlockTLScript.py", "BlockTLScript.py")
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
        if datasubdir == 'thumbnail':
            W = int(args[2].strip('W '))
            H = int(args[3].strip('H '))
            break

    return [W,H]

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

        elif is_wholechip:
            block['id_str'] = 'wholechip'

        base_args.append("--output-dir=%s" % SIGPROC_RESULTS)
        base_args.append("--basecaller-output-dir=%s" % BASECALLER_RESULTS)

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

            # Remove leading space
            args = [entry.strip() for entry in args]

            #ignore thumbnail
            if args[0] =='thumbnail':
                continue

            #autoanalyze
            autoanalyze = int(args[4].split(':')[1].strip()) == 1
            #analyzeearly
            analyzeearly = int(args[5].split(':')[1].strip()) == 1
            # Define Block dictionary object
            #   id_str contains a unique id string
            #   datasubdir contains name of block directory (i.e. 'X0_Y128')
            #   jobcmd contains array of Analysis command line arguments
            #   jobid contains job id returned when job is queued
            #   status contains job status string
            block = {'id_str':'',
                    'datasubdir':'',
                    'jobcmd':[],
                    'jobid':None,
                    'status':None}

            block['datasubdir'] = "%s_%s" % (args[0].strip(),args[1].strip())
            block['id_str'] = block['datasubdir']

            base_args = env['analysisArgs'].strip().split(' ')
            # raw data dir is last element in analysisArgs and needs the block subdirectory appended
            base_args[-1] = os.path.join(base_args[-1], block['datasubdir'])
            rawDirectory = base_args[-1]
            base_args.append("--libraryKey=%s" % env["libraryKey"])
            base_args.append("--no-subdir")
#            base_args.append("--output-dir=../%s/block_%s" % (SIGPROC_RESULTS, block['id_str']))
#            base_args.append("--basecaller-output-dir=../%s/block_%s" % (BASECALLER_RESULTS, block['id_str']))
            base_args.append("--output-dir=%s" % SIGPROC_RESULTS)
            base_args.append("--basecaller-output-dir=%s" % BASECALLER_RESULTS)

            block['jobcmd'] = base_args

            if (autoanalyze and analyzeearly) or os.path.isdir(rawDirectory):
                print base_args
                print block
                blocks.append(block)

    return blocks

def spawn_cluster_job(rpath):
    out_path = "%s/drmaa_stdout_block.html" % rpath
    err_path = "%s/drmaa_stderr_block.txt" % rpath
    logout = open(os.path.join(out_path), "w")
    logout.write("<html><head><meta http-equiv=refresh content='5'; URL=''></head><pre> \n")
    logout.close()
    cwd = os.getcwd()

    jt_nativeSpecification = ""
    jt_remoteCommand = "python"
    jt_workingDirectory = os.path.join(cwd, rpath)
    jt_outputPath = ":" + os.path.join(cwd, out_path)
    jt_errorPath = ":" + os.path.join(cwd, err_path)
    if is_wholechip:
        jt_args = ['BlockTLScript.py', 'ion_params_00.json']
    elif is_thumbnail:
        jt_args = ['BlockTLScript.py', 'ion_params_00.json']
    else:
        jt_args = ['../BlockTLScript.py', 'ion_params_00.json']
    jt_joinFiles = False

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

def submitBlockJob(blockObj, env):

    if is_wholechip:
        resultDir = "./"
    elif is_thumbnail:
        resultDir = "./"
    else:
        resultDir = './%s%s' % ('block_', blockObj['id_str'])

        if not os.path.exists(resultDir):
            os.mkdir(resultDir)

        #write block specific ion_params_00.json
        if not os.path.exists("%s/ion_params_00.json" % resultDir):

            file_in = open("ion_params_00.json", 'r')
            TMP_PARAMS = json.loads(file_in.read())
            file_in.close()

            TMP_PARAMS["pathToData"] = os.path.join(TMP_PARAMS["pathToData"], blockObj['id_str'])
            TMP_PARAMS["analysisArgs"] = ' '.join(blockObj['jobcmd'])
            file_out = open("%s/ion_params_00.json" % resultDir, 'w')
            json.dump(TMP_PARAMS, file_out)
            file_out.close()

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


    jobid = spawn_cluster_job(resultDir)

    return jobid


def runFullChip(env):
    STATUS = None
    basefolder = 'plugin_out'
    if not os.path.isdir(basefolder):
        os.umask(0000)   #grant write permission to plugin user
        os.mkdir(basefolder)
        os.umask(0002)

    libsff = "%s/%s_%s.sff" % (BASECALLER_RESULTS, env['expName'], env['resultsName'])
    tfsff = "%s/%s_%s.tf.sff" % (BASECALLER_RESULTS, env['expName'], env['resultsName'])
    fastqpath = "%s/%s_%s.fastq" % (BASECALLER_RESULTS, env['expName'], env['resultsName'])

    #launcher = PluginRunner()

    #-------------------------------------------------------------
    # Gridded data processing
    #-------------------------------------------------------------
    write_version()

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

        job_list = []
        # Launch multiple Block analysis jobs
        for block in blocks:
            block['jobid'] = submitBlockJob(block,env)
            printtime("Submitted block (%s) analysis job with job ID (%s)" % (block['id_str'], str(block['jobid'])))

            if block['jobid'] != -1:
                job_list.append(block['jobid'])

        # write job id's to file
        f = open('job_list.txt','w')
        for jobid in job_list:
            f.write(jobid+'\n')
        f.close()

        # Watch status of jobs.  As they finish, run the merge script and remove
        # the job from the list.

        pl_started = False
        while len(job_list) > 0:
            for job in job_list:
                block = [block for block in blocks if block['jobid'] == job][0]
                #check status of jobid
                try:
                    block['status'] = jobserver.jobstatus(block['jobid'])
                except:
                    traceback.print_exc()
                    continue
                if block['status']=='done' or block['status']=='failed':
                    printtime("Job %s has ended with status %s" % (str(block['jobid']),block['status']))
                    job_list.remove(block['jobid'])
#                else:
#                    printtime("Job %s has status %s" % (str(block['jobid']),block['status']))

                # Hack
                if is_thumbnail and not pl_started:
                    f = open('progress.txt')
                    text = f.read()
                    f.close()       
                    matches = re.findall(r"wellfinding = green", text)

                    if len(matches) != 0:
                        pl_started = True
                        for plugin in sorted(env['plugins'], key=lambda plugin: plugin["name"],reverse=True):
                            if plugin['name'] == 'rawPlots' or plugin['name'] == 'separator' or plugin['name'] == 'chipNoise':
                                runPlug = True
                                printtime("Plugin %s is enabled" % plugin['name'])

                                try:
                                    #https://iontorrent.jira.com/wiki/display/TS/Plugin+json+file+format
                                    env['report_root_dir'] = os.getcwd()
                                    env['analysis_dir'] = os.getcwd()
                                    env['sigproc_dir'] = os.path.join(env['report_root_dir'],SIGPROC_RESULTS)
                                    env['basecaller_dir'] = os.path.join(env['report_root_dir'],BASECALLER_RESULTS)
                                    env['alignment_dir'] = os.path.join(env['report_root_dir'],ALIGNMENT_RESULTS)
                                    env['testfrag_key'] = 'ATCG'
                                    printtime("RAWDATA: %s" % env['pathToRaw'])
                                    start_json = make_plugin_json(env,plugin,env['primary_key'],basefolder,url_root)
                                    ret = pluginserver.pluginStart(start_json)
                                    printtime('plugin %s started ...' % plugin['name'])
                                except:
                                    printtime('plugin %s failed...' % plugin['name'])
                                    traceback.print_exc()

            time.sleep (5)
            printtime("waiting for %d blocks to be finished" % len(job_list))

    printtime("All jobs processed")

    tfKey = "ATCG"
    libKey = env['libraryKey']
    floworder = env['flowOrder']
    printtime("Using flow order: %s" % floworder)
    printtime("Using library key: %s" % libKey)
    tfmapperstats_outputfile = os.path.join(BASECALLER_RESULTS,"TFMapper.stats")
    merged_bead_mask_path = os.path.join(SIGPROC_RESULTS, 'MaskBead.mask')
    QualityPath = os.path.join(BASECALLER_RESULTS,'quality.summary')
    peakOut = 'raw_peak_signal'
    beadPath = os.path.join(SIGPROC_RESULTS,'bfmask.stats')
    alignmentSummaryPath = os.path.join(ALIGNMENT_RESULTS,'alignment.summary')
    BaseCallerJsonPath = os.path.join(BASECALLER_RESULTS,'BaseCaller.json')
    block_tfsff = "rawtf.sff"
    block_libsff = "rawlib.sff"

    #--------------------------------------------------------
    # Start merging results files
    #--------------------------------------------------------
    if is_thumbnail or is_wholechip:
        printtime("MERGING: THUMBNAIL OR 31X - skipping merge process.")

      #  res = models.Results.objects.get(pk=env['primary_key'])
      #  res.metaData["thumb"] = 1
        #res.timeStamp = datetime.datetime.now()
      #  res.save()
      #  printtime("thumbnail: "+str(res.metaData["thumb"]))

    if runFromRaw:
        printtime("PROCESSING FROM RAW")
        if not is_thumbnail and not is_wholechip:
            #####################################################
            # Grab one of the processParameters.txt files       #
            #####################################################
            printtime("Merging processParameters.txt")

            for subdir in dirs:
                subdir = os.path.join(SIGPROC_RESULTS,subdir)
                ppfile = os.path.join(subdir,'processParameters.txt')
                printtime(ppfile)
                if os.path.isfile(ppfile):
                    processParametersMerge.processParametersMerge(ppfile,True)
                    break

    ## BASECALLER
    if runFromWells:
        printtime("PROCESSING FROM WELLS")
        if not is_thumbnail and not is_wholechip:
            ############################################
            # Merge individual quality.summary files #
            ############################################
            printtime("Merging individual quality.summary files")

            config_out = ConfigParser.RawConfigParser()
            config_out.optionxform = str # don't convert to lowercase
            config_out.add_section('global')

            numberkeys = ['Number of 50BP Reads',
                          'Number of 100BP Reads',
                          'Number of 150BP Reads',
                          'Number of Reads at Q0',
                          'Number of Bases at Q0',
                          'Number of 50BP Reads at Q0',
                          'Number of 100BP Reads at Q0',
                          'Number of 150BP Reads at Q0',
                          'Number of Reads at Q17',
                          'Number of Bases at Q17',
                          'Number of 50BP Reads at Q17',
                          'Number of 150BP Reads at Q17',
                          'Number of 100BP Reads at Q17',
                          'Number of Reads at Q20',
                          'Number of Bases at Q20',
                          'Number of 50BP Reads at Q20',
                          'Number of 100BP Reads at Q20',
                          'Number of 150BP Reads at Q20']

            maxkeys = ['Max Read Length at Q0',
                       'Max Read Length at Q17',
                       'Max Read Length at Q20']

            meankeys = ['System SNR',
                        'Mean Read Length at Q0',
                        'Mean Read Length at Q17',
                        'Mean Read Length at Q20']

            config_in = MyConfigParser()
            config_in.optionxform = str # don't convert to lowercase
            doinit = True
            for i,subdir in enumerate(dirs):
                if isbadblock(subdir, "Merging quality.summary"):
                    continue
                summaryfile=os.path.join(BASECALLER_RESULTS, subdir, 'quality.summary')
                if os.path.exists(summaryfile):
                    printtime("INFO: process %s" % summaryfile)
                    config_in.read(summaryfile)
                    for key in numberkeys:
                        value_in = config_in.get('global',key)
                        if doinit:
                            value_out = 0
                        else:
                            value_out = config_out.get('global', key)
                        config_out.set('global', key, int(value_in) + int(value_out))
                    for key in maxkeys:
                        value_in = config_in.get('global',key)
                        if doinit:
                            value_out = 0
                        else:
                            value_out = config_out.get('global', key)
                        config_out.set('global', key, max(int(value_in),int(value_out)))
                    for key in meankeys:
                        value_in = config_in.get('global',key)
                        if doinit:
                            value_out = 0
                        else:
                            value_out = config_out.get('global', key)
                        config_out.set('global', key, float(value_out)+float(value_in)/len(dirs))
                    doinit = False
                else:
                    printtime("ERROR: skipped %s" % summaryfile)

            with open(QualityPath, 'wb') as configfile:
                config_out.write(configfile)

            #################################################
            # Merge individual block bead metrics files     #
            #################################################
            printtime("Merging individual block bead metrics files")

            try:
                _tmpfile = os.path.join(SIGPROC_RESULTS,'bfmask.bin')
                cmd = 'BeadmaskMerge -i bfmask.bin -o ' + _tmpfile
                for subdir in dirs:
                    subdir = os.path.join(SIGPROC_RESULTS,subdir)
                    if isbadblock(subdir, "Merging individual block bead metrics files"):
                        continue
                    bfmaskbin = os.path.join(subdir,'bfmask.bin')
                    if os.path.exists(bfmaskbin):
                        cmd = cmd + ' %s' % subdir
                    else:
                        printtime("ERROR: skipped %s" % bfmaskbin)
                printtime("DEBUG: Calling '%s'" % cmd)
                subprocess.call(cmd,shell=True)
            except:
                printtime("BeadmaskMerge failed (test fragments)")

            ###############################################
            # Merge individual block MaskBead files       #
            ###############################################
            printtime("Merging MaskBead.mask files")

            try:
                bfmaskfolders = []
                for subdir in dirs:
                    subdir = os.path.join(SIGPROC_RESULTS,subdir)
                    printtime("DEBUG: %s:" % subdir)

                    if isbadblock(subdir, "Merging MaskBead.mask files"):
                        continue

                    bfmaskbead = os.path.join(subdir,'MaskBead.mask')
                    if not os.path.exists(bfmaskbead):
                        printtime("ERROR: Merging MaskBead.mask files: skipped %s" % bfmaskbead)
                        continue

                    bfmaskfolders.append(subdir)

                offset_str = "use_blocks"
                MaskMerge.main_merge('MaskBead.mask', bfmaskfolders, merged_bead_mask_path, True, offset_str)
            except:
                printtime("Merging MaskBead.mask files failed")

            ##################################################
            #generate TF Metrics                             #
            #look for both keys and append same file         #
            ##################################################
            
            printtime("Merging TFMapper metrics and generating TF plots")
            
            try:
                TFPipeline.mergeBlocks(BASECALLER_RESULTS,dirs,floworder)
            
            except:
                printtime("ERROR: Merging TFMapper metrics failed")
            
            
            #printtime("Calling TFMapper")
            #
            #try:
            #    cmd = "TFMapper"
            #    cmd += " --logfile TFMapper.log"
            #    cmd += " --output-dir=%s" % (BASECALLER_RESULTS)
            #    cmd += " --wells-dir=%s" % (SIGPROC_RESULTS)
            #    cmd += " --sff-dir=%s" % (BASECALLER_RESULTS)
            #    cmd += " --tfkey=%s" % (tfKey)
            #    cmd += " %s" % (block_tfsff)
            #    for subdir in dirs:
            #        _subdir = os.path.join(BASECALLER_RESULTS,subdir)
            #        if isbadblock(_subdir, "TFMapper tf files"):
            #            continue
            #        rawtfsff = os.path.join(_subdir,block_tfsff)
            #        if os.path.exists(rawtfsff):
            #            cmd = cmd + ' %s' % subdir
            #        else:
            #            printtime("ERROR: skipped %s" % rawtfsff)
            #    cmd = cmd + " > %s" % tfmapperstats_outputfile
            #    printtime("DEBUG: Calling '%s'" % cmd)
            #    os.system(cmd)
            #except:
            #    printtime("ERROR: TFMapper failed")
            #
            ########################################################
            #generate the TF Metrics including plots               #
            ########################################################
            #printtime("generate the TF Metrics including plots")
            #
            #tfMetrics = None
            #if os.path.exists(tfmapperstats_outputfile):
            #    try:
            #        # Q17 TF Read Length Plot
            #        tfMetrics = parseTFstats.generateMetricsData(tfmapperstats_outputfile)
            #        tfGraphs.Q17(tfMetrics)
            #        tfGraphs.genCafieIonograms(tfMetrics,floworder)
            #    except Exception:
            #        printtime("ERROR: Metrics Gen Failed")
            #        traceback.print_exc()
            #else:
            #    printtime("ERROR: %s doesn't exist" % tfmapperstats_outputfile)
            #    tfMetrics = None

            ###############################################
            # Merge individual block bead stats files     #
            ###############################################
            printtime("Merging bfmask.stats files")

            try:
                bfmaskstatsfiles = []
                for subdir in dirs:
                    subdir = os.path.join(SIGPROC_RESULTS,subdir)
                    if isbadblock(subdir, "Merging bfmask.stats files"):
                        continue
                    bfmaskstats = os.path.join(subdir,'bfmask.stats')
                    if os.path.exists(bfmaskstats):
                        bfmaskstatsfiles.append(subdir)
                    else:
                        printtime("ERROR: Merging bfmask.stats files: skipped %s" % bfmaskstats)

                StatsMerge.main_merge(bfmaskstatsfiles, True)
                #TODO
                shutil.move('bfmask.stats', SIGPROC_RESULTS)
            except:
                printtime("No bfmask.stats files were found to merge")

            ###############################################
            # Merge BaseCaller.json files                 #
            ###############################################
            printtime("Merging BaseCaller.json files")

            try:
                basecallerfiles = []
                for subdir in dirs:
                    subdir = os.path.join(BASECALLER_RESULTS,subdir)
                    printtime("DEBUG: %s:" % subdir)
                    if isbadblock(subdir, "Merging BaseCaller.json files"):
                        continue
                    basecallerjson = os.path.join(subdir,'BaseCaller.json')
                    if os.path.exists(basecallerjson):
                        basecallerfiles.append(subdir)
                    else:
                        printtime("ERROR: Merging BaseCaller.json files: skipped %s" % basecallerjson)

                mergeBaseCallerJson.merge(basecallerfiles,BASECALLER_RESULTS)
            except:
                printtime("Merging BaseCaller.json files failed")


            ########################################
            # Merge individual block SFF files     #
            ########################################
            printtime("Merging Library SFF files")
            try:
                cmd = 'SFFMerge'
                cmd = cmd + ' -i rawlib.sff'
                cmd = cmd + ' -o %s ' % libsff
                for subdir in dirs:
                    subdir = os.path.join(BASECALLER_RESULTS,subdir)
                    if isbadblock(subdir, "Merging Library SFF files"):
                        continue
                    rawlibsff = os.path.join(subdir,'rawlib.sff')
                    if os.path.exists(rawlibsff):
                        cmd = cmd + ' %s' % subdir
                    else:
                        printtime("ERROR: skipped %s" % rawlibsff)
                printtime("DEBUG: Calling '%s'" % cmd)
                subprocess.call(cmd,shell=True)
            except:
                printtime("SFFMerge failed (library)")

            printtime("Merging Test Fragment SFF files")
            try:
                cmd = 'SFFMerge'
                cmd = cmd + ' -i rawtf.sff'
                cmd = cmd + ' -o %s ' % tfsff
                for subdir in dirs:
                    subdir = os.path.join(BASECALLER_RESULTS,subdir)
                    if isbadblock(subdir, "Merging Test Fragment SFF files"):
                        continue
                    rawtfsff = os.path.join(subdir,'rawtf.sff')
                    if os.path.exists(rawtfsff):
                        cmd = cmd + ' %s' % subdir
                    else:
                        printtime("ERROR: skipped %s" % rawtfsff)
                printtime("DEBUG: Calling '%s'" % cmd)
                subprocess.call(cmd,shell=True)
            except:
                printtime("SFFMerge failed (test fragments)")


        ########################################################
        #Make Bead Density Plots                               #
        ########################################################
        printtime("Make Bead Density Plots (composite report)")

        bfmaskPath = os.path.join(SIGPROC_RESULTS,'bfmask.bin')
        maskpath = os.path.join(SIGPROC_RESULTS,'MaskBead.mask')

        if os.path.isfile(bfmaskPath):
            com = "BeadmaskParse -m MaskBead %s" % bfmaskPath
            os.system(com)
            #TODO
            try:
                shutil.move('MaskBead.mask', maskpath)
            except:
                printtime("ERROR: MaskBead.mask already moved")
        else:
            printtime("Warning: %s doesn't exists." % bfmaskPath)

        if os.path.exists(maskpath):
            try:
                # Makes Bead_density_contour.png
                beadDensityPlot.genHeatmap(maskpath, BASECALLER_RESULTS) # todo, takes too much time
      #          os.remove(maskpath)
            except:
                traceback.print_exc()
        else:
            printtime("Warning: no MaskBead.mask file exists.")

        ##################################################
        # Create zip of files
        ##################################################

        #sampled sff
        #make_zip(libsff.replace(".sff",".sampled.sff")+'.zip', libsff.replace(".sff",".sampled.sff"))

        #library sff
        #make_zip(libsff + '.zip', libsff )

        #tf sff
        #make_zip(tfsff + '.zip', tfsff)

        #fastq zip
        #make_zip(fastqpath + '.zip', fastqpath)

        #sampled fastq
        #make_zip(fastqpath.replace(".fastq",".sampled.fastq")+'.zip', fastqpath.replace(".fastq",".sampled.fastq"))

    ## do ALIGNMENT
    if runFromSFF:
        printtime("PROCESSING FROM SFF")
        if not is_thumbnail and not is_wholechip:

            #############################################
            # Merge individual block bam files   #
            #############################################
            printtime("Merging bam files")
            try:
        #        cmd = 'picard-tools MergeSamFiles'
                cmd = 'java -Xmx2g -jar /opt/picard/picard-tools-current/MergeSamFiles.jar'
                for subdir in dirs:
                    if isbadblock(subdir, "Merging bam files"):
                        continue
                    bamfile = os.path.join(ALIGNMENT_RESULTS, subdir, "rawlib.bam")
                    if os.path.exists(bamfile):
                        cmd = cmd + ' I=%s' % bamfile
                    else:
                        printtime("ERROR: skipped %s" % bamfile)
                cmd = cmd + ' O=%s/%s_%s.bam' % (ALIGNMENT_RESULTS, env['expName'], env['resultsName'])
                cmd = cmd + ' ASSUME_SORTED=true'
                cmd = cmd + ' USE_THREADING=true'
                cmd = cmd + ' VALIDATION_STRINGENCY=LENIENT'
                printtime("DEBUG: Calling '%s'" % cmd)
                subprocess.call(cmd,shell=True)
            except:
                printtime("bam file merge failed")

            ##################################################
            #Call alignStats on merged bam file              #
            ##################################################
            printtime("Call alignStats on merged bam file")

            try:
                cmd = "alignStats -i %s/%s_%s.bam" % (ALIGNMENT_RESULTS, env['expName'], env['resultsName'])
                cmd = cmd + " -g /results/referenceLibrary/tmap-f2/%s/%s.info.txt" % (env["libraryName"], env["libraryName"])
                cmd = cmd + " -n 12 -l 20 -m 400 -q 7,10,17,20,47 -s 0 -a alignTable.txt"
                cmd = cmd + " --outputDir %s" % ALIGNMENT_RESULTS
                cmd = cmd + " 2>> " + os.path.join(ALIGNMENT_RESULTS, "alignStats_out.txt")
                printtime("DEBUG: Calling '%s'" % cmd)
                os.system(cmd)
            except:
                printtime("alignStats failed")

    ### end alignment



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
        procParams = None
        filterPath = None

        mycwd = os.getcwd()
        jobserver.uploadmetrics(
            os.path.join(mycwd,tfmapperstats_outputfile),
            procParams,
            os.path.join(mycwd,beadPath),
            filterPath,
            os.path.join(mycwd,alignmentSummaryPath),
            STATUS,
            os.path.join(mycwd,peakOut),
            os.path.join(mycwd,QualityPath),
            os.path.join(mycwd,BaseCallerJsonPath),
            os.path.join(mycwd,'primary.key'),
            os.path.join(mycwd,'uploadStatus'),
            "Completed", True)
        # this will replace the five progress squares with a re-analysis button

    except:
        traceback.print_exc()

    '''
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
    '''

    #TODO, barcode processing
    ########################################################
    # Zip up and move sff, fastq, bam, bai files           #
    # Move zip files to results directory                  #
    ########################################################
    if os.path.exists(DIR_BC_FILES):
        os.chdir(DIR_BC_FILES)
        filestem = "%s_%s" % (env['expName'], env['resultsName'])
        alist = os.listdir('./')
        extlist = ['sff','bam','bai','fastq']
        for ext in extlist:
            filelist = fnmatch.filter(alist, "*." + ext)
            zipname = filestem + '.barcode.' + ext + '.zip'
            for afile in filelist:
                make_zip(zipname, afile)
                os.rename(afile,os.path.join("../",afile))

        # Move zip files up one directory to results directory
        for ext in extlist:
            zipname = filestem + '.barcode.' + ext + '.zip'
            if os.path.isfile(zipname):
                os.rename(zipname,os.path.join("../",zipname))

        os.chdir('../')

    ########################################################
    # Run experimental script                              #
    ########################################################

    #TODO
    if is_blockprocessing:
        printtime('Skipping Plugins')
        return

    printtime('Running Plugins')

    #Plugins will run in reverse alphabetical order order.
    for plugin in sorted(env['plugins'], key=lambda plugin: plugin["name"],reverse=True):
        if plugin != '':
            runPlug = True
            printtime("Plugin %s is enabled" % plugin['name'])

            if not plugin['autorun']:
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

            if not runPlug:
                printtime("     did not run after all.")
                continue    #skip to next plugin

            try:
                #https://iontorrent.jira.com/wiki/display/TS/Plugin+json+file+format
                env['report_root_dir'] = os.getcwd()
                env['analysis_dir'] = os.getcwd()
                env['sigproc_dir'] = os.path.join(env['report_root_dir'],SIGPROC_RESULTS)
                env['basecaller_dir'] = os.path.join(env['report_root_dir'],BASECALLER_RESULTS)
                env['alignment_dir'] = os.path.join(env['report_root_dir'],ALIGNMENT_RESULTS)
                env['testfrag_key'] = 'ATCG'
                printtime("RAWDATA: %s" % env['pathToRaw'])
                start_json = make_plugin_json(env,plugin,env['primary_key'],basefolder,url_root)
                ret = pluginserver.pluginStart(start_json)
            except:
                printtime('plugin %s failed...' % plugin['name'])
                traceback.print_exc()

    return 0

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
                      "Hostname", "PID", "Working Directory", "Job ID",
                      "Job Name", "Analysis Start Time"]
    _MARGINS = 4
    _TABSIZE = 4
    _max_sum = max(map(lambda (a,b): len(a) + len(b), zip(python_data,python_data_labels)))
    _info_width = _max_sum + _MARGINS + _TABSIZE
    print('*'*_info_width)
    for d,l in zip(python_data, python_data_labels):
        spacer = ' '*(_max_sum - (len(l) + len(d)) + _TABSIZE)
        print('* %s%s%s *' % (str(l),spacer,str(d).replace('\n',' ')))
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
#    try:
#        # SHOULD BE iondb.settings.QMASTERHOST, but this always seems to be "localhost"
#        act_qmaster = os.path.join(iondb.settings.SGE_ROOT, iondb.settings.SGE_CELL, 'common', 'act_qmaster')
#        master_node = open(act_qmaster).readline().strip()
#    except IOError:
    master_node = "localhost"

    env['master_node'] = master_node

    env['net_location'] = EXTERNAL_PARAMS.get('net_location','http://' + master_node )
    env['url_path'] = EXTERNAL_PARAMS.get('url_path','output/Home')

    env['blockArgs'] = EXTERNAL_PARAMS.get('blockArgs')

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

    SIGPROC_RESULTS="sigproc_results"
    BASECALLER_RESULTS="basecaller_results"
    ALIGNMENT_RESULTS="alignment_results"
    SIGPROC_RESULTS="./"
    BASECALLER_RESULTS="./"
    ALIGNMENT_RESULTS="./"

    # define entry point
    if env['blockArgs'] == "fromRaw":
        runFromRaw = True
        runFromWells = True
        runFromSFF = True
    elif env['blockArgs'] == "fromWells":
        runFromRaw = False
        runFromWells = True
        runFromSFF = True
        r = subprocess.call(["ln", "-s", os.path.join("../Auto_LOT-184-16530-1.25chip-iemolbio_266_tn_921", SIGPROC_RESULTS), SIGPROC_RESULTS]) # tn
#        r = subprocess.call(["ln", "-s", os.path.join("../Auto_LOT-184-16530-1.25chip-iemolbio_266_920", SIGPROC_RESULTS), SIGPROC_RESULTS])

    elif env['blockArgs'] == "fromSFF":
        runFromRaw = False
        runFromWells = False
        runFromSFF = True
    else:
        runFromRaw = True
        runFromWells = True
        runFromSFF = True

    initreports()
    logout = open("ReportLog.html", "w")
    logout.close()

    sys.stdout.flush()
    sys.stderr.flush()

    try:
        jobserver = xmlrpclib.ServerProxy("http://%s:%d" % (JOBSERVER_HOST, JOBSERVER_PORT), allow_none=True)
    except (socket.error, xmlrpclib.Fault):
        traceback.print_exc()

    try:
        pluginserver = xmlrpclib.ServerProxy("http://%s:%d" % (PLUGINSERVER_HOST, PLUGINSERVER_PORT), allow_none=True)
    except (socket.error, xmlrpclib.Fault):
        traceback.print_exc()


    runFullChip(env)

    #todo jobserver.close() ?

    getExpLogMsgs(env)
    printtime("Run Complete")
    sys.exit(0)
