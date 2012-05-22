#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

"""
Proton on-instrument full chip composite analysis control script
This is a stand-alone script which does not rely on any additional files
from the Ion repository (does not require ion-dbreports package).  It does
require the block-level analysis control script.
"""

__version__ = filter(str.isdigit, "$Revision: 22985 $")

import os
import tempfile

# matplotlib/numpy compatibility
os.environ['HOME'] = tempfile.mkdtemp()
from matplotlib import use
use("Agg")

import ConfigParser
import datetime
import shutil
import socket
import subprocess
import sys
import time
import numpy
import StringIO
import traceback
import json

import drmaa
import pylab
import math
import string
import random
import argparse
import ConfigParser
from subprocess import CalledProcessError

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

def merge(blockDirs, resultsDir):
    '''mergeBaseCallerJson.merge - Combine BaseCaller.json metrics from multiple blocks'''
    
    combinedJson = {'Phasing' : {'CF' : 0,
                                 'IE' : 0,
                                 'DR' : 0},
                    "BeadSummary" : {"lib" : {"badKey"      : 0,
                                              "highPPF"     : 0,
                                              "highRes"     : 0,
                                              "key"         : "TCAG",
                                              "polyclonal"  : 0,
                                              "short"       : 0,
                                              "valid"       : 0,
                                              "zero"        : 0},
                                     "tf" :  {"badKey"      : 0,
                                              "highPPF"     : 0,
                                              "highRes"     : 0,
                                              "key"         : "ATCG",
                                              "polyclonal"  : 0,
                                              "short"       : 0,
                                              "valid"       : 0,
                                              "zero"        : 0 }}}
    numBlocks = 0.0
    
    for dir in blockDirs:
        try:
            file = open(os.path.join(dir,'BaseCaller.json'), 'r')
            blockJson = json.load(file)
            file.close()
            
            blockCF             = blockJson['Phasing']['CF']
            blockIE             = blockJson['Phasing']['IE']
            blockDR             = blockJson['Phasing']['DR']
            
            blockLibbadKey      = blockJson['BeadSummary']['lib']['badKey']
            blockLibhighPPF     = blockJson['BeadSummary']['lib']['highPPF']
            blockLibhighRes     = blockJson['BeadSummary']['lib']['highRes']
            blockLibkey         = blockJson['BeadSummary']['lib']['key']
            blockLibpolyclonal  = blockJson['BeadSummary']['lib']['polyclonal']
            blockLibshort       = blockJson['BeadSummary']['lib']['short']
            blockLibvalid       = blockJson['BeadSummary']['lib']['valid']
            blockLibzero        = blockJson['BeadSummary']['lib']['zero']
            
            blockTFbadKey      = blockJson['BeadSummary']['tf']['badKey']
            blockTFhighPPF     = blockJson['BeadSummary']['tf']['highPPF']
            blockTFhighRes     = blockJson['BeadSummary']['tf']['highRes']
            blockTFkey         = blockJson['BeadSummary']['tf']['key']
            blockTFpolyclonal  = blockJson['BeadSummary']['tf']['polyclonal']
            blockTFshort       = blockJson['BeadSummary']['tf']['short']
            blockTFvalid       = blockJson['BeadSummary']['tf']['valid']
            blockTFzero        = blockJson['BeadSummary']['tf']['zero']

            combinedJson['Phasing']['CF'] += blockCF
            combinedJson['Phasing']['IE'] += blockIE
            combinedJson['Phasing']['DR'] += blockDR
            numBlocks += 1.0
            
            combinedJson['BeadSummary']['lib']['badKey']        += blockLibbadKey
            combinedJson['BeadSummary']['lib']['highPPF']       += blockLibhighPPF
            combinedJson['BeadSummary']['lib']['highRes']       += blockLibhighRes
            combinedJson['BeadSummary']['lib']['key']           = blockLibkey
            combinedJson['BeadSummary']['lib']['polyclonal']    += blockLibpolyclonal
            combinedJson['BeadSummary']['lib']['short']         += blockLibshort
            combinedJson['BeadSummary']['lib']['valid']         += blockLibvalid
            combinedJson['BeadSummary']['lib']['zero']          += blockLibzero

            combinedJson['BeadSummary']['tf']['badKey']        += blockTFbadKey
            combinedJson['BeadSummary']['tf']['highPPF']       += blockTFhighPPF
            combinedJson['BeadSummary']['tf']['highRes']       += blockTFhighRes
            combinedJson['BeadSummary']['tf']['key']           = blockTFkey
            combinedJson['BeadSummary']['tf']['polyclonal']    += blockTFpolyclonal
            combinedJson['BeadSummary']['tf']['short']         += blockTFshort
            combinedJson['BeadSummary']['tf']['valid']         += blockTFvalid
            combinedJson['BeadSummary']['tf']['zero']          += blockTFzero

        except:
            print 'mergeBaseCallerJson: Pass block ' + dir

    if numBlocks > 0:
        combinedJson['Phasing']['CF'] /= numBlocks
        combinedJson['Phasing']['IE'] /= numBlocks
        combinedJson['Phasing']['DR'] /= numBlocks

    file = open(os.path.join(resultsDir,'BaseCaller.json'), 'w')
    file.write(json.dumps(combinedJson,indent=4))
    file.close()
    
def processParametersMerge(ppfilename, verbose):

    process_parameter_file = 'processParameters.txt'


    # Output file
    config_out = ConfigParser.RawConfigParser()
    config_out.optionxform = str # don't convert to lowercase
    config_out.add_section('global')
    
    # Input file
    config_pp = ConfigParser.RawConfigParser()

    if verbose:
        print "Reading", ppfilename

    config_pp.read(ppfilename)
    version = config_pp.get('global','Version')
    build = config_pp.get('global','Build')
    svnrev = config_pp.get('global','SvnRev')
    runid = config_pp.get('global','RunId')
    datadirectory = config_pp.get('global','dataDirectory')
    chip = config_pp.get('global','Chip')
    floworder = config_pp.get('global','flowOrder')
    librarykey = config_pp.get('global','libraryKey')
    cyclesProcessed = config_pp.get('global','cyclesProcessed')
    framesProcessed = config_pp.get('global','framesProcessed')
    
    config_out.set('global','Version',version)
    config_out.set('global','Build',build)
    config_out.set('global','SvnRev',svnrev)
    config_out.set('global','RunId',runid)
    config_out.set('global','dataDirectory',datadirectory)
    config_out.set('global','Chip',chip)
    config_out.set('global','flowOrder',floworder)
    config_out.set('global','libraryKey',librarykey)
    config_out.set('global','cyclesProcessed',cyclesProcessed)
    config_out.set('global','framesProcessed',framesProcessed)
    
    with open(process_parameter_file, 'wb') as configfile:
        if verbose:
            print "Writing", process_parameter_file
        config_out.write(configfile)
        
def main_merge(blockfolder, verbose):

    process_parameter_file = 'processParameters.txt'
    stats_file = 'bfmask.stats'

    # Output file
    config_out = ConfigParser.RawConfigParser()
    config_out.optionxform = str # don't convert to lowercase
    config_out.add_section('global')

    for i,folder in enumerate(blockfolder):

        infile = os.path.join(folder, stats_file)

        if verbose:
            print "Reading", infile

        config = ConfigParser.RawConfigParser()
        config.read(infile)

        keys = ['Total Wells', 'Excluded Wells', 'Empty Wells',
                'Pinned Wells', 'Ignored Wells', 'Bead Wells', 'Dud Beads',
                'Ambiguous Beads', 'Live Beads',
                'Test Fragment Beads', 'Library Beads',
                'TF Filtered Beads (read too short)',
                'TF Filtered Beads (fail keypass)',
                'TF Filtered Beads (too many positive flows)',
                'TF Filtered Beads (poor signal fit)',
                'TF Validated Beads',
                'Lib Filtered Beads (read too short)',
                'Lib Filtered Beads (fail keypass)',
                'Lib Filtered Beads (too many positive flows)',
                'Lib Filtered Beads (poor signal fit)',
                'Lib Validated Beads']

        if i==0:

            config_pp = ConfigParser.RawConfigParser()
            config_pp.read(os.path.join(folder, process_parameter_file))
            chip = config_pp.get('global', 'Chip')
            size = chip.split(',')
            config_out.set('global','Start Row', '0')
            config_out.set('global','Start Column', '0')
            config_out.set('global','Width', int(size[0]))
            config_out.set('global','Height', int(size[1]))

            config_out.set('global','Percent Template-Positive Library Beads', '0') # TODO

            for key in keys:
                config_out.set('global', key, '0')

        for key in keys:
            value_in = config.get('global', key)
            value_out = config_out.get('global', key)
            config_out.set('global', key, int(value_in) + int(value_out))

    with open(stats_file, 'wb') as configfile:
        if verbose:
            print "Writing", stats_file

        config_out.write(configfile)

def merge(folder, infile, out_list, verbose, offset_str):

    infile = os.path.join(folder,infile)

    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(folder, 'processParameters.txt'))
    if offset_str == "use_blocks":
        size = config.get('global', 'Block')
    elif offset_str == "use_analysis_regions":
        size = config.get('global', 'Analysis Region')
    else:
        print "MaskMerge: ERROR: offset string not known"
        sys.exit(1)

    offset = size.split(',')
    offsetx = int(offset[0])
    offsety = int(offset[1])

    if verbose:
        print "MaskMerge: Reading "+str(infile)

    beadlist = numpy.loadtxt(infile, dtype='int', comments='#')


    # ignore block length
    WIDTH=int(beadlist[0,0])
    HEIGHT=int(beadlist[0,1])
    if verbose:
        print "MaskMerge: block size:", WIDTH, HEIGHT

    # add offset to current block data, ignore first column which contains the block size
    beadlist[1:,0]+=offsety
    beadlist[1:,1]+=offsetx

    if verbose:
        print "MaskMerge: Append block with offsets x: "+str(offsetx)+" y: "+str(offsety)

    # remove first element, extend list
    l = beadlist.tolist()
    del l[0]
    out_list.extend(l)

def main_merge(inputfile, blockfolder, outputfile, verbose, offset_str):

    print "MaskMerge: started"

    if verbose:
        print "MaskMerge: in:",inputfile
        print "MaskMerge: out:",outputfile

    out_list = list()

    # add block data to outputfile
    for i,folder in enumerate(blockfolder):

        if i==0:

            config = ConfigParser.RawConfigParser()
            config.read(os.path.join(folder, 'processParameters.txt'))
            chip = config.get('global', 'Chip')
            size = chip.split(',')
            sizex = size[0]
            sizey = size[1]

            if verbose:
                print "MaskMerge: chip size:",sizex,sizey
 
            if verbose:
                print "MaskMerge: write header"
            # open to write file size
            f = open(outputfile, 'w')
            f.write(sizex+" "+sizey+"\n")
            f.close()

        merge(folder,inputfile,out_list,verbose,offset_str)

    if verbose:
        print "MaskMerge: write",outputfile

    # append data
    f_handle = file(outputfile, 'a')
    outdata = numpy.array(out_list)
    numpy.savetxt(f_handle, outdata, fmt='%1.1i')
    f_handle.close()

def isbadblock(blockdir, message):
    if os.path.exists(os.path.join(blockdir,'badblock.txt')):
        printtime("WARNING: %s: skipped %s" % (message,blockdir))
        return True
    return False

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


def parse_log(text):
    """Take the raw text of the experiment log, and return a dictionary
    of the entries contained in the log, parsed into the appropriate
    datatype.
    """
    def filter_non_printable(str):
        if isinstance(str,basestring):
            return ''.join([c for c in str if ord(c) > 31 or ord(c) == 9])
        else:
            return str
    def clean_name(name):
        no_ws = CLEAN_RE.sub("_", name)
        return no_ws.lower()
    def extract_entries(text):
        ret = []
        for line in text.split('\n'):
            match = ENTRY_RE.match(line)
            if match is not None:
                d = match.groupdict()
                ret.append((clean_name(d['name']),d['value'].strip()))
        return ret
    ret = {}
    entries = extract_entries(text)
    for name,value in entries:
        # utf-8 replace code to ensure we don't crash on invalid characters
        #ret[name] = ENTRY_MAP.get(name, lre.text_parse)(value.decode("ascii","ignore"))
        ret[name] = filter_non_printable(ENTRY_MAP.get(name, lre.text_parse)(value.decode("ascii","ignore")))
    #For the oddball repeating keyword: BlockStatus
    #create an array of them.
    ret['blocks'] = []
    for line in text.split('\n'):
        if line.startswith('BlockStatus') or line.startswith('RegionStatus') or line.startswith('TileStatus'):
            ret['blocks'].append (line)
    return ret


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
#        exp_json = json.loads(env['exp_json'])
#        log = json.loads(exp_json['log'])
#        blockstatus = log['blocks']
        blockstatus = env['blocks']
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
    logout.write("<html><pre> \n")
    logout.close()
    cwd = os.getcwd()

    jt = _session.createJobTemplate()

    #SGE
    sge_queue = 'all.q'
    if is_thumbnail:
        sge_queue = 'thumbnail.q'
    printtime("Use "+ sge_queue)
#    jt.nativeSpecification = "-pe ion_pe 1 -q " + sge_queue
#   jt.nativeSpecification = "-pe ion_pe 1 -q " + sge_queue + " -l h_vmem=10000M"
    jt.nativeSpecification = ""

    #TORQUE
    #jt.nativeSpecification = ""

    jt.remoteCommand = "python"
    jt.workingDirectory = os.path.join(cwd, rpath)
    jt.outputPath = ":" + os.path.join(cwd, out_path)
    jt.errorPath = ":" + os.path.join(cwd, err_path)
    if is_wholechip:
        jt.args = ['localBLanalysis.py', 'ion_params_00.json']
    elif is_thumbnail:
        jt.args = ['localBLanalysis.py', 'ion_params_00.json']
    else:
        jt.args = ['../localBLanalysis.py', 'ion_params_00.json']
    jt.joinFiles = False
    jobid = _session.runJob(jt)
    _session.deleteJobTemplate(jt)

    return jobid

def submitBlockJob(blockObj, env):

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
        TMP_PARAMS["pathToData"] = os.path.join(TMP_PARAMS["pathToData"], blockObj['id_str'])

    TMP_PARAMS["analysisArgs"] = ' '.join(blockObj['jobcmd'])
    file_out = open("%s/ion_params_00.json" % resultDir, 'w')
    json.dump(TMP_PARAMS, file_out)
    file_out.close()

    jobid = spawn_cluster_job(resultDir)

    return jobid


def runFullChip(env):
    STATUS = None
#    basefolder = 'plugin_out'
#    if not os.path.isdir(basefolder):
#        os.umask(0000)   #grant write permission to plugin user
#        os.mkdir(basefolder)
#        os.umask(0002)

    libsff = "%s/%s_%s.sff" % (BASECALLER_RESULTS, env['expName'], env['resultsName'])
    tfsff = "%s/%s_%s.tf.sff" % (BASECALLER_RESULTS, env['expName'], env['resultsName'])
    fastqpath = "%s/%s_%s.fastq" % (BASECALLER_RESULTS, env['expName'], env['resultsName'])
    libKeyArg = "--libraryKey=%s" % env["libraryKey"]

#    launcher = PluginRunner()

    #-------------------------------------------------------------
    # Gridded data processing
    #-------------------------------------------------------------
#    write_version()

    printtime("RUNNING FULL CHIP MULTI-BLOCK ANALYSIS")
    # List of block objects to analyze
    blocks = GetBlocksToAnalyze(env)
    dirs = ['block_%s' % block['id_str'] for block in blocks]
#    dirs = ['%s/block_%s' % (SIGPROC_RESULTS, block['id_str']) for block in blocks]

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

        # Launch multiple Block analysis jobs
        for block in blocks:
            block['jobid'] = submitBlockJob(block,env)
            printtime("Submitted block (%s) analysis job with job ID (%s)" % (block['id_str'], str(block['jobid'])))

        job_list=[block['jobid'] for block in blocks]

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
                block['status'] = _session.jobStatus(block['jobid'])
                if block['status']==drmaa.JobState.DONE or block['status']==drmaa.JobState.FAILED:
                    printtime("Job %s has ended with status %s" % (str(block['jobid']),block['status']))
                    job_list.remove(block['jobid'])
#                else:
#                    printtime("Job %s has status %s" % (str(block['jobid']),block['status']))

#                # Hack
#                if is_thumbnail and not pl_started:
#                    f = open('progress.txt')
#                    text = f.read()
#                    f.close()       
#                    matches = re.findall(r"wellfinding = green", text)
#
#                    if len(matches) != 0:
#                        pl_started = True
#                        for plugin in sorted(env['plugins'], key=lambda plugin: plugin["name"],reverse=True):
#                            if plugin['name'] == 'rawPlots' or plugin['name'] == 'separator' or plugin['name'] == 'chipNoise':
#                                runPlug = True
#                                printtime("Plugin %s is enabled" % plugin['name'])
#
#                                try:
#                                    #https://iontorrent.jira.com/wiki/display/TS/Plugin+json+file+format
#                                    env['report_root_dir'] = os.getcwd()
#                                    env['analysis_dir'] = os.path.join(env['report_root_dir'],SIGPROC_RESULTS)
#                                    env['basecaller_dir'] = os.path.join(env['report_root_dir'],BASECALLER_RESULTS)
#                                    env['alignment_dir'] = os.path.join(env['report_root_dir'],ALIGNMENT_RESULTS)
#                                    env['testfrag_key'] = 'ATCG'
#                                    printtime("RAWDATA: %s" % env['pathToRaw'])
#                                    start_json = make_plugin_json(env,plugin,env['primary_key'],basefolder,url_root)
#                                    ret = launcher.callPluginXMLRPC(start_json, iondb.settings.IPLUGIN_HOST, iondb.settings.IPLUGIN_PORT)
#                                    printtime('plugin %s started ...' % plugin['name'])
#                                except:
#                                    printtime('plugin %s failed...' % plugin['name'])
#                                    traceback.print_exc()

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

    #--------------------------------------------------------
    # Start merging results files
    #--------------------------------------------------------
#    if is_thumbnail or is_wholechip:
#        printtime("MERGING: THUMBNAIL OR 31X - skipping merge process.")
#
#        res = models.Results.objects.get(pk=env['primary_key'])
#        res.metaData["thumb"] = 1
#        #res.timeStamp = datetime.datetime.now()
#        res.save()
#        printtime("thumbnail: "+str(res.metaData["thumb"]))

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
                    processParametersMerge(ppfile,True)
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
                subdir = os.path.join(BASECALLER_RESULTS,subdir)
                if isbadblock(subdir, "Merging quality.summary"):
                    continue
                summaryfile=os.path.join(subdir, 'quality.summary')
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
            printtime("Calling TFMapper")

            block_tfsff = "rawtf.sff"
            block_libsff = "rawlib.sff"

            try:
                cmd = "TFMapper -m 0 --logfile TFMapper.log --cafie-metrics"
                cmd += " --output-dir=%s" % (BASECALLER_RESULTS)
                cmd += " --wells-dir=%s" % (SIGPROC_RESULTS)
                cmd += " --sff-dir=%s" % (BASECALLER_RESULTS)
                cmd += " --tfkey=%s" % (tfKey)
                cmd += " --libkey=%s" % (libKey)
                cmd += " %s" % (block_tfsff)
                for subdir in dirs:
                    _subdir = os.path.join(BASECALLER_RESULTS,subdir)
                    if isbadblock(_subdir, "TFMapper tf files"):
                        continue
                    rawtfsff = os.path.join(_subdir,block_tfsff)
                    if os.path.exists(rawtfsff):
                        cmd = cmd + ' %s' % subdir
                    else:
                        printtime("ERROR: skipped %s" % rawtfsff)
                cmd = cmd + " > %s" % tfmapperstats_outputfile
                printtime("DEBUG: Calling '%s'" % cmd)
                os.system(cmd)
            except:
                printtime("ERROR: TFMapper failed")

            ########################################################
            #generate the TF Metrics including plots               #
            ########################################################
#            printtime("generate the TF Metrics including plots")
#
#            tfMetrics = None
#            if os.path.exists(tfmapperstats_outputfile):
#                try:
#                    # Q17 TF Read Length Plot
#                    tfMetrics = parseTFstats.generateMetricsData(tfmapperstats_outputfile)
#                    tfGraphs.Q17(tfMetrics)
#                    tfGraphs.genCafieIonograms(tfMetrics,floworder)
#                except Exception:
#                    printtime("ERROR: Metrics Gen Failed")
#                    traceback.print_exc()
#            else:
#                printtime("ERROR: %s doesn't exist" % tfmapperstats_outputfile)
#                tfMetrics = None

            ###############################################
            # Merge individual block bead stats files     #
            ###############################################
#            printtime("Merging bfmask.stats files")
#
#            try:
#                bfmaskstatsfiles = []
#                for subdir in dirs:
#                    subdir = os.path.join(SIGPROC_RESULTS,subdir)
#                    if isbadblock(subdir, "Merging bfmask.stats files"):
#                        continue
#                    bfmaskstats = os.path.join(subdir,'bfmask.stats')
#                    if os.path.exists(bfmaskstats):
#                        bfmaskstatsfiles.append(subdir)
#                    else:
#                        printtime("ERROR: Merging bfmask.stats files: skipped %s" % bfmaskstats)
#
#                StatsMerge.main_merge(bfmaskstatsfiles, True)
#                #TODO
#                shutil.move('bfmask.stats', SIGPROC_RESULTS)
#            except:
#                printtime("No bfmask.stats files were found to merge")

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
                cmd = 'SFFProtonMerge'
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
                printtime("SFFProtonMerge failed (library)")

            printtime("Merging Test Fragment SFF files")
            try:
                cmd = 'SFFProtonMerge'
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
                printtime("SFFProtonMerge failed (test fragments)")


        ########################################################
        #Make Bead Density Plots                               #
        ########################################################
#        printtime("Make Bead Density Plots (composite report)")
#
#        bfmaskPath = os.path.join(SIGPROC_RESULTS,'bfmask.bin')
#        maskpath = os.path.join(SIGPROC_RESULTS,'MaskBead.mask')
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
#            printtime("Warning: %s doesn't exists." % bfmaskPath)
#
#        if os.path.exists(maskpath):
#            try:
#                # Makes Bead_density_contour.png
#                beadDensityPlot.genHeatmap(maskpath, BASECALLER_RESULTS) # todo, takes too much time
#      #          os.remove(maskpath)
#            except:
#                traceback.print_exc()
#        else:
#            printtime("Warning: no MaskBead.mask file exists.")

        ##################################################
        # Create zip of files
        ##################################################
#
#        #sampled sff
#        make_zip(libsff.replace(".sff",".sampled.sff")+'.zip', libsff.replace(".sff",".sampled.sff"))
#
#        #library sff
#        make_zip(libsff + '.zip', libsff )
#
#        #tf sff
#        make_zip(tfsff + '.zip', tfsff)
#
#        #fastq zip
#        make_zip(fastqpath + '.zip', fastqpath)
#
#        #sampled fastq
#        make_zip(fastqpath.replace(".fastq",".sampled.fastq")+'.zip', fastqpath.replace(".fastq",".sampled.fastq"))
#
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
                cmd = 'java -Xmx8g -jar /opt/picard/picard-tools-current/MergeSamFiles.jar'
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
                cmd = cmd + ' CREATE_INDEX=true'
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

    return 0

if __name__=="__main__":
    #
    # This string will be appended verbatim to the Analysis binary invocation
    #
    ANALYSIS_CMD_OPTIONS = ' -k off --ppf-filter off --cr-filter off --clonal-filter-solve off --beadfind-minlivesnr 4 --beadfind-lagone-filt 1 --gopt=/opt/ion/config/gopt_316.param'

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
    
    #--------------------------------------------
    # cmd line arg parsing
    #--------------------------------------------
    if len(sys.argv) != 2:
        print
        print "Usage:"
        print
        print "%s <full path to raw data directory>" % sys.argv[0]
        print
        sys.exit(0)
    else:
        top_level_raw_dir = sys.argv[1]
        
    if not os.path.exists(top_level_raw_dir):
        print "ERROR: invalid path - %s" % top_level_raw_dir
        sys.exit(0)
    
    REPORT_ARCHIVE = "/results/local_analysis"
    if not os.path.exists(REPORT_ARCHIVE):
        print "ERROR: invalid report archive path - %s" % REPORT_ARCHIVE
        sys.exit(0)
    
    #--------------------------------------------
    # Define 'env' variable from data parsed from explog.txt
    #--------------------------------------------
    env = {}
    try:
        file_in=open(os.path.join(top_level_raw_dir,'explog.txt'),'r')
        #explog = file_in.readlines()
    except:
        print traceback.format_exc()
        sys.exit(0)

    keywords={
        'Project':'project',
        'Sample':'sample',
        'ChipType':'chiptype',
        'Library':'libraryName',
        'Experiment Name':'expName',
        'LibraryKeySequence':'libraryKey',
        'Image Map':'flowOrder',
        'Flows':'flows'}
    
    for keyword in keywords:
        for line in file_in:
            if keyword == line.split(":")[0].strip():
                arg = line.split(":")[1].strip()
                env[keywords[keyword]] = arg.strip()
        file_in.seek(0)
        
    env['blocks'] = []
    for line in file_in:
        if line.startswith('BlockStatus'):
            env['blocks'].append (line)
    file_in.seek(0)
    
    env['pathToRaw']    = top_level_raw_dir
    env['pathToData']   = top_level_raw_dir
    env['align_full']   = True
    env['analysisArgs'] = "Analysis %s %s" % (ANALYSIS_CMD_OPTIONS,top_level_raw_dir)
    env['exp_json']     = json.dumps(file_in.read(),indent=4)
    env['resultsName']  = "comp_%s_%s" % (env['expName'],''.join(random.choice(string.letters+string.digits) for i in xrange(6)))
    env['flowOrder']    = env['flowOrder'].upper()
    env['sfftrim']      = True
    env['site_name']    = "Proton Instrument"
    env['reverse_primer_dict']  = {'name':'Ion Kit',
                               'sequence':'ATCACCGACTGCCCATAGAGAGGCTGAGAC',
                               'qual_cutoff':9,
                               'qual_window':30,
                               'adapter_cutoff':16
                               }
    if '' == env['libraryKey']:
        env['libraryKey'] = 'TCAG'
    env['blockArgs'] = ''
        
    file_in.close()
    
    # Create a Report directory
    report_dir = os.path.join(REPORT_ARCHIVE,env['resultsName'])
    os.mkdir(report_dir,0775)
    
    # Copy files to Report directory
    if os.path.isfile('localBLanalysis.py'):
        shutil.copyfile('localBLanalysis.py',os.path.join(report_dir,'localBLanalysis.py'))
    else:
        print "ERROR: localBLanalysis.py file is not in current directory"
        sys.exit(0)
        
    os.chdir(report_dir)
    
    #--------------------------------------------
    # Create ersatz json params file
    #--------------------------------------------
    file_out=open("ion_params_00.json", 'w')
    contents = json.dumps(env)
    file_out.write(contents)
    file_out.close()

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
    elif env['blockArgs'] == "fromSFF":
        runFromRaw = False
        runFromWells = False
        runFromSFF = True
    else:
        runFromRaw = True
        runFromWells = True
        runFromSFF = True
    
    #--------------------------------------------
    # DRMAA initialization
    #--------------------------------------------
    try:
        os.environ["SGE_ROOT"] = '/var/lib/gridengine'
        os.environ["SGE_CELL"] = 'iontorrent'
        os.environ["SGE_CLUSTER_NAME"] = 'p6444'
        os.environ["SGE_QMASTER_PORT"] = '6444'
        os.environ["SGE_EXECD_PORT"] = '6445'
        os.environ["DRMAA_LIBRARY_PATH"] = '/usr/lib/libdrmaa.so.1.0'
        _session = drmaa.Session()
        _session.initialize()
        print 'drmaa session initialized'
    except:
        print "Unexpected error:", sys.exc_info()
        sys.exit(1)
        
    #--------------------------------------------
    # Start the analysis pipeline
    #--------------------------------------------
    is_blockprocessing = True
    is_wholechip = False
    is_thumbnail = False
    if not runFullChip (env):
        print "Processing completed successfully"
    else:
        print "There were errors during processing"
    
    #--------------------------------------------
    # Clean up
    #--------------------------------------------
    _session.exit()
    print 'drmaa session exit'
    
    sys.exit(0)
