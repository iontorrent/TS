#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision: 17459 $")

import os

# import /etc/torrentserver/cluster_settings.py, provides PLUGINSERVER_HOST, PLUGINSERVER_PORT
import sys
sys.path.append('/etc')
from torrentserver.cluster_settings import *

import ConfigParser
import StringIO
import datetime
import shutil
import socket
import xmlrpclib
import subprocess
import time
import os.path
from collections import deque

from ion.plugin.remote import call_launchPluginsXMLRPC
from ion.plugin.constants import RunLevel, RunType

from ion.reports.plotters import *
from ion.utils.aggregate_alignment import *

sys.path.append('/opt/ion/')

import traceback
import json


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

def write_version():
    a = subprocess.Popen('ion_versionCheck.py --ion', shell=True, stdout=subprocess.PIPE)
    ret = a.stdout.readlines()
    f = open('version.txt','w')
    for i in ret[:len(ret)-1]:
#    for i in ret:
        f.write(i)
    f.close()

def parse_metrics(fileIn):
    """Takes a text file where a '=' is the delimter 
    in a key value pair and return a python dict of those values """
    
    f = open(fileIn, 'r')
    data = f.readlines()
    f.close()
    ret = {}
    for line in data:
        l = line.strip().split('=')
        key = l[0].strip()
        value = l[-1].strip()
        ret[key]=value
    return ret

def initTLReport(basefolder):
    if not os.path.isdir(basefolder):
        oldmask = os.umask(0000)   #grant write permission to plugin user
        os.mkdir(basefolder)
        os.umask(oldmask)

    # Begin report writing
    os.umask(0002)
    #TMPL_DIR = os.path.join(distutils.sysconfig.get_python_lib(),'ion/web/db/writers')
    TMPL_DIR = '/usr/share/ion/web/db/writers'
    templates = [
        # DIRECTORY, SOURCE_FILE, DEST_FILE or None for same as SOURCE
        (TMPL_DIR, "report_layout.json", None),
        (TMPL_DIR, "parsefiles.php", None),
        (TMPL_DIR, "format_whole.php", "Default_Report.php",), ## Renamed during copy
        #(os.path.join(distutils.sysconfig.get_python_lib(), 'ion', 'reports',  "BlockTLScript.py", None)
    ]
    for (d,s,f) in templates:
        if not f: f=s
        # If owner is different copy fails - unless file is removed first
        if os.access(f, os.F_OK):
            os.remove(f)
        shutil.copy(os.path.join(d,s), f)

def initBlockReport(blockObj,SIGPROC_RESULTS,BASECALLER_RESULTS,ALIGNMENT_RESULTS,pluginbasefolder,from_sigproc_analysis=False):
    """Performs initialization for both report writing and background report generation."""

    printtime("INIT BLOCK REPORT")
    
    if blockObj['id_str'] == "wholechip":
        resultDir = "."
    elif blockObj['id_str'] == "thumbnail":
        resultDir = "."
    else:
        resultDir = '%s%s' % ('block_', blockObj['id_str'])

        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
        
        block_pluginbasefolder = os.path.join(resultDir,pluginbasefolder)    
        if not os.path.isdir(block_pluginbasefolder):
            oldmask = os.umask(0000)   #grant write permission to plugin user
            os.mkdir(block_pluginbasefolder)
            os.umask(oldmask)  

        _SIGPROC_RESULTS = os.path.join(SIGPROC_RESULTS, resultDir)
        _BASECALLER_RESULTS = os.path.join(BASECALLER_RESULTS, resultDir)
        _ALIGNMENT_RESULTS = os.path.join(ALIGNMENT_RESULTS, resultDir)

        if from_sigproc_analysis:
            if not os.path.isdir(_SIGPROC_RESULTS):
                try:
                    os.mkdir(_SIGPROC_RESULTS)
                except:
                    traceback.print_exc()

        if not os.path.isdir(_BASECALLER_RESULTS):
            try:
                os.mkdir(_BASECALLER_RESULTS)
            except:
                traceback.print_exc()

        
        try:
          os.symlink(os.path.join("..",_SIGPROC_RESULTS), os.path.join(resultDir,SIGPROC_RESULTS))
          os.symlink(os.path.join("..",_BASECALLER_RESULTS), os.path.join(resultDir,BASECALLER_RESULTS))
#        os.symlink(os.path.join("..",_ALIGNMENT_RESULTS), os.path.join(resultDir,ALIGNMENT_RESULTS))
        except:
          printtime("couldn't create symbolic link")

        file_in = open("ion_params_00.json", 'r')
        TMP_PARAMS = json.loads(file_in.read())
        file_in.close()

        # update path to data
        TMP_PARAMS["pathToData"] = os.path.join(TMP_PARAMS["pathToData"], blockObj['datasubdir'])
        TMP_PARAMS["mark_duplicates"] = False

        #write block specific ion_params_00.json
        file_out = open("%s/ion_params_00.json" % resultDir, 'w')
        json.dump(TMP_PARAMS, file_out)
        file_out.close()

        cwd = os.getcwd()
        try:
            os.symlink(os.path.join(cwd,"Default_Report.php"), os.path.join(resultDir,"Default_Report.php"))
        except:
            printtime("couldn't create symbolic link")

    return resultDir


def create_index_file(composite_bam_filepath, composite_bai_filepath):
    try:
        cmd = 'samtools index %s %s' % (composite_bam_filepath,composite_bai_filepath)
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("index creation failed: %s %s" % (composite_bam_filepath, composite_bai_filepath))
        traceback.print_exc()


def isbadblock(blockdir, message):
    # process blockstatus.txt
    # printtime("WARNING: %s: skipped %s" % (message,blockdir))
    return False


def get_datasets_basecaller(BASECALLER_RESULTS):
    datasets_basecaller_path = os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json")

    if not os.path.exists(datasets_basecaller_path):
        printtime("ERROR: %s does not exist" % datasets_basecaller_path)
        raise Exception("ERROR: %s does not exist" % datasets_basecaller_path)

    datasets_basecaller = {}
    try:
        f = open(datasets_basecaller_path,'r')
        datasets_basecaller = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % datasets_basecaller_path)
        raise Exception("ERROR: problem parsing %s" % datasets_basecaller_path)
    return datasets_basecaller


def printheader():
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
                      "Job Name", "Start Time"]
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

def get_plugins_to_run(plugins, report_type):
    """ Sort out runtype and runlevel of each plugin and return plugins appropriate for this analysis """
    blocklevel = False
    plugins_to_run = {}
    printtime("Gettings plugins to run, report type = %s" % report_type)    
    for name in plugins.keys():
        plugin = plugins[name]
        
        # default is run on wholechip and thumbnail, but not composite
        selected = report_type in [RunType.FULLCHIP, RunType.THUMB]          
        if plugin.get('runtype',''):
            selected = (report_type in plugin['runtype'])
    
        # TODO hardcoded specials
        if (report_type == RunType.COMPOSITE) and (plugin['name'] in ['torrentscout','contourPlots','seq_dependent_errors']):
            selected = True
        if (report_type == RunType.THUMB) and (plugin['name'] in ['rawPlots', 'separator', 'chipNoise']):
            plugin['runlevel'] = [RunLevel.SEPARATOR, RunLevel.DEFAULT]
        
        if selected:            
            plugin['runlevel'] = plugin.get('runlevel') if plugin.get('runlevel') else [RunLevel.DEFAULT]
            printtime("Plugin %s is enabled, runlevels=%s" % (plugin['name'],','.join(plugin['runlevel'])))
            plugins_to_run[name] = plugin
  
            # check if have any blocklevel plugins        
            if report_type == RunType.COMPOSITE and RunLevel.BLOCK in plugin['runlevel']:
                blocklevel = True
  
    return plugins_to_run, blocklevel


def runplugins(plugins, env, level = RunLevel.DEFAULT, params={}):
    printtime("Starting plugins runlevel=%s" % level )
    params.setdefault('run_mode', 'pipeline') ## Plugins launched here come from pipeline
    try:
        pluginserver = xmlrpclib.ServerProxy("http://%s:%d" % (PLUGINSERVER_HOST, PLUGINSERVER_PORT), allow_none=True)
        # call ionPlugin xmlrpc function to launch selected plugins
        # note that dependency plugins may be added to the plugins dict
        plugins, msg = call_launchPluginsXMLRPC(env['primary_key'], plugins, env['net_location'], env['username'], level, params, pluginserver)
        print msg
    except:
        traceback.print_exc()

    return plugins  


def merge_bam_files(bamfilelist,composite_bam_filepath,composite_bai_filepath,mark_duplicates,method="samtools"):

    if method=='samtools':
        merge_bam_files_samtools(bamfilelist,composite_bam_filepath,composite_bai_filepath,mark_duplicates)

    if method=='picard':
        merge_bam_files_picard(bamfilelist,composite_bam_filepath,composite_bai_filepath,mark_duplicates)


def merge_bam_files_samtools(bamfilelist,composite_bam_filepath,composite_bai_filepath,mark_duplicates):

    try:
        for bamfile in bamfilelist:
            cmd = 'samtools view -H %s > %s.header.sam' % (bamfile,bamfile,)
            printtime("DEBUG: Calling '%s'" % cmd)
            subprocess.call(cmd,shell=True)

        cmd = 'java -Xmx8g -jar /opt/picard/picard-tools-current/MergeSamFiles.jar'
        for bamfile in bamfilelist:
            cmd = cmd + ' I=%s.header.sam' % bamfile
        cmd = cmd + ' O=%s.header.sam' % (composite_bam_filepath)
        cmd = cmd + ' VERBOSITY=WARNING' # suppress INFO on stderr
        cmd = cmd + ' QUIET=true' # suppress job-summary on stderr
        cmd = cmd + ' VALIDATION_STRINGENCY=SILENT'
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)

        if len(bamfilelist) == 1:
            # Usage: samtools reheader <in.header.sam> <in.bam>
            if mark_duplicates:
                cmd = 'samtools reheader %s.header.sam %s' % (composite_bam_filepath, bamfilelist[0])
            else:
                cmd = 'samtools reheader %s.header.sam %s > %s' % (composite_bam_filepath, bamfilelist[0], composite_bam_filepath)
        else:
            # Usage: samtools merge [-nr] [-h inh.sam] <out.bam> <in1.bam> <in2.bam> [...]
            cmd = 'samtools merge -l1 -p8'
            if mark_duplicates:
                cmd += ' - '
            else:
                cmd += ' %s' % (composite_bam_filepath)
            for bamfile in bamfilelist:
                cmd += ' %s' % bamfile
            cmd += ' -h %s.header.sam' % composite_bam_filepath

        if mark_duplicates:
            json_name = ('BamDuplicates.%s.json')%(os.path.normpath(composite_bam_filepath)) if os.path.normpath(composite_bam_filepath)!='rawlib.bam' else 'BamDuplicates.json'
            cmd += ' | BamDuplicates -i stdin -o %s -j %s' % (composite_bam_filepath, json_name)
        
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)

        if composite_bai_filepath:
            create_index_file(composite_bam_filepath, composite_bai_filepath)
    except:
        printtime("bam file merge failed")
        traceback.print_exc()
        return 1

def merge_bam_files_picard(bamfilelist,composite_bam_filepath,composite_bai_filepath,mark_duplicates):

    try:
#        cmd = 'picard-tools MergeSamFiles'
        if mark_duplicates:
            cmd = 'java -Xmx8g -jar /usr/local/bin/MarkDuplicates.jar M=%s.markduplicates.metrics.txt' % composite_bam_filepath
        else:
            cmd = 'java -Xmx8g -jar /opt/picard/picard-tools-current/MergeSamFiles.jar'

        for bamfile in bamfilelist:
            cmd = cmd + ' I=%s' % bamfile
        cmd = cmd + ' O=%s' % (composite_bam_filepath)
        cmd = cmd + ' ASSUME_SORTED=true'
        if composite_bai_filepath:
            cmd = cmd + ' CREATE_INDEX=true'
        cmd = cmd + ' USE_THREADING=true'
        cmd = cmd + ' VERBOSITY=WARNING' # suppress INFO on stderr
        cmd = cmd + ' QUIET=true' # suppress job-summary on stderr
        cmd = cmd + ' VALIDATION_STRINGENCY=SILENT'
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("bam file merge failed")
        traceback.print_exc()
        return 1

    try:
        if composite_bai_filepath:
            if not os.path.exists(composite_bai_filepath):
                # picard is using .bai , we want .bam.bai
                srcbaifilepath = composite_bam_filepath.replace(".bam",".bai")
                if os.path.exists(srcbaifilepath):
                    os.rename(srcbaifilepath, composite_bai_filepath)
                else:
                    printtime("ERROR: %s doesn't exists" % srcbaifilepath)
    except:
        traceback.print_exc()
        return 1

def remove_unneeded_block_files(blockdirs):
    return
    for blockdir in blockdirs:
        try:
            bamfile = os.path.join(blockdir,'basecaller_results','rawlib.basecaller.bam')
            if os.path.exists(bamfile):
                os.remove(bamfile)

            recalibration_dir = os.path.join(blockdir,'basecaller_results','recalibration')
            shutil.rmtree(recalibration_dir, ignore_errors=True)
        except:
            printtime("remove unneeded block files failed")
            traceback.print_exc()

def bam2fastq_command(BAMName,FASTQName):
    com = "java -Xmx8g -jar /opt/picard/picard-tools-current/SamToFastq.jar"
    com += " I=%s" % BAMName
    com += " F=%s" % FASTQName
    return com

def merge_raw_key_signals(filelist,composite_file):

    mergedKeyPeak = {}
    mergedKeyPeak['Test Fragment'] = 0
    mergedKeyPeak['Library'] = 0

    N = 0
    merged_key_signal_sum = 0
    for xfile in filelist:
        try:
            keyPeak = parse_metrics(xfile)
            library_key_signal = int(keyPeak['Library'])
            merged_key_signal_sum += library_key_signal
            N += 1
        except:
            printtime(traceback.format_exc())
            continue
    if N > 0:
        mergedKeyPeak['Library'] = merged_key_signal_sum/N

    try:
        f = open(composite_file,'w')
        f.write('Test Fragment = %s\n' % mergedKeyPeak['Test Fragment'])
        f.write('Library = %s\n' % mergedKeyPeak['Library'])
        f.close()
    except:
        printtime(traceback.format_exc())

    return 0
