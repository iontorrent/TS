#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision: 17459 $")

import os
import tempfile

# matplotlib/numpy compatibility
os.environ['HOME'] = tempfile.mkdtemp()
from matplotlib import use
use("Agg")

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
import sys
import time
from collections import deque

from ion.utils.plugin_json import *

# Import analysis-specific packages. We try to follow the from A import B
# convention when B is a module in package A, or import A when
# A is a module.
#from ion.analysis import cafie,sigproc
#from ion.fileformats import sff
#from ion.reports import tfGraphs, parseTFstats
from ion.reports import blast_to_ionogram, plotKey, \
    parseBeadfind, parseProcessParams, beadDensityPlot, \
    libGraphs, beadHistogram, plotRawRegions, trimmedReadLenHisto


from ion.reports.plotters import *
from ion.utils.aggregate_alignment import *

sys.path.append('/opt/ion/')

import math
from scipy import cluster, signal, linalg
import traceback
import json

import dateutil.parser

class MyConfigParser(ConfigParser.RawConfigParser):
    def read(self, filename):
        try:
            text = open(filename).read()
        except IOError:
            pass
        else:
            afile = StringIO.StringIO("[global]\n" + text)
            self.readfp(afile, filename)

def add_status(process, status, message=""):
    f = open("blockstatus.txt", 'a')
    f.write(process+"="+str(status)+" "+str(message)+"\n")
    f.close()


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

def initreports(SIGPROC_RESULTS, BASECALLER_RESULTS, ALIGNMENT_RESULTS):
    """Performs initialization for both report writing and background
    report generation."""
    printtime("INIT REPORTS")
    #
    # Begin report writing
    #

    basefolder = 'plugin_out'
    if not os.path.isdir(basefolder):
        oldmask = os.umask(0000)   #grant write permission to plugin user
        os.mkdir(basefolder)
        os.umask(oldmask)

    if not os.path.isdir(SIGPROC_RESULTS):
        try:
            os.mkdir(SIGPROC_RESULTS)
        except:
            traceback.print_exc()

    if not os.path.isdir(BASECALLER_RESULTS):
        try:
            os.mkdir(BASECALLER_RESULTS)
        except:
            traceback.print_exc()

    if not os.path.isdir(ALIGNMENT_RESULTS):
        try:
            os.mkdir(ALIGNMENT_RESULTS)
        except:
            traceback.print_exc()

    # create symbolic links for merge process
    mycwd=os.path.basename(os.getcwd())
    if "block_" in mycwd:

        _SIGPROC_RESULTS = os.path.join('..', SIGPROC_RESULTS, mycwd)
        _BASECALLER_RESULTS = os.path.join('..', BASECALLER_RESULTS, mycwd)
        _ALIGNMENT_RESULTS = os.path.join('..', ALIGNMENT_RESULTS, mycwd)

        r = subprocess.call(["ln", "-s", os.path.join('..', mycwd, SIGPROC_RESULTS), _SIGPROC_RESULTS])
        r = subprocess.call(["ln", "-s", os.path.join('..', mycwd, BASECALLER_RESULTS), _BASECALLER_RESULTS])
        r = subprocess.call(["ln", "-s", os.path.join('..', mycwd, ALIGNMENT_RESULTS), _ALIGNMENT_RESULTS])

    # Begin report writing
    os.umask(0002)
    #TMPL_DIR = os.path.join(distutils.sysconfig.get_python_lib(),'ion/web/db/writers')
    TMPL_DIR = '/usr/share/ion/web/db/writers'
    templates = [
        # DIRECTORY, SOURCE_FILE, DEST_FILE or None for same as SOURCE
        (TMPL_DIR, "report_layout.json", None),
        (TMPL_DIR, "parsefiles.php", None),
        (TMPL_DIR, "log.html", None),
        (TMPL_DIR, "alignment_summary.html", os.path.join(ALIGNMENT_RESULTS,"alignment_summary.html")),
        (TMPL_DIR, "csa.php", None),
        (TMPL_DIR, "format_whole.php", "Default_Report.php",), ## Renamed during copy
        #(os.path.join(distutils.sysconfig.get_python_lib(), 'ion', 'reports',  "BlockTLScript.py", None)
    ]
    for (d,s,f) in templates:
        if not f: f=s
        # If owner is different copy fails - unless file is removed first
        if os.access(f, os.F_OK):
            os.remove(f)
        shutil.copy(os.path.join(d,s), f)

def isbadblock(blockdir, message):
    if os.path.exists(os.path.join(blockdir,'badblock.txt')):
        printtime("WARNING: %s: skipped %s" % (message,blockdir))
        return True
    return False

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


def getparameter(parameterfile=None):

    #####################################################################
    # Load the analysis parameters and metadata from a json file passed in on the
    # command line with --params=<json file name>
    # we expect this loop to iterate only once. This is more elegant than
    # trying to index into ARGUMENTS.
    #####################################################################
    EXTERNAL_PARAMS = {}
    env = {}
    if parameterfile:
        env["params_file"] = parameterfile
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

    #get the experiment json data
    env['exp_json'] = EXTERNAL_PARAMS.get('exp_json')

    env['pgmName'] = EXTERNAL_PARAMS.get('pgmName','unknownPGM')

    #this will get the exp data from the database
    exp_json = json.loads(env['exp_json'])
    exp_log_json = json.loads(exp_json['log'])

    env['flows'] = EXTERNAL_PARAMS.get('flows')
    env['notes'] = exp_json['notes']
    env['start_time'] = exp_log_json['start_time']

    env['blockArgs'] = EXTERNAL_PARAMS.get('blockArgs')

    # get command line args
    env['analysisArgs'] = EXTERNAL_PARAMS.get("analysisArgs")

    env['basecallerArgs'] = EXTERNAL_PARAMS.get("basecallerArgs","")

    #previousReports
    env['previousReport'] = EXTERNAL_PARAMS.get("previousReport","")

    # this is the library name for the run taken from the library field in the database
    env["libraryName"] = EXTERNAL_PARAMS.get("libraryName", "none")
    if env["libraryName"]=="":
        env["libraryName"] = "none"
    dtnow = datetime.datetime.now()
    # the time at which the analysis was started, mostly for debugging purposes
    env["report_start_time"] = dtnow.strftime("%c")
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
    #plan
    env['plan'] = EXTERNAL_PARAMS.get('plan', {})
    # skipChecksum?
    env['skipchecksum'] = EXTERNAL_PARAMS.get('skipchecksum',False)
    # Do Full Align?
    env['align_full'] = EXTERNAL_PARAMS.get('align_full')
    # Check to see if a SFFTrim should be done
    env['sfftrim'] = EXTERNAL_PARAMS.get('sfftrim')
    # Get SFFTrim args
    env['sfftrim_args'] = EXTERNAL_PARAMS.get('sfftrim_args')

    env['flowOrder'] = EXTERNAL_PARAMS.get('flowOrder',"0").strip()
    # If flow order is missing, assume classic flow order:
    if env['flowOrder'] == "0":
        env['flowOrder'] = "TACG"
        printtime("ERROR: floworder redefine required.  set to TACG")

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

    #get the name of the site
    env['site_name'] = EXTERNAL_PARAMS.get('site_name')

    env['pe_forward'] = EXTERNAL_PARAMS.get('pe_forward','')
    env['pe_reverse'] = EXTERNAL_PARAMS.get('pe_reverse','')
    env['isReverseRun'] = EXTERNAL_PARAMS.get('isReverseRun',False)

    env['runID'] = EXTERNAL_PARAMS.get('runid','ABCDE')


    env['tfKey'] = EXTERNAL_PARAMS.get('tfKey','')
    if env['tfKey'] == "":
        env['tfKey'] = "ATCG"

    SIGPROC_RESULTS = "sigproc_results"
    BASECALLER_RESULTS = "basecaller_results"
    ALIGNMENT_RESULTS = "alignment_results"
    SIGPROC_RESULTS = "./"
    BASECALLER_RESULTS = "./"
    ALIGNMENT_RESULTS = "./"
    env['SIGPROC_RESULTS'] = SIGPROC_RESULTS
    env['BASECALLER_RESULTS'] = BASECALLER_RESULTS
    env['ALIGNMENT_RESULTS'] = ALIGNMENT_RESULTS

    # Sub directory to contain fastq files for barcode enabled runs
    env['DIR_BC_FILES'] = 'bc_files'
    env['sam_parsed'] = EXTERNAL_PARAMS.get('sam_parsed')

    # Parse barcode_args (originates from GlobalConfig.barcode_args json)
    barcode_args = EXTERNAL_PARAMS.get('barcode_args',"")
    barcode_args = json.loads(barcode_args)
    for key in barcode_args:
        env['barcodesplit_'+key] = str(barcode_args[key])
    env['tmap_version'] = EXTERNAL_PARAMS.get('tmap_version')
    env['url_path'] = EXTERNAL_PARAMS.get('url_path')
    env['net_location'] = EXTERNAL_PARAMS.get('net_location')
    # net_location is set on masternode (in views.py) with "http://" + str(socket.getfqdn())
    env['master_node'] = env['net_location'].replace('http://','')

    sys.stdout.flush()
    sys.stderr.flush()

    return env

def getThumbnailSize(explog):
    # expLog.txt
    exp_json = json.loads(explog)
    log = json.loads(exp_json['log'])
    blockstatus = log.get('blocks', [])
    if not blockstatus:
        print >>sys.stderr, "ERROR: No blocks found in explog"
    W = 0
    H = 0
    for line in blockstatus:
        # Remove keyword; divide argument by comma delimiter into an array
        args = line.strip().replace('BlockStatus:','').split(',')
        datasubdir = "%s_%s" % (args[0].strip(),args[1].strip())
        if datasubdir == 'thumbnail':
            W = int(args[2].strip('W '))
            H = int(args[3].strip('H '))

    return [W,H]

def getBlocksFromExpLog(explog, excludeThumbnail=False):
    '''Returns array of block dictionary objects defined in explog.txt'''
    blocks = []
    # expLog.txt contents from Experiment.log field
    exp_json = json.loads(explog)
    log = json.loads(exp_json['log'])
    # contains regular blocks and a thumbnail block
    blockstatus = log.get('blocks', [])
    if not blockstatus:
        print >>sys.stderr, "ERROR: No blocks found in explog"
    for line in blockstatus:
        # Remove keyword; divide argument by comma delimiter into an array
        args = line.strip().replace('BlockStatus:','').split(',')

        # Remove leading space
        args = [entry.strip() for entry in args]

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
                'autoanalyze':False,
                'analyzeearly':False,
                'status':None}

        if args[0] =='thumbnail':
            block['datasubdir'] = 'thumbnail'
            if excludeThumbnail:
                continue
        else:
            block['datasubdir'] = "%s_%s" % (args[0].strip(),args[1].strip())
        block['autoanalyze'] = int(args[4].split(':')[1].strip()) == 1
        block['analyzeearly'] = int(args[5].split(':')[1].strip()) == 1
        block['id_str'] = block['datasubdir']
        print "explog: " + str(block)
        blocks.append(block)

    return blocks


def runplugins(env, basefolder, url_root):
    printtime('Running Plugins')

    try:
        pluginserver = xmlrpclib.ServerProxy("http://%s:%d" % (PLUGINSERVER_HOST, PLUGINSERVER_PORT), allow_none=True)
    except (socket.error, xmlrpclib.Fault):
        traceback.print_exc()

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
                env['sigproc_dir'] = os.path.join(env['report_root_dir'],env['SIGPROC_RESULTS'])
                env['basecaller_dir'] = os.path.join(env['report_root_dir'],env['BASECALLER_RESULTS'])
                env['alignment_dir'] = os.path.join(env['report_root_dir'],env['ALIGNMENT_RESULTS'])
                env['testfrag_key'] = 'ATCG'
                printtime("RAWDATA: %s" % env['pathToRaw'])
                start_json = make_plugin_json(env,plugin,env['primary_key'],basefolder,url_root)
                ret = pluginserver.pluginStart(start_json)
            except:
                printtime('plugin %s failed...' % plugin['name'])
                traceback.print_exc()


def run_selective_plugins(plugin_set,env,basefolder,url_root):
    try:
        pluginserver = xmlrpclib.ServerProxy("http://%s:%d" % (PLUGINSERVER_HOST, PLUGINSERVER_PORT), allow_none=True)
    except (socket.error, xmlrpclib.Fault):
        traceback.print_exc()

    for plugin in sorted(env['plugins'], key=lambda plugin: plugin["name"],reverse=True):
        if plugin['name'] in plugin_set:
            printtime("Plugin %s is enabled" % plugin['name'])

            try:
                #https://iontorrent.jira.com/wiki/display/TS/Plugin+json+file+format
                env['report_root_dir'] = os.getcwd()
                env['analysis_dir'] = os.getcwd()
                env['sigproc_dir'] = os.path.join(env['report_root_dir'],env['SIGPROC_RESULTS'])
                env['basecaller_dir'] = os.path.join(env['report_root_dir'],env['BASECALLER_RESULTS'])
                env['alignment_dir'] = os.path.join(env['report_root_dir'],env['ALIGNMENT_RESULTS'])
                env['testfrag_key'] = 'ATCG'
                printtime("RAWDATA: %s" % env['pathToRaw'])
                start_json = make_plugin_json(env,plugin,env['primary_key'],basefolder,url_root)
                ret = pluginserver.pluginStart(start_json)
                printtime('plugin %s started ...' % plugin['name'])
            except:
                printtime('plugin %s failed...' % plugin['name'])
                traceback.print_exc()
