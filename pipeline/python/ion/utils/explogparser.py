#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision: 17459 $")

import os
import sys
import datetime
import traceback
import json
import re
import logregexp as lre


def getparameter(parameterfile=None):

    #####################################################################
    # Load the analysis parameters and metadata from a json file passed in on the
    # command line with --params=<json file name>
    # we expect this loop to iterate only once. This is more elegant than
    # trying to index into ARGUMENTS.
    #####################################################################
    EXTERNAL_PARAMS = {}
    warnings = ''
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
            EXTERNAL_PARAMS[k] = str(v.encode('utf-8'))

    # Where the raw data lives (generally some path on the network)
    pathprefix = str(EXTERNAL_PARAMS['pathToData'])
    env['prefix'] = pathprefix

    #get the experiment json data
    env['exp_json'] = EXTERNAL_PARAMS.get('exp_json')

    env['pgmName'] = EXTERNAL_PARAMS.get('pgmName','unknownPGM')

    #this will get the exp data from the database
    exp_json = json.loads(env['exp_json'])
    if not isinstance(exp_json['log'],dict):
        exp_log_json = json.loads(exp_json['log'])
    else:
        exp_log_json = exp_json['log']

    env['flows'] = EXTERNAL_PARAMS.get('flows')
    env['notes'] = exp_json['notes']
    env['start_time'] = exp_log_json['start_time']

    env['blockArgs'] = EXTERNAL_PARAMS.get('blockArgs')

    #is it a thumbnail
    env['doThumbnail'] = EXTERNAL_PARAMS.get("doThumbnail")
    # get command line args
    env['beadfindArgs'] = EXTERNAL_PARAMS.get("beadfindArgs","")
    env['analysisArgs'] = EXTERNAL_PARAMS.get("analysisArgs","")
    env['basecallerArgs'] = EXTERNAL_PARAMS.get("basecallerArgs","")
    env['prebasecallerArgs'] = EXTERNAL_PARAMS.get("prebasecallerArgs","")
    env['doBaseRecal'] = EXTERNAL_PARAMS.get("doBaseRecal",False)
    
    #previousReports
    env['previousReport'] = EXTERNAL_PARAMS.get("previousReport","")

    # this is the library name for the run taken from the library field in the database
    env["libraryName"] = EXTERNAL_PARAMS.get("libraryName", "")
    if not env["libraryName"]:
        env["libraryName"] = "none"
        warnings += "WARNING: libraryName redefine required.  set to none\n"
    dtnow = datetime.datetime.now()
    # the time at which the analysis was started, mostly for debugging purposes
    env["report_start_time"] = dtnow.strftime("%c")
    # name of current analysis
    env['resultsName'] = EXTERNAL_PARAMS.get("resultsName")
    # name of current experiment
    env['expName'] = EXTERNAL_PARAMS.get("expName")
    # user-input part of experiment name: R_2012_04_30_15_22_02_user_FOZ-389--R145025-CC409_allt_12878-asr
    # would get FOZ-389--R145025-CC409_allt_12878-asr
    env['shortRunName'] = env['expName'].split('_',8)[-1]
    #library key input
    env['libraryKey'] = EXTERNAL_PARAMS.get("libraryKey","")
    if not env['libraryKey']:
        env['libraryKey'] = "TCAG"
        warnings += "WARNING: libraryKey redefine required.  set to TCAG\n"
    #path to the raw data
    env['pathToRaw'] = EXTERNAL_PARAMS.get("pathToData")
    #plugins
    env['plugins'] = EXTERNAL_PARAMS.get("plugins")
    #plan
    env['plan'] = EXTERNAL_PARAMS.get('plan', {})
    #eas
    env['experimentAnalysisSettings'] = EXTERNAL_PARAMS.get('experimentAnalysisSettings', {})
    # skipChecksum?
    env['skipchecksum'] = EXTERNAL_PARAMS.get('skipchecksum',False)
    # Do Full Align?
    env['align_full'] = EXTERNAL_PARAMS.get('align_full')

    env['flowOrder'] = EXTERNAL_PARAMS.get('flowOrder',"0").strip()
    # If flow order is missing, assume classic flow order:
    if env['flowOrder'] == "0":
        env['flowOrder'] = "TACG"
        warnings += "WARNING: floworder redefine required.  set to TACG\n"

    env['oninstranalysis'] = False
    try:
        if exp_log_json['oninstranalysis'] =="yes":
            env['oninstranalysis'] = True
    except:
        pass

    env['project'] = EXTERNAL_PARAMS.get('project')
    env['sample'] = EXTERNAL_PARAMS.get('sample')
    env['chipType'] = EXTERNAL_PARAMS.get('chiptype')
    env['barcodeId'] = EXTERNAL_PARAMS.get('barcodeId','')
    barcodeSamples = EXTERNAL_PARAMS.get('barcodeSamples','')
    env['barcodeSamples'] = json.loads(barcodeSamples) if barcodeSamples else {}
    
    env['reverse_primer_dict'] = EXTERNAL_PARAMS.get('reverse_primer_dict')
    env['rawdatastyle'] = EXTERNAL_PARAMS.get('rawdatastyle', 'single')

    #extra JSON
    env['extra'] = EXTERNAL_PARAMS.get('extra', '{}')
    # Aligner options
    env['aligner_opts_extra'] = EXTERNAL_PARAMS.get('aligner_opts_extra', '{}')
    env['mark_duplicates'] = EXTERNAL_PARAMS.get('mark_duplicates')
    env['realign'] = EXTERNAL_PARAMS.get('realign')

    #get the name of the site
    env['site_name'] = EXTERNAL_PARAMS.get('site_name')

    env['runID'] = EXTERNAL_PARAMS.get('runid','ABCDE')


    env['tfKey'] = EXTERNAL_PARAMS.get('tfKey','')
    if not env['tfKey']:
        env['tfKey'] = "ATCG"
        warnings += "WARNING: tfKey redefine required.  set to ATCG\n"

    env['SIGPROC_RESULTS'] = "sigproc_results"
    env['BASECALLER_RESULTS'] = "basecaller_results"
    env['ALIGNMENT_RESULTS'] = "./"

    # Sub directory to contain files for barcode enabled runs
    env['DIR_BC_FILES'] = os.path.join(env['BASECALLER_RESULTS'],'bc_files')
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

    # figure out report type
    if env['rawdatastyle'] == 'single':
        env['report_type'] = 'wholechip'
    else:
        if "thumbnail" in env['pathToRaw']:
           env['report_type'] = 'thumbnail'
        else:
           env['report_type'] = 'composite'

    env['username'] = EXTERNAL_PARAMS.get('username')

    return env, warnings

def getparameter_minimal(parameterfile=None):
    
    EXTERNAL_PARAMS = {}    
    if not parameterfile:    
        parameterfile = 'ion_params_00.json'
    afile = open(parameterfile, 'r')
    EXTERNAL_PARAMS = json.loads(afile.read())
    afile.close()
    for k,v in EXTERNAL_PARAMS.iteritems():
        if isinstance(v, unicode):
            EXTERNAL_PARAMS[k] = str(v)
    
    env = EXTERNAL_PARAMS
    env['SIGPROC_RESULTS'] = "sigproc_results"
    env['BASECALLER_RESULTS'] = "basecaller_results"
    env['ALIGNMENT_RESULTS'] = "./"
    env['pathToRaw'] = ""
  
    env['libraryKey'] = EXTERNAL_PARAMS.get('libraryKey', "TCAG")
    env['tfKey'] = EXTERNAL_PARAMS.get('tfKey', "ATCG")
    
    env['report_type'] = ""
  
    return env


def load_log_path(filepath):
    (head,tail) = os.path.split(filepath)
    return load_log(head, tail)
    
def load_log(folder,logName):
    """Retrieve the contents of the experiment log found in ``folder``,
    or return ``None`` if no log can be found."""
    fname = os.path.join(folder, logName)
    if os.path.isfile(fname):
        infile = None
        try:
            infile = open(fname)
            text = infile.read()
        except IOError:
            text = None
        finally:
            if infile is not None:
                infile.close()
            if len(text) == 0:
                text = None
    else:
        text = None
    return text

ENTRY_MAP = {
    "gain": lre.float_parse,
    "datacollect_version": lre.int_parse,
    "liveview_version": lre.int_parse,
    "firmware_version": lre.int_parse,
    "fpga_version": lre.int_parse,
    "driver_version": lre.int_parse,
    "script_version": lre.dot_separated_parse,
    "board_version": lre.int_parse,
    "kernel_build": lre.kernel_parse,
    "prerun": lre.yn_parse,
    # TODO: Enumerate things that will break if this is changed
    # It is used to parse job files produced by the PGM, yes?
    "cycles": lre.int_parse,
    "livechip": lre.yn_parse,
    "continuous_clocking": lre.yn_parse,
    "auto_cal": lre.yn_parse,
    "frequency": lre.int_parse,
    "oversample": lre.oversample_parse,
    "frame_time": lre.float_parse,
    "num_frames": lre.int_parse,
    "autoanalyze": lre.yn_parse,
    "dac": lre.dac_parse,
    "cal_chip_hist": lre.space_separated_parse,
    "cal_chip_high_low_inrange": lre.cal_range_parse,
    "prebeadfind": lre.yn_parse,
    "flows": lre.int_parse,
    "analyzeearly": lre.yn_parse,
#    "chiptype": lre.chip_parse,    
}

def parse_log(text):
    """Take the raw text of the experiment log, and return a dictionary
    of the entries contained in the log, parsed into the appropriate
    datatype.
    """

    ENTRY_RE = re.compile(r'^(?P<name>[^:]+)[:](?P<value>.*)$')
    CLEAN_RE = re.compile(r'\s|[/]+')

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
    ret['blocks'] = []
    ret['thumbnails'] = []
    #For the oddball repeating keyword: BlockStatus create an array of them.compatibility:
    for line in text.split('\n'):
        if line.startswith('BlockStatus'):
            ret['blocks'].append(line.strip().replace('BlockStatus:',''))
    #new format
    for k,v in ret.iteritems():
        if k.startswith('block_'):
            ret['blocks'].append(v)
        if k.startswith('thumbnail_'):
            ret['thumbnails'].append(v)

    return ret

def getBlocksFromExpLogJson(explog, excludeThumbnail=False):
    '''Returns array of block dictionary objects defined in explog.txt'''
    # expLog.txt contents from Experiment.log field
    exp_json = json.loads(explog)
    log = json.loads(exp_json['log'])
    return getBlocksFromExpLogDict(log,excludeThumbnail)

def getBlocksFromExpLogDict(explogdict, excludeThumbnail=False):
    '''Returns array of block dictionary objects defined in explog.txt'''
    blocks = []
    # contains regular blocks and a thumbnail block
    blockstatus = explogdict.get('blocks', [])
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
        #   jobid contains job id returned when job is queued
        #   status contains job status string
        block = {'id_str':'',
                'datasubdir':'',
                'jobid':None,
                'autoanalyze':False,
                'analyzeearly':False,
                'status':None}

        if args[0] =='thumbnail' or (args[0] == '0' and args[1] == '0'):
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


