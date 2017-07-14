#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Loads Fullchip analysis pipeline
Uses output from there to generate charts and graphs and dumps to current directory
Adds key metrics to database
"""

__version__ = filter(str.isdigit, "$Revision$")

import os
import tempfile

# matplotlib/numpy compatibility
os.environ['HOME'] = tempfile.mkdtemp()
# from matplotlib import use
# use("Agg")

# import /etc/torrentserver/cluster_settings.py, provides JOBSERVER_HOST, JOBSERVER_PORT
# import /etc/torrentserver/cluster_settings.py, provides PLUGINSERVER_HOST, PLUGINSERVER_PORT
import sys
sys.path.append('/etc')
from torrentserver.cluster_settings import *

import re
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
from ion.utils.file_exists import file_exists
from ion.utils.compress import make_zip
from ion.utils.blockprocessing import printtime

from ion.plugin.constants import RunLevel, RunType
from ion.plugin.remote import call_launchPluginsXMLRPC

from collections import deque
from urlparse import urlunsplit

from ion.reports.plotters import *
sys.path.append('/opt/ion/')

# 
#
# Analysis implementation details
#
# 


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


def initTLReport():
    plugin_folder = 'plugin_out'
    if not os.path.isdir(plugin_folder):
        oldmask = os.umask(0000)  # grant write permission to plugin user
        os.mkdir(plugin_folder)
        os.umask(oldmask)

    # Begin report writing
    os.umask(0002)
    # TMPL_DIR = os.path.join(distutils.sysconfig.get_python_lib(),'ion/web/db/writers')
    TMPL_DIR = '/usr/share/ion/web'
    templates = [
        # DIRECTORY, SOURCE_FILE, DEST_FILE or None for same as SOURCE
        (TMPL_DIR, "report_layout.json", None),
        (TMPL_DIR, "parsefiles.php", None),
        (TMPL_DIR, "format_whole.php", "Default_Report.php",),  # Renamed during copy
        #(os.path.join(distutils.sysconfig.get_python_lib(), 'ion', 'reports',  "BlockTLScript.py", None)
    ]
    for (d, s, f) in templates:
        if not f: f = s
        # If owner is different copy fails - unless file is removed first
        if os.access(f, os.F_OK):
            os.remove(f)
        shutil.copy(os.path.join(d, s), f)


def initBlockReport(blockObj, SIGPROC_RESULTS, BASECALLER_RESULTS, ALIGNMENT_RESULTS, from_sigproc_analysis=False):
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

        # TS-10734: plugin folder for IRU only
        plugin_folder = os.path.join(resultDir, 'plugin_out')
        if not os.path.isdir(plugin_folder):
            oldmask = os.umask(0000)  # grant write permission to plugin user
            os.mkdir(plugin_folder)
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
            os.symlink(os.path.join("..", _SIGPROC_RESULTS), os.path.join(resultDir, SIGPROC_RESULTS))
            os.symlink(os.path.join("..", _BASECALLER_RESULTS), os.path.join(resultDir, BASECALLER_RESULTS))
#        os.symlink(os.path.join("..",_ALIGNMENT_RESULTS), os.path.join(resultDir,ALIGNMENT_RESULTS))
        except:
            printtime("couldn't create symbolic link")

        file_in = open("ion_params_00.json", 'r')
        TMP_PARAMS = json.loads(file_in.read())
        file_in.close()

        # update path to data
        TMP_PARAMS["pathToData"] = os.path.join(TMP_PARAMS["pathToData"], blockObj['datasubdir'])
        TMP_PARAMS["mark_duplicates"] = False

        # write block specific ion_params_00.json
        file_out = open("%s/ion_params_00.json" % resultDir, 'w')
        json.dump(TMP_PARAMS, file_out)
        file_out.close()

        cwd = os.getcwd()
        try:
            os.symlink(os.path.join(cwd, "Default_Report.php"), os.path.join(resultDir, "Default_Report.php"))
        except:
            printtime("couldn't create symbolic link")

    return resultDir


def get_plugins_to_run(plugins, report_type):
    """ Sort out runtypes and runlevels of each plugin and return plugins appropriate for this analysis """
    blocklevel = False
    plugins_to_run = {}
    printtime("Get plugins to run, report type = %s" % report_type)
    for name in plugins.keys():
        plugin = plugins[name]

        # default is run on wholechip and thumbnail, but not composite
        selected = report_type in [RunType.FULLCHIP, RunType.THUMB]
        if plugin.get('runtypes'):
            selected = (report_type in plugin['runtypes'])

        if selected:
            plugin['runlevels'] = plugin.get('runlevels') if plugin.get('runlevels') else [RunLevel.DEFAULT]
            printtime("Plugin %s is enabled, runlevels=%s" % (plugin['name'], ','.join(plugin['runlevels'])))
            plugins_to_run[name] = plugin

            # check if have any blocklevel plugins
            if report_type == RunType.COMPOSITE and RunLevel.BLOCK in plugin['runlevels']:
                blocklevel = True
        else:
            printtime("Plugin %s (runtypes=%s) is not enabled for %s report" % (plugin['name'], ','.join(plugin.get('runtypes','')), report_type))

    return plugins_to_run, blocklevel


def runplugins(plugins, env, level=RunLevel.DEFAULT, params={}):
    printtime("Starting plugins runlevel=%s" % level)
    params.setdefault('run_mode', 'pipeline')  # Plugins launched here come from pipeline
    try:
        pluginserver = xmlrpclib.ServerProxy("http://%s:%d" % (PLUGINSERVER_HOST, PLUGINSERVER_PORT), allow_none=True)
        # call ionPlugin xmlrpc function to launch selected plugins
        # note that dependency plugins may be added to the plugins dict
        plugins, msg = call_launchPluginsXMLRPC(env['primary_key'], plugins, env['net_location'], env['username'], level, params, pluginserver)
        print msg
    except:
        traceback.print_exc()

    return plugins


def get_pgm_log_files(rawdatadir):
    # Create a tarball of the pgm raw data log files for inclusion into CSA.
    # tarball it now before the raw data gets deleted.
    # inst diagnostic files are always in toplevel raw data dir:
    if 'thumbnail' in rawdatadir:
        rawdatadir = rawdatadir.replace('thumbnail', '')
    files = [
        'explog_final.txt',
        'explog.txt',
        'InitLog.txt',
        'InitLog1.txt',
        'InitLog2.txt',
        'RawInit.txt',
        'RawInit.jpg',
        'InitValsW3.txt',
        'InitValsW2.txt',
        'Controller',
        'debug',
        'Controller_1',
        'debug_1',
        'chipCalImage.bmp.bz2',
        'InitRawTrace0.png',
    ]
    for afile in files:
        if os.path.exists(os.path.join(rawdatadir, afile)):
            make_zip('pgm_logs.zip', os.path.join(rawdatadir, afile), arcname=afile)

    return


def GetBlocksToAnalyze(env):
    blocks = []

    if is_thumbnail or is_single:

        if is_thumbnail:
            blocks.append({'id_str': 'thumbnail', 'datasubdir': 'thumbnail', 'jobid': None, 'status': None})

        elif is_single:
            blocks.append({'id_str': 'wholechip', 'datasubdir': '', 'jobid': None, 'status': None})

    else:
        explogblocks = explogparser.getBlocksFromExpLogJson(env['exp_json'], excludeThumbnail=True)
        for block in explogblocks:
            rawDirectory = os.path.join(env['pathToRaw'], block['datasubdir'])
            toProcess = (block['autoanalyze'] and block['analyzeearly']) or os.path.isdir(rawDirectory)

            if env.get('chipBlocksOverride') and toProcess:
                if env['chipBlocksOverride'] == '510':
                    toProcess = block['id_str'].endswith('Y0')

            if toProcess:
                print "block: " + str(block)
                blocks.append(block)
            else:
                print "skip block: " + str(block)

    print blocks
    return blocks


def hash_matches(full_filename):
    ret = False
    try:
        with open(full_filename, 'rb') as f:
            binary_content = f.read()
            md5sum = hashlib.md5(binary_content).hexdigest()

        head, tail = os.path.split(full_filename)
        with open(os.path.join(head, 'MD5SUMS'), 'r') as f:
            lines = f.readlines()
        expected_md5sums = {}
        for line in lines:
            ahash, filename = line.split()
            expected_md5sums[filename] = ahash
        ret = (expected_md5sums[tail] == md5sum)
    except:
        traceback.print_exc()
        pass
    return ret


def get_mem_usage():

    meminfo = {}
    with open('/proc/meminfo') as f:
        for line in f:
            name, value = line.split(':')
            meminfo[name] = int(re.findall(r'\d+', value)[0])

    # Transform from kB to MB
    mem_total = meminfo['MemTotal']/1024
    mem_free = meminfo['MemFree']/1024
    mem_used = (meminfo['MemTotal']-meminfo['MemFree'])/1024
    mem_buffers = meminfo['Buffers']/1024
    mem_cached = meminfo['Cached']/1024
    mem_total_free = mem_free+mem_buffers+mem_cached
    return "Memory [MB]  Total: {0:6d}   Used: {1:6d}   Free: {2:6d}   Buffers: {3:6d}   Cached: {4:6d}   TotalFree: {5:6d}".format(mem_total, mem_used, mem_free, mem_buffers, mem_cached, mem_total_free)


def write_jobid_list(block_job_dict, merge_job_dict=None):
    # save job ids to file, Services page job termination and job info functions read this
    job_list = {}
    for block, jobid in block_job_dict.items():
        job_list[block] = {'block_processing': jobid}

    if merge_job_dict:
        job_list['merge'] = merge_job_dict

    with open('job_list.json', 'w') as f:
        f.write(json.dumps(job_list, indent=2))


def spawn_cluster_job(rpath, scriptname, args, holds=None):
    out_path = "%s/drmaa_stdout_block.txt" % rpath
    err_path = "%s/drmaa_stderr_block.txt" % rpath
    cwd = os.getcwd()

    # SGE
    sge_queue = 'all.q'
    if is_thumbnail:
        sge_queue = 'thumbnail_worker.q'
    jt_nativeSpecification = "-pe ion_pe 1 -q " + sge_queue

    printtime("Use " + sge_queue)

    # TORQUE
    # jt_nativeSpecification = ""

    jt_remoteCommand = "python"
    jt_workingDirectory = os.path.join(cwd, rpath)
    jt_outputPath = ":" + os.path.join(cwd, out_path)
    jt_errorPath = ":" + os.path.join(cwd, err_path)
    jt_args = [os.path.join('/usr/bin', scriptname)] + args
    jt_joinFiles = False

    if holds != None and len(holds) > 0:
        jt_nativeSpecification += " -hold_jid "
        for holdjobid in holds:
            jt_nativeSpecification += "%s," % holdjobid

    # TODO remove debug output
    print jt_remoteCommand
    print jt_workingDirectory
    print jt_outputPath
    print jt_errorPath
    print jt_args
    print jt_nativeSpecification

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


if __name__ == "__main__":

    blockprocessing.printheader()

    env, warn = explogparser.getparameter()
    print warn

    debug_mode = False

    try:
        primary_key = open("primary.key").readline()
        primary_key = primary_key.split(" = ")
        env['primary_key'] = primary_key[1]
        printtime(env['primary_key'])
    except:
        printtime("Error, unable to get the primary key")

    # assemble the URL path for this analysis result, relative to the webroot directory: (/var/www/)
    # <output dir>/<Location name>/<analysis dir>
    url_root = os.path.join(env['url_path'], os.path.basename(os.getcwd()))

    printtime("DEBUG url_root string %s" % url_root)
    printtime("DEBUG net_location string %s" % env['net_location'])
    printtime("DEBUG master_node string %s" % env['master_node'])

    report_config = ConfigParser.RawConfigParser()
    report_config.optionxform = str  # don't convert to lowercase
    report_config.add_section('global')

    is_thumbnail = False
    is_single = False
    is_composite = False

    if env['rawdatastyle'] == 'single':
        is_single = True
        report_config.set('global', 'Type', '31x')
    else:
        if "thumbnail" in env['pathToRaw']:
            is_thumbnail = True
            report_config.set('global', 'Type', 'Thumbnail')
        else:
            is_composite = True
            report_config.set('global', 'Type', 'Composite')

    with open('ReportConfiguration.txt', 'wb') as reportconfigfile:
        report_config.write(reportconfigfile)

    # drops a zip file of the pgm log files
    get_pgm_log_files(env['pathToRaw'])

    # Generate a system information file for diagnostics purposes.
    try:
        com = "/usr/bin/ion_sysinfo 2>&1 >> ./sysinfo.txt"
        os.system(com)
    except:
        print traceback.format_exc()

    # 
    # Code to start full chip analysis                          #
    # 
    os.umask(0002)

    '''
    [
    os.path.join(sigproc_dir, 'analysis_return_code.txt'),
    os.path.join(sigproc_dir, 'analysis.bfmask.bin'),
    os.path.join(sigproc_dir, 'analysis.bfmask.stats'),
    os.path.join(sigproc_dir, 'processParameters.txt'),
    os.path.join(sigproc_dir, 'avgNukeTrace_%s.txt' % env['tfKey']),
    os.path.join(sigproc_dir, 'avgNukeTrace_%s.txt' % env['libraryKey']),
    os.path.join(sigproc_dir, '1.wells'),
    ]
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

    # define entry point
    print "blockArgs '"+str(env['blockArgs'])+"'"
    print "previousReport: '"+str(env['previousReport'])+"'"

    if is_thumbnail:
        initlogfilepath = os.path.join(env['pathToRaw'], '..', 'InitLog.txt')
        initlog1filepath = os.path.join(env['pathToRaw'], '..', 'InitLog1.txt')
        initlog2filepath = os.path.join(env['pathToRaw'], '..', 'InitLog2.txt')
        explogfilepath = os.path.join(env['pathToRaw'], '..', 'explog.txt')
        explogfinalfilepath = os.path.join(env['pathToRaw'], '..', 'explog_final.txt')
    else:
        initlogfilepath = os.path.join(env['pathToRaw'], 'InitLog.txt')
        initlog1filepath = os.path.join(env['pathToRaw'], 'InitLog1.txt')
        initlog2filepath = os.path.join(env['pathToRaw'], 'InitLog2.txt')
        explogfilepath = os.path.join(env['pathToRaw'], 'explog.txt')
        explogfinalfilepath = os.path.join(env['pathToRaw'], 'explog_final.txt')

    '''
    Instrument from raw + TS from wells
    re-analysis, from wells (BaseReport / no BaseReport(use OIA results))
    re-analysis, from raw (checkbox OnTS/OnInstrument)
    '''

    reference_selected = False
    for barcode_name, barcode_info in sorted(env['barcodeInfo'].iteritems()):
        if barcode_info['referenceName']:
            reference_selected = True
            pass

    if env['blockArgs'] == "fromRaw":
        doSigproc = True

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
        doSigproc = False

        if env['previousReport']:
            root_target = env['previousReport']
            sigproc_target = os.path.join(env['previousReport'], env['SIGPROC_RESULTS'])
        else:
            if env['oninstranalysis']:
                if is_composite:
                    root_target = os.path.join(env['pathToRaw'])
                    sigproc_target = os.path.join(env['pathToRaw'], 'onboard_results', env['SIGPROC_RESULTS'])
                elif is_thumbnail:
                    root_target = os.path.join(env['pathToRaw'], '..')
                    sigproc_target = os.path.join(env['pathToRaw'], '..', 'onboard_results', env['SIGPROC_RESULTS'], 'block_thumbnail')
            else:
                printtime("ERROR: previousReport not set")

        initlogfilepath = os.path.join(root_target, 'InitLog.txt')
        initlog1filepath = os.path.join(root_target, 'InitLog1.txt')
        initlog2filepath = os.path.join(root_target, 'InitLog2.txt')
        explogfilepath = os.path.join(root_target, 'explog.txt')
        explogfinalfilepath = os.path.join(root_target, 'explog_final.txt')

        os.symlink(sigproc_target, env['SIGPROC_RESULTS'])

        if not os.path.isdir(env['BASECALLER_RESULTS']):
            try:
                os.mkdir(env['BASECALLER_RESULTS'])
            except:
                traceback.print_exc()

        # copy files used in displaying report
        report_files = ['analysis.bfmask.stats', 'Bead_density_raw.png', 'Bead_density_contour.png', 'Bead_density_20.png',
                        'Bead_density_70.png', 'Bead_density_200.png', 'Bead_density_1000.png']
        for filepath in report_files:
            try:
                if os.path.exists(os.path.join(sigproc_target, filepath)):
                    shutil.copy(os.path.join(sigproc_target, filepath), ".")
            except:
                printtime(traceback.format_exc())

    else:
        printtime("WARNING: start point not defined, create new report from raw data")
        doSigproc = True

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

    # copy InitLog*.txt and explog.txt into report directory
    for filepath in [initlogfilepath, initlog1filepath, initlog2filepath, explogfilepath]:
        try:
            if os.path.exists(filepath):
                shutil.copy(filepath, ".")
            else:
                printtime("WARN: Cannot copy %s: doesn't exist" % filepath)
        except:
            printtime(traceback.format_exc())

    initTLReport()

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
            primary_key_file = os.path.join(os.getcwd(), 'primary.key')
            jobserver.updatestatus(primary_key_file, status, True)
            printtime("TLStatus %s\tpid %d\tpk file %s started in %s" %
                     (status, os.getpid(), primary_key_file, debugging_cwd))
        except:
            traceback.print_exc()

    set_result_status('Started')

    #-------------------------------------------------------------
    # Initialize plugins
    #-------------------------------------------------------------
    plugins, blocklevel_plugins = get_plugins_to_run(env['plugins'], env['report_type'])
    plugins_params = {}

    #-------------------------------------------------------------
    # Gridded data processing
    #-------------------------------------------------------------
    blockprocessing.write_version()

    printtime("RUNNING FULL CHIP MULTI-BLOCK ANALYSIS")
    # List of block objects to analyze
    blocks = GetBlocksToAnalyze(env)
    dirs = ['block_%s' % block['id_str'] for block in blocks]
    # dirs = ['%s/block_%s' % (SIGPROC_RESULTS, block['id_str']) for block in blocks]

    # 
    # Create block reports                              #
    # 

    # TODO
    doblocks = 1
    if doblocks:

        block_job_dict = {}
        merge_job_dict = {}

        result_dirs = {}
        for block in blocks:
            result_dirs[block['id_str']] = initBlockReport(block, env['SIGPROC_RESULTS'], env['BASECALLER_RESULTS'], env['ALIGNMENT_RESULTS'], from_sigproc_analysis=(env['blockArgs'] == "fromRaw"))

        # create a list of blocks
        blocks_to_process = []
        blocks_to_process.extend(blocks)
        timeout = 3*60*60
        if is_composite:
            if doSigproc:
                timeout = 1 * 1*60  # sec , skip OIA, process diagnosis blocks
            else:
                timeout = 20*60*60  # sec

        plugins_params['block_dirs'] = [os.path.join(env['report_root_dir'], result_dirs[block['id_str']]) for block in blocks_to_process]

        while len(blocks_to_process) > 0 and timeout > 0:

            printtime('waiting for %s block(s) to schedule    %s' % (str(len(blocks_to_process)), get_mem_usage()))
            sys.stdout.flush()
            sys.stderr.flush()

            # wait 10 sec before looking for new files
            timeout -= 10
            time.sleep(10)

            blocks_to_process_ready = []
            if doSigproc:
                for block in blocks_to_process:
                    if is_single:
                        data_file = os.path.join(env['pathToRaw'], 'acq_0000.dat')
                    elif is_thumbnail:
                        data_file = os.path.join(env['pathToRaw'], '..', block['id_str'], 'acq_0000.dat')  # TODO
                    else:  # is_composite
                        data_file = os.path.join(env['pathToRaw'], block['id_str'], 'acq_0000.dat')

                    if os.path.exists(data_file):
                        blocks_to_process_ready.append(block)
                    else:
                        if debug_mode:
                            printtime("missing %s" % data_file)

            else:
                for block in blocks_to_process:
                    if is_composite:
                        # look for all analysis_return_code.txt files, this is the last file beeing transfered
                        data_file = os.path.join(env['SIGPROC_RESULTS'], 'block_'+block['id_str'], 'analysis_return_code.txt')
                    else:  # is_thumbnail or is_single:
                        data_file = os.path.join(env['SIGPROC_RESULTS'], 'analysis_return_code.txt')

                    if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
                        blocks_to_process_ready.append(block)
                    else:
                        if debug_mode:
                            printtime("missing %s" % data_file)

            printtime('try to schedule %s new block(s)' % str(len(blocks_to_process_ready)))

            for block in blocks_to_process_ready:

                wait_list = []
                try:
                    if env['blockArgs'] == "fromRaw":
                        block_tlscript_options = ['--do-sigproc', '--do-basecalling']
                    else:
                        block_tlscript_options = ['--do-basecalling']

                    block['jobid'] = spawn_cluster_job(result_dirs[block['id_str']], 'BlockTLScript.py', block_tlscript_options, wait_list)
                    block_job_dict[block['id_str']] = str(block['jobid'])
                    printtime("Submitted block (%s) job with job ID (%s)" % (block['id_str'], str(block['jobid'])))
                except:
                    printtime("submitting job for block (%s) failed" % block['id_str'])

                blocks_to_process.remove(block)

            if len(blocks_to_process_ready) > 0:
                write_jobid_list(block_job_dict)

        if timeout <= 0:
            printtime("Error: timeout while processing blocks")

        if is_composite:
            merge_job_dict['merge'] = spawn_cluster_job('.', 'MergeTLScript.py', ['--do-sigproc', '--do-basecalling', '--do-zipping'], block_job_dict.values())
            printtime("Submitted composite merge job with job ID (%s)" % (str(merge_job_dict['merge'])))
        else:
            merge_job_dict['merge'] = spawn_cluster_job('.', 'MergeTLScript.py', ['--do-zipping'], block_job_dict.values())
            printtime("Submitted zipping job with job ID (%s)" % (str(merge_job_dict['merge'])))

        # update file now that all jobs are launched
        write_jobid_list(block_job_dict, merge_job_dict)

        # multilevel plugins preprocessing level
        plugins = runplugins(plugins, env, RunLevel.PRE, plugins_params)

        # Watch status of jobs.  As they finish remove the job from the list.

        pl_started = False
        block_job_list = block_job_dict.values()
        while len(block_job_list) > 0:
            for job in block_job_list:
                block = [block for block in blocks if block['jobid'] == job][0]
                # check status of jobid
                try:
                    block['status'] = jobserver.jobstatus(block['jobid'])
                except:
                    traceback.print_exc()
                    continue

                if blocklevel_plugins and (block['status'] == 'done'):
                    plugins_params['blockId'] = block['id_str']
                    plugins = runplugins(plugins, env, RunLevel.BLOCK, plugins_params)

                if block['status'] == 'done' or block['status'] == 'failed' or block['status'] == "DRMAA BUG":
                    printtime("Job %s has ended with status %s" % (str(block['jobid']), block['status']))
                    block_job_list.remove(block['jobid'])
#                else:
#                    printtime("Job %s has status %s" % (str(block['jobid']),block['status']))

            if os.path.exists(os.path.join(env['SIGPROC_RESULTS'], 'separator.mask.bin')) and not pl_started:
                plugins = runplugins(plugins, env, RunLevel.SEPARATOR, plugins_params)
                pl_started = True

            printtime("waiting for %d blocks to be finished    %s" % (len(block_job_list), get_mem_usage()))
            time.sleep(10)

        merge_job_list = merge_job_dict.keys()
        while len(merge_job_list) > 0:
            for key in merge_job_list:

                # check status of jobid
                try:
                    jid = merge_job_dict[key]
                    merge_status = jobserver.jobstatus(jid)
                except:
                    traceback.print_exc()
                    continue

                if merge_status == 'done' or merge_status == 'failed' or merge_status == "DRMAA BUG":
                    printtime("Job %s, %s has ended with status %s" % (key, jid, merge_status))
                    merge_job_list.remove(key)
                    break

            printtime("waiting for %d merge jobs to be finished    %s" % (len(merge_job_list), get_mem_usage()))
            time.sleep(10)

    printtime("All jobs processed")

    # 
    # ParseFiles and Upload Metrics                         #
    # 
    printtime("Attempting to Upload to Database")

    # attempt to upload the metrics to the Django database
    try:
        mycwd = os.getcwd()

        ionParamsPath = os.path.join('.', 'ion_params_00.json')
        BaseCallerJsonPath = os.path.join(env['BASECALLER_RESULTS'], 'BaseCaller.json')
        tfmapperstats_outputfile = os.path.join(env['BASECALLER_RESULTS'], "TFStats.json")
        ionstats_basecaller_json_path = os.path.join(env['BASECALLER_RESULTS'], 'ionstats_basecaller.json')
        peakOut = os.path.join('.', 'raw_peak_signal')
        beadPath = os.path.join(env['SIGPROC_RESULTS'], 'analysis.bfmask.stats')
        procPath = os.path.join(env['SIGPROC_RESULTS'], 'processParameters.txt')
        ionstats_alignment_json_path = os.path.join(env['ALIGNMENT_RESULTS'], 'ionstats_alignment.json')
        reportLink = True

        if is_thumbnail or is_single:
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
        elif is_composite:
            try:
                raw_return_code_file = os.path.join(env['BASECALLER_RESULTS'], "composite_return_code.txt")
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
            os.path.join(mycwd, tfmapperstats_outputfile),
            os.path.join(mycwd, procPath),
            os.path.join(mycwd, beadPath),
            os.path.join(mycwd, ionstats_alignment_json_path),
            os.path.join(mycwd, ionParamsPath),
            os.path.join(mycwd, peakOut),
            os.path.join(mycwd, ionstats_basecaller_json_path),
            os.path.join(mycwd, BaseCallerJsonPath),
            os.path.join(mycwd, 'primary.key'),
            os.path.join(mycwd, 'uploadStatus'),
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

    # 
    # Wait for final files to be written before proceeding #
    # 
    # copy explog_final.txt into report directory
    try:
        if file_exists(explogfinalfilepath, block=300, delay=2):
            shutil.copy(explogfinalfilepath, ".")
        else:
            printtime("WARN: Cannot copy %s: doesn't exist" % explogfinalfilepath)
    except:
        printtime(traceback.format_exc())

    # 
    # Write checksum_status.txt to raw data directory      #
    # 
    if is_single:
        raw_return_code_file = os.path.join(env['SIGPROC_RESULTS'], "analysis_return_code.txt")
    elif is_composite:
        raw_return_code_file = os.path.join(env['BASECALLER_RESULTS'], "composite_return_code.txt")

    try:
        # never overwrite existing checksum_status.txt file, needed in R&D to keep important data
        if not os.path.exists(os.path.join(env['pathToRaw'], "checksum_status.txt")):

            printtime("INFO: chipType: %s" % env['chipType'])

            keep_raw_data = False
            try:

                if os.path.exists('/opt/ion/.ion-internal-server'):

                    if is_composite:
                        aq17 = 0
                        if reference_selected:
                            if os.path.exists(ionstats_alignment_json_path):
                                afile = open(ionstats_alignment_json_path, 'r')
                                ionstats_alignment = json.load(afile)
                                afile.close()
                                aq17 = int(ionstats_alignment['AQ17']['num_bases'])
                            else:
                                printtime("ERROR: keep_raw_data check failed: ionstats_alignment.json is missing")
                        else:
                            printtime("INFO: reference library not set, skip AQ17 check")

                        if env['chipType'] == 'P2.2.2':
                            if aq17 > 3000000000:
                                keep_raw_data = True

                        if env['chipType'] == 'P1.1.17':
                            if aq17 > 23000000000:
                                keep_raw_data = True

                        if env['chipType'] == '530':
                            if aq17 > 7500000000:
                                keep_raw_data = True

                        if env['chipType'] == '521':
                            if aq17 > 2600000000:
                                keep_raw_data = True

            except:
                printtime("ERROR: keep_raw_data check failed")
                traceback.print_exc()

            if (is_single or is_composite) and os.path.isfile(raw_return_code_file):
                if keep_raw_data:
                    printtime("INFO: keep raw %s data" % env['chipType'])
                    f = open(raw_return_code_file+".keep", 'w')
                    f.write(str(99))
                    f.close()

                    shutil.copyfile(raw_return_code_file+".keep", os.path.join(env['pathToRaw'], "checksum_status.txt"))
                else:
                    shutil.copyfile(raw_return_code_file, os.path.join(env['pathToRaw'], "checksum_status.txt"))

    except:
        traceback.print_exc()

    # default plugin level
    plugins = runplugins(plugins, env, RunLevel.DEFAULT, plugins_params)

    getExpLogMsgs(explogfinalfilepath)
    get_pgm_log_files(env['pathToRaw'])

    # multilevel plugins postprocessing
    plugins = runplugins(plugins, env, RunLevel.POST, plugins_params)
    # plugins last level - plugins in this level will wait for all previously launched plugins to finish
    plugins = runplugins(plugins, env, RunLevel.LAST, plugins_params)

    # 
    # Record disk space usage for the Result directory #
    # 
    try:
        jobserver.resultdiskspace(env['primary_key'])
    except:
        traceback.print_exc()

    printtime("Run Complete")
    sys.exit(0)
