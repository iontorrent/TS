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
from collections import deque

from ion.utils.plugin_json import *
from ion.plugin.remote import runPlugin

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
        (TMPL_DIR, "log.html", None),
        (TMPL_DIR, "alignment_summary.html", None),
        (TMPL_DIR, "format_whole.php", "Default_Report.php",), ## Renamed during copy
        #(os.path.join(distutils.sysconfig.get_python_lib(), 'ion', 'reports',  "BlockTLScript.py", None)
    ]
    for (d,s,f) in templates:
        if not f: f=s
        # If owner is different copy fails - unless file is removed first
        if os.access(f, os.F_OK):
            os.remove(f)
        shutil.copy(os.path.join(d,s), f)

def initBlockReport(blockObj,SIGPROC_RESULTS,BASECALLER_RESULTS,ALIGNMENT_RESULTS,oninstranalysis=False):
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

        _SIGPROC_RESULTS = os.path.join(SIGPROC_RESULTS, resultDir)
        _BASECALLER_RESULTS = os.path.join(BASECALLER_RESULTS, resultDir)
        _ALIGNMENT_RESULTS = os.path.join(ALIGNMENT_RESULTS, resultDir)

        if not oninstranalysis:
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
        TMP_PARAMS["pathToData"] = os.path.join(TMP_PARAMS["pathToData"], blockObj['id_str'])
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

        try:
            os.symlink(os.path.join(cwd,"parsefiles.php"), os.path.join(resultDir,"parsefiles.php"))
        except:
            printtime("couldn't create symbolic link")

        try:
            os.symlink(os.path.join(cwd,"log.html"), os.path.join(resultDir,"log.html"))
        except:
            printtime("couldn't create symbolic link")

        try:
            os.symlink(os.path.join(cwd,"DefaultTFs.conf"), os.path.join(resultDir,"DefaultTFs.conf"))
        except:
            printtime("couldn't create symbolic link")

        try:
            os.symlink(os.path.join(cwd,"barcodeList.txt"), os.path.join(resultDir,"barcodeList.txt"))
        except:
            printtime("couldn't create symbolic link")

    return resultDir

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

def get_plugins_to_run(plugins, report_type):
  """ Sort out runtype and runlevel of each plugin and return plugins appropriate for this analysis """
  printtime("Gettings plugins to run, report type = %s" % report_type)
  plugin_list=[]
  for plugin in sorted(plugins, key=lambda plugin: plugin["name"],reverse=True):  

    runPlug = True
 #   printtime("Plugin %s is enabled, AutoRun is %s" % (plugin['name'], plugin['autorun']))
    if not plugin['autorun']:
      continue    #skip to next plugin

    # Exclude this plugin if non-blank entry does not match run info:
    for label in ['project','sample','libraryName','chipType']:
      if plugin[label] != None and plugin[label] != "":
        runPlug = False
        for i in plugin[label].split(','):
            i = i.strip()
            if i in env[label]:
                runPlug = True
                break
            if not runPlug:              
              continue    #skip to next plugin
  
  
    # check if this plugin was selected to run for this run type
    if report_type == 'wholechip' or report_type == 'thumbnail':
      selected = True
    else:
      selected = False  
      
    if ('runtype' in plugin.keys()) and plugin['runtype']:
      selected = (report_type in plugin['runtype'])

    # TODO hardcoded specials
    if (report_type == 'composite') and (plugin['name'] in ['torrentscout','contourPlots','seq_dependent_errors']):
      selected = True
    if (report_type == 'thumbnail') and (plugin['name'] in ['rawPlots', 'separator', 'chipNoise']):
      plugin['runlevel'] = 'separator, default'          
    
    if selected:                  
      printtime("Plugin %s is enabled" % plugin['name'])
      if not ( ('runlevel' in plugin.keys()) and plugin['runlevel'] ):
      # fill in default value if runlevel is missing 
        plugin['runlevel'] = 'default'      
      plugin['hold_jid'] = []
      plugin_list.append(plugin)

  return plugin_list


def runplugins(plugins, env, basefolder, url_root, level = 'default'): 
    try:
        pluginserver = xmlrpclib.ServerProxy("http://%s:%d" % (PLUGINSERVER_HOST, PLUGINSERVER_PORT), allow_none=True)
    except (socket.error, xmlrpclib.Fault):
        traceback.print_exc()
        
    env['runlevel'] = level
    
    # 'last' plugin level waits for ALL previously launched plugins
    hold_last = []
    if 'last' in level:
      for plugin in plugins:
          hold_last+= plugin['hold_jid']
    
    retries = 1 # pipeline doesn't retry plugins: retries = 1    
    once = True
    nrun = 0
    for i,plugin in enumerate(plugins):      
      if level in plugin['runlevel'] and plugin['name'] != '':                    
          if once:
              printtime("Starting plugins runlevel=%s in basefolder= %s" % (level,basefolder) )
              once = False
          if 'last' in level:
              plugin['hold_jid'] = hold_last          
          
          start_json = make_plugin_json(env,plugin,env['primary_key'],basefolder,url_root) 
                   
          plugin, msg = runPlugin(plugin, start_json, level, pluginserver, retries)
          printtime(msg)           
          # save needed info for multilevel plugins
          plugins[i] = plugin          
          nrun += 1              
                  
    if nrun > 0:      
      printtime('Launched %i Plugins runlevel = %s' % (nrun,level))
      
    return plugins  


def run_selective_plugins(plugin_set,env,basefolder,url_root):
    try:
        pluginserver = xmlrpclib.ServerProxy("http://%s:%d" % (PLUGINSERVER_HOST, PLUGINSERVER_PORT), allow_none=True)
    except (socket.error, xmlrpclib.Fault):
        traceback.print_exc()

    for plugin in sorted(env['plugins'], key=lambda plugin: plugin["name"],reverse=True):
        if plugin['name'] in plugin_set:
            printtime("Plugin %s is enabled" % plugin['name'])

            try:
                env['report_root_dir'] = os.getcwd()
                start_json = make_plugin_json(env,plugin,env['primary_key'],basefolder,url_root)
                pluginserver.pluginStart(start_json)
                printtime('plugin %s started ...' % plugin['name'])
            except:
                printtime('plugin %s failed...' % plugin['name'])
                traceback.print_exc()



def merge_bam_files(bamfilelist,composite_bam_filepath,composite_bai_filepath,mark_duplicates):

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
        cmd = cmd + ' CREATE_INDEX=true'
        cmd = cmd + ' USE_THREADING=true'
        cmd = cmd + ' VALIDATION_STRINGENCY=SILENT'
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("bam file merge failed")
        traceback.print_exc()
        return 1

    try:
        # picard is using .bai , we want .bam.bai
        srcbaifilepath = composite_bam_filepath.replace(".bam",".bai")
        if os.path.exists(srcbaifilepath):
            os.rename(srcbaifilepath, composite_bai_filepath)
        else:
            printtime("ERROR: %s doesn't exists" % srcbaifilepath)
    except:
        traceback.print_exc()
        return 1

def merge_sff_files(sfffilelist,composite_sff_filepath):
    try:
        cmd = 'SFFMerge'
        cmd += ' -o %s ' % composite_sff_filepath
        for sfffile in sfffilelist:
            cmd += ' %s' % sfffile
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("SFFMerge failed")

    return 0
