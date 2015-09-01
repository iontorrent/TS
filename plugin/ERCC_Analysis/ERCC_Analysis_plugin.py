#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
import subprocess
import json
import time
import traceback
from glob import glob
from optparse import OptionParser
from subprocess import *
import re
import simplejson

# set up for Django rendering with TSS installed apps
from django.conf import settings
from django.template.loader import render_to_string
from django.conf import global_settings
global_settings.LOGGING_CONFIG=None

#
# ----------- custom tag additions -------------
# Use template.base.add_to_builtins("django_tags") to add from django_tags.py (in cwd)
# Below is an example - may not be used in the current coverageAnalysis templates
#
from django import template
register = template.Library()

@register.filter 
def toMega(value):
  return float(value) / 1000000

template.builtins.append(register) 

# global data collecters common to functions
pluginParams = {}
pluginResult = {}
pluginReport = {}
barcodeSummary = []
barcodeReport = {}
help_dictionary = {}

# max plugin out file ext length for addressing max filename length (w/ barcode), e.g. ".amplicon.cov.xls"
max_fileext_len = 17
max_filename_len = 255

#
# -------------------- customize code for this plugin here ----------------
#

def addAutorunParams(plan=None):
  '''Additional parameter set up for automated runs, e.g. to add defaults for option only in the GUI.'''
  config = pluginParams['config']
  config['minrsquared'] = '0.9'
  config['mincounts'] = '10'
  config['erccpool'] = '1'
  config['barcode'] = ''
  config['fwdonlyreads'] = 'No'
  if plan:
    if 'runType' in plan and plan['runType'] == "RNA":
      config['fwdonlyreads'] = 'Yes'


def furbishPluginParams():
  '''Complete/rename/validate user parameters.'''
  config = pluginParams['config']
  if config['barcode'] == "All": config['barcode'] = ''
  if not 'fwdonlyreads' in config: config['fwdonlyreads'] = 'No'


def configReport():
  '''Returns a dictionary based on the plugin config parameters that as reported in results.json.'''
  # This is to avoid outputting hidden or aliased values. If not needed just pass back a copy of config.
  #return pluginParams['config'].copy()
  config = pluginParams['config']
  return {
    "Launch Mode" : config['launch_mode'],
    "Reference Genome" : pluginParams['genome_id'],
    "Passing R-square" : config['minrsquared'],
    "Min. Read Counts" : config['mincounts'],
    "ERCC Pool Used" :   config['erccpool'],
    "Barcodes Of Interest" : config['barcode'] if config['barcode'] else 'All',
    "barcoded" : 'true' if pluginParams['barcoded'] else 'false'
  }


def printStartupMessage():
  '''Output the standard start-up message. Customized for additional plugin input and options.'''
  printlog('')
  printtime('Started %s' % pluginParams['plugin_name'])
  config = pluginParams['config']
  printlog('Plugin run parameters:')
  printlog('  Plugin version:   %s' % pluginParams['cmdOptions'].version)
  printlog('  Launch mode:      %s' % config['launch_mode'])
  printlog('  Run is barcoded:  %s' % ('Yes' if pluginParams['barcoded'] else 'No'))
  printlog('  Plan reference:   %s' % pluginParams['genome_id'])
  printlog('  Ignore rev reads: %s' % config['fwdonlyreads'])
  printlog('  Passing R-square: %s' % config['minrsquared'])
  printlog('  Min. read counts: %s' % config['mincounts'])
  printlog('  ERCC pool used:   %s' % config['erccpool'])
  printlog('  Barcodes checked: %s' % (config['barcode'] if config['barcode'] else 'All'))
  printlog('Data files used:')
  printlog('  Parameters:     %s' % pluginParams['jsonInput'])
  printlog('  Reference:      %s' % "ERCC92/ERCC92.fasta")
  printlog('  Root alignment: %s' % pluginParams['bamroot'])
  printlog('')


def run_plugin(skiprun=False,barcode=""):
  '''Wrapper for making command line calls to perform the specific plugin analyis.'''
  # first part is pretty much boiler plate - grab the key parameters that most plugins use
  logopt = pluginParams['cmdOptions'].logopt
  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['output_dir']
  output_url = pluginParams['output_url']
  output_prefix = pluginParams['output_prefix']
  bamfile = pluginParams['bamfile']
  config = pluginParams['config']
  sample = sampleName(barcode,'None')
  fwd_reads = "-f" if config['fwdonlyreads'] == 'Yes' else ""

  # before accepting given (unaligned) BAM, check if aligned BAM exists, which may have ERCC aligned already
  alnbam = os.path.join( pluginParams['analysis_dir'], "rawlib.bam" )
  if os.path.exists(alnbam):
    printlog("Warning: Processed using mapped BAM file. (Possibly pre-aligned for ERCC reads.)")
    bamfile = alnbam

  # link from source BAM since pipeline uses the name as output file stem
  linkbam = os.path.join(output_dir,output_prefix+".bam")
  createlink(bamfile,linkbam)
  #createlink(bamfile+'.bai',linkbam+'.bai')
  bamfile = linkbam

  # pluginParams['reference'] not used here - uses local ERCC reference when remapping is required
  reference = '%s/ERCC92/ERCC92.fasta' % plugin_dir

  # skip the actual and assume all the data already exists in this file for processing
  if skiprun:
    printlog("Skipped analysis - generating report on in-situ data")
  else:
    # option for cmd-line detailed HTML report generation
    runcmd = '%s %s %s -D "%s" -H "%s" -B "%s" -F "%s" -N "%s" -R "%s" -M "%s" -T "%s" "%s" %s' % (
        os.path.join(plugin_dir,'run_erccanalysis.sh'), pluginParams['logopt'], fwd_reads, output_dir, pluginParams['report_name'],
        barcode, output_prefix, sample, reference, config['mincounts'], config['minrsquared'], bamfile, config['erccpool'] )
    if logopt: printlog('\n$ %s\n'%runcmd)
    if( os.system(runcmd) ):
      raise Exception("Failed running run_coverage_analysis.sh. Refer to Plugin Log.")

  if pluginParams['cmdOptions'].cmdline: return ({},{})

  # Link report page resources. This is necessary as the plugin code is inaccesible from URLs directly.
  #createlink( os.path.join(plugin_dir,'lifegrid'), output_dir )

  # Optional: Delete intermediate files after successful run. These should not be required to regenerate any of the
  # report if the skip-analysis option. Temporary file deletion is also disabled when the --keep_temp option is used.
  deleteTempFiles([ '*.log', '*.sam', '*.bam', '*.bam.bai', '*.fastq', '*.dat' ])

  # Parse out stats from results text file to dict AND convert unacceptible characters to underscores in keys to avoid Django issues
  resultData = parseToDict( os.path.join(output_dir,output_prefix+'.stats.txt'), ":" )

  # Catch for soft-error report
  if 'Failed Analysis' in resultData:
    raise Exception("CATCH:"+resultData['Failed Analysis'])

  # No report needed data here since it is created by the run using own template
  return (resultData,{})


def run_meta_plugin():
  '''Create barcode x target reads matrix and text version of barcode summary table (after analysis completes for individual barcodes).'''
  if pluginParams['cmdOptions'].cmdline: return
  printtime("Collating barcodes summary data...")
  bctable = []
  bcresults = pluginResult['barcodes']
  # iterate barcodeSummary[] to maintain barcode processing order
  for bcdata in barcodeSummary:
    bcname = bcdata['barcode_name']
    bcline = "%s\t%s\t%s\t%s\t%s\t%s"%(bcname,bcdata['sample'],bcdata['passes'],bcdata['detected_targets'],bcdata['ercc_targets'],bcdata['r_squared'])
    bctable.append(bcline)

  if len(bctable) > 0:
    bctabfile = pluginParams['prefix']+".bc_summary.xls"
    bcline = "Barcode ID\tSample Name\tPasses\tTargets Detected\tOn Target\tR-Squared"
    with open(os.path.join(pluginParams['results_dir'],bctabfile),'w') as outfile:
      outfile.write(bcline+'\n')
      for bcline in bctable:
        outfile.write(bcline+'\n')
    barcodeReport.update({"bctable":bctabfile})


def updateBarcodeSummaryReport(barcode,autoRefresh=False):
  '''Create barcode summary (progress) report. Called before, during and after barcodes are being analysed.'''
  global barcodeSummary
  if pluginParams['cmdOptions'].cmdline: return
  # no barcode means either non have been ceated yet or this is a refresh after all have been processed (e.g. for meta data)
  if barcode != "":
    resultData = pluginResult['barcodes'][barcode]
    # check for error status
    errMsg = resultData.get('Error','')
    sample = sampleName(barcode,'None')
    # barcodes_json dictionary is firm-coded in Kendo table template that we are using for main report styling
    if errMsg != "":
      detailsLink = "<span class='help' title='%s' style='color:red'>%s</span>" % ( errMsg, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "passes" : "NA",
        "detected_targets": "NA",
        "ercc_targets": "NA",
        "r_squared": "NA"
      })
    else:
      detailsLink = "<a target='_parent' href='%s' class='help'><span title='Click to view the detailed report for barcode %s'>%s</span><a>" % (
        os.path.join(barcode,pluginParams['report_name']), barcode, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "passes" : resultData['Passes Correlation Threshold'],
        "detected_targets": resultData['ERCC Targets Detected'],
        "ercc_targets": resultData['Percent ERCC tracking reads'],
        "r_squared": resultData['Dose-response R-squared']
      })
  render_context = {
    "autorefresh" : autoRefresh,
    "run_name" : pluginParams['prefix'],
    "barcode_results" : simplejson.dumps(barcodeSummary)
  }
  render_context.update(renderOptions())
  # extra report items, e.g. file links from barcodes summary page
  if barcodeReport:
    render_context.update(barcodeReport)
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'barcode_summary.html', render_context )


def renderOptions():
  '''Support method to generate list of rendering options and echoed user options.'''
  return pluginParams['config']


def createIncompleteReport(errorMsg=""):
  '''Called to create an incomplete or error report page for non-barcoded runs.'''
  render_context = {
    "autorefresh" : (errorMsg == ""),
    "run_name": pluginParams['prefix'],
    "Sample_Name": sampleName(),
    "Error": errorMsg }
  render_context.update(renderOptions())
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'incomplete.html', render_context )


def createDetailReport(resultData,reportData):
  '''Called to create the main report (for un-barcoded run or for each barcode).'''
  # Code uses its own HTML template
  pass


def createBlockReport():
  '''Called at the end of run to create a block.html report. Use 'pass' if not wanted.'''
  # Code creates its own block report for non-bacoded
  if pluginParams['cmdOptions'].cmdline: return
  printtime("Creating block report...")
  if pluginParams['barcoded']:
    render_context = {
      "run_name" : pluginParams['prefix'],
      "barcode_results" : simplejson.dumps(barcodeSummary)
    }
    render_context.update( renderOptions() )
    tplate = 'barcode_block.html'
    createReport( pluginParams['block_report'], tplate, render_context )
  # Code uses its own HTML template for detailed reports


def createErrorReport(errMsg="",overwriteSummary=False):
  '''Create both a block report page and optionally overwrite the existing (barcode) summary report.'''
  global pluginResult
  if errMsg.startswith("CATCH:"):
    errMsg = errMsg[6:]
    printlog("Analysis failed: "+errMsg)
  else:
    printlog("ERROR: "+errMsg)
  pluginResult.update({ 'Error': errMsg })
  if overwriteSummary == True:
    createIncompleteReport(errMsg)
  render_context = pluginResult.copy()
  render_context.update(renderOptions())
  createReport( pluginParams['block_report'], 'failed_nonbarcode_block.html', render_context )


#
# --------------- Base code for standard plugin runs -------------
#

def parseCmdArgs():
  '''Process standard command arguments. Customized for additional debug and other run options.'''
  # standard run options here - do not remove
  parser = OptionParser()
  parser.add_option('-B', '--bam', help='Filepath to root alignment BAM file. Default: rawlib.bam', dest='bamfile', default='')
  parser.add_option('-P', '--prefix', help='Output file name prefix for output files. Default: '' => Use analysis folder name or "output".', dest='prefix', default='')
  parser.add_option('-R', '--reference_fasta', help='Path to fasta file for the whole reference', dest='reference', default='')
  parser.add_option('-U', '--results_url', help='URL for access to files in the output directory', dest='results_url', default='')
  parser.add_option('-V', '--version', help='Version string for tracking in output', dest='version', default='')
  parser.add_option('-X', '--min_bc_bam_size', help='Minimum file size required for barcode BAM processing', type="int", dest='minbamsize', default=0)
  parser.add_option('-c', '--cmdline', help='Run command line only. Reports will not be generated using the HTML templates.', action="store_true", dest='cmdline')
  parser.add_option('-d', '--scraper', help='Create a scraper folder of links to output files using name prefix (-P).', action="store_true", dest='scraper')
  parser.add_option('-k', '--keep_temp', help='Keep intermediate files. By default these are deleted after a successful run.', action="store_true", dest='keep_temp')
  parser.add_option('-l', '--log', help='Output extra progress Log information to STDERR during a run.', action="store_true", dest='logopt')
  parser.add_option('-p', '--purge_results', help='Remove all folders and most files from output results folder.', action="store_true", dest='purge_results')
  parser.add_option('-s', '--skip_analysis', help='Skip re-generation of existing files but make new report.', action="store_true", dest='skip_analysis')
  parser.add_option('-x', '--stop_on_error', help='Stop processing barcodes after one fails. Otherwise continue to the next.', action="store_true", dest='stop_on_error')

  (cmdOptions, args) = parser.parse_args()
  if( len(args) != 1 ):
    printerr('Takes only one argument (parameters.json file)')
    raise TypeError(os.path.basename(__file__)+" takes exactly one argument (%d given)."%len(args))
  with open(args[0]) as jsonFile:
    jsonParams = json.load(jsonFile)
  global pluginParams
  pluginParams['cmdOptions'] = cmdOptions
  pluginParams['jsonInput'] = args[0]
  pluginParams['jsonParams'] = jsonParams

def emptyResultsFolder():
  '''Purge everything in output folder except for specifically named files.'''
  if not pluginParams['cmdOptions'].purge_results: return
  # Dangerous - replace with something safer if it becomes obvious (e.g. putting output in subfolder?)
  results_dir = pluginParams['results_dir']
  if results_dir == '/': return
  logopt = pluginParams['cmdOptions'].logopt
  cwd = os.path.realpath(os.getcwd())
  if logopt or os.path.exists( os.path.join(results_dir,pluginParams['report_name']) ):
    printlog("Purging old results...")
  for root,dirs,files in os.walk(results_dir,topdown=False):
    for name in files:
      # these are the exceptions - partial names and in the to level results
      if root == results_dir:
        start = os.path.basename(name)[:10]
        if start == "drmaa_stdo" or start == "ion_plugin" or start == "startplugi" or start == 'barcodes.j':
          continue
      fname = os.path.join(root,name)
      if logopt and root == results_dir:
        printlog("Removing file %s"%fname)
      os.system('rm -f "%s"'%fname)
    for name in dirs:
      fname = os.path.realpath(os.path.join(root,name))
      if fname.startswith(cwd):
        printlog("Warning: Leaving folder %s as in cwd path."%fname)
        continue
      if logopt:
        printlog("Removing directory %s"%fname)
      os.system('rm -rf "%s"'%fname)
  if logopt: printlog("")

def parseToDict(filein,sep=None):
  ret = {}
  if os.path.exists(filein):
    with open(filein) as fin:
      for line in fin:
        line = line.strip()
        # ignore lines being with non-alphanum (for comments, etc)
        if line == "" or not line[0].isalnum():
          continue
        kvp = line.split(sep,1)
        if len(kvp) > 1:
          ret[kvp[0].strip()] = kvp[1].strip()
  else:
    printerr("parseToDict() could not open "+filein)
  return ret

def printerr(msg):
  cmd = os.path.basename(__file__)
  sys.stderr.write( '%s: ERROR: %s\n' % (cmd,msg) )
  sys.stderr.flush()

def printlog(msg):
  sys.stderr.write(msg)
  sys.stderr.write('\n')
  sys.stderr.flush()

def printtime(msg):
  # use unix 'date' command so output format is identical to called script
  runtime = Popen( ["date"], stdout=PIPE, shell=False )
  dtm = runtime.communicate()[0]
  printlog( '(%s) %s'%(dtm.strip(),msg) )

def createlink(srcPath,destPath):
  # using system call as os.symlink() only seems to handle one file at a time and has other limitations
  if not srcPath:
    printlog("WARNING: Failed to create symlink as source path is empty.")
    return False
  elif not os.path.exists(srcPath):
    printlog("WARNING: Failed to create symlink as source path '%s' was not found."%srcPath)
    return False
  elif not destPath:
    printlog("WARNING: Failed to create symlink as destination path is empty.")
    return False
  os.system('ln -s "%s" "%s"'%(srcPath,destPath))
  if pluginParams['cmdOptions'].logopt:
    printlog("Created symlink %s -> %s"%(destPath,srcPath))
  return True
  
def deleteTempFiles(tmpFiles):
  if tmpFiles == None or pluginParams['cmdOptions'].keep_temp: return
  output_dir = pluginParams['output_dir']
  for filename in tmpFiles:
    flist = glob( os.path.join(output_dir,filename) )
    for f in flist:
      if pluginParams['cmdOptions'].logopt:
        printlog("Deleting file %s"%f)
      os.system('rm -f "%s"'%f)

def createReport(reportName,reportTemplate,reportData):
  # configure django to use the templates folder and various installed apps
  if not settings.configured:
    plugin_dir = pluginParams['plugin_dir'] if 'plugin_dir' in pluginParams else os.path.realpath(__file__)
    settings.configure( DEBUG=False, TEMPLATE_DEBUG=False,
      INSTALLED_APPS=('django.contrib.humanize',),
      TEMPLATE_DIRS=(os.path.join(plugin_dir,'templates'),) )

  with open(reportName,'w') as bcsum:
    bcsum.write( render_to_string(reportTemplate,safeKeys(reportData)) )

def sampleNames():
  try:
    if pluginParams['barcoded']:
      samplenames = {}
      bcsamps = pluginParams['jsonParams']['plan']['barcodedSamples']
      if isinstance(bcsamps,basestring):
        bcsamps = json.loads(bcsamps)
      for bcname in bcsamps:
        for bc in bcsamps[bcname]['barcodes']:
          samplenames[bc] = bcname if bcname != 'Unknown' else ''
    else:
      samplenames = jsonParams['expmeta']['sample']
  except:
    return ""
  return samplenames

def sampleName(barcode='',default=''):
  if not 'sample_names' in pluginParams:
    return default
  sample_names = pluginParams['sample_names']
  if isinstance(sample_names,basestring):
    return sample_names if sample_names else default
  return sample_names.get(barcode,default) if barcode else default

def targetFiles():
  trgfiles = {}
  try:
    if pluginParams['barcoded']:
      bcbeds = pluginParams['config']['barcodetargetregions']
      if isinstance(bcbeds,basestring):
        bcmaps = bcbeds.split(';')
        for m in bcmaps:
          kvp = m.split('=',1)
          if kvp[0] and kvp[1]:
            trgfiles[kvp[0]] = kvp[1]
      else:
        for bc in bcbeds:
          trgfiles[bc] = bcbeds[bc]
  except:
    pass
  return trgfiles

def loadPluginParams():
  '''Process default command args and json parameters file to extract TSS plugin environment.'''
  global pluginParams
  parseCmdArgs()

  # copy typical environment data needed for analysis
  jsonParams = pluginParams['jsonParams']
  pluginParams['plugin_name'] = jsonParams['runinfo'].get('plugin_name','')
  pluginParams['plugin_dir'] = jsonParams['runinfo'].get('plugin_dir','.')
  pluginParams['genome_id'] = jsonParams['runinfo'].get('library','')
  pluginParams['run_name'] = jsonParams['expmeta'].get('run_name','')
  pluginParams['analysis_name'] = jsonParams['expmeta'].get('results_name',pluginParams['plugin_name'])
  pluginParams['analysis_dir'] = jsonParams['runinfo'].get('analysis_dir','.')
  pluginParams['results_dir'] = jsonParams['runinfo'].get('results_dir','.')
  pluginParams['logopt'] = '-l' if pluginParams['cmdOptions'].logopt else ''

  # some things not yet in startplugin.json are provided or over-writen by cmd args
  copts = pluginParams['cmdOptions']
  pluginParams['reference'] = copts.reference if copts.reference != "" else jsonParams['runinfo'].get('reference','')
  pluginParams['bamroot']   = copts.bamfile   if copts.bamfile != "" else '%s/rawlib.bam' % pluginParams['analysis_dir']
  pluginParams['prefix']    = copts.prefix    if copts.prefix != "" else pluginParams['analysis_name']
  pluginParams['results_url'] = copts.results_url if copts.results_url != "" else os.path.join(
  jsonParams['runinfo'].get('url_root','.'),'plugin_out',pluginParams['plugin_name']+'_out' )

  # check for non-supported de novo runs
  #if not pluginParams['genome_id'] or not pluginParams['reference']:
  #  printerr("Requires a reference sequence for coverage analysis.")
  #  raise Exception("CATCH:Do not know how to analyze coverage without reference sequence for library '%s'"%pluginParams.get('genome_id',""))

  # set up for barcoded vs. non-barcodedruns
  pluginParams['bamfile'] = pluginParams['bamroot']
  pluginParams['output_dir'] = pluginParams['results_dir']
  pluginParams['output_url'] = pluginParams['results_url']
  pluginParams['output_prefix'] = pluginParams['prefix']
  pluginParams['bamname'] = os.path.basename(pluginParams['bamfile'])
  pluginParams['barcoded'] = os.path.exists(pluginParams['analysis_dir']+'/barcodeList.txt')
  pluginParams['sample_names'] = sampleNames()

  # disable run skip if no report exists => plugin has not been run before
  pluginParams['report_name'] = pluginParams['plugin_name']+'.html'
  pluginParams['block_report'] = os.path.join(pluginParams['results_dir'],pluginParams['plugin_name']+'_block.html')
  if not os.path.exists( os.path.join(pluginParams['results_dir'],pluginParams['report_name']) ):
    if pluginParams['cmdOptions'].skip_analysis:
      printlog("Warning: Skip analysis option ignorred as previous output appears to be missing.")
      pluginParams['cmdOptions'].skip_analysis = False

  # set up plugin specific options depending on auto-run vs. plan vs. GUI
  config = pluginParams['config'] = jsonParams['pluginconfig'].copy() if 'pluginconfig' in jsonParams else {}
  launchmode = config.get('launch_mode','')
  if not launchmode:
    if jsonParams['runinfo'].get('run_mode','') == 'manual': launchmode = 'Manual'
  if launchmode == 'Manual':
    pass
  elif 'plan' in jsonParams:
    if launchmode:
      furbishPluginParams()
    else:
      config['launch_mode'] = 'Autostart with plan configuration'
    addAutorunParams(jsonParams['plan'])
  else:
    config['launch_mode'] = 'Autostart with default configuration'
    addAutorunParams()

  # complete plugin customization
  furbishPluginParams()

  # plugin configuration becomes basis of results.json file
  global pluginResult, pluginReport
  pluginResult = configReport()
  if pluginParams['barcoded']:
    pluginResult['barcodes'] = {}
    pluginReport['barcodes'] = {}


def fileName(filepath):
  filepath = os.path.basename(filepath)
  return os.path.splitext(filepath)[0]

def writeDictToJsonFile(data,filename):
  with open(filename,'w') as outfile:
    json.dump(data,outfile,indent=2,sort_keys=True)
    if pluginParams['cmdOptions'].logopt:
      printlog("Created JSON file '%s'"%filename)

def safeKeys(indict):
  # Recursive method to return a dictionary with non alpha-numeric characters in dictionary key names replaced with underscores.
  # Expects indict to be a json-compatible dictionary or array of dictionaries.
  # A non-dicionary object (reference) is returned, i.e. no copy is made as with arrays and dicionaries.
  # lists and tuple (subclass) objects are returned as ordinary lists
  if isinstance(indict,(list,tuple)):
    nlist = []
    for item in indict:
      nlist.append(safeKeys(item))
    return nlist
  if not isinstance(indict,dict):
    return indict
  retdict = {}
  for key,value in indict.iteritems():
    retdict[re.sub(r'[^0-9A-Za-z]','_',key)] = safeKeys(value)
  return retdict
  
def testRun(outdir,prefix):
  # default for testing framework
  testout = os.path.join(outdir,prefix+"_test.out")
  with open(testout,'w') as f:
    f.write("This is a test file.\n")
  printlog('Created %s'%testout)

def ensureFilePrefix(prependLen=0):
  global pluginParams
  prefix = pluginParams['prefix']
  maxfn = prependLen + len(prefix) + max_fileext_len
  if maxfn <= max_filename_len: return
  # clip prefix to maximum size for allowed (before prepend/append)
  prefix = prefix[:max_filename_len-maxfn]
  maxfn = len(prefix)
  # use nearest '_' if doesn't reduce the length of name by more than 70%
  uslen = prefix.rfind('_')
  if uslen >= 0.7*maxfn:
    prefix = prefix[:uslen]
  printlog("WARNING: Output file name stem shortened to ensure output file name length <= %d characters.\nNew stem = %s\n" % (max_filename_len,prefix))
  pluginParams['prefix'] = prefix

def runForBarcodes():
  global pluginParams, pluginResult, pluginReport
  # read barcode ids
  barcodes = []
  try:
    bcfileName = pluginParams['analysis_dir']+'/barcodeList.txt'
    with open(bcfileName) as bcfile:
      for line in bcfile:
        if line.startswith('barcode '):
          barcodes.append(line.split(',')[1])
  except:
    printerr("Reading barcode list file '%s'" % bcfileName)
    raise
  # iterate over listed barcodes to pre-test barcode files
  bcoi = pluginParams['config']['barcode'].split(',')
  have_bcoi = (bcoi[0] != '' and bcoi[0] != 'All')
  numGoodBams = 0
  numAnalBams = 0
  maxBarcodeLen = 0
  minFileSize = pluginParams['cmdOptions'].minbamsize
  (bcBamPath,bcBamRoot) = os.path.split(pluginParams['bamroot'])
  bcBamFile = []
  for barcode in barcodes:
    # before accepting given (unaligned) BAM, check if aligned BAM exists, which may have ERCC aligned already
    bcbam = os.path.join( pluginParams['analysis_dir'], "%s_rawlib.bam"%barcode )
    if not os.path.exists(bcbam):
      bcbam = os.path.join( bcBamPath, "%s_%s"%(barcode,bcBamRoot) )
    else:
      numAnalBams += 1
    # special case for ERCC plugin: allows run on single specific barcode
    if have_bcoi and not barcode in bcoi:
      bcbam = ": Barcode excluded by user selection"
    elif not os.path.exists(bcbam):
      bcbam = ": BAM file not found"
    elif os.stat(bcbam).st_size < minFileSize:
      bcbam = ": BAM file too small"
    else:
      if( len(barcode) > maxBarcodeLen ):
        maxBarcodeLen = len(barcode)
      numGoodBams += 1
    bcBamFile.append(bcbam)

  ensureFilePrefix(maxBarcodeLen+1)

  printlog("Processing %d barcodes...\n" % numGoodBams)
  if numAnalBams > 0:
    printlog("Warning: %d barcodes will be processed using mapped BAM files. (Possibly pre-aligned for ERCC reads.)" % numAnalBams)
  pluginReport['num_barcodes_processed'] = numGoodBams
  pluginReport['num_barcodes_failed'] = 0

  # create initial (empty) barcodes summary report
  updateBarcodeSummaryReport("",True)

  # iterate over all barcodes and process the valid ones
  skip_analysis = pluginParams['cmdOptions'].skip_analysis
  stop_on_error = pluginParams['cmdOptions'].stop_on_error
  create_scraper = pluginParams['cmdOptions'].scraper
  postout = False; # just for logfile prettiness
  for barcode in barcodes:
    sample = sampleName(barcode)
    bamfile = bcBamFile.pop(0)
    if bamfile[0] == ":":
      if postout:
        postout = False
        printlog("")
      printlog("Skipping %s%s%s" % (barcode,('' if sample == '' else ' (%s)'%sample),bamfile))
      # special error case for bad user barcode selection
      if have_bcoi and barcode in bcoi:
        errMsg = "User specified barcode not available for analysis."
        printerr(errMsg)
        # Record as barcode failure as this may have be unexpected
        pluginReport['num_barcodes_failed'] += 1
        pluginResult['barcodes'][barcode] = { "Sample Name" : sample, "Error" : str(errMsg) }
        pluginReport['barcodes'][barcode] = {}
        updateBarcodeSummaryReport(barcode,True)
    else:
      postout = True
      printlog("\nProcessing %s%s...\n" % (barcode,('' if sample == '' else ' (%s)'%sample)))
      pluginParams['bamfile'] = bamfile
      pluginParams['output_dir'] = os.path.join(pluginParams['results_dir'],barcode)
      pluginParams['output_url'] = os.path.join(pluginParams['results_url'],barcode)
      pluginParams['output_prefix'] = barcode+"_"+pluginParams['prefix']
      if not os.path.exists(pluginParams['output_dir']):
         os.makedirs(pluginParams['output_dir'])
      try:
        (resultData,reportData) = run_plugin(skip_analysis,barcode)
        pluginResult['barcodes'][barcode] = resultData
        pluginReport['barcodes'][barcode] = reportData
        createDetailReport(resultData,reportData)
        if create_scraper:
          createScraperLinksFolder( pluginParams['output_dir'], pluginParams['output_prefix'] )
      except Exception, e:
        errMsg = str(e)
        if errMsg.startswith("CATCH:"): errMsg = errMsg[6:]
        pluginResult['barcodes'][barcode] = { "Sample Name" : sample, "Error" : errMsg }
        pluginReport['barcodes'][barcode] = {}
        # Do not trace expected errors as failures
        if str(e).startswith("CATCH:"): 
          printlog('Analysis of barcode %s failed: %s'%(barcode,errMsg))
        else:
          printerr('Analysis of barcode %s failed:'%barcode)
          pluginReport['num_barcodes_failed'] += 1
          if stop_on_error: raise
          traceback.print_exc()
      updateBarcodeSummaryReport(barcode,True)

  run_meta_plugin()
  updateBarcodeSummaryReport("")
  if create_scraper:
    createScraperLinksFolder( pluginParams['results_dir'], pluginParams['prefix'] )


def runNonBarcoded():
  global pluginResult, pluginReport
  ensureFilePrefix()
  try:
    createIncompleteReport()
    (resultData,pluginReport) = run_plugin( pluginParams['cmdOptions'].skip_analysis )
    pluginResult.update(resultData)
    createDetailReport(pluginResult,pluginReport)
  except Exception, e:
    errMsg = str(e)
    createErrorReport(errMsg,True)
    if not errMsg.startswith("CATCH:"): raise
  if pluginParams['cmdOptions'].scraper:
    createScraperLinksFolder( pluginParams['output_dir'], pluginParams['output_prefix'] )

def createScraperLinksFolder(outdir,rootname):
  '''Make links to all files matching <outdir>/<rootname>.* to <outdir>/scraper/link.*'''
  # rootname is a file path relative to outdir and should not contain globbing characters
  scrapeDir = os.path.join(outdir,'scraper')
  if pluginParams['cmdOptions'].logopt:
    printlog("Creating scraper folder %s"%scrapeDir)
  if not os.path.exists(scrapeDir):
    os.makedirs(scrapeDir)
  subroot =  os.path.basename(rootname)+'.'
  flist = glob( os.path.join(outdir,rootname)+'.*' )
  for f in flist:
    lname = os.path.basename(f).replace(subroot,'link.')
    createlink( f, os.path.join(scrapeDir,lname) )

def wrapup():
  '''Called at very end of run for final data dump and clean up.'''
  if not 'Error' in pluginResult: createBlockReport()
  printtime("Writing results.json...")
  jsonfile = os.path.join(pluginParams['results_dir'],"results.json")
  writeDictToJsonFile(pluginResult,jsonfile)

def plugin_main():
  '''Main entry point for script. Returns unix-like 0/1 for success/failure.'''
  try:
    loadPluginParams()
    printStartupMessage()
  except Exception, e:
    printerr("Failed to set up run parameters.")
    emsg = str(e)
    if emsg[:6] == 'CATCH:':
      emsg = emsg[6:]
      printlog('ERROR: %s'%emsg)
      createIncompleteReport(emsg)
      return 0
    else:
      traceback.print_exc()
      return 1
  try:
    if not pluginParams['cmdOptions'].skip_analysis:
      emptyResultsFolder()
    if pluginParams['barcoded']:
      runForBarcodes()
      if pluginReport['num_barcodes_processed'] == 0:
        createErrorReport("No barcode alignment files were found for this barcoded run.",True)
        return 1
      elif pluginReport['num_barcodes_processed'] == pluginReport['num_barcodes_failed']:
        createErrorReport("Analysis failed for all barcode alignments.")
        return 1
    else:
      runNonBarcoded()
    wrapup()
  except Exception, e:
    traceback.print_exc()
    wrapup()  # call only if suitable partial results are available, including some error status
    return 1
  return 0

if __name__ == "__main__":
  exit(plugin_main())

