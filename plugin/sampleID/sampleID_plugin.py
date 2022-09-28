#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

# NOTE: This plugin uses litle of barcodes.json as it makes its own alignments for sample tracking targets

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
#
from django import template
register = template.Library()

@register.filter 
def toMega(value):
  return float(value) / 1000000

template.builtins.append(register) 

# defines exceptional barcode IDs to check for
NONBARCODED = "nonbarcoded"
NOMATCH = "nomatch"

# global data collecters common to functions
barcodeInput = {}
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
  pass


def furbishPluginParams():
  '''Complete/rename/validate user parameters.'''
  # For example, HTML form posts do not add unchecked option values
  pass


def configReport():
  '''Returns a dictionary based on the plugin config parameters that as reported in results.json.'''
  # This is to avoid outputting hidden or aliased values. If not needed just pass back a copy of config.
  #return pluginParams['config'].copy()
  config = pluginParams['config']
  return {
    "Launch Mode" : config['launch_mode'],
    "Reference Genome" : pluginParams['genome_id']
  }


def printStartupMessage():
  '''Output the standard start-up message. Customized for additional plugin input and options.'''
  printlog('')
  printtime('Started %s' % pluginParams['plugin_name'])
  config = pluginParams['config']
  printlog('Run configuration:')
  printlog('  Plugin version:   %s' % pluginParams['cmdOptions'].version)
  printlog('  Launch mode:      %s' % config['launch_mode'])
  printlog('  Parameters:       %s' % pluginParams['jsonInput'])
  printlog('  Barcodes:         %s' % pluginParams['barcodeInput'])
  printlog('  Output folder:    %s' % pluginParams['results_dir'])
  printlog('  Output file stem: %s' % pluginParams['prefix'])
  printlog('  Reference genome: %s' % pluginParams['genome_id'])
  printlog('  Tracking set ID:  %s' % pluginParams['track_set'])
  printlog('')


def run_plugin(skiprun=False,barcode=""):
  '''Wrapper for making command line calls to perform the specific plugin analyis.'''
  # first part is pretty much boiler plate - grab the key parameters that most plugins use
  logopt = pluginParams['cmdOptions'].logopt
  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['output_dir']
  output_url = pluginParams['output_url']
  output_prefix = pluginParams['output_prefix']
  config = pluginParams['config']
  barcodeData = barcodeInput[barcode]
  sample = barcodeData.get('sample','None')
  reference = barcodeData.get('reference_fullpath','')

  # link from source BAM since pipeline uses the name as output file stem
  linkbam = os.path.join(output_dir,output_prefix+".bam")
  createlink(barcodeData['bam_filepath'],linkbam)
  createlink(barcodeData['bam_filepath']+'.bai',linkbam+'.bai')
  bamfile = linkbam

  # for legacy createDetailReport() code
  pluginParams['bamfile'] = bamfile
  pluginParams['genome_id'] = barcodeData['reference']

  # Hard-coded path to sample ID target BED file - may be user definable later
  track_set = pluginParams['track_set']
  regionsBed = os.path.join(plugin_dir,'targets',track_set+'_sampleID_regions.bed')
  lociBed = os.path.join(plugin_dir,'targets',track_set+'_sampleID_loci.bed')

  # skip the actual and assume all the data already exists in this file for processing
  if skiprun:
    printlog("Skipped analysis - generating report on in-situ data")
  else:
    # option for cmd-line detailed HTML report generation
    printtime("Running coverage analysis pipeline...")
    runcmd = '%s %s -D "%s" -F "%s" -N "%s" "%s" "%s" "%s" "%s"' % (
        os.path.join(plugin_dir,'run_sampleid.sh'), pluginParams['logopt'], output_dir,
        pluginParams['prefix'], sample, reference, bamfile, regionsBed, lociBed )
    if logopt: printlog('\n$ %s\n'%runcmd)
    if( os.system(runcmd) ):
      raise Exception("Failed running run_sampleid.sh. Refer to Plugin Log.")

  if pluginParams['cmdOptions'].cmdline: return ({},{})
  printtime("Generating report...")
  try:
    # Link report page resources. This is necessary as the plugin code is inaccesible from URLs directly.
    createlink( os.path.join(plugin_dir,'slickgrid'), output_dir )
    createlink(os.path.join(plugin_dir, 'scripts'), output_dir)
    createlink(os.path.join(plugin_dir, 'css'), output_dir)
    createlink(os.path.join(plugin_dir, 'kendo'), output_dir)
    createlink( os.path.join(plugin_dir,'lifegrid'), output_dir )
    createlink( os.path.join(plugin_dir,'scripts','igv.php3'), output_dir )
    createlink( regionsBed, os.path.join(output_dir,'tracking_regions.bed') )
    createlink( lociBed, os.path.join(output_dir,'tracking_loci.bed') )
    createlink(os.path.join(plugin_dir, 'scripts'), pluginParams['results_dir'])
    createlink(os.path.join(plugin_dir, 'css'), pluginParams['results_dir'])
    createlink(os.path.join(plugin_dir, 'kendo'), pluginParams['results_dir'])
  except Exception as Err:
    printtime("Symbolic linking Error: %s", Err)

  # Optional: Delete intermediate files after successful run. These should not be required to regenerate any of the
  # report if the skip-analysis option. Temporary file deletion is also disabled when the --keep_temp option is used.
  #deleteTempFiles([ '*.bam', '*.bam.bai', '*.bed' ])

  # Parse out stats from results text file to dict AND convert unacceptible characters to underscores in keys to avoid Django issues
  resultData = parseToDict( os.path.join(output_dir,'read_stats.txt'), ":" )
  resultData['target_cov'] = parseToDict( os.path.join(output_dir,'on_target_stats.txt'), ":" )
  resultData['loci_cov'] = parseToDict( os.path.join(output_dir,'on_loci_stats.txt'), ":" )

  # Collect other output data to pluginReport, which is anything else that is used to generate the report
  reportData = {
    "run_name" : output_prefix,
    "barcode_name" : barcode,
    "output_dir" : output_dir,
    "output_url" : output_url,
    "output_prefix" : output_prefix,
    "allele_table" : 'allele_counts.xls'
  }
  if not pluginParams['cmdOptions'].isDx:
    reportData["file_links"] = 'filelinks.xls'
  return (resultData,reportData)


def run_meta_plugin():
  '''Create barcode x target reads matrix and text version of barcode summary table (after analysis completes for individual barcodes).'''
  if pluginParams['cmdOptions'].cmdline: return
  printlog("")
  printtime("Collating barcodes summary data...")
  bctable = []
  bcresults = pluginResult['barcodes']
  bcreports = pluginReport['barcodes']
  # iterate barcodeSummary[] to maintain barcode processing order
  for bcdata in barcodeSummary:
    bcname = bcdata['barcode_name']
    bcrep = bcreports[bcname]
    bcline = "%s\t%s\t%s\t%s\t%s"%(bcname,bcdata['sample'],bcdata['sample_id'],bcdata['coverage'],bcdata['pc_tracking_reads'])
    bctable.append(bcline)

  if len(bctable) > 0:
    bctabfile = pluginParams['prefix']+".bc_summary.xls"
    bcline = "Barcode ID\tSample Name\tSample ID\tCoverage\tTracking Reads"
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
    reportData = pluginReport['barcodes'][barcode]
    errMsg = resultData.get('Error','')
    sample = resultData['Sample Name']
    if sample == '': sample = 'None'
    # barcodes_json dictionary is firm-coded in Kendo table template that we are using for main report styling
    if errMsg != "":
      detailsLink = "<span class='help' title='%s' style='color:red'>%s</span>" % ( errMsg, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "sample_ID" : "NA",
        "coverage": "NA",
        "pc_tracking_reads": "NA"
      })
    else:
      detailsLink = "<a target='_parent' href='%s' class='help'><span title='Click to view the detailed report for barcode %s'>%s</span><a>" % (
        os.path.join(barcode,pluginParams['report_name']), barcode, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "sample_id" : resultData['Sample ID'],
        "coverage": resultData['loci_cov']['Coverage at 20x'],
        "pc_tracking_reads": resultData['Percent reads in sample ID regions']
      })
  render_context = {
    "autorefresh" : autoRefresh,
    "isDx": pluginParams['cmdOptions'].isDx,
    "run_name" : pluginParams['prefix'],
    "barcode_results" : simplejson.dumps(barcodeSummary)
  }
  # extra report items, e.g. file links from barcodes summary page
  if barcodeReport:
    render_context.update(barcodeReport)
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'barcode_summary.html', render_context )


def createIncompleteReport(errorMsg=""):
  '''Called to create an incomplete or error report page for non-barcoded runs.'''
  if pluginParams['barcoded']:
    sample = 'None'
  else:
    barcodeData = barcodeInput[NONBARCODED]
    if barcodeData: sample = barcodeData.get('sample','None')
    if sample == 'none': sample = 'None'
  render_context = {
    "autorefresh" : (errorMsg == ""),
    "run_name": pluginParams['prefix'],
    "Sample_Name": sample,
    "Error": errorMsg }
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'incomplete.html', render_context )


def createDetailReport(resultData,reportData):
  '''Called to create the main report (for un-barcoded run or for each barcode).'''
  if pluginParams['cmdOptions'].cmdline: return
  logopt = pluginParams['cmdOptions'].logopt
  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['output_dir']
  output_prefix = pluginParams['output_prefix']
  html_report = os.path.join(output_dir,pluginParams['report_name'])
  render_context = resultData.copy()
  reportData["isDx"] = pluginParams['cmdOptions'].isDx
  render_context.update(reportData)
  createReport( html_report, 'report.html', render_context )

  # PDF has to be created here, after report created (w/o File Links) and before filelinks file is created
  pdf_report = output_prefix+'.sampleID.pdf'
  runcmd = '%s --load-error-handling ignore --no-background "%s" "%s"' % (
    os.path.join(plugin_dir,'bin','wkhtmltopdf-amd64'), html_report, os.path.join(output_dir,pdf_report) )
  if logopt: printlog('\n$ %s\n'%runcmd)
  xcmd = Popen( [os.path.join(plugin_dir,'bin','wkhtmltopdf-amd64'), '--load-error-handling', 'ignore',
    html_report, os.path.join(output_dir,pdf_report)], shell=False, stdout=PIPE, stderr=PIPE )
  xcmd.communicate()
  if xcmd.poll():
    printlog("Warning: Failed to PDF report file.")
    pdf_report = ''

  # Create file links file - here so File Links table does not appear in PDF (consisent with previous version)
  with open(os.path.join(pluginParams['output_dir'],'filelinks.xls'),"w") as flinks:
    #if pdf_report != '':
    flinks.write( 'Text\tLink\n' )
    flinks.write( 'Download a hard-copy A4 PDF image of this report page.\t'+pdf_report+'\n' )
    flinks.write( 'Download all variant calls as a table file.\tallele_counts.xls\n' )
    flinks.write( 'Download the tracking target regions file.\ttracking_regions.bed\n' )
    flinks.write( 'Download the tracking loci regions (SNPs) file.\ttracking_loci.bed\n' )
    flinks.write( 'Download the aligned tracking reads (BAM) file.\t'+output_prefix+'.bam\n' )
    flinks.write( 'Download the aligned tracking reads index (BAI) file.\t'+output_prefix+'.bam.bai\n' )

  # Create auxiliary report files, e.g. igv_session file for linking out Allele Coverage table to IGV
  runcmd = '%s -r "%s" -b "%s" -g "%s" -s igv_session.xml' % ( os.path.join(plugin_dir,'scripts','create_igv_link.py'),
    output_dir, pluginParams['bamfile'], pluginParams['genome_id'] )
  if logopt: printlog('\n$ %s\n'%runcmd)
  if( os.system(runcmd) ):
    printlog("Warning: Failed to create igv_session.xml file.")

  # Re-create report with file links added
  render_context.update( {"full_report" : 'yes'} )
  createReport( html_report, 'report.html', render_context )

def createBlockReport():
  '''Called at the end of run to create a block.html report. Use 'pass' if not wanted.'''
  if pluginParams['cmdOptions'].cmdline: return
  printtime("Creating block report...")
  if pluginParams['barcoded']:
    render_context = {
      "run_name" : pluginParams['prefix'],
      "barcode_results" : simplejson.dumps(barcodeSummary)
    }
    tplate = 'barcode_block.html'
  else:
    render_context = pluginResult.copy()
    render_context.update(pluginReport)
    tplate = 'report_block.html'
  createReport( pluginParams['block_report'], tplate, render_context )


def createProgressReport(progessMsg,last=False):
  '''General method to write a message directly to the block report, e.g. when starting prcessing of a new barcode.'''
  createReport( pluginParams['block_report'], "progress_block.html", { "progress_text" : progessMsg, "refresh" : "" if last else "refresh" } )


#
# --------------- Base code for standard plugin runs -------------
#

def getOrderedBarcodes():
  if NONBARCODED in barcodeInput:
    return []
  barcodes = {}
  for barcode in barcodeInput:
    if barcode == NOMATCH: continue
    barcodes[barcode] = barcodeInput[barcode]["barcode_index"]
  return sorted(barcodes,key=barcodes.get)

def parseCmdArgs():
  '''Process standard command arguments. Customized for additional debug and other run options.'''
  # standard run options here - do not remove
  parser = OptionParser()
  parser.add_option('-V', '--version', help='Version string for tracking in output', dest='version', default='')
  parser.add_option('-X', '--min_bc_bam_size', help='Minimum file size required for barcode BAM processing', type="int", dest='minbamsize', default=0)
  parser.add_option('-c', '--cmdline', help='Run command line only. Reports will not be generated using the HTML templates.', action="store_true", dest='cmdline')
  parser.add_option('-d', '--scraper', help='Create a scraper folder of links to output files using name prefix (-P).', action="store_true", dest='scraper')
  parser.add_option('-k', '--keep_temp', help='Keep intermediate files. By default these are deleted after a successful run.', action="store_true", dest='keep_temp')
  parser.add_option('-l', '--log', help='Output extra progress Log information to STDERR during a run.', action="store_true", dest='logopt')
  parser.add_option('-p', '--purge_results', help='Remove all folders and most files from output results folder.', action="store_true", dest='purge_results')
  parser.add_option('-s', '--skip_analysis', help='Skip re-generation of existing files but make new report.', action="store_true", dest='skip_analysis')
  parser.add_option('-x', '--stop_on_error', help='Stop processing barcodes after one fails. Otherwise continue to the next.', action="store_true", dest='stop_on_error')
  parser.add_option('-i', '--isDx', help='platform specific environmental variable', dest='isDx', default='')

  (cmdOptions, args) = parser.parse_args()
  if( len(args) != 2 ):
    printerr('Takes two file arguments: startplugin.json barcodes.json')
    raise TypeError(os.path.basename(__file__)+" takes exactly two arguments (%d given)."%len(args))
  with open(args[0]) as jsonFile:
    jsonParams = json.load(jsonFile)
  global pluginParams, barcodeInput
  with open(args[1]) as jsonFile:
    barcodeInput = json.load(jsonFile)
  pluginParams['cmdOptions'] = cmdOptions
  pluginParams['jsonInput'] = args[0]
  pluginParams['barcodeInput'] = args[1]
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
  # -nf prevents both warning and odd behavior where target exists and is a directory
  os.system('ln -snf "%s" "%s"'%(srcPath,destPath))
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

  # get FILEPATH_OUTPUT_STEM or create old default if not available
  pluginParams['prefix'] = jsonParams['expmeta'].get('output_file_name_stem','')
  if not pluginParams['prefix']:
    pluginParams['prefix'] = jsonParams['expmeta'].get('run_name','auto')
    if 'results_name' in jsonParams['expmeta']:
      pluginParams['prefix'] += "_" + jsonParams['expmeta']['results_name']

  # TODO: replace this with url_plugindir when available from startplugin.json
  resurl = jsonParams['runinfo'].get('results_dir','.')
  if pluginParams['cmdOptions'].isDx:
    plgpos = resurl.find('plugins')
  else:
    plgpos = resurl.find('plugin_out')
  if plgpos >= 0:
    resurl = os.path.join( jsonParams['runinfo'].get('url_root','.'), resurl[plgpos:] )
  pluginParams['results_url'] = resurl

  pluginParams['barcoded'] = NONBARCODED not in barcodeInput

  # disable run skip if no report exists => plugin has not been run before
  pluginParams['report_name'] = pluginParams['plugin_name']+'.html'
  pluginParams['block_report'] = os.path.join(pluginParams['results_dir'],pluginParams['plugin_name']+'_block.html')
  if not os.path.exists( os.path.join(pluginParams['results_dir'],pluginParams['report_name']) ):
    if pluginParams['cmdOptions'].skip_analysis and not pluginParams['cmdOptions'].cmdline:
      printlog("Warning: Skip analysis option ignorred as previous output appears to be missing.")
      pluginParams['cmdOptions'].skip_analysis = False

  # set up plugin specific options depending on auto-run vs. plan vs. GUI
  config = pluginParams['config'] = jsonParams['pluginconfig'].copy() if 'pluginconfig' in jsonParams else {}
  launchmode = config.get('launch_mode','')
  if launchmode == 'Manual':
    furbishPluginParams()
  elif 'plan' in jsonParams:
    # assume that either plan.html or config.html has partially defined the config if launch_mode is defined
    if launchmode:
      furbishPluginParams()
    else:
      config['launch_mode'] = 'Autostart with plan configuration'
    addAutorunParams(jsonParams['plan'])
  else:
    config['launch_mode'] = 'Autostart with default configuration'
    addAutorunParams()

  # check for non-supported reference alignments
  refid = pluginParams['genome_id']
  if refid.startswith("hg19") or refid.startswith("GRCh37"):
    pluginParams['track_set'] = "KIDDAME"
  elif refid.startswith("GRCh38") or refid.startswith("hg38"):
    pluginParams['track_set'] = "hg38_KIDDAME"
  else:
    pluginParams['track_set'] = "KIDDAME"
    printlog("WARNING: This plugin is intended to be used with reads compatible with a hg19 or GRCh38 reference.")

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

def runNonBarcoded():
  global pluginResult, pluginReport
  try:
    ensureFilePrefix()
    barcodeData = barcodeInput[NONBARCODED]
    sample = barcodeData.get('sample','')
    sampleTag = ' (%s)'%sample if sample else ''
    printlog("\nProcessing nonbarcoded%s...\n" % sampleTag)
    createIncompleteReport()
    pluginParams['output_dir'] = pluginParams['results_dir']
    pluginParams['output_url'] = pluginParams['results_url']
    pluginParams['output_prefix'] = pluginParams['prefix']
    (resultData,pluginReport) = run_plugin( pluginParams['cmdOptions'].skip_analysis, NONBARCODED )
    pluginResult.update(resultData)
    createDetailReport(pluginResult,pluginReport)
  except Exception, e:
    errMsg = str(e)
    createIncompleteReport(errMsg)
    if not errMsg.startswith("CATCH:"): raise
  if pluginParams['cmdOptions'].scraper:
    createScraperLinksFolder( pluginParams['output_dir'], pluginParams['output_prefix'] )

def runForBarcodes():
  # iterate over listed barcodes to pre-test barcode files
  global pluginParams, pluginResult, pluginReport
  barcodes = getOrderedBarcodes()

  numGoodBams = 0
  numAnalBams = 0
  maxBarcodeLen = 0
  numInvalidBarcodes = 0
  minFileSize = pluginParams['cmdOptions'].minbamsize
  barcodeIssues = []
  for barcode in barcodes:
    barcodeData = barcodeInput[barcode]
    bcbam = barcodeData['bam_filepath']
    errMsg = ''
    if not barcodeData['aligned']:
      errMsg = "BAM file is not aligned (required for samtools pileup)"
    elif barcodeData['filtered']:
      errMsg = "Filtered (not enough reads)"
    elif not os.path.exists(bcbam):
      errMsg = "BAM file not found at " + bcbam
    elif os.stat(bcbam).st_size < minFileSize:
      errMsg = "BAM file too small"
    else:
      if( len(barcode) > maxBarcodeLen ):
        maxBarcodeLen = len(barcode)
      numGoodBams += 1
      # match former behavior of tracking whether BAM was aligned
      if barcodeData.get('aligned',False):
        numAnalBams += 1
    barcodeIssues.append(errMsg)

  ensureFilePrefix(maxBarcodeLen+1)
  pluginReport['num_barcodes_processed'] = numGoodBams
  pluginReport['num_barcodes_invalid'] = numInvalidBarcodes
  pluginReport['num_barcodes_failed'] = 0

  skip_analysis = pluginParams['cmdOptions'].skip_analysis
  stop_on_error = pluginParams['cmdOptions'].stop_on_error
  create_scraper = pluginParams['cmdOptions'].scraper

  # create initial (empty) barcodes summary report
  printlog("Processing %d barcodes...\n" % numGoodBams)
  updateBarcodeSummaryReport("",True)

  # iterate over all barcodes and process the valid ones
  postout = False; # just for logfile prettiness
  barcodeProcessed = 0
  for barcode in barcodes:
    barcodeData = barcodeInput[barcode]
    sample = barcodeData.get('sample','')
    sampleTag = ' (%s)'%sample if sample else ''
    barcodeError = barcodeIssues.pop(0)
    if barcodeError:
      if postout:
        postout = False
        printlog("")
      printlog("Skipping %s%s: %s" % (barcode,sampleTag,barcodeError))
      # for error messages to appear in barcode table
      perr = barcodeError.find('ERROR:')+6
      if perr >= 6:
        pluginResult['barcodes'][barcode] = { "Sample Name" : sample, "Error" : barcodeError[perr:].strip() }
        pluginReport['barcodes'][barcode] = {}
        updateBarcodeSummaryReport(barcode,True)
    else:
      try:
        postout = True
        printlog("\nProcessing %s%s...\n" % (barcode,sampleTag))
        pluginParams['output_dir'] = os.path.join(pluginParams['results_dir'],barcode)
        pluginParams['output_url'] = os.path.join(pluginParams['results_url'],barcode)
        pluginParams['output_prefix'] = barcode+"_"+pluginParams['prefix']
        if not os.path.exists(pluginParams['output_dir']):
           os.makedirs(pluginParams['output_dir'])
        barcodeProcessed += 1
        createProgressReport( "Processing barcode %d of %d..." % (barcodeProcessed,numGoodBams) )
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

  createProgressReport( "Compiling barcode summary report...", True )
  run_meta_plugin()
  updateBarcodeSummaryReport("")
  if create_scraper:
    createScraperLinksFolder( pluginParams['results_dir'], pluginParams['prefix'] )

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
      createProgressReport("Analysis failed.")
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
        createIncompleteReport("No barcode alignment files were found for this barcoded run.")
        createProgressReport("No barcode alignment files were found.")
        return 1
      elif pluginReport['num_barcodes_processed'] == pluginReport['num_barcodes_failed']:
        createIncompleteReport("Analysis failed for all barcode alignments.")
        createProgressReport("Analysis failed for all barcodes.")
        return 1
    else:
      runNonBarcoded()
    createProgressReport( "Analysis completed successfully." )
    wrapup()
  except Exception, e:
    traceback.print_exc()
    wrapup()  # call only if suitable partial results are available, including some error status
    return 1
  return 0

if __name__ == "__main__":
  exit(plugin_main())

