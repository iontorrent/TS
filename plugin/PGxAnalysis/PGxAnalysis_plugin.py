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
import simplejson
import re
from ion.utils import compress

# set up for Django rendering with TSS installed apps
from django.conf import settings
from django.template.loader import render_to_string
from django.conf import global_settings
global_settings.LOGGING_CONFIG=None

#
# ----------- custom tag additions -------------
# Use template.base.add_to_builtins("django_tags") to add from django_tags.py (in cwd)
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
barcodeData = []
barcodeReport = {}
numPass =""
totalSamples=""
numAvgCov=""
numUniformity=""
variantCallerName=""
coverageAnalysisName=""
hotspotsFile=""
targetsFile=""
#
# -------------------- customize code for this plugin here ----------------
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
  parser.add_option('-d', '--scraper', help='Create a scraper folder of links to output files using name prefix (-P).', action="store_true", dest='scraper')
  parser.add_option('-k', '--keep_temp', help='Keep intermediate files. By default these are deleted after a successful run.', action="store_true", dest='keep_temp')
  parser.add_option('-l', '--log', help='Output extra progress Log information to STDERR during a run.', action="store_true", dest='logopt')
  parser.add_option('-s', '--skip_analysis', help='Skip re-generation of existing files but make new report.', action="store_true", dest='skip_analysis')
  parser.add_option('-x', '--stop_on_error', help='Stop processing barcodes after one fails. Otherwise continue to the next.', action="store_true", dest='stop_on_error')

  # add other run options here (these should override or account for things not in the json parameters file)

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


def addAutorunParams(plan=None):
  '''Additional parameter set up for automated runs, e.g. to add defaults for option only in the GUI.'''
  pass

def RunCommand( command , errorExit=1):
        global progName, haveBed, noerrWarn, noerrWarnFile
        command = command.strip()
        #WriteLog( " $ %s\n" % command )
        stat = os.system( command )
        if( stat ) != 0:
                if errorExit != 0:
                        sys.stderr.write( "ERROR: resource failed with status %d:\n" % stat )
                else:
                        sys.stderr.write( "INFO: return status %d:\n" % stat)
                        errorExit = 1
                sys.stderr.write( "$ %s\n" % command )
                sys.exit(errorExit)


def furbishPluginParams():
  '''Complete/rename/validate user parameters.'''
  # For example, HTML form posts do not add unchecked option values
  pass


def printStartupMessage():
  '''Output the standard start-up message. Customized for additional plugin input and options.'''
  printlog('')
  printtime('%s plugin running...' % pluginParams['plugin_name'])

  printlog('Alignment plugin run options:')
  if pluginParams['cmdOptions'].version != "":
    printlog('  Plugin version    : %s' % pluginParams['cmdOptions'].version)
  printlog('  Plugin start mode : %s' % pluginParams['start_mode'])
  printlog('  Run is barcoded   : %s' % ('Yes' if pluginParams['barcoded'] else 'No'))

  printlog('Data files used:')
  if pluginParams['cmdOptions'].logopt:
    printlog('  Parameters file   : %s' % pluginParams['jsonInput'])
  printlog('  Alignment file    : %s' % pluginParams['bamroot'])
  printlog('')


def updateBarcodeSummaryReport(autoRefresh=False):
  '''Create barcode summary (progress) report. Called before, during and after barcodes are being analysed.'''
  global barcodeData, numPass, numAvgCov, numUniformity, totalSamples, variantCallerName,coverageAnalysisName
  render_context = {
    "autorefresh" : autoRefresh,
    "run_name" : pluginParams['prefix'],
    "library_type" : "Ampliseq DNA",
    "genome_name" : "Human hg19",
    "targets_name" : os.path.basename(targetsFile),
    "hotspots_name" : os.path.basename(hotspotsFile),
    "numPass" : numPass,
    "numAvgCov" : numAvgCov,
    "numUniformity" : numUniformity,
    "totalSamples" : totalSamples,
    "variantCallerName" : variantCallerName,
    "run_name" : pluginParams['prefix'],
    "barcodeData" : barcodeData,
    "coverageAnalysisName" : coverageAnalysisName
  }
  # extra report items, e.g. file links from barcodes summary page
  if barcodeReport:
    render_context.update(barcodeReport)
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'barcode_summary.html', render_context )


def createIncompleteReport(errorMsg=""):
  '''Called to create an incomplete or error report page for non-barcoded runs.'''
  sample = pluginParams['sample_names'] if isinstance(pluginParams['sample_names'],basestring) else ''
  render_context = {
    "autorefresh" : (errorMsg == ""),
    "run_name": pluginParams['prefix'],
    "Sample_Name": sample,
    "Error": errorMsg }
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'incomplete.html', render_context )


def createDetailReport(resultData,reportData):
  '''Called to create the main report (for un-barcoded run or for each barcode).'''
  render_context = resultData.copy()
  render_context.update(reportData)
  createReport( os.path.join(pluginParams['output_dir'],pluginParams['report_name']), 'report.html', render_context )


def createBlockReport():
  '''Called at the end of run to create a block.html report. Use 'pass' if not wanted.'''
  printtime("Creating block report...")
  if pluginParams['barcoded']:
    render_context = { "barcode_results" : simplejson.dumps(barcodeSummary) }
    tplate = 'barcode_block.html'
  else:
    render_context = pluginResult.copy()
    render_context.update(pluginReport)
    tplate = 'report_block.html'
  createReport( pluginParams['block_report'], tplate, render_context )


#
# --------------- Base code for standard plugin runs -------------
#

def emptyResultsFolder():
  '''Purge everything in output folder except for specifically named files.'''
  # Dangerous - replace with something safer if it becomes obvious (e.g. putting output in subfolder?)
  logopt = pluginParams['cmdOptions'].logopt
  results_dir = pluginParams['results_dir']
  if results_dir == '/': return
  printlog("Purging old results...")
  for root,dirs,files in os.walk(results_dir,topdown=False):
    for name in files:
      # these are the exceptions - partial names and in the to level results
      if root == results_dir:
        start = os.path.basename(name)[:10]
        if start == "drmaa_stdo" or start == "ion_plugin" or start == "startplugi":
          continue
      fname = os.path.join(root,name)
      if logopt and root == results_dir:
        printlog("Removing file %s"%fname)
      os.system('rm -f "%s"'%fname)
    for name in dirs:
      fname = os.path.join(root,name)
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
        key = kvp[0].strip().replace(' ','_')
        ret[key] = line
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
  printlog( '(%s) %s'%(time.strftime('%X'),msg) )

def createlink(srcPath,destPath,srcFile=None,destFile=None ):
  # os.symlink to be performed using 2, 3 or 4 args (to track and avoid code repetition)
  if srcFile:
    srcPath = os.path.join(srcPath,srcFile)
  if destFile:
    destPath = os.path.join(destPath,destFile)
  elif srcFile:
    destPath = os.path.join(destPath,srcFile)
  if os.path.exists(destPath):
    os.unlink(destPath)
  if os.path.exists(srcPath):
    os.symlink(srcPath,destPath)
    if pluginParams['cmdOptions'].logopt:
      printlog("\nCreated symlink %s -> %s"%(destPath,srcPath))
    return True
  elif pluginParams['cmdOptions'].logopt:
    printlog("\nFailed to create symlink %s -> %s (source file does not exist)"%(destPath,srcPath))
  return False
  
def deleteTempFiles(tmpFiles):
  if tmpFiles == None or pluginParams['cmdOptions'].keep_temp:
    return
  output_dir = pluginParams['output_dir']
  for filename in tmpFiles:
    flist = glob( os.path.join(output_dir,filename) )
    for f in flist:
      if pluginParams['cmdOptions'].logopt:
        printlog("Deleting file %s"%f)
      os.unlink(f)

def createReport(reportName,reportTemplate,reportData):
  with open(reportName,'w') as bcsum:
    bcsum.write( render_to_string(reportTemplate,reportData) )

def sampleNames():
  try:
    if pluginParams['barcoded']:
      samplenames = {}
      bcsamps = pluginParams['jsonParams']['plan']['barcodedSamples']
      if isinstance(bcsamps,basestring):
        bcsamps = json.loads(bcsamps)
      for bcname in bcsamps:
        for bc in bcsamps[bcname]['barcodes']:
          samplenames[bc] = bcname
    else:
      samplenames = jsonParams['expmeta']['sample']
  except:
    return ""
  return samplenames

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
  pluginParams['hotspots_file'] = jsonParams['plan'].get('regionfile', '.') 
  pluginParams['regions_file'] = jsonParams['plan'].get('bedfile', '.') 
  
  # some things not yet in startplugin.json are provided or over-writen by cmd args
  copts = pluginParams['cmdOptions']
  pluginParams['reference'] = copts.reference if copts.reference != "" else jsonParams['runinfo'].get('reference','')
  pluginParams['bamroot']   = copts.bamfile   if copts.bamfile != "" else '%s/rawlib.bam' % pluginParams['analysis_dir']
  pluginParams['prefix']    = copts.prefix    if copts.prefix != "" else pluginParams['analysis_name']
  pluginParams['results_url'] = copts.results_url if copts.results_url != "" else os.path.join(
  jsonParams['runinfo'].get('url_root','.'),'plugin_out',pluginParams['plugin_name']+'_out' )

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
  if 'launch_mode' in config:
    pluginParams['start_mode'] = 'Manual start'
  elif 'plan' in jsonParams:
    pluginParams['start_mode'] = 'Autostart with plan configuration'
    addAutorunParams(jsonParams['plan'])
  else:
    pluginParams['start_mode'] = 'Autostart with default configuration'
    addAutorunParams()

  # plugin configuration becomes basis of results.json file
  global pluginResult, pluginReport
  pluginResult = pluginParams['config'].copy()
  pluginResult['barcoded'] = pluginParams['barcoded']
  if pluginParams['barcoded']:
    pluginResult['barcodes'] = {}
    pluginReport['barcodes'] = {}

  # configure django to use the templates folder and various installed apps
  if not settings.configured:
    settings.configure( DEBUG=False, TEMPLATE_DEBUG=False,
      INSTALLED_APPS=('django.contrib.humanize',),
      TEMPLATE_DIRS=(os.path.join(pluginParams['plugin_dir'],'templates'),) )


def writeDictToJsonFile(data,filename):
  with open(filename,'w') as outfile:
    json.dump(data,outfile)

def testRun(outdir,prefix):
  # default for testing framework
  testout = os.path.join(outdir,prefix+"_test.out")
  with open(testout,'w') as f:
    f.write("This is a test file.\n")
  printlog('Created %s'%testout)

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
  numGoodBams = 0
  numUnalBams = 0
  minFileSize = pluginParams['cmdOptions'].minbamsize
  (bcBamPath,bcBamRoot) = os.path.split(pluginParams['bamroot'])
  validBarcodes = []
  for barcode in barcodes:
    # use unmapped BAM if there else mapped BAM (unmapped may not be present on Proton)
    bcbam = os.path.join( bcBamPath, "%s_%s"%(barcode,bcBamRoot) )
    if not os.path.exists(bcbam):
      bcbam = os.path.join( pluginParams['analysis_dir'], "%s_rawlib.bam"%barcode )
      numUnalBams += 1
    if not os.path.exists(bcbam):
      bcbam = ": BAM file not found"
      numUnalBams -= 1
    elif os.stat(bcbam).st_size < minFileSize:
      bcbam = ": BAM file too small"
    else:
      numGoodBams += 1
      validBarcodes.append(barcode)

  printlog("Processing %d barcodes...\n" % numGoodBams)
  if numUnalBams > 0:
    printlog("Warning: %d barcodes will be processed using mapped BAM files. (Unmapped BAMs were not available.)\n" % numUnalBams)
  pluginReport['num_barcodes_processed'] = numGoodBams
  pluginReport['num_barcodes_failed'] = 0

  # iterate over all barcodes and process the valid ones
  skip_analysis = pluginParams['cmdOptions'].skip_analysis
  stop_on_error = pluginParams['cmdOptions'].stop_on_error
  create_scraper = pluginParams['cmdOptions'].scraper
  sample_names = pluginParams['sample_names']
  postout = False; # just for logfile prettiness
  sampleNamesFile = ("%s/sampleNames.txt" % pluginParams['results_dir'])
  sampleNamesFW = open(sampleNamesFile, 'w')
  for barcode in barcodes:
    sample = sample_names[barcode] if barcode in sample_names else ''
    sampleNamesFW.write("%s:%s\n" %(barcode, sample))
  sampleNamesFW.close()
  binDir = os.path.join(pluginParams['plugin_dir'], 'bin')
  outDir = pluginParams['results_dir']
  analysisDir = pluginParams['analysis_dir']
  global hotspotsFile
  hotspotsFile = pluginParams['hotspots_file']
  global targetsFile
  targetsFile = pluginParams['regions_file']
  printlog("hotspots file is %s " %hotspotsFile)
  pluginOutDir = os.path.join(analysisDir, 'plugin_out')
  global variantCallerName 
  if not filter(re.compile(r'variantCaller_out*').search, os.listdir(pluginOutDir)):
	printerr("Variant Caller plugin has to be run before launching the PGX Analysis plugin. Please run Torrent Variant Caller plugin")
	return
  variantCallerName = max(filter(re.compile(r'variantCaller_out*').search, os.listdir(pluginOutDir)))
  global coverageAnalysisName
  if not filter(re.compile(r'coverageAnalysis_out*').search, os.listdir(pluginOutDir)):
	printerr("Coverage Analysis plugin has to be run before launching the PGX Analysis plugin. Please run the Coverage Analysis plugin")	
	return
  coverageAnalysisName = max(filter(re.compile(r'coverageAnalysis_out*').search, os.listdir(pluginOutDir)))  
  printlog(variantCallerName)
  printlog(coverageAnalysisName)
  variantCallerDir = os.path.join(pluginOutDir, variantCallerName)
  printlog("variantcaller dir is %s" % variantCallerDir)
  coverageAnalysisDir = os.path.join(pluginOutDir, coverageAnalysisName)
  
  hotspotsFileVC = ""
  resultsJsonFile = os.path.join(variantCallerDir, "results.json")
  if not os.path.isfile(resultsJsonFile):
	printerr("VariantCaller results are not ready. Please wait for the variant Caller plugin to finish and then launch the PGx plugin")
        return

  covAnalysisResultsJsonFile = os.path.join(coverageAnalysisDir, "results.json")
  if not os.path.isfile(covAnalysisResultsJsonFile):
	printerr("Coverage Analysis results are not ready. Please wait for the Coverage Analysis plugin to finish and then launch the PGx plugin")
        return
  targetsFileVC = ""
  with open(resultsJsonFile) as fin:
	for line in fin:
		if "hotspots_bed" in line and ":" in line and "type" not in line :
			kvp = line.split(":")
			hotspotsFileVC = (os.path.basename(kvp[1].strip()))
			if "," in hotspotsFileVC:
				hotspotsFileVC = hotspotsFileVC[:-2]
			else:
				hotspotsFileVC = hotspotsFileVC[:-1]
			hotspotsFileVC = os.path.join(variantCallerDir, hotspotsFileVC)
		if "targets_bed" in line and ":" in line and "type" not in line :
			kvp = line.split(":")
			targetsFileVC = (os.path.basename(kvp[1].strip()))
			if "," in targetsFileVC:
				targetsFileVC = targetsFileVC[:-2]
			else:
				targetsFileVC = targetsFileVC[:-1]
				
			targetsFileVC = os.path.join(variantCallerDir, targetsFileVC)
  if not hotspotsFileVC:
	printerr("Cannot obtain the hotspots file used by the VariantCaller. Trying to obtain the hotspots file from plan")
  else:
	 hotspotsFile = hotspotsFileVC
  if not hotspotsFile:
	printerr("The plan is not set up with a hotspots file.")
	return
  
  if not targetsFileVC:
	printerr("Cannot obtain the Target Regions file used by the VariantCaller. Trying to obtain the regions file from plan")
  else:
	 targetsFile = targetsFileVC
 
  cmd = ("java -jar %s/PGX_Analysis.jar %s %s %s %s %s %s %s %s" % (binDir, hotspotsFile, outDir, bcfileName, analysisDir, variantCallerDir, coverageAnalysisDir, binDir, sampleNamesFile));
 
  printlog(cmd) 
  RunCommand(cmd);

  # parse out data in results text file to dict AND coverts spaces to underscores in keys to avoid Django issues
  statsfile = 'summary.txt'
  analysisData = parseToDict( os.path.join(outDir,statsfile), "\t" )
  global numPass, numUniformity, numAvgCov, totalSamples  
  totalSamples = numGoodBams
  numPass = numGoodBams
  numAvgCov = 0
  numUniformity = 0
  for keys,values in analysisData.items():
    printlog(keys)
    printlog(values)
  for file in os.listdir("%s/cnvCalls" %outDir):
	if file.endswith(".log"):
		cnvCallsDir = os.path.join(outDir,"cnvCalls")
		filein = os.path.join(cnvCallsDir, file)
    		printlog("filein is %s " % filein)
		with open(filein) as fin:
			sep = "="
      			for line in fin:
				if("valid Samples =" in line):
					kvp = line.split(sep);
					totalSamples = kvp[1].strip()
				elif("CNV Calling =" in line):
					kvp = line.split(sep);
					numPass = kvp[1].strip()
				elif("Average coverage" in line):
					kvp = line.split(sep);
					numAvgCov = kvp[1].strip()
				elif("Uniformate Rate" in line):
					kvp = line.split(sep);
					numUniformity = kvp[1].strip()
  zipfilename = "%s/cnvExports.zip" % outDir
  cnvExportsDir = "%s/cnvExports" % outDir 
  for file in os.listdir(cnvExportsDir):
	if file.endswith("_cn.txt"):
		filein = os.path.join(cnvExportsDir, file) 
		compress.make_zip(zipfilename, filein, arcname=os.path.basename(filein), use_sys_zip = False)
 
  vcfZipFilename = "%s/%s.vcf.zip" % (outDir, pluginParams['prefix'])
  mergedVcfsDir = "%s/merged_VCFs" % outDir
  for file in os.listdir(mergedVcfsDir):
	if file.endswith(".gz") or file.endswith(".tbi"):
		filein = os.path.join(mergedVcfsDir, file) 
		compress.make_zip(vcfZipFilename, filein, arcname=os.path.basename(filein), use_sys_zip = False)
   
  
  global barcodeData
 
  for barcode in validBarcodes:
  	barcode_entry = {}
  	sample = sample_names[barcode] if barcode in sample_names else ''
  	barcode_entry['name'] = barcode
	if barcode in analysisData:
		barcodeLine = analysisData[barcode]
		kvp = barcodeLine.split("\t")
        	#key = kvp[0].strip()
			
		if sample=='':	
			barcode_entry['sample'] = 'none'
		else:
			barcode_entry['sample'] = sample		
		if len(kvp) < 8 and kvp[2].strip() == 'null':
			barcode_entry['hotspots_variants_total'] = "none" 
			barcode_entry['novel_variants_total'] = "none"
			barcode_entry['exon9_cnv'] = kvp[3].strip()
			barcode_entry['gene_cnv'] = kvp[4].strip()
			barcode_entry['exon9_cnv_confidence'] = kvp[5].strip()
			barcode_entry['gene_cnv_confidence'] = kvp[6].strip()

		else:
			barcode_entry['hotspots_variants_total'] = "%d/%s" %(int(kvp[2].strip()) - int(kvp[4].strip()) - int(kvp[5].strip()), kvp[2].strip())
			barcode_entry['novel_variants_total'] = int(kvp[3].strip()) - int(kvp[2].strip())		
			barcode_entry['exon9_cnv'] = kvp[6].strip()
			barcode_entry['gene_cnv'] = kvp[7].strip()
			barcode_entry['exon9_cnv_confidence'] = kvp[8].strip()
			barcode_entry['gene_cnv_confidence'] = kvp[9].strip()
	
  		barcodeData.append(barcode_entry)

  updateBarcodeSummaryReport()

  if create_scraper:
    createScraperLinksFolder( pluginParams['results_dir'], pluginParams['prefix'] )
  #createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'block_barcodes.html', render_context )

def runNonBarcoded():
  printerr('Analysis not supported for non-barcoded runs')
  pluginResult.update({ 'Error': 'Analysis not supported for non-barcoded runs' })
  createIncompleteReport('Analysis not supported for non-barcoded runs')
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
  #if not 'Error' in pluginResult: createBlockReport()
  printtime("Writing results.json...")
  writeDictToJsonFile(pluginResult,os.path.join(pluginParams['results_dir'],"results.json"))

def plugin_main():
  '''Main entry point for script. Returns unix-like 0/1 for success/failure.'''
  try:
    loadPluginParams()
    printStartupMessage()
  except Exception, e:
    printerr("Failed to set up run parameters.")
    traceback.print_exc()
    return 1
  try:
    if not pluginParams['cmdOptions'].skip_analysis:
      emptyResultsFolder()
    if pluginParams['barcoded']:
      runForBarcodes()
      if pluginReport['num_barcodes_processed'] == 0:
      	printlog("WARNING: No barcode alignment files were found for this barcoded run.")
      elif pluginReport['num_barcodes_processed'] == pluginReport['num_barcodes_failed']:
      	printlog("ERROR: Analysis failed for all barcode alignments.")
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

