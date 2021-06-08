#!/usr/bin/env python
# Copyright (C) 2019 Thermo Fisher Scientific, Inc. All Rights Reserved.

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
import copy

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
def lt(a,b):
  return float(a) < float(b);
@register.filter 
def gt(a,b):
  return float(a) > float(b);

template.builtins.append(register) 

# defines exceptional barcode IDs to check for
NONBARCODED = "nonbarcoded"
NOMATCH = "nomatch"

# global data collecters common to functions
pluginParams = {}
pluginResult = {}
pluginReport = {}
barcodeSummary = []
barcodeReport = {}
barcodeInput = {}
help_dictionary = {}
barcodeLibraries = {}
prefix_info={}
suffix_info={}
bc_details ={}
# Specifies whether to assume default reference or disallow barcode-specific no referecne
barcode_no_reference_as_default = True

# max plugin out file ext length for addressing max filename length (w/ barcode), e.g. ".amplicon.cov.xls"
max_fileext_len = 17
max_filename_len = 255

#
# -------------------- customize code for this plugin here ----------------
#

def addAutorunParams(plan=None):
  '''Additional parameter set up for auto-mated runs, e.g. to add defaults for option only in the GUI.'''
  # Note: this function may be redundant in later TSS versions, where pluginconfig already consolidates input from all sources
  config = pluginParams['config']
  # defaults for auto-run to match config settings from GUI
  if plan: 
    # examine plan library type and ensure non-appropriate options for the library type are set to default values
    runtype = plan['runType']
    config['librarytype'] = runtype
    config['librarytype_id'] = plan['runTypeDescription']

    # check for required targets file
    bedfile = plan['bedfile']
    if bedfile == 'None': bedfile = ''
    config['targetregions'] = bedfile
    bedname = fileName(bedfile)
    target_id = os.path.basename(bedfile)
    if target_id[-4:] == ".bed": target_id = target_id[:-4]
    config['targetregions_id'] = target_id
    # catch no default reference Plan set up
    genome = pluginParams['genome_id']
    if genome == 'None': genome = ''
    if not genome:
      raise Exception("CATCH: Cannot run plugin with no default reference.")
    # check for bacode-specific targets and detect barcode-specific mode
    trgstr = ''
    for bc,data in barcodeLibraries.items():
      setbed = data['bedfile']
      # AMPS_DNA_RNA is an exception - must have barcode-specific targets
      if runtype == 'AMPS_DNA_RNA' or setbed != bedfile:
        trgstr += bc+'='+setbed+';'
    if trgstr:
      config['barcodebeds'] = 'Yes'
      config['barcodetargetregions'] = trgstr
    elif not bedfile:
      config['padtargets'] = '0'
      config['sampleid'] = 'No'
      if runtype != 'WGNM' and runtype != 'RNA' and runtype != 'GENS':
        printlog('\nWARNING: %s analysis requires the plan to specify a targets file. Defaulting to Whole Genome analysis.\n'%config['librarytype_id'])
        config['librarytype'] = 'WGNM'
        config['librarytype_id'] = 'Whole Genome'
  else:
    raise Exception("CATCH:Automated analysis requires a Plan to specify Run Type.")    


def furbishPluginParams():
  '''Additional parameter set up for this plugin.'''
  config = pluginParams['config']
  # since HTML form posts do not add unchecked options add them in here
  if 'barcodebeds' not in config: config['barcodebeds'] = 'No'
  if 'sampleid' not in config: config['sampleid'] = 'No'
  if 'uniquemaps' not in config: config['uniquemaps'] = 'No'
  if 'nonduplicates' not in config: config['nonduplicates'] = 'No'


def configReport():
  '''Returns a dictionary based on the plugin config parameters that as reported in results.json.'''
  # This is to avoid outputting hidden or aliased values. If not needed just pass back a copy of config.
  #return pluginParams['config'].copy()
  config = pluginParams['config']
  return {
    "Launch Mode" : config['launch_mode'],
    "Reference Genome" : pluginParams['genome_id'],
    "Library Type" : config['librarytype_id'],
    "Targeted Regions" : config['targetregions_id'],
    "Barcode-specific Targets" : config['barcodebeds'],
    "Sample Tracking" : config['sampleid'],
    "barcoded" : "true" if pluginParams['barcoded'] else "false"
  }


def printStartupMessage():
  '''Output the standard start-up message. Customized for additional plugin input and options.'''
  printlog('')
  printtime('Started %s' % pluginParams['plugin_name'])
  config = pluginParams['config']
  #for k, v in config.iteritems():
  #	printlog("tt: %s %s"%(k,v))
  printlog('Plugin run parameters:')
  printlog('  Plugin version:   %s' % pluginParams['cmdOptions'].version)
  printlog('  Launch mode:      %s' % config['launch_mode'])
  printlog('  Run is barcoded:  %s' % ('Yes' if pluginParams['barcoded'] else 'No'))
  printlog('  Reference Name:   %s' % pluginParams['genome_id'])
  printlog('  Library  Type:     %s' % config['librarytype_id'])
  printlog('  Target Regions:   %s' % config['targetregions_id'])
  #if config['barcodebeds'] == 'Yes':
    #target_files = pluginParams['target_files']
    #for bctrg in sorted(target_files):
      #trg = fileName(target_files[bctrg])
      #if trg == "": trg = "None"
      #printlog('    %s  %s' % (bctrg,trg))
  printlog('Data files used:')
  printlog('  Parameters:     %s' % pluginParams['jsonInput'])
  printlog('  Reference:      %s' % pluginParams['reference'])
  printlog('  Root Alignment: %s' % pluginParams['bamroot'])
  printlog('  Target Regions: %s' % config['targetregions_id'])
  printlog('')

def barcodeSpecifics(barcode=""):
  '''Process the part of the plan that customizes barcodes to different references.'''
  if not barcode:
    barcode = NOMATCH
  if not barcode in bc_details:
    return { "filtered" : True }

  barcodeData = bc_details[barcode]
  sample = barcodeData.get('sample','')
  if sample == 'none': sample = ''
  filtered = barcodeData.get('filtered',True)
  if barcode == NOMATCH: filtered = True
  reference = barcodeData['reference']
  refPath = barcodeData['reference_fullpath']
  genomeUrl = barcodeData.get('genome_urlpath','')
  if not genomeUrl:
    idx = refPath.find('referenceLibrary/')
    if idx >= 0:
      genomeUrl = os.path.join("/output",refPath[idx+17])

  # special exception to allow full IGV annotation
  if reference == 'hg19': genomeUrl = ''

  # bedfile override only for manual run - logic for allowed overrides elsewhere
  if pluginParams['manual_run']:
    target_files = pluginParams['target_files']
    if barcode in target_files:
      bedfile = target_files[barcode]
    else:
      bedfile = pluginParams['config']['targetregions']
  else:
    bedfile = barcodeData['target_region_filepath']
  return {
    "filtered" : filtered,
    "sample" : sample,
    "reference" : reference,
    "refpath" : refPath,
    "nuctype" : barcodeData.get('nucleotide_type','DNA'),
    "refurl" : '{http_host}'+genomeUrl if genomeUrl else reference,
    "bedfile" : bedfile,
    "bamfile" : barcodeData.get('bam_filepath','') 
   }

def get_param_config_name(meta_dict):
    my_configuration = meta_dict.get('configuration', None)
    if meta_dict.get('custom', False):
        if my_configuration is None:
            return 'customized'
        if my_configuration.endswith(' (from external)'):
            return my_configuration
        
        return '%s (from external)' %my_configuration

    return my_configuration if my_configuration is not None else 'unknown configuration'
    
    
def run_plugin(skiprun=False,barcode=""):
  '''Wrapper for making command line calls to perform the specific plugin analyis.'''
  # first part is pretty much boiler plate - grab the key parameters that most plugins use
  logopt = pluginParams['cmdOptions'].logopt
  config = pluginParams['config']
  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['output_dir']
  output_url = pluginParams['output_url']
  output_prefix = pluginParams['output_prefix']
  bamfile = pluginParams['bamfile']
  libtype = config['librarytype']
  genome = pluginParams['genome_url']
  reference = pluginParams['reference']
  bedfile = pluginParams['bedfile']
  barcodeData = barcodeSpecifics(barcode)
  tvc_config = config['tvc_config']
  tvc_param_file = os.path.join(output_dir, "local_parameters.json")
  writeDictToJsonFile(tvc_config,tvc_param_file)
  configure_file = get_param_config_name(tvc_config['meta'])
  cmd = "cp %s/local_parameters.json %s/../local_parameters.json"%(output_dir,output_dir)
  os.system(cmd)

  
  sample = pluginParams['sample_names']
  if barcode: 
        sample = sample.get(barcode,'')
  if sample == '': sample = 'None'

  # account for reference and library type overrides, e.g. for AmpliSeq-DNA/RNA
  allow_no_target = pluginParams['allow_no_target']
  libnuc_type = ''
  if barcode and (config['barcodebeds'] == 'Yes' or pluginParams['barcoderefs'] == 'Yes'):
    (libtype,genome,reference,target) = barcodeLibrary(barcode)
    # barcode-specific reference MUST be defined
    if not reference:
      raise Exception("Barcode-specified reference expected but none provided.")

  # link from source BAM since pipeline uses the name as output file stem
  linkbam = os.path.join(output_dir,output_prefix+".bam")
  createlink(bamfile,linkbam)
  createlink(bamfile+'.bai',linkbam+'.bai')
  bamfile = linkbam
  num_thread = 16
  # Run-type flags used to customize this barcode report - not necessarily the same as in renderOptions()
  samp_track = (config['sampleid'] == 'No')
  trg_stats = (bedfile != "")
  amp_stats = 1
  bas_stats = 1
  trg_type = 1 
  statsfile = output_prefix+'.cov.stats.json'
  
  # skip the actual and assume all the data already exists in this file for processing
  if skiprun:
    printlog("Skipped analysis - generating report on in-situ data")
  else:
    # Pre-run modification of BED files is done here to avoid redundancy of repeating for target assigned barcodes
    # Note: The stand-alone command can perform the (required) annotation but does not handle padding
    rptopt = 'molecularCoverageAnalysis.html' if pluginParams['cmdOptions'].cmdline else ''
    printtime("Running Molecular Coverage Analysis pipeline...")
    runcmd = '%s %s %s "%s" "%s" "%s" "%s" %s  "%s" %s ' % (
        os.path.join(plugin_dir,'run_molecular_coverage.sh'), plugin_dir, bamfile, reference, sample, libtype , bedfile, num_thread, output_dir, output_prefix)
    if( os.system(runcmd) ):
      raise Exception("Failed running run_molecular_coverage.sh. Refer to Plugin Log.")
  printtime("Generating report...")
  try:
    # Link report page resources. This is necessary as the plugin code is inaccesible from URLs directly.
    createlink( os.path.join(plugin_dir,'flot'), output_dir )
    createlink( os.path.join(plugin_dir,'lifechart'), output_dir )
    createlink( os.path.join(plugin_dir,'css'), pluginParams['results_dir'] )
    createlink( os.path.join(plugin_dir,'scripts'), pluginParams['results_dir'] )
    createlink( os.path.join(plugin_dir,'bin','igv.php3'), output_dir )
    createlink( os.path.join(plugin_dir,'bin','zipReport.php3'), output_dir )
  except Exception as Err:
    printtime("Symbolic linking Error: %s", Err)
  # Optional: Delete intermediate files after successful run. These should not be required to regenerate any of the
  # report if the skip-analysis option. Temporary file deletion is also disabled when the --keep_temp option is used.
  #deleteTempFiles([ '*.bam', '*.bam.bai', '*.tmp' ])

  # Create an annotated list of files as used to create the file links table.
  # - Could be handled in the HTML template directly but external code is re-used to match cmd-line reports.

  # Parse out stats from results text file to dict AND convert unacceptible characters to underscores in keys to avoid Django issues
  trgtype = '.amplicon.cov'
  resultData = parseToDict( os.path.join(output_dir,statsfile), ":" )

  # Collect other output data to pluginReport, which is anything else that is used to generate the report
  reportData = {
    "library_type" : config['librarytype_id'],
    "libnuc_type" : barcodeData['nuctype'],
    "configure_file" : configure_file,
    "run_name" : output_prefix,
    "barcode_name" : barcode,
    "samp_track" : samp_track,
    "output_dir" : output_dir,
    "output_url" : output_url,
    "output_prefix" : output_prefix,
    "help_dict" : helpDictionary(),
    "stats_txt" : checkOutputFileURL(statsfile),
    "finecov_tsv" :  checkOutputFileURL(output_prefix+trgtype+'.xls'),
    "base_cov_tsv" : checkOutputFileURL(output_prefix+'.mol.cov.xls'),
    "bed_link" : re.sub( r'^.*/uploads/BED/(\d+)/.*', r'/rundb/uploadstatus/\1/', bedfile ),
    "file_links" : checkOutputFileURL('filelinks.xls'),
    "bam_link" : checkOutputFileURL(output_prefix+'.bam'),
    "bai_link" : checkOutputFileURL(output_prefix+'.bam.bai'),
    "aux_ttc"  : checkOutputFileURL('tmc.aux.ttc.xls')
  }
  return (resultData,reportData)

def checkFileURL(fileURL):
  '''coverageAnalysis helper method to return "" if the provided file URL does not exist'''
  if os.path.exists(os.path.join(pluginParams['output_dir'],fileURL)):
    return fileURL
  return ""

def checkOutputFileURL(fileURL):
  '''coverageAnalysis helper method to return "" if the provided file URL does not exist'''
  if os.path.exists(os.path.join(pluginParams['output_dir'],fileURL)):
    return fileURL
  return ""

''' Add GC to bed files , temporary turn off for '''

'''
def modifyBedFiles(bedfile,reference):
  #coverageAnalysis method to merged padded and GC annotated BED files, creating them if they do not already exist.
  if not bedfile: return ('','')
  # files will be created or found in this results subdir
  bedDir = os.path.join(pluginParams['results_dir'],"local_beds")
  if not os.path.exists(bedDir): os.makedirs(bedDir)
  if bedfile == 'contigs':
    # special option to create annotated BED files for whole contigs
    rootbed = fileName(reference)
    gcbed = os.path.join(bedDir,"%s.contigs.gc.bed"%rootbed)
    if os.path.exists(gcbed):
      printlog("Adopting GC annotated contigs %s"%os.path.basename(gcbed))
    else:
      printtime("Creating GC annotated contigs %s"%os.path.basename(gcbed))
      if os.system( '%s -g "%s.fai" "%s" > "%s"' % (
          os.path.join(pluginParams['plugin_dir'],'bed','gcAnnoBed.pl'), reference, reference, gcbed ) ):
        raise Exception("Failed to annotate contig regions using gcAnnoBed.pl")
    mergbed = annobed = gcbed
  else:
    # the pair of files returned are dependent on the Library Type and padded options
    rootbed = fileName(bedfile)
    mergbed = bedfile.replace('unmerged','merged',1)
    annobed = bedfile if pluginParams['is_ampliseq'] else mergbed
    gcbed = os.path.join(bedDir,"%s.gc.bed"%rootbed)
    if os.path.exists(gcbed):
      printlog("Adopting GC annotated targets %s"%os.path.basename(gcbed))
    else:
      printtime("Creating GC annotated targets %s"%os.path.basename(gcbed))
      if os.system( '%s -s -w -f 4,8 -t "%s" "%s" "%s" > "%s"' % (
          os.path.join(pluginParams['plugin_dir'],'bed','gcAnnoBed.pl'), bedDir, annobed, reference, gcbed ) ):
        raise Exception("Failed to annotate target regions using gcAnnoBed.pl")
    annobed = gcbed
  # padding removes any merged detail fields to give only coordinates bed file
  try:
    padval = int(float(pluginParams['config']['padtargets']))
  except:
    padval = pluginParams['config']['padtargets'] = 0
  if padval > 0:
    padbed = os.path.join(bedDir,"%s.pad%s.bed"%(rootbed,padval))
    if os.path.exists(padbed):
      printlog("Adopting padded targets %s"%os.path.basename(padbed))
    else:
      printtime("Creating padded targets %s"%os.path.basename(padbed))
      if os.system( '%s %s "%s" "%s" %d "%s"' % (
          os.path.join(pluginParams['plugin_dir'],'bed','padbed.sh'), pluginParams['logopt'],
          mergbed, reference+'.fai', padval, padbed ) ):
        raise Exception("Failed to pad target regions using padbed.sh")
    mergbed = padbed
  return (mergbed,annobed)
'''

def run_meta_plugin():
  '''Create barcode x target reads matrix and text version of barcode summary table (after analysis completes for individual barcodes).'''
  if pluginParams['cmdOptions'].cmdline: return
  printtime("Collating barcodes summary data...")
  # get correct file/type for reads matrix
  fileext2 = ""
  renderOpts = renderOptions()
  fieldid = '9'
  typestr = 'amplicon'
  fileext = '.'+typestr+'.cov.xls'
  bctable = []
  reportFiles = []
  bcresults = pluginResult['barcodes']
  bcreports = pluginReport['barcodes']
  # iterate barcodeSummary[] to maintain barcode processing order
  refs = []
  for bcdata in barcodeSummary:
    bcname = bcdata['barcode_name']
    bcrep = bcreports[bcname]
    #printlog(" final barcode %s : %s" % (bcname,bcrep))
    if 'output_dir' not in bcrep: continue
    try:
      bcline = bcname+"\t"+bcdata['sample']+"\t"+bcdata['funcfam']+"\t"+bcdata['funcratio']+"\t"+bcdata['uniformity']+"\t"+bcdata['famsize']+"\t"+bcdata['conversion']
      bctable.append(bcline)
    except Exception, Err:
      printtime("Error during barcode summary processing: %s", Err)
    reportfile = os.path.join(bcrep['output_dir'],bcrep['output_prefix']+fileext)
    reportfile2 = os.path.join(bcrep['output_dir'],bcrep['output_prefix']+fileext2)
    if os.path.exists(reportfile):
      reportFiles.append(reportfile)
    elif fileext2 and os.path.exists(reportfile2):
      reportFiles.append(reportfile2)

  if len(bctable) > 0:
    bctabfile = pluginParams['prefix']+".bc_summary.xls"
    bcline = "Barcode ID\tSample name\tMedian functional molecules\tMedian functional ratio\tMolecular based uniformity\tMedian average molecular size\tR2F conversion ratio";
    with open(os.path.join(pluginParams['results_dir'],bctabfile),'w') as outfile:
      outfile.write(bcline+'\n')
      for bcline in bctable:
        outfile.write(bcline+'\n')
    barcodeReport.update({"bctable":bctabfile})
  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['results_dir']
  createlink( os.path.join(plugin_dir,'bin','zipReport.php3'), output_dir )
  createlink( os.path.join(plugin_dir,'bin','download.php3'), output_dir )


def updateBarcodeSummaryReport(barcode,autoRefresh=False):
  '''Create barcode summary (progress) report. Called before, during and after barcodes are being analysed.'''
  global barcodeSummary
  if pluginParams['cmdOptions'].cmdline: return
  renderOpts = renderOptions()
  numErrs = pluginReport['num_barcodes_failed'] + pluginReport['num_barcodes_invalid']
  # no barcode means either non have been ceated yet or this is a refresh after all have been processed (e.g. for meta data)
  if barcode != "":
    resultData = pluginResult['barcodes'][barcode]
    reportData = pluginReport['barcodes'][barcode]
    # check for error status
    errMsg = ""
    errZero = False
    if 'Error' in resultData:
      errMsg = resultData['Error']
      errZero = "no mapped" in errMsg or "no read" in errMsg
    sample = resultData.get('Sample Name', None)
    if sample == '' : sample = 'None'
    # barcodes_json dictoonary is firmcoded in Kendo table template that we are using for main report styling
    if errMsg != "":
      detailsLink = "<span class='help' title=\"%s\" style='color:red'>%s</span>" % ( errMsg, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "funcfam": "NA",
		"lod": "NA",
		"allfam": "NA",
		"famloss": "NA",
		"funcratio": "NA",
		"famsize": "NA",
		"conversion": "NA",
		"uniformity": "NA"
      })
    else:
      is_default_eq_beta = "-"

      detailsLink = "<a target='_parent' href='%s' class='help'><span title='Click to view the detailed report for barcode %s'>%s</span><a>" % (
        os.path.join(barcode,pluginParams['report_name']), barcode, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
		"funcfam": resultData.get('Median_funcfam_coverage_per_amplicon',"NA"),
		"lod": resultData.get('LOD',"NA"),
		"allfam": resultData.get('Median_allfam_coverage_per_amplicon',"NA"),
		"famloss": resultData.get('Median_fam_loss_by_strand_bias',"NA"),
		"famsize": resultData.get('Median_average_fam_size',"NA"),
		"funcratio": resultData.get('Functional molecules %',"NA"),
		"conversion": resultData.get('Median_conversion_rate',"NA"),
		"uniformity": resultData.get('Fam_uniformity_of_amplicon_coverage',"NA")
      })
  render_context = {
    "autorefresh" : autoRefresh,
    "run_name" : pluginParams['prefix'],
    "barcode_results" : simplejson.dumps(barcodeSummary),
    "help_dict" : helpDictionary(),
    "Error" : '%d barcodes failed analysis.'%numErrs if numErrs > 0 else ''
  }
  render_context.update(renderOpts)
  config = pluginParams['config']
  tplate = 'barcode_%s_summary.html'%(config['librarytype'])
  # extra report items, e.g. file links from barcodes summary page
  if barcodeReport:
    render_context.update(barcodeReport)
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), tplate, render_context )


def renderOptions():
  '''coverageAnalysis support method to generate list of condensed rendering options for (barcode) summary reports.'''
  config = pluginParams['config']
  librarytype = config.get('librarytype',"NA")
  targets = 'Barcode-specific' if config['barcodebeds'] == 'Yes' else config['targetregions_id']
  if targets == 'None': targets = ""
  #trg_stats = pluginParams['have_targets']
  configure_file = get_param_config_name(config['tvc_config']['meta'])
  return {
    "runType" : librarytype,
    "library_type" : config.get('librarytype_id',"NA"),
    "target_regions" : targets,
    "configure_file" : configure_file
  }
  

def createIncompleteReport(errorMsg=""):
  '''Called to create an incomplete or error report page for non-barcoded runs.'''
  sample = 'None'
  if 'sample_names' in pluginParams and isinstance(pluginParams['sample_names'],basestring):
    sample = pluginParams['sample_names']
  render_context = {
    "autorefresh" : (errorMsg == ""),
    "run_name": pluginParams['prefix'],
    "Sample Name": sample,
    "Error": errorMsg }
  render_context.update(renderOptions())
  if errorMsg == "" :
    createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'incomplete.html', render_context )
  else :
    createReport( pluginParams['block_report'] , 'error_block.html', render_context )


def createDetailReport(resultData,reportData):
  '''Called to create the main report (for un-barcoded run or for each barcode).'''
  if pluginParams['cmdOptions'].cmdline: return
  output_dir = pluginParams['output_dir']
  render_context = resultData.copy()
  render_context.update(reportData)
  config = pluginParams['config']
  tplate = 'report_%s.html'%(config['librarytype'])

  createReport( os.path.join(output_dir,pluginParams['report_name']), tplate, render_context )

  # create temporary html report and convert to PDF for the downloadable zip report
  html_report = os.path.join(pluginParams['output_dir'],'tmp_report.html')
  tplate = 'report_%s_pdf.html'%(config['librarytype'])
  createReport( html_report, tplate, render_context )
  pdf_report = pluginParams['output_prefix']+'.summary.pdf'
  xcmd = Popen( [os.path.join(pluginParams['plugin_dir'],'bin','wkhtmltopdf-amd64'), '--load-error-handling', 'ignore',
    '--page-height', '2000', html_report, os.path.join(output_dir,pdf_report)], shell=False, stdout=PIPE, stderr=PIPE )
  xcmd.communicate()
  if xcmd.poll():
    printlog("Warning: Failed to PDF report summary file.")
    pdf_report = ''
  else:
    deleteTempFiles([html_report])


def createBlockReport():
  '''Called at the end of run to create a block.html report. Use 'pass' if not wanted.'''
  config = pluginParams['config']
  if pluginParams['cmdOptions'].cmdline: return
  printtime("Creating block report...")
  if pluginParams['barcoded']:
    numErrs = pluginReport['num_barcodes_failed'] + pluginReport['num_barcodes_invalid']
    render_context = {
      "run_name" : pluginParams['prefix'],
      "barcode_results" : simplejson.dumps(barcodeSummary),
      "help_dict" : helpDictionary(),
      "Error" : '%d barcodes failed analysis.'%numErrs if numErrs > 0 else ''
    }
    render_context.update( renderOptions() )
    tplate = 'barcode_%s_block.html'%(config['librarytype'])
  else:
    render_context = pluginResult.copy()
    render_context.update(pluginReport)
    tplate = 'report_block.html'
  createReport( pluginParams['block_report'], tplate, render_context )


def createProgressReport(progessMsg):
  '''General method to write a message directly to the block report, e.g. when starting prcessing of a new barcode.'''
  createReport( pluginParams['block_report'], "progress_block.html", { "progress_text" : progessMsg } )


def helpDictionary():
  '''coverageAnalysis method to load a dictionary for on-line help in the reports.'''
  global help_dictionary
  if not help_dictionary:
    with open(os.path.join(pluginParams['plugin_dir'],'templates','help_dict.json')) as jsonFile:
      help_dictionary = json.load(jsonFile)
  return help_dictionary

#def load_dataset_info():
#  ''' load the prefix and suffix information from datasets_basecaller.json '''
#  global prefix_info,suffix_info
#  if not prefix_info:
#     with open(os.path.join(pluginParams['basecaller_dir'],'datasets_basecaller.json')) as jsonFile:
#        all_info = json.load(jsonFile)
#  for k, v in all_info['read_groups'].items():
#        if 'barcode_name' in v.keys():
#             bn = v['barcode_name']
#             p_tag = v['mol_tag_prefix']
#             s_tag = v['mol_tag_suffix']
#             prefix_info[bn] = p_tag
#             suffix_info[bn] = s_tag
#  return (prefix_info , suffix_info)

def handle_ts_default_configuration(jsonParams):
    pluginconfig = jsonParams.get('pluginconfig', {})
    my_configuration = pluginconfig.get('meta', {}).get('configuration', None)
    
    # TS builtin configuration renders startsplugin['pluginconfig'] = {'meta': {'configuration': configuration_name}}
    is_ts_builtin_configuration = pluginconfig == {'meta': {'configuration': my_configuration}}
    
    if (not is_ts_builtin_configuration) or (my_configuration is None):
        # Not the case of TS builtin configuration (TS-17735)
        return 
    
    # Try the the following variantCaller plugin paths
    # 1) TS default plugin dir (should always work since the feature works on a Torrent Server only.)
    # 2) Relative to this file (assume the plugin repo)
    try_vc_plugin_path_list = ['/results/plugins/variantCaller', 
                                os.path.join(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))), 'variantCaller')]
    # Read VC plugin builtin parameter sets
    parameter_sets_list = []
    for try_vc_plugin_path in try_vc_plugin_path_list:
        parameter_sets_json_path = os.path.join(try_vc_plugin_path, 'pluginMedia/parameter_sets/parameter_sets.json')
        try:
            with open(parameter_sets_json_path, 'r') as f_json:
                parameter_sets_list = json.load(f_json)
            if parameter_sets_list:
                # Now I get the tvc parameter_sets
                break
        except:
            pass

    if not parameter_sets_list:
        printerr('Unable to retrieve the variantCaller parameter sets for TS default configuration.' )                    
        exit(1)
        return   

    # See if I can find a match.
    for parameter_set in parameter_sets_list:
        if parameter_set.get('meta', {}).get('configuration', None) == my_configuration:
            # Find a match. Let's use it.
            jsonParams['pluginconfig']['tvc_config'] = copy.deepcopy(parameter_set)
            printlog('Found the TS builtin configuration "%s" from the variantCaller parameter sets. The TVC configuration "%s" will be used.' %(my_configuration, my_configuration))
            return 

    # The configuration is not found in parameter_sets_list. Perhaps a legacy configuration?
    # See if I can find any replacement.
    for parameter_set in parameter_sets_list:
        if my_configuration in parameter_set.get('meta', {}).get('replaces', []):
            # Find a replacement. Let's use it.
            jsonParams['pluginconfig']['tvc_config'] = copy.deepcopy(parameter_set)
            new_configuration = parameter_set.get('meta', {}).get('configuration', 'Unknown Configuration')
            printlog('TS builtin configuration "%s" is replaced by the variantCaller configuration "%s" that will be used.' %(my_configuration, new_configuration))            
            return
    
    printerr('Unable to find the TS builtin configuration "%s" or and a replacement from the variantCaller parameter sets.' %my_configuration)                    
    exit(1)
    return 

#
# --------------- Base code for standard plugin runs -------------
#

def parseCmdArgs():
  '''Process standard command arguments. Customized for additional debug and other run options.'''
  # standard run options here - do not remove
  parser = OptionParser()
  parser.add_option('-B', '--bam', help='Filepath to root alignment BAM file. Default: rawlib.bam', dest='bamfile', default='')
  parser.add_option('-P', '--prefix', help='Output file name prefix for output files. Default: '' => Use analysis folder name or "output".', dest='prefix', default='')
  parser.add_option('-Q', '--reference_url', help='Relative URL to fasta file for the whole reference', dest='reference_url', default='')
  parser.add_option('-R', '--reference_fasta', help='Path to fasta file for the whole reference', dest='reference', default='')
  parser.add_option('-U', '--results_url', help='URL for access to files in the output directory', dest='results_url', default='')
  parser.add_option('-V', '--version', help='Version string for tracking in output', dest='version', default='')
  parser.add_option('-X', '--min_bc_bam_size', help='Minimum file size required for barcode BAM processing', type="int", dest='minbamsize', default=0)
  parser.add_option('-c', '--cmdline', help='Run command line only. Reports will not be generated using the HTML templates.', action="store_true", dest='cmdline')
  parser.add_option('-d', '--scraper', help='Create a scraper folder of links to output files using name prefix (-P).', action="store_true", dest='scraper')
  parser.add_option('-k', '--keep_temp', help='Keep intermediate files. By default these are deleted after a successful run.', action="store_true", dest='keep_temp')
  parser.add_option('-l', '--log', help='Output extra progress Log information to STDERR during a run.', action="store_true", dest='logopt')
  parser.add_option('-s', '--skip_analysis', help='Skip re-generation of existing files but make new report.', action="store_true", dest='skip_analysis')
  parser.add_option('-x', '--stop_on_error', help='Stop processing barcodes after one fails. Otherwise continue to the next.', action="store_true", dest='stop_on_error')

  (cmdOptions, args) = parser.parse_args()
  if( len(args) != 2 ):
    printerr('Usage requires exactly two file arguments (startplugin.json barcodes.json)')
    exit(1)
  with open(args[0]) as jsonFile:
    jsonParams = json.load(jsonFile)
  # Handle the TS default configuration (TS-17735)
  handle_ts_default_configuration(jsonParams)

  global pluginParams, bc_details
  with open(args[1]) as jsonFile:
    bc_details = json.load(jsonFile)

  pluginParams['cmdOptions'] = cmdOptions
  pluginParams['jsonInput'] = args[0]
  pluginParams['jsonParams'] = jsonParams
  pluginParams['barcodeInput'] = args[1]  
  pluginParams['barcoded'] = os.path.exists(args[1])
   
def emptyResultsFolder():
  '''Purge everything in output folder except for specifically named files.'''
  results_dir = pluginParams['results_dir']
  if results_dir == '/': return
  logopt = pluginParams['cmdOptions'].logopt
  if logopt or os.path.exists( os.path.join(results_dir,pluginParams['report_name']) ):
    printlog("Purging old results...")
  for fname in os.listdir(results_dir):
    # avoid specific files needed to launch run
    if not os.path.isdir(fname):
      if "log" in fname: continue
      start = os.path.basename(fname)[:10]
      if start == "drmaa_stdo" or start == "ion_plugin" or start == "startplugi" or start == "barcodes.j":
        continue
    if logopt:
      if os.path.islink(fname):
        printlog("Removing symlink %s"%fname)
      elif os.path.isdir(fname):
        printlog("Removing directory %s"%fname)
      else:
        printlog("Removing file %s"%fname)
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
      os.unlink(f)

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
  except Exception,e:
    return ""
  return samplenames

def targetFiles():
  trgfiles = {}
  try:
    if pluginParams['barcoded']:
      bcbeds = pluginParams['config']['barcodetargetregions']
      if isinstance(bcbeds,basestring):
        bcmaps = bcbeds.split(';')
        for m in bcmaps:
          kvp = m.split('=',1)
          if not kvp[0]: continue
          if kvp[1] == 'None': kvp[1] = ''
          trgfiles[kvp[0]] = kvp[1]
      else:
        for bc in bcbeds:
          trgfiles[bc] = bcbeds[bc]
  except:
    pass
  return trgfiles

def barcodeLibrary(barcode=""):
  '''Process the part of the plan that customizes barcodes to different references.'''
  global barcodeLibraries
  if not barcodeLibraries:
    def_genome = pluginParams['genome_id']
    def_genomeURL = pluginParams['genome_url']
    def_reference = pluginParams['reference']
    # grab plan settings for each barcode
    bcsamps = pluginParams['jsonParams']['plan']['barcodedSamples']
    if isinstance(bcsamps,basestring):
      bcsamps = json.loads(bcsamps)
    for bcname in bcsamps:
      if not 'barcodeSampleInfo' in bcsamps[bcname]: continue
      for (bc,data) in bcsamps[bcname]['barcodeSampleInfo'].items():
          genome = data.get('reference','')
          if barcode_no_reference_as_default and not genome:
            genome = def_genome
          trg = data.get('targetRegionBedFile','')
          if trg == 'None': trg = ''
          barcodeLibraries[bc] = {
            'library': 'AMPS_RNA' if data.get('nucleotideType','') == 'RNA' else 'AMPS',
            'genome': def_genomeURL.replace(def_genome,genome) if genome else '',
            'reference': def_reference.replace(def_genome,genome) if genome else '',
            'bedfile': trg
          }
  # allow call with no barcode to set up - for initialization
  if not barcode: return ('','','','')
  # GUI shouldn't allow barcodes to be specified that are not speciied in the Plan here, but just in case...
  if barcode in barcodeLibraries:
    bc = barcodeLibraries[barcode]
    return ( bc['library'], bc['genome'], bc['reference'], bc['bedfile'] )
  return ( pluginParams['genome_id'], pluginParams['genome_url'], pluginParams['reference'], '' )

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
  pluginParams['basecaller_dir'] = jsonParams['runinfo'].get('basecaller_dir','.')
  pluginParams['logopt'] = '-l' if pluginParams['cmdOptions'].logopt else ''

  # some things not yet in startplugin.json are provided or over-writen by cmd args
  copts = pluginParams['cmdOptions']
  pluginParams['reference'] = copts.reference if copts.reference != "" else jsonParams['runinfo'].get('reference','')
  pluginParams['bamroot']   = copts.bamfile   if copts.bamfile != "" else '%s/rawlib.bam' % pluginParams['analysis_dir']
  pluginParams['prefix']    = copts.prefix    if copts.prefix != "" else pluginParams['analysis_name']
  pluginParams['results_url'] = copts.results_url if copts.results_url != "" else os.path.join(
  jsonParams['runinfo'].get('url_root','.'),'plugin_out',pluginParams['plugin_name']+'_out' )
  # set URL to local .fasta (later IGV .genome if this becomes available) file or default to genome name
  pluginParams['genome_url'] = '{http_host}/auth'+copts.reference_url if copts.reference_url != "" else pluginParams['genome_id']
 
      
  # check for non-supported de novo runs
  if not pluginParams['genome_id'] or not pluginParams['reference']:
    printerr("Requires a reference sequence for coverage analysis.")
    raise Exception("CATCH:Do not know how to analyze coverage without reference sequence for library '%s'"%pluginParams.get('genome_id',""))

  # set up for barcoded vs. non-barcodedruns
  pluginParams['bamfile'] = pluginParams['bamroot']
  pluginParams['output_dir'] = pluginParams['results_dir']
  pluginParams['output_url'] = pluginParams['results_url']
  pluginParams['output_prefix'] = pluginParams['prefix']
  pluginParams['bamname'] = os.path.basename(pluginParams['bamfile'])
  #pluginParams['barcoded'] = (os.path.exists(pluginParams['results_dir']+'/barcodes.json'))
  #pluginParams['barcoded'] = (os.path.exists(pluginParams['analysis_dir']+'/barcodeList.txt'))
  #pluginParams['barcoded'] = (os.path.exists(pluginParams['results_dir']+'/barcodes.json'))
  #pluginParams['barcoded'] = (os.path.exists(pluginParams['results_dir']+'/barcodes.json'))
  #pluginParams['barcoded'] = True
  pluginParams['sample_names'] = sampleNames()

  # disable run skip if no report exists => plugin has not been run before
  pluginParams['report_name'] = pluginParams['plugin_name']+'.html'
  pluginParams['block_report'] = os.path.join(pluginParams['results_dir'],pluginParams['plugin_name']+'_block.html')
  if not os.path.exists( os.path.join(pluginParams['results_dir'],pluginParams['report_name']) ):
    if pluginParams['cmdOptions'].skip_analysis:
      printlog("Warning: Skip analysis option ignorred as previous output appears to be missing.")
      pluginParams['cmdOptions'].skip_analysis = False

  # set up for barcode-specific libraries (target and/or reference): targets may be overriden by GUI
  pluginParams['barcoderefs'] = 'No'
  barcodeLibrary()
  defref = pluginParams['reference']
  for bc,data in barcodeLibraries.items():
    setref = data['reference']
    if setref and setref != defref:
      pluginParams['barcoderefs'] = 'Yes'
      break;

  # set up plugin specific options depending on auto-run vs. plan vs. GUI
  config = pluginParams['config'] = jsonParams['pluginconfig'].copy() if 'pluginconfig' in jsonParams else {}
  if 'barcodebeds' not in config: config['barcodebeds'] = 'No'
  if 'sampleid' not in config: config['sampleid'] = 'No'

  pluginParams['manual_run'] = config.get('launch_mode','') == 'Manual'
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
    raise Exception("The plugin can't run without any configuration")

  if "targetregions" in config :
    for bc in bc_details :
      bc_details[bc]['target_region_filepath']=config['targetregions']
  if "barcodetargetregions" in config :
    tlist = config['barcodetargetregions'].split(";")
    for t in tlist :
      l = t.split("=")
      if l[0] in bc_details :
        bc_details[l[0]]['target_region_filepath']=l[1]

  # code to handle single or per-barcode target files
  if 'librarytype' not in config: config['library'] =  jsonParams['runType']
  runtype = config['librarytype']
  if runtype[:7] == 'AMPS_HD' :
    config['librarytype']= 'ampliseq_hd'
  elif runtype == 'TAG_SEQUENCING' :
    config['librarytype']= 'tagseq'
  else : 	
    raise Exception("CATCH:Unable to analyze molecular coverage for '%s'"%config['librarytype_id'])
  pluginParams['is_ampliseq'] = (runtype[:4] == 'AMPS' or runtype == 'TARS_16S')
  pluginParams['target_files'] = targetFiles()
  pluginParams['have_targets'] = (config['targetregions'] or pluginParams['target_files'])
  if 'targetregions' in config :	pluginParams['target_files']=config['targetregions']
  pluginParams['allow_no_target'] = (runtype == 'GENS' or runtype == 'WGNM' or runtype == 'RNA')

  # loading tvc_configuration file
  if 'tvc_config' not in config :
    tvc_json = os.path.join(pluginParams['analysis_dir'], 'inputs','vc','VariantCallerActor','VariantCallerActor.json')
    if not os.path.exists(tvc_json):
        printlog('No tvc config file' %(tvc_json))
        exit(1)
    with open(tvc_json) as jsonFile:
        tvc_config = json.load(jsonFile)
        config['tvc_config'] = tvc_config
        config['tvc_config']['meta']['configuration']='customized'
  else :
    tvc_config = config['tvc_config']

  tvc_param_file = os.path.join(pluginParams['results_dir'], 'local_parameters.json')
  writeDictToJsonFile(tvc_config,tvc_param_file)

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
  #try:
  #  bcfileName = pluginParams['analysis_dir']+'/barcodeList.txt'
  #  with open(bcfileName) as bcfile:
  #    for line in bcfile:
  #      if line.startswith('barcode '):
  #        barcodes.append(line.split(',')[1])
  #except:
  #  printerr("Reading barcode list file '%s'" % bcfileName)
  #  raise
  # RULES: A barcode-specific target set to None ('') assumes the default target UNLESS require_specific_targets is set (for library type),
  # in which case an error for the barcode is set. Otherwise if the barcode-specific target resolves to None then the barcode is skipped
  # for library types requiring a target. For AMPS_DNA_RNA it is a reported error if the barcode has no targets.
  runtype = pluginParams['config']['librarytype']
  disallow_no_target = not pluginParams['allow_no_target']
  have_targets = pluginParams['have_targets']
  default_target = pluginParams['config']['targetregions']
  #default_hotspot = pluginParams['config']['hotspotregions']
  target_files = pluginParams['target_files']
  #same_target_flag = int( pluginParams['config']['starget'])
  #same_hotspot_flag = int( pluginParams['config']['shotspot'])
  # Pre-check BAM and BED files and compatibility - barcodes are either skipped or produce summary table errors
  numGoodBams = 0
  numInvalidBarcodes = 0
  maxBarcodeLen = 0
  minFileSize = pluginParams['cmdOptions'].minbamsize
  (bcBamPath,bcBamRoot) = os.path.split(pluginParams['bamroot'])
  bcBamFile = []
  for barcode in sorted(bc_details) :
    #barcodeData = barcodeSpecifics(barcode)
    #if(barcodeData['filtered']) : continue
    #bcbam = barcodeData['bamfile']
    #printlog("barcode")
    details = bc_details[barcode]
    bcbam = details['bam_filepath']
    printlog("barcode %s : bam  %s ...\n" % (barcode,bcbam))
    ntype = details['nucleotide_type']
    if not os.path.exists(bcbam):
      bcBamFile.append("ERROR: BAM file not found")
      continue
    if os.stat(bcbam).st_size < minFileSize:
      bcBamFile.append("ERROR: BAM file too small")
      continue
    if ntype == "RNA":
      bcBamFile.append("ERROR: The plugin does not support RNA barcodes")
      continue
    if barcode in bc_details :
      #bedfile = bc_details[barcode]['target_region_filepath']
      bedfile = bc_details[barcode].get('target_region_filepath','')
    else :
      bedfile = default_target

    if not bedfile:
      # Special cases where ONLY explicitly specified barcodes override the default
      if barcode in target_files:
        #if runtype == 'AMPS_DNA_RNA':
        #  numInvalidBarcodes += 1
        #  bcBamFile.append(":\nERROR: Targets file is not specified but required for this Library Type.")
        #  continue
        if default_target:
          bcBamFile.append("ERROR: Default target regions overriden by barcode-specific 'None'.")
          continue
        if disallow_no_target:
          bcBamFile.append("ERROR: No specific or default target regions for barcode.")
          continue
      elif not default_target and disallow_no_target:
        bcBamFile.append("ERROR: No default target regions for barcode.")
        continue
      bedfile = default_target
    ckbb = checkBamBed(bcbam,bedfile)
    if ckbb:
      bcBamFile.append("ERROR: "+ckbb)
      numInvalidBarcodes += 1;
      continue
    if( len(barcode) > maxBarcodeLen ):
      maxBarcodeLen = len(barcode)
    numGoodBams += 1
    bcBamFile.append(bcbam)

  ensureFilePrefix(maxBarcodeLen+1)

  printlog("Processing %d barcodes...\n" % numGoodBams)
  pluginReport['num_barcodes_processed'] = numGoodBams
  pluginReport['num_barcodes_failed'] = 0
  pluginReport['num_barcodes_invalid'] = numInvalidBarcodes

  # create initial (empty) barcodes summary report
  updateBarcodeSummaryReport("",True)

  # iterate over all barcodes and process the valid ones
  skip_analysis = pluginParams['cmdOptions'].skip_analysis
  stop_on_error = pluginParams['cmdOptions'].stop_on_error
  create_scraper = pluginParams['cmdOptions'].scraper
  sample_names = pluginParams['sample_names']
  postout = False; # just for logfile prettiness
  barcodeProcessed = 0
  for barcode in sorted(bc_details):
    sample = bc_details[barcode]['sample'] if barcode in bc_details else ''
    bamfile = bcBamFile.pop(0)
    if bamfile[:6] == "ERROR:":
      if postout:
        printlog("")
      printlog("Skipping %s%s%s" % (barcode,('' if sample == '' else ' (%s)'%sample),bamfile))
      # for error messages to appear in barcode table
      perr = bamfile.find('ERROR:')+6
      if perr >= 6:
        pluginResult['barcodes'][barcode] = { "Sample Name" : sample, "Error" : bamfile[perr:].strip() }
        pluginReport['barcodes'][barcode] = {}
        updateBarcodeSummaryReport(barcode,True)
    else:
      try:
        postout = True
        printlog("\nProcessing %s%s...\n" % (barcode,('' if sample == '' else ' (%s)'%sample)))
        if barcode in barcodeLibraries:
          printlog('Reference File: %s' % barcodeLibraries[barcode]['reference'])
        if have_targets:
          target_file =  bc_details[barcode]['target_region_filepath'] if barcode in bc_details else ''
          if not target_file: target_file = default_target
          pluginParams['bedfile'] = target_file
          if not pluginParams['is_ampliseq']:
            target_file = target_file.replace('unmerged','merged',1)
          if target_file != "None" and target_file != "": printlog('Target Regions: %s' % target_file)
        else:
          pluginParams['bedfile'] = ''
        
        pluginParams['bamfile'] = bamfile
        #hotspotfile =  bc_details[barcode]['hotspot_filepath'] if barcode in bc_details else ''
	#if not hotspotfile : hotspotfile= default_hotspot
	#pluginParams['hotspotfile'] = hotspotfile
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
        printerr('Analysis of barcode %s failed:'%barcode)
        pluginReport['num_barcodes_failed'] += 1
        pluginResult['barcodes'][barcode] = { "Sample Name" : sample, "Error" : str(e) }
        pluginReport['barcodes'][barcode] = {}
        if stop_on_error: raise
        traceback.print_exc()
      updateBarcodeSummaryReport(barcode,True)		

  createProgressReport( "Compiling barcode summary report..." )
  run_meta_plugin()
  updateBarcodeSummaryReport("")
  if create_scraper:
    createScraperLinksFolder( pluginParams['results_dir'], pluginParams['prefix'] )


def runNonBarcoded():
  '''Run the plugin script once for non-barcoded reports.'''
  global pluginResult, pluginReport
  ensureFilePrefix()
  try:
    pluginParams['bedfile'] = pluginParams['config']['targetregions'] if pluginParams['have_targets'] else ''
    createIncompleteReport()
    ckbb = checkBamBed(pluginParams['bamfile'],pluginParams['bedfile'])
    if ckbb: raise Exception(ckbb)
    (resultData,pluginReport) = run_plugin( pluginParams['cmdOptions'].skip_analysis )
    pluginResult.update(resultData)
    createDetailReport(pluginResult,pluginReport)
  except Exception, e:
    printerr('Analysis failed')
    pluginResult.update({ 'Error': str(e) })
    createIncompleteReport(str(e))
    raise
  if pluginParams['cmdOptions'].scraper:
    createScraperLinksFolder( pluginParams['output_dir'], pluginParams['output_prefix'] )

def checkBamBed(bamfile,bedfile):
  '''Return error message if the provided BED file is not suitable for BAM file or other issues with BAM file.'''
  if not bedfile: return ""
  runcmd = Popen( [os.path.join(pluginParams['plugin_dir'],'bin','checkBamBed.pl'), bamfile, bedfile], stdout=PIPE, shell=False )
  errMsg = runcmd.communicate()[0]
  if runcmd.poll():
    raise Exception("Failed running checkBamBed.pl. Refer to Plugin Log.")
  return errMsg.strip()

def createScraperLinksFolder(outdir,rootname):
  '''Make links to all files matching <outdir>/<rootname>.* to <outdir>/scraper/link.*'''
  # rootname is a file path relative to outdir and should not contain globbing characters
  scrapeDir = os.path.join(outdir,'scraper')
  if pluginParams['cmdOptions'].logopt:
    printlog("Creating scraper folder %s" % scrapeDir)
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
#  if pluginParams['is_ampliseq']:
#    runcmd = Popen( [os.path.join(pluginParams['plugin_dir'],'scripts','addMeanBarcodeStats.py'),
#      jsonfile, "Amplicons reading end-to-end"], stdout=PIPE, shell=False )
#    runcmd.communicate()
#    runcmd = Popen( [os.path.join(pluginParams['plugin_dir'],'scripts','addMeanBarcodeStats.py'),
#      jsonfile, "Percent end-to-end reads"], stdout=PIPE, shell=False )
#    runcmd.communicate()

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
      printlog('ERROR: %s' % emsg)
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
        if pluginReport['num_barcodes_invalid'] > 0:
          printlog("ERROR: All barcodes had invalid target regions specified.")
          return 1;
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

