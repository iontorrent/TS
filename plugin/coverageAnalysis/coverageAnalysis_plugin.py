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
barcodeLibraries = {}

# option to allow missing DNA/RNA targets to be analyzed as whole genome
barcode_no_target_as_whole_genome = False

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
  config['librarytype'] = 'wholegenome'
  config['librarytype_id'] = 'Whole Genome'
  config['targetregions'] = ''
  config['targetregions_id'] = 'None'
  config['sampleid'] = 'No'
  config['trimreads'] = 'No'
  config['padtargets'] = '0'
  config['uniquemaps'] = 'No'
  config['nonduplicates'] = 'Yes'
  config['barcodebeds'] = 'No'
  config['barcodetargetregions'] = ''
  # set defaults for derived settings in case of early error set up, e.g. via renderOptions()
  pluginParams['is_ampliseq'] = False
  pluginParams['have_targets'] = False
  # extract things from the plan if provided - for coverageAnalysis auto-run w/o a plan leads to early exit
  if plan: 
    runtype = plan['runType']
    if runtype == 'AMPS':
      config['librarytype'] = 'ampliseq'
      config['librarytype_id'] = 'Ion AmpliSeq'
    elif runtype == 'AMPS_EXOME':
      config['librarytype'] = 'ampliseq-exome'
      config['librarytype_id'] = 'Ion AmpliSeq Exome'
    elif runtype == 'AMPS_RNA':
      config['librarytype'] = 'ampliseq-rna'
      config['librarytype_id'] = 'Ion AmpliSeq RNA'
    elif runtype == 'AMPS_DNA_RNA':
      config['librarytype'] = 'ampliseq-dna-rna'
      config['librarytype_id'] = 'Ion AmpliSeq DNA/RNA'
      # extract barcode-specific targets from plan
      config['barcodebeds'] = 'Yes'
      barcodeLibrary()
      trgstr = ''
      for bc,data in barcodeLibraries.items():
          trgstr += bc+'='+data['bedfile']+';'
      config['barcodetargetregions'] = trgstr
    elif runtype == 'TARS':
      config['librarytype'] = 'targetseq'
      config['librarytype_id'] = 'Ion TargetSeq'
    elif runtype == 'WGNM':
      config['librarytype'] = 'wholegenome'
      config['librarytype_id'] = 'Whole Genome'
    else:
      config['librarytype_id'] = "[%s]"%runtype
      raise Exception("CATCH:Do not know how to analyze coverage for unsupported plan runType: '%s'"%runtype)
    if runtype[0:4] == "AMPS":
      config['nonduplicates'] = 'No'
    bedfile = plan['bedfile']
    if bedfile != "":
      config['targetregions'] = bedfile
      config['targetregions_id'] = fileName(bedfile)
    elif runtype != 'WGNM':
      printlog('\nWARNING: %s analysis requires the plan to specify a targets file. Defaulting to Whole Genome analysis.\n'%config['librarytype_id'])
      config['librarytype'] = 'wholegenome'
      config['librarytype_id'] = 'Whole Genome'
  else:
    raise Exception("CATCH:Automated analysis requires a Plan to specify Run Type.")    


def addPluginParams():
  '''Additional parameter set up for this plugin.'''
  config = pluginParams['config']
  pluginParams['is_ampliseq'] = (config['librarytype'][:8] == 'ampliseq')
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
    "Target Padding" : config['padtargets'],
    "Use Only Uniquely Mapped Reads" : config['uniquemaps'],
    "Use Only Non-duplicate Reads" : config['nonduplicates'],
    "barcoded" : "true" if pluginParams['barcoded'] else "false"
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
  printlog('  Reference Name:   %s' % pluginParams['genome_id'])
  printlog('  Library Type:     %s' % config['librarytype_id'])
  printlog('  Target Regions:   %s' % config['targetregions_id'])
  printlog('  Barcoded Targets: %s' % config['barcodebeds'])
  if config['barcodebeds'] == 'Yes':
    target_files = pluginParams['target_files']
    for bctrg in sorted(target_files):
      printlog('    %s  %s' % (bctrg,fileName(target_files[bctrg])))
  printlog('  Target Padding:   %s' % config['padtargets'])
  printlog('  Sample Tracking:  %s' % config['sampleid'])
  printlog('  Uniquely Mapped:  %s' % config['uniquemaps'])
  printlog('  Non-duplicate:    %s' % config['nonduplicates'])
  printlog('Data files used:')
  printlog('  Parameters:     %s' % pluginParams['jsonInput'])
  printlog('  Reference:      %s' % pluginParams['reference'])
  printlog('  Root Alignment: %s' % pluginParams['bamroot'])
  if pluginParams['is_ampliseq']:
    printlog('  Target Regions: %s' % config['targetregions'])
  else:
    printlog('  Target Regions: %s' % config['targetregions'].replace('unmerged','merged',1))
  printlog('')


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
  librarytype = config['librarytype']
  genome = pluginParams['genome_url']
  reference = pluginParams['reference']
  bedfile = pluginParams['bedfile']

  sample = pluginParams['sample_names']
  if barcode: sample = sample.get(barcode,'')
  if sample == '': sample = 'None'

  # account for reference and library type overrides, e.g. for AmpliSeq-DNA/RNA
  libnuc_type = ''
  if barcode and librarytype == 'ampliseq-dna-rna':
    (librarytype,genome,reference,target) = barcodeLibrary(barcode)
    libnuc_type = 'RNA' if librarytype == 'ampliseq-rna' else 'DNA'

  # special case: AmpliSeq should fail without target but 'None' => make whole genome report
  if bedfile == 'None':
    librarytype = 'wholegenome'
    bedfile = ''

  # link from source BAM since pipeline uses the name as output file stem
  linkbam = os.path.join(output_dir,output_prefix+".bam")
  createlink(bamfile,linkbam)
  createlink(bamfile+'.bai',linkbam+'.bai')
  bamfile = linkbam

  # Run-type flags used to customize the report
  samp_track = (config['sampleid'] == 'Yes')
  rna_stats = (librarytype == 'ampliseq-rna')
  trg_stats = (bedfile != "")
  amp_stats = trg_stats and pluginParams['is_ampliseq']
  wgn_stats = not trg_stats
  bas_stats = not rna_stats
  trg_type = 1 if amp_stats else 0
  if rna_stats: trg_type = 2

  # Hard-coded path to sample ID target BED file
  sampleidBed = os.path.join(os.path.dirname(plugin_dir),'sampleID','targets','KIDDAME_sampleID_regions.bed') if samp_track else ''

  # skip the actual and assume all the data already exists in this file for processing
  if skiprun:
    printlog("Skipped analysis - generating report on in-situ data")
  else:
    # Pre-run modification of BED files is done here to avoid redundancy of repeating for target assigned barcodes
    # The stand-alone command can perform the (required) annotation but does not handle padding
    (mergeBed,annoBed) = modifyBedFiles(bedfile,reference)
    # option for cmd-line detailed HTML report generation
    rptopt = '-R coverageAnalysis.html' if pluginParams['cmdOptions'].cmdline else ''
    ampopt = '-a' if amp_stats else ''
    filtopts = '-u' if config['uniquemaps'] == 'Yes' else ''
    if config['nonduplicates'] == 'Yes': filtopts += ' -d'
    if rna_stats: ampopt = '-r'
    printtime("Running coverage analysis pipeline...")
    runcmd = '%s %s %s %s -c -D "%s" -A "%s" -B "%s" -C "%s" -L "%s" -N "%s" -p %s -S "%s" %s "%s" "%s"' % (
        os.path.join(plugin_dir,'run_coverage_analysis.sh'), pluginParams['logopt'], ampopt,
        filtopts, output_dir, annoBed, mergeBed, bedfile, genome, sample, config['padtargets'], sampleidBed, rptopt, reference, bamfile )
    if logopt: printlog('\n$ %s\n'%runcmd)
    if( os.system(runcmd) ):
      raise Exception("Failed running run_coverage_analysis.sh. Refer to Plugin Log.")

  if pluginParams['cmdOptions'].cmdline: return ({},{})
  printtime("Generating report...")

  # Link report page resources. This is necessary as the plugin code is inaccesible from URLs directly.
  createlink( os.path.join(plugin_dir,'flot'), output_dir )
  createlink( os.path.join(plugin_dir,'lifechart'), output_dir )
  createlink( os.path.join(plugin_dir,'scripts','igv.php3'), output_dir )
  createlink( os.path.join(plugin_dir,'scripts','zipReport.php3'), output_dir )
  if annoBed: createlink( annoBed, output_dir )
   
  # Optional: Delete intermediate files after successful run. These should not be required to regenerate any of the
  # report if the skip-analysis option. Temporary file deletion is also disabled when the --keep_temp option is used.
  #deleteTempFiles([ '*.bam', '*.bam.bai', '*.bed' ])

  # Create an annotated list of files as used to create the file links table.
  # - Could be handled in the HTML template directly but external code is re-used to match cmd-line reports.

  # Parse out stats from results text file to dict AND convert unacceptible characters to underscores in keys to avoid Django issues
  statsfile = output_prefix+'.stats.cov.txt'
  resultData = parseToDict( os.path.join(output_dir,statsfile), ":" )

  # Collect other output data to pluginReport, which is anything else that is used to generate the report
  trgtype = '.amplicon.cov' if amp_stats else '.target.cov'
  reportData = {
    "library_type" : librarytype,
    "libnuc_type" : libnuc_type,
    "run_name" : output_prefix,
    "barcode_name" : barcode,
    "samp_track" : samp_track,
    "output_dir" : output_dir,
    "output_url" : output_url,
    "output_prefix" : output_prefix,
    "num_stat_tables" : 1 if (rna_stats or wgn_stats) else 2,
    "amp_stats" : amp_stats,
    "rna_stats" : rna_stats,
    "trg_stats" : trg_stats,
    "wgn_stats" : wgn_stats,
    "bas_stats" : bas_stats,
    "trg_type" : trg_type,
    "help_dict" : helpDictionary(),
    "stats_txt" : checkFileURL(statsfile),
    "overview_png" : checkFileURL(output_prefix+'.covoverview.png'),
    "rep_overview_png" : checkFileURL(output_prefix+'.repoverview.png'),
    "scat_gc_png" :  checkFileURL(output_prefix+'.gc.png'),
    "scat_len_png" : checkFileURL(output_prefix+'.ln.png'),
    "rep_gc_png" :   checkFileURL(output_prefix+'.gc_rep.png'),
    "rep_len_png" :  checkFileURL(output_prefix+'.ln_rep.png'),
    "rep_pool_png" : checkFileURL(output_prefix+'.pool.png'),
    "finecov_tsv" :  checkFileURL(output_prefix+trgtype+'.xls'),
    "overview_tsv" : checkFileURL(output_prefix+'.covoverview.xls'),
    "base_cov_tsv" : checkFileURL(output_prefix+'.base.cov.xls'),
    "chr_cov_tsv" :  checkFileURL(output_prefix+'.chr.cov.xls'),
    "wgn_cov_tsv" :  checkFileURL(output_prefix+'.wgn.cov.xls'),
    "bed_link" : re.sub( r'^.*/uploads/BED/(\d+)/.*', r'/rundb/uploadstatus/\1/', bedfile ),
    "aux_bbc" : checkFileURL('tca_auxiliary.bbc'),
    "aux_cbc" : checkFileURL('tca_auxiliary.cbc'),
    "aux_ttc" : checkFileURL('tca_auxiliary.ttc.xls'),
    "file_links" : checkFileURL('filelinks.xls'),
    "bam_link" : checkFileURL(output_prefix+'.bam'),
    "bai_link" : checkFileURL(output_prefix+'.bam.bai')
  }
  return (resultData,reportData)

def checkFileURL(fileURL):
  '''coverageAnalysis helper method to return "" if the provided file URL does not exist'''
  if os.path.exists(os.path.join(pluginParams['output_dir'],fileURL)):
    return fileURL
  return ""

def modifyBedFiles(bedfile,reference):
  '''coverageAnalysis method to merged padded and GC annotated BED files, creating them if they do not already exist.'''
  if not bedfile: return ('','')
  # files will be created or found in this results subdir
  bedDir = os.path.join(pluginParams['results_dir'],"local_beds")
  if not os.path.exists(bedDir): os.makedirs(bedDir)
  # the pair of files returned are dependent on the Library Type and padded options
  rootbed = fileName(bedfile)
  mergbed = bedfile.replace('unmerged','merged',1)
  annobed = bedfile if pluginParams['is_ampliseq'] else mergbed
  # do not re-do GC annotation on same BED file - moderately expensive for large BEDs
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


def run_meta_plugin():
  '''Create barcode x target reads matrix and text version of barcode summary table (after analysis completes for individual barcodes).'''
  if pluginParams['cmdOptions'].cmdline: return
  printtime("Collating barcodes summary data...")
  # get correct file/type for reads matrix
  renderOpts = renderOptions()
  if renderOpts['trg_stats']:
    fieldid = '9'
    typestr = 'amplicon' if pluginParams['is_ampliseq'] else 'target'
    fileext = '.'+typestr+'.cov.xls'
  else:
    fieldid = 'chrom'
    typestr = 'contig'
    fileext = '.chr.cov.xls'
  bctable = []
  reportFiles = []
  bcresults = pluginResult['barcodes']
  bcreports = pluginReport['barcodes']
  # iterate barcodeSummary[] to maintain barcode processing order
  refs = []
  for bcdata in barcodeSummary:
    bcname = bcdata['barcode_name']
    bcrep = bcreports[bcname]
    if 'output_dir' not in bcrep: continue
    bcline = bcname+"\t"+bcdata['sample']+"\t"+bcdata['mapped_reads']
    if renderOpts['trg_stats']: bcline += "\t"+bcdata['on_target']
    if renderOpts['samp_track']: bcline += "\t"+bcdata['sample_target']
    if renderOpts['bas_stats']: bcline += "\t"+bcdata['mean_depth']+"\t"+bcdata['uniformity']
    if renderOpts['mixed_stats']:
      if bcname in barcodeLibraries and 'reference' in barcodeLibraries[bcname]:
        rfile = barcodeLibraries[bcname]['reference'] + '.fai'
        if not rfile in refs: refs.append(rfile)
    bctable.append(bcline)
    reportfile = os.path.join(bcrep['output_dir'],bcrep['output_prefix']+fileext)
    if os.path.exists(reportfile):
      reportFiles.append(reportfile)

  if len(bctable) > 0:
    bctabfile = pluginParams['prefix']+".bc_summary.xls"
    bcline = "Barcode ID\tSample Name\tMapped Reads"
    if renderOpts['trg_stats']: bcline += "\tOn Target"
    if renderOpts['samp_track']: bcline += "\tSampleID"
    if renderOpts['bas_stats']: bcline += "\tMean Depth\tUniformity"
    with open(os.path.join(pluginParams['results_dir'],bctabfile),'w') as outfile:
      outfile.write(bcline+'\n')
      for bcline in bctable:
        outfile.write(bcline+'\n')
    barcodeReport.update({"bctable":bctabfile})

  if len(reportFiles) > 0:
    # collect references
    if renderOpts['mixed_stats']:
      if len(refs) > 0:
        reference = ",".join(refs)
      else:
        reference = "-"
        printlog("WARNING: No references were matched to specified barcodes.\n- Barcode matrix targets may not be correctly ordered.");
    else:
      reference = pluginParams['reference']+'.fai'
    bcmatrix = pluginParams['prefix']+".bcmatrix.xls"
    with open(os.path.join(pluginParams['results_dir'],bcmatrix),'w') as outfile:
      runcmd = Popen( [os.path.join(pluginParams['plugin_dir'],'scripts','barcodeMatrix.pl'),
        reference, fieldid] + reportFiles, stdout=outfile )
      runcmd.communicate()
      if runcmd.poll():
        raise Exception("Failed to create barcode x %s reads matrix."%typestr)
      barcodeReport.update({ "bcmatrix":bcmatrix, "bcmtype":typestr })

  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['results_dir']
  createlink( os.path.join(plugin_dir,'scripts','zipReport.php3'), output_dir )


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
    if 'Error' in resultData:
      errMsg = resultData['Error']
    sample = resultData['Sample Name']
    if sample == '': sample = 'None'
    # barcodes_json dictoonary is firmcoded in Kendo table template that we are using for main report styling
    if errMsg != "":
      detailsLink = "<span class='help' title=\"%s\" style='color:red'>%s</span>" % ( errMsg, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "mapped_reads": "NA",
        "on_target": "NA",
        "sample_target": "NA",
        "mean_depth": "NA",
        "uniformity": "NA"
      })
    else:
      detailsLink = "<a target='_parent' href='%s' class='help'><span title='Click to view the detailed report for barcode %s'>%s</span><a>" % (
        os.path.join(barcode,pluginParams['report_name']), barcode, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "mapped_reads": resultData.get('Number of mapped reads',"NA"),
        "on_target": resultData.get('Percent reads on target',"NA"),
        "sample_target": resultData.get('Percent sample tracking reads',"NA"),
        "mean_depth": resultData.get('Average base coverage depth',"NA"),
        "uniformity": resultData.get('Uniformity of base coverage',"NA")
      })
  render_context = {
    "autorefresh" : autoRefresh,
    "run_name" : pluginParams['prefix'],
    "barcode_results" : simplejson.dumps(barcodeSummary),
    "help_dict" : helpDictionary(),
    "Error" : '%d barcodes failed analysis.'%numErrs if numErrs > 0 else ''
  }
  render_context.update(renderOpts)
  # extra report items, e.g. file links from barcodes summary page
  if barcodeReport:
    render_context.update(barcodeReport)
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'barcode_summary.html', render_context )


def renderOptions():
  '''coverageAnalysis support method to generate list of condensed rendering options and values.'''
  config = pluginParams['config']
  targets = 'Barcode-specific' if config['barcodebeds'] == 'Yes' else config['targetregions_id']
  if targets == 'None': targets = ""
  filter_options = []
  if pluginParams['is_ampliseq'] and config['sampleid'] == 'Yes': filter_options.append('Sample tracking')
  if not config['librarytype'] == 'ampliseq-rna' and config['uniquemaps'] == 'Yes': filter_options.append('Uniquely mapped')
  if not pluginParams['is_ampliseq'] and config['nonduplicates'] == 'Yes': filter_options.append('Non-duplicate')
  # note that these general settings may have been overriden for AmpliSeq-DNA/RNA
  return {
    "library_type" : config['librarytype_id'],
    "target_regions" : targets,
    "target_padding" : config['padtargets'],
    "filter_options" : ', '.join(filter_options),
    "samp_track" : (config['sampleid'] == 'Yes'),
    "mixed_stats" : (config['librarytype'] == "ampliseq-dna-rna"),
    "trg_stats" : pluginParams['have_targets'],
    "bas_stats" : (config['librarytype'] != 'ampliseq-rna')
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
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'incomplete.html', render_context )


def createDetailReport(resultData,reportData):
  '''Called to create the main report (for un-barcoded run or for each barcode).'''
  if pluginParams['cmdOptions'].cmdline: return
  output_dir = pluginParams['output_dir']
  render_context = resultData.copy()
  render_context.update(reportData)
  createReport( os.path.join(output_dir,pluginParams['report_name']), 'report.html', render_context )

  # create temporary html report and convert to PDF for the downloadable zip report
  html_report = os.path.join(pluginParams['output_dir'],'tmp_report.html')
  createReport( html_report, 'report_pdf.html', render_context )
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
    tplate = 'barcode_block.html'
  else:
    render_context = pluginResult.copy()
    render_context.update(pluginReport)
    tplate = 'report_block.html'
  createReport( pluginParams['block_report'], tplate, render_context )


def helpDictionary():
  '''coverageAnalysis method to load a dictionary for on-line help in the reports.'''
  global help_dictionary
  if not help_dictionary:
    with open(os.path.join(pluginParams['plugin_dir'],'templates','help_dict.json')) as jsonFile:
      help_dictionary = json.load(jsonFile)
  return help_dictionary


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
  # Dangerous - replace with something safer if it becomes obvious (e.g. putting output in subfolder?)
  results_dir = pluginParams['results_dir']
  if results_dir == '/': return
  logopt = pluginParams['cmdOptions'].logopt
  if logopt or os.path.exists( os.path.join(results_dir,pluginParams['report_name']) ):
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
      os.unlink(fname)
    for name in dirs:
      fname = os.path.join(root,name)
      if logopt:
        printlog("Removing directory %s"%fname)
      os.rmdir(fname)
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
          if not kvp[1]:
            kvp[1] = 'None' if barcode_no_target_as_whole_genome else ''
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
  defTrg = 'None' if barcode_no_target_as_whole_genome else ''
  if not barcodeLibraries:
    def_genome = pluginParams['genome_id']
    def_genomeURL = pluginParams['genome_url']
    def_reference = pluginParams['reference']
    # grab plan settings for each barcode
    bcsamps = pluginParams['jsonParams']['plan']['barcodedSamples']
    if isinstance(bcsamps,basestring):
      bcsamps = json.loads(bcsamps)
    for bcname in bcsamps:
      for (bc,data) in bcsamps[bcname]['barcodeSampleInfo'].items():
          genome = data.get('reference',def_genome)
          barcodeLibraries[bc] = {
            'library': 'ampliseq-rna' if data.get('nucleotideType','') == 'RNA' else 'ampliseq-dna',
            'genome': def_genomeURL.replace(def_genome,genome),
            'reference': def_reference.replace(def_genome,genome),
            'bedfile': data.get('targetRegionBedFile',defTrg)
          }
  # allow call with no barcode (for initiation)
  if not barcode: return ('','','','')
  # GUI shouldn't allow barcodes to be specified that are not speciied in the Plan here, but just in case...
  if barcode in barcodeLibraries:
    bc = barcodeLibraries[barcode]
    return ( bc['library'], bc['genome'], bc['reference'], bc['bedfile'] )
  return ( pluginParams['genome_id'], pluginParams['genome_url'], pluginParams['reference'], defTrg )

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
    pass
  elif 'plan' in jsonParams:
    config['launch_mode'] = 'Autostart with plan configuration'
    addAutorunParams(jsonParams['plan'])
  else:
    config['launch_mode'] = 'Autostart with default configuration'
    addAutorunParams()

  # add extra plugin customization
  addPluginParams()

  # code to handle single or per-barcode target files
  pluginParams['target_files'] = targetFiles()
  pluginParams['have_targets'] = (config['targetregions'] or pluginParams['target_files'])

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
  # grab barcoded-target information and ensure plan is checked for multi-references
  have_targets = pluginParams['have_targets']
  default_target = pluginParams['config']['targetregions']
  check_targets = (have_targets and not default_target)
  runtype = pluginParams['config']['librarytype']
  if runtype == 'ampliseq-dna-rna':
    barcodeLibrary()
    if barcode_no_target_as_whole_genome: check_targets = false
  target_files = pluginParams['target_files']
  # pre-check specified barcodes file presence
  numGoodBams = 0
  numNoTargets = 0
  maxBarcodeLen = 0
  minFileSize = pluginParams['cmdOptions'].minbamsize
  (bcBamPath,bcBamRoot) = os.path.split(pluginParams['bamroot'])
  bcBamFile = []
  for barcode in barcodes:
    bcbam = os.path.join( bcBamPath, "%s_%s"%(barcode,bcBamRoot) )
    bedfile = target_files.get(barcode,'')
    if not os.path.exists(bcbam):
      bcbam = ": BAM file not found"
    elif check_targets and not bedfile:
      # in this case distinguish selected 'None' vs. not used
      if runtype == 'ampliseq-dna-rna' and barcode in target_files:
        numNoTargets += 1
        bcbam = ":\nERROR: Targets file is not specified but required for run type 'Ion AmpliSeq DNA/RNA'."
      else:
        bcbam = ": No assigned or default target regions for barcode."
    elif os.stat(bcbam).st_size < minFileSize:
      bcbam = ": BAM file too small"
    else:
      if( len(barcode) > maxBarcodeLen ):
        maxBarcodeLen = len(barcode)
      numGoodBams += 1
    bcBamFile.append(bcbam)

  ensureFilePrefix(maxBarcodeLen+1)

  printlog("Processing %d barcodes...\n" % numGoodBams)
  pluginReport['num_barcodes_processed'] = numGoodBams
  pluginReport['num_barcodes_failed'] = 0
  pluginReport['num_barcodes_invalid'] = numNoTargets

  # create initial (empty) barcodes summary report
  updateBarcodeSummaryReport("",True)

  # iterate over all barcodes and process the valid ones
  skip_analysis = pluginParams['cmdOptions'].skip_analysis
  stop_on_error = pluginParams['cmdOptions'].stop_on_error
  create_scraper = pluginParams['cmdOptions'].scraper
  sample_names = pluginParams['sample_names']
  postout = False; # just for logfile prettiness
  for barcode in barcodes:
    sample = sample_names[barcode] if barcode in sample_names else ''
    bamfile = bcBamFile.pop(0)
    if bamfile[0] == ":":
      if postout:
        postout = False
        printlog("")
      printlog("Skipping %s%s%s" % (barcode,('' if sample == '' else ' (%s)'%sample),bamfile))
      # for error messages to appear in barcode table
      perr = bamfile.find('ERROR:')+6
      if perr >= 6:
        pluginResult['barcodes'][barcode] = { "Sample Name" : sample, "Error" : bamfile[perr:] }
        pluginReport['barcodes'][barcode] = {}
        updateBarcodeSummaryReport(barcode,True)
    else:
      try:
        postout = True
        printlog("\nProcessing %s%s...\n" % (barcode,('' if sample == '' else ' (%s)'%sample)))
        if barcode in barcodeLibraries:
          printlog('Reference File: %s' % barcodeLibraries[barcode]['reference'])
        if have_targets:
          target_file = target_files[barcode] if barcode in target_files else default_target
          pluginParams['bedfile'] = target_file
          if not pluginParams['is_ampliseq']:
            target_file = target_file.replace('unmerged','merged',1)
          printlog('Target Regions: %s' % target_file)
        else:
          pluginParams['bedfile'] = ''
        pluginParams['bamfile'] = bamfile
        pluginParams['output_dir'] = os.path.join(pluginParams['results_dir'],barcode)
        pluginParams['output_url'] = os.path.join(pluginParams['results_url'],barcode)
        pluginParams['output_prefix'] = barcode+"_"+pluginParams['prefix']
        if not os.path.exists(pluginParams['output_dir']):
           os.makedirs(pluginParams['output_dir'])
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

  run_meta_plugin()
  updateBarcodeSummaryReport("")
  if create_scraper:
    createScraperLinksFolder( pluginParams['results_dir'], pluginParams['prefix'] )


def runNonBarcoded():
  global pluginResult, pluginReport
  ensureFilePrefix()
  try:
    pluginParams['bedfile'] = pluginParams['config']['targetregions'] if pluginParams['have_targets'] else ''
    createIncompleteReport()
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
  if pluginParams['is_ampliseq']:
    runcmd = Popen( [os.path.join(pluginParams['plugin_dir'],'scripts','addMeanBarcodeStats.py'),
      jsonfile, "Amplicons reading end-to-end"], stdout=PIPE, shell=False )
    runcmd.communicate()
    if runcmd.poll():
      printlog("Warning: Failed to modify results.json using addMeanBarcodeStats.py.")

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

