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
#
from django import template
register = template.Library()

@register.filter 
def toMega(value):
  return float(value) / 1000000

template.builtins.append(register) 

# defines exceptional bacode names to check against
NONBARCODED = "nonbarcoded"
NOMATCH = "nomatch"

# global data collecters common to functions
barcodeInput = {}
pluginParams = {}
pluginResult = {}
pluginReport = {}
barcodeSummary = []
barcodeReport = {}

# ratio for minimal to maximum barcode BAM size for barcode to be ignored - 0 => disabled
barcode_filter = 0.0

# max plugin out file ext length for addressing max filename length (w/ barcode), e.g. ".STARTBowtie2.RNAmetrics.png"
max_fileext_len = 33
max_filename_len = 255

#
# -------------------- customize code for this plugin here ----------------
#

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

  # add other run options here (these should override or account for things not in the json parameters file)

  (cmdOptions, args) = parser.parse_args()
  if( len(args) != 2 ):
    printerr('Takes only two file arguments: startplugin.json barcodes.json')
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


def addAutorunParams(plan=None):
  '''Additional parameter set up for automated runs, e.g. to add defaults for option only in the GUI.'''
  config = pluginParams['config']
  if not 'reference' in config: config['reference'] = ""
  if not 'genome' in config: config['genome'] = ""
  config['fraction_of_reads'] = "1"
  config['cutadapt'] = "None"
  config['fpkm_thres'] = "0.3"


def furbishPluginParams():
  '''Complete/rename/validate user parameters.'''
  # For example, HTML form posts do not add unchecked option values
  config = pluginParams['config']
  config['fpkm_thres'] = "0.3"
  pass


def printStartupMessage():
  '''Output the standard start-up message. Customized for additional plugin input and options.'''
  printlog('')
  printtime('Started %s\n' % pluginParams['plugin_name'])
  config = pluginParams['config']
  printlog('Run configuration:')
  printlog('  Plugin version:   %s' % pluginParams['cmdOptions'].version)
  printlog('  Launch mode:      %s' % config['launch_mode'])
  printlog('  Parameters:       %s' % pluginParams['jsonInput'])
  printlog('  Barcodes:         %s' % pluginParams['barcodeInput'])
  printlog('  Output folder:    %s' % pluginParams['results_dir'])
  printlog('  Output file stem: %s' % pluginParams['prefix'])
  printlog('Run parameters:')
  printlog('  Reference name:   %s' % config['genome'])
  printlog('  Sampled Reads:    %s' % config['fraction_of_reads'])
  printlog('  Adapter sequence: %s' % config['cutadapt'])
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
  barcodeData = barcodeSpecifics(barcode)

  # link from source BAM since pipeline uses the name as output file stem
  linkbam = os.path.join(output_dir,output_prefix+".bam")
  createlink(barcodeData['bamfile'],linkbam)
  bamfile = linkbam

  # user parameters
  genome = config['genome']
  reference = config['reference']
  fracreads = config['fraction_of_reads']
  adapter   = config['cutadapt']
  fpkm_thres = config['fpkm_thres']

  # scratch location
  ref_idxdir = "/results/plugins/scratch/RNASeqAnalysis"

  # skip the actual and assume all the data already exists in this file for processing
  run_log = 'rnaseq.log'
  if skiprun:
    printlog("Skipped analysis - generating report on in situ data")
  else:
    runplugin = Popen( [ os.path.join(plugin_dir,'run_rnaseqanalysis.py'),
      genome, reference, bamfile, barcodeData['sample'], adapter, fracreads, fpkm_thres,
      output_dir, output_prefix, run_log, ref_idxdir ], stdout=PIPE )
    runplugin.communicate()
    if runplugin.poll():
      raise Exception("Failed running run_rnaseqanalysis.py. See file %s."%run_log)

  if pluginParams['cmdOptions'].cmdline: return ({},{})
  printtime("Generating report...")

  # link over output resources here
  createlink( os.path.join(plugin_dir,'lifechart'), output_dir )

  # delete temporary/intermeddiate files after successful run (unless option to keep these is given)
  deleteTempFiles([ 'Log.progress.out', 'alignSTAR.bam', 'unmapped_remapBowtie2.bam', '*.fastq', '*.alignmentSummary.txt', '*.RNAmetrics.pdf' ])

  # parse out data in results text file to dict
  statsfile  = output_prefix+'.stats.txt'
  resultData = parseToDict( os.path.join(pluginParams['output_dir'],statsfile), ":" )

  # create html page of all result files available for download
  dlPage = "download_page.htm"
  if os.system( '%s "%s" "%s"' % ( os.path.join(pluginParams['plugin_dir'],'scripts','createFileIndex.sh'), 
      output_dir, os.path.join(output_dir,dlPage) ) ):
    printlog("WARNING: Failed to create file download page using createFileIndex.sh")

  cufPage = "cuflinks.htm"
  if os.system( '%s "%s" "%s"' % ( os.path.join(pluginParams['plugin_dir'],'scripts','createFileIndex.sh'), 
      os.path.join(output_dir,"output_cufflinks"), os.path.join(output_dir,cufPage) ) ):
    printlog("WARNING: Failed to create file cufflinks download page using createFileIndex.sh")

  # collect other output data to pluginReport, which is anything else that is used to generate the report
  reportData = {
    "run_name" : pluginParams['output_prefix'],
    "output_dir" : output_dir,
    "output_url" : output_url,
    "output_prefix" : output_prefix,
    "fpkm_thres" : fpkm_thres,
    "stats" : statsfile,
    "genereads" : output_prefix+".genereads.xls",
    "plot_generep" : output_prefix+".generep.png",
    "rnareads" : output_prefix+".rnareads.xls",
    "plot_rnareads" : output_prefix+".rnareads.png",
    "plot_transcov" : output_prefix+".STARBowtie2.RNAmetrics.png",
    "plot_isoexpr" : output_prefix+".geneisoexp.png",
    "dl_files" : dlPage,
    "cufflinks" : cufPage,
    "run_log" : run_log
  }
  return (resultData,reportData)


def run_meta_plugin():
  '''Wrapper for making command line calls to perform plugin meta-analyis. This is called after analysis has completed for individual barcodes.'''
  if pluginParams['cmdOptions'].cmdline: return
  printlog("")
  printtime("Creating barcodes summary report...")

  bcresults = pluginResult['barcodes']
  bcreports = pluginReport['barcodes']
  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['results_dir']

  # collect barcode statistics from the barcode summary table data and lists of output files
  warnMissingPlot = False
  bctable = []
  geneReadFiles = []
  distReadFiles = []
  featReadFiles = []
  cuffIsofFiles = []
  for bcdata in barcodeSummary:
    bcname = bcdata['barcode_name']
    bcrep = bcreports[bcname]
    if 'output_dir' not in bcrep: continue
    bcline = bcname+"\t"+bcdata['sample']+"\t"+bcdata['total_reads']+"\t"+bcdata['aligned_reads']+"\t"
    bcline += bcdata['pc_aliged_reads']+"\t"+bcdata['mean_read_length']+"\t"+bcdata['genes_detected']+"\t"+bcdata['isoforms_detected']
    bctable.append(bcline)
    reportfile = os.path.join(bcrep['output_dir'],bcrep['output_prefix']+".genereads.xls")
    if os.path.exists(reportfile):
      geneReadFiles.append(reportfile)
    reportfile = os.path.join(bcrep['output_dir'],bcrep['output_prefix']+".STARBowtie2.RNAmetrics.txt")
    if os.path.exists(reportfile):
      distReadFiles.append("'"+reportfile+"'")
    reportfile = os.path.join(bcrep['output_dir'],bcrep['output_prefix']+".rnareads.xls")
    if os.path.exists(reportfile):
      featReadFiles.append(reportfile)
    reportfile = os.path.join(bcrep['output_dir'],bcrep['output_prefix']+".isoforms.fpkm_tracking")
    if os.path.exists(reportfile):
      cuffIsofFiles.append(reportfile)

  # output text version of barcode summary table
  bctabfile = pluginParams['prefix']+".bc_summary.xls"
  if len(bctable) > 0:
    bcline = "Barcode ID\tSample Name\tTotal Reads\tAligned Reads\tPercent Aligned\tMean Read Length\tGenes Detected\tIsoforms Detected"
    with open(os.path.join(output_dir,bctabfile),'w') as outfile:
      outfile.write(bcline+'\n')
      for bcline in bctable:
        outfile.write(bcline+'\n')
    barcodeReport.update({"bctable":bctabfile})

  # parameters governing plots (may become available to user in future version)
  fpkm_thres =  pluginParams['config']['fpkm_thres']
  hm_samples = "250"
  hm_minrpm  = "100"
  hm_bcmrds  = "100000"
  hm_bcmexp  = "1000"
  hm_minfpkm = "100"
  minrvalue  = "0.4"

  # comparative analysis (plots and files) over all barcodes
  runR = "R --no-save --slave --vanilla --args"

  numReports = len(geneReadFiles)
  if numReports > 0:
    # generate barcode x gene reads and RPM reads tables
    typestr = "gene"
    bcmatrix = pluginParams['prefix']+".bcmatrix.xls"
    p_bcmatrix = os.path.join(output_dir,bcmatrix)
    with open(p_bcmatrix,'w') as outfile:
      runcmd = Popen( [os.path.join(plugin_dir,'scripts','geneMatrix.pl')] + geneReadFiles, stdout=outfile )
      runcmd.communicate()
      if runcmd.poll():
        raise Exception("Failed to create barcode x %s reads matrix."%typestr)
    rpmbcmatrix = pluginParams['prefix']+".rpm.bcmatrix.xls"
    with open(os.path.join(output_dir,rpmbcmatrix),'w') as outfile:
      runcmd = Popen( [os.path.join(plugin_dir,'scripts','geneMatrix.pl'), '-r'] + geneReadFiles, stdout=outfile )
      runcmd.communicate()
      if runcmd.poll():
        raise Exception("Failed to create barcode x %s RPM matrix."%typestr)

    # create correlation matrix plots from RPM reads matrix: generates the r-value matrix required for heatmap
    cpairsPlot = pluginParams['prefix']+".corpairs.png"
    cpairsTitle = "log2 RPM pair correlation plots" if numReports > 1 else "log2 RPM density plot"
    rvalueMatrix = pluginParams['prefix']+".rvalues.xls"
    if os.system( '%s "%s" "%s" %d "%s" "%s" < %s' % ( runR,
        rpmbcmatrix, cpairsPlot, numReports, cpairsTitle, os.path.join(output_dir,rvalueMatrix),
        os.path.join(plugin_dir,'scripts','plot_cormatrix.R') ) ):
      # temporarily avoid htseq-count issue of not finding any genes
      warnMissingPlot = True
      printlog("WARNING: Failed to create barcode RPM paired correlation plots using plot_cormatrix.R")

    # create overlaid transcript normalized coverage plot
    transcovPlot = pluginParams['prefix']+".transcov.png"
    if os.system( '%s "%s" %s < %s' % ( runR,
        os.path.join(output_dir,transcovPlot), " ".join(distReadFiles),
        os.path.join(plugin_dir,'scripts','plot_distcov.R') ) ):
      warnMissingPlot = True
      printlog("WARNING: Failed to create barcode transcript coverage plot using plot_distcov.R")

    # create barchart of mapped reads
    alignmentPlot = pluginParams['prefix']+".mapreads.png"
    if os.system( '%s "%s" "%s" "Reads Alignment Summary" "Million Reads" 0.000001 < %s' % ( runR,
        os.path.join(output_dir,bctabfile), os.path.join(output_dir,alignmentPlot),
        os.path.join(plugin_dir,'scripts','plot_thbar.R') ) ):
      warnMissingPlot = True
      printlog("WARNING: Failed to create barcode read alignment plot using plot_thbar.R")

    # create alignment distribution matrix and barchart plot of this
    featMatrix = pluginParams['prefix']+".feature_reads.xls"
    with open(os.path.join(output_dir,featMatrix),'w') as outfile:
      runcmd = Popen( [os.path.join(plugin_dir,'scripts','featureMatrix.pl')] + featReadFiles, stdout=outfile )
      runcmd.communicate()
      if runcmd.poll():
        warnMissingPlot = True
        printlog("WARNING: Failed to create barcode x feature alignment distribution matrix.")

    featPlot = pluginParams['prefix']+".feature_reads.png"
    if os.system( '%s "%s" "%s" "Alignment Distribution" "Million Base Reads" 0.000001 < %s' % ( runR,
        os.path.join(output_dir,featMatrix), os.path.join(output_dir,featPlot),
        os.path.join(plugin_dir,'scripts','plot_hbar.R') ) ):
      warnMissingPlot = True
      printlog("WARNING: Failed to create feature alignment distribution plot using plot_hbar.R")

    # create overlaid gene log10 distribution frequency curve (w/o genes with 0 reads)
    genepdfPlot = pluginParams['prefix']+".genepdf.png"
    if os.system( '%s "%s" "%s" %d "Distribution of Gene Reads" < %s' % ( runR,
        os.path.join(output_dir,bcmatrix), os.path.join(output_dir,genepdfPlot), numReports,
        os.path.join(plugin_dir,'scripts','plot_multi_pdf.R') ) ):
      # temporarily avoid htseq-count issue of not finding any genes
      printlog("WARNING: Failed to create gene read pdf plot using plot_multi_pdf.R")

    # create cufflinks isoform FPKM matrix
    isofMatrix = pluginParams['prefix']+".isoforms.xls"
    with open(os.path.join(output_dir,isofMatrix),'w') as outfile:
      runcmd = Popen( [os.path.join(plugin_dir,'scripts','isoformMatrix.pl')] + cuffIsofFiles, stdout=outfile )
      runcmd.communicate()
      if runcmd.poll():
        warnMissingPlot = True
        printlog("WARNING: Failed to create barcode x isoforms representation matrix.")

    # create isoform distribution matrix from cufflinks isoform FPKM matrix
    isofDistplot = pluginParams['prefix']+".isohist.png"
    if os.system( '%s "%s" "%s" "Distribution of Isoform Reads" %s < %s' % ( runR,
        os.path.join(output_dir,isofMatrix), os.path.join(output_dir,isofDistplot), fpkm_thres,
        os.path.join(plugin_dir,'scripts','plot_multi_isofrq.R') ) ):
      warnMissingPlot = True
      printlog("WARNING: Failed to create isoforms distribution plot using plot_multi_isofrq.R")

  # the remaining plots require that there are at least 2 barcodes to compare
  rvalueHeatmap = ''
  genevarHeatmap = ''
  isofHeatmap = ''
  if numReports > 1:
    # create heatmap plot from r-value matrix
    rvalueHeatmap = pluginParams['prefix']+".corbc.hm.png"
    if os.system( '%s "%s" "%s" "Correlation Heatmap" "r-value" %s < %s' % ( runR,
        os.path.join(output_dir,rvalueMatrix), os.path.join(output_dir,rvalueHeatmap), minrvalue,
        os.path.join(plugin_dir,'scripts','plot_corbc_heatmap.R') ) ):
      warnMissingPlot = True
      printlog("WARNING: Failed to create barcode heatmap plots using plot_corbc_heatmap.R")

    # create heatmaplot of top hm_samples (250) variant genes vs. barcode
    genevarHeatmap = pluginParams['prefix']+".genebc.hm.png"
    if os.system( '%s "%s" "%s" "Gene Representation Heatmap" "Representation: log10(RPM+1)" %s %s %s < %s' % ( runR,
        os.path.join(output_dir,bcmatrix), os.path.join(output_dir,genevarHeatmap), hm_samples, hm_bcmrds, hm_minrpm,
        os.path.join(plugin_dir,'scripts','plot_genebc_heatmap.R') ) ):
      warnMissingPlot = True
      printlog("WARNING: Failed to create barcode heatmap plots using plot_genebc_heatmap.R")

    # create heatmap plot for hm_samples (250) isoforms showing largest BC variation from cufflinks isoform FPKM matrix
    isofHeatmap = pluginParams['prefix']+".isoheatmap.png"
    if os.system( '%s "%s" "%s" "Transcript Isoform Representation Heatmap" "Representation: log10(FPKM+1)" %s %s %s %s < %s' % ( runR,
        os.path.join(output_dir,isofMatrix), os.path.join(output_dir,isofHeatmap), hm_samples, fpkm_thres, hm_bcmexp, hm_minfpkm,
        os.path.join(plugin_dir,'scripts','plot_isobc_heatmap.R') ) ):
      warnMissingPlot = True
      printlog("WARNING: Failed to create isoforms representation heatmap plot using plot_isobc_heatmap.R")

  # record output files for use in barcode summary report
  # (p_bcmatrix used for passing to php script for possible use with interactive utilities)
  if numReports > 0:
    barcodeReport.update({
      "bcmtype" : typestr,
      "fpkm_thres" : fpkm_thres,
      "hm_samples" : hm_samples,
      "hm_bcmrds" : hm_bcmrds,
      "hm_minrpm" : hm_minrpm,
      "hm_bcmexp" : hm_bcmexp,
      "hm_minfpkm" : hm_minfpkm,
      "minrvalue" : minrvalue,
      "bcmatrix" : bcmatrix,
      "p_bcmatrix" : p_bcmatrix,
      "rpmbcmatrix" : rpmbcmatrix,
      "rvalmatrix" : rvalueMatrix,
      "featmatrix" : featMatrix,
      "isofmatrix" : isofMatrix,
      "transcovplot" : transcovPlot,
      "genepdfplot" : genepdfPlot,
      "mapreadsplot" : alignmentPlot,
      "alignfeatplot" : featPlot,
      "isofdistplot" : isofDistplot,
      "rvalheatmap" : rvalueHeatmap,
      "geneheatmap" : genevarHeatmap,
      "isofheatmap" : isofHeatmap,
      "cpairsplot" : cpairsPlot,
      "warning" : ("Some plots are unavailable due to insufficient assigned reads for one or more barcodes." if warnMissingPlot else "")
    })

  # create symlink for js/css - the (empty) tabs on report page will not appear until this exists
  createlink( os.path.join(plugin_dir,'lifechart'), output_dir )


def updateBarcodeSummaryReport(barcode,autoRefresh=False):
  '''Create barcode summary (progress) report. Called before, during and after barcodes are being analysed.'''
  global barcodeSummary
  if pluginParams['cmdOptions'].cmdline: return
  if barcode != "":
    resultData = pluginResult['barcodes'][barcode]
    reportData = pluginReport['barcodes'][barcode]
    sample = resultData['Sample Name']
    if sample == '': sample = 'None'
    # Populate the specific barcodes_json list as requied for the Kendo table code
    if 'Error' in resultData:
      detailsLink = "<span class='help' title='%s' style='color:red'>%s</span>" % ( resultData['Error'], barcode )
      run_log = os.path.join(pluginParams['output_dir'],'rnaseq.log')
      if os.path.exists( run_log ):
        detailsLink = "<a target='_parent' href='%s'>%s<a>" % ( os.path.join(barcode,'rnaseq.log'), detailsLink )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "total_reads": "NA",
        "aligned_reads": "NA",
        "pc_aliged_reads": "NA",
        "mean_read_length": "NA",
        "genes_detected": "NA",
        "isoforms_detected": "NA"
      })
    else:
      detailsLink = "<a target='_parent' href='%s' class='help'><span title='Click to view the detailed report for barcode %s'>%s</span><a>" % (
        os.path.join(barcode,pluginParams['report_name']), barcode, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "total_reads": resultData.get("Total Reads","NA"),
        "aligned_reads": resultData.get("Aligned Reads","NA"),
        "pc_aliged_reads": resultData.get("Pct Aligned","NA"),
        "mean_read_length": resultData.get("Mean Read Length","NA"),
        "genes_detected": resultData.get("Genes with 10+ reads","NA"),
        "isoforms_detected": resultData.get("Isoforms Detected","NA")
      })
  config = pluginParams['config']
  render_context = {
    "autorefresh" : autoRefresh,
    "run_name" : pluginParams['prefix'],
    "ref_name" : config['genome'],
    "adapter" : config['cutadapt'],
    "sampreads" : str(100*float(config['fraction_of_reads']))+'%',
    "num_barcodes_filtered" : pluginReport['num_barcodes_filtered'],
    "barcode_filter" : pluginReport['barcode_filter'],
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
    barcodeData = barcodeSpecifics(NONBARCODED)
    if barcodeData: sample = barcodeData.get('sample','None')
    if sample == 'none': sample = 'None'
  config = pluginParams['config']
  render_context = {
    "autorefresh" : (errorMsg == ""),
    "run_name": pluginParams['prefix'],
    "ref_name" : config['genome'],
    "adapter" : config['cutadapt'],
    "sampreads" : str(100*float(config['fraction_of_reads']))+'%',
    "Sample_Name": sample,
    "Error": errorMsg }
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'incomplete.html', render_context )


def createDetailReport(resultData,reportData):
  '''Called to create the main report (for un-barcoded run or for each barcode).'''
  if pluginParams['cmdOptions'].cmdline: return
  render_context = resultData.copy()
  render_context.update(reportData)
  createReport( os.path.join(pluginParams['output_dir'],pluginParams['report_name']), 'report.html', render_context )


def createBlockReport():
  '''Called at the end of run to create a block.html report. Use 'pass' if not wanted.'''
  if pluginParams['cmdOptions'].cmdline: return
  printtime("Creating block report...")
  if pluginParams['barcoded']:
    config = pluginParams['config']
    numErrs = pluginReport['num_barcodes_failed'] + pluginReport['num_barcodes_invalid']
    render_context = {
      "run_name" : pluginParams['prefix'],
      "barcode_results" : simplejson.dumps(barcodeSummary),
      "ref_name" : config['genome'],
      "adapter" : config['cutadapt'],
      "sampreads" : str(100*float(config['fraction_of_reads']))+'%',
      "Error" : '%d barcodes failed analysis.'%numErrs if numErrs > 0 else ''
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

def barcodeSpecifics(barcode=""):
  '''Process the part of the plan that customizes barcodes to different references.'''
  if not barcode:
    barcode = NOMATCH
  if not barcode in barcodeInput:
    return { "filtered" : True }

  barcodeData = barcodeInput[barcode]
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

  bedfile = barcodeData['target_region_filepath']

  return {
    "filtered" : filtered,
    "sample" : sample,
    "reference" : reference,
    "refpath" : refPath,
    "nuctype" : barcodeData.get('nucleotide_type','DNA'),
    "refurl" : '{http_host}'+genomeUrl if genomeUrl else reference,
    "bedfile" : bedfile,
    "bamfile" : barcodeData.get('bam_filepath','') }

def checkBarcode(barcode,relBamSize=0):
  '''Checks if a specific barcode is set up correctly for analysis.'''
  # This plugin does not care about whether reads were mapped to a reference or targeted regions
  barcodeData = barcodeSpecifics(barcode)
  if barcodeData['filtered']:
    return "Filtered (not enough reads)"
  #if not barcodeData['reference']:
  #  return "Analysis requires alignment to a reference"
  if not os.path.exists(barcodeData['bamfile']):
    return "BAM file not found at " + barcodeData['bamfile']
  fileSize = os.stat(barcodeData['bamfile']).st_size
  if fileSize < pluginParams['cmdOptions'].minbamsize:
    return "BAM file too small"
  if fileSize < relBamSize:
    return "BAM file too small relative to largest"
  #return checkTarget( barcodeData['bedfile'], barcodeData['bamfile'] )
  return ""

def emptyResultsFolder():
  '''Purge everything in output folder except for specifically named files.'''
  if not pluginParams['cmdOptions'].purge_results: return
  results_dir = pluginParams['results_dir']
  if results_dir == '/': return
  logopt = pluginParams['cmdOptions'].logopt
  if logopt or os.path.exists( os.path.join(results_dir,pluginParams['report_name']) ):
    printlog("Purging old results...")
  for fname in os.listdir(results_dir):
    # avoid specific files needed to launch run
    if not os.path.isdir(fname):
      start = os.path.basename(fname)[:10]
      if start == "drmaa_stdo" or start == "ion_plugin" or start == "startplugi" or start == 'barcodes.j':
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
  printlog( '(%s) %s'%(time.strftime('%X'),msg) )

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
  noErrMsg = "2> /dev/null" if pluginParams['cmdOptions'].skip_analysis else ""
  os.system('ln -s "%s" "%s" %s'%(srcPath,destPath,noErrMsg))
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
  # render the report template
  # NOTE: non-alphnum chars in keys are converted to underscores to avoid Django issues
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

  # get FILEPATH_OUTPUT_STEM or create old default if not available
  pluginParams['prefix'] = jsonParams['expmeta'].get('output_file_name_stem','')
  if not pluginParams['prefix']:
    pluginParams['prefix'] = jsonParams['expmeta'].get('run_name','auto')
    if 'results_name' in jsonParams['expmeta']:
      pluginParams['prefix'] += "_" + jsonParams['expmeta']['results_name']

  # TODO: replace this with url_plugindir when available from startplugin.json
  resurl = jsonParams['runinfo'].get('results_dir','.')
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

  # extra test for whether this is a supported genome
  if config['reference'] == "" and launchmode != 'Manual':
    raise Exception("CATCH: Analysis requires a reference sequence is specified using plugin configuration.")
  if config['genome'] != "hg19" and config['genome'] != "mm10":
    raise Exception("CATCH: Analysis is currently only allowed for the 'hg19' and 'mm10' genomes.")

  # plugin configuration becomes basis of results.json file
  global pluginResult, pluginReport
  pluginResult = pluginParams['config'].copy()
  pluginResult['barcoded'] = pluginParams['barcoded']
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
    barcodeData = barcodeSpecifics(NONBARCODED)
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
    printerr('Analysis failed')
    pluginResult.update({ 'Error': str(e) })
    createIncompleteReport(str(e))
    raise
  if pluginParams['cmdOptions'].scraper:
    createScraperLinksFolder( pluginParams['output_dir'], pluginParams['output_prefix'] )

def runForBarcodes():
  # iterate over listed barcodes to pre-test barcode files
  global pluginParams, pluginResult, pluginReport
  barcodes = getOrderedBarcodes()

  # scan for largest BAM file size to set relative minimum
  relBamSize = 0
  if barcode_filter > 0:
    maxBamSize = 0
    for barcode in barcodes:
      bcbam = barcodeInput[barcode].get('bam_filepath','')
      if os.path.exists(bcbam):
        fsiz = os.stat(bcbam).st_size
        if fsiz > maxBamSize: maxBamSize = fsiz
    relBamSize = int(barcode_filter * maxBamSize)

  numGoodBams = 0
  numBamSmall = 0
  maxBarcodeLen = 0
  numInvalidBarcodes = 0
  barcodeIssues = []
  for barcode in barcodes:
    errmsg = checkBarcode(barcode,relBamSize)
    if not errmsg:
      if( len(barcode) > maxBarcodeLen ):
        maxBarcodeLen = len(barcode)
      numGoodBams += 1
    elif errmsg[:6] == "ERROR:":
      errmsg = "\n"+errmsg
      numInvalidBarcodes += 1
      if errmsg.find('relative to largest'):
        numBamSmall += 1
    barcodeIssues.append(errmsg)

  ensureFilePrefix(maxBarcodeLen+1)
  pluginReport['num_barcodes_processed'] = numGoodBams
  pluginReport['num_barcodes_invalid'] = numInvalidBarcodes
  pluginReport['num_barcodes_failed'] = 0
  pluginReport['num_barcodes_filtered'] = numBamSmall
  pluginReport['barcode_filter'] = 100*barcode_filter
  pluginResult['Barcodes filtered'] = str(numBamSmall)

  skip_analysis = pluginParams['cmdOptions'].skip_analysis
  stop_on_error = pluginParams['cmdOptions'].stop_on_error
  create_scraper = pluginParams['cmdOptions'].scraper

  # create initial (empty) barcodes summary report
  if numBamSmall > 1:
    printlog("WARNING: %d bamfiles discounted as too small compared to largest BAM file.\n"%numBamSmall)
  printlog("Processing %d barcodes..." % numGoodBams)
  updateBarcodeSummaryReport("",True)

  # iterate over all barcodes and process the valid ones
  postout = False; # just for logfile prettiness
  barcodeProcessed = 0
  for barcode in barcodes:
    barcodeData = barcodeSpecifics(barcode)
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
        printlog('Reference File: %s' % barcodeData['refpath'])
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
        pluginResult['barcodes'][barcode] = { "Sample_Name" : sample, "Error" : str(e) }
        pluginReport['barcodes'][barcode] = {}
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
  writeDictToJsonFile(pluginResult,os.path.join(pluginParams['results_dir'],"results.json"))

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
        printlog("WARNING: No barcode alignment files were found for this barcoded run.")
        createProgressReport("No barcode alignment files were found.")
      elif pluginReport['num_barcodes_processed'] == pluginReport['num_barcodes_failed']:
        printlog("ERROR: Analysis failed for all barcodes.")
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

