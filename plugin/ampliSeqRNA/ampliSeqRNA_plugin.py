#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

# NOTE: Code for GUI barcode-specifc targets override is included but not currently utilized
# for ampliSeqRNA, since this currently does not make sense with respect to comparing barcodes.

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

# defines exceptional barcode IDs to check for
NONBARCODED = "nonbarcoded"
NOMATCH = "nomatch"

# max plugin out file ext length for addressing max filename length (w/ barcode), e.g. ".amplicon.cov.xls"
max_fileext_len = 17
max_filename_len = 255

# ratio for minimal to maximum barcode BAM size for barcode to be ignored
barcode_filter = 0.1

# flag to create diferential expression matrix and significance ratio threshold for report summary
create_DE_matrix = False
difexp_thresh = 2

#
# ----------- inialize django and custom tag additions -------------
#
from django import template
register = template.Library()

@register.filter 
def toMega(value):
  return float(value) / 1000000

template.builtins.append(register) 

# global data collecters common to functions
barcodeInput = {}
pluginParams = {}
pluginResult = {}
pluginReport = {}
barcodeSummary = []
barcodeReport = {}
help_dictionary = {}
warningMessage = ""

#
# -------------------- customize code for this plugin here ----------------
#

def addAutorunParams(plan=None):
  '''Additional parameter set up for auto-mated runs, e.g. to add defaults for option only in the GUI.'''
  # Note: this function may be redundant in later TSS versions, where pluginconfig already consolidates input from all sources
  config = pluginParams['config']
  # defaults for auto-run to match config settings from GUI
  config['librarytype'] = 'ampiseq-rna'
  config['librarytype_id'] = 'Ion AmpliSeq RNA'
  config['targetregions'] = ''
  config['targetregions_id'] = 'None'
  config['barcodebeds'] = 'No'
  config['barcodetargetregions'] = ''
  # GUI-only options that might be set by plan.html
  if config.get('filterbarcodes','') == '': config['filterbarcodes'] = 'Yes'
  if config.get('ercc','') == '': config['ercc'] = 'No'
  if config.get('uniquemaps','') == '': config['uniquemaps'] = 'No'
  # extract things from the plan if provided - for coverageAnalysis auto-run w/o a plan leads to early exit
  if plan: 
    runtype = plan['runType']
    if runtype != 'AMPS_RNA':
      config['librarytype_id'] = "[%s]"%runtype
      raise Exception("CATCH:Do not know how to analyze coverage for unsupported plan runType: '%s'"%runtype)
    bedfile = plan['bedfile']
    if bedfile != "":
      config['targetregions'] = bedfile
      config['targetregions_id'] = fileName(bedfile)
    else:
      raise Exception("CATCH:Automated analysis requires a targets region to be specified by the Plan.")    
  else:
    raise Exception("CATCH:Automated analysis requires a Plan to specify Run Type.")    


def furbishPluginParams():
  '''Complete/rename/validate user parameters.'''
  # For example, HTML form posts do not add unchecked option values
  config = pluginParams['config']
  config['barcodebeds'] = 'Yes' if config.get('barcodebeds',False) else 'No'
  config['filterbarcodes'] = 'Yes' if config.get('filterbarcodes',False) else 'No'
  config['ercc'] = 'Yes' if config.get('ercc',False) else 'No'
  config['uniquemaps'] = 'Yes' if config.get('uniquemaps',False) else 'No'


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
    "Filter Barcodes" : config['filterbarcodes'],
    "Barcode-specific Targets" : config['barcodebeds'],
    "ERCC Tracking" : config['ercc'],
    "Use Only Uniquely Mapped Reads" : config['uniquemaps'],
    "barcoded" : "true" if pluginParams['barcoded'] else "false"
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
  printlog('Run parameters:')
  printlog('  Library Type:     %s' % config['librarytype_id'])
  printlog('  Reference Name:   %s' % pluginParams['genome_id'])
  printlog('  Target Regions:   %s' % config['targetregions_id'])
  printlog('  Filter Barcodes:  %s' % config['filterbarcodes'])
  # the GUI option to override barcode-specific targets is not used
  if config['barcodebeds'] == 'Yes' and pluginParams['manual_run']:
    target_files = pluginParams['target_files']
    for bctrg in sorted(target_files):
      printlog('    %s  %s' % (bctrg,fileName(target_files[bctrg])))
  printlog('  ERCC Tracking:    %s' % config['ercc'])
  printlog('  Uniquely Mapped:  %s' % config['uniquemaps'])
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
  librarytype = config['librarytype']
  barcodeData = barcodeSpecifics(barcode)

  # link from source BAM since pipeline uses the name as output file stem
  linkbam = os.path.join(output_dir,output_prefix+".bam")
  createlink(barcodeData['bamfile'],linkbam)
  createlink(barcodeData['bamfile']+'.bai',linkbam+'.bai')
  bamfile = linkbam
  bedfile = barcodeData['bedfile']

  # read filters - some hard coded for now
  filtopts = '-a -R 17'
  if config['uniquemaps'] == 'Yes': filtopts += ' -u'

  # skip the actual and assume all the data already exists in this file for processing
  if skiprun:
    printlog("Skipped analysis - generating report on in-situ data")
  else:
    # Pre-run modification of BED files is done here to avoid redundancy of repeating for target assigned barcodes
    # The stand-alone command can perform the (required) annotation. (The mergeBed filepath is not used.)
    reference = barcodeData['refpath']
    (mergeBed,annoBed,erccBed) = modifyBedFiles(bedfile,reference)
    runcmd = '%s %s %s -D "%s" -L "%s" -N "%s" -S "%s" -T "%s" "%s" "%s" "%s"' % (
        os.path.join(plugin_dir,'run_ampliseqrna.sh'), pluginParams['logopt'], filtopts,
        output_dir, barcodeData['reference'], barcodeData['sample'], erccBed, fileName(bedfile),
        reference, bamfile, annoBed )
    if logopt: printlog('\n$ %s\n'%runcmd)
    if( os.system(runcmd) ):
      raise Exception("Failed running run_ampliseqrna.sh. Refer to Plugin Log.")

  if pluginParams['cmdOptions'].cmdline: return ({},{})
  printtime("Generating report...")

  # Link report page resources. This is necessary as the plugin code is inaccesible from URLs directly.
  createlink( os.path.join(plugin_dir,'flot'), output_dir )
  createlink( os.path.join(plugin_dir,'lifechart'), output_dir )
   
  # Optional: Delete intermediate files after successful run. These should not be required to regenerate any of the
  # report if the skip-analysis option. Temporary file deletion is also disabled when the --keep_temp option is used.
  #deleteTempFiles([ '*.bam', '*.bam.bai', '*.bed' ])

  # Create an annotated list of files as used to create the file links table.
  # - Could be handled in the HTML template directly but external code is re-used to match cmd-line reports.

  # Parse out stats from results text file to dict AND convert unacceptible characters to underscores in keys to avoid Django issues
  statsfile = output_prefix+'.stats.cov.txt'
  resultData = parseToDict( os.path.join(output_dir,statsfile), ":" )

  # Collect other output data to pluginReport, which is anything else that is used to generate the report
  trgtype = '.amplicon.cov'
  reportData = {
    "library_type" : config['librarytype_id'],
    "run_name" : output_prefix,
    "barcode_name" : barcode,
    "ercc_track" : (config['ercc'] == 'Yes'),
    "output_dir" : output_dir,
    "output_url" : output_url,
    "output_prefix" : output_prefix,
    "help_dict" : helpDictionary(),
    "stats_txt" : checkOutputFileURL(statsfile),
    "rep_overview_png" : checkOutputFileURL(output_prefix+'.repoverview.png'),
    "finecov_tsv" : checkOutputFileURL(output_prefix+trgtype+'.xls'),
    "bed_link" : re.sub( r'^.*/uploads/BED/(\d+)/.*', r'/rundb/uploadstatus/\1/', bedfile ),
    "file_links" : checkOutputFileURL('filelinks.xls'),
    "bam_link" : checkOutputFileURL(output_prefix+'.bam'),
    "bai_link" : checkOutputFileURL(output_prefix+'.bam.bai')
  }
  return (resultData,reportData)

def checkOutputFileURL(fileURL):
  '''coverageAnalysis helper method to return "" if the provided file URL does not exist'''
  if os.path.exists(os.path.join(pluginParams['output_dir'],fileURL)):
    return fileURL
  return ""

def modifyBedFiles(bedfile,reference):
  '''ampliSeqRNA method to return merged, GC annotated, and ERCC BED files, creating them if they do not already exist.'''
  if not bedfile: return ('','','')
  # files will be created or found in this results subdir
  bedDir = os.path.join(pluginParams['results_dir'],"local_beds")
  if not os.path.exists(bedDir): os.makedirs(bedDir)
  # the pair of files returned are dependent on the Library Type
  rootbed = fileName(bedfile)
  mergbed = bedfile.replace('unmerged','merged',1)
  # do not re-do GC annotation on same BED file - moderately expensive for large BEDs
  gcbed = os.path.join(bedDir,"%s.gc.bed"%rootbed)
  if os.path.exists(gcbed):
    printlog("Adopting GC annotated targets %s"%os.path.basename(gcbed))
  else:
    printtime("Creating GC annotated targets %s"%os.path.basename(gcbed))
    if os.system( '%s -s -w -f 4,8 -t "%s" "%s" "%s" > "%s"' % (
        os.path.join(pluginParams['plugin_dir'],'bed','gcAnnoBed.pl'),
        bedDir, bedfile, reference, gcbed ) ):
      raise Exception("Failed to annotate target regions using gcAnnoBed.pl")
  annobed = gcbed
  erccbed = ''
  if pluginParams['config']['ercc'] == 'Yes':
    erccbed = os.path.join(bedDir,"%s.ercc.bed"%rootbed)
    num_ercc = countFileLines(erccbed)
    if num_ercc >= 0:
      printlog("Adopting %d ERCC targets from %s"%(num_ercc,os.path.basename(erccbed)))
    elif num_ercc < 0:
      printtime("Creating ERCC targets file %s"%os.path.basename(erccbed))
      if os.system( "awk '$1~/^ERCC-/ {print}' '%s' > '%s'" % (bedfile,erccbed) ):
        raise Exception("Failed to create ERCC targets file using awk command")
      num_ercc = countFileLines(erccbed)
      if num_ercc > 0:
        printlog("  %d ERCC targets detected in target panel."%num_ercc)
      if num_ercc == 0:
        printlog("WARNING: No ERCC targets were detected in targets panel!")
  return (mergbed,annobed,erccbed)

def countFileLines(fpath):
  '''Utility function to return number of lines in a file or -1 if the file does not exist.'''
  if os.path.exists(fpath):
    nlines = 0
    with open(fpath) as f:
      for line in f: nlines += 1
    return nlines
  else:
    return -1

def run_meta_plugin():
  '''Create barcode x target reads matrix files and derived files and plots.'''
  global warningMessage
  if pluginParams['cmdOptions'].cmdline: return
  printlog("")
  printtime("Creating barcodes summary report...")

  # collect barcode statistics from the barcode summary table data and lists of output files
  renderOpts = renderOptions()
  typestr = 'amplicon'
  fileext = '.'+typestr+'.cov.xls'
  bctable = []
  readstable = []
  reportFiles = []
  bclist = ''
  bcresults = pluginResult['barcodes']
  bcreports = pluginReport['barcodes']
  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['results_dir']

  # iterate barcodeSummary[] to maintain barcode processing order
  for bcdata in barcodeSummary:
    bcname = bcdata['barcode_name']
    if bclist: bclist += ','
    bclist += bcname
    bcrep = bcreports[bcname]
    bcline = bcname+"\t"+bcdata['sample']+"\t"+bcdata['mapped_reads']+"\t"+bcdata['valid_target']+"\t"+bcdata['detected_target']
    if renderOpts['ercc_track']: bcline += "\t"+bcdata['ercc_target']
    bctable.append(bcline)
    reportfile = os.path.join( bcrep['output_dir'], bcrep['output_prefix']+fileext )
    bcline = bcname+"\t"+bcdata['sample']+"\t"+bcdata['total_reads']+"\t"+bcdata['mapped_reads']+"\t"+bcdata['ontrg_reads']+"\t"+bcdata['valid_reads']
    if renderOpts['ercc_track']: bcline += "\t"+bcdata['ercc_reads']
    else: bcline += "\t0"
    readstable.append(bcline)
    if os.path.exists(reportfile):
      reportFiles.append(reportfile)

  if len(bctable) > 0:
    bctabfile = pluginParams['prefix']+".bc_summary.xls"
    bcline = "Barcode ID\tSample Name\tMapped Reads\tValid Reads\tTargets Detected"
    if renderOpts['ercc_track']: bcline += "\tERCC"
    with open(os.path.join(output_dir,bctabfile),'w') as outfile:
      outfile.write(bcline+'\n')
      for bcline in bctable:
        outfile.write(bcline+'\n')
    readsfile = pluginParams['prefix']+".reads_summary.xls"
    with open(os.path.join(output_dir,readsfile),'w') as outfile:
      outfile.write('Barcode ID\tSample Name\tTotal Reads\tMapped Reads\tOn Target Reads\tAssigned Reads\tERCC Reads\n')
      for bcline in readstable:
        outfile.write(bcline+'\n')
    barcodeReport.update({"bctable":bctabfile,"readstable":readsfile})

  # comparative analysis (plots and files) over all barcodes
  # NOTE: Here pluginParams['reference'] is used as the single common reference.
  runR = "R --no-save --slave --vanilla --args"

  numReports = len(reportFiles)
  if numReports > 0:
    bcmatrix = pluginParams['prefix']+".bcmatrix.xls"
    p_bcmatrix = os.path.join(output_dir,bcmatrix)
    with open(p_bcmatrix,'w') as outfile:
      runcmd = Popen( [os.path.join(plugin_dir,'scripts','barcodeMatrix.pl'),
        '-A', 'A_', pluginParams['reference']+'.fai', '9'] + reportFiles, stdout=outfile )
      runcmd.communicate()
      if runcmd.poll():
        raise Exception("Failed to create barcode x %s reads matrix."%typestr)
    rpmbcmatrix = pluginParams['prefix']+".rpm.bcmatrix.xls"
    with open(os.path.join(output_dir,rpmbcmatrix),'w') as outfile:
      runcmd = Popen( [os.path.join(plugin_dir,'scripts','barcodeMatrix.pl'),
        '-A', 'A_', pluginParams['reference']+'.fai', '12'] + reportFiles, stdout=outfile )
      runcmd.communicate()
      if runcmd.poll():
        raise Exception("Failed to create barcode x %s RPM matrix."%typestr)
    
    # convert normalized RPM to CHP format for TAC software
    rpm2log = os.path.join(plugin_dir,"scripts","log2rpm.pl");
    log2rpm = pluginParams['prefix']+".log2rpm.xls";
    chp_con = os.path.join(plugin_dir,"tac","convert2chp.py");
    chp_zip = pluginParams['prefix']+".log2rpm.zip";
    if os.system('%s %d "%s" > "%s"' % (rpm2log, numReports, os.path.join(output_dir,rpmbcmatrix), os.path.join(output_dir,log2rpm))):
      raiseWarning("Failed to create log2(rpm+1) files for %s." % rpmbcmatrix)
    else:
      chp_dir = "convert2chp"
      if os.system('python %s -A "%s" -n "%s" -P %s -V "%s" -m %s -i %s -o %s' % ( chp_con, fileName(pluginParams['config']['targetregions_id']),
	pluginParams['prefix']+"_", pluginParams['plugin_name'], pluginParams['cmdOptions'].version,
        'RPM-normalized', os.path.join(output_dir,log2rpm), os.path.join(output_dir,chp_dir)) ):
        raiseWarning("Failed to create CHP files for %s."%log2rpm)
      else:
        os.system('zip -r %s %s'%(chp_zip,chp_dir))

    deSummary = ""
    derTable = ""
    if create_DE_matrix:
      if numReports > 1:
        derTable = pluginParams['prefix']+".deratio.xls"
        if os.system( '%s -N 1000000 -S RPM "%s" -a -M %d > "%s"' % (
            os.path.join(plugin_dir,'scripts','tableDE.pl'), bcmatrix, numReports, derTable) ):
          raise Exception("Failed to create differential expression matrix using tableDE.pl")
        if os.system( 'awk \'NR==1;NR>1{print|"sort -k 6,6nr -k 1,1d -k 2,2d"}\' "%s" > sort.xls.tmp; mv sort.xls.tmp "%s"' % (derTable,derTable) ):
          raise Exception("Failed to sort differential expression matrix using awk command.")
        with open(os.path.join(output_dir,derTable),'r') as infile:
          nline = 0
          nde = 0
          for line in infile:
            nline += 1
            if nline == 1: continue
            fields = line.split('\t')
            if float(fields[len(fields)-1]) >= difexp_thresh: nde += 1
          if nline > 1:
            deSummary = "%d targets (%.2f%%) showed differential expression at %s-fold or greater." % (nde,100*float(nde)/(nline-1),difexp_thresh)
            pluginResult['Differentially expressed targets'] = str(nde)

    # create barchart of mapped reads - works with 1 barcode
    alignmentPlot = pluginParams['prefix']+".mapreads.png"
    if os.system( '%s "%s" "%s" "Reads Alignment Summary" "Million Reads" 0.000001 < %s' % ( runR,
        os.path.join(output_dir,readsfile), os.path.join(output_dir,alignmentPlot),
        os.path.join(plugin_dir,'scripts','plot_reads_hbar.R') ) ):
      raiseWarning("Failed to create barcode read alignment plot using plot_reads_hbar.R")

    # create overlaid gene log10 distribution frequency curve (w/o genes with 0 reads)
    genepdfPlot = pluginParams['prefix']+".genepdf.png"
    if os.system( '%s "%s" "%s" %d "Distribution of Gene Reads" < %s' % ( runR,
        os.path.join(output_dir,bcmatrix), os.path.join(output_dir,genepdfPlot), numReports,
        os.path.join(plugin_dir,'scripts','plot_multi_pdf.R') ) ):
      raiseWarning("Failed to create gene read pdf plot using plot_multi_pdf.R")

    # create correlation matrix plots from RPM reads matrix: generates the r-value matrix required for heatmap
    cpairsPlot = pluginParams['prefix']+".corpairs.png"
    cpairsTitle = "log2 RPM pair correlation plots" if numReports > 1 else "log2 RPM density plot"
    rvalueMatrix = pluginParams['prefix']+".rvalues.xls"
    if os.system( '%s "%s" "%s" %d "%s" "%s" < %s' % ( runR,
        rpmbcmatrix, cpairsPlot, numReports, cpairsTitle, os.path.join(output_dir,rvalueMatrix),
        os.path.join(plugin_dir,'scripts','plot_cormatrix.R') ) ):
      raiseWarning("Failed to create barcode RPM paired correlation plots using plot_cormatrix.R")

    # create heatmap plot from r-value matrix
    rvalueHeatmap = pluginParams['prefix']+".corbc.hm.png"
    if numReports > 1 and os.system( '%s "%s" "%s" "Correlation Heatmap" "r-value" 0.4 < %s' % ( runR,
        os.path.join(output_dir,rvalueMatrix), os.path.join(output_dir,rvalueHeatmap),
        os.path.join(plugin_dir,'scripts','plot_corbc_heatmap.R') ) ):
      raiseWarning("Failed to create barcode heatmap plots using plot_corbc_heatmap.R")

    # create heatmaplot of top 250 variant genes vs. barcode
    genevarHeatmap = pluginParams['prefix']+".genebc.hm.png"
    if numReports > 1 and os.system( '%s "%s" "%s" "Gene Representation Heatmap" "Representation: log10(RPM+1)" 250 10000 100 %d < %s' % ( runR,
        os.path.join(output_dir,bcmatrix), os.path.join(output_dir,genevarHeatmap), numReports,
        os.path.join(plugin_dir,'scripts','plot_genebc_heatmap.R') ) ):
      raiseWarning("Failed to create barcode heatmap plots using plot_genebc_heatmap.R")

    # give warning about missing plots if not one already
    if warningMessage == "" and numReports < 2:
      warningMessage = "Some plots unavailable without multiple barcodes."

    # record output files for use in barcode summary report
    # (p_bcmatrix used for passing to php script for interactive utilities)
    barcodeReport.update({
      "bclist" : bclist,
      "bcmtype" : typestr,
      "bcmatrix" : bcmatrix,
      "p_bcmatrix" : p_bcmatrix,
      "rpmbcmatrix" : rpmbcmatrix,
      "rpmchp" : chp_zip,
      "rvaluematrix" : rvalueMatrix,
      "readmaps" : readsfile,
      "featmatrix" : rvalueMatrix,
      "genepdfplot" : genepdfPlot,
      "mapreadsplot" : alignmentPlot,
      "heatmapplot" : rvalueHeatmap,
      "genebcplot" : genevarHeatmap,
      "cpairsplot" : cpairsPlot
    })

def raiseWarning(msg):
  global warningMessage
  #raise Exception(msg)
  printlog("WARNING: "+msg)
  warningMessage = "Run completed with warnings. Refer to Plugin Log.";


def updateBarcodeSummaryReport(barcode,autoRefresh=False):
  '''Create barcode summary (progress) report. Called before, during and after barcodes are being analysed.'''
  global barcodeSummary
  if pluginParams['cmdOptions'].cmdline: return
  renderOpts = renderOptions()
  # no barcode means either non have been ceated yet or this is a refresh after all have been processed (e.g. for meta data)
  if barcode != "":
    resultData = pluginResult['barcodes'][barcode]
    reportData = pluginReport['barcodes'][barcode]
    sample = resultData['Sample Name']
    if sample == '': sample = 'None'
    # check for error status - fill 0's if error due to 0 reads
    errMsg = ""
    errZero = False
    if 'Error' in resultData:
      errMsg = resultData['Error']
      errZero = "no mapped" in errMsg or "no read" in errMsg
    # barcodes_json dictoonary is firmcoded in Kendo table template that we are using for main report styling
    if errMsg != "":
      detailsLink = "<span class='help' title='%s' style='color:red'>%s</span>" % ( errMsg, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "total_reads": "0" if errZero else "NA",
        "mapped_reads": "0" if errZero else "NA",
        "ontrg_reads": "0" if errZero else "NA",
        "valid_reads": "0" if errZero else "NA",
        "ercc_reads": "0" if errZero else "NA",
        "valid_target": "0.00%" if errZero else "NA",
        "detected_target": "0.00%" if errZero else "NA",
        "ercc_target": "NA"
      })
    else:
      detailsLink = "<a target='_parent' href='%s' class='help'><span title='Click to view the detailed report for barcode %s'>%s</span><a>" % (
        os.path.join(barcode,pluginParams['report_name']), barcode, barcode )
      numTargets = int(resultData['Number of amplicons'])
      pcDetected = 100*float(resultData['Amplicons with at least 10 reads'])/numTargets if numTargets > 0 else 0
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "sample" : sample,
        "total_reads": resultData['Number of total reads'],
        "mapped_reads": resultData['Number of mapped reads'],
        "ontrg_reads": resultData['Number of on-target reads'],
        "valid_reads": resultData['Number of assigned reads'],
        "ercc_reads": resultData['Number of ERCC tracking reads'] if renderOpts['ercc_track'] else "NA",
        "valid_target": resultData['Percent assigned reads'],
        "detected_target": ("%.2f"%pcDetected)+"%",
        "ercc_target": resultData['Percent ERCC tracking reads'] if renderOpts['ercc_track'] else "NA"
      })
  render_context = {
    "autorefresh" : autoRefresh,
    "run_name" : pluginParams['prefix'],
    "barcode_results" : simplejson.dumps(barcodeSummary),
    "num_barcodes_filtered" : pluginReport['num_barcodes_filtered'],
    "barcode_filter" : pluginReport['barcode_filter'],
    "help_dict" : helpDictionary()
  }
  render_context.update(renderOpts)
  # extra report items, e.g. file links from barcodes summary page
  if barcodeReport:
    render_context.update(barcodeReport)
    if warningMessage != "":
      render_context.update({"WarningMsg":warningMessage})
  createReport( os.path.join(pluginParams['results_dir'],pluginParams['report_name']), 'barcode_summary.html', render_context )


def renderOptions():
  '''coverageAnalysis support method to generate list of condensed rendering options and values.'''
  config = pluginParams['config']
  targets = 'Barcode-specific' if config['barcodebeds'] == 'Yes' else config['targetregions_id']
  if targets == 'None': targets = ""
  filter_options = []
  if config['ercc'] == 'Yes': filter_options.append('ERCC tracking')
  if config['uniquemaps'] == 'Yes': filter_options.append('Uniquely mapped')
  # extra filters may become a user option
  filter_options.append('Alignment length (17+)')
  return {
    "library_type" : config['librarytype_id'],
    "target_regions" : targets,
    "filter_options" : ', '.join(filter_options),
    "ercc_track" : (config['ercc'] == 'Yes')
  }
  

def createIncompleteReport(errorMsg=""):
  '''Called to create an incomplete or error report page for non-barcoded runs.'''
  if pluginParams['barcoded']:
    sample = 'None'
  else:
    barcodeData = barcodeSpecifics(NONBARCODED)
    if barcodeData: sample = barcodeData.get('sample','None')
    if sample == 'none': sample = 'None'
  render_context = {
    "autorefresh" : (errorMsg == ""),
    "run_name": pluginParams['prefix'],
    "Sample_Name": sample,
    "Error": errorMsg }
  render_context.update(renderOptions())
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
    render_context = {
      "run_name" : pluginParams['prefix'],
      "barcode_results" : simplejson.dumps(barcodeSummary),
      "help_dict" : helpDictionary() }
    render_context.update( renderOptions() )
    tplate = 'barcode_block.html'
  else:
    render_context = pluginResult.copy()
    render_context.update(pluginReport)
    tplate = 'report_block.html'
  createReport( pluginParams['block_report'], tplate, render_context )


def createProgressReport(progessMsg,last=False):
  '''General method to write a message directly to the block report, e.g. when starting prcessing of a new barcode.'''
  createReport( pluginParams['block_report'], "progress_block.html", { "progress_text" : progessMsg, "refresh" : "last" if last else "" } )


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

  # check to see if the barcode was excluded using the frame work barcodes configuration table
  selected = True
  fwbctable = pluginParams['config'].get('barcodetable',None)
  if fwbctable:
    for bc in fwbctable:
      if bc.get('barcode_name',"") == barcode:
        selected = bc.get('selected',True)
        break

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
    "selected" : selected,
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
  barcodeData = barcodeSpecifics(barcode)
  if barcodeData['filtered']:
    return "Filtered (not enough reads)"
  if not barcodeData['selected']:
    return "Excluded from analysis"
  if not barcodeData['reference']:
    return "Analysis requires alignment to a reference"
  if not os.path.exists(barcodeData['bamfile']):
    return "BAM file not found at " + barcodeData['bamfile']
  fileSize = os.stat(barcodeData['bamfile']).st_size
  if fileSize < pluginParams['cmdOptions'].minbamsize:
    return "BAM file too small"
  if fileSize < relBamSize:
    return "BAM file too small relative to largest"
  return checkTarget( barcodeData['bedfile'], barcodeData['bamfile'] )

def checkTarget(bedfile,bamfile):
  '''Checks that the given target is valid for library type, etc. Return error msg or "".'''
  # no bed file is replaced by default bed file in manual mode only
  # - whether having no bedfile is specifically allowed depends on library type
  default_bedfile = pluginParams['config'].get('targetregions','')
  disallow_no_target = not pluginParams['allow_no_target']
  if not bedfile:
    if pluginParams['manual_run']:
      if default_bedfile:
        return "Default target regions overriden by barcode-specific 'None'."
      if disallow_no_target:
        return "No specific or default target regions for barcode."
    elif disallow_no_target:
      return "No default target regions for barcode."
    bedfile = default_bedfile
  ckbb = checkBamBed(bamfile,bedfile)
  if ckbb:
    # signal a more serious error to be reported in the main report against the barcode
    return "ERROR: "+ckbb
  return ""

def checkBamBed(bamfile,bedfile):
  '''Return error message if the provided BED file is not suitable for BAM file or other issues with BAM file.'''
  if not bedfile: return ""
  runcmd = Popen( [os.path.join(pluginParams['plugin_dir'],'scripts','checkBamBed.pl'), bamfile, bedfile], stdout=PIPE, shell=False )
  errMsg = runcmd.communicate()[0]
  errMsg.strip()
  # Hard error will kill run. Soft error (errMsg != "") will just kill for current barcode.
  if runcmd.poll():
    raise Exception("Detected issue with BAM/BED files: %s" % errMsg)
  return errMsg

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

  (cmdOptions, args) = parser.parse_args()
  if( len(args) != 2 ):
    printerr('Usage requires exactly two file arguments (startplugin.json barcodes.json)')
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
      fname = os.path.realpath(os.path.join(root,name))
      if fname.startswith(cwd): continue
      if logopt:
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

  with open(reportName,'w') as bcsum:
    bcsum.write( render_to_string(reportTemplate,safeKeys(reportData)) )

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
  pluginParams['logopt'] = '-l' if pluginParams['cmdOptions'].logopt else ''
  pluginParams['plugin_name'] = jsonParams['runinfo'].get('plugin_name','')
  pluginParams['plugin_dir'] = jsonParams['runinfo'].get('plugin_dir','.')
  pluginParams['run_name'] = jsonParams['expmeta'].get('run_name','')
  pluginParams['analysis_name'] = jsonParams['expmeta'].get('results_name',pluginParams['plugin_name'])
  pluginParams['analysis_dir'] = jsonParams['runinfo'].get('analysis_dir','.')
  pluginParams['results_dir'] = jsonParams['runinfo'].get('results_dir','.')
  pluginParams['report_name'] = pluginParams['plugin_name']+'.html'
  pluginParams['block_report'] = os.path.join(pluginParams['results_dir'],pluginParams['plugin_name']+'_block.html')

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
    pluginParams['results_url'] = os.path.join( jsonParams['runinfo'].get('url_root','.'), resurl[plgpos:] )

  pluginParams['barcoded'] = not NONBARCODED in barcodeInput

  # disable run skip if no report exists => plugin has not been run before
  if not os.path.exists( os.path.join(pluginParams['results_dir'],pluginParams['report_name']) ):
    if pluginParams['cmdOptions'].skip_analysis and not pluginParams['cmdOptions'].cmdline:
      printlog("Warning: Skip analysis option ignorred as previous output appears to be missing.")
      pluginParams['cmdOptions'].skip_analysis = False

  # set up plugin specific options depending on auto-run vs. plan vs. GUI
  config = pluginParams['config'] = jsonParams['pluginconfig'].copy() if 'pluginconfig' in jsonParams else {}
  launchmode = config.get('launch_mode','')
  pluginParams['manual_run'] = launchmode == 'Manual'
  if pluginParams['manual_run']:
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

  # store manual target overrides to dictionary: must be done before call to barcodeSpecifics()
  pluginParams['target_files'] = targetFiles()

  # scan barcodes to check number genome/targets - primarily for log header
  if pluginParams['barcoded']:
    reference = '.'
    genome_id = '.'
    target_id = '.'
    for barcode in barcodeInput:
      if barcode == NOMATCH: continue
      barcodeData = barcodeSpecifics(barcode)
      if not barcodeData['selected']: continue
      if target_id == '.':
        target_id = barcodeData['bedfile']
      elif target_id != barcodeData['bedfile']:
        target_id = "Barcode specific"
      if genome_id == '.':
        genome_id = barcodeData['reference']
        reference = barcodeData['refpath']
      elif genome_id != barcodeData['reference']:
        genome_id = "Barcode specific"
        break
  else:
    barcodeData = barcodeSpecifics(NONBARCODED)
    genome_id = barcodeData['reference']
    target_id = barcodeData['bedfile']
    reference = barcodeData['refpath']
  pluginParams['genome_id'] = genome_id if genome_id else 'None'
  pluginParams['reference'] = reference if reference else 'None'
  if target_id: target_id = fileName(target_id)
  if not pluginParams['manual_run']:
    config['targetregions_id'] = target_id if target_id else 'None'
    config['barcodebeds'] = "Yes" if target_id == "Barcode specific" else "No"

  # preset some (library) dependent flags
  runtype = config['librarytype']
  pluginParams['is_ampliseq'] = (runtype[:4] == 'AMPS' or runtype == 'TARS_16S')
  pluginParams['allow_no_target'] = (runtype == 'GENS' or runtype == 'WGNM' or runtype == 'RNA')

  # check for non-supported de novo runs
  if pluginParams['genome_id'].lower == 'none':
    printerr("Requires a reference sequence for coverage analysis.")
    raise Exception("CATCH:Cannot run plugin without reads aligned to a reference.")

  # check for ampliSeqRNA usage error with bracode-specific references/targets
  if genome_id == "Barcode specific":
    printerr("Barcodes are not all aligned to the same reference.")
    raise Exception("CATCH:This plugin is not suitable for (comparing) barcode reads aligned to different references.")
  if target_id == "Barcode specific":
    printerr("Barcode-specific targets detected.")
    raise Exception("CATCH:This plugin is not suitable for (comparing) barcode reads aligned to different sets of targets.")

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
  ensureFilePrefix()
  try:
    barcodeData = barcodeSpecifics(NONBARCODED)
    sample = barcodeData.get('sample','')
    sampleTag = ' (%s)'%sample if sample else ''
    printlog("\nProcessing nonbarcoded%s...\n" % sampleTag)
    printlog('Reference File: %s' % barcodeData['refpath'])
    bedfile = barcodeData['bedfile']
    if bedfile:
      printlog('Target Regions: %s' % bedfile)
    errmsg = checkBarcode(NONBARCODED)
    if errmsg:
      perr = errmsg.find('ERROR:')+6
      if perr >= 6: errmsg = errmsg[perr:]
      raise Exception(errmsg)
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
    createScraperLinksFolder( pluginParams['results_dir'], pluginParams['prefix'] )

def runForBarcodes():
  # iterate over listed barcodes to pre-test barcode files
  global pluginParams, pluginResult, pluginReport
  barcodes = getOrderedBarcodes()

  # scan for largest BAM file size to set relative minimum
  relBamSize = 0
  if pluginParams['config']['filterbarcodes'] == 'Yes':
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

  # create symlink for js/css - the (empty) tabs on report page will not appear until this exists
  createlink( os.path.join(pluginParams['plugin_dir'],'lifechart'), pluginParams['results_dir'] )

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
        bedfile = barcodeData['bedfile']
        if bedfile:
          if not pluginParams['is_ampliseq']: bedfile.replace('unmerged','merged',1)
          printlog('Target Regions: %s' % bedfile)
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
        printlog("WARNING: No barcode alignment files were found for this barcoded run.")
        createProgressReport("No barcode alignment files were found.")
      elif pluginReport['num_barcodes_processed'] == pluginReport['num_barcodes_failed']:
        printlog("ERROR: Analysis failed for all barcodes.")
        createProgressReport("Analysis failed for all barcodes.")
        return 1
    else:
      runNonBarcoded()
    createProgressReport( "Analysis completed successfully.", True )
    wrapup()
  except Exception, e:
    traceback.print_exc()
    wrapup()  # call only if suitable partial results are available, including some error status
    return 1
  return 0

if __name__ == "__main__":
  exit(plugin_main())

