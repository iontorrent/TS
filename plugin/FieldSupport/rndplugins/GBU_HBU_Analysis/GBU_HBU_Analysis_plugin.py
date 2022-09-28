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
import pandas as pd

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

#
# -------------------- customize code for this plugin here ----------------
#

def addAutorunParams(plan=None):
  '''Additional parameter set up for auto-mated runs, e.g. to add defaults for option only in the GUI.'''
  # get most recent coveageAnalysis plugin output only - GUI also allows for selection of coverageAnalysis_dev etc.
  # Note: Dependent plugin folder may be in startplugin.json if there at autorun ?
  plugbase = os.path.dirname(pluginParams['results_dir'])
  tcapaths = sorted( glob(os.path.join(plugbase,"coverageAnalysis_out.*")), key=os.path.getmtime )
  if not len(tcapaths):
    raise Exception("CATCH: Autorun is dependent on availablity of a completed coverageAnalysis report.")
  config = pluginParams['config']
  config['coverage_analysis_path'] = tcapaths[-1]
  config['coverage_analysis_to_use'] = os.path.basename(config['coverage_analysis_path'])
  config['dup_resolve'] = "Mean"
  furbishPluginParams()


def furbishPluginParams():
  '''Complete/rename/validate user parameters.'''
  # these are for general options that could be enabled in the GUI later
  config = pluginParams['config']
  # original APD_plugin version used amplicon MRD, which is incorrect but can be reproduced setting this value to True
  config['use_amplicon_mrd'] = False
  # allows some user-modified GENE_ID values to be automaticaly corrected for master list look up
  config['autocorrect_geneid'] = True
  # these could be employed later
  config['librarytype'] = "AMPS"
  config['librarytype_id'] = "AmpliSeq DNA"
  config['filterbarcodes'] = "No"
  config['barcodebeds'] = 'Yes'
  if not 'targetregions' in config:
    config['targetregions'] = ""
    config['targetregions_id'] = ""
    config['barcodetargetregions'] = ""
  # from GUI coverage_analysis_path has extra '/' on the end
  if not 'coverage_analysis_to_use' in config:
    config['coverage_analysis_to_use'] = os.path.basename(os.path.dirname(config['coverage_analysis_path']))
  if not 'dup_resolve' in config:
    config['dup_resolve'] = "Mean"


def configReport():
  '''Returns a dictionary based on the plugin config parameters that as reported in results.json.'''
  # This is to avoid outputting hidden or aliased values. If not needed just pass back a copy of config.
  #return pluginParams['config'].copy()
  config = pluginParams['config']
  return {
    "Launch Mode" : config['launch_mode'],
    "Dependent Plugin" : config['coverage_analysis_to_use'],
    "Replicate Resolution" : config['dup_resolve']
  }


def printStartupMessage():
  '''Output the standard start-up message. Customized for additional plugin input and options.'''
  #printlog('')
  #printtime('Started %s' % pluginParams['plugin_name'])
  config = pluginParams['config']
  printlog('Run configuration:')
  printlog('  Plugin version:   %s' % pluginParams['cmdOptions'].version)
  printlog('  Launch mode:      %s' % config['launch_mode'])
  printlog('  Parameters:       %s' % pluginParams['jsonInput'])
  printlog('  Barcodes:         %s' % pluginParams['barcodeInput'])
  printlog('  Dependent plugin: %s' % config['coverage_analysis_to_use'])
  printlog('  Rep resolution:   %s' % config['dup_resolve'])
  printlog('  Output folder:    %s' % pluginParams['results_dir'])
  printlog('  Output file stem: %s' % pluginParams['prefix'])
  printlog('')


def run_plugin(skiprun=False,barcode=""):
  '''Wrapper for making command line calls to perform the specific plugin analyis.'''
  # first part is pretty much boiler plate - grab the key parameters that most plugins use
  logopt = pluginParams['cmdOptions'].logopt
  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['output_dir']
  output_url = pluginParams['output_url']
  #output_prefix = pluginParams['output_prefix']
  output_prefix = barcode
  config = pluginParams['config']
  tcaDir = os.path.join(config['coverage_analysis_path'],barcode)
  mbrd = barcodeData['mean_base_read_depth'] if config['use_amplicon_mrd'] else ""
  fixg = "-a" if config['autocorrect_geneid'] else ""
  wigo = "-d"
  # intersection of panel vs. CDS to mask undigested pimer (on ends) is always appropriate
  # (better completely eliminated by new BBCtools option)
  cdsIntersect = "-i"; # if pluginParams['PhoenixOncologyQC'] else ""
  barcodeData = barcodeSpecifics(barcode)
  sample = barcodeData['sample']
  bedfile = barcodeData['bedfile']
  padding = "5"

  # make panel name part of output prefix for tracking and down-stream resolution of source
  panel = fileName(bedfile)
  if panel[0] >= '0' and panel[0] <= '9':
    panel = 'PP'+panel
  output_prefix += '_'+panel+'.GBU'
  
  # link from source BAM since pipeline uses the name as output file stem
  linkbam = os.path.join(output_dir,output_prefix+".bam")
  createlink(barcodeData['bamfile'],linkbam)
  createlink(barcodeData['bamfile']+'.bai',linkbam+'.bai')
  bamfile = linkbam

  # skip the actual and assume all the data already exists in this file for processing
  if skiprun:
    printlog("Skipped analysis - generating report on in-situ data")
  else:  
    runcmd = '%s -g %s %s %s -D "%s" -F "%s" -M "%s" -N "%s" -P %s "%s" "%s"' % (
      os.path.join(plugin_dir,'run_gbu_calc.sh'), fixg, cdsIntersect, wigo, output_dir,
      output_prefix, mbrd, sample, padding, tcaDir, bedfile )

    if logopt: printlog('\n$ %s\n'%runcmd)
    if( os.system(runcmd) ):
      raise Exception("Failed running run_gbu_calc.sh. Refer to Plugin Log.")

  if pluginParams['cmdOptions'].cmdline: return ({},{})
  printtime("Generating report...")

  # Link report page resources. This is necessary as the plugin code is inaccesible from URLs directly.
  createlink( os.path.join(plugin_dir,'lifechart'), output_dir )
  createlink( os.path.join(plugin_dir,'slickgrid'), output_dir )
  createlink( os.path.join(plugin_dir,'flot'), output_dir )

  # Optional: Delete intermediate files after successful run. These should not be required to regenerate any of the
  # report if the skip-analysis option. Temporary file deletion is also disabled when the --keep_temp option is used.
  deleteTempFiles([ '*.bed' ])

  # Parse out stats from results text file to dict AND convert unacceptible characters to underscores in keys to avoid Django issues
  statsfile = output_prefix+'.stats.txt'
  resultData = parseToDict( os.path.join(output_dir,statsfile), ":" )

  # Collect other output data to pluginReport, which is anything else that is used to generate the report
  reportData = {
    "library_type" : config['librarytype_id'],
    "run_name" : output_prefix,
    "barcode_name" : barcode,
    "output_dir" : output_dir,
    "output_url" : output_url,
    "output_prefix" : output_prefix,
    "help_dict" : helpDictionary(),
    "stats_txt" : checkOutputFileURL(statsfile),
    "unmatched_genes_link" : checkOutputFileURL(output_prefix+'.unmatched_geneids.xls'),
    "amp_gbu_link" : checkOutputFileURL(output_prefix+'.amp.gbu.csv'),
    "amp_pbu_link" : checkOutputFileURL(output_prefix+'.amp.pbu.txt'),
    "cds_gbu_link" : checkOutputFileURL(output_prefix+'.cds.gbu.csv'),
    "cds_pbu_link" : checkOutputFileURL(output_prefix+'.cds.pbu.txt'),
    "cdspad_gbu_link" : checkOutputFileURL(output_prefix+'.cds_pad%s.gbu.csv'%padding),
    "hbu_link1" : checkOutputFileURL(output_prefix+'.gene.hbu.0.1x.csv'),
    "hbu_link2" : checkOutputFileURL(output_prefix+'.gene.hbu.0.2x.csv'),
    "cdspad_gbu_down_link" : checkOutputFileURL(output_prefix+'.cds_pad%s.gbu.down.csv'%padding),
    "hbu_down_link1" : checkOutputFileURL(output_prefix+'.gene.hbu.0.1x.down.csv'),
    "hbu_down_link2" : checkOutputFileURL(output_prefix+'.gene.hbu.0.2x.down.csv'),
    "hbu_cov_link" : checkOutputFileURL(output_prefix+'.gene.hbu.cov.csv'),
    "cdspad_pbu_link" : checkOutputFileURL(output_prefix+'.cds_pad%s.pbu.txt'%padding),
    "bed_link" : re.sub( r'^.*/uploads/BED/(\d+)/.*', r'/rundb/uploadstatus/\1/', bedfile ),
    "bam_link" : checkOutputFileURL(output_prefix+'.bam'),
    "bai_link" : checkOutputFileURL(output_prefix+'.bam.bai')
  }
  return (resultData,reportData)

def checkOutputFileURL(fileURL):
  '''coverageAnalysis helper method to return "" if the provided file URL does not exist'''
  if os.path.exists(os.path.join(pluginParams['output_dir'],fileURL)):
    return fileURL
  return ""

def merge_mgbu_mhbu(gbustats,hbustats1,hbustats2,ghbustats):
    gbu = pd.read_csv(gbustats,sep='\t')
    hbu1 = pd.read_csv(hbustats1,sep='\t')
    hbu2 = pd.read_csv(hbustats2,sep='\t')
    gbu = gbu[['Gene','RackID','AmpCov','wmGBU','wmPGIBU']]
#    hbu1 = hbu1[['Gene','wmGBU','wmPGIBU']]
    hbu2 = hbu2[['Gene','wmGBU','wmPGIBU']]
#    ghbu = pd.merge(gbu,hbu1,how='outer',on='Gene')
#    ghbu = pd.merge(ghbu,hbu2,how='outer',on='Gene')
    ghbu = pd.merge(gbu,hbu2,how='outer',on='Gene')
#    ghbu.columns = ['Gene','RackID','AmpCov','wmGBU','wmPGIBU','wmHBU_0.1x','wmPHIBU_0.1x','wmHBU_0.2x','wmPHIBU_0.2x']
    ghbu.columns = ['Gene','RackID','AmpCov','wmGBU','wmPGIBU','wmHBU','wmPHIBU']
    ghbu.to_csv(ghbustats,sep='\t',header=True,index=False)

def run_meta_plugin():
  '''Create barcode x target reads matrix files and derived files and plots.'''
  if pluginParams['cmdOptions'].cmdline: return
  printlog("")
  printtime("Creating barcodes summary report...")

  # collect barcode statistics from the barcode summary table data and lists of output files
  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['results_dir']
  prefix     = pluginParams['prefix']
  bcreports  = pluginReport['barcodes']
  config     = pluginParams['config']
  bclist = ''
  bctable = []
  GBUreportFiles = []
  HBUreportFiles1 = []
  HBUreportFiles2 = []

  # iterate barcodeSummary[] to maintain barcode processing order
  for bcdata in barcodeSummary:
    bcname = bcdata['barcode_name']
    if bclist: bclist += ','
    bclist += bcname
    bcrep = bcreports[bcname]
    bcline = bcname+"\t"+bcdata['sample']+"\t"+bcdata['panel_name']+"\t"+bcdata['num_genes']+"\t"+bcdata['Number_of_amplicons']+"\t"+bcdata['cds_genes']+"\t"+bcdata['cds_regions']+"\t"+bcdata['pgi_cov']+"\t"+bcdata['amp_mbrd']+"\t"+bcdata['pbu_amp']+"\t"+bcdata['pbu_cds_5']
    bctable.append(bcline)
    gbureportfile = os.path.join( bcrep['output_dir'], bcrep['cdspad_gbu_link'] )
    if os.path.exists(gbureportfile):
      GBUreportFiles.append(gbureportfile)
    hbureportfile1 = os.path.join( bcrep['output_dir'], bcrep['hbu_link1'] )
    if os.path.exists(hbureportfile1):
      HBUreportFiles1.append(hbureportfile1)
    hbureportfile2 = os.path.join( bcrep['output_dir'], bcrep['hbu_link2'] )
    if os.path.exists(hbureportfile2):
      HBUreportFiles2.append(hbureportfile2)

  if len(bctable) > 0:
    bctabfile = prefix+".bc_summary.xls"
    bcline = "Barcode ID\tSample Name\tPanel Name\tGenes\tAmplicons\tMatched Genes\tCDS Regions\tPGI Cov (CDS+5)\tMBRD (Amp)\tPBU(Amp)\tPBU (CDS+5)";
    with open(os.path.join(output_dir,bctabfile),'w') as outfile:
      outfile.write(bcline+'\n')
      for bcline in bctable:
        outfile.write(bcline+'\n')
    barcodeReport.update({"bctable":bctabfile})

  if len(GBUreportFiles):
    printtime("Creating weighted GBU and WIG files...")
    gbustats = prefix+".mgbu.xls"
    hbustats1 = prefix+".mhbu.0.1x.xls"
    hbustats2 = prefix+".mhbu.0.2x.xls"
    ghbustats = prefix+".mghbu.xls"
    geneRack = os.path.join(output_dir,"gene_rack.xls")
    wigZip   = prefix+"_wig"
    wigDir   = os.path.join(output_dir,wigZip)
    os.system("mkdir -p %s"%wigDir)
    if not os.path.exists(geneRack): geneRack = "-"
    wMBRD = "-w" if config.get('weight_gbu_mbrd',"") == "Yes" else ""
    os.system("%s %s -d -C %s -R %s -W %s %s > %s" % ( os.path.join(plugin_dir,"collectGBUstats.pl"),
      wMBRD, wigDir, geneRack, config['dup_resolve'], " ".join(GBUreportFiles), os.path.join(output_dir,gbustats) ))
    os.system("%s %s -d -R %s -W %s %s > %s" % ( os.path.join(plugin_dir,"collectGBUstats.pl"),
      wMBRD, geneRack, config['dup_resolve'], " ".join(HBUreportFiles1), os.path.join(output_dir,hbustats1) ))
    os.system("%s %s -d -R %s -W %s %s > %s" % ( os.path.join(plugin_dir,"collectGBUstats.pl"),
      wMBRD, geneRack, config['dup_resolve'], " ".join(HBUreportFiles2), os.path.join(output_dir,hbustats2) ))  
    # merge mgbu and mhbu files
    merge_mgbu_mhbu(gbustats,hbustats1,hbustats2,ghbustats)
    # create zip for output and delete original
    os.system("cp %s %s/genes.ghbu"%(os.path.join(output_dir,ghbustats),wigDir))
    os.system("zip -j -r %s %s"%(wigDir,wigDir))
    os.system("rm -rf %s"%wigDir)
    barcodeReport.update({"ghbustats":ghbustats, "wigzip":wigZip+".zip"})


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
    panel_name = resultData['Panel Target Regions']
    show_block = "display"
    if pluginParams['PhoenixOncologyQC']:
      if panel_name.startswith('PP'): show_block = ""
      panel_name = "QC Plate "+panel_name
    if errMsg != "":
      detailsLink = "<span class='help' title='%s' style='color:red'>%s</span>" % ( errMsg, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "show_block" : show_block,
        "sample" : sample,
        "panel_name": panel_name,
        "num_genes": "0" if errZero else "NA",
        "Number_of_amplicons": "0" if errZero else "NA",
        #"amps_mult_genes": "0" if errZero else "NA",
        "cds_genes": "0" if errZero else "NA",
        "cds_regions": "0" if errZero else "NA",
        "pgi_cov": "0.00%" if errZero else "NA",
        "amp_mbrd": "0" if errZero else "NA",
        "pbu_amp": "0.00%" if errZero else "NA",
        #"pbu_cds": "0.00%" if errZero else "NA",
        #"pgipbu_cds_5": "0.00%" if errZero else "NA",
        "pbu_cds_5": "0.00%" if errZero else "NA",
        "pbu_hotspot": "0.00%" if errZero else "NA"
      })
    else:
      detailsLink = "<a target='_parent' href='%s' class='help'><span title='Click to view the detailed report for barcode %s'>%s</span><a>" % (
        os.path.join(barcode,pluginParams['report_name']), barcode, barcode )
      barcodeSummary.append({
        "index" : len(barcodeSummary),
        "barcode_name" : barcode,
        "barcode_details" : detailsLink,
        "show_block" : show_block,
        "sample" : sample,
        "panel_name": panel_name,
        "num_genes": resultData['Number of genes'],
        "Number_of_amplicons": resultData['Number of amplicons'],
        #"amps_mult_genes": resultData['Number of amplicons in multiple genes'],
        "cds_genes": resultData['Number of CDS genes matched'],
        "cds_regions": resultData['Number of CDS regions covered'],
        "pgi_cov": resultData['Panel-gene-intersection (CDS+5)'],
        "amp_mbrd": resultData['Panel Mean Base Read Depth'],
        "pbu_amp": resultData['Panel Base Uniformity (Design)'],
        #"pbu_cds": resultData['Panel Base Uniformity (CDS)'],
        #"pgipbu_cds_5": resultData['PGI Panel Base Uniformity (CDS+5)'],
        "pbu_cds_5": resultData['Panel Base Uniformity (CDS+5)'],
        "pbu_hotspot": resultData['PGI Panel Base Uniformity (Hotspot)']    # For hotspots, PGI uniformity is the same as panel uniformity
      })
  render_context = {
    "autorefresh" : autoRefresh,
    "barcode_results" : simplejson.dumps(barcodeSummary)
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
  isqc = ('PhoenixOncologyQC' in pluginParams and pluginParams['PhoenixOncologyQC'])
  return {
    "run_name" : pluginParams['prefix'],
    "coverage_analysis_source" : config['coverage_analysis_to_use'],
    "library_type" : config['librarytype_id'],
    "target_regions" : "Phoenix Oncology QC barcode-specific" if isqc else targets,
    "dup_resolve" : config['dup_resolve'] if isqc else "",
    "help_dict" : helpDictionary()
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
  target_files = pluginParams['target_files']
  if pluginParams['manual_run'] and target_files:
    if barcode in target_files:
      bedfile = target_files[barcode]
    else:
      bedfile = pluginParams['config']['targetregions']
  else:
    bedfile = barcodeData['target_region_filepath']
  return {
    "mean_base_read_depth" : barcodeData.get('mean_base_read_depth',"0"),
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
  if not barcodeData['reference']:
    return "Analysis requires alignment to a reference"
  if not os.path.exists(barcodeData['bamfile']):
    return "BAM file not found at " + barcodeData['bamfile']
  fileSize = os.stat(barcodeData['bamfile']).st_size
  if fileSize < pluginParams['cmdOptions'].minbamsize:
    return "BAM file too small"
  if fileSize < relBamSize:
    return "BAM file too small relative to largest"
  config = pluginParams['config']
  if not os.path.exists( os.path.join(config['coverage_analysis_path'],barcode,"tca_auxiliary.bbc") ):
    return "No base coverage (BBC) file available for analysis"
  #return checkTarget( barcodeData['bedfile'], barcodeData['bamfile'] )
  return "" if barcodeData.get('bedfile','') else "No specific or default target regions for barcode."

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
  parser.add_option('-a', '--autorun', help='Simulate an autorun for a manual launch.', action="store_true", dest='autorun')
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
  if pluginParams['cmdOptions'].autorun and launchmode == 'Manual':
    launchmode = ''
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
  pluginParams['target_files'] = {}; #targetFiles()

  # reconfigure barcodes to dependent plugin run
  importBarcodeConfiguration( config['coverage_analysis_path'] )

  # scan barcodes to check number genome/targets - primarily for log header
  if pluginParams['barcoded']:
    reference = '.'
    genome_id = '.'
    target_id = '.'
    for barcode in barcodeInput:
      if barcode == NOMATCH: continue
      barcodeData = barcodeSpecifics(barcode)
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

  # plugin configuration becomes basis of results.json file
  global pluginResult, pluginReport
  pluginResult = configReport()
  if pluginParams['barcoded']:
    pluginResult['barcodes'] = {}
    pluginReport['barcodes'] = {}

def importBarcodeConfiguration(tcaReportPath):
  '''Replaces barcode specific targets with those from an ancester plugin, possibly further filtered by current barcode set.'''
  global barcodeInput, pluginParams
  # moved start header here as this prerun set up can take significant time
  printlog('')
  printtime('Started %s' % pluginParams['plugin_name'])
  try:
    printtime("Checking coverageAnalysis report as Panel or QC Plate run...")
    if not os.path.exists(tcaReportPath):
      raise Exception("CATCH:Plugin requires an existing coverageAnalysis output folder.")
    output_dir = pluginParams['results_dir']
    # phoenix_qc_plate_full_racks.pl does not specifiy output folder so assumes CWD is plugin output_dir
    runcmd = '%s "%s" ' % ( os.path.join(pluginParams['plugin_dir'],'phoenix_qc_plate_full_racks.pl'), tcaReportPath )
    if( os.system(runcmd) ):
      raise Exception("Failed running phoenix_qc_plate_full_racks.pl.")
    bc2rackFile = os.path.join(output_dir,"barcode_panel.tsv")
    bc2rack = 0
    if os.path.exists(bc2rackFile):
      with open(bc2rackFile) as fin:
        bc2rack = dict(line.split() for line in fin)
    if bc2rack:
      printlog("Barcodes using QC panel plates identified.")
      pluginParams['PhoenixOncologyQC'] = True
      os.system("mkdir local_beds; mv *.bed local_beds")
    else:
      printlog("- No barcode to QC racks identified; standard panels assumed.")
      pluginParams['PhoenixOncologyQC'] = False
    
    printtime("Checking coverageAnalysis report for analyzed barcodes...")

    # use the pluginconfig to build a list bed file sources - unfortunately results.json only has file names
    bedPaths = {}
    src_def_targets = False
    with open(os.path.join(tcaReportPath,"startplugin.json")) as f:
      src_config = json.load(f)['pluginconfig']
      launch_mode = src_config.get('launch_mode','')
      src_def_targets = src_config.get('targetregions','')
      if src_def_targets:
        bedPaths[src_config['targetregions_id']] = src_def_targets
      src_targets = src_config.get('barcodetargetregions','').split(';')
      for kev in src_targets:
        if not kev: continue
        kvp = kev.split('=')
        bedname = os.path.basename(kvp[1])
        bedPaths[bedname[:-4]] = kvp[1]

    # use ancester results.json for barcodes processed to results
    # - edits barcodeInput assuming it has been loaded for this plugin config
    with open(os.path.join(tcaReportPath,"results.json")) as f:
      fjson = json.load(f)
      if fjson['Library Type'] != "AmpliSeq DNA":
        printlog("\nWARNING: Library Type (report) for %s may be unsuitable for GBU/PBU evaluation."%fjson['Library Type'])
      src_barcodes = fjson['barcodes']
      for barcode in barcodeInput:
        # additonal barcode-specific data extracted from results.json
        if 'Average base coverage depth' in src_barcodes[barcode]:
          barcodeInput[barcode]['mean_base_read_depth'] = src_barcodes[barcode]['Average base coverage depth']
        else:
          barcodeInput[barcode]['mean_base_read_depth'] = 'NA'
        # if no explicit TCA config (autorun w/o config) keep bed file source from barcodes.json
        bed = barcodeInput[barcode]['target_region_filepath']
        if src_def_targets:
          if barcode in src_barcodes:
            bed = src_barcodes[barcode].get('Target Regions',"")
            if bed: bed = bedPaths.get(bed,"")
          elif launch_mode:
            # if manual or plan config exists then assume barcode BED explicitly set to None
            bed = ""
        # filter used barcodes to those for full racks for Phoenix Oncology QC runs
        # - if no explicit plan config then assume QC rack always overrides configuration
        if bc2rack and (not bedPaths or bed):
          bed = bc2rack.get(barcode,"")
          if bed: bed = os.path.join(output_dir,"local_beds",bed+".bed")
        # overwrite barcode bed or set to "" if not analyzed
        barcodeInput[barcode]['target_region_filepath'] = bed
  except Exception, e:
    printerr('WARNING: Failed to parse previous barcode-specific target bed files.')
    #pluginResult.update({ 'Error': str(e) })
    #return False
    raise
  printlog("")
  return True

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
      if pluginParams['PhoenixOncologyQC']:
        printlog('Target Regions: Amplicons dispensed to QC plate %s' % fileName(bedfile))
      else:
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
  #createlink( os.path.join(pluginParams['plugin_dir'],'lifechart'), pluginParams['results_dir'] )

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
          if pluginParams['PhoenixOncologyQC']:
            printlog('Target Regions: Amplicons dispensed to QC plate %s' % fileName(bedfile))
          else:
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
  # delete BED files extracted for QC reports
  if pluginParams['PhoenixOncologyQC']:
    os.system("rm -rf '%s'"%os.path.join(pluginParams['results_dir'],"local_beds"))

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

