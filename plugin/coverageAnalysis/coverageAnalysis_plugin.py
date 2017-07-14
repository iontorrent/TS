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
barcodeInput = {}
pluginParams = {}
pluginResult = {}
pluginReport = {}
barcodeSummary = []
barcodeReport = {}
help_dictionary = {}

# Specifies whether to assume default reference or disallow barcode-specific no referecne
barcode_no_reference_as_default = True

# max plugin out file ext length for addressing max filename length (w/ barcode), e.g. ".amplicon.cov.xls"
max_fileext_len = 17
max_filename_len = 255

#
# -------------------- customize code for this plugin here ----------------
#

def addAutorunParams(plan=None):
  '''Additional parameter set up for autoruns to fill in for defaults usually set customization GUI.'''
  config = pluginParams['config']
  config['librarytype'] = 'WGNM'
  config['librarytype_id'] = 'Whole Genome'
  config['targetregions'] = ''
  config['targetregions_id'] = 'None'
  config['trimreads'] = 'No'
  if not 'sampleid' in config: config['sampleid'] = 'No'
  if not 'padtargets' in config: config['padtargets'] = '0'
  if not 'uniquemaps' in config: config['uniquemaps'] = 'No'
  if not 'nonduplicates' in config: config['nonduplicates'] = 'Yes'
  config['minalignlen'] = '0'
  config['minmapqual'] = '0'
  config['barcodebeds'] = 'No'
  config['barcodetargetregions'] = ''
  # extract things from the plan if provided - for coverageAnalysis auto-run w/o a plan leads to early exit
  if plan: 
    config['librarytype'] = runtype = plan['runType']
    if runtype == 'AMPS':
      config['librarytype_id'] = 'AmpliSeq DNA'
    elif runtype == 'AMPS_EXOME':
      config['librarytype_id'] = 'AmpliSeq Exome'
    elif runtype == 'AMPS_RNA':
      config['librarytype_id'] = 'AmpliSeq RNA'
    elif runtype == 'AMPS_DNA_RNA':
      config['librarytype_id'] = 'AmpliSeq DNA+RNA'
    elif runtype == 'TARS':
      config['librarytype_id'] = 'TargetSeq'
    elif runtype == 'TAG_SEQUENCING':
      config['librarytype_id'] = 'Tag Sequencing'
    elif runtype == 'TARS_16S':
      config['librarytype_id'] = '16S Targeted Sequencing'
      config['sampleid'] = 'No'
      config['uniquemaps'] = 'No'
    elif runtype == 'RNA':
      config['librarytype_id'] = 'RNA-Seq'
      config['sampleid'] = 'No'
      config['uniquemaps'] = 'No'
    elif runtype == 'WGNM':
      config['librarytype_id'] = 'Whole Genome'
    elif runtype == 'GENS':
      config['librarytype_id'] = 'Generic Sequencing'
    else:
      config['librarytype_id'] = "[%s]"%runtype
      raise Exception("CATCH:Do not know how to analyze coverage for unsupported plan runType: '%s'"%runtype)
    if runtype[:4] == 'AMPS' or runtype == 'TARS_16S' or runtype == 'RNA' or runtype == 'TAG_SEQUENCING':
      config['nonduplicates'] = 'No'
      config['padtargets'] = '0'
    else:
      config['sampleid'] = 'No'
  else:
    raise Exception("CATCH:Automated analysis requires a Plan to specify Run Type.")    


def furbishPluginParams():
  '''Additional parameter set up for configured and semi-configured runs.'''
  config = pluginParams['config']
  # HTML form posts typically do not add unchecked options...
  if 'barcodebeds' not in config: config['barcodebeds'] = 'No'
  if 'sampleid' not in config: config['sampleid'] = 'No'
  if 'uniquemaps' not in config: config['uniquemaps'] = 'No'
  if 'nonduplicates' not in config: config['nonduplicates'] = 'No'
  if 'minalignlen' not in config: config['minalignlen'] = '0'
  if 'minmapqual' not in config: config['minmapqual'] = '0'


def configReport():
  '''Returns a dictionary based on the plugin config parameters that are reported in results.json.'''
  config = pluginParams['config']
  return {
    "Launch Mode" : config['launch_mode'],
    "Library Type" : config['librarytype_id'],
    "Reference Genome" : pluginParams['genome_id'],
    "Targeted Regions" : config['targetregions_id'],
    "Sample Tracking" : config['sampleid'],
    "Target Padding" : config['padtargets'],
    "Use Only Uniquely Mapped Reads" : config['uniquemaps'],
    "Use Only Non-duplicate Reads" : config['nonduplicates'],
    "Min Aligned Length" : config['minalignlen'],
    "Min Mapping Quality" : config['minmapqual'],
    "barcoded" : "true" if pluginParams['barcoded'] else "false"
  }


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
  printlog('  Library Type:     %s' % config['librarytype_id'])
  printlog('  Reference Name:   %s' % pluginParams['genome_id'])
  printlog('  Target Regions:   %s' % config['targetregions_id'])
  # echo out manually specified targets to facilitate copy/paste for re-run
  if config['barcodebeds'] == 'Yes' and pluginParams['manual_run']:
    target_files = pluginParams['target_files']
    for bctrg in sorted(target_files):
      trg = fileName(target_files[bctrg])
      if trg == "": trg = "None"
      printlog('    %s   %s' % (bctrg,trg))
  printlog('  Target Padding:   %s' % config['padtargets'])
  printlog('  Sample Tracking:  %s' % config['sampleid'])
  printlog('  Uniquely Mapped:  %s' % config['uniquemaps'])
  printlog('  Non-duplicate:    %s' % config['nonduplicates'])
  printlog('  Min Align Length: %s' % config['minalignlen'])
  printlog('  Min Map Quality:  %s' % config['minmapqual'])
  printlog('')


def run_plugin(skiprun=False,barcode=""):
  '''Wrapper for making command line calls to perform the specific plugin analyis.'''
  # grab the key input parameters
  logopt = pluginParams['cmdOptions'].logopt
  plugin_dir = pluginParams['plugin_dir']
  output_dir = pluginParams['output_dir']
  output_url = pluginParams['output_url']
  output_prefix = pluginParams['output_prefix']
  config = pluginParams['config']
  librarytype = config['librarytype']
  barcodeData = barcodeSpecifics(barcode)

  # use nucType and targets to distinguish barcode-specific run-types
  bedfile = barcodeData['bedfile']
  if librarytype == 'AMPS_DNA_RNA':
    librarytype = 'AMPS_RNA' if barcodeData['nuctype'] == 'RNA' else 'AMPS'
  elif librarytype == 'RNA' and bedfile != '':
    librarytype = 'AMPS_RNA'

  # link from source BAM since pipeline uses the name as output file stem
  linkbam = os.path.join(output_dir,output_prefix+".bam")
  createlink(barcodeData['bamfile'],linkbam)
  createlink(barcodeData['bamfile']+'.bai',linkbam+'.bai')
  bamfile = linkbam

  # Run-type flags used to customize the detailed (barcode) report
  samp_track = (config['sampleid'] == 'Yes')
  trg_stats = (bedfile != "")
  amp_stats = (trg_stats and pluginParams['is_ampliseq'])
  rna_stats = (librarytype == 'AMPS_RNA' or librarytype == 'RNA')
  chr_stats = (librarytype == 'RNA' and not trg_stats)
  wgn_stats = ((librarytype == 'WGNM' or librarytype == 'GENS') and not trg_stats)
  bas_stats = ((wgn_stats or trg_stats) and not rna_stats)
  trg_type = 1 if amp_stats else 0
  if rna_stats: trg_type = 2
  if chr_stats: trg_type = 3
  if wgn_stats: trg_type = 4

  # Check sample tracking is appropriate to reference and set path to sample ID target BED file
  sampleidBed = ''
  if samp_track:
    if not librarytype.startswith('AMPS'):
      printlog("WARNING: Sample Tracking option ignored. This is only available for AmpliSeq Libraries.")
      samp_track = False
    elif pluginParams['genome_id'] != "hg19":
      printlog("WARNING: Sample Tracking option ignored. This is only available for reads aligned to the hg19 reference.")
      samp_track = False
    else:
      sampleidBed = os.path.join(os.path.dirname(plugin_dir),'sampleID','targets','KIDDAME_sampleID_regions.bed')

  # skip the actual and assume all the data already exists in this file for processing
  if skiprun:
    printlog("Skipped analysis - generating report on in-situ data")
  else:
    # Pre-run modification of BED files is done here to avoid redundancy of repeating for target assigned barcodes
    # Note: The stand-alone command can perform the (required) annotation but does not handle padding
    sample = barcodeData['sample']
    if not sample: sample = 'None'
    reference = barcodeData['refpath']
    genomeUrl = barcodeData['refurl']
    (mergeBed,annoBed) = modifyBedFiles( ('contigs' if wgn_stats or chr_stats else bedfile), reference )
    ampopt = '-t'
    if amp_stats: ampopt = '-a'
    if rna_stats: ampopt = '-r'
    if chr_stats: ampopt = '-c -r'
    if wgn_stats: ampopt = '-c -t'
    filtopts = '-u' if config['uniquemaps'] == 'Yes' else ''
    if config['nonduplicates'] == 'Yes': filtopts += ' -d'
    rptopt = '-R coverageAnalysis.html' if pluginParams['cmdOptions'].cmdline else ''
    printtime("Running coverage analysis pipeline...")
    runcmd = '%s %s %s %s -D "%s" -A "%s" -B "%s" -C "%s" -L "%s" -M %s -N "%s" -P %s -Q %s -S "%s" %s "%s" "%s"' % (
        os.path.join(plugin_dir,'run_coverage_analysis.sh'), pluginParams['logopt'], ampopt,
        filtopts, output_dir, annoBed, mergeBed, bedfile, genomeUrl, config['minalignlen'],
        sample, config['padtargets'], config['minmapqual'], sampleidBed, rptopt, reference, bamfile )
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
  if bedfile and annoBed: createlink( annoBed, output_dir )
   
  # Optional: Delete intermediate files after successful run. These should not be required to regenerate any of the
  # report if the skip-analysis option. Temporary file deletion is also disabled when the --keep_temp option is used.
  #deleteTempFiles([ '*.bam', '*.bam.bai', '*.tmp' ])

  # Create an annotated list of files as used to create the file links table.
  # - Could be handled in the HTML template directly but external code is re-used to match cmd-line reports.

  # Parse out stats from results text file to dict AND convert unacceptible characters to underscores in keys to avoid Django issues
  statsfile = output_prefix+'.stats.cov.txt'
  resultData = parseToDict( os.path.join(output_dir,statsfile), ":" )

  # Collect other output data to pluginReport, which is anything else that is used to generate the report
  trgtype = '.target.cov'
  if amp_stats or librarytype == 'RNA': trgtype = '.amplicon.cov'
  if chr_stats: trgtype = '.contig.cov'
  reportData = {
    "library_type" : config['librarytype_id'],
    "libnuc_type" : barcodeData['nuctype'],
    "run_name" : output_prefix,
    "barcode_name" : barcode,
    "samp_track" : samp_track,
    "output_dir" : output_dir,
    "output_url" : output_url,
    "output_prefix" : output_prefix,
    "amp_stats" : amp_stats,
    "rna_stats" : rna_stats,
    "trg_stats" : trg_stats,
    "chr_stats" : chr_stats,
    "wgn_stats" : wgn_stats,
    "bas_stats" : bas_stats,
    "trg_type" : trg_type,
    "help_dict" : helpDictionary(),
    "stats_txt" : checkOutputFileURL(statsfile),
    "overview_png" : checkOutputFileURL(output_prefix+'.covoverview.png'),
    "rep_overview_png" : checkOutputFileURL(output_prefix+'.repoverview.png'),
    "scat_gc_png" :  checkOutputFileURL(output_prefix+'.gc.png'),
    "scat_len_png" : checkOutputFileURL(output_prefix+'.ln.png'),
    "rep_gc_png" :   checkOutputFileURL(output_prefix+'.gc_rep.png'),
    "rep_len_png" :  checkOutputFileURL(output_prefix+'.ln_rep.png'),
    "rep_pool_png" : checkOutputFileURL(output_prefix+'.pool.png'),
    "finecov_tsv" :  checkOutputFileURL(output_prefix+trgtype+'.xls'),
    "overview_tsv" : checkOutputFileURL(output_prefix+'.covoverview.xls'),
    "base_cov_tsv" : checkOutputFileURL(output_prefix+'.base.cov.xls'),
    "chr_cov_tsv" :  checkOutputFileURL(output_prefix+'.chr.cov.xls'),
    "wgn_cov_tsv" :  checkOutputFileURL(output_prefix+'.wgn.cov.xls'),
    "bed_link" : re.sub( r'^.*/uploads/BED/(\d+)/.*', r'/rundb/uploadstatus/\1/', bedfile ),
    "bed_anno" : checkOutputFileURL(os.path.basename(annoBed)),
    "aux_bbc" : checkOutputFileURL('tca_auxiliary.bbc'),
    "aux_ttc" : checkOutputFileURL('tca_auxiliary.ttc.xls'),
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
  '''coverageAnalysis method to merged padded and GC annotated BED files, creating them if they do not already exist.'''
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


def run_meta_plugin():
  '''Create barcode x target reads matrix and text version of barcode summary table (after analysis completes for individual barcodes).'''
  if pluginParams['cmdOptions'].cmdline: return
  printlog("")
  printtime("Collating barcodes summary data...")
  # get correct file/type for reads matrix
  fileext2 = ""
  renderOpts = renderOptions()
  if renderOpts['runType'] == "RNA":
    fieldid = '9'
    typestr = 'contig'
    fileext = '.contig.cov.xls'
    if renderOpts['trg_stats']: fileext2 = '.amplicon.cov.xls'
  elif renderOpts['runType'] == "WGNM":
    fieldid = 'chrom'
    typestr = 'chromosome'
    fileext = '.chr.cov.xls'
  else:
    # includes 'GENS' runType
    fieldid = '9'
    typestr = 'amplicon' if pluginParams['is_ampliseq'] or renderOpts['runType'] == "RNA" else 'target'
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
    if 'output_dir' not in bcrep: continue
    bcline = bcname+"\t"+bcdata['sample']+"\t"+bcdata['mapped_reads']
    if renderOpts['trg_stats']: bcline += "\t"+bcdata['on_target']
    if renderOpts['samp_track']: bcline += "\t"+bcdata['sample_target']
    if renderOpts['bas_stats']: bcline += "\t"+bcdata['mean_depth']+"\t"+bcdata['uniformity']
    rfile = barcodeInput[bcname]['reference_fullpath'] + '.fai'
    if not rfile in refs: refs.append(rfile)
    bctable.append(bcline)
    reportfile = os.path.join(bcrep['output_dir'],bcrep['output_prefix']+fileext)
    reportfile2 = os.path.join(bcrep['output_dir'],bcrep['output_prefix']+fileext2)
    if os.path.exists(reportfile):
      reportFiles.append(reportfile)
    elif fileext2 and os.path.exists(reportfile2):
      reportFiles.append(reportfile2)

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
    reference = ",".join(refs)
    bcmatrix = pluginParams['prefix']+".bcmatrix.xls"
    with open(os.path.join(pluginParams['results_dir'],bcmatrix),'w') as outfile:
      runcmd = Popen( [os.path.join(pluginParams['plugin_dir'],'scripts','barcodeMatrix.pl'),
        reference, fieldid] + reportFiles, stdout=outfile )
      runcmd.communicate()
      if runcmd.poll():
        raise Exception("Failed to create barcode x %s reads matrix."%typestr)
      fieldstr = "mean base read depth for" if typestr == 'target' else "reads assigned to"
      barcodeReport.update({ "bcmatrix":bcmatrix, "bcmtype":typestr, "bcmfield":fieldstr })

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
    errZero = False
    if 'Error' in resultData:
      errMsg = resultData['Error']
      errZero = "no mapped" in errMsg or "no read" in errMsg
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
        "mapped_reads": "0" if errZero else "NA",
        "on_target": "0.00%" if errZero else "NA",
        "sample_target": "0.00%" if errZero else "NA",
        "mean_depth": "0.00" if errZero else "NA",
        "uniformity": "100.00%" if errZero else "NA"
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
  '''Support method to generate list of rendering options for (barcode) summary reports.'''
  config = pluginParams['config']
  librarytype = config['librarytype']
  filter_options = []
  ampliseq = pluginParams.get('is_ampliseq')
  if ampliseq and config['sampleid'] == 'Yes': filter_options.append('Sample tracking')
  if not librarytype == 'AMPS_RNA' and config['uniquemaps'] == 'Yes': filter_options.append('Uniquely mapped')
  if not ampliseq and config['nonduplicates'] == 'Yes': filter_options.append('Non-duplicate')
  if config['minalignlen'] and int(config['minalignlen']):
    filter_options.append('Minimum aligned length = %d'%int(config['minalignlen']))
  if config['minmapqual'] and int(config['minmapqual']):
    filter_options.append('Minimum mapping quality = %d'%int(config['minmapqual']))
  trg_stats = config['targetregions_id'] != 'None'
  return {
    "runType" : librarytype,
    "library_type" : config['librarytype_id'],
    "target_regions" : config['targetregions_id'],
    "target_padding" : config['padtargets'],
    "filter_options" : ', '.join(filter_options),
    "samp_track" : (config['sampleid'] == 'Yes'),
    "mixed_stats" : (librarytype == "AMPS_DNA_RNA"),
    "chr_stats" : (librarytype == "RNA" and not trg_stats),
    "wgn_stats" : (librarytype == 'WGNM' or librarytype == 'GENS'),
    "trg_stats" : trg_stats,
    "bas_stats" : (librarytype != 'AMPS_RNA' and librarytype != 'RNA')
  }
  

def createIncompleteReport(errorMsg=""):
  '''Called to create an incomplete or error report page for non-barcoded runs.'''
  if not pluginParams['barcoded']:
    barcodeData = barcodeSpecifics(NONBARCODED)
    if barcodeData: sample = barcodeData.get('sample','None')
    if sample == 'none': sample = 'None'
  else:
    sample = 'None'
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
    '-R', '0', html_report, os.path.join(output_dir,pdf_report)], shell=False, stdout=PIPE, stderr=PIPE )
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
    render_context.update(renderOptions())
    tplate = 'barcode_block.html'
  else:
    render_context = pluginResult.copy()
    render_context.update(pluginReport)
    tplate = 'report_block.html'
  createReport( pluginParams['block_report'], tplate, render_context )


def createProgressReport(progessMsg,last=False):
  '''General method to write a message directly to the block report, e.g. when starting prcessing of a new barcode.'''
  createReport( pluginParams['block_report'], "progress_block.html", { "progress_text" : progessMsg, "refresh" : "last" if last else "" } )


def fatalErrorReport(msg):
    printerr(msg)
    pluginResult.update({ 'Error': msg })
    createIncompleteReport(msg)
    createProgressReport(msg,True)


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
    exit(1)
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
    "bamfile" : barcodeData.get('bam_filepath','') }

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

  # scan barcodes to check number genome/targets - mainly for log header
  if pluginParams['barcoded']:
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
      elif genome_id != barcodeData['reference']:
        genome_id = "Barcode specific"
        break
  else:
    barcodeData = barcodeSpecifics(NONBARCODED)
    genome_id = barcodeData['reference']
    target_id = barcodeData['bedfile']

  pluginParams['genome_id'] = genome_id if genome_id else 'None'
  target_id = os.path.basename(target_id)
  if target_id[-4:] == ".bed": target_id = target_id[:-4]
  config['targetregions_id'] = target_id if target_id else 'None'
  config['barcodebeds'] = "Yes" if target_id == "Barcode specific" else "No"
 
  # preset some (library) dependent flags
  runtype = config['librarytype']
  pluginParams['is_ampliseq'] = (runtype[:4] == 'AMPS' or runtype == 'TARS_16S' or runtype == 'TAG_SEQUENCING')
  pluginParams['allow_no_target'] = (runtype == 'GENS' or runtype == 'WGNM' or runtype == 'RNA')

  # early catch for unsuitable setup - to approximate 5.0 behavior
  if pluginParams['genome_id'].lower == 'none':
    raise Exception("CATCH: Cannot run plugin without reads aligned to any reference.")

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
  '''Run the plugin script once for non-barcoded reports.'''
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
      if not pluginParams['is_ampliseq']: bedfile.replace('unmerged','merged',1)
      printlog('Target Regions: %s' % bedfile)
    ensureFilePrefix()
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
    fatalErrorReport(str(e))
    raise
  if pluginParams['cmdOptions'].scraper:
    createScraperLinksFolder( pluginParams['results_dir'], pluginParams['prefix'] )

def runForBarcodes():
  # iterate over listed barcodes to pre-test barcode files
  global pluginParams, pluginResult, pluginReport
  barcodes = getOrderedBarcodes()
  numGoodBams = 0
  numInvalidBarcodes = 0
  numInvalidBAMs = 0
  maxBarcodeLen = 0
  barcodeIssues = []
  for barcode in barcodes:
    errmsg = checkBarcode(barcode)
    if errmsg:
      # distinguish errors due to targets vs. BAM
      numInvalidBarcodes += 1
      if errmsg[:6] == "ERROR:":
        errmsg = "\n"+errmsg
        numInvalidBAMs += 1
    else:
      if( len(barcode) > maxBarcodeLen ):
        maxBarcodeLen = len(barcode)
      numGoodBams += 1
    barcodeIssues.append(errmsg)

  ensureFilePrefix(maxBarcodeLen+1)
  pluginReport['num_barcodes_processed'] = numGoodBams
  pluginReport['num_barcodes_invalid'] = numInvalidBarcodes
  pluginReport['num_barcodes_badbams'] = numInvalidBAMs
  pluginReport['num_barcodes_failed'] = 0

  skip_analysis = pluginParams['cmdOptions'].skip_analysis
  stop_on_error = pluginParams['cmdOptions'].stop_on_error
  create_scraper = pluginParams['cmdOptions'].scraper

  # create initial (empty) barcodes summary report
  printlog("Processing %d barcodes..." % numGoodBams)
  updateBarcodeSummaryReport("",True)

  # iterate over all barcodes and process the valid ones
  postout = True; # just for logfile prettiness
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

def checkBarcode(barcode):
  '''Checks if a specific barcode is set up correctly for analysis.'''
  barcodeData = barcodeSpecifics(barcode) 
  if barcodeData['filtered']:
    return "ERROR: Filtered (not enough reads)"
  if not barcodeData['reference']:
    return "ERROR: Analysis requires alignment to a reference"
  if not os.path.exists(barcodeData['bamfile']):
    return "ERROR: BAM file not found at " + barcodeData['bamfile']
  if os.stat(barcodeData['bamfile']).st_size < pluginParams['cmdOptions'].minbamsize:
    return "ERROR: BAM file too small"
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
  #if runcmd.poll():
  #  raise Exception("Detected issue with BAM/BED files: %s" % errMsg)
  return errMsg

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
    runcmd = Popen( [os.path.join(pluginParams['plugin_dir'],'scripts','addMeanBarcodeStats.py'),
      jsonfile, "Percent end-to-end reads"], stdout=PIPE, shell=False )
    runcmd.communicate()

def plugin_main():
  '''Main entry point for script. Returns unix-like 0/1 for success/failure.'''
  try:
    loadPluginParams()
    printStartupMessage()
  except Exception, e:
    printerr("Failed to set up run parameters.")
    emsg = str(e)
    if emsg[:6] == 'CATCH:':
      fatalErrorReport(emsg[6:])
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
        if pluginReport['num_barcodes_badbams'] == 0:
          fatalErrorReport("All barcodes had invalid target regions specified.")
        elif pluginReport['num_barcodes_invalid'] == pluginReport['num_barcodes_badbams']:
          fatalErrorReport("No valid barcode alignment files were found for this barcoded run.")
        else:
          fatalErrorReport("All barcodes had invalid alignment files vs. target regions specified.")
        return 1
      elif pluginReport['num_barcodes_processed'] == pluginReport['num_barcodes_failed']:
        fatalErrorReport("Analysis failed for all barcode alignments.")
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

