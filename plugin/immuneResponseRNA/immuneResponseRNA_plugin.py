#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

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
import shutil

# set up for Django rendering with TSS installed apps
from django.conf import settings
from django.template.loader import render_to_string
from django.conf import global_settings

global_settings.LOGGING_CONFIG = None

# max plugin out file ext length for addressing max filename length (w/ barcode), e.g. ".amplicon.cov.xls"
max_fileext_len = 17
max_filename_len = 255

# flag to create diferential expression matrix and significance ratio threshold for report summary
create_DE_matrix = False
difexp_thresh = 2

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
# ratio for minimal to maximum barcode BAM size for barcode to be ignored. 0 = disabled
barcode_filter = 0.0

# global data collecters common to functions
jsonBarcode = {}
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
    # Note: this function may be redundant in later TSS versions, where pluginconfig already consolidates input from all sources
    config = pluginParams['config']
    # defaults for auto-run to match config settings from GUI
    config['librarytype'] = 'ampiseq-rna'
    config['librarytype_id'] = 'Ion AmpliSeq RNA'
    config['target_regions'] = ''
    config['target_regions_name'] = 'None'
    config['user_target_regions'] = ''
    config['user_target_regions_name'] = 'None'
    config['barcodebeds'] = 'No'
    config['barcodetargetregions'] = ''
    config['time_of_analysis'] = time.strftime("%Y-%m-%d %H:%M")
    # GUI-only options that might be set by plan.html
    if config.get('filterbarcodes', '') == '': config['filterbarcodes'] = 'Yes'
    if config.get('ercc', '') == '': config['ercc'] = 'No'
    if config.get('uniquemaps', '') == '': config['uniquemaps'] = 'No'
    if config.get('user_target_regions', '') == '': config['user_target_regions'] = 'None'
    # extract things from the plan if provided - for coverageAnalysis auto-run w/o a plan leads to early exit
    if plan:
        runtype = plan['runType']
        if runtype != 'AMPS_RNA':
            config['librarytype_id'] = "[%s]" % runtype
            raise Exception("CATCH:Do not know how to analyze coverage for unsupported plan runType: '%s'" % runtype)
        bedfile = plan['bedfile']
        if bedfile != "":
            config['target_regions'] = bedfile
            config['target_regions_name'] = fileName(bedfile)
        else:
            raise Exception("CATCH:Automated analysis requires a targets region to be specified by the Plan.")
    else:
        raise Exception("CATCH:Automated analysis requires a Plan to specify Run Type.")


def furbishPluginParams():
    '''Complete/rename/validate user parameters.'''
    # For example, HTML form posts do not add unchecked option values
    config = pluginParams['config']
    if config.get('barcodebeds', '') == '': config['barcodebeds'] = 'No'
    if config.get('filterbarcodes', '') == '': config['filterbarcodes'] = 'No'
    if config.get('ercc', '') == '': config['ercc'] = 'No'
    if config.get('uniquemaps', '') == '': config['uniquemaps'] = 'No'
    if config.get('barcode_list', '') == '': config['barcode_list'] = None
    if config.get('user_target_regions', '') == '': config['user_target_regions'] = 'None'
    if config.get('time_of_analysis', '') == '': config['time_of_analysis'] = time.strftime("%Y-%m-%d %H:%M")

def configReport():
    '''Returns a dictionary based on the plugin config parameters that as reported in results.json.'''
    # This is to avoid outputting hidden or aliased values. If not needed just pass back a copy of config.
    # return pluginParams['config'].copy()
    config = pluginParams['config']
    return {
        "Launch Mode": config['launch_mode'],
        "Reference Genome": pluginParams['genome_id'],
        "Library Type": config['librarytype_id'],
        "Targeted Regions": config['target_regions_name'],
        "Filter Barcodes": config['filterbarcodes'],
        "Barcode-specific Targets": config['barcodebeds'],
        "ERCC Tracking": config['ercc'],
        "barcode_list": config['barcode_list'],
        "user_target_regions": config['user_target_regions'],
        "Use Only Uniquely Mapped Reads": config['uniquemaps'],
        "barcoded": "true" if pluginParams['barcoded'] else "false"
    }


def printStartupMessage():
    '''Output the standard start-up message. Customized for additional plugin input and options.'''
    printlog('')
    printtime('Started %s' % pluginParams['plugin_name'])
    config = pluginParams['config']
    printlog('Plugin run parameters:')
    printlog('  Run Report_ID:    %s' % pluginParams.get('jsonParams').get('runinfo').get('pk'))
    printlog('  Plugin version:   %s' % pluginParams['cmdOptions'].version)
    printlog('  Launch mode:      %s' % config['launch_mode'])
    printlog('  Run is barcoded:  %s' % ('Yes' if pluginParams['barcoded'] else 'No'))
    printlog('  Reference Name:   %s' % pluginParams['genome_id'])
    printlog('  Library Type:     %s' % config['librarytype_id'])
    printlog('  Target Regions:   %s' % config['target_regions_name'])
    printlog('  Filter Barcodes:  %s' % config['filterbarcodes'])
    printlog('  Group1 barcodes: %s' % ','.join(pluginParams['group1_barcodes']))
    printlog('  Group2 barcodes: %s' % ','.join(pluginParams['group2_barcodes']))
    # this option is currently disabled from the GUI (but is supported in the code if needed later)
    # printlog('  Barcoded Targets: %s' % config['barcodebeds'])
    if config['barcodebeds'] == 'Yes':
        target_files = pluginParams['target_files']
        for bctrg in sorted(target_files):
            printlog('    %s  %s' % (bctrg, fileName(target_files[bctrg])))
    printlog('  ERCC Tracking:    %s' % config['ercc'])
    printlog('  Uniquely Mapped:  %s' % config['uniquemaps'])
    printlog('Data files used:')
    printlog('  Parameters:     %s' % pluginParams['jsonInput'])
    printlog('  Reference:      %s' % pluginParams['reference'])
    printlog('  Root Alignment: %s' % pluginParams['bamroot'])
    printlog('  Target Regions from config: %s' % config['target_regions'])
    printlog('  Target Regions from pluginParam: %s' % pluginParams.get('config').get('target_regions'))
    printlog('  User Target Regions: %s' % pluginParams['user_target_regions'])
    printlog('  Time of analysis: %s' % config['time_of_analysis'])
    printlog('')
    for (key, val) in config.items():
        printlog('config_K: %s;    Val: %s' % (key, val))
    for (key, val) in pluginParams.items():
        printlog('pluginParam_K: %s;    Val: %s' % (key, val))


def run_plugin(skiprun=False, barcode=""):
    '''Wrapper for making command line calls to perform the specific plugin analyis.'''
    # first part is pretty much boiler plate - grab the key parameters that most plugins use
    logopt = pluginParams['cmdOptions'].logopt
    plugin_dir = pluginParams['plugin_dir']
    output_dir = pluginParams['output_dir']
    output_url = pluginParams['output_url']
    output_prefix = pluginParams['output_prefix']
    bamfile = pluginParams['bamfile']
    bedfile = pluginParams['bedfile']
    config = pluginParams['config']
    sample = sampleName(barcode, 'None')

    # link from source BAM since pipeline uses the name as output file stem
    linkbam = os.path.join(output_dir, output_prefix + ".bam")
    createlink(bamfile, linkbam)
    createlink(bamfile + '.bai', linkbam + '.bai')
    bamfile = linkbam

    # read filters - some hard coded for now
    filtopts = '-a -R 17'
    if config['uniquemaps'] == 'Yes': filtopts += ' -u'

    # skip the actual and assume all the data already exists in this file for processing
    if skiprun:
        printlog("Skipped analysis - generating report on in-situ data")
    else:
        # Pre-run modification of BED files is done here to avoid redundancy of repeating for target assigned barcodes
        # The stand-alone command can perform the (required) annotation
        (mergeBed, annoBed, erccBed) = modifyBedFiles(bedfile)
        runcmd = '%s %s %s -D "%s" -L "%s" -N "%s" -S "%s" -T "%s" "%s" "%s" "%s"' % (
            os.path.join(plugin_dir, 'run_immuneResponseRNA.sh'), pluginParams['logopt'], filtopts,
            output_dir, pluginParams['genome_id'], sample, erccBed, fileName(bedfile),
            pluginParams['reference'], bamfile, annoBed)
        if logopt: printlog('\n$ %s\n' % runcmd)
        if (os.system(runcmd)):
            raise Exception("Failed running run_immuneResponseRNA.sh. Refer to Plugin Log.")

    if pluginParams['cmdOptions'].cmdline: return ({}, {})
    printtime("Generating report...")

    # Link report page resources. This is necessary as the plugin code is inaccesible from URLs directly.
    createlink(os.path.join(plugin_dir, 'flot'), output_dir)
    createlink(os.path.join(plugin_dir, 'lifechart'), output_dir)

    # Optional: Delete intermediate files after successful run. These should not be required to regenerate any of the
    # report if the skip-analysis option. Temporary file deletion is also disabled when the --keep_temp option is used.
    # deleteTempFiles([ '*.bam', '*.bam.bai', '*.bed' ])

    # Create an annotated list of files as used to create the file links table.
    # - Could be handled in the HTML template directly but external code is re-used to match cmd-line reports.

    # Parse out stats from results text file to dict AND convert unacceptible characters to underscores in keys to avoid Django issues
    statsfile = output_prefix + '.stats.cov.txt'
    resultData = parseToDict(os.path.join(output_dir, statsfile), ":")

    # Collect other output data to pluginReport, which is anything else that is used to generate the report
    trgtype = '.amplicon.cov'
    reportData = {
        "library_type": config['librarytype_id'],
        "run_name": output_prefix,
        "barcode_name": barcode,
        "ercc_track": (config['ercc'] == 'Yes'),
        "output_dir": output_dir,
        "output_url": output_url,
        "output_prefix": output_prefix,
        "help_dict": helpDictionary(),
        "stats_txt": checkFileURL(statsfile),
        "rep_overview_png": checkFileURL(output_prefix + '.repoverview.png'),
        "finecov_tsv": checkFileURL(output_prefix + trgtype + '.xls'),
        "bed_link": re.sub(r'^.*/uploads/BED/(\d+)/.*', r'/rundb/uploadstatus/\1/', bedfile),
        "file_links": checkFileURL('filelinks.xls'),
        "bam_link": checkFileURL(output_prefix + '.bam'),
        "bai_link": checkFileURL(output_prefix + '.bam.bai')
    }
    return (resultData, reportData)


def checkFileURL(fileURL):
    '''coverageAnalysis helper method to return "" if the provided file URL does not exist'''
    if os.path.exists(os.path.join(pluginParams['output_dir'], fileURL)):
        return fileURL
    return ""


def modifyBedFiles(bedfile):
    '''coverageAnalysis method to return merged, GC annotated, and ERCC BED files, creating them if they do not already exist.'''
    if not bedfile: return ('', '', '')
    # files will be created or found in this results subdir
    bedDir = os.path.join(pluginParams['results_dir'], "local_beds")
    if not os.path.exists(bedDir): os.makedirs(bedDir)
    # the pair of files returned are dependent on the Library Type
    rootbed = fileName(bedfile)
    mergbed = bedfile.replace('unmerged', 'merged', 1)
    # do not re-do GC annotation on same BED file - moderately expensive for large BEDs
    gcbed = os.path.join(bedDir, "%s.gc.bed" % rootbed)
    if os.path.exists(gcbed):
        printlog("Adopting GC annotated targets %s" % os.path.basename(gcbed))
    else:
        printtime("Creating GC annotated targets %s" % os.path.basename(gcbed))
        if os.system('%s -s -w -f 4,8 -t "%s" "%s" "%s" > "%s"' % (
                os.path.join(pluginParams['plugin_dir'], 'bed', 'gcAnnoBed.pl'),
                bedDir, bedfile, pluginParams['reference'], gcbed)):
            raise Exception("Failed to annotate target regions using gcAnnoBed.pl")
    annobed = gcbed
    erccbed = ''
    if pluginParams['config']['ercc'] == 'Yes':
        erccbed = os.path.join(bedDir, "%s.ercc.bed" % rootbed)
        num_ercc = countFileLines(erccbed)
        if num_ercc >= 0:
            printlog("Adopting %d ERCC targets from %s" % (num_ercc, os.path.basename(erccbed)))
        elif num_ercc < 0:
            printtime("Creating ERCC targets file %s" % os.path.basename(erccbed))
            if os.system("awk '$1~/^ERCC-/ {print}' '%s' > '%s'" % (bedfile, erccbed)):
                raise Exception("Failed to create ERCC targets file using awk command")
            num_ercc = countFileLines(erccbed)
            if num_ercc > 0:
                printlog("  %d ERCC targets detected in target panel." % num_ercc)
            if num_ercc == 0:
                printlog("WARNING: No ERCC targets were detected in targets panel!")
    return (mergbed, annobed, erccbed)


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
    if pluginParams['cmdOptions'].cmdline: return
    printtime("Collating barcodes summary data...")

    # collect barcode statistics from the barcode summary table data and lists of output files
    renderOpts = renderOptions()
    typestr = 'amplicon'
    fileext = '.' + typestr + '.cov.xls'
    bctable = []
    readstable = []
    reportFiles = []
    bclist = ''
    bcresults = pluginResult['barcodes']
    bcreports = pluginReport['barcodes']
    plugin_dir = pluginParams['plugin_dir']
    output_dir = pluginParams['results_dir']
    bc_samples = getBarcodeSampleFromJson()

    #- Copy the gDNA and H2O data from plugin annotation dir
    gDNA_H2O_data = "background_expression_gDNA_H2O_neg_control_201609.xlsx"
    if os.path.isfile(os.path.join(plugin_dir, 'annotations', gDNA_H2O_data)) :
        shutil.copy(os.path.join(plugin_dir, 'annotations/background_expression_gDNA_H2O_neg_control_201609.xlsx'), os.path.join(output_dir, gDNA_H2O_data))
    else:
        gDNA_H2O_data = None

    # iterate barcodeSummary[] to maintain barcode processing order
    for bcdata in barcodeSummary:
        bcname = bcdata['barcode_name']
        if bclist: bclist += ','
        bclist += bcname
        bcrep = bcreports[bcname]
        bcline = bcname + "\t" + bcdata['sample'] + "\t" + bcdata['mapped_reads'] + "\t" + bcdata[
            'valid_target'] + "\t" + bcdata['detected_target'] + "\t" + bcdata['target1'] + "\t" + bcdata['target2']
        if renderOpts['ercc_track']: bcline += "\t" + bcdata['ercc_target']
        bctable.append(bcline)
        reportfile = os.path.join(bcrep['output_dir'], bcrep['output_prefix'] + fileext)
        bcline = bcname + "\t" + bcdata['sample'] + "\t" + bcdata['total_reads'] + "\t" + bcdata[
            'mapped_reads'] + "\t" + bcdata['ontrg_reads'] + "\t" + bcdata['valid_reads']
        if renderOpts['ercc_track']:
            bcline += "\t" + bcdata['ercc_reads']
        else:
            bcline += "\t0"
        readstable.append(bcline)
        if os.path.exists(reportfile):
            reportFiles.append(reportfile)

    if len(bctable) > 0:
        bctabfile = pluginParams['prefix'] + ".bc_summary.xls"
        bcline = "Barcode ID\tSample Name\tMapped Reads\tOn Target\tTargets Detected\tTargets >= 1 reads\tTarget >= 2 reads"
        if renderOpts['ercc_track']: bcline += "\tERCC"
        with open(os.path.join(pluginParams['results_dir'], bctabfile), 'w') as outfile:
            outfile.write(bcline + '\n')
            for bcline in bctable:
                outfile.write(bcline + '\n')
        readsfile = pluginParams['prefix'] + ".reads_summary.xls"
        with open(os.path.join(pluginParams['results_dir'], readsfile), 'w') as outfile:
            outfile.write(
                'Barcode ID\tSample Name\tTotal Reads\tMapped Reads\tOn Target Reads\tAssigned Reads\tERCC Reads\n')
            for bcline in readstable:
                outfile.write(bcline + '\n')
        barcodeReport.update({"bctable": bctabfile, "readstable": readsfile})

    # comparative analysis (plots and files) over all barcodes
    runR = "R --no-save --slave --vanilla --args"
    r_util_fun = os.path.join(plugin_dir, 'scripts', 'utilityFunctions.R')

    numReports = len(reportFiles)
    if numReports > 0:
        bcmatrix = pluginParams['prefix'] + ".bcmatrix.xls"
        p_bcmatrix = os.path.join(pluginParams['results_dir'], bcmatrix)
        with open(p_bcmatrix, 'w') as outfile:
            runcmd = Popen([os.path.join(plugin_dir, 'scripts', 'barcodeMatrix.pl'), '-m', pluginParams['jsonBarcode'],
                            '-A', 'A_', pluginParams['reference'] + '.fai',  '9'] + reportFiles, stdout=outfile)
            runcmd.communicate()
            if runcmd.poll():
                raise Exception("Failed to create barcode x %s reads matrix." % typestr)
        rpmbcmatrix = pluginParams['prefix'] + ".rpm.bcmatrix.xls"
        with open(os.path.join(pluginParams['results_dir'], rpmbcmatrix), 'w') as outfile:
            runcmd = Popen([os.path.join(plugin_dir, 'scripts', 'barcodeMatrix.pl'), '-m', pluginParams['jsonBarcode'],
                            '-A', 'A_', pluginParams['reference'] + '.fai', '12'] + reportFiles, stdout=outfile)
            runcmd.communicate()
            if runcmd.poll():
                raise Exception("Failed to create barcode x %s RPM matrix." % typestr)

        deSummary = ""
        derTable = ""
        if create_DE_matrix:
            if numReports > 1:
                derTable = pluginParams['prefix'] + ".deratio.xls"
                if os.system('%s -N 1000000 -S RPM "%s" -a -M %d > "%s"' % (
                        os.path.join(plugin_dir, 'scripts', 'tableDE.pl'), bcmatrix, numReports, derTable)):
                    raise Exception("Failed to create differential expression matrix using tableDE.pl")
                if os.system('awk \'NR==1;NR>1{print|"sort -k 6,6nr -k 1,1d -k 2,2d"}\' "%s" > sort.xls.tmp; mv sort.xls.tmp "%s"' % (
                        derTable, derTable)):
                    raise Exception("Failed to sort differential expression matrix using awk command.")
                with open(os.path.join(pluginParams['results_dir'], derTable), 'r') as infile:
                    nline = 0
                    nde = 0
                    for line in infile:
                        nline += 1
                        if nline == 1: continue
                        fields = line.split('\t')
                        if float(fields[len(fields) - 1]) >= difexp_thresh: nde += 1
                    if nline > 1:
                        deSummary = "%d targets (%.2f%%) showed differential expression at %s-fold or greater." % (
                        nde, 100 * float(nde) / (nline - 1), difexp_thresh)
                        pluginResult['Differentially expressed targets'] = str(nde)

        #############################################
        ##- top5 and bot5 plots are not displayed
        ##- perform housekeeping-scaled (normalzed) RPM as well in this script.
        topExpPlot = pluginParams['prefix'] + ".topExp.png"
        botExpPlot = pluginParams['prefix'] + ".botExp.png"
        housekeepingExpPlot = pluginParams['prefix'] + ".housekeepingPlot.png"
        hk_scaled_log2_count = pluginParams['prefix'] + '.mean_hk_scaled_log2_count.xls'
        if os.system('%s "%s" "%s" "%s" "%s" "%s" "%s" "Expression characteristics" "R" 0.001 "%s" "%s" "%s" < %s' % (
                runR,
                os.path.join(pluginParams['results_dir'], rpmbcmatrix),
                numReports,
                os.path.join(output_dir, topExpPlot),
                os.path.join(output_dir, botExpPlot),
                os.path.join(output_dir, housekeepingExpPlot),
                os.path.join(output_dir, hk_scaled_log2_count),
                r_util_fun,
                pluginParams.get('config').get('target_regions'),
                pluginParams.get('jsonParams').get('runinfo').get('pk'),
                os.path.join(plugin_dir, 'scripts', 'doSampleCharacteristics.R'))):
            raise Exception("Failed to create top 5% expression plot using doSampleCharacteristics.R")

        #############################################
        # create correlation matrix plots from RPM reads matrix: generates the r-value matrix required for heatmap
        cpairsPlot = pluginParams['prefix'] + ".corpairs.png"
        pcaSamplePlot = pluginParams['prefix'] + ".pcaSamplePlot.png"
        cpairsTitle = "log2 RPM pair correlation plots" if numReports > 1 else "log2 RPM density plot"
        rvalueMatrix = pluginParams['prefix'] + ".rvalues.xls"
        if numReports < 2:
            cpairsPlot = 'None'
            pcaSamplePlot = 'None'
        elif os.system('%s "%s" "%s" "%s" %d "%s" "%s" < %s' % (
                runR,
                rpmbcmatrix, cpairsPlot, pcaSamplePlot, numReports, cpairsTitle,
                os.path.join(output_dir, rvalueMatrix),
                os.path.join(plugin_dir, 'scripts', 'plot_cormatrix.R'))):
            raise Exception("Failed to create barcode RPM paired correlation plots using plot_cormatrix.R")

        #############################################
        # create heatmap plot from r-value matrix
        rvalueHeatmap = pluginParams['prefix'] + ".corbc.hm.png"
        if numReports < 2:
            rvalueHeatmap = 'None'
        elif os.system('%s "%s" "%s" "Sample Correlation Heatmap" "r-value" 0.4 < %s' % (
                runR,
                os.path.join(output_dir, rvalueMatrix),
                os.path.join(output_dir, rvalueHeatmap),
                os.path.join(plugin_dir, 'scripts', 'plot_corbc_heatmap.R'))):
            raise Exception("Failed to create barcode heatmap plots using plot_corbc_heatmap.R")

        #############################################
        # create heatmaplot of top 500 variant genes vs. barcode
        #- use rdthresh (count threshold) = 0 and minRPM=0 to plot all the targets
        genevarHeatmap = pluginParams['prefix'] + ".genebc.hm.png"
        if numReports < 2:
            genevarHeatmap = 'None'
        elif os.system('%s "%s" "%s" "%s" "Clustering All Targets" "Representation: log10(RPM+1)" 500 0 0 %d < %s' % (
                runR,
                os.path.join(output_dir, bcmatrix),
                os.path.join(output_dir, genevarHeatmap),
                r_util_fun, numReports,
                os.path.join(plugin_dir, 'scripts', 'plot_genebc_heatmap.R'))):
            raise Exception("Failed to create barcode heatmap plots using plot_genebc_heatmap.R")

        #############################################
        ##- Create hierachical clustering heatmap based on user uploaded bed file containing genes of interest
        ##- The bed file should be a subset of the entire panel bed file
        heatmap_user_target_regions_plot = 'None'
        fold_change_user_target_plot = pluginParams['prefix'] + '.fold_change_user_target.png'
        if numReports > 1 and pluginParams['user_target_regions'] != 'None':
            heatmap_user_target_regions_plot = pluginParams['prefix'] + ".heatmap_user_target_regions.png"
            if os.system('%s "%s" "%s" "%s" "%s" "Clustering based on genes of interest" "log2(RPM+1)" %d "%s" "%s" "%s" < %s' % (
                runR,
                os.path.join(output_dir, rpmbcmatrix),
                os.path.join(output_dir, heatmap_user_target_regions_plot),
                r_util_fun,
                pluginParams['user_target_regions'],
                numReports,
                ','.join(pluginParams['group1_barcodes']),
                ','.join(pluginParams['group2_barcodes']),
                os.path.join(output_dir, fold_change_user_target_plot),
                os.path.join(plugin_dir, 'scripts', 'doAnalysis_userTargetRegions.R'))):
                raise Exception("Failed to create heatmap plots from genes of interest using doAnalysis_userTargetRegions.R")

        #############################################
        # create overlaid gene log10 distribution frequency curve (w/o genes with 0 reads)
        genepdfPlot = pluginParams['prefix'] + ".genepdf.png"
        if os.system('%s "%s" "%s" %d "Distribution of Gene Expression" < %s' % (
                runR,
                os.path.join(output_dir, bcmatrix),
                os.path.join(output_dir, genepdfPlot),
                numReports,
                os.path.join(plugin_dir, 'scripts', 'plot_multi_pdf.R'))):
            raise Exception("Failed to create gene read pdf plot using plot_multi_pdf.R")

        # create barchart of mapped reads
        # alignmentPlot = pluginParams['prefix']+".mapreads.png"
        # if os.system( '%s "%s" "%s" "Reads Alignment Summary" "Million Reads" 0.000001 < %s' % ( runR,
        #    os.path.join(output_dir,readsfile), os.path.join(output_dir,alignmentPlot),
        #    os.path.join(plugin_dir,'scripts','plot_reads_hbar.R') ) ):
        #  raise Exception("Failed to create barcode read alignment plot using plot_reads_hbar.R")

        #############################################
        ##- Differential expression analysis only if there is 2 groups defined
        fcp_volcano_plot = 'None'
        fcp_data_file = 'None'
        if len(pluginParams['group1_barcodes']) > 1  and len(pluginParams['group2_barcodes']) > 1:
            fcp_volcano_plot = pluginParams['prefix'] + ".fcp_volcano_plot.png"
            fcp_data_file = pluginParams['prefix'] + ".foldchange_pvalue.xls"
            if os.system('%s "%s" "%s" "%s" "%s" "Fold change and p value" %d "%s" "%s" "%s" < %s' % (
                runR,
                os.path.join(output_dir, rpmbcmatrix),
                os.path.join(output_dir, fcp_volcano_plot),
                r_util_fun,
                pluginParams['user_target_regions'],
                numReports,
                ','.join(pluginParams['group1_barcodes']),
                ','.join(pluginParams['group2_barcodes']),
                os.path.join(output_dir, fcp_data_file),
                os.path.join(plugin_dir, 'scripts', 'doFoldChange_t_test.R'))):
                raise Exception("Failed to create heatmap plots from genes of interest using doFoldChange_t_test.R")

        #############################################
        ##- convert normalized rpm to CHP format for TAC software
        ##- Housekeeping normalized
        tac_dir = os.path.join(plugin_dir, 'tac')
        chp_converter = os.path.join(tac_dir, 'convert2chp.py')
        file_for_chp = pluginParams['prefix'] + '.mean_hk_scaled_log2.forCHP.tab'
        file_chp_zip = pluginParams['prefix'] + '.mean_hk_scaled_log2.chp.zip'
        chp_dir = pluginParams['prefix'] + '.mean_hk_scaled_log2_CHP'
        if os.system('python %s -m %s -i %s -o %s' % (chp_converter, "housekeeping-normalized", file_for_chp, chp_dir)):
            raise Exception("Failed to create CHP files for %s." % file_for_chp)
        else:
            os.system("zip -r " + file_chp_zip + " " + chp_dir)

        #############################################
        ##- convert normalized rpm to CHP format for TAC software
        ##- RPM values NOT housekeeping normalized
        file_for_chp2 = pluginParams['prefix'] + '_rpm_forCHP.tab'
        file_chp_zip2 = pluginParams['prefix'] + '_rpm_log2.chp.zip'
        chp_dir2 = pluginParams['prefix'] + '.log2_rpm_CHP'
        if os.system('python %s -m %s -i %s -o %s' % (chp_converter, 'RPM-normalized', file_for_chp2, chp_dir2)):
            raise Exception("Failed to create CHP files for %s." % file_for_chp2)
        else:
            os.system("zip -r " + file_chp_zip2 + " " + chp_dir2)

        # record output files for use in barcode summary report
        # (p_bcmatrix used for passing to php script for interactive utilities)
        barcodeReport.update({
            "bclist": bclist,
            "bcmtype": typestr,
            "bcmatrix": bcmatrix,
            "p_bcmatrix": p_bcmatrix,
            "rpmbcmatrix": rpmbcmatrix,
            "chp_housekeeping_zip": file_chp_zip,
            "chp_rpm_zip": file_chp_zip2,
            "hk_scaled_log2_count": hk_scaled_log2_count,
            "rvaluematrix": rvalueMatrix,
            "readmaps": readsfile,
            "featmatrix": rvalueMatrix,
            "genepdfplot": genepdfPlot,
            "mapreadsplot": topExpPlot,
            "bot5expplot": botExpPlot,
            "housekeepingplot": housekeepingExpPlot,
            "pcasampleplot": pcaSamplePlot,
            "heatmapplot": rvalueHeatmap,
            "heatmap_user_target_regions_plot" : heatmap_user_target_regions_plot,
            #"fold_change_user_target_plot": fold_change_user_target_plot,
            "genebcplot": genevarHeatmap,
            "cpairsplot": cpairsPlot,
            'fcp_volcano_plot': fcp_volcano_plot,
            'gDNA_H2O_data': gDNA_H2O_data
        })

    # create symlink for js/css - the (empty) tabs on report page will not appear until this exists
    createlink(os.path.join(plugin_dir, 'lifechart'), output_dir)


def updateBarcodeSummaryReport(barcode, autoRefresh=False):
    '''Create barcode summary (progress) report. Called before, during and after barcodes are being analysed.'''
    global barcodeSummary
    if pluginParams['cmdOptions'].cmdline: return
    renderOpts = renderOptions()
    # no barcode means either non have been ceated yet or this is a refresh after all have been processed (e.g. for meta data)
    if barcode != "":
        resultData = pluginResult['barcodes'][barcode]
        reportData = pluginReport['barcodes'][barcode]
        errMsg = resultData.get('Error', '')
        sample = sampleName(barcode, 'None')
        # barcodes_json dictoonary is firmcoded in Kendo table template that we are using for main report styling
        if errMsg != "":
            detailsLink = "<span class='help' title='%s' style='color:red'>%s</span>" % (errMsg, barcode)
            barcodeSummary.append({
                "index": len(barcodeSummary),
                "barcode_name": barcode,
                "barcode_details": detailsLink,
                "sample": sample,
                "total_reads": "NA",
                "mapped_reads": "NA",
                "ontrg_reads": "NA",
                "valid_reads": "NA",
                "ercc_reads": "NA",
                "valid_target": "NA",
                "detected_target": "NA",
                "target1": "NA",
                "target2": "NA",
                "ercc_target": "NA",
                "time_of_analysis": "NA"
            })
        else:
            detailsLink = "<a target='_parent' href='%s' class='help'><span title='Click to view the detailed report for barcode %s'>%s</span><a>" % (
                os.path.join(barcode, pluginParams['report_name']), barcode, barcode)
            numTargets = int(resultData['Number of amplicons'])
            #pcDetected = 100 * float(
            #    resultData['Amplicons with at least 10 reads']) / numTargets if numTargets > 0 else 0
            num_detected = resultData['Amplicons with at least 10 reads']
            barcodeSummary.append({
                "index": len(barcodeSummary),
                "barcode_name": barcode,
                "barcode_details": detailsLink,
                "sample": sample,
                "total_reads": resultData['Number of total reads'],
                "mapped_reads": resultData['Number of mapped reads'],
                "ontrg_reads": resultData['Number of on-target reads'],
                "valid_reads": resultData['Number of assigned reads'],
                "ercc_reads": resultData['Number of ERCC tracking reads'] if renderOpts['ercc_track'] else "NA",
                "valid_target": resultData['Percent assigned reads'],
                "target1": resultData['Amplicons with at least 1 reads'],
                "target2": resultData['Amplicons with at least 2 reads'],
                #"detected_target": ("%.2f" % pcDetected) + "%",
                "detected_target": num_detected,
                "ercc_target": resultData['Percent ERCC tracking reads'] if renderOpts['ercc_track'] else "NA"
            })
    render_context = {
        "autorefresh": autoRefresh,
        "run_name": pluginParams['prefix'],
        "barcode_results": simplejson.dumps(barcodeSummary),
        "num_barcodes_filtered": pluginReport['num_barcodes_filtered'],
        "barcode_filter": pluginReport['barcode_filter'],
        "help_dict": helpDictionary()
    }
    render_context.update(renderOpts)
    # extra report items, e.g. file links from barcodes summary page
    if barcodeReport:
        render_context.update(barcodeReport)
    createReport(os.path.join(pluginParams['results_dir'], pluginParams['report_name']), 'barcode_summary.html',
                 render_context)


def renderOptions():
    '''coverageAnalysis support method to generate list of condensed rendering options and values.'''
    config = pluginParams['config']
    targets = 'Barcode-specific' if config['barcodebeds'] == 'Yes' else config['target_regions_name']
    if targets == 'None': targets = ""
    filter_options = []
    if config['ercc'] == 'Yes': filter_options.append('ERCC tracking')
    if config['uniquemaps'] == 'Yes': filter_options.append('Uniquely mapped')
    # extra filters may become a user option
    filter_options.append('Alignment length at least 50% of insert size.')
    sample_grouping = False
    if (len(pluginParams['group1_barcodes']) > 1 and len(pluginParams['group2_barcodes']) > 1):
        sample_grouping = True
    return {
        "library_type": config['librarytype_id'],
        "target_regions": targets,
        "filter_options": ', '.join(filter_options),
        "ercc_track": (config['ercc'] == 'Yes'),
        'sample_grouping': sample_grouping,
        'user_target_regions_name': config['user_target_regions_name'],
        'time_of_analysis': config['time_of_analysis']
    }


def createIncompleteReport(errorMsg=""):
    '''Called to create an incomplete or error report page for non-barcoded runs.'''
    render_context = {
        "autorefresh": (errorMsg == ""),
        "run_name": pluginParams['prefix'],
        "Sample_Name": sampleName(),
        'user_target_regions_name': config['user_target_regions_name'],
        'time_of_analysis': config['time_of_analysis'],
        "Error": errorMsg}
    render_context.update(renderOptions())
    createReport(os.path.join(pluginParams['results_dir'], pluginParams['report_name']), 'incomplete.html',
                 render_context)


def createDetailReport(resultData, reportData):
    '''Called to create the main report (for un-barcoded run or for each barcode).'''
    if pluginParams['cmdOptions'].cmdline: return
    render_context = resultData.copy()
    render_context.update(reportData)
    createReport(os.path.join(pluginParams['output_dir'], pluginParams['report_name']), 'report.html', render_context)


def createBlockReport():
    '''Called at the end of run to create a block.html report. Use 'pass' if not wanted.'''
    if pluginParams['cmdOptions'].cmdline: return
    printtime("Creating block report...")
    if pluginParams['barcoded']:
        render_context = {
            "run_name": pluginParams['prefix'],
            "barcode_results": simplejson.dumps(barcodeSummary),
            "help_dict": helpDictionary()}
        render_context.update(renderOptions())
        tplate = 'barcode_block.html'
    else:
        render_context = pluginResult.copy()
        render_context.update(pluginReport)
        tplate = 'report_block.html'
    createReport(pluginParams['block_report'], tplate, render_context)


def createProgressReport(progessMsg):
    '''General method to write a message directly to the block report, e.g. when starting prcessing of a new barcode.'''
    createReport(pluginParams['block_report'], "progress_block.html", {"progress_text": progessMsg})


def helpDictionary():
    '''coverageAnalysis method to load a dictionary for on-line help in the reports.'''
    global help_dictionary
    if not help_dictionary:
        with open(os.path.join(pluginParams['plugin_dir'], 'templates', 'help_dict.json')) as jsonFile:
            help_dictionary = json.load(jsonFile)
    return help_dictionary


#
# --------------- Base code for standard plugin runs -------------
#

def parseCmdArgs():
    '''Process standard command arguments. Customized for additional debug and other run options.'''
    # standard run options here - do not remove
    parser = OptionParser()
    parser.add_option('-B', '--bam', help='Filepath to root alignment BAM file. Default: rawlib.bam', dest='bamfile',
                      default='')
    parser.add_option('-P', '--prefix',
                      help='Output file name prefix for output files. Default: '' => Use analysis folder name or "output".',
                      dest='prefix', default='')
    parser.add_option('-R', '--reference_fasta', help='Path to fasta file for the whole reference', dest='reference',
                      default='')
    parser.add_option('-U', '--results_url', help='URL for access to files in the output directory', dest='results_url',
                      default='')
    parser.add_option('-V', '--version', help='Version string for tracking in output', dest='version', default='')
    parser.add_option('-X', '--min_bc_bam_size', help='Minimum file size required for barcode BAM processing',
                      type="int", dest='minbamsize', default=0)
    parser.add_option('-c', '--cmdline',
                      help='Run command line only. Reports will not be generated using the HTML templates.',
                      action="store_true", dest='cmdline')
    parser.add_option('-d', '--scraper',
                      help='Create a scraper folder of links to output files using name prefix (-P).',
                      action="store_true", dest='scraper')
    parser.add_option('-k', '--keep_temp',
                      help='Keep intermediate files. By default these are deleted after a successful run.',
                      action="store_true", dest='keep_temp')
    parser.add_option('-l', '--log', help='Output extra progress Log information to STDERR during a run.',
                      action="store_true", dest='logopt')
    parser.add_option('-p', '--purge_results', help='Remove all folders and most files from output results folder.',
                      action="store_true", dest='purge_results')
    parser.add_option('-s', '--skip_analysis', help='Skip re-generation of existing files but make new report.',
                      action="store_true", dest='skip_analysis')
    parser.add_option('-x', '--stop_on_error',
                      help='Stop processing barcodes after one fails. Otherwise continue to the next.',
                      action="store_true", dest='stop_on_error')

    (cmdOptions, args) = parser.parse_args()
    if (len(args) != 2):
        printerr('Takes only two file arguments: startplugin.json barcodes.json')
        raise TypeError(os.path.basename(__file__) + " takes exactly two arguments (%d given)." % len(args))
    with open(args[0]) as jsonFile:
        jsonParams = json.load(jsonFile)
    global pluginParams, jsonBarcode
    with open(args[1]) as jsonFile:
        jsonBarcode = json.load(jsonFile)
    pluginParams['cmdOptions'] = cmdOptions
    pluginParams['jsonInput'] = args[0]
    pluginParams['jsonBarcode'] = args[1]
    pluginParams['jsonParams'] = jsonParams


def emptyResultsFolder():
    '''Purge everything in output folder except for specifically named files.'''
    if not pluginParams['cmdOptions'].purge_results: return
    results_dir = pluginParams['results_dir']
    if results_dir == '/': return
    logopt = pluginParams['cmdOptions'].logopt
    cwd = os.path.realpath(os.getcwd())
    if logopt or os.path.exists(os.path.join(results_dir, pluginParams['report_name'])):
        printlog("Purging old results...")
    for root, dirs, files in os.walk(results_dir, topdown=False):
        for name in files:
            # these are the exceptions - partial names and in the to level results
            if root == results_dir:
                start = os.path.basename(name)[:10]
                if start == "drmaa_stdo" or start == "ion_plugin" or start == "startplugi" or start == 'barcodes.j':
                    continue
            fname = os.path.realpath(os.path.join(root, name))
            if fname.startswith(cwd): continue
            # if logopt and root == results_dir:
            #  printlog("Removing file %s"%fname)
            printlog("Removing file %s" % fname)
            os.system('rm -f "%s"' % fname)
        for name in dirs:
            fname = os.path.realpath(os.path.join(root, name))
            if fname.startswith(cwd):
                printlog("Warning: Leaving folder %s as in cwd path." % fname)
                continue
            if logopt:
                printlog("Removing directory %s" % fname)
            os.system('rm -rf "%s"' % fname)
    if logopt: printlog("")


def parseToDict(filein, sep=None):
    ret = {}
    if os.path.exists(filein):
        with open(filein) as fin:
            for line in fin:
                line = line.strip()
                # ignore lines being with non-alphanum (for comments, etc)
                if line == "" or not line[0].isalnum():
                    continue
                kvp = line.split(sep, 1)
                if len(kvp) > 1:
                    ret[kvp[0].strip()] = kvp[1].strip()
    else:
        printerr("parseToDict() could not open " + filein)
    return ret


def printerr(msg):
    cmd = os.path.basename(__file__)
    sys.stderr.write('%s: ERROR: %s\n' % (cmd, msg))
    sys.stderr.flush()


def printlog(msg):
    sys.stderr.write(msg)
    sys.stderr.write('\n')
    sys.stderr.flush()


def printtime(msg):
    # use unix 'date' command so output format is identical to called script
    runtime = Popen(["date"], stdout=PIPE, shell=False)
    dtm = runtime.communicate()[0]
    printlog('(%s) %s' % (dtm.strip(), msg))


def createlink(srcPath, destPath):
    # using system call as os.symlink() only seems to handle one file at a time and has other limitations
    if not srcPath:
        printlog("WARNING: Failed to create symlink as source path is empty.")
        return False
    elif not os.path.exists(srcPath):
        printlog("WARNING: Failed to create symlink as source path '%s' was not found." % srcPath)
        return False
    elif not destPath:
        printlog("WARNING: Failed to create symlink as destination path is empty.")
        return False
    noErrMsg = "2> /dev/null" if pluginParams['cmdOptions'].skip_analysis else ""
    os.system('ln -s "%s" "%s" %s' % (srcPath, destPath, noErrMsg))
    if pluginParams['cmdOptions'].logopt:
        printlog("Created symlink %s -> %s" % (destPath, srcPath))
    return True


def deleteTempFiles(tmpFiles):
    if tmpFiles == None or pluginParams['cmdOptions'].keep_temp: return
    output_dir = pluginParams['output_dir']
    for filename in tmpFiles:
        flist = glob(os.path.join(output_dir, filename))
        for f in flist:
            if pluginParams['cmdOptions'].logopt:
                printlog("Deleting file %s" % f)
            os.unlink(f)


def createReport(reportName, reportTemplate, reportData):
    # configure django to use the templates folder and various installed apps
    if not settings.configured:
        plugin_dir = pluginParams['plugin_dir'] if 'plugin_dir' in pluginParams else os.path.realpath(__file__)
        settings.configure(DEBUG=False, TEMPLATE_DEBUG=False,
                           INSTALLED_APPS=('django.contrib.humanize',),
                           TEMPLATE_DIRS=(os.path.join(plugin_dir, 'templates'),))

    with open(reportName, 'w') as bcsum:
        bcsum.write(render_to_string(reportTemplate, safeKeys(reportData)))


def sampleNames():
    try:
        if pluginParams['barcoded']:
            samplenames = {}
            bcsamps = pluginParams['jsonParams']['plan']['barcodedSamples']
            if isinstance(bcsamps, basestring):
                bcsamps = json.loads(bcsamps)
            for bcname in bcsamps:
                for bc in bcsamps[bcname]['barcodes']:
                    samplenames[bc] = bcname if bcname != 'Unknown' else ''
        else:
            samplenames = jsonParams['expmeta']['sample']
    except:
        return ""
    return samplenames


def sampleName(barcode='', default=''):
    if not 'sample_names' in pluginParams:
        return default
    sample_names = pluginParams['sample_names']
    if isinstance(sample_names, basestring):
        return sample_names if sample_names else default
    return sample_names.get(barcode, default) if barcode else default

def getBarcodeSampleFromJson():
    '''process barcodes.json file to get sample names for barcode. If sample name is not set, use barcode ID
        if duplicate sample_name, use barcode index with sample_name
        replace space with underscore
    '''
    with open(pluginParams['jsonBarcode']) as json_file:
        json_barcodes = json.load(json_file)
    barcodes = json_barcodes.keys()
    barcodes.sort()
    bc_sample = dict()
    sample_name = dict()
    for bc in barcodes:
        barcode_detail = getBarcodeDetail(bc)
        sample = barcode_detail.get('sample_name')
        sample = re.sub(r' +', r'_', sample)
        if sample == '' or sample == 'none' or sample is None:
            sample = bc
        bc_index = barcode_detail.get('barcode_index')
        if (sample_name.get(sample) is not None) :
            sample = "IDx" + bc_index + "_" + sample
        else:
            sample_name[sample] = sample
        bc_sample[bc] = sample

    return(bc_sample)


def getBarcodeDetail(barcode=''):
    '''Generate barcode detail information from jsonBarcode data'''
    if not barcode:
        barcode = NOMATCH
    if not barcode in jsonBarcode:
        return { "filtered" : True }

    barcodeData = jsonBarcode[barcode]
    sample_name = barcodeData.get('sample','')
    sample_name = sample_name.replace(' ', '_')
    if sample_name == 'none' or sample_name == '':
        sample_name = barcode
    filtered = barcodeData.get('filtered', True)
    if barcode == NOMATCH: filtered = True
    reference = barcodeData['reference']
    reference_path = barcodeData['reference_fullpath']
    genome_url = barcodeData.get('genome_urlpath','')
    idx = -1
    if not genome_url:
        idx = reference_path.find('referenceLibrary/')
    if idx >= 0:
        genome_url = os.path.join("/output", reference_path[idx + 17])

    # special exception to allow full IGV annotation
    if reference == 'hg19': genome_url = ''

    target_region_file = barcodeData['target_region_filepath']

    return {
        "filtered" : filtered,
        "sample_name" : sample_name,
        "reference" : reference,
        "reference_path" : reference_path,
        "nuc_type" : barcodeData.get('nucleotide_type', 'DNA'),
        "reference_url" : '{http_host}' + genome_url if genome_url else reference,
        "target_region_file" : target_region_file,
        "bam_file" : barcodeData.get('bam_filepath', ''),
        "read_count" : barcodeData.get('read_count', 1),
        "barcode_index" : barcodeData.get('barcode_index') }

def targetFiles():
    trgfiles = {}
    try:
        if pluginParams['barcoded']:
            bcbeds = pluginParams['config']['barcodetargetregions']
            if isinstance(bcbeds, basestring):
                bcmaps = bcbeds.split(';')
                for m in bcmaps:
                    kvp = m.split('=', 1)
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
    ##- analysis_dir is the dir of the report
    ##- result_dir is the plugin output dir
    jsonParams = pluginParams['jsonParams']
    pluginParams['plugin_name'] = jsonParams['runinfo'].get('plugin_name', '')
    pluginParams['plugin_dir'] = jsonParams['runinfo'].get('plugin_dir', '.')
    pluginParams['genome_id'] = jsonParams['runinfo'].get('library', '')
    pluginParams['run_name'] = jsonParams['expmeta'].get('run_name', '')
    pluginParams['analysis_name'] = jsonParams['expmeta'].get('results_name', pluginParams['plugin_name'])
    pluginParams['analysis_dir'] = jsonParams['runinfo'].get('analysis_dir', '.')
    pluginParams['results_dir'] = jsonParams['runinfo'].get('results_dir', '.')
    pluginParams['logopt'] = '-l' if pluginParams['cmdOptions'].logopt else ''

    # some things not yet in startplugin.json are provided or over-writen by cmd args
    copts = pluginParams['cmdOptions']
    pluginParams['reference'] = copts.reference if copts.reference != "" else jsonParams['runinfo'].get('reference', '')
    pluginParams['bamroot'] = copts.bamfile if copts.bamfile != "" else '%s/rawlib.bam' % pluginParams['analysis_dir']
    pluginParams['prefix'] = copts.prefix if copts.prefix != "" else pluginParams['analysis_name']
    pluginParams['results_url'] = copts.results_url if copts.results_url != "" else os.path.join(
        jsonParams['runinfo'].get('url_root', '.'), 'plugin_out', pluginParams['plugin_name'] + '_out')

    # check for non-supported de novo runs
    if not pluginParams['genome_id'] or not pluginParams['reference']:
        printerr("Requires a reference sequence for coverage analysis.")
        raise Exception(
            "CATCH:Do not know how to analyze coverage without reference sequence for library '%s'" % pluginParams.get(
                'genome_id', ""))

    # set up for barcoded vs. non-barcodedruns
    pluginParams['bamfile'] = pluginParams['bamroot']
    pluginParams['output_dir'] = pluginParams['results_dir']
    pluginParams['output_url'] = pluginParams['results_url']
    pluginParams['output_prefix'] = pluginParams['prefix']
    pluginParams['bamname'] = os.path.basename(pluginParams['bamfile'])
    pluginParams['barcoded'] = os.path.exists(pluginParams['analysis_dir'] + '/barcodeList.txt')
    pluginParams['sample_names'] = sampleNames()

    # disable run skip if no report exists => plugin has not been run before
    pluginParams['report_name'] = pluginParams['plugin_name'] + '.html'
    pluginParams['block_report'] = os.path.join(pluginParams['results_dir'],
                                                pluginParams['plugin_name'] + '_block.html')
    if not os.path.exists(os.path.join(pluginParams['results_dir'], pluginParams['report_name'])):
        if pluginParams['cmdOptions'].skip_analysis:
            printlog("Warning: Skip analysis option ignorred as previous output appears to be missing.")
            pluginParams['cmdOptions'].skip_analysis = False

    # set up plugin specific options depending on auto-run vs. plan vs. GUI
    config = pluginParams['config'] = jsonParams['pluginconfig'].copy() if 'pluginconfig' in jsonParams else {}
    launchmode = config.get('launch_mode', '')
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

    # code to handle single or per-barcode target files
    pluginParams['target_files'] = targetFiles()
    pluginParams['have_targets'] = (config['target_regions'] or pluginParams['target_files'])

    pluginParams['user_target_regions'] = (config['user_target_regions'] or 'None')
    pluginParams['user_target_regions_name'] = (fileName(config['user_target_regions']) or 'None')

    ###- get a list of barcode and their grouping info
    barcode_list = None
    if config.get('barcode_list') is not None:
        barcode_list = pluginParams['barcode_list'] = (config['barcode_list'] or None)
    else:
        pluginParams['barcode_list'] = None
        config['barcode_list'] = None
    group1_barcodes = []
    group2_barcodes = []
    bc_samples = getBarcodeSampleFromJson()
    if barcode_list is not None:
        for bc in barcode_list.split(','):
            group = config[bc]
            if group == 'group1':
                #group1_barcodes.extend([bc])
                group1_barcodes.extend([ bc_samples[bc] ])
            elif group == 'group2':
                #group2_barcodes.extend([bc])
                group2_barcodes.extend([ bc_samples[bc] ])
    pluginParams['group1_barcodes'] = group1_barcodes
    pluginParams['group2_barcodes'] = group2_barcodes


    # plugin configuration becomes basis of results.json file
    global pluginResult, pluginReport
    pluginResult = configReport()
    if pluginParams['barcoded']:
        pluginResult['barcodes'] = {}
        pluginReport['barcodes'] = {}


def fileName(filepath):
    filepath = os.path.basename(filepath)
    return os.path.splitext(filepath)[0]


def writeDictToJsonFile(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=2, sort_keys=True)
        if pluginParams['cmdOptions'].logopt:
            printlog("Created JSON file '%s'" % filename)


def safeKeys(indict):
    # Recursive method to return a dictionary with non alpha-numeric characters in dictionary key names replaced with underscores.
    # Expects indict to be a json-compatible dictionary or array of dictionaries.
    # A non-dicionary object (reference) is returned, i.e. no copy is made as with arrays and dicionaries.
    # lists and tuple (subclass) objects are returned as ordinary lists
    if isinstance(indict, (list, tuple)):
        nlist = []
        for item in indict:
            nlist.append(safeKeys(item))
        return nlist
    if not isinstance(indict, dict):
        return indict
    retdict = {}
    for key, value in indict.iteritems():
        retdict[re.sub(r'[^0-9A-Za-z]', '_', key)] = safeKeys(value)
    return retdict


def testRun(outdir, prefix):
    # default for testing framework
    testout = os.path.join(outdir, prefix + "_test.out")
    with open(testout, 'w') as f:
        f.write("This is a test file.\n")
    printlog('Created %s' % testout)


def ensureFilePrefix(prependLen=0):
    global pluginParams
    prefix = pluginParams['prefix']
    maxfn = prependLen + len(prefix) + max_fileext_len
    if maxfn <= max_filename_len: return
    # clip prefix to maximum size for allowed (before prepend/append)
    prefix = prefix[:max_filename_len - maxfn]
    maxfn = len(prefix)
    # use nearest '_' if doesn't reduce the length of name by more than 70%
    uslen = prefix.rfind('_')
    if uslen >= 0.7 * maxfn:
        prefix = prefix[:uslen]
    printlog(
        "WARNING: Output file name stem shortened to ensure output file name length <= %d characters.\nNew stem = %s\n" % (
        max_filename_len, prefix))
    pluginParams['prefix'] = prefix


def runForBarcodes():
    global pluginParams, pluginResult, pluginReport
    # read barcode ids
    barcodes = []
    try:
        bcfileName = pluginParams['analysis_dir'] + '/barcodeList.txt'
        with open(bcfileName) as bcfile:
            for line in bcfile:
                if line.startswith('barcode '):
                    barcodes.append(line.split(',')[1])
    except:
        printerr("Reading barcode list file '%s'" % bcfileName)
        raise
    # grab barcoded-target information
    have_targets = pluginParams['have_targets']
    default_target = pluginParams['config']['target_regions']
    check_targets = (have_targets and not default_target)
    target_files = pluginParams['target_files']
    # iterate over listed barcodes to pre-test barcode files
    numGoodBams = 0
    maxBarcodeLen = 0
    minFileSize = pluginParams['cmdOptions'].minbamsize
    (bcBamPath, bcBamRoot) = os.path.split(pluginParams['bamroot'])
    bcBamFile = []
    # for bacode filtering first make pass to find largest barcode BAM file size
    minBamSize = minFileSize
    numBamSmall = 0
    if pluginParams['config']['filterbarcodes'] == 'Yes':
        maxBamSize = 0
        for barcode in barcodes:
            bcbam = os.path.join(bcBamPath, "%s_%s" % (barcode, bcBamRoot))
            if os.path.exists(bcbam):
                fsiz = os.stat(bcbam).st_size
                if fsiz > maxBamSize: maxBamSize = fsiz
        minBamSize = int(barcode_filter * maxBamSize)
    # pre-apply BAM file filters
    for barcode in barcodes:
        bcbam = os.path.join(bcBamPath, "%s_%s" % (barcode, bcBamRoot))
        if not os.path.exists(bcbam):
            bcbam = ": BAM file not found"
        elif check_targets and barcode not in target_files:
            bcbam = ": No assigned or default target regions for barcode."
        elif os.stat(bcbam).st_size < minBamSize:
            if minBamSize == minFileSize:
                bcbam = ": BAM file too small"
            else:
                bcbam = ": BAM file too small relative to largest"
                numBamSmall += 1
        else:
            if (len(barcode) > maxBarcodeLen):
                maxBarcodeLen = len(barcode)
            numGoodBams += 1
        bcBamFile.append(bcbam)

    ensureFilePrefix(maxBarcodeLen + 1)

    if numBamSmall > 1:
        printlog("WARNING: %d bamfiles discounted as too small compared to largest BAM file.\n" % numBamSmall)
    printlog("Processing %d barcodes...\n" % numGoodBams)
    pluginReport['num_barcodes_processed'] = numGoodBams
    pluginReport['num_barcodes_failed'] = 0
    pluginReport['num_barcodes_filtered'] = numBamSmall
    pluginReport['barcode_filter'] = 100 * barcode_filter

    pluginResult['Barcodes filtered'] = str(numBamSmall)

    # create initial (empty) barcodes summary report
    createlink(os.path.join(pluginParams['plugin_dir'], 'lifechart'), pluginParams['results_dir'])
    updateBarcodeSummaryReport("", True)

    # iterate over all barcodes and process the valid ones
    skip_analysis = pluginParams['cmdOptions'].skip_analysis
    stop_on_error = pluginParams['cmdOptions'].stop_on_error
    create_scraper = pluginParams['cmdOptions'].scraper
    postout = False;  # just for logfile prettiness
    barcodeProcessed = 0
    for barcode in barcodes:
        sample = sampleName(barcode)
        bamfile = bcBamFile.pop(0)
        if bamfile[0] == ":":
            if postout:
                postout = False
                printlog("")
            printlog("Skipping %s%s%s" % (barcode, ('' if sample == '' else ' (%s)' % sample), bamfile))
        else:
            postout = True
            printlog("\nProcessing %s%s...\n" % (barcode, ('' if sample == '' else ' (%s)' % sample)))
            if have_targets:
                target_file = target_files[barcode] if barcode in target_files else default_target
                pluginParams['bedfile'] = target_file
                target_file = target_file.replace('unmerged', 'merged', 1)
                printlog('Target Regions: %s' % target_file)
            else:
                pluginParams['bedfile'] = ''
            pluginParams['bamfile'] = bamfile
            pluginParams['output_dir'] = os.path.join(pluginParams['results_dir'], barcode)
            pluginParams['output_url'] = os.path.join(pluginParams['results_url'], barcode)
            pluginParams['output_prefix'] = barcode + "_" + pluginParams['prefix']
            if not os.path.exists(pluginParams['output_dir']):
                os.makedirs(pluginParams['output_dir'])
            try:
                barcodeProcessed += 1
                createProgressReport("Processing barcode %d of %d..." % (barcodeProcessed, numGoodBams))
                (resultData, reportData) = run_plugin(skip_analysis, barcode)
                pluginResult['barcodes'][barcode] = resultData
                pluginReport['barcodes'][barcode] = reportData
                createDetailReport(resultData, reportData)
                if create_scraper:
                    createScraperLinksFolder(pluginParams['output_dir'], pluginParams['output_prefix'])
            except Exception, e:
                printerr('Analysis of barcode %s failed:' % barcode)
                pluginReport['num_barcodes_failed'] += 1
                pluginResult['barcodes'][barcode] = {"Sample Name": sample, "Error": str(e)}
                pluginReport['barcodes'][barcode] = {}
                if stop_on_error: raise
                traceback.print_exc()
            updateBarcodeSummaryReport(barcode, True)

    createProgressReport("Compiling barcode summary report...")
    run_meta_plugin()
    updateBarcodeSummaryReport("")
    if create_scraper:
        createScraperLinksFolder(pluginParams['results_dir'], pluginParams['prefix'])


def runNonBarcoded():
    global pluginResult, pluginReport
    ensureFilePrefix()
    try:
        pluginParams['bedfile'] = pluginParams['config']['target_regions'] if pluginParams['have_targets'] else ''
        createIncompleteReport()
        (resultData, pluginReport) = run_plugin(pluginParams['cmdOptions'].skip_analysis)
        pluginResult.update(resultData)
        createDetailReport(pluginResult, pluginReport)
    except Exception, e:
        printerr('Analysis failed')
        pluginResult.update({'Error': str(e)})
        createIncompleteReport(str(e))
        raise
    if pluginParams['cmdOptions'].scraper:
        createScraperLinksFolder(pluginParams['output_dir'], pluginParams['output_prefix'])


def createScraperLinksFolder(outdir, rootname):
    '''Make links to all files matching <outdir>/<rootname>.* to <outdir>/scraper/link.*'''
    # rootname is a file path relative to outdir and should not contain globbing characters
    scrapeDir = os.path.join(outdir, 'scraper')
    if pluginParams['cmdOptions'].logopt:
        printlog("Creating scraper folder %s" % scrapeDir)
    if not os.path.exists(scrapeDir):
        os.makedirs(scrapeDir)
    subroot = os.path.basename(rootname) + '.'
    flist = glob(os.path.join(outdir, rootname) + '.*')
    for f in flist:
        lname = os.path.basename(f).replace(subroot, 'link.')
        createlink(f, os.path.join(scrapeDir, lname))


def wrapup():
    '''Called at very end of run for final data dump and clean up.'''
    if not 'Error' in pluginResult: createBlockReport()
    printtime("Writing results.json...")
    jsonfile = os.path.join(pluginParams['results_dir'], "results.json")
    writeDictToJsonFile(pluginResult, jsonfile)


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
        wrapup()
    except Exception, e:
        traceback.print_exc()
        wrapup()  # call only if suitable partial results are available, including some error status
        return 1
    return 0


if __name__ == "__main__":
    exit(plugin_main())
