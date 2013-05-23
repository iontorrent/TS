#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

from ion.utils.blockprocessing import printtime
from ion.utils.blockprocessing import isbadblock
from ion.utils.blockprocessing import MyConfigParser
from ion.utils import blockprocessing

import traceback
import os
import ConfigParser
import tempfile
# matplotlib/numpy compatibility
os.environ['HOME'] = tempfile.mkdtemp()
from matplotlib import use
use("Agg",warn=False)

import subprocess
import sys
import time
import json
import math
from collections import deque
from urlparse import urlunsplit

#from ion.reports.plotters import *
from ion.utils.aggregate_alignment import *
from ion.utils import ionstats_plots
from ion.utils import ionstats


def align(
    libraryName,
    lib_path,
    align_full,
    sam_parsed,
    bidirectional,
    mark_duplicates,
    realign,
    skip_sorting,
    aligner_opts_extra,
    logfile,
    output_dir,
    output_basename):
    #     Input -> output_basename.bam
    #     Output -> output_dir/output_basename.bam

    try:
        cmd = "alignmentQC.py"
        #cmd = "alignmentQC.pl"
        cmd += " --logfile %s" % logfile
        cmd += " --output-dir %s" % output_dir
        cmd += " --input %s" % lib_path
        cmd += " --genome %s" % libraryName
        #cmd += " --max-plot-read-len %s" % str(int(800))
        cmd += " --out-base-name %s" % output_basename
        #cmd += " --skip-alignStats"
        #cmd += " --threads 8"
        #cmd += " --server-key 13"

        #if align_full:
        #    cmd += " --align-all-reads"
        #if sam_parsed:
        #    cmd  += ' -p 1'
        if realign:
            cmd += " --realign"
        if skip_sorting:
            cmd += " --skip-sorting"
        if bidirectional:
            cmd += ' --bidirectional'
        if aligner_opts_extra:
            cmd += ' --aligner-opts-extra "%s"' % aligner_opts_extra
        if mark_duplicates:
            cmd += ' --mark-duplicates'

        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        if ret != 0:
            printtime("ERROR: alignmentQC.py failed, return code: %d" % ret)
            raise RuntimeError('exit code: %d' % ret)
    except:
        raise





def alignment_unmapped_bam(
        BASECALLER_RESULTS,
        ALIGNMENT_RESULTS,
        align_full,
        libraryName,
        flows,
        realign,
        aligner_opts_extra,
        mark_duplicates,
        bidirectional,
        sam_parsed):

    printtime("Attempt to align")

    skip_sorting = False

    datasets_basecaller_path = os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json")

    if not os.path.exists(datasets_basecaller_path):
        printtime("ERROR: %s does not exist" % datasets_basecaller_path)
        raise Exception("ERROR: %s does not exist" % datasets_basecaller_path)

    datasets_basecaller = {}
    try:
        f = open(datasets_basecaller_path,'r')
        datasets_basecaller = json.load(f)
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % datasets_basecaller_path)
        raise Exception("ERROR: problem parsing %s" % datasets_basecaller_path)

    for dataset in datasets_basecaller["datasets"]:
        if not os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
            continue

        #-----------------------------------
        # Analysis - (Runs for barcoded runs too)
        #-----------------------------------
        try:
            align(
                libraryName,
                os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']),
                align_full,
                sam_parsed,
                bidirectional,
                mark_duplicates,
                realign,
                skip_sorting,
                aligner_opts_extra,
                logfile=os.path.join(ALIGNMENT_RESULTS,dataset['file_prefix']+'.alignmentQC_out.txt'),
                output_dir=ALIGNMENT_RESULTS,
                output_basename=dataset['file_prefix'])
        except:
            traceback.print_exc()

        printtime("Barcode processing, rename")


    alignment_post_processing(libraryName, BASECALLER_RESULTS, ALIGNMENT_RESULTS, flows, mark_duplicates)

    printtime("**** Alignment completed ****")



def alignment_post_processing(
        libraryName,
        BASECALLER_RESULTS,
        ALIGNMENT_RESULTS,
        flows,
        mark_duplicates):


    datasets_basecaller = {}
    try:
        f = open(os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"),'r')
        datasets_basecaller = json.load(f)
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"))
        traceback.print_exc()
        return

    try:
        graph_max_x = int(50 * math.ceil(0.014 * int(flows)))
    except:
        graph_max_x = 800



    alignment_file_list = []

    for dataset in datasets_basecaller["datasets"]:
        if not os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
            continue

        ionstats.generate_ionstats_alignment(
                os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam'),
                os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json'),
                graph_max_x)
        ionstats2alignstats(libraryName,
                os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json'),
                os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.alignment.summary'))

        alignment_file_list.append(os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json'))

    # In Progress: merge ionstats alignment results
    ionstats.reduce_stats(alignment_file_list,os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'))    
    ionstats2alignstats(libraryName,
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            os.path.join(ALIGNMENT_RESULTS,'alignment.summary'))

    # Special legacy post-processing.
    # Generate merged rawlib.bam on barcoded runs

    composite_bam_filename = os.path.join(ALIGNMENT_RESULTS,'rawlib.bam')
    if not os.path.exists(composite_bam_filename):

        bam_file_list = []
        for dataset in datasets_basecaller["datasets"]:
            bam_name = os.path.join(ALIGNMENT_RESULTS,os.path.basename(dataset['file_prefix'])+'.bam')
            if os.path.exists(bam_name):
                bam_file_list.append(bam_name)

        blockprocessing.merge_bam_files(bam_file_list,composite_bam_filename,composite_bam_filename+'.bai',mark_duplicates)

    # Generate alignment_barcode_summary.csv
    #TODO: use datasets_basecaller.json + *.ionstats_alignment.json instead of barcodeList.txt and alignment_*.summary
    barcodelist_path = 'barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../../../barcodeList.txt'
    if os.path.exists(barcodelist_path):
        printtime("Barcode processing, aggregate")
        aggregate_alignment ("./",barcodelist_path)

    # These graphs are likely obsolete
    #makeAlignGraphs()

    # In Progress: Use ionstats alignment results to generate plots
    ionstats_plots.alignment_rate_plot2(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            'alignment_rate_plot.png', graph_max_x)
    ionstats_plots.base_error_plot(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            'base_error_plot.png', graph_max_x)
    ionstats_plots.old_aq_length_histogram(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            'Filtered_Alignments_Q10.png', 'AQ10', 'red')
    ionstats_plots.old_aq_length_histogram(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            'Filtered_Alignments_Q17.png', 'AQ17', 'yellow')
    ionstats_plots.old_aq_length_histogram(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            'Filtered_Alignments_Q20.png', 'AQ20', 'green')
    ionstats_plots.old_aq_length_histogram(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'), 
            'Filtered_Alignments_Q47.png', 'AQ47', 'purple')

def ionstats2alignstats(libraryName,ionstats_file,alignstats_file):

    stats = json.load(open(ionstats_file,'r'))

    config_out = ConfigParser.RawConfigParser()
    config_out.optionxform = str # don't convert to lowercase
    config_out.add_section('global')


    config_out.set('global', 'Total number of Reads', int(stats['full']['num_reads']))

    config_out.set('global', 'Total Mapped Reads', int(stats['aligned']['num_reads']))
    config_out.set('global', 'Total Mapped Target Bases', int(stats['aligned']['num_bases']))


    quallist = ['Q7', 'Q10', 'Q17', 'Q20', 'Q30', 'Q47']

    for q in quallist:
        config_out.set('global', 'Filtered %s Longest Alignment' % q, int(stats['A'+q]['max_read_length']))
        config_out.set('global', 'Filtered %s Mean Alignment Length' % q, int(stats['A'+q]['mean_read_length']))
        config_out.set('global', 'Filtered Mapped Bases in %s Alignments' % q, int(stats['A'+q]['num_bases']))
        config_out.set('global', 'Filtered %s Alignments' % q, int(stats['A'+q]['num_reads']))
        config_out.set('global', 'Filtered %s Coverage Percentage' % q, 'N/A')
        config_out.set('global', 'Filtered %s Mean Coverage Depth' % q, '%.1f' % (float(stats['A'+q]['num_bases'])/float(3095693981)) ) # 3095693981 TODO

        summ = 0
        for (i,x) in enumerate(stats['A'+q]['read_length_histogram']):
            summ += x
            if (i+1)%50 == 0:
                config_out.set('global', 'Filtered %s%s Reads' % (i+1, q), stats['A'+q]['num_reads'] - summ)

    genomeinfodict = {}
    try:
        genomeinfofilepath = '/results/referenceLibrary/tmap-f3/%s/%s.info.txt' % (libraryName,libraryName)
        with open(genomeinfofilepath) as genomeinfofile:
            for line in genomeinfofile:
                key, value = line.partition("\t")[::2]
                genomeinfodict[key.strip()] = value.strip()
    except:
        traceback.print_exc()

    config_out.set('global', 'Genome', genomeinfodict['genome_name'])
    config_out.set('global', 'Genome Version', genomeinfodict['genome_version'])
    config_out.set('global', 'Genomesize', genomeinfodict['genome_length'])
    config_out.set('global', 'Index Version', genomeinfodict['index_version'])

    with open(alignstats_file, 'wb') as configfile:
        config_out.write(configfile)





def merge_alignment_stats(dirs, BASECALLER_RESULTS, ALIGNMENT_RESULTS, flows):

    datasets_basecaller = {}
    try:
        f = open(os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"),'r')
        datasets_basecaller = json.load(f)
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"))
        traceback.print_exc()
        return



    try:
        graph_max_x = int(50 * math.ceil(0.014 * int(flows)))
    except:
        graph_max_x = 800




    ########################################################
    # Merge ionstats_alignment.json
    # First across blocks, then across barcoded
    ########################################################

    try:
        composite_filename_list = []
        for dataset in datasets_basecaller["datasets"]:
            composite_filename = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json')
            barcode_filename_list = [os.path.join(dir,ALIGNMENT_RESULTS,dataset['file_prefix']+'.ionstats_alignment.json') for dir in dirs]
            barcode_filename_list = [filename for filename in barcode_filename_list if os.path.exists(filename)]
            ionstats.reduce_stats(barcode_filename_list,composite_filename)
            if os.path.exists(composite_filename):
                composite_filename_list.append(composite_filename)

        ionstats.reduce_stats(composite_filename_list,os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'))
    except:
        printtime("ERROR: Failed to merge ionstats_alignment.json")
        traceback.print_exc()

    # Use ionstats alignment results to generate plots
    ionstats_plots.alignment_rate_plot2(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            'alignment_rate_plot.png', graph_max_x)
    ionstats_plots.base_error_plot(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            'base_error_plot.png', graph_max_x)
    ionstats_plots.old_aq_length_histogram(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            'Filtered_Alignments_Q10.png', 'AQ10', 'red')
    ionstats_plots.old_aq_length_histogram(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            'Filtered_Alignments_Q17.png', 'AQ17', 'yellow')
    ionstats_plots.old_aq_length_histogram(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'),
            'Filtered_Alignments_Q20.png', 'AQ20', 'green')
    ionstats_plots.old_aq_length_histogram(
            os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'), 
            'Filtered_Alignments_Q47.png', 'AQ47', 'purple')



    # Generate alignment_barcode_summary.csv
    barcodelist_path = 'barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../../barcodeList.txt'
    if not os.path.exists(barcodelist_path):
        barcodelist_path = '../../../../barcodeList.txt'
    if os.path.exists(barcodelist_path):
        printtime("Barcode processing, aggregate")
        aggregate_alignment ("./",barcodelist_path)




def merge_alignment_bigdata(dirs, BASECALLER_RESULTS, ALIGNMENT_RESULTS, mark_duplicates):

    datasets_json = {}
    try:
        f = open(os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"),'r')
        datasets_json = json.load(f)
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"))
        traceback.print_exc()
        return

    for dataset in datasets_json['datasets']:
        # Merge BAMs
        try:
            block_bam_list = [os.path.join(dir,ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam') for dir in dirs]
            block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
            composite_bam_filename = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam')
            if block_bam_list:
                blockprocessing.merge_bam_files(block_bam_list,composite_bam_filename,composite_bam_filename+'.bai',mark_duplicates)
        except:
            printtime("ERROR: merging %s unsuccessful" % (dataset['file_prefix']+'.bam'))

