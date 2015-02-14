#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

from ion.utils.blockprocessing import printtime
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
    referenceName,
    lib_path,
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
        cmd += " --logfile %s" % logfile
        cmd += " --output-dir %s" % output_dir
        cmd += " --input %s" % lib_path
        cmd += " --genome %s" % referenceName
        #cmd += " --max-plot-read-len %s" % str(int(800))
        cmd += " --out-base-name %s" % output_basename
        #cmd += " --skip-alignStats"
        #cmd += " --threads 8"
        #cmd += " --server-key 13"

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


def merge_barcoded_alignment_bams(ALIGNMENT_RESULTS, basecaller_datasets, method):

    try:
        composite_bam_filename = os.path.join(ALIGNMENT_RESULTS,'rawlib.bam')

        bam_file_list = []
        for dataset in basecaller_datasets["datasets"]:
            bam_name = os.path.join(ALIGNMENT_RESULTS,os.path.basename(dataset['file_prefix'])+'.bam')
            if os.path.exists(bam_name):
                bam_file_list.append(bam_name)
            else:
                printtime("WARNING: exclude %s from merging into %s" % (bam_name,composite_bam_filename))

        composite_bai_filename = composite_bam_filename+'.bai'
        mark_duplicates = False
        blockprocessing.merge_bam_files(bam_file_list, composite_bam_filename, composite_bai_filename, mark_duplicates, method)
    except:
        traceback.print_exc()
        printtime("ERROR: Generate merged %s on barcoded run failed" % composite_bam_filename)

    printtime("Finished barcode merging of %s" % ALIGNMENT_RESULTS)


def alignment_unmapped_bam(
        BASECALLER_RESULTS,
        basecaller_datasets,
        ALIGNMENT_RESULTS,
        realign,
        aligner_opts_extra,
        mark_duplicates,
        create_index,
        bidirectional,
        activate_barcode_filter,
        barcodeInfo):

    printtime("Attempt to align")

    skip_sorting = False
    
    for dataset in basecaller_datasets["datasets"]:

        # filter out based on flag
        if activate_barcode_filter:
            keep_dataset = False
            for rg_name in dataset["read_groups"]:
                if not basecaller_datasets["read_groups"][rg_name].get('filtered',False):
                    keep_dataset = True
            if not keep_dataset:
                continue

        # needed for unfiltered data
        if int(dataset["read_count"]) == 0:
            continue

        '''
        sample, barcode = dataset['dataset_name'].split('/')
        barcode_name = barcode.replace('No_barcode_match','no_barcode')
        reference = barcodeInfo[barcode_name]['referenceName'] # TODO
        '''

        read_group = dataset['read_groups'][0]
        reference = basecaller_datasets['read_groups'][read_group]['reference']
        print '%s' % reference
        if not reference:
            continue

        try:
            align(
                reference,
                os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']),
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

        if create_index:
            try:
                composite_bam_filepath = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam')
                composite_bai_filepath = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam.bai')
                blockprocessing.create_index_file(composite_bam_filepath, composite_bai_filepath)
            except:
                traceback.print_exc()

    printtime("**** Alignment completed ****")



def create_ionstats(
        BASECALLER_RESULTS,
        ALIGNMENT_RESULTS,
        basecaller_meta_information,
        basecaller_datasets,
        graph_max_x,
        activate_barcode_filter,
        evaluate_hp):

    # TEST
    basecaller_bam_file_list = []
    alignment_bam_file_list = []


    ionstats_alignment_file_list = []
    if evaluate_hp:
        ionstats_alignment_h5_file_list = []

    ionstats_basecaller_file_list = []

    for dataset in basecaller_datasets["datasets"]:

        keep_dataset = False
        for rg_name in dataset["read_groups"]:
            if not basecaller_datasets["read_groups"][rg_name].get('filtered',False):
                keep_dataset = True
        filtered = not keep_dataset

        # filter out based on flag
        if activate_barcode_filter:
            if filtered:
                continue

        # skip non-existing bam file
        if int(dataset["read_count"]) == 0:
            continue

        read_group = dataset['read_groups'][0]
        reference = basecaller_datasets['read_groups'][read_group]['reference']
        if reference and not filtered:

            # TEST
            alignment_bam_file_list.append(os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam'))

            ionstats.generate_ionstats_alignment(
                [os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam')],
                os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json'),
                os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_error_summary.h5') if evaluate_hp else None,
                basecaller_meta_information if evaluate_hp else None,
                graph_max_x)

            ionstats_alignment_file_list.append(os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json'))
            if evaluate_hp:
                ionstats_alignment_h5_file_list.append(os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_error_summary.h5'))
        else:

            # TEST
            basecaller_bam_file_list.append(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']))

            ionstats.generate_ionstats_basecaller(
                [os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])],
                os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'),
                graph_max_x)

            ionstats_basecaller_file_list.append(os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'))


    # Merge ionstats files from individual (barcoded) datasets
    if len(ionstats_alignment_file_list) > 0:
        ionstats.reduce_stats(ionstats_alignment_file_list,os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json'))
    else: # barcode classification filtered all barcodes or no reads available
        # TODO: ionstats needs to produce initial json file
        try:
            #cmd = "echo $'@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"
            cmd  = "echo  '@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"

            printtime("DEBUG: Calling '%s':" % cmd)
            ret = subprocess.call(cmd,shell=True)
            if ret != 1:
                printtime("ERROR: empty bam file generation failed, return code: %d" % ret)
                raise RuntimeError('exit code: %d' % ret)

            ionstats.generate_ionstats_alignment(
                ['empty_dummy.bam'],
                os.path.join(ALIGNMENT_RESULTS, 'ionstats_alignment.json'),
                os.path.join(ALIGNMENT_RESULTS, 'ionstats_error_summary.h5') if evaluate_hp else None,
                basecaller_meta_information if evaluate_hp else None,
                graph_max_x)

        except:
            pass

    if len(ionstats_basecaller_file_list) > 0:
        ionstats.reduce_stats(ionstats_basecaller_file_list,os.path.join(BASECALLER_RESULTS,'ionstats_tmp_basecaller.json'))
    else: # barcode classification filtered all barcodes or no reads available
        # TODO: ionstats needs to produce initial json file
        try:
            #cmd = "echo $'@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"
            cmd  = "echo  '@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"

            printtime("DEBUG: Calling '%s':" % cmd)
            ret = subprocess.call(cmd,shell=True)
            if ret != 1:
                printtime("ERROR: empty bam file generation failed, return code: %d" % ret)
                raise RuntimeError('exit code: %d' % ret)

            ionstats.generate_ionstats_basecaller(
                ['empty_dummy.bam'],
                os.path.join(BASECALLER_RESULTS, 'ionstats_tmp_basecaller.json'),
                graph_max_x)
        except:
            pass


    ionstatslist = []
    a = os.path.join(ALIGNMENT_RESULTS,'ionstats_alignment.json')
    b = os.path.join(BASECALLER_RESULTS,'ionstats_tmp_basecaller.json')
    if os.path.exists(a):
        ionstatslist.append(a)
    if os.path.exists(b):
        ionstatslist.append(b)
    if len(ionstatslist) > 0:
        ionstats.reduce_stats( ionstatslist, os.path.join(BASECALLER_RESULTS,'ionstats_basecaller_with_aligninfos.json'))
        ionstats.reduce_stats( reversed(ionstatslist), os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json'))
    if evaluate_hp and len(ionstats_alignment_h5_file_list) > 0 and basecaller_meta_information:
        ionstats.reduce_stats_h5(ionstats_alignment_h5_file_list,os.path.join(ALIGNMENT_RESULTS,'ionstats_error_summary.h5'))

    '''
    # TEST: TS-8745, single call to ionstats alignment operating on all the per-barcode BAMs at once
    if len(alignment_bam_file_list) > 0:
        ionstats.generate_ionstats_alignment(
                alignment_bam_file_list,
                os.path.join(ALIGNMENT_RESULTS, 'test_ionstats_alignment.json'),
                os.path.join(ALIGNMENT_RESULTS, 'test_ionstats_error_summary.h5'),
                basecaller_meta_information,
                graph_max_x)
    # TODO: ionstats basecaller doesn't support multiple input files
    if len(basecaller_bam_file_list) > 0:
        ionstats.generate_ionstats_basecaller(
                basecaller_bam_file_list,
                os.path.join(ALIGNMENT_RESULTS, 'test2_ionstats_basecaller.json'),
                os.path.join(ALIGNMENT_RESULTS, 'test2_ionstats_error_summary.h5'), # TODO, not needed
                basecaller_meta_information,
                graph_max_x)
    '''

def plot_main_report_histograms(BASECALLER_RESULTS,ALIGNMENT_RESULTS,basecaller_datasets,graph_max_x):

    ionstats_folder = BASECALLER_RESULTS
    ionstats_file = 'ionstats_basecaller.json'

    # Plot new read length histogram
    ionstats_plots.read_length_histogram(
        os.path.join(ionstats_folder,ionstats_file),
        os.path.join(BASECALLER_RESULTS,'readLenHisto2.png'),
        graph_max_x)

    for dataset in basecaller_datasets["datasets"]:

        keep_dataset = False
        for rg_name in dataset["read_groups"]:
            if not basecaller_datasets["read_groups"][rg_name].get('filtered',False):
                keep_dataset = True
        if not keep_dataset:
            continue

        read_group = dataset['read_groups'][0]
        reference = basecaller_datasets['read_groups'][read_group]['reference']

        if reference:
            ionstats_folder = ALIGNMENT_RESULTS
            ionstats_file = 'ionstats_alignment.json'
        else:
            ionstats_folder = BASECALLER_RESULTS
            ionstats_file = 'ionstats_basecaller.json'

        # Plot read length sparkline
        ionstats_plots.read_length_sparkline(
            os.path.join(ionstats_folder, dataset['file_prefix']+'.'+ionstats_file),
            os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.sparkline.png'),
            graph_max_x)

        # Plot higher detail barcode specific histogram
        ionstats_plots.old_read_length_histogram(
            os.path.join(ionstats_folder, dataset['file_prefix']+'.'+ionstats_file),
            os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.read_len_histogram.png'),
            graph_max_x)


def create_plots(ionstats_filepath,flows):

    try:
        graph_max_x = int(50 * math.ceil(0.014 * int(flows)))
    except:
        graph_max_x = 800

    # Use ionstats alignment results to generate plots
    ionstats_plots.alignment_rate_plot2(ionstats_filepath,'alignment_rate_plot.png', graph_max_x)

    ionstats_plots.base_error_plot(ionstats_filepath,'base_error_plot.png', graph_max_x)

    ionstats_plots.old_aq_length_histogram(ionstats_filepath,'Filtered_Alignments_Q10.png', 'AQ10', 'red')
    ionstats_plots.old_aq_length_histogram(ionstats_filepath,'Filtered_Alignments_Q17.png', 'AQ17', 'yellow')
    ionstats_plots.old_aq_length_histogram(ionstats_filepath,'Filtered_Alignments_Q20.png', 'AQ20', 'green')
    ionstats_plots.old_aq_length_histogram(ionstats_filepath,'Filtered_Alignments_Q47.png', 'AQ47', 'purple')


def merge_ionstats(dirs, BASECALLER_RESULTS, ALIGNMENT_RESULTS, basecaller_datasets):

    # Merge *ionstats_alignment.json files across blocks

    # DEBUG: check if merging is commutative

    try:
        # DEBUG
        composite_filename_list = []
        composite_h5_filename_list = []

        for dataset in basecaller_datasets["datasets"]:

            # filter out based on flag
            keep_dataset = False
            for rg_name in dataset["read_groups"]:
                if not basecaller_datasets["read_groups"][rg_name].get('filtered',False):
                    keep_dataset = True
            if not keep_dataset:
                printtime("INFO: filter out %s" % rg_name)
                continue

            read_group = dataset['read_groups'][0]
            reference = basecaller_datasets['read_groups'][read_group]['reference']

            if reference:
                ionstats_folder = ALIGNMENT_RESULTS
                ionstats_file = 'ionstats_alignment.json'
            else:
                ionstats_folder = BASECALLER_RESULTS
                ionstats_file = 'ionstats_basecaller.json'

            block_filename_list = [os.path.join(dir,ionstats_folder,dataset['file_prefix']+'.'+ionstats_file) for dir in dirs]
            block_filename_list = [filename for filename in block_filename_list if os.path.exists(filename)] # TODO, remove this check and provide list with valid blocks
            composite_filename = os.path.join(ionstats_folder, dataset['file_prefix']+'.composite_allblocks_'+ionstats_file)
            ionstats.reduce_stats(block_filename_list, composite_filename)
            composite_filename_list.append(composite_filename)

            if reference:
                block_h5_filename_list = [os.path.join(dir,ALIGNMENT_RESULTS,dataset['file_prefix']+'.ionstats_error_summary.h5') for dir in dirs]
                block_h5_filename_list = [filename for filename in block_h5_filename_list if os.path.exists(filename)]  # TODO, remove this check and provide list with valid blocks
                composite_h5_filename = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_error_summary.h5')
                ionstats.reduce_stats_h5(block_h5_filename_list, composite_h5_filename)
                composite_h5_filename_list.append(composite_h5_filename)


        block_filename_list = [os.path.join(dir,ALIGNMENT_RESULTS,'ionstats_alignment.json') for dir in dirs]
        block_filename_list = [filename for filename in block_filename_list if os.path.exists(filename)]
        composite_filename = os.path.join(ALIGNMENT_RESULTS, 'composite_allblocks_ionstats_alignment.json')
        ionstats.reduce_stats(block_filename_list, composite_filename)

        block_h5_filename_list = [os.path.join(dir,ALIGNMENT_RESULTS,'ionstats_error_summary.h5') for dir in dirs]
        block_h5_filename_list = [filename for filename in block_h5_filename_list if os.path.exists(filename)]
        composite_filename = os.path.join(ALIGNMENT_RESULTS, 'ionstats_error_summary.h5') # composite_allblocks
        if len(block_h5_filename_list):
            ionstats.reduce_stats_h5(block_h5_filename_list, composite_filename)

        # DEBUG: this is used to check if merging is commutative, the length check is necessary in case  all datasets are 'filtered' (e.g.)
        if len(composite_filename_list) > 0:
            ionstats.reduce_stats(composite_filename_list,os.path.join(ALIGNMENT_RESULTS,'composite_allbarcodes_ionstats_alignment.json'))
        if len(composite_h5_filename_list) > 0:
            ionstats.reduce_stats_h5(composite_h5_filename_list,os.path.join(ALIGNMENT_RESULTS,'composite_allbarcodes_ionstats_error_summary.h5'))

    except:
        printtime("ERROR: Failed to merge ionstats_alignment.json")
        traceback.print_exc()



def merge_bams(dirs, BASECALLER_RESULTS, ALIGNMENT_RESULTS, basecaller_datasets, mark_duplicates):

    for dataset in basecaller_datasets['datasets']:

        try:
            read_group = dataset['read_groups'][0]
            reference = basecaller_datasets['read_groups'][read_group]['reference']

            filtered = True
            for rg_name in dataset["read_groups"]:
                if not basecaller_datasets["read_groups"][rg_name].get('filtered',False):
                    filtered = False

            if reference and not filtered:
                bamdir = ALIGNMENT_RESULTS
                bamfile = dataset['file_prefix']+'.bam'
            else:
                bamdir = BASECALLER_RESULTS
                bamfile = dataset['basecaller_bam']
            block_bam_list = [os.path.join(blockdir, bamdir, bamfile) for blockdir in dirs]
            block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
            composite_bam_filepath = os.path.join(bamdir, bamfile)
            if block_bam_list:
                if reference and not filtered:
                    composite_bai_filepath = composite_bam_filepath+'.bai'
                    blockprocessing.merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates)
                else:
                    composite_bai_filepath=""
                    blockprocessing.merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates=False, method='samtools')

        except:
            print traceback.format_exc()
            printtime("ERROR: merging %s unsuccessful" % bamfile)


