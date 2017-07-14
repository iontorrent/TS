#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved


import subprocess
import os
import sys
import traceback
import json
import re
import time
import dateutil.parser
import math
import shlex
import copy
from collections import defaultdict

from ion.utils.blockprocessing import isbadblock
from ion.utils.blockprocessing import printtime

from ion.reports import mergeBaseCallerJson
from ion.utils import blockprocessing


def basecaller_cmd(basecallerArgs,
                   SIGPROC_RESULTS,
                   libKey,
                   tfKey,
                   runID,
                   BASECALLER_RESULTS,
                   block_offset_xy,
                   datasets_pipeline_path,
                   adapter):
    if basecallerArgs:
        cmd = basecallerArgs
    else:
        cmd = "BaseCaller"
        printtime("ERROR: BaseCaller command not specified, using default: 'BaseCaller'")

    cmd += " --input-dir=%s" % (SIGPROC_RESULTS)
    cmd += " --librarykey=%s" % (libKey)
    cmd += " --tfkey=%s" % (tfKey)
    cmd += " --run-id=%s" % (runID)
    cmd += " --output-dir=%s" % (BASECALLER_RESULTS)
    cmd += " --block-offset %d,%d" % block_offset_xy
    cmd += " --datasets=%s" % (datasets_pipeline_path)
    cmd += " --trim-adapter %s" % (adapter)

    phase_estimates_json = os.path.join(SIGPROC_RESULTS, "PhaseEstimates.json")
    if os.path.exists(phase_estimates_json):
        cmd += " --phase-estimation-file %s" % phase_estimates_json

    return cmd


def basecalling(
    block_offset_xy,
    SIGPROC_RESULTS,
    basecallerArgs,
    libKey,
    tfKey,
    runID,
    adaptersequence,
    BASECALLER_RESULTS,
    barcodeId,
    barcodeInfo,
    library,
    notes,
    site_name,
    platform,
    instrumentName,
    chipType,
):

    if not os.path.exists(BASECALLER_RESULTS):
        os.mkdir(BASECALLER_RESULTS)

    ''' Step 1: Generate datasets_pipeline.json '''

    # New file, datasets_pipeline.json, contains the list of all active result files.
    # Tasks like post_basecalling, alignment, plugins, must process each specified file and merge results

    datasets_pipeline_path = os.path.join(BASECALLER_RESULTS, "datasets_pipeline.json")

    try:
        generate_datasets_json(
            barcodeId,
            barcodeInfo,
            library,
            runID,
            notes,
            site_name,
            platform,
            instrumentName,
            chipType,
            datasets_pipeline_path,
        )
    except:
        printtime('ERROR: Generation of %s unsuccessful' % datasets_pipeline_path)
        traceback.print_exc()

    ''' Step 2: Invoke BaseCaller '''

    try:

        cmd = basecaller_cmd(basecallerArgs,
                             SIGPROC_RESULTS,
                             libKey,
                             tfKey,
                             runID,
                             BASECALLER_RESULTS,
                             block_offset_xy,
                             datasets_pipeline_path,
                             adaptersequence)

        printtime("DEBUG: Calling '%s':" % cmd)
        proc = subprocess.Popen(shlex.split(cmd.encode('utf8')), shell=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout_value, stderr_value = proc.communicate()
        ret = proc.returncode
        sys.stdout.write("%s" % stdout_value)
        sys.stderr.write("%s" % stderr_value)

        # Ion Reporter
        try:
            basecaller_log_path = os.path.join(BASECALLER_RESULTS, 'basecaller.log')
            with open(basecaller_log_path, 'a') as f:
                if stdout_value: f.write(stdout_value)
                if stderr_value: f.write(stderr_value)
        except IOError:
            traceback.print_exc()

        if ret != 0:
            printtime('ERROR: BaseCaller failed with exit code: %d' % ret)
            raise Exception("BaseCaller failed with exit code: %d" % ret)
    except:
        printtime('ERROR: BaseCaller failed')
        traceback.print_exc()
        raise Exception("BaseCaller failed")

    printtime("Finished basecaller processing")


def merge_barcoded_basecaller_bams(BASECALLER_RESULTS, basecaller_datasets, method):

    try:
        composite_bam_filename = os.path.join(BASECALLER_RESULTS, 'rawlib.basecaller.bam')
        if not os.path.exists(composite_bam_filename):  # TODO

            bam_file_list = []
            for dataset in basecaller_datasets["datasets"]:
                print os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])
                if os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
                    bam_file_list.append(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']))

            composite_bai_filepath = ""
            mark_duplicates = False
            blockprocessing.merge_bam_files(bam_file_list, composite_bam_filename, composite_bai_filepath, mark_duplicates, method)
    except:
        traceback.print_exc()
        printtime("ERROR: Generate merged %s on barcoded run failed" % composite_bam_filename)

    printtime("Finished basecaller barcode merging")


# 

def initalize_combined_readgroup(combined_readgroup):

    combined_readgroup['Q20_bases'] = 0
    combined_readgroup['total_bases'] = 0
    combined_readgroup['read_count'] = 0
    if "filtered" in combined_readgroup:
        combined_readgroup['filtered'] = True if 'barcode_sequence' in combined_readgroup else False
    else:
        combined_readgroup['filtered'] = False
    if "barcode_sequence" in combined_readgroup:
        combined_readgroup['barcode_bias'] = [-1]
        combined_readgroup['barcode_distance_hist'] = [0, 0, 0, 0, 0]
        combined_readgroup['barcode_errors_hist'] = [0, 0, 0]
        combined_readgroup['barcode_match_filtered'] = 0
        combined_readgroup['num_blocks_filtered'] = 0
        combined_readgroup['barcode_adapter_filtered'] = 0


def combine_read_groups(combined_readgroup, current_readgroup_dict, read_group):

    combined_readgroup['Q20_bases'] += current_readgroup_dict.get(read_group, {}).get("Q20_bases", 0)
    combined_readgroup['total_bases'] += current_readgroup_dict.get(read_group, {}).get("total_bases", 0)
    combined_readgroup['read_count'] += current_readgroup_dict.get(read_group, {}).get("read_count", 0)
    combined_readgroup['filtered'] &= current_readgroup_dict.get(read_group, {}).get("filtered", True)
    if current_readgroup_dict.get(read_group, {}).get("filtered", False):
        combined_readgroup['num_blocks_filtered'] += 1
    if "barcode_sequence" in combined_readgroup:
        combined_readgroup['barcode_adapter_filtered'] += current_readgroup_dict.get(read_group, {}).get("barcode_adapter_filtered", 0)
        combined_readgroup['barcode_match_filtered'] += current_readgroup_dict.get(read_group, {}).get("barcode_match_filtered", 0)
        error_hist = current_readgroup_dict.get(read_group, {}).get("barcode_errors_hist", [0, 0, 0])
        for hist_idx in range(len(error_hist)):
            combined_readgroup['barcode_errors_hist'][hist_idx] += error_hist[hist_idx]
        distance_hist = current_readgroup_dict.get(read_group, {}).get("barcode_distance_hist", [0, 0, 0, 0, 0])

        for hist_idx in range(len(distance_hist)):
            combined_readgroup['barcode_distance_hist'][hist_idx] += distance_hist[hist_idx]
        barcode_bias = current_readgroup_dict.get(read_group, {}).get("barcode_bias", [-1])
        if (combined_readgroup['barcode_bias'] == [-1]) & (barcode_bias != [-1]):
            combined_readgroup['barcode_bias'] = len(barcode_bias) * [0]
        if barcode_bias != [-1]:
            for bias_idx in range(len(barcode_bias)):
                combined_readgroup['barcode_bias'][bias_idx] += barcode_bias[bias_idx] * current_readgroup_dict.get(read_group, {}).get("read_count", 0)


def compute_read_group_averages(combined_readgroup):

    if "barcode_sequence" in combined_readgroup:
        if combined_readgroup['barcode_bias'] == [-1]:
            combined_readgroup['barcode_bias'] = [0]
        if combined_readgroup['read_count'] > 0:
            for bias_idx in range(len(combined_readgroup['barcode_bias'])):
                combined_readgroup['barcode_bias'][bias_idx] /= combined_readgroup['read_count']

# 


def merge_datasets_basecaller_json(dirs, BASECALLER_RESULTS):

    # 
    # Merge datasets_basecaller.json                       #
    # 

    block_datasets_json = []
    combined_datasets_json = {}

    for dir in dirs:
        current_datasets_path = os.path.join(dir, BASECALLER_RESULTS, 'datasets_basecaller.json')
        try:
            f = open(current_datasets_path, 'r')
            block_datasets_json.append(json.load(f))
            f.close()
        except:
            printtime("ERROR: skipped %s" % current_datasets_path)

    if (not block_datasets_json) or ('datasets' not in block_datasets_json[0]) or ('read_groups' not in block_datasets_json[0]):
        printtime("merge_basecaller_results: no block contained a valid datasets_basecaller.json, aborting")
        return

    combined_datasets_json = copy.deepcopy(block_datasets_json[0])

    # Merging dataset entries
    for dataset_idx in range(len(combined_datasets_json['datasets'])):
        combined_datasets_json['datasets'][dataset_idx]['read_count'] = 0
        for current_datasets_json in block_datasets_json:
            combined_datasets_json['datasets'][dataset_idx]['read_count'] += current_datasets_json['datasets'][dataset_idx].get("read_count", 0)

    # Merging read group entries
    for read_group in combined_datasets_json['read_groups'].iterkeys():
        initalize_combined_readgroup(combined_datasets_json['read_groups'][read_group])
        for current_datasets_json in block_datasets_json:
            combine_read_groups(combined_datasets_json['read_groups'][read_group], current_datasets_json['read_groups'], read_group)
        compute_read_group_averages(combined_datasets_json['read_groups'][read_group])

    # And merging information for control barcode datasets & read groups if available
    if "IonControl" in combined_datasets_json:
        # Merging dataset entries
        for dataset_idx in range(len(combined_datasets_json['IonControl']['datasets'])):
            combined_datasets_json['IonControl']['datasets'][dataset_idx]['read_count'] = 0
            for current_datasets_json in block_datasets_json:
                combined_datasets_json['IonControl']['datasets'][dataset_idx]['read_count'] += current_datasets_json['IonControl']['datasets'][dataset_idx].get("read_count", 0)
        # Merging read group entries
        for read_group in combined_datasets_json['IonControl']['read_groups'].iterkeys():
            initalize_combined_readgroup(combined_datasets_json['IonControl']['read_groups'][read_group])
            for current_datasets_json in block_datasets_json:
                combine_read_groups(combined_datasets_json['IonControl']['read_groups'][read_group], current_datasets_json['IonControl']['read_groups'], read_group)
            compute_read_group_averages(combined_datasets_json['IonControl']['read_groups'][read_group])

    # Barcode filters -------------------------------------------------------
    # Potential filters 1) frequency filter 2) minreads filter 3) error histogram filter
    # No use to attempt filtering here if filtering is done per block or json entries are missing
    if "barcode_filters" in combined_datasets_json and (combined_datasets_json['barcode_filters']['filter_postpone'] != 0):
        # Loop through read groups to compute combined filtering threshold
        max_reads = 0
        for read_group in combined_datasets_json['read_groups'].iterkeys():
            if "barcode_sequence" in combined_datasets_json['read_groups'][read_group]:
                max_reads = max(max_reads, combined_datasets_json['read_groups'][read_group]['read_count'])
        filter_threshold = combined_datasets_json['barcode_filters']['filter_minreads']
        filter_threshold = max(filter_threshold, math.floor(max_reads*combined_datasets_json['barcode_filters']['filter_frequency']))

        # Doing the actual filtering - exclude no-match read group
        for read_group in combined_datasets_json['read_groups']:
            filter_me = (combined_datasets_json['read_groups'][read_group]['sample'] == 'none')
            if ("barcode_sequence" in combined_datasets_json['read_groups'][read_group]) and filter_me:
                if combined_datasets_json['read_groups'][read_group]['read_count'] <= filter_threshold:
                    combined_datasets_json['read_groups'][read_group]['filtered'] = True
                if (not combined_datasets_json['read_groups'][read_group]['filtered']) and (combined_datasets_json['barcode_filters']['filter_errors_hist'] > 0):
                    av_errors = (combined_datasets_json['read_groups'][read_group]['barcode_errors_hist'][1] + 2*combined_datasets_json['read_groups'][read_group]['barcode_errors_hist'][2]) / combined_datasets_json['read_groups'][read_group]['read_count']
                    combined_datasets_json['read_groups'][read_group]['filtered'] = (av_errors > combined_datasets_json['barcode_filters']['filter_errors_hist'])
    # ----------------------------------------------------------------------

    try:
        f = open(os.path.join(BASECALLER_RESULTS, 'datasets_basecaller.json'), "w")
        json.dump(combined_datasets_json, f, indent=4)
        f.close()
    except:
        printtime("ERROR: Failed to write merged datasets_basecaller.json")
        traceback.print_exc()


def merge_basecaller_json(dirs, BASECALLER_RESULTS):

    printtime("Merging BaseCaller.json files")

    try:
        basecallerfiles = []
        for subdir in dirs:
            subdir = os.path.join(BASECALLER_RESULTS, subdir)
            printtime("DEBUG: %s:" % subdir)
            if isbadblock(subdir, "Merging BaseCaller.json files"):
                continue
            basecallerjson = os.path.join(subdir, 'BaseCaller.json')
            if os.path.exists(basecallerjson):
                basecallerfiles.append(subdir)
            else:
                printtime("ERROR: Merging BaseCaller.json files: skipped %s" % basecallerjson)

        mergeBaseCallerJson.merge(basecallerfiles, BASECALLER_RESULTS)
    except:
        traceback.print_exc()
        printtime("Merging BaseCaller.json files failed")

    printtime("Finished merging basecaller stats")


def generate_datasets_json(
        barcodeId,
        barcodeInfo,
        library,
        runID,
        notes,
        site_name,
        platform,
        instrumentName,
        chipType,
        datasets_json_path
        ):

    # TS-6135: ignore optional LB field, TODO: track library in database

    if not site_name:
        site_name = ""
    if not notes:
        notes = ""

    datasets = {
        "meta": {
            "format_name": "Dataset Map",
            "format_version": "1.0",
            "generated_by": "basecaller.py",
            "creation_date": dateutil.parser.parse(time.asctime()).isoformat()
        },
        "sequencing_center":  "%s/%s" % (''.join(ch for ch in site_name if ch.isalnum()), instrumentName),
        "datasets": [],
        "read_groups": {}
    }

    # get no barcode sample name and reference
    sample = barcodeInfo['no_barcode']['sample']
    reference = barcodeInfo['no_barcode']['referenceName']

    # Scenario 1. No barcodes.
    if len(barcodeInfo) == 1:
        datasets["datasets"].append({
            "dataset_name": sample,
            "file_prefix": "rawlib",
            "read_groups": [runID, ]
        })
        datasets["read_groups"][runID] = {
            "index": 0,
            "sample": sample,
            #"library"           : library,
            "reference": reference,
            "description": ''.join(ch for ch in notes if ch.isalnum() or ch == " "),
            "platform_unit":  "%s/%s" % (platform, chipType.replace('"', ""))
        }

    # Scenario 2. Barcodes present
    else:
        datasets["barcode_config"] = {}
        # TODO: not needed for calibration
        datasets["datasets"].append({
            "dataset_name": sample + "/No_barcode_match",
            "file_prefix": "nomatch_rawlib",
            "read_groups": [runID+".nomatch", ]
        })

        datasets["read_groups"][runID+".nomatch"] = {
            "index": 0,
            "sample": sample,
            #"library"           : library,
            #"reference"         : reference,
            "reference": "",
            "description": ''.join(ch for ch in notes if ch.isalnum() or ch == " "),
            "platform_unit":  "%s/%s/%s" % (platform, chipType.replace('"', ""), "nomatch")
        }
        datasets["barcode_config"]["barcode_id"] = barcodeId

        try:
            for barcode_name, barcode_info in sorted(barcodeInfo.iteritems()):

                if barcode_name == 'no_barcode':
                    continue

                datasets["read_groups"][runID+"."+barcode_name] = {
                    "barcode_name": barcode_name,
                    "barcode_sequence": barcode_info['sequence'],
                    "barcode_adapter": barcode_info['adapter'],
                    "index": barcode_info['index'],
                    "sample": barcode_info['sample'],
                    #"library"           : library,
                    "reference": barcode_info['referenceName'],
                    "description": ''.join(ch for ch in notes if ch.isalnum() or ch == " "),
                    "platform_unit":  "%s/%s/%s" % (platform, chipType.replace('"', ""), barcode_name)
                }

        except:
            print traceback.format_exc()
            datasets["read_groups"] = {}

        try:

            if 'calibration' in datasets_json_path:

                # create groups of barcodes with same references

                referencedict = defaultdict(list)

                for barcode_name, barcode_info in sorted(barcodeInfo.iteritems()):

                    if barcode_name == 'no_barcode':
                        continue

                    if barcode_info['referenceName']:
                        if barcode_info['calibrate']:
                            referencedict[barcode_info['referenceName']].append(barcode_name)
                        else:
                            referencedict['no_calibration'].append(barcode_name)
                    else:
                        # TODO: not needed for calibration
                        referencedict['no_reference'].append(barcode_name)

                for reference, bclist in referencedict.iteritems():
                    datasets["datasets"].append({
                        "dataset_name": reference,
                        "file_prefix": '%s_rawlib' % reference,
                        "read_groups": [runID+"."+barcode_name for barcode_name in bclist]
                    })

                print referencedict

            else:

                for barcode_name, barcode_info in sorted(barcodeInfo.iteritems()):

                    if barcode_name == 'no_barcode':
                        continue

                    datasets["datasets"].append({
                        "dataset_name": barcode_info['sample'] + "/" + barcode_name,
                        "file_prefix": '%s_rawlib' % barcode_name,
                        "read_groups": [runID+"."+barcode_name]
                    })

        except:
            print traceback.format_exc()
            datasets["datasets"] = []

    f = open(datasets_json_path, "w")
    json.dump(datasets, f, indent=4)
    f.close()
