#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from ion.utils.blockprocessing import printtime

import subprocess
import os
import sys
import traceback
import json
import re
import time
import dateutil
from shutil import move
import glob
import math
import shlex
import copy

from ion.utils import TFPipeline
from ion.utils.blockprocessing import isbadblock
from ion.reports import MaskMerge

from ion.reports import mergeBaseCallerJson
from ion.utils import blockprocessing
from ion.utils import ionstats

from ion.reports import wells_beadogram
from ion.utils import ionstats_plots

def basecaller_cmd(basecallerArgs,
                   SIGPROC_RESULTS,
                   libKey,
                   tfKey,
                   runID,
                   BASECALLER_RESULTS,
                   block_col_offset,
                   block_row_offset,
                   datasets_pipeline_path,
                   adapter,
                   barcodesplit_filter,
                   barcodesplit_filter_minreads):
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
    cmd += " --block-col-offset %d" % (block_col_offset)
    cmd += " --block-row-offset %d" % (block_row_offset)
    cmd += " --datasets=%s" % (datasets_pipeline_path)
    cmd += " --trim-adapter %s" % (adapter)
    cmd += " --barcode-filter %s" % barcodesplit_filter
    cmd += " --barcode-filter-minreads %s" % barcodesplit_filter_minreads

    return cmd


def basecalling(
      SIGPROC_RESULTS,
      basecallerArgs,
      libKey,
      tfKey,
      runID,
      floworder,
      reverse_primer_dict,
      BASECALLER_RESULTS,
      barcodeId,
      barcodeSamples,
      barcodesplit_filter,
      barcodesplit_filter_minreads,
      DIR_BC_FILES,
      barcodeList_path,
      barcodeMask_path,
      libraryName,
      sample,
      site_name,
      notes,
      start_time,
      chipType,
      expName,
      resultsName,
      pgmName
      ):

    if not os.path.exists(BASECALLER_RESULTS):
        os.mkdir(BASECALLER_RESULTS)


    ''' Step 1: Generate datasets_pipeline.json '''

    # New file, datasets_pipeline.json, contains the list of all active result files.
    # Tasks like post_basecalling, alignment, plugins, must process each specified file and merge results
    # Temporarily generated in BASECALLER_RESULTS directory from barcodeList.txt.
    # Eventually will replace barcodeList.txt altogether.
    
    datasets_pipeline_path = os.path.join(BASECALLER_RESULTS,"datasets_pipeline.json")
    datasets_basecaller_path = os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json")
    
    try:
        generate_datasets_json(
            barcodeId,
            barcodeSamples,
            barcodeList_path,
            datasets_pipeline_path,
            runID,
            libraryName,
            sample,
            site_name,
            notes,
            chipType,
            expName,
            resultsName,
            pgmName
        )
    except:
        printtime('ERROR: Generation of barcode_files.json unsuccessful')
        traceback.print_exc()



    ''' Step 2: Invoke BaseCaller '''

    try:
        [(x,y)] = re.findall('block_X(.*)_Y(.*)',os.getcwd())
        if x.isdigit():
            block_col_offset = int(x)
        else:
            block_col_offset = 0

        if y.isdigit():
            block_row_offset = int(y)
        else:
            block_row_offset = 0
    except:
        block_col_offset = 0
        block_row_offset = 0

    try:
        # 3' adapter details
        adapter = reverse_primer_dict['sequence']
        # TODO: provide barcode_filter via datasets.json

        cmd = basecaller_cmd(basecallerArgs,
                             SIGPROC_RESULTS,
                             libKey,
                             tfKey,
                             runID,
                             BASECALLER_RESULTS,
                             block_col_offset,
                             block_row_offset,
                             datasets_pipeline_path,
                             adapter,
                             barcodesplit_filter,
                             barcodesplit_filter_minreads)

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
            raise
        #ignore rest of operations
        if '--calibration-training' in basecallerArgs:
            printtime('training mode: ignore filtering')
            return
    except:
        printtime('ERROR: BaseCaller failed')
        traceback.print_exc()
        raise



    ''' Step 3: Apply barcode filtering: just move the filtered files to a different directory '''

    # This approach to barcode filtering needs rethinking. On proton, filtering should happen after block merge

    try:
        DIR_BC_FILTERED = os.path.join(BASECALLER_RESULTS,'bc_filtered')
        if not os.path.exists(DIR_BC_FILTERED):
            os.mkdir(DIR_BC_FILTERED)

        f = open(datasets_basecaller_path,'r')
        datasets_basecaller = json.load(f);
        f.close()
        
        for dataset in datasets_basecaller["datasets"]:
            
            keep_dataset = False
            for rg_name in dataset["read_groups"]:
                if not datasets_basecaller["read_groups"][rg_name].get('filtered',False):
                    keep_dataset = True
            if keep_dataset:
                continue
            
            filtered_file = os.path.join(BASECALLER_RESULTS, dataset["basecaller_bam"])
            printtime ("filter_barcodes: removing %s" % filtered_file)
            try:
                move(filtered_file, DIR_BC_FILTERED)
            except:
                traceback.print_exc()

    except:
        printtime ("Barcode filtering failed")
        traceback.print_exc()
    
    
    try:
        wells_beadogram.generate_wells_beadogram(BASECALLER_RESULTS, SIGPROC_RESULTS)
    except:
        printtime ("Wells beadogram generation failed")
        traceback.print_exc()

    
    printtime("Finished basecaller processing")



def post_basecalling(BASECALLER_RESULTS,expName,resultsName,flows):

    datasets_basecaller_path = os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json")
    if not os.path.exists(datasets_basecaller_path):
        printtime("ERROR: %s does not exist" % datasets_basecaller_path)
        raise Exception("ERROR: %s does not exist" % datasets_basecaller_path)
    
    datasets_basecaller = {}
    try:
        f = open(datasets_basecaller_path,'r')
        datasets_basecaller = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % datasets_basecaller_path)
        raise Exception("ERROR: problem parsing %s" % datasets_basecaller_path)

    try:
        graph_max_x = int(50 * math.ceil(0.014 * int(flows)))
    except:
        graph_max_x = 400

    quality_file_list = []
    for dataset in datasets_basecaller["datasets"]:
        if not os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
            continue
                
        # Call ionstats utility to generate alignment-independent metrics for current unmapped BAM
        ionstats.generate_ionstats_basecaller(
                os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']),
                os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'),
                graph_max_x)
        
        # Plot read length sparkline
        ionstats_plots.read_length_sparkline(
                os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'),
                os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.sparkline.png'),
                graph_max_x)
        
        quality_file_list.append(os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'))
        
    # Merge ionstats_basecaller files from individual barcodes/dataset
    ionstats_basecaller_path = os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json')
    ionstats.reduce_stats(quality_file_list, ionstats_basecaller_path)

    # Read the merged ionstats in order to set a consistent read length histogram width for more detailed images
    try:
        with open(ionstats_basecaller_path) as f:
            ionstats_basecaller = json.load(f)
    except Exception:
        printtime("ERROR: problem parsing %s" % ionstats_basecaller_path)
        raise Exception("ERROR: problem parsing %s" % ionstats_basecaller_path)
    full_graph_x = max(50, ionstats_basecaller["full"]["max_read_length"])
    if full_graph_x % 50:
        full_graph_x += 50 - full_graph_x % 50

    # Generate legacy stats file: quality.summary
    ionstats.generate_legacy_basecaller_files(
            os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json'),
            os.path.join(BASECALLER_RESULTS,''))

    # Plot classic read length histogram
    ionstats_plots.old_read_length_histogram(
            os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json'),
            os.path.join(BASECALLER_RESULTS,'readLenHisto.png'),
            full_graph_x)
    
    # Plot new read length histogram
    ionstats_plots.read_length_histogram(
            os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json'),
            os.path.join(BASECALLER_RESULTS,'readLenHisto2.png'),
            graph_max_x)

    # Plot quality value histogram
    ionstats_plots.quality_histogram(
        os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json'),
        os.path.join(BASECALLER_RESULTS,'quality_histogram.png'))

    # Plot higher detail barcode specific histograms
    for dataset in datasets_basecaller["datasets"]:
        if not os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
            continue
        ionstats_plots.old_read_length_histogram(
                os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'),
                os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.read_len_histogram.png'),
                full_graph_x)

    printtime("Finished basecaller post processing")


def merge_barcoded_basecaller_bams(BASECALLER_RESULTS):

    datasets_basecaller_path = os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json")

    if not os.path.exists(datasets_basecaller_path):
        printtime("ERROR: %s does not exist" % datasets_basecaller_path)
        raise Exception("ERROR: %s does not exist" % datasets_basecaller_path)
    
    datasets_basecaller = {}
    try:
        f = open(datasets_basecaller_path,'r')
        datasets_basecaller = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % datasets_basecaller_path)
        raise Exception("ERROR: problem parsing %s" % datasets_basecaller_path)

    try:
        composite_bam_filename = os.path.join(BASECALLER_RESULTS,'rawlib.basecaller.bam')
        if not os.path.exists(composite_bam_filename):

            bam_file_list = []
            for dataset in datasets_basecaller["datasets"]:
                print os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])
                if os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
                    bam_file_list.append(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']))

            blockprocessing.merge_bam_files(bam_file_list,composite_bam_filename,composite_bam_filename+'.bai',False)
    except:
        traceback.print_exc()
        printtime("ERROR: Generate merged rawlib.basecaller.bam on barcoded runs failed")

    printtime("Finished basecaller barcode merging")


def tf_processing(
      tf_basecaller_bam_path,
      tfKey,
      floworder,
      BASECALLER_RESULTS,
      analysis_dir):


    ##################################################
    #generate TF Metrics                             #
    ##################################################

    printtime("Calling TFPipeline.processBlock")
    TFPipeline.processBlock(tf_basecaller_bam_path, BASECALLER_RESULTS, tfKey, floworder, analysis_dir)
    printtime("Completed TFPipeline.processBlock")

    printtime("Finished tf processing")



def merge_basecaller_stats(dirs, BASECALLER_RESULTS, SIGPROC_RESULTS, flows, floworder):

    ########################################################
    # Merge datasets_basecaller.json                       #
    ########################################################
    
    block_datasets_json = []
    combined_datasets_json = {}
    
    for dir in dirs:
        current_datasets_path = os.path.join(dir,BASECALLER_RESULTS,'datasets_basecaller.json')
        try:
            f = open(current_datasets_path,'r')
            block_datasets_json.append(json.load(f))
            f.close()
        except:
            printtime("ERROR: skipped %s" % current_datasets_path)
    
    if (not block_datasets_json) or ('datasets' not in block_datasets_json[0]) or ('read_groups' not in block_datasets_json[0]):
        printtime("merge_basecaller_results: no block contained a valid datasets_basecaller.json, aborting")
        return

    combined_datasets_json = copy.deepcopy(block_datasets_json[0])
    
    for dataset_idx in range(len(combined_datasets_json['datasets'])):
        combined_datasets_json['datasets'][dataset_idx]['read_count'] = 0
        for current_datasets_json in block_datasets_json:
            combined_datasets_json['datasets'][dataset_idx]['read_count'] += current_datasets_json['datasets'][dataset_idx].get("read_count",0)
    
    for read_group in combined_datasets_json['read_groups'].iterkeys():
        combined_datasets_json['read_groups'][read_group]['Q20_bases'] = 0;
        combined_datasets_json['read_groups'][read_group]['total_bases'] = 0;
        combined_datasets_json['read_groups'][read_group]['read_count'] = 0;
        combined_datasets_json['read_groups'][read_group]['filtered'] = True if 'nomatch' not in read_group else False
        for current_datasets_json in block_datasets_json:
            combined_datasets_json['read_groups'][read_group]['Q20_bases'] += current_datasets_json['read_groups'].get(read_group,{}).get("Q20_bases",0)
            combined_datasets_json['read_groups'][read_group]['total_bases'] += current_datasets_json['read_groups'].get(read_group,{}).get("total_bases",0)
            combined_datasets_json['read_groups'][read_group]['read_count'] += current_datasets_json['read_groups'].get(read_group,{}).get("read_count",0)
            combined_datasets_json['read_groups'][read_group]['filtered'] &= current_datasets_json['read_groups'].get(read_group,{}).get("filtered",True)
    
    try:
        f = open(os.path.join(BASECALLER_RESULTS,'datasets_basecaller.json'),"w")
        json.dump(combined_datasets_json, f, indent=4)
        f.close()
    except:
        printtime("ERROR; Failed to write merged datasets_basecaller.json")
        traceback.print_exc()



    ########################################################
    # Merge ionstats_basecaller.json:                      #
    # First across blocks, then across barcodes            #
    ########################################################

    try:
        composite_filename_list = []
        for dataset in combined_datasets_json["datasets"]:
            composite_filename = os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json')
            barcode_filename_list = [os.path.join(dir,BASECALLER_RESULTS,dataset['file_prefix']+'.ionstats_basecaller.json') for dir in dirs]
            barcode_filename_list = [filename for filename in barcode_filename_list if os.path.exists(filename)]
            ionstats.reduce_stats(barcode_filename_list,composite_filename)
            if os.path.exists(composite_filename):
                composite_filename_list.append(composite_filename)

        ionstats.reduce_stats(composite_filename_list,os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json'))
        ionstats.generate_legacy_basecaller_files(
                os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json'),
                os.path.join(BASECALLER_RESULTS,''))
    except:
        printtime("ERROR: Failed to merge ionstats_basecaller.json")
        traceback.print_exc()



    ########################################################
    # write composite return code                          #
    ########################################################

    try:
        if len(dirs)==96:
            composite_return_code=96
            for subdir in dirs:

                blockstatus_return_code_file = os.path.join(subdir,"blockstatus.txt")
                if os.path.exists(blockstatus_return_code_file):

                    with open(blockstatus_return_code_file, 'r') as f:
                        text = f.read()
                        if 'Basecaller=0' in text:
                            composite_return_code-=1

            composite_return_code_file = os.path.join(BASECALLER_RESULTS,"composite_return_code.txt")
            if not os.path.exists(composite_return_code_file):
                printtime("DEBUG: create %s" % composite_return_code_file)
                os.umask(0002)
                f = open(composite_return_code_file, 'a')
                f.write(str(composite_return_code))
                f.close()
            else:
                printtime("DEBUG: skip generation of %s" % composite_return_code_file)
    except:
        traceback.print_exc()


    ##################################################
    #generate TF Metrics                             #
    #look for both keys and append same file         #
    ##################################################

    printtime("Merging TFMapper metrics and generating TF plots")
    try:
        TFPipeline.mergeBlocks(BASECALLER_RESULTS,dirs,floworder)
    except:
        printtime("ERROR: Merging TFMapper metrics failed")

    
    ###############################################
    # Merge BaseCaller.json files                 #
    ###############################################
    printtime("Merging BaseCaller.json files")

    try:
        basecallerfiles = []
        for subdir in dirs:
            subdir = os.path.join(BASECALLER_RESULTS,subdir)
            printtime("DEBUG: %s:" % subdir)
            if isbadblock(subdir, "Merging BaseCaller.json files"):
                continue
            basecallerjson = os.path.join(subdir,'BaseCaller.json')
            if os.path.exists(basecallerjson):
                basecallerfiles.append(subdir)
            else:
                printtime("ERROR: Merging BaseCaller.json files: skipped %s" % basecallerjson)

        mergeBaseCallerJson.merge(basecallerfiles,BASECALLER_RESULTS)
    except:
        printtime("Merging BaseCaller.json files failed")


    ###############################################
    # Generate composite plots
    ###############################################

    printtime("Build composite basecaller graphs")
    try:
        graph_max_x = int(50 * math.ceil(0.014 * int(flows)))
    except:
        graph_max_x = 400

    # Plot read length sparkline
    for dataset in combined_datasets_json["datasets"]:
        ionstats_plots.read_length_sparkline(
                os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'),
                os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.sparkline.png'),
                graph_max_x)

    # Plot classic read length histogram
    ionstats_plots.old_read_length_histogram(
            os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json'),
            os.path.join(BASECALLER_RESULTS,'readLenHisto.png'),
            graph_max_x)
    
    # Plot new read length histogram
    ionstats_plots.read_length_histogram(
            os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json'),
            os.path.join(BASECALLER_RESULTS,'readLenHisto2.png'),
            graph_max_x)

    # Plot quality value histogram
    ionstats_plots.quality_histogram(
        os.path.join(BASECALLER_RESULTS,'ionstats_basecaller.json'),
        os.path.join(BASECALLER_RESULTS,'quality_histogram.png'))
    

    try:
        wells_beadogram.generate_wells_beadogram(BASECALLER_RESULTS, SIGPROC_RESULTS)
    except:
        printtime ("ERROR: Wells beadogram generation failed")
        traceback.print_exc()

    printtime("Finished merging basecaller stats")

def merge_basecaller_bam(dirs, BASECALLER_RESULTS):

    datasets_basecaller = {}
    try:
        f = open(os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"),'r')
        datasets_basecaller = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"))
        traceback.print_exc()
        return

    # Iterate over datasets. Could be one for non-barcoded runs or multiple for barcoded runs
    
    for dataset in datasets_basecaller['datasets']:
        if 'basecaller_bam' not in dataset:
            continue
        
        ###############################################
        # Merge Per-barcode Unmapped BAMs             #
        ###############################################
        
        try:
            block_bam_list = [os.path.join(dir,BASECALLER_RESULTS, dataset['basecaller_bam']) for dir in dirs]
            block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
            composite_bam_filename = os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])
            if block_bam_list:
                blockprocessing.merge_bam_files(block_bam_list,composite_bam_filename,composite_bam_filename+'.bai',False)    
        except:
            printtime("ERROR: merging %s unsuccessful" % dataset['basecaller_bam'])

    ## Note! on barcoded runs, barcode files are NOT subsequently merged into one multi-barcode BAM. 

    printtime("Finished merging basecaller BAM files")



def generate_datasets_json(
        barcodeId,
        barcodeSamples,
        barcodeList_path,
        datasets_json_path,
        runID,
        libraryName,
        sample,
        site_name,
        notes,
        chipType,
        expName,
        resultsName,
        pgmName
        ):

    if not sample or barcodeId:
        sample = "None"
    if not libraryName:
        libraryName = ""
    if not site_name:
        site_name = ""
    if not notes:
        notes = ""
    
    datasets = {
        "meta" : {
            "format_name"       : "Dataset Map",
            "format_version"    : "1.0",
            "generated_by"      : "basecaller.py",
            "creation_date"     : dateutil.parser.parse(time.asctime()).isoformat()
        },
        "sequencing_center" :  "%s/%s" % (''.join(ch for ch in site_name if ch.isalnum()), pgmName),
        "datasets" : [],
        "read_groups" : {}
    }
    
    # Scenario 1. Barcodes present
    if barcodeId:
        try:
            datasets["barcode_config"] = {}
            f = open(barcodeList_path, 'r')
            for line in f:
                if line.startswith('score_mode'):
                    datasets["barcode_config"]["score_mode"] = int(line.lstrip("score_mode "))
                if line.startswith('score_cutoff'):
                    datasets["barcode_config"]["score_cutoff"] = float(line.lstrip("score_cutoff "))
                if line.startswith('barcode'):
                    record = line.lstrip("barcode ").rstrip().split(",")
                    # use any per barcode sample names entered during Planning
                    bcsample = [k for k,v in barcodeSamples.items() if record[1] in v.get('barcodes',[])]
                    if len(bcsample) == 1:
                        bcsample = bcsample[0]
                    else:
                        bcsample = 'None'
                        
                    datasets["datasets"].append({
                        "dataset_name"      : bcsample + "/" + record[1],
                        "file_prefix"       : '%s_rawlib' % record[1],
                        "read_groups"       : [runID+"."+record[1],]
                    })
                    datasets["read_groups"][runID+"."+record[1]] = {
                        "barcode_name"      : record[1],
                        "barcode_sequence"  : record[2],
                        "barcode_adapter"   : record[3],
                        "index"             : int(record[0]),
                        "sample"            : bcsample,
                        "library"           : libraryName+"/"+record[1],
                        "description"       : ''.join(ch for ch in notes if ch.isalnum() or ch == " "),
                        "platform_unit"     :  "PGM/%s/%s" % (chipType.replace('"',""),record[1])
                    }
            f.close()

            datasets["datasets"].append({
                "dataset_name"      : sample + "/No_barcode_match",
                "file_prefix"       : "nomatch_rawlib",
                "read_groups"       : [runID+".nomatch",]
            })
            datasets["read_groups"][runID+".nomatch"] = {
                "index"             : 0,
                "sample"            : sample,
                "library"           : libraryName+"/No_barcode_match",
                "description"       : ''.join(ch for ch in notes if ch.isalnum() or ch == " "),
                "platform_unit"     :  "PGM/%s/%s" % (chipType.replace('"',""),"nomatch")
            }
            datasets["barcode_config"]["barcode_id"] = barcodeId
        
        except:
            print traceback.format_exc()
            datasets["read_groups"] = {}
            datasets["datasets"] = []
    

    # Scenario 2. No barcodes.
    if not datasets["datasets"]:
        datasets["datasets"].append({
            "dataset_name"      : sample,
            "file_prefix"       : "rawlib",
            "read_groups"       : [runID,]
        })
        datasets["read_groups"][runID] = {
            "index"             : 0,
            "sample"            : sample,
            "library"           : libraryName,
            "description"       : ''.join(ch for ch in notes if ch.isalnum() or ch == " "),
            "platform_unit"     :  "PGM/%s" % (chipType.replace('"',""))
        }
        

    f = open(datasets_json_path,"w")
    json.dump(datasets, f, indent=4)
    f.close()
    


''' Marcin's temporary runner for testing basecalling and post_basecalling'''
if __name__=="__main__":
    
    env = {
        'SIGPROC_RESULTS'       : '../sigproc_results',
        'basecallerArgs'        : '/home/msikora/Documents/BaseCaller',
        'libraryKey'            : 'TCAG',
        'tfKey'                 : 'ATCG',
        'runID'                 : 'ABCDE',
        'flowOrder'             : 'TACGTACGTCTGAGCATCGATCGATGTACAGC',
        'reverse_primer_dict'   : {'adapter_cutoff':16,'sequence':'ATCACCGACTGCCCATAGAGAGGCTGAGAC','qual_window':30,'qual_cutoff':9},
        'BASECALLER_RESULTS'    : 'basecaller_results',
        'barcodeId'             : 'IonExpress',
        #'barcodeId'             : '',
        'barcodesplit_filter'   : 0.01,
        'barcodesplit_filter_minreads'  : 0,
        'DIR_BC_FILES'          : 'basecaller_results/bc_files',
        'libraryName'           : 'hg19',
        'sample'                : 'My-sample',
        'site_name'             : 'My-site',
        'notes'                 : 'My-notes',
        'start_time'            : time.asctime(),
        'align_full'            : False,
        'flows'                 : 260,
        'aligner_opts_extra'    : '',
        'mark_duplicates'       : False,
        'ALIGNMENT_RESULTS'     : './',
        'sam_parsed'            : False,
        'chipType'              : '316B',
        'expName'               : 'My-experiment',
        'resultsName'           : 'My-results',
        'pgmName'               : 'B19'
    }

    
    basecalling(
        env['SIGPROC_RESULTS'],
        env['basecallerArgs'],
        env['libraryKey'],
        env['tfKey'],
        env['runID'],
        env['flowOrder'],
        env['reverse_primer_dict'],
        env['BASECALLER_RESULTS'],
        env['barcodeId'],
        env.get('barcodeSamples',''),
        env['barcodesplit_filter'],
        env['barcodesplit_filter_minreads'],
        env['DIR_BC_FILES'],
        os.path.join("barcodeList.txt"),
        os.path.join(env['BASECALLER_RESULTS'], "barcodeMask.bin"),
        env['libraryName'],
        env['sample'],
        env['site_name'],
        env['notes'],
        env['start_time'],
        env['chipType'],
        env['expName'],
        env['resultsName'],
        env['pgmName']
    )
    
    post_basecalling(env['BASECALLER_RESULTS'],env['expName'],env['resultsName'],env['flows'])

    
    tf_processing(
        os.path.join(env['BASECALLER_RESULTS'], "rawtf.basecaller.bam"),
        env['tfKey'],
        env['flowOrder'],
        env['BASECALLER_RESULTS'],
        '.')

    
    if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed')):
        post_basecalling(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed'),env['expName'],env['resultsName'],env['flows'])
    
    if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed')):
        post_basecalling(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed'),env['expName'],env['resultsName'],env['flows'])
    

    
    #from ion.utils import alignment
    import alignment
    bidirectional = False
    

    if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed")):
        alignment.alignment_unmapped_bam(
            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed"),
            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed"),
            env['align_full'],
            env['libraryName'],
            env['flows'],
            env['aligner_opts_extra'],
            env['mark_duplicates'],
            bidirectional,
            env['sam_parsed'])

    if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed")):
        alignment.alignment_unmapped_bam(
            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed"),
            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed"),
            env['align_full'],
            env['libraryName'],
            env['flows'],
            env['aligner_opts_extra'],
            env['mark_duplicates'],
            bidirectional,
            env['sam_parsed'])

    alignment.alignment_unmapped_bam(
        env['BASECALLER_RESULTS'],
        env['ALIGNMENT_RESULTS'],
        env['align_full'],
        env['libraryName'],
        env['flows'],
        env['aligner_opts_extra'],
        env['mark_duplicates'],
        bidirectional,
        env['sam_parsed'])
