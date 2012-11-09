#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from ion.utils.blockprocessing import printtime
from ion.utils.blockprocessing import MyConfigParser

import ConfigParser
import StringIO
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

# for barcode handling, TODO
from ion.utils.aggregate_alignment import *

from ion.reports import trimmedReadLenHisto

from ion.utils import TFPipeline
from ion.utils.blockprocessing import isbadblock
from ion.reports import MaskMerge, \
    StatsMerge, plotRawRegions, plotKey
from ion.reports import mergeBaseCallerJson
from ion.utils import blockprocessing

from ion.reports import quality_histogram
from ion.reports import wells_beadogram
from ion.reports import read_length_sparkline

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
      barcodesplit_filter,
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

        # 3' adapter details
        qual_cutoff = reverse_primer_dict['qual_cutoff']
        qual_window = reverse_primer_dict['qual_window']
        adapter_cutoff = reverse_primer_dict['adapter_cutoff']
        adapter = reverse_primer_dict['sequence']
        cmd += " --flow-order %s" % (floworder)
        cmd += " --trim-qual-cutoff %s" % (qual_cutoff)
        cmd += " --trim-qual-window-size %s" % (qual_window)
        cmd += " --trim-adapter-cutoff %s" % (adapter_cutoff)
        cmd += " --trim-adapter %s" % (adapter)
        # TODO: provide via datasets.json
        if barcodesplit_filter:
            cmd += " --barcode-filter %s" % barcodesplit_filter

        cmd += " >> %s 2>&1" % os.path.join(BASECALLER_RESULTS, 'basecaller.log')

        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        blockprocessing.add_status("BaseCaller", ret)
    except:
        printtime('ERROR: BaseCaller failed')
        traceback.print_exc()



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
                if datasets_basecaller["read_groups"][rg_name].get('filtered',False) == False:
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
        quality_histogram.generate_quality_histogram(
            os.path.join(BASECALLER_RESULTS,'BaseCaller.json'),
            os.path.join(BASECALLER_RESULTS,'quality_histogram.png'))
    except:
        printtime ("Quality histogram generation failed")
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
        open('badblock.txt', 'w').close()
        return
    
    datasets_basecaller = {}
    try:
        f = open(datasets_basecaller_path,'r')
        datasets_basecaller = json.load(f);
        f.close()
    except:
        printtime("ERROR: problem parsing %s" % datasets_basecaller_path)
        traceback.print_exc()
        open('badblock.txt', 'w').close()
        return

    try:
        graph_max_x = int(50 * math.ceil(0.014 * int(flows)))
    except:
        graph_max_x = 400

    input_prefix_list = []
    for dataset in datasets_basecaller["datasets"]:
        if not os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
            continue
        
        dataset['sff'] = dataset['file_prefix']+'.sff'
        dataset['fastq'] = dataset['file_prefix']+'.fastq'

        try:
            com = "bam2sff"
            com += " -o %s"  % os.path.join(BASECALLER_RESULTS, dataset['sff'])
            com += " %s"     % os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])
            printtime("DEBUG: Calling '%s'" % com)
            subprocess.call(com,shell=True)
        except:
            printtime('Failed bam2sff')

        try:
            com = "SFFSummary"
            com += " -o %s"         % os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.quality.summary')
            com += " --sff-file %s" % os.path.join(BASECALLER_RESULTS, dataset['sff'])
            com += " --read-length 50,100,150"
            com += " --min-length 0,0,0"
            com += " --qual 0,17,20"
            com += " -d %s"         % os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.readLen.txt')
            printtime("DEBUG: Calling '%s'" % com)
            ret = subprocess.call(com,shell=True)
            input_prefix_list.append(os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.'))
            #blockprocessing.add_status("SFFSummary", ret)
        except:
            printtime('Failed SFFSummary')

        # In the future, make fastq from bam
        # Tried picard, but its suspiciously slow. Need to test bamtools
        # java -Xmx8g -jar /opt/picard/picard-tools-current/SamToFastq.jar I=IonXpress_033_rawlib.basecaller.bam F=result.fastq
            
        try:
            com = "SFFRead"
            com += " -q %s"         % os.path.join(BASECALLER_RESULTS, dataset['fastq'])
            com += " %s"            % os.path.join(BASECALLER_RESULTS, dataset['sff'])
    
            printtime("DEBUG: Calling '%s'" % com)
            ret = subprocess.call(com,shell=True)
            #blockprocessing.add_status("SFFRead", ret)
        except:
            printtime('Failed SFFRead')
        
        # Creating links to legacy names

        if dataset.has_key('legacy_prefix'):
            
            link_src = [
                os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']),
                os.path.join(BASECALLER_RESULTS, dataset['sff']),
                os.path.join(BASECALLER_RESULTS, dataset['fastq'])]
            link_dst = [
                os.path.join(BASECALLER_RESULTS, dataset['legacy_prefix']+'.basecaller.bam'),
                os.path.join(BASECALLER_RESULTS, dataset['legacy_prefix']+'.sff'),
                os.path.join(BASECALLER_RESULTS, dataset['legacy_prefix']+'.fastq')]
            for (src,dst) in zip(link_src,link_dst):
                try:
                    os.symlink(os.path.relpath(src,os.path.dirname(dst)),dst)
                except:
                    printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))
                    
        # Plot read length sparkline
        
        try:
            readlen_path = os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.readLen.txt')
            sparkline_path = os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.sparkline.png')
            read_length_sparkline.read_length_sparkline(readlen_path,sparkline_path,graph_max_x)
        except:
            printtime("Failed to create %s" % sparkline_path)

    try:
        merge_quality_summary(input_prefix_list, BASECALLER_RESULTS+'/')
    except:
        traceback.print_exc()

    printtime("make the read length histogram")
    try:
        filepath_readLenHistogram = os.path.join(BASECALLER_RESULTS,'readLenHisto.png')
        filepath_readlentxt = os.path.join(BASECALLER_RESULTS,'readLen.txt')
        trimmedReadLenHisto.trimmedReadLenHisto(filepath_readlentxt,filepath_readLenHistogram)
        filepath_readLenHistogram2 = os.path.join(BASECALLER_RESULTS,'readLenHisto2.png')
        read_length_sparkline.read_length_histogram(filepath_readlentxt,filepath_readLenHistogram2,graph_max_x)
        
    except:
        printtime("Failed to create %s" % filepath_readLenHistogram)


    # Special legacy post-processing.
    # Generate merged rawlib.basecaller.bam and rawlib.sff on barcoded runs

    composite_bam_filename = os.path.join(BASECALLER_RESULTS,'%s_%s.basecaller.bam'%(expName,resultsName))
    composite_bam_legacy_name = os.path.join(BASECALLER_RESULTS,'rawlib.basecaller.bam')
    if not os.path.exists(composite_bam_filename):

        bam_file_list = []
        for dataset in datasets_basecaller["datasets"]:
            if os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
                bam_file_list.append(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']))

        blockprocessing.merge_bam_files(bam_file_list,composite_bam_filename,composite_bam_filename+'.bai',False)
        
        composite_sff_filename = os.path.join(BASECALLER_RESULTS,'%s_%s.sff'%(expName,resultsName))
        composite_sff_legacy_name = os.path.join(BASECALLER_RESULTS,'rawlib.sff')
        composite_fastq_filename = os.path.join(BASECALLER_RESULTS,'%s_%s.fastq'%(expName,resultsName))
        composite_fastq_legacy_name = os.path.join(BASECALLER_RESULTS,'rawlib.fastq')

        try:
            com = "bam2sff"
            com += " -o %s"  % composite_sff_filename
            com += " %s"     % composite_bam_filename
            printtime("DEBUG: Calling '%s'" % com)
            subprocess.call(com,shell=True)
        except:
            printtime('Failed bam2sff')

        try:
            com = "SFFRead"
            com += " -q %s"         % composite_fastq_filename
            com += " %s"            % composite_sff_filename
            printtime("DEBUG: Calling '%s'" % com)
            ret = subprocess.call(com,shell=True)
        except:
            printtime('Failed SFFRead')

        link_src = [
            composite_bam_filename,
            composite_sff_filename,
            composite_fastq_filename]
        link_dst = [
            composite_bam_legacy_name,
            composite_sff_legacy_name,
            composite_fastq_legacy_name]
        for (src,dst) in zip(link_src,link_dst):
            try:
                os.symlink(os.path.relpath(src,os.path.dirname(dst)),dst)
            except:
                printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))

    printtime("Finished basecaller post processing")




def tf_processing(
      SIGPROC_RESULTS,
      tf_basecaller_bam_path,
      libKey,
      tfKey,
      floworder,
      BASECALLER_RESULTS,
      analysis_dir):


    ##################################################
    #generate TF Metrics                             #
    ##################################################

    if os.path.exists(os.path.join(BASECALLER_RESULTS, 'rawtf.sff')):
        os.rename(os.path.join(BASECALLER_RESULTS, 'rawtf.sff'), os.path.join(BASECALLER_RESULTS, 'rawtf.old.sff'))

    try:
        com = "bam2sff"
        com += " -o %s"  % os.path.join(BASECALLER_RESULTS, 'rawtf.sff')
        com += " %s"     % os.path.join(BASECALLER_RESULTS, 'rawtf.basecaller.bam')
        printtime("DEBUG: Calling '%s'" % com)
        subprocess.call(com,shell=True)
    except:
        printtime('Failed bam2sff')


    printtime("Calling TFPipeline.processBlock")
    TFPipeline.processBlock(tf_basecaller_bam_path, BASECALLER_RESULTS, tfKey, floworder, analysis_dir)
    printtime("Completed TFPipeline.processBlock")



    ########################################################
    #Generate Raw Data Traces for lib and TF keys          #
    ########################################################
    printtime("Generate Raw Data Traces for lib and TF keys(iontrace_Test_Fragment.png, iontrace_Library.png)")

    tfRawPath = os.path.join(SIGPROC_RESULTS, 'avgNukeTrace_%s.txt' % tfKey)
    libRawPath = os.path.join(SIGPROC_RESULTS, 'avgNukeTrace_%s.txt' % libKey)
    peakOut = 'raw_peak_signal'

    if os.path.exists(tfRawPath):
        try:
            kp = plotKey.KeyPlot(tfKey, floworder, 'Test Fragment')
            kp.parse(tfRawPath)
            kp.dump_max(os.path.join('.',peakOut))
            kp.plot()
        except:
            printtime("TF key graph didn't render")
            traceback.print_exc()
    else:
        printtime("ERROR: %s is missing" % tfRawPath)

    if os.path.exists(libRawPath):
        try:
            kp = plotKey.KeyPlot(libKey, floworder, 'Library')
            kp.parse(libRawPath)
            kp.dump_max(os.path.join('.',peakOut))
            kp.plot()
        except:
            printtime("Lib key graph didn't render")
            traceback.print_exc()
    else:
        printtime("ERROR: %s is missing" % libRawPath)


    ########################################################
    # Make per region key incorporation traces             #
    ########################################################
    printtime("Make per region key incorporation traces")
    perRegionTF = "averagedKeyTraces_TF.txt"
    perRegionLib = "averagedKeyTraces_Lib.txt"
    if os.path.exists(perRegionTF):
        pr = plotRawRegions.PerRegionKey(tfKey, floworder,'TFTracePerRegion.png')
        pr.parse(perRegionTF)
        pr.plot()

    if os.path.exists(perRegionLib):
        pr = plotRawRegions.PerRegionKey(libKey, floworder,'LibTracePerRegion.png')
        pr.parse(perRegionLib)
        pr.plot()

    printtime("Finished tf processing")



''' Merge quality.summary and also readLen.txt '''

def merge_quality_summary(input_prefix_list, output_prefix):

    printtime("Merging quality.summary and readLen.txt files: %s..." % output_prefix)

    summary_file_list = [input_prefix+'quality.summary' for input_prefix in input_prefix_list]
    readlen_file_list = [input_prefix+'readLen.txt' for input_prefix in input_prefix_list]

    try:
        config_out = ConfigParser.RawConfigParser()
        config_out.optionxform = str # don't convert to lowercase
        config_out.add_section('global')

        numberkeys = ['Number of 50BP Reads',
                      'Number of 100BP Reads',
                      'Number of 150BP Reads',
                      'Number of Reads at Q0',
                      'Number of Bases at Q0',
                      'Number of 50BP Reads at Q0',
                      'Number of 100BP Reads at Q0',
                      'Number of 150BP Reads at Q0',
                      'Number of Reads at Q17',
                      'Number of Bases at Q17',
                      'Number of 50BP Reads at Q17',
                      'Number of 150BP Reads at Q17',
                      'Number of 100BP Reads at Q17',
                      'Number of Reads at Q20',
                      'Number of Bases at Q20',
                      'Number of 50BP Reads at Q20',
                      'Number of 100BP Reads at Q20',
                      'Number of 150BP Reads at Q20']

        maxkeys = ['Max Read Length at Q0',
                   'Max Read Length at Q17',
                   'Max Read Length at Q20']

        meankeys = ['System SNR']
#                    'Mean Read Length at Q0',
#                    'Mean Read Length at Q17',
#                    'Mean Read Length at Q20']

        config_in = MyConfigParser()
        config_in.optionxform = str # don't convert to lowercase

        # initialize
        for key in numberkeys + maxkeys + meankeys:
            config_out.set('global',key,0)

        adict={}
        for summary_file in summary_file_list:
            try:
                if os.path.exists(summary_file):
                    printtime("INFO: process %s" % summary_file)
                    # make sure all values are there
                    try:
                        config_in.read(summary_file)
                        for key in numberkeys + maxkeys + meankeys:
                            adict[key] = config_in.get('global',key)
                    except:
                        printtime("INFO: missing key(s) in %s" % summary_file)
                        traceback.print_exc()
                        continue

                    for key in numberkeys:
                        value_out = config_out.getint('global', key) + int(adict[key])
                        config_out.set('global', key, value_out)

                    for key in maxkeys:
                        value_out = max (config_out.getint('global', key), int(adict[key]))
                        config_out.set('global', key, value_out)

                    for key in meankeys:
                        if len(input_prefix_list) > 0:
                            value_out = float(config_out.get('global', key)) + float(adict[key])/len(input_prefix_list)
                        else:
                            value_out = 0
                        config_out.set('global', key, value_out)
                else:
                    printtime("ERROR: %s doesn't exist" % summary_file)
            except:
                traceback.print_exc()
                continue

        # Metrics that need more careful merging, see meankeys which is returning floats 
        for quality in ['Q0','Q17','Q20']:
            total_reads = config_out.getint('global','Number of Reads at %s' % quality)
            total_bases = config_out.getint('global','Number of Bases at %s' % quality)
            if total_reads == 0:
                value_out = 0
            else:
                value_out = total_bases / total_reads
            config_out.set('global','Mean Read Length at %s' % quality, value_out)


        with open(output_prefix + "quality.summary", 'wb') as configfile:
            config_out.write(configfile)

    except:
        traceback.print_exc()


    try:
        # merge readLen.txt, essentially concatenate
        rl_out = open(output_prefix+"readLen.txt",'w')
        rl_out.write("read\ttrimLen\n")
        for readlen_file in readlen_file_list:
            if os.path.exists(readlen_file):
                printtime("INFO: process %s" % readlen_file)
                rl_in = open(readlen_file,'r')
                first_line = rl_in.readline()
                for line in rl_in:
                    rl_out.write(line)
                rl_in.close()
            else:
                printtime("ERROR: skipped %s" % readlen_file)
        rl_out.close()
    except:
        traceback.print_exc()


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

    combined_datasets_json = dict(block_datasets_json[0])
    
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
    # Merge quality.summary and readLen.txt:               #
    # First across blocks, then across barcodes            #
    ########################################################

    composite_prefix_list = []
    for dataset in combined_datasets_json["datasets"]:

        # Create barcode.quality.summary and barcode.readLen.txt by merging across blocks
        composite_prefix = os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.')
        barcode_prefix_list = [os.path.join(dir,BASECALLER_RESULTS,dataset['file_prefix']+'.') for dir in dirs]
        barcode_prefix_list = [prefix for prefix in barcode_prefix_list if os.path.exists(prefix+"quality.summary")]
        merge_quality_summary(barcode_prefix_list, composite_prefix)
        
        if os.path.exists(composite_prefix+"quality.summary"):
            composite_prefix_list.append(composite_prefix)

    # Create quality.summary and readLen.txt by merging across barcodes
    merge_quality_summary(composite_prefix_list, BASECALLER_RESULTS+'/')


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
                        if 'BaseCaller=0' in text:
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

    for dataset in combined_datasets_json["datasets"]:
        try:
            readlen_path = os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.readLen.txt')
            sparkline_path = os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.sparkline.png')
            read_length_sparkline.read_length_sparkline(readlen_path,sparkline_path,graph_max_x)
        except:
            printtime("ERROR: Failed to create %s" % sparkline_path)

    try:
        filepath_readLenHistogram = os.path.join(BASECALLER_RESULTS,'readLenHisto.png')
        filepath_readlentxt = os.path.join(BASECALLER_RESULTS,'readLen.txt')
        trimmedReadLenHisto.trimmedReadLenHisto(filepath_readlentxt,filepath_readLenHistogram)
        filepath_readLenHistogram2 = os.path.join(BASECALLER_RESULTS,'readLenHisto2.png')
        read_length_sparkline.read_length_histogram(filepath_readlentxt,filepath_readLenHistogram2,graph_max_x)
    except:
        printtime("ERROR: Failed to create %s" % filepath_readLenHistogram)
        traceback.print_exc()
    
    try:
        quality_histogram.generate_quality_histogram(
            os.path.join(BASECALLER_RESULTS,'BaseCaller.json'),
            os.path.join(BASECALLER_RESULTS,'quality_histogram.png'))
    except:
        printtime ("ERROR: Quality histogram generation failed")
        traceback.print_exc()

    try:
        wells_beadogram.generate_wells_beadogram(BASECALLER_RESULTS, SIGPROC_RESULTS)
    except:
        printtime ("ERROR: Wells beadogram generation failed")
        traceback.print_exc()

    printtime("Finished merging basecaller stats")

def merge_basecaller_bam_only(dirs, BASECALLER_RESULTS):

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

        ###############################################
        # Provide symlinks for legacy naming conv.    #
        ###############################################

        if dataset.has_key('legacy_prefix'):
            src = os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])
            dst = os.path.join(BASECALLER_RESULTS, dataset['legacy_prefix']+'.basecaller.bam')
            try:
                os.symlink(os.path.relpath(src,os.path.dirname(dst)),dst)
            except:
                printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))

    ## Note! on barcoded runs, barcode files are NOT subsequently merged into one multi-barcode BAM. 

    printtime("Finished merging basecaller BAM files")
        

def merge_basecaller_bigdata(dirs, BASECALLER_RESULTS):

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

        ###############################################
        # Generate Per-barcode SFF                    #
        ###############################################
    
        if not os.path.exists(os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])):
            continue

        dataset['sff'] = dataset['file_prefix']+'.sff'
        dataset['fastq'] = dataset['file_prefix']+'.fastq'

        try:
            com = "bam2sff"
            com += " -o %s"  % os.path.join(BASECALLER_RESULTS, dataset['sff'])
            com += " %s"     % os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])
            printtime("DEBUG: Calling '%s'" % com)
            subprocess.call(com,shell=True)
        except:
            printtime('Failed bam2sff')

        ###############################################
        # Generate Per-barcode FASTQ                  #
        ###############################################

        try:
            com = "SFFRead"
            com += " -q %s"         % os.path.join(BASECALLER_RESULTS, dataset['fastq'])
            com += " %s"            % os.path.join(BASECALLER_RESULTS, dataset['sff'])
    
            printtime("DEBUG: Calling '%s'" % com)
            ret = subprocess.call(com,shell=True)
            #blockprocessing.add_status("SFFRead", ret)
        except:
            printtime('Failed SFFRead')
        
        ###############################################
        # Provide symlinks for legacy naming conv.    #
        ###############################################

        if dataset.has_key('legacy_prefix'):
            link_src = [
                os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']),
                os.path.join(BASECALLER_RESULTS, dataset['sff']),
                os.path.join(BASECALLER_RESULTS, dataset['fastq'])]
            link_dst = [
                os.path.join(BASECALLER_RESULTS, dataset['legacy_prefix']+'.basecaller.bam'),
                os.path.join(BASECALLER_RESULTS, dataset['legacy_prefix']+'.sff'),
                os.path.join(BASECALLER_RESULTS, dataset['legacy_prefix']+'.fastq')]
            for (src,dst) in zip(link_src,link_dst):
                try:
                    os.symlink(os.path.relpath(src,os.path.dirname(dst)),dst)
                except:
                    printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))

    ## Note! on barcoded runs, barcode files are NOT subsequently merged into one multi-barcode BAM. 

    printtime("Finished merging basecaller big files")



def generate_datasets_json(
        barcodeId,
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

    if not sample:
        sample = "NOSM"
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
                    datasets["datasets"].append({
                        "dataset_name"      : sample + "/" + record[1],
                        "legacy_prefix"     : "bc_files/%s_rawlib" % record[1],
                        "file_prefix"       : '%s_%s_%s' % (record[1],expName,resultsName),
                        "read_groups"       : [runID+"."+record[1],]
                    })
                    datasets["read_groups"][runID+"."+record[1]] = {
                        "barcode_name"      : record[1],
                        "barcode_sequence"  : record[2],
                        "barcode_adapter"   : record[3],
                        "index"             : int(record[0]),
                        "sample"            : sample,
                        "library"           : libraryName+"/"+record[1],
                        "description"       : ''.join(ch for ch in notes if ch.isalnum() or ch == " "),
                        "platform_unit"     :  "PGM/%s/%s" % (chipType.replace('"',""),record[1])
                    }
            f.close()

            datasets["datasets"].append({
                "dataset_name"      : sample + "/No_barcode_match",
                "legacy_prefix"     : "bc_files/%s_rawlib" % "nomatch",
                "file_prefix"       : "%s_%s_%s" % ("nomatch",expName,resultsName),
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
            "legacy_prefix"     : "rawlib",
            "file_prefix"       : "%s_%s" % (expName,resultsName),
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
        'sfftrim_args'          : '',
        'barcodeId'             : 'IonExpress',
        #'barcodeId'             : '',
        'barcodesplit_filter'   : 0.01,
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
        env['barcodesplit_filter'],
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
        env['SIGPROC_RESULTS'],
        os.path.join(env['BASECALLER_RESULTS'], "rawtf.basecaller.bam"),
        env['libraryKey'],
        env['tfKey'],
        env['flowOrder'],
        env['BASECALLER_RESULTS'])

    
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
