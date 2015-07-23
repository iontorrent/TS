#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

from ion.utils.blockprocessing import printtime
from ion.utils import blockprocessing

import traceback
import os
import subprocess
import sys
import time
import multiprocessing
import ionstats


def alignTFs(basecaller_bam_filename,bam_filename,fasta_path):

    com1 = "tmap mapall -n 12 -f %s -r %s -Y -v stage1 map4" % (fasta_path, basecaller_bam_filename)
    com2 = "samtools view -Sb -o %s - 2>> /dev/null" % bam_filename
    p1 = subprocess.Popen(com1, stdout=subprocess.PIPE, shell=True)
    p2 = subprocess.Popen(com2, stdin=p1.stdout, shell=True)
    p2.communicate()
    p1.communicate()

    if p1.returncode != 0:
        raise subprocess.CalledProcessError(p1.returncode, com1)
    if p2.returncode != 0:
        # Assumption: samtools view only fails when there are zero reads.
        printtime("Command %s failed, presumably because there are no TF reads" % (com2))
        raise Exception('No TF reads found')        
        #raise subprocess.CalledProcessError(p2.returncode, com2)


def align(
    blocks,
    alignerArgs,
    ionstatsArgs,
    referenceName,
    basecaller_meta_information,
    library_key,
    graph_max_x,
    readFile,
    do_realign,
    do_ionstats,
    do_sorting,
    do_mark_duplicates,
    do_indexing,
    logfile,
    output_dir,
    output_basename):
    # Input  : readFile
    # Output : output_dir/output_basename.bam

    try:

        if 'tmap' in alignerArgs:
            aligner = 'tmap'
            if '...' in alignerArgs:
                alist = alignerArgs.split('...')
                cmd = alist[0]
                tmap_stage_options = alist[1]
            else:
                cmd = 'tmap mapall'
                tmap_stage_options = 'stage1 map4'
        elif 'bowtie2' in alignerArgs:
            aligner = 'bowtie2'
            cmd = alignerArgs
        else:
            printtime("ERROR: Aligner command not specified")
            raise


        threads = multiprocessing.cpu_count()
        bamBase = os.path.normpath(output_dir + "/" + output_basename)
        bamFile = bamBase + ".bam"

        blocks=[] # TODO
        if aligner == 'tmap':
            referenceFastaFile = '/results/referenceLibrary/tmap-f3/' + referenceName + '/' + referenceName + '.fasta'
            if blocks:
                mergecmd = 'java -Xmx8g -jar /opt/picard/picard-tools-current/picard.jar MergeSamFiles'
                bamdir = '.'
                bamfile = readFile
                block_bam_list = [os.path.join(blockdir, bamdir, bamfile) for blockdir in blocks]
                block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
                for bamfile in block_bam_list:
                    mergecmd = mergecmd + ' I=%s' % bamfile
                mergecmd = mergecmd + ' O=/dev/stdout'
                mergecmd = mergecmd + ' VERBOSITY=WARNING' # suppress INFO on stderr
                mergecmd = mergecmd + ' QUIET=true' # suppress job-summary on stderr
                mergecmd = mergecmd + ' VALIDATION_STRINGENCY=SILENT'
                cmd = mergecmd + " | " + cmd
            cmd += " -n %d" % threads
            cmd += " -f %s" % referenceFastaFile
            if blocks:
                cmd += " -i bam"
            else:
                cmd += " -r %s" % readFile
            cmd += " -v"
            cmd += " -Y"
            cmd += " -u --prefix-exclude 5"  # random seed based on read name after ignoring first 5 characters
            if do_realign:
                cmd += " --do-realign"
            cmd += " -o 2" # -o 0: SAM, -o 2: uncompressed BAM
            cmd += " %s" % tmap_stage_options
            cmd += " 2>> " + logfile

        elif aligner == 'bowtie2':
            referenceFastaDir  = '/results/referenceLibrary/bowtie2/' + referenceName + '/' + referenceName
            cmd="java -Xmx8g -jar /opt/picard/picard-tools-current/picard.jar SamToFastq I=%s F=/dev/stdout" % readFile
            cmd+=" | /results/plugins/bowtielauncher/bowtie2 -p%d -x %s -U /dev/stdin" % (threads, referenceFastaDir)
            cmd+=" | samtools view -ubS -"

        if do_ionstats:
            bam_filenames=["/dev/stdin"]
            ionstats_alignment_filename="%s.ionstats_alignment.json" % bamBase
            ionstats_alignment_h5_filename="%s.ionstats_error_summary.h5" % bamBase

            ionstats_cmd = ionstats.generate_ionstats_alignment_cmd(
                               ionstatsArgs,
                               bam_filenames,
                               ionstats_alignment_filename,
                               ionstats_alignment_h5_filename,
                               basecaller_meta_information,
                               library_key,
                               graph_max_x)

            cmd += " | tee >(%s)" % ionstats_cmd

        if do_sorting:
            if do_mark_duplicates:
                #TODO: implement alternative, maybe with named pipes
                cmd += " | samtools sort -m 1000M -l1 -@12 -o - -"
                json_name = 'BamDuplicates.%s.json' % bamBase if bamBase!='rawlib' else 'BamDuplicates.json'
                cmd = "BamDuplicates -i <(%s) -o %s -j %s" % (cmd, bamFile, json_name)
            else:
#                cmd += " | ( samtools sort -m 1000M -l1 -@12 - %s <&0 & )" % bamBase
                cmd += " | samtools sort -m 1000M -l1 -@12 - %s" % bamBase
        else:
            cmd += " > %s.bam" % bamBase


        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.Popen(['/bin/bash', '-c', cmd]).wait()
        if ret != 0:
            printtime("ERROR: alignment failed, return code: %d" % ret)
            raise RuntimeError('exit code: %d' % ret)


        # TODO: piping into samtools index or create index in sort process ?
        if do_indexing and do_sorting:
            cmd = "samtools index " + bamFile
            print cmd
            subprocess.call(cmd,shell=True)

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

# this becomes the main pipeline function
#TODO : pass also block dirs for multi-bam tmap process
def process_datasets(
        blocks,
        alignmentArgs,
        ionstatsArgs,
        BASECALLER_RESULTS,
        basecaller_meta_information,
        library_key,
        graph_max_x,
        basecaller_datasets,
        ALIGNMENT_RESULTS,
        do_realign,
        do_ionstats,
        do_mark_duplicates,
        do_indexing,
        barcodeInfo):

    printtime("Attempt to align")

    do_sorting = True

    # compare with pipeline/python/ion/utils/ionstats.py
    ionstats_basecaller_file_list = []
    ionstats_alignment_file_list = []
    ionstats_basecaller_filtered_file_list = []
    ionstats_alignment_filtered_file_list = []

    for dataset in basecaller_datasets["datasets"]:

        read_group = dataset['read_groups'][0]
        reference = basecaller_datasets['read_groups'][read_group]['reference']
        #print "DEBUG: reference: %s' % reference

        filtered = True
        for rg_name in dataset["read_groups"]:
            if not basecaller_datasets["read_groups"][rg_name].get('filtered',False):
                filtered = False

        # skip non-existing bam file
        if int(dataset["read_count"]) == 0:
            continue

        if reference:

            # merge unmapped bam files TODO move into align
            try:
                bamdir = BASECALLER_RESULTS
                bamfile = dataset['basecaller_bam']
                block_bam_list = [os.path.join(blockdir, bamdir, bamfile) for blockdir in blocks]
                block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
                composite_bam_filepath = os.path.join(bamdir, bamfile)
                if block_bam_list:
                    composite_bai_filepath=""
                    mark_duplicates=False
                    method='samtools'
                    blockprocessing.merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates, method)
            except:
                traceback.print_exc()
                printtime("ERROR: merging %s unsuccessful" % bamfile)


            try:
                align(
                    blocks,
                    alignmentArgs,
                    ionstatsArgs,
                    reference,
                    basecaller_meta_information,
                    library_key,
                    graph_max_x,
                    os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']),
                    do_realign,
                    do_ionstats,
                    do_sorting,
                    do_mark_duplicates,
                    do_indexing,
                    logfile=os.path.join(ALIGNMENT_RESULTS,dataset['file_prefix']+'.alignmentQC_out.txt'),
                    output_dir=ALIGNMENT_RESULTS,
                    output_basename=dataset['file_prefix'])
            except:
                traceback.print_exc()

            if filtered:
                ionstats_alignment_filtered_file_list.append(os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json'))
            else:
                ionstats_alignment_file_list.append(os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json'))

            '''
            if do_indexing:
                try:
                    composite_bam_filepath = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam')
                    composite_bai_filepath = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam.bai')
                    blockprocessing.create_index_file(composite_bam_filepath, composite_bai_filepath)
                except:
                    traceback.print_exc()
            '''

        else:

            # merge unmapped bam file without reference
            try:
                bamdir = BASECALLER_RESULTS
                bamfile = dataset['basecaller_bam']
                block_bam_list = [os.path.join(blockdir, bamdir, bamfile) for blockdir in blocks]
                block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
                composite_bam_filepath = os.path.join(bamdir, bamfile)
                if block_bam_list:
                    composite_bai_filepath=""
                    mark_duplicates=False
                    method='samtools'
                    blockprocessing.merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates, method)
            except:
                traceback.print_exc()
                printtime("ERROR: merging %s unsuccessful" % bamfile)


            if do_ionstats:
                # TODO: move ionstats basecaller into basecaller
                ionstats.generate_ionstats_basecaller(
                    [os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])],
                    os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'),
                    library_key,
                    graph_max_x)

                if filtered:
                    ionstats_basecaller_filtered_file_list.append(os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'))
                else:
                    ionstats_basecaller_file_list.append(os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'))

    if do_ionstats:

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
                if ret != 0:
                    printtime("ERROR: empty bam file generation failed, return code: %d" % ret)
                    raise RuntimeError('exit code: %d' % ret)

                ionstats.generate_ionstats_alignment(
                    ionstatsArgs,
                    ['empty_dummy.bam'],
                    os.path.join(ALIGNMENT_RESULTS, 'ionstats_alignment.json'),
                    os.path.join(ALIGNMENT_RESULTS, 'ionstats_error_summary.h5'),
                    basecaller_meta_information,
                    library_key,
                    graph_max_x)

            except:
                raise

        if len(ionstats_basecaller_file_list) > 0:
            ionstats.reduce_stats(ionstats_basecaller_file_list,os.path.join(BASECALLER_RESULTS,'ionstats_tmp_basecaller.json'))
        else: # barcode classification filtered all barcodes or no reads available
            # TODO: ionstats needs to produce initial json file
            try:
                #cmd = "echo $'@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"
                cmd  = "echo  '@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"

                printtime("DEBUG: Calling '%s':" % cmd)
                ret = subprocess.call(cmd,shell=True)
                if ret != 0:
                    printtime("ERROR: empty bam file generation failed, return code: %d" % ret)
                    raise RuntimeError('exit code: %d' % ret)

                ionstats.generate_ionstats_basecaller(
                    ['empty_dummy.bam'],
                    os.path.join(BASECALLER_RESULTS, 'ionstats_tmp_basecaller.json'),
                    library_key,
                    graph_max_x)
            except:
                raise


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
#        if len(ionstats_alignment_h5_file_list) > 0:
#            ionstats.reduce_stats_h5(ionstats_alignment_h5_file_list,os.path.join(ALIGNMENT_RESULTS,'ionstats_error_summary.h5'))


    printtime("**** Alignment completed ****")

