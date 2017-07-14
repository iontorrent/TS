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


def alignTFs(basecaller_bam_filename, bam_filename, fasta_path):

    com1 = "tmap mapall -n 12 -f %s -r %s -Y -v stage1 map4" % (fasta_path, basecaller_bam_filename)
    com2 = "samtools view -Sb -o %s - 2>> /dev/null" % bam_filename
    printtime("DEBUG: Calling '%s | %s':" % (com1, com2))
    p1 = subprocess.Popen(com1, stdout=subprocess.PIPE, shell=True)
    p2 = subprocess.Popen(com2, stdin=p1.stdout, shell=True)
    p2.communicate()
    p1.communicate()

    if p1.returncode != 0:
        raise subprocess.CalledProcessError(p1.returncode, com1)
    if p2.returncode != 0:
        # Assumption: samtools view only fails when there are zero reads.
        printtime("Command '%s | %s' failed, presumably because there are no TF reads" % (com1, com2))
        raise Exception('No TF reads found')
        # raise subprocess.CalledProcessError(p2.returncode, com2)


# has to support 4 modes:
# 1 block/dataset,                   ionstats           (no reference or nomatch BC)
# 1 block/dataset,        alignment, ionstats, indexing
# N blocks,        merge,          , ionstats           (no reference or nomatch BC)
# N blocks,        merge, alignment, ionstats, indexing
#                                                       alignment includes sorting and bamdup and ( indexing )
#
#
# Input  : blocks                         # e.g.  [] or ['block_X0_Y0', ...]
# Input  : basecaller_bam_filename        # e.g.  os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])
# Output : output_dir/output_basename.bam # output_dir/output_basename.bam
# outputs files in multiple directories, merged unmapped bams and merged mapped bams
# output merged (un/mapped) bam file, (base/align) ionstats file, index file
#
def align(
    blocks,
    basecaller_bam_filename,  # e.g. 'basecaller_results/IonXpress001.basecaller.bam'
    alignerArgs,
    ionstatsArgs,
    referenceName,
    basecaller_meta_information,
    library_key,
    graph_max_x,
    do_realign,
    do_ionstats,
    do_sorting,
    do_mark_duplicates,
    do_indexing,
    output_dir,
    output_basename,
    threads=0):

    try:

        threads = threads or multiprocessing.cpu_count()
        bamBase = os.path.normpath(output_dir + "/" + output_basename)
        bamFile = bamBase + ".bam"

        printtime("reference:            %s" % referenceName)
        printtime("input blocks:         %s" % blocks)
        printtime("input reads:          %s" % basecaller_bam_filename)
        printtime("output dir:           %s" % output_dir)
        printtime("output basename:      %s" % output_basename)
        printtime("full output base:     %s" % bamBase)
        printtime("full output file:     %s" % bamFile)  # TODO: not always used

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

        if not referenceName:

            # 1. create merged unmapped bam, 2. call ionstats
            # TODO: long term: move ionstats basecaller into basecaller binary

            cmd = ""
            composite_bam_filepath = bamBase+'.basecaller.bam'
            if blocks:
                bamdir = '.'  # TODO , do we need this ?
                bamfile = basecaller_bam_filename
                block_bam_list = [os.path.join(blockdir, bamdir, bamfile) for blockdir in blocks]
                block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
                if len(block_bam_list) >= 2:
                    blockprocessing.extract_and_merge_bam_header(block_bam_list, composite_bam_filepath)
                    cmd = 'samtools cat -h %s.header.sam -o /dev/stdout' % (composite_bam_filepath)
                    for blockbamfile in block_bam_list:
                        cmd = cmd + ' %s' % blockbamfile
                elif len(block_bam_list) == 1:
#                    cmd = "samtools reheader %s.header.sam %s -" % (composite_bam_filepath,block_bam_list[0])
                    cmd = "cat %s" % (block_bam_list[0])
                else:
                    return
                '''
                if block_bam_list:
                    composite_bai_filepath=""
                    mark_duplicates=False
                    method='samtools'
                    blockprocessing.merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates, method)
                '''
                bam_filenames = ["/dev/stdin"]
            else:
                bam_filenames = [basecaller_bam_filename]
            if do_ionstats:
                ionstats_cmd = ionstats.generate_ionstats_basecaller_cmd(
                    bam_filenames,
                    bamBase+'.ionstats_basecaller.json',
                    library_key,
                    graph_max_x)

                if blocks:
                    cmd += " | tee >(%s)" % ionstats_cmd
                    cmd += " > %s" % composite_bam_filepath
                else:
                    cmd = ionstats_cmd

            printtime("DEBUG: Calling '%s':" % cmd)
            ret = subprocess.Popen(['/bin/bash', '-c', cmd]).wait()
            if ret != 0:
                printtime("ERROR: unmapped bam merging failed, return code: %d" % ret)
                raise RuntimeError('exit code: %d' % ret)

            return

        if aligner == 'tmap':
            referenceFastaFile = '/results/referenceLibrary/tmap-f3/' + referenceName + '/' + referenceName + '.fasta'
            if blocks:
                bamdir = '.'  # TODO , do we need this ?
                bamfile = basecaller_bam_filename
#                printtime("DEBUG: BLOCKS for BAMFILE %s: %s" % (bamfile, blocks))
                block_bam_list = [os.path.join(blockdir, bamdir, bamfile) for blockdir in blocks]
#                printtime("DEBUG: block_bam_list: %s" % block_bam_list)
                block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
#                printtime("DEBUG: block_bam_list: %s" % block_bam_list)
                printtime("blocks with reads:    %s" % len(block_bam_list))
                if len(block_bam_list) >= 2:
                    blockprocessing.extract_and_merge_bam_header(block_bam_list, basecaller_bam_filename)
                    mergecmd = 'samtools cat -h %s.header.sam -o /dev/stdout' % basecaller_bam_filename
                    for blockbamfile in block_bam_list:
                        mergecmd = mergecmd + ' %s' % blockbamfile
                    '''
                    mergecmd = 'java -Xmx8g -jar /opt/picard/picard-tools-current/picard.jar MergeSamFiles'
                    for blockbamfile in block_bam_list:
                        mergecmd = mergecmd + ' I=%s' % blockbamfile
                    mergecmd = mergecmd + ' O=/dev/stdout'
                    mergecmd = mergecmd + ' VERBOSITY=WARNING' # suppress INFO on stderr
                    mergecmd = mergecmd + ' QUIET=true' # suppress job-summary on stderr
                    mergecmd = mergecmd + ' VALIDATION_STRINGENCY=SILENT'
                    '''
                    cmd = mergecmd + " | " + cmd
                    cmd += " -n %d" % threads
                    cmd += " -f %s" % referenceFastaFile
                    cmd += " -i bam"
                elif len(block_bam_list) == 1:
                    cmd += " -n %d" % threads
                    cmd += " -f %s" % referenceFastaFile
                    cmd += " -r %s" % block_bam_list[0]
                else:
                    printtime("ERROR: all blocks filtered")
                    return
            else:
                cmd += " -n %d" % threads
                cmd += " -f %s" % referenceFastaFile
                cmd += " -r %s" % basecaller_bam_filename
            cmd += " -v"
            cmd += " -Y"
            cmd += " -u --prefix-exclude 5"  # random seed based on read name after ignoring first 5 characters
            if do_realign:
                cmd += " --do-realign"
            cmd += " -o 2"  # -o 0: SAM, -o 2: uncompressed BAM
            cmd += " %s" % tmap_stage_options
            cmd += " 2>> " + bamBase + '.alignmentQC_out.txt'  # logfile

        elif aligner == 'bowtie2':
            referenceFastaDir = '/results/referenceLibrary/bowtie2/' + referenceName + '/' + referenceName
            cmd = "java -Xmx8g -jar /opt/picard/picard-tools-current/picard.jar SamToFastq I=%s F=/dev/stdout" % basecaller_bam_filename
            cmd += " | /results/plugins/bowtielauncher/bowtie2 -p%d -x %s -U /dev/stdin" % (threads, referenceFastaDir)
            cmd += " | samtools view -ubS -"

        if do_ionstats:
            bam_filenames = ["/dev/stdin"]
            ionstats_alignment_filename = "%s.ionstats_alignment.json" % bamBase
            ionstats_alignment_h5_filename = "%s.ionstats_error_summary.h5" % bamBase

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
                # TODO: implement alternative, maybe with named pipes
                cmd += " | samtools sort -m 1000M -l1 -@12 -o - -"
                json_name = 'BamDuplicates.%s.json' % bamBase if bamBase != 'rawlib' else 'BamDuplicates.json'
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
            printtime("DEBUG: Calling '%s':" % cmd)
            subprocess.call(cmd, shell=True)

            '''
            if do_indexing:
                try:
                    composite_bam_filepath = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam')
                    composite_bai_filepath = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam.bai')
                    blockprocessing.create_index_file(composite_bam_filepath, composite_bai_filepath)
                except:
                    traceback.print_exc()
            '''

    except:
        raise


def align_dataset_parallel(
        dataset,
        blocks,
        reference,
        alignmentArgs,
        ionstatsArgs,
        BASECALLER_RESULTS,
        basecaller_meta_information,
        library_key,
        graph_max_x,
        ALIGNMENT_RESULTS,
        do_realign,
        do_ionstats,
        do_mark_duplicates,
        do_indexing,
        align_threads
    ):

    do_sorting = True

    try:
        # process block by block
        if reference and len(blocks) > 1 and int(dataset["read_count"]) > 20000000:
            printtime("DEBUG: TRADITIONAL BLOCK PROCESSING ------ prefix: %20s ----------- reference: %20s ---------- reads: %10s ----------" % (dataset['file_prefix'], reference, dataset["read_count"]))
          # start alignment for each block and current barcode with reads
          # TODO: in how many blocks are reads with this barcode
            for block in blocks:
                printtime("DEBUG: ALIGN ONLY ONE BLOCK: %s" % block)
                align(
                    [block],
                    os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']),
                    alignmentArgs,
                    ionstatsArgs,
                    reference,
                    basecaller_meta_information,
                    library_key,
                    graph_max_x,
                    do_realign,
                    do_ionstats=False,
                    do_sorting=do_sorting,
                    do_mark_duplicates=False,
                    do_indexing=False,
                    output_dir=os.path.join(block, ALIGNMENT_RESULTS),
                    output_basename=dataset['file_prefix'],
                    threads=align_threads)
    
            bamdir = '.'  # TODO , do we need this ?
            bamBase = dataset['file_prefix']
            bamfile = dataset['file_prefix'] + ".bam"
    
            block_bam_list = [os.path.join(blockdir, bamdir, bamfile) for blockdir in blocks]
            block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
            printtime("blocks with reads:    %s" % len(block_bam_list))
    
            bamFile = dataset['file_prefix'] + ".bam"
            composite_bam_filepath = dataset['file_prefix'] + ".bam"
    
            blockprocessing.extract_and_merge_bam_header(block_bam_list, composite_bam_filepath)
            # Usage: samtools merge [-nr] [-h inh.sam] <out.bam> <in1.bam> <in2.bam> [...]
            cmd = 'samtools merge -l1 -@8'
            if do_ionstats:
                cmd += ' - '
            else:
                cmd += ' %s' % (composite_bam_filepath)
            for bamfile in block_bam_list:
                cmd += ' %s' % bamfile
            cmd += ' -h %s.header.sam' % composite_bam_filepath
    
            if do_ionstats:
                bam_filenames = ["/dev/stdin"]
                ionstats_alignment_filename = "%s.ionstats_alignment.json" % bamBase      # os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json')
                ionstats_alignment_h5_filename = "%s.ionstats_error_summary.h5" % bamBase  # os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_error_summary.h5')
    
                ionstats_cmd = ionstats.generate_ionstats_alignment_cmd(
                    ionstatsArgs,
                    bam_filenames,
                    ionstats_alignment_filename,
                    ionstats_alignment_h5_filename,
                    basecaller_meta_information,
                    library_key,
                    graph_max_x)
    
                cmd += " | tee >(%s)" % ionstats_cmd
    
            if do_mark_duplicates:
                json_name = 'BamDuplicates.%s.json' % bamBase if bamBase != 'rawlib' else 'BamDuplicates.json'
                cmd = "BamDuplicates -i <(%s) -o %s -j %s" % (cmd, bamFile, json_name)
            else:
                cmd += " > %s.bam" % bamBase
    
            printtime("DEBUG: Calling '%s':" % cmd)
            ret = subprocess.Popen(['/bin/bash', '-c', cmd]).wait()
            if ret != 0:
                printtime("ERROR: merging failed, return code: %d" % ret)
                raise RuntimeError('exit code: %d' % ret)
    
            # TODO: piping into samtools index or create index in sort process ?
            if do_indexing and do_sorting:
                cmd = "samtools index " + bamFile
                printtime("DEBUG: Calling '%s':" % cmd)
                subprocess.call(cmd, shell=True)
    
        else:
            printtime("DEBUG: MERGED BLOCK PROCESSING ----------- prefix: %20s ----------- reference: %20s ---------- reads: %10s ----------" % (dataset['file_prefix'], reference, dataset["read_count"]))
            # TODO: try a python multiprocessing pool
            align(
                blocks,
                os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam']),
                alignmentArgs,
                ionstatsArgs,
                reference,
                basecaller_meta_information,
                library_key,
                graph_max_x,
                do_realign,
                do_ionstats,
                do_sorting,
                do_mark_duplicates,
                do_indexing,
                output_dir=ALIGNMENT_RESULTS if reference else BASECALLER_RESULTS,
                output_basename=dataset['file_prefix'],
                threads=align_threads)
    except:
        traceback.print_exc()


def align_dataset_parallel_wrap(args):
    return align_dataset_parallel(*args)


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

    parallel_datasets = 1
    try:
        memTotalGb = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/(1024*1024*1024)
        if memTotalGb > 70:
            parallel_datasets = 2
    except:
        pass

    align_threads = multiprocessing.cpu_count() / parallel_datasets
    printtime("Attempt to align")
    printtime("DEBUG: PROCESS DATASETS blocks: '%s', parallel datasets: %d" % (blocks, parallel_datasets))

    # TODO: compare with pipeline/python/ion/utils/ionstats.py
    ionstats_basecaller_file_list = []
    ionstats_alignment_file_list = []
    ionstats_basecaller_filtered_file_list = []
    ionstats_alignment_filtered_file_list = []
    
    align_dataset_args = []

    for dataset in basecaller_datasets["datasets"]:

        read_group = dataset['read_groups'][0]
        reference = basecaller_datasets['read_groups'][read_group]['reference']
        # print "DEBUG: reference: %s' % reference

        filtered = True
        for rg_name in dataset["read_groups"]:
            if not basecaller_datasets["read_groups"][rg_name].get('filtered', False):
                filtered = False

        # skip non-existing bam file
        if int(dataset["read_count"]) == 0:
            continue

        align_dataset_args.append((
            dataset,
            blocks,
            reference,
            alignmentArgs,
            ionstatsArgs,
            BASECALLER_RESULTS,
            basecaller_meta_information,
            library_key,
            graph_max_x,
            ALIGNMENT_RESULTS,
            do_realign,
            do_ionstats,
            do_mark_duplicates,
            do_indexing,
            align_threads
        ))

        if reference:
            if filtered:
                ionstats_alignment_filtered_file_list.append(os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json'))
            else:
                ionstats_alignment_file_list.append(os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json'))
        else:
            if filtered:
                ionstats_basecaller_filtered_file_list.append(os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'))
            else:
                ionstats_basecaller_file_list.append(os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+'.ionstats_basecaller.json'))

    # do alignment in multiprocessing pool
    pool = multiprocessing.Pool(processes=parallel_datasets)
    pool.map(align_dataset_parallel_wrap, align_dataset_args)


    if do_ionstats:

        # Merge ionstats files from individual (barcoded) datasets
        if len(ionstats_alignment_file_list) > 0:
            ionstats.reduce_stats(ionstats_alignment_file_list, os.path.join(ALIGNMENT_RESULTS, 'ionstats_alignment.json'))
        else:  # barcode classification filtered all barcodes or no reads available
            # TODO: ionstats needs to produce initial json file
            try:
                # cmd = "echo $'@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"
                cmd = "echo  '@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"

                printtime("DEBUG: Calling '%s':" % cmd)
                ret = subprocess.call(cmd, shell=True)
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
            ionstats.reduce_stats(ionstats_basecaller_file_list, os.path.join(BASECALLER_RESULTS, 'ionstats_tmp_basecaller.json'))
        else:  # barcode classification filtered all barcodes or no reads available
            # TODO: ionstats needs to produce initial json file
            try:
                # cmd = "echo $'@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"
                cmd = "echo  '@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"

                printtime("DEBUG: Calling '%s':" % cmd)
                ret = subprocess.call(cmd, shell=True)
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
        a = os.path.join(ALIGNMENT_RESULTS, 'ionstats_alignment.json')
        b = os.path.join(BASECALLER_RESULTS, 'ionstats_tmp_basecaller.json')
        if os.path.exists(a):
            ionstatslist.append(a)
        if os.path.exists(b):
            ionstatslist.append(b)
        if len(ionstatslist) > 0:
            ionstats.reduce_stats(ionstatslist, os.path.join(BASECALLER_RESULTS, 'ionstats_basecaller_with_aligninfos.json'))
            ionstats.reduce_stats(reversed(ionstatslist), os.path.join(BASECALLER_RESULTS, 'ionstats_basecaller.json'))
#        if len(ionstats_alignment_h5_file_list) > 0:
#            ionstats.reduce_stats_h5(ionstats_alignment_h5_file_list,os.path.join(ALIGNMENT_RESULTS,'ionstats_error_summary.h5'))

    printtime("**** Alignment completed ****")
