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
import ion


def _get_total_memory_gb():
    return (
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024 * 1024)
    )


def _add_read_queue_size(input_cmd, queue_size=50000):
    """
    only add when mapall option is used and
    "-q" and "--reads-queue-size" is not already used.

    there is no other long command starting with q either.

    :param input_cmd:
    :param queue_size:
    :return:
    """
    if "mapall" not in input_cmd:
        return input_cmd

    if ("-q" in input_cmd) or ("--reads-queue-size" in input_cmd):
        return input_cmd

    return input_cmd + "-q %d" % queue_size


def alignTFs(basecaller_bam_filename, bam_filename, fasta_path):

    com1 = "tmap mapall -n 12 -f %s -r %s -Y -v stage1 map4" % (
        fasta_path,
        basecaller_bam_filename,
    )
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
        printtime(
            "Command '%s | %s' failed, presumably because there are no TF reads"
            % (com1, com2)
        )
        raise Exception("No TF reads found")
        # raise subprocess.CalledProcessError(p2.returncode, com2)


def process_tmap_bed_file_args(input_args, bam_filename, barcode_info):
    """ modify tmap args based on bam_filename and barcode_info

    append bed-file when target region is defined in either per sample or in default. In case of barcoded runs,
    if no specific sample is defined in plan, it is assumed to be DNA type.

    scenario to account for:

    * bed files defined in per-sample settings
    * bed files are defined in default but not per-sample settings.
    * bed files are defined in default and additional BAM detected with per-sample settings.
    * all detected BAM files are assumed to be 'DNA' unless contradicted in per-sample settings

    trim off `--bed-file` when

    * barcode_info is {}
    * basecaller_bam_filename start with "nomatch"; this will not affect non-barcoded run.
    * if matched barcode is found, ignore non-DNA type

    input_args: alignerArgs
    bam_filename: basecaller_bam_filename
    barcode_info: barcode_info
    """

    def _get_barcode_bed_file(bam_filename, barcode_info):
        """
        'no_barcode' is used by non-barcoded run and the default/fallback values

        :param bam_filename:
        :param barcode_info:
        :return:
        """
        default_bed_file = ""
        if "no_barcode" in barcode_info:
            default_bed_file = barcode_info["no_barcode"].get("targetRegionBedFile", "")

        for barcode, info in barcode_info.items():
            if bam_filename.startswith(barcode):
                # exist at first match
                nuc_type = info.get("nucleotideType", "")
                if nuc_type == "RNA":
                    # if RNA is specified, return empty string
                    return ""
                else:
                    # everything else is considered DNA, even nuc_type is empty/not defined.
                    return info.get("targetRegionBedFile", "")

        # return default bed file when no match is found
        return default_bed_file

    def _trim_bed_file_args(args):
        # only look for --bed-file surrounded by spaces
        return args.replace(" --bed-file ", " ")

    def _add_target_region(args, target_region_file):
        # only look for --bed-file surrounded by spaces
        return args.replace(
            " --bed-file ", " --bed-file {bed} ".format(bed=target_region_file)
        )

    # tmap and 'bed-file' specific
    if "tmap" not in input_args or "--bed-file" not in input_args:
        return input_args

    # ignore nomatch barcode BAMs
    if bam_filename.startswith("nomatch"):
        return _trim_bed_file_args(input_args)

    # ignore if barcode is empty
    if not barcode_info:
        return _trim_bed_file_args(input_args)

    target_region_file = _get_barcode_bed_file(bam_filename, barcode_info)

    # trim tmap arg if target_region is not found or append bed files to tmap args
    if target_region_file:
        return _add_target_region(input_args, target_region_file)
    else:
        return _trim_bed_file_args(input_args)


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
    threads=0,
    barcode_info={},
):

    try:

        threads = threads or multiprocessing.cpu_count()
        memGb = _get_total_memory_gb()
        bamBase = os.path.normpath(output_dir + "/" + output_basename)
        bamFile = bamBase + ".bam"

        printtime("reference:            %s" % referenceName)
        printtime("input blocks:         %s" % blocks)
        printtime("input reads:          %s" % basecaller_bam_filename)
        printtime("output dir:           %s" % output_dir)
        printtime("output basename:      %s" % output_basename)
        printtime("full output base:     %s" % bamBase)
        printtime("full output file:     %s" % bamFile)  # TODO: not always used

        # process args before the splits
        alignerArgs = process_tmap_bed_file_args(
            input_args=alignerArgs,
            bam_filename=output_basename,
            barcode_info=barcode_info,
        )

        if "tmap" in alignerArgs:
            aligner = "tmap"
            if "..." in alignerArgs:
                alist = alignerArgs.split("...")
                cmd = alist[0]
                if memGb and memGb <= 40:
                    cmd = _add_read_queue_size(cmd, 50000)
                tmap_stage_options = alist[1]
            else:
                cmd = "tmap mapall"
                if memGb and memGb <= 40:
                    cmd = _add_read_queue_size(cmd, 50000)
                tmap_stage_options = "stage1 map4"
        elif "bowtie2" in alignerArgs:
            aligner = "bowtie2"
            cmd = alignerArgs
        else:
            printtime("ERROR: Aligner command not specified")
            raise

        if not referenceName:

            # 1. create merged unmapped bam, 2. call ionstats
            # TODO: long term: move ionstats basecaller into basecaller binary

            cmd = ""
            composite_bam_filepath = bamBase + ".basecaller.bam"
            if blocks:
                bamdir = "."  # TODO , do we need this ?
                bamfile = basecaller_bam_filename
                block_bam_list = [
                    os.path.join(blockdir, bamdir, bamfile) for blockdir in blocks
                ]
                block_bam_list = [
                    block_bam_filename
                    for block_bam_filename in block_bam_list
                    if os.path.exists(block_bam_filename)
                ]
                if len(block_bam_list) >= 2:
                    blockprocessing.extract_and_merge_bam_header(
                        block_bam_list, composite_bam_filepath
                    )
                    cmd = "samtools cat -h %s.header.sam -o /dev/stdout" % (
                        composite_bam_filepath
                    )
                    for blockbamfile in block_bam_list:
                        cmd = cmd + " %s" % blockbamfile
                elif len(block_bam_list) == 1:
                    #                    cmd = "samtools reheader %s.header.sam %s -" % (composite_bam_filepath,block_bam_list[0])
                    cmd = "cat %s" % (block_bam_list[0])
                else:
                    return
                """
                if block_bam_list:
                    composite_bai_filepath=""
                    mark_duplicates=False
                    method='samtools'
                    blockprocessing.merge_bam_files(block_bam_list, composite_bam_filepath, composite_bai_filepath, mark_duplicates, method)
                """
                bam_filenames = ["/dev/stdin"]
            else:
                bam_filenames = [basecaller_bam_filename]

            if do_ionstats:
                ionstats_cmd = ionstats.generate_ionstats_basecaller_cmd(
                    bam_filenames,
                    bamBase + ".ionstats_basecaller.json",
                    library_key,
                    graph_max_x,
                )

                if blocks:
                    cmd += " | tee >(%s)" % ionstats_cmd
                    cmd += " > %s" % composite_bam_filepath
                else:
                    cmd = ionstats_cmd

            printtime("DEBUG: Calling '%s':" % cmd)
            ret = subprocess.Popen(["/bin/bash", "-c", cmd]).wait()
            if ret != 0:
                printtime("ERROR: unmapped bam merging failed, return code: %d" % ret)
                raise RuntimeError("exit code: %d" % ret)

            return

        if aligner == "tmap":
            referenceFastaFile = (
                ion.referenceBasePath + referenceName + "/" + referenceName + ".fasta"
            )
            if blocks:
                bamdir = "."  # TODO , do we need this ?
                bamfile = basecaller_bam_filename
                #                printtime("DEBUG: BLOCKS for BAMFILE %s: %s" % (bamfile, blocks))
                block_bam_list = [
                    os.path.join(blockdir, bamdir, bamfile) for blockdir in blocks
                ]
                #                printtime("DEBUG: block_bam_list: %s" % block_bam_list)
                block_bam_list = [
                    block_bam_filename
                    for block_bam_filename in block_bam_list
                    if os.path.exists(block_bam_filename)
                ]
                #                printtime("DEBUG: block_bam_list: %s" % block_bam_list)
                printtime("blocks with reads:    %s" % len(block_bam_list))
                if len(block_bam_list) >= 2:
                    blockprocessing.extract_and_merge_bam_header(
                        block_bam_list, basecaller_bam_filename
                    )
                    mergecmd = (
                        "samtools cat -h %s.header.sam -o /dev/stdout"
                        % basecaller_bam_filename
                    )
                    for blockbamfile in block_bam_list:
                        mergecmd = mergecmd + " %s" % blockbamfile
                    """
                    mergecmd = 'java -Xmx8g -jar ' + ion.picardPath + ' MergeSamFiles'
                    for blockbamfile in block_bam_list:
                        mergecmd = mergecmd + ' I=%s' % blockbamfile
                    mergecmd = mergecmd + ' O=/dev/stdout'
                    mergecmd = mergecmd + ' VERBOSITY=WARNING' # suppress INFO on stderr
                    mergecmd = mergecmd + ' QUIET=true' # suppress job-summary on stderr
                    mergecmd = mergecmd + ' VALIDATION_STRINGENCY=SILENT'
                    """
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
            cmd += (
                " -u --prefix-exclude 5"
            )  # random seed based on read name after ignoring first 5 characters
            if do_realign:
                cmd += " --do-realign"
            cmd += " -o 2"  # -o 0: SAM, -o 2: uncompressed BAM
            cmd += " %s" % tmap_stage_options
            cmd += " 2>> " + bamBase + ".alignmentQC_out.txt"  # logfile

        elif aligner == "bowtie2":
            referenceFastaDir = (
                "/results/referenceLibrary/bowtie2/"
                + referenceName
                + "/"
                + referenceName
            )
            cmd = "java -Xmx8g -jar %s SamToFastq I=%s F=/dev/stdout" % (
                ion.picardPath,
                basecaller_bam_filename,
            )
            cmd += (
                " | /results/plugins/bowtielauncher/bowtie2 -p%d -x %s -U /dev/stdin"
                % (threads, referenceFastaDir)
            )
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
                graph_max_x,
            )

            cmd += " | tee >(%s)" % ionstats_cmd

        # use number align_threads if smaller than 12
        samtool_threads = min(threads, 12)
        if do_sorting:
            if do_mark_duplicates:
                # use '-T' option to avoid temp file name collision among barcodes
                cmd += " | samtools sort -m 1000M -l1 -@%d -T tmp_%s -O bam -o - -" % (
                    samtool_threads,
                    bamBase,
                )
                json_name = (
                    "BamDuplicates.%s.json" % bamBase
                    if bamBase != "rawlib"
                    else "BamDuplicates.json"
                )
                cmd = "BamDuplicates -i <(%s) -o %s -j %s" % (cmd, bamFile, json_name)
            else:
                # use '-T' option to avoid temp file name collision among barcodes
                cmd += " | samtools sort -m 1000M -l1 -@{thread_num} -T tmp_{output_prefix} -O bam -o - - > {output_prefix}.bam".format(
                    thread_num=samtool_threads, output_prefix=bamBase
                )
        else:
            cmd += " > %s.bam" % bamBase

        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.Popen(["/bin/bash", "-c", cmd]).wait()
        if ret != 0:
            printtime("ERROR: alignment failed, return code: %d" % ret)
            raise RuntimeError("exit code: %d" % ret)

        # TODO: piping into samtools index or create index in sort process ?
        if do_indexing and do_sorting:
            cmd = "samtools index " + bamFile
            printtime("DEBUG: Calling '%s':" % cmd)
            subprocess.call(cmd, shell=True)

            """
            if do_indexing:
                try:
                    composite_bam_filepath = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam')
                    composite_bai_filepath = os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.bam.bai')
                    blockprocessing.create_index_file(composite_bam_filepath, composite_bai_filepath)
                except:
                    traceback.print_exc()
            """

    except Exception:
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
    align_threads,
    barcodeInfo,
):

    do_sorting = True

    try:
        # process block by block
        memTotalGb = _get_total_memory_gb()
        maxReads = 20000000
        if memTotalGb > 140:
            maxReads = 60000000
        if reference and len(blocks) > 1 and int(dataset["read_count"]) > maxReads:
            printtime(
                "DEBUG: TRADITIONAL BLOCK PROCESSING ------ prefix: %20s ----------- reference: %20s ---------- reads: %10s ----------"
                % (dataset["file_prefix"], reference, dataset["read_count"])
            )
            # start alignment for each block and current barcode with reads
            # TODO: in how many blocks are reads with this barcode
            for block in blocks:
                printtime("DEBUG: ALIGN ONLY ONE BLOCK: %s" % block)
                align(
                    [block],
                    os.path.join(BASECALLER_RESULTS, dataset["basecaller_bam"]),
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
                    output_basename=dataset["file_prefix"],
                    threads=align_threads,
                    barcode_info=barcodeInfo,
                )

            bamdir = "."  # TODO , do we need this ?
            bamBase = dataset["file_prefix"]
            bamfile = dataset["file_prefix"] + ".bam"

            block_bam_list = [
                os.path.join(blockdir, bamdir, bamfile) for blockdir in blocks
            ]
            block_bam_list = [
                block_bam_filename
                for block_bam_filename in block_bam_list
                if os.path.exists(block_bam_filename)
            ]
            printtime("blocks with reads:    %s" % len(block_bam_list))

            bamFile = dataset["file_prefix"] + ".bam"
            composite_bam_filepath = dataset["file_prefix"] + ".bam"

            blockprocessing.extract_and_merge_bam_header(
                block_bam_list, composite_bam_filepath
            )
            # Usage: samtools merge [-nr] [-h inh.sam] <out.bam> <in1.bam> <in2.bam> [...]
            cmd = "samtools merge -l1 -@8"
            if do_ionstats:
                cmd += " - "
            else:
                cmd += " %s" % (composite_bam_filepath)
            for bamfile in block_bam_list:
                cmd += " %s" % bamfile
            cmd += " -h %s.header.sam" % composite_bam_filepath

            if do_ionstats:
                bam_filenames = ["/dev/stdin"]
                ionstats_alignment_filename = (
                    "%s.ionstats_alignment.json" % bamBase
                )  # os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_alignment.json')
                ionstats_alignment_h5_filename = (
                    "%s.ionstats_error_summary.h5" % bamBase
                )  # os.path.join(ALIGNMENT_RESULTS, dataset['file_prefix']+'.ionstats_error_summary.h5')

                ionstats_cmd = ionstats.generate_ionstats_alignment_cmd(
                    ionstatsArgs,
                    bam_filenames,
                    ionstats_alignment_filename,
                    ionstats_alignment_h5_filename,
                    basecaller_meta_information,
                    library_key,
                    graph_max_x,
                )

                cmd += " | tee >(%s)" % ionstats_cmd

            if do_mark_duplicates:
                json_name = (
                    "BamDuplicates.%s.json" % bamBase
                    if bamBase != "rawlib"
                    else "BamDuplicates.json"
                )
                cmd = "BamDuplicates -i <(%s) -o %s -j %s" % (cmd, bamFile, json_name)
            else:
                cmd += " > %s.bam" % bamBase

            printtime("DEBUG: Calling '%s':" % cmd)
            ret = subprocess.Popen(["/bin/bash", "-c", cmd]).wait()
            if ret != 0:
                printtime("ERROR: merging failed, return code: %d" % ret)
                raise RuntimeError("exit code: %d" % ret)

            # TODO: piping into samtools index or create index in sort process ?
            if do_indexing and do_sorting:
                cmd = "samtools index " + bamFile
                printtime("DEBUG: Calling '%s':" % cmd)
                subprocess.call(cmd, shell=True)

        else:
            printtime(
                "DEBUG: MERGED BLOCK PROCESSING ----------- prefix: %20s ----------- reference: %20s ---------- reads: %10s ----------"
                % (dataset["file_prefix"], reference, dataset["read_count"])
            )
            # TODO: try a python multiprocessing pool
            align(
                blocks,
                os.path.join(BASECALLER_RESULTS, dataset["basecaller_bam"]),
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
                output_basename=dataset["file_prefix"],
                threads=align_threads,
                barcode_info=barcodeInfo,
            )
    except Exception:
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
    barcodeInfo,
):

    parallel_datasets = 1
    memTotalGb = _get_total_memory_gb()
    try:
        if memTotalGb > 140:
            parallel_datasets = 4
        elif memTotalGb >= 70:
            parallel_datasets = 2
    except Exception:
        pass

    align_threads = multiprocessing.cpu_count() / parallel_datasets
    if memTotalGb <= 40:
        # reduce number of CPU (1 vCPU = 2 cores)
        align_threads = align_threads - 2
    printtime("Attempt to align")
    printtime(
        "DEBUG: PROCESS DATASETS blocks: '%s', parallel datasets: %d"
        % (blocks, parallel_datasets)
    )

    # TODO: compare with pipeline/python/ion/utils/ionstats.py
    ionstats_basecaller_file_list = []
    ionstats_alignment_file_list = []
    ionstats_basecaller_filtered_file_list = []
    ionstats_alignment_filtered_file_list = []

    align_dataset_args = []

    for dataset in basecaller_datasets["datasets"]:

        read_group = dataset["read_groups"][0]
        reference = basecaller_datasets["read_groups"][read_group]["reference"]
        # print "DEBUG: reference: %s' % reference

        filtered = True
        for rg_name in dataset["read_groups"]:
            if not basecaller_datasets["read_groups"][rg_name].get("filtered", False):
                filtered = False

        # skip non-existing bam file
        if int(dataset["read_count"]) == 0:
            continue

        align_dataset_args.append(
            (
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
                align_threads,
                barcodeInfo,
            )
        )

        if reference:
            if filtered:
                ionstats_alignment_filtered_file_list.append(
                    os.path.join(
                        ALIGNMENT_RESULTS,
                        dataset["file_prefix"] + ".ionstats_alignment.json",
                    )
                )
            else:
                ionstats_alignment_file_list.append(
                    os.path.join(
                        ALIGNMENT_RESULTS,
                        dataset["file_prefix"] + ".ionstats_alignment.json",
                    )
                )
        else:
            if filtered:
                ionstats_basecaller_filtered_file_list.append(
                    os.path.join(
                        BASECALLER_RESULTS,
                        dataset["file_prefix"] + ".ionstats_basecaller.json",
                    )
                )
            else:
                ionstats_basecaller_file_list.append(
                    os.path.join(
                        BASECALLER_RESULTS,
                        dataset["file_prefix"] + ".ionstats_basecaller.json",
                    )
                )

    # do alignment in multiprocessing pool
    pool = multiprocessing.Pool(processes=parallel_datasets)
    pool.map(align_dataset_parallel_wrap, align_dataset_args)

    if do_ionstats:

        # Merge ionstats files from individual (barcoded) datasets
        if len(ionstats_alignment_file_list) > 0:
            ionstats.reduce_stats(
                ionstats_alignment_file_list,
                os.path.join(ALIGNMENT_RESULTS, "ionstats_alignment.json"),
            )
        else:  # barcode classification filtered all barcodes or no reads available
            # TODO: ionstats needs to produce initial json file
            try:
                # cmd = "echo $'@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"
                cmd = "echo  '@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"

                printtime("DEBUG: Calling '%s':" % cmd)
                ret = subprocess.call(cmd, shell=True)
                if ret != 0:
                    printtime(
                        "ERROR: empty bam file generation failed, return code: %d" % ret
                    )
                    raise RuntimeError("exit code: %d" % ret)

                ionstats.generate_ionstats_alignment(
                    ionstatsArgs,
                    ["empty_dummy.bam"],
                    os.path.join(ALIGNMENT_RESULTS, "ionstats_alignment.json"),
                    os.path.join(ALIGNMENT_RESULTS, "ionstats_error_summary.h5"),
                    basecaller_meta_information,
                    library_key,
                    graph_max_x,
                )

            except Exception:
                raise

        if len(ionstats_basecaller_file_list) > 0:
            ionstats.reduce_stats(
                ionstats_basecaller_file_list,
                os.path.join(BASECALLER_RESULTS, "ionstats_tmp_basecaller.json"),
            )
        else:  # barcode classification filtered all barcodes or no reads available
            # TODO: ionstats needs to produce initial json file
            try:
                # cmd = "echo $'@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"
                cmd = "echo  '@HD\tVN:1.5\tSO:coordinate\n@SQ\tSN:ref\tLN:4\n@RG\tID:filename\tSM:filename' | samtools view -F4 -S -b - > empty_dummy.bam"

                printtime("DEBUG: Calling '%s':" % cmd)
                ret = subprocess.call(cmd, shell=True)
                if ret != 0:
                    printtime(
                        "ERROR: empty bam file generation failed, return code: %d" % ret
                    )
                    raise RuntimeError("exit code: %d" % ret)

                ionstats.generate_ionstats_basecaller(
                    ["empty_dummy.bam"],
                    os.path.join(BASECALLER_RESULTS, "ionstats_tmp_basecaller.json"),
                    library_key,
                    graph_max_x,
                )
            except Exception:
                raise

        ionstatslist = []
        a = os.path.join(ALIGNMENT_RESULTS, "ionstats_alignment.json")
        b = os.path.join(BASECALLER_RESULTS, "ionstats_tmp_basecaller.json")
        if os.path.exists(a):
            ionstatslist.append(a)
        if os.path.exists(b):
            ionstatslist.append(b)
        if len(ionstatslist) > 0:
            ionstats.reduce_stats(
                ionstatslist,
                os.path.join(
                    BASECALLER_RESULTS, "ionstats_basecaller_with_aligninfos.json"
                ),
            )
            ionstats.reduce_stats(
                reversed(ionstatslist),
                os.path.join(BASECALLER_RESULTS, "ionstats_basecaller.json"),
            )
    #        if len(ionstats_alignment_h5_file_list) > 0:
    #            ionstats.reduce_stats_h5(ionstats_alignment_h5_file_list,os.path.join(ALIGNMENT_RESULTS,'ionstats_error_summary.h5'))

    printtime("**** Alignment completed ****")
