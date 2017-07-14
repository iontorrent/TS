#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

import os
import sys
import argparse
import traceback
import json
import subprocess
import xmlrpclib
import math
import re

from ion.utils import blockprocessing
from ion.utils import explogparser
from ion.utils import sigproc
from ion.utils import basecaller
from ion.utils import alignment
from ion.utils import ionstats_plots
from ion.utils import ionstats

from ion.utils.blockprocessing import printtime

sys.path.append('/etc')
from torrentserver import cluster_settings

from ion.reports import wells_beadogram

from ion.utils.compress import make_zip

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-s', '--do-sigproc', dest='do_sigproc',
                        action='store_true', help='signal processing')
    parser.add_argument('-b', '--do-basecalling',
                        dest='do_basecalling', action='store_true', help='base calling')
    parser.add_argument('-a', '--do-alignment', dest='do_alignment', action='store_true', help='alignment')
    parser.add_argument('-z', '--do-zipping', dest='do_zipping', action='store_true', help='zipping')

    args = parser.parse_args()

    if args.verbose:
        print "MergeTLScript:", args

    if not args.do_sigproc and not args.do_basecalling and not args.do_zipping:
        parser.print_help()
        sys.exit(1)

    # ensure we permit read/write for owner and group output files.
    os.umask(0002)

    blockprocessing.printheader()
    env, warn = explogparser.getparameter()

    blockprocessing.write_version()
    sys.stdout.flush()
    sys.stderr.flush()

    #-------------------------------------------------------------
    # Connect to Job Server
    #-------------------------------------------------------------
    try:
        jobserver = xmlrpclib.ServerProxy("http://%s:%d" %
                                         (cluster_settings.JOBSERVER_HOST, cluster_settings.JOBSERVER_PORT),
                                          verbose=False, allow_none=True)
        primary_key_file = os.path.join(os.getcwd(), 'primary.key')
    except:
        traceback.print_exc()

    if env['rawdatastyle'] == 'single' or "thumbnail" in env['pathToRaw']:
        is_composite = False
    else:
        is_composite = True

    # Speedup Flags
    do_unfiltered_processing = True and not is_composite
    do_merge_all_barcodes = True and not is_composite
    do_ionstats_spatial_accounting = False

    reference_selected = False
    for barcode_name, barcode_info in sorted(env['barcodeInfo'].iteritems()):
        if barcode_info['referenceName']:
            reference_selected = True
            break

    try:
        graph_max_x = int(50 * math.ceil(0.014 * int(env['flows'])))
    except:
        traceback.print_exc()
        graph_max_x = 400

    def set_result_status(status):
        try:
            if os.path.exists(primary_key_file):
                jobserver.updatestatus(primary_key_file, status, True)
                printtime("MergeTLStatus %s\tpid %d\tpk file %s started" %
                         (status, os.getpid(), primary_key_file))
        except:
            traceback.print_exc()

    explogblocks = explogparser.getBlocksFromExpLogJson(env['exp_json'], excludeThumbnail=True)
    blocks_to_process = []
    for block in explogblocks:
        toProcess = block['autoanalyze'] and block['analyzeearly']
        if env.get('chipBlocksOverride') and toProcess:
            if env['chipBlocksOverride'] == '510':
                toProcess = block['id_str'].endswith('Y0')
        if toProcess:
            blocks_to_process.append(block)

    number_of_total_blocks = len(blocks_to_process)
    dirs = ['block_%s' % block['id_str'] for block in blocks_to_process]
    if not is_composite:
        dirs = []

    if args.do_sigproc:

        set_result_status('Merge Heatmaps')

        # In 5.4 we apply a  mask in the bead find stage but also here to be backward compatible
        # with from-basecall reanalyses of 5.2 and older OIA results
        exclusionMaskFile = 'exclusionMask_%s.txt' % env.get('chipType', '').lower()

        sigproc.mergeSigProcResults(
            dirs,
            env['SIGPROC_RESULTS'],
            env['shortRunName'],
            exclusionMaskFile)

        '''
        # write composite return code, not needed anymore ?
        try:
            composite_return_code=number_of_total_blocks
            for subdir in dirs:

                blockstatus_return_code_file = os.path.join(subdir,"blockstatus.txt")
                if os.path.exists(blockstatus_return_code_file):

                    with open(blockstatus_return_code_file, 'r') as f:
                        text = f.read()
                        if 'Analysis=0' in text:
                            composite_return_code-=1

            composite_return_code_file = os.path.join(SIGPROC_RESULTS,"analysis_return_code.txt")
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
        '''

    if args.do_basecalling:

        set_result_status('Merge Basecaller Results')

        # write composite return code
        try:
            composite_return_code = number_of_total_blocks
            for subdir in dirs:

                blockstatus_return_code_file = os.path.join(subdir, "blockstatus.txt")
                if os.path.exists(blockstatus_return_code_file):

                    with open(blockstatus_return_code_file, 'r') as f:
                        text = f.read()
                        if 'Basecaller=0' in text:
                            composite_return_code -= 1
                        else:
                            with open(os.path.join(subdir, "sigproc_results", "analysis_return_code.txt"), 'r') as g:
                                return_code_text = g.read()
                                # TODO
                                corner_P1_blocks = [
                                    'block_X0_Y0', 'block_X14168_Y0', 'block_X0_Y9324', 'block_X14168_Y9324']
                                corner_P0_blocks = [
                                    'block_X0_Y0', 'block_X7040_Y0', 'block_X0_Y4648', 'block_X7040_Y4648']
                                if return_code_text == "3" and subdir in corner_P0_blocks + corner_P1_blocks:
                                    printtime("INFO: suppress non-critical error in %s" % subdir)
                                    composite_return_code -= 1

            composite_return_code_file = os.path.join(env['BASECALLER_RESULTS'], "composite_return_code.txt")
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

        # update blocks to process
        try:
            newlist = []
            for subdir in dirs:

                analysis_return_code_file = os.path.join(
                    subdir, env['SIGPROC_RESULTS'], "analysis_return_code.txt")
                if os.path.exists(analysis_return_code_file):
                    with open(analysis_return_code_file, 'r') as f:
                        text = f.read()
                        if '0' == text:
                            newlist.append(subdir)
            dirs = newlist
        except:
            traceback.print_exc()

        try:
            sigproc.mergeAvgNukeTraces(dirs, env['SIGPROC_RESULTS'], env['libraryKey'], 'Library Beads')
        except:
            printtime("Warning: mergeAvgNukeTraces '%s' 'Library Beads' failed" % env['libraryKey'])

        try:
            sigproc.mergeAvgNukeTraces(dirs, env['SIGPROC_RESULTS'], env['tfKey'], 'Test Fragment Beads')
        except:
            printtime("Warning: mergeAvgNukeTraces '%s' 'Test Fragment Beads' failed" % env['tfKey'])

        try:
            sigproc.generate_raw_data_traces(
                env['libraryKey'],
                env['tfKey'],
                env['flowOrder'],
                env['SIGPROC_RESULTS'])
        except:
            traceback.print_exc()

        try:
            # Only merge standard json files
            basecaller.merge_datasets_basecaller_json(
                dirs,
                env['BASECALLER_RESULTS'])

            basecaller.merge_basecaller_json(
                dirs,
                env['BASECALLER_RESULTS'])
        except:
            traceback.print_exc()
            printtime("ERROR: Merge Basecaller Results failed")

        try:
            printtime("INFO: merging rawtf.basecaller.bam")
            block_bam_list = [os.path.join(adir, env['BASECALLER_RESULTS'], 'rawtf.basecaller.bam')
                              for adir in dirs]
            block_bam_list = [
                block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
            composite_bam_filename = os.path.join(env['BASECALLER_RESULTS'], 'rawtf.basecaller.bam')
            if block_bam_list:
                blockprocessing.merge_bam_files(
                    block_bam_list, composite_bam_filename, composite_bai_filepath="", mark_duplicates=False, method='picard')
        except:
            print traceback.format_exc()
            printtime("ERROR: merging rawtf.basecaller.bam unsuccessful")

        if do_unfiltered_processing:

            basecaller_datasets = blockprocessing.get_datasets_basecaller(env['BASECALLER_RESULTS'])

            try:
                os.mkdir(os.path.join(env['BASECALLER_RESULTS'], 'unfiltered.untrimmed'))

                basecaller.merge_datasets_basecaller_json(
                    dirs,
                    os.path.join(env['BASECALLER_RESULTS'], "unfiltered.untrimmed"))

                blockprocessing.merge_bams(
                    dirs,
                    os.path.join(env['BASECALLER_RESULTS'], "unfiltered.untrimmed"),
                    basecaller_datasets,
                    'picard')
            except:
                print traceback.format_exc()

            try:
                os.mkdir(os.path.join(env['BASECALLER_RESULTS'], 'unfiltered.trimmed'))

                basecaller.merge_datasets_basecaller_json(
                    dirs,
                    os.path.join(env['BASECALLER_RESULTS'], "unfiltered.trimmed"))

                blockprocessing.merge_bams(
                    dirs,
                    os.path.join(env['BASECALLER_RESULTS'], "unfiltered.trimmed"),
                    basecaller_datasets,
                    'picard')
            except:
                print traceback.format_exc()

    if args.do_zipping:
        # at least one barcode has a reference
        if not reference_selected:
            printtime("INFO: reference not selected")

        # make sure pre-conditions for alignment step are met
        if not os.path.exists(os.path.join(env['BASECALLER_RESULTS'], "datasets_basecaller.json")):
            printtime("ERROR: alignment pre-conditions not met")
            sys.exit(1)

        basecaller_datasets = blockprocessing.get_datasets_basecaller(env['BASECALLER_RESULTS'])

        try:
            c = open(os.path.join(env['BASECALLER_RESULTS'], "BaseCaller.json"), 'r')
            basecaller_meta_information = json.load(c)
            c.close()
        except:
            traceback.print_exc()
            raise
        if is_composite or not do_ionstats_spatial_accounting:
            basecaller_meta_information = None

        if reference_selected:
            set_result_status('Alignment')

        do_ionstats = True
        do_indexing = True

        alignment.process_datasets(
            dirs,
            env['alignmentArgs'],
            env['ionstatsArgs'],
            env['BASECALLER_RESULTS'],
            basecaller_meta_information,
            env['libraryKey'],
            graph_max_x,
            basecaller_datasets,
            env['ALIGNMENT_RESULTS'],
            env['realign'],
            do_ionstats,
            env['mark_duplicates'],
            do_indexing,
            env['barcodeInfo'])

        '''

    ### OLD per block processing
    if args.do_alignment:

        try:
            c = open(os.path.join(env['BASECALLER_RESULTS'], "BaseCaller.json"),'r')
            basecaller_meta_information = json.load(c)
            c.close()
        except:
            traceback.print_exc()
            raise

        basecaller_datasets = blockprocessing.get_datasets_basecaller(env['BASECALLER_RESULTS'])

        set_result_status('Alignment')

        if isBlock:
            do_indexing=False
            do_ionstats=True
        else:
            do_indexing=True
            do_ionstats=True

        try:
            blocks=[]
            alignment.process_datasets(
                blocks,
                env['alignmentArgs'],
                env['ionstatsArgs'],
                env['BASECALLER_RESULTS'],
                basecaller_meta_information if do_ionstats_spatial_accounting else None,
                env['libraryKey'],
                graph_max_x,
                basecaller_datasets,
                env['ALIGNMENT_RESULTS'],
                env['realign'],
                do_ionstats,
                env['mark_duplicates'],
                do_indexing,
                env['barcodeInfo'])
            add_status("Alignment", 0)
        except:
            traceback.print_exc()
            add_status("Alignment", 1)
            printtime ("ERROR: Alignment failed")
            sys.exit(1)
    ### OLD per block processing

        else:

            set_result_status('Merge Bam Files')

            # this includes indexing with samtools if reference is specified
            blockprocessing.merge_bams(
                dirs,
                env['BASECALLER_RESULTS'],
                env['ALIGNMENT_RESULTS'],
                basecaller_datasets,
                env['mark_duplicates'])


            # generates:
            #     ionstats_basecaller.json
            #     ionstats_alignment.json
            # compare with merged results
            # '' '
            try:
                set_result_status('Create Statistics')
                ionstats.create_ionstats(
                    env['ionstatsArgs'],
                    env['BASECALLER_RESULTS'],
                    env['ALIGNMENT_RESULTS'],
                    basecaller_meta_information,
                    basecaller_datasets,
                    env['libraryKey'],
                    graph_max_x,
                    do_ionstats_spatial_accounting)
            except:
                traceback.print_exc()
            # '' '


            # generates:
            #     ionstats_basecaller.json
            #     ionstats_alignment.json
            #     ionstats_error_summary.h5

            try:
                set_result_status('Merge Statistics')
                ionstats.merge_ionstats(
                    dirs,
                    env['BASECALLER_RESULTS'],
                    env['ALIGNMENT_RESULTS'],
                    basecaller_datasets,
                    basecaller_meta_information,
                    env['ionstatsArgs'],
                    env['libraryKey'],
                    graph_max_x)
            except:
                traceback.print_exc()
        '''

        try:

            # Plot classic read length histogram (also used for Read Length Details view)
            ionstats_plots.old_read_length_histogram(
                os.path.join(env['BASECALLER_RESULTS'], 'ionstats_basecaller.json'),
                os.path.join(env['BASECALLER_RESULTS'], 'readLenHisto.png'),
                graph_max_x)

            # Plot new read length histogram
            ionstats_plots.read_length_histogram(
                os.path.join(env['BASECALLER_RESULTS'], 'ionstats_basecaller.json'),
                os.path.join(env['BASECALLER_RESULTS'], 'readLenHisto2.png'),
                graph_max_x)

            for dataset in basecaller_datasets["datasets"]:

                if int(dataset["read_count"]) == 0:
                    continue

                read_group = dataset['read_groups'][0]
                reference = basecaller_datasets['read_groups'][read_group]['reference']

                if reference:
                    ionstats_folder = env['ALIGNMENT_RESULTS']
                    ionstats_file = 'ionstats_alignment.json'
                else:
                    ionstats_folder = env['BASECALLER_RESULTS']
                    ionstats_file = 'ionstats_basecaller.json'

                # Plot read length sparkline
                ionstats_plots.read_length_sparkline(
                    os.path.join(ionstats_folder, dataset['file_prefix']+'.'+ionstats_file),
                    os.path.join(env['BASECALLER_RESULTS'], dataset['file_prefix']+'.sparkline.png'),
                    graph_max_x)

                # Plot higher detail barcode specific histogram
                ionstats_plots.old_read_length_histogram(
                    os.path.join(ionstats_folder, dataset['file_prefix']+'.'+ionstats_file),
                    os.path.join(env['BASECALLER_RESULTS'], dataset['file_prefix']+'.read_len_histogram.png'),
                    graph_max_x)

        except:
            traceback.print_exc()

        if reference_selected:
            try:
                # Use ionstats alignment results to generate plots
                ionstats_plots.alignment_rate_plot2(
                    'ionstats_alignment.json', 'alignment_rate_plot.png', graph_max_x)
                ionstats_plots.base_error_plot('ionstats_alignment.json', 'base_error_plot.png', graph_max_x)
                ionstats_plots.old_aq_length_histogram(
                    'ionstats_alignment.json', 'Filtered_Alignments_Q10.png', 'AQ10', 'red')
                ionstats_plots.old_aq_length_histogram(
                    'ionstats_alignment.json', 'Filtered_Alignments_Q17.png', 'AQ17', 'yellow')
                ionstats_plots.old_aq_length_histogram(
                    'ionstats_alignment.json', 'Filtered_Alignments_Q20.png', 'AQ20', 'green')
                ionstats_plots.old_aq_length_histogram(
                    'ionstats_alignment.json', 'Filtered_Alignments_Q47.png', 'AQ47', 'purple')
            except:
                traceback.print_exc()

        try:
            wells_beadogram.generate_wells_beadogram(env['BASECALLER_RESULTS'], env['SIGPROC_RESULTS'])
        except:
            printtime("ERROR: Wells beadogram generation failed")
            traceback.print_exc()

        set_result_status('TF Processing')

        try:
            # TODO basecaller_results/datasets_tf.json might contain read_count : 0
            if os.path.exists(os.path.join(env['BASECALLER_RESULTS'], 'rawtf.basecaller.bam')):

                # input
                tf_basecaller_bam_filename = os.path.join(env['BASECALLER_RESULTS'], 'rawtf.basecaller.bam')
                tf_reference_filename = os.path.join(
                    "/results/referenceLibrary/TestFragment", env['tfKey'], "DefaultTFs.fasta")

                # These files will be created
                tfbam_filename = os.path.join(env['BASECALLER_RESULTS'], "rawtf.bam")
                ionstats_tf_filename = os.path.join(env['BASECALLER_RESULTS'], "ionstats_tf.json")
                tfstatsjson_path = os.path.join(env['BASECALLER_RESULTS'], "TFStats.json")

                printtime("TF: Mapping '%s'" % tf_basecaller_bam_filename)
                alignment.alignTFs(tf_basecaller_bam_filename, tfbam_filename, tf_reference_filename)

                ionstats.generate_ionstats_tf(tfbam_filename, tf_reference_filename, ionstats_tf_filename)

                ionstats_plots.tf_length_histograms(ionstats_tf_filename, '.')

                ionstats.generate_legacy_tf_files(ionstats_tf_filename, tfstatsjson_path)

        except:
            traceback.print_exc()
            printtime("No data to analyze Test Fragments")
            f = open(os.path.join(env['BASECALLER_RESULTS'], 'TFStats.json'), 'w')
            f.write(json.dumps({}))
            f.close()

        # Process unfiltered reads
        if do_unfiltered_processing:
            set_result_status('Process Unfiltered BAM')

            do_ionstats = False
            create_index = False

            for unfiltered_directory in [
                os.path.join(env['BASECALLER_RESULTS'], 'unfiltered.untrimmed'),
                os.path.join(env['BASECALLER_RESULTS'], 'unfiltered.trimmed')
            ]:
                try:

                    if os.path.exists(unfiltered_directory):

                        unfiltered_basecaller_meta_information = None
                        unfiltered_basecaller_datasets = blockprocessing.get_datasets_basecaller(
                            unfiltered_directory)

                        # TODO, don't generate this file
                        if env['barcodeId']:
                            basecaller.merge_barcoded_basecaller_bams(
                                unfiltered_directory,
                                unfiltered_basecaller_datasets,
                                'picard')

                        if reference_selected:
                            alignment.process_datasets(
                                dirs,
                                env['alignmentArgs'],
                                env['ionstatsArgs'],
                                unfiltered_directory,
                                unfiltered_basecaller_meta_information,
                                env['libraryKey'],
                                graph_max_x,
                                unfiltered_basecaller_datasets,
                                unfiltered_directory,
                                env['realign'],
                                env['mark_duplicates'],
                                do_ionstats,
                                create_index,
                                env['barcodeInfo'])

                            '''
                            # Legacy post-processing. Generate merged rawlib.bam on barcoded runs
                            # Incompatible with multiple references
                            if do_merge_all_barcodes and env['barcodeId']:
                                alignment.merge_barcoded_alignment_bams(
                                    unfiltered_directory,
                                    unfiltered_basecaller_datasets,
                                    'picard')
                            '''

                except:
                    traceback.print_exc()

        '''
        # Legacy post-processing. Generate merged rawlib.bam on barcoded runs
        # Incompatible with multiple references

        if do_merge_all_barcodes:

            if env['barcodeId'] and reference_selected:

                try:
                    alignment.merge_barcoded_alignment_bams(
                        env['ALIGNMENT_RESULTS'],
                        basecaller_datasets,
                        'samtools')
                except:
                    traceback.print_exc()
        '''

        #'''
        # move frequency filtered barcodes into subdirectory
        try:
            os.mkdir("frequency_filtered_barcodes")
        except:
            printtime(traceback.format_exc())

        import glob
        import shutil

        for dataset in basecaller_datasets["datasets"]:

            filtered = True  # default
            for rg_name in dataset["read_groups"]:
                if not basecaller_datasets["read_groups"][rg_name].get('filtered', False):
                    filtered = False
            if filtered and int(dataset["read_count"]) > 0:
                try:
                    printtime("INFO FILTERED: %s" % dataset)
                    for data in glob.glob(dataset['file_prefix']+".*"):
                        shutil.move(data, "frequency_filtered_barcodes")
                except:
                    printtime(traceback.format_exc())
        #'''

        # Create links with official names to all downloadable data files

        set_result_status('Create Download Links')

        download_links = 'download_links'

        try:
            os.mkdir(download_links)
        except:
            printtime(traceback.format_exc())

        prefix_list = [dataset['file_prefix'] for dataset in basecaller_datasets.get("datasets", [])]

        link_task_list = [
            ('bam',             env['ALIGNMENT_RESULTS']),
            ('bam.bai',         env['ALIGNMENT_RESULTS']),
            ('basecaller.bam',  env['BASECALLER_RESULTS']), ]

        for extension, base_dir in link_task_list:
            for prefix in prefix_list:
                try:
                    filename = "%s/%s%s_%s.%s" % (download_links, re.sub(
                        'rawlib$', '', prefix), env['expName'], env['resultsName'], extension)
                    src = os.path.join(base_dir, prefix+'.'+extension)
                    if os.path.exists(src):
                        os.symlink(os.path.relpath(src, os.path.dirname(filename)), filename)
                except:
                    printtime("ERROR: target: %s" % filename)
                    traceback.print_exc()

        src = os.path.join(env['BASECALLER_RESULTS'], 'rawtf.bam')
        dst = os.path.join(download_links, "%s_%s.rawtf.bam" % (env['expName'], env['resultsName']))
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
            except:
                printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))
                printtime(traceback.format_exc())

    printtime("MergeTLScript exit")
    sys.exit(0)
