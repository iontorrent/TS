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
from ion.utils import TFPipeline
from ion.utils import ionstats_plots

from ion.utils.blockprocessing import printtime

from torrentserver import cluster_settings

from ion.reports import wells_beadogram

from ion.utils.compress import make_zip

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-s', '--do-sigproc', dest='do_sigproc', action='store_true', help='signal processing')
    parser.add_argument('-b', '--do-basecalling', dest='do_basecalling', action='store_true', help='base calling')
    parser.add_argument('-a', '--do-alignment', dest='do_alignment', action='store_true', help='alignment')
    parser.add_argument('-z', '--do-zipping', dest='do_zipping', action='store_true', help='zipping')

    args = parser.parse_args()

    if args.verbose:
        print "MergeTLScript:",args

    if not args.do_sigproc and not args.do_basecalling and not args.do_zipping:
        parser.print_help()
        sys.exit(1)

    #ensure we permit read/write for owner and group output files.
    os.umask(0002)

    blockprocessing.printheader()
    env,warn = explogparser.getparameter()

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
        primary_key_file = os.path.join(os.getcwd(),'primary.key')
    except:
        traceback.print_exc()


    is_thumbnail = False
    is_single = False
    is_composite = False

    if env['rawdatastyle'] == 'single':
        is_single = True
    else:
        if "thumbnail" in env['pathToRaw']:
           is_thumbnail = True
        else:
           is_composite = True

    do_unfiltered_processing = is_thumbnail or is_single
    reference_selected = env['referenceName'] and env['referenceName']!='none'

    def set_result_status(status):
        try:
            if os.path.exists(primary_key_file):
                jobserver.updatestatus(primary_key_file, status, True)
                printtime("MergeTLStatus %s\tpid %d\tpk file %s started" % 
                    (status, os.getpid(), primary_key_file))
        except:
            traceback.print_exc()


    blocks = explogparser.getBlocksFromExpLogJson(env['exp_json'], excludeThumbnail=True)
    dirs = ['block_%s' % block['id_str'] for block in blocks]

    do_merged_alignment = False

    if args.do_sigproc:

        set_result_status('Merge Heatmaps')
        
        chipType = env.get('chipType','')
        exclusionMaskFile = ''
        if chipType.startswith('P1.1'):
            exclusionMaskFile = 'exclusionMask_P1_1.txt'
        elif chipType.startswith('P1.0'):
            exclusionMaskFile = 'exclusionMask_P1_0.txt'

        sigproc.mergeSigProcResults(
            dirs,
            env['SIGPROC_RESULTS'],
            env['shortRunName'],
            exclusionMaskFile)


    if args.do_basecalling:

        set_result_status('Merge Basecaller Results')

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
            basecaller.merge_basecaller_stats(
                dirs,
                env['BASECALLER_RESULTS'])
        except:
            traceback.print_exc()
            printtime("ERROR: Merge Basecaller Results failed")

        try:
            c = open(os.path.join(env['BASECALLER_RESULTS'], "BaseCaller.json"),'r')
            basecaller_meta_information = json.load(c)
            c.close()
        except:
            traceback.print_exc()
            raise

        basecaller_datasets = blockprocessing.get_datasets_basecaller(env['BASECALLER_RESULTS'])


        try:
            RECALIBRATION_RESULTS = os.path.join(env['BASECALLER_RESULTS'],"recalibration")
            if not os.path.isdir(RECALIBRATION_RESULTS):
                os.makedirs(RECALIBRATION_RESULTS)            
            cmd = "calibrate --hpmodelMerge"
            printtime("DEBUG: Calling '%s':" % cmd)
            ret = subprocess.call(cmd,shell=True)
        except:
            traceback.print_exc()
            printtime("ERROR: Merge Basecaller Results failed")


        try:
            graph_max_x = int(50 * math.ceil(0.014 * int(env['flows'])))
        except:
            traceback.print_exc()
            graph_max_x = 400

        try:
            printtime("INFO: merging rawtf.basecaller.bam")
            block_bam_list = [os.path.join(adir, env['BASECALLER_RESULTS'], 'rawtf.basecaller.bam') for adir in dirs]
            block_bam_list = [block_bam_filename for block_bam_filename in block_bam_list if os.path.exists(block_bam_filename)]
            composite_bam_filename = os.path.join(env['BASECALLER_RESULTS'], 'rawtf.basecaller.bam')
            if block_bam_list:
                blockprocessing.merge_bam_files(block_bam_list,composite_bam_filename,composite_bai_filepath="",mark_duplicates=False,method='picard')
        except:
            print traceback.format_exc()
            printtime("ERROR: merging rawtf.basecaller.bam unsuccessful")

        if do_unfiltered_processing:
            try:
                os.mkdir(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed'))

                basecaller.merge_datasets_basecaller_json(
                    dirs,
                    os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed"))

                basecaller.merge_bams(
                    dirs,
                    os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed"),
                    basecaller_datasets,
                    'picard')
            except:
                print traceback.format_exc()


            try:
                os.mkdir(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed'))

                basecaller.merge_datasets_basecaller_json(
                    dirs,
                    os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed"))

                basecaller.merge_bams(
                    dirs,
                    os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed"),
                    basecaller_datasets,
                    'picard')
            except:
                print traceback.format_exc()


    if args.do_alignment:
        # at least one barcode has a reference
        if not reference_selected:
            printtime ("INFO: reference not selected")

        #make sure pre-conditions for alignment step are met
        if not os.path.exists(os.path.join(env['BASECALLER_RESULTS'], "datasets_basecaller.json")):
            printtime ("ERROR: alignment pre-conditions not met")
            sys.exit(1)

        basecaller_datasets = blockprocessing.get_datasets_basecaller(env['BASECALLER_RESULTS'])

        if do_merged_alignment:

            set_result_status('Alignment')

            create_index = True

            try:
                bidirectional = False
                activate_barcode_filter = True
                alignment.alignment_unmapped_bam(
                    env['BASECALLER_RESULTS'],
                    basecaller_datasets,
                    env['ALIGNMENT_RESULTS'],
                    env['realign'],
                    env['aligner_opts_extra'],
                    env['mark_duplicates'],
                    create_index,
                    bidirectional,
                    activate_barcode_filter,
                    env['barcodeInfo'])
#                add_status("Alignment", 0)
            except:
                traceback.print_exc()
#                add_status("Alignment", 1)

        else:

            set_result_status('Merge Bam Files')

            # this includes indexing with samtools
            if reference_selected:
                alignment.merge_bams(
                    dirs,
                    env['BASECALLER_RESULTS'],
                    env['ALIGNMENT_RESULTS'],
                    basecaller_datasets,
                    env['mark_duplicates'])
            else:
                basecaller.merge_bams(
                    dirs,
                    env['BASECALLER_RESULTS'],
                    basecaller_datasets,
                    'samtools')


    if args.do_zipping:

        set_result_status('Create Statistics')

        try:
            c = open(os.path.join(env['BASECALLER_RESULTS'], "BaseCaller.json"),'r')
            basecaller_meta_information = json.load(c)
            c.close()
        except:
            traceback.print_exc()
            raise
        if is_composite:
            basecaller_meta_information = None

        basecaller_datasets = blockprocessing.get_datasets_basecaller(env['BASECALLER_RESULTS'])

        try:
            graph_max_x = int(50 * math.ceil(0.014 * int(env['flows'])))
        except:
            traceback.print_exc()
            graph_max_x = 400

        # read length histograms require all reads
        activate_barcode_filter = False

        try:
            alignment.create_ionstats(
                    env['BASECALLER_RESULTS'],
                    env['ALIGNMENT_RESULTS'],
                    basecaller_meta_information,
                    basecaller_datasets,
                    graph_max_x,
                    activate_barcode_filter)
        except:
            traceback.print_exc()


        if is_composite and os.path.exists('/opt/ion/.ion-internal-server'):

            try:
                set_result_status('Merge Statistics')
                alignment.merge_ionstats(
                    dirs,
                    env['BASECALLER_RESULTS'],
                    env['ALIGNMENT_RESULTS'],
                    basecaller_datasets)
            except:
                traceback.print_exc()


        try:

            # generate sparklines for each barcode based on TODO (which file?)
            alignment.plot_main_report_histograms(
                env['BASECALLER_RESULTS'],
                env['ALIGNMENT_RESULTS'],
                basecaller_datasets,
                graph_max_x)

            # Plot classic read length histogram (also used for Read Length Details view)
            ionstats_plots.old_read_length_histogram(
                os.path.join(env['BASECALLER_RESULTS'],'ionstats_basecaller.json'),
                os.path.join(env['BASECALLER_RESULTS'],'readLenHisto.png'),
                graph_max_x)

        except:
            traceback.print_exc()

        if reference_selected:
            try:
                alignment.create_plots('ionstats_alignment.json', graph_max_x)
            except:
                traceback.print_exc()

        try:
            wells_beadogram.generate_wells_beadogram(env['BASECALLER_RESULTS'], env['SIGPROC_RESULTS'])
        except:
            printtime ("ERROR: Wells beadogram generation failed")
            traceback.print_exc()

        set_result_status('TF Processing')

        try:
            TFPipeline.processBlock(
                os.path.join(env['BASECALLER_RESULTS'], 'rawtf.basecaller.bam'),
                env['BASECALLER_RESULTS'],
                env['tfKey'],
                env['flowOrder'],
                '.')
            #add_status("TF Processing", 0)
        except:
            traceback.print_exc()
            #add_status("TF Processing", 1)


        # Process unfiltered reads

        if do_unfiltered_processing:
            set_result_status('Process Unfiltered BAM')

            bidirectional = False
            activate_barcode_filter = False
            create_index = False

            for unfiltered_directory in [
                                         os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed'),
                                         os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed')
                                        ]:
                try:

                    if os.path.exists(unfiltered_directory):

                        unfiltered_basecaller_datasets = blockprocessing.get_datasets_basecaller(unfiltered_directory)

                        # TODO, don't generate this file
                        if env['barcodeId']:
                            basecaller.merge_barcoded_basecaller_bams(
                                unfiltered_directory,
                                unfiltered_basecaller_datasets,
                                'picard')

                        if reference_selected:
                            alignment.alignment_unmapped_bam(
                                unfiltered_directory,
                                unfiltered_basecaller_datasets,
                                unfiltered_directory,
                                env['realign'],
                                env['aligner_opts_extra'],
                                env['mark_duplicates'],
                                create_index,
                                bidirectional,
                                activate_barcode_filter,
                                env['barcodeInfo'])

                            # TODO, don't generate this file
                            if env['barcodeId']:
                                alignment.merge_barcoded_alignment_bams(
                                    unfiltered_directory,
                                    unfiltered_basecaller_datasets,
                                    'picard')

                except:
                    traceback.print_exc()


        # Legacy post-processing. Generate merged rawlib.bam on barcoded runs

        if is_thumbnail or is_single:

            if env['barcodeId'] and reference_selected:

                try:
                    alignment.merge_barcoded_alignment_bams(
                        env['ALIGNMENT_RESULTS'],
                        basecaller_datasets,
                        'samtools')
                except:
                    traceback.print_exc()


        # Create links with official names to all downloadable data files

        set_result_status('Create Download Links')

        download_links = 'download_links'

        try:
            os.mkdir(download_links)
        except:
            printtime(traceback.format_exc())

        prefix_list = [dataset['file_prefix'] for dataset in basecaller_datasets.get("datasets",[])]

        link_task_list = [
            ('bam',             env['ALIGNMENT_RESULTS']),
            ('bam.bai',         env['ALIGNMENT_RESULTS']),
            ('basecaller.bam',  env['BASECALLER_RESULTS']),]

        for extension,base_dir in link_task_list:
            for prefix in prefix_list:
                try:
                    filename = "%s/%s%s_%s.%s" % (download_links, re.sub('rawlib$','',prefix), env['expName'], env['resultsName'], extension)
                    src = os.path.join(base_dir, prefix+'.'+extension)
                    if os.path.exists(src):
                        os.symlink(os.path.relpath(src,os.path.dirname(filename)),filename)
                except:
                    printtime("ERROR: target: %s" % filename)
                    traceback.print_exc()


        src = os.path.join(env['BASECALLER_RESULTS'], 'rawtf.bam')
        dst = os.path.join(download_links, "%s_%s.rawtf.bam" % (env['expName'], env['resultsName']))
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                os.symlink(os.path.relpath(src,os.path.dirname(dst)),dst)
            except:
                printtime("ERROR: Unable to symlink '%s' to '%s'" % (src, dst))
                printtime(traceback.format_exc())


    printtime("MergeTLScript exit")
    sys.exit(0)
