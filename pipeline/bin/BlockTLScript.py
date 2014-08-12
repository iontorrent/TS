#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

import os
import sys
import glob
import subprocess
import argparse
import shutil
import traceback
import fnmatch
import xmlrpclib
import json

from ion.reports import beadDensityPlot
from ion.utils import blockprocessing
from ion.utils import explogparser
from ion.utils import sigproc
from ion.utils import basecaller
from ion.utils import alignment
from ion.utils import flow_space_recal
from ion.utils import ionstats_plots

from ion.utils.blockprocessing import printtime

from torrentserver import cluster_settings

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-s', '--do-sigproc', dest='do_sigproc', action='store_true', help='signal processing')
    parser.add_argument('-b', '--do-basecalling', dest='do_basecalling', action='store_true', help='base calling')
    parser.add_argument('-a', '--do-alignment', dest='do_alignment', action='store_true', help='alignment')

    args = parser.parse_args()

    if args.verbose:
        print "BlockTLScript:",args

    if not args.do_sigproc and not args.do_basecalling and not args.do_alignment:
        parser.print_help()
        sys.exit(1)

    #ensure we permit read/write for owner and group output files.
    os.umask(0002)

    blockprocessing.printheader()
    env,warn = explogparser.getparameter()
    print warn

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

    reference_selected = env['referenceName'] and env['referenceName']!='none'

    if os.path.exists(primary_key_file):
        isProtonBlock = False
    else:
        isProtonBlock = True


    def set_result_status(status):
        try:
            if not isProtonBlock:
                if status == 'Processing finished':
                    # filter out this status (31x and Thumbnail)
                    return
                jobserver.updatestatus(primary_key_file, status, True)
                printtime("BlockTLStatus %s\tpid %d\tpk file %s started" % 
                    (status, os.getpid(), primary_key_file))
            else: # Proton Block
                if status == 'Beadfind':
                    f = open('progress.txt','w')
                    f.write('wellfinding = yellow\n')
                    f.write('signalprocessing = grey\n')
                    f.write('basecalling = grey\n')
                    f.write('alignment = grey')
                    f.close()
                elif status == 'Signal Processing':
                    f = open('progress.txt','w')
                    f.write('wellfinding = green\n')
                    f.write('signalprocessing = yellow\n')
                    f.write('basecalling = grey\n')
                    f.write('alignment = grey')
                    f.close()
                elif status == 'Base Calling':
                    f = open('progress.txt','w')
                    f.write('wellfinding = green\n')
                    f.write('signalprocessing = green\n')
                    f.write('basecalling = yellow\n')
                    f.write('alignment = grey')
                    f.close()
                elif status == 'Alignment':
                    f = open('progress.txt','w')
                    f.write('wellfinding = green\n')
                    f.write('signalprocessing = green\n')
                    f.write('basecalling = green\n')
                    f.write('alignment = yellow')
                    f.close()
                elif status == 'Processing finished':
                    f = open('progress.txt','w')
                    f.write('wellfinding = green\n')
                    f.write('signalprocessing = green\n')
                    f.write('basecalling = green\n')
                    f.write('alignment = green')
                    f.close()
        except:
            traceback.print_exc()


    def add_status(process, status, message=""):
        f = open("blockstatus.txt", 'a')
        f.write(process+"="+str(status)+" "+str(message)+"\n")
        f.close()


    def upload_analysismetrics(bfmaskstatspath):

        print("Starting Upload Analysis Metrics")
        try:
            if not isProtonBlock:
                full_bfmaskstatspath = os.path.join(os.getcwd(), bfmaskstatspath)
                result = jobserver.uploadanalysismetrics(full_bfmaskstatspath, primary_key_file)
                printtime("Completed Upload Analysis Metrics: %s" % result)
        except Exception as err:
            printtime("Error during analysis metrics upload %s" % err)
            traceback.print_exc()


    def update_bfmask_artifacts(bfmaskPath, bfmaskstatspath, outputdir, plot_title):

        printtime("Make Bead Density Plots")
        try:
            beadDensityPlot.genHeatmap(bfmaskPath, bfmaskstatspath, outputdir, plot_title)
        except IOError as err:
            printtime("Bead Density Plot file error: %s" % err)
        except Exception as err:
            printtime("Bead Density Plot generation failure: %s" % err)
            traceback.print_exc()



    if args.do_sigproc:

        set_result_status('Beadfind')
        status = sigproc.beadfind(
                    env['beadfindArgs'],
                    env['libraryKey'],
                    env['tfKey'],
                    env['pathToRaw'],
                    env['SIGPROC_RESULTS'])
        add_status("Beadfind", status)

        if status != 0:
            printtime("ERROR: Beadfind finished with status '%s'" % status)
            sys.exit(1)


        ########################################################
        #Make Bead Density Plots                               #
        ########################################################
        bfmaskPath = os.path.join(env['SIGPROC_RESULTS'],"bfmask.bin")
        bfmaskstatspath = os.path.join(env['SIGPROC_RESULTS'],"bfmask.stats")
        try:
            upload_analysismetrics(bfmaskstatspath)
            update_bfmask_artifacts(bfmaskPath, bfmaskstatspath, "./", plot_title=env['shortRunName'])
        except:
            traceback.print_exc()

        set_result_status('Signal Processing')
        status = sigproc.sigproc(
                    env['analysisArgs'],
                    env['libraryKey'],
                    env['tfKey'],
                    env['pathToRaw'],
                    env['SIGPROC_RESULTS'])
        add_status("Analysis", status)

        # write return code into file
        try:
            f = open(os.path.join(env['SIGPROC_RESULTS'],"analysis_return_code.txt"), 'w')
            f.write(str(status))
            f.close()
            os.chmod(os.path.join(env['SIGPROC_RESULTS'],"analysis_return_code.txt"), 0775)
        except:
            traceback.print_exc()

        if status != 0:
            printtime("ERROR: Analysis finished with status '%s'" % status)
            sys.exit(1)



    if args.do_basecalling:

        # Generate files needed for regular and fromWells reports
        sigproc.generate_raw_data_traces(
            env['libraryKey'],
            env['tfKey'],
            env['flowOrder'],
            env['SIGPROC_RESULTS'])

        # Update or generate Bead density plots
        bfmaskPath = os.path.join(env['SIGPROC_RESULTS'],"analysis.bfmask.bin")
        bfmaskstatspath = os.path.join(env['SIGPROC_RESULTS'],"analysis.bfmask.stats")
        try:
            upload_analysismetrics(bfmaskstatspath)
            update_bfmask_artifacts(bfmaskPath, bfmaskstatspath, "./", plot_title=env['shortRunName'])
        except:
            traceback.print_exc()

        #make sure pre-conditions for basecaller step are met
        if not os.path.exists(os.path.join(env['SIGPROC_RESULTS'],'1.wells')):
            printtime ("ERROR: missing %s" % os.path.join(env['SIGPROC_RESULTS'],'1.wells') )
            printtime ("ERROR: basecaller pre-conditions not met")
            add_status("Pre Basecalling Step", status=1)
            sys.exit(1)

        if not os.path.exists(os.path.join(env['SIGPROC_RESULTS'],'analysis_return_code.txt')):
            printtime ("ERROR: missing %s" % os.path.join(env['SIGPROC_RESULTS'],'analysis_return_code.txt'))
            printtime ("ERROR: basecaller pre-conditions not met")
            add_status("Pre Basecalling Step", status=1)
            sys.exit(1)

        try:
            with open(os.path.join(env['SIGPROC_RESULTS'],'analysis_return_code.txt'), 'r') as f:
                return_code = f.read()
                if not int(return_code) == 0:
                    printtime ("ERROR: basecaller pre-conditions not met")
                    add_status("Pre Basecalling Step", status=1)
                    raise Exception("Analysis failed with %s" % return_code)
        except:
            traceback.print_exc()
            sys.exit(1)


        ########################################################
        # Flow Space Recalibration and re-basecalling          #
        ########################################################
        additional_basecallerArgs = ""
        if env['doBaseRecal'] and reference_selected:
            printtime("DEBUG: Flow Space Recalibration is enabled with Reference: %s" % env['referenceName'])
            set_result_status('Flow Space Recalibration')
            try:

                # Default options to produce smaller basecaller results
                prebasecallerArgs = env['prebasecallerArgs']
                if not "--calibration-training=" in prebasecallerArgs:
                    prebasecallerArgs = prebasecallerArgs + " --calibration-training=2000000"
                if not "--flow-signals-type" in prebasecallerArgs:
                    prebasecallerArgs = prebasecallerArgs + " --flow-signals-type scaled-residual"

                basecaller.basecalling(
                    env['SIGPROC_RESULTS'],
                    prebasecallerArgs,
                    env['libraryKey'],
                    env['tfKey'],
                    env['runID'],
                    env['reverse_primer_dict'],
                    os.path.join(env['BASECALLER_RESULTS'], 'recalibration'),
                    env['barcodeId'],
                    env['barcodeInfo'],
                    env['library'],
                    env['notes'],
                    env['site_name'],
                    env['platform'],
                    env['instrumentName'],
                    env['chipType'])

                basecaller_recalibration_datasets = blockprocessing.get_datasets_basecaller(os.path.join(env['BASECALLER_RESULTS'],'recalibration'))

                # file containing dimension info (offsets, rows, cols) and flow info for stratification
                try:
                    c = open(os.path.join(env['BASECALLER_RESULTS'], "recalibration", 'BaseCaller.json'),'r')
                    chipflow = json.load(c)
                    c.close()
                except:
                    traceback.print_exc()
                    raise

                # Recalibrate
                for dataset in basecaller_recalibration_datasets["datasets"]:

                    read_group = dataset['read_groups'][0]

                    if basecaller_recalibration_datasets['read_groups'][read_group].get('filtered',False):
                        continue

                    if not basecaller_recalibration_datasets['read_groups'][read_group].get('read_count',0) > 0:
                        continue

                    barcode_name = basecaller_recalibration_datasets['read_groups'][read_group].get('barcode_name','no_barcode')
                    if not env['barcodeInfo'][barcode_name]['calibrate']:
                        continue

                    referenceName = basecaller_recalibration_datasets['read_groups'][read_group]['reference']

                    if not referenceName:
                        continue

                    readsFile = os.path.join(env['BASECALLER_RESULTS'],'recalibration',dataset['basecaller_bam'])

                    printtime("DEBUG: Work starting on %s" % readsFile)
                    RECALIBRATION_RESULTS = os.path.join(env['BASECALLER_RESULTS'],"recalibration", dataset['file_prefix'])
                    os.makedirs(RECALIBRATION_RESULTS)
                    sample_map_path = os.path.join(RECALIBRATION_RESULTS, "samplelib.bam")

                    alignment.align(
                        referenceName,
                        readsFile,
                        bidirectional=False,
                        mark_duplicates=False,
                        realign=False,
                        skip_sorting=True,
                        aligner_opts_extra="",
                        logfile=os.path.join(RECALIBRATION_RESULTS,"alignmentQC_out.txt"),
                        output_dir=RECALIBRATION_RESULTS,
                        output_basename="samplelib")

                    # Generate both hpTable and hpModel.
                    flow_space_recal.calibrate(
                        RECALIBRATION_RESULTS,
                        sample_map_path,
                        env['recalibArgs'],
                        chipflow)

                # merge step, calibrate collects the training data saved for each barcode,
                # calculate and generate hpTable and hpModel files for the whole dataset
                flow_space_recal.HPaggregation(
                    os.path.join(env['BASECALLER_RESULTS'],"recalibration"),
                    env['recalibArgs'])

                hptable = os.path.join(env['BASECALLER_RESULTS'], "recalibration", "hpTable.txt")
                printtime("hptable: %s" % hptable)
                additional_basecallerArgs = " --calibration-file " + hptable + " --phase-estimation-file " + os.path.join(env['BASECALLER_RESULTS'], "recalibration", "BaseCaller.json") + " --model-file " + os.path.join(env['BASECALLER_RESULTS'], "recalibration", "hpModel.txt")
                add_status("Recalibration", 0)
            except:
                traceback.print_exc()
                add_status("Recalibration", 1)
                printtime ("ERROR: Recalibration failed")
                sys.exit(1)

        else:
            printtime("DEBUG: Flow Space Recalibration is disabled, Reference: '%s'" % env['referenceName'])
            updated_basecallerArgs = env['basecallerArgs']


        set_result_status('Base Calling')
        try:
            basecaller.basecalling(
                env['SIGPROC_RESULTS'],
                env['basecallerArgs'] + additional_basecallerArgs,
                env['libraryKey'],
                env['tfKey'],
                env['runID'],
                env['reverse_primer_dict'],
                env['BASECALLER_RESULTS'],
                env['barcodeId'],
                env['barcodeInfo'],
                env['library'],
                env['notes'],
                env['site_name'],
                env['platform'],
                env['instrumentName'],
                env['chipType'],
            )
            add_status("Basecaller", 0)
        except:
            traceback.print_exc()
            add_status("Basecaller", 1)
            printtime ("ERROR: Basecaller failed")
            sys.exit(1)



    if args.do_alignment:

        do_merged_alignment = isProtonBlock and False
        if do_merged_alignment:
            sys.exit(0)

        try:
            c = open(os.path.join(env['BASECALLER_RESULTS'], "BaseCaller.json"),'r')
            basecaller_meta_information = json.load(c)
            c.close()
        except:
            traceback.print_exc()
            raise

        basecaller_datasets = blockprocessing.get_datasets_basecaller(env['BASECALLER_RESULTS'])

        # update filtered flag
        if isProtonBlock:
            composite_basecaller_datasets = blockprocessing.get_datasets_basecaller(os.path.join('..',env['BASECALLER_RESULTS']))
            for rg_name in basecaller_datasets["read_groups"]:
                block_filtered_flag = basecaller_datasets["read_groups"][rg_name].get('filtered',False)
                composite_filtered_flag = composite_basecaller_datasets["read_groups"][rg_name].get('filtered',False)
                basecaller_datasets["read_groups"][rg_name]['filtered'] = composite_filtered_flag or block_filtered_flag

        activate_barcode_filter = True

        if reference_selected:

            set_result_status('Alignment')

            if isProtonBlock:
                create_index = False
            else:
                create_index = True

            try:
                bidirectional = False
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
                add_status("Alignment", 0)
            except:
                traceback.print_exc()
                add_status("Alignment", 1)
                printtime ("ERROR: Alignment failed")
                sys.exit(1)


        if isProtonBlock and os.path.exists('/opt/ion/.ion-internal-server'):

            printtime("create block level ionstats")

            try:
                import math
                graph_max_x = int(50 * math.ceil(0.014 * int(env['flows'])))
            except:
                traceback.print_exc()
                graph_max_x = 400

            try:
                # Plot classic read length histogram (also used for Read Length Details view)
                # TODO: change word alignment
                alignment.create_ionstats(
                    env['BASECALLER_RESULTS'],
                    env['ALIGNMENT_RESULTS'],
                    basecaller_meta_information,
                    basecaller_datasets,
                    graph_max_x,
                    activate_barcode_filter)

                ionstats_plots.old_read_length_histogram(
                    #os.path.join('ionstats_alignment.json'),
                    os.path.join(env['BASECALLER_RESULTS'],'ionstats_basecaller.json'),
                    os.path.join(env['BASECALLER_RESULTS'],'readLenHisto.png'),
                    graph_max_x)
            except:
                traceback.print_exc()

        set_result_status('Processing finished')

    printtime("BlockTLScript exit")
    sys.exit(0)
