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

from ion.reports import beadDensityPlot
from ion.utils import blockprocessing
from ion.utils import explogparser
from ion.utils import sigproc
from ion.utils import basecaller
from ion.utils import alignment
from ion.utils import flow_space_recal
from ion.utils import handle_legacy_report

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

        if env['doThumbnail']:
            oninstrumentanalysis = False
        else:
            oninstrumentanalysis = env['oninstranalysis']


        if not oninstrumentanalysis:

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
            update_bfmask_artifacts(bfmaskPath, bfmaskstatspath, env['SIGPROC_RESULTS'], plot_title=env['shortRunName'])
            update_bfmask_artifacts(bfmaskPath, bfmaskstatspath, "./", plot_title=env['shortRunName'])
        except:
            traceback.print_exc()


        if not oninstrumentanalysis:

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

        # In case of from-wells or from-basecaller reanalysis of a legacy report
        # some adjustments may be needed
        handle_legacy_report.handle_sigproc(env['SIGPROC_RESULTS'])

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
            update_bfmask_artifacts(bfmaskPath, bfmaskstatspath, env['SIGPROC_RESULTS'], plot_title=env['shortRunName'])
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
        if env['libraryName']!='none' and len(env['libraryName'])>=1 and env['doBaseRecal']:
            printtime("DEBUG: Flow Space Recalibration is enabled with Reference: %s" % env['libraryName'])
            set_result_status('Flow Space Recalibration')
            try:
                qvtable = flow_space_recal.base_recalib(
                                                env['SIGPROC_RESULTS'],
                                                env['prebasecallerArgs'],
                                                env['libraryKey'],
                                                env['tfKey'],
                                                env['runID'],
                                                env['flowOrder'],
                                                env['reverse_primer_dict'],
                                                env['BASECALLER_RESULTS'],
                                                env['barcodeId'],
                                                env['barcodeSamples'],
                                                env.get('barcodesplit_filter',0),
                                                env.get('barcodesplit_filter_minreads',0),
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
                                                env['pgmName'],
                                                env['tmap_version'],
                                                "datasets_basecaller.json", # file containing all available datasets
                                                "BaseCaller.json" #file containing dimension info (offsets, rows, cols) and flow info for stratification
                                               )
                printtime("QVTable: %s" % qvtable)
                additional_basecallerArgs = " --calibration-file " + qvtable + " --phase-estimation-file " + os.path.join(env['BASECALLER_RESULTS'], "recalibration", "BaseCaller.json") + " --model-file " + os.path.join(env['BASECALLER_RESULTS'], "recalibration", "hpModel.txt")
                add_status("Recalibration", 0)
            except:
                traceback.print_exc()
                add_status("Recalibration", 1)

        else:
            printtime("DEBUG: Flow Space Recalibration is disabled, Reference: '%s'" % env['libraryName'])
            updated_basecallerArgs = env['basecallerArgs']


        set_result_status('Base Calling')
        try:
            basecaller.basecalling(
                env['SIGPROC_RESULTS'],
                env['basecallerArgs'] + additional_basecallerArgs,
                env['libraryKey'],
                env['tfKey'],
                env['runID'],
                env['flowOrder'],
                env['reverse_primer_dict'],
                env['BASECALLER_RESULTS'],
                env['barcodeId'],
                env['barcodeSamples'],
                env.get('barcodesplit_filter',0),
                env.get('barcodesplit_filter_minreads',0),
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
            add_status("Basecaller", 0)
        except:
            traceback.print_exc()
            add_status("Basecaller", 1)


        set_result_status('Post Basecalling')
        try:
            basecaller.post_basecalling(env['BASECALLER_RESULTS'],env['expName'],env['resultsName'],env['flows'])
            add_status("PostBasecaller", 0)
        except:
            traceback.print_exc()
            add_status("PostBasecaller", 1)

        if not isProtonBlock:
            set_result_status('Merge Barcoded Basecaller Bams')
            try:
                basecaller.merge_barcoded_basecaller_bams(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed'))
                add_status("MergeBarcodedBasecallerBamsUntrimmed", 0)
            except:
                traceback.print_exc()
                add_status("MergeBarcodedBasecallerBamsUntrimmed", 1)
            try:
                basecaller.merge_barcoded_basecaller_bams(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed'))
                add_status("MergeBarcodedBasecallerBamsTrimmed", 0)
            except:
                traceback.print_exc()
                add_status("MergeBarcodedBasecallerBamsTrimmed", 1)
            try:
                basecaller.merge_barcoded_basecaller_bams(env['BASECALLER_RESULTS'])
                add_status("MergeBarcodedBasecallerBams", 0)
            except:
                traceback.print_exc()
                add_status("MergeBarcodedBasecallerBams", 1)


        set_result_status('TF Processing')
        try:
            basecaller.tf_processing(
                os.path.join(env['BASECALLER_RESULTS'], "rawtf.basecaller.bam"),
                env['tfKey'],
                env['flowOrder'],
                env['BASECALLER_RESULTS'],
                '.')
            add_status("TF Processing", 0)
        except:
            traceback.print_exc()
            add_status("TF Processing", 1)


        ##################################################
        # Unfiltered BAM
        ##################################################

        if not isProtonBlock:
            try:
                if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed')):
                    basecaller.post_basecalling(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed'),env['expName'],env['resultsName'],env['flows'])

                if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed')):
                    basecaller.post_basecalling(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed'),env['expName'],env['resultsName'],env['flows'])

            except IndexError:
                printtime("Error, unfiltered handling")
                traceback.print_exc()

    do_merged_alignment = False
    if args.do_alignment and not do_merged_alignment:

        #make sure pre-conditions for alignment step are met
        # TODO if not os.path.exists(os.path.join(env['BASECALLER_RESULTS'], "rawlib.basecaller.bam")):
        if not os.path.exists(os.path.join(env['BASECALLER_RESULTS'], "ionstats_basecaller.json")):
            printtime ("ERROR: alignment pre-conditions not met")
            add_status("Pre Alignment Step", status=1)
            sys.exit(1)
        
        if env['libraryName']=='none' or len(env['libraryName'])<1:
            # skip alignment when no library
            printtime("DEBUG: No Reference Library selected (libraryName = %s)",env['libraryName'])        
        else:
            set_result_status('Alignment')
            
            bidirectional = False
            
            ##################################################
            # Unfiltered BAM
            ##################################################

            if not isProtonBlock:

                try:
                    if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed")):
                        alignment.alignment_unmapped_bam(
                            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed"),
                            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed"),
                            env['align_full'],
                            env['libraryName'],
                            env['flows'],
                            env['realign'],
                            env['aligner_opts_extra'],
                            env['mark_duplicates'],
                            bidirectional,
                            env['sam_parsed'])
                        #add_status("Alignment", 0)
                except:
                    traceback.print_exc()
                    #add_status("Alignment", 1)

                try:
                    if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed")):
                        alignment.alignment_unmapped_bam(
                            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed"),
                            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed"),
                            env['align_full'],
                            env['libraryName'],
                            env['flows'],
                            env['realign'],
                            env['aligner_opts_extra'],
                            env['mark_duplicates'],
                            bidirectional,
                            env['sam_parsed'])
                        #add_status("Alignment", 0)
                except:
                    traceback.print_exc()
                    #add_status("Alignment", 1)

            try:
                alignment.alignment_unmapped_bam(
                    env['BASECALLER_RESULTS'],
                    env['ALIGNMENT_RESULTS'],
                    env['align_full'],
                    env['libraryName'],
                    env['flows'],
                    env['realign'],
                    env['aligner_opts_extra'],
                    env['mark_duplicates'],
                    bidirectional,
                    env['sam_parsed'])
                add_status("Alignment", 0)
            except:
                traceback.print_exc()
                add_status("Alignment", 1)

        set_result_status('Processing finished')

    printtime("BlockTLScript exit")
    sys.exit(0)
