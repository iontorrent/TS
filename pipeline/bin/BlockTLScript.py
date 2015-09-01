#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

__version__ = filter(str.isdigit, "$Revision$")

import os
import sys
import argparse
import traceback
import xmlrpclib
import json

from ion.reports import beadDensityPlot
from ion.utils import blockprocessing
from ion.utils import explogparser
from ion.utils import sigproc
from ion.utils import basecaller
from ion.utils import alignment
from ion.utils import flow_space_recal

from ion.utils.blockprocessing import printtime

sys.path.append('/etc')
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

    # Speedup Flags
    do_ionstats_spatial_accounting  = False

    reference_selected = False
    for barcode_name,barcode_info in sorted(env['barcodeInfo'].iteritems()):
        if barcode_info['referenceName']:
            reference_selected = True
            pass

    if os.path.exists(primary_key_file):
        isBlock = False
    else:
        isBlock = True

    try:
        import math
        graph_max_x = int(50 * math.ceil(0.014 * int(env['flows'])))
    except:
        traceback.print_exc()
        graph_max_x = 400

    def set_result_status(status):
        try:
            if not isBlock:
                if status == 'Processing finished':
                    # filter out this status (31x and Thumbnail)
                    return
                jobserver.updatestatus(primary_key_file, status, True)
                printtime("BlockTLStatus %s\tpid %d\tpk file %s started" % 
                    (status, os.getpid(), primary_key_file))
            else: # Block
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
            if not isBlock:
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
            # write return code into file
            try:
                f = open(os.path.join(env['SIGPROC_RESULTS'],"analysis_return_code.txt"), 'w')
                f.write(str(status))
                f.close()
                os.chmod(os.path.join(env['SIGPROC_RESULTS'],"analysis_return_code.txt"), 0775)
            except:
                traceback.print_exc()
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
        # Flow Space Recalibration and pre-basecalling          #
        ########################################################
        #
        #          'no_recal' : 'No Calibration'
        #    'standard_recal' : 'Default Calibration'
        #       'panel_recal' : 'Calibration Standard'
        #

        additional_basecallerArgs = ""
        if (env['doBaseRecal'] == "standard_recal" and reference_selected) or env['doBaseRecal'] == "panel_recal":
            printtime("DEBUG: Flow Space Recalibration is enabled, Mode: %s" % env['doBaseRecal'])
            set_result_status('Flow Space Recalibration')
            try:

                # Default options to produce smaller basecaller results
                prebasecallerArgs = env['prebasecallerArgs']

                if env['doBaseRecal'] == "panel_recal":
                    prebasecallerArgs = prebasecallerArgs + " --calibration-training=0"
                    prebasecallerArgs = prebasecallerArgs + " --calibration-panel /opt/ion/config/datasets_calibration.json"
                else:
                    if not "--calibration-training=" in prebasecallerArgs:
                        prebasecallerArgs = prebasecallerArgs + " --calibration-training=100000"

                basecaller.basecalling(
                    env['SIGPROC_RESULTS'],
                    prebasecallerArgs,
                    env['libraryKey'],
                    env['tfKey'],
                    env['runID'],
                    env['reverse_primer_dict']['sequence'],
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
                if env['doBaseRecal'] == "panel_recal":
                    basecaller_recalibration_datasets = basecaller_recalibration_datasets['IonControl']

                # file containing dimension info (offsets, rows, cols) and flow info for stratification
                try:
                    c = open(os.path.join(env['BASECALLER_RESULTS'], "recalibration", 'BaseCaller.json'),'r')
                    chipflow = json.load(c)
                    c.close()
                except:
                    traceback.print_exc()
                    raise

                calibration_input_bams = ""
                
                for dataset in basecaller_recalibration_datasets["datasets"]:

                    if not dataset.get('read_count',0) > 0:
                        continue

                    read_group = dataset['read_groups'][0]
                    referenceName = basecaller_recalibration_datasets['read_groups'][read_group]['reference']
                    #referenceName = dataset['reference']
                    if not referenceName:
                        continue

                    basecaller_bam = os.path.join(env['BASECALLER_RESULTS'],'recalibration',dataset['basecaller_bam'])

                    # --- Alignment of individual calibration BAM files in recalibration directory                   
                    printtime("DEBUG: Work starting on %s" % basecaller_bam)
                    RECALIBRATION_RESULTS = os.path.join(env['BASECALLER_RESULTS'],"recalibration")
                    calibration_bam_base = os.path.join(RECALIBRATION_RESULTS,dataset['file_prefix'])
                    blocks=[]
                    basecaller_meta_information=None
                    
                    if len(calibration_input_bams)>0:
                        calibration_input_bams += ","
                    calibration_input_bams += calibration_bam_base + ".bam"

                    alignment.align(
                        blocks,
                        basecaller_bam,
                        env['alignmentArgs'],
                        env['ionstatsArgs'],
                        referenceName,
                        basecaller_meta_information,
                        env['libraryKey'],
                        graph_max_x,
                        do_realign=False,
                        do_ionstats=False,
                        do_sorting=False,
                        do_mark_duplicates=False,
                        do_indexing=False,
                        output_dir=RECALIBRATION_RESULTS,
                        output_basename=dataset['file_prefix'])

                # Do not call Calibration if we did not generate any aligned input BAMs but gracefully continue
                additional_basecallerArgs += " --phase-estimation-file " + os.path.join(env['BASECALLER_RESULTS'], "recalibration", "BaseCaller.json")
                if env['doBaseRecal'] == "panel_recal":
                    additional_basecallerArgs += " --calibration-panel /opt/ion/config/datasets_calibration.json"
                    
                if len(calibration_input_bams)>0:
                    # Call Calibration module to process aligned training BAM files
                    flow_space_recal.calibrate(
                        env['BASECALLER_RESULTS'],
                        calibration_input_bams,
                        env['recalibArgs'],
                        chipflow)
                
                    additional_basecallerArgs  += " --calibration-json " + os.path.join(env['BASECALLER_RESULTS'],"Calibration.json")
                    
                printtime("Finished Recalibration")
                add_status("Recalibration", 0)
            except:
                traceback.print_exc()
                add_status("Recalibration", 1)
                printtime ("ERROR: Recalibration failed")
                sys.exit(1)

        else:
            printtime("DEBUG: Flow Space Recalibration is disabled")
            updated_basecallerArgs = env['basecallerArgs']


        set_result_status('Base Calling')
        try:
            basecaller.basecalling(
                env['SIGPROC_RESULTS'],
                env['basecallerArgs'] + additional_basecallerArgs,
                env['libraryKey'],
                env['tfKey'],
                env['runID'],
                env['reverse_primer_dict']['sequence'],
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


        set_result_status('Processing finished')

    printtime("BlockTLScript exit")
    sys.exit(0)
