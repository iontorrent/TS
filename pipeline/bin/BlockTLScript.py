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
        debugging_cwd = os.getcwd()
    except:
        traceback.print_exc()
    
    def set_result_status(status):
        try:
            primary_key_file = os.path.join(os.getcwd(),'primary.key')
            jobserver.updatestatus(primary_key_file, status, True)
            printtime("TLStatus %s\tpid %d\tpk file %s started in %s" % 
                (status, os.getpid(), primary_key_file, debugging_cwd))
        except:
            traceback.print_exc()

    

    if args.do_sigproc:

        set_result_status('Signal Processing')
        # create analysis progress bar file
        f = open('progress.txt','w')
        f.write('wellfinding = yellow\n')
        f.write('signalprocessing = grey\n')
        f.write('basecalling = grey\n')
        f.write('sffread = grey\n')
        f.write('alignment = grey')
        f.close()

        if env['doThumbnail']:
            analysisArgs = env['thumbnailAnalysisArgs']
            oninstrumentanalysis = False
        else:
            analysisArgs = env['analysisArgs']
            oninstrumentanalysis = env['oninstranalysis']

        sigproc.sigproc(
            analysisArgs,
            env['libraryKey'],
            env['tfKey'],
            env['pathToRaw'],
            env['SIGPROC_RESULTS'],
            env['shortRunName'],
            oninstrumentanalysis)

    # In case of from-wells or from-basecaller reanalysis of a legacy report, some adjustments may be needed
    handle_legacy_report.handle_sigproc(env['SIGPROC_RESULTS'])

    if args.do_basecalling:

        #make sure pre-conditions for basecaller step are met
        if not os.path.exists(os.path.join(env['SIGPROC_RESULTS'],'1.wells')):
            printtime ("ERROR: missing %s" % os.path.join(env['SIGPROC_RESULTS'],'1.wells') )
            printtime ("ERROR: basecaller pre-conditions not met")
            sys.exit(1)

        set_result_status('Base Calling')
        # create analysis progress bar file
        f = open('progress.txt','w')
        f.write('wellfinding = green\n')
        f.write('signalprocessing = green\n')
        f.write('basecalling = yellow\n')
        f.write('sffread = grey\n')
        f.write('alignment = grey')
        f.close()

        if env['doThumbnail']:
            basecallerArgs = env['thumbnailBasecallerArgs']
        else:
            basecallerArgs = env['basecallerArgs']

        basecaller.basecalling(
            env['SIGPROC_RESULTS'],
            basecallerArgs,
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
        
        
        ########################################################################
        #
        # NEW PIPELINE BRANCH: Flow Space Recalibration and re-basecalling
        #
        ########################################################################
        if env['doBaseRecal']:
            printtime("DEBUG: Flow Space Recalibration is enabled")
#            # Replace *.basecaller.bam files with links, only if recalibration is on - Marcin
#            try:
#                f = open(os.path.join(BASECALLER_RESULTS,"datasets_basecaller.json"),'r')
#                datasets_basecaller = json.load(f);
#                f.close()
#                for dataset in datasets_basecaller["datasets"]:
#                    bam_filename_primary = os.path.join(BASECALLER_RESULTS, dataset['basecaller_bam'])
#                    bam_filename_backup = os.path.join(BASECALLER_RESULTS, dataset['file_prefix']+".old.basecaller.bam")
#                    if os.path.exists(bam_filename_primary):
#                        shutil.move(bam_filename_primary,bam_filename_backup)
#                        os.symlink(os.path.relpath(bam_filename_backup,os.path.dirname(bam_filename_primary)),bam_filename_primary)
#                        
#            except:
#                printtime("ERROR: Replacing basecaller.bam files with links failed")
#                traceback.print_exc()
            
            if env['libraryName']=='none' or len(env['libraryName'])<1:
                printtime("DEBUG: No reference specified - Flow Space Recalibration skipped")
            else:
                printtime("DEBUG: Flow Space Recalibration with Reference: %s" % env['libraryName'])
                try:
                    flow_space_recal.base_recalib(env['BASECALLER_RESULTS'],
                                                  env['runID'],
                                                  env['align_full'],
                                                  env['DIR_BC_FILES'],
                                                  env['libraryName'],
                                                  env['sample'],
                                                  env['chipType'],
                                                  env['site_name'],
                                                  env['flows'],
                                                  env['notes'],
                                                  env['barcodeId'],
                                                  env['aligner_opts_extra'],
                                                  env['mark_duplicates'],
                                                  env['start_time'],
                                                  env['tmap_version'],
                                                  "datasets_basecaller.json", # file containing all available datasets
                                                  "BaseCaller.json" #file containing dimension info (offsets, rows, cols) and flow info for stratification
                                                  )
                except:
                    printtime("ERROR: Flow Space Recalibration Failed")
        else:
            printtime("DEBUG: Flow Space Recalibration is disabled")
        
        # create analysis progress bar file
        f = open('progress.txt','w')
        f.write('wellfinding = green\n')
        f.write('signalprocessing = green\n')
        f.write('basecalling = green\n')
        f.write('sffread = yellow\n')
        f.write('alignment = grey')
        f.close()

        # Work in progress: migrate post-basecaller steps to use datasets_*.json to drive processing
        basecaller.post_basecalling(env['BASECALLER_RESULTS'],env['expName'],env['resultsName'],env['flows'])


        basecaller.tf_processing(
            env['SIGPROC_RESULTS'],
            os.path.join(env['BASECALLER_RESULTS'], "rawtf.basecaller.bam"),
            env['libraryKey'],
            env['tfKey'],
            env['flowOrder'],
            env['BASECALLER_RESULTS'],
            '.')


        ##################################################
        # Unfiltered BAM
        ##################################################
        
        try:
            if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed')):
                basecaller.post_basecalling(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed'),env['expName'],env['resultsName'],env['flows'])
            
            if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed')):
                basecaller.post_basecalling(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed'),env['expName'],env['resultsName'],env['flows'])
   
        except IndexError:
            printtime("Error, unfiltered handling")
            traceback.print_exc()
            
            
    if args.do_alignment:

        #make sure pre-conditions for alignment step are met
        if not os.path.exists(os.path.join(env['BASECALLER_RESULTS'], "rawlib.basecaller.bam")):
            printtime ("ERROR: alignment pre-conditions not met")
        
        if env['libraryName']=='none' or len(env['libraryName'])<1:
            # skip alignment when no library
            printtime("DEBUG: No Reference Library selected (libraryName = %s)",env['libraryName'])        
        else:
            set_result_status('Alignment')
            # create analysis progress bar file
            f = open('progress.txt','w')
            f.write('wellfinding = green\n')
            f.write('signalprocessing = green\n')
            f.write('basecalling = green\n')
            f.write('sffread = green\n')
            f.write('alignment = yellow')
            f.close()
            
            bidirectional = False
            
            ##################################################
            # Unfiltered BAM
            ##################################################

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


        # create analysis progress bar file
        f = open('progress.txt','w')
        f.write('wellfinding = green\n')
        f.write('signalprocessing = green\n')
        f.write('basecalling = green\n')
        f.write('sffread = green\n')
        f.write('alignment = green')
        f.close()

    printtime("BlockTLScript exit")
    sys.exit(0)
