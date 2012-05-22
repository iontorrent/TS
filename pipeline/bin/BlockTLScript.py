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

from ion.utils import blockprocessing
from ion.utils import sigproc
from ion.utils import basecaller
from ion.utils import alignment

from ion.utils.blockprocessing import printtime

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


    blockprocessing.printheader()
    env = blockprocessing.getparameter()

    if not os.path.exists("ReportLog.html"):
        blockprocessing.initreports(env['SIGPROC_RESULTS'], env['BASECALLER_RESULTS'], env['ALIGNMENT_RESULTS'])
        logout = open("ReportLog.html", "w")
        logout.close()

    blockprocessing.write_version()
    sys.stdout.flush()
    sys.stderr.flush()

    fastq_path = os.path.join(env['BASECALLER_RESULTS'], "rawlib.fastq")
    libsff_path = os.path.join(env['BASECALLER_RESULTS'], "rawlib.sff")
    tfsff_path = os.path.join(env['BASECALLER_RESULTS'], "rawtf.sff")

    if args.do_sigproc:

        # create analysis progress bar file
        f = open('progress.txt','w')
        f.write('wellfinding = yellow\n')
        f.write('signalprocessing = grey\n')
        f.write('basecalling = grey\n')
        f.write('sffread = grey\n')
        f.write('alignment = grey')
        f.close()

        sigproc.sigproc(
            env['analysisArgs'],
            env['pathToRaw'],
            env['SIGPROC_RESULTS'])


    if args.do_basecalling:

        #make sure pre-conditions for basecaller step are met
        if not os.path.exists(os.path.join(env['SIGPROC_RESULTS'],'1.wells')):
            printtime ("ERROR: basecaller pre-conditions not met")
            sys.exit(1)

        # create analysis progress bar file
        f = open('progress.txt','w')
        f.write('wellfinding = green\n')
        f.write('signalprocessing = green\n')
        f.write('basecalling = yellow\n')
        f.write('sffread = grey\n')
        f.write('alignment = grey')
        f.close()

        basecaller.basecalling(
            env['SIGPROC_RESULTS'],
            env['previousReport'],
            env['basecallerArgs'],
            env['libraryKey'],
            env['tfKey'],
            env['runID'],
            env['flowOrder'],
            env['reverse_primer_dict'],
            env['BASECALLER_RESULTS'])

        # create analysis progress bar file
        f = open('progress.txt','w')
        f.write('wellfinding = green\n')
        f.write('signalprocessing = green\n')
        f.write('basecalling = green\n')
        f.write('sffread = yellow\n')
        f.write('alignment = grey')
        f.close()

        generate_beadsummary=True
        skipsfftrim=True

        basecaller.post_basecalling(
            libsff_path,
            env['reverse_primer_dict'],
            skipsfftrim,
            env['sfftrim_args'],
            env['libraryKey'],
            env['flowOrder'],
            env['barcodeId'],
            env['barcodesplit_filter'],
            env['DIR_BC_FILES'],
            os.path.join("barcodeList.txt"),
            os.path.join(env['SIGPROC_RESULTS'], "bfmask.bin"),
            os.path.join(env['BASECALLER_RESULTS'], "barcodeMask.bin"),
            generate_beadsummary,
            env['BASECALLER_RESULTS'])

        basecaller.tf_processing(
            env['SIGPROC_RESULTS'],
            tfsff_path,
            env['libraryKey'],
            env['tfKey'],
            env['flowOrder'],
            env['BASECALLER_RESULTS'])


        ##################################################
        # Unfiltered SFF
        ##################################################
        top_dir = os.getcwd()
        try:
            unfiltered_dir = "unfiltered"
            if os.path.exists(unfiltered_dir):

                #change to the unfiltered dir
                os.chdir(os.path.join(top_dir,unfiltered_dir))

                #trim status
                for status in ["untrimmed","trimmed"]:
                    if not os.path.exists(status):
                        os.makedirs(status)

                #grab the first file named untrimmed.sff
                sff_untrimmed_path = glob.glob("*.untrimmed.sff")[0]
                sff_trimmed_path = glob.glob("*.trimmed.sff")[0]
                #sff_trimmed_path = sff_untrimmed_path[:-4]+".trimmed.sff"
                fastq_untrimmed_path = sff_untrimmed_path.replace(".sff",".fastq")

                #copy
                if os.path.exists(sff_untrimmed_path):
                    printtime ("DEBUG: move file %s" % sff_untrimmed_path)
                    shutil.copy(sff_untrimmed_path,os.path.join(top_dir,unfiltered_dir,"untrimmed"))
                    #shutil.copy(sff_untrimmed_path,os.path.join(top_dir,unfiltered_dir,"trimmed",sff_untrimmed_path[:-4]+".trimmed.sff"))
                else:
                    printtime ("ERROR: Move: File not found: %s" % sff_untrimmed_path)
                    
                if os.path.exists(sff_trimmed_path):
                    printtime ("DEBUG: move file %s" % sff_trimmed_path)
                    shutil.copy(sff_trimmed_path,os.path.join(top_dir,unfiltered_dir,"trimmed"))
                else:
                    printtime ("ERROR: Move: File not found: %s" % sff_trimmed_path)

                #trim status
                for status in ["untrimmed","trimmed"]:
                    os.chdir(os.path.join(top_dir,unfiltered_dir,status))

                    printtime("Trim Status:" + status)

                    skipsfftrim=True
                    if status == "untrimmed":
                        skipsfftrim=True
                        sff_path=sff_untrimmed_path
                    if status == "trimmed":
                        #skipsfftrim=False
                        skipsfftrim=True
                        sff_path=sff_trimmed_path

                    generate_beadsummary=False

                    basecaller.post_basecalling(
                        sff_path,
                        env['reverse_primer_dict'],
                        skipsfftrim,
                        env['sfftrim_args'],
                        env['libraryKey'],
                        env['flowOrder'],
                        env['barcodeId'],
                        env['barcodesplit_filter'],
                        env['DIR_BC_FILES'],
                        os.path.join("..","..", "barcodeList.txt"),
                        os.path.join("..","..", env['SIGPROC_RESULTS'], "bfmask.bin"),
                        "barcodeMask.bin",
                        generate_beadsummary,
                        BASECALLER_RESULTS=".")

            else:
                printtime("Directory unfiltered does not exist")

        except IndexError:
            printtime("Error, unfiltered handling")
            traceback.print_exc()
        os.chdir(top_dir)

    if args.do_alignment:

        #make sure pre-conditions for basecaller step are met
        if not os.path.exists(libsff_path):
            printtime ("ERROR: alignment pre-conditions not met")
            sys.exit(1)
        
        if env['libraryName']=='none' or len(env['libraryName'])<1:
          # skip alignment when no library
          printtime("DEBUG: No Reference Library selected (libraryName = %s)",env['libraryName'])        
        else:
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
          # Unfiltered SFF
          ##################################################

          unfiltered_dir = "unfiltered"
          if os.path.exists(unfiltered_dir):

              top_dir = os.getcwd()

              #change to the unfiltered dir
              os.chdir(os.path.join(top_dir,unfiltered_dir))

              #trim status
              for status in ["untrimmed","trimmed"]:
                  if not os.path.exists(status):
                      os.makedirs(status)
                  os.chdir(os.path.join(top_dir,unfiltered_dir,status))

                  printtime("Trim Status:" + status)

                  try:
                      unfiltered_sff = glob.glob("*." + status + ".sff")[0]
                  except:
                      printtime("ERROR: unfiltered sff file not found")
                      unfiltered_sff = ""

                  try:
                      unfiltered_fastq = glob.glob("*." + status + ".fastq")[0]
                  except:
                      printtime("ERROR: unfiltered fastq file not found")
                      unfiltered_fastq = ""

                  alignment.alignment(
                      unfiltered_sff,
                      unfiltered_fastq,
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
                      env['start_time'],
                      env['ALIGNMENT_RESULTS'],
                      bidirectional,
                      env['sam_parsed'])

              os.chdir(top_dir)
          else:
              printtime("Directory unfiltered does not exist")


          alignment.alignment(
              libsff_path,
              fastq_path,
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
              env['start_time'],
              env['ALIGNMENT_RESULTS'],
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

    sys.exit(0)
