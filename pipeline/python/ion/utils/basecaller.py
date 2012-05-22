#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from ion.utils.blockprocessing import printtime
from ion.utils.blockprocessing import MyConfigParser

import ConfigParser
import StringIO
import subprocess
import os
import sys
import traceback

from ion.reports import trimmedReadLenHisto

from ion.utils import TFPipeline
from ion.utils.blockprocessing import isbadblock
from ion.reports import MaskMerge, mergeBaseCallerJson, \
    StatsMerge, plotRawRegions, plotKey
from ion.utils import blockprocessing

def basecalling(
      SIGPROC_RESULTS,
      previousReport,
      basecallerArgs,
      libKey,
      tfKey,
      runID,
      floworder,
      reverse_primer_dict,
      BASECALLER_RESULTS):

    try:
        if basecallerArgs:
            cmd = basecallerArgs
        else:
            cmd = "BaseCaller"
        if previousReport:
            cmd += " --input-dir=%s" % (os.path.join(previousReport,SIGPROC_RESULTS))
        else:
            cmd += " --input-dir=%s" % (SIGPROC_RESULTS)
        cmd += " --librarykey=%s" % (libKey)
        cmd += " --tfkey=%s" % (tfKey)
        cmd += " --run-id=%s" % (runID)
        cmd += " --output-dir=%s" % (BASECALLER_RESULTS)

        # 3' adapter details
        qual_cutoff = reverse_primer_dict['qual_cutoff']
        qual_window = reverse_primer_dict['qual_window']
        adapter_cutoff = reverse_primer_dict['adapter_cutoff']
        adapter = reverse_primer_dict['sequence']
        cmd += " --flow-order %s" % (floworder)
        cmd += " --trim-qual-cutoff %s" % (qual_cutoff)
        cmd += " --trim-qual-window-size %s" % (qual_window)
        cmd += " --trim-adapter-cutoff %s" % (adapter_cutoff)
        cmd += " --trim-adapter %s" % (adapter)
        cmd += " --trim-min-read-len 5"
        cmd += " --bead-summary"

        cmd += " >> ReportLog.html 2>&1"

        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        blockprocessing.add_status("BaseCaller", ret)
    except:
        printtime('ERROR: BaseCaller failed')
        traceback.print_exc()


def post_basecalling(
      libsff_path,
      reverse_primer_dict,
      skipsfftrim,
      sfftrim_args,
      libKey,
      floworder,
      barcodeId,
      barcodesplit_filter,
      DIR_BC_FILES,
      barcodeList_path,
      bfmask_path,
      barcodeMask_path,
      generate_beadsummary,
      BASECALLER_RESULTS):

    if not os.path.exists(libsff_path):
        printtime("ERROR: %s does not exist" % libsff_path)
        open('badblock.txt', 'w').close()
        return


    ##################################################
    # Trim the SFF file if it has been requested     #
    ##################################################

    if not skipsfftrim:
        printtime("Attempting to trim the SFF file")

        libsff_untrimmed_path = libsff_path
        (head,tail) = os.path.split(libsff_untrimmed_path)
        libsff_trimmed_path = os.path.join(head,tail[:-4] + ".trimmed.sff")

        try:
            com = "SFFTrim"
            com += " --in-sff %s" % (libsff_untrimmed_path)
            com += " --out-sff %s" % (libsff_trimmed_path)
            if sfftrim_args:
                printtime("using non default args '%s'" % sfftrim_args)
                com += " " + sfftrim_args
            else:
                printtime("no special args found, using default args")

                # 3' adapter details
                qual_cutoff = reverse_primer_dict['qual_cutoff']
                qual_window = reverse_primer_dict['qual_window']
                adapter_cutoff = reverse_primer_dict['adapter_cutoff']
                adapter = reverse_primer_dict['sequence']

                com += " --flow-order %s" % (floworder)
                com += " --key %s" % (libKey)
                com += " --qual-cutoff %s" % (qual_cutoff)
                com += " --qual-window-size %s" % (qual_window)
                com += " --adapter-cutoff %s" % (adapter_cutoff)
                com += " --adapter %s" % (adapter)
                com += " --min-read-len 5"
                if generate_beadsummary:
                    com += " --bead-summary %s" % (os.path.join(BASECALLER_RESULTS, 'BaseCaller.json'))

            printtime("DEBUG: Calling '%s':" % com)
            ret = subprocess.call(com,shell=True)
            blockprocessing.add_status("SFFTrim", ret)
        except:
            printtime('Failed SFFTrim')
            traceback.print_exc()


        if os.path.exists(libsff_untrimmed_path):
            printtime ("DEBUG: remove untrimmed file %s" % libsff_untrimmed_path)
            os.remove(libsff_untrimmed_path)
        else:
            printtime ("ERROR: untrimmed file not found: %s" % libsff_untrimmed_path)

        if os.path.exists(libsff_trimmed_path):
            printtime ("DEBUG: Renaming %s to %s" % (libsff_trimmed_path,libsff_path))
            os.rename(libsff_trimmed_path,libsff_path)
    else:
        printtime("Not attempting to trim the SFF")


    #####################################################
    # Barcode trim SFF if barcodes have been specified  #
    # Creates one fastq per barcode, plus unknown reads #
    #####################################################

    if barcodeId != '':
        try:

            (head,tail) = os.path.split(libsff_path)
            libsff_bctrimmed_path = os.path.join(head,tail[:-4] + ".bctrimmed.sff")
            
            if not os.path.exists(DIR_BC_FILES):
              os.mkdir(DIR_BC_FILES)

            com = "barcodeSplit"
            com += " -s"
            com += " -i %s" % libsff_path
            com += " -b %s" % barcodeList_path
            com += " -k %s" % bfmask_path
            com += " -f %s" % floworder
            com += " -l %s" % barcodesplit_filter
            com += " -c %s" % barcodeMask_path
            com += " -d %s" % DIR_BC_FILES

            printtime("DEBUG: Calling '%s'" % com)
            ret = subprocess.call(com,shell=True)
            blockprocessing.add_status("barcodeSplit", ret)

            if int(ret) != 0:
                printtime("ERROR Failed barcodeSplit with return code %d" % int(ret))
            else:

                # barcodeSplit is producing "bctrimmed_"+libsff_path , rename

                (head,tail) = os.path.split(libsff_path)
                bcsff = os.path.join(DIR_BC_FILES,head,"bctrimmed_"+tail)
                if os.path.exists(bcsff):
                    printtime ("Renaming %s to %s" % (bcsff, libsff_bctrimmed_path))
                    os.rename(bcsff,libsff_bctrimmed_path)
                else:
                    printtime ("ERROR: Renaming: File not found: %s" % bcsff)

                if os.path.exists(libsff_path):
                    printtime ("DEBUG: remove file %s" % libsff_path)
                    os.remove(libsff_path)
                else:
                    printtime ("ERROR: Remove: File not found: %s" % libsff_path)
 
                #rename: libsff_path contains now the trimmed/bctrimmed data
                if os.path.exists(libsff_bctrimmed_path):
                    printtime ("Renaming %s to %s" % (libsff_bctrimmed_path,libsff_path))
                    os.rename(libsff_bctrimmed_path,libsff_path)

        except:
            printtime("ERROR Failed barcodeSplit")
            traceback.print_exc()

        # implement barcode filtering by moving filtered files
        if float(barcodesplit_filter) > 0:
            from ion.utils.filter_barcodes import filter_barcodes
            filter_barcodes(DIR_BC_FILES)

    ##################################################
    # Once we have the new SFF, run SFFSummary
    # to get the predicted quality scores
    ##################################################

    try:
        com = "SFFSummary"
        com += " -o %s" % os.path.join(BASECALLER_RESULTS, 'quality.summary')
        com += " --sff-file %s" % libsff_path
        com += " --read-length 50,100,150"
        com += " --min-length 0,0,0"
        com += " --qual 0,17,20"
        com += " -d %s" % os.path.join(BASECALLER_RESULTS, 'readLen.txt')

        printtime("DEBUG: Calling '%s'" % com)
        ret = subprocess.call(com,shell=True)
        blockprocessing.add_status("SFFSummary", ret)
    except:
        printtime('Failed SFFSummary')


    printtime("make the read length histogram")
    try:
        filepath_readLenHistogram = os.path.join(BASECALLER_RESULTS,'readLenHisto.png')
        trimmedReadLenHisto.trimmedReadLenHisto('readLen.txt',filepath_readLenHistogram)
    except:
        printtime("Failed to create %s" % filepath_readLenHistogram)


    #####################################################
    # make keypass.fastq file -c(cut key) -k(key flows) #
    #####################################################

    try:
        com = "SFFRead"
        com += " -q %s" % libsff_path.replace(".sff",".fastq")
        com += " %s" % libsff_path
        com += " > %s" % os.path.join(BASECALLER_RESULTS, 'keypass.summary')

        printtime("DEBUG: Calling '%s'" % com)
        ret = subprocess.call(com,shell=True)
        blockprocessing.add_status("SFFRead", ret)
    except:
        printtime('Failed SFFRead')
        printtime('Failed to convert SFF ' + str(libsff_path) + ' to fastq')


def tf_processing(
      SIGPROC_RESULTS,
      tfsff_path,
      libKey,
      tfKey,
      floworder,
      BASECALLER_RESULTS):


    ##################################################
    #generate TF Metrics                             #
    ##################################################

    printtime("Calling TFPipeline.processBlock")
    TFPipeline.processBlock(tfsff_path, BASECALLER_RESULTS, SIGPROC_RESULTS, tfKey, floworder)
    printtime("Completed TFPipeline.processBlock")



    ########################################################
    #Generate Raw Data Traces for lib and TF keys          #
    ########################################################
    printtime("Generate Raw Data Traces for lib and TF keys(iontrace_Test_Fragment.png, iontrace_Library.png)")

    tfRawPath = 'avgNukeTrace_%s.txt' % tfKey
    libRawPath = 'avgNukeTrace_%s.txt' % libKey
    peakOut = 'raw_peak_signal'

    if os.path.exists(tfRawPath):
        try:
            kp = plotKey.KeyPlot(tfKey, floworder, 'Test Fragment')
            kp.parse(tfRawPath)
            kp.dump_max(peakOut)
            kp.plot()
        except:
            printtime("TF key graph didn't render")
            traceback.print_exc()

    if os.path.exists(libRawPath):
        try:
            kp = plotKey.KeyPlot(libKey, floworder, 'Library')
            kp.parse(libRawPath)
            kp.dump_max(peakOut)
            kp.plot()
        except:
            printtime("Lib key graph didn't render")
            traceback.print_exc()


    ########################################################
    # Make per region key incorporation traces             #
    ########################################################
    printtime("Make per region key incorporation traces")
    perRegionTF = "averagedKeyTraces_TF.txt"
    perRegionLib = "averagedKeyTraces_Lib.txt"
    if os.path.exists(perRegionTF):
        pr = plotRawRegions.PerRegionKey(tfKey, floworder,'TFTracePerRegion.png')
        pr.parse(perRegionTF)
        pr.plot()

    if os.path.exists(perRegionLib):
        pr = plotRawRegions.PerRegionKey(libKey, floworder,'LibTracePerRegion.png')
        pr.parse(perRegionLib)
        pr.plot()


def mergeBasecallerResults(dirs, QualityPath, merged_bead_mask_path, floworder, libsff, tfsff, BASECALLER_RESULTS):
    ############################################
    # Merge individual quality.summary files #
    ############################################
    printtime("Merging individual quality.summary files")

    config_out = ConfigParser.RawConfigParser()
    config_out.optionxform = str # don't convert to lowercase
    config_out.add_section('global')

    numberkeys = ['Number of 50BP Reads',
                  'Number of 100BP Reads',
                  'Number of 150BP Reads',
                  'Number of Reads at Q0',
                  'Number of Bases at Q0',
                  'Number of 50BP Reads at Q0',
                  'Number of 100BP Reads at Q0',
                  'Number of 150BP Reads at Q0',
                  'Number of Reads at Q17',
                  'Number of Bases at Q17',
                  'Number of 50BP Reads at Q17',
                  'Number of 150BP Reads at Q17',
                  'Number of 100BP Reads at Q17',
                  'Number of Reads at Q20',
                  'Number of Bases at Q20',
                  'Number of 50BP Reads at Q20',
                  'Number of 100BP Reads at Q20',
                  'Number of 150BP Reads at Q20']

    maxkeys = ['Max Read Length at Q0',
               'Max Read Length at Q17',
               'Max Read Length at Q20']

    meankeys = ['System SNR',
                'Mean Read Length at Q0',
                'Mean Read Length at Q17',
                'Mean Read Length at Q20']

    config_in = MyConfigParser()
    config_in.optionxform = str # don't convert to lowercase
    doinit = True
    for i,subdir in enumerate(dirs):
        if isbadblock(subdir, "Merging quality.summary"):
            continue
        summaryfile=os.path.join(BASECALLER_RESULTS, subdir, 'quality.summary')
        if os.path.exists(summaryfile):
            printtime("INFO: process %s" % summaryfile)
            config_in.read(summaryfile)
            for key in numberkeys:
                value_in = config_in.get('global',key)
                if doinit:
                    value_out = 0
                else:
                    value_out = config_out.get('global', key)
                config_out.set('global', key, int(value_in) + int(value_out))
            for key in maxkeys:
                value_in = config_in.get('global',key)
                if doinit:
                    value_out = 0
                else:
                    value_out = config_out.get('global', key)
                config_out.set('global', key, max(int(value_in),int(value_out)))
            for key in meankeys:
                value_in = config_in.get('global',key)
                if doinit:
                    value_out = 0
                else:
                    value_out = config_out.get('global', key)
                config_out.set('global', key, float(value_out)+float(value_in)/len(dirs))
            doinit = False
        else:
            printtime("ERROR: skipped %s" % summaryfile)

    with open(QualityPath, 'wb') as configfile:
        config_out.write(configfile)

    ##################################################
    #generate TF Metrics                             #
    #look for both keys and append same file         #
    ##################################################

    printtime("Merging TFMapper metrics and generating TF plots")

    try:
        TFPipeline.mergeBlocks(BASECALLER_RESULTS,dirs,floworder)

    except:
        printtime("ERROR: Merging TFMapper metrics failed")


    ###############################################
    # Merge BaseCaller.json files                 #
    ###############################################
    printtime("Merging BaseCaller.json files")

    try:
        basecallerfiles = []
        for subdir in dirs:
            subdir = os.path.join(BASECALLER_RESULTS,subdir)
            printtime("DEBUG: %s:" % subdir)
            if isbadblock(subdir, "Merging BaseCaller.json files"):
                continue
            basecallerjson = os.path.join(subdir,'BaseCaller.json')
            if os.path.exists(basecallerjson):
                basecallerfiles.append(subdir)
            else:
                printtime("ERROR: Merging BaseCaller.json files: skipped %s" % basecallerjson)

        mergeBaseCallerJson.merge(basecallerfiles,BASECALLER_RESULTS)
    except:
        printtime("Merging BaseCaller.json files failed")


    ########################################
    # Merge individual block SFF files     #
    ########################################
    printtime("Merging Library SFF files")
    try:
        cmd = 'SFFProtonMerge'
        cmd = cmd + ' -i rawlib.sff'
        cmd = cmd + ' -o %s ' % libsff
        for subdir in dirs:
            subdir = os.path.join(BASECALLER_RESULTS,subdir)
            if isbadblock(subdir, "Merging Library SFF files"):
                continue
            rawlibsff = os.path.join(subdir,'rawlib.sff')
            if os.path.exists(rawlibsff):
                cmd = cmd + ' %s' % subdir
            else:
                printtime("ERROR: skipped %s" % rawlibsff)
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("SFFProtonMerge failed (library)")

    printtime("Merging Test Fragment SFF files")
    try:
        cmd = 'SFFProtonMerge'
        cmd = cmd + ' -i rawtf.sff'
        cmd = cmd + ' -o %s ' % tfsff
        for subdir in dirs:
            subdir = os.path.join(BASECALLER_RESULTS,subdir)
            if isbadblock(subdir, "Merging Test Fragment SFF files"):
                continue
            rawtfsff = os.path.join(subdir,'rawtf.sff')
            if os.path.exists(rawtfsff):
                cmd = cmd + ' %s' % subdir
            else:
                printtime("ERROR: skipped %s" % rawtfsff)
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("SFFProtonMerge failed (test fragments)")
