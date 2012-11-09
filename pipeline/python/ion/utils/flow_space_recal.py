#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from ion.utils.blockprocessing import printtime
import shutil
import traceback
import subprocess
import os
import fnmatch
import json

from ion.utils import blockprocessing
from ion.utils import alignment

def sampleReads(inputBasesFile,
                sampleBasesFile,
                samplingFraction):
    '''Selects a subset of reads from input and writes them to output file'''
    try:
        if samplingFraction < 0.5:
            cmd = "samtools view -bh"
            cmd += " -s %0.2f" % samplingFraction        
            cmd += " %s" % inputBasesFile
            cmd += " > %s" % sampleBasesFile
            printtime("DEBUG: Calling '%s':" % cmd)
            ret = subprocess.call(cmd,shell=True)
            #blockprocessing.add_status("Reads sampling with samtools", ret)
            if ret != 0:
                raise RuntimeError('exit code: %d' % ret)
            printtime("Finished reads sampling processing with status: %d" % ret)
        else:
            inputPathList = inputBasesFile.split(os.path.sep)
            outputPathList = sampleBasesFile.split(os.path.sep)
            #find first index both  do not agree
            diffIndex = 0
            diffLimit = len(inputPathList)
            if diffLimit > len(outputPathList):
                diffLimit  = len(outputPathList)
            diffLimit = diffLimit - 1
#            print "diffLimit: %d" %diffLimit
            while diffIndex < diffLimit:
#                print "inputPathList[%d]: %s; outputPathList[%d]: %s" % (diffIndex, inputPathList[diffIndex], diffIndex, outputPathList[diffIndex])
                if inputPathList[diffIndex] == outputPathList[diffIndex]:
                    diffIndex = diffIndex + 1
                else:
                    break
#            print "diffIndex: %d" %diffIndex
#            print "len(outputPathList): %d; len(inputPathList): %d" % (len(outputPathList), len(inputPathList))
            #reconstruct the symbolic path
            outputLinkName = ""
            for i in range(1, len(outputPathList) - diffIndex):
                if outputLinkName == "":
                    outputLinkName = ".."
                else:
                    outputLinkName = os.path.join(outputLinkName, "..")
#            print "outputLinkName: %s" % outputLinkName
            for j in range(diffIndex, len(inputPathList)):
                if outputLinkName == "":
                    outputLinkName = inputPathList[j]
                else:
                    outputLinkName = os.path.join(outputLinkName, inputPathList[j])
#            print "outputLinkName: %s" % outputLinkName
            os.symlink(outputLinkName,sampleBasesFile)
            printtime("DEBUG: creating link %s -> %s" % (sampleBasesFile, outputLinkName))
    except:
        printtime('ERROR: Reads sampling')
        raise    
    return

def QVtable(dir_recalibration,
            genome_path,
            sampleBAMFile,
            qvtablefile,
            xMin,
            xMax,
            xCuts,
            yMin,
            yMax,
            yCuts,
            flowSpan):
    '''Generates a QV table from the mapped sample reads'''
    try:
        cmd = "java -jar /usr/local/lib/java/FlowspaceCalibration.jar"
        cmd += " I=%s" % sampleBAMFile
        cmd += " R=%s" % genome_path
        cmd += " O=%s" % os.path.join(dir_recalibration, 'sample.csv')
        cmd += " F=%s" % os.path.join(dir_recalibration, 'sample.flow.csv')
        cmd += " Q=%s" % qvtablefile
        cmd += " X_MIN=%d" % xMin #X_MAX=3391 =0 Y_MAX=3791 Y_MIN=0 X_CUTS=1 Y_CUTS=1 FLOW_SPAN=520
        cmd += " X_MAX=%d" % xMax
        cmd += " X_CUTS=%d" % xCuts
        cmd += " Y_MIN=%d" % yMin
        cmd += " Y_MAX=%d" % yMax
        cmd += " Y_CUTS=%d" % yCuts
        cmd += " FLOW_SPAN=%d" % flowSpan
        cmd += " VALIDATION_STRINGENCY=SILENT NUM_THREADS=16 > %s 2>&1" % os.path.join(dir_recalibration, 'flowQVtable.log')
        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        #blockprocessing.add_status("Flow QV table", ret)
        if ret != 0:
            raise RuntimeError('exit code: %d' % ret)
    except:
        printtime('ERROR: flow QV table failed')
        raise
    printtime("Finished flow QV table")
    return

def recalBases(lib_path,
               recal_path,
               qvtable):
    '''Recalibrate unmapped reads and output recallib.sff(.bam)'''
    try:
        cmd = "SeqBoost"
        cmd += " -t %s" % qvtable
        cmd += " -o %s" % recal_path 
        cmd += " -f true %s" % lib_path
        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        #blockprocessing.add_status("Reads recalibration", ret)
        if ret != 0:
            raise RuntimeError('exit code: %d' % ret)
    except:
        printtime('ERROR: recalibration failed')
        raise
    
    printtime("Finished recalibration processing")
    return

def flowQVTraining(
    runID,
    align_full,
    DIR_BC_FILES,
    libraryName,
    sample,
    chipType,
    site_name,
    flows,
    notes,
    barcodeId,
    aligner_opts_extra,
    mark_duplicates,
    start_time,
    tmap_version,
    lib_path,
    RECALIBRATION_RESULTS,
    qvtable,
    sample_path,
    sample_map_path,
    xMin,
    xMax,
    xCuts,
    yMin,
    yMax,
    yCuts,
    flowSpan,
    samplingFraction):
        
    try:
        # Sample subset of reads for mapping
        #     Input -> rawlib.sff
        #     Output -> samplelib.sff
        sampleReads (lib_path,sample_path,samplingFraction)
    except:
        raise
        
    try:
        # Map sampled reads using tmap
        #     Input -> samplelib.sff
        #     Output -> recalibration/samplelib.bam
        cmd = "alignmentQC.pl"
        cmd += " --logfile %s" % os.path.join(RECALIBRATION_RESULTS,"alignmentQC_out.txt")
        cmd += " --output-dir %s" % RECALIBRATION_RESULTS
        cmd += " --input %s" % sample_path
        cmd += " --genome %s" % libraryName
        cmd += " --max-plot-read-len %s" % str(int(400))
        cmd += " --out-base-name samplelib"
        cmd += " --skip-alignStats"
        cmd += " >> %s 2>&1" % os.path.join(RECALIBRATION_RESULTS, 'alignment.log')
        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        if ret != 0:
            raise RuntimeError('exit code: %d' % ret)
    except:
        raise
        
    try:
        # Flow QV table generation
        #     Input -> recalibration/samplelib.bam, genome_path
        #     Output -> QVtable file
        genome_path = "/results/referenceLibrary/%s/%s/%s.fasta" % (tmap_version,libraryName,libraryName)
        QVtable(RECALIBRATION_RESULTS,genome_path,sample_map_path,qvtable,xMin,xMax,xCuts,yMin,yMax,yCuts,flowSpan)
    except:
        raise

def recalibration(
    lib_path,
    qvtable,
    recal_path,
):
        
    try:
        # Reads recalibration
        #     Input -> rawlib.sff, QVtable
        #     Output -> recallib.sff
        recalBases(lib_path,recal_path,qvtable)
    except:
        raise

def base_recalib(
    BASECALLER_RESULTS,
    runID,
    align_full,
    DIR_BC_FILES,
    libraryName,
    sample,
    chipType,
    site_name,
    flows,
    notes,
    barcodeId,
    aligner_opts_extra,
    mark_duplicates,
    start_time,
    tmap_version,
    dataset_name,
    chipflow_name):
    '''Do flow space recalibration for all basecall files in a report heirarchy'''

    # load datasets_basecaller.json
    try:
        f = open(os.path.join(BASECALLER_RESULTS, dataset_name),'r')
        datasets_basecaller = json.load(f);
        f.close()
                
    except:
        printtime("ERROR: load " + dataset_name)
        traceback.print_exc()
    
    try:            
        c = open(os.path.join(BASECALLER_RESULTS, chipflow_name),'r')
        chipflow = json.load(c)
        c.close()
    except:
        printtime("ERROR: load " + chiflow_name)
        traceback.print_exc()
        
    #collect dimension and flow info
    xMin = chipflow["BaseCaller"]['block_col_offset']
    xMax = chipflow["BaseCaller"]['block_col_size'] + xMin -1 
    yMin = chipflow["BaseCaller"]['block_row_offset']
    yMax = chipflow["BaseCaller"]['block_row_size'] + yMin - 1
    yCuts = 2
    xCuts = 2
    numFlows = chipflow["BaseCaller"]['num_flows']
    flowSpan = numFlows/2
    #print("xMin: %d; xMax: %d; xCuts: %d; yMin: %d; yMax: %d; yCuts: %d; numFlows: %d" % (xMin, xMax, xCuts, yMin, yMax, yCuts, numFlows));
    
    try:
        #get total number of reads
        totalReads = 0
        if len(datasets_basecaller["datasets"]) > 1:
            for dataset in datasets_basecaller["datasets"]:
                totalReads = totalReads+dataset["read_count"]
        else:
            totalReads = datasets_basecaller["datasets"][0]["read_count"]
        samplingFraction = 1;
        if totalReads >0:
            samplingFraction = 1.0 * 2000000 / totalReads
        
        printtime("totalReads: %d; samplingFraction: %0.2f" %(totalReads, samplingFraction))
    except:
        printtime("ERROR: calculate samplingFraction ")
        traceback.print_exc()
    
    if len(datasets_basecaller["datasets"]) > 1:
    ################################################################################
    #
    # Barcode Basecall File Processing
    #
    ################################################################################
        printtime("Processing barcode run...")
        wdir = os.getcwd()
        try:
            os.chdir(BASECALLER_RESULTS)
            printtime("Starting barcode basecalls recalibration at %s" % os.getcwd() )
            for dataset in datasets_basecaller["datasets"]:
                read_count = dataset['read_count']
                if (read_count != 0):
                    barcodeString = os.path.split(dataset['file_prefix'])[-1]
                    readsFile = "%s.basecaller.bam" % barcodeString
                    #add protection that readsFile might not exist
                    if not os.path.isfile(readsFile):
                        continue
                    
                    printtime("DEBUG: Work starting on %s" % readsFile)
                    RECALIBRATION_RESULTS = os.path.join(barcodeString,'recalibration')
                    os.makedirs(RECALIBRATION_RESULTS)
                    qvtable = os.path.join(RECALIBRATION_RESULTS, "flowQVtable.txt")
                    recal_path = os.path.join(RECALIBRATION_RESULTS, "%s.basecaller.rc.bam" % barcodeString)
                    sample_path = os.path.join(RECALIBRATION_RESULTS, "samplelib.basecaller.bam")
                    sample_map_path = os.path.join(RECALIBRATION_RESULTS, "samplelib.bam")
                    try:
                        flowQVTraining(
                            runID,
                            align_full,
                            DIR_BC_FILES,
                            libraryName,
                            sample,
                            chipType,
                            site_name,
                            flows,
                            notes,
                            '', # disable alignment of barcode files.
                            aligner_opts_extra,
                            mark_duplicates,
                            start_time,
                            tmap_version,
                            readsFile,
                            RECALIBRATION_RESULTS,
                            qvtable,
                            sample_path,
                            sample_map_path,
                            xMin,
                            xMax,
                            xCuts,
                            yMin,
                            yMax,
                            yCuts,
                            flowSpan,
                            samplingFraction)
                    except:
                        raise
         
            #QV aggregation
            try:
                cmd = "java -classpath /usr/local/lib/java/FlowspaceCalibration.jar"
                cmd += " org.iontorrent.flowspace.HPTableParser"
                cmd += " -f ./"
                cmd += " -s %d" % flowSpan
                cmd += " -o flowQVtable.txt"
                printtime("DEBUG: Calling '%s':" % cmd)
                ret = subprocess.call(cmd,shell=True)
                #blockprocessing.add_status("Flow QV aggregation", ret)
            except:
                printtime('ERROR: Flow QV aggregation failed')
                raise
            printtime("Finished Flow QV aggregation")
            
            for dataset in datasets_basecaller["datasets"]:
                read_count = dataset["read_count"]
                if (read_count != 0):
                    barcodeString = os.path.split(dataset["file_prefix"])[-1]
                    readsFile = "%s.basecaller.bam" % barcodeString
                    if not os.path.isfile(readsFile):
                        continue
                    printtime("DEBUG: Recalibration rork starting on %s" % readsFile)
                    RECALIBRATION_RESULTS = os.path.join(barcodeString,'recalibration')
                    qvtable = os.path.join(RECALIBRATION_RESULTS, "flowQVtable.txt")
                    recal_path = os.path.join(RECALIBRATION_RESULTS, "%s_rawlib.basecaller.rc.bam" % barcodeString)
                    shutil.copyfile("flowQVtable.txt", os.path.join(RECALIBRATION_RESULTS, "flowQVtable.txt"))
                    try:
                        recalibration(
                            readsFile,
                            qvtable,
                            recal_path)
                    except:
                        raise
                    # Change symbolic link to new basecall file.
                    if os.path.isfile(recal_path):
                        shutil.move(readsFile, barcodeString + ".old.basecaller.bam")
                        os.symlink(recal_path,readsFile)
                        printtime("DEBUG:Created %s -> %s" % (recal_path,readsFile))
                    else:
                        printtime("WARNING: recalibrated reads file does not exist; cannot link to %s" % recal_path)  
        except:
            printtime("WARNING: Recalibration is not performed.")
            traceback.print_exc()
        finally:
            os.chdir(wdir)
    else:  
    ################################################################################
    #
    # Composite Basecall File Processing
    #
    ################################################################################
        printtime("Processing non-barcode run...")
        try:
            run_prefix = os.path.split(datasets_basecaller["datasets"][0]["file_prefix"])[-1]
            RECALIBRATION_RESULTS = os.path.join(BASECALLER_RESULTS,"recalibration")
            try:
                os.mkdir(RECALIBRATION_RESULTS)
                os.chmod(RECALIBRATION_RESULTS,0775)
            except:
                traceback.print_exc()
                
            printtime("currenct work dir is %s" % os.getcwd())
                
            lib_path = os.path.join(BASECALLER_RESULTS,"%s.basecaller.bam" % run_prefix)
            #add protection that lib_path might not exist
            if not os.path.isfile(lib_path):
                return
            
            qvtable = os.path.join(RECALIBRATION_RESULTS, "flowQVtable.txt")
            recal_path = os.path.join(RECALIBRATION_RESULTS, "%s.basecaller.rc.bam" % run_prefix)
            sample_path = os.path.join(RECALIBRATION_RESULTS, "samplelib.basecaller.bam")
            sample_map_path = os.path.join(RECALIBRATION_RESULTS, "samplelib.bam")
            
            try:
                flowQVTraining(
                    runID,
                    align_full,
                    DIR_BC_FILES,
                    libraryName,
                    sample,
                    chipType,
                    site_name,
                    flows,
                    notes,
                    '', # barcodeId is blank so we disable alignment of barcodes here
                    aligner_opts_extra,
                    mark_duplicates,
                    start_time,
                    tmap_version,
                    lib_path,
                    RECALIBRATION_RESULTS,
                    qvtable,
                    sample_path,
                    sample_map_path,
                    xMin,
                    xMax,
                    xCuts,
                    yMin,
                    yMax,
                    yCuts,
                    flowSpan,
                    samplingFraction)
                
                recalibration(
                    lib_path,
                    qvtable,
                    recal_path)
            except:
                raise
                
            # Change the symbolic link to the recalibrated reads file
            if os.path.isfile(recal_path):
                wdir = os.getcwd()
                try:
                    os.chdir(BASECALLER_RESULTS)
                    shutil.move("%s.basecaller.bam" % run_prefix, "%s.old.basecaller.bam" % run_prefix)
                    os.symlink(os.path.join("recalibration", "%s.basecaller.rc.bam" % run_prefix), "%s.basecaller.bam" % run_prefix)
                    printtime("DEBUG:Created %s -> %s" % (os.path.join("recalibration", "%s.basecaller.rc.bam" % run_prefix), "%s.basecaller.bam" % run_prefix))
                except:
                    traceback.print_exc()
                    os.chdir(wdir)
                    raise
                finally:
                    os.chdir(wdir)
            else:
                printtime("WARNING: recalibrated reads file does not exist; cannot link to %s" % recal_path)

        except:
            printtime("WARNING: Recalibration is not performed.")
            traceback.print_exc()
    return
