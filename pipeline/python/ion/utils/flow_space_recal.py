#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from ion.utils.blockprocessing import printtime
import traceback
import subprocess
import os
import json


from ion.utils import alignment
from ion.utils import basecaller

def HPtable(dir_recalibration,
            sampleBAMFile,
            xMin,
            xMax,
            xCuts,
            yMin,
            yMax,
            yCuts,
            numFlows,
            flowCuts):
    '''Generates HP table from the mapped sample reads'''
    try:
        cmd = "calibrate --skipDroop"
        cmd += " -i %s" % sampleBAMFile
        cmd += " -o %s" % dir_recalibration
        cmd += " --xMin %d" % xMin #X_MAX=3391 =0 Y_MAX=3791 Y_MIN=0 X_CUTS=1 Y_CUTS=1 FLOW_SPAN=520
        cmd += " --xMax %d" % xMax
        cmd += " --xCuts %d" % xCuts
        cmd += " --yMin %d" % yMin
        cmd += " --yMax %d" % yMax
        cmd += " --yCuts %d" % yCuts
        cmd += " --numFlows %d" % numFlows
        cmd += " --flowCuts %d" % flowCuts
        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        if ret == 0:
            printtime("Finished HP table")
        else:
            raise RuntimeError('HP table exit code: %d' % ret)
    except:
        printtime('ERROR: HP table failed')
        raise

def QVtable(dir_recalibration,
            genome_path,
            sampleBAMFile,
            xMin,
            xMax,
            xCuts,
            yMin,
            yMax,
            yCuts,
            flowSpan):
    '''Generates a QV table from the mapped sample reads'''
    try:
        cmd = "java -jar /usr/local/share/java/FlowspaceCalibration.jar"
        cmd += " I=%s" % sampleBAMFile
        cmd += " R=%s" % genome_path
        cmd += " O=%s" % os.path.join(dir_recalibration, 'sample.csv')
        cmd += " F=%s" % os.path.join(dir_recalibration, 'sample.flow.csv')
        cmd += " X_MIN=%d" % xMin #X_MAX=3391 =0 Y_MAX=3791 Y_MIN=0 X_CUTS=1 Y_CUTS=1 FLOW_SPAN=520
        cmd += " X_MAX=%d" % xMax
        cmd += " X_CUTS=%d" % xCuts
        cmd += " Y_MIN=%d" % yMin
        cmd += " Y_MAX=%d" % yMax
        cmd += " Y_CUTS=%d" % yCuts
        cmd += " FLOW_SPAN=%d" % flowSpan
        cmd += " VALIDATION_STRINGENCY=SILENT NUM_THREADS=16 MAX_QUEUE_SIZE=8192 > %s 2>&1" % os.path.join(dir_recalibration, 'flowQVtable.log')
        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        if ret == 0:
            printtime("Finished flow QV table")
        else:
            raise RuntimeError('Flow QV table exit code: %d' % ret)
    except:
        printtime('ERROR: flow QV table failed')
        raise

def HPaggregation(dir_recalibration
                 ):
    try:
        cmd = "calibrate --performMerge"
        cmd += " -o %s" % dir_recalibration
        cmd += " --mergeParentDir %s" % dir_recalibration
        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        if ret == 0:
            printtime("Finished HP training")
        else:
            raise RuntimeError('HP training exit code: %d' % ret)
    except:
        printtime('ERROR: HP training failed')
        raise

def QVaggregation(dir_recalibration,
                  flowSpan,
                  qvtable
                 ):
    try:
        cmd = "java -classpath /usr/local/share/java/FlowspaceCalibration.jar"
        cmd += " org.iontorrent.sam2flowgram.flowspace.PerturbationTableParser"
        cmd += " -f %s" % dir_recalibration
        cmd += " -s %d" % flowSpan
        cmd += " -o %s" % qvtable
        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        if ret == 0:
            printtime("Finished flow QV aggregation")
        else:
            raise RuntimeError('Flow QV aggregation exit code: %d' % ret)
    except:
        printtime('ERROR: flow QV aggregation failed')
        raise

def base_recalib(
      SIGPROC_RESULTS,
      basecallerArgs,
      libKey,
      tfKey,
      runID,
      floworder,
      reverse_primer_dict,
      BASECALLER_RESULTS,
      barcodeId,
      barcodeSamples,
      barcodesplit_filter,
      barcodesplit_filter_minreads,
      DIR_BC_FILES,
      barcodeList_path,
      barcodeMask_path,
      libraryName,
      sample,
      site_name,
      notes,
      start_time,
      chipType,
      expName,
      resultsName,
      pgmName,
      tmap_version,
      dataset_name,
      chipflow_name
    ):
    '''Do flow space recalibration for all basecall files in a report heirarchy'''
#    frame = inspect.currentframe()
#    args, _, _, values = inspect.getargvalues(frame)
#    print 'function name "%s"' % inspect.getframeinfo(frame)[2]
#    for i in args:
#        print "    %s = %s" % (i, values[i])
#    #overwrite reverse_primer_dict
#    if not reverse_primer_dict:
#        reverse_primer_dict = {'adapter_cutoff':16,'sequence':'ATCACCGACTGCCCATAGAGAGGCTGAGAC','qual_window':30,'qual_cutoff':9}
#

    try:
        # Produce smaller basecaller results
                #                      basecallerArgs + " --calibration-training=2000000 --flow-signals-type scaled-residual",
        if not "--calibration-training=" in basecallerArgs:
            basecallerArgs = basecallerArgs + " --calibration-training=2000000"
        if not "--flow-signals-type" in basecallerArgs:
            basecallerArgs = basecallerArgs + " --flow-signals-type scaled-residual"
        basecaller.basecalling(
                      SIGPROC_RESULTS,
                      basecallerArgs,
                      libKey,
                      tfKey,
                      runID,
                      floworder,
                      reverse_primer_dict,
                      os.path.join(BASECALLER_RESULTS, "recalibration"),
                      barcodeId,
                      barcodeSamples,
                      barcodesplit_filter,
                      barcodesplit_filter_minreads,
                      DIR_BC_FILES,
                      barcodeList_path,
                      barcodeMask_path,
                      libraryName,
                      sample,
                      site_name,
                      notes,
                      start_time,
                      chipType,
                      expName,
                      resultsName,
                      pgmName)

        # load datasets_basecaller.json
        try:
            f = open(os.path.join(BASECALLER_RESULTS, "recalibration", dataset_name),'r')
            datasets_basecaller = json.load(f);
            f.close()
        except:
            printtime("ERROR: load " + dataset_name)
            traceback.print_exc()
            raise

        try:
            c = open(os.path.join(BASECALLER_RESULTS, "recalibration", chipflow_name),'r')
            chipflow = json.load(c)
            c.close()
        except:
            printtime("ERROR: load " + chipflow_name)
            traceback.print_exc()
            raise

        #collect dimension and flow info
        xMin = chipflow["BaseCaller"]['block_col_offset']
        xMax = chipflow["BaseCaller"]['block_col_size'] + xMin -1
        yMin = chipflow["BaseCaller"]['block_row_offset']
        yMax = chipflow["BaseCaller"]['block_row_size'] + yMin - 1
        yCuts = 2
        xCuts = 2
        numFlows = chipflow["BaseCaller"]['num_flows']
        flowSpan = numFlows/2
        flowCuts = 2
#        print("xMin: %d; xMax: %d; xCuts: %d; yMin: %d; yMax: %d; yCuts: %d; numFlows: %d" % (xMin, xMax, xCuts, yMin, yMax, yCuts, numFlows));

        try:
            for dataset in datasets_basecaller["datasets"]:
                read_count = dataset['read_count']
                if (read_count == 0):
                    continue
#                readsFile = os.path.join(BASECALLER_RESULTS,'recalibration',os.path.split(dataset['basecaller_bam'])[-1])
                readsFile = os.path.join(BASECALLER_RESULTS,'recalibration',dataset['basecaller_bam'])
                runname_prefix = os.path.split(dataset['file_prefix'])[-1]
                #add protection that readsFile might not exist
                if not os.path.isfile(readsFile):
                    printtime("WARNING: missing file: %s" % readsFile)
                    continue

                printtime("DEBUG: Work starting on %s" % readsFile)
                RECALIBRATION_RESULTS = os.path.join(BASECALLER_RESULTS,"recalibration", runname_prefix)
                os.makedirs(RECALIBRATION_RESULTS)
                sample_map_path = os.path.join(RECALIBRATION_RESULTS, "samplelib.bam")
                try:
                    alignment.align(
                        libraryName,
                        readsFile,
                        align_full=False,
                        sam_parsed=False,
                        bidirectional=False,
                        mark_duplicates=False,
                        realign=False,
                        skip_sorting=True,
                        aligner_opts_extra="",
                        logfile=os.path.join(RECALIBRATION_RESULTS,"alignmentQC_out.txt"),
                        output_dir=RECALIBRATION_RESULTS,
                        output_basename="samplelib")
                except:
                    traceback.print_exc()
                    raise

                try:
                    # Flow QV table generation
                    #     Input -> recalibration/samplelib.bam, genome_path
                    #     Output -> QVtable file
                    #genome_path = "/results/referenceLibrary/%s/%s/%s.fasta" % (tmap_version,libraryName,libraryName)
                    #QVtable(RECALIBRATION_RESULTS,genome_path,sample_map_path,xMin,xMax,xCuts,yMin,yMax,yCuts,flowSpan)
                    HPtable(RECALIBRATION_RESULTS,sample_map_path,xMin,xMax,xCuts,yMin,yMax,yCuts,numFlows,flowCuts)
                except:
                    traceback.print_exc()
                    raise

            #create flowQVtable.txt
            try:
                qvtable = os.path.join(BASECALLER_RESULTS, "recalibration", "flowQVtable.txt")
                #QVaggregation(
                #    os.path.join(BASECALLER_RESULTS,"recalibration"),
                #    flowSpan,
                #    qvtable
                #)
                HPaggregation(os.path.join(BASECALLER_RESULTS,"recalibration"))
                
            except:
                printtime('ERROR: Flow QV aggregation failed')
                raise

        except:
            traceback.print_exc()
            raise


    except Exception as err:
        printtime("WARNING: Recalibration is not performed: %s" % err)
        raise

    return qvtable
