#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from ion.utils.blockprocessing import printtime
import traceback
import subprocess


def calibrate(dir_recalibration, sampleBAMFile, recalibArgs, chipflow):
    try:
        if recalibArgs:
            cmd = recalibArgs
        else:
            cmd = "calibrate --skipDroop"

        # default parameters
        xMin = chipflow["BaseCaller"]['block_col_offset']
        xMax = chipflow["BaseCaller"]['block_col_size'] + xMin -1
        yMin = chipflow["BaseCaller"]['block_row_offset']
        yMax = chipflow["BaseCaller"]['block_row_size'] + yMin - 1
        yCuts = 2
        xCuts = 2
        numFlows = chipflow["BaseCaller"]['num_flows']
        flowCuts = 2

        if "--xMin" not in cmd:
            cmd += " --xMin %d" % xMin #X_MAX=3391 =0 Y_MAX=3791 Y_MIN=0 X_CUTS=1 Y_CUTS=1 FLOW_SPAN=520
        if "--xMax" not in cmd:
            cmd += " --xMax %d" % xMax
        if "--xCuts" not in cmd:
            cmd += " --xCuts %d" % xCuts
        if "--yMin" not in cmd:
            cmd += " --yMin %d" % yMin
        if "--yMax" not in cmd:
            cmd += " --yMax %d" % yMax
        if "--yCuts" not in cmd:
            cmd += " --yCuts %d" % yCuts
        if "--numFlows" not in cmd:
            cmd += " --numFlows %d" % numFlows
        if "--flowCuts" not in cmd:
            cmd += " --flowCuts %d" % flowCuts

        cmd += " -i %s" % sampleBAMFile
        cmd += " -o %s" % dir_recalibration

        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        if ret == 0:
            printtime("Finished HP table")
        else:
            raise RuntimeError('HP table exit code: %d' % ret)
    except:
        printtime('ERROR: HP training failed')
        traceback.print_exc()
        raise


def HPaggregation(dir_recalibration, recalibArgs):
    try:
        if recalibArgs:
            cmd = recalibArgs
        else:
            cmd = "calibrate"
        cmd += " --performMerge"
        cmd += " -o %s" % dir_recalibration
        cmd += " --mergeParentDir %s" % dir_recalibration

        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd,shell=True)
        if ret == 0:
            printtime("Finished HP merging")
        else:
            raise RuntimeError('HP merging exit code: %d' % ret)
    except:
        printtime('ERROR: HP merging failed')
        traceback.print_exc()
        raise

