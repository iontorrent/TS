#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
import subprocess
import traceback
import time
import numpy
import ConfigParser

from ion.reports import beadDensityPlot, StatsMerge, plotKey
from ion.utils.blockprocessing import printtime, isbadblock
from ion.utils import blockprocessing


def beadfind_cmd(beadfindArgs, libKey, tfKey, pathtorawblock, SIGPROC_RESULTS, block_offset_xy):
    if beadfindArgs:
        cmd = beadfindArgs  # e.g /home/user/Beadfind -xyz
    else:
        cmd = "justBeadFind"
        printtime("ERROR: Beadfind command not specified, using default: 'justBeadFind'")

    cmd += " --librarykey=%s" % (libKey)
    cmd += " --tfkey=%s" % (tfKey)
    cmd += " --no-subdir"
    cmd += " --output-dir=%s" % (SIGPROC_RESULTS)
    # justBeadFind is currently internally deriving the block offset 
    #cmd += " --block-offset %d,%d" % block_offset_xy
    cmd += " %s" % pathtorawblock

    return cmd


def beadfind(beadfindArgs, libKey, tfKey, pathtorawblock, SIGPROC_RESULTS, block_offset_xy):

    cmd = beadfind_cmd(beadfindArgs, libKey, tfKey, pathtorawblock, SIGPROC_RESULTS, block_offset_xy)
    cmd += " >> %s 2>&1" % os.path.join(SIGPROC_RESULTS, 'sigproc.log')

    if not os.path.exists(SIGPROC_RESULTS):
        os.mkdir(SIGPROC_RESULTS)

    printtime("Beadfind command: " + cmd)
    proc = subprocess.Popen(cmd, shell=True)
    status = proc.wait()

    return status


def sigproc_cmd(analysisArgs, libKey, tfKey, pathtorawblock, SIGPROC_RESULTS):

    if analysisArgs:
        cmd = analysisArgs  # e.g /home/user/Analysis --flowlimit 80
    else:
        cmd = "Analysis"
        printtime("ERROR: Analysis command not specified, using default: 'Analysis'")

    cmd += " --librarykey=%s" % (libKey)
    cmd += " --tfkey=%s" % (tfKey)
    cmd += " --no-subdir"
    cmd += " --output-dir=%s" % (SIGPROC_RESULTS)
    cmd += " %s" % pathtorawblock

    return cmd


def sigproc(analysisArgs, libKey, tfKey, pathtorawblock, SIGPROC_RESULTS):

    cmd = sigproc_cmd(analysisArgs, libKey, tfKey, pathtorawblock, SIGPROC_RESULTS)
    cmd += " >> %s 2>&1" % os.path.join(SIGPROC_RESULTS, 'sigproc.log')

    if not os.path.exists(SIGPROC_RESULTS):
        os.mkdir(SIGPROC_RESULTS)

    printtime("Analysis command: " + cmd)
    proc = subprocess.Popen(cmd, shell=True)
    status = proc.wait()

    return status


def generate_raw_data_traces(libKey, tfKey, floworder, SIGPROC_RESULTS):
    # 
    # Generate Raw Data Traces for lib and TF keys          #
    # 
    printtime("Generate Raw Data Traces for lib and TF keys(iontrace_Test_Fragment.png, iontrace_Library.png) and raw_peak_signal file")

    tfRawPath = os.path.join(SIGPROC_RESULTS, 'avgNukeTrace_%s.txt' % tfKey)
    libRawPath = os.path.join(SIGPROC_RESULTS, 'avgNukeTrace_%s.txt' % libKey)
    peakOut = 'raw_peak_signal'

    if os.path.exists(tfRawPath):
        try:
            kp = plotKey.KeyPlot(tfKey, floworder, 'Test Fragment')
            kp.parse(tfRawPath)
            kp.dump_max(os.path.join('.', peakOut))
            kp.plot()
        except:
            printtime("TF key graph didn't render")
            traceback.print_exc()
    else:
        printtime("ERROR: %s is missing" % tfRawPath)

    if os.path.exists(libRawPath):
        try:
            kp = plotKey.KeyPlot(libKey, floworder, 'Library')
            kp.parse(libRawPath)
            kp.dump_max(os.path.join('.', peakOut))
            kp.plot()
        except:
            printtime("Lib key graph didn't render")
            traceback.print_exc()
    else:
        printtime("ERROR: %s is missing" % libRawPath)


def mergeSigProcResults(dirs, SIGPROC_RESULTS, plot_title, exclusionMask=''):

    bfmaskPath = os.path.join(SIGPROC_RESULTS, 'analysis.bfmask.bin')
    bfmaskstatspath = os.path.join(SIGPROC_RESULTS, 'analysis.bfmask.stats')

    # 
    # Merge individual block bead metrics files and generate bead stats  #
    # 
    printtime("Merging individual block bead metrics files")

    try:
        cmd = 'BeadmaskMerge -i analysis.bfmask.bin -o ' + bfmaskPath
        if exclusionMask:
            cmd += ' -e %s' % exclusionMask

        for subdir in dirs:
            subdir = os.path.join(SIGPROC_RESULTS, subdir)
            if isbadblock(subdir, "Merging individual block bead metrics files"):
                continue
            bfmaskbin = os.path.join(subdir, 'analysis.bfmask.bin')
            if os.path.exists(bfmaskbin):
                cmd = cmd + ' %s' % subdir
            else:
                printtime("ERROR: skipped %s" % bfmaskbin)
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd, shell=True)
    except:
        printtime("BeadmaskMerge failed")

    ''' Not needed: BeadmaskMerge will generate analysis.bfmask.stats with exclusion mask applied

    ###############################################
    # Merge individual block bead stats files     #
    ###############################################
    printtime("Merging analysis.bfmask.stats files")

    try:
        bfmaskstatsfiles = []
        for subdir in dirs:
            subdir = os.path.join(SIGPROC_RESULTS,subdir)
            if isbadblock(subdir, "Merging analysis.bfmask.stats files"):
                continue
            bfmaskstats = os.path.join(subdir,'analysis.bfmask.stats')
            if os.path.exists(bfmaskstats):
                bfmaskstatsfiles.append(bfmaskstats)
            else:
                printtime("ERROR: Merging bfmask.stats files: skipped %s" % bfmaskstats)

        StatsMerge.main_merge(bfmaskstatsfiles, bfmaskstatspath, True)
    except:
        printtime("ERROR: No analysis.bfmask.stats files were found to merge")
        traceback.print_exc()
    '''

    # 
    # Make Bead Density Plots                               #
    # 
    printtime("Make Bead Density Plots (composite report)")

    printtime("DEBUG: generate composite heatmap")
    if os.path.exists(bfmaskPath):
        try:
            beadDensityPlot.genHeatmap(bfmaskPath, bfmaskstatspath, "./", plot_title)
        except:
            traceback.print_exc()
    else:
        printtime("Warning: no heatmap generated.")

    printtime("Finished mergeSigProcResults")

'''
def mergeRawPeakSignals(dirs):

    ###############################################
    # Merge raw_peak_signal files                 #
    ###############################################
    printtime("Merging raw_peak_signal files")

    try:
        raw_peak_signal_files = []
        for subdir in dirs:
            printtime("DEBUG: %s:" % subdir)
            if isbadblock(subdir, "Merging raw_peak_signal files"):
                continue
            raw_peak_signal_file = os.path.join(subdir,'raw_peak_signal')
            if os.path.exists(raw_peak_signal_file):
                raw_peak_signal_files.append(raw_peak_signal_file)
            else:
                printtime("ERROR: Merging raw_peak_signal files: skipped %s" % raw_peak_signal_file)
        composite_raw_peak_signal_file = "raw_peak_signal"
        blockprocessing.merge_raw_key_signals(raw_peak_signal_files, composite_raw_peak_signal_file)
    except:
        printtime("Merging raw_peak_signal files failed")

    printtime("Finished mergeRawPeakSignals")
'''


def mergeAvgNukeTraces(dirs, SIGPROC_RESULTS, key, beads):

    # 
    # Merging avgNukeTrace_*.txt files            #
    # 
    printtime("Merging avgNukeTrace_*.txt files")

    try:
        output_trace_file = os.path.join(SIGPROC_RESULTS, 'avgNukeTrace_%s.txt' % key)
        sumAvgNukeTraceData = None
        sumWells = 0
        config = ConfigParser.RawConfigParser()

        for subdir in dirs:
            try:
                input_trace_file = os.path.join(subdir, SIGPROC_RESULTS, 'avgNukeTrace_%s.txt' % key)
                if os.path.exists(input_trace_file):
                    config.read(os.path.join(subdir, SIGPROC_RESULTS, 'bfmask.stats'))
                    wells = config.getint('global', beads)
                    labels = numpy.genfromtxt(input_trace_file, delimiter=' ',  usecols=[0], dtype=str)
                    currentAvgNukeTraceData = numpy.genfromtxt(input_trace_file, delimiter=' ')[:, 1:]
                else:
                    continue
            except:
                traceback.print_exc()
                continue

            if sumAvgNukeTraceData == None:
                sumAvgNukeTraceData = currentAvgNukeTraceData * wells
            else:
                sumAvgNukeTraceData += currentAvgNukeTraceData * wells
            sumWells += wells

        AvgNukeTraceData = sumAvgNukeTraceData / sumWells
        AvgNukeTraceTable = numpy.column_stack((labels, AvgNukeTraceData.astype('|S10')))
        numpy.savetxt(output_trace_file, AvgNukeTraceTable, fmt='%s')

    except:
        traceback.print_exc()
        printtime("ERROR: Merging %s failed" % output_trace_file)

    printtime("Finished mergeAvgNukeTraces")
