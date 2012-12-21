#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import subprocess
import traceback
import time
import shlex

from ion.reports import beadDensityPlot, StatsMerge, plotKey
from ion.utils.blockprocessing import printtime, isbadblock
from ion.utils import blockprocessing



def beadfind(beadfindArgs, libKey, tfKey, pathtorawblock, SIGPROC_RESULTS):

    if beadfindArgs:
        cmd = beadfindArgs  # e.g /home/user/Beadfind -xyz
    else:
        cmd = "justBeadFind"
        printtime("ERROR: Beadfind command not specified, using default: 'justBeadFind'")

    cmd += " --librarykey=%s" % (libKey)
    cmd += " --tfkey=%s" % (tfKey)
    cmd += " --no-subdir"
    cmd += " --output-dir=%s" % (SIGPROC_RESULTS)
    cmd += " %s" % pathtorawblock

    printtime("Beadfind command: " + cmd)
    proc = subprocess.Popen(shlex.split(cmd.encode('utf8')), shell=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout_value, stderr_value = proc.communicate()
    status = proc.returncode
    sys.stdout.write("%s" % stdout_value)
    sys.stderr.write("%s" % stderr_value)

    # Ion Reporter
    try:
        sigproc_log_path = os.path.join(SIGPROC_RESULTS, 'sigproc.log')
        with open(sigproc_log_path, 'a') as f:
            if stdout_value: f.write(stdout_value)
            if stderr_value: f.write(stderr_value)
    except IOError:
        traceback.print_exc()

    return status


def sigproc(analysisArgs, libKey, tfKey, pathtorawblock, SIGPROC_RESULTS):

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

    printtime("Analysis command: " + cmd)
    proc = subprocess.Popen(shlex.split(cmd.encode('utf8')), shell=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout_value, stderr_value = proc.communicate()
    status = proc.returncode
    sys.stdout.write("%s" % stdout_value)
    sys.stderr.write("%s" % stderr_value)

    # Ion Reporter
    try:
        sigproc_log_path = os.path.join(SIGPROC_RESULTS, 'sigproc.log')
        with open(sigproc_log_path, 'a') as f:
            if stdout_value: f.write(stdout_value)
            if stderr_value: f.write(stderr_value)
    except IOError:
        traceback.print_exc()

    return status


def generate_raw_data_traces(libKey, tfKey, floworder, SIGPROC_RESULTS):
    ########################################################
    #Generate Raw Data Traces for lib and TF keys          #
    ########################################################
    printtime("Generate Raw Data Traces for lib and TF keys(iontrace_Test_Fragment.png, iontrace_Library.png)")

    tfRawPath = os.path.join(SIGPROC_RESULTS, 'avgNukeTrace_%s.txt' % tfKey)
    libRawPath = os.path.join(SIGPROC_RESULTS, 'avgNukeTrace_%s.txt' % libKey)
    peakOut = 'raw_peak_signal'

    if os.path.exists(tfRawPath):
        try:
            kp = plotKey.KeyPlot(tfKey, floworder, 'Test Fragment')
            kp.parse(tfRawPath)
            kp.dump_max(os.path.join('.',peakOut))
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
            kp.dump_max(os.path.join('.',peakOut))
            kp.plot()
        except:
            printtime("Lib key graph didn't render")
            traceback.print_exc()
    else:
        printtime("ERROR: %s is missing" % libRawPath)


def mergeSigProcResults(dirs, SIGPROC_RESULTS, plot_title):

    bfmaskPath = os.path.join(SIGPROC_RESULTS,'analysis.bfmask.bin')
    bfmaskstatspath = os.path.join(SIGPROC_RESULTS,'analysis.bfmask.stats')

    ########################################################
    # write composite return code                          #
    ########################################################

    try:
        if len(dirs)==96:
            composite_return_code=96
            for subdir in dirs:

                blockstatus_return_code_file = os.path.join(subdir,"blockstatus.txt")
                if os.path.exists(blockstatus_return_code_file):

                    with open(blockstatus_return_code_file, 'r') as f:
                        text = f.read()
                        if 'Analysis=0' in text:
                            composite_return_code-=1

            composite_return_code_file = os.path.join(SIGPROC_RESULTS,"analysis_return_code.txt")
            if not os.path.exists(composite_return_code_file):
                printtime("DEBUG: create %s" % composite_return_code_file)
                os.umask(0002)
                f = open(composite_return_code_file, 'a')
                f.write(str(composite_return_code))
                f.close()
            else:
                printtime("DEBUG: skip generation of %s" % composite_return_code_file)
    except:
        traceback.print_exc()

    #################################################
    # Merge individual block bead metrics files     #
    #################################################
    printtime("Merging individual block bead metrics files")

    try:
        cmd = 'BeadmaskMerge -i analysis.bfmask.bin -o ' + bfmaskPath
        for subdir in dirs:
            subdir = os.path.join(SIGPROC_RESULTS,subdir)
            if isbadblock(subdir, "Merging individual block bead metrics files"):
                continue
            bfmaskbin = os.path.join(subdir,'analysis.bfmask.bin')
            if os.path.exists(bfmaskbin):
                cmd = cmd + ' %s' % subdir
            else:
                printtime("ERROR: skipped %s" % bfmaskbin)
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("BeadmaskMerge failed")



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


    ########################################################
    #Make Bead Density Plots                               #
    ########################################################
    printtime("Make Bead Density Plots (composite report)")

    printtime("DEBUG: generate composite heatmap")
    if os.path.exists(bfmaskPath):
        try:
            # Makes Bead_density_contour.png, TODO have to read multiple blocks
            beadDensityPlot.genHeatmap(bfmaskPath, bfmaskstatspath, SIGPROC_RESULTS, plot_title)
        except:
            traceback.print_exc()
    else:
        printtime("Warning: no heatmap generated.")


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


    printtime("Finished sigproc merging")
