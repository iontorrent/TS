#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
import subprocess
import traceback
import shutil
import time

from ion.reports import beadDensityPlot, StatsMerge
from ion.utils.blockprocessing import printtime, isbadblock
from ion.utils import blockprocessing


def update_bfmask_artifacts(bfmaskPath, bfmaskstatspath, SIGPROC_RESULTS, plot_title):
    import xmlrpclib
    from torrentserver import cluster_settings
    print("Starting Upload Analysis Metrics")
    cwd = os.getcwd()
    bfmaskstatspath = os.path.join(cwd, bfmaskstatspath)
    print(bfmaskPath)
    print(bfmaskstatspath)
    try:
        jobserver = xmlrpclib.ServerProxy("http://%s:%d" % 
            (cluster_settings.JOBSERVER_HOST, cluster_settings.JOBSERVER_PORT), 
            verbose=False, allow_none=True)
        primary_key_file = os.path.join(cwd,'primary.key')

        result = jobserver.uploadanalysismetrics(bfmaskstatspath, primary_key_file)
        print(result)
        print("Compelted Upload Analysis Metrics")
    except Exception as err:
        print("Error during analysis metrics upload %s" % err)
        traceback.print_exc()

    printtime("Make Bead Density Plots")
    try:
        beadDensityPlot.genHeatmap(bfmaskPath, bfmaskstatspath, SIGPROC_RESULTS, plot_title)
    except IOError as err:
        printtime("Bead Density Plot file error: %s" % err)
    except Exception as err:
        printtime("Bead Density Plot generation failure: %s" % err)
        traceback.print_exc()



def sigproc(analysisArgs, libKey, tfKey, pathtorawblock, SIGPROC_RESULTS, plot_title, oninstrumentanalysis):
    printtime("RUNNING SINGLE BLOCK ANALYSIS")

    if not oninstrumentanalysis:
        if analysisArgs:
            cmd = analysisArgs  # e.g /home/user/Analysis --flowlimit 80
        else:
            cmd = "Analysis"
            printtime("ERROR: Analysis command not specified, using default: 'Analysis'")

        sigproc_log_path = os.path.join(SIGPROC_RESULTS, 'sigproc.log')
        open(sigproc_log_path, 'w').close()
        
        sigproc_log = open(sigproc_log_path)

        cmd += " --librarykey=%s" % (libKey)
        cmd += " --tfkey=%s" % (tfKey)
        cmd += " --no-subdir"
        cmd += " --output-dir=%s" % (SIGPROC_RESULTS)
        cmd += " %s" % pathtorawblock
        cmd += " >> %s 2>&1" % sigproc_log_path

        printtime("Analysis command: " + cmd)
        proc = subprocess.Popen(cmd, shell=True)
        while proc.poll() is None:  
            where = sigproc_log.tell()
            lines = sigproc_log.readlines()
            if lines:
                if any(l.startswith("Beadfind Complete") for l in lines):
                    printtime("Beadfind is complete.")
                    bfmaskPath = os.path.join(SIGPROC_RESULTS,"bfmask.bin")
                    bfmaskstatspath = os.path.join(SIGPROC_RESULTS,"bfmask.stats")
                    update_bfmask_artifacts(bfmaskPath, bfmaskstatspath, SIGPROC_RESULTS, plot_title)
            else:
                sigproc_log.seek(where)
                time.sleep(1)
        sigproc_log.close()
        status = proc.wait()

        blockprocessing.add_status("Analysis", status)

        # write return code into file
        try:
            os.umask(0002)
            f = open(os.path.join(SIGPROC_RESULTS,"analysis_return_code.txt"), 'w')
            f.write(str(status))
            f.close()
        except:
            traceback.print_exc()

        if status == 2:
            printtime("Analysis finished with status '%s'" % status)
            try:
                com = "ChkDat"
                com += " -r %s" % (pathtorawblock)
#                printtime("DEBUG: Calling '%s':" % com)
#                ret = subprocess.call(com,shell=True)
            except:
                traceback.print_exc()

    ########################################################
    #Make Bead Density Plots                               #
    ########################################################
    bfmaskPath = os.path.join(SIGPROC_RESULTS,"analysis.bfmask.bin")
    bfmaskstatspath = os.path.join(SIGPROC_RESULTS,"analysis.bfmask.stats")
    update_bfmask_artifacts(bfmaskPath, bfmaskstatspath, SIGPROC_RESULTS, plot_title)
    printtime("Finished single block analysis")



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

    printtime("Finished sigproc merging")
