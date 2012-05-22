#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import subprocess
import traceback
import shutil

from ion.reports import processParametersMerge
from ion.reports import beadDensityPlot, MaskMerge, StatsMerge
from ion.utils.blockprocessing import printtime, isbadblock
from ion.utils import blockprocessing

def sigproc(analysisArgs, pathtorawblock, SIGPROC_RESULTS):
    printtime("RUNNING SINGLE BLOCK ANALYSIS")

    command = "%s >> ReportLog.html 2>&1" % (analysisArgs)
    printtime("Analysis command: " + command)
    sys.stdout.flush()
    sys.stderr.flush()
    status = subprocess.call(command,shell=True)
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
#            printtime("DEBUG: Calling '%s':" % com)
#            ret = subprocess.call(com,shell=True)
        except:
            traceback.print_exc()

    ########################################################
    #Make Bead Density Plots                               #
    ########################################################
    printtime("Make Bead Density Plots")
    bfmaskPath = os.path.join(SIGPROC_RESULTS,"analysis.bfmask.bin")
    maskpath = os.path.join(SIGPROC_RESULTS,"MaskBead.mask")

    if os.path.isfile(bfmaskPath):
        com = "BeadmaskParse"
        com += " -m MaskBead"
        com += " %s" % bfmaskPath
        ret = subprocess.call(com,shell=True)
        blockprocessing.add_status("BeadmaskParse", ret)
        try:
            shutil.move('MaskBead.mask', maskpath)
        except:
            printtime("ERROR: MaskBead.mask already moved")
    else:
        printtime("Warning: no analysis.bfmask.bin file exists.")

    if os.path.exists(maskpath):
        try:
            # Makes Bead_density_contour.png
            beadDensityPlot.genHeatmap(maskpath, SIGPROC_RESULTS)
#            os.remove(maskpath)
        except:
            traceback.print_exc()
    else:
        printtime("Warning: no MaskBead.mask file exists.")

    printtime("Finished single block analysis")


def mergeSigProcResults(dirs, pathToRaw, skipchecksum, SIGPROC_RESULTS):
    #####################################################
    # Grab one of the processParameters.txt files       #
    #####################################################
    printtime("Merging processParameters.txt")

    for subdir in dirs:
        subdir = os.path.join(SIGPROC_RESULTS,subdir)
        ppfile = os.path.join(subdir,'processParameters.txt')
        printtime(ppfile)
        if os.path.isfile(ppfile):
            processParametersMerge.processParametersMerge(ppfile,True)
            break



    ########################################################
    # write composite return code                          #
    ########################################################
    composite_return_code=0

    for subdir in dirs:
        if subdir == "block_X0_Y9331":
            continue
        if subdir == "block_X14168_Y9331":
            continue
        if subdir == "block_X0_Y0":
            continue
        if subdir == "block_X14168_Y0":
            continue

        try:
            f = open(os.path.join(SIGPROC_RESULTS,subdir,"analysis_return_code.txt"), 'r')
            analysis_return_code = int(f.read(1))
            f.close()
            if analysis_return_code!=0:
                printtime("DEBUG: errors in %s " % subdir)
                composite_return_code=1
                break
        except:
            traceback.print_exc()

    csp = os.path.join(pathToRaw,'checksum_status.txt')
    if not os.path.exists(csp) and not skipchecksum and len(dirs)==96:
        printtime("DEBUG: create checksum_status.txt")
        try:
            os.umask(0002)
            f = open(csp, 'w')
            f.write(str(composite_return_code))
            f.close()
        except:
            traceback.print_exc()
    else:
        printtime("DEBUG: skip generation of checksum_status.txt")


    #################################################
    # Merge individual block bead metrics files     #
    #################################################
    printtime("Merging individual block bead metrics files")

    try:
        _tmpfile = os.path.join(SIGPROC_RESULTS,'bfmask.bin')
        cmd = 'BeadmaskMerge -i bfmask.bin -o ' + _tmpfile
        for subdir in dirs:
            subdir = os.path.join(SIGPROC_RESULTS,subdir)
            if isbadblock(subdir, "Merging individual block bead metrics files"):
                continue
            bfmaskbin = os.path.join(subdir,'bfmask.bin')
            if os.path.exists(bfmaskbin):
                cmd = cmd + ' %s' % subdir
            else:
                printtime("ERROR: skipped %s" % bfmaskbin)
        printtime("DEBUG: Calling '%s'" % cmd)
        subprocess.call(cmd,shell=True)
    except:
        printtime("BeadmaskMerge failed (test fragments)")



    ###############################################
    # Merge individual block bead stats files     #
    ###############################################
    printtime("Merging bfmask.stats files")

    try:
        bfmaskstatsfiles = []
        for subdir in dirs:
            subdir = os.path.join(SIGPROC_RESULTS,subdir)
            if isbadblock(subdir, "Merging bfmask.stats files"):
                continue
            bfmaskstats = os.path.join(subdir,'bfmask.stats')
            if os.path.exists(bfmaskstats):
                bfmaskstatsfiles.append(subdir)
            else:
                printtime("ERROR: Merging bfmask.stats files: skipped %s" % bfmaskstats)

        StatsMerge.main_merge(bfmaskstatsfiles, True)
        #TODO
        shutil.move('bfmask.stats', SIGPROC_RESULTS)
    except:
        printtime("No bfmask.stats files were found to merge")

    ###############################################
    # Merge individual block MaskBead files       #
    ###############################################
#    printtime("Merging MaskBead.mask files")
#
#    try:
#        bfmaskfolders = []
#        for subdir in dirs:
#            subdir = os.path.join(SIGPROC_RESULTS,subdir)
#            printtime("DEBUG: %s:" % subdir)
#
#            if isbadblock(subdir, "Merging MaskBead.mask files"):
#                continue
#
#            bfmaskbead = os.path.join(subdir,'MaskBead.mask')
#            if not os.path.exists(bfmaskbead):
#                printtime("ERROR: Merging MaskBead.mask files: skipped %s" % bfmaskbead)
#                continue
#
#            bfmaskfolders.append(subdir)
#
#        offset_str = "use_blocks"
#        MaskMerge.main_merge('MaskBead.mask', bfmaskfolders, merged_bead_mask_path, True, offset_str)
#    except:
#        printtime("Merging MaskBead.mask files failed")


    ########################################################
    #Make Bead Density Plots                               #
    ########################################################
    printtime("Make Bead Density Plots (composite report)")

    bfmaskPath = os.path.join(SIGPROC_RESULTS,'bfmask.bin')
    maskpath = os.path.join(SIGPROC_RESULTS,'MaskBead.mask')

    # skip if merged MaskBead.mask exists TODO
    printtime("generate MaskBead.mask")
    if os.path.isfile(bfmaskPath):
        com = "BeadmaskParse -m MaskBead %s" % bfmaskPath
        os.system(com)
        #TODO
        try:
            shutil.move('MaskBead.mask', maskpath)
        except:
            printtime("ERROR: MaskBead.mask already moved")
    else:
        printtime("Warning: %s doesn't exists." % bfmaskPath)

    printtime("generate graph")
    if os.path.exists(maskpath):
        try:
            # Makes Bead_density_contour.png
            beadDensityPlot.genHeatmap(maskpath, SIGPROC_RESULTS) # todo, takes too much time
  #          os.remove(maskpath)
        except:
            traceback.print_exc()
    else:
        printtime("Warning: no MaskBead.mask file exists.")

