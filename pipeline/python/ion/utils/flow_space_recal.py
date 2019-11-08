#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
from ion.utils.blockprocessing import printtime
import traceback
import subprocess


def calibrate(dir_recalibration, sampleBAMFile, recalibArgs, chipflow):
    try:
        if recalibArgs:
            cmd = recalibArgs
        else:
            cmd = "Calibration"

        # default parameters
        block_offset_x = chipflow["BaseCaller"]["block_col_offset"]
        block_offset_y = chipflow["BaseCaller"]["block_row_offset"]
        block_size_x = chipflow["BaseCaller"]["block_col_size"]
        block_size_y = chipflow["BaseCaller"]["block_row_size"]

        if "--block-offset" not in cmd:
            cmd += " --block-offset %d,%d" % (block_offset_x, block_offset_y)
        if "--block-size" not in cmd:
            cmd += " --block-size %d,%d" % (block_size_x, block_size_y)

        cmd += " -i %s" % sampleBAMFile
        cmd += " -o %s" % dir_recalibration

        printtime("DEBUG: Calling '%s':" % cmd)
        ret = subprocess.call(cmd, shell=True)
        if ret == 0:
            printtime(
                "Calibration generated: %s"
                % (os.path.join(dir_recalibration, "Calibration.json"))
            )
        else:
            raise RuntimeError("Calibration exit code: %d" % ret)
    except Exception:
        printtime("ERROR: HP training failed")
        traceback.print_exc()
        raise
