#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
from ion.utils.blockprocessing import printtime


def handle_sigproc(SIGPROC_RESULTS):

    legacy_impostor_list = [
        (os.path.join(SIGPROC_RESULTS,"analysis.bfmask.bin"),os.path.join(SIGPROC_RESULTS,"bfmask.bin")),
        (os.path.join(SIGPROC_RESULTS,"analysis.bfmask.stats"),os.path.join(SIGPROC_RESULTS,"bfmask.stats")),
        (os.path.join(SIGPROC_RESULTS,"Bead_density_1000.png"),os.path.join(SIGPROC_RESULTS,"Bead_density_contour.png")),
        (os.path.join(SIGPROC_RESULTS,"Bead_density_200.png"),os.path.join(SIGPROC_RESULTS,"Bead_density_contour.png")),
        (os.path.join(SIGPROC_RESULTS,"Bead_density_70.png"),os.path.join(SIGPROC_RESULTS,"Bead_density_contour.png")),
        (os.path.join(SIGPROC_RESULTS,"Bead_density_20.png"),os.path.join(SIGPROC_RESULTS,"Bead_density_contour.png")),
    ]

    for required_file, replacement_file in legacy_impostor_list:
        if not os.path.exists(required_file):
            try:
                os.symlink(os.path.relpath(replacement_file,os.path.dirname(required_file)),required_file)
            except:
                printtime("ERROR: Unable to symlink '%s' to '%s'" % (replacement_file, required_file))

if __name__=="__main__":
    
    handle_sigproc('.')

