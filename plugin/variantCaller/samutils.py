# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import pysam
import os

def GetLibraryFromSam(samfile):
    head, tail = os.path.split( samfile )
    base, extension = os.path.splitext( samfile )
    extension = extension.lower()
    f = None
    if extension == "sam":
        f = pysam.Samfile(samfile, "r")
    else:
        f = pysam.Samfile(samfile, "rb")
        
    return f.header['SQ'][0]['SN']
    
    
