#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import sys
import os
from djangoinit import *
from iondb.rundb import models

def getProtonDefaults():
    # return Proton default values
    gc = models.GlobalConfig.objects.get()
    
    gc.default_command_line      = "Analysis --clonal-filter-bkgmodel on --region-size=216x224 --beadfind-minlivesnr 3 --bkg-bfmask-update off --gpuWorkLoad 0"
    gc.analysisthumbnailargs     = "Analysis --clonal-filter-bkgmodel on --region-size=100x100 --beadfind-minlivesnr 3 --bkg-bfmask-update off --gpuWorkLoad 0 --beadfind-thumbnail 1 --bkg-debug-param 1"
    gc.basecallerargs            = "BaseCaller --beverly-filter 0.04,0.04,8 --keypass-filter on --phasing-residual-filter=2.0 --trim-qual-cutoff 100.0 --trim-adapter-cutoff 16"
    gc.basecallerthumbnailargs   = "BaseCaller --beverly-filter 0.04,0.04,8 --keypass-filter on --phasing-residual-filter=2.0 --trim-qual-cutoff 100.0 --trim-adapter-cutoff 16"    
    return gc    
    
def getPGMDefaults():
    # return PGM default values
    gc = models.GlobalConfig.objects.get()
    
    gc.default_command_line      = "Analysis"
    gc.analysisthumbnailargs     = ""
    gc.basecallerargs            = "BaseCaller"
    gc.basecallerthumbnailargs   = ""
    return gc
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Set Global Config analysis entries to default values")
    parser.add_argument('--pgm',default=False,action='store_true',help="Set defaults for PGM instrument data")
    parser.add_argument('--proton',default=False,action='store_true',help="Set defaults for Proton instrument data")
    
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
        
    # Parse command line
    args = parser.parse_args()
    if args.pgm and args.proton:
        print "Error.  Specify one or the other, not both"
        sys.exit(1)
    else:         
         if args.pgm:
            gc = getPGMDefaults()
            gc.save()
         elif args.proton:
            gc = getProtonDefaults()
            gc.save()   

    print "NOTE: In most cases, it is necessary to close and reopen your browser for the updated options to appear in the Start Analysis launch page."
