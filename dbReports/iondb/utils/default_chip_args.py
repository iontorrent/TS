# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb import models

def default_chip_args(chipName):
    # Default cmdline args for this chip
    if chipName in ['314','316','318']:
        # PGM defaults
        args = {
            "beadfindArgs"            : "justBeadFind",
            "analysisArgs"            : "Analysis --from-beadfind",
            "basecallerArgs"          : "BaseCaller --trim-qual-cutoff 15 --trim-qual-window-size 30 --trim-adapter-cutoff 16",
            "thumbnailBeadfindArgs"   : "",
            "thumbnailAnalysisArgs"   : "",
            "thumbnailBasecallerArgs" : ""
        }
    elif chipName == '900':
        # ProtonI defaults
        args = {
            "beadfindArgs"            : "justBeadFind --beadfind-minlivesnr 3 --region-size=216x224",
            "analysisArgs"            : "Analysis --from-beadfind --clonal-filter-bkgmodel on --region-size=216x224 --bkg-bfmask-update off --gpuWorkLoad 0",
            "basecallerArgs"          : "BaseCaller --keypass-filter on --phasing-residual-filter=2.0 --trim-qual-cutoff 15 --trim-qual-window-size 30 --trim-adapter-cutoff 16 --num-unfiltered 1000",
            "thumbnailBeadfindArgs"   : "justBeadFind --beadfind-minlivesnr 3 --region-size=100x100 --beadfind-thumbnail 1",
            "thumbnailAnalysisArgs"   : "Analysis --from-beadfind --clonal-filter-bkgmodel on --region-size=100x100 --bkg-bfmask-update off --gpuWorkLoad 0 --bkg-debug-param 1 --beadfind-thumbnail 1",
            "thumbnailBasecallerArgs" : "BaseCaller --keypass-filter on --phasing-residual-filter=2.0 --trim-qual-cutoff 15 --trim-qual-window-size 30 --trim-adapter-cutoff 16 --num-unfiltered 100000"
        }
    else:
        # Unknown chip, give some basic defaults
        args = {
            "beadfindArgs"            : "justBeadFind",
            "analysisArgs"            : "Analysis --from-beadfind",
            "basecallerArgs"          : "BaseCaller",
            "thumbnailBeadfindArgs"   : "justBeadFind --beadfind-thumbnail 1",
            "thumbnailAnalysisArgs"   : "Analysis --from-beadfind",
            "thumbnailBasecallerArgs" : "BaseCaller"
        }    
        print "WARNING: Chip default args not found for chip name = %s." % chipName
        
    return args

    
def restore_default_chip_args():
    # Restore default cmdline args
    chips = models.Chip.objects.all()
    for c in chips:
        args = default_chip_args(c.name)
        for key,value in args.items():
            setattr(c, key.lower(), value)
        c.save()
    
