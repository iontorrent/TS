# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb import models
import subprocess
import traceback


def default_chip_args(chipName):

    try:
        # Determine slots by number of cpu sockets installed
        p1 = subprocess.Popen("/usr/bin/lscpu", stdout=subprocess.PIPE)
        p2 = subprocess.Popen(
            ["grep", "^CPU socket"], stdin=p1.stdout, stdout=subprocess.PIPE
        )
        sockets = p2.stdout.read().strip().split(":")[1]
        sockets = int(sockets)
    except Exception:
        sockets = 2
        print(traceback.format_exc())

    # Default cmdline args for this chip
    if chipName in ["314", "316", "318", "314v2", "316v2", "318v2"]:
        # PGM defaults
        args = {
            "slots": 1,
            "beadfindArgs": "justBeadFind",
            "analysisArgs": "Analysis --from-beadfind",
            "prebasecallerArgs": "BaseCaller --trim-qual-cutoff 15 --trim-qual-window-size 30 --trim-adapter-cutoff 16 --calibration-training=100000 --flow-signals-type scaled-residual",
            "basecallerArgs": "BaseCaller --trim-qual-cutoff 15 --trim-qual-window-size 30 --trim-adapter-cutoff 16",
            "thumbnailBeadfindArgs": "",
            "thumbnailAnalysisArgs": "",
            "prethumbnailBasecallerArgs": "",
            "thumbnailBasecallerArgs": "",
        }
    elif chipName in [
        "900",
        "900v2",
        "P1.0.19",
        "P1.1.16",
        "P1.1.17",
        "P1.2.18",
        "P2.0.16",
        "P2.1.16",
        "P2.2.16",
    ]:
        # Proton defaults
        args = {
            "slots": 1,
            "beadfindArgs": "justBeadFind --beadfind-minlivesnr 3 --region-size=216x224 --total-timeout 600",
            "analysisArgs": "Analysis --from-beadfind --clonal-filter-bkgmodel on --region-size=216x224 --bkg-bfmask-update off --gpuWorkLoad 1 --total-timeout 600",
            "prebasecallerArgs": "BaseCaller --keypass-filter on --phasing-residual-filter=2.0 --trim-qual-cutoff 15 --trim-qual-window-size 30 --trim-adapter-cutoff 16 --num-unfiltered 1000 --calibration-training=100000 --flow-signals-type scaled-residual",
            "basecallerArgs": "BaseCaller --keypass-filter on --phasing-residual-filter=2.0 --trim-qual-cutoff 15 --trim-qual-window-size 30 --trim-adapter-cutoff 16 --num-unfiltered 1000",
            "thumbnailBeadfindArgs": "justBeadFind --beadfind-minlivesnr 3 --region-size=100x100 --beadfind-thumbnail 1",
            "thumbnailAnalysisArgs": "Analysis --from-beadfind --clonal-filter-bkgmodel on --region-size=100x100 --bkg-bfmask-update off --gpuWorkLoad 1 --bkg-debug-param 1 --beadfind-thumbnail 1",
            "prethumbnailBasecallerArgs": "BaseCaller --keypass-filter on --phasing-residual-filter=2.0 --trim-qual-cutoff 15 --trim-qual-window-size 30 --trim-adapter-cutoff 16 --num-unfiltered 100000 --calibration-training=100000 --flow-signals-type scaled-residual",
            "thumbnailBasecallerArgs": "BaseCaller --keypass-filter on --phasing-residual-filter=2.0 --trim-qual-cutoff 15 --trim-qual-window-size 30 --trim-adapter-cutoff 16 --num-unfiltered 100000",
        }
    else:
        # Unknown chip, give some basic defaults
        args = {
            "slots": 1,
            "beadfindArgs": "justBeadFind",
            "analysisArgs": "Analysis --from-beadfind",
            "prebasecallerArgs": "BaseCaller --calibration-training=100000 --flow-signals-type scaled-residual",
            "basecallerArgs": "BaseCaller",
            "thumbnailBeadfindArgs": "justBeadFind --beadfind-thumbnail 1",
            "thumbnailAnalysisArgs": "Analysis --from-beadfind",
            "prethumbnailBasecallerArgs": "BaseCaller --calibration-training=100000 --flow-signals-type scaled-residual",
            "thumbnailBasecallerArgs": "BaseCaller",
        }
        print("WARNING: Chip default args not found for chip name = %s." % chipName)

    return args


def restore_default_chip_args():
    # Restore default cmdline args
    chips = models.Chip.objects.all()
    for c in chips:
        args = default_chip_args(c.name)
        for key, value in list(args.items()):
            setattr(c, key.lower(), value)
        c.save()
