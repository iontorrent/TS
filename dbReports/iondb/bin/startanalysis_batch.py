#!/usr/bin/env python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

"""
Look for existing experiment and analyze with latest parameters
"""

import os
import time
import urllib
import re
import argparse

# Django related import
os.environ["DJANGO_SETTINGS_MODULE"] = "iondb.settings"
from iondb.rundb import models


def get_build_number(analysis_arg):
    """get Analysis build number"""

    p = os.popen("%s --version" % analysis_arg)
    for line in p:
        if line.startswith("Version ="):
            m = re.search("\(\w+\)", line)
            buildnum = m.group(0).strip("(").strip(")")
            return buildnum

    # return empty in case "Build" is not specified.
    return ""


def get_report_timestamp(timestring=None):
    """ get report launch time stamp """
    if timestring and not timestring.isspace():
        timestamp = timestring
    else:
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    return timestamp


def generate_report_name(exp, timestring, ebr, gpu, note, analysis_arg):
    """ report name: <exp name>_<build num>_<time stamp>_<ebr>_<gpu>_<note>"""

    report_name = "%s_%s_%s_%s_%s" % (
        exp.pretty_print_no_space(),
        get_build_number(analysis_arg),
        get_report_timestamp(timestring),
        ebr,
        gpu,
    )

    if note:
        report_name = "%s_%s" % (report_name, note)

    return report_name


def get_exp_from_name(name):
    """get the exp name from the report name"""
    exp = models.Experiment.objects.filter(expName__exact=name)
    return exp[0]


def generate_post(run_name, timestamp, ebr_opt, gpu_opt, note_opt):
    """mirror this functions from crawler.py"""

    exp = get_exp_from_name(run_name)

    if ebr_opt.lower() == "no_recal":
        ebr_str = "noebr"
    elif ebr_opt.lower() == "double":
        ebr_opt = "standard_recal"
        ebr_str = "doubleCal_ebr"
    else:
        ebr_str = "ebr"

    if int(gpu_opt) == 0:
        gpu_arg = " --gpuWorkLoad 0"
        gpu_str = "noGPU"
    elif int(gpu_opt) == 1:
        gpu_arg = " --gpuWorkLoad 1"
        gpu_str = "GPU"
    elif int(gpu_opt) == 2:
        gpu_arg = " --sigproc-compute-flow 20,20:1 --gpu-flow-by-flow true --num-regional-samples 200 --gpuWorkLoad 1"
        gpu_str = "GPU_newPipeline"
    else:
        gpu_arg = ""
        gpu_str = ""

    if note_opt.lower() == "tn":
        is_thumbnail = True
    else:
        is_thumbnail = False

    if note_opt.lower() == "fcwells":
        block_args = "fromWells"
    else:
        block_args = "fromRaw"

    # source the args
    plan_args = exp.plan.get_default_cmdline_args()
    eas, eas_created = exp.get_or_create_EAS(reusable=True)
    if exp.plan and exp.plan.latestEAS:
        exp.plan.latestEAS = eas
        exp.plan.save()

    # reset the args to latest values
    for key, value in list(plan_args.items()):
        setattr(eas, key, value)

    # set analysis args
    if is_thumbnail:
        beadfindargs = plan_args["thumbnailbeadfindargs"]
        analysisargs = plan_args["thumbnailanalysisargs"]
        prebasecallerargs = plan_args["prethumbnailbasecallerargs"]
        calibrateargs = plan_args["thumbnailcalibrateargs"]
        basecallerargs = plan_args["thumbnailbasecallerargs"]
        alignmentargs = plan_args["thumbnailalignmentargs"]
    else:
        beadfindargs = plan_args["beadfindargs"]
        analysisargs = plan_args["analysisargs"]
        prebasecallerargs = plan_args["prebasecallerargs"]
        calibrateargs = plan_args["calibrateargs"]
        basecallerargs = plan_args["basecallerargs"]
        alignmentargs = plan_args["alignmentargs"]

    # replace binary
    """
    beadfindargs = re.sub(
        'justBeadFind',
        '/results/justBeadFind.915d576',
        beadfindargs)
    analysisargs = re.sub(
        'Analysis',
        '/results/Analysis.915d576',
        analysisargs)
    """

    # replace Analysis args
    m = re.search("--gpuWorkLoad.{2}", analysisargs)
    if m:
        amended_analysisargs = re.sub(m.group(0), gpu_arg, analysisargs)
    else:
        amended_analysisargs = analysisargs + gpu_arg

    if ebr_str == "doubleCal_ebr":
        calibrateargs = calibrateargs + " --double-fit true"
        prebasecallerargs = prebasecallerargs + " --linear-hp-thres 0"
        basecallerargs = basecallerargs + " --linear-hp-thres 0"

    # save the args back
    if is_thumbnail:
        eas.thumbnailbeadfindargs = beadfindargs
        eas.thumbnailanalysisargs = amended_analysisargs
        eas.prethumbnailbasecallerargs = prebasecallerargs
        eas.thumbnailcalibrateargs = calibrateargs
        eas.thumbnailbasecallerargs = basecallerargs
        eas.thumbnailalignmentargs = alignmentargs
    else:
        eas.beadfindargs = beadfindargs
        eas.analysisargs = amended_analysisargs
        eas.prebasecallerargs = prebasecallerargs
        eas.calibrateargs = calibrateargs
        eas.basecallerargs = basecallerargs
        eas.alignmentargs = alignmentargs
    eas.save()

    report_name = generate_report_name(
        exp, timestamp, ebr_str, gpu_str, note_opt, amended_analysisargs
    )

    params = urllib.urlencode(
        {
            "report_name": report_name,
            "path": exp.expDir,
            "do_thumbnail": str(is_thumbnail),
            "do_base_recal": ebr_opt,
            "blockArgs": block_args,
            "realign": "False",
        }
    )

    # start analysis with urllib
    try:
        print("Start Analysis: %s" % exp)
        f = urllib.urlopen("http://127.0.0.1/report/analyze/%s/0/" % exp.pk, params)
    except Exception:
        f = None
        print("Can not start analysis for %s" % exp)

    if f:
        error_code = f.getcode()
        if error_code != 200:
            print("failed to start anlaysis %s" % exp)
    return


if __name__ == "__main__":
    # commandline arg parsing
    parser = argparse.ArgumentParser(description="batch start analysis from CLI")

    parser.add_argument(
        "--run-name", "-r", required=True, help="run name, a.k.a. expName"
    )
    parser.add_argument(
        "--timestamp",
        "-t",
        default=time.strftime("%Y%m%d%H%M%S", time.localtime()),
        help="timestamp of the analysis. If empty, use the launch time",
    )
    parser.add_argument(
        "--ebr",
        "-e",
        default="standard_recal",
        help="enable basecaller recalibration, i.e. no_recal will disable calibration process",
    )
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=1,
        help="enable GPU for pipeline. This is often use to modify the analysis arguments",
    )
    parser.add_argument(
        "--note",
        "-n",
        default="",
        help="distinguish thumbnail vs fullchip analysis, i.e. tn, fcwells, ...",
    )
    cmd_args = vars(parser.parse_args())

    # begin the main function
    generate_post(
        cmd_args["run_name"],
        cmd_args["timestamp"],
        cmd_args["ebr"],
        cmd_args["gpu"],
        cmd_args["note"],
    )
