#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

"""
Look for existing experiments and analysis them again
"""

import os
import sys
import time
import urllib
import re
import getopt

# Django related import
os.environ['DJANGO_SETTINGS_MODULE'] = "iondb.settings"
from iondb.rundb import models


def get_default_analysis(exp):
    """analysis args from Chip"""

    chip_type = exp.chipType
    args = models.AnalysisArgs.objects.filter(
        chipType=chip_type, chip_default=True)

    if not args:
        # could be old chip_type -> "314R", 316D, ...
        chip_type = chip_type.strip('"')[0:3]
        args = models.AnalysisArgs.objects.filter(
            chipType=chip_type, chip_default=True)
        return args[0].analysisargs
    else:
        return 'Analysis'


def get_build_number(analysis_arg):
    """get Analysis build number"""

    p = os.popen("%s --version" % analysis_arg)
    for line in p:
        if line.startswith('Version ='):
            m = re.search('\(\w+\)', line)
            buildnum = m.group(0).strip('(').strip(')')
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

    report_name = '%s_%s_%s_%s_%s' % (
        exp.pretty_print_no_space(),
        get_build_number(analysis_arg),
        get_report_timestamp(timestring),
        ebr, gpu)

    if note:
        report_name = '%s_%s' % (report_name, note)

    return report_name


def get_exp_from_name(name):
    """get the exp name from the report name"""
    exp = models.Experiment.objects.filter(expName__exact=name)
    return exp[0]


def generate_post(run_name, timestamp, ebrOpt, gpu, note):
    """mirror this functions from crawler.py"""

    exp = get_exp_from_name(run_name)

    if ebrOpt.lower() == 'no_recal':
        ebrStr = 'noebr'
    else:
        ebrStr = 'ebr'

    if int(gpu) == 0:
        gpuArgs = ' --gpuWorkLoad 0'
        gpuStr = 'noGPU'
    elif int(gpu) == 1:
        gpuArgs = ' --gpuWorkLoad 1'
        gpuStr = 'GPU'
    elif int(gpu) == 2:
        gpuArgs = ' --sigproc-compute-flow 20,20:1 --gpu-flow-by-flow true --gpuWorkLoad 1'
        gpuStr = 'GPU_newPipeline'
    else:
        gpuArgs = ''
        gpuStr = ''

    if note.lower() == 'tn':
        doThumbnail = "True"
    else:
        doThumbnail = "False"

    if note.lower() == 'fcwells':
        blockArgs = "fromWells"
    else:
        blockArgs = "fromRaw"

    # source the args
    args = exp.plan.get_default_cmdline_args()
    eas, eas_created = exp.get_or_create_EAS(reusable=True)
    if exp.plan and exp.plan.latestEAS:
        exp.plan.latestEAS = eas
        exp.plan.save()

    # reset the args to latest values
    for key, value in args.items():
        setattr(eas, key, value)

    # set analysis args
    if doThumbnail == 'False':
        beadfindargs = args['beadfindargs']
        analysisargs = args['analysisargs']
    else:
        beadfindargs = args['thumbnailbeadfindargs']
        analysisargs = args['thumbnailanalysisargs']

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
        newArgs = re.sub(m.group(0), gpuArgs, analysisargs)
    else:
        newArgs = analysisargs + gpuArgs

    # save the analysis back
    if doThumbnail == "False":
        eas.beadfindargs = beadfindargs
        eas.analysisargs = newArgs
    else:
        eas.thumbnailbeadfindargs = beadfindargs
        eas.thumbnailanalysisargs = newArgs
    eas.save()

    report_name = generate_report_name(exp, timestamp, ebrStr, gpuStr, note, newArgs)

    params = urllib.urlencode({'report_name': report_name,
                               'path': exp.expDir,
                               'do_thumbnail': doThumbnail,
                               'do_base_recal': ebrOpt,
                               'blockArgs': blockArgs,
                               'realign': 'False'})

    # start analysis with urllib
    try:
        print("Start Analysis: %s" % exp)
        f = urllib.urlopen(
            'http://127.0.0.1/report/analyze/%s/0/' % exp.pk, params)
    except:
        f = None
        print("Can not start analysis for %s" % exp)

    if f:
        error_code = f.getcode()
        if error_code != 200:
            print("failed to start anlaysis %s" % exp)
    return


if __name__ == '__main__':
    argv = sys.argv[1:]
    run_name = ""
    timestamp = ""
    ebr = "True"
    gpu = 1
    note = ""

    usage = "startanalysis_batch.py" + \
            " -r experimentName" + \
            " -t timestamp (if empty, use the launch time)" + \
            " -e ebr (true/false)" + \
            " -g gpu (1/0)" + \
            " -n note (tn/fcwells)"
    try:
        opts, args = getopt.getopt(
            argv,
            "hr:t:e:g:n:",
            ["experiment=", "timestamp=", "ebr=", "gpu=", "note="])
    except getopt.GetoptError:
        print usage
        sys.exit(2)

    if opts.__len__() == 0:
        print usage
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print usage
            sys.exit(2)
        elif opt == "-r":
            run_name = arg
            print ("runName is %s" % run_name)
        elif opt == "-t":
            timestamp = arg
        elif opt == "-e":
            ebr = arg
        elif opt == "-g":
            gpu = arg
        elif opt == "-n":
            note = arg

    generate_post(run_name, timestamp, ebr, gpu, note)
