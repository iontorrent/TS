#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

"""
Look for old experiments and analysis them again

Usage:

    /opt/ion/iondb/bin/startanalysis_batch.py <experiment_name> [ebr (true/false) ] [note] [prefix] [project_name]

The first input arguement is the expName in the database.
The second and optional input argument is the basecalling recalibration switch, default value is 'True'.
The third and optional input is a user short note, such as 'fromWell'.
The fourth and optional input is prefix to the report name. It is "PGM" by default.
The fifth and optional input is the project name, By default, it is set to what is defined in explog.txt
"""

# mirror import from crawler.py
import datetime
import glob
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib
import copy
import re

from iondb.bin.djangoinit import *
from django import db
from django.db import connection
from django.core.exceptions import ObjectDoesNotExist

from twisted.internet import reactor
from twisted.internet import task
from twisted.web import xmlrpc, server

from iondb.rundb import models
from iondb.rundb import tasks
from ion.utils.explogparser import load_log
from ion.utils.explogparser import parse_log
from iondb.utils.crawler_utils import getFlowOrder
from iondb.utils.crawler_utils import folder_mtime
from iondb.utils.crawler_utils import tdelt2secs

from time import strftime

# use default crawler functions
import crawler

class BatchAnalysisLog(crawler.CrawlLog):
    BASE_LOG_NAME = "startanal.log"

logger = BatchAnalysisLog()

def get_default_analysis(exp):
    """analysis args from Chip"""
    chip_type = exp.chipType
    args = models.AnalysisArgs.objects.filter(chipType=chip_type, chip_default=True)
    if not args:
        # could be old chip_type -> "314R", 316D, ...
        chip_type = chip_type.strip('"')[0:3]
        args = models.AnalysisArgs.objects.filter(chipType=chip_type, chip_default=True)

    if args:
        return args[0].analysisargs
    else:
        return 'Analysis'

def get_build_number(exp):
    """get Analysis build number"""

    # analysis args does not always start with "Analysis"
    analysis_arg = get_default_analysis(exp)

    p = os.popen("%s --version" % analysis_arg)
    for line in p:
        if ("Version = " in line):
            head, sep, tail = line.partition('(')
            head1, sep, tail1=tail.partition(')')
            buildnum=head1.rstrip(')\n')
            return buildnum

    # return empty in case "Build" is not specified.
    return ""
def get_report_timestamp():

    timestamp = strftime("%Y%m%d%H%M%S",time.localtime())
    return timestamp

def generate_report_name(exp, use_recal, gpu, note):
    """generate report with a input prefix and build number as suffix"""
    buildnum = get_build_number(exp)
    timestamp = get_report_timestamp()
    if use_recal.lower() == 'false':
        ebr = "noebr"
    else:
        ebr = "ebr"
    if int(gpu) == 0:
        gpuUsage = 'noGPU'
    elif int(gpu) == 1:
        gpuUsage = 'GPU'
    if note != '':
        report_name = '%s_%s_%s_%s_%s_%s' %(exp.pretty_print_no_space(), buildnum,timestamp, ebr,gpuUsage, note)
    else:
        report_name = '%s_%s_%s_%s_%s' %(exp.pretty_print_no_space(), buildnum,timestamp, ebr,gpuUsage)

    return report_name

def generate_project_name(exp):
    """generate project name from parsing explog or input"""
    if len(sys.argv) > 5:
        project_name = sys.argv[5]
    else:
        fpath = load_log(exp.expDir, 'explog.txt')
        explog = parse_log(fpath)
        crawler.exp_kwargs(explog, exp.expDir, logger.errors)
        project_name = explog.get('project', '')
    return project_name

def get_exp_from_name(name):
    """get the exp name from the report name"""
    exp = models.Experiment.objects.filter(expName__exact=name)
    return exp[0]

def generate_post(run_name, ebr, gpu, note):
    """mirror this functions from crawler.py"""

    exp = get_exp_from_name(run_name)

    #report_name = generate_report_name(exp)
    project_names = generate_project_name(exp)

    # comment out the use_recal arg check, use the one parsed from command line
    """
    try:
        gc = models.GlobalConfig.objects.all().order_by('pk')[0]
        use_recal = gc.base_recalibrate
    except models.GlobalConfig.DoesNotExist:
        use_recal = False
    """
    use_recal = ebr
    if int(gpu) == 0:
        gpuArgs = ' --gpuWorkLoad 0'
    else:
        gpuArgs = ' --gpuWorkLoad 1'
    if note.lower() == 'tn':
        doThumbnail = "True"
    else:
        doThumbnail = "False"
    # source the args
    plan = exp.plan
    args = plan.get_default_cmdline_args()
    if re.search ("--gpuWorkLoad",args['analysisargs']):
        newAnalysisArgs = re.sub(re.search("--gpuWorkLoad.{2}",args['analysisargs']).group(0),gpuArgs,args['analysisargs'])
    else:
        newAnalysisArgs = args['analysisargs'] + gpuArgs
    #print ("default base arg is %s" % args['basecallerargs'])
    #print ("new analysis args is %s" % newAnalysisArgs)
    
    # update args
    eas, eas_created = exp.get_or_create_EAS(reusable=True)
    eas.analysisargs = newAnalysisArgs
    eas.beadfindArgs = args['beadfindargs']
    eas.basecallerargs = args['basecallerargs']
    eas.save()

    report_name = generate_report_name(exp, use_recal, gpu, note)
    params = urllib.urlencode({'report_name':report_name,
                               'tf_config':'',
                               'path':exp.expDir,
                               'submit': ['Start Analysis'],
                               'do_thumbnail':doThumbnail,
                               'project_names':project_names,
                               'do_base_recal':use_recal,
                               'realign':'False'})
    # start analysis with urllib
    try:
        print("Start Analysis: %s" % exp)
        f = urllib.urlopen('http://127.0.0.1/report/analyze/%s/0/' % exp.pk, params)
    except:
        f = None
        print("Can not start analysis for %s" % exp)

    if f:
        error_code = f.getcode()
        if error_code != 200:
            print("failed to start anlaysis %s" % exp)
    return



def print_usage():
    print __doc__



if __name__ == '__main__':
    ebr = "True"
    note = ''
    gpu = 1
    if len(sys.argv) > 4:
        run_name = sys.argv[1]
        ebr = sys.argv[2]
        gpu = sys.argv[3]
        note = sys.argv[4]
    elif len(sys.argv) > 3:
        run_name = sys.argv[1]
        ebr = sys.argv[2]
        gpu = sys.argv[3]
    elif len(sys.argv) > 2:
        run_name = sys.argv[1]
        ebr = sys.argv[2]
    elif len(sys.argv) > 1:
        run_name = sys.argv[1]
    else:
        print_usage()
        sys.exit(2)

    generate_post(run_name,ebr, gpu, note)
