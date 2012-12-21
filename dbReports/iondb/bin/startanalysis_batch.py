#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

"""
Look for old experiments and analysis them again

Usage:

    /opt/ion/iondb/bin/startanalysis_batch.py <experiment_name> <prefix> <project_name>

The first input arguement is the expName in the database. The second
and optional input argument will be the prefix to the report name.
By default, it is set to "Batch". The third input is also optional.
By default, it is set to what is defined in explog.txt
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

# use default crawler functions
import crawler

def get_default_analysis(exp):
    """analysis args from Chip"""
    # chip_type -> "314R", 316D, ...
    chip_type = exp.chipType.strip('"')[0:3]
    
    # chip.name -> 314, 316, 318, 900, ...
    chips = models.Chip.objects.all()
    for chip in chips:
        if chip.name == chip_type:
            return chip.analysisargs
    
    # if chip is not found
    return 'Analysis'

def get_build_number(exp):
    """get Analysis build number"""
    
    # analysis args does not always start with "Analysis"
    analysis_arg = get_default_analysis(exp)
    
    p = os.popen("%s --version" % analysis_arg)
    for line in p:
        if ("Version = " in line):
            head, sep, tail = line.partition('(')
            head, sep, tail1 =tail.partition(') (')
            buildnum=tail1.rstrip(')\n')
            return buildnum
    
    # return empty in case "Build" is not specified.
    return ""

def generate_report_name(exp):
    """generate report with a input prefix and build number as suffix"""
    if len(sys.argv) > 2:
        prefix = sys.argv[2]
    else:
        prefix = "Batch"
    buildnum = get_build_number(exp)
    report_name = '%s_%s_Build_%s' %(prefix, exp.pretty_print_no_space(), buildnum)
    return report_name

def generate_project_name(exp):
    """generate project name from parsing explog or input"""
    if len(sys.argv) > 3:
        project_name = sys.argv[3]
    else:
        fpath = load_log(exp.expDir, 'explog.txt')
        explog = parse_log(fpath)
        crawler.exp_kwargs(explog, exp.expDir)
        project_name = explog.get('project', '')
    return project_name

def get_exp_from_name(name):
    """get the exp name from the report name"""
    exp = models.Experiment.objects.filter(expName__exact=name)
    return exp[0]

def generate_post(run_name):
    """mirror this functions from crawler.py"""

    exp = get_exp_from_name(run_name)
    
    report_name = generate_report_name(exp)
    project_names = generate_project_name(exp)
    
    try:
        gc = models.GlobalConfig.objects.all().order_by('pk')[0]
        use_recal = gc.base_recalibrate
    except models.GlobalConfig.DoesNotExist:
        use_recal = False
    
    params = urllib.urlencode({'report_name':report_name,
                               'tf_config':'',
                               'path':exp.expDir,
                               'submit': ['Start Analysis'],
                               'do_thumbnail':"False",
                               'project_names':project_names,
                               'do_base_recal':use_recal})
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
    if len(sys.argv) > 1:
        run_name = sys.argv[1]
    else:
        print_usage()
        sys.exit(2)
    
	
    generate_post(run_name)
