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
from logging import handlers as loghandlers
import os
from os import path
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from djangoinit import *

from django import db

from twisted.internet import reactor
from twisted.internet import task
from twisted.web import xmlrpc,server

from iondb.rundb import models
from django.db import connection
from ion.utils.explogparser import load_log
from ion.utils.explogparser import parse_log

import urllib
import string

# use default crawler functions
import crawler

def get_default_command():
    """get command line argument from GlobalConfig
    it does not, however, observed the Chip specific argument"""
    gc = models.GlobalConfig.objects.all()
    ret = 'Analysis'
    if len(gc)>0:
        for i in gc:
            ret = i.default_command_line
    return ret

def get_default_basecaller_command():
    """get basecaller args"""
    gc = models.GlobalConfig.objects.all()
    ret = 'BaseCaller'
    if len(gc)>0:
        for i in gc:
            ret = i.basecallerargs
    return ret

def get_build_number():
    """get Analysis build number"""
    p = os.popen("%s --version" %(get_default_command()))
    for line in p:
	if ("Build:" in line):
		head, sep, tail = line.partition('(')
		buildnum=tail.rstrip(')\n')
    return buildnum

def generate_report_name(exp):
    """generate report with a input prefix and build number as suffix"""
    if len(sys.argv) > 2:
        prefix = sys.argv[2]
    else:
        prefix = "Batch"
    buildnum = get_build_number()
    report_name = '%s_%s_Build_%s' %(prefix, exp.pretty_print(), buildnum)
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

def generate_http_post(exp, server):
    """mirror this functions from crawler.py"""
    params = urllib.urlencode({'report_name':generate_report_name(exp),
                               'tf_config':'',
                               'path':exp.expDir,
                               'args':get_default_command(),
                               'basecallerArgs':get_default_basecaller_command(),
                               'submit': ['Start Analysis'],
                               'do_thumbnail':"False",
                               'forward_list':"None",
                               'reverse_list':"None",
                               'project_names':generate_project_name(exp)})
    try:
        print("Start Analysis: %s" % exp)
        f = urllib.urlopen('http://%s/rundb/newanalysis/%s/0' % (server, exp.pk), params)
    except:
        print("Can not start analysis for %s" % exp)
    return

def get_exp_from_report_name(report):
    """get the exp name from the report name"""
    exp = models.Experiment.objects.filter(expName__exact=report)
    return exp[0]

def print_usage():
    print __doc__



if __name__ == '__main__':
    if len(sys.argv) > 1:
        runname = sys.argv[1]
    else:
        print_usage()
        sys.exit(2)
    
    exp = get_exp_from_report_name(runname)	
    generate_http_post(exp, "localhost")
