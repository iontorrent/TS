#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

"""
Look for old experiments and analysis them again

Usage:

    /opt/ion/iondb/bin/startanalysis_batch.py <experiment_name> <prefix>

The first input arguement is the expName in the database. The second
and optional input argument will be the prefix to the report name.
By default, it is set to "Batch".
"""

import datetime
import json
import logging
from logging import handlers as loghandlers
import os
from os import path
import re
import sys
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django import db

from djangoinit import *
from iondb.rundb import models, views
import gc
import httplib, urllib

#get django models
from iondb.rundb import models

def get_default_command():
    """get command line argument from GlobalConfig
    it does not, however, observed the Chip specific argument"""
    gc = models.GlobalConfig.objects.all()
    ret = None
    if len(gc)>0:
        for i in gc:
            ret = i.default_command_line
    else:
        ret = 'Analysis'
    return ret

def generate_http_post(exp, server, prefix, buildnum):
    """mirror this functions from crawler.py"""
    report_name = '%s_%s_%s_Build_%s' %(prefix, exp.pretty_print(), exp.pk, buildnum)
    params = urllib.urlencode({'report_name':report_name,
                               'tf_config':'',
                               'path':exp.expDir,
                               'args':get_default_command(),
                               'submit': ['Start Analysis'],
                               'do_thumbnail':"False",
                               'forward_list':"None",
                               'reverse_list':"None"})
    print params
    headers = {"Content-type": "text/html",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    try:
        f = urllib.urlopen('http://%s/rundb/newanalysis/%s/0' % (server, exp.pk), params)
    except:
        f = urllib.urlopen('https://%s/rundb/newanalysis/%s/0' % (server, exp.pk), params)
    return 

def get_exp_from_report_name(report):
    """get the exp name from the report name"""
    exp = models.Experiment.objects.filter(expName__exact=report)
    return exp[0]

def GetBuildNumber():
    """get Analysis build number"""
    p = os.popen("%s --version" %(get_default_command()))
    for line in p:
	if ("Build:" in line):
		head, sep, tail = line.partition('(')
		buildnum=tail.rstrip(')\n')
    return buildnum

if __name__ == '__main__':
    # get build number
    buildnum=GetBuildNumber()
    runname = sys.argv[1]
    if len(sys.argv) > 2:
        prefix = sys.argv[2]
    else:
        prefix = "Batch"
    
    exp = get_exp_from_report_name(runname)	
    
    print "Starting analysis for : ", exp
    generate_http_post(exp, "localhost", prefix, buildnum)

