#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *
from django import db
from django.db import transaction
import sys
import os
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

from django.db import models
from iondb.rundb import models
from socket import gethostname
import datetime

def best_result(exp,metric):
    try:
        rset = exp.results_set.all()
        rset = rset.exclude(libmetrics__i50Q17_reads=0)
        rset = rset.exclude(libmetrics=None)
        rset = rset.order_by('-libmetrics__%s' % metric)[0]
    except IndexError:
        rset = None
    return rset
    
def best_lib_metrics(res,metric):
    try:
        ret = res.libmetrics_set.all().order_by('-%s' % metric)[0]
        ret = getattr(res.libmetrics_set.all()[0],metric)
    except IndexError:
        ret = None
    return ret
    
if __name__=="__main__":
    metric = sys.argv[1]
    if '-h' in metric:
        res = models.Results
        met = models.LibMetrics._meta.get_all_field_names()
        print 'Possible Metrics'
        for m in met:
            print m
        sys.exit(0)
    timerange = datetime.datetime.now() - datetime.timedelta(days=365)
    exp = models.Experiment.objects.filter(date__gt=timerange)
    # get best result for each experiment
    res = [best_result(e,metric) for e in exp if best_result(e,metric) is not None]
    # sort the best results of the best experiments and return the best 20
    res = sorted(res,key=lambda r: best_lib_metrics(r,metric),reverse=True)[0]
    # get the library metrics for the top 20 reports
    lbms = getattr(res.libmetrics_set.all()[0],metric)
    print "%s = %s" % (metric,lbms)
    print res.experiment
    print res
    print res.reportLink
    print res.experiment.expDir


    
