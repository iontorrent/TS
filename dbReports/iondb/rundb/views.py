# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
"""
Views
=====

The ``views`` module contains all the Python functions which handle requests
to the Torrent PC Analysis Suite frontend. Each function is a Django "view,"
in that the Django system will forward HTTP requests to the view, which will
then return a Django HTTP response, which is then passed on to the user.

The views in this module serve several purposes:

* Starting new analyses.
* Finding and sorting experiments.
* Finding and sorting experiment analysis reports.
* Monitoring the status of background processes (such as the `crawler` or the
  job `server`.

Not all functions contained in ``views`` are actual Django views, only those
that take ``request`` as their first argument and appear in a ``urls`` module
are in fact Django views.
"""

import datetime
import tempfile
import csv
import os
import re
import urllib
import logging
import traceback
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login
from django.db.models import Q
from tastypie.bundle import Bundle
from iondb.rundb import tasks

import json
# Handles serialization of decimal and datetime objects

#for sorting a list of lists by a key
from operator import itemgetter

from django import shortcuts, template
from django.conf import settings
from django.core.paginator import Paginator, InvalidPage, EmptyPage
from django.core import urlresolvers
from django import http

os.environ['MPLCONFIGDIR'] = '/tmp'

from iondb.rundb import forms, models
from iondb.utils import tables, toBoolean


FILTERED = None # contains the last filter for the reports page for csv export

logger = logging.getLogger(__name__)


@login_required
def tf_csv(request):
    """Return a comma separated values list of all test fragment metrics."""
    #tbl = models.Results.to_pretty_table(models.Results.objects.all())
    global FILTERED
    if FILTERED == None:
        tbl = models.Results.to_pretty_table(models.Results.objects.all())
    else:
        tbl = models.Results.to_pretty_table(FILTERED)
    ret = http.HttpResponse(tables.table2csv(tbl), mimetype='text/csv')
    now = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    ret['Content-Disposition'] = 'attachment; filename=metrics_%s.csv' % now
    return ret

def remove_experiment(request, page=None):
    """TODO: Blocked on modifying the database schema"""
    pass

@login_required
@csrf_exempt
def displayReport(request, pk):
    ctx = {}
    ctx = template.RequestContext(request, ctx)
    return shortcuts.render_to_response("rundb/reports/report.html", context_instance=ctx)


@login_required
def blank(request, **kwargs):
    """
    just render a blank template
    """
    return shortcuts.render_to_response("rundb/reports/30_default_report.html",
        {'tab':kwargs['tab']})

def getCSV(request):
    CSVstr = ""
    try:
        table = models.Results.to_pretty_table(models.Results.objects.all())
        for k in table:
            for val in k:
                CSVstr += '%s,'%val
            CSVstr = CSVstr[:(len(CSVstr)-1)] + '\n'
    except Exception as inst:
        pass
    ret = http.HttpResponse(CSVstr, mimetype='text/csv')
    now = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    ret['Content-Disposition'] = 'attachment; filename=metrics_%s.csv' % now
    return ret

# ============================================================================
# Global configuration processing and helpers
# ============================================================================

@login_required
def PDFGen(request, pkR):
    from iondb.utils import makePDF
    pkR = pkR[:len(pkR)-4]
    return http.HttpResponse(makePDF.getPDF(pkR), mimetype="application/pdf")

@login_required
def PDFGenOld(request, pkR):
    from iondb.utils import makePDF
    pkR = pkR[:len(pkR)-4]
    return http.HttpResponse(makePDF.getOldPDF(pkR), mimetype="application/pdf")


@login_required
def viewLog(request, pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    try:
        log = []
        for datum in ret.metaData["Log"]:
            logList = []
            for dat in datum:
                logList.append("%s: %s"%(dat, datum[dat]))
            log.append(logList)
    except:
        log = [["no actions have been taken on this report."]]
    ctxd = {"log":log}
    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_reportLog.html",
                                        context_instance=context)



def barcodeData(filename,metric=None):
    """
    Read the barcode alignment summary file, parse it and return data for the API and graph.
    if no metric is given return all the data
    """
    dictList = []

    #FIX, we got bugs
    if not os.path.exists(filename):
        logger.error("Barcode data file does not exist: '%s'", filename)
        return False

    try:
        fh = open(filename, "rU")
        reader = csv.DictReader(fh)
        for row in reader:
            if metric:
                if metric in row:
                    try:
                        d = {"axis": int(row["Index"]),
                             "name": row["ID"],
                             "value" : float(row[metric]),
                             "sequence" : row["Sequence"],
                             "adapter": '',
                             "annotation" : '',
                        }
                        if "Adapter" in row:
                            d["adapter"] = row["Adapter"]
                        if "Adapter" in row and "Annotation" in row:
                            d["annotation"] = row["Annotation"]
                        elif "Annotation" in row:
                            # V1 had only Annotation column, but it was really adapter sequence
                            d["adapter"] = row["Annotation"]
                        dictList.append(d)
                    except (KeyError, ValueError) as e:
                        ## Could have truncated data!
                        logger.exception(row)
                else:
                    logger.error("Metric missing: '%s'", metric)
            else:
                del row[""] ## Delete empty string (from trailing comma)
                #return a list of dicts where each dict is one row
                dictList.append(row)
    except (IOError, csv.Error) as e:
        ## Could have truncated data!
        logger.exception(e)
    except:
        logger.exception()
        return False

    if not dictList:
        logger.warn("Empty Metric List")
        return False

    #now sort by the "axis" label
    if metric:
        dictList = sorted(dictList, key=itemgetter('axis'))
    else:
        dictList = sorted(dictList, key=itemgetter('Index'))

    return dictList


def pretty(d, indent=0):
    if d:
        for key, value in d.iteritems():
            logger.debug('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                logger.debug('\t' * (indent+1) + str(value))
