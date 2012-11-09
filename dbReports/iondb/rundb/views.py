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
from os import path
import re
import socket
import StringIO
import subprocess
import sys
import urllib
import xmlrpclib
import stat
import logging
import traceback
import forms
import string
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import logout, authenticate, login
from django.db.models import Q
from ion.utils.TSversion import findVersions
from tastypie.bundle import Bundle

import json
from django.utils import simplejson
from django.core import serializers
# Handles serialization of decimal and datetime objects
from django.core.serializers.json import DjangoJSONEncoder

#for sorting a list of lists by a key
from operator import itemgetter

from iondb.rundb.ajax import render_to_json
from iondb.backup import rawDataStorageReport
from django import http, shortcuts, template
from django.conf import settings
from django.core.paginator import Paginator, InvalidPage, EmptyPage
from django.core import urlresolvers
from django import http
from django.forms.models import model_to_dict
from django.db import transaction
from django.http import HttpResponsePermanentRedirect
from django.template.context import RequestContext

os.environ['MPLCONFIGDIR'] = '/tmp'

from iondb.anaserve import client
from iondb.rundb import forms, models
from iondb.utils import tables
from iondb.backup import devices
from iondb.backup.archiveExp import Experiment
from twisted.internet import reactor
from twisted.web import xmlrpc, server

from iondb.rundb import publishers, tasks
from iondb.plugins.manager import PluginManager

FILTERED = None # contains the last filter for the reports page for csv export

logger = logging.getLogger(__name__)

def flattenString(string):
    return string.replace("\n"," ").replace("\r"," ").strip()

def toBoolean(val, default=True):
    """convert strings from CSV to Python bool
    if they have an empty string - default to true unless specified otherwise
    """
    if default:
        trueItems = ["true", "t", "yes", "y", "1", "" ]
        falseItems = ["false", "f", "no", "n", "none", "0" ]
    else:
        trueItems = ["true", "t", "yes", "y", "1", "on"]
        falseItems = ["false", "f", "no", "n", "none", "0", "" ]

    if str(val).strip().lower() in trueItems:
        return True
    if str(val).strip().lower() in falseItems:
        return False
    
def seconds2htime(s):
    """Convert a number of seconds to a dictionary of days, hours, minutes,
    and seconds.

    >>> seconds2htime(90061)
    {"days":1,"hours":1,"minutes":1,"seconds":1}
    """
    days = int(s / (24 * 3600))
    s -= days * 24 * 3600
    hours = int(s / 3600)
    s -= hours * 3600
    minutes = int(s / 60)
    s -= minutes * 60
    s = int(s)
    return {"days":days, "hours":hours, "minutes":minutes, "seconds":s}


def base_context_processor(request):
    """This is a hook which adds the returned dictionary to the context for
    every view in the project.  It is used to allow the site name to be
    added to the base_template in every view.
    Namespace any items added here with 'base_' in order to avoid conflict.
    """
    site_name = models.GlobalConfig.objects.all().order_by('id')[0].site_name
    messages = models.Message.objects.filter(route="").filter(
        Q(status="unread", expires="read") | ~Q(expires="read"))
    from iondb.rundb.api import MessageResource
    resource = MessageResource()
    msg_list = [resource.full_dehydrate(Bundle(message)) for message in messages]
    serialized_messages = resource.serialize(None, msg_list, "application/json")
    if msg_list: logger.debug("Found %d global messages" % len(msg_list))
    logger.debug("Global messages are %s" % serialized_messages)
    return {"base_site_name": site_name, "global_messages": serialized_messages}


def message_binding_processor(request):
    """This is called for every request but only performs work for
    those which have bindings for specific message routes.
    """
    if hasattr(request, "message_bindings"):
        messages = models.Message.objects.bound(*request.message_bindings) \
            .filter(Q(status="unread", expires="read") | ~Q(expires="read"))
        bound_messages = list(messages)
        messages.update(status="read")
        if bound_messages: logger.debug("Found %d bound messages" % len(bound_messages))
        return {"bound_messages": bound_messages}
    else:
        logger.debug("No messages found")
        return {}


def bind_messages(*bindings):
    """This returns a function which takes a view and decorates it with
    a wrapper function which adds the binding information passed to
    this function to the request object for that view so that the
    message_binding_processor can access the specific bindings for the
    view that goes through it.  This feels like a serious hack...
    """
    def bind_view(view_func):
        logger.debug("Binding %s to %s" % (str(bindings),
                                           str(view_func.__name__)))
        def bound_view(request, *args, **kwargs):
            request.message_bindings = bindings
            return view_func(request, *args, **kwargs)
        return bound_view
    return bind_view

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
def crawler_status(request):
    """Determine the crawler's status by attempting to query it over
    XMLRPC. If the ``crawler_status`` is unable to contact the crawler
    (for example, because the crawler is not running), then crawler is
    reported to be offline. Otherwise, ``crawler_status`` provides information
    on recently discovered experiment data, crawler uptime, and the crawler's
    current state (for example, "working" or "sleeping").
    """
    url = "http://127.0.0.1:%d" % settings.CRAWLER_PORT
    cstat = xmlrpclib.ServerProxy(url)
    try:
        raw_elapsed = cstat.time_elapsed()
        elapsed = seconds2htime(raw_elapsed)
        nfound = cstat.experiments_found()
        raw_exprs = cstat.prev_experiments()
        exprs = []
        for r in raw_exprs:
            try:
                exp = models.Experiment.objects.get(expName=r)
            except (models.Experiment.DoesNotExist,
                    models.Experiment.MultipleObjectsReturned):
                exp = r
            exprs.append(exp)
        folder = cstat.current_folder()
        state = cstat.state()
        hostname = cstat.hostname()
        result = [folder, elapsed, exprs, nfound, state, hostname]
        keys = ["folder", "elapsed", "exprs", "nfound", "state", "hostname"]
        result_pairs = zip(keys, result)
    except socket.error:
        result_pairs = ()
    ctx = template.RequestContext(request, {"result_dict":dict(result_pairs)})
    return ctx

@login_required
def single_experiment(request, pk):
    """Present the user with information about a single experiment."""
    try:
        pk = int(pk)
    except:
        return http.HttpResponseNotFound()
    e = shortcuts.get_object_or_404(models.Experiment, pk=pk)
    ctx = template.RequestContext(request, {"e":e})
    return shortcuts.render_to_response("rundb/ion_single_experiment.html",
                                        context_instance=ctx)

def escape_re(match):
    """Replace each match with the match escaped.

    >>> escape_re(re.match('^(match me)$','match me'))
    '\\\\match me'
    """
    return "\\%s" % match.group(0)

def wildcardify(terms):
    """Convert a wildcard search string into a regular expression.
    First, escape special regexp characters. Replace all whitespace with '\\s*'.
    Then replace escaped Kleene stars with '.*'. Parentheses are left as-is.

    >>> wildcardify("anything after this*")
    "anything\\\\s*after\\\\s*this.*"
    """
    terms = re.sub(r"[$^\*?+{}]|\[|\.]", escape_re, terms)
    terms = re.sub(r'\s+', r'\s*', terms)
    terms = re.sub(r"\\\*", ".*", terms)
    return terms

def search_and_sort(qset, getURL, search, sort, model, default_sort, namefield):
    """Apply searching and sorting criteria from the forms ``search`` and
    ``sort`` to the Django queryset ``qset``. Field names are taken from
    ``model``. The default field name to sort on is specified by
    ```default_sort``. The field to search is specified by ``namefield``.
    """
    if search.is_valid():
        terms = search.cleaned_data['searchterms'].strip().lower()
        terms = wildcardify(terms)
        kwargs = {}
        kwargs[namefield + "__iregex"] = terms
        qset = qset.filter(**kwargs)
        urlparams = urllib.urlencode(search.cleaned_data)
        if getURL is not None:
            getURL = "%s&%s" % (getURL, urlparams)
        else:
            getURL = urlparams
    is_ordered = False
    order_field = None
    if sort.is_valid():
        field = sort.cleaned_data['sortfield']
        name_map = dict((name.lower(), name) for name in
                        model._meta.get_all_field_names())
        key = field.lstrip('-').lower()
        if field.startswith('-'):
            pref = '-'
        else:
            pref = ''
        if key in name_map:
            qset = qset.order_by(pref + name_map[key], default_sort)
            is_ordered = True
            urlparams = urllib.urlencode(sort.cleaned_data)
            if getURL is not None:
                getURL = "%s&%s" % (getURL, urlparams)
            else:
                getURL = urlparams
    if not is_ordered:
        qset = qset.order_by(default_sort)
    return qset, getURL

@login_required
def experiment(request, page=None):
    """Display experiments, with filters, sorting, searching, and pagination
    applied."""
    exp = models.Experiment.objects.all().select_related()
    if 'submit' in request.GET:
        d = request.GET
    else:
        d = None
    filter = forms.ExperimentFilter(d)
    search = forms.SearchForm(d)
    sort = forms.SortForm(d)        
    fbtime = datetime.datetime.now()
    getURL = None
    if filter.is_valid():
        d = dict(filter.cleaned_data)
        d['submit'] = True
        getURL = urllib.urlencode(d)
        project = filter.cleaned_data['project']
        if project != 'None':
            exp = exp.filter(results__projects__name=project).distinct()
        sample = filter.cleaned_data['sample']
        if sample != 'None':
            exp = exp.filter(sample=sample)
        library = filter.cleaned_data['library']
        if library != 'None':
            exp = exp.filter(library=library)
        rig = filter.cleaned_data['pgm']
        if rig != 'None':
            exp = exp.filter(pgmName=rig)
        store = filter.cleaned_data['storage']
        if store != 'None':
            exp = exp.filter(storage_options=store)
        starred = filter.cleaned_data['starred']
        if starred:
            exp = exp.filter(star=starred)


        dateStart = filter.cleaned_data['date_start']
        exp = exp.filter(date__gte=dateStart)

        dateEnd = filter.cleaned_data['date_end']
        dateEndDt = datetime.datetime(dateEnd.year, dateEnd.month, dateEnd.day,
                                      23, 59, 59, 999999)
        exp = exp.filter(date__lte=dateEndDt)

    exp, getURL = search_and_sort(exp, getURL, search, sort,
                                 models.Experiment, "-date", "expName")

    paginator = Paginator(exp, models.GlobalConfig.objects.all()[0].records_to_display)
    try:
        page = int(request.GET.get('page', '1'))
    except ValueError:
        page = 1
    try:
        ctx = paginator.page(page)
    except (EmptyPage, InvalidPage):
        ctx = paginator.page(paginator.num_pages)


    #only return current indexes
    refgenomes = models.ReferenceGenome.objects.filter(index_version=settings.TMAP_VERSION,enabled=True)

    storages = models.ReportStorage.objects.all()
    storage = storages.filter(default=True)[0]   #Select default ReportStorage obj.

    '''if storage.dirPath[:len(storage.dirPath)-1] != '/':
        disk = os.statvfs(storage.dirPath+'/')
    else:'''
    disk = os.statvfs(storage.dirPath)
        
    #values in bytes
    teraBytes  = 1099511627776
    gigaBytes  = 1073741824
    totalStore = ( disk.f_bsize * disk.f_blocks ) / gigaBytes
    freeStore  = ( disk.f_bsize * disk.f_bavail ) / gigaBytes
    usedStore  = ( disk.f_bsize * (disk.f_blocks - disk.f_bavail) ) / gigaBytes

    force_desktop = request.GET.get('force_desktop', 'false').lower() == 'true'

    context = template.RequestContext(request, {"exp":ctx, "paginator":ctx,
                                                "filterform":filter,
                                                "searchform":search,
                                                "sortform":sort,
                                                "getURL":getURL,
                                                "refgenomes" : refgenomes,
                                                "totalStore" : totalStore,
                                                "freeStore" : freeStore,
                                                "usedStore" :  usedStore,
                                                "force_desktop" : force_desktop
                                                 })


    return shortcuts.render_to_response("rundb/ion_experiment.html",
                                        context_instance=context)

@login_required
def reports(request, page=None):
    """Display reports, with sorting, filtering, searching, and pagination
    applied."""
    rep = models.Results.objects.all().order_by('-timeStamp').select_related()
    if 'submit' in request.GET:
        d = request.GET
    else:
        d = None
    filter = forms.ReportFilter(d)
    search = forms.SearchForm(d)
    sort = forms.SortForm(d)
    if filter.is_valid():
        d = dict(filter.cleaned_data)
        d['submit'] = True
        getURL = urllib.urlencode(d)
        status = filter.cleaned_data['status']
        if status != 'None':
            rep = rep.filter(status=status)
        flows = filter.cleaned_data['cycles']
        if flows != 'None':
            rep = rep.filter(processedCycles=flows)
        temp = filter.cleaned_data['template']
        if temp != 'None':
            rep = rep.filter(tfmetrics__name=temp)

        project = filter.cleaned_data['project']
        if project != 'None':
            rep = rep.filter(projects__name=project)
        sample = filter.cleaned_data['sample']
        if sample != 'None':
            rep = rep.filter(experiment__sample=sample)
        library = filter.cleaned_data['library']
        if library != 'None':
            rep = rep.filter(experiment__library=library)
        dateStart = filter.cleaned_data['date_start']
        rep = rep.filter(timeStamp__gte=dateStart)

        dateEnd = filter.cleaned_data['date_end']
        dateEndDt = datetime.datetime(dateEnd.year, dateEnd.month, dateEnd.day,
                                      23, 59, 59, 999999)
        rep = rep.filter(timeStamp__lte=dateEndDt)

    else:
        getURL = None
    rep, getURL = search_and_sort(rep, getURL, search, sort, models.Results,
                                 '-timeStamp', "resultsName")
    global FILTERED
    FILTERED = rep
    if search.is_valid():
        terms = search.cleaned_data['searchterms'].strip().lower()
        terms = wildcardify(terms)
        rep = rep.filter(resultsName__iregex=terms)
        urlparams = urllib.urlencode(search.cleaned_data)
        if getURL is not None:
            getURL = "%s&%s" % (getURL, urlparams)
        else:
            getURL = urlparams

    if sort.is_valid():
        field = sort.cleaned_data['sortfield']
        name_map = dict((name.lower(), name) for name in
                        models.Results._meta.get_all_field_names())
        key = field.lstrip('-')
        if field.startswith('-'):
            pref = '-'
        else:
            pref = ''
        if key in name_map:
            rep = rep.order_by(pref + name_map[key], "-timeStamp")
            urlparams = urllib.urlencode(sort.cleaned_data)
            if getURL is not None:
                getURL = "%s&%s" % (getURL, urlparams)
            else:
                getURL = urlparams
    paginator = Paginator(rep, models.GlobalConfig.objects.all()[0].records_to_display)

    try:
        page = int(request.GET.get('page', '1'))
    except ValueError:
        page = 1

    try:
        ctx = paginator.page(page)
    except (EmptyPage, InvalidPage):
        ctx = paginator.page(paginator.num_pages)
        
    rets = models.Results.objects.all()
    bkL = models.dm_reports.objects.all().order_by('pk').reverse()
    if len(bkL) > 0:
        bkExist = "True"
        if bkL[len(bkL)-1].location != (None or ''):
            bkLocExist = "True"
        else:
            bkLocExist = "False"
    else:
        bkExist = "False"
        bkLocExist = "False"
    ctxd = {"rep":ctx, "paginator":ctx, "filterform":filter, "getURL":getURL,
            "searchform":search, "sortform":sort, "use_content2" : True, "bkL": bkExist, "bkLE": bkLocExist}
    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_report.html",
                                        context_instance=context)

def loadScript(fileIn):
    """Read a file with filename ``fileIn`` and return its contents as 
    a string."""
    f = open(fileIn, 'r')
    ret = f.read()
    f.close()
    return ret

def create_runid(name):
    '''Returns 5 char string hashed from input string'''
    #Copied from TS/Analysis/file-io/ion_util.c
    def DEKHash(key):
        hash = len(key)
        for i in key:
            hash = ((hash << 5) ^ (hash >> 27)) ^ ord(i)
        return (hash & 0x7FFFFFFF)

    def base10to36(num):
        str = ''
        for i in range(5):
            digit=num % 36
            if digit < 26:
                str = chr(ord('A') + digit) + str
            else:
                str = chr(ord('0') + digit - 26) + str
            num /= 36
        return str
        
    return base10to36(DEKHash(name))
    
def build_result(experiment, name, server, location):
    """Initialize a new `Results` object named ``name``
    representing an analysis of ``experiment``. ``server`` specifies
    the ``models.reportStorage`` for the location in which the report output
    will be stored, and ``location`` is the
    ``models.Location`` object for that file server's location.
    """
    # Final "" element forces trailing '/'
    # reportLink is used in calls to dirname, which would otherwise resolve to parent dir
    link = path.join(server.webServerPath, location.name, "%s_%%03d" % name, "")
    j = lambda l: path.join(link, l)
    storages = models.ReportStorage.objects.all()
    storage = storages.filter(default=True)[0]   #Select default ReportStorage obj.
        
    kwargs = {
        "experiment":experiment,
        "resultsName":name,
        "sffLink":j("%s_%s.sff" % (experiment, name)),
        "fastqLink":path.join(link,"basecaller_results", "%s_%s.fastq" % (experiment, name)),
        "reportLink": link, # Default_Report.php is implicit via Apache DirectoryIndex
        "status":"Pending", # Used to be "Started"
        "tfSffLink":j("%s_%s.tf.sff" % (experiment, name)),
        "tfFastq":"_",
        "log":j("log.html"),
        "analysisVersion":"_",
        "processedCycles":"0",
        "processedflows":"0",
        "framesProcessed":"0",
        "timeToComplete":0,
        "reportstorage":storage,
        }
    ret = models.Results(**kwargs)
    ret.save()
    for k, v in kwargs.iteritems():
        if hasattr(v, 'count') and v.count("%03d") == 1:
            v = v % ret.pk
            setattr(ret, k, v)
    ret.save()
    return ret

def create_meta(experiment, result):

    """Build the contents of a report metadata file (``expMeta.dat``)."""
    lines = ("Run Name = %s" % experiment.expName,
             "Run Date = %s" % experiment.date,
             "Run Cycles = %s" % experiment.cycles,
             "Run Flows = %s" % experiment.flows,
             "Project = %s" % ','.join(p.name for p in result.projects.all()),
             "Sample = %s" % experiment.sample,
             "Library = %s" % result.reference,
             "PGM = %s" % experiment.pgmName,
             "Flow Order = %s" % (experiment.flowsInOrder.strip() if experiment.flowsInOrder.strip() != '0' else 'TACG'),
             "Library Key = %s" % (experiment.libraryKey if experiment.libraryKey != "" else experiment.reverselibrarykey),
             "TF Key = %s" % "ATCG", # TODO, is it really unique?
             "Chip Check = %s" % get_chipcheck_status(experiment),
             "Chip Type = %s" % experiment.chipType,
             "Chip Data = %s" % experiment.rawdatastyle,
             "Notes = %s" % experiment.notes,
             "Barcode Set = %s" % experiment.barcodeId,
             "Analysis Name = %s" % result.resultsName,
             "Analysis Date = %s" % datetime.date.today(),
             "Analysis Flows = %s" % result.processedflows,
             "runID = %s" % result.runid,
             )
    return ('expMeta.dat', '\n'.join(lines))
    
def create_bc_conf(barcodeId,fname):
    """
    Creates a barcodeList file for use in barcodeSplit binary.
    
    Danger here is if the database returns a blank, or no lines, then the
    file will be written with no entries.  The use of this empty file later
    will generate no fastq files, except for the nomatch.fastq file.
    
    See C source code BarCode.h for list of valid keywords
    """
    # Retrieve the list of barcodes associated with the given barcodeId
    db_barcodes = models.dnaBarcode.objects.filter(name=barcodeId).order_by("index")
    lines = []
    for db_barcode in db_barcodes:
        lines.append('barcode %d,%s,%s,%s,%s,%s,%d,%s' % (db_barcode.index,db_barcode.id_str,db_barcode.sequence,db_barcode.adapter,db_barcode.annotation,db_barcode.type,db_barcode.length,db_barcode.floworder))
    if db_barcodes:
        lines.insert(0,"file_id %s" % db_barcodes[0].name)
        lines.insert(1,"score_mode %s" % str(db_barcodes[0].score_mode))
        lines.insert(2,"score_cutoff %s" % str(db_barcodes[0].score_cutoff))
    return (fname, "\n".join(lines))

def get_chipcheck_status(exp):
    """
    Load the explog stored in the log field in the experiment
    table into a python dict.  Check if `calibratepassed` is set
    """
    data = exp.log
    if data.get('calibratepassed', 'Not Found'):
        return 'Passed'
    else:
        return 'Failed'

def tfs_from_db():
    """
    Build the contents of a report TF file (``DefaultTFs.conf``),
    using the TF template data stored in the database.
    """
    tfs = models.Template.objects.all()
    lines = ["%s,%s,%s" % (tf.name, tf.key, tf.sequence,) for tf in tfs if tf.isofficial]
    return lines

def create_tf_conf(tfConfig):
    """
    Build the contents of the report TF file (``DefaultTFs.conf``)
    using the contents of an uploaded file.
    """
    fname = "DefaultTFs.conf"
    if tfConfig is not None:
        if tfConfig.size > 1024 * 1024 * 1024:
            raise ValueError("Uploaded TF config file too large (%d bytes)"
                             % tfConfig.size)
        buf = StringIO.StringIO()
        for chunk in tfConfig.chunks():
            buf.write(chunk)
        ret = (fname, buf.getvalue())
        buf.close()
        return ret
    else:
        lines = tfs_from_db()
        return (fname, "\n".join(lines))

def create_pk_conf(pk):
    """
    Build the contents of the report primary key file (``primary.key``).
    """
    text = "ResultsPK = %d" % pk
    return ("primary.key", text)

class BailException(Exception):
    """
    Raised when an error is encountered with report creation. These errors
    may include failure to contact the job server, or attempting to create
    an analysis in a directory which can't be read from or written to
    by the job server.
    """
    def __init__(self, msg):
        super(BailException, self).__init__()
        self.msg = msg

@login_required
@csrf_exempt
def displayReport(request, pk):
    ctx = {}
    ctx = template.RequestContext(request, ctx)
    return shortcuts.render_to_response("rundb/reports/report.html", context_instance=ctx)

def _createReport(request, pk, reportpk):
    """
    Send a report to the job server.
    
    If ``createReport`` receives a `GET` request, it displays a form
    to the user.

    If ``createReport`` receives a `POST` request, it will attempt
    to validate a ``RunParamsForm``. If the form fails to validate, it
    re-displays the form to the user, with error messages explaining why
    the form did not validate (using standard Django form error messages).

    If the ``RunParamsForm`` is valid, ``createReport`` will go through
    the following process. If at any step the process fails, ``createReport``
    raises and then catches ``BailException``, which causes an error message
    to be displayed to the user.

    * Attempt to contact the job server. If this does not raise a socket
      error or an ``xmlrpclib.Fault`` exception, then ``createReport`` will
      check with job server to make sure the job server can write to the
      report's intended working directory.
    * If the user uploaded a template file (for use as ``DefaultTFs.conf``),
      then ``createReport`` will check that the file is under 1MB in size.
      If the file is too big, ``createReport`` bails.
    * Finally, ``createReport`` contacts the job server and instructs it
      to run the report.

    When contacting the job server, ``createReport`` will attempt to
    figure out where the appropriate job server is listening. First,
    ``createReport`` checks to see if these is an entry in
    ``settings.JOB_SERVERS`` for the report's location. If it doesn't
    find an entry in ``settings.JOB_SERVERS``, it attempts to connect
    to `127.0.0.1` on the port given by ``settings.JOBSERVER_PORT``.
    """
    def bail(result, err):
        result.status = err
        raise BailException(err)

    exp = shortcuts.get_object_or_404(models.Experiment, pk=pk)
    try:
        rig = models.Rig.objects.get(name=exp.pgmName)
        loc = rig.location
    except ObjectDoesNotExist:
        #If there is a rig try to use the location set by it
        loc = models.Location.objects.filter(defaultlocation=True)
        if not loc:
            #if there is not a default, just take the first one
            loc = models.Location.objects.all().order_by('pk')
        if loc:
            loc = loc[0]
        else:
            logger.critical("There are no Location objects, at all")
            raise ObjectDoesNotExist("There are no Location objects, at all.")

    # Always use the default ReportStorage object
    storages = models.ReportStorage.objects.all()
    storage = storages.filter(default=True)[0]   #Select default ReportStorage obj.
    start_error = None
    
    # Get list of Reports associated with the ChipBarCode of this experiment
    # This is the link between Reports of Paired-End experiments

    rev_report_list = [("None","None")]
    fwd_report_list = [("None","None")]
    javascript = ""
    
    # alignment reference
    references = [("none","none")]
    for r in models.ReferenceGenome.objects.filter(index_version=settings.TMAP_VERSION,enabled=True):
      references.append((r.short_name, r.short_name+" ("+r.name+")"))        

    if exp.chipBarcode:
        #fix-TS-3900 pe_experiments = models.Experiment.objects.filter(chipBarcode=exp.chipBarcode)        
        pe_experiments = models.Experiment.objects.filter(chipBarcode__iexact=exp.chipBarcode)
        
        logger.debug("chipBarcode: %s" % exp.chipBarcode)
        for pe_exp in pe_experiments:
            results = pe_exp.sorted_results_with_reports()
            for result in results:
                # skip Paired-End Reports
                if result.metaData.get('paired','') == '':
                    if result.experiment.isReverseRun:
                        rev_report_list.append((result.get_report_dir(),result.resultsName))
                    else:
                        fwd_report_list.append((result.get_report_dir(),result.resultsName))

    isProton = True if exp.chipType == "900" else False

    #get the list of report addresses
    resultList = models.Results.objects.filter(experiment=exp).order_by("timeStamp")
    previousReports = []
    simple_version = re.compile(r"^(\d+\.?\d*)")
    for r in resultList:
        #try to get the version the major version the report was generated with
        try:
            versions = dict(v.split(':') for v in r.analysisVersion.split(",") if v)
            version = simple_version.match(versions['db']).group(1)
        except Exception:
            #just fail to 2.2
            version = "2.2"
        previousReports.append(( r.get_report_dir(), 
                        r.resultsName + " [" + str(r.get_report_dir()) + "]", 
                        version
        ))

    if request.method == 'POST':
        rpf = forms.RunParamsForm(request.POST, request.FILES)
        
        rpf.fields['previousReport'].widget.choices = previousReports
        rpf.fields['forward_list'].widget.choices = fwd_report_list
        rpf.fields['reverse_list'].widget.choices = rev_report_list
        rpf.fields['reference'].widget.choices = references

        #send some js to the page
        previousReportDir = get_initial_arg(reportpk)
        if previousReportDir:
            rpf.fields['blockArgs'].initial = "fromWells"
            javascript = """
            $("#fromWells").click();
            """
            javascript += '$("#id_previousReport").val("'+previousReportDir +'");'

        # validate the form
        if rpf.is_valid():
            chiptype_arg = exp.chipType
            ufResultsName = rpf.cleaned_data['report_name']
            resultsName = ufResultsName.strip().replace(' ', '_')

            result = build_result(exp, resultsName, storage, loc)

            webRootPath = result.web_root_path(loc)
            tfConfig = rpf.cleaned_data['tf_config']
            tfKey = rpf.cleaned_data['tfKey']
            blockArgs = rpf.cleaned_data['blockArgs']
            doThumbnail = rpf.cleaned_data['do_thumbnail']
            doBaseRecal = rpf.cleaned_data['do_base_recal']
            ts_job_type = ""
            if doThumbnail and exp.chipType == "900":
                ts_job_type = 'thumbnail'
                result.metaData["thumb"] = 1

            args = flattenString(rpf.cleaned_data['args'])
            basecallerArgs= flattenString(rpf.cleaned_data['basecallerArgs'])
            thumbnailBasecallerArgs= flattenString(rpf.cleaned_data['thumbnailBasecallerArgs'])
            thumbnailAnalysisArgs= flattenString(rpf.cleaned_data['thumbnailAnalysisArgs'])

            #do a full alignment?
            align_full = rpf.cleaned_data['align_full']
            #If libraryKey was set, then override the value taken from the explog.txt on the PGM
            libraryKey = rpf.cleaned_data['libraryKey']
            #ionCrawler may modify the path to raw data in the path variable passed thru URL
            exp.expDir = rpf.cleaned_data['path']
            aligner_opts_extra = rpf.cleaned_data['aligner_opts_extra']
            mark_duplicates = rpf.cleaned_data['mark_duplicates']
            result.runid = create_runid(resultsName + "_" + str(result.pk))
            previousReport = rpf.cleaned_data['previousReport']
            
            if rpf.cleaned_data['reference']:
              result.reference = rpf.cleaned_data['reference']
            else:
              result.reference = exp.library
            
            #attach project(s)
            projectNames = get_project_names(rpf, exp)                    
            username = request.user.username
            for name in projectNames.split(','):
              if name:                                  
                try:
                  p = models.Project.objects.get(name = name)            
                except models.Project.DoesNotExist:              
                  p = models.Project()
                  p.name = name    
                  p.creator = models.User.objects.get(username = username) 
                  p.save()
                  models.EventLog.objects.add_entry(p, "Created project name= %s during report creation." % p.name, request.user.username)  
                result.projects.add(p)
                models.EventLog.objects.add_entry(p, "Add result (%s) during report creation." % result.pk, request.user.username)  
            
            result.save()
            try:
                # Set the select fields here for the case when Bail exception is caught and form is reloaded.
                rpf.fields['forward_list'].widget.choices = fwd_report_list
                rpf.fields['reverse_list'].widget.choices = rev_report_list
                
                # Default control script definition
                scriptname='TLScript.py'
                
                # Check the reverse_list and forward_list elements
                select_forward = rpf.cleaned_data['forward_list'] or "None"
                select_reverse = rpf.cleaned_data['reverse_list'] or "None"
                if select_forward != "None" and select_reverse != "None":
                    # Paired end processing request
                    scriptname='PEScript.py'
                    ts_job_type='PairedEnd'
                    logger.debug("Forward %s" % request.POST.get('forward_list'))
                    logger.debug("Reverse %s" % request.POST.get('reverse_list'))
                    result.metaData["paired"] = 1
                    result.save()
                    
                scriptpath=os.path.join('/usr/lib/python2.6/dist-packages/ion/reports',scriptname)
                try:
                    with open(scriptpath,"r") as f:
                        script=f.read()
                except Exception as error:
                    bail(result,"Error reading %s\n%s" % (scriptpath,error.args))
                
                # check if path to raw data is there
                files = []
                try:
                    bk = models.Backup.objects.get(experiment=exp)
                except:
                    bk = False

                #------------------------------------------------
                # Tests to determine if raw data still available:
                #------------------------------------------------
                # Data directory is located on this server
                logger.debug("Start Analysis on %s" % exp.expDir)
                if ts_job_type == "thumbnail":
                    # thumbnail raw data is special in that despite there being a backup object for the dataset, thumbnail data is not deleted.
                    if rpf.cleaned_data['blockArgs'] != "fromWells" and rpf.cleaned_data['blockArgs'] != "fromSFF" and not path.exists(os.path.join(exp.expDir,'thumbnail')):
                        bail(result, "No path to raw data")
                else:
                    if bk and (rpf.cleaned_data['blockArgs'] != "fromWells" and rpf.cleaned_data['blockArgs'] != "fromSFF"):
                        if str(bk.backupPath) == 'DELETED':
                            bail(result, "The analysis cannot start because the raw data has been deleted.")
                            logger.warn("The analysis cannot start because the raw data has been deleted.")
                        else:
                            try:
                                datfiles = os.listdir(exp.expDir)
                                logger.debug("Got a list of files")
                            except:
                                logger.debug(traceback.format_exc())
                                bail(result,
                                     "The analysis cannot start because the raw data has been archived to %s.  Please mount that drive to make the data available." % (str(bk.backupPath),))
                    if rpf.cleaned_data['blockArgs'] != "fromWells" and rpf.cleaned_data['blockArgs'] != "fromSFF" and not path.exists(exp.expDir):
                        bail(result, "No path to raw data")
                
                try:
                    host = "127.0.0.1"
                    conn = client.connect(host, settings.JOBSERVER_PORT)
                    to_check = path.dirname(webRootPath)
                except (socket.error, xmlrpclib.Fault):
                    bail(result, "Failed to contact job server.")
                # prepare the directory in which the results' outputs will
                # be written
                # copy TF config to new path if it exists
                try:
                    files.append(create_tf_conf(tfConfig))
                except ValueError as ve:
                    bail(result, str(ve))
                # write meta data to folder for report
                files.append(create_meta(exp, result))
                files.append(create_pk_conf(result.pk))
                # write barcodes file to folder
                if exp.barcodeId and exp.barcodeId is not '':
                    files.append(create_bc_conf(exp.barcodeId,"barcodeList.txt"))
                # tell the analysis server to start the job
                params = makeParams(exp, args, blockArgs, doThumbnail, resultsName, result, align_full, libraryKey,
                                                        os.path.join(storage.webServerPath, loc.name), aligner_opts_extra,
                                                        mark_duplicates,
                                                        select_forward, select_reverse,
                                                        basecallerArgs, result.runid,
                                                        previousReport, tfKey,
                                                        thumbnailAnalysisArgs, thumbnailBasecallerArgs, doBaseRecal)
                chip_dict = {}
                try:
                    chips = models.Chip.objects.all()
                    chip_dict = dict((c.name, '-pe ion_pe %s' % str(c.slots)) for c in chips)
                except:
                    chip_dict = {} # just in case we can't read from the db
                try:
                    conn.startanalysis(resultsName, script, params, files,
                                       webRootPath, result.pk, chiptype_arg, chip_dict, ts_job_type)
                except (socket.error, xmlrpclib.Fault):
                    bail(result, "Failed to contact job server.")
                # redirect the user to the report started page
                return result
            except BailException as be:
                start_error = be.msg
                logger.exception("Aborted createReport for result %d: '%s'", result.pk, start_error)
                result.delete()
    # fall through if not valid...

    if request.method == 'GET':

        rpf = forms.RunParamsForm()
        rpf.fields['path'].initial = path.join(exp.expDir)
        rpf.fields['align_full'].initial = False
        #rpf.fields['mark_duplicates'].initial = False

        #if there is a library Key for the exp use that instead of the default
        if exp.isReverseRun:
            if exp.reverselibrarykey:
                rpf.fields['libraryKey'].initial = exp.reverselibrarykey
        else:
            if exp.libraryKey:
                rpf.fields['libraryKey'].initial = exp.libraryKey

        args = models.GlobalConfig.objects.all()[0].get_default_command()
        chipargs = get_chip_args(exp.chipType)
        alist = [args, chipargs]
        alist = [a for a in alist if a is not None]
        rpf.fields['args'].initial = " ".join(alist)

        rpf.fields['forward_list'].widget.choices = fwd_report_list
        rpf.fields['reverse_list'].widget.choices = rev_report_list
        rpf.fields['previousReport'].widget.choices = previousReports
        rpf.fields['reference'].widget.choices = references        
        rpf.fields['reference'].initial = exp.library        
        rpf.fields['project_names'].initial = get_project_names(rpf, exp)
        
        #send some js to the page
        previousReportDir = get_initial_arg(reportpk)
        if previousReportDir:
            rpf.fields['blockArgs'].initial = "fromWells"
            javascript = """
            $("#fromWells").click();
            """
            javascript += '$("#id_previousReport").val("'+previousReportDir +'");'


    ctx = {"rpf": rpf, "expName":exp.pretty_print_no_space, "start_error":start_error, "javascript" : javascript,
           "isProton":isProton, "pk":pk, "reportpk":reportpk, "isexpDir": os.path.exists(exp.expDir)}
    ctx = template.RequestContext(request, ctx)
    return ctx
@login_required
@csrf_exempt
def createReport(request, pk, reportpk):
    """
    Send a report to the job server.
    
    If ``createReport`` receives a `GET` request, it displays a form
    to the user.

    If ``createReport`` receives a `POST` request, it will attempt
    to validate a ``RunParamsForm``. If the form fails to validate, it
    re-displays the form to the user, with error messages explaining why
    the form did not validate (using standard Django form error messages).

    If the ``RunParamsForm`` is valid, ``createReport`` will go through
    the following process. If at any step the process fails, ``createReport``
    raises and then catches ``BailException``, which causes an error message
    to be displayed to the user.

    * Attempt to contact the job server. If this does not raise a socket
      error or an ``xmlrpclib.Fault`` exception, then ``createReport`` will
      check with job server to make sure the job server can write to the
      report's intended working directory.
    * If the user uploaded a template file (for use as ``DefaultTFs.conf``),
      then ``createReport`` will check that the file is under 1MB in size.
      If the file is too big, ``createReport`` bails.
    * Finally, ``createReport`` contacts the job server and instructs it
      to run the report.

    When contacting the job server, ``createReport`` will attempt to
    figure out where the appropriate job server is listening. First,
    ``createReport`` checks to see if these is an entry in
    ``settings.JOB_SERVERS`` for the report's location. If it doesn't
    find an entry in ``settings.JOB_SERVERS``, it attempts to connect
    to `127.0.0.1` on the port given by ``settings.JOBSERVER_PORT``.
    """
    templateName = "rundb/ion_run.html"
        
    result = _createReport(request, pk, reportpk)
    if isinstance(result, RequestContext):
        return shortcuts.render_to_response(templateName,
                                        context_instance=result)
    if (request.method == 'POST'):
        url = urlresolvers.reverse('report-started', args=(result.pk,))
        return HttpResponsePermanentRedirect(url)    
    
def get_chip_args(chipType):
    """Get the chip specific arguments to use when launching a run"""
    chips = models.Chip.objects.all()
    for chip in chips:
        if chip.name in chipType:
            return chip.args.strip()
    return None

def get_initial_arg(pk):
    """
    Builds the initial arg string for rerunning from wells
    """
    if int(pk) != 0:
        try:
            report = models.Results.objects.get(pk=pk)
            ret = report.get_report_dir()
            return ret
        except models.Results.DoesNotExist:
            return ""
    else:
        return ""
        
def get_project_names(rpf, exp):
    names = ''
    # get projects from form
    try:
      names = rpf.cleaned_data['project_names']
    except:
      pass  
    if len(names) > 1: return names
    # get projects from previous report
    if len(exp.sorted_results()) > 0: 
      names = exp.sorted_results()[0].projectNames()
    if len(names) > 1: return names           
    # get projects from Plan
    if exp.selectedPlanGUID:
      try:
        planObj = models.PlannedExperiment.objects.get(planGUID=exp.selectedPlanGUID)
        names = [p.name for p in planObj.projects.all()]  
        names = ','.join(names)  
      except:
        pass
    if len(names) > 1: return names    
    # last try: get from explog  
    try:
      names = exp.log['project']  
    except:
      pass
    return names    

def _report_started(request, pk):
    """
    Inform the user if a report sent to the job server was successfully
    started.
    """
    try:
        pk = int(pk)
    except (TypeError, ValueError):
        return http.HttpResponseNotFound()
    result = shortcuts.get_object_or_404(models.Results, pk=pk)
    report = result.reportLink
    log = path.join(path.dirname(result.reportLink), "log.html")
    ctxd = {"name":result.resultsName, "pk":result.pk,
            "link":report, "log":log,
            "status":result.status}
    ctx = template.RequestContext(request, ctxd)
    return ctx

@login_required
def report_started(request, pk):
    """
    Inform the user if a report sent to the job server was successfully
    started.
    """
    ctx = _report_started(request, pk)
    tmplname = "rundb/ion_successful_start_analysis.html"
    return shortcuts.render_to_response(tmplname, context_instance=ctx)

def get_plugins_dict(pg, plan={}, plan_selected_only = False):
    """
    Build a list containing dictionaries of plugin information.
    will only put the plugins in the list that are selected in the 
    interface
    """
    ret = []
    if len(pg) > 0:
        selected = []           
        selectedInput = {}
        if plan and 'selectedPlugins' in plan.keys():
          selectedPlugins = json.loads(plan['selectedPlugins'])
          selectedNames = []
          if 'planplugins' in selectedPlugins.keys():
              selected += selectedPlugins['planplugins']
          if 'planuploaders' in selectedPlugins.keys():
              selected += selectedPlugins['planuploaders']
          for planplugin in selected:
              selectedNames.append(planplugin['name'])            
              if 'userInput' in planplugin.keys():
                  selectedInput[planplugin['name']] = planplugin['userInput']
              else:
                  selectedInput[planplugin['name']] = {}
        
        for p in pg:
          # Exclude plugins based on Plan selection if any
          if plan_selected_only and (len(selected) > 0) and (p.name not in selectedNames):
              continue            
        
          params = {'name':p.name,
                'path':p.path,
                'version':p.version,
                'id':p.id,
                'project':p.project,
                'sample':p.sample,
                'libraryName':p.libraryName,
                'chipType':p.chipType,
                'autorun':p.autorun,
                'pluginconfig': json.dumps(p.config),
                'userInput':  selectedInput.get(p.name, '')
                } 
                
          for key in p.pluginsettings.keys():
              params[key] = p.pluginsettings[key]
          
          ret.append(params)   
    return ret    


def makeParams(exp, args, blockArgs, doThumbnail, resultsName, result, align_full, libraryKey,
                                url_path, aligner_opts_extra, mark_duplicates,
                                select_forward, select_reverse, basecallerArgs, runid,
                                previousReport,tfKey,
                                thumbnailAnalysisArgs, thumbnailBasecallerArgs, doBaseRecal):
    """Build a dictionary of analysis parameters, to be passed to the job
    server when instructing it to run a report.  Any information that a job
    will need to be run must be constructed here and included inside the return.  
    This includes any special instructions for flow control in the top level script."""
    gc = models.GlobalConfig.objects.all().order_by('id')[0]    
    pathToData = path.join(exp.expDir)
    if doThumbnail and exp.chipType == "900":
        pathToData = path.join(pathToData,'thumbnail')
    defaultLibKey = gc.default_library_key

    ##logger.debug("...views.makeParams() gc.default_library_key=%s;" % defaultLibKey)
    expName = exp.expName

    #get the exp data for sam metadata
    exp_filter = models.Experiment.objects.filter(pk=exp.pk)
    exp_json = serializers.serialize("json", exp_filter)
    exp_json = json.loads(exp_json)
    exp_json = exp_json[0]["fields"]

    #now get the plan and return that
    try:
        if exp.plan:
            planObj = [exp.plan]
        elif exp.selectedPlanGUID:
            # Why isn't the FK set?
            planObj = models.PlannedExperiment.objects.filter(planGUID=exp.selectedPlanGUID)
        else:
            # Fallback to explog data... crawler should be setting this up
            planId = exp.log.get("pending_run_short_id",exp.log.get("planned_run_short_id", {}))
            if planId:
                planObj = models.PlannedExperiment.objects.filter(planShortID=planId)
                # Broken - ShortID is not unique once plan is used, may get wrong plan here.
            else:
                planObj = []
    except: ## (KeyError, ValueError, TypeError):
        logger.exception("Failed to extract plan data from exp '%s'", exp.expName)
        planObj = []

    if planObj:
        plan_json = serializers.serialize("json", planObj)
        plan_json = json.loads(plan_json)
        plan = plan_json[0]["fields"]
    else:
        plan = {}

    try:
        pg = models.Plugin.objects.filter(selected=True,active=True).exclude(path='')
        plugins = get_plugins_dict(pg, plan, True)
    except:
        logger.exception("Failed to get list of active plugins")
        plugins = ""

    site_name = gc.site_name
    barcode_args = gc.barcode_args

    libraryName = result.reference

    skipchecksum = False
    fastqpath = result.fastqLink.strip().split('/')[-1]

    #TODO: remove the libKey from the analysis args, assign this in the TLScript. To make this more fluid

    #if the librayKey was set by createReport use that value. If not use the value from the PGM
    if not libraryKey:
        if exp.isReverseRun:
            libraryKey = exp.reverselibrarykey
        else:
            libraryKey = exp.libraryKey

    if libraryKey == None or len(libraryKey) < 1:
        libraryKey = defaultLibKey

    # floworder field sometimes has whitespace appended (?)  So strip it off
    flowOrder = exp.flowsInOrder.strip()
    # Set the default flow order if its not stored in the dbase.  Legacy support
    if flowOrder == '0' or flowOrder == None or flowOrder == '':
        flowOrder = "TACG"

    # Set the barcodeId
    if exp.barcodeId:
        barcodeId = exp.barcodeId
    else:
        barcodeId = ''
    project = ','.join(p.name for p in result.projects.all())
    sample = exp.sample
    chipType = exp.chipType
    #net_location = gc.web_root
    #get the hostname try to get the name from global config first
    if gc.web_root:
        net_location = gc.web_root
    else:
        #if a hostname was not found in globalconfig.webroot then use what the system reports
        net_location = "http://" + str(socket.getfqdn())
    
    # Get the 3' adapter sequence
    adapterSequence = exp.forward3primeadapter
    if exp.isReverseRun:
        adapterSequence = exp.reverse3primeadapter
        
    try:
        adapter_primer_dicts = models.ThreePrimeadapter.objects.filter(sequence=adapterSequence)
    except:
        adapter_primer_dicts = None
        
    #the adapter_primer_dicts should not be empty or none
    if not adapter_primer_dicts or adapter_primer_dicts.count() == 0:
        if exp.isReverseRun:
            try:
                adapter_primer_dict = models.ThreePrimeadapter.objects.get(direction="Reverse", isDefault=True)
            except (models.ThreePrimeadapter.DoesNotExist,
                    models.ThreePrimeadapter.MultipleObjectsReturned):
                    
                #ok, there should be a default in db, but just in case... I'm keeping the previous logic for fail-safe
                adapter_primer_dict = {'name':'Reverse Ion Kit',
                                       'sequence':'CTGAGTCGGAGACACGCAGGGATGAGATGG',
                                       'direction': 'Reverse',
                                       'qual_cutoff':9,
                                       'qual_window':30,
                                       'adapter_cutoff':16
                                        }                
        else:     
            try:         
                adapter_primer_dict = models.ThreePrimeadapter.objects.get(direction="Forward", isDefault=True)
            except (models.ThreePrimeadapter.DoesNotExist,
                    models.ThreePrimeadapter.MultipleObjectsReturned):
                
                #ok, there should be a default in db, but just in case... I'm keeping the previous logic for fail-safe
                adapter_primer_dict = {'name':'Ion Kit',
                                       'sequence':'ATCACCGACTGCCCATAGAGAGGCTGAGAC',
                                       'direction': 'Forward',
                                       'qual_cutoff':9,
                                       'qual_window':30,
                                       'adapter_cutoff':16
                                        }
    else:
        adapter_primer_dict = adapter_primer_dicts[0]


    rawdatastyle = exp.rawdatastyle

    #if args are passed use them, if not default to the global config default
    #TODO, do chip stuff here too
    if args:
        analysisArgs = args
    else:
        analysisArgs = flattenString(gc.default_command_line)

    #if basecallerArgs are not provied, default to the global config defauly
    if not basecallerArgs:
        basecallerargs = flattenString(gc.basecallerargs)

    #get the thumbnail analysis args
    if not thumbnailAnalysisArgs:
        thumbnailAnalysisArgs = flattenString(gc.analysisthumbnailargs)

    #get the thumbnail basecaller args
    if not thumbnailBasecallerArgs:
        thumbnailBasecallerArgs = flattenString(gc.basecallerthumbnailargs)

    ret = {'pathToData':pathToData,
           'analysisArgs':analysisArgs,
           'basecallerArgs' : basecallerArgs,
           'blockArgs':blockArgs,
           'libraryName':libraryName,
           'resultsName':resultsName,
           'expName':expName,
           'libraryKey':libraryKey,
           'plugins':plugins,
           'fastqpath':fastqpath,
           'skipchecksum':skipchecksum,
           'flowOrder':flowOrder,
           'align_full' : align_full,
           'project':project,
           'sample':sample,
           'chiptype':chipType,
           'barcodeId':barcodeId,
           'net_location':net_location,
           'exp_json': json.dumps(exp_json,cls=DjangoJSONEncoder),
           'site_name': site_name,
           'url_path':url_path,
           'reverse_primer_dict':adapter_primer_dict,
           'rawdatastyle':rawdatastyle,
           'aligner_opts_extra':aligner_opts_extra,
           'mark_duplicates' : mark_duplicates,
           'plan': plan,
           'flows':exp.flows,
           'pgmName':exp.pgmName,
           'pe_forward':select_forward,
           'pe_reverse':select_reverse,
           'isReverseRun':exp.isReverseRun,
           'barcode_args':json.dumps(barcode_args,cls=DjangoJSONEncoder),
           'tmap_version':settings.TMAP_VERSION,
           'runid':runid,
           'previousReport':previousReport,
           'tfKey': tfKey,
           'thumbnailAnalysisArgs': thumbnailAnalysisArgs,
           'thumbnailBasecallerArgs' : thumbnailBasecallerArgs,
           'doThumbnail' : doThumbnail,
           'sam_parsed' : True if os.path.isfile('/opt/ion/.ion-internal-server') else False,
           'doBaseRecal':doBaseRecal,
    }
    
    return ret

@login_required
def current_jobs(request):
    """
    Display status information about any job servers listed in
    ``settings.JOB_SERVERS`` (or the local job server if appropriate),
    as well as information about any jobs (reports) in progress.
    """
    jservers = [(socket.gethostname(), socket.gethostbyname(socket.gethostname()))]
    servers = []
    jobs = []
    for server_name, ip in jservers:
        short_name = "%s (%s)" % (server_name, ip)
        try:
            conn = client.connect(ip, settings.JOBSERVER_PORT)
            running = conn.running()
            uptime = seconds2htime(conn.uptime())
            nrunning = len(running)
            servers.append((server_name, ip, True, nrunning, uptime,))
            server_up = True
        except (socket.error, xmlrpclib.Fault):
            servers.append((server_name, ip, False, 0, 0,))
            server_up = False
        if server_up:
            for name, pid, pk, atype, stat in running:
                try:
                    result = models.Results.objects.get(pk=pk)
                    experiment = result.experiment
                    jobs.append((short_name, name, pid, result, atype, stat,
                                 result, experiment))
                except:
                    pass
            jobs.sort(key=lambda x: int(x[2]))
    ctxd = {"jobs":jobs, "servers":servers}
    ctx = template.RequestContext(request, ctxd)
    return ctx

@login_required
def blank(request, **kwargs):
    """
    just render a blank template
    """
    return shortcuts.render_to_response("rundb/reports/30_default_report.html", 
        {'tab':kwargs['tab']})

@login_required
def edit_template(request, pk):
    """
    Make changes to an existing test fragment template database record,
    or create a new one if ``pk`` is zero.
    """
    if int(pk) != 0:
        tf = shortcuts.get_object_or_404(models.Template, pk=pk)
    else:
        tf = models.Template()

    if request.method == "POST":
        rfd = forms.AddTemplate(request.POST)
        if rfd.is_valid():
            tf.name = rfd.cleaned_data['name'].replace(' ', '_')
            tf.key = rfd.cleaned_data['key']
            tf.sequence = rfd.cleaned_data['sequence']
            tf.isofficial = rfd.cleaned_data['isofficial']
            tf.comments = rfd.cleaned_data['comments']
            tf.save()
            url = urlresolvers.reverse("ion-references")
            return http.HttpResponsePermanentRedirect(url)
        else:
            ctxd = {"temp":rfd, "name":tf.name}
            ctx = template.RequestContext(request, ctxd)
            return shortcuts.render_to_response("rundb/ion_edit_template.html",
                                                context_instance=ctx)
    elif request.method == "GET":
        if int(pk) == 0:
            temp = forms.AddTemplate()
            temp.fields['pk'].initial = 0
            ctxd = {"temp":temp, "name":"New Template"}
            ctx = template.RequestContext(request, ctxd)
            return shortcuts.render_to_response("rundb/ion_edit_template.html",
                                                context_instance=ctx)
        else:
            temp = forms.AddTemplate()
            temp.fields['name'].initial = tf.name.replace(' ', '_')
            temp.fields['key'].initial = tf.key
            temp.fields['sequence'].initial = tf.sequence
            temp.fields['isofficial'].initial = tf.isofficial
            temp.fields['comments'].initial = tf.comments
            temp.fields['pk'].initial = tf.pk

            ctxd = {"temp":temp, "name":tf.name}
            ctx = template.RequestContext(request, ctxd)
            return shortcuts.render_to_response("rundb/ion_edit_template.html",
                                                context_instance=ctx)

@login_required
def db_backup(request):
    
    def get_servers():
        fileservers = models.FileServer.objects.all()
        ret = []
        for fs in fileservers:
            if path.exists(fs.filesPrefix):
                ret.append(fs)
        return ret
    
    def areyourunning():
        uptime = 0.0
        try:
            astat = xmlrpclib.ServerProxy("http://127.0.0.1:%d" % settings.IARCHIVE_PORT)
            logger.debug ("Sending xmplrpc status_check")
            uptime = astat.status_check()
            daemon_status = True
        except (socket.error, xmlrpclib.Fault):
            logger.warn (traceback.format_exc())
            daemon_status = False
        except:
            logger.exception(traceback.format_exc())
            daemon_status = False
        return daemon_status
    
    def removeOnly():
        '''Makes xmlrpc call to the ionArchive daemon to get the remove_only flag'''
        try:
            astat = xmlrpclib.ServerProxy("http://127.0.0.1:%d" % settings.IARCHIVE_PORT)
            logger.debug ("Sending xmplrpc remove_only_mode")
            remove_only_status = astat.remove_only_mode()
        except:
            logger.exception(traceback.format_exc())
            remove_only_status = True
        return remove_only_status
    
    # Determine if ionArchive service is running
    daemon_status = areyourunning()
    
    #Note: there is always only 1 BackupConfig object configured.
    bk = models.BackupConfig.objects.all().order_by('pk')
    # populate dictionary with experiments ready to be archived
    # Get all experiments in database sorted by date
    experiments = models.Experiment.objects.all().order_by('date')
    # Ignore experiments already archived/deleted
    experiments = experiments.exclude(expName__in = models.Backup.objects.all().values('backupName'))
    # Ignore all experiments marked Keep
    #experiments = experiments.exclude(storage_options='KI')
    # Filter out experiments already (deleted or archived)
    # This will include Runs that are not yet slated to be processed
    #experiments = experiments.exclude(user_ack='D')
    # This will exclude Runs that are not yet slated to be processed
    #experiments = experiments.exclude(user_ack='U').exclude(user_ack='D')
    
#    if removeOnly():
#        # Filter out experiments marked Archive
#        experiments = experiments.exclude(storage_options='A')
        
    # make dictionary, one array per file server of archiveExperiment objects
    to_archive = {}
    fs_stats = {}
    # only generate list of experiments if Archiving is enabled
    if bk[0].online:
        servers = get_servers()
        for fs in servers:
            # Statistics for this server
            fs_stats[fs.filesPrefix] = fs.percentfull
            
            # Experiments for this server
            explist = []
            for exp in experiments:
                if exp.expDir.startswith(fs.filesPrefix):
                    E = Experiment(exp,
                                   str(exp.expName),
                                   str(exp.date),
                                   str(exp.star),
                                   str(exp.storage_options),
                                   str(exp.user_ack),
                                   str(exp.expDir),
                                   exp.pk,
                                   str(exp.rawdatastyle))
                    explist.append(E)
                # limit number in list to configured limit
                #if len(explist) >= bk[0].number_to_backup:
                #    break
            to_archive[fs.filesPrefix] = explist
        
    ctx = template.RequestContext(request, {"backups":bk, "to_archive":to_archive, "status":daemon_status,"fs_stats":fs_stats})
    return ctx

#def backup(request):
#    """
#    Displays the current status of the Archive server.  Makes use of
#    xmlrpc to communicate with the server.
#    """
#    bk = models.BackupConfig.objects.all().order_by('pk')
#    url = "http://127.0.0.1:%d" % settings.IARCHIVE_PORT
#    to_archive = {}
#    try:
#        astat = xmlrpclib.ServerProxy(url)
#        to_archive = astat.next_to_archive()
#        status = True
#    except (socket.error, xmlrpclib.Fault):
#        status = False
#    ctx = template.RequestContext(request, {"backups":bk, "to_archive":to_archive, "status":status})
#    return ctx

def edit_backup(request, pk):
    """
    Handles any changes to the backup configuration
    """
    if int(pk) != 0:
        bk = shortcuts.get_object_or_404(models.BackupConfig, pk=pk)
        exists = True
    else:
        bk = models.BackupConfig()
        exists = False

    if request.method == "POST":
        ebk = forms.EditBackup(request.POST)
        if ebk.is_valid():
            if ebk.cleaned_data['archive_directory'] != None:
                bk.name = ebk.cleaned_data['archive_directory'].strip().split('/')[-1]
                bk.backup_directory = ebk.cleaned_data['archive_directory']
            else:
                bk.name = 'None'
                bk.backup_directory = 'None'
            bk.location = models.Location.objects.all()[0]
            bk.number_to_backup = ebk.cleaned_data['number_to_archive']
            bk.timeout = ebk.cleaned_data['timeout']
            bk.backup_threshold = ebk.cleaned_data['percent_full_before_archive']
            bk.grace_period = int(ebk.cleaned_data['grace_period'])
            bk.bandwidth_limit = int(ebk.cleaned_data['bandwidth_limit'])
            bk.email = ebk.cleaned_data['email']
            bk.online = ebk.cleaned_data['enabled']
            bk.save()
            url = urlresolvers.reverse("ion-daemon")
            return http.HttpResponsePermanentRedirect(url)
        else:
            ctxd = {"temp":ebk}
            ctx = template.RequestContext(request, ctxd)
            return shortcuts.render_to_response("rundb/ion_edit_backup.html",
                                                context_instance=ctx)
    elif request.method == "GET":
        temp = forms.EditBackup()
        if int(pk) == 0:
            #temp.fields['archive_directory'].choices = get_dir_choices()
            temp.fields['number_to_archive'].initial = 10
            temp.fields['timeout'].initial = 60
            temp.fields['percent_full_before_archive'].initial = 90
            temp.fields['grace_period'].initial = 72
            ctxd = {"temp":temp, "name":"New Archive Configuration"}
            ctx = template.RequestContext(request, ctxd)
            return shortcuts.render_to_response("rundb/ion_edit_backup.html",
                                                context_instance=ctx)
        else:
            #temp.fields['backup_directory'].choices = get_dir_choices()
            temp.fields['archive_directory'].initial = bk.backup_directory
            temp.fields['number_to_archive'].initial = bk.number_to_backup
            temp.fields['timeout'].initial = bk.timeout
            temp.fields['percent_full_before_archive'].initial = bk.backup_threshold
            temp.fields['grace_period'].initial = bk.grace_period
            temp.fields['bandwidth_limit'].initial = bk.bandwidth_limit
            temp.fields['email'].initial = bk.email
            temp.fields['enabled'].initial = bk.online
            ctxd = {"temp":temp}
            ctx = template.RequestContext(request, ctxd)
            return shortcuts.render_to_response("rundb/ion_edit_backup.html",
                                                context_instance=ctx)
            


def getChipZip(request, path):
    from django.core.servers.basehttp import FileWrapper
    import tarfile
    import zipfile
    
    try:
        name = os.path.basename(path)
        name = name.split(".")[0]
        
        # initialize zip archive file
        zipfilename = os.path.join("/tmp","%s.zip" % name)
        zipobj = zipfile.ZipFile(zipfilename, mode='w', allowZip64=True)
        
        # open tar.bz2 file, extract all members and write to zip archive
        tf = tarfile.open(os.path.join(path))
        for tarobj in tf.getmembers():
            contents = tf.extractfile(tarobj)
            zipobj.writestr(tarobj.name, contents.read())
        zipobj.close()
        
        response = http.HttpResponse(FileWrapper(open(zipfilename)), mimetype='application/zip')
        response['Content-Disposition'] = 'attachment; filename=%s' % os.path.basename(zipfilename)
        os.unlink(zipfilename)
        return response
    except Exception as inst:
        logger.exception(traceback.format_exc())
        ctxd = {
        "error_state":1,
        "error":[['Error','%s'%inst],['Error type','%s'%type(inst)]],
        "locations_list":[],
        "base_site_name":'Error',
        "files":[],
        }
        ctx = template.RequestContext(request, ctxd)
        return shortcuts.render_to_response("rundb/ion_chips.html", context_instance=ctx)

def getChipLog(request, path):
    import tarfile
    import shutil
    try:
        tf = tarfile.open('%s'%path)
        ti = tf.extractfile('InitLog.txt')
        logLines = []
        for line in ti.readlines():
            logLines.append(line)
        tf.close()
    except:
        logger.exception(traceback.format_exc())
        
    ctxd = {"lineList":logLines}
    ctx = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response('rundb/ion_chipLog.html', context_instance=ctx)

def getChipPdf(request, path):
    import tarfile
    import re
    from django.core.servers.basehttp import FileWrapper
    
    def runFromShell(cmd1):
        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p1.communicate()
        return p1
    
    # File I/O setup
    tmpstem = os.path.basename(path).split('.')[0]
    tmphtml = os.path.join('/tmp',tmpstem+'.html')
    tmppdf = os.path.join('/tmp',tmpstem+'.pdf')
    tf = tarfile.open(path)
    ti = tf.extractfile('InitLog.txt')
    
    # regular expressions for the string parsing
    ph = re.compile('\d*\)\sW2\spH=\d*.\d*')
    phpass = re.compile('(W2\sCalibrate\sPassed\sPH=)(\d*.\d*)')
    adding = re.compile('(\d*\)\sAdding\s)(\d*.\d*)(\sml)')
    datefilter = re.compile('Sun|Mon|Tue|Wed|Thu|Fri|Sat') #Simple filter for the line with the date on it.
    namefilter = re.compile('(Name:\s*)([a-z][a-z0-9_]*)',re.IGNORECASE)
    serialfilter = re.compile('(Serial)(.*?)(:)(\s*)([a-z][a-z0-9_]*)',re.IGNORECASE)
    surfacefilter = re.compile('(surface)(=)((?:[a-z][a-z]+))(\\s+)',re.IGNORECASE)
    rawtracefilter = re.compile('(RawTraces)(\s+)((?:[a-z][a-z]*[0-9]+[a-z0-9]*))(:)(\s+)([+-]?\d*\.\d+)(?![-+0-9\.])',re.IGNORECASE)
    
    # initialize variables
    initialph = ''
    finalph = ''
    totalAdded = 0.0
    iterationBuffer = ''
    rawtraceBuffer = ''
    totalIterations = 0
    startdate = ''
    enddate = ''
    pgmname = ''
    serialnumber = ''
    calstatus = ''
    rawtracestartdate = ''
    
    # Log file parsing
    for line in ti.readlines():
        test = namefilter.match(line)
        if test:
            pgmname = test.group(2)
            
        test = serialfilter.match(line)
        if test:
            serialnumber = test.group(5)

        test = ph.match(line)
        if test:
            if initialph == '': initialph = test.group().split('=')[1]
            iterationBuffer += "<tr><td>%s</td></tr>\n" % test.group()
            totalIterations += 1
        
        test = adding.match(line)
        if test:
            iterationBuffer += "<tr><td>%s</td></tr>\n" % test.group()
            totalAdded += float(test.group(2))
            
        test = phpass.match(line)
        if test:
            finalph = test.group(2)
            calstatus = 'PASSED'
            
        if datefilter.match(line):
            if startdate == '': startdate = line.strip()
            
        test = surfacefilter.match(line)
        if test:
            surface = test.group(3) 
        
        test = rawtracefilter.match(line)
        if test:
            rawtraceBuffer += "<tr><td>%s %s: %s</td></tr>\n" % (test.group(1),test.group(3),test.group(6))
            
    # Find the end date of the Chip Calibration - we need multilines to identify the end date entry
    #TODO: fix assumption that line endings are always newline char.
    ti.seek(0)
    contents = ti.read()
    enddatefilter = re.compile('^(W2 Calibrate Passed.*$\n)(Added.*$\n)([Sun|Mon|Tue|Wed|Thu|Fri|Sat].*$)',re.MULTILINE)
    m = enddatefilter.search(contents,re.MULTILINE)
    if m:
        enddate = m.group(3)
    
    startrawfilter = re.compile('([Sun|Mon|Tue|Wed|Thu|Fri|Sat].*$\n)(RawTraces.*$)',re.MULTILINE)
    m = startrawfilter.search(contents,re.MULTILINE)
    if m:
        rawtracestartdate = m.group(1)
        rawtraceBuffer = ('<tr><td>Raw Traces</td><td></td><td>%s</td></tr>' % rawtracestartdate) + rawtraceBuffer
        
    tf.close()
    
    f = open(tmphtml, 'w')
    f.write('<html>\n')
    f.write("<img src='/var/www/site_media/images/logo_top_right_banner.png' alt='lifetechnologies, inc.'/>")
    # If there are sufficient errors in parsing, display an error banner
    if calstatus == '' and finalph == '':
        f.write('<table width="100%">')
        f.write('<tr><td></td></tr>')
        f.write('<tr><td align=center><hr /><font color=red><i><h2>* * * Error parsing InitLog.txt * * *</h2></i></font><hr /></td></tr>')
        f.write('<tr><td></td></tr>')
        f.write("</table>")
    else:
        f.write('<table width="100%">')
        f.write('<tr><td></td></tr>')
        f.write('<tr><td align=center><hr /><i><h2>Instrument Installation Report</h2></i><hr /></td></tr>')
        f.write('<tr><td></td></tr>')
        f.write("</table>")
        
    f.write('<table width="100%">')
    f.write('<tr><td>Instrument Name</td><td>%s</td></tr>\n' % (pgmname))
    f.write('<tr><td>Serial Number</td><td>%s</td></tr>\n' % (serialnumber))
    f.write('<tr><td>Chip Surface</td><td>%s</td></tr>\n' % (surface))
    f.write("<tr><td>Initial pH:</td><td>%s</td><td>%s</td></tr>\n" % (initialph,startdate))
    f.write("<tr><td></td></tr>\n") # puts a small line space
    f.write(iterationBuffer)
    f.write("<tr><td></td></tr>\n") # puts a small line space
    f.write('<tr><td>Total Hydroxide Added:</td><td>%0.2f ml</td></tr>\n' % totalAdded)
    f.write('<tr><td>Total Iterations:</td><td>%d</td></tr>\n' % totalIterations)
    f.write("<tr><td>Final pH:</td><td>%s</td><td>%s</td></tr>\n" % (finalph,enddate))
    f.write("<tr><td></td></tr>\n") # puts a small line space
    f.write(rawtraceBuffer)
    f.write("<tr><td></td></tr>\n") # puts a small line space
    f.write('<tr><td>Instrument Installation Status:</td><td><i><font color="#00aa00">%s</font></i></td></tr>\n' % calstatus)
    f.write("</table>")
        
    f.write('<br /><br />\n')
    f.write('<table frame="box">\n')
    f.write('<tr><th align="left">Acknowledged by:</th></tr>')
    f.write('<tr><td align="left"><br />\n')
    f.write('__________________________________________________________<br />Customer Signature</td>')
    f.write('<td align="left"><br />___________________<br />Date\n')
    f.write('</td></tr>')
    f.write('<tr><td align="left"><br />\n')
    f.write('__________________________________________________________<br />Life Tech FSE Signature</td>')
    f.write('<td align="left"><br />___________________<br />Date\n')
    f.write('</td></tr></table></html>\n')
    f.close()
    pdf_cmd = ['/opt/ion/iondb/bin/wkhtmltopdf-amd64',str(tmphtml),str(tmppdf)]
    p1 = runFromShell(pdf_cmd)
    os.unlink(tmphtml)
    response = http.HttpResponse(FileWrapper(open(tmppdf)), mimetype='application/pdf')
    response['Content-Disposition'] = 'attachment; filename=%s' % (os.path.basename(tmppdf))
    # Can we delete the pdf file now?  Yes,on my dev box...
    os.unlink(tmppdf)
    
    return response

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

def get_dir_choices():
    """
    Return the list of directories in the /media directory
    """
    basicChoice = [(None, 'None')]
    for choice in devices.to_media(devices.disk_report()):
        basicChoice.append(choice)
    return tuple(basicChoice)

def get_loc_choices():
    """
    return a list of Locations to select
    """
    basicChoice = []
    for loc in models.Location.objects.all():
        basicChoice.append((loc, loc))
    return tuple(basicChoice)

@login_required
def about(request):
    """
    Generates the information on the `About` tab
    """
    versions, meta_version = findVersions()
    ctxd = {"about":versions, "meta": meta_version, "user": request.user}
    ctx = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_about.html",
                                        context_instance=ctx, mimetype="text/html")

# ============================================================================
# Global configuration processing and helpers
# ============================================================================


def config_contacts(request, context):
    """Essentially but not actually a context processor to handle user contact
    information on the global config page.
    """
    updated = False
    contacts = {"lab_contact": None, "it_contact": None}
    for profile in models.UserProfile.objects.filter(user__username__in=contacts.keys()):
        if request.method == "POST" and "lab_contact-name" in request.POST:
            form = forms.UserProfileForm(data=request.POST, instance=profile,
                                         prefix=profile.user.username)
            if form.is_valid():
                form.save()
                updated = True
        else:
            form = forms.UserProfileForm(instance=profile,
                                         prefix=profile.user.username)
        contacts[profile.user.username] = form
    if updated:
        tasks.contact_info_flyaway.delay()
    context.update({"contacts": contacts})


def config_site_name(request, context, config):
    """The site name will be automatically loaded on the page, so all we have
    to do here is check whether we should update it, and if so, do so.
    """
    if request.method == "POST" and "site_name" in request.POST:
        config.site_name = request.POST["site_name"]
        config.save()
        context.update({"base_site_name": request.POST["site_name"]})


@bind_messages("config")
@login_required
def global_config(request):
    """
    Renders the Config tab.
    """
    globalconfig = models.GlobalConfig.objects.all().order_by('pk')[0]
    ctx = template.RequestContext(request, {})
    # As the config page takes on more responsibility, things that need updating
    # can be updated from here:
    config_contacts(request, ctx)
    config_site_name(request, ctx, globalconfig)

    emails = models.EmailAddress.objects.all().order_by('pk')

    # Rescan Publishers
    publishers.purge_publishers()
    publishers.search_for_publishers(globalconfig)

    # Rescan Plugins
    pluginmanager = PluginManager(globalconfig)
    pluginmanager.rescan() ## Find new, remove missing

    # Refresh pubs and plugs, as some may have been added or removed
    plugs = models.Plugin.objects.filter(active=True).order_by('-date').exclude(path='')
    pubs = models.Publisher.objects.all().order_by('name')

    ctx.update({"config":globalconfig,
                "email":emails,
                "plugin":plugs,
                "use_precontent":True,
                "use_content2":True,
                "use_content3":True,
                "publishers": pubs})

    return shortcuts.render_to_response("rundb/ion_global_config.html",
                                        context_instance=ctx)


@login_required
def edit_email(request, pk):
    """
    Simple view for adding email addresses for nightly email
    """
    if int(pk) != 0:
        em = shortcuts.get_object_or_404(models.EmailAddress, pk=pk)
    else:
        em = models.EmailAddress()

    if request.method == "POST":
        rfd = forms.EditEmail(request.POST)
        if rfd.is_valid():
            em.email = rfd.cleaned_data['email_address']
            em.selected = rfd.cleaned_data['selected']
            em.save()
            url = urlresolvers.reverse("ion-config")
            return http.HttpResponsePermanentRedirect(url)
        else:
            ctxd = {"temp":rfd, "name":em.email}
            ctx = template.RequestContext(request, ctxd)
            return shortcuts.render_to_response("rundb/ion_edit_email.html",
                                                context_instance=ctx)
    elif request.method == "GET":
        if int(pk) == 0:
            temp = forms.EditEmail()
            ctxd = {"temp":temp, "name":"New Email"}
            ctx = template.RequestContext(request, ctxd)
            return shortcuts.render_to_response("rundb/ion_edit_email.html",
                                                context_instance=ctx)
        else:
            temp = forms.EditEmail()
            temp.fields['email_address'].initial = em.email
            temp.fields['selected'].initial = em.selected
            ctxd = {"temp":temp, "name":em.email}
            ctx = template.RequestContext(request, ctxd)
            return shortcuts.render_to_response("rundb/ion_edit_email.html",
                                                context_instance=ctx)


def get_best_results(sortfield, timeframe, ret_num):
    """
    Returns the `ret_num` best analysis for all runs in `timeframe` based on `sortfield`
    """
    timerange = datetime.datetime.now() - datetime.timedelta(days=timeframe)
    exp = models.Experiment.objects.filter(date__gt=timerange)
    # get best result for each experiment
    res = [e.best_result(sortfield) for e in exp if e.best_result(sortfield) is not None]
    # sort the best results of the best experiments and return the best 20
    if ret_num is None:
        res = sorted(res, key=lambda r: getattr(r.best_lib_by_value(sortfield), sortfield), reverse=True)
    else:
        res = sorted(res, key=lambda r: getattr(r.best_lib_by_value(sortfield), sortfield), reverse=True)[:ret_num]
    return res

@login_required
def best_runs(request):
    """
    Generates the `best_runs` page.  Gets the best results for all experiments
    and displays the best 20.  Also, sets any that make it on that list to `Archive`
    so that the raw data will not be deleted
    """
    if 'submit' in request.GET:
        d = request.GET
    else:
        d = None
    sort = forms.BestRunsSort(d)
    sortfield = 'i100Q17_reads'
    if sort.is_valid():
        sortfield = sort.cleaned_data['library_metrics']
    res = get_best_results(sortfield, 365, 20)
    # mark anything on the best runs page to _at least_ 'Archive'
    for r in res:
        e = models.Experiment.objects.get(pk=r.experiment.pk)
        try:
            bk = models.Backup.objects.get(name=e.expName)
        except:
            bk = False
        if not bk:
            if e.storage_options == 'D':
                e.storage_options = 'A'
                e.save()
    # get the library metrics for the top 20 reports
    lbms = [r.best_lib_metrics for r in res]
    tfms = [r.best_metrics for r in res]
    ctx = template.Context({"reports":
                                [(r, t, l) for r, t, l in zip(res, tfms, lbms)]})
    ctxd = {"reports":ctx, "sort":sort}
    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_best.html",
                                        context_instance=context)

@login_required
def stats_sys(request):
    """
    Generates the stats page on system configuration
    """

    # Run a script on the server to generate text
    networkCMD = [os.path.join("/usr/bin", "ion_netinfo")]
    p = subprocess.Popen(networkCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stats_network = stdout.splitlines(True)

    statsCMD = [os.path.join("/usr/bin", "ion_sysinfo")]
    q = subprocess.Popen(statsCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = q.communicate()
    stats = stdout.splitlines(True)
    
    stats_dm = rawDataStorageReport.storage_report()

    # Create filename for the report
    reportFileName = "/tmp/stats_sys.txt"

    # Stuff the variable into the context object
    ctx = template.Context({"stats_network":stats_network,
                            "stats_network_cmd":networkCMD[0],
                            "stats":stats,
                            "stats_cmd":statsCMD[0],
                            "stats_dm":stats_dm,
                            "reportFilePath":reportFileName,
                            "use_precontent":True,
                            "use_content2":True,
                            "use_content3":True, })

    # Generate a file from the report
    try:
        os.unlink(reportFileName)
    except:
        logger.exception("Error! Could not delete '%s'", reportFileName)

    outfile = open(reportFileName, 'w')
    for line in stats_network:
        outfile.write(line)
    for line in stats:
        outfile.write(line)
    for line in stats_dm:
        outfile.write(line)
    outfile.close()
    # Set permissions so anyone can read/overwrite/destroy
    try:
        os.chmod(reportFileName,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
    except:
        logger.exception("Could not chmod '%s'", reportFileName)

    return shortcuts.render_to_response("rundb/ion_stats_sys.html",
                                        context_instance=ctx)

def run_tracker(request):
    """
    Generates the context for the runtracker page - not implemented
    """
    if 'submit' in request.GET:
        d = request.GET
    else:
        d = None
    sort = forms.BestRunsSort(d)
    ctxd = {"sort":sort}
    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_run_tracker.html",
                                        context_instance=context)

@login_required
def servers(request):
    """
    Renders the `Servers` tab by calling the view for each section
    """
    def process_status(process):
        return subprocess.Popen("service %s status" % process,
                  shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processes = [
         "ionJobServer",
         "ionCrawler",
         "ionPlugin",
         "ionArchive",
         "celeryd",
         "RSM_Launch",
         "dhcp3-server",
         "ntp"
    ]
    proc_set = dict((p, process_status(p)) for p in processes)
    for name, proc in proc_set.items():
        stdout, stderr = proc.communicate()
        proc_set[name] = proc.returncode == 0
        logger.info("%s out = '%s' err = %s''" % (name, stdout, stderr))
    # tomcat specific status code so that we don't need root privilege
    def complicated_status(filename, parse):
        try:
            if os.path.exists(filename):
                data = open(filename).read()
                pid = parse(data)
                proc = subprocess.Popen("ps %d" % pid, shell=True)
                proc.communicate()
                return proc.returncode == 0
        except Exception as err:
            return False
    proc_set["tomcat6"] = complicated_status("/var/run/tomcat6.pid", int)
    # pids should contain something like '[{rabbit@TSVMware,18442}].'
    proc_set["RabbitMQ"] = complicated_status("/var/lib/rabbitmq/pids",
                              lambda x: int(x[x.rindex(',')+1:x.rindex('}')]))

    cs = crawler_status(request)
    job = current_jobs(request)
    #archive = backup(request)
    archive = db_backup(request)

    ctxd = {"crawler":cs,
            "processes": sorted(proc_set.items()),
            "jobs":job,
            "archive":archive,
            "bk":reportOpt,
            "use_precontent":True,
            "use_content2":True,
            "use_content3":True
            }
    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_servers.html",
                                        context_instance=context)
    
@login_required
def PDFGen(request, pkR):
    from iondb.backup import makePDF
    pkR = pkR[:len(pkR)-4]
    return http.HttpResponse(makePDF.getPDF(pkR), mimetype="application/pdf")

def PDFGenOld(request, pkR):
    from iondb.backup import makePDF
    pkR = pkR[:len(pkR)-4]
    return http.HttpResponse(makePDF.getOldPDF(pkR), mimetype="application/pdf")

def getCSA(request, pk):
    '''Python replacement for the pipeline/web/db/writers/csa.php script'''
    try:
        from django.core.servers.basehttp import FileWrapper
        from iondb.backup import makePDF
        from ion.utils import makeCSA
        
        # From the database record, get the report information
        result = models.Results.objects.get(pk=pk)
        reportDir = result.get_report_dir()
        rawDataDir = result.experiment.expDir
        
        # Generate report PDF file.
        # This will create a file named backupPDF.pdf in results directory
        makePDF.makePDFdir(pk, reportDir)
        
        csaFullPath = makeCSA.makeCSA(reportDir,rawDataDir)
        csaFileName = os.path.basename(csaFullPath)
        
        response = http.HttpResponse(FileWrapper (open(csaFullPath)), mimetype='application/zip')
        response['Content-Disposition'] = 'attachment; filename=%s' % csaFileName
        
    except:
        logger.error(traceback.format_exc())
        response = http.HttpResponse(status=500)
        
    return response

def jobDetails(request):
    import commands
    file3 = open('/tmp/jobDetailLog.txt', 'w')
    file3.write('log opened...')
    
    if request.method == 'GET':
        file = open('/tmp/testLog3.txt', 'w')
        output = commands.getstatusoutput('qstat -f')[1]
        file.write(output)
        file.close()
        file = open('/tmp/testLog3.txt', 'r')
        jID = []
        qList = []
        currentQ = ''
        i = 0
        pendingCheck = False
        #Note: in order to only display only the queue name (no node name), just switch the first "string.split(line, ' ')"...
        #...into "string.split(line, '@')" below.
        for line in file.readlines():
            if '@' in line:
                qList.append([string.split(line, ' ')[0], 0, int(string.split(line, '/')[1]), int(string.split(line, '/')[2][:3]),
                              (100*int(string.split(line, '/')[1]))/int(string.split(line, '/')[2][:4])])
            if '#' in line:
                pendingCheck = True
            tempjID = 0
            line = string.strip(line, ' ')
            i = 1
            while i > 0:
                try:
                    tempjID = int(line[:i])
                    i += 1
                except:
                    jID.append(tempjID)
                    try:
                        if pendingCheck:
                            tempStat = string.split(commands.getstatusoutput('qstat -j %s'%tempjID)[1], '\n')
                            for stat in tempStat:
                                if 'hard_queue_list' in stat:
                                    for q in qList:
                                        if q[0] in stat:
                                            q[1] += 1
                    except:
                        pass
                        
                    i = 0
        file2 = open('/tmp/testLog4.txt', 'w')
        for ID in jID:
            if ID != 0:
                try:
                    file2.write(commands.getstatusoutput('qstat -j %s'%ID)[1])
                    file2.write('\nNEW\n')
                except:
                    file2.write('problem with job #%s'%ID)
                    file2.write('\nNEW\n')
        file2.close()
        file2 = open('/tmp/testLog4.txt', 'r')
        jobInfoList = []
        infoList = []
        keyList = []
        tempKey = []
        tempInfo = []
        tempList = []
        go = True
        for line in file2.readlines():
            #file3.write('%s\n'%tempList[len(tempList)-1])
            if go:
                if not 'NEW' in line:
                    if line[:5] == 'sched':
                        go = False
                    if go:
                        tempList.append('%s'%line)
                        line = string.replace(line, ' ', '')
                        #don't use rfind; some values contain ':', while the keys don't.
                        div = string.find(line, ':')
                        if div != -1:
                            tempKey.append('%s'%line[:div])
                            tempInfo.append('%s'%line[(div+1):])
                        else:
                            tempKey.append('')
                            tempInfo.append('')
                else:
                    jobInfoList.append({"JList":tempList, "KList":tempKey, "IList":tempInfo})
                    infoList.append(tempInfo)
                    keyList.append(tempKey)
                    tempKey = []
                    tempInfo = []
                    tempList = []
            elif 'NEW' in line:
                go = True
                jobInfoList.append({"JList":tempList, "KList":tempKey, "IList":tempInfo})
                infoList.append(tempInfo)
                keyList.append(tempKey)
                tempKey = []
                tempInfo = []
                tempList = []
        
        file3.write('\n\nJList: \n\n%s\n'%jobInfoList)
        tlJobs = []
        blocks = []
        qListParent = []
        for job in jobInfoList:
            tempIndex = 0
            for line in job['JList']:
                if line[:10] == 'hard_queue':
                    qListParent.append([])
                    qListParent[len(qListParent)-1].append(job['IList'][tempIndex][:(len(job['IList'][tempIndex])-1)])
                    qListParent[len(qListParent)-1].append(0)
                    qListParent[len(qListParent)-1].append(0)
                    qListParent[len(qListParent)-1].append(0)
                tempIndex += 1
            if 'tl.q\n' in job['IList']:
                file3.write('\n\n%s is tl.q job for result #%s.\n\n'%(job['IList'][1], '?'))
                for j in job['JList']:
                    if 'cwd' in j:
                        str = string.replace(j, ' ', '')
                        str = string.replace(str, '\n', '')
                        str = string.split(str, ':')[1]
                        pk = str[(string.rfind(str, '_')+1):]
                        pk = pk[:3]
                try:
                    #file = open('/tmp/jobInfoTestLog_%s.txt'%pk, 'a+')
                    file = open('%s/job_list.txt'%(shortcuts.get_object_or_404(models.Results, pk=pk).get_report_dir()))
                    tempSubJobList = []
                    jid = 0
                    arg = ''
                    numDone = 0
                    contents = file.readlines()
                    numJobs = len(contents)
                    for line in contents:
                        try:
                            jid = int(line)
                        except:
                            pass
                        if '%s'%256 != '%s'%commands.getstatusoutput('qstat -j %s'%jid)[0]:
                            for j in jobInfoList:
                                try:
                                    if int(j['IList'][1]) == jid:
                                        blockArg = string.split(j['IList'][15], '/')
                                        if len(blockArg[len(blockArg)-1]) > 2:
                                            blockArg = blockArg[len(blockArg)-1]
                                        else:
                                            blockArg = 'No associated block.'
                                        '''if not blockArg in blocks:
                                            blocks.append(bloackArg)'''
                                        lineArg = string.split(j['IList'][24], ',')[1]
                                        if lineArg == '--do-sigproc\n':
                                            arg = 'Signal Processing'
                                        elif lineArg == '--do-basecalling\n':
                                            arg = 'Basecalling'
                                        elif lineArg == '--do-alignment\n':
                                            arg = 'Alignment'
                                        elif lineArg == '--do-zipping\n':
                                            arg = 'Zipping'
                                        else:
                                            arg = 'Other (arg: %s)'%lineArg
                                except Exception as inst:
                                    pass
                        else:
                            arg = 'Done'
                        if jid != 0 and arg != '':
                            done = 'N'
                            if '%s'%256 == '%s'%commands.getstatusoutput('qstat -j %s'%jid)[0]:
                                done = 'Y'
                                numDone += 1
                            try:
                                if done != 'Y':
                                    tempSubJobList.append([jid, arg, done, blockArg])
                                    if not blockArg in blocks:
                                        blocks.append('%s'%blockArg)
                                else:
                                    tempSubJobList.append([jid, arg, done, 'Done'])
                                    if not 'Done' in blocks:
                                        blocks.append('Done')
                            except:
                                tempSubJobList.append([jid, arg, done, 'No associated block.'])
                            jid = 0
                            arg = ''
                    tlJobs.append({"job":job, "pk":pk, "jid":job["IList"][1], "Results":shortcuts.get_object_or_404(models.Results, pk=pk), "subJobs":tempSubJobList,
                                   "finished":'%s'%((numDone*100)/numJobs), "blocks":blocks})
                    blocks = []
                except:
                    pass
        ctxd = {"JList":jobInfoList, "KList":keyList, "IList":infoList, 'tlJobs':tlJobs, "qList":qList}
        context = template.RequestContext(request, ctxd)
        return shortcuts.render_to_response("rundb/ion_jobDetails.html", context_instance=context)
    
    elif request.method == 'POST':
        url = '/rundb/servers'
        return http.HttpResponsePermanentRedirect(url)

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

def validate_barcode(barCodeDict):
    """validate the barcode, return what failed"""

    failed = []
    requiredList = ["sequence"]

    for req in requiredList:
        if req in barCodeDict:
            if not barCodeDict[req]:
                failed.append( (req, "Required column is empty") )
        else:
            failed.append( (req, "Required column is missing") )


    nucOnly = ["sequence","floworder","adapter"]

    for nuc in nucOnly:
        if nuc in barCodeDict:
            if not set(barCodeDict[nuc].upper()).issubset("ATCG"):
                failed.append( (nuc, "Must have A, T, C, G only") )

    if 'score_mode' in barCodeDict:
            if barCodeDict["score_mode"]:
                try:
                    int(str(barCodeDict["score_mode"]))
                except ValueError:
                    failed.append( ("score_mode","score_mode must be a whole number") )

    if 'score_cutoff' in barCodeDict:
        if barCodeDict["score_cutoff"]:
            if barCodeDict["score_cutoff"]:
                try:
                    float(str(barCodeDict["score_cutoff"]))
                except ValueError:
                    failed.append( ("score_cutoff","score_cutoff must be a number") )

    if 'id_str' in barCodeDict:
        if not set(barCodeDict["id_str"]).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"):
            failed.append(("id_str", "str_id must only have letters, numbers, or the characters _ . - "))

    #do not let the index be set to zero. Zero is reserved.
    if 'index' in barCodeDict:
        if barCodeDict["index"] == "0":
            failed.append(("index", "index must not contain a 0. Indices should start at 1."))

    return failed


def add_barcode(request):
    """add the barcodes, with CSV validation"""

    if request.method == 'POST':
        name = request.POST.get('name', '')
        file = request.FILES['postedfile']

        destination = tempfile.NamedTemporaryFile(delete=False)

        for chunk in file.chunks():
            destination.write(chunk)

        file.close()
        destination.close()

        #check to ensure it is not empty
        headerCheck = open(destination.name, "rU")
        firstCSV = []

        for firstRow in csv.reader(headerCheck):
            firstCSV.append(firstRow)

        headerCheck.close()

        if not firstRow:
            os.unlink(destination.name)
            return http.HttpResponse(json.dumps({"status":"Error: Barcode file is empty"}) , mimetype="text/html")

        expectedHeader = ["id_str","type","sequence","floworder","index","annotation","adapter","score_mode","score_cutoff"]

        if sorted(firstCSV[0]) != sorted(expectedHeader):
            os.unlink(destination.name)
            return http.HttpResponse(json.dumps({"status":"Barcode csv header is not as expected. Please try again starting with the provided example"}) , mimetype="text/html")

        #test if the barcode set name has been used before
        barCodeSet = models.dnaBarcode.objects.filter(name=name)
        if barCodeSet:
            os.unlink(destination.name)
            return http.HttpResponse(json.dumps({"status":"Error: Barcode set with the same name already exists"}) , mimetype="text/html")

        index = 0
        barCodes = []
        failed = {}

        file = open(destination.name, "rU")
        reader = csv.DictReader(file)

        for index, row in enumerate(reader, start=1):

            invalid = validate_barcode(row)
            if invalid:
                #don't make dna object or add it to the list
                failed[index] = invalid
                continue

            newBarcode = models.dnaBarcode()

            #set the name
            newBarcode.name = name

            #set index this can be overwritten later
            newBarcode.index = index

            #fields that have to be uppercase
            nucs = ["sequence","floworder","adapter"]

            #set the values for the objects
            for key, value in row.items():

                #strip the strings
                value = str(value)
                value = value.strip()

                #uppercase if a nuc
                if key in nucs:
                    value = value.upper()

                if value:
                    setattr(newBarcode, key, value)

            #make a id_str if one is not provided
            if not newBarcode.id_str:
                newBarcode.id_str = str(name) + "_" + str(index)

            #now set a default
            newBarcode.length = len(newBarcode.sequence)

            #append to our list for later saving
            barCodes.append(newBarcode)

        #now close and remove the temp file
        destination.close()
        os.unlink(destination.name)

        if index > 384:
            return http.HttpResponse(json.dumps({"status":"Error: Too many barcodes! There must be 384 or less. Please reload the page and try again with fewer barcodes."}) , mimetype="text/html")

        if index == 0:
            return http.HttpResponse(json.dumps({"status":"Error: There must be at least one barcode! Please reload the page and try again with more barcodes."}) , mimetype="text/html")

        usedID = []
        for barCode in barCodes:
            if barCode.id_str not in usedID:
                usedID.append(barCode.id_str)
            else:
                error = {"status" : "Duplicate id_str for barcodes named: " + str(barCode.id_str) + "." }
                return http.HttpResponse(json.dumps(error) , mimetype="text/html")

        usedIndex = []
        for barCode in barCodes:
            if barCode.index not in usedIndex:
                usedIndex.append(barCode.index)
            else:
                error = {"status" : "Duplicate index: " + barCode.index + "." }
                return http.HttpResponse(json.dumps(error) , mimetype="text/html")

        if failed:
            r = {"status": "Barcodes validation failed. The barcode set has not been saved.",  "failed" : failed }
            return http.HttpResponse(json.dumps(r) , mimetype="text/html")

        #saving to db needs to be the last thing to happen
        for barCode in barCodes:
            try:
                barCode.save()
            except:
                return http.HttpResponse(json.dumps({"status":"Error saving barcode to database!"}) , mimetype="text/html")

        r = {"status": "Barcodes Uploaded! The barcode set will be listed on the references page.",  "failed" : failed }
        return http.HttpResponse(json.dumps(r) , mimetype="text/html")

    elif request.method == 'GET':
        ctx = template.RequestContext(request, {})
        return shortcuts.render_to_response("rundb/ion_barcode.html",
                                            context_instance=ctx)


@login_required
def save_barcode(request):
    """save barcode is still used by the edit barcode page"""

    if request.method == 'POST':
        name = request.POST.get('name', False)
        codes = request.POST.getlist('codes[]')
        if len(codes) > 384:
            return http.HttpResponse(json.dumps({"error":"too many barcodes must be 384 or less"}) , mimetype="text/html")

        #need to find the next number to is as the index
        def nextIndex(name):
            barCodeSet = models.dnaBarcode.objects.filter(name=name).order_by("-index")
            if barCodeSet:
                return barCodeSet[0].index + 1
            else:
                return 1

        for code in codes:
            bar = models.dnaBarcode(name=name, sequence=code)
            if models.dnaBarcode.objects.filter(name=name).count() < 384:
                bar.length = len(code)
                bar.type = "none"
                bar.index = nextIndex(name)
                bar.floworder = "none"
                bar.save()
            else:
                return http.HttpResponse(json.dumps({"error":"too many barcodes in the database must be 384 or less"}) , mimetype="text/html")


        return http.HttpResponse(json.dumps({}) , mimetype="text/html")

@login_required
def edit_barcode(request, name):
    """
    Simple view to display the edit barcode page
    """
    barcodes = models.dnaBarcode.objects.filter(name=name).order_by("index")
    ctxd = {"barcodes":barcodes}
    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_edit_barcodes.html",
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

@login_required
def graph_iframe(request,pk):
    """
    Make a Protovis graph from the requested metric
    """
    metric = request.GET.get('metric',False)

    result = shortcuts.get_object_or_404(models.Results, pk=pk)

    barcodeSummary = "alignment_barcode_summary.csv"
    data = barcodeData(os.path.join(result.get_report_dir(),barcodeSummary),metric)

    ctxd = { "data" : json.dumps(data) }
    context = template.RequestContext(request, ctxd)

    return shortcuts.render_to_response("rundb/ion_graph_iframe.html", context_instance=context)

@login_required
def plugin_iframe(request,pk):
    """
    load files into iframes
    """

    def openfile(fname):
        """strip lines """
        try:
            with open(fname, 'r') as f:
                content = f.read()
        except:
            logger.exception("Failed to open '%s'", fname)
            return False
        return content

    # Used in javascript, must serialize to json
    plugin = shortcuts.get_object_or_404(models.Plugin, pk=pk)
    #make json to send to the template
    plugin_json = json.dumps({'pk':pk,'model':str(plugin._meta), 'fields': model_to_dict(plugin)},cls=DjangoJSONEncoder)

    # If you set more than one of these, 
    # behavior is undefined. (one returned randomly)
    dispatch_table = {
        'report': 'instance.html',
        'config': 'config.html',
        'about': 'about.html',
    }
    for (k,v) in dispatch_table.items():
        if request.GET.get(k, False):
            fname = os.path.join(plugin.path, v)
            break
    else:
        logger.error("plugin iframe requires report, config or about")
        raise http.Http404()

    content = openfile(fname)
    if not content:
        raise http.Http404()

    index_version = settings.TMAP_VERSION

    report = request.GET.get('report', False)

    ctxd = {"plugin":plugin_json , "file" : content, "report" : report ,"tmap" : str(index_version) }
    context = template.RequestContext(request, ctxd)

    return shortcuts.render_to_response("rundb/ion_plugin_iframe.html",
                                            context_instance=context)

@login_required
def planning(request):
    """
    Run registation planning page
    """

    plans = models.PlannedExperiment.objects.filter(planExecuted=False).order_by("-date")

    ctxd = {"plans":plans}
    ctx = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_planning.html",
                                        context_instance=ctx, mimetype="text/html")

def validate_plan(plan):
    """validate a dict that is a plan"""

    failed = []
    requiredList = ["planName"]

    for req in requiredList:
        if req in plan:
            if not plan[req]:
                failed.append( (req, "Required column is empty") )
        else:
            failed.append( (req, "Required column is missing") )

    charCheckList = ["planName","sample","project"]
    for charCheck in charCheckList:
        if charCheck in plan:
            if not set(plan[charCheck]).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"):
                failed.append((charCheck, "Must only have letters, numbers, or the characters _ . - "))

    if 'cycles' in plan:
        if plan["cycles"]:
            try:
                minCheck = int(str(plan["cycles"]))
                if minCheck < 1:
                    failed.append( ("cycles","flows must be at least 1") )
            except ValueError:
                failed.append( ("cycles","flows must be a whole number") )

    alphaNumList = ["chipBarcode","seqKitBarcode", "forwardLibraryKey", "forward3PrimeAdapter", "reverseLibraryKey", "reverse3PrimeAdapter"]
    for alphaNum in alphaNumList:
        if alphaNum in plan:
            if not set(plan[alphaNum]).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
                failed.append((alphaNum, "Must only have letters or numbers"))

    checkList = ["autoAnalyze","preAnalysis"]
    checkWord = ["false", "False", "FALSE","f", "no", "n", "none", "0" , "true", "True", "TRUE","t", "yes", "y", "1", "" ]
    for check in checkList:
        if check in plan:
            if not plan[check] in checkWord:
                failed.append((check, 'Must contain only one of the values for false ("false", "False", "FALSE","f", "no", "n", "none", "0" ) or for true ("true", "True", "TRUE","t", "yes", "y", "1", "") '))

    checkList = ["isPairedEnd"]
    checkWord = ["false", "False", "FALSE", "f", "no", "No", "NO", "n", "none", "0", "true", "True", "TRUE", "t", "yes", "Yes", "YES", "y", "1", "" ]
    for check in checkList:
        if check in plan:
            if not plan[check] in checkWord:
                failed.append((check, 'Must contain only one of the values for false ("false", "False", "FALSE", "f", "no", "No", "NO", "n", "none", "0", "") or for true ("true", "True", "TRUE", "t", "yes", "Yes", "YES", "y", "1") '))

    if 'notes' in plan:
        if not set(plan["notes"]).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.- "):
            failed.append((alphaNum, "Must only have letters, numbers, spaces, or the characters _ . - "))

    #runType: items in the list
    runTypes = models.RunType.objects.all().order_by("id")
    runTypes = [r.runType for r in runTypes]
    if 'runType' in plan:
        if plan['runType']:
            if not plan['runType'] in runTypes:
                failed.append(("runType", "Must be a valid runType, default is GENS",plan['runType'],runTypes))

    #barcodeId: items in the list
    barcodes = models.dnaBarcode.objects.values_list('name').distinct().order_by('name')
    barcodes = [b[0] for b in barcodes]

    if 'barcodeID' in plan:
        if plan["barcodeID"]:
            if not plan["barcodeID"] in barcodes:
                failed.append(("barcodeID", "Must be a valid barcode id of an existing barcode set"))

    if plan.get("library", False):
        if not models.ReferenceGenome.objects.all().filter(
            short_name=plan["library"], index_version=settings.TMAP_VERSION):
            failed.append(("library", "Must be a valid genome reference library short form name. (also known as reference sequence)"))

    return failed

@login_required
def add_plans(request):
    """Take care of adding plans.  The POST method will do the CSV parsing and database additions"""

    if request.method == 'POST':
        #this POST method will verify the data sent in was good

        file = request.FILES['postedfile']

        destination = tempfile.NamedTemporaryFile(delete=False)

        for chunk in file.chunks():
            destination.write(chunk)

        destination.close()

        #check to ensure it is not empty
        headerCheck = open(destination.name, "rU")
        firstCSV = []

        for firstRow in csv.reader(headerCheck):
            firstCSV.append(firstRow)

        headerCheck.close()

        if not firstRow:
            os.unlink(destination.name)
            return http.HttpResponse(json.dumps({"status":"Error: Plan file is empty"}) , mimetype="text/html")

        expectedHeader = [ "planName", "sample", "project", "flows",
                          "runType", "library", "barcodeId",
                          "seqKitBarcode", "autoAnalyze", "preAnalysis",
                          "bedfile", "regionfile", "notes",
                          "forwardLibraryKey", "forward3PrimeAdapter",
                          "isPairedEnd", "reverseLibraryKey",
                          "reverse3PrimeAdapter"
                         ]

        if sorted(firstCSV[0]) != sorted(expectedHeader):
            os.unlink(destination.name)
            return http.HttpResponse(json.dumps({"status":"Plan csv header is not as expected. Please try again starting with the provided example"}) , mimetype="text/html")

        index = 0
        failed = {}
        plans = []

        file = open(destination.name, "rU")
        reader = csv.DictReader(file)

        for index, row in enumerate(reader, start=1):

            invalid = validate_plan(row)
            if invalid:
                #don't make dna object or add it to the list
                failed[index] = invalid
                continue

            newPlan = models.PlannedExperiment()

            #save it to get the pk
            for key, value in row.items():

                #convert to bool
                if key == 'autoAnalyze' or key == 'preAnalysis':
                    value = toBoolean(value)
                if key == 'isPairedEnd':
                    value = toBoolean(value, False)                    
            

                #don't save if the cycles is blank
                if key == 'flows' and value == '':
                    pass
                elif key =='forwardLibraryKey':
                    setattr(newPlan, 'libraryKey', value)
                elif key == 'forward3PrimeAdapter':
                    setattr(newPlan, 'forward3primeadapter', value)
                elif key == 'isPairedEnd':
                    setattr(newPlan, 'isReverseRun', value)
                elif key == 'reverseLibraryKey':
                    setattr(newPlan, 'reverselibrarykey', value)
                elif key == 'reverse3PrimeAdapter':
                    setattr(newPlan, 'reverse3primeadapter', value)
                else:
                    setattr(newPlan, key, value)                    
                
            #default runtype is GENS
            if not newPlan.runType:
                newPlan.runType = "GENS"

            newPlan.usePreBeadfind  = True
            newPlan.usePostBeadfind = True

            plans.append(newPlan)

        #now remove the file and close others
        file.close()
        os.unlink(destination.name)

        if index == 0:
            return http.HttpResponse(json.dumps({"status":"Error: There must be at least one Plan! Please reload the page and try again."}) , mimetype="text/html")

        if failed:
            r = {"status": "Plan validation failed. The plans set has not been saved.",  "failed" : failed }
            return http.HttpResponse(json.dumps(r) , mimetype="text/html")

        for plan in plans:
            try:
                plan.save()
            except:
                return http.HttpResponse(json.dumps({"status":"Error saving plan to database!","plan":plan.planName }) , mimetype="text/html")

        r = {"status": "Plans Uploaded! The plans will be listed on the planning page.",  "failed" : failed }
        return http.HttpResponse(json.dumps(r) , mimetype="text/html")


    ctx = template.RequestContext(request, {} )
    return shortcuts.render_to_response("rundb/ion_addplans.html",context_instance=ctx, mimetype="text/html")

def get_plan_data():

    data = {}
    data["runTypes"] = models.RunType.objects.all().order_by("id")
    data["barcodes"] = models.dnaBarcode.objects.values('name').distinct().order_by('name')
    data["references"] = models.ReferenceGenome.objects.all().filter(index_version=settings.TMAP_VERSION)

    allFiles = models.Content.objects.filter(publisher__name="BED",path__contains="/unmerged/detail/")
    bedFiles, hotspotFiles = [], []
    for file in allFiles:
        if file.meta.get("hotspot", False):
            hotspotFiles.append(file)
        else:
            bedFiles.append(file)

    data["bedFiles"] = bedFiles
    data["hotspotFiles"] = hotspotFiles

    data["seqKits"] = models.KitInfo.objects.filter(kitType='SequencingKit')
    data["libKits"] = models.KitInfo.objects.filter(kitType='LibraryKit')
    
    data["variantfrequencies"] = models.VariantFrequencies.objects.all().order_by("name")
    
    #is the ion reporter upload plugin installed?
    try:
        IRupload = models.Plugin.objects.get(name="IonReporterUploader",selected=True,active=True,autorun=True)
    ##    status, IRworkflows = tasks.IonReporterWorkflows()
    except models.Plugin.DoesNotExist:
        IRupload = False
     ##   status = False
     ##   IRworkflows = "Plugin Does Not Exist"
    except models.Plugin.MultipleObjectsReturned:
        IRupload = False
     ##  status = False
     ##  IRworkflows = "Multiple active versions of IonReporterUploader installed. Please uninstall all but one."

    data["IRupload"] = IRupload
    ##data["status"] = status
    ##data["IRworkflows"] = IRworkflows

    #the entry marked as the default will be on top of the list
    data["forwardLibKeys"] = models.LibraryKey.objects.filter(direction='Forward').order_by('-isDefault', 'name')
    data["forward3Adapters"] = models.ThreePrimeadapter.objects.filter(direction='Forward').order_by('-isDefault', 'name')
    data["reverseLibKeys"] = models.LibraryKey.objects.filter(direction='Reverse').order_by('-isDefault', 'name')
    data["reverse3Adapters"] = models.ThreePrimeadapter.objects.filter(direction='Reverse').order_by('-isDefault', 'name')

    return data


def get_allApplProduct_data():    
    data = get_base_planTemplate_data()
    
    for appl in models.RunType.objects.all():
        
        applType = appl.runType

        try:
            #we should only have 1 default per application. TODO: add logic to ApplProduct to ensure that
            defaultApplProduct = models.ApplProduct.objects.filter(isActive = True, isDefault = True, applType = appl)
            if defaultApplProduct[0]:
                
                applData = {}

                applData["runType"] = defaultApplProduct[0].applType
                applData["reference"] = defaultApplProduct[0].defaultGenomeRefName
                applData["targetBedFile"] = defaultApplProduct[0].defaultTargetRegionBedFileName
                applData["hotSpotBedFile"] = defaultApplProduct[0].defaultHotSpotRegionBedFileName
                applData["seqKit"] = defaultApplProduct[0].defaultSequencingKit
                applData["libKit"] = defaultApplProduct[0].defaultLibraryKit
                applData["peSeqKit"] = defaultApplProduct[0].defaultPairedEndSequencingKit
                applData["peLibKit"] = defaultApplProduct[0].defaultPairedEndLibraryKit
                applData["chipType"] = defaultApplProduct[0].defaultChipType
                applData["isPairedEndSupported"] = defaultApplProduct[0].isPairedEndSupported
                applData["isDefaultPairedEnd"] = defaultApplProduct[0].isDefaultPairedEnd
                applData['defaultVariantFrequency'] = defaultApplProduct[0].defaultVariantFrequency
                    
                #20120619-TODO-add compatible plugins, default plugins
            
                data[applType] = applData
                
                ##if applType == 'AMPS':
                  ##  pretty(data[applType])
            else:
                data[applType] = 'none'
        except:
            data[applType] = 'none'

    
    return data

def get_base_planTemplate_data():
    data = {}
    
    data["runTypes"] = models.RunType.objects.all().order_by("id")
    data["barcodes"] = models.dnaBarcode.objects.values('name').distinct().order_by('name')
    data["references"] = models.ReferenceGenome.objects.all().filter(index_version=settings.TMAP_VERSION)

    allFiles = models.Content.objects.filter(publisher__name="BED",path__contains="/unmerged/detail/")
    bedFiles, hotspotFiles = [], []
    for file in allFiles:
        if file.meta.get("hotspot", False):
            hotspotFiles.append(file)
        else:
            bedFiles.append(file)

    data["bedFiles"] = bedFiles
    data["hotspotFiles"] = hotspotFiles

    data["seqKits"] = models.KitInfo.objects.filter(kitType='SequencingKit')
    data["libKits"] = models.KitInfo.objects.filter(kitType='LibraryKit')
    
    data["variantfrequencies"] = models.VariantFrequencies.objects.all().order_by("name")

    #is the ion reporter upload plugin installed?
    try:
        IRupload = models.Plugin.objects.get(name="IonReporterUploader",selected=True,active=True,autorun=True)
        ##status, IRworkflows = tasks.IonReporterWorkflows()
    except models.Plugin.DoesNotExist:
        IRupload = False
        ##status = False
        ##IRworkflows = "Plugin Does Not Exist"
    except models.Plugin.MultipleObjectsReturned:
        IRupload = False
        ##status = False
        ##IRworkflows = "Multiple active versions of IonReporterUploader installed. Please uninstall all but one."

    data["IRupload"] = IRupload
    ##data["status"] = status
    ##data["IRworkflows"] = IRworkflows

    #the entry marked as the default will be on top of the list
    data["forwardLibKeys"] = models.LibraryKey.objects.filter(direction='Forward').order_by('-isDefault', 'name')
    data["forward3Adapters"] = models.ThreePrimeadapter.objects.filter(direction='Forward').order_by('-isDefault', 'name')
    data["reverseLibKeys"] = models.LibraryKey.objects.filter(direction='Reverse').order_by('-isDefault', 'name')
    data["reverse3Adapters"] = models.ThreePrimeadapter.objects.filter(direction='Reverse').order_by('-isDefault', 'name')

    #chip types
    data['chipTypes'] = models.Chip.objects.all().order_by('name')
    #QC
    data['qcTypes'] = models.QCType.objects.all().order_by('qcName')
    #plugins
    data['plugins'] = models.Plugin.objects.filter(active=True).exclude(name__contains="Uploader").order_by('name', '-version')
    #project
    data['projects'] = models.Project.objects.filter(public=True).order_by('name')
    #uploader plugins
    data['uploaders'] = models.Plugin.objects.filter(active=True, name__contains="Uploader").order_by('name', '-version')
    
    return data


@login_required
def add_plan(request):
    """add one plan"""

    ctxd = get_plan_data()

    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_addeditplan.html",
                                        context_instance=context)

 
def pretty(d, indent=0):
    if d:
        for key, value in d.iteritems():
            logger.debug('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                logger.debug('\t' * (indent+1) + str(value))
        
@login_required
def edit_plan(request, id):
    """
    Simple view to display the edit barcode page
    """
    ctxd = get_plan_data()

    #get the existing plan
    ctxd["plan"] = shortcuts.get_object_or_404(models.PlannedExperiment,pk=id)

    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_addeditplan.html",
                                        context_instance=context)
def _edit_experiment(request, id):
    """
    Simple view to display the edit barcode page
    """

    #get the info to populate the page
    data = get_plan_data()

    #get the existing plan
    exp = shortcuts.get_object_or_404(models.Experiment,pk=id)

    selectedReference = exp.library

    #for paired-end reverse run, separate library key and 3' adapter needed
    selectedForwardLibKey = exp.libraryKey
    selectedForward3PrimeAdapter = exp.forward3primeadapter
    
    selectedReverseLibKey = exp.reverselibrarykey
    selectedReverse3PrimerAdapter = exp.reverse3primeadapter

    #the entry marked as the default will be on top of the list
    forwardLibKeys = models.LibraryKey.objects.filter(direction='Forward').order_by('-isDefault', 'name')
    forward3Adapters = models.ThreePrimeadapter.objects.filter(direction='Forward').order_by('-isDefault', 'name')
    reverseLibKeys = models.LibraryKey.objects.filter(direction='Reverse').order_by('-isDefault', 'name')
    reverse3Adapters = models.ThreePrimeadapter.objects.filter(direction='Reverse').order_by('-isDefault', 'name')


    ctxd = {
            "bedFiles":data["bedFiles"],
            "hotspotFiles" : data["hotspotFiles"],
            "libKits": data["libKits"],
            "seqKits" : data["seqKits"],
            "variantfrequencies": data["variantfrequencies"],
            "runTypes": data["runTypes"],
            "barcodes" : data["barcodes"],
            "references" : data["references"],

            "exp":exp,
            "selectedReference" : selectedReference,
            "selectedForwardLibKey" : selectedForwardLibKey,
            "selectedForward3PrimeAdapter" : selectedForward3PrimeAdapter,
            "selectedReverseLibKey" : selectedReverseLibKey,
            "selectedReverse3PrimerAdapter" : selectedReverse3PrimerAdapter,
            "forwardLibKeys" : forwardLibKeys,
            "forward3Adapters" : forward3Adapters, 
            "reverseLibKeys" : reverseLibKeys, 
            "reverse3Adapters" : reverse3Adapters
             }

    context = template.RequestContext(request, ctxd)
    return context

@login_required
def edit_experiment(request, id):
    """
    Simple view to display the edit barcode page
    """
    context = _edit_experiment(request, id)
    return shortcuts.render_to_response("rundb/ion_editexp.html", context_instance=context)

@login_required
def exp_ack(request):
    if request.method == 'POST':

        runPK = request.POST.get('runpk', False)
        runState = request.POST.get('runstate', False)

        if not runPK:
            return http.HttpResponse(json.dumps({"status":"error, no runPK POSTed"}) , mimetype="application/json")

        if not runState:
            return http.HttpResponse(json.dumps({"status":"error, no runState POSTed"}) , mimetype="application/json")

        try:
            exp = models.Experiment.objects.get(pk=runPK)
        except :
            return http.HttpResponse(json.dumps({"status":"error, could find the run"}) , mimetype="application/json")

        try:
            exp.user_ack = runState
            exp.save()
        except :
            return http.HttpResponse(json.dumps({"status":"error, could not modify the user_ack state for " + str(exp) }) , mimetype="application/json")

        try:
            host = "127.0.0.1"
            conn = client.connect(host, settings.IARCHIVE_PORT)
            user_ack = conn.user_ack()
        except :
            return http.HttpResponse(json.dumps({"status":"error, could not connect the the backup process over xmlrpc" }) , mimetype="application/json")

        return http.HttpResponse(json.dumps({"runState": runState, "user_ack" : exp.user_ack , "runPK" : runPK, "user_ack": user_ack }) , mimetype="application/json")

@login_required
def add_edit_barcode(request, barCodeSet):
    """
    Simple view to display the edit barcode page
    """

    def nextIndex(name):
        barCodeSet = models.dnaBarcode.objects.filter(name=name).order_by("-index")
        if barCodeSet:
            return barCodeSet[0].index + 1
        else:
            return 1

    barcodeID = request.GET.get('barcode', '')

    #if there is a barcode do a look up for it
    if barcodeID:
        barcode = models.dnaBarcode.objects.get(pk=int(barcodeID))
        index = barcode.index
        #get a list of all the other barcodes minus this one
        others = models.dnaBarcode.objects.filter(name=barCodeSet)
        others = others.exclude(pk=int(barcodeID))
    else:
        barcode = False
        index = nextIndex(barCodeSet)
        #get a list of all the other barcodes
        others = models.dnaBarcode.objects.filter(name=barCodeSet)


    otherList = []
    for other in others:
        otherList.append(other.id_str)

    ctxd = {"barcode" : barcode, "barCodeSet": barCodeSet , "index" : index , "otherList" : json.dumps(otherList) }

    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_barcode_addedit.html",
                                        context_instance=context)

@login_required
def pluginZipUpload(request):
    """file upload for plupload"""
    if request.method == 'POST':
        name = request.REQUEST.get('name','')
        uploaded_file = request.FILES['file']
        if not name:
            name = uploaded_file.name
        name,ext = os.path.splitext(name)

        #check to see if a user has uploaded a file before, and if they have
        #not, make them a upload directory

        upload_dir = "/results/plugins/scratch/"

        if not os.path.exists(upload_dir):
            return render_to_json({"error":"upload path does not exist"})

        dest_path = '%s%s%s%s' % (upload_dir,os.sep,name,ext)

        chunk = request.REQUEST.get('chunk','0')
        chunks = request.REQUEST.get('chunks','0')

        debug = [chunk, chunks]

        with open(dest_path,('wb' if chunk==0 else 'ab')) as f:
            for content in uploaded_file.chunks():
                f.write(content)

        if int(chunk) + 1 >= int(chunks):
            #the upload has finished
            pass

        return render_to_json({"chuck posted":debug})

    else:
        return render_to_json({"method":"only post here"})


@login_required
def account(request):
    (userprofile, created) = models.UserProfile.objects.get_or_create(user=request.user)
    if request.method == "POST":
        form = forms.UserProfileForm(data=request.POST, instance=userprofile)
        if form.is_valid():
            form.save()
            updated = True
    else:
        form = forms.UserProfileForm(instance=userprofile)

    context = template.RequestContext(request, {'form': form,})
    return shortcuts.render_to_response("rundb/ion_account.html",
                                        context_instance=context)

# NO login_required - new users
def registration(request):
    if request.method == 'POST': # If the form has been submitted...
        form = forms.UserRegistrationForm(request.POST)
        if form.is_valid(): # All validation rules pass
            # create user from cleaned_data
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password1']

            new_user = models.User.objects.create_user(username, email, password)
            new_user.is_active = True # WARNING: Self-register active user!
            try:
                group = models.Group.objects.get(name='ionusers')
                new_user.groups.add(group)
            except models.Group.DoesNotExist:
                log.warn("Group ionusers not found. " + \
                         "New user %s will lack permission to do anything beyond look!", username)
            new_user.save()


            # Log user in.
            # Note: TODO More secure - deactivate user account and require approval before login.
            user = authenticate(username=username, password=password)
            if user is not None and user.is_active:
                login(request, user)
                # Redirect to home when login successful
                return shortcuts.redirect(urlresolvers.reverse('homepage'))

            # Otherwise redirect to a success page. Sending to login for now...
            return shortcuts.redirect(urlresolvers.reverse('login'))
    else:
        # Blank Form
        form = forms.UserRegistrationForm()

    context = template.RequestContext(request, {'form': form,})
    return shortcuts.render_to_response("rundb/ion_account_reg.html",
                                        context_instance=context)







