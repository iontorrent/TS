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
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.models import User

try:
    import json
except:
    import simplejson as json

#for sorting a list of lists by a key
from operator import itemgetter

sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

from iondb.rundb.ajax import render_to_json
from iondb.backup import rawDataStorageReport
from django import http, shortcuts, template
from django.conf import settings
from django.core.paginator import Paginator, InvalidPage, EmptyPage
from django.core import urlresolvers
from django import http
from django.core import serializers

from django.utils.datastructures import SortedDict

os.environ['MPLCONFIGDIR'] = '/tmp'
import matplotlib

from iondb.anaserve import client
from iondb.rundb import forms
from iondb.rundb import models
from iondb.utils import tables
from iondb.backup import devices
from iondb.backup.archiveExp import Experiment
from twisted.internet import reactor
from twisted.web import xmlrpc, server

from iondb.rundb import publishers
from iondb.rundb import tasks

FILTERED = None # contains the last filter for the reports page for csv export

logger = logging.getLogger(__name__)

def toBoolean(val):
    """convert strings from CSV to Python bool
    if they have an empty string - default to true
    """
    trueItems = ["true", "t", "yes", "y", "1", "" ]
    if str(val).strip().lower() in trueItems:
        return True
    falseItems = ["false", "f", "no", "n", "none", "0" ]
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
    site_name = models.GlobalConfig.objects.get(pk=1).site_name
    return {"base_site_name": site_name}


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
            exp = exp.filter(project=project)
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

    storages = models.ReportStorage.objects.all().order_by('id')
    storage = storages[len(storages) - 1]

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
            rep = rep.filter(experiment__project=project)
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

    ctxd = {"rep":ctx, "paginator":ctx, "filterform":filter, "getURL":getURL,
            "searchform":search, "sortform":sort, }
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

def build_result(experiment, name, server, location):
    """Initialize a new `Results` object named ``name``
    representing an analysis of ``experiment``. ``server`` specifies
    the ``models.reportStorage`` for the location in which the report output
    will be stored, and ``location`` is the
    ``models.Location`` object for that file server's location.
    """
    link = path.join(server.webServerPath, location.name, "%s_%%03d" % name)
    j = lambda l: path.join(link, l)
    report_name = "Default_Report.php"
    if get_alternative():
        report_name = 'Detailed_Report.php'
    storages = models.ReportStorage.objects.all().order_by('id')
    storage = storages[len(storages) - 1]
    kwargs = {
        "experiment":experiment,
        "resultsName":name,
        "sffLink":j("%s_%s.sff" % (experiment, name)),
        "fastqLink":j("%s_%s.fastq" % (experiment, name)),
        "reportLink":j(report_name),
        "status":"Started",
        "tfSffLink":j("%s_%s.tf.sff" % (experiment, name)),
        "tfFastq":"_",
        "log":j("log.html"),
        "analysisVersion":"_",
        "processedCycles":"0",
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

def create_meta(experiment, resultsName):
    """Build the contents of a report metadata file (``expMeta.dat``)."""
    lines = ("Run Name = %s" % experiment.expName,
             "Run Date = %s" % experiment.date,
             "Analysis Name = %s" % resultsName,
             "Analysis Date = %s" % datetime.date.today(),
             "Analysis Cycles = %s" % experiment.cycles, #TODO: change this to flows
             "Analysis Flows = %s" % experiment.flows,
             "Project = %s" % experiment.project,
             "Sample = %s" % experiment.sample,
             "Library = %s" % experiment.library,
             "PGM = %s" % experiment.pgmName,
             "Chip Check = %s" % get_chipcheck_status(experiment),
             "Chip Type = %s" % experiment.chipType,
             "Chip Data = %s" % experiment.rawdatastyle,
             "Notes = %s" % experiment.notes,
             "Barcode Set = %s" % experiment.barcodeId,
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
    db_barcodes = models.dnaBarcode.objects.filter(name=barcodeId)
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
        loc = loc[0]

        #if there is not a default, just take the first one
        if not loc:
            loc = models.Location.objects.all()
            loc = loc[0]


    # Always use the last, latest ReportStorage object
    storages = models.ReportStorage.objects.all().order_by('id')
    storage = storages[len(storages) - 1]
    start_error = None
    if request.method == 'POST':
        rpf = forms.RunParamsForm(request.POST, request.FILES)
        # validate the form
        if rpf.is_valid():
            # This handles the takeovernode button
            tn = rpf.cleaned_data['takeover_node']
            if tn:
                chiptype_arg = 'takeover'
            else:
                chiptype_arg = exp.chipType
            ufResultsName = rpf.cleaned_data['report_name']
            resultsName = ufResultsName.strip().replace(' ', '_')

            result = build_result(exp, resultsName, storage, loc)
            # load script and determine the location/url of the results
            # analysis
            script = models.RunScript.objects.all()[0].script
            webRootPath = result.web_root_path(loc)
            #BPP - variable not used anywhere.  delete?
            #linkPath = result.web_path(loc)
            tfConfig = rpf.cleaned_data['tf_config']
            blockArgs = rpf.cleaned_data['blockArgs']
            doThumbnail = rpf.cleaned_data['do_thumbnail']
            ts_job_type = ""
            if doThumbnail:
                ts_job_type = 'thumbnail'
            args = rpf.cleaned_data['args']
            #do a full alignment?
            align_full = rpf.cleaned_data['align_full']
            #Check to see if SFFTrim should be used
            sfftrim = models.GlobalConfig.objects.all()[0].sfftrim
            #get sff args
            sfftrim_args = models.GlobalConfig.objects.all()[0].sfftrim_args
            #If libraryKey was set, then override the value taken from the explog.txt on the PGM
            libraryKey = rpf.cleaned_data['libraryKey']
            #ionCrawler may modify the path to raw data in the path variable passed thru URL
            exp.expDir = rpf.cleaned_data['path']

            aligner_opts_extra = rpf.cleaned_data['aligner_opts_extra']

            result.save()
            try:
                # create a results object
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
                if bk and 'from-wells' not in args:
                    if str(bk.backupPath) == 'DELETED':
                        bail(result, "The analysis cannot start because the raw data has been deleted.")
                    else:
                        if not path.exists(exp.expDir):
                            bail(result, "The analysis cannot start because the raw data has been archived to %s.  Please mount that drive to make the data available."
                                 % (str(bk.backupPath),))
                if 'from-wells' not in args and not path.exists(exp.expDir):
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
                files.append(create_meta(exp, resultsName))
                files.append(create_pk_conf(result.pk))
                # write barcodes file to folder
                if exp.barcodeId and exp.barcodeId is not '':
                    files.append(create_bc_conf(exp.barcodeId,"barcodeList.txt"))
                # tell the analysis server to start the job
                params = makeParams(exp, args, blockArgs, doThumbnail, resultsName, result, align_full, sfftrim, sfftrim_args, libraryKey,
                                                        os.path.join(storage.webServerPath, loc.name),aligner_opts_extra)
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
                url = urlresolvers.reverse('report-started', args=(result.pk,))
                return http.HttpResponsePermanentRedirect(url)
            except BailException as be:
                start_error = be.msg
                result.delete()
    # fall through if not valid...
    elif request.method == 'GET':
        rpf = forms.RunParamsForm()
        rpf.fields['path'].initial = path.join(exp.expDir)
        rpf.fields['align_full'].initial = False

        extra = get_initial_arg(reportpk)
        args = models.GlobalConfig.objects.all()[0].get_default_command()
        chipargs = get_chip_args(exp.chipType)
        alist = [args, extra, chipargs]
        alist = [a for a in alist if a is not None]
        rpf.fields['args'].initial = " ".join(alist)
    ctx = {"rpf": rpf, "expName":exp.expName, "start_error":start_error}
    ctx = template.RequestContext(request, ctx)
    return shortcuts.render_to_response("rundb/ion_run_start_template_cp.html",
                                        context_instance=ctx)
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
        report = models.Results.objects.get(pk=pk)
        #ret = report.reportLink.strip().split('/')
        #ret.pop(-1)
        #ret.pop(1)
        #ret = '/'.join(ret).strip()[1:]
        #storage = storage.dirPath
        ret = path.join(report.get_report_dir(), '1.wells')
        cmd = '--from-wells=%s' % ret
        return cmd
    else:
        return None

def report_started(request, pk):
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
    tmplname = "rundb/ion_successful_start_analysis.html"
    return shortcuts.render_to_response(tmplname, context_instance=ctx)

def get_selected_plugins():
    """
    Build a list containing dictionaries of plugin information.
    will only put the plugins in the list that are selected in the 
    interface
    """
    try:
        pg = models.Plugin.objects.filter(selected=True).exclude(path='')
    except:
        return ""
    ret = []
    if len(pg) > 0:
        ret = [{'name':p.name,
                'path':p.path,
                'version':p.version,
                'project':p.project,
                'sample':p.sample,
                'libraryName':p.libraryName,
                'chipType':p.chipType,
                'autorun':p.autorun} for p in pg]
        return ret
    else:
        return ""

def get_alternative():
    """
    Secret sauce
    """
    try:
        ret = settings.ALTERNATIVE
    except:
        ret = False
    return ret

def makeParams(expOb, args, blockArgs, doThumbnail, resultsName, result, align_full , sfftrim, sfftrim_args, libraryKey,
                                url_path, aligner_opts_extra):
    """Build a dictionary of analysis parameters, to be passed to the job
    server when instructing it to run a report.  Any information that a job
    will need to be run must be constructed here and included inside the return.  
    This includes any special instructions for flow control in the top level script."""
    gc = models.GlobalConfig.objects.all()[0]
    plugins = get_selected_plugins()
    exp = expOb
    pathToData = path.join(exp.expDir)
    if doThumbnail and exp.chipType == "900":
        pathToData = path.join(pathToData,'thumbnail')
    defaultLibKey = gc.default_library_key
    expName = exp.expName

    #get the exp data for sam metadata
    exp_filter = models.Experiment.objects.filter(pk=exp.pk)
    exp_json = serializers.serialize("json", exp_filter)
    exp_json = json.loads(exp_json)
    exp_json = exp_json[0]["fields"]

    site_name = models.GlobalConfig.objects.get(pk=1).site_name

    try:
        libraryName = models.ReferenceGenomes.objects.get(short_name=exp.library).name
    except:
        libraryName = exp.library

    skipchecksum = False

    #TODO: remove the libKey from the analysis args, assign this in the TLScript. To make this more fluid

    #if the librayKey was set by createReport use that value. If not use the value from the PGM
    if not libraryKey:
        libraryKey = exp.libraryKey

    if libraryKey == None or len(libraryKey) < 1:
        libraryKey = defaultLibKey

    #TODO: SEE IF WE CAN TAKE OFF -C off
    if len(args.strip()) != 0:
        analysisArgs = args.strip() + " %s" % (pathToData)
    else:
        analysisArgs = "%s" % (pathToData)

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
    project = exp.project
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
    if exp.reverse_primer:
        primer_name = exp.reverse_primer
    else:
        primer_name = 'Ion Kit'
    try:
        reverse_primer_dict = models.ThreePrimeadapter.objects.get(name=primer_name)
    except:
        reverse_primer_dict = None
    if not reverse_primer_dict:
        reverse_primer_dict = {'name':'Ion Kit',
                               'sequence':'ATCACCGACTGCCCATAGAGAGGCTGAGAC',
                               'qual_cutoff':9,
                               'qual_window':30,
                               'adapter_cutoff':16
                               }
    rawdatastyle = exp.rawdatastyle
    
    ret = {'pathToData':pathToData,
           'analysisArgs':analysisArgs,
           'blockArgs':blockArgs,
           'libraryName':libraryName,
           'resultsName':resultsName,
           'expName':expName,
           'libraryKey':libraryKey,
           'plugins':plugins,
           'skipchecksum':skipchecksum,
           'flowOrder':flowOrder,
           'align_full' : align_full,
           'sfftrim' : sfftrim,
           'sfftrim_args' : sfftrim_args,
           'project':project,
           'sample':sample,
           'chiptype':chipType,
           'barcodeId':barcodeId,
           'net_location':net_location,
           'exp_json': json.dumps(exp_json),
           'site_name': site_name,
           'url_path':url_path,
           'reverse_primer_dict':reverse_primer_dict,
           'rawdatastyle':rawdatastyle,
           'aligner_opts_extra':aligner_opts_extra 
           }

    return ret

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

def blank(request, **kwargs):
    """
    just render a blank template
    """
    return shortcuts.render_to_response("rundb/ion_blank.html", {'tab':kwargs['tab']})

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
    
    # Determine if ionArchive service is running
    daemon_status = areyourunning()
    
    #Note: there is always only 1 BackupConfig object configured.
    bk = models.BackupConfig.objects.all().order_by('pk')
    # populate dictionary with experiments ready to be archived
    # Get all experiments in database sorted by date
    experiments = models.Experiment.objects.all().order_by('date')
    # Filter out all experiments marked Keep
    experiments = experiments.exclude(storage_options='KI').exclude(expName__in = models.Backup.objects.all().values('backupName'))
    # Filter out experiments marked 'U' or 'D'
    experiments = experiments.exclude(user_ack='U').exclude(user_ack='D')
    
    # make dictionary, one array per file server of archiveExperiment objects
    to_archive = {}
    servers = get_servers()
    for fs in servers:
        explist = []
        for exp in experiments:
            if fs.filesPrefix in exp.expDir:
                #TODO: remove this dependency on the rig object (pgmName)
                #if the rig no longer exists, the experiment cannot be deleted
                location = models.Rig.objects.get(name=exp.pgmName).location
                E = Experiment(exp,
                               str(exp.expName),
                               str(exp.date),
                               str(exp.star),
                               str(exp.storage_options),
                               str(exp.user_ack),
                               str(exp.expDir),
                               location,
                               exp.pk)
                explist.append(E)
            # limit number in list to configured limit
            if len(explist) >= bk[0].number_to_backup:
                break
        to_archive[fs.filesPrefix] = explist
        
    ctx = template.RequestContext(request, {"backups":bk, "to_archive":to_archive, "status":daemon_status})
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

def findVersions():
    """
    find the version of the packages
    """
    packages =  ["ion-alignment",
                 "ion-analysis",
                 "ion-dbreports",
                 "ion-docs",
                 "ion-gpu",
                 "ion-pgmupdates",
                 "ion-plugins",
                 "ion-referencelibrary",
                 "ion-rsmts",
                 "ion-sampledata",
                 "ion-tsconfig",
                 "ion-tsups",
                 "ion-publishers",
                 "ion-onetouchupdater",
                 "ion-pipeline",
                 "ion-usbmount",
                 "tmap",
                 "ion-torrentR"]

    packages = sorted(packages)

    ret = SortedDict()
    for package in packages:
        #command for version checking
        com = "dpkg -s %s | grep Version | head -1" % package
        try:
            a = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #just get the version number
            tolist = a.stdout.readlines()[0].strip().split("Version: ")[1]
            ret[package] = tolist
        except:
            pass

    from ion import version
    meta_version = version

    return ret, meta_version

def about(request):
    """
    Generates the information on the `About` tab
    """
    ret, meta_version = findVersions()
    ctxd = {"about":ret, "meta": meta_version, "user": request.user}
    ctx = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_about.html",
                                        context_instance=ctx, mimetype="text/html")

def search_for_plugins(config, basepath):
    """
    Searches for new plugins
    """
    def create_new(name, version, path):
        p = models.Plugin()
        p.name = name
        p.version = version
        p.date = datetime.datetime.now()
        p.selected = False
        p.path = path
        p.autorun = False
        p.save()

    def get_version(pdir, name, defaultpluginscript):
        command = "grep VERSION %s/%s | head -1 | awk '{print $1}' " % (pdir, defaultpluginscript)
        try:
            a = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            version = a.stdout.readlines()[0].strip()
            version = version.split("=")[1]
            version = version.replace('"', '')
        except:
            version = "0"
        return version

    def update_version(plugin, version):
        plugin.version = version
        plugin.save()

    def set_pluginconfig (plugin, pdir, configfile):
        """if there is data in pluginconfig json
        set the plugin.config value in the database
        """
        pluginconfigfile = os.path.join(pdir, configfile)
        if not os.path.exists(pluginconfigfile):
            return
        logger.info("Setting pluginconfig: '%s'", pluginconfigfile)
        try:
            with open(pluginconfigfile) as f:
                config = f.read()
            config = json.loads(config)
            plugin.config = config
            plugin.save()
            return
        except:
            logger.exception("Failed to load pluginconfig from '%s'", pluginconfigfile)
            return

    default_script = config.default_plugin_script
    basedir = "/results/plugins" # FIXME - get from globalconfig
    if path.exists(basedir):
        # only list files in the 'plugin' directory if they are actually folders
        folder_list = [i for i in os.listdir(basedir) if (path.isdir(path.join(basedir, i)) and i != "scratch")]
        for pname in folder_list:
            full_path = path.join(basedir, pname)
            try:
                p = models.Plugin.objects.get(name=pname.strip())
            except:
                create_new(pname,
                           get_version(full_path, pname, default_script),
                           full_path)
                p = models.Plugin.objects.get(name=pname.strip())
                if not p.config:
                    set_pluginconfig (p,
                                      path.join(basedir, pname),
                                      "pluginconfig.json")
                continue
            current_version = get_version(path.join(basedir, pname),
                                          pname,
                                          default_script)

            if p.version != current_version:
                update_version(p, current_version)

            # update paths in place
            if p.path != full_path:
                p.path = path
                p.save()

            if not p.config:
                set_pluginconfig (p,
                             path.join(basedir, pname),
                             "pluginconfig.json")
    return None

def purge_plugins(plugin_path):
    """
    Removes records from plugin table which no longer have corresponding folder
    on the file system.  If the folder does not exist, we assume that the plugin
    has been deleted.  In any case, one cannot execute the plugin if the plugin
    folder has been removed.
    """
    # get all plugin records from table
    plugins = models.Plugin.objects.all().exclude(path='')
    # for each record, test for corresponding folder
    for plugin in plugins:    
        # if folder does not exist
        if not os.path.isdir(plugin.path):
            ## FIXME - mark as inactive / missing / disabled
            #plugin.autorun = False
            #plugin.selected = False
            #plugin.path = ''
            #plugin.save()
            # Remove plugin record
            plugin.delete()
    return


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
        config = config.get(pk=1)
        config.site_name = request.POST["site_name"]
        config.save()
        context.update({"base_site_name": request.POST["site_name"]})


def global_config(request):
    """
    Renders the Config tab.
    """
    globalconfig = models.GlobalConfig.objects.all().order_by('pk')
    ctx = template.RequestContext(request, {})
    # As the config page takes on more responsibility, things that need updating
    # can be updated from here:
    config_contacts(request, ctx)
    config_site_name(request, ctx, globalconfig)
    
    emails = models.EmailAddress.objects.all().order_by('pk')
    purge_plugins('/results/plugins')
    publishers.purge_publishers()
    for c in globalconfig:
        search_for_plugins(c, c.plugin_folder)
        publishers.search_for_publishers(c)
    plugs = models.Plugin.objects.all().order_by('pk')
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


def how_are_you(request, host, port):
    """Open a socket to the given host and port, if we can :), if not :(
    """
    try:
        s = socket.create_connection((host, int(port)), 10)
        s.close()
        status = ":)"
    except Exception as complaint:
        logger.warn(complaint)
        status  = ":("
    return http.HttpResponse('{"feeling":"%s"}' % status,
                        mimetype='application/javascript')


def fetch_remote_content(request, url):
    """Perform an HTTP GET request on the given url and forward it's result.
    This is specifically for remote requests which need to originate from this
    server rather than from the client connecting to the torrent server.
    """
    try:
        remote = urllib.urlopen(url)
        data = remote.read()
        remote.close()
    except Exception as complaint:
        logger.warn(complaint)
        data = ""
    return http.HttpResponse(data)


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

def stats_sys(request):
    """
    Generates the stats page on system configuration
    """

    # Run a script on the server to generate text
    networkCMD = [os.path.join("/opt/ion/iondb/bin", "networkchecker")]
    p = subprocess.Popen(networkCMD, shell=True, stdout=subprocess.PIPE)
    p.wait()
    stats_network = p.stdout.readlines()

    statsCMD = [os.path.join("/opt/ion/iondb/bin", "collectInfo.sh")]
    q = subprocess.Popen(statsCMD, shell=True, stdout=subprocess.PIPE)
    q.wait()
    stats = q.stdout.readlines()
    
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
        print "Error! Could not delete ", reportFileName

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
        print "FOOBAR! could not chmod ", reportFileName

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

def servers(request):
    """
    Renders the `Servers` tab by calling the view for each section
    """
    cs = crawler_status(request)
    job = current_jobs(request)
    #archive = backup(request)
    archive = db_backup(request)

    ctxd = {"crawler":cs,
            "jobs":job,
            "archive":archive,
            "use_precontent":True,
            "use_content2":True,
            "use_content3":True
            }
    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_servers.html",
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
        return shortcuts.render_to_response("rundb/ion_barcode.html")


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
        #print >> sys.stderr, "File does not exist: ", filename
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
                        #print >> sys.stderr, "ERROR: ", e
                        ## Could have truncated data!
                        pass
                else:
                    print sys.stderr, "Metric missing: ", metric
            else:
                del row[""] ## Delete empty string (from trailing comma)
                #return a list of dicts where each dict is one row
                dictList.append(row)
    except (IOError, csv.Error) as e:
        print >> sys.stderr, "ERROR: ", e
        ## Could have truncated data!
        pass
    except:
        print >> sys.stderr, "ERROR: ", sys.exc_info()
        return False

    if not dictList:
        print >> sys.stderr, "WARNING: Empty Metric List"
        return False

    #now sort by the "axis" label
    if metric:
        dictList = sorted(dictList, key=itemgetter('axis'))
    else:
        dictList = sorted(dictList, key=itemgetter('Index'))

    return dictList

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

def plugin_iframe(request,pk):
    """
    if a plugin has instance.html or instance.json provide this from a django template
    one weakness of this method is that there is not an easy way to provide static content with
    the plugin
     """

    def openfile(file):
        """strip lines """
        try:
            file = open(file)
        except:
            return False
        file = file.read()
        return file

    plugin = shortcuts.get_object_or_404(models.Plugin, pk=pk)
    #make json to send to the template
    plugin_json = models.Plugin.objects.filter(pk=pk)
    plugin_json = serializers.serialize("json", plugin_json)
    plugin_json = json.loads(plugin_json)[0]
    plugin_json = json.dumps(plugin_json)

    report = request.GET.get('report', False)
    config = request.GET.get('config', False)
    about  = request.GET.get('about', False)

    file = openfile(os.path.join(plugin.path, "instance.html"))

    if config:
        file = openfile(os.path.join(plugin.path, "config.html"))
    if about:
        file = openfile(os.path.join(plugin.path, "about.html"))

    ctxd = {"plugin":plugin_json , "file" : file, "report" : report }
    context = template.RequestContext(request, ctxd)

    return shortcuts.render_to_response("rundb/ion_plugin_iframe.html",
                                            context_instance=context)

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

    alphaNumList = ["chipBarcode","seqKitBarcode"]
    for alphaNum in alphaNumList:
        if alphaNum in plan:
            if not set(plan[alphaNum]).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
                failed.append((alphaNum, "Must only have letters or numbers"))

    checkList = ["autoAnalyze","preAnalysis"]
    checkWord = ["false", "f", "no", "n", "none", "0" , "true", "t", "yes", "y", "1", "" ]
    for check in checkList:
        if check in plan:
            if not plan[check] in checkWord:
                failed.append((check, 'Must contain only one of the values for false ("false", "f", "no", "n", "none", "0" ) or for true ("true", "t", "yes", "y", "1", "") '))

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

    #library: items in the list
    libraries = models.ReferenceGenome.objects.all().filter(index_version=settings.TMAP_VERSION).order_by('pk')
    libraries = [l.short_name for l in libraries]
    if 'library' in plan:
        if plan["library"]:
            if not plan["library"] in libraries:
                failed.append(("library", "Must be a valid genome reference library short form name. (also known as reference sequence)"))


    return failed

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
                          "bedfile", "regionfile", "notes"
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

                #don't save if the cycles is blank
                if key == 'flows' and value == '':
                    pass
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


    return shortcuts.render_to_response("rundb/ion_addplans.html")

def get_plan_data():

    runTypes = models.RunType.objects.all().order_by("id")
    barcodes = models.dnaBarcode.objects.values('name').distinct().order_by('name')
    references = models.ReferenceGenome.objects.all().filter(index_version=settings.TMAP_VERSION).order_by('pk')

    allFiles = models.Content.objects.filter(publisher__name="BED",path__contains="/unmerged/detail/")
    bedFiles, hotspotFiles = [], []
    for file in allFiles:
        if file.meta.get("hotspot", False):
            hotspotFiles.append(file)
        else:
            bedFiles.append(file)

    seqKits = models.SequencingKit.objects.all()
    libKits = models.LibraryKit.objects.all()

    variantfrequencies = models.VariantFrequencies.objects.all().order_by("name")

    return runTypes, barcodes, references, bedFiles, hotspotFiles, seqKits, libKits, variantfrequencies

def add_plan(request):
    """add one plan"""

    runTypes, barcodes, references, bedFiles, hotspotFiles, seqKits, libKits, variantfrequencies = get_plan_data()

    ctxd = {"bedFiles":bedFiles, "hotspotFiles" : hotspotFiles,
            "libKits": libKits, "seqKits" : seqKits,
            "variantfrequencies": variantfrequencies,
            "runTypes":runTypes, "barcodes" : barcodes, "references" : references}

    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_addeditplan.html",
                                        context_instance=context)

def edit_plan(request, id):
    """
    Simple view to display the edit barcode page
    """
    runTypes, barcodes, references, bedFiles, hotspotFiles, seqKits, libKits, variantfrequencies = get_plan_data()

    #get the existing plan
    plan = shortcuts.get_object_or_404(models.PlannedExperiment,pk=id)

    ctxd = {"plan":plan,
            "bedFiles":bedFiles, "hotspotFiles" : hotspotFiles,
            "libKits": libKits, "seqKits" : seqKits,
            "variantfrequencies": variantfrequencies,
            "runTypes":runTypes, "barcodes" : barcodes, "references" : references}

    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_addeditplan.html",
                                        context_instance=context)
def edit_experiment(request, id):
    """
    Simple view to display the edit barcode page
    """

    #get the info to populate the page
    runTypes, barcodes, references, bedFiles, hotspotFiles, seqKits, libKits, variantfrequencies = get_plan_data()

    #get the existing plan
    exp = shortcuts.get_object_or_404(models.Experiment,pk=id)


    selectedReference = exp.library

    ctxd = {"exp":exp,
            "bedFiles":bedFiles, "hotspotFiles" : hotspotFiles,
            "libKits": libKits, "seqKits" : seqKits,
            "variantfrequencies": variantfrequencies,
            "runTypes":runTypes, "barcodes" : barcodes,
            "references" : references,
            "selectedReference" : selectedReference }

    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_editexp.html",
                                        context_instance=context)

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


def test_task(request, argument):
    from django.http import HttpResponse
    #async = tasks.test_task.delay(argument)
    async = tasks.downloadGenome.delay(argument, "genome")
    #result = async.get()
    result = "waiting"
    task_id = async.task_id
    state = async.state
    html = """<html><body>Your argument was %s.<br />
    Your Celery task ID is %s<br />
    Your Celery task state is %s
    </body></html>
    """ % (result, task_id, state)
    return HttpResponse(html)
