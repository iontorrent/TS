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


def wildcardify(terms):
    """Convert a wildcard search string into a regular expression.
    First, escape special regexp characters. Replace all whitespace with '\\s*'.
    Then replace escaped Kleene stars with '.*'. Parentheses are left as-is.

    >>> wildcardify("anything after this*")
    "anything\\\\s*after\\\\s*this.*"
    """
    
    def escape_re(match):
        """Replace each match with the match escaped.
    
        >>> escape_re(re.match('^(match me)$','match me'))
        '\\\\match me'
        """
        return "\\%s" % match.group(0)
    
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
        
    bk = models.dm_reports.get()
    if bk.exists():
        bkExist = "True"
        if bk.location != (None or ''):
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
    from iondb.backup import makePDF
    pkR = pkR[:len(pkR)-4]
    return http.HttpResponse(makePDF.getPDF(pkR), mimetype="application/pdf")

@login_required
def PDFGenOld(request, pkR):
    from iondb.backup import makePDF
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



@login_required
def add_plans(request):
    """Take care of adding plans.  The POST method will do the CSV parsing and database additions"""
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


def pretty(d, indent=0):
    if d:
        for key, value in d.iteritems():
            logger.debug('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                logger.debug('\t' * (indent+1) + str(value))


def get_default_cmdline_args(chipType):
    chips = models.Chip.objects.all()
    for c in chips:
        if chipType.startswith(c.name):
            analysisArgs    = c.analysisargs
            basecallerArgs  = c.basecallerargs
            beadfindArgs    = c.beadfindargs
            thumbnailAnalysisArgs   = c.thumbnailanalysisargs
            thumbnailBasecallerArgs = c.thumbnailbasecallerargs
            thumbnailBeadfindArgs   = c.thumbnailbeadfindargs
            break
    # loop finished without finding chipType: provide basic defaults
    else:        
        analysisArgs = thumbnailAnalysisArgs = 'Analysis'
        basecallerArgs = thumbnailBasecallerArgs = 'BaseCaller'        
        beadfindArgs = thumbnailBeadfindArgs = 'justBeadFind'
    
    args = {
      'analysisArgs':   analysisArgs,      
      'basecallerArgs': basecallerArgs,
      'beadfindArgs':   beadfindArgs,
      'thumbnailAnalysisArgs':    thumbnailAnalysisArgs,
      'thumbnailBasecallerArgs':  thumbnailBasecallerArgs,
      'thumbnailBeadfindArgs':    thumbnailBeadfindArgs      
    }
    return args

