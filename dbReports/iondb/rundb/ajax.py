# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

"""
AJAX Views And API
==================

The :mod:`rundb.ajax` module provides views intended for use with 
asynchronous HTTP requests. However, rather than returning XML,
these views return JSON. This ends up being simpler to work with both
on the client and server sides.

The module also provides a "JSON API" - an easily extensible, super-simple
remote procedure call framework.
"""

# python standard lib
import json
from os import path
import re
import urlparse
import logging
from django.core.urlresolvers import reverse

# django
from django import http,shortcuts
from django.conf import settings
from django.core import serializers
from django.core.serializers.json import DjangoJSONEncoder
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache

# local
import models
from iondb.anaserve import  client as anaclient

_http_re = re.compile(r'^HTTP/1\.\d\s+(\d{3}) [a-zA-Z0-9_ ]+$')
_http404 = http.HttpResponseNotFound

_charset = 'utf-8'

_API_METHODS = {}


logger = logging.getLogger(__name__)


def jsonapi(fn):
    """Decorator to convert a normal Python function into a function
    callable through the JSON API.
    
    >>> @jsonapi
    ... def jsapi_strip(a): return render_to_json(a.strip())
    # jsapi_strip can now be called through the JSON API
    """
    argnames = set(fn.func_code.co_varnames[:fn.func_code.co_argcount])
    def ret(js):
        for k in js:
            if k not in argnames:
                raise ValueError("Invalid argument: %s" % k)
        argd = dict((str(k),v) for k,v in js.iteritems())
        return fn(**argd)
    ret.func_name = fn.func_name
    _API_METHODS[ret.func_name] = ret
    return ret
    

def enc(s):
    """UTF-8 encode a string."""
    return s.encode('utf-8')

def render_to_json(data,is_json=False):
    """Create a JSON response from a data dictionary and return a
    Django response object."""
    if not is_json:
        js = json.dumps(data, cls=DjangoJSONEncoder)
    else:
        js = data
    mime = mimetype="application/json;charset=utf-8"
    response = http.HttpResponse(enc(js), content_type=mime)
    return response

@login_required
def analysis_liveness(request, pk):
    """Determine if an analysis has been successfully started.
    """
    try:
        pk = int(pk)
    except (TypeError,ValueError):
        return http.HttpResponseNotFound

    result = shortcuts.get_object_or_404(models.Results, pk=pk)
    if settings.TEST_INSTALL:
        rl = path.join(path.dirname(result.reportLink), "ion_params_00.json")
    else:
        rl = result.reportLink
        
    url = urlparse.urlparse(rl)
    loc = result.server_and_location()

    web_path = result.web_path(loc)
#    report = result.reportLink
    report = "/report/%s" % pk
    save_path = result.web_root_path(loc)

    if web_path:
        log = reverse("report_log", args=(pk,))
        report_exists = result.report_exist()
    else:
        log = False
        report_exists = False

    ip = "127.0.0.1"
    if ip is None:
        return http.HttpResponseNotFound("no job server found for %s"
                                         % loc.name)
    proxy = anaclient.connect(ip,10000)
    #if the job is finished, it will be shown as 'failed'
    success,status = proxy.status(save_path, result.pk)
    return render_to_json({"success":success, "log":log, "report":report, "exists" : report_exists,
                           "status":status})

@login_required
def starRun(request, pk, set):
    """Allow user to 'star' a run thus making it magic"""
    try:
        pk = int(pk)
    except (TypeError,ValueError):
        return http.HttpResponseNotFound

    exp = shortcuts.get_object_or_404(models.Experiment, pk=pk)
    exp.star = bool(int(set))
    exp.save()
    return http.HttpResponse()

@login_required
def change_library(request, pk):
    """changes the library for a run """
    if request.method == 'POST':
        try:
            pk = int(pk)
        except (TypeError, ValueError):
            return http.HttpResponseNotFound

        exp = shortcuts.get_object_or_404(models.Experiment, pk=pk)

        log = exp.log
        lib = request.POST.get('lib',"none")
        log["library"] = str(lib)
        exp.log = log
        exp.library = lib
        exp.save()
        
        return http.HttpResponse()

    elif request.method == 'GET':
        return http.HttpResponse()


@login_required
def delete_barcode(request, pk):
    """delete a barcode"""
    if request.method == 'POST':
        try:
            pk = int(pk)
        except (TypeError, ValueError):
            return http.HttpResponseNotFound

        barcode = shortcuts.get_object_or_404(models.dnaBarcode, pk=pk)
        barcode.delete()

        return http.HttpResponse()

    elif request.method == 'GET':
        return http.HttpResponse()

@login_required
def delete_barcode_set(request, name):
    """delete a set of barcodes"""
    if request.method == 'POST':

        barcode = shortcuts.get_list_or_404(models.dnaBarcode, name=name)
        for code in barcode:
            code.delete()

        return http.HttpResponse()

    elif request.method == 'GET':
        return http.HttpResponse()

@never_cache
@login_required
def progress_bar(request,pk):
    '''gets the current status from the current experiment table
    based on the experiment pk and updates the ftpStatus progress bar'''
    try:
        pk = int(pk)
    except (TypeError,ValueError):
        return http.HttpResponseNotFound
    exp = shortcuts.get_object_or_404(models.Experiment, pk=pk)
    try:
        value = int(exp.ftpStatus)
    except ValueError:
        if exp.ftpStatus == "Complete":
            value = 100
        else:
            value = 0
    return render_to_json({"value":value})

@never_cache
@login_required
def progressbox(request,pk):
    '''reads in the progress.txt file inside a running analysis
    and returns the colors of the boxes for the faux progress bar
    for the running analysis'''
    try:
        pk = int(pk)
    except (TypeError,ValueError):
        return http.HttpResponseNotFound
    res = shortcuts.get_object_or_404(models.Results, pk=pk)

    prefix = res.get_report_path()
    prefix = path.split(prefix)[0]
    prefix = path.join(prefix,'progress.txt')
    try:
        f = open(prefix,'r')
        data = f.readlines()
        f.close()
    except IOError:
        # Empty response, leaves progress grey
        return render_to_json({"value":{}, "status": res.status })
    ret = {}


    for line in data:
        line = line.strip().split('=')
        key = line[0].strip()
        value = line[-1].strip()
        ret[key]=value

    return render_to_json({"value":ret, "status": res.status})


# JSON API stuff
@login_required
def apibase(request):
    """Make a call to the JSON API.

    Calls are made by passing in data through the ``request``'s GET or
    POST data. The calling convention is as follows:

    #. :func:`apibase` extracts two parameters from the request data:
      
      #. ``methodname``: The name of the method, as it appears in this file.
      #. ``params``: A string of JSON-encoded keyword arguments to the method.

    #. The ``methodname`` is limited to functions decorated with the
       :func:`jsonapi` decorator. If :func:`apibase` cannot resolve the
       ``methodname`` to one of these functions, it returns and HTTP 404 error.
    #. The function then decodes the JSON-encoded ``params`` string,
       and checks (to a limited degree) that the parameters match the 
       signature for ``methodname``.
    #. Finally, :func:`jsonapi` calls the function designated by ``methodname``
       with the decoded parameters and returns the result of the call.

    """
    if request.method == "GET":
        d = request.GET
    else:
        d = request.POST
    if "methodname" not in d or "params" not in d:
        return _http404("JSON API call requires both 'methodname' and 'params'")
    methodname = d["methodname"]
    try:
        params = json.loads(d["params"])
    except ValueError:
        return _http404("Invalid JSON encoding for params.")
    if methodname not in _API_METHODS:
        return _http404("Method does not exist.")
    f = _API_METHODS[methodname]
    return f(params)
        
@jsonapi
def report_info(pk):
    """Serialize the ``models.Results`` object denoted by ``pk`` and
    return the resulting JSON string.
    """
    pk = int(pk)
    report = shortcuts.get_object_or_404(models.Results,pk=pk)
    expr = report.experiment
    serialize = lambda x: serializers.serialize("json",x)
    ret = [report,expr]
    ret.extend(report.tfmetrics_set.all())
    return render_to_json(serialize(ret), True)

#@jsonapi
#def locate_report(pk):
#    """Return the file path to the working directory of the ``models.Results``
#   object denoted by ``pk``."""
#   pk = int(pk)
#   report = shortcuts.get_object_or_404(models.Results,pk=pk)
#    server,loc = report.server_and_location()
#    ret = report.web_root_path(server,loc)
#    return render_to_json(ret)

@jsonapi
def last_report():
    """Return the primary key of the ``models.Results`` object more recently
    created."""
    try:
        ret = models.Results.objects.latest('timeStamp').pk
    except models.Results.DoesNotExist:
        ret = None
    return render_to_json(ret)

