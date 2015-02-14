# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import xmlrpclib
import subprocess
import socket
import logging
import os
import string
import json
import traceback
import stat
import tempfile
import csv
import httplib2
import urlparse
import time
import datetime
from django.utils import timezone
from pprint import pformat
from celery.task import task
import celery.exceptions
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.shortcuts import render_to_response, get_object_or_404, get_list_or_404
from django.template import RequestContext, Context
from django.http import Http404, HttpResponsePermanentRedirect, HttpResponse, HttpResponseRedirect, \
    HttpResponseNotFound, HttpResponseBadRequest, HttpResponseNotAllowed, HttpResponseServerError
from django.core.servers.basehttp import FileWrapper
from django.core.urlresolvers import reverse
from django.forms.models import model_to_dict
import ion.utils.TSversion
from iondb.rundb.forms import EmailAddress as EmailAddressForm, UserProfileForm
from iondb.rundb.forms import AmpliseqLogin
from iondb.rundb import tasks, publishers
from iondb.anaserve import client
from iondb.plugins.manager import pluginmanager
from django.contrib.auth.models import User
from iondb.rundb.models import dnaBarcode, Plugin, PluginResult, GlobalConfig,\
    EmailAddress, Publisher, Location, Experiment, Results, Template, UserProfile, \
    FileMonitor, EventLog, ContentType, Cruncher
from iondb.rundb.data import rawDataStorageReport, reportLogStorage
from iondb.rundb.configure.genomes import search_for_genomes
from iondb.rundb.plan import ampliseq
# Handles serialization of decimal and datetime objects
from django.core.serializers.json import DjangoJSONEncoder
from iondb.rundb.configure.util import plupload_file_upload
from ion.utils import makeCSA
from django.core import urlresolvers
from iondb.utils.raid import get_raid_status, get_raid_status_json, load_raid_status_json
from iondb.servelocation import serve_wsgi_location
from ion.plugin.remote import call_pluginStatus
logger = logging.getLogger(__name__)

from json import encoder
encoder.FLOAT_REPR = lambda x: format(x, '.15g')

@login_required
def configure(request):
    '''Wrapper'''
    return configure_about(request)


@login_required
def configure_about(request):
    '''
    Populate about page values
    CLI to get TS version:
    python -c "from ion.utils import TSversion; _, version = TSversion.findVersions(); print version"
    '''
    reload(ion.utils.TSversion)
    versions, meta = ion.utils.TSversion.findVersions()
    osversion = ion.utils.TSversion.findOSversion()
    ctxd = {"versions": versions, "meta": meta, "osversion": osversion}
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/about.html", context_instance = ctx)


@login_required
def configure_ionreporter(request):
    iru = Plugin.objects.get(name__iexact='IonReporterUploader', active=True)
    ctxd = { "iru" : iru }

    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/ionreporter.html", context_instance=ctx)


def timeout_raid_info():
    async_result = tasks.get_raid_stats.delay()
    try:
        raidinfo = async_result.get(timeout=30)
        if async_result.failed():
            raidinfo = None
    except celery.exceptions.TimeoutError as err:
        logger.warning("RAID status check timed out, taking longer than 30 seconds.")
        raidinfo = None
    return raidinfo


def timeout_raid_info_json():
    async_result = tasks.get_raid_stats_json.delay()
    err_msg = None
    try:
        raidinfo = async_result.get(timeout=30)
        if async_result.failed():
            err_msg = "RAID status check failed."
            logger.error(err_msg)
            raidinfo = None
    except celery.exceptions.TimeoutError as err:
        err_msg = "RAID status check timed out, taking longer than 30 seconds."
        logger.warning(err_msg)
        raidinfo = None
    return raidinfo, err_msg

def sort_drive_array_for_display(raidstatus):
    # sorts array to be displayed matching physical arrangement in enclosure
    slots=12
    for d in raidstatus:
        try:
            adapter_id = d.get('adapter_id','')
            drives = d.get('drives',[])
            if not adapter_id or len(drives) < slots:
                continue

            if adapter_id.startswith("PERC H710"):
                d['cols'] = 4
            elif adapter_id.startswith("PERC H810"):
                ncols = 4
                temp = [drives[i:i+(slots/ncols)] for i in range(0, slots, slots/ncols)]
                d['drives'] = sum( map(list, zip(*temp)), [])
                d['cols'] = ncols
        except:
            pass

@login_required
def configure_services(request):

    servers = get_servers()
    jobs = current_jobs() + current_plugin_jobs()
    crawler = _crawler_status(request)
    processes = process_set()

    # RAID Info
    raidinfo, raid_err_msg = timeout_raid_info_json()
    raid_status_updated = datetime.datetime.now()

    raid_status = None
    if raidinfo:
        raid_status = get_raid_status(raidinfo)
    elif raid_err_msg:
        # attempt to load previously generated raidstatus file
        contents = load_raid_status_json()
        raid_status = contents.get('raid_status')
        raid_status_updated = contents.get('date')

    if raid_status:
        sort_drive_array_for_display(raid_status)

    ctxd = {
        "processes": processes,
        "servers": servers,
        "jobs": jobs,
        "crawler": crawler,
        "raid_status": raid_status,
        "raid_status_updated": raid_status_updated,
        "raid_err_msg": raid_err_msg,
        'crunchers': Cruncher.objects.all().order_by('name'),
        }
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/services.html", context_instance=ctx)


@login_required
def configure_references(request):
    search_for_genomes()
    ctx = RequestContext(request)
    return render_to_response("rundb/configure/references.html", context_instance=ctx)


@login_required
def configure_plugins(request):
    # Rescan Plugins
    ## Find new, remove missing plugins
    pluginmanager.rescan()
    ctx = RequestContext(request, {})
    config_publishers(request, ctx)
    return render_to_response("rundb/configure/plugins.html", context_instance=ctx)


@login_required
def configure_plugins_plugin_install(request):
    ctx = RequestContext(request, {})
    return render_to_response("rundb/configure/modal_configure_plugins_plugin_install.html", context_instance=ctx)


@login_required
def configure_plugins_plugin_configure(request, action, pk):
    """
    load files into iframes
    """

    def openfile(fname):
        """strip lines """
        try:
            f = open(fname, 'r')
        except:
            logger.exception("Failed to open '%s'", fname)
            return False
        content = f.read()
        f.close()
        return content

    # Used in javascript, must serialize to json
    plugin = get_object_or_404(Plugin, pk=pk)
    #make json to send to the template
    plugin_json = json.dumps({'pk': pk, 'model': str(plugin._meta), 'fields': model_to_dict(plugin)}, cls=DjangoJSONEncoder)

    # If you set more than one of these,
    # behavior is undefined. (one returned randomly)
    dispatch_table = {
        'report': 'instance.html',
        'config': 'config.html',
        'about': 'about.html',
        'plan': 'plan.html'
    }

    fname = os.path.join(plugin.path, dispatch_table[action])

    content = openfile(fname)
    if not content:
        raise Http404()

    index_version = settings.TMAP_VERSION

    report = request.GET.get('report', False)
    results_json = {}
    if report:
        # Used in javascript, must serialize to json
        results_obj = get_object_or_404(Results, pk=report)
        #make json to send to the template
        results_json = json.dumps({'pk': report, 'model': str(results_obj._meta), 'fields': model_to_dict(results_obj)}, cls=DjangoJSONEncoder)

    ctxd = {"plugin": plugin_json, "file": content, "report": report, "tmap": str(index_version),
            "results" : results_json, "action" : action }

    context = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/modal_configure_plugins_plugin_configure.html", context_instance=context)


@login_required
def configure_plugins_pluginmedia(request, action, pk, path):
    # note action is ignored, just serve pluginmedia on relative path

    plugin = get_object_or_404(Plugin, pk=pk)
    fspath = plugin.path

    # Map given relative URL to fspath, then back to transformed URL
    prefix = "/results/plugins/"
    if not fspath.startswith(prefix):
        raise Http404("Invalid Plugin path '%s'", fspath)
    if not os.path.exists(os.path.join(fspath, "pluginMedia")):
        raise Http404("Request for pluginMedia for plugin with no pluginMedia folder")

    # Security Note: May allows access to any plugin file with ".."

    # Strip prefix from filesystem path, replace with "/pluginMedia/"
    servepath="plugins/" + fspath[len(prefix):] + "/pluginMedia/" + path
    # example: "/plugins/" + "examplePlugin" + "/" + "pluginMedia/img.png"
    # example: "/plugins/" + "instances/12345667/exampleZeroConf" + "/" + "pluginMedia/img.png"
    logger.debug("Redirecting pluginMedia request '%s' to '%s'", path, servepath)

    # Serve redirects to /private/plugins/,
    # which is mapped to /results/plugins/ in apache
    return serve_wsgi_location(request, servepath)


@login_required
def configure_plugins_plugin_uninstall(request, pk):
    #TODO: See about pulling this out into a common methods
    _type = 'plugin'
    plugin = get_object_or_404(Plugin, pk=pk)
    type = "Plugin"
    action = reverse('api_dispatch_uninstall', kwargs={'api_name': 'v1', 'resource_name': 'plugin', 'pk': pk})

    ctx = RequestContext(request, {
        "id": pk, "method": "DELETE", 'methodDescription': 'Delete', "readonly": False, 'type': type, 'action': action, 'plugin': plugin
    })
    return render_to_response("rundb/configure/modal_confirm_plugin_uninstall.html", context_instance=ctx)


@login_required
def configure_plugins_plugin_zip_upload(request):
    return plupload_file_upload(request, "/results/plugins/scratch/")

@login_required
def configure_plugins_plugin_enable(request, pk, set):
    """Allow user to enable a plugin for use in the analysis"""
    try:
        pk = int(pk)
    except (TypeError,ValueError):
        return HttpResponseNotFound

    plugin = get_object_or_404(Plugin, pk=pk)
    plugin.selected = bool(int(set))
    plugin.save()
    return HttpResponse()

@login_required
def configure_plugins_plugin_autorun(request, pk):
    """
    toogle autorun for a plugin
    """
    if request.method == 'POST':
        try:
            pk = int(pk)
        except (TypeError, ValueError):
            return HttpResponseNotFound

        plugin = get_object_or_404(Plugin, pk=pk)

        # Ignore request if marked AUTORUNDISABLE
        if not plugin.autorunMutable:
            # aka autorun disable, so make sure it is off
            if plugin.autorun:
                plugin.autorun = False
                plugin.save()
            return HttpResponse() # NotModified, Invalid, Conflict?

        checked = request.POST.get('checked',"false").lower()

        if checked == "false":
            plugin.autorun = False
        if checked == "true":
            plugin.autorun = True

        plugin.save()

        return HttpResponse()
    else:
        return HttpResponseNotAllowed(['POST'])

@login_required
def configure_plugins_plugin_refresh(request, pk):
    plugin = get_object_or_404(Plugin, pk=pk)
    url = reverse('api_dispatch_info', kwargs={'resource_name': 'plugin', 'api_name': 'v1', 'pk': int(pk)})
    url += '?use_cache=false'
    ctx = RequestContext(request, {'plugin':plugin, 'action':url, 'method':'get'})
    return render_to_response("rundb/configure/plugins/modal_refresh.html", context_instance=ctx)

@login_required
def configure_plugins_plugin_usage(request, pk):
    plugin = get_object_or_404(Plugin, pk=pk)
    pluginresults = plugin.pluginresult_set.filter(endtime__isnull=False)
    ctx = RequestContext(request, {'plugin':plugin, 'pluginresults': pluginresults})
    return render_to_response("rundb/configure/plugins/plugin_usage.html", context_instance=ctx)

@login_required
def configure_configure(request):
    ctx = RequestContext(request, {})
    emails = EmailAddress.objects.all().order_by('pk')
    enable_nightly = GlobalConfig.get().enable_nightly_email
    ctx.update({"email": emails, "enable_nightly": enable_nightly})
    config_contacts(request, ctx)
    config_site_name(request, ctx)
    return render_to_response("rundb/configure/configure.html", context_instance=ctx)


@login_required
def config_publishers(request, ctx):
    globalconfig = GlobalConfig.get()
    # Rescan Publishers
    publishers.purge_publishers()
    publishers.search_for_publishers(globalconfig)
    pubs = Publisher.objects.all().order_by('name')
    ctx.update({"publishers": pubs})


def _crawler_status(request):
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
        exp_errors = cstat.exp_errors()
        for err in exp_errors:
            err[0] = datetime.datetime.strptime(str(err[0]), "%Y%m%dT%H:%M:%S").replace(tzinfo=timezone.utc)

        ret = {
            'elapsed': seconds2htime(raw_elapsed),
            'hostname': cstat.hostname(),
            'state': cstat.state(),
            'errors': exp_errors
        }
    except socket.error:
        ret = {}

    return ret


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
    return {"days": days, "hours": hours, "minutes": minutes, "seconds": s}

def get_servers():
    jservers = [(socket.gethostname(), socket.gethostbyname(socket.gethostname()))]
    servers = []
    for server_name, ip in jservers:
        try:
            conn = client.connect(ip, settings.JOBSERVER_PORT)
            nrunning = conn.n_running()
            uptime = seconds2htime(conn.uptime())
            servers.append((server_name, ip, True, nrunning, uptime,))
        except (socket.error, xmlrpclib.Fault):
            servers.append((server_name, ip, False, 0, 0,))
    return servers

def current_jobs():
    """
    Get list of running jobs from job server
    """
    jobs = []
    try:
        host = "127.0.0.1"
        conn = client.connect(host, settings.JOBSERVER_PORT)
        running = conn.running()
        runs = dict((r[2], r) for r in running)
        
        results = Results.objects.filter(pk__in=runs.keys()).order_by('pk')
        for result in results:
            name, pid, pk, atype, stat = runs[result.pk]
            jobs.append({
                'name': name,
                'resultsName': result.resultsName,
                'pid': pid,
                'type': 'analysis',
                'status': stat,
                'pk': result.pk,
                'report_exist': result.report_exist(),
                'report_url': reverse('report', args=(pk,)),
                'term_url': reverse('control_job', args=(pk,'term'))
            })
    except (socket.error, xmlrpclib.Fault):
        pass

    return jobs

def current_plugin_jobs():
    """
    Get list of active pluginresults from database then connect to ionPlugin and get drmaa status per jobid.
    """
    jobs = []
    running = PluginResult.objects.filter(state__in=PluginResult.RUNNING_STATES).order_by('pk')
    if running:
        # get job status from drmaa
        jobids = running.values_list('jobid', flat=True)
        try:
            job_status = call_pluginStatus(list(jobids))
            for i, pr in enumerate(list(running)):
                if job_status[i] != 'DRMAA BUG':
                    jobs.append({
                        'name': pr.plugin.name,
                        'resultsName': pr.result.resultsName,
                        'pid': pr.jobid,
                        'type': 'plugin',
                        'status': job_status[i],
                        'pk': pr.pk,
                        'report_exist': True,
                        'report_url': reverse('report', args=(pr.result.pk,)),
                        'term_url': "/rundb/api/v1/pluginresult/%d/control/" % pr.pk
                    })
        except:
            pass
            
    return jobs

def process_set():
    def process_status(process):
        return subprocess.Popen("service %s status" % process,
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processes = [
        "ionJobServer",
        "ionCrawler",
        "ionPlugin",
        "RSM_Launch",
        "dhcp3-server",
        "ntp"
    ]
    proc_set = dict((p, process_status(p)) for p in processes)
    for name, proc in proc_set.items():
        stdout, stderr = proc.communicate()
        proc_set[name] = proc.returncode == 0
        logger.debug("%s out = '%s' err = %s''" % (name, stdout, stderr))
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
    proc_set["RabbitMQ"] = complicated_status("/var/run/rabbitmq/pid", int)

    for node in ['celerybeat', 'celery_w1', 'celery_plugins', 'celery_periodic', 'celery_slowlane', 'celery_transfer', 'celery_diskutil']:
        proc_set[node] = complicated_status("/var/run/celery/%s.pid" % node, int)
    return sorted(proc_set.items())


@login_required
def references_TF_edit(request, pk=None):

    if pk:
        tf = get_object_or_404(Template, pk=pk)
        ctx = RequestContext(request, {
            'id': pk, 'method': 'PUT', 'methodDescription': 'Edit', 'readonly': False, 'action': reverse('api_dispatch_detail', kwargs={'resource_name': 'template', 'api_name': 'v1', 'pk': int(pk)}), 'tf': tf
        })
    else:
        ctx = RequestContext(request, {
            'id': pk, 'method': 'POST', 'methodDescription': 'Add', 'readonly': False, 'action': reverse('api_dispatch_list', kwargs={'resource_name': 'template', 'api_name': 'v1'})
        })
    return render_to_response("rundb/configure/modal_references_edit_TF.html", context_instance=ctx)


@login_required
def references_TF_delete(request, pk):
    tf = get_object_or_404(Template, pk=pk)
    _type = 'TestFragment'
    ctx = RequestContext(request, {
        "id": pk, "ids": json.dumps([pk]), "names": tf.name, "method": "DELETE", 'methodDescription': 'Delete', "readonly": False, 'type': _type, 'action': reverse('api_dispatch_detail', kwargs={'resource_name': 'template', 'api_name': 'v1', 'pk': int(pk)}), 'actions': json.dumps([])
    })
    return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)


@login_required
def references_barcodeset(request, barCodeSetId):
    barCodeSetName = get_object_or_404(dnaBarcode, pk=barCodeSetId).name
    ctx = RequestContext(request, {'name': barCodeSetName, 'barCodeSetId': barCodeSetId})
    return render_to_response("rundb/configure/references_barcodeset.html", context_instance=ctx)


@login_required
def references_barcodeset_add(request):
    if request.method == 'GET':
        ctx = RequestContext(request, {})
        return render_to_response("rundb/configure/modal_references_add_barcodeset.html", context_instance=ctx)
    elif request.method == 'POST':
        return _add_barcode(request)


def _add_barcode(request):
    """add the barcodes, with CSV validation"""

    if request.method == 'POST':
        name = request.POST.get('name', '')
        postedfile = request.FILES['postedfile']

        barCodeSet = dnaBarcode.objects.filter(name=name)
        if barCodeSet:
            return HttpResponse(json.dumps({"status": "Error: Barcode set with the same name already exists"}), mimetype="text/html")

        expectedHeader = ["id_str", "type", "sequence", "floworder", "index", "annotation", "adapter"]

        barCodes = []
        failed = {}
        nucs = ["sequence", "floworder", "adapter"]  # fields that have to be uppercase
        reader = csv.DictReader(postedfile.read().splitlines())
        for index, row in enumerate(reader, start=1):
            invalid = _validate_barcode(row)
            if invalid:  # don't make dna object or add it to the list
                failed[index] = invalid
                continue
            newBarcode = dnaBarcode(name=name, index=index)
            for key in expectedHeader:  # set the values for the objects
                value = row.get(key, None)
                if value:
                    value = value.strip()  # strip the strings
                    if key in nucs:  # uppercase if a nuc
                        value = value.upper()
                    setattr(newBarcode, key, value)
            if not newBarcode.id_str:  # make a id_str if one is not provided
                newBarcode.id_str = str(name) + "_" + str(index)
            newBarcode.length = len(newBarcode.sequence)  # now set a default
            barCodes.append(newBarcode)  # append to our list for later saving

        if failed:
            r = {"status": "Barcodes validation failed. The barcode set has not been saved.", "failed": failed}
            return HttpResponse(json.dumps(r), mimetype="text/html")
        if not barCodes:
            return HttpResponse(json.dumps({"status": "Error: There must be at least one barcode! Please reload the page and try again with more barcodes."}), mimetype="text/html")
        usedID = []
        for barCode in barCodes:
            if barCode.id_str not in usedID:
                usedID.append(barCode.id_str)
            else:
                error = {"status": "Duplicate id_str for barcodes named: " + str(barCode.id_str) + "."}
                return HttpResponse(json.dumps(error), mimetype="text/html")
        usedIndex = []
        for barCode in barCodes:
            if barCode.index not in usedIndex:
                usedIndex.append(barCode.index)
            else:
                error = {"status": "Duplicate index: " + barCode.index + "."}
                return HttpResponse(json.dumps(error), mimetype="text/html")

        #saving to db needs to be the last thing to happen
        for barCode in barCodes:
            try:
                barCode.save()
            except:
                logger.exception("Error saving barcode to database")
                return HttpResponse(json.dumps({"status": "Error saving barcode to database!"}), mimetype="text/html")
        r = {"status": "Barcodes Uploaded! The barcode set will be listed on the references page.", "failed": failed, 'success': True}
        return HttpResponse(json.dumps(r), mimetype="text/html")


def _validate_barcode(barCodeDict):
    """validate the barcode, return what failed"""
    failed = []
    requiredList = ["sequence"]

    for req in requiredList:
        if req in barCodeDict:
            if not barCodeDict[req]:
                failed.append((req, "Required column is empty"))
        else:
            failed.append((req, "Required column is missing"))
    nucOnly = ["sequence", "floworder", "adapter"]
    for nuc in nucOnly:
        if nuc in barCodeDict:
            if not set(barCodeDict[nuc].upper()).issubset("ATCG"):
                failed.append((nuc, "Must have A, T, C, G only: '%s'" % barCodeDict[nuc]))
    if 'id_str' in barCodeDict:
        if not set(barCodeDict["id_str"]).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"):
            failed.append(("id_str", "str_id must only have letters, numbers, or the characters _ . - "))
    #do not let the index be set to zero. Zero is reserved.
    if 'index' in barCodeDict:
        if barCodeDict["index"] == "0":
            failed.append(("index", "index must not contain a 0. Indices should start at 1."))
    return failed


@login_required
def references_barcodeset_delete(request, barCodeSetId):
    barCodeSetName = get_object_or_404(dnaBarcode, pk=barCodeSetId).name
    """delete a set of barcodes"""
    if request.method == 'POST':
        dnaBarcode.objects.filter(name=barCodeSetName).delete()
        return HttpResponse()
    elif request.method == 'GET':
        #TODO: See about pulling this out into a common methods
        _type = 'dnabarcode'
        type = "Barcode Set"
        pks = []
        actions = []
        ctx = RequestContext(request, {
            "id": barCodeSetName, "ids": json.dumps(pks), "method": "POST", 'methodDescription': 'Delete', "readonly": False, 'type': type, 'action': reverse('references_barcodeset_delete', args=[barCodeSetId, ]), 'actions': json.dumps(actions)
        })
        return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)


@login_required
def references_barcode_add(request, barCodeSetId):
    return references_barcode_edit(request, barCodeSetId, None)


@login_required
def references_barcode_edit(request, barCodeSetId, pk):
    dna = get_object_or_404(dnaBarcode, pk=barCodeSetId)
    barCodeSetName = dna.name

    def nextIndex(name):
        barCodeSetName = dnaBarcode.objects.filter(name=name).order_by("-index")
        if barCodeSetName:
            return barCodeSetName[0].index + 1
        else:
            return 1
    #if there is a barcode do a look up for it
    if pk:
        barcode = dnaBarcode.objects.get(pk=int(pk))
        index = barcode.index
        #get a list of all the other barcodes minus this one
        others = dnaBarcode.objects.filter(name=barCodeSetName)
        others = others.exclude(pk=int(pk))
    else:
        barcode = False
        index = nextIndex(barCodeSetName)
        #get a list of all the other barcodes
        others = dnaBarcode.objects.filter(name=barCodeSetName)

    otherList = []
    for other in others:
        otherList.append(other.id_str)

    ctxd = {"barcode": barcode, "barCodeSetName": barCodeSetName, "index": index, "otherList": json.dumps(otherList)}
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/modal_references_addedit_barcode.html", context_instance=ctx)


@login_required
def references_barcode_delete(request, barCodeSetId, pks):
    #TODO: See about pulling this out into a common methods
    pks = pks.split(',')
    barcodes = get_list_or_404(dnaBarcode, pk__in=pks)
    _type = 'dnabarcode'
    type = "Barcode"
    actions = []
    names = ', '.join([x.id_str for x in barcodes])
    for pk in pks:
        actions.append(reverse('api_dispatch_detail', kwargs={'resource_name': _type, 'api_name': 'v1', 'pk': int(pk)}))

    ctx = RequestContext(request, {
        "id": pks[0], "ids": json.dumps(pks), "names": names, "method": "DELETE", 'methodDescription': 'Delete', "readonly": False, 'type': type, 'action': actions[0], 'actions': json.dumps(actions)
    })
    return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)


@login_required
def control_job(request, pk, signal):
    """Send ``signal`` to the job denoted by ``pk``, where ``signal``
    is one of

    * ``"term"`` - terminate (permanently stop) the job.
    * ``"stop"`` - stop (pause) the job.
    * ``"cont"`` - continue (resume) the job.
    """
    pk = int(pk)
    if signal not in set(("term", "stop", "cont")):
        return HttpResponseNotFound("No such signal")
    result = get_object_or_404(Results, pk=pk)
    loc = result.server_and_location()
    ip = '127.0.0.1'  # assume, webserver and jobserver on same appliance
    conn = client.connect(ip, settings.JOBSERVER_PORT)
    result.status = 'TERMINATED'
    result.save()
    return render_to_json(conn.control_job(pk, signal))


def enc(s):
    """UTF-8 encode a string."""
    return s.encode('utf-8')


def render_to_json(data, is_json=False):
    """Create a JSON response from a data dictionary and return a
    Django response object."""
    if not is_json:
        js = json.dumps(data)
    else:
        js = data
    mime = mimetype = "application/json;charset=utf-8"
    response = HttpResponse(enc(js), content_type=mime)
    return response


@login_required
def configure_system_stats(request):
    ctx = RequestContext(request, {'url': reverse('configure_system_stats_data'), 'type': 'GET'})
    return render_to_response("rundb/configure/configure_system_stats_loading.html", context_instance=ctx)


@login_required
def configure_system_stats_data(request):
    """
    Generates the stats page on system configuration
    """
    # Run a script on the server to generate text
    # If this is change, mirror that change in
    # rundb/configure/configure_system_stats_loading.html
    networkCMD = ["/usr/bin/ion_netinfo"]
    logger.info("Calling netinfo script")
    p = subprocess.Popen(networkCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        stats_network = stdout.splitlines(True)
    else:
        stats_network = []

    statsCMD = ["/usr/bin/ion_sysinfo"]
    logger.info("Calling ion_sysinfo script")
    q = subprocess.Popen(statsCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = q.communicate()
    if q.returncode == 0:
        stats = stdout.splitlines(True)
    else:
        stats = []

    logger.info("Calling (celery) disk_check_status")
    disk_check_status = tasks.disk_check_status.delay().get(timeout=5)

    # MegaCli64 needs root privilege to access RAID controller - we use a celery task
    # to do the work since they execute with root privilege
    logger.info("Calling timeout_raid_info")
    raid_stats = timeout_raid_info()

    logger.info("Calling storage_report")
    stats_dm = rawDataStorageReport.storage_report()

    # Create filename for the report
    reportFileName = "/tmp/stats_sys.txt"

    # Stuff the variable into the context object
    ctx = Context({"stats_network": stats_network,
                   "stats": stats,
                   "stats_dm": stats_dm,
                   "raid_stats": raid_stats,
                   "reportFilePath": reportFileName,
                   "disk_check_status": disk_check_status,
                   })

    # Generate a file from the report
    logger.info("Writing %s" % reportFileName)
    outfile = open(reportFileName, 'w')
    for line in stats_network:
        outfile.write(line)
    for line in stats:
        outfile.write(line)
    for line in stats_dm:
        outfile.write(line)
    for line in raid_stats:
        outfile.write(line)
    outfile.close()
    
    # Set permissions so anyone can read/overwrite/destroy
    try:
        os.chmod(reportFileName,
                 stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
    except:
        logger.exception("Could not chmod '%s'", reportFileName)

    return render_to_response("rundb/configure/configure_system_stats.html", context_instance=ctx)


def raid_info(request, index=0):
    # display RAID info for a drives array from saved file /var/spool/ion/raidstatus.json
    # index is the adapter/enclosure row clicked on services page
    contents = load_raid_status_json()
    try:
        array_status = contents['raid_status'][int(index)]['drives']
    except:
        array_status = []
    return render_to_response("rundb/configure/modal_raid_info.html", {"array_status": array_status})

def config_contacts(request, context):
    """Essentially but not actually a context processor to handle user contact
    information on the global config page.
    """
    updated = False
    contacts = {"lab_contact": None, "it_contact": None}
    for profile in UserProfile.objects.filter(user__username__in=contacts.keys()):
        if request.method == "POST" and str(profile.user) + "-name" in request.POST:
            try:
                profile.name = request.POST.get(str(profile.user) + "-name", "")
                profile.phone_number = request.POST.get(str(profile.user) + "-phone_number", "")
                profile.user.email = request.POST.get(str(profile.user) + "-email", "")
                profile.user.save()
                profile.save()
                updated = True
            except:
                logger.exception("Error while saving contact info for %s" % profile.name)
        else:
            contacts[profile.user.username] = {'name': profile.name,
                                               'phone_number': profile.phone_number,
                                               'email': profile.user.email}
    if updated:
        tasks.contact_info_flyaway.delay()
    context.update({"contacts": contacts})


def config_site_name(request, context):
    """The site name will be automatically loaded on the page, so all we have
    to do here is check whether we should update it, and if so, do so.
    """
    if request.method == "POST" and "site_name" in request.POST:
        config = GlobalConfig.get()
        config.site_name = request.POST["site_name"]
        config.save()
        context.update({"base_site_name": request.POST["site_name"]})


@login_required
def edit_email(request, pk=None):
    if pk is None:
        context = {"name": "Add Email",
                   "method": "POST",
                   "url": "/rundb/api/v1/emailaddress/",
                   "form": EmailAddressForm()
                   }
    else:
        email = get_object_or_404(EmailAddress, pk=pk)
        context = {"name": "Edit Email",
                   "method": "PUT",
                   "url": "/rundb/api/v1/emailaddress/%s/" % pk,
                   "form": EmailAddressForm(instance=email)
                   }
    return render_to_response(
        "rundb/configure/modal_configure_edit_email.html",
        context_instance=RequestContext(request, context)
    )

@login_required
def delete_email(request, pk=None):
    email = get_object_or_404(EmailAddress, pk=pk)
    if request.method == 'POST':
        email.delete()
        return HttpResponse()
    elif request.method == 'GET':
        ctx = RequestContext(request, {
            "id": email.email, "ids": json.dumps([]),
            "method": "POST", 'methodDescription': 'Delete', "readonly": False,
            'type': "Email Address",
            'action': reverse('delete_email', args=[pk, ]), 'actions': json.dumps([])
        })
        return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)


def get_sge_jobs():
    jobs = {}
    args = ['qstat', '-u', 'www-data', '-s']
    options = (('r', 'running'),
               ('p', 'pending'),
               ('s', 'suspended'),
               ('z', 'done'))

    for opt, status in options:
        p1 = subprocess.Popen(args + [opt], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = p1.stdout.readlines()
        jobs[status] = [l.split()[0] for l in stdout if l.split()[0].isdigit()]

    return jobs


@login_required
def jobStatus(request, jid):
    args = ['qstat', '-j', jid]
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    detailed = p1.stdout.readlines()
    status = 'Running'
    if not detailed:
        # try finished jobs
        args = ['qacct', '-j', jid]
        p1 = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        detailed = p1.stdout.readlines()
        if detailed:
            status = 'done, exit_status='
            for line in detailed:
                if 'exit_status' in line.split()[0]:
                    status += line.split()[1]
        else:
            status = 'not found'

    ctxd = {'jid': jid, 'status': status, 'jInfo': detailed}
    context = RequestContext(request, ctxd)
    return render_to_response('rundb/ion_jobStatus.html', context_instance=context)


@login_required
def jobDetails(request, jid):
    pk = request.GET.get('result_pk')
    result = get_object_or_404(Results, pk=pk)
    job_list_json = os.path.join(result.get_report_path(), 'job_list.json')

    # job_list.json is written by TLScript and will not be available until all jobs are launched
    if not os.path.exists(job_list_json):
        context = RequestContext(request, {'TLS_jid': jid, 'summary': None})
        return render_to_response('rundb/configure/services_jobDetails.html', context_instance=context)

    with open(job_list_json, 'r') as f:
        jobs = json.load(f)

    current_jobs = get_sge_jobs()

    for block, subjobs in jobs.items():
        block_status = 'pending'
        for job in subjobs:
            subjob_jid = subjobs[job]
            # get job status
            status = 'done'
            for st, job_list in current_jobs.items():
                if subjob_jid in job_list:
                    status = st

            # processing status for the block: show the job that's currently running
            if status == 'running':
                block_status = job
            elif status == 'done':
                if block == 'merge' and job == 'merge/zipping':
                    block_status = 'done'
                elif job == 'alignment':
                    block_status = 'done'

        jobs[block]['status'] = block_status

    # summary count how many blocks in each category
    summary_keys = ['pending', 'sigproc', 'basecaller', 'alignment', 'done']
    summary_values = [0] * len(summary_keys)
    num_blocks = len(jobs) - 1  # don't count merge block
    for block in jobs:
        if block != 'merge' and jobs[block]['status'] != 'not found':
            indx = summary_keys.index(jobs[block]['status'])
            summary_values[indx] += 1

    context = RequestContext(request, {'TLS_jid': jid, 'jobs': jobs, 'summary': zip(summary_keys, summary_values), 'num_blocks': num_blocks})
    return render_to_response('rundb/configure/services_jobDetails.html', context_instance=context)


@login_required
def queueStatus(request):
    # get cluster queue status
    args = ['qstat', '-g', 'c', '-ext']
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = p1.stdout.readlines()
    queues = []
    for line in stdout:
        sl = line.split()
        if len(sl) > 1 and '.q' in sl[0]:
            queues.append({
                'name': sl[0],
                'pending': 0,
                'used': sl[2],
                'avail': sl[4],
                'error': sl[18],
                'total': sl[5],
            })

    # get pending jobs per queue
    args = ['qstat', '-u', 'www-data', '-q']
    for queue in queues:
        p1 = subprocess.Popen(args + [queue['name']], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = p1.stdout.readlines()
        for line in stdout:
            sl = line.split()
            if sl[0].isdigit() and 'qw' in sl[4]:
                queue['pending'] += 1

    context = RequestContext(request, {'queues': queues})
    return render_to_response('rundb/configure/modal_services_queueStatus.html', context_instance=context)


@login_required
def configure_account(request):
    """ Show form for this user account. If user is admin, show list of users waiting for approval. """

    (userprofile, created) = UserProfile.objects.get_or_create(user=request.user)
    updated = False
    if request.method == "POST":
        form = UserProfileForm(data=request.POST, instance=userprofile)
        if form.is_valid():
            form.save()
            updated = True
    else:
        form = UserProfileForm(instance=userprofile)

    # superuser (admin) only - show list of users waiting activation
    if request.user.is_superuser:
        approve = User.objects.filter(is_active=False)
    else:
        approve = None

    context = RequestContext(request, {'form': form, 'approve': approve, 'updated': updated})
    return render_to_response("rundb/configure/account.html",
                              context_instance=context)


@login_required
def system_support_archive(request):
    try:
        path, name = makeCSA.make_ssa()
        response = HttpResponse(FileWrapper(open(path)),
                                    mimetype='application/zip')
        response['Content-Disposition'] = 'attachment; filename=%s' % name
        return response
    except:
        logger.exception("Failed to create System Support Archive")
        return HttpResponseServerError(traceback.format_exc())


def get_ampliseq_designs(user, password):
    h = httplib2.Http(disable_ssl_certificate_validation=settings.DEBUG)
    h.add_credentials(user, password)
    url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/design/list")
    response, content = h.request(url)
    if response['status'] == '200':
        design_data = json.loads(content)
        designs = design_data.get('AssayDesigns', [])
        for design in designs:
            solutions = []
            for solution in design.get('DesignSolutions', []):
                version, data, meta = ampliseq.handle_versioned_plans(solution)
                solutions.append(data)
            design['DesignSolutions'] = solutions
        return response, designs
    else:
        return response, {}

def get_ampliseq_fixed_designs(user, password):
    h = httplib2.Http(disable_ssl_certificate_validation=settings.DEBUG)
    h.add_credentials(user, password)
    url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/tmpldesign/list/active")
    response, content = h.request(url)
    if response['status'] == '200':
        designs = json.loads(content)
        fixed = []
        for template in designs.get('TemplateDesigns', []):
            version, data, meta = ampliseq.handle_versioned_plans(template)
            fixed.append(data)
        return response, fixed
    else:
        return response, None


@require_POST
@login_required
def configure_ampliseq_logout(request):
    if "ampliseq_username" in request.session:
        del request.session["ampliseq_username"]
    if "ampliseq_password" in request.session:
        del request.session["ampliseq_password"]
    return HttpResponseRedirect(urlresolvers.reverse("configure_ampliseq"))


@login_required
def configure_ampliseq(request, pipeline=None):
    ctx = {
        'designs': None
    }
    form = AmpliseqLogin()
    if request.method == 'POST':
        form = AmpliseqLogin(request.POST)
        if form.is_valid():
            request.session['ampliseq_username'] = form.cleaned_data['username']
            request.session['ampliseq_password'] = form.cleaned_data['password']
            if pipeline:
                return HttpResponseRedirect(urlresolvers.reverse("configure_ampliseq", args=[pipeline]))
            else:
                return HttpResponseRedirect(urlresolvers.reverse("configure_ampliseq"))

    if 'ampliseq_username' in request.session:
            username = request.session['ampliseq_username']
            password = request.session['ampliseq_password']
            try:
                response, designs = get_ampliseq_designs(username, password)
                if response['status'].startswith("40"):
                    request.session.pop('ampliseq_username')
                    request.session.pop('ampliseq_password')
                    ctx['http_error'] = 'Your user name or password is invalid.<br> You may need to log in to <a href="https://ampliseq.com/">AmpliSeq.com</a> and check your credentials.'
                    fixed = None
                else:
                    response, fixed = get_ampliseq_fixed_designs(username, password)
            except httplib2.HttpLib2Error as err:
                logger.error("There was a connection error when contacting ampliseq: %s" % err)
                ctx['http_error'] = "Could not connect to AmpliSeq.com"
                fixed = None

            pipe_types = {
                "RNA": "AMPS_RNA",
                "DNA": "AMPS",
                "exome": "AMPS_EXOME"
            }
            target = pipe_types.get(pipeline, None)
            def match(design):
                return not target or design['plan']['runType'] == target

            if fixed is not None:
                form = None
                ctx['ordered_solutions'] = []
                ctx['fixed_solutions'] = filter(lambda x: x['status'] == "ORDERABLE" and
                    match(x), fixed)
                ctx['unordered_solutions'] = []
                for design in designs:
                    for solution in design['DesignSolutions']:
                        if match(solution):
                            if solution.get('ordered', False):
                                ctx['ordered_solutions'].append((design, solution))
                            else:
                                ctx['unordered_solutions'].append((design, solution))
                ctx['designs_pretty'] = json.dumps(designs, indent=4, sort_keys=True)
                ctx['fixed_designs_pretty'] = json.dumps(fixed, indent=4, sort_keys=True)
    ctx['form'] = form
    ctx['ampliseq_account_update'] = timezone.now() < timezone.datetime(2013, 11, 30, tzinfo=timezone.utc)
    ctx['ampliseq_url'] = settings.AMPLISEQ_URL
    return render_to_response("rundb/configure/ampliseq.html", ctx,
        context_instance=RequestContext(request))


@login_required
def configure_ampliseq_download(request):
    if 'ampliseq_username' in request.session:
        username = request.session['ampliseq_username']
        password = request.session['ampliseq_password']
        solutions = request.POST.getlist("solutions")
        fixed_solutions = request.POST.getlist("fixed_solutions")
        for ids in solutions:
            design_id, solution_id = ids.split(",")
            meta = '{"choice":"%s"}' % request.POST.get(solution_id + "_instrument_choice", "None")
            start_ampliseq_solution_download(design_id, solution_id, meta, (username, password))
        for ids in fixed_solutions:
            design_id, reference = ids.split(",")
            meta = '{"reference":"%s", "choice": "%s"}' % (reference.lower(), request.POST.get(design_id + "_instrument_choice", "None"))
            start_ampliseq_fixed_solution_download(design_id, meta, (username, password))

    if request.method == "POST":
        if solutions or fixed_solutions:
            return HttpResponseRedirect(urlresolvers.reverse("configure_ampliseq_download"))
        else:
            return HttpResponseRedirect(urlresolvers.reverse("configure_ampliseq"))

    downloads = FileMonitor.objects.filter(tags__contains="ampliseq_template").order_by('-created')
    ctx = {
        'downloads': downloads
    }
    return render_to_response("rundb/configure/ampliseq_download.html", ctx,
        context_instance=RequestContext(request))

def start_ampliseq_solution_download(design_id, solution_id, meta, auth):
    url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/design/{0}/solutions/{1}/download/results".format(design_id, solution_id))
    monitor = FileMonitor(url=url, tags="ampliseq_template", status="Queued")
    monitor.save()
    t = tasks.download_something.apply_async(
        (url, monitor.id), {"auth": auth},
        link=tasks.ampliseq_zip_upload.subtask((meta,)))

def start_ampliseq_fixed_solution_download(design_id, meta, auth):
    url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/tmpldesign/{0}/download/results".format(design_id))
    monitor = FileMonitor(url=url, tags="ampliseq_template", status="Queued")
    monitor.save()
    t = tasks.download_something.apply_async(
        (url, monitor.id), {"auth": auth},
        link=tasks.ampliseq_zip_upload.subtask((meta,)))

@login_required
def cache_status(request):
    from django import http
    import memcache

    host = memcache._Host(settings.CACHES['default']['LOCATION'])
    host.connect()
    host.send_cmd("stats")

    stats = {}

    while 1:
        try:
            line = host.readline().split(None, 2)
        except socket.timeout:
            break
        if line[0] == "END":
            break
        stat, key, value = line
        try:
            # convert to native type, if possible
            value = int(value)
            if key == "uptime":
                value = datetime.timedelta(seconds=value)
            elif key == "time":
                value = datetime.datetime.fromtimestamp(value)
        except ValueError:
            pass
        stats[key] = value

    host.close_socket()
    total_get = (stats.get('get_hits', 0) + stats.get('get_misses', 0))
    ctx = dict(
        stats=stats,
        items=sorted(stats.iteritems()),
        hit_rate=(100 * stats.get('get_hits', 0) / total_get) if total_get else 0,
        total_get=total_get,
        time=datetime.datetime.now()
    )
    return render_to_response("rundb/configure/cache_status.html", ctx)


def cluster_info_refresh(request):
    try:
        t = tasks.check_cluster_status()
        t.get(timeout = 120)
    except:
        return HttpResponseServerError(traceback.format_exc())
    return HttpResponse()


def cluster_info_log(request, pk):
    nodes = Cruncher.objects.all()
    ct = ContentType.objects.get_for_model(Cruncher)
    title = "Cluster Info log for %s" % nodes.get(pk=pk).name
    
    ctx = RequestContext(request, {"title": title, "pk": pk, "cttype": ct.id})
    return render_to_response("rundb/common/modal_event_log.html", context_instance=ctx)


def cluster_info_history(request):
    nodes = Cruncher.objects.all().values_list('name',flat=True)
    ctx = RequestContext(request, {'nodes':nodes})
    return render_to_response("rundb/configure/clusterinfo_history.html", context_instance=ctx)