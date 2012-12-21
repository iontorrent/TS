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

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import render_to_response, get_object_or_404, get_list_or_404
from django.template import RequestContext, Context
from django.http import Http404, HttpResponsePermanentRedirect, HttpResponse,\
    HttpResponseNotFound, HttpResponseBadRequest, HttpResponseNotAllowed
from django.core.servers.basehttp import FileWrapper
from django.core.urlresolvers import reverse
from django.forms.models import model_to_dict
import ion.utils.TSversion
from iondb.rundb.configure.archiver_utils import exp_list
from iondb.rundb.configure.archiver_utils import areyourunning
from iondb.rundb.forms import EditBackup, EmailAddress as EmailAddressForm, EditReportBackup,\
    bigPruneEdit, EditPruneLevels, UserProfileForm
from iondb.rundb import tasks, publishers
from iondb.anaserve import client
from iondb.plugins.manager import pluginmanager
from django.contrib.auth.models import User
from iondb.rundb.models import dnaBarcode, Plugin, BackupConfig, GlobalConfig,\
    EmailAddress, Publisher, Location, dm_reports, dm_prune_group, Experiment,\
    Results, Template, UserProfile, dm_prune_field
from iondb.anaserve import client as anaclient
from iondb.backup import rawDataStorageReport, reportLogStorage
from iondb.rundb.configure.genomes import search_for_genomes
# Handles serialization of decimal and datetime objects
from django.core.serializers.json import DjangoJSONEncoder
from iondb.rundb.configure.util import plupload_file_upload
from ion.utils import makeCSA
from django.core import urlresolvers


logger = logging.getLogger(__name__)

# Handler to send logging to ionArchive's logger handler for /var/log/ion/reportsLog.log
socketHandler = logging.handlers.SocketHandler('localhost', settings.DM_LOGGER_PORT)



@login_required
def configure(request):
    return configure_about(request)


@login_required
def configure_about(request):
    reload(ion.utils.TSversion)
    versions, meta = ion.utils.TSversion.findVersions()
    ctxd = {"versions": versions, "meta": meta}
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/about.html", context_instance=ctx)


@login_required
def configure_services(request):
    jobs = current_jobs(request)
    crawler = _crawler_status(request)
    processes = process_set()
    backups = BackupConfig.objects.all().order_by('pk')
    iastatus = areyourunning()
    to_archive, fs_stats = exp_list(backups[0])
    autoArchive = GlobalConfig.objects.all().order_by('pk')[0].auto_archive_ack
    ctxd = {"processes": processes, "jobs": jobs, "crawler": crawler, "backups": backups, "to_archive": to_archive, "iastatus": iastatus, "fs_stats": fs_stats, "autoArchive": autoArchive}
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

    ctxd = {"plugin": plugin_json, "file": content, "report": report, "tmap": str(index_version)}
    context = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/modal_configure_plugins_plugin_configure.html", context_instance=context)


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
def configure_configure(request):
    ctx = RequestContext(request, {})
    emails = EmailAddress.objects.all().order_by('pk')
    ctx.update({"email": emails})
    config_contacts(request, ctx)
    config_site_name(request, ctx)
    return render_to_response("rundb/configure/configure.html", context_instance=ctx)


@login_required
def config_publishers(request, ctx):
    globalconfig = GlobalConfig.objects.all().order_by('pk')[0]
    # Rescan Publishers
    publishers.purge_publishers()
    publishers.search_for_publishers(globalconfig)
    pubs = Publisher.objects.all().order_by('name')
    ctx.update({"publishers": pubs})


def get_dir_choices():
    from iondb.backup import devices
    basicChoice = [(None, 'None')]
    for choice in devices.to_media(devices.disk_report()):
        basicChoice.append(choice)
    return tuple(basicChoice)


@login_required
def edit_backup(request, pk):
    """
    Handles any changes to the backup configuration
    """
    if int(pk) != 0:
        bk = get_object_or_404(BackupConfig, pk=pk)
    else:
        bk = BackupConfig()

    if request.method == "POST":
        ebk = EditBackup(request.POST)
        if ebk.is_valid():
            if ebk.cleaned_data['archive_directory'] is not None:
                bk.name = ebk.cleaned_data['archive_directory'].strip().split('/')[-1]
                bk.backup_directory = ebk.cleaned_data['archive_directory']
            else:
                bk.name = 'None'
                bk.backup_directory = 'None'
            bk.location = Location.objects.all()[0]
            bk.number_to_backup = ebk.cleaned_data['number_to_archive']
            bk.timeout = ebk.cleaned_data['timeout']
            bk.backup_threshold = ebk.cleaned_data['percent_full_before_archive']
            bk.grace_period = int(ebk.cleaned_data['grace_period'])
            bk.bandwidth_limit = int(ebk.cleaned_data['bandwidth_limit'])
            bk.email = ebk.cleaned_data['email']
            bk.online = ebk.cleaned_data['enabled']
            bk.save()
            url = reverse("configure_services")
            return HttpResponsePermanentRedirect(url)
        else:
            ctxd = {"temp": ebk}
            ctx = RequestContext(request, ctxd)
            return render_to_response("rundb/configure/edit_backup.html",
                                      context_instance=ctx)
            #return render_to_response("rundb/configure/modal_configure_edit_backup.html",
            #                                    context_instance=ctx)

    elif request.method == "GET":
        temp = EditBackup()
        if int(pk) == 0:
            #temp.fields['archive_directory'].choices = get_dir_choices()
            temp.fields['number_to_archive'].initial = 10
            temp.fields['timeout'].initial = 60
            temp.fields['percent_full_before_archive'].initial = 90
            temp.fields['grace_period'].initial = 72
            ctxd = {"temp": temp, "name": "New Archive Configuration"}
            ctx = RequestContext(request, ctxd)
            return render_to_response("rundb/configure/edit_backup.html",
                                      context_instance=ctx)
            #return render_to_response("rundb/configure/modal_configure_edit_backup.html",
            #                                    context_instance=ctx)
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
            ctxd = {"temp": temp}
            ctx = RequestContext(request, ctxd)
            return render_to_response("rundb/configure/edit_backup.html",
                                      context_instance=ctx)
            #return render_to_response("rundb/configure/modal_configure_edit_backup.html",
            #                                    context_instance=ctx)


def _configure_report_data_mgmt(request, pk=None):
    logger = logging.getLogger("reportLogger")
    logger.addHandler(socketHandler)

    def getRuleList(grps):
        rList = []
        for j, grp in enumerate(grps, start=1):
            grp.idStr = "%02d" % (j)
            rList.append([grp.name, 'T' if grp.editable else 'F', grp.idStr, grp.pk])
        return rList

    def getReportStorageSavings(days):
        result = []
        for day in days:
            try:
                kbtotal = reportLogStorage.getSavedSpace(day)
            except:
                kbtotal = 0
                logger.error(traceback.format_exc())

            gbtotal = round(kbtotal/(1024 * 1024), 3)
            result.append([day, kbtotal, gbtotal])

        return result

    if not pk:
        qs = dm_reports.objects.all().order_by('-pk')
        if qs.exists():
            model = qs[0]
            model.save()
        else:
            model = dm_reports()
            model.save()
        pk = model.pk

    model = get_object_or_404(dm_reports, pk=pk)

    reportStr = getReportStorageSavings([1, 7, 14, 30, 90, 365])
    grps = dm_prune_group.objects.all().order_by('pk')
    rList = getRuleList(grps)
    if request.method == "POST":
        form = EditReportBackup(request.POST)
        if form.is_valid():
            _autoType = form.cleaned_data['autoAction']
            if model.autoType != _autoType:
                model.autoType = _autoType
                logger.info("dm_reports configuration changed: autoType set to %s" % model.autoType)

            _autoAge = form.cleaned_data['autoDays']
            if model.autoAge != _autoAge:
                model.autoAge = _autoAge
                logger.info("dm_reports configuration changed: autoAge set to %d" % model.autoAge)

            _pruneLevel = form.cleaned_data['pruneLevel']
            if model.pruneLevel != _pruneLevel:
                model.pruneLevel = _pruneLevel
                logger.info("dm_reports configuration changed: pruneLevel set to %s" % model.pruneLevel)

            _location = form.cleaned_data['location']
            if model.location != _location:
                model.location = _location
                logger.info("dm_reports configuration changed: archive location set to %s" % model.location)

            _autoPrune = form.cleaned_data['autoPrune']
            if model.autoPrune != _autoPrune:
                model.autoPrune = _autoPrune
                logger.info("dm_reports configuration changed: auto-action set to %s" % model.autoPrune)

            model.save()
            url = reverse('configure_configure')
            return HttpResponsePermanentRedirect(url)
    else:
        form = EditReportBackup()
        form.fields['location'].initial = model.location
        form.fields['autoPrune'].initial = model.autoPrune
        form.fields['autoDays'].initial = model.autoAge
        if '%s' % model.pruneLevel == '':
            form.fields['pruneLevel'].initial = ['No-op']
        else:
            form.fields['pruneLevel'].initial = model.pruneLevel
        form.fields['autoAction'].initial = model.autoType
    ctxd = {"form": form, "spaceSaved": reportStr, "ruleList": rList}
    ctx = RequestContext(request, ctxd)
    return ctx


@login_required
def configure_report_data_mgmt_prunegroups(request, pk=None):
    ctx = _configure_report_data_mgmt(request, pk)
    return ctx if isinstance(ctx, HttpResponsePermanentRedirect) else render_to_response("rundb/configure/blocks/configure_report_data_mgmt_prunegroups.html", context_instance=ctx)


@login_required
def configure_report_data_mgmt(request, pk=None):
    ctx = _configure_report_data_mgmt(request, pk)
    return ctx if isinstance(ctx, HttpResponsePermanentRedirect) else render_to_response("rundb/configure/configure_report_data_mgmt.html", context_instance=ctx)


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
        elapsed = seconds2htime(raw_elapsed)
        nfound = cstat.experiments_found()
        raw_exprs = cstat.prev_experiments()
        exprs = []
        for r in raw_exprs:
            try:
                exp = Experiment.objects.get(expName=r)
            except (Experiment.DoesNotExist,
                    Experiment.MultipleObjectsReturned):
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
    ctx = RequestContext(request, {"result_dict": dict(result_pairs)})
    return ctx


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
            runs = dict((r[2], r) for r in running)
            results = Results.objects.select_related('experiment').filter(pk__in=runs.keys()).order_by('pk')
            for result in results:
                name, pid, pk, atype, stat = runs[result.pk]
                jobs.append((short_name, name, pid, atype, stat,
                             result, result.experiment))
    ctxd = {"jobs": jobs, "servers": servers}
    return ctxd


def process_set():
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
                                              lambda x: int(x[x.rindex(',') + 1:x.rindex('}')]))
    return sorted(proc_set.items())


def exp_ack(request):
    if request.method == 'POST':

        runPK = request.POST.get('runpk', False)
        runState = request.POST.get('runstate', False)

        if not runPK:
            return HttpResponse(json.dumps({"status": "error, no runPK POSTed"}), mimetype="application/json")

        if not runState:
            return HttpResponse(json.dumps({"status": "error, no runState POSTed"}), mimetype="application/json")

        try:
            exp = Experiment.objects.get(pk=runPK)
        except:
            return HttpResponse(json.dumps({"status": "error, could find the run"}), mimetype="application/json")

        try:
            exp.user_ack = runState
            exp.save()
        except:
            return HttpResponse(json.dumps({"status": "error, could not modify the user_ack state for " + str(exp)}), mimetype="application/json")

        try:
            host = "127.0.0.1"
            conn = client.connect(host, settings.IARCHIVE_PORT)
            user_ack = conn.user_ack()
        except:
            return HttpResponse(json.dumps({"status": "error, could not connect the the backup process over xmlrpc"}), mimetype="application/json")

        return HttpResponse(json.dumps({"runState": runState, "user_ack": exp.user_ack, "runPK": runPK, "user_ack": user_ack}), mimetype="application/json")


@login_required
def enableArchive(request, pk, set):
    """Allow user to enable the archive tool"""
    try:
        pk = int(pk)
    except (TypeError, ValueError):
        return HttpResponseNotFound

    archive = get_object_or_404(BackupConfig, pk=pk)
    archive.online = bool(int(set))
    archive.save()
    return HttpResponse()


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
        destination = tempfile.NamedTemporaryFile(delete=False)
        for chunk in postedfile.chunks():
            destination.write(chunk)
        postedfile.close()
        destination.close()
        #check to ensure it is not empty
        headerCheck = open(destination.name, "rU")
        firstCSV = []
        for firstRow in csv.reader(headerCheck):
            firstCSV.append(firstRow)
        headerCheck.close()
        if not firstRow:
            os.unlink(destination.name)
            return HttpResponse(json.dumps({"status": "Error: Barcode file is empty"}), mimetype="text/html")
        expectedHeader = ["id_str", "type", "sequence", "floworder", "index", "annotation", "adapter", "score_mode", "score_cutoff"]
        if sorted(firstCSV[0]) != sorted(expectedHeader):
            os.unlink(destination.name)
            return HttpResponse(json.dumps({"status": "Barcode csv header is not as expected. Please try again starting with the provided example"}), mimetype="text/html")
        #test if the barcode set name has been used before
        barCodeSet = dnaBarcode.objects.filter(name=name)
        if barCodeSet:
            os.unlink(destination.name)
            return HttpResponse(json.dumps({"status": "Error: Barcode set with the same name already exists"}), mimetype="text/html")
        index = 0
        barCodes = []
        failed = {}
        file = open(destination.name, "rU")
        reader = csv.DictReader(file)
        for index, row in enumerate(reader, start=1):
            invalid = _validate_barcode(row)
            if invalid:  # don't make dna object or add it to the list
                failed[index] = invalid
                continue
            newBarcode = dnaBarcode()
            newBarcode.name = name  # set the name
            newBarcode.index = index  # set index this can be overwritten later
            nucs = ["sequence", "floworder", "adapter"]  # fields that have to be uppercase
            for key, value in row.items():  # set the values for the objects
                value = str(value)
                value = value.strip()  # strip the strings
                if key in nucs:  # uppercase if a nuc
                    value = value.upper()
                if value:
                    setattr(newBarcode, key, value)
            if not newBarcode.id_str:  # make a id_str if one is not provided
                newBarcode.id_str = str(name) + "_" + str(index)
            newBarcode.length = len(newBarcode.sequence)  # now set a default
            barCodes.append(newBarcode)  # append to our list for later saving

        destination.close()  # now close and remove the temp file
        os.unlink(destination.name)
        if index > 384:
            return HttpResponse(json.dumps({"status": "Error: Too many barcodes! There must be 384 or less. Please reload the page and try again with fewer barcodes."}), mimetype="text/html")
        if index == 0:
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
        if failed:
            r = {"status": "Barcodes validation failed. The barcode set has not been saved.", "failed": failed}
            return HttpResponse(json.dumps(r), mimetype="text/html")
        #saving to db needs to be the last thing to happen
        for barCode in barCodes:
            try:
                barCode.save()
            except:
                return HttpResponse(json.dumps({"status": "Error saving barcode to database!"}), mimetype="text/html")
        r = {"status": "Barcodes Uploaded! The barcode set will be listed on the references page.", "failed": failed}
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
                failed.append((nuc, "Must have A, T, C, G only"))
    if 'score_mode' in barCodeDict:
        if barCodeDict["score_mode"]:
            try:
                int(str(barCodeDict["score_mode"]))
            except ValueError:
                failed.append(("score_mode", "score_mode must be a whole number"))
    if 'score_cutoff' in barCodeDict:
        if barCodeDict["score_cutoff"]:
            if barCodeDict["score_cutoff"]:
                try:
                    float(str(barCodeDict["score_cutoff"]))
                except ValueError:
                    failed.append(("score_cutoff", "score_cutoff must be a number"))
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
def change_storage(request, pk, value):
    """changes the storage option for run raw data"""
    if request.method == 'POST':
        try:
            pk = int(pk)
        except (TypeError, ValueError):
            return HttpResponseNotFound

        exp = get_object_or_404(Experiment, pk=pk)

        # When changing from Archive option to Delete option, need to reset
        # the user acknowledge field
        if exp.storage_options == 'A' and value == 'D':
            exp.user_ack = 'U'

        exp.storage_options = value
        exp.save()
        return HttpResponse()


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
    conn = anaclient.connect(ip, settings.JOBSERVER_PORT)
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
    p = subprocess.Popen(networkCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stats_network = stdout.splitlines(True)

    statsCMD = ["/usr/bin/ion_sysinfo"]
    q = subprocess.Popen(statsCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = q.communicate()
    stats = stdout.splitlines(True)

    stats_dm = rawDataStorageReport.storage_report()

    # Create filename for the report
    reportFileName = "/tmp/stats_sys.txt"

    # Stuff the variable into the context object
    ctx = Context({"stats_network": stats_network,
                   "stats": stats,
                   "stats_dm": stats_dm,
                   "reportFilePath": reportFileName,
                   "use_precontent": True,
                   "use_content2": True,
                   "use_content3": True, })

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

    return render_to_response("rundb/configure/configure_system_stats.html", context_instance=ctx)


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
def configure_report_data_mgmt_editPruneGroups(request):
    logger = logging.getLogger("reportLogger")
    logger.addHandler(socketHandler)

    reportOptList = dm_reports.objects.all().order_by('-pk')
    bk = reportOptList[0]
    groupList = dm_prune_group.objects.all().order_by('pk')
    ruleList = dm_prune_field.objects.all().order_by('pk')
    readonlyGroupList = dm_prune_group.objects.filter(editable=False).order_by('pk')
    editableGroupList = dm_prune_group.objects.filter(editable=True).order_by('pk')
    rulesUsedByReadonlyGroups = set(filter(None, set([item for g in readonlyGroupList for item in g.ruleNums.split(',')])))
    fieldPKList = []
    for field in ruleList:
        fieldPKList.append('%s' % field.pk)
    if request.method == 'GET':
        temp = []
        for grp in groupList:
            kwargs = {"pk": grp.pk}
            temp.append(bigPruneEdit(**kwargs))
            tempList = string.split('%s' % grp.ruleNums, ',')
            list = []
            for num in tempList:
                if num in fieldPKList:
                    list.append('%s' % grp.pk + ':' + '%s' % num)
            temp[-1].fields['checkField'].initial = list

        #for rTemp in temp:
        #    logger.error(rTemp.fields['checkField'].widget.choices)
        #    logger.error(rTemp.fields['checkField'].initial)

        ctxd = {"bk": bk, "groups": groupList, "fields": ruleList, "temp": temp, "selected": bk.pruneLevel, "rulesUsedByReadonlyGroups":rulesUsedByReadonlyGroups}
        context = RequestContext(request, ctxd)
        return render_to_response("rundb/configure/modal_configure_report_data_mgmt_edit_pruning_config.html",
                                  context_instance=context)

    elif request.method == 'POST':
        # This field contains pk of rules to remove: from dm_prune_rules table and from prune_groups that reference it
        #TODO: TS-4965: log rule removal, prune group edits, new field addition
        # Remove rule(s) marked for removal
        removeList = set(request.POST.getlist('remField'))
        removeNames = []    # string list of rules removed
        logger.info(removeList)
        removeList = removeList - rulesUsedByReadonlyGroups  #removing any rules that are used by uneditable dm_prune_groups
        
        for pk in removeList:
            rule = dm_prune_field.objects.get(pk=pk)
            name = rule.rule
            removeNames.append(name)
            rule.delete()
            logger.info("prune_field deleted: %s" % name)

        # Edit prune group objects and remove rules marked for removal
        for pgrp in editableGroupList:
            newList = []    # new list to contain valid rules only
            for rule in pgrp.ruleNums.split(','):
                if len(rule) > 0 and rule not in removeList:
                    newList.append(int(rule))
            newNums = ','.join(['%d' % i for i in newList])
            #pks can be in any order so we sort them to see if they are different.
            if sorted(newList) != sorted([int(i) for i in pgrp.ruleNums.split(',') if len(i) > 0]):
                pgrp.ruleNums = newNums
                pgrp.save()
                logger.info("dm_prune_group edited: %s (removed %s)" % (pgrp.name, removeNames))

        checkList = (request.POST.getlist('checkField'))
        for grp in editableGroupList:
            newList = []    # new list to contain valid rules only
            for box in checkList:
                if ':' in box:
                    opt = string.split(box, ':')
                    if str(grp.pk) == str(opt[0]):
                        newList.append(int(opt[1]))
                else:
                    logger.debug('checkField list entry is: \'%s\'' % box)
            newNums = ','.join(['%d' % i for i in newList])
            #pks can be in any order so we sort them to see if they are different.
            if sorted(newList) != sorted([int(i) for i in grp.ruleNums.split(',') if len(i) > 0]):
                grp.ruleNums = newNums
                grp.save()
                logger.info("dm_prune_group edited(rule change): %s" % grp.name)

        addString = request.POST['newField']
        #in case someone tries to enter a list...
        #don't want to give the impression that entering '*.bam, *.bai' would work, since each rule is for individual file (type)s.
        addString = string.replace(addString, ' ', '')
        addString = string.replace(addString, '"', '')
        addString = string.replace(addString, '[', '')
        addString = string.replace(addString, ']', '')
        addString = string.replace(addString, "'", '')
        addString = string.replace(addString, ",", '')

        if addString != '':
            obj = dm_prune_field()
            obj.rule = addString
            obj.save()
            logger.info("dm_prune_field created: %s" % obj.rule)

        url = reverse('configure_report_data_mgmt')
        return HttpResponsePermanentRedirect(url)


@login_required
def configure_report_data_mgmt_remove_pruneGroup(request, pk):
    logger = logging.getLogger("reportLogger")
    logger.addHandler(socketHandler)

    pgrp = dm_prune_group.objects.get(pk=pk)

    if request.method == 'GET':
        type = "Prune Group"
        pks = []
        actions = []
        ctx = RequestContext(request, {
            "id": pk, "ids": json.dumps(pks), "method": "POST", "names": pgrp.name, 'methodDescription': 'Delete', "readonly": False, 'type': type, 'action': reverse('configure_report_data_mgmt_remove_prune_group', args=[pk, ]), 'actions': json.dumps(actions)
        })
        return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)
        pass
    elif request.method == "POST":
        #Remove the specified prune_group object
        try:
            name = pgrp.name
            pgrp.delete()
            logger.info("dm_prune_group deleted: %s" % name)
        except:
            raise

        # If the removed prune group was also marked as the default prune group level, then clear the default prune group level in configuration object
        reportOptList = dm_reports.objects.all().order_by('-pk')
        bk = reportOptList[0]
        if name == bk.pruneLevel:
            bk.pruneLevel = 'No-op'
            bk.save()
            logger.info("dm_reports configuration change: default prune level from %s to No-op" % name)

        return HttpResponse(json.dumps({"status": "success"}), mimetype="application/json")


def getRules(nums):
    '''nums is array of pks of prune_field'''
    ruleString = []
    for num in nums:
        rule = dm_prune_field.objects.get(pk=num)
        ruleString.append(rule.rule)
    return ruleString


@login_required
def configure_report_data_mgmt_pruneEdit(request):
    logger = logging.getLogger("reportLogger")
    logger.addHandler(socketHandler)

    grps = dm_prune_group.objects.all().order_by('pk').reverse()
    ruleList = dm_prune_field.objects.all().order_by('pk')
    reportOptList = dm_reports.objects.all().order_by('-pk')
    bk = reportOptList[0]
    if request.method == 'GET':
        temp = EditPruneLevels(request.GET)
        kwargs = {'pk': 0}
        rTemp = bigPruneEdit(**kwargs)
        ctxd = {"groups": grps, "fields": ruleList, "bk": bk, "temp": temp, "ruleTemp": rTemp}
        context = RequestContext(request, ctxd)
        return render_to_response("rundb/configure/modal_configure_report_data_mgmt_add_prune_group.html",
                                  context_instance=context)
    elif request.method == 'POST':
        name = request.POST['name'] if request.POST['name'] != '' else 'Untitled'

        try:
            #check for an existing prune group with the given name, return an error if found.
            grp = dm_prune_group.objects.get(name=name)
            return HttpResponseBadRequest("Error: Cannot create Duplicate Prune Group!")
        except:
            # Adding a new prune group
            grp = dm_prune_group()
            checkList = (request.POST.getlist('checkField'))
            ruleNums = []
            for box in checkList:
                opt = string.split(box, ':')
                ruleNums.append(int(opt[1]))
            grp.ruleNums = ','.join(['%d' % i for i in ruleNums])
            grp.name = name
            grp.save()
            logger.info("dm_prune_group created: %s - Rules = %s" % (grp.name, getRules(ruleNums)))

            url = reverse('configure_report_data_mgmt')
            return HttpResponsePermanentRedirect(url)


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
        return HttpResponse(status=500)
        
