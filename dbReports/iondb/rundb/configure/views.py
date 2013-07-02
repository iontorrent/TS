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
from pprint import pformat
from celery.task import task
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render_to_response, get_object_or_404, get_list_or_404
from django.template import RequestContext, Context
from django.http import Http404, HttpResponsePermanentRedirect, HttpResponse, HttpResponseRedirect, \
    HttpResponseNotFound, HttpResponseBadRequest, HttpResponseNotAllowed, HttpResponseServerError
from django.core.servers.basehttp import FileWrapper
from django.core.urlresolvers import reverse
from django.forms.models import model_to_dict
import ion.utils.TSversion
from iondb.rundb.configure.archiver_utils import exp_list, disk_usage_stats
from iondb.rundb.forms import EmailAddress as EmailAddressForm, EditReportBackup,\
    bigPruneEdit, EditPruneLevels, UserProfileForm
from iondb.rundb.forms import AmpliseqLogin
from iondb.rundb import tasks, publishers
from iondb.anaserve import client
from iondb.plugins.manager import pluginmanager
from django.contrib.auth.models import User
from iondb.rundb.models import dnaBarcode, Plugin, GlobalConfig,\
    EmailAddress, Publisher, Location, dm_reports, dm_prune_group, Experiment,\
    Results, Template, UserProfile, dm_prune_field, FileServer, DMFileSet, DMFileStat,\
    DownloadMonitor, EventLog, ContentType
from iondb.rundb.data import rawDataStorageReport, reportLogStorage
from iondb.rundb.configure.genomes import search_for_genomes
from iondb.rundb.plan import ampliseq
# Handles serialization of decimal and datetime objects
from django.core.serializers.json import DjangoJSONEncoder
from iondb.rundb.configure.util import plupload_file_upload
from ion.utils import makeCSA
from django.core import urlresolvers
from iondb.rundb.data import tasks as dmtasks
from iondb.rundb.data import dmactions
import iondb.rundb.data.dmactions_types as dmactions_types
from iondb.rundb.data.data_management import update_files_in_use
from iondb.rundb.data import exceptions as DMExceptions
from iondb.utils.files import disk_attributes
logger = logging.getLogger(__name__)


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
    fs_stats = disk_usage_stats()

    if os.path.exists("/opt/ion/.ion-internal-server"):
        # split Data Management tables per fileserver
        dm_tables = []
        for path in FileServer.objects.all().order_by('pk').values_list('filesPrefix', flat=True):
            dm_tables.append({
                "filesPrefix": path,
                "url": "/rundb/api/v1/compositedatamanagement/?format=json&expDir__startswith=%s" % path
            })
    else:
        dm_tables = [{
            "filesPrefix": "",
            "url": "/rundb/api/v1/compositedatamanagement/?format=json"
        }]

    gc = GlobalConfig.objects.all().order_by('pk')[0]
    dm_filesets = DMFileSet.objects.filter(version=settings.RELVERSION).order_by('pk')
    archive_paths = []
    for bdir in set(dm_filesets.values_list('backup_directory', flat=True)):
        if bdir and bdir != 'None':
            try:
                total, availSpace, freeSpace, bsize = disk_attributes(bdir)
                total_gb = float(total*bsize)/(1024*1024*1024)
                avail_gb = float(availSpace*bsize)/(1024*1024*1024)
                #free_gb = float(freeSpace*bsize)/(1024*1024*1024)
                percentfull = 100-(float(availSpace)/float(total)*100) if total > 0 else 0
            except:
                percentfull = ""
                total_gb = ""
                avail_gb = ""
            archive_paths.append({
                'backup_directory': bdir,
                'exists':os.path.exists(bdir),
                'id': dm_filesets.filter(backup_directory=bdir)[0].id,
                'percentfull': percentfull,
                'disksize': total_gb,
                'diskfree': avail_gb,
                })

    ctxd = {
        "processes": processes,
        "jobs": jobs,
        "crawler": crawler,
        "autoArchive": gc.auto_archive_ack,
        "autoArchiveEnable": gc.auto_archive_enable,
        "dm_filesets": dm_filesets,
        "dm_tables": dm_tables,
        "archive_paths": archive_paths,
        "fs_stats": fs_stats,
        }
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/services.html", context_instance=ctx)

@login_required
@staff_member_required
def dm_configuration(request):
    config = GlobalConfig.objects.all()[0]
    dm_contact, created = User.objects.get_or_create(username='dm_contact')

    if request.method == 'GET':
        dm_filesets = DMFileSet.objects.filter(version=settings.RELVERSION).order_by('pk')
        backup_dirs = get_dir_choices()

        config.email = dm_contact.email

        ctx = RequestContext(request, {
            "dm_filesets": dm_filesets,
            "config": config,
            "backup_dirs": backup_dirs,
            "categories": dmactions_types.FILESET_TYPES
        })
        return render_to_response("rundb/configure/dm_configuration.html", context_instance=ctx)

    elif request.method == 'POST':
        dm_filesets = DMFileSet.objects.all()
        log = 'SAVED Data Management Configuration<br>'
        data = json.loads(request.raw_post_data)
        try:
            for key, value in data.items():
                if key == 'filesets':
                    for category, params in value.items():
                        dm_filesets.filter(type=category).update(**params)
                        log += '<b>%s:</b> %s<br>' % (category, json.dumps(params).translate(None, "{}\"\'") )
                elif key == 'email':
                    dm_contact.email = value
                    dm_contact.save()
                    log += '<b>Email:</b> %s<br>' % value
                elif key == 'auto_archive_ack':
                    GlobalConfig.objects.all().update(auto_archive_ack = True if value=='True' else False)
                    log += '<b>Auto Acknowledge Delete:</b> %s<br>' % value
            _add_dm_configuration_log(request, log)
        except Exception as e:
            logger.exception("dm_configuration: error: %s" % str(e))
            return HttpResponseServerError("Error: %s" % str(e))

        return HttpResponse()

def _add_dm_configuration_log(request, log):
    # add log entry. Want to save with the DMFileSet class, not any single object, so use fake object_pk.
    ct = ContentType.objects.get_for_model(DMFileSet)
    ev = EventLog(object_pk=0, content_type=ct, username=request.user.username, text=log)
    ev.save()

def delete_ack(request):
    runPK = request.POST.get('runpk', False)
    runState = request.POST.get('runstate', False)

    if not runPK:
        return HttpResponse(json.dumps({"status": "error, no runPK POSTed"}), mimetype="application/json")
    if not runState:
        return HttpResponse(json.dumps({"status": "error, no runState POSTed"}), mimetype="application/json")

    # Also change the experiment user_ack value.
    exp = Results.objects.get(pk=runPK).experiment
    exp.user_ack = runState
    exp.save()

    # If multiple reports per experiment update all sigproc action_states.
    results_pks = exp.results_set.values_list('pk', flat=True)
    ret = DMFileStat.objects.filter(result__pk__in=results_pks, dmfileset__type=dmactions_types.SIG).update(action_state=runState)

    for result in exp.results_set.all():
        msg = '%s deletion ' % dmactions_types.SIG
        msg += 'is Acknowledged' if runState == 'A' else ' Acknowledgement is removed'
        EventLog.objects.add_entry(result, msg, username=request.user.username)

    return HttpResponse(json.dumps({"runState": runState, "count": ret, "runPK": runPK}), mimetype="application/json")

def dm_log(request, pk=None):
    if request.method == 'GET':
        selected = get_object_or_404(Results, pk=pk)
        ct = ContentType.objects.get_for_model(selected)
        title = "Data Management Actions for %s (%s):" % (selected.resultsName, pk)
        ctx = RequestContext(request, {"title": title, "pk": pk, "cttype": ct.id})
        return render_to_response("rundb/common/modal_event_log.html", context_instance=ctx)

def dm_configuration_log(request):
    if request.method == 'GET':
        ct = ContentType.objects.get_for_model(DMFileSet)
        title = "Data Management Configuration History"
        ctx = RequestContext(request, {"title": title, "pk": 0, "cttype": ct.id})
        return render_to_response("rundb/common/modal_event_log.html", context_instance=ctx)
    elif request.method == 'POST':
        log = request.POST.get('log')
        _add_dm_configuration_log(request, log)
        return HttpResponse()

def dm_history(request):
    logs = EventLog.objects.for_model(Results)
    usernames = set(logs.values_list('username', flat=True))
    ctx = RequestContext(request, {'usernames':usernames})
    return render_to_response("rundb/configure/dm_history.html", context_instance=ctx)

def dm_actions(request, results_pks):
    results = Results.objects.filter(pk__in=results_pks.split(','))

    # update disk space info if needed
    to_update = DMFileStat.objects.filter(diskspace=None, result__in=results_pks.split(','))
    for dmfilestat in to_update:
        dmtasks.update_dmfilestats_diskspace.delay(dmfilestat)
    if len(to_update) > 0:
        time.sleep(2)
    
    dm_files_info = []    
    for category in dmactions_types.FILESET_TYPES:
        info = ''
        for result in results:
            dmfilestat = result.dmfilestat_set.get(dmfileset__type=category)
            if not info:
                info = {
                'category':dmfilestat.dmfileset.type,
                'description':dmfilestat.dmfileset.description,
                'action_state': dmfilestat.get_action_state_display(),
                'diskspace': dmfilestat.diskspace,
                'in_process': dmfilestat.in_process()
            }
            else:
                # multiple results
                if info['action_state'] != dmfilestat.get_action_state_display():
                    info['action_state'] = '*'
                if (info['diskspace'] is not None) and (dmfilestat.diskspace is not None):
                    info['diskspace'] += dmfilestat.diskspace
                else:
                    info['diskspace'] = None
                info['in_process'] &= dmfilestat.in_process()

        dm_files_info.append(info)

    if len(results) == 1:
        name = "Report Name: %s" % result.resultsName
        subtitle = "Run Name: %s" % result.experiment.expName
    else:
        # multiple results (available from Project page)
        name = "Selected %s results." % len(results)
        subtitle = "(%s)" % ', '.join(results_pks.split(','))

    backup_dirs = get_dir_choices()[1:]
    ctxd = {
        "dm_files_info": dm_files_info,
        "name": name,
        "subtitle": subtitle,
        "results_pks": results_pks,
        "backup_dirs": backup_dirs,
        "isDMDebug": os.path.exists("/opt/ion/.ion-dm-debug"),
        }
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/modal_dm_actions.html", context_instance=ctx)

def dm_action_selected(request, results_pks, action):
    '''
    file categories to process: data['categories']
    user log entry comment: data['comment']
    results_pks could contain more than 1 result
    '''
    logger = logging.getLogger('data_management')

    data = json.loads(request.raw_post_data)
    logger.info("dm_action_selected: request '%s' on report(s): %s" % (action, results_pks))

    '''
    organize the dmfilestat objects by result_id, we make multiple dbase queries
    but it keeps them organized.  Most times, this will be a single query anyway.
    '''
    dmfilestat_dict = {}
    try:
        # update any dmfilestats in use by running analyses
        update_files_in_use()

        backup_directory = data['backup_dir'] if data['backup_dir'] != 'default' else None

        for resultPK in results_pks.split(','):
            logger.debug("Matching dmfilestats containg %s reportpk" % resultPK)
            dmfilestat_dict[resultPK] = DMFileStat.objects.select_related() \
                .filter(dmfileset__type__in=data['categories'], result__id=int(resultPK))

            # validate export/archive destination folders
            if action in ['export', 'archive']:
                for dmfilestat in dmfilestat_dict[resultPK]:
                    dmactions.destination_validation(dmfilestat, backup_directory, manual_action=True)

            # validate files not in use
            try:
                for dmfilestat in dmfilestat_dict[resultPK]:
                    dmactions.action_validation(dmfilestat, action, data['confirmed'])
            except DMExceptions.FilesInUse as e:
                # warn if exporting files currently in use, allow to proceed if confirmed
                if action=='export':
                    if not data['confirmed']:
                        return HttpResponse(json.dumps({'warning':str(e)+'<br>Exporting now may produce incomplete data set.'}), mimetype="application/json")
                else:
                    raise e
            except DMExceptions.BaseInputLinked as e:
                # warn if deleting basecaller files used in any other re-analysis started from BaseCalling
                if not data['confirmed']:
                    return HttpResponse(json.dumps({'warning':str(e)}), mimetype="application/json")

            # warn if archiving data marked Keep
            if action=='archive' and dmfilestat.getpreserved():
                if not data['confirmed']:
                    return HttpResponse(json.dumps({'warning':'%s currently marked Keep.' % dmfilestat.dmfileset.type}), mimetype="application/json")

        async_task_result = dmtasks.action_group.delay(request.user, data['categories'], action, dmfilestat_dict, data['comment'], backup_directory, data['confirmed'])

        if async_task_result:
            logger.debug(async_task_result)

    except DMExceptions.SrcDirDoesNotExist as e:
        dmfilestat.setactionstate('DD')
        msg = "Source directory %s no longer exists. Setting action_state to Deleted" % e.message
        logger.info(msg)
        EventLog.objects.add_entry(filestat.result, msg, username=request.user.username)
    except Exception as e:
        logger.error("dm_action_selected: error: %s" % str(e))
        return HttpResponseServerError("%s" % str(e))

    test = {'pks':results_pks, 'action':action, 'data':data}
    return HttpResponse(json.dumps(test), mimetype="application/json");


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
    from iondb.utils import devices
    basicChoice = [(None, 'None')]
    for choice in devices.to_media(devices.disk_report()):
        basicChoice.append(choice)
    return tuple(basicChoice)

def _configure_report_data_mgmt(request, pk=None):
    logger = logging.getLogger("data_management")

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
                mbtotal = reportLogStorage.getSavedSpace(day)
            except:
                mbtotal = 0
                logger.error(traceback.format_exc())

            gbtotal = round(mbtotal/(1024), 3)
            result.append([day, mbtotal, gbtotal])

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
    proc_set["RabbitMQ"] = complicated_status("/var/run/rabbitmq/pid", int)

    for node in ['celerybeat', 'celery_w1', 'celery_plugins', 'celery_periodic']:
        proc_set[node] = complicated_status("/var/run/celery/%s.pid" % node, int)
    return sorted(proc_set.items())


@login_required
def preserve_data(request):
    # Sets flag to preserve data for a single DMFileStat object
    if request.method == 'POST':

        reportPK = request.POST.get('reportpk', False)
        expPK = request.POST.get('exppk', False)
        keep = True if request.POST.get('keep')=='true' else False
        dmtype = request.POST.get('type', '')

        if dmtype == 'sig':
            typeStr = dmactions_types.SIG
        elif dmtype == 'base':
            typeStr = dmactions_types.BASE
        elif dmtype == 'out':
            typeStr = dmactions_types.OUT
        elif dmtype == 'intr':
            typeStr = dmactions_types.INTR
        else:
            return HttpResponse(json.dumps({"status": "error, unknown DMFileStat type"}), mimetype="application/json")

        try:
            if reportPK:
                if dmtype == 'sig':
                    results = Results.objects.get(pk=reportPK).experiment.results_set.all()
                else:
                    results = Results.objects.filter(pk=reportPK)
            elif expPK:
                results = Experiment.objects.get(pk=expPK).results_set.all()
            else:
                return HttpResponse(json.dumps({"status": "error, no object pk specified"}), mimetype="application/json")

            for result in results:
                filestat = result.get_filestat(typeStr)
                filestat.setpreserved(keep)
                if keep:
                    msg = '%s marked exempt from auto-action' % typeStr
                else:
                    msg = '%s no longer exempt from auto-action' % typeStr
                EventLog.objects.add_entry(filestat.result, msg, username=request.user.username)
        except Exception as err:
            return HttpResponse(json.dumps({"status": "error, %s" % err}), mimetype="application/json")

        return HttpResponse(json.dumps({"reportPK": reportPK, "type":typeStr, "keep":filestat.getpreserved()}), mimetype="application/json")


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
    p = subprocess.Popen(networkCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        stats_network = stdout.splitlines(True)
    else:
        stats_network = []

    statsCMD = ["/usr/bin/ion_sysinfo"]
    q = subprocess.Popen(statsCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = q.communicate()
    if q.returncode == 0:
        stats = stdout.splitlines(True)
    else:
        stats = []

    # MegaCli64 needs root privilege to access RAID controller - we use a celery task
    # to do the work since they execute with root privilege
    async_result = tasks.get_raid_stats.delay()
    #wait for task to complete
    raid_stats = async_result.get(timeout=20)
    if async_result.failed():
        raid_stats = "get_raid_stats task failed"

    stats_dm = rawDataStorageReport.storage_report()

    # Create filename for the report
    reportFileName = "/tmp/stats_sys.txt"

    # Stuff the variable into the context object
    ctx = Context({"stats_network": stats_network,
                   "stats": stats,
                   "stats_dm": stats_dm,
                   "raid_stats": raid_stats,
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

def raid_info(request):
    # display RAID info from /usr/bin/ion_raidinfo
    import re
    ENTRY_RE = re.compile(r'^(?P<name>[^:]+)[:](?P<value>.*)$')
    def parse(text):
        ret = []
        for line in slot.splitlines():
            match = ENTRY_RE.match(line)
            if match is not None:
                d = match.groupdict()
                #ret[d['name'].strip()] = d['value'].strip()
                ret.append((d['name'].strip(), d['value'].strip()))
        return ret

    # MegaCli64 needs root privilege to access RAID controller - we use a celery task
    # to do the work since they execute with root privilege
    async_result = tasks.get_raid_stats.delay()
    raid_stats = async_result.get(timeout=20)
    if async_result.failed():
        raid_stats = "get_raid_stats task failed"

    raid_stats = ' '.join(raid_stats).split("=====")
    stats = []
    if len(raid_stats) > 1:
        for slot in raid_stats:
            parsed_stats = parse(slot)
            if len(parsed_stats) > 0:
                stats.append(parsed_stats)

    ctx = RequestContext(request, {"raid_stats": stats})
    return render_to_response("rundb/configure/configure_raid_info.html", context_instance=ctx)

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

@login_required
def configure_report_data_mgmt_editPruneGroups(request):
    logger = logging.getLogger("data_management")

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
    logger = logging.getLogger("data_management")

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
    logger = logging.getLogger("data_management")

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
        return HttpResponseServerError(traceback.format_exc())


def get_ampliseq_designs(user, password):
    h = httplib2.Http(disable_ssl_certificate_validation=settings.DEBUG)
    h.add_credentials(user, password)
    url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/design/list")
    response, content = h.request(url)
    if response['status'] == '200':
        data = json.loads(content)
        designs = data.get('AssayDesigns', [])
        for design in designs:
            solutions = []
            for solution in design.get('DesignSolutions', []):
                solutions.append(ampliseq.handle_versioned_plans(solution)[1])
            design['DesignSolutions'] = solutions
        return response, designs
    else:
        return response, {}

def get_ampliseq_fixed_designs(user, password):
    h = httplib2.Http(disable_ssl_certificate_validation=settings.DEBUG)
    h.add_credentials(user, password)
    url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/tmpldesign/list/active")
    response, content = h.request(url)
    logger.warning(response)
    if response['status'] == '200':
        data = json.loads(content)
        fixed = [ampliseq.handle_versioned_plans(d)[1] for d in data.get('TemplateDesigns', [])]
        return response, fixed
    else:
        return response, None


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
                    ctx['http_error'] = "Your user name or password is invalid."
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
                        if solution.get('ordered', False):
                            ctx['ordered_solutions'].append((design, solution))
                        else:
                            ctx['unordered_solutions'].append((design, solution))
                ctx['designs_pretty'] = json.dumps(designs, indent=4, sort_keys=True)
                ctx['fixed_designs_pretty'] = json.dumps(fixed, indent=4, sort_keys=True)
    ctx['form'] = form
    return render_to_response("rundb/configure/ampliseq.html", ctx,
        context_instance=RequestContext(request))


@login_required
def configure_ampliseq_download(request):
    if 'ampliseq_username' in request.session:
        username = request.session['ampliseq_username']
        password = request.session['ampliseq_password']
        for ids in request.POST.getlist("solutions"):
            design_id, solution_id = ids.split(",")
            meta = '{}'
            start_ampliseq_solution_download(design_id, solution_id, meta, (username, password))
        for ids in request.POST.getlist("fixed_solutions"):
            design_id, reference = ids.split(",")
            meta = '{"reference":"%s"}' % reference.lower()
            start_ampliseq_fixed_solution_download(design_id, meta, (username, password))

    if request.method == "POST":
        return HttpResponseRedirect(urlresolvers.reverse("configure_ampliseq_download"))

    downloads = DownloadMonitor.objects.filter(tags__contains="ampliseq_template").order_by('-created')
    for download in downloads:
        if download.size:
            download.percent_progress = "{0:.2%}".format(float(download.progress) / download.size)
        else:
            download.percent_progress = "..."
    ctx = {
        'downloads': downloads
    }
    return render_to_response("rundb/configure/ampliseq_download.html", ctx,
        context_instance=RequestContext(request))

def start_ampliseq_solution_download(design_id, solution_id, meta, auth):
    url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/design/{0}/solutions/{1}/download/results".format(design_id, solution_id))
    monitor = DownloadMonitor(url=url, tags="ampliseq_template", status="Queued")
    monitor.save()
    t = tasks.download_something.apply_async(
        (url, monitor.id), {"auth": auth},
        link=tasks.ampliseq_zip_upload.subtask((meta,)))

def start_ampliseq_fixed_solution_download(design_id, meta, auth):
    url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/tmpldesign/{0}/download/results".format(design_id))
    monitor = DownloadMonitor(url=url, tags="ampliseq_template", status="Queued")
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
