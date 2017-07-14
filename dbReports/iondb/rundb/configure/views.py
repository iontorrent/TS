# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import xmlrpclib
import subprocess
import socket
import logging
import os
import json
import traceback
import stat
import csv
import codecs
import re
import httplib2
import urlparse
import datetime
from django.utils import timezone
import celery.exceptions
from django.conf import settings
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.admin.views.decorators import staff_member_required
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
from iondb.rundb.models import dnaBarcode, Plugin, PluginResult, PluginResultJob, GlobalConfig, Chip, AnalysisArgs, \
                               EmailAddress, Publisher, Results, Template, UserProfile, model_to_csv, IonMeshNode, \
                               FileMonitor, ContentType, Cruncher, SharedServer, RunType, RUNNING_STATES
from iondb.rundb.data import rawDataStorageReport
from iondb.rundb.configure.genomes import search_for_genomes
from iondb.rundb.plan import ampliseq
from iondb.rundb.plan.ampliseq import AmpliSeqPanelImport
# Handles serialization of decimal and datetime objects
from django.core.serializers.json import DjangoJSONEncoder
from iondb.rundb.configure.util import plupload_file_upload
from ion.utils import makeCSA
from ion.utils import makeSSA
from django.core import urlresolvers
from django.template.loader import get_template
from iondb.utils.raid import get_raid_status, load_raid_status_json
from iondb.servelocation import serve_wsgi_location
from ion.plugin.remote import call_pluginStatus
from iondb.utils.utils import convert, cidr_lookup
from iondb.bin.IonMeshDiscoveryManager import IonMeshDiscoveryManager
from iondb.rundb.configure import updateProducts
from iondb.utils.nexenta_nms import this_is_nexenta, get_all_torrentnas_data, has_nexenta_cred
from iondb.rundb.configure import ampliseq_design_parser
logger = logging.getLogger(__name__)

from json import encoder
encoder.FLOAT_REPR = lambda x: format(x, '.15g')

ionMeshDiscoveryManager = IonMeshDiscoveryManager()


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
    return render_to_response("rundb/configure/about.html", context_instance=ctx)


@login_required
def configure_ionreporter(request):
    iru = Plugin.objects.get(name__iexact='IonReporterUploader', active=True)
    ctxd = {"iru": iru}

    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/ionreporter.html", context_instance=ctx)


def timeout_raid_info_json():
    '''Call celery task to query RAID card status'''
    raidinfo = None
    err_msg = None
    try:
        async_result = tasks.get_raid_stats_json.delay()

        raidinfo = async_result.get(timeout=45)
        if async_result.failed():
            err_msg = "RAID status check failed."
            logger.error(err_msg)
            raidinfo = None
    except celery.exceptions.TimeoutError as err:
        err_msg = "RAID status check timed out, taking longer than 45 seconds."
        logger.warning(err_msg)
    except Exception as err:
        err_msg = "Failed RAID status check: %s" % err
        logger.error(err_msg)

    return raidinfo, err_msg


def sort_drive_array_for_display(raidstatus):
    '''sorts array to be displayed matching physical arrangement in enclosure'''
    slots = 12
    for drv in raidstatus:
        try:
            adapter_id = drv.get('adapter_id', '')
            drives = drv.get('drives', [])
            if not adapter_id or len(drives) < slots:
                continue

            if adapter_id.startswith("PERC H710"):
                drv['cols'] = 4
            elif adapter_id.startswith("PERC H810"):
                ncols = 4
                temp = [drives[i:i+(slots/ncols)] for i in range(0, slots, slots/ncols)]
                drv['drives'] = sum(map(list, zip(*temp)), [])
                drv['cols'] = ncols
        except:
            pass


def get_torrent_nas_info():
    # Torrent NAS information

    def get_health_state(health_str):
        # web page will use color to highlight states
        _ERROR_STATES = ["FAULTED", "UNAVAIL"]
        _WARN_STATES = ["DEGRADED"]

        state = 'good'
        if health_str in _ERROR_STATES:
            state = 'error'
        elif health_str in _WARN_STATES:
            state = 'warning'
        return state

    info = []
    data, err = get_all_torrentnas_data()
    for error in err:
        logger.error(error)
    for nas in data:
        nasInfo = { "ipaddress": nas["ipaddress"], "volumes": [] }
        for name in nas['volumes']:
            volume = dict(nas[name])
            volume['name'] = name

            # volume health info and states
            volume['health'] = nas[name]['state'][0]
            volume['state'] = get_health_state(volume['health'])

            # info per disk
            volume['disks'] = []
            config = volume.pop('config')
            for line in config[3:]:
                l = re.compile(r'\W+').split(line.strip(), 5)
                volume['disks'].append({
                    'name': l[0],
                    'health': l[1],
                    'state': get_health_state(l[1]),
                    'err': ' / '.join([l[2],l[3],l[4]]),
                    'info': l[5] if len(l) > 5 else ''
                })

            # extra info to display on page
            extra = []
            if volume.get('status'):
                extra.append(' '.join(volume['status']) )
            if volume['errors'][0] != "No known data errors":
                extra.append(' '.join(volume['errors']) )
            if volume.get('action'):
                extra.append(' '.join(volume['action']) )
            volume['extra_info'] = '<br>'.join(extra)
            # make html links
            links  = re.compile(r'(https?://[^\s]+)')
            volume['extra_info'] = links.sub(r'<a href="\1" target="_blank">\1</a>', volume['extra_info'])

            nasInfo["volumes"].append(volume)
        info.append(nasInfo)

    return info


def torrent_nas_section(request):
    try:
        nas_info = get_torrent_nas_info()
    except Exception as e:
        logger.error(traceback.format_exc())
        return HttpResponseServerError("Failed to get Torrent NAS Info: %s" % repr(e))
    else:
        ctx = RequestContext(request, {"nas_info": nas_info})
        return render_to_response("rundb/configure/services_torrentNAS.html", context_instance=ctx)


@login_required
def configure_services(request):
    '''Render the service tab'''
    servers = get_servers()
    jobs = current_jobs() + current_plugin_jobs()
    crawler = _crawler_status(request)
    processes = process_set()

    # RAID Info
    raidJson = load_raid_status_json()
    raid_status = raidJson.get('raid_status')
    raid_status_updated = raidJson.get('date')

    if raid_status:
        sort_drive_array_for_display(raid_status)

    ctxd = {
        "processes": processes,
        "servers": servers,
        "jobs": jobs,
        "crawler": crawler,
        "raid_status": raid_status,
        "raid_status_updated": raid_status_updated,
        'crunchers': Cruncher.objects.all().order_by('name'),
        'show_nas_info': has_nexenta_cred()
        }
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/services.html", context_instance=ctx)


@login_required
def configure_references(request):
    '''Render reference tab'''
    search_for_genomes()
    ctx = RequestContext(request)
    return render_to_response("rundb/configure/references.html", context_instance=ctx)


@login_required
def configure_plugins(request):
    '''Render plugin tab'''
    # Rescan Plugins
    ## Find new, remove missing plugins

    # force a refresh of the cache
    pluginServer = xmlrpclib.ServerProxy(settings.IPLUGIN_STR)
    pluginServer.GetSupportedPlugins(list(), True, False)

    # Rescan Publishers
    publishers.search_for_publishers()
    pubs = Publisher.objects.all().order_by('name')

    #hide the "Upload" button if upload.html doesn't exists
    for p in pubs:
        p.uploadLink = True
        if not os.path.exists(os.path.join(p.path,"upload.html")):
            p.uploadLink = False

    # perform a check here to see if the user is a super user, but really we should be looking for a more refined permission set
    # but the interface needs to be able to support refined permissions before this happens
    can_upgrade = request.user.is_superuser

    ctx = RequestContext(request, {"publishers": pubs, "can_upgrade": can_upgrade})
    return render_to_response("rundb/configure/plugins.html", context_instance=ctx)


@login_required
def configure_plugins_plugin_install_to_version(request, pk, version):
    """
    View rendering for the install to version modal interface
    :param request: Request
    :param pk: The primary key for the plugin
    :param version: The version to install
    :return: a renderer
    """

    plugin = get_object_or_404(Plugin, pk=pk)
    action = reverse('api_dispatch_install_to_version', kwargs={'api_name': 'v1', 'resource_name': 'plugin', 'pk': pk, 'version': version})
    ctx = RequestContext(request, {
        "id": pk, "method": "POST", 'methodDescription': 'Upgrade', "readonly": False, 'InstallVersion': version, 'action': action, 'plugin': plugin
    })
    return render_to_response("rundb/configure/modal_confirm_plugin_install_to_version.html", context_instance=ctx)


@login_required
def configure_plugins_plugin_install(request):
    '''Render plugin install tab'''
    ctxd = {
        "what": "plugin",
        "file_filters": [("deb", "Debian Package"), ("zip", "Compressed Zip files")],
        "pick_label": "a Plugin File",
        "plupload_url": reverse("configure_plugins_plugin_zip_upload"),
        "install_url": reverse('api_dispatch_install', kwargs={'api_name': 'v1', 'resource_name': 'plugin'})
    }
    return render_to_response("rundb/configure/modal_plugin_or_publisher_install.html", context_instance=RequestContext(request, ctxd))


@login_required
def configure_publisher_install(request):
    '''Render publisher install tab'''
    ctxd = {
        "what": "publisher",
        "file_filters": [("zip", "Compressed Zip files")],
        "pick_label": "a Publisher ZIP File",
        "plupload_url": reverse("configure_plugins_plugin_zip_upload"),
        "install_url": reverse('api_dispatch_install', kwargs={'api_name': 'v1', 'resource_name': 'publisher'})
    }
    return render_to_response("rundb/configure/modal_plugin_or_publisher_install.html", context_instance=RequestContext(request, ctxd))


@login_required
def configure_plugins_plugin_upgrade(request, pk):
    """
    This method will take a web request to upgrade a plugin
    """

    plugin = get_object_or_404(Plugin, pk=pk)
    action = reverse('api_dispatch_upgrade', kwargs={'api_name': 'v1', 'resource_name': 'plugin', 'pk': pk})
    ctx = RequestContext(request, {
        "id": pk, "method": "POST", 'methodDescription': 'Upgrade', "readonly": False, 'UpgradeTo': plugin.availableVersions[-1], 'action': action, 'plugin': plugin
    })
    return render_to_response("rundb/configure/modal_confirm_plugin_upgrade.html", context_instance=ctx)


@login_required
def configure_plugins_plugin_configure(request, action, pk):
    """
    load files into iframes
    """

    def openfile(fname):
        """strip lines """
        try:
            fhandle = codecs.open(fname, 'r', 'utf-8')
        except:
            logger.exception("Failed to open '%s'", fname)
            return False
        content = fhandle.read()
        fhandle.close()
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

    applicationGroup = request.GET.get('applicationGroup', "")
    runTypeId = request.GET.get('runTypeId', "")
    runType = ""
    plan_json = {}
    if runTypeId:
        runType_obj = get_object_or_404(RunType, pk = runTypeId)
        runType = runType_obj.runType
    plan_json = json.dumps({'applicationGroup' : applicationGroup, 'runType' : runType}, cls=DjangoJSONEncoder)

    ctxd = {"plugin": plugin_json, "file": content, "report": report, "tmap": str(index_version),
            "results": results_json, "plan" : plan_json, "action": action}

    context = RequestContext(request, ctxd)

    # modal_configure_plugins_plugin_configure.html will be removed in a newer release as it places a script outside the
    # html tag.
    # Add in the js vars just before the closing head tag.
    plugin_js_script = get_template("rundb/configure/plugins/plugin_configure_js.html").render(context).replace("\n",
                                                                                                                "")
    context["file"] = context["file"].replace("</head>", plugin_js_script + "</head>", 1)


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
    servepath = "plugins/" + fspath[len(prefix):] + "/pluginMedia/" + path
    # example: "/plugins/" + "examplePlugin" + "/" + "pluginMedia/img.png"
    # example: "/plugins/" + "instances/12345667/exampleZeroConf" + "/" + "pluginMedia/img.png"
    logger.debug("Redirecting pluginMedia request '%s' to '%s'", path, servepath)

    # Serve redirects to /private/plugins/,
    # which is mapped to /results/plugins/ in apache
    return serve_wsgi_location(request, servepath)


@login_required
def configure_plugins_plugin_uninstall(request, pk):
    """
    Disables a plugin from the system
    :param request:
    :param pk: The primary key of the plugin to be disabled
    :return:
    """
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
    except (TypeError, ValueError):
        return HttpResponseNotFound

    plugin = get_object_or_404(Plugin, pk=pk)
    plugin.selected = bool(int(set))
    plugin.save()
    return HttpResponse()


@login_required
def configure_plugins_plugin_default_selected(request, pk, set):
    """Allow user to enable a plugin for use in the analysis"""
    try:
        pk = int(pk)
    except (TypeError, ValueError):
        return HttpResponseNotFound

    plugin = get_object_or_404(Plugin, pk=pk)
    plugin.defaultSelected = bool(int(set))
    plugin.save()
    return HttpResponse()


@login_required
def configure_plugins_plugin_refresh(request, pk):
    plugin = get_object_or_404(Plugin, pk=pk)
    url = reverse('api_dispatch_info', kwargs={'resource_name': 'plugin', 'api_name': 'v1', 'pk': int(pk)})
    url += '?use_cache=false'
    ctx = RequestContext(request, {'plugin': plugin, 'action': url, 'method': 'get'})
    return render_to_response("rundb/configure/plugins/modal_refresh.html", context_instance=ctx)


@login_required
def configure_plugins_plugin_usage(request, pk):
    plugin = get_object_or_404(Plugin, pk=pk)
    pluginresults = plugin.pluginresult_set
    ctx = RequestContext(request, {'plugin': plugin, 'pluginresults': pluginresults})
    return render_to_response("rundb/configure/plugins/plugin_usage.html", context_instance=ctx)


@login_required
def configure_configure(request):
    """
    Handles the render and post for server configuration
    """
    ctx = RequestContext(request, {})
    emails = EmailAddress.objects.all().order_by('pk')
    enable_nightly = GlobalConfig.get().enable_nightly_email
    ctx.update({"email": emails, "enable_nightly": enable_nightly})
    config_contacts(request, ctx)
    config_site_name(request, ctx)

    # handle post request for setting a time zone
    if request.method == "POST" and "zone_select" in request.POST:
        try:
            selected_timezone = request.POST["zone_select"] + '/' + request.POST["city_select"]
            subprocess.check_call(['sudo', 'timedatectl', 'set-timezone', selected_timezone])
            logger.info("Successfully set the timezone to " + selected_timezone)
        except subprocess.CalledProcessError as exc:
            logger.error(str(exc))
            raise Http404()

    # handle the request for the timezones
    else:
        get_timezone_info(request, ctx)

    # get current list of
    return render_to_response("rundb/configure/configure.html", context_instance=ctx)


@permission_required('user.is_staff')
def configure_mesh(request):
    """
    Handles view requests for the ion mesh configuration page.
    :param request: The http request
    :return: A http response
    """

    ctx=RequestContext(request, {
        "mesh_computers": sorted(ionMeshDiscoveryManager.getMeshComputers()),
        "ion_mesh_nodes_table": IonMeshNode.objects.all(),
        "system_id": settings.SYSTEM_UUID
    })

    return render_to_response("rundb/configure/configure_mesh.html", context_instance=ctx)


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


def seconds2htime(_seconds):
    """Convert a number of seconds to a dictionary of days, hours, minutes,
    and seconds.
    >>> seconds2htime(90061)
    {"days":1,"hours":1,"minutes":1,"seconds":1}
    """
    days = int(_seconds / (24 * 3600))
    _seconds -= days * 24 * 3600
    hours = int(_seconds / 3600)
    _seconds -= hours * 3600
    minutes = int(_seconds / 60)
    _seconds -= minutes * 60
    _seconds = int(_seconds)
    return {"days": days, "hours": hours, "minutes": minutes, "seconds": _seconds}


def get_servers():

    # attempt to get an ipaddress for this server by attempting to open a connection to another ip address but fallback to the older methods
    try:
        ipaddress = [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
    except:
        ipaddress = socket.gethostbyname(socket.gethostname())

    jservers = [(socket.gethostname(), ipaddress)]
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
                'term_url': reverse('control_job', args=(pk, 'term'))
            })
    except (socket.error, xmlrpclib.Fault):
        pass

    return jobs


def current_plugin_jobs():
    """
    Get list of active pluginresults from database then connect to ionPlugin and get drmaa status per jobid.
    """
    jobs = []
    running = PluginResultJob.objects.filter(state__in=RUNNING_STATES).order_by('pk')
    if running:
        # get job status from drmaa
        jobids = running.values_list('grid_engine_jobid', flat=True)
        try:
            job_status = call_pluginStatus(list(jobids))
            for i, prj in enumerate(list(running)):
                if job_status[i] not in ['DRMAA BUG', 'job finished normally']:
                    jobs.append({
                        'name': prj.plugin_result.plugin.name+"-"+prj.run_level,
                        'resultsName': prj.plugin_result.result.resultsName,
                        'pid': prj.grid_engine_jobid,
                        'type': 'plugin',
                        'status': job_status[i],
                        'pk': prj.plugin_result.pk,
                        'report_exist': True,
                        'report_url': reverse('report', args=(prj.plugin_result.result.pk,)),
                        'term_url': "/rundb/api/v1/pluginresult/%d/stop/" % prj.plugin_result.pk
                    })
        except:
            pass

    return jobs


def process_set():
    def process_status(process):
        return subprocess.Popen("service %s status" % process, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def simple_status(name):
        proc = process_status(name)
        stdout, stderr = proc.communicate()
        logger.debug("%s out = '%s' err = %s''" % (name, stdout, stderr))
        return proc.returncode == 0

    def upstart_status(name):
        # Upstart jobs status command always returns 0.
        proc = subprocess.Popen("service %s status" % name, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        logger.debug("%s out = '%s' err = %s''" % (name, stdout, stderr))
        return "start/running" in stdout

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

    processes = [
        "ionJobServer",
        "ionCrawler",
        "ionPlugin",
        "RSM_Launch",
        "ntp"
    ]

    proc_set = {}
    for name in processes:
        proc_set[name] = simple_status(name)

    # get the DjangoFTP status
    proc_set['DjangoFTP'] = upstart_status("DjangoFTP")

    # get the dhcp service status
    proc_set['dhcp'] = simple_status('isc-dhcp-server')

    # get the tomcat service status
    proc_set["tomcat"] = complicated_status("/var/run/tomcat7.pid", int)

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
def references_barcodeset(request, barcodesetid):
    barCodeSetName = get_object_or_404(dnaBarcode, pk=barcodesetid).name
    ctx = RequestContext(request, {'name': barCodeSetName, 'barCodeSetId': barcodesetid})
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

        #Trim off barcode and barcode kit leading and trailing spaces and update the log file if exists
        if ((len(name) - len(name.lstrip())) or (len(name) - len(name.rstrip()))):
            name = name.strip()
            logger.warning("The Barcode Set Name (%s) contains Leading/Trailing spaces and got trimmed." % name)

        barCodeSet = dnaBarcode.objects.filter(name=name)
        if barCodeSet:
            return HttpResponse(json.dumps({"status": "Error: Barcode set with the same name already exists"}), mimetype="text/html")

        expectedHeader = ["id_str", "type", "sequence", "floworder", "index", "annotation", "adapter"]

        barCodes = []
        failed = {}
        nucs = ["sequence", "floworder", "adapter"]  # fields that have to be uppercase
        reader = csv.DictReader(postedfile.read().splitlines())
        for index, row in enumerate(reader, start=1):
            invalid = _validate_barcode(row, barcodeSetName=name)
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
                    if key == "type":
                        value = value.lower()
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


def _validate_barcode(barCodeDict, barcodeSetName=None):
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
        barcode = barCodeDict["id_str"]
         #Trim off barcode and barcode kit leading and trailing spaces and update the log file if exists
        if ((len(barcode) - len(barcode.lstrip())) or (len(barcode) - len(barcode.rstrip()))):
            logger.warning("The BarcodeName (%s) of BarcodeSetName (%s) contains Leading/Trailing spaces and got trimmed" % (barcode, barcodeSetName))
            barcode = barcode.strip()
        if not set(barcode).issubset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"):
            failed.append(("id_str", "str_id must only have letters, numbers, or the characters _ . - "))
    #do not let the index be set to zero. Zero is reserved.
    if 'index' in barCodeDict:
        if barCodeDict["index"] == "0":
            failed.append(("index", "index must not contain a 0. Indices should start at 1."))
    return failed


def reference_barcodeset_csv(request, barcodesetid):
    """Get a csv for a barcode set"""
    barcodeset_name = get_object_or_404(dnaBarcode, pk=barcodesetid).name
    filename = '%s_%s' % (barcodeset_name, str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    barcode_csv_filename = filename.replace(" ", "_")
    response = HttpResponse(mimetype='text/csv')
    response['Content-Disposition'] = 'attachment; filename=%s.csv' % barcode_csv_filename
    response.write(model_to_csv(dnaBarcode.objects.filter(name=barcodeset_name), ['index', 'id_str', 'sequence', 'annotation', 'adapter']))
    return response

@login_required
def references_barcodeset_delete(request, barcodesetid):
    barCodeSetName = get_object_or_404(dnaBarcode, pk=barcodesetid).name
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
            "id": barCodeSetName, "ids": json.dumps(pks), "method": "POST", 'methodDescription': 'Delete', "readonly": False, 'type': type, 'action': reverse('references_barcodeset_delete', args=[barcodesetid, ]), 'actions': json.dumps(actions)
        })
        return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)


@login_required
def references_barcode_add(request, barcodesetid):
    return references_barcode_edit(request, barcodesetid, None)


@login_required
def references_barcode_edit(request, barcodesetid, pk):
    dna = get_object_or_404(dnaBarcode, pk=barcodesetid)
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
def references_barcode_delete(request, barcodesetid, pks):
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
    networkCMD = "/usr/bin/ion_netinfo"
    logger.info("Calling netinfo script")
    p = subprocess.Popen(networkCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        stats_network = stdout.splitlines(True)
    else:
        stats_network = []

    statsCMD = "/usr/bin/ion_sysinfo"
    logger.info("Calling ion_sysinfo script")
    q = subprocess.Popen(statsCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = q.communicate()
    if q.returncode == 0:
        stats = stdout.splitlines(True)
    else:
        stats = []

    # MegaCli64 needs root privilege to access RAID controller - we use a celery task
    # to do the work since they execute with root privilege
    logger.info("Calling timeout_raid_info")
    raidCMD = "sudo /usr/bin/ion_raidinfo"
    q = subprocess.Popen(raidCMD, shell=True, stdout=subprocess.PIPE)
    stdout, stderr = q.communicate()
    raid_stats = stdout if q.returncode == 0 else stderr

    logger.info("Calling storage_report")
    stats_dm = rawDataStorageReport.storage_report()

    # read in the contents of environment
    environment_file = '/etc/environment'
    environment = list()
    environment.append('==================================================================')
    environment.append('Environment (/etc/environment)')
    environment.append('==================================================================')
    if os.path.exists(environment_file):
        with open(environment_file, 'r') as environment_handle:
            environment += environment_handle.readlines()

    # Create filename for the report
    reportFileName = "/tmp/stats_sys.txt"

    # Stuff the variable into the context object
    ctx = Context({"stats_network": stats_network,
                   "stats": stats,
                   "stats_dm": stats_dm,
                   "raid_stats": raid_stats,
                   "environment": "\n".join([line.strip() for line in environment]),
                   "reportFilePath": reportFileName,
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
    for line in environment:
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


def raid_info_refresh(request):
    try:
        raidinfo = tasks.get_raid_stats_json.delay().get(timeout=45)
        tasks.post_check_raid_status(raidinfo)
    except celery.exceptions.TimeoutError as err:
        return HttpResponseServerError("RAID status check timed out, taking longer than 45 seconds.")
    except Exception as err:
        logger.error(traceback.format_exc())
        return HttpResponseServerError("Failed RAID status check: %s" % err)

    return HttpResponse()


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


def get_timezone_info(request, context):
    lines = ''
    with open ('/etc/timezone', 'r') as file:
        lines = [line.rstrip('\n') for line in file]
    TZ = ''.join(lines)
    current_zone, current_city = TZ.split("/", 1)
    context.update({'current_TZ': {"zones": current_zone, "cities": current_city}})
    all_zones = []
    all_cities = []
    zones = os.listdir('/usr/share/zoneinfo')
    for zone in zones:
        zonedir = os.path.join('/usr/share/zoneinfo/' + zone)
        if os.path.isdir(zonedir):
            all_zones.append(zone)
            if zone == context['current_TZ']['zones']:
                cities = os.listdir(zonedir)
                for city in cities:
                    citydir = os.path.join(zonedir + '/' + city)
                    if os.path.isfile(citydir):
                        all_cities.append(city)
                    elif os.path.isdir(citydir):
                        inner_dir = os.listdir(citydir)
                        for inner_city in inner_dir:
                            all_cities.append(city + '/' + inner_city)
    all_zones.sort()
    all_cities.sort()
    context.update({'all_TZ':  {"zones": all_zones, "cities": all_cities}})


def auto_detect_timezone(request):
    from urllib2 import urlopen
    from contextlib import closing

    current_zone = []
    current_city = []
    url = 'http://geoip.nekudo.com/api'
    try:
        with closing(urlopen(url, timeout=1)) as response:
            location = json.loads(response.read())
            timezone = location['location']['time_zone']
            current_zone, current_city = timezone.split("/", 1)
    except:
        return HttpResponse("error", status=404)
    return  HttpResponse(json.dumps({"current_zone": [current_zone], "current_city": [current_city]}), mimetype="application/json")


def get_all_cities(request, zone=None):
    if not zone:
        return HttpResponse("error", status=404)
    all_cities = []
    cities = os.listdir(os.path.join('/usr/share/zoneinfo/' + zone))

    for city in cities:
        if os.path.isfile(os.path.join('/usr/share/zoneinfo/' + zone + '/' + city)):
            all_cities.append(city)
    all_cities.sort()
    return  HttpResponse(json.dumps({"cities": all_cities}), mimetype="application/json")


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
                if block == 'merge' and job == 'merge':
                    block_status = 'done'
                elif job == 'alignment':
                    block_status = 'done'

        jobs[block]['status'] = block_status

    # summary count how many blocks in each category
    #summary_keys = ['pending', 'sigproc', 'basecaller', 'alignment', 'done']
    summary_keys = ['pending', 'block_processing', 'done']
    summary_values = [0] * len(summary_keys)
    num_blocks = len(jobs) - 1  # don't count merge block
    for block in jobs:
        if block != 'merge' and jobs[block]['status'] in summary_keys:
            indx = summary_keys.index(jobs[block]['status'])
            summary_values[indx] += 1

    context = RequestContext(request, {'TLS_jid': jid, 'jobs': jobs, 'summary': zip(summary_keys, summary_values), 'num_blocks': num_blocks})
    return render_to_response('rundb/configure/services_jobDetails.html', context_instance=context)


@login_required
def queueStatus(request):
    # get cluster queue status
    args = ['qstat', '-g', 'c', '-ext']
    output = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = output.stdout.readlines()
    queues = []
    for line in stdout:
        splitline = line.split()
        if len(splitline) > 1 and '.q' in splitline[0]:
            queues.append({
                'name': splitline[0],
                'pending': 0,
                'used': splitline[2],
                'avail': splitline[4],
                'error': splitline[18],
                'total': splitline[5],
            })

    # get pending jobs per queue
    args = ['qstat', '-u', 'www-data', '-q']
    for queue in queues:
        output = subprocess.Popen(args + [queue['name']], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = output.stdout.readlines()
        for line in stdout:
            splitline = line.split()
            if splitline[0].isdigit() and 'qw' in splitline[4]:
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
        path, name = makeSSA.makeSSA()
        response = HttpResponse(FileWrapper(open(path)),
                                    mimetype='application/zip')
        response['Content-Disposition'] = 'attachment; filename=%s' % name
        return response
    except:
        logger.exception("Failed to create System Support Archive")
        return HttpResponseServerError(traceback.format_exc())

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
    ctx = {'designs': None}
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
        # Get the list of ampliseq designs panels both custom and fixed for display
        username = request.session['ampliseq_username']
        password = request.session['ampliseq_password']
        try:
            response, ctx = ampliseq_design_parser.get_ampliseq_designs(username, password, pipeline, ctx)
            if response['status'].startswith("40"):
                request.session.pop('ampliseq_username')
                request.session.pop('ampliseq_password')
                ctx['http_error'] = 'Your user name or password is invalid.<br> You may need to log in to ' \
                                    '<a href="https://ampliseq.com/">AmpliSeq.com</a> and check your credentials.'
                fixed = None

            else:
                ctx = ampliseq_design_parser.get_ampliseq_fixed_designs(username, password, pipeline, ctx)
                fixed = ctx['fixed_solutions']
        except httplib2.HttpLib2Error as err:
            logger.error("There was a connection error when contacting ampliseq: %s" % err)
            ctx['http_error'] = "Could not connect to AmpliSeq.com"
            fixed = None
        if fixed:
            form = None

    ctx['form'] = form
    ctx['ampliseq_account_update'] = timezone.now() < timezone.datetime(2013, 11, 30, tzinfo=timezone.utc)
    ctx['ampliseq_url'] = settings.AMPLISEQ_URL

    s5_chips = Chip.objects.filter(isActive=True, name__in=["510", "520", "530", "540"], instrumentType= "S5").values_list('name', flat=True).order_by('name')
    ctx['s5_chips'] = s5_chips
    
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
    tasks.download_something.apply_async(
        (url, monitor.id), {"auth": auth},
        link=tasks.ampliseq_zip_upload.subtask((meta,)))


def start_ampliseq_fixed_solution_download(design_id, meta, auth):
    url = urlparse.urljoin(settings.AMPLISEQ_URL, "ws/tmpldesign/{0}/download/results".format(design_id))
    monitor = FileMonitor(url=url, tags="ampliseq_template", status="Queued")
    monitor.save()
    tasks.download_something.apply_async(
        (url, monitor.id), {"auth": auth},
        link=tasks.ampliseq_zip_upload.subtask((meta,)))


@login_required
def cache_status(request):
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
        _, key, value = line
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
        thandle = tasks.check_cluster_status()
        thandle.get(timeout=300)
        tasks.post_run_nodetests()
    except:
        return HttpResponseServerError(traceback.format_exc())
    return HttpResponse()


@login_required
@staff_member_required
def cluster_ctrl(request, name, action):
    error = tasks.cluster_ctrl_task.delay(action, name, request.user.username).get(timeout=20)
    if error:
        return HttpResponse(json.dumps({"error": error}), mimetype="application/json")

    return HttpResponse(json.dumps({"status": "%s is %sd" % (name, action.capitalize())}), mimetype="application/json")


def cluster_info_log(request, pk):
    nodes = Cruncher.objects.all()
    ct_obj = ContentType.objects.get_for_model(Cruncher)
    title = "Cluster Info log for %s" % nodes.get(pk=pk).name

    ctx = RequestContext(request, {"title": title, "pk": pk, "cttype": ct_obj.id})
    return render_to_response("rundb/common/modal_event_log.html", context_instance=ctx)


def cluster_info_history(request):
    nodes = Cruncher.objects.all().values_list('name', flat=True)
    ctx = RequestContext(request, {'nodes': nodes})
    return render_to_response("rundb/configure/clusterinfo_history.html", context_instance=ctx)


@login_required
def configure_analysisargs(request):
    chips = Chip.objects.filter(isActive=True).order_by("name")
    ctx = RequestContext(request, {'chips': chips})
    return render_to_response("rundb/configure/manage_analysisargs.html", context_instance=ctx)


@login_required
def configure_analysisargs_action(request, pk, action):
    if pk:
        obj = get_object_or_404(AnalysisArgs, pk=pk)
    else:
        obj = AnalysisArgs(name='new_parameters')

    if request.method == 'GET':
        chips = Chip.objects.filter(isActive=True).order_by("name")
        args_for_uniq_validation = AnalysisArgs.objects.all()
        display_name = "%s (%s)" % (obj.description, obj.name) if obj else ''
        if action == "copy":
            obj.name = obj.description = ''
        elif action == "edit":
            args_for_uniq_validation = args_for_uniq_validation.exclude(pk=obj.pk)

        ctxd = {
            'obj': obj,
            'chips': chips,
            'args_action': action,
            'display_name': display_name,
            'args_for_uniq_validation': args_for_uniq_validation
            #'uniq_names': json.dumps(args_for_uniq_validation.values_list('name', flat=True)),
            #'uniq_descriptions': json.dumps(args_for_uniq_validation.values_list('description', flat=True))
        }
        return render_to_response("rundb/configure/modal_analysisargs_details.html", context_instance=RequestContext(request, ctxd))

    elif request.method == 'POST':
        params = request.POST.dict()
        params['isSystem'] = params['chip_default'] = False
        params['lastModifiedUser'] = request.user
        if action == 'copy':
            params['creator'] = request.user
            obj.pk = None

        for key, val in params.items():
            setattr(obj, key, val)
        obj.save()
        return HttpResponseRedirect(urlresolvers.reverse("configure_analysisargs"))


def get_local_ip():
    '''Returns an array of IP addresses used by this host'''
    ip_array = []
    cmd = "ifconfig"
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = p.communicate()
    if p.returncode == 0:
        for line in [line for line in stdout.splitlines() if "inet addr" in line]:
            address = line.split(':', 1)[1].split()[0].strip()
            ip_array.append(address)
    return ip_array


def get_nas_devices(request):
    '''Returns json object containing list of IP addresses/hostnames of direct-connected
    NAS devices'''
    # ===== get local instrument network ports from dhcp.conf =====
    addresses = []
    with open('/etc/dhcp/dhcpd.conf') as fh:
        for subnet in [line for line in fh.readlines() if line.startswith('subnet')]:
            addresses.append([subnet.split()[1].strip(), subnet.split()[3].strip()])

    ## ===== Include LAN subnet =====
    #import socket, struct
    #"""Read the default gateway directly from /proc."""
    #with open("/proc/net/route") as fh:
    #    for line in fh:
    #        fields = line.strip().split()
    #        if fields[1] != '00000000' or not int(fields[3], 16) & 2:
    #            continue
    #        else:
    #            lan_address = socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    ## Append the LAN address
    #addresses.append(lan_address)

    logger.info("scanning these subnets for NAS: %s", addresses)

    # ===== Get all servers in given address range =====
    devices = []
    for address in addresses:
        # nmap cmdline returns addresses of given range.  Output:
        # "Nmap scan report for localhost (127.0.0.1)"
        cmd = "/usr/bin/nmap -sn %s/%s" % (address[0], cidr_lookup(address[1]))
        logger.info("CMD: %s", cmd)
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = p.communicate()
        if p.returncode == 0:
            for line in [line for line in stdout.splitlines() if "Nmap scan" in line]:
                devices.append(line.replace('(', '').replace(')', '').split()[4])
        else:
            pass

    # ===== Filter out addresses of this host =====
    ip_list = get_local_ip()
    devices = list(set(devices) - set(ip_list))
    logger.info("scanning these devices for shares: %s", devices)

    # ===== Get all servers with NFS mounts =====
    nas_devices = []
    stderr = ""
    for device in devices:
        logger.info("scanning %s", device)
        # Not the best way: - using showmount and a time limit
        cmd = "/usr/bin/timelimit -t 9 -T 1 showmount -e %s" % (device)
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode == 0:
            # Filter out S5 and Proton instrument shared volumes:
            if '/results' in stdout or '/sw_results' in stdout:
                continue
            nas_devices.append(device)
        elif p.returncode > 128:
            stderr = "Request timed out"

    myjson = json.dumps({
        'error': stderr,
        'devices': nas_devices,
    })
    return HttpResponse(myjson, mimetype="application/json")


def get_avail_mnts(request, ip=None):
    '''Returns json object containing list of shares available for the given
    IP address or hostname.'''
    #resolve the hostname, determine the IP address
    p = subprocess.Popen(['host', ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        if 'domain name pointer' in stdout:
            hostname = stdout.split()[-1].strip()
            hostname = hostname[:-1]    # Last character is always '.'
            ip_address = ip
        else:
            hostname = ip
            ip_address = stdout.split()[-1].strip()
    else:
        hostname = ip
        ip_address = ip
    #determine available mountpoints
    mymounts = []
    cmd = "/usr/bin/timelimit -t 9 -T 1 showmount -e %s" % (hostname)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        answers = stdout.split(':', 1)[1].strip()
        for line in [s.strip() for s in answers.splitlines()]:
            mymounts.append(line)
    elif p.returncode > 128:
        stderr = "Request timed out"
    logger.info("Mounts for %s are %s" % (hostname, mymounts))
    #response
    mnts_json = json.dumps({
        'hostname': hostname,
        'address': ip_address,
        'mount_dir': mymounts,
        'error': stderr,
        })
    return  HttpResponse(mnts_json, mimetype="application/json")


def get_current_mnts(request):
    '''Returns json object containing list of current NFS mounted directories'''
    mymounts = []
    p = subprocess.Popen(['mount', '-t', 'nfs'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        for line in [s.strip() for s in stdout.splitlines()]:
            mymounts.append(line)
    #response
    mnts_json = json.dumps({
        'mount_dir': mymounts,
        'error': stderr,
        })
    return HttpResponse(mnts_json, mimetype="application/json")


def add_nas_storage(request):
    """
    Edits all_local, adding specified mount entry, calls ansible-playbook
    :parameter request: The web request
    """

    try:
        data = json.loads(request.body)
        servername = data.get('servername')
        sharename = data.get('sharename')
        mountpoint = data.get('mountpoint')
        logger.info("Handling request for %s:%s on %s" % (servername, sharename, mountpoint))

        # create an ansible playbook and which will mount the drive
        subprocess.check_call(['sudo', '/opt/ion/iondb/bin/ion_add_nfs_mount.py', servername, sharename, mountpoint])

        # Probe nas unit to identify Nexenta appliance
        if this_is_nexenta(servername):
            logger.info("%s is TorrentNAS", servername)
            # Update credentials file
            subprocess.call(['sudo', '/opt/ion/iondb/bin/write_nms_access.py', '--id', '%s' % servername])
    except:
        logger.error(traceback.format_exc())
    return HttpResponse()


def remove_nas_storage(request):
    '''
    Edits all_local, removing specified mount entry, calls ansible-playbook
    Removes entry from nms_access file
    '''
    try:
        data = json.loads(request.body)
        servername = data.get('servername')
        mountpoint = data.get('mountpoint')
        logger.info("Handling request to remove %s", mountpoint)  # create an ansible playbook and which will mount the drive
        subprocess.check_call(['sudo', '/opt/ion/iondb/bin/ion_remove_nfs_mount.py', mountpoint])
        # Update credentials file
        subprocess.call(['sudo', '/opt/ion/iondb/bin/write_nms_access.py', '--remove', '--id', '%s' % servername])
    except:
        logger.error(traceback.format_exc())
    return HttpResponse()


def check_nas_perms(request):
    logger.debug("User %s is authorized? %s" % (request.user, request.user.is_superuser))
    myjson = json.dumps({'error': '', 'authorized': request.user.is_superuser})
    return HttpResponse(myjson, mimetype="application/json")


@login_required
def offcycle_updates(request):
    ctx = {
        'user_can_update': request.user.is_active and request.user.is_staff,
        'plugins': updateProducts.get_update_plugins(),
        'products': updateProducts.get_update_products(),
        'instruments': updateProducts.get_update_packages()
    }
    return render_to_response("rundb/configure/offcycleUpdates.html", context_instance=RequestContext(request, ctx))


@staff_member_required
def offcycle_updates_install_product(request, name, version):
    try:
        updateProducts.update_product(name, version)
        return HttpResponse()
    except Exception as err:
        return HttpResponseServerError(err)

@staff_member_required
def offcycle_updates_install_package(request, name, version):
    try:
        updateProducts.update_package(name, version)
        return HttpResponse()
    except Exception as err:
        return HttpResponseServerError(err)
