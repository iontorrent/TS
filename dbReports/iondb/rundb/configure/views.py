# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import codecs
import datetime
import json
import logging
import os
import re
import socket
import stat
import subprocess
import traceback
import urlparse
import xmlrpclib

import celery.exceptions
import ion.utils.TSversion
import psutil
import requests
from django.conf import settings
from django.contrib.auth.decorators import permission_required
from django.contrib.auth.models import User
from django.core import urlresolvers
from django.core.cache import cache

# Handles serialization of decimal and datetime objects
from django.core.serializers.json import DjangoJSONEncoder
from django.core.servers.basehttp import FileWrapper
from django.core.urlresolvers import reverse
from django.forms.models import model_to_dict
from django.http import (
    Http404,
    HttpResponse,
    HttpResponseRedirect,
    HttpResponseNotFound,
    HttpResponseServerError,
)
from django.shortcuts import render_to_response, get_object_or_404
from django.template import RequestContext
from django.template.loader import get_template
from django.utils import timezone
from django.utils.translation import ugettext as _, ugettext_lazy
from django.views.decorators.http import require_POST
from ion.plugin.remote import call_pluginStatus
from ion.utils import makeSSA

from iondb.anaserve import client
from iondb.bin.IonMeshDiscoveryManager import IonMeshDiscoveryManager, getLocalComputer
from iondb.product_integration.models import ThermoFisherCloudAccount
from iondb.rundb import tasks, publishers, publisher_types, labels
from iondb.rundb.configure import ampliseq_design_parser
from iondb.rundb.configure import updateProducts
from iondb.rundb.configure.genomes import search_for_genomes
from iondb.rundb.configure.util import plupload_file_upload
from iondb.rundb.data import rawDataStorageReport
from iondb.rundb.forms import EmailAddress as EmailAddressForm, UserProfileForm
from iondb.rundb.json_lazy import LazyJSONEncoder
from iondb.rundb.models import (
    dnaBarcode,
    Plugin,
    PluginResultJob,
    Chip,
    AnalysisArgs,
    EmailAddress,
    Publisher,
    Results,
    Template,
    UserProfile,
    model_to_csv,
    IonMeshNode,
    FileMonitor,
    ContentType,
    Cruncher,
    RunType,
    RUNNING_STATES,
    ReferenceGenome,
    GlobalConfig,
)
from iondb.servelocation import serve_wsgi_location
from iondb.utils import validation, i18n_errors
from iondb.utils.nexenta_nms import (
    this_is_nexenta,
    get_all_torrentnas_data,
    has_nexenta_cred,
    load_torrentnas_status_json,
)
from iondb.utils.raid import load_raid_status_json
from iondb.utils.utils import ManagedPool
from iondb.utils.utils import cidr_lookup, service_status, services_views
from iondb.utils.utils import authenticate_fetch_url

logger = logging.getLogger(__name__)

from json import encoder

encoder.FLOAT_REPR = lambda x: format(x, ".15g")

TIMEOUT_LIMIT_SEC = settings.REQUESTS_TIMEOUT_LIMIT_SEC


def configure(request):
    """Wrapper"""
    return configure_about(request)


def configure_about(request):
    """
    Populate about page values
    CLI to get TS version:
    python -c "from ion.utils import TSversion; _, version = TSversion.findVersions(); print version"
    """
    reload(ion.utils.TSversion)
    versions, meta = ion.utils.TSversion.findVersions()
    osversion = ion.utils.TSversion.findOSversion()
    versions.update(ion.utils.TSversion.offcycleVersions())
    productname = ugettext_lazy("product.name")  # "Torrent Suite"
    productversion = meta
    ctxd = {
        "versions": versions,
        "productname": productname,
        "productversion": productversion,
        "osversion": osversion,
    }
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/about.html", context_instance=ctx)


def configure_ionreporter(request):
    try:
        iru = Plugin.objects.get(name__iexact="IonReporterUploader", active=True)
    except Plugin.DoesNotExist:
        iru = None
    ctxd = {"iru": iru}
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/ionreporter.html", context_instance=ctx)


def timeout_raid_info_json():
    """Call celery task to query RAID card status"""
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
    """sorts array to be displayed matching physical arrangement in enclosure"""
    slots = 12
    for drv in raidstatus:
        try:
            adapter_id = drv.get("adapter_id", "")
            drives = drv.get("drives", [])
            if not adapter_id or len(drives) < slots:
                continue

            if adapter_id.startswith("PERC H710"):
                drv["cols"] = 4
            elif adapter_id.startswith("PERC H810"):
                ncols = 4
                temp = [
                    drives[i : i + (slots / ncols)]
                    for i in range(0, slots, slots / ncols)
                ]
                drv["drives"] = sum(map(list, zip(*temp)), [])
                drv["cols"] = ncols
        except Exception:
            pass


def get_torrent_nas_info(refresh=True):
    # Torrent NAS information

    def get_health_state(health_str):
        # web page will use color to highlight states
        _ERROR_STATES = ["FAULTED", "UNAVAIL"]
        _WARN_STATES = ["DEGRADED"]

        state = "good"
        if health_str in _ERROR_STATES:
            state = "error"
        elif health_str in _WARN_STATES:
            state = "warning"
        return state

    info = []
    if not has_nexenta_cred():
        return info

    if refresh:
        data, err = get_all_torrentnas_data()
        for error in err:
            logger.error(error)
    else:
        data = load_torrentnas_status_json()

    for nas in data:
        nasInfo = {"ipaddress": nas["ipaddress"], "volumes": []}
        for name in nas["volumes"]:
            volume = dict(nas[name])
            volume["name"] = name

            # volume health info and states
            volume["health"] = nas[name]["state"][0]
            volume["state"] = get_health_state(volume["health"])

            # info per disk
            volume["disks"] = []
            config = volume.pop("config")
            for line in config[3:]:
                l = re.compile(r"\W+").split(line.strip(), 5)
                volume["disks"].append(
                    {
                        "name": l[0],
                        "health": l[1],
                        "state": get_health_state(l[1]),
                        "err": " / ".join([l[2], l[3], l[4]]),
                        "info": l[5] if len(l) > 5 else "",
                    }
                )

            # extra info to display on page
            extra = []
            if volume.get("status"):
                extra.append(" ".join(volume["status"]))
            if volume["errors"][0] != "No known data errors":
                extra.append(" ".join(volume["errors"]))
            if volume.get("action"):
                extra.append(" ".join(volume["action"]))
            volume["extra_info"] = "<br>".join(extra)
            # make html links
            links = re.compile(r"(https?://[^\s]+)")
            volume["extra_info"] = links.sub(
                r'<a href="\1" target="_blank">\1</a>', volume["extra_info"]
            )

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
        return render_to_response(
            "rundb/configure/services_torrentNAS.html", context_instance=ctx
        )


def get_memory_info():
    """ Returns the system total mem in GB, used mem in GB, the percent used, and the generation datetime
        https://psutil.readthedocs.io/en/latest/#memory
        """
    mem = psutil.virtual_memory()
    return {
        "total": mem.total / 1000000000.0,
        "used": (mem.total - mem.available) / 1000000000.0,
        "percentage": 100 * ((mem.total - mem.available) / float(mem.total)),
        "generated": datetime.datetime.now(),
    }


def configure_services(request):
    """Render the service tab"""
    servers = get_servers()
    jobs = current_jobs() + current_plugin_jobs()
    crawler = _crawler_status(request)

    system_services = []
    telemetry_services = []

    for name, status in process_set():
        if name in ["RSM_Launch", "deeplaser"]:
            telemetry_services.append([name, status])
        else:
            system_services.append([name, status])

    # RAID Info
    raidJson = load_raid_status_json()
    raid_status = raidJson.get("raid_status")
    raid_status_updated = raidJson.get("date")

    if raid_status:
        sort_drive_array_for_display(raid_status)

    gc = GlobalConfig.objects.get()

    ctxd = {
        "system_services": system_services,
        "telemetry_services": telemetry_services,
        "telemetry_enabled": gc.telemetry_enabled,
        "servers": servers,
        "jobs": jobs,
        "crawler": crawler,
        "raid_status": raid_status,
        "raid_status_updated": raid_status_updated,
        "crunchers": Cruncher.objects.all().order_by("name"),
        "show_nas_info": has_nexenta_cred(),
        "memory_info": get_memory_info(),
    }
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/services.html", context_instance=ctx)


def configure_telemetry_services_toggle(request):
    """ Switch telemetry_services on and off """
    if request.method == "POST":
        gc = GlobalConfig.objects.get()
        if gc.telemetry_enabled:
            gc.telemetry_enabled = False
        else:
            gc.telemetry_enabled = True
        gc.save()

    return HttpResponseRedirect(urlresolvers.reverse("configure_services"))


def configure_references(request):
    """Render reference tab"""
    search_for_genomes()
    ctxd = {
        "active_references": ReferenceGenome.objects.filter(
            enabled=True, index_version=settings.TMAP_VERSION
        ).order_by("short_name"),
        "publisher_types": json.dumps(publisher_types.get_publisher_types()),
    }
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/configure/references.html", context_instance=ctx)


def configure_plugins(request):
    """Render plugin tab"""

    # Rescan Publishers
    publishers.search_for_publishers()
    pubs = Publisher.objects.all().order_by("name")

    # hide the "Upload" button if upload.html doesn't exists
    for p in pubs:
        p.uploadLink = True
        if not os.path.exists(os.path.join(p.path, "upload.html")):
            p.uploadLink = False

    # perform a check here to see if the user is a super user, but really we should be looking for a more refined permission set
    # but the interface needs to be able to support refined permissions before this happens
    can_upgrade = request.user.is_superuser

    plugin_perms = {}
    # All Plugins
    plugin_perms[
        "rescan_plugins"
    ] = request.user.is_authenticated()  # Any Auth'd user can Rescan
    plugin_perms[
        "rescan_plugin"
    ] = request.user.is_authenticated()  # Any Auth'd user can Rescan
    plugin_perms[
        "about_plugin"
    ] = request.user.is_authenticated()  # Any Auth'd user can View About
    plugin_perms[
        "usage_plugin"
    ] = request.user.is_authenticated()  # Any Auth'd user can View Usage

    plugin_perms["install_plugin"] = request.user.is_superuser  # Admin
    plugin_perms["configure_plugin"] = request.user.is_superuser  # Admin
    plugin_perms["upgrade_plugin"] = request.user.is_superuser  # Admin
    plugin_perms["install_to_version_plugin"] = request.user.is_superuser  # Admin
    plugin_perms["uninstall_plugin"] = request.user.is_superuser  # Admin

    logger.debug("plugin_perms %s", plugin_perms)

    ctx = RequestContext(
        request,
        {
            "publishers": pubs,
            "can_upgrade": can_upgrade,
            "Plugin": labels.Plugin,
            "Publisher": labels.Publisher,
            "plugin_perms": plugin_perms,
        },
    )
    return render_to_response("rundb/configure/plugins.html", context_instance=ctx)


def configure_plugins_plugin_install_to_version(request, pk, version):
    """
    View rendering for the install to version modal interface
    :param request: Request
    :param pk: The primary key for the plugin
    :param version: The version to install
    :return: a renderer
    """

    plugin = get_object_or_404(Plugin, pk=pk)
    action = reverse(
        "api_dispatch_install_to_version",
        kwargs={
            "api_name": "v1",
            "resource_name": "plugin",
            "pk": pk,
            "version": version,
        },
    )
    _installVersionedName = Plugin(name=plugin.name, version=version).versionedName()
    ctx = RequestContext(
        request,
        {
            "method": "POST",
            "action": action,
            "i18n": {
                "title": ugettext_lazy(
                    "configure_plugins_plugin_install_to_version.title"
                ),  # 'Confirm Install Plugin'
                "confirmmsg": ugettext_lazy(
                    "configure_plugins_plugin_install_to_version.messages.confirmmsg.singular"
                )
                % {  # 'Are you sure you want to install %(versionedName)s?'
                    "versionedName": _installVersionedName
                },
                "submit": ugettext_lazy(
                    "configure_plugins_plugin_install_to_version.action.submit"
                ),  # 'Yes, Upgrade!'
                "cancel": ugettext_lazy("global.action.modal.cancel"),
                "submitmsg": ugettext_lazy(
                    "configure_plugins_plugin_install_to_version.messages.submitmsg"
                ),  # 'Now upgrading, please wait.'
            },
        },
    )
    return render_to_response(
        "rundb/configure/modal_confirm_plugin_install_to_version.html",
        context_instance=ctx,
    )


def configure_plugins_plugin_install(request):
    """Render plugin install tab"""
    ctxd = {
        "i18n": {
            "title": ugettext_lazy(
                "configure_plugins_plugin_install.title"
            ),  # 'Install or Upgrade Plugin'
            "fields": {
                "pickfile": {
                    "label": ugettext_lazy(
                        "configure_plugins_plugin_install.fields.pickfile.label"
                    ),  # 'Upload an Updates File'
                    "select": ugettext_lazy(
                        "global.fields.pickfile.select"
                    ),  # Select File
                    "helptext": ugettext_lazy(
                        "global.fields.pickfile.helptext"
                    ),  # 'In order to provide a better uploading experience an HTML5 compatible browser are required for file uploading. You may need to contact your local system administrator for assistance.'
                }
            },
            "submit": ugettext_lazy(
                "configure_plugins_plugin_install.action.submit"
            ),  # 'Upload and Install'
            "cancel": ugettext_lazy("global.action.modal.cancel"),
            "submitmsg": ugettext_lazy(
                "configure_plugins_plugin_install.messages.submitmsg"
            ),  # 'Attempting to install ...'
            "submitsuccessmsg": ugettext_lazy(
                "configure_plugins_plugin_install.messages.submitsuccessmsg"
            ),  # 'Started Plugin install'
        },
        "what": "plugin",
        "file_filters": [("deb", labels.FileTypes.deb), ("zip", labels.FileTypes.zip)],
        "plupload_url": reverse("configure_plugins_plugin_zip_upload"),
        "install_url": reverse(
            "api_dispatch_install", kwargs={"api_name": "v1", "resource_name": "plugin"}
        ),
    }
    return render_to_response(
        "rundb/configure/modal_plugin_or_publisher_install.html",
        context_instance=RequestContext(request, ctxd),
    )


def configure_publisher_install(request):
    """Render publisher install tab"""
    ctxd = {
        "i18n": {
            "title": ugettext_lazy(
                "configure_publisher_install.title"
            ),  # 'Install or Upgrade Publisher'
            "fields": {
                "pickfile": {
                    "label": ugettext_lazy(
                        "configure_publisher_install.fields.pickfile.label"
                    ),  # 'Upload an Updates File'
                    "select": ugettext_lazy(
                        "global.fields.pickfile.select"
                    ),  # Select File
                    "helptext": ugettext_lazy(
                        "global.fields.pickfile.helptext"
                    ),  # 'In order to provide a better uploading experience an HTML5 compatible browser are required for file uploading. You may need to contact your local system administrator for assistance.'
                }
            },
            "submit": ugettext_lazy(
                "configure_publisher_install.action.submit"
            ),  # 'Upload and Install'
            "cancel": ugettext_lazy("global.action.modal.cancel"),
            "submitmsg": ugettext_lazy(
                "configure_publisher_install.messages.submitmsg"
            ),  # 'Attempting to install ...'
            "submitsuccessmsg": ugettext_lazy(
                "configure_publisher_install.messages.submitsuccessmsg"
            ),  # 'Started Publisher install'
        },
        "what": "publisher",
        "file_filters": [("zip", labels.FileTypes.zip)],
        "plupload_url": reverse("configure_plugins_plugin_zip_upload"),
        "install_url": reverse(
            "api_dispatch_install",
            kwargs={"api_name": "v1", "resource_name": "publisher"},
        ),
    }
    return render_to_response(
        "rundb/configure/modal_plugin_or_publisher_install.html",
        context_instance=RequestContext(request, ctxd),
    )


def configure_plugins_plugin_upgrade(request, pk):
    """
    This method will take a web request to upgrade a plugin
    """

    plugin = get_object_or_404(Plugin, pk=pk)
    action = reverse(
        "api_dispatch_upgrade",
        kwargs={"api_name": "v1", "resource_name": "plugin", "pk": pk},
    )
    ctx = RequestContext(
        request,
        {
            "method": "POST",
            "action": action,
            "i18n": {
                "title": ugettext_lazy(
                    "configure_plugins_plugin_upgrade.title"
                ),  # 'Confirm Upgrade Plugin'
                "confirmmsg": ugettext_lazy(
                    "configure_plugins_plugin_upgrade.messages.confirmmsg.singular"
                )
                % {  # 'Are you sure you want to upgrade %(versionedName)s to %(UpgradeToVersion)s?'
                    "versionedName": plugin.versionedName(),
                    "UpgradeToVersion": plugin.availableVersions[0],
                },
                "submit": ugettext_lazy(
                    "configure_plugins_plugin_upgrade.action.submit"
                ),  # 'Yes, Upgrade!'
                "cancel": ugettext_lazy("global.action.modal.cancel"),
                "submitmsg": ugettext_lazy(
                    "configure_plugins_plugin_upgrade.messages.submitmsg"
                ),  # 'Now upgrading, please wait.'
            },
        },
    )
    return render_to_response(
        "rundb/configure/modal_confirm_plugin_upgrade.html", context_instance=ctx
    )


def configure_plugins_plugin_configure(request, action, pk):
    """
    load files into iframes
    """

    def openfile(fname):
        """strip lines """
        try:
            fhandle = codecs.open(fname, "r", "utf-8")
        except Exception:
            logger.exception("Failed to open '%s'", fname)
            return False
        content = fhandle.read()
        fhandle.close()
        return content

    # Used in javascript, must serialize to json
    plugin = get_object_or_404(Plugin, pk=pk)
    # make json to send to the template
    plugin_json = json.dumps(
        {"pk": pk, "model": str(plugin._meta), "fields": model_to_dict(plugin)},
        cls=DjangoJSONEncoder,
    )

    # If you set more than one of these,
    # behavior is undefined. (one returned randomly)
    dispatch_table = {
        "report": "instance.html",
        "config": "config.html",
        "about": "about.html",
        "plan": "plan.html",
    }

    fname = os.path.join(plugin.path, dispatch_table[action])

    content = openfile(fname) or ""
    index_version = settings.TMAP_VERSION

    report = request.GET.get("report", False)
    results_json = {}
    if report:
        # Used in javascript, must serialize to json
        results_obj = get_object_or_404(Results, pk=report)
        # make json to send to the template
        results_json = json.dumps(
            {
                "pk": report,
                "model": str(results_obj._meta),
                "fields": model_to_dict(results_obj),
            },
            cls=DjangoJSONEncoder,
        )
        applicationGroup = results_obj.experiment.plan.applicationGroup if results_obj.experiment.plan else None
        plan_json = json.dumps(
            {
                "applicationGroup": applicationGroup.name if applicationGroup else "",
                "runType": results_obj.experiment.plan.runType if results_obj.experiment.plan else "",
            }
        )
    else:
        applicationGroup = request.GET.get("applicationGroup", "")
        runTypeId = request.GET.get("runTypeId", "")
        runType = request.GET.get("runType", "")
        plan_json = {}
        if runTypeId:
            runType_obj = get_object_or_404(RunType, pk=runTypeId)
            runType = runType_obj.runType
        plan_json = json.dumps(
            {"applicationGroup": applicationGroup, "runType": runType},
            cls=DjangoJSONEncoder,
        )

    ctxd = {
        "plugin": plugin_json,
        "file": content,
        "report": report,
        "tmap": str(index_version),
        "results": results_json,
        "plan": plan_json,
        "action": action,
    }

    context = RequestContext(request, ctxd)

    # modal_configure_plugins_plugin_configure.html will be removed in a newer release as it places a script outside the html tag.
    # Add in the js vars just before the closing head tag.
    plugin_js_script = (
        get_template("rundb/configure/plugins/plugin_configure_js.html")
        .render(context)
        .replace("\n", "")
    )
    context["file"] = context["file"].replace(
        "</head>", plugin_js_script + "</head>", 1
    )

    return render_to_response(
        "rundb/configure/modal_configure_plugins_plugin_configure.html",
        context_instance=context,
    )


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
    servepath = "plugins/" + fspath[len(prefix) :] + "/pluginMedia/" + path
    # example: "/plugins/" + "examplePlugin" + "/" + "pluginMedia/img.png"
    # example: "/plugins/" + "instances/12345667/exampleZeroConf" + "/" + "pluginMedia/img.png"
    logger.debug("Redirecting pluginMedia request '%s' to '%s'", path, servepath)

    # Serve redirects to /private/plugins/,
    # which is mapped to /results/plugins/ in apache
    return serve_wsgi_location(request, servepath)


def configure_plugins_plugin_uninstall(request, pk):
    """
    Disables a plugin from the system
    :param request:
    :param pk: The primary key of the plugin to be disabled
    :return:
    """
    # TODO: See about pulling this out into a common methods
    plugin = get_object_or_404(Plugin, pk=pk)
    action = reverse(
        "api_dispatch_uninstall",
        kwargs={"api_name": "v1", "resource_name": "plugin", "pk": pk},
    )

    ctx = RequestContext(
        request,
        {
            "method": "DELETE",
            "action": action,
            "i18n": {
                "title": ugettext_lazy(
                    "configure_plugins_plugin_uninstall.title"
                ),  # 'Confirm Uninstall Plugin'
                "confirmmsg": ugettext_lazy(
                    "configure_plugins_plugin_uninstall.messages.confirmmsg.singular"
                )
                % {  # 'Are you sure you want to uninstall %(versionedName)s Plugin (%(id)s)?'
                    "id": str(pk),
                    "versionedName": plugin.versionedName(),
                },
                "submit": ugettext_lazy(
                    "configure_plugins_plugin_uninstall.action.submit"
                ),  # 'Yes, Uninstall!'
                "cancel": ugettext_lazy("global.action.modal.cancel"),
                "submitmsg": ugettext_lazy(
                    "configure_plugins_plugin_uninstall.messages.submitmsg"
                ),  # 'Now uninstalling, please wait.'
            },
        },
    )
    return render_to_response(
        "rundb/configure/modal_confirm_plugin_uninstall.html", context_instance=ctx
    )


def configure_plugins_plugin_zip_upload(request):
    return plupload_file_upload(request, "/results/plugins/scratch/")


def configure_offline_bundle_upload(request):
    return plupload_file_upload(request, "/results/uploads/offcycle/")


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


def configure_plugins_plugin_usage(request, pk):
    plugin = get_object_or_404(Plugin, pk=pk)
    pluginresults = plugin.pluginresult_set
    ctx = RequestContext(
        request,
        {
            "plugin": plugin,
            "pluginresults": pluginresults,
            "Plugin": labels.Plugin,
            "PluginResult": labels.PluginResult,
        },
    )
    return render_to_response(
        "rundb/configure/plugins/plugin_usage.html", context_instance=ctx
    )


def configure_configure(request):
    """
    Handles the render and post for server configuration
    """
    ctx = RequestContext(request, {})
    emails = EmailAddress.objects.all().order_by("pk")
    enable_nightly = GlobalConfig.get().enable_nightly_email
    ctx.update(
        {
            "email": emails,
            "enable_nightly": enable_nightly,
            "EmailAddress": labels.EmailAddress,
            "GlobalConfig": labels.GlobalConfig,
        }
    )
    config_contacts(request, ctx)
    config_site_name(request, ctx)

    # handle post request for setting a time zone
    if request.method == "POST" and "zone_select" in request.POST:
        try:
            selected_timezone = (
                request.POST["zone_select"] + "/" + request.POST["city_select"]
            )
            subprocess.check_call(
                ["sudo", "timedatectl", "set-timezone", selected_timezone]
            )
            logger.info("Successfully set the timezone to " + selected_timezone)
        except subprocess.CalledProcessError as exc:
            logger.error(str(exc))
            raise Http404()

    # handle the request for the timezones
    else:
        get_timezone_info(request, ctx)

    # get current list of
    return render_to_response("rundb/configure/configure.html", context_instance=ctx)


def finish_sharedservers_mesh_link():
    # Finishes linking any SharedServers that were converted to IonMeshNode in migration 0324
    # This function will only do anything if there are any unlinked servers left
    from iondb.security.models import SecureString
    from iondb.rundb.models import _generate_key

    nodes = IonMeshNode.objects.filter(system_id__startswith="sharedserver_")
    for mesh_node in nodes:
        try:
            secure_string = SecureString.objects.get(
                pk=mesh_node.system_id.split("_")[1]
            )
            decrypted = json.loads(secure_string.decrypted)
            api_url = "http://%s/rundb/api/v1/" % mesh_node.hostname

            s = IonMeshNode.SetupMeshSDKSession(
                auth=(decrypted["username"], decrypted["password"])
            )

            # retrieve remote server info
            r = s.get(api_url + "ionmeshnode/system_id/", timeout=10)
            r.raise_for_status()
            remote_system_id = r.json()["system_id"]

            # make sure this node doesn't already exist
            exist = IonMeshNode.objects.filter(system_id=remote_system_id)
            if exist:
                # update node name to the original SharedServer name and delete duplicate entry
                mesh_node.delete()
                exist[0].name = mesh_node.name
                exist[0].save(update_fields=["name"])
            else:
                # set up mesh node on remote server
                local_info = {
                    "system_id": settings.SYSTEM_UUID,
                    "hostname": getLocalComputer(),
                    "apikey": _generate_key(),
                }
                r = s.post(api_url + "ionmeshnode/exchange_keys/", data=local_info)
                r.raise_for_status()

                # finish updating local node
                mesh_node.system_id = remote_system_id
                mesh_node.apikey_remote = r.json()["apikey"]
                mesh_node.apikey_local = local_info["apikey"]
                mesh_node.save()

            # clean up temporary username/password store
            secure_string.delete()

        except Exception:
            logger.debug(
                "Error linking IonMeshNode %s: %s"
                % (mesh_node.name, traceback.format_exc())
            )


@permission_required("user.is_staff", raise_exception=True)
def link_mesh_node(request):
    """
    Add or update mesh node
    """
    name = request.POST.get("name")
    hostname = request.POST.get("hostname")
    username = request.POST.get("userid")
    password = request.POST.get("pswrd")
    api_url = "http://%s/rundb/api/v1/" % hostname
    error = ""

    # set up remote session
    sdk_session = IonMeshNode.SetupMeshSDKSession(auth=(username, password))
    r = None

    try:
        r = sdk_session.get(api_url + "ionmeshnode/system_id/", timeout=30)
        r.raise_for_status()
    except requests.ConnectionError as exc:
        error = ugettext_lazy(
            "entity.IonMeshNode.fields.status.choices.connection_error.description"
        )  # "Could not make a connection to the remote server"
    except requests.exceptions.Timeout:
        error = ugettext_lazy(
            "entity.IonMeshNode.fields.status.choices.timeout.description"
        )  # "Timeout while trying to get response from remote server"
    except requests.HTTPError as exc:
        if r.status_code == 401:
            error = ugettext_lazy(
                "entity.IonMeshNode.fields.status.choices.unauthorized.description"
            )  # "Invalid user credentials"
        else:
            logger.error(unicode(exc))
            error = ugettext_lazy(
                "configure_mesh.setup.messages.error"
            )  # "Unable to set up Mesh link. Ensure remote server has the same version of Torrent Suite as this server."
    except Exception as exc:
        error = unicode(exc)
        logger.error(traceback.format_exc())
    else:
        remote_system_id = r.json()["system_id"]
        if remote_system_id == settings.SYSTEM_UUID:
            error = ugettext_lazy(
                "configure_mesh.setup.messages.system_id.localhost"
            )  # "Adding your own local host into mesh is not allowed. Please check your hostname/address"
        else:
            node, created = IonMeshNode.create(system_id=remote_system_id)
            if not created and node.hostname != hostname:
                error = ugettext_lazy("configure_mesh.setup.messages.exists") % {
                    "mesh_node_name": node.name
                }  # "This system has already been setup in the mesh as '" + node.name + "'."
            else:
                try:
                    # set up mesh node on remote server
                    local_info = {
                        "system_id": settings.SYSTEM_UUID,
                        "hostname": getLocalComputer(),
                        "apikey": node.apikey_local,
                    }
                    r = sdk_session.post(
                        api_url + "ionmeshnode/exchange_keys/", data=local_info
                    )
                    r.raise_for_status()
                except:
                    error = ugettext_lazy(
                        "configure_mesh.setup.messages.error"
                    )  # "Unable to set up Mesh link. Ensure remote server has the same version of Torrent Suite as this server."
                    logger.error(traceback.format_exc())
                    if not node.hostname:
                        node.delete()
                else:
                    # set up mesh node on this server
                    node.name = name
                    node.hostname = hostname
                    node.apikey_remote = r.json()["apikey"]
                    # handle edge case: host was entered previously via SharedServer migration
                    duplicate = IonMeshNode.objects.filter(
                        hostname=node.hostname
                    ).exclude(pk=node.pk)
                    duplicate.delete()
                    try:
                        node.save()
                    except Exception as exc:
                        error = unicode(exc)
                        logger.error(traceback.format_exc())

    return HttpResponse(
        json.dumps({"error": error}, cls=LazyJSONEncoder), mimetype="application/json"
    )


@permission_required("user.is_staff", raise_exception=True)
def delete_mesh_node(request, pk):
    node = get_object_or_404(IonMeshNode, pk=pk)
    url = "http://%s/rundb/api/v1/ionmeshnode/" % node.hostname
    sdk_session = IonMeshNode.SetupMeshSDKSession(apikey=node.apikey_remote)

    # remove remote entry
    remote_exception = None
    try:
        r = sdk_session.delete(url + "?system_id=" + settings.SYSTEM_UUID)
        r.raise_for_status()
    except Exception as exc:
        remote_exception = exc

    # remove local entry
    if not remote_exception or "force" in request.GET:
        node.delete()

    if remote_exception:
        return HttpResponse(unicode(remote_exception), status=500)
    else:
        return HttpResponse()


@permission_required("user.is_staff", raise_exception=True)
def configure_mesh(request):
    """
    Handles view requests for the ion mesh configuration page.
    :param request: The http request
    :return: A http response
    """
    finish_sharedservers_mesh_link()

    mesh_hosts = []

    # Fetch hosts that may have been posted to this TS by an S5 host
    mesh_hosts += [
        i["hostname"] for i in json.loads(cache.get("auto_discovered_hosts_json", "[]"))
    ]

    try:
        ionMeshDiscoveryManager = IonMeshDiscoveryManager()
        # Fetch hosts that may have been found via avahi
        mesh_hosts += ionMeshDiscoveryManager.getMeshComputers()
    except Exception as err:
        logger.error("IonMeshDiscoveryManager returned error: " + str(err))

    ctx = RequestContext(
        request,
        {
            "mesh_hosts": sorted(mesh_hosts),
            "ion_mesh_nodes_table": IonMeshNode.objects.all(),
            "system_id": settings.SYSTEM_UUID,
            "IonMeshNode": labels.IonMeshNode,
            "IonMeshNodeStatus": labels.IonMeshNodeStatus,
            "User": labels.User,
        },
    )

    return render_to_response(
        "rundb/configure/configure_mesh.html", context_instance=ctx
    )


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
            err[0] = datetime.datetime.strptime(str(err[0]), "%Y%m%dT%H:%M:%S").replace(
                tzinfo=timezone.utc
            )

        ret = {
            "elapsed": seconds2htime(raw_elapsed),
            "hostname": cstat.hostname(),
            "state": cstat.state(),
            "errors": exp_errors,
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
        ipaddress = [
            (s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close())
            for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]
        ][0][1]
    except Exception:
        ipaddress = socket.gethostbyname(socket.gethostname())

    jservers = [(socket.gethostname(), ipaddress)]
    servers = []
    for server_name, ip in jservers:
        try:
            conn = client.connect(ip, settings.JOBSERVER_PORT)
            nrunning = conn.n_running()
            uptime = seconds2htime(conn.uptime())
            servers.append((server_name, ip, True, nrunning, uptime))
        except (socket.error, xmlrpclib.Fault):
            servers.append((server_name, ip, False, 0, 0))
    return servers


def current_jobs():
    """
    Get list of running jobs from job server
    """
    jobs = []
    try:
        running = client.get_running_jobs(
            settings.JOBSERVER_HOST, settings.JOBSERVER_PORT
        )
        runs = dict((r[2], r) for r in running)

        results = Results.objects.filter(pk__in=list(runs.keys())).order_by("pk")
        for result in results:
            name, pid, pk, atype, stat = runs[result.pk]
            jobs.append(
                {
                    "name": name,
                    "resultsName": result.resultsName,
                    "pid": pid,
                    "type": "analysis",
                    "status": stat,
                    "pk": result.pk,
                    "report_exist": result.report_exist(),
                    "report_url": reverse("report", args=(pk,)),
                    "term_url": reverse("control_job", args=(pk, "term")),
                }
            )
    except (socket.error, xmlrpclib.Fault):
        pass

    return jobs


def current_plugin_jobs():
    """
    Get list of active pluginresults from database then connect to ionPlugin and get drmaa status per jobid.
    """
    jobs = []
    running = PluginResultJob.objects.filter(state__in=RUNNING_STATES).order_by("pk")
    if running:
        # get job status from drmaa
        jobids = running.values_list("grid_engine_jobid", flat=True)
        try:
            job_status = call_pluginStatus(list(jobids))
            for i, prj in enumerate(list(running)):
                if job_status[i] not in ["DRMAA BUG", "job finished normally"]:
                    jobs.append(
                        {
                            "name": prj.plugin_result.plugin.name + "-" + prj.run_level,
                            "resultsName": prj.plugin_result.result.resultsName,
                            "pid": prj.grid_engine_jobid,
                            "type": "plugin",
                            "status": job_status[i],
                            "pk": prj.plugin_result.pk,
                            "report_exist": True,
                            "report_url": reverse(
                                "report", args=(prj.plugin_result.result.pk,)
                            ),
                            "term_url": "/rundb/api/v1/pluginresult/%d/stop/"
                            % prj.plugin_result.pk,
                        }
                    )
        except Exception:
            pass

    return jobs


def process_set():
    """get list of ion services and processa and output a sorted list"""

    proc_set = service_status(services_views())
    return sorted(proc_set.items())


def references_TF_edit(request, pk=None):

    if pk:
        tf = get_object_or_404(Template, pk=pk)
        ctx = RequestContext(
            request,
            {
                "id": pk,
                "method": "PUT",
                "methodDescription": "Edit",
                "readonly": False,
                "action": reverse(
                    "api_dispatch_detail",
                    kwargs={
                        "resource_name": "template",
                        "api_name": "v1",
                        "pk": int(pk),
                    },
                ),
                "tf": tf,
                "title": ugettext_lazy("references_TF_edit.edit.title"),
                "TestFragment": labels.TestFragment,
                "submitLabel": ugettext_lazy("global.action.save"),
            },
        )
    else:
        ctx = RequestContext(
            request,
            {
                "id": pk,
                "method": "POST",
                "methodDescription": "Add",
                "readonly": False,
                "action": reverse(
                    "api_dispatch_list",
                    kwargs={"resource_name": "template", "api_name": "v1"},
                ),
                "title": ugettext_lazy("references_TF_edit.add.title"),
                "TestFragment": labels.TestFragment,
                "submitLabel": ugettext_lazy("global.action.save"),
            },
        )
    return render_to_response(
        "rundb/configure/modal_references_edit_TF.html", context_instance=ctx
    )


def references_TF_delete(request, pk):
    tf = get_object_or_404(Template, pk=pk)
    _type = "TestFragment"
    ctx = RequestContext(
        request,
        {
            "id": pk,
            "ids": json.dumps([pk]),
            "method": "DELETE",
            "action": reverse(
                "api_dispatch_detail",
                kwargs={"resource_name": "template", "api_name": "v1", "pk": int(pk)},
            ),
            "actions": json.dumps([]),
            "items": None,
            "isMultiple": False,
            "i18n": {
                "title": _(
                    "testfragment.modal_confirm_delete.title.singular"
                ),  # 'Confirm Delete Test Fragment'
                "confirmmsg": _(
                    "testfragment.modal_confirm_delete.messages.confirmmsg.singular"
                )
                % {"testFragmentId": pk, "testFragmentName": tf.name},
                "submit": _("testfragment.modal_confirm_delete.action.submit"),
                "cancel": _("testfragment.modal_confirm_delete.action.cancel"),
            },
        },
    )
    return render_to_response(
        "rundb/common/modal_confirm_delete.html", context_instance=ctx
    )


def references_barcodeset(request, barcodesetid):
    barcode = get_object_or_404(dnaBarcode, pk=barcodesetid)
    ctx = RequestContext(
        request,
        {
            "name": barcode.name,
            "barCodeSetId": barcodesetid,
            "system": barcode.system,
            "title": ugettext_lazy("references_barcodeset.title").format(
                barCodeSetName=barcode.name
            ),  # 'Barcodes in {name}'
            "BarcodeSet": labels.BarcodeSet,
            "Barcode": labels.Barcode,
        },
    )
    return render_to_response(
        "rundb/configure/references_barcodeset.html", context_instance=ctx
    )


def references_barcodeset_add(request):
    if request.method == "GET":
        ctx = RequestContext(request, {"BarcodeSet": labels.BarcodeSet})
        return render_to_response(
            "rundb/configure/modal_references_add_barcodeset.html", context_instance=ctx
        )
    elif request.method == "POST":
        name = request.POST.get("name", "")
        postedfile = request.FILES["postedfile"]

        # Trim off barcode and barcode kit leading and trailing spaces and update the log file if exists
        if (len(name) - len(name.lstrip())) or (len(name) - len(name.rstrip())):
            name = name.strip()
            logger.warning(
                "The Barcode Set Name (%s) contains Leading/Trailing spaces and got trimmed."
                % name
            )

        if dnaBarcode.objects.filter(name=name):
            return HttpResponse(
                json.dumps(
                    {
                        "status": validation.invalid_entity_field_unique(
                            labels.BarcodeSet.verbose_name,
                            labels.BarcodeSet.name.verbose_name,
                        )
                    },
                    cls=LazyJSONEncoder,
                ),
                mimetype="text/html",
            )

        bar_codes, failed = dnaBarcode.from_csv(postedfile, name)

        if failed:
            r = {
                "status": i18n_errors.validationerrors_cannot_save(
                    labels.BarcodeSet.verbose_name
                ),
                "failed": failed,
            }
            return HttpResponse(
                json.dumps(r, cls=LazyJSONEncoder), mimetype="text/html"
            )
        if not bar_codes:
            r = {
                "status": ugettext_lazy(
                    "references_barcodeset_add.messages.error.barcoderequired"
                )
            }  # "Error: There must be at least one barcode! Please reload the page and try again with more barcodes."
            return HttpResponse(
                json.dumps(r, cls=LazyJSONEncoder), mimetype="text/html"
            )
        used_id = []
        for bar_code in bar_codes:
            if bar_code.id_str not in used_id:
                used_id.append(bar_code.id_str)
            else:
                error = {
                    "status": validation.invalid_entity_field_unique_value(
                        labels.Barcode.verbose_name,
                        Barcode.id_str.verbose_name,
                        unicode(barCode.id_str),
                    )
                }
                return HttpResponse(
                    json.dumps(error, cls=LazyJSONEncoder), mimetype="text/html"
                )
        used_index = []
        for bar_code in bar_codes:
            if bar_code.index not in used_index:
                used_index.append(bar_code.index)
            else:
                error = {
                    "status": validation.invalid_entity_field_unique_value(
                        labels.Barcode.verbose_name,
                        labels.Barcode.index.verbose_name,
                        unicode(bar_code.index),
                    )
                }
                return HttpResponse(
                    json.dumps(error, cls=LazyJSONEncoder), mimetype="text/html"
                )

        # saving to db needs to be the last thing to happen
        for bar_code in bar_codes:
            try:
                bar_code.save()
            except Exception as exc:
                logger.exception("Error saving barcode to database")
                return HttpResponse(
                    json.dumps(
                        {
                            "status": i18n_errors.fatal_internalerror_during_save_of(
                                Barcode.verbose_name
                            )
                        },
                        cls=LazyJSONEncoder,
                    ),
                    mimetype="text/html",
                )
        r = {
            "status": ugettext_lazy("references_barcodeset_add.messages.success"),
            "failed": failed,
            "success": True,
        }  # "Barcodes Uploaded! The barcode set will be listed on the references page."
        return HttpResponse(json.dumps(r, cls=LazyJSONEncoder), mimetype="text/html")


def reference_barcodeset_csv(request, barcodesetid):
    """Get a csv for a barcode set"""
    barcodeset_name = get_object_or_404(dnaBarcode, pk=barcodesetid).name
    filename = "%s_%s" % (
        barcodeset_name,
        str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    barcode_csv_filename = filename.replace(" ", "_")
    response = HttpResponse(mimetype="text/csv")
    response["Content-Disposition"] = (
        "attachment; filename=%s.csv" % barcode_csv_filename
    )
    barcodes = dnaBarcode.objects.filter(name=barcodeset_name)
    columns = ["index", "id_str", "sequence", "annotation", "adapter"]

    # if any of the barcodes have an end adapter or sequence this will also needed to be added to the csv
    for barcode in barcodes:
        if barcode.end_adapter or barcode.end_sequence:
            columns += ["end_adapter", "end_sequence"]
            break

    response.write(model_to_csv(barcodes, columns))
    return response


def references_barcodeset_delete(request, barcodesetid):
    barCodeSet = get_object_or_404(dnaBarcode, pk=barcodesetid)
    barCodeSetName = barCodeSet.name
    """delete a set of barcodes"""
    if request.method == "POST":
        dnaBarcode.objects.filter(name=barCodeSetName).delete()
        return HttpResponse()
    elif request.method == "GET":
        # TODO: See about pulling this out into a common methods
        ctx = RequestContext(
            request,
            {
                "id": barCodeSetName,
                "ids": json.dumps([]),
                "method": "POST",
                "action": reverse("references_barcodeset_delete", args=[barcodesetid]),
                "actions": json.dumps([]),
                "items": None,
                "isMultiple": False,
                "i18n": {
                    "title": _(
                        "barcodeset.modal_confirm_delete.title.singular"
                    ),  # 'Confirm Delete Barcode Set'
                    "confirmmsg": _(
                        "barcodeset.modal_confirm_delete.messages.confirmmsg.singular"
                    )
                    % {"barCodeSetName": barCodeSetName},
                    "submit": _("barcodeset.modal_confirm_delete.action.submit"),
                    "cancel": _("barcodeset.modal_confirm_delete.action.cancel"),
                },
            },
        )
        return render_to_response(
            "rundb/common/modal_confirm_delete.html", context_instance=ctx
        )


def references_barcode_add(request, barcodesetid):
    return references_barcode_edit(request, barcodesetid, None)


def references_barcode_edit(request, barcodesetid, pk):
    dna = get_object_or_404(dnaBarcode, pk=barcodesetid)
    barCodeSetName = dna.name

    def nextIndex(name):
        barCodeSetName = dnaBarcode.objects.filter(name=name).order_by("-index")
        if barCodeSetName:
            return barCodeSetName[0].index + 1
        else:
            return 1

    # if there is a barcode do a look up for it
    if pk:
        barcode = dnaBarcode.objects.get(pk=int(pk))
        index = barcode.index
        # get a list of all the other barcodes minus this one
        others = dnaBarcode.objects.filter(name=barCodeSetName)
        others = others.exclude(pk=int(pk))
        title = ugettext_lazy("references_barcode_edit.title").format(
            barCodeSetName=barCodeSetName
        )  # 'Edit barcode in set <strong>{barCodeSetName}</strong>'

    else:
        barcode = False
        index = nextIndex(barCodeSetName)
        # get a list of all the other barcodes
        others = dnaBarcode.objects.filter(name=barCodeSetName)
        title = ugettext_lazy("references_barcode_add.title").format(
            barCodeSetName=barCodeSetName
        )  # 'Add new barcode in set <strong>{barCodeSetName}</strong>'

    otherList = []
    for other in others:
        otherList.append(other.id_str)

    ctxd = {
        "barcode": barcode,
        "barCodeSetName": barCodeSetName,
        "index": index,
        "otherList": json.dumps(otherList),
        "title": title,
        "Barcode": labels.Barcode,
    }
    ctx = RequestContext(request, ctxd)
    return render_to_response(
        "rundb/configure/modal_references_addedit_barcode.html", context_instance=ctx
    )


def references_barcode_delete(request, barcodesetid, pk):
    # TODO: See about pulling this out into a common methods
    barcode = get_object_or_404(dnaBarcode, pk=pk)
    action = reverse(
        "api_dispatch_detail",
        kwargs={"resource_name": "dnabarcode", "api_name": "v1", "pk": int(pk)},
    )

    ctx = RequestContext(
        request,
        {
            "id": pk,
            "ids": json.dumps([pk]),
            "method": "DELETE",
            "action": action,
            "actions": json.dumps([action]),
            "items": None,
            "isMultiple": False,
            "i18n": {
                "title": _(
                    "barcode.modal_confirm_delete.title.singular"
                ),  # 'Confirm Delete Barcode'
                "confirmmsg": _(
                    "barcode.modal_confirm_delete.messages.confirmmsg.singular"
                )
                % {"barCodeIdStr": barcode.id_str, "barCodePk": pk},
                "submit": _("barcode.modal_confirm_delete.action.submit"),
                "cancel": _("barcode.modal_confirm_delete.action.cancel"),
            },
        },
    )
    return render_to_response(
        "rundb/common/modal_confirm_delete.html", context_instance=ctx
    )


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
    ip = "127.0.0.1"  # assume, webserver and jobserver on same appliance
    conn = client.connect(ip, settings.JOBSERVER_PORT)
    result.status = "TERMINATED"
    result.save()
    return render_to_json(conn.control_job(pk, signal))


def enc(s):
    """UTF-8 encode a string."""
    return s.encode("utf-8")


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


def configure_system_stats(request):
    ctx = RequestContext(
        request, {"url": reverse("configure_system_stats_data"), "type": "GET"}
    )
    return render_to_response(
        "rundb/configure/configure_system_stats_loading.html", context_instance=ctx
    )


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
    environment_file = "/etc/environment"
    environment = list()
    environment.append(
        "=================================================================="
    )
    environment.append("Environment (/etc/environment)")
    environment.append(
        "=================================================================="
    )
    if os.path.exists(environment_file):
        with open(environment_file, "r") as environment_handle:
            environment += environment_handle.readlines()

    # Create filename for the report
    reportFileName = "/tmp/stats_sys.txt"

    # Stuff the variable into the context object
    ctx = RequestContext(
        request,
        {
            "stats_network": stats_network,
            "stats": stats,
            "stats_dm": stats_dm,
            "raid_stats": raid_stats,
            "environment": "\n".join([line.strip() for line in environment]),
            "reportFilePath": reportFileName,
        },
    )

    # Generate a file from the report
    logger.info("Writing %s" % reportFileName)
    outfile = open(reportFileName, "w")
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
        os.chmod(
            reportFileName,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IRGRP
            | stat.S_IWGRP
            | stat.S_IROTH
            | stat.S_IWOTH,
        )
    except Exception:
        logger.exception("Could not chmod '%s'", reportFileName)

    return render_to_response(
        "rundb/configure/configure_system_stats.html", context_instance=ctx
    )


def raid_info(request, index=0):
    # display RAID info for a drives array from saved file /var/spool/ion/raidstatus.json
    # index is the adapter/enclosure row clicked on services page
    contents = load_raid_status_json()
    try:
        array_status = contents["raid_status"][int(index)]["drives"]
    except Exception:
        array_status = []
    return render_to_response(
        "rundb/configure/modal_raid_info.html",
        {"array_status": array_status},
        context_instance=RequestContext(request),
    )


def raid_info_refresh(request):
    try:
        raidinfo = tasks.get_raid_stats_json.delay().get(timeout=45)
        tasks.post_check_raid_status(raidinfo)
    except celery.exceptions.TimeoutError as err:
        return HttpResponseServerError(
            "RAID status check timed out, taking longer than 45 seconds."
        )
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
    for profile in UserProfile.objects.filter(user__username__in=list(contacts.keys())):
        if request.method == "POST" and str(profile.user) + "-name" in request.POST:
            try:
                profile.name = request.POST.get(str(profile.user) + "-name", "")
                profile.phone_number = request.POST.get(
                    str(profile.user) + "-phone_number", ""
                )
                profile.user.email = request.POST.get(str(profile.user) + "-email", "")
                profile.user.save()
                profile.save()
                updated = True
            except Exception:
                logger.exception(
                    "Error while saving contact info for %s" % profile.name
                )
        else:
            contacts[profile.user.username] = {
                "name": profile.name,
                "phone_number": profile.phone_number,
                "email": profile.user.email,
            }
    if updated:
        tasks.contact_info_flyaway.delay()
    context.update({"contacts": contacts, "UserProfile": labels.UserProfile})


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
    lines = ""
    with open("/etc/timezone", "r") as file:
        lines = [line.rstrip("\n") for line in file]
    TZ = "".join(lines)
    current_zone, current_city = TZ.split("/", 1)
    context.update({"current_TZ": {"zones": current_zone, "cities": current_city}})
    all_zones = []
    all_cities = []
    zones = os.listdir("/usr/share/zoneinfo")
    for zone in zones:
        zonedir = os.path.join("/usr/share/zoneinfo/" + zone)
        if os.path.isdir(zonedir):
            all_zones.append(zone)
            if zone == context["current_TZ"]["zones"]:
                cities = os.listdir(zonedir)
                for city in cities:
                    citydir = os.path.join(zonedir + "/" + city)
                    if os.path.isfile(citydir):
                        all_cities.append(city)
                    elif os.path.isdir(citydir):
                        inner_dir = os.listdir(citydir)
                        for inner_city in inner_dir:
                            all_cities.append(city + "/" + inner_city)
    all_zones.sort()
    all_cities.sort()
    context.update({"all_TZ": {"zones": all_zones, "cities": all_cities}})


def auto_detect_timezone(request):
    from urllib2 import urlopen
    from contextlib import closing

    current_zone = []
    current_city = []
    url = "http://geoip.nekudo.com/api"
    try:
        with closing(urlopen(url, timeout=1)) as response:
            location = json.loads(response.read())
            timezone = location["location"]["time_zone"]
            current_zone, current_city = timezone.split("/", 1)
    except Exception:
        return HttpResponse("error", status=404)
    return HttpResponse(
        json.dumps({"current_zone": [current_zone], "current_city": [current_city]}),
        mimetype="application/json",
    )


def get_all_cities(request, zone=None):
    if not zone:
        return HttpResponse("error", status=404)
    all_cities = []
    cities = os.listdir(os.path.join("/usr/share/zoneinfo/" + zone))

    for city in cities:
        if os.path.isfile(os.path.join("/usr/share/zoneinfo/" + zone + "/" + city)):
            all_cities.append(city)
    all_cities.sort()
    return HttpResponse(json.dumps({"cities": all_cities}), mimetype="application/json")


def edit_email(request, pk=None):
    if pk is None:
        context = {
            "name": ugettext_lazy("configure_configure.email.add_email.title"),
            "method": "POST",
            "url": "/rundb/api/v1/emailaddress/",
            "form": EmailAddressForm(),
        }
    else:
        email = get_object_or_404(EmailAddress, pk=pk)
        context = {
            "name": ugettext_lazy("configure_configure.email.edit_email.title"),
            "method": "PUT",
            "url": "/rundb/api/v1/emailaddress/%s/" % pk,
            "form": EmailAddressForm(instance=email),
        }
    return render_to_response(
        "rundb/configure/modal_configure_edit_email.html",
        context_instance=RequestContext(request, context),
    )


def delete_email(request, pk=None):
    email = get_object_or_404(EmailAddress, pk=pk)
    if request.method == "POST":
        email.delete()
        return HttpResponse()
    elif request.method == "GET":
        ctx = RequestContext(
            request,
            {
                "id": email.email,
                "ids": json.dumps([]),
                "method": "POST",
                "action": reverse("delete_email", args=[pk]),
                "actions": json.dumps([]),
                "i18n": {
                    "title": _(
                        "emailaddress.modal_confirm_delete.title.singular"
                    ),  # 'Confirm Delete Email Address'
                    "confirmmsg": _(
                        "emailaddress.modal_confirm_delete.messages.confirmmsg.singular"
                    )
                    % {"emailAddress": email.email, "emailId": pk},
                    "submit": _("emailaddress.modal_confirm_delete.action.submit"),
                    "cancel": _("emailaddress.modal_confirm_delete.action.cancel"),
                },
            },
        )
        return render_to_response(
            "rundb/common/modal_confirm_delete.html", context_instance=ctx
        )


def get_sge_jobs():
    jobs = {}
    args = ["qstat", "-u", "www-data", "-s"]
    options = (("r", "running"), ("p", "pending"), ("s", "suspended"), ("z", "done"))

    for opt, status in options:
        p1 = subprocess.Popen(
            args + [opt], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout = p1.stdout.readlines()
        jobs[status] = [l.split()[0] for l in stdout if l.split()[0].isdigit()]

    return jobs


def jobStatus(request, jid):
    args = ["qstat", "-j", jid]
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    detailed = p1.stdout.readlines()
    status = "Running"
    if not detailed:
        # try finished jobs
        args = ["qacct", "-j", jid]
        p1 = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        detailed = p1.stdout.readlines()
        if detailed:
            status = "done, exit_status="
            for line in detailed:
                if "exit_status" in line.split()[0]:
                    status += line.split()[1]
        else:
            status = "not found"

    ctxd = {"jid": jid, "status": status, "jInfo": detailed}
    context = RequestContext(request, ctxd)
    return render_to_response("rundb/ion_jobStatus.html", context_instance=context)


def jobDetails(request, jid):
    pk = request.GET.get("result_pk")
    result = get_object_or_404(Results, pk=pk)
    job_list_json = os.path.join(result.get_report_path(), "job_list.json")

    # job_list.json is written by TLScript and will not be available until all jobs are launched
    if not os.path.exists(job_list_json):
        context = RequestContext(request, {"TLS_jid": jid, "summary": None})
        return render_to_response(
            "rundb/configure/services_jobDetails.html", context_instance=context
        )

    with open(job_list_json, "r") as f:
        jobs = json.load(f)

    current_jobs = get_sge_jobs()

    for block, subjobs in list(jobs.items()):
        block_status = "pending"
        for job in subjobs:
            subjob_jid = subjobs[job]
            # get job status
            status = "done"
            for st, job_list in list(current_jobs.items()):
                if subjob_jid in job_list:
                    status = st

            # processing status for the block: show the job that's currently running
            if status == "running":
                block_status = job
            elif status == "done":
                if block == "merge" and job == "merge":
                    block_status = "done"
                elif job == "alignment":
                    block_status = "done"

        jobs[block]["status"] = block_status

    # summary count how many blocks in each category
    # summary_keys = ['pending', 'sigproc', 'basecaller', 'alignment', 'done']
    summary_keys = ["pending", "block_processing", "done"]
    summary_values = [0] * len(summary_keys)
    num_blocks = len(jobs) - 1  # don't count merge block
    for block in jobs:
        if block != "merge" and jobs[block]["status"] in summary_keys:
            indx = summary_keys.index(jobs[block]["status"])
            summary_values[indx] += 1

    context = RequestContext(
        request,
        {
            "TLS_jid": jid,
            "jobs": jobs,
            "summary": zip(summary_keys, summary_values),
            "num_blocks": num_blocks,
        },
    )
    return render_to_response(
        "rundb/configure/services_jobDetails.html", context_instance=context
    )


def queueStatus(request):
    # get cluster queue status
    args = ["qstat", "-g", "c", "-ext"]
    output = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = output.stdout.readlines()
    queues = []
    for line in stdout:
        splitline = line.split()
        if len(splitline) > 1 and ".q" in splitline[0]:
            queues.append(
                {
                    "name": splitline[0],
                    "pending": 0,
                    "used": splitline[2],
                    "avail": splitline[4],
                    "error": splitline[18],
                    "total": splitline[5],
                }
            )

    # get pending jobs per queue
    args = ["qstat", "-u", "www-data", "-q"]
    for queue in queues:
        output = subprocess.Popen(
            args + [queue["name"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout = output.stdout.readlines()
        for line in stdout:
            splitline = line.split()
            if splitline[0].isdigit() and "qw" in splitline[4]:
                queue["pending"] += 1

    context = RequestContext(request, {"queues": queues})
    return render_to_response(
        "rundb/configure/modal_services_queueStatus.html", context_instance=context
    )


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
        approve = User.objects.filter(
            is_active=False, userprofile__needs_activation=True
        )
    else:
        approve = None

    try:
        tfc_account = ThermoFisherCloudAccount.objects.get(
            user_account_id=request.user.id
        )
    except Exception:
        tfc_account = None

    context = RequestContext(
        request,
        {
            "form": form,
            "approve": approve,
            "updated": updated,
            "tfc_account": tfc_account,
            "User": labels.User,
            "SuperUser": labels.SuperUser,
            "StaffUser": labels.StaffUser,
        },
    )
    return render_to_response("rundb/configure/account.html", context_instance=context)


def system_support_archive(request):
    try:
        path, name = makeSSA.makeSSA()
        response = HttpResponse(FileWrapper(open(path)), mimetype="application/zip")
        response["Content-Disposition"] = "attachment; filename=%s" % name
        return response
    except Exception:
        logger.exception("Failed to create System Support Archive")
        return HttpResponseServerError(traceback.format_exc())


@require_POST
def configure_ampliseq_logout(request):
    if "ampliseq_username" in request.session:
        del request.session["ampliseq_username"]
    if "ampliseq_password" in request.session:
        del request.session["ampliseq_password"]
    return HttpResponseRedirect(urlresolvers.reverse("configure_ampliseq"))


# Ignore the below view : used for local dev testing
def configure_ampliseq_local(request):
    """View for ampliseq.com importing stuff"""

    tfc_account = None
    http_error = False
    panelLists = ["ready-to-use", "on-demand", "made-to-order", "your-design"]
    chemistryTypes = ["ampliseq", "ampliseqHD"]
    tfc_account = True

    ctx = {"designs": None, "tfc_account_setup": tfc_account is None}

    if tfc_account:
        ctx["ampliseq_account_update"] = timezone.now() < timezone.datetime(
            2013, 11, 30, tzinfo=timezone.utc
        )
        ctx["ampliseq_url"] = settings.AMPLISEQ_URL
        ctx["panelLists"] = panelLists
        ctx["pipeline"] = None
        ctx["chemistryTypes"] = chemistryTypes

    return render_to_response(
        "rundb/configure/ampliseq.html", ctx, context_instance=RequestContext(request)
    )


def get_ctx_ampliseq(request, panelTab="on-demand"):
    http_error = ""
    # get the thermo fisher cloud account model
    tfc_account = get_ampliseq_act_info(request)
    ampliseq_url = settings.AMPLISEQ_URL

    ctx = {
        "designs": None,
        "tfc_account_setup": tfc_account is None,
        "panelTab": panelTab,
    }

    if tfc_account:
        # Get the list of ampliseq designs panels both custom and fixed for display
        username = tfc_account.username
        password = tfc_account.get_ampliseq_password()
        try:
            base_url = os.path.join(settings.AMPLISEQ_URL, "ws/design/list")
            response = authenticate_fetch_url(username=username, password=password, base_url=base_url)
        except Exception as exc:
            if exc.message == 401:
                http_error = (
                    "Your user name or password is invalid.<br> You may need to log in to "
                    '<a href="https://ampliseq.com/">AmpliSeq.com</a> and check your credentials.'
                )
                tfc_account.delete()
            elif exc.message == 500:
                http_error = (
                    "Error Code-500 : Internal Server Error when contacting ampliseq.com"
                )
            else:
                tfc_account = None
                logger.error("Unknown error: %s" % str(exc))
                http_error = (
                    "Could not connect to ampliseq.com"
                )

        ctx["ampliseq_account_update"] = timezone.now() < timezone.datetime(
            2013, 11, 30, tzinfo=timezone.utc
        )
        ctx["ampliseq_url"] = settings.AMPLISEQ_URL

    if http_error or not tfc_account:
        ctx = {
            "designs": None,
            "tfc_account_setup": tfc_account is None,
            "http_error": http_error,
        }

    if tfc_account:
        ctx["ampliseq_account_update"] = timezone.now() < timezone.datetime(
            2013, 11, 30, tzinfo=timezone.utc
        )
        ctx["ampliseq_url"] = settings.AMPLISEQ_URL
        ctx["recommended_apps"] = get_recommended_apps(request)

    return ctx


def configure_ampliseq(request):
    """View for ampliseq.com importing stuff"""

    ctx = get_ctx_ampliseq(request)

    return render_to_response(
        "rundb/configure/ampliseq.html", ctx, context_instance=RequestContext(request)
    )


def get_ampliseq_act_info(request):
    # get the thermo fisher cloud account model
    tfc_account = None

    try:
        tfc_account = ThermoFisherCloudAccount.objects.get(
            user_account_id=request.user.id
        )
    except ThermoFisherCloudAccount.DoesNotExist:
        tfc_account = None

    return tfc_account


# This method will be called by the multiprocessing pool to get the api response
def get_all_AS_panels(args):
    results = ampliseq_design_parser.ampliseq_concurrent_api_call(*args)
    return results


def get_ampliseq_HD_panels(request):
    tfc_account = get_ampliseq_act_info(request)
    username = tfc_account.username
    password = tfc_account.get_ampliseq_password()
    api_endpoints = {
        "fixed": "ws/tmpldesign/list/active/",
        "others": "ws/design/list",
        "hd": "ws/design/list/?ampliseq-hd=true",
    }
    api_endpoints_list = [
        (username, password, api_endpoints["fixed"]),
        (username, password, api_endpoints["others"]),
        (username, password, api_endpoints["hd"]),
    ]

    # query the api concurrently and get the response
    with ManagedPool(processes=len(api_endpoints_list)) as pool:
        results = pool.map(get_all_AS_panels, api_endpoints_list)

    solutions = []

    for result in results:
        solutions.extend(result.get("ordered_solutions", []))
        solutions.extend(result.get("unordered_solutions", []))
        solutions.extend(result.get("fixed_solutions", []))

    return solutions


"""Local data access for testing"""


def get_local_designs(panelType="ampliseq_local_data"):
    jsonPath = "/opt/ion/iondb/rundb/fixtures/" + panelType + ".json"

    panelDict = {}
    with open(jsonPath, "r") as fh:
        try:
            panelDict = json.load(fh)
        except ValueError:
            logger.exception(
                "Unable to load %s. Probably not a valid JSON file." % jsonPath
            )

    return panelDict


"""Local data access for testing"""


def configure_ampliseq_getGridData_local(request):
    panelType = "ampliseq_local_data"
    ctx = {"designs": None}

    ctx["ampliseq_account_update"] = timezone.now() < timezone.datetime(
        2013, 11, 30, tzinfo=timezone.utc
    )
    ctx["ampliseq_url"] = settings.AMPLISEQ_URL
    panelDict = get_local_designs(panelType=panelType)

    ctx = json.dumps(panelDict, cls=DjangoJSONEncoder)

    return HttpResponse(ctx, content_type="application/json")


def get_recommended_apps(request):
    solutions = get_ampliseq_HD_panels(request)
    recommended_apps = [
        sol.get("recommended_application")
        for sol in solutions
        if sol.get("recommended_application")
    ]
    recommended_apps = list(set(recommended_apps))

    return recommended_apps


def configure_ampliseq_getGridData(request):
    """View for ampliseq.com Grid style"""
    solutions = get_ampliseq_HD_panels(request)

    ctx = json.dumps(
        {"total": len(solutions), "objects": solutions}, cls=DjangoJSONEncoder
    )

    return HttpResponse(ctx, content_type="application/json")


def configure_ampliseq_download(request):
    # get the thermo fisher cloud account model

    tfc_account = get_ampliseq_act_info(request)

    if tfc_account:
        # Get the list of ampliseq designs panels both custom and fixed for display
        username = tfc_account.username
        password = tfc_account.get_ampliseq_password()
        solutions = request.POST.getlist("solutions")
        fixed_solutions = request.POST.getlist("fixed_solutions")

        redirect_panel_tab = "on-demand"
        panel_tab = request.POST.getlist("panel_tab")
        if panel_tab:
            redirect_panel_tab = panel_tab[0]

        meta = {
            "upload_type": publisher_types.AMPLISEQ,
            "username": request.user.username,
        }
        for ids in solutions:
            if len(ids) > 0:
                design_id, solution_id = ids.split(",")
                meta["choice"] = request.POST.get(
                    solution_id + "_instrument_choice", "None"
                )
                start_ampliseq_solution_download(
                    design_id, solution_id, json.dumps(meta), (username, password)
                )
        for ids in fixed_solutions:
            if len(ids) > 0:
                design_id, reference = ids.split(",")
                meta["choice"] = request.POST.get(
                    design_id + "_instrument_choice", "None"
                )
                meta["reference"] = reference
                start_ampliseq_fixed_solution_download(
                    design_id, json.dumps(meta), (username, password)
                )
    ctx = get_ctx_ampliseq(request, panelTab=redirect_panel_tab)

    return render_to_response(
        "rundb/configure/ampliseq.html", ctx, context_instance=RequestContext(request)
    )


def start_ampliseq_solution_download(design_id, solution_id, meta, auth):
    url = urlparse.urljoin(
        settings.AMPLISEQ_URL,
        "ws/design/{0}/solutions/{1}/download/results".format(design_id, solution_id),
    )
    monitor = FileMonitor(url=url, tags="ampliseq_template", status="Queued")
    monitor.save()
    tasks.download_something.apply_async(
        (url, monitor.id),
        {"auth": auth},
        link=tasks.ampliseq_zip_upload.subtask((meta,)),
    )


def start_ampliseq_fixed_solution_download(design_id, meta, auth):
    url = urlparse.urljoin(
        settings.AMPLISEQ_URL, "ws/tmpldesign/{0}/download/results".format(design_id)
    )
    monitor = FileMonitor(url=url, tags="ampliseq_template", status="Queued")
    monitor.save()
    tasks.download_something.apply_async(
        (url, monitor.id),
        {"auth": auth},
        link=tasks.ampliseq_zip_upload.subtask((meta,)),
    )


def cache_status(request):
    import memcache

    host = memcache._Host(settings.CACHES["default"]["LOCATION"])
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
    total_get = stats.get("get_hits", 0) + stats.get("get_misses", 0)
    ctx = dict(
        stats=stats,
        items=sorted(stats.items()),
        hit_rate=(100 * stats.get("get_hits", 0) / total_get) if total_get else 0,
        total_get=total_get,
        time=datetime.datetime.now(),
    )
    return render_to_response(
        "rundb/configure/cache_status.html",
        context_instance=RequestContext(request, ctx),
    )


def cluster_info_refresh(request):
    try:
        thandle = tasks.check_cluster_status()
        thandle.get(timeout=300)
        tasks.post_run_nodetests()
    except Exception:
        return HttpResponseServerError(traceback.format_exc())
    return HttpResponse()


@permission_required("user.is_staff", raise_exception=True)
def cluster_ctrl(request, name, action):
    error = tasks.cluster_ctrl_task.delay(action, name, request.user.username).get(
        timeout=20
    )
    if error:
        return HttpResponse(json.dumps({"error": error}), mimetype="application/json")

    return HttpResponse(
        json.dumps({"status": "%s is %sd" % (name, action.capitalize())}),
        mimetype="application/json",
    )


def cluster_info_log(request, pk):
    nodes = Cruncher.objects.all()
    ct_obj = ContentType.objects.get_for_model(Cruncher)
    title = "Cluster Info log for %s" % nodes.get(pk=pk).name  # TODO: i18n
    eventlog_messages_empty = _("eventlog.messages.empty") % {"title": title}
    ctx = RequestContext(
        request,
        {
            "title": title,
            "pk": pk,
            "cttype": ct_obj.id,
            "eventlog_messages_empty": eventlog_messages_empty,
        },
    )
    return render_to_response("rundb/common/modal_event_log.html", context_instance=ctx)


def cluster_info_history(request):
    nodes = Cruncher.objects.all().values_list("name", flat=True)
    ctx = RequestContext(request, {"nodes": nodes})
    return render_to_response(
        "rundb/configure/clusterinfo_history.html", context_instance=ctx
    )


def configure_analysisargs(request):
    chips = Chip.objects.filter(isActive=True).order_by("name")
    ctx = RequestContext(request, {"chips": chips})
    return render_to_response(
        "rundb/configure/manage_analysisargs.html", context_instance=ctx
    )


def configure_analysisargs_action(request, pk, action):
    if pk:
        obj = get_object_or_404(AnalysisArgs, pk=pk)
    else:
        obj = AnalysisArgs(name="new_parameters")

    if request.method == "GET":
        chips = Chip.objects.filter(isActive=True).order_by("name")
        args_for_uniq_validation = AnalysisArgs.objects.all()
        display_name = "%s (%s)" % (obj.description, obj.name) if obj else ""
        if action == "copy":
            obj.name = obj.description = ""
        elif action == "edit":
            args_for_uniq_validation = args_for_uniq_validation.exclude(pk=obj.pk)

        ctxd = {
            "obj": obj,
            "chips": chips,
            "args_action": action,
            "display_name": display_name,
            "args_for_uniq_validation": args_for_uniq_validation
            # 'uniq_names': json.dumps(args_for_uniq_validation.values_list('name', flat=True)),
            # 'uniq_descriptions': json.dumps(args_for_uniq_validation.values_list('description', flat=True))
        }
        return render_to_response(
            "rundb/configure/modal_analysisargs_details.html",
            context_instance=RequestContext(request, ctxd),
        )

    elif request.method == "POST":
        params = request.POST.dict()
        params["isSystem"] = params["chip_default"] = False
        params["lastModifiedUser"] = request.user
        if action == "copy":
            params["creator"] = request.user
            obj.pk = None

        for key, val in list(params.items()):
            setattr(obj, key, val)
        obj.save()
        return HttpResponseRedirect(urlresolvers.reverse("configure_analysisargs"))


def get_local_ip():
    """Returns an array of IP addresses used by this host"""
    ip_array = []
    cmd = "ifconfig"
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = p.communicate()
    if p.returncode == 0:
        for line in [line for line in stdout.splitlines() if "inet" in line]:
            address = line.split()[1]
            if ":" in address:
                address = address.split(":")[1]
            ip_array.append(address)
    return ip_array


def get_nas_devices(request):
    """Returns json object containing list of IP addresses/hostnames of direct-connected
    NAS devices"""
    # ===== get local instrument network ports from dhcp.conf =====
    addresses = []
    with open("/etc/dhcp/dhcpd.conf") as fh:
        for subnet in [line for line in fh.readlines() if line.startswith("subnet")]:
            addresses.append([subnet.split()[1].strip(), subnet.split()[3].strip()])

    ## ===== Include LAN subnet =====
    # import socket, struct
    # """Read the default gateway directly from /proc."""
    # with open("/proc/net/route") as fh:
    #    for line in fh:
    #        fields = line.strip().split()
    #        if fields[1] != '00000000' or not int(fields[3], 16) & 2:
    #            continue
    #        else:
    #            lan_address = socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    ## Append the LAN address
    # addresses.append(lan_address)

    logger.info("scanning these subnets for NAS: %s", addresses)

    # ===== Get all servers in given address range =====
    devices = []
    for address in addresses:
        # nmap cmdline returns addresses of given range.  Output:
        # "Nmap scan report for localhost (127.0.0.1)"
        cmd = "/usr/bin/nmap -sn %s/%s" % (address[0], cidr_lookup(address[1]))
        logger.info("CMD: %s", cmd)
        p = subprocess.Popen(
            cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, _ = p.communicate()
        if p.returncode == 0:
            for line in [line for line in stdout.splitlines() if "Nmap scan" in line]:
                devices.append(line.replace("(", "").replace(")", "").split()[-1])
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
        p = subprocess.Popen(
            cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate()
        if p.returncode == 0:
            # Filter out S5 and Proton instrument shared volumes:
            if "/results" in stdout or "/sw_results" in stdout:
                continue
            nas_devices.append(device)
        elif p.returncode > 128:
            stderr = "Request timed out"

    myjson = json.dumps({"error": stderr, "devices": nas_devices})
    return HttpResponse(myjson, mimetype="application/json")


def get_avail_mnts(request, ip=None):
    """Returns json object containing list of shares available for the given
    IP address or hostname."""
    # resolve the hostname, determine the IP address
    p = subprocess.Popen(["host", ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        if "domain name pointer" in stdout:
            hostname = stdout.split()[-1].strip()
            hostname = hostname[:-1]  # Last character is always '.'
            ip_address = ip
        else:
            hostname = ip
            ip_address = stdout.split()[-1].strip()
    else:
        hostname = ip
        ip_address = ip
    # determine available mountpoints
    mymounts = []
    cmd = "/usr/bin/timelimit -t 9 -T 1 showmount -e %s" % (hostname)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        answers = stdout.split(":", 1)[1].strip()
        for line in [s.strip() for s in answers.splitlines()]:
            mymounts.append(line)
    elif p.returncode > 128:
        stderr = "Request timed out"
    logger.info("Mounts for %s are %s" % (hostname, mymounts))
    # response
    mnts_json = json.dumps(
        {
            "hostname": hostname,
            "address": ip_address,
            "mount_dir": mymounts,
            "error": stderr.split(":")[-1],
        }
    )
    return HttpResponse(mnts_json, mimetype="application/json")


def get_current_mnts(request):
    """Returns json object containing list of current NFS mounted directories"""
    mymounts = []
    p = subprocess.Popen(
        ["mount", "-t", "nfs,nfs3,nfs4"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = p.communicate()
    if p.returncode == 0:
        for line in [s.strip() for s in stdout.splitlines()]:
            mymounts.append(line)
    # response
    mnts_json = json.dumps({"mount_dir": mymounts, "error": stderr})
    return HttpResponse(mnts_json, mimetype="application/json")


def add_nas_storage(request):
    """
    Edits all_local, adding specified mount entry, calls ansible-playbook
    :parameter request: The web request
    """

    try:
        data = json.loads(request.body)
        servername = data.get("servername")
        sharename = data.get("sharename")
        mountpoint = data.get("mountpoint")
        logger.info(
            "Handling request for %s:%s on %s" % (servername, sharename, mountpoint)
        )

        # create an ansible playbook and which will mount the drive
        subprocess.check_call(
            [
                "sudo",
                "/opt/ion/iondb/bin/ion_add_nfs_mount.py",
                servername,
                sharename,
                mountpoint,
            ]
        )

        # Probe nas unit to identify Nexenta appliance
        if this_is_nexenta(servername):
            logger.info("%s is TorrentNAS", servername)
            # Update credentials file
            subprocess.call(
                [
                    "sudo",
                    "/opt/ion/iondb/bin/write_nms_access.py",
                    "--id",
                    "%s" % servername,
                ]
            )
    except Exception:
        logger.error(traceback.format_exc())
    return HttpResponse()


def remove_nas_storage(request):
    """
    Edits all_local, removing specified mount entry, calls ansible-playbook
    Removes entry from nms_access file
    """
    try:
        data = json.loads(request.body)
        servername = data.get("servername")
        mountpoint = data.get("mountpoint")
        logger.info(
            "Handling request to remove %s", mountpoint
        )  # create an ansible playbook and which will mount the drive
        subprocess.check_call(
            ["sudo", "/opt/ion/iondb/bin/ion_remove_nfs_mount.py", mountpoint]
        )
        # Update credentials file
        subprocess.call(
            [
                "sudo",
                "/opt/ion/iondb/bin/write_nms_access.py",
                "--remove",
                "--id",
                "%s" % servername,
            ]
        )
    except Exception:
        logger.error(traceback.format_exc())
    return HttpResponse()


def check_nas_perms(request):
    logger.debug(
        "User %s is authorized? %s" % (request.user, request.user.is_superuser)
    )
    myjson = json.dumps({"error": "", "authorized": request.user.is_superuser})
    return HttpResponse(myjson, mimetype="application/json")


def offcycle_updates(request, offcycle_type="online"):
    ctx = {
        "user_can_update": request.user.is_active and request.user.is_staff,
        "plugins": updateProducts.get_update_plugins(),
        "products": updateProducts.get_update_products(offcycle_type=offcycle_type),
        "instruments": updateProducts.get_update_packages(),
    }
    if offcycle_type == "manual":
        return HttpResponse()

    return render_to_response(
        "rundb/configure/offcycleUpdates.html",
        context_instance=RequestContext(request, ctx),
    )


@permission_required("user.is_staff", raise_exception=True)
def offcycle_updates_install_product(request, name, version):
    try:
        updateProducts.update_product(name, version)
        return HttpResponse()
    except Exception as err:
        return HttpResponseServerError(err)


@permission_required("user.is_staff", raise_exception=True)
def offcycle_updates_install_package(request, name, version):
    try:
        updateProducts.update_package(name, version)
        return HttpResponse()
    except Exception as err:
        return HttpResponseServerError(err)


@permission_required("user.is_staff", raise_exception=True)
def configure_offline_bundle(request):
    """Render offline offcycle install tab"""
    offcycle_localPath = settings.OFFCYCLE_UPDATE_PATH_LOCAL
    if not os.path.exists(offcycle_localPath):
        os.makedirs(offcycle_localPath)
        os.chmod(offcycle_localPath, 0o777)

    ctxd = {
        "i18n": {
            "title": ugettext_lazy(
                "configure_offline_bundle.title"
            ),  # 'Install Updates'
            "fields": {
                "pickfile": {
                    "label": ugettext_lazy(
                        "configure_offline_bundle.fields.pickfile.label"
                    ),  # 'Upload an Updates File'
                    "select": ugettext_lazy(
                        "global.fields.pickfile.select"
                    ),  # Select File
                    "helptext": ugettext_lazy(
                        "global.fields.pickfile.helptext"
                    ),  # 'In order to provide a better uploading experience an HTML5 compatible browser are required for file uploading. You may need to contact your local system administrator for assistance.'
                }
            },
            "submit": ugettext_lazy(
                "configure_offline_bundle.action.submit"
            ),  # 'Upload and Install'
            "cancel": ugettext_lazy("global.action.modal.cancel"),
            "submitmsg": ugettext_lazy(
                "configure_offline_bundle.messages.submitmsg"
            ),  # 'The install process may take a few minutes to complete...'
            "submitsuccessmsg": ugettext_lazy(
                "configure_offline_bundle.messages.submitsuccessmsg"
            ),  # 'Started Install Updates'
            "submitprocessingmsg": ugettext_lazy(
                "configure_offline_bundle.messages.submitprocessingmsg"
            ),  # 'Installing, please wait ...'
        },
        "what": "Install Updates",
        "file_filters": [
            ("deb", labels.FileTypes.deb),
            ("zip", labels.FileTypes.zip),
            ("json", labels.FileTypes.json),
        ],
        "plupload_url": reverse("configure_offline_bundle_upload"),
        "install_url": reverse("configure_offline_install"),
        "install_product_url": reverse(
            "offcycle_updates", kwargs={"offcycle_type": "manual"}
        ),
    }

    return render_to_response(
        "rundb/configure/modal_plugin_or_publisher_install.html",
        context_instance=RequestContext(request, ctxd),
    )


@permission_required("user.is_staff", raise_exception=True)
def configure_offline_install(request, **kwargs):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            offcycle_localPath = settings.OFFCYCLE_UPDATE_PATH_LOCAL
            # get the misc deb file name location
            filename = os.path.join(offcycle_localPath, data["file"])
            actualFileName = data.get("actualFileName", None)
            # check to see if there is an extension
            if not "." in filename:
                return HttpResponseServerError("Cannot identify file type.")

            # get the file extensions
            extension = filename.split(".")[-1]

            if os.stat(filename).st_size == 0:
                raise Exception(
                    "Invalid file (%s) uploaded. Please check the file content and try again "
                    % actualFileName
                )

            # parse the extension
            if extension in ["zip", "json"]:
                updateProducts.InstallProducts(filename, extension, actualFileName)
            elif extension == "deb":
                installPackages = settings.SUPPORTED_INSTALL_PACKAGES
                # since deb file has some version attached to it, also some has (underscore, hypen) after the package name
                # which is not consistent, so perform the reverse comparison
                isSupported = [
                    name
                    for name, description in installPackages
                    if name in actualFileName
                ]
                if not isSupported and "ion-plugin" not in actualFileName:
                    errMsg = "The selected deb pacakge is not supported currently."
                    return HttpResponseServerError(errMsg)
                updateProducts.InstallDeb(filename, actualFileName=actualFileName)
            else:
                return HttpResponseServerError(
                    "The extension " + extension + " is not a valid file type."
                )
        except Exception as err:
            return HttpResponseServerError(str(err))

        finally:
            if "filename" in vars() and filename and os.path.exists(filename):
                os.remove(filename)

    return HttpResponse()
