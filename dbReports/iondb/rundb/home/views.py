# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# -*- coding: utf-8 -*-

import datetime
import json
import os
import socket
import subprocess
from multiprocessing import Pool

import pytz
from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.http import Http404
from django.http import HttpResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from django.utils import timezone
from ion import version as TS_version

from iondb.rundb.configure.views import process_set, get_torrent_nas_info, load_raid_status_json
from iondb.rundb.home.definitions import *
from iondb.rundb.home.runs import get_runs_list
from iondb.rundb.models import FileServer, Rig, Cruncher, DMFileStat, NewsPost, GlobalConfig
from iondb.utils import devices
from iondb.utils.files import get_disk_attributes_gb, is_mounted


def is_live(ip):
    # test is the instrument online
    try:
        socket.gethostbyaddr(ip)
    except:
        # try ping
        ping = ['ping', '-c 2', '-W 1', ip]
        try:
            subprocess.check_call(ping)
        except:
            return False

    return True


def get_instrument_info(rig):
    # returns dict with Rig info
    instr = {
        "name": rig.name,
        "type": rig.type or 'PGM',
        "last_init": format_date(rig.last_init_date),
        "last_clean": format_date(rig.last_clean_date),
        "last_experiment": format_date(rig.last_experiment),
        "alarms": rig.alarms.values() if rig.alarms else [],
        "image_url": ''
    }
    if instr['type'] == "Raptor": instr['type'] = "S5"
    if instr['type'] in INSTRUMENT_TYPES:
        instr['image_url'] = "resources/img/instrument_icons/%s.png" % instr['type'].lower()

    if is_live(rig.host_address):
        instr['display_state'] = rig.get_display_state() or CONNECTED

        if rig.alarms:
            instr['status'] = ALARM
        else:
            instr['status'] = CONNECTED
    else:
        instr['display_state'] = instr['status'] = OFFLINE

    return instr


def get_disk_usage():
    disk_usage = []
    paths = []
    for fs_path, percentfull in FileServer.objects.order_by('pk').values_list('filesPrefix', 'percentfull'):
        if os.path.exists(fs_path):
            disk_usage.append((fs_path, percentfull))
            paths.append(fs_path)
            mounted = is_mounted(os.path.realpath(fs_path))
            if mounted:
                paths.append(mounted)

    archive_folders = devices.to_media(devices.disk_report())
    for bdir, name in archive_folders:
        mounted = is_mounted(bdir)
        if mounted and bdir not in paths:
            disk_usage.append((bdir, get_disk_attributes_gb(bdir).get('percentfull')))
            paths.append(bdir)

    return disk_usage


def get_storage_status():
    # Torrent NAS and RAID info, if any
    # this does not actually check the status directly but loads cached info
    def combined_state(states):
        if 'error' in states:
            return 'error'
        elif 'warning' in states:
            return 'warning'
        else:
            return 'good'

    status = {
        'show_nas': False,
        'show_raid': False
    }
    nasInfo = get_torrent_nas_info(refresh=False)
    if nasInfo:
        status['show_nas'] = True
        nas_states = sum([[volume['state'] for volume in nas['volumes']] for nas in nasInfo], [])
        status['nas_status'] = combined_state(nas_states)

    raidInfo = load_raid_status_json()
    if raidInfo and raidInfo['raid_status']:
        status['show_raid'] = True
        raid_states = [encl['status'] for encl in raidInfo['raid_status']]
        status['raid_status'] = combined_state(raid_states)

    return status


def format_date(date):
    obj_date = None
    try:
        obj_date = datetime.datetime.strptime(date, "%Y/%m/%d %H:%M:%S")
    except:
        try:
            obj_date = datetime.datetime.strptime(date, "%Y_%m_%d_%H_%M_%S")
        except:
            pass

    if obj_date:
        return obj_date.strftime("%b %d %Y %I:%M %p")
    else:
        return date


@login_required
def dashboard_fragments(request, skip_runs=False):
    """ Returns the dashboard sections as html in a json object"""
    time_span = request.GET.get("time_span", "24hours")
    now = datetime.datetime.now(pytz.UTC)

    DASHBOARD_TIME_SPANS = {
        "hour": now - datetime.timedelta(hours=1),
        "today": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "24hours": now - datetime.timedelta(hours=24),
        "7days": now - datetime.timedelta(days=7),
        # Used for testing only. Do not expose to the UI.
        "__all__": datetime.datetime(year=1971, month=1, day=1),
    }
    if time_span not in DASHBOARD_TIME_SPANS:
        raise Http404("Time span %s not available!" % time_span)

    # runs section
    if skip_runs:
        runs_context = {
            # Runs Section
            "runs": {
                "time_span": time_span,
                "stages": DASHBOARD_STAGES,
                "runs": [],
                "error": ""
            },
        }
    else:
        try:
            runs = get_runs_list(DASHBOARD_TIME_SPANS[time_span])
            runs_error = None
        except Exception as err:
            runs = []
            runs_error = str(err)

        runs_context = {
            # Runs Section
            "runs": {
                "time_span": time_span,
                "stages": DASHBOARD_STAGES,
                "runs": runs,
                "error": runs_error
            },
        }

    # software update
    update_status = GlobalConfig.get().ts_update_status

    # services
    services_down = []
    for process, state in process_set():
        if not state:
            services_down.append(process)

    show_cluster = False
    nodes_down = []
    if Cruncher.objects.count() > 0:
        show_cluster = True
        nodes_down = Cruncher.objects.exclude(state='G').values_list('name', flat=True)

    # storage status
    storage = get_storage_status()

    # data management
    disk_usage = get_disk_usage()

    dm_active_jobs = DMFileStat.objects.filter(action_state__in=['AG', 'DG', 'EG', 'SA', 'SE', 'SD', 'IG']).values_list(
        'action_state', flat=True)

    # instruments
    rigs = Rig.objects.exclude(host_address='')
    num_rigs = len(rigs)
    if num_rigs > 1:
        pool = Pool(processes=min(num_rigs, 50))
        instruments = pool.map(get_instrument_info, rigs)
    else:
        instruments = [get_instrument_info(rig) for rig in rigs]

    instr_connected = sum([instr['status'] == CONNECTED for instr in instruments])
    instr_offline = sum([instr['status'] == OFFLINE for instr in instruments])
    instr_alarm = sum([instr['status'] == ALARM for instr in instruments])

    summary_context = {
        # Summary Section
        "summary": {
            "ts_version": TS_version,
            "update_status": update_status,
            "instruments": {
                "connected": instr_connected,
                "offline": instr_offline,
                "alerts": instr_alarm,
            },
            "services": {
                "url": reverse("configure_services"),
                "number_services_down": len(services_down),
                "services_down": services_down,
                "show_cluster": True if show_cluster else False,
                "number_nodes_down": len(nodes_down) if show_cluster else "",
                "show_nas": storage['show_nas'],
                "nas_status": storage.get('nas_status', ''),
                "show_raid": storage['show_raid'],
                "raid_status": storage.get('raid_status', ''),
            },
            "data_management": {
                "url": reverse("datamanagement"),
                "disk_usage": disk_usage,
                "show_path": len(disk_usage) > 1,
                "dm_jobs": [
                    ("archive in progress", sum([s == 'AG' for s in dm_active_jobs])),
                    ("export in progress", sum([s == 'EG' for s in dm_active_jobs])),
                    ("delete in progress", sum([s == 'DG' for s in dm_active_jobs])),
                    ("import in progress", sum([s == 'IG' for s in dm_active_jobs])),
                    ("archive pending", sum([s == 'SA' for s in dm_active_jobs])),
                    ("export pending", sum([s == 'SE' for s in dm_active_jobs])),
                    ("delete pending", sum([s == 'SD' for s in dm_active_jobs])),
                ]
            }
        },
    }

    instruments_context = {
        "instruments": sorted(instruments, key=lambda x: (x['status'], x['name'].lower()))
    }

    return HttpResponse(json.dumps({
        "summary": render_to_string("rundb/home/fragments/summary.html", summary_context),
        "runs": render_to_string("rundb/home/fragments/runs.html", runs_context),
        "instruments": render_to_string("rundb/home/fragments/instruments.html", instruments_context)
    }), content_type="application/json")


@login_required
def dashboard(request):
    """ Renders the TS dashboard
    """
    initial_fragments = json.loads(dashboard_fragments(request, skip_runs=True).content)
    context = {
        "disable_messages": True,
        # Summary Section
        "initial_summary_html": initial_fragments["summary"],
        # Runs Section
        "initial_runs_html": "<span class='muted'>Loading...</span>",
        # Instruments Section
        "initial_instruments_html": initial_fragments["instruments"],
    }
    return render(request, "rundb/home/dashboard.html", context)


@login_required
def news(request):
    profile = request.user.userprofile
    ctx = {
        "articles": list(NewsPost.objects.all().order_by('-updated')),
        "last_read": profile.last_read_news_post,
        "is_updating": GlobalConfig.get().check_news_posts,

    }
    profile.last_read_news_post = timezone.now()
    profile.save()
    return render(request, "rundb/home/news.html", ctx)
