# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# -*- coding: utf-8 -*-

import datetime
import json
import os
import socket
import subprocess
from multiprocessing import Pool

import pytz
from django.core.urlresolvers import reverse
from django.http import Http404
from django.http import HttpResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from ion import version as TS_version

from iondb.rundb.configure.views import process_set
from iondb.rundb.dashboard.definitions import *
from iondb.rundb.dashboard.runs import get_runs_list
from iondb.rundb.models import FileServer, GlobalConfig, Rig, Cruncher, DMFileStat


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
        "alarms": rig.alarms.values() if rig.alarms else []
    }
    if instr['type'] == "Raptor": instr['type'] = "S5"
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


def dashboard_fragments(request):
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
    runs = get_runs_list(DASHBOARD_TIME_SPANS[time_span])
    runs_context = {
        # Runs Section
        "runs": {
            "time_span": time_span,
            "stages": DASHBOARD_STAGES,
            "runs": runs,
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

    # data management
    disk_usage = {}
    for fs in FileServer.objects.all().order_by('pk'):
        if os.path.exists(fs.filesPrefix):
            disk_usage[fs.filesPrefix] = fs.percentfull

    dm_active_jobs = DMFileStat.objects.filter(action_state__in=['AG', 'DG', 'EG', 'SA', 'SE', 'SD', 'IG']).values_list(
        'action_state', flat=True)

    # instruments
    rigs = Rig.objects.exclude(host_address='')
    num_rigs = len(rigs)
    if num_rigs > 1:
        pool = Pool(processes=min(num_rigs, 20))
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
        "summary": render_to_string("rundb/dashboard/fragments/summary.html", summary_context),
        "runs": render_to_string("rundb/dashboard/fragments/runs.html", runs_context),
        "instruments": render_to_string("rundb/dashboard/fragments/instruments.html", instruments_context)
    }), content_type="application/json")


def dashboard(request):
    """ Renders the TS dashboard
    """
    initial_fragments = json.loads(dashboard_fragments(request).content)
    context = {
        "disable_messages": True,
        # Summary Section
        "initial_summary_html": initial_fragments["summary"],
        # Runs Section
        "initial_runs_html": "<span class='muted'>Loading...</span>",
        # Instruments Section
        "initial_instruments_html": initial_fragments["instruments"],
    }
    return render(request, "rundb/dashboard/dashboard.html", context)
