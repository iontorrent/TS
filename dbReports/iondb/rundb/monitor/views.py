# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render_to_response, render
from django.template.context import RequestContext
from django.core.paginator import Paginator, InvalidPage

from iondb.rundb.api import MonitorResultResource
from iondb.rundb.models import GlobalConfig
from iondb.rundb import models
from iondb.utils import utils

import datetime
import json
import logging
import requests
logger = logging.getLogger(__name__)

@login_required
def monitor(request):
    """This is a the main entry point to the Monitor tab."""
    pageSize = GlobalConfig.get().records_to_display
    resource = MonitorResultResource()
    objects = resource.get_object_list(request)
    paginator = resource._meta.paginator_class(
        request.GET,
        objects,
        resource_uri=resource.get_resource_uri(), 
        limit=pageSize, 
        max_limit=resource._meta.max_limit, 
        collection_name=resource._meta.collection_name
    )
    to_be_serialized = paginator.page()
    to_be_serialized['objects'] = [
        resource.full_dehydrate(resource.build_bundle(obj=obj, request=request)) 
        for obj in to_be_serialized['objects']
    ]
    serialized_exps = resource.serialize(None, to_be_serialized, 'application/json')

    context = {
        'initial_runs': serialized_exps,
        'pageSize' : pageSize
    }
    return render_to_response("rundb/monitor/monitor.html", context,
                              context_instance=RequestContext(request))

def instruments(request):
    queryset = models.MonitorData.objects.all()
    # Make a new model if none exist.
    if len(queryset) == 0:
        m1 = models.MonitorData()
        m1.testDat = json.loads('{"test1":{"test2":"inner"},"test2.5":{"test3":"alsoinner"},"test4":"outer"}')
        m1.name = "Debug"
        m1.save()
        queryset = models.MonitorData.objects.all()
    # DEBUG: use this to just keep one treeview model, so it's clean.
    '''else:
        m1 = queryset[0]
        m1.testDat = json.loads('{"Example1":{"Example2":{"Example3":"n","Example4":"n"},"Example5":"n"},"Example6":"n"}')
        m1.name = "Debug"
        m1.save()
        queryset = models.MonitorData.objects.all()'''
    # Clean excess models.
    '''for i in range(2,20):
        try:
            m = models.MonitorData.objects.get(id=i)
            m.delete()
        except:
            pass'''
    # Prepare Monitor Data sets for being passed to the web page.
    queryset = models.MonitorData.objects.all()
    monitors = []
    for mon in queryset:
        #monitors += json.dumps(mon.testDat)
        monitors.append(mon)
    # Prepare Rig data for being passed to the web page.
    queryset = models.Rig.objects.all()
    rigs = []
    for rig in queryset:
        rigs.append(rig)
    context = { 'extra_data' : monitors, 'rigs' : rigs }
    return render_to_response("rundb/monitor/instruments.html", context, context_instance=RequestContext(request))

def getSoftware(request):
    req = requests.get('http://updates.ite/BB/updates/SoftwareVersionList.txt')
    sofText = req.text
    return HttpResponse(sofText)


def get_int(querydict, key, default=0):
    value = default
    try:
        if key in querydict:
            value = int(querydict.get(key))
    except ValueError:
        pass
    return value


def chef(request):
    querydict = request.GET
    days = get_int(querydict, "days", 7)
    size = get_int(querydict, "size", 10)
    page = get_int(querydict, "page", 1)

    days_ago = datetime.datetime.now() - datetime.timedelta(days=days)

    # monitoring page should show Chef plans that are currently in progress or have recently finished Chef processing, 
    # regardless if they are being or have been sequenced
    plans = models.PlannedExperiment.objects.filter(experiment__chefLastUpdate__gte=days_ago)    
    sampleSets = models.SampleSet.objects.filter(libraryPrepInstrumentData__lastUpdate__gte=days_ago)

    libprep_done = []
    chef_table = []
    # planned runs may have both Template and Library prep data
    for plan in plans:
        data = {
            'planName': plan.planDisplayedName,
            'sampleSetName': ', '.join(plan.sampleSets.order_by('displayedName').values_list('displayedName', flat=True)),
            'last_updated' : plan.experiment.chefLastUpdate,
            'instrumentName': plan.experiment.chefInstrumentName,
            'template_prep_progress': plan.experiment.chefProgress,
            'template_prep_status': plan.experiment.chefStatus,
            'template_prep_operation_mode': plan.experiment.chefOperationMode,
            'template_prep_remaining_time': utils.convert_seconds_to_hhmmss_string(plan.experiment.chefRemainingSeconds),
            'template_prep_estimated_end_time': utils.convert_seconds_to_datetime_string(plan.experiment.chefLastUpdate, plan.experiment.chefRemainingSeconds)           
        }

        samplesets_w_libprep = plan.sampleSets.filter(libraryPrepInstrumentData__isnull=False)
        if samplesets_w_libprep:
            libprep_done.extend(list(samplesets_w_libprep.values_list('pk', flat=True)))
            data.update({
                'lib_prep_progress': samplesets_w_libprep[0].libraryPrepInstrumentData.progress,
                'lib_prep_status': samplesets_w_libprep[0].libraryPrepInstrumentData.instrumentStatus
            })
        chef_table.append(data)

    # add sample sets with Library prep only
    for sampleSet in sampleSets:
        if sampleSet.pk not in libprep_done:
            chef_table.append({
                'planName': '',
                'sampleSetName': sampleSet.displayedName,
                'sample_prep_type': 'Library Prep',
                'last_updated' : sampleSet.libraryPrepInstrumentData.lastUpdate,
                'instrumentName': sampleSet.libraryPrepInstrumentData.instrumentName,
                'lib_prep_progress': sampleSet.libraryPrepInstrumentData.progress,
                'lib_prep_status': sampleSet.libraryPrepInstrumentData.instrumentStatus,
                'operation_mode': sampleSet.libraryPrepInstrumentData.operationMode,
                'remaining_time': utils.convert_seconds_to_hhmmss_string(sampleSet.libraryPrepInstrumentData.remainingSeconds),
                'estimated_end_time': utils.convert_seconds_to_datetime_string(sampleSet.libraryPrepInstrumentData.lastUpdate, sampleSet.libraryPrepInstrumentData.remainingSeconds)
            })

    chef_table = sorted(chef_table, key=lambda plan: plan['last_updated'], reverse=True)

    chef_pager = Paginator(chef_table, size, 0)
    try:
        chef_page = chef_pager.page(page)
    except InvalidPage:
        chef_page = chef_pager.page(1)

    context = {
        "plans": chef_page
    }
    return render(request, "rundb/monitor/chef.html", context)
