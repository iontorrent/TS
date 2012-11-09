# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.contrib.auth.decorators import login_required
from django.shortcuts import render_to_response
from django.template.context import RequestContext

from iondb.rundb.api import MonitorExperimentResource
from iondb.rundb.models import GlobalConfig

import logging
logger = logging.getLogger(__name__)

@login_required
def monitor(request):
    """This is a the main entry point to the Monitor tab."""
    pageSize = GlobalConfig.objects.all()[0].records_to_display
    resource = MonitorExperimentResource()
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
    return render_to_response("rundb/monitor/runs_in_progress.html", context,
                              context_instance=RequestContext(request))
