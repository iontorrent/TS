# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django import http, template
from django.core import urlresolvers
from django.core import serializers
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import RequestContext
from django.core.exceptions import ObjectDoesNotExist
from django.utils import simplejson
from tastypie.bundle import Bundle
import json
import datetime
import string
import cStringIO
import csv
from os import path
from iondb.rundb.models import (
    Experiment,
    Results,
    Project,
    Location,
    ReportStorage,
    EventLog
)
from iondb.rundb.api import CompositeExperimentResource, ProjectResource
from iondb.rundb.views import build_result, _edit_experiment, _report_started
from iondb.anaserve import client
from iondb.rundb import models

from django.http import HttpResponse, HttpResponseServerError
from datetime import datetime
import logging
from django.core.urlresolvers import reverse
from django.db.models.query_utils import Q
from urllib import unquote_plus
logger = logging.getLogger(__name__)

def get_search_parameters():
    experiment_params = {
        'sample':[],
        'library':[],
        'flows':[],
        'chipType': [],
        'pgmName': [],
    }
    report_params = {
        'status':[],
        'processedflows':[],
    }
    for key in experiment_params.keys():
        experiment_params[key] = list(Experiment.objects.values_list(key, flat=True).distinct(key).order_by(key))
    for key in report_params.keys():
        report_params[key] = list(Results.objects.values_list(key, flat=True).distinct(key).order_by(key))
    combined_params = {
        'flows': sorted(set(experiment_params['flows'] + report_params['processedflows'])),
        'projects': Project.objects.values_list('name', flat=True).distinct('name').order_by('name')
    }
    del experiment_params['flows']
    del report_params['processedflows']
    return {'experiment': experiment_params,
            'report': report_params,
            'combined': combined_params,
    }

@login_required
def rundb_redirect(request):
    ## Old /rundb/ page redirects to /data/, keeps args
    url = reverse('data') or '/data/'
    args = request.META.get('QUERY_STRING', '')
    if args:
        url = "%s?%s" % (url, args)
    return redirect(url, permanent=True)

@login_required
def data(request):
    """This is a the main entry point to the Data tab."""
    pageSize = models.GlobalConfig.objects.all()[0].records_to_display
    resource = CompositeExperimentResource()
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
    #This needs some fixing...
    data = {
        'search': get_search_parameters(),
        'inital_query': serialized_exps,
        'pageSize' : pageSize
    }
    return render_to_response("rundb/data/data.html", data,
                              context_instance=RequestContext(request))
def data_table(request):
    data = {
        'search': get_search_parameters()
    }    
    return render_to_response("rundb/data/completed_table.html", data,
                              context_instance=RequestContext(request))
    
def getCSV(request):
    CSVstr = ""
    
    if request.method == "GET":
        qDict = request.GET
    elif request.method == "POST":
        qDict = request.POST
    else:
        raise ValueError('Unsupported HTTP METHOD')
    
    try:
        base_object_list = models.Results.objects.select_related('experiment').prefetch_related('libmetrics_set', 'tfmetrics_set', 'analysismetrics_set', 'pluginresult_set__plugin') \
        .exclude(experiment__expName__exact="NONE_ReportOnly_NONE")
        if qDict.get('results__projects__name', None) is not None:
            base_object_list = base_object_list.filter(projects__name__exact = qDict.get('results__projects__name', None))
        if qDict.get('sample', None) is not None:
            base_object_list = base_object_list.filter(experiment__sample__exact = qDict.get('sample', None))
        if qDict.get('chipType', None) is not None:
            base_object_list = base_object_list.filter(experiment__chipType__exact = qDict.get('chipType', None))
        if qDict.get('pgmName', None) is not None:
            base_object_list = base_object_list.filter(experiment__pgmName__exact = qDict.get('pgmName', None))
        if qDict.get('library', None) is not None:
            base_object_list = base_object_list.filter(experiment__library__exact = qDict.get('library', None))
        if qDict.get('flows', None) is not None:
            base_object_list = base_object_list.filter(experiment__flows__exact = qDict.get('flows', None))
        if qDict.get('star', None) is not None:
            base_object_list = base_object_list.filter(experiment__star__exact = bool(qDict.get('star', None)))
        
        name = qDict.get('all_text', None)
        if name is not None:
            qset = (
                Q(experiment__expName__icontains=name) |
                Q(resultsName__icontains=name) |
                Q(experiment__notes__icontains=name)
            )
            base_object_list = base_object_list.filter(qset)

        date = qDict.get('all_date', None)
        logger.debug("Got all_date='%s'" % str(date))
        if date is not None:
            date = unquote_plus(date)
            date = date.split(',')
            logger.debug("Got all_date='%s'" % str(date))
            qset = (
                Q(experiment__date__range=date) |
                Q(timeStamp__range=date)
            )
            base_object_list = base_object_list.filter(qset)
        status = qDict.get('result_status', None)
        if status is not None:
            if status == "Completed":
                qset = Q(status="Completed")
            elif status == "Progress":
                qset = Q(status__in=("Pending", "Started", 
                    "Signal Processing", "Base Calling", "Alignment"))
            elif status == "Error":
                # This list may be incomplete, but a coding standard needs to
                # be established to make these more coherent and migration
                # written to normalize the exisitng entries
                qset = Q(status__in=(
                    "Failed to contact job server.",
                    "Error",
                    "TERMINATED",
                    "Checksum Error",
                    "Moved",
                    "CoreDump",
                    "ERROR",
                    "Error in alignmentQC.pl",
                    "MissingFile",
                    "Separator Abort",
                    "PGM Operation Error",
                    "Error in Reads sampling with samtools",
                    "No Live Beads",
                    "Error in Analysis",
                    "Error in BaseCaller"
                    ))
            base_object_list = base_object_list.filter(qset)        
        base_object_list.distinct()            
        table = models.Results.to_pretty_table(base_object_list)
        CSVstr = cStringIO.StringIO()
        writer = csv.writer(CSVstr)
        writer.writerows(table)
        CSVstr.seek(0)
    except Exception as err:
        logger.error("During result CSV generation: %s" % err)
        raise
    ret = http.HttpResponse(CSVstr, mimetype='text/csv')
    now = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    ret['Content-Disposition'] = 'attachment; filename=metrics_%s.csv' % now
    return ret

def projects(request):
    ctx = template.RequestContext(request)
    return render_to_response("rundb/data/projects.html", context_instance=ctx)


def project_view(request, pk=None):
    pr = ProjectResource()
    project = pr.obj_get(pk = pk)
    
    pr_bundle = pr.build_bundle(obj=project, request=request)
    
    return render_to_response("rundb/data/modal_project_details.html", {
        # Other things here.
        "project_json": pr.serialize(None, pr.full_dehydrate(pr_bundle), 'application/json')
        , "project" : project
        , "method":"GET"
        , "readonly":True
        , 'action': reverse('api_dispatch_detail', kwargs={'resource_name':'project', 'api_name':'v1', 'pk':int(pk)}, args={'format':'json'})
    })
    
def project_add(request, pk=None):
    otherList = [p.name for p in Project.objects.all()]    
    ctx = template.RequestContext(request, {
                                            'id':pk
                                            , 'otherList' : json.dumps(otherList)
                                            , 'method':'POST'
                                            , 'methodDescription': 'Add'
                                            , 'readonly':False
                                            , 'action': reverse('api_dispatch_list', kwargs={'resource_name':'project', 'api_name':'v1'})
                                            })
    return render_to_response("rundb/data/modal_project_details.html", context_instance=ctx)

def project_delete(request, pk=None):
    pr = ProjectResource()
    project = pr.obj_get(pk = pk)
    _type = 'project';
    ctx = template.RequestContext(request, { 
                                            "id":pk
                                            , "name": project.name
                                            , "method":"DELETE"
                                            , 'methodDescription': 'Delete'
                                            , "readonly":False
                                            , 'type':_type
                                            , 'action': reverse('api_dispatch_detail', kwargs={'resource_name':_type, 'api_name':'v1', 'pk':int(pk)})
                                            })
    return render_to_response("rundb/data/modal_confirm_delete.html", context_instance=ctx)

def project_edit(request, pk=None):
    pr = ProjectResource()
    project = pr.obj_get(pk = pk)
    pr_bundle = pr.build_bundle(obj=project, request=request)
    
    otherList = [p.name for p in Project.objects.all()] 
    return render_to_response("rundb/data/modal_project_details.html", {
        # Other things here.
        "project_json": pr.serialize(None, pr.full_dehydrate(pr_bundle), 'application/json')
        , "project" : project
        , "id":pk
        , 'otherList' : json.dumps(otherList)
        , "method":"PATCH"
        , 'methodDescription': 'Edit' 
        , "readonly":True 
        , 'action': reverse('api_dispatch_detail', kwargs={'resource_name':'project', 'api_name':'v1', 'pk':int(pk)})
    })

def project_log(request, pk=None):
    if request.method == 'GET':
        selected = get_object_or_404(Project, pk=pk)
        log = EventLog.objects.for_model(Project).filter(object_pk = pk)
        ctx = template.RequestContext(request, {"project":selected, "event_log":log})
        return render_to_response("rundb/data/modal_project_log.html", context_instance=ctx)
    if request.method == 'POST':
        try: 
            pk = int(pk)
        except:  
            pk = request.REQUEST['url'].split('/')[-2]
#        logger.debug('project_log post %s pk=%s' % (request.REQUEST, pk))        
        try:    
            project = Project.objects.get(pk=pk)   
            if request.REQUEST['type'] == 'PATCH':
                message = 'Edit project name= %s.' % project.name
            elif request.REQUEST['type'] == 'DELETE':
                message = 'Delete project requested.'
            elif request.REQUEST['type'] == 'POST':
                message = 'Created project name= %s.' % project.name                        
            EventLog.objects.add_entry(project, message, request.user.username)
        except Exception as e:
            logger.exception(e)
        return HttpResponse()

def project_results(request, pk):
    selected = get_object_or_404(Project, pk=pk)
    ctx = template.RequestContext(request, {"project":selected})
    return render_to_response("rundb/data/project_results.html", context_instance=ctx)

def results_from_project(request, results_pks, project_pk):
    action = 'Remove'
    try:
        _results_to_project_helper(request, results_pks, project_pk, action)
        return HttpResponse()
    except:     
        return HttpResponseServerError('Errors occurred while processing your request')
        
def results_to_project(request, results_pks):
    if request.method == 'GET':
        ctx = template.RequestContext(request, {
                                                "results_pks":results_pks
                                                #TODO: write url using url dispatcher and format accordingly
                                                , "action": urlresolvers.reverse('results_to_project', args=[results_pks,])
#                                                , "action": '/data/results/%s/combine/project/%s/' % (results_pks, project_pk)
                                                , "method": 'POST'})
        return render_to_response("rundb/data/modal_projects_select.html", context_instance=ctx)        
    if request.method == 'POST':
        json_data = simplejson.loads(request.raw_post_data)
        try:
            project_pks = json_data['projects']
        except KeyError:
            return HttpResponseServerError("Missing 'projects' attribute!")        
        action = 'Add'
        try:
            for project_pk in project_pks:
                _results_to_project_helper(request, int(project_pk), results_pks, action)
            return HttpResponse()
        except Exception as e:
            logger.exception(e)
            raise     
            return HttpResponseServerError('Errors occurred while processing your request')
    
def _results_to_project_helper(request, project_pk, result_pks, action):
    project = Project.objects.get(pk=project_pk)
    
    for result_pk in result_pks.split(','):
        result = Results.objects.get(pk = int(result_pk))
        if action == 'Add': 
            result.projects.add(project)  
        elif action == 'Remove':
            result.projects.remove(project)
    
    # log project history
    message = '%s results (%s).' % (action, result_pks)
    EventLog.objects.add_entry(project, message, request.user.username)
    
def validate_results_to_combine(selected_results):
    # validate selected reports
    warnings = []        
    ver_map = {'analysis':'an','alignment':'al','dbreports':'db', 'tmap' : 'tm' }        
    for result in selected_results:            
        version = {}
        for name, shortname in ver_map.iteritems():
            version[name] = next(( v.split(':')[1].strip() for v in result.analysisVersion.split(',') if v.split(':')[0].strip() == shortname ), '')
            setattr(result, name+"_version", version[name])
        result.floworder = result.experiment.flowsInOrder
        result.barcodeId = result.experiment.barcodeId
        
    if len(set([getattr(r,'tmap_version') for r in selected_results])) > 1:
        warnings.append("Selected results have different TMAP versions.")
    if len(set([getattr(r,'alignment_version') for r in selected_results])) > 1:
        warnings.append("Selected results have different Alignment versions.")  
    if len(set([r.floworder for r in selected_results if r.resultsType != 'CombinedAlignments'])) > 1:
        warnings.append("Selected results have different FlowOrder Sequences.")
    if len(set([r.barcodeId for r in selected_results if r.resultsType != 'CombinedAlignments'])) > 1:
        warnings.append("Selected results have different Barcode Sets.")    
    return warnings        
            
def results_to_combine(request, results_pks, project_pk): 
    if request.method == 'GET':        
        selected_results = Results.objects.filter(id__in=results_pks.split(',')).order_by('-timeStamp')
        warnings = validate_results_to_combine(selected_results)        
        
        ctx = template.RequestContext(request, {
                                                "results_pks":results_pks
                                                , "project_pk": project_pk
                                                , "selected_results": selected_results
                                                , "warnings": warnings
                                                #TODO: write url using url dispatcher and format accordingly
                                                , "action": urlresolvers.reverse('results_to_combine', args=(results_pks,project_pk))
#                                                , "action": '/data/results/%s/combine/project/%s/' % (results_pks, project_pk)
                                                , "method": 'POST'})
        return render_to_response("rundb/data/modal_combine_results.html", context_instance=ctx)
    if request.method == 'POST':
        try:        
            json_data = simplejson.loads(request.raw_post_data)
            result =_combine_results_sendto_project(project_pk, json_data, request.user.username)
            ctx = _report_started(request, result.pk)
            return render_to_response("rundb/reports/analysis_started.html",  context_instance=ctx)
        except Exception as e:
            return HttpResponseServerError("Error: %s" % e)
        
def _combine_results_sendto_project(project_pk, json_data, username=''):
    project = Project.objects.get(id=project_pk)
    projectName = project.name
    
    name = json_data['name']
    mark_duplicates = json_data['mark_duplicates']
    ids_to_merge = json_data['selected_pks']
         
    # check reference and flow order the same in all selected results    
    for pk in ids_to_merge:
        result = Results.objects.get(pk = pk)
        if pk == ids_to_merge[0]:
            reference = result.reference
            floworder = result.experiment.flowsInOrder
        elif not reference == result.reference:          
            raise Exception("Selected results do not have the same Alignment Reference.")
        elif not floworder == result.experiment.flowsInOrder:
            floworder = ''
                
    # create new entry in DB for combined Result
    delim = ':'
    filePrefix = "CombineAlignments" # this would normally be Experiment name that's prefixed to all filenames
    result = create_combined_result('CA_%s_%s' % (name,projectName))
    result.projects.add(project)
    result.resultsType = 'CombinedAlignments'
    result.parentIDs = delim + delim.join(ids_to_merge) + delim  
    result.reference = reference
    result.sffLink = path.join(result.reportLink, "%s_%s.sff" % (filePrefix, result.resultsName))
    result.save()
    
    # gather parameters to pass to merging script
    links = []
    bams = []  
    names = []
    plan = {}
    parents = Results.objects.filter(id__in=ids_to_merge).order_by('-timeStamp')
    for parent in parents:
        links.append(parent.reportLink)
        names.append(parent.resultsName)
        bamFile = path.split(parent.sffLink)[1].rstrip('.sff') + '.bam'
        bams.append(path.join(parent.get_report_dir(), bamFile))
        
        #need Plan info for VariantCaller etc. plugins: but which plan to use??
        try:        
            planObj = [parent.experiment.plan]  
            plan_json = serializers.serialize("json", planObj)
            plan_json = json.loads(plan_json)
            plan = plan_json[0]["fields"]
        except: 
            pass   
    
    try:
        genome = models.ReferenceGenome.objects.all().filter(short_name=reference,index_version=settings.TMAP_VERSION,enabled=True)[0]
        if path.exists(genome.info_text()):
            genomeinfo = genome.info_text()
        else:
            genomeinfo = ""    
    except:
        genomeinfo = ""   
        
    params = {
      'resultsName':result.resultsName,
      'parentIDs':ids_to_merge,
      'parentNames':names,
      'parentLinks':links,
      'parentBAMs':bams,
      'libraryName': result.reference,
      'tmap_version': settings.TMAP_VERSION,
      'mark_duplicates': mark_duplicates,
      'plan': plan,
      'run_name': filePrefix,
      'genomeinfo': genomeinfo,
      'warnings': validate_results_to_combine(parents)
    }
    
    scriptpath='/usr/lib/python2.6/dist-packages/ion/reports/combineReports.py'
    
    try:
        with open(scriptpath, "r") as f:
            script = f.read()
    except Exception as error:
        result.status = "Error reading %s\n%s" % (scriptpath, error.args)
        raise Exception(result.status)      
    
    files = [] 
     # do we need expMeta?   
    lines = ("Project = %s" % ','.join(p.name for p in result.projects.all()),
             "Library = %s" % result.reference,
             "Analysis Name = %s" % result.resultsName,
             "Flow Order = %s" % floworder,
             "Run Name = %s" % filePrefix
             )
    files.append(('expMeta.dat', '\n'.join(lines)))
    files.append(("primary.key", "ResultsPK = %s" % result.pk))
    
    webRootPath = result.web_root_path(_location()) 
      
    try:
        host = "127.0.0.1"
        conn = client.connect(host, settings.JOBSERVER_PORT)
        conn.startanalysis(result.resultsName, script, params, files, webRootPath, result.pk, '',{},'combineAlignments')
    except:
        result.status = "Failed to contact job server."
        raise Exception(result.status)
    
    
    # log project history
    message = 'Combine results %s into name= %s (%s), auto-assign to project name= %s (%s).' % (ids_to_merge, result.resultsName, result.pk, projectName, project_pk)
    EventLog.objects.add_entry(project, message, username)
          
    return result


def create_combined_result(resultsName):
    # create Results entry in DB without any Experiment (creates blank Exp)
    exp = _blank_Exp('NONE_ReportOnly_NONE')
    last = 0
    otherResults = Results.objects.filter(resultsName__contains=resultsName).order_by('pk')
    if otherResults:
        lastName = otherResults[len(otherResults)-1].resultsName
        last = int(lastName.split('_')[-1])     
    resultsName = "%s_%03d" % (resultsName, last+1)
    
    storages = ReportStorage.objects.all().order_by('id')
    storage = storages[len(storages) - 1]

    result = build_result(exp, resultsName, storage, _location())
    return result

def _location():
    loc = Location.objects.filter(defaultlocation=True)
    if not loc:
        #if there is not a default, just take the first one
        loc = Location.objects.all().order_by('pk')
        if not loc:    
            logger.critical("There are no Location objects")
            raise ObjectDoesNotExist("There are no Location objects, at all.")  
    return loc[0]   

def _blank_Exp(blankName):
    # create blank experiment if doesn't already exist
    try:
        ret = Experiment.objects.get(expName=blankName)   
    except Experiment.DoesNotExist:     
        kwargs = {}
        for field in Experiment._meta.fields: 
            if field.name == 'expName':
                kwargs[field.name] = blankName      
            elif field.name == 'date':
                kwargs[field.name] = datetime.now()
            elif not field.null:                      
                if field.get_internal_type() == 'CharField' or field.get_internal_type() == 'TextField':
                    kwargs[field.name] = ""
            elif field.get_internal_type() == 'BooleanField':
                kwargs[field.name] = False
            elif field.get_internal_type() == 'IntegerField':
                kwargs[field.name] = 0  
            else: # only id should remain 
                logging.debug(field.name, field.get_internal_type())
                pass
        kwargs['cycles'] = 1
        kwargs['flows'] = 1
        kwargs['barcodeId'] = '' 
        ret = Experiment(**kwargs)
        ret.save()
    return ret

@login_required
def experiment_edit(request, pk):
    context = _edit_experiment(request, pk)
    return render_to_response("rundb/data/modal_experiment_edit.html", context_instance=context)

