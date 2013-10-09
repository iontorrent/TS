# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django import http, template
from django.core import urlresolvers
from django.core import serializers
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import RequestContext
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned

import json
import cStringIO
import csv
from os import path
from iondb.rundb.models import (
    Experiment,
    Results,
    Project,
    Location,
    ReportStorage,
    EventLog, GlobalConfig, ReferenceGenome, RunType, dnaBarcode, Content, KitInfo, ContentType,
    VariantFrequencies, Plugin, LibraryKey, ThreePrimeadapter, ExperimentAnalysisSettings, Sample)
from iondb.rundb.api import CompositeExperimentResource, ProjectResource
from iondb.rundb.report.views import build_result, _report_started
from iondb.rundb import forms
from iondb.anaserve import client
from iondb.rundb.data import dmactions_types

from django.http import HttpResponse, HttpResponseServerError
from datetime import datetime
import logging
from django.core.urlresolvers import reverse
from django.db.models.query_utils import Q
from urllib import unquote_plus
logger = logging.getLogger(__name__)


def get_search_parameters():
    experiment_params = {
        'flows': [],
        'chipType': [],
        'pgmName': [],
    }
    report_params = {
        'processedflows': [],
    }

    eas_keys = [('library','reference')]
    
    for key in experiment_params.keys():
        experiment_params[key] = list(Experiment.objects.values_list(key, flat=True).distinct(key).order_by(key))

    experiment_params['sample'] = list(Sample.objects.filter(status = "run").values_list('name', flat= True).order_by('name'))

    for expkey,key in eas_keys:
        experiment_params[expkey] = list(ExperimentAnalysisSettings.objects.values_list(key, flat=True).distinct(key).order_by(key))        
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


def data_context(request):
    pageSize = GlobalConfig.objects.all()[0].records_to_display
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
    _data = {
        'search': get_search_parameters(),
        'inital_query': serialized_exps,
        'pageSize': pageSize
    }
    return _data


@login_required
def data(request):
    """This is a the main entry point to the Data tab."""
    context = cache.get("data_tab_context")
    if context is None:
        context = data_context(request)
        cache.set("data_tab_context", context, 29)
    return render_to_response("rundb/data/data.html", context,
                              context_instance=RequestContext(request))


def data_table(request):
    data = {
        'search': get_search_parameters()
    }
    return render_to_response("rundb/data/completed_table.html", data,
                              context_instance=RequestContext(request))

def _makeCSVstr(object_list):
    table = Results.to_pretty_table(object_list)
    CSVstr = cStringIO.StringIO()
    writer = csv.writer(CSVstr)
    writer.writerows(table)
    CSVstr.seek(0)
    return CSVstr

def getCSV(request):
    CSVstr = ""

    if request.method == "GET":
        qDict = request.GET
    elif request.method == "POST":
        qDict = request.POST
    else:
        raise ValueError('Unsupported HTTP METHOD')

    try:
        base_object_list = Results.objects.select_related('experiment').prefetch_related('libmetrics_set', 'tfmetrics_set', 'analysismetrics_set', 'pluginresult_set__plugin') \
            .exclude(experiment__expName__exact="NONE_ReportOnly_NONE")
        if qDict.get('results__projects__name', None) is not None:
            base_object_list = base_object_list.filter(projects__name__exact=qDict.get('results__projects__name', None))
        if qDict.get('samples__name', None) is not None:
            base_object_list = base_object_list.filter(experiment__samples__name__exact=qDict.get('samples__name', None))
        if qDict.get('chipType', None) is not None:
            base_object_list = base_object_list.filter(experiment__chipType__exact=qDict.get('chipType', None))
        if qDict.get('pgmName', None) is not None:
            base_object_list = base_object_list.filter(experiment__pgmName__exact=qDict.get('pgmName', None))
        if qDict.get('results__eas__reference', None) is not None:
            base_object_list = base_object_list.filter(eas__reference__exact=qDict.get('results__eas__reference', None))
        if qDict.get('flows', None) is not None:
            base_object_list = base_object_list.filter(experiment__flows__exact=qDict.get('flows', None))
        if qDict.get('star', None) is not None:
            base_object_list = base_object_list.filter(experiment__star__exact=bool(qDict.get('star', None)))

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
        CSVstr = _makeCSVstr(base_object_list)

    except Exception as err:
        logger.error("During result CSV generation: %s" % err)
        raise
    ret = http.HttpResponse(CSVstr, mimetype='text/csv')
    now = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    ret['Content-Disposition'] = 'attachment; filename=metrics_%s.csv' % now
    return ret

def get_project_CSV(request, results_pks, project_pk):
    projectName = Project.objects.get(id=project_pk).name
    
    base_object_list = Results.objects.select_related('experiment').prefetch_related('libmetrics_set', 'tfmetrics_set', 'analysismetrics_set', 'pluginresult_set__plugin')
    base_object_list = base_object_list.filter(projects = project_pk)
    base_object_list = base_object_list.filter(id__in=results_pks.split(',')).order_by('-timeStamp')
    CSVstr = _makeCSVstr(base_object_list)
    ret = http.HttpResponse(CSVstr, mimetype='text/csv')
    ret['Content-Disposition'] = 'attachment; filename=%s_metrics_%s.csv' % (projectName, str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    return ret

def projects(request):
    ctx = template.RequestContext(request)
    return render_to_response("rundb/data/projects.html", context_instance=ctx)


def project_view(request, pk=None):
    pr = ProjectResource()
    base_bundle = pr.build_bundle(request=request)
    project = pr.obj_get(bundle=base_bundle, pk=pk)

    pr_bundle = pr.build_bundle(obj=project, request=request)

    return render_to_response("rundb/data/modal_project_details.html", {
        # Other things here.
        "project_json": pr.serialize(None, pr.full_dehydrate(pr_bundle), 'application/json'), "project": project, "method": "GET", "readonly": True, 'action': reverse('api_dispatch_detail', kwargs={'resource_name': 'project', 'api_name': 'v1', 'pk': int(pk)}, args={'format': 'json'})
    })


def project_add(request, pk=None):
    otherList = [p.name for p in Project.objects.all()]
    ctx = template.RequestContext(request, {
        'id': pk, 'otherList': json.dumps(otherList), 'method': 'POST', 'methodDescription': 'Add', 'readonly': False, 'action': reverse('api_dispatch_list', kwargs={'resource_name': 'project', 'api_name': 'v1'})
    })
    return render_to_response("rundb/data/modal_project_details.html", context_instance=ctx)


def project_delete(request, pk=None):
    pr = ProjectResource()
    base_bundle = pr.build_bundle(request=request)
    project = pr.obj_get(bundle=base_bundle, pk=pk)
    _type = 'project'
    ctx = template.RequestContext(request, {
        "id": pk, "name": project.name, "method": "DELETE", 'methodDescription': 'Delete', "readonly": False, 'type': _type, 'action': reverse('api_dispatch_detail', kwargs={'resource_name': _type, 'api_name': 'v1', 'pk': int(pk)})
    })
    return render_to_response("rundb/data/modal_confirm_delete.html", context_instance=ctx)


def project_edit(request, pk=None):
    pr = ProjectResource()
    base_bundle = pr.build_bundle(request=request)
    project = pr.obj_get(bundle=base_bundle, pk=pk)
    pr_bundle = pr.build_bundle(obj=project, request=request)

    otherList = [p.name for p in Project.objects.all()]
    return render_to_response("rundb/data/modal_project_details.html", {
        # Other things here.
        "project_json": pr.serialize(None, pr.full_dehydrate(pr_bundle), 'application/json'), "project": project, "id": pk, 'otherList': json.dumps(otherList), "method": "PATCH", 'methodDescription': 'Edit', "readonly": True, 'action': reverse('api_dispatch_detail', kwargs={'resource_name': 'project', 'api_name': 'v1', 'pk': int(pk)})
    })


def project_log(request, pk=None):
    if request.method == 'GET':
        selected = get_object_or_404(Project, pk=pk)
        ct = ContentType.objects.get_for_model(selected)
        title = "Project History for %s (%s):" % (selected.name, pk)
        ctx = template.RequestContext(request, {"title": title, "pk": pk, "cttype": ct.id})
        return render_to_response("rundb/common/modal_event_log.html", context_instance=ctx)
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
    thumbs_exist = Results.objects.filter(metaData__contains='thumb').exists()
    ctx = template.RequestContext(request, {"project": selected, 'filter_thumbnails': thumbs_exist})
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
            "results_pks": results_pks,
            "action": urlresolvers.reverse('results_to_project', args=[results_pks, ]),
            "method": 'POST'})
        return render_to_response("rundb/data/modal_projects_select.html", context_instance=ctx)
    if request.method == 'POST':
        json_data = json.loads(request.raw_post_data)
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
        result = Results.objects.get(pk=int(result_pk))
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
    ver_map = {'analysis': 'an', 'alignment': 'al', 'dbreports': 'db', 'tmap': 'tm'}
    barcoded = False
    for r in selected_results:
        version = {}
        for name, shortname in ver_map.iteritems():
            version[name] = next((v.split(':')[1].strip() for v in r.analysisVersion.split(',') if v.split(':')[0].strip() == shortname), '')
            setattr(r, name + "_version", version[name])
        # starting with TS3.6 we don't have separate alignment or tmap packages
        if not version['tmap']: r.tmap_version = version['analysis']
        if not version['alignment']: r.alignment_version = version['analysis']
        
        r.barcodeId = r.eas.barcodeKitName
        if r.barcodeId:
            r.barcodedSamples = json.dumps(r.eas.barcodedSamples)
            barcoded = True
        else:
            r.sample = r.experiment.get_sample()

    if len(set([getattr(r, 'tmap_version') for r in selected_results])) > 1:
        warnings.append("Selected results have different TMAP versions.")
    if len(set([getattr(r, 'alignment_version') for r in selected_results])) > 1:
        warnings.append("Selected results have different Alignment versions.")
    if len(set([r.experiment.flowsInOrder for r in selected_results if r.resultsType != 'CombinedAlignments'])) > 1:
        warnings.append("Selected results have different FlowOrder Sequences.")
    if len(set([r.barcodeId for r in selected_results if r.resultsType != 'CombinedAlignments'])) > 1:
        warnings.append("Selected results have different Barcode Sets.")
        barcoded = False
    
    if barcoded:
        if len(set([r.barcodedSamples for r in selected_results if r.resultsType != 'CombinedAlignments'])) > 1:
            warnings.append("Selected results have different Samples.")
    else:
        if len(set([r.sample for r in selected_results if r.resultsType != 'CombinedAlignments'])) > 1:
            warnings.append("Selected results have different Samples.")
        
    return warnings


def results_to_combine(request, results_pks, project_pk):
    if request.method == 'GET':
        selected_results = Results.objects.filter(id__in=results_pks.split(',')).order_by('-timeStamp')
        warnings = validate_results_to_combine(selected_results)

        ctx = template.RequestContext(request, {
            "results_pks": results_pks, "project_pk": project_pk, "selected_results": selected_results,
            "warnings": warnings,
            "action": urlresolvers.reverse('results_to_combine', args=(results_pks, project_pk)),
            "method": 'POST'})
        return render_to_response("rundb/data/modal_combine_results.html", context_instance=ctx)
    if request.method == 'POST':
        try:
            json_data = json.loads(request.raw_post_data)
            result = _combine_results_sendto_project(project_pk, json_data, request.user.username)
            ctx = _report_started(request, result.pk)
            return render_to_response("rundb/reports/analysis_started.html", context_instance=ctx)
        except Exception as e:
            return HttpResponseServerError("%s" % e)


def _combine_results_sendto_project(project_pk, json_data, username=''):
    project = Project.objects.get(id=project_pk)
    projectName = project.name

    name = json_data['name']
    mark_duplicates = json_data['mark_duplicates']
    ids_to_merge = json_data['selected_pks']
    parents = Results.objects.filter(id__in=ids_to_merge).order_by('-timeStamp')

    # test if reports can be combined and get common field values
    for parent in parents:
        if parent.dmfilestat_set.get(dmfileset__type=dmactions_types.OUT).action_state == 'DD':
            raise Exception("Output Files for %s are Deleted." % parent.resultsName)
        
        if parent.pk == parents[0].pk:
            reference = parent.reference
            floworder = parent.experiment.flowsInOrder
            barcodeId = parent.eas.barcodeKitName
            barcodeSamples = parent.eas.barcodedSamples
        else:
            if not reference == parent.reference:
                raise Exception("Selected results do not have the same Alignment Reference.")
            if not floworder == parent.experiment.flowsInOrder:
                floworder = ''
            if not barcodeId == parent.eas.barcodeKitName:
                barcodeId = ''
            if not barcodeSamples == parent.eas.barcodedSamples:
                barcodeSamples = {}

    # create new entry in DB for combined Result
    delim = ':'
    filePrefix = "CombineAlignments"  # this would normally be Experiment name that's prefixed to all filenames
    result, exp = create_combined_result('CA_%s_%s' % (name, projectName))
    result.resultsType = 'CombinedAlignments'
    result.parentIDs = delim + delim.join(ids_to_merge) + delim
    result.reference = reference
    result.sffLink = path.join(result.reportLink, "%s_%s.sff" % (filePrefix, result.resultsName))
    
    result.projects.add(project)

    # add ExperimentAnalysisSettings
    eas_kwargs = {
            'date' : datetime.now(),
            'experiment' : exp,
            'isEditable' : False,
            'isOneTimeOverride' : True,
            'status' : 'run',
            'reference': reference,
            'barcodeKitName': barcodeId,
            'barcodedSamples': barcodeSamples,
            'targetRegionBedFile': '',
            'hotSpotRegionBedFile': '',
            'isDuplicateReads': mark_duplicates
    }
    eas = ExperimentAnalysisSettings(**eas_kwargs)
    eas.save()    
    result.eas = eas
    
    result.save()

    # gather parameters to pass to merging script
    links = []
    bams = []
    names = []
    plan = {}
    bamFile = 'rawlib.bam'
    for parent in parents:
        links.append(parent.reportLink)
        names.append(parent.resultsName)

        # BAM files location
        dmfilestat = parent.dmfilestat_set.get(dmfileset__type=dmactions_types.OUT)
        reportDir = dmfilestat.archivepath if dmfilestat.action_state == 'AD' else parent.get_report_dir()
        bams.append(path.join(reportDir, bamFile))

        #need Plan info for VariantCaller etc. plugins: but which plan to use??
        try:
            planObj = [parent.experiment.plan]
            plan_json = serializers.serialize("json", planObj)
            plan_json = json.loads(plan_json)
            plan = plan_json[0]["fields"]
        except:
            pass

    try:
        genome = ReferenceGenome.objects.all().filter(short_name=reference, index_version=settings.TMAP_VERSION, enabled=True)[0]
        if path.exists(genome.info_text()):
            genomeinfo = genome.info_text()
        else:
            genomeinfo = ""
    except:
        genomeinfo = ""

    eas_json = serializers.serialize("json", [eas])
    eas_json = json.loads(eas_json)
    eas_json = eas_json[0]["fields"]
    
    params = {
        'resultsName': result.resultsName,
        'parentIDs': ids_to_merge,
        'parentNames': names,
        'parentLinks': links,
        'parentBAMs': bams,
        'libraryName': result.reference,
        'tmap_version': settings.TMAP_VERSION,
        'mark_duplicates': mark_duplicates,
        'plan': plan,
        'run_name': filePrefix,
        'genomeinfo': genomeinfo,
        'flowOrder': floworder,
        'project': projectName,
        'barcodeId': barcodeId,
        'barcodeSamples': json.dumps(barcodeSamples),
        'experimentAnalysisSettings': eas_json,
        'warnings': validate_results_to_combine(parents)
    }

    from distutils.sysconfig import get_python_lib
    scriptpath = path.join(get_python_lib(), 'ion', 'reports', 'combineReports.py')

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
        conn.startanalysis(result.resultsName, script, params, files, webRootPath, result.pk, '', {}, 'combineAlignments')
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
    # check resultsName for invalid chars to be safe
    validChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    resultsName = ''.join([c for c in resultsName if c in validChars])

    otherResults = Results.objects.filter(resultsName__contains=resultsName).order_by('pk')
    if otherResults:
        lastName = otherResults[len(otherResults) - 1].resultsName
        last = int(lastName.split('_')[-1])
    resultsName = "%s_%03d" % (resultsName, last + 1)

    storages = ReportStorage.objects.all().order_by('id')
    storage = storages[len(storages) - 1]

    result = build_result(exp, resultsName, storage, _location())
    return result, exp


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
            else:  # only id should remain
                logging.debug(field.name, field.get_internal_type())
        kwargs['cycles'] = 1
        kwargs['flows'] = 1
        kwargs['ftpStatus'] = 'Complete'
        ret = Experiment(**kwargs)
        ret.save()
    return ret


@login_required
def experiment_edit(request, pk):
    exp = get_object_or_404(Experiment, pk=pk)    
    eas, eas_created = exp.get_or_create_EAS(editable=True)
    plan = exp.plan
    
    barcodes = {}
    for bc in dnaBarcode.objects.order_by('name', 'index').values('name', 'id_str','sequence'):
        barcodes.setdefault(bc['name'],[]).append(bc)
    
    # get list of plugins to run
    plugins = Plugin.objects.filter(selected=True,active=True).exclude(path='')
    selected_names = [pl['name'] for pl in eas.selectedPlugins.values()]
    plugins_list = list(plugins.filter(name__in=selected_names))
    
    if request.method == 'GET':
        exp_form = forms.ExperimentSettingsForm(instance=exp)
        eas_form = forms.AnalysisSettingsForm(instance=eas)
        
        # Application, i.e. runType
        if plan:
            exp_form.fields['runtype'].initial = plan.runType
            exp_form.fields['sampleTubeLabel'].initial = plan.sampleTubeLabel
            
        # Library Kit name - can get directly or from kit barcode    
        libraryKitName = ''
        if eas.libraryKitName:
             libraryKitName = eas.libraryKitName
        elif eas.libraryKitBarcode:
             libkitset = KitInfo.objects.filter(kitType='LibraryKit',kitpart__barcode = eas.libraryKitBarcode)
             if len(libkitset) == 1:
                libraryKitName = libkitset[0].name
        exp_form.fields['libraryKitname'].initial = libraryKitName
        
        # Sequencing Kit name - can get directly or from kit barcode
        if not exp.sequencekitname and exp.sequencekitbarcode:
            seqkitset = KitInfo.objects.filter(kitType='SequencingKit',kitpart__barcode = exp.sequencekitbarcode)
            if len(seqkitset) == 1:
                exp_form.fields['sequencekitname'] = seqkitset[0].name
        
        exp_form.fields['libraryKey'].initial = eas.libraryKey
        if len(exp.samples.all()) > 0:
            exp_form.fields['sample'].initial = exp.samples.all()[0].id
        exp_form.fields['barcodedSamples'].initial = eas.barcodedSamples
        
        exp_form.fields['mark_duplicates'].initial = eas.isDuplicateReads
        
        # plugins with optional userInput
        eas_form.fields['plugins'].initial = [plugin.id for plugin in plugins_list]
        pluginsUserInput = {}
        for plugin in plugins_list:
            pluginsUserInput[str(plugin.id)] = eas.selectedPlugins.get(plugin.name, {}).get('userInput','')
        eas_form.fields['pluginsUserInput'].initial = json.dumps(pluginsUserInput)
    
    if request.method == 'POST': 
        exp_form = forms.ExperimentSettingsForm(request.POST, instance=exp)
        eas_form = forms.AnalysisSettingsForm(request.POST, instance=eas)

        if exp_form.is_valid() and eas_form.is_valid():
            # save Plan
            if plan:
                plan.runType = exp_form.cleaned_data['runtype']
                plan.sampleTubeLabel = exp_form.cleaned_data['sampleTubeLabel']

                plan.save()
                
            # save Experiment
            exp_form.save()
            
            # save ExperimentAnalysisSettings
            eas = eas_form.save(commit=False)
            eas.libraryKey = exp_form.cleaned_data['libraryKey']
            eas.libraryKitName = exp_form.cleaned_data['libraryKitname']
            eas.barcodedSamples = exp_form.cleaned_data['barcodedSamples']
            eas.isDuplicateReads = exp_form.cleaned_data['mark_duplicates']
            
            # plugins
            form_plugins_list = list(eas_form.cleaned_data['plugins'])
            pluginsUserInput = json.loads(eas_form.cleaned_data['pluginsUserInput'])
            selectedPlugins = {}
            for plugin in form_plugins_list:
                selectedPlugins[plugin.name] = {
                     "id" : str(plugin.id),
                     "name" : plugin.name,
                     "version" : plugin.version,
                     "features": plugin.pluginsettings.get('features',[]),
                     "userInput": pluginsUserInput.get(str(plugin.id),'')
                }
            eas.selectedPlugins = selectedPlugins
            
            eas.save()
            
            # save single non-barcoded sample or barcoded samples
            if not eas.barcodeKitName:
                sampleId = exp_form.cleaned_data['sample']
                if sampleId:
                    sample = Sample.objects.get(pk=sampleId)
                    exp.samples.clear()
                    exp.samples.add(sample)
            elif eas.barcodedSamples:
                exp.samples.clear()
                for value in eas.barcodedSamples.values():
                    sampleId = value['id']
                    sample = Sample.objects.get(pk=sampleId)
                    exp.samples.add(sample)

        else:
            return HttpResponseServerError('%s %s' % (exp_form.errors, eas_form.errors))
        
    ctxd = {"exp_form": exp_form, "eas_form": eas_form, "pk":pk, "name": exp.expName, "barcodes":json.dumps(barcodes)}
    return render_to_response("rundb/data/modal_experiment_edit.html", context_instance=template.RequestContext(request, ctxd))
    
