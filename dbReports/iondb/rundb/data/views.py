# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django import http, template
from django.core import urlresolvers
from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.core.cache import cache
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render_to_response, get_object_or_404, redirect, render
from django.template import RequestContext
from django.template.loader import render_to_string
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned
from django.core.servers.basehttp import FileWrapper
from django.core.serializers.json import DjangoJSONEncoder
from django.forms.models import model_to_dict

import json
import cStringIO
import csv
import os
import tempfile
import shutil
import subprocess
import glob
import time
import traceback
from django.views.generic import ListView
from iondb.rundb.models import (
    Experiment,
    Results,
    Project,
    Location,
    ReportStorage,
    EventLog, GlobalConfig, ReferenceGenome, dnaBarcode, KitInfo, ContentType, Plugin, ExperimentAnalysisSettings, Sample,
    DMFileSet, DMFileStat, FileServer, IonMeshNode)
from iondb.rundb.api import CompositeExperimentResource, ProjectResource
from iondb.rundb.report.analyze import build_result
from iondb.rundb.report.views import _report_started
from iondb.rundb.report import file_browse
from iondb.rundb import forms
from iondb.anaserve import client
from iondb.rundb.data import dmactions_types
from iondb.rundb.data import tasks as dmtasks
from iondb.rundb.data import dmactions
from iondb.rundb.data.data_management import update_files_in_use
from iondb.rundb.data import exceptions as DMExceptions
from iondb.rundb.data.data_import import find_data_to_import, data_import
from iondb.utils.files import get_disk_attributes_gb, is_mounted
from iondb.rundb.data.dmfilestat_utils import dm_category_stats, get_keepers_diskspace

from django.http import HttpResponse, HttpResponseServerError, HttpResponseNotFound
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

    eas_keys = [('library', 'reference')]

    for key in experiment_params.keys():
        experiment_params[key] = list(Experiment.objects.values_list(key, flat=True).distinct(key).order_by(key))
        experiment_params[key] = [value for value in experiment_params[key] if len(str(value).strip()) > 0]

    experiment_params['sample'] = list(Sample.objects.filter(
        status="run").values_list('name', flat=True).order_by('name'))

    for expkey, key in eas_keys:
        experiment_params[expkey] = list(
            ExperimentAnalysisSettings.objects.values_list(key, flat=True).distinct(key).order_by(key)
        )
        experiment_params[expkey] = [value for value in experiment_params[expkey] if len(str(value).strip()) > 0]
    for key in report_params.keys():
        report_params[key] = list(Results.objects.values_list(key, flat=True).distinct(key).order_by(key))
    combined_params = {
        'flows': sorted(set(experiment_params['flows'] + report_params['processedflows'])),
        'projects': Project.objects.values_list('name', flat=True).distinct('name').order_by('name')
    }
    mesh_params = {
        'nodes': IonMeshNode.objects.all()
    }
    del experiment_params['flows']
    del report_params['processedflows']
    return {'experiment': experiment_params,
            'report': report_params,
            'combined': combined_params,
            'mesh': mesh_params
            }


@login_required
def rundb_redirect(request):
    # Old /rundb/ page redirects to /data/, keeps args
    url = reverse('data') or '/data/'
    args = request.META.get('QUERY_STRING', '')
    if args:
        url = "%s?%s" % (url, args)
    return redirect(url, permanent=True)


def get_serialized_exps(request, pageSize):
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
    return serialized_exps


def data_context(request):
    pageSize = GlobalConfig.get().records_to_display
    context = {
        'search': get_search_parameters(),
        'inital_query': get_serialized_exps(request, pageSize),
        'pageSize': pageSize
    }
    return context


@login_required
def data(request):
    """This is a the main entry point to the Data tab."""
    context = cache.get("data_tab_context")
    if context is None:
        context = data_context(request)
        cache.set("data_tab_context", context, 29)
    return render(request, "rundb/data/data.html", context)


class ExperimentListView(ListView):

    """This is a class based view using the Django ListView generic view.
    It shows Experiment objects and data from their representative report.
    """
    queryset = Experiment.objects.select_related(
        "repResult", "repResult__qualitymetrics", "repResult__eas"
    ).exclude(repResult=None).order_by('-repResult__timeStamp')

    template_name = "rundb/data/fast.html"
    paginate_by = 30

    def get_object(self):
        exp = super(ExperimentListView, self).get_object()
        if exp.has_status:
            exp.in_progress = exp.ftpStatus.isdigit()
            if exp.in_progress:
                exp.progress_percent = 100 * float(exp.ftpStatus) / float(exp.flows)
        return exp

    def get_context_data(self, **kwargs):
        context = super(ExperimentListView, self).get_context_data(**kwargs)
        context['show_status'] = any(e.has_status for e in self.object_list)
        return context


class ResultsListView(ListView):

    """This ListView shows Results objects and is meant to be quick and light weight
    """
    queryset = Results.objects.select_related(
        "experiment", "qualitymetrics", "eas"
    ).order_by('-timeStamp')

    template_name = "rundb/data/results_list.html"
    paginate_by = 30

    def get_object(self):
        result = super(ResultsListView, self).get_object()
        if result.experiment.has_status():
            result.experiment.in_progress = result.experiment.ftpStatus.isdigit()
            if result.experiment.in_progress:
                result.experiment.progress_percent = 100 * \
                    float(result.experiment.ftpStatus) / float(result.experiment.flows)
        return result

    def get_context_data(self, **kwargs):
        context = super(ResultsListView, self).get_context_data(**kwargs)
        context['show_status'] = any(r.experiment.has_status for r in self.object_list)
        return context


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
    # Use the CompositeExperimentResource to generate a queryset from the request args
    resource_instance = CompositeExperimentResource()
    request_bundle = resource_instance.build_bundle(request=request)
    experiment_queryset = resource_instance.obj_get_list(request_bundle)

    # The CSV is generated from a Results queryset so we need to get a Results queryset from the Experiment queryset
    experiment_ids = experiment_queryset.values_list('id', flat=True)
    results_queryset = Results.objects.filter(experiment_id__in=experiment_ids)

    # Now we can directly return a csv file response
    response = http.HttpResponse(_makeCSVstr(results_queryset), mimetype='text/csv')
    response['Content-Disposition'] = 'attachment; filename=metrics_%s.csv' % str(
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    return response


def get_project_CSV(request, project_pk, result_pks):
    projectName = Project.objects.get(id=project_pk).name
    result_ids = result_pks.split(",")
    base_object_list = Results.objects.select_related('experiment').prefetch_related(
        'libmetrics_set', 'tfmetrics_set', 'analysismetrics_set', 'pluginresult_set__plugin')
    base_object_list = base_object_list.filter(id__in=result_ids).order_by('-timeStamp')
    CSVstr = _makeCSVstr(base_object_list)
    ret = http.HttpResponse(CSVstr, mimetype='text/csv')
    ret['Content-Disposition'] = 'attachment; filename=%s_metrics_%s.csv' % (
        projectName, str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
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


def get_result_metrics(result):
    metrics = [
        result.timeStamp,
        result.experiment.chipType,
        result.qualitymetrics.q0_bases,
        result.qualitymetrics.q0_reads,
        result.qualitymetrics.q0_mean_read_length,
        result.qualitymetrics.q20_bases,
        result.qualitymetrics.q20_reads,
        result.qualitymetrics.q20_mean_read_length,
        result.eas.reference,
        result.libmetrics.total_mapped_target_bases,
        result.libmetrics.total_mapped_reads
    ]
    return metrics


def project_compare_context(pk):
    project = get_object_or_404(Project, pk=pk)
    results = Results.objects.filter(projects=project).select_related()
    context = {
        "project": project,
        "results": results,
        "result_ids": [str(r.id) for r in results]
    }
    return context


def project_compare_make_latex(pk):
    context = project_compare_context(pk)
    latex_template = render_to_string("rundb/data/print_multi_report.tex", context)

    directory = tempfile.mkdtemp(prefix="project_compare_", dir="/tmp")
    latex_path = os.path.join(directory, "comparison_report.tex")
    pdf_path = os.path.join(directory, "comparison_report.pdf")
    with open(latex_path, 'w') as latex_file:
        latex_file.write(latex_template)
    logger.debug("Creating PDF directory %s" % directory)
    cmd = [
        "pdflatex", latex_path,
        "-output-directory", directory,
        "-interaction",
        "nonstopmode"
    ]
    logger.debug(' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory)
    stdout, stderr = proc.communicate()
    if 0 <= proc.returncode <= 1:
        return directory, pdf_path
    else:
        logger.error("PDF stdout: %s" % stdout)
        raise Exception("Project Comparison PDF generation failure")


def project_compare_pdf(request, pk):
    project = get_object_or_404(Project, pk=pk)
    directory, pdf_path = project_compare_make_latex(pk)
    if not os.path.exists(pdf_path):
        return HttpResponse(open(os.path.join(directory, "comparison_report.log")).read(), "text/plain")
    response = HttpResponse(FileWrapper(open(pdf_path)),
                            content_type="application/pdf")
    response['Content-Length'] = os.path.getsize(pdf_path)
    response['Content-Disposition'] = "attachment; filename=project_compare_%s.pdf" % project.name
    return response


def project_compare(request, pk):
    context = project_compare_context(pk)

    return render(request, "rundb/data/project_compare.html", context)


def get_value(obj, path, default=None):
    attributes = path.split('.')
    length = len(attributes)
    value = default
    missing = False
    for i in xrange(length - 1):
        if hasattr(obj, attributes[i]):
            obj = getattr(obj, attributes[i])
        else:
            missing = True
            break
    if not missing and length > 0:
        value = getattr(obj, attributes[-1], default)
    return value


def project_compare_csv(request, pk):
    comparison = project_compare_context(pk)
    out_obj = cStringIO.StringIO()
    writer = csv.writer(out_obj)
    table = [
        ('Result Name', 'resultsName'),
        ('Status', 'status'),
        ('Date', 'timeStamp'),
        ('Chip', 'experiment.chipType'),
        ('Total Bases', 'qualitymetrics.q0_bases'),
        ('Total Reads', 'qualitymetrics.q0_reads'),
        ('Key Signal', 'libmetrics.aveKeyCounts'),
        ('Loading', 'analysismetrics.loading'),
        ('Mean Read Len.', 'qualitymetrics.q0_mean_read_length'),
        ('Median Read Len.', 'qualitymetrics.q0_median_read_length'),
        ('Mode Read Len.', 'qualitymetrics.q0_mode_read_length'),
        ('Q20 Bases', 'qualitymetrics.q20_bases'),
        ('Q20 Reads', 'qualitymetrics.q20_reads'),
        ('Q20 Read Len.', 'qualitymetrics.q20_mean_read_length'),
        ('Reference', 'eas.reference'),
        ('Aligned Bases', 'libmetrics.total_mapped_target_bases'),
        ('Aligned Reads', 'libmetrics.total_mapped_reads')
    ]
    # above we define the header name and column's value path toether in a tuple
    # for visual clarity and to help catch typos when making changes
    # here we separate them again for use in the loop.
    header, columns = zip(*table)
    writer.writerow(header)
    for result in comparison['results']:
        row = [get_value(result, c, '') for c in columns]
        writer.writerow(row)

    filename = "project_compare_{0}.csv".format(comparison['project'].name)
    csv_content = out_obj.getvalue()
    size = len(csv_content)
    out_obj.close()
    response = HttpResponse(csv_content, content_type="text/csv")
    response['Content-Length'] = size
    response['Content-Disposition'] = "attachment; filename=" + filename
    return response


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
        json_data = json.loads(request.body)
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


def validate_results_to_combine(selected_results, override_samples=False):
    # validate selected reports
    warnings = []
    ver_map = {'analysis': 'an', 'alignment': 'al', 'dbreports': 'db', 'tmap': 'tm'}
    common = {}
    for i, r in enumerate(selected_results):
        version = {}
        for name, shortname in ver_map.iteritems():
            version[name] = next((v.split(':')[1].strip()
                                 for v in r.analysisVersion.split(',') if v.split(':')[0].strip() == shortname), '')
            setattr(r, name + "_version", version[name])
        # starting with TS3.6 we don't have separate alignment or tmap packages
        if not version['tmap']: r.tmap_version = version['analysis']
        if not version['alignment']: r.alignment_version = version['analysis']

        if not common:
            if r.resultsType != 'CombinedAlignments' or i == (len(selected_results) - 1):
                common = {
                    'floworder': r.experiment.flowsInOrder,
                    'barcodeId': r.eas.barcodeKitName,
                    'barcodedSamples': r.eas.barcodedSamples if r.eas.barcodeKitName else {},
                    'sample': r.experiment.get_sample() if not r.eas.barcodeKitName else ''
                }

    if len(set([getattr(r, 'tmap_version') for r in selected_results])) > 1:
        warnings.append("Selected results have different TMAP versions.")
    if len(set([getattr(r, 'alignment_version') for r in selected_results])) > 1:
        warnings.append("Selected results have different Alignment versions.")
    if len(set([r.experiment.flowsInOrder for r in selected_results if r.resultsType != 'CombinedAlignments'])) > 1:
        warnings.append("Selected results have different FlowOrder Sequences.")
        common['floworder'] = ''

    barcodeSet = set(
        [r.eas.barcodeKitName for r in selected_results if r.resultsType != 'CombinedAlignments'])
    if len(barcodeSet) > 1:
        warnings.append("Selected results have different Barcode Sets.")
        # allow merging for sub-sets of barcodes, e.g. "IonCode" and "IonCode Barcodes 1-32"
        minstr = min(barcodeSet, key=len)
        common['barcodeId'] = minstr if all(s.startswith(minstr) for s in barcodeSet) else ""

    if not override_samples:
        if common['barcodeId']:
            if len(set([json.dumps(r.eas.barcodedSamples) for r in selected_results if r.resultsType != 'CombinedAlignments'])) > 1:
                warnings.append("Selected results have different Samples.")
                common['barcodedSamples'] = {}
        else:
            if len(set([r.experiment.get_sample() for r in selected_results if r.resultsType != 'CombinedAlignments'])) > 1:
                warnings.append("Selected results have different Samples.")
                common['sample'] = ''

    return warnings, common


def results_to_combine(request, results_pks, project_pk):
    if request.method == 'GET':
        selected_results = Results.objects.filter(id__in=results_pks.split(',')).order_by('-timeStamp')
        warnings, common = validate_results_to_combine(selected_results)
        barcodes = ''
        if common['barcodeId']:
            barcodes = dnaBarcode.objects.filter(name=common['barcodeId']).order_by('name', 'index')

        ctx = template.RequestContext(request, {
            "results_pks": results_pks, "project_pk": project_pk, "selected_results": selected_results,
            "warnings": warnings, "barcodes": barcodes,
            "action": urlresolvers.reverse('results_to_combine', args=(results_pks, project_pk)),
            "method": 'POST'})
        return render_to_response("rundb/data/modal_combine_results.html", context_instance=ctx)
    if request.method == 'POST':
        try:
            json_data = json.loads(request.body)
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
    override_samples = json_data.get('override_samples', False) == 'on'

    parents = Results.objects.filter(id__in=ids_to_merge).order_by('-timeStamp')

    # test if reports can be combined and get common field values
    for parent in parents:
        if parent.dmfilestat_set.get(dmfileset__type=dmactions_types.OUT).action_state == 'DD':
            raise Exception("Output Files for %s are Deleted." % parent.resultsName)

        if parent.pk == parents[0].pk:
            reference = parent.reference
        else:
            if not reference == parent.reference:
                raise Exception("Selected results do not have the same Alignment Reference.")

    warnings, common = validate_results_to_combine(parents, override_samples)
    floworder = common['floworder']
    barcodeId = common['barcodeId']
    if override_samples:
        sample = json_data.get('sample', '')
        barcodedSamples = json_data.get('barcodedSamples', {})
        if barcodedSamples and common['barcodedSamples']:
            # try to update with original barcodeSampleInfo
            for sample_name, value in barcodedSamples.items():
                for barcode in value['barcodes']:
                    barcodeSampleInfo = [v.get('barcodeSampleInfo', {}).get(barcode) for v in common[
                                         'barcodedSamples'].values() if v.get('barcodeSampleInfo', {}).get(barcode)]
                    if barcodeSampleInfo:
                        barcodedSamples[sample_name].setdefault(
                            'barcodeSampleInfo', {})[barcode] = barcodeSampleInfo[0]
        else:
            barcodedSamples = json.dumps(barcodedSamples, cls=DjangoJSONEncoder)
    else:
        sample = common['sample']
        barcodedSamples = common['barcodedSamples']

    # create new entry in DB for combined Result
    delim = ':'
    filePrefix = "CombineAlignments"  # this would normally be Experiment name that's prefixed to all filenames
    result, exp = create_combined_result('CA_%s_%s' % (name, projectName))
    result.resultsType = 'CombinedAlignments'
    result.parentIDs = delim + delim.join(ids_to_merge) + delim
    result.reference = reference
    result.sffLink = os.path.join(result.reportLink, "%s_%s.sff" % (filePrefix, result.resultsName))

    result.projects.add(project)

    # add ExperimentAnalysisSettings
    eas_kwargs = {
        'date': datetime.now(),
        'experiment': exp,
        'isEditable': False,
        'isOneTimeOverride': True,
        'status': 'run',
        'reference': reference,
        'barcodeKitName': barcodeId,
        'barcodedSamples': barcodedSamples,
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
    bamFile = 'rawlib.bam'
    for parent in parents:
        links.append(parent.reportLink)
        names.append(parent.resultsName)

        # BAM files location
        dmfilestat = parent.dmfilestat_set.get(dmfileset__type=dmactions_types.OUT)
        reportDir = dmfilestat.archivepath if dmfilestat.action_state == 'AD' else parent.get_report_dir()
        bams.append(os.path.join(reportDir, bamFile))

    # need Plan section for plugins, use latest report's plan
    latest_w_plan = parents.filter(experiment__plan__isnull=False)
    if not latest_w_plan:
        grandparents = sum([v.split(delim) for v in parents.values_list('parentIDs', flat=True)], [])
        grandparents = [v for v in set(grandparents) if v]
        latest_w_plan = Results.objects.filter(
            id__in=grandparents, experiment__plan__isnull=False).order_by('-timeStamp')

    plan_json = model_to_dict(latest_w_plan[0].experiment.plan) if latest_w_plan else {}

    try:
        genome = ReferenceGenome.objects.all().filter(
            short_name=reference, index_version=settings.TMAP_VERSION, enabled=True)[0]
        if os.path.exists(genome.info_text()):
            genomeinfo = genome.info_text()
        else:
            genomeinfo = ""
    except:
        genomeinfo = ""

    eas_json = model_to_dict(eas)

    barcodedSamples_reference_names = eas.barcoded_samples_reference_names
    # use barcodedSamples' selected reference if NO plan default reference is specified
    reference = result.reference
    if not result.reference and barcodedSamples_reference_names:
        reference = barcodedSamples_reference_names[0]

    params = {
        'resultsName': result.resultsName,
        'parentIDs': ids_to_merge,
        'parentNames': names,
        'parentLinks': links,
        'parentBAMs': bams,
        'referenceName': reference,
        'tmap_version': settings.TMAP_VERSION,
        'mark_duplicates': mark_duplicates,
        'plan': plan_json,
        'run_name': filePrefix,
        'genomeinfo': genomeinfo,
        'flowOrder': floworder,
        'project': projectName,
        'barcodeId': barcodeId,
        "barcodeSamples_referenceNames": barcodedSamples_reference_names,
        'sample': sample,
        'override_samples': override_samples,
        'experimentAnalysisSettings': eas_json,
        'warnings': warnings,
        'runid': result.runid
    }
    params = json.dumps(params, cls=DjangoJSONEncoder)

    from distutils.sysconfig import get_python_lib
    scriptpath = os.path.join(get_python_lib(), 'ion', 'reports', 'combineReports.py')

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

    webRootPath = result.web_root_path(Location.getdefault())

    try:
        host = "127.0.0.1"
        conn = client.connect(host, settings.JOBSERVER_PORT)
        conn.startanalysis(result.resultsName, script, params,
                           files, webRootPath, result.pk, '', {}, 'combineAlignments')
    except:
        result.status = "Failed to contact job server."
        raise Exception(result.status)

    # log project history
    message = 'Combine results %s into name= %s (%s), auto-assign to project name= %s (%s).' % (
        ids_to_merge, result.resultsName, result.pk, projectName, project_pk)
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

    storage = ReportStorage.objects.filter(default=True)[0]

    result = build_result(exp, resultsName, storage, Location.getdefault())
    return result, exp


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
    for bc in dnaBarcode.objects.order_by('name', 'index').values('name', 'id_str', 'sequence'):
        barcodes.setdefault(bc['name'], []).append(bc)

    # get list of plugins to run
    plugins = Plugin.objects.filter(selected=True, active=True).exclude(path='')
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
            libkitset = KitInfo.objects.filter(kitType='LibraryKit', kitpart__barcode=eas.libraryKitBarcode)
            if len(libkitset) == 1:
                libraryKitName = libkitset[0].name
        exp_form.fields['libraryKitname'].initial = libraryKitName

        # Sequencing Kit name - can get directly or from kit barcode
        if not exp.sequencekitname and exp.sequencekitbarcode:
            seqkitset = KitInfo.objects.filter(
                kitType='SequencingKit', kitpart__barcode=exp.sequencekitbarcode)
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
            pluginsUserInput[str(plugin.id)] = eas.selectedPlugins.get(plugin.name, {}).get('userInput', '')
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
                    "id": str(plugin.id),
                    "name": plugin.name,
                    "version": plugin.version,
                    "features": plugin.pluginsettings.get('features', []),
                    "userInput": pluginsUserInput.get(str(plugin.id), '')
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

    ctxd = {"exp_form": exp_form, "eas_form": eas_form, "pk":
            pk, "name": exp.expName, "barcodes": json.dumps(barcodes)}
    return render_to_response("rundb/data/modal_experiment_edit.html", context_instance=template.RequestContext(request, ctxd))


@login_required
def datamanagement(request):
    gc = GlobalConfig.get()

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

    dm_filesets = DMFileSet.objects.filter(version=settings.RELVERSION).order_by('pk')
    for dmfileset in dm_filesets:
        if dmfileset.backup_directory in ['None', None, '']:
            dmfileset.mounted = False
        else:
            dmfileset.mounted = bool(is_mounted(dmfileset.backup_directory)) and os.path.exists(
                dmfileset.backup_directory)

    # Disk Usage section
    fs_stats = {}
    for path in FileServer.objects.all().order_by('pk').values_list('filesPrefix', flat=True):
        try:
            if os.path.exists(path):
                fs_stats[path] = get_disk_attributes_gb(path)
                # get space used by data marked Keep
                keeper_used = get_keepers_diskspace(path)
                keeper_used = float(sum(keeper_used.values())) / 1024  # gbytes
                total_gb = fs_stats[path]['disksize']
                fs_stats[path]['percentkeep'] = 100 * (keeper_used / total_gb) if total_gb > 0 else 0
        except:
            logger.error(traceback.format_exc())

    archive_stats = {}
    backup_dirs = get_dir_choices()[1:]
    for bdir, name in backup_dirs:
        try:
            mounted = is_mounted(bdir)  # This will return mountpoint path
            if mounted and bdir not in archive_stats and bdir not in fs_stats:
                archive_stats[bdir] = get_disk_attributes_gb(bdir)
        except:
            logger.error(traceback.format_exc())

    ctxd = {
        "autoArchive": gc.auto_archive_ack,
        "autoArchiveEnable": gc.auto_archive_enable,
        "dm_filesets": dm_filesets,
        "dm_tables": dm_tables,
        "archive_stats": archive_stats,
        "fs_stats": fs_stats,
        "dm_stats": dm_category_stats()
    }

    ctx = template.RequestContext(request, ctxd)
    return render_to_response("rundb/data/data_management.html", context_instance=ctx)


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
                    'category': dmfilestat.dmfileset.type,
                    'description': dmfilestat.dmfileset.description,
                    'action_state': dmfilestat.get_action_state_display(),
                    'keep': dmfilestat.getpreserved(),
                    'diskspace': dmfilestat.diskspace,
                    'in_process': dmfilestat.in_process()
                }
            else:
                # multiple results
                if info['action_state'] != dmfilestat.get_action_state_display():
                    info['action_state'] = '*'
                if info['keep'] != dmfilestat.getpreserved():
                    info['keep'] = '*'
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
    return render_to_response("rundb/data/modal_dm_actions.html", context_instance=ctx)


def dm_action_selected(request, results_pks, action):
    '''
    file categories to process: data['categories']
    user log entry comment: data['comment']
    results_pks could contain more than 1 result
    '''
    logger = logging.getLogger('data_management')

    data = json.loads(request.body)
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
            logger.debug("Matching dmfilestats contain %s reportpk" % resultPK)
            dmfilestat_dict[resultPK] = DMFileStat.objects.select_related() \
                .filter(dmfileset__type__in=data['categories'], result__id=int(resultPK))

            for dmfilestat in dmfilestat_dict[resultPK]:
                # validate export/archive destination folders
                if action in ['export', 'archive']:
                    dmactions.destination_validation(dmfilestat, backup_directory, manual_action=True)

                # validate files not in use
                try:
                    dmactions.action_validation(dmfilestat, action, data['confirmed'])
                except DMExceptions.FilesInUse as e:
                    # warn if exporting files currently in use, allow to proceed if confirmed
                    if action == 'export':
                        if not data['confirmed']:
                            return HttpResponse(json.dumps({'warning': str(e) + '<br>Exporting now may produce incomplete data set.'}), mimetype="application/json")
                    else:
                        raise e
                except DMExceptions.BaseInputLinked as e:
                    # warn if deleting basecaller files used in any other re-analysis started from BaseCalling
                    if not data['confirmed']:
                        return HttpResponse(json.dumps({'warning': str(e)}), mimetype="application/json")

                # warn if archiving data marked Keep
                if action == 'archive' and dmfilestat.getpreserved():
                    if not data['confirmed']:
                        return HttpResponse(json.dumps({'warning': '%s currently marked Keep.' % dmfilestat.dmfileset.type}), mimetype="application/json")
                    else:
                        dmfilestat.setpreserved(False)

                # if further processing an archived dataset, error if archive drive is not mounted
                if dmfilestat.isarchived() and not os.path.exists(dmfilestat.archivepath):
                    return HttpResponseServerError("%s archive location %s is not available." % (dmfilestat.dmfileset.type, dmfilestat.archivepath))

        async_task_result = dmtasks.action_group.delay(request.user.username, data[
                                                       'categories'], action, dmfilestat_dict, data['comment'], backup_directory, data['confirmed'])

        if async_task_result:
            logger.debug(async_task_result)

    except DMExceptions.SrcDirDoesNotExist as e:
        dmfilestat.setactionstate('DD')
        msg = "Source directory %s no longer exists. Setting action_state to Deleted" % e.message
        logger.info(msg)
        EventLog.objects.add_entry(dmfilestat.result, msg, username=request.user.username)
    except Exception as e:
        logger.error("dm_action_selected: error: %s" % str(e))
        return HttpResponseServerError("%s" % str(e))

    test = {'pks': results_pks, 'action': action, 'data': data}
    return HttpResponse(json.dumps(test), mimetype="application/json")


@login_required
@staff_member_required
def dm_configuration(request):
    def isdiff(value1, value2):
        return str(value1) != str(value2)

    config = GlobalConfig.get()
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
        return render_to_response("rundb/data/dm_configuration.html", context_instance=ctx)

    elif request.method == 'POST':
        dm_filesets = DMFileSet.objects.all()
        log = 'SAVED Data Management Configuration<br>'
        data = json.loads(request.body)
        changed = False
        html = lambda s: '<span style="color:#3A87AD;">%s</span>' % s
        try:
            for key, value in data.items():
                if key == 'filesets':
                    for category, params in value.items():
                        # log += '<b>%s:</b> %s<br>' % (category,
                        # json.dumps(params).translate(None, "{}\"\'") )
                        dmfileset = dm_filesets.filter(type=category)
                        current_params = dmfileset.values(*params.keys())[0]
                        changed_params = [
                            key for key, value in params.items() if isdiff(value, current_params.get(key))]
                        if len(changed_params) > 0:
                            dmfileset.update(**params)
                            changed = True
                        log += '<b>%s:</b> ' % category
                        for key, value in params.items():
                            log_txt = ' %s: %s,' % (key, value)
                            if key in changed_params:
                                log_txt = html(log_txt)
                            log += log_txt
                        log = log[:-1] + '<br>'
                elif key == 'email':
                    log_txt = '<b>Email:</b> %s' % value
                    if isdiff(value, dm_contact.email):
                        changed = True
                        dm_contact.email = value
                        dm_contact.save()
                        log_txt = html(log_txt)
                    log += log_txt + '<br>'
                elif key == 'auto_archive_ack':
                    log_txt = '<b>Auto Acknowledge Delete:</b> %s' % value
                    if isdiff(value, config.auto_archive_ack):
                        changed = True
                        config.auto_archive_ack = True if value == 'True' else False
                        config.save()
                        log_txt = html(log_txt)
                    log += log_txt + '<br>'
            if changed:
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
    ret = DMFileStat.objects.filter(
        result__pk__in=results_pks, dmfileset__type=dmactions_types.SIG).update(action_state=runState)

    for result in exp.results_set.all():
        msg = '%s deletion ' % dmactions_types.SIG
        msg += 'is Acknowledged' if runState == 'A' else ' Acknowledgement is removed'
        EventLog.objects.add_entry(result, msg, username=request.user.username)

    return HttpResponse(json.dumps({"runState": runState, "count": ret, "runPK": runPK}), mimetype="application/json")


@login_required
def preserve_data(request):
    # Sets flag to preserve data for a single DMFileStat object
    if request.method == 'POST':

        reportPK = request.POST.get('reportpk', False)
        expPK = request.POST.get('exppk', False)
        keep = True if request.POST.get('keep') == 'true' else False
        dmtype = request.POST.get('type', '')

        if dmtype == 'sig':
            typeStr = dmactions_types.SIG
        elif dmtype == 'base':
            typeStr = dmactions_types.BASE
        elif dmtype == 'out':
            typeStr = dmactions_types.OUT
        elif dmtype == 'intr':
            typeStr = dmactions_types.INTR
        elif dmtype == 'reanalysis':
            typeStr = dmactions_types.SIG
        else:
            return HttpResponseServerError("error, unknown DMFileStat type")

        try:
            if reportPK:
                if dmtype == 'sig':
                    expPKs = Results.objects.filter(
                        pk__in=reportPK.split(',')).values_list('experiment', flat=True)
                    results = Results.objects.filter(experiment__in=expPKs)
                else:
                    results = Results.objects.filter(pk__in=reportPK.split(','))
            elif expPK:
                results = Experiment.objects.get(pk=expPK).results_set.all()
            else:
                return HttpResponseServerError("error, no object pk specified")

            msg = '%s marked exempt from auto-action' if keep else '%s no longer exempt from auto-action'
            msg = msg % typeStr
            for result in results:
                filestat = result.get_filestat(typeStr)
                filestat.setpreserved(keep)

                if dmtype == 'reanalysis':
                    # Keep BASE category data for Proton fullchip for re-analysis from on-instrument files
                    if result.experiment.log.get('oninstranalysis', '') == 'yes' and not result.isThumbnail:
                        result.get_filestat(dmactions_types.BASE).setpreserved(keep)

                EventLog.objects.add_entry(result, msg, username=request.user.username)
        except Exception as err:
            return HttpResponseServerError("error, %s" % err)

        return HttpResponse(json.dumps({"reportPK": reportPK, "type": typeStr, "keep": filestat.getpreserved()}), mimetype="application/json")


def get_dir_choices():
    from iondb.utils import devices
    basicChoice = [(None, 'None')] + devices.to_media(devices.disk_report())

    # add selected directories to choices
    for choice in set(DMFileSet.objects.exclude(backup_directory__in=['', 'None']).values_list('backup_directory', flat=True)):
        if choice and not (choice, choice) in basicChoice:
            choice_str = choice if bool(is_mounted(choice)) else '%s (not mounted)' % choice
            basicChoice.append((choice, choice_str))

    return tuple(basicChoice)


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
    ctx = RequestContext(request, {'usernames': usernames})
    return render_to_response("rundb/data/dm_history.html", context_instance=ctx)


def dm_list_files(request, resultPK, action):
    """Returns the list of files that are selected for the given file categories for the given Report"""
    data = json.loads(request.body)
    dmfilestat = DMFileStat.objects.select_related() \
        .filter(dmfileset__type__in=data['categories'], result__id=int(resultPK))
    dmfilestat = dmfilestat[0]

    # Hack - generate serialized json file for the DataXfer plugin
    dmactions.write_serialized_json(dmfilestat.result, dmfilestat.result.get_report_dir())

    to_process, to_keep = dmactions.get_file_list(dmfilestat)
    payload = {
        'files_to_transfer': to_process,
        'start_dirs': [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir],
    }
    return HttpResponse(json.dumps(payload), mimetype="application/json")


@login_required
def browse_backup_dirs(request, path):
    from iondb.utils import devices

    def bread_crumbs(path):
        crumbs = []
        while path != '/':
            if os.path.ismount(path):
                crumbs.insert(0, (path, path))
                break
            head, tail = os.path.split(path)
            crumbs.insert(0, (path, tail))
            path = head
        return crumbs

    backup_dirs = get_dir_choices()[1:]
    breadcrumbs = []
    dir_info = []
    file_info = []
    path_allowed = True
    if path:
        if not os.path.isabs(path): path = os.path.join('/', path)
        # only allow directories inside mount points
        path_allowed = any([path.startswith(d) for d, n in backup_dirs])
        if not path_allowed:
            return HttpResponseServerError("Directory not allowed: %s" % path)

    exclude_archived = request.GET.get('exclude_archived', 'false')

    if path and path_allowed:
        breadcrumbs = bread_crumbs(path)
        try:
            dirs, files = file_browse.list_directory(path)
        except Exception as e:
            return HttpResponseServerError(str(e))
        dirs.sort()
        files.sort()
        for name, full_dir_path, stat in dirs:
            dir_path = full_dir_path
            if exclude_archived == 'true':
                if name == 'archivedReports' or name == 'exportedReports' or glob.glob(os.path.join(full_dir_path, 'serialized_*.json')):
                    dir_path = ''

            date = datetime.fromtimestamp(stat.st_mtime)
            try:
                size = file_browse.dir_size(full_dir_path)
            except:
                size = ''
            dir_info.append((name, dir_path, date, size))
        for name, full_file_path, stat in files:
            file_path = os.path.join(path, name)
            date = datetime.fromtimestamp(stat.st_mtime)
            try:
                size = file_browse.format_units(stat.st_size)
            except:
                size = ''
            file_info.append((name, file_path, date, size))

    ctxd = {"backup_dirs": backup_dirs, "selected_path": path, "breadcrumbs": breadcrumbs,
            "dirs": dir_info, "files": file_info, "exclude_archived": exclude_archived}
    return render_to_response("rundb/data/modal_browse_dirs.html", ctxd)


def import_data(request):
    if request.method == "GET":
        backup_dirs = get_dir_choices()
        return render_to_response("rundb/data/modal_import_data.html", context_instance=RequestContext(request, {"backup_dirs": backup_dirs}))
    elif request.method == "POST":
        postData = json.loads(request.body)
        for result in postData:
            name = result.pop('name')
            copy_data = bool(result.pop('copy_data', False))
            copy_report = bool(result.pop('copy_report', False))
            async_result = data_import.delay(name, result, request.user.username, copy_data, copy_report)
        return HttpResponse()


def import_data_find(request, path):
    # search directory tree for importable data
    if not os.path.isabs(path): path = os.path.join('/', path.strip())
    if path and os.path.exists(path):
        found_results = find_data_to_import(path)
        if len(found_results) == 0:
            return HttpResponseNotFound('Did not find any data exported or archived with TS version 4.0 or later in %s.' % path)
        else:
            results_list = []
            for result in found_results:
                results_list.append({
                    'name': result['name'],
                    'report': result['categories'].get(dmactions_types.OUT),
                    'basecall': result['categories'].get(dmactions_types.BASE),
                    'sigproc': result['categories'].get(dmactions_types.SIG)
                })
            ctxd = {
                "dm_types": [dmactions_types.OUT, dmactions_types.BASE, dmactions_types.SIG],
                'results_list': results_list
            }
            return render_to_response("rundb/data/modal_import_data.html", ctxd)
    else:
        return HttpResponseNotFound('Cannot access path: %s.' % path)


def import_data_log(request, path):
    # display log file
    if not os.path.isabs(path): path = os.path.join('/', path)
    contents = ''
    with open(path, 'rb') as f:
        contents = f.read()

    response = '<div><pre>%s</pre></div>' % contents
    # add some js to overwrite modal header
    response += '<script type="text/javascript">$(function() { $("#modal_report_log .modal-header").find("h2").text("Data Import Log");});</script>'
    return HttpResponse(response, mimetype='text/plain')


def dmactions_jobs(request):
    active_dmfilestats = DMFileStat.objects.filter(
        action_state__in=['AG', 'DG', 'EG', 'SA', 'SE', 'SD', 'IG']).select_related('result', 'dmfileset')
    active_logs = EventLog.objects.filter(
        object_pk__in=active_dmfilestats.values_list('result__pk', flat=True))
    dmactions_jobs = []
    for dmfilestat in active_dmfilestats:
        d = {
            'pk': dmfilestat.pk,
            'state': dmfilestat.get_action_state_display(),
            'diskspace': "%.1f" % (dmfilestat.diskspace or 0),
            'result_pk': dmfilestat.result.pk,
            'resultsName': dmfilestat.result.resultsName,
            'category': dmfilestat.dmfileset.type,
            'destination': dmfilestat.archivepath if dmfilestat.archivepath else ''
        }
        # parse date and comment from logs
        for log in active_logs.filter(object_pk=d['result_pk']).order_by('-created'):
            if d['category'] in log.text and d['state'] in log.text:
                d['started_on'] = log.created
                d['username'] = log.username
                try:
                    d['comment'] = log.text.split('User Comment:')[1].strip()
                except:
                    d['comment'] = log.text

        dmactions_jobs.append(d)

    ctx = json.dumps({'objects': dmactions_jobs, 'total': len(dmactions_jobs)}, cls=DjangoJSONEncoder)
    return HttpResponse(ctx, content_type="application/json")


def cancel_pending_dmaction(request, pk):
    dmfilestat = DMFileStat.objects.get(pk=pk)
    msg = 'Canceled %s for %s' % (dmfilestat.get_action_state_display(), dmfilestat.dmfileset.type)
    if dmfilestat.action_state in ['SA', 'SE']:
        dmfilestat.setactionstate('L')
        EventLog.objects.add_entry(dmfilestat.result, msg, username=request.user.username)
    return HttpResponse()
