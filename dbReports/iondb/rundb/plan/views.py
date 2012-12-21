# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django import http
from datetime import datetime
from django.contrib.auth.decorators import login_required
from django.template import RequestContext
from django.shortcuts import render_to_response, get_object_or_404, \
    get_list_or_404
from django.conf import settings
from django.db import transaction
from django.http import HttpResponse

from iondb.rundb.models import PlannedExperiment, RunType, ApplProduct, \
    ReferenceGenome, Content, KitInfo, VariantFrequencies, dnaBarcode, \
    LibraryKey, ThreePrimeadapter, Chip, QCType, Project, Plugin, \
    PlannedExperimentQC

from traceback import format_exc
import json
import simplejson
import uuid

import logging
from django.core import serializers
from iondb.rundb.api import PlannedExperimentResource, RunTypeResource, \
    dnaBarcodeResource, ChipResource
import re
from django.core.urlresolvers import reverse

from iondb.utils import toBoolean
from iondb.rundb.plan.views_helper import get_projects, dict_bed_hotspot
from iondb.rundb.plan.plan_csv_writer import get_template_data_for_batch_planning
from iondb.rundb.plan.plan_csv_validator import validate_csv_plan

import os
import string
import traceback
import tempfile
import csv

logger = logging.getLogger(__name__)


@login_required
def plans(request):
    """
    plan template home page
    """
    ctxd = {}

    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/plan/plans.html", context_instance=ctx, mimetype="text/html")


@login_required
def planned(request):
    ctx = RequestContext(request)
    return render_to_response("rundb/plan/planned.html", context_instance=ctx)


@login_required
def delete_plan_template(request, pks=None):
    #TODO: See about pulling this out into a common methods
    pks = pks.split(',')
    _type = 'plannedexperiment'
    planTemplates = get_list_or_404(PlannedExperiment, pk__in=pks)
    _typeDescription = "Template" if planTemplates[0].isReusable is True else "Planned Run"
    actions = []
    for pk in pks:
        actions.append(reverse('api_dispatch_detail', kwargs={'resource_name': _type, 'api_name': 'v1', 'pk': int(pk)}))
    names = ', '.join([x.planName for x in planTemplates])
    ctx = RequestContext(request, {
        "id": pks[0], "name": names, "ids": json.dumps(pks), "names": names, "method": "DELETE", 'methodDescription': 'Delete', "readonly": False, 'type': _typeDescription, 'action': actions[0], 'actions': json.dumps(actions)
    })
    return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)


@login_required
def add_plan_template(request, code):

    """prepare data to guide user in plan template creation"""
    return _add_plan(request, code, "New")


@login_required
def add_plan_no_template(request, code):
    """
    Create a planned run *without* a template via wizard
    """
    return _add_plan(request, code, "Plan Run New")


def _add_plan(request, code, intent):
    """prepare data to guide user in plan template creation"""

    #logger.debug("TIMING START - _add_plan for either plan or template...");

    isForTemplate = True
    if (intent == "Plan Run New"):
        isForTemplate = False
    data = _get_allApplProduct_data(isForTemplate)

    logger.debug("views.add_planTemplate()... code=%s" % str(code))

    ctxd = {
        "intent": intent,
        "planTemplateData": data,
        "selectedPlanTemplate": None
    }

    if code == '1':
        #logger.debug("AMPLISEQ add_plan.. ")
        ctxd["selectedApplProductData"] = data["AMPS"]
    elif code == '2':
        #logger.debug("TARGETSEQ add_plan.. ")
        ctxd["selectedApplProductData"] = data["TARS"]

    elif code == '3':
        #logger.debug("WHOLE GENOME add_plan.. ")
        ctxd["selectedApplProductData"] = data["WGNM"]

    elif code == '4':
        #logger.debug("RNA add_plan.. ")
        ctxd["selectedApplProductData"] = data["RNA"]

    elif code == '5':
        #logger.debug("AMPLISEQ RNA add_plan.. ")
        ctxd["selectedApplProductData"] = data["AMPS_RNA"]

    if "selectedApplProductData" not in ctxd:
        #logger.debug("GENERIC add_plan.. ")
        ctxd["selectedApplProductData"] = data["GENS"]

    context = RequestContext(request, ctxd)

    #logger.debug("TIMING END - _add_plan for either plan or template...");

    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context)


def _get_runtype_json(request, runType):
    rtr = RunTypeResource()
    rt = rtr.obj_get(runType=runType)
    rtr_bundle = rtr.build_bundle(obj=rt, request=request)
    rt_json = rtr.serialize(None, rtr.full_dehydrate(rtr_bundle), 'application/json')
    return rt_json


def _get_dnabarcode_json(request, barcodeId):
    dnar = dnaBarcodeResource()
    dna = dnar.obj_get_list(name=barcodeId, request=request).order_by('index')
    dna_bundles = [dnar.build_bundle(obj=x, request=request) for x in dna]
    dna_json = dnar.serialize(None, [dnar.full_dehydrate(bundle) for bundle in dna_bundles], 'application/json')
    return dna_json


def _get_chiptype_json(request, chipType):
    chipResource = ChipResource()
    chipResource_serialize_json = None
    if chipType:
        chip = chipResource.obj_get(name=chipType)
        chipResource_bundle = chipResource.build_bundle(obj=chip, request=request)
        chipResource_serialize_json = chipResource.serialize(None, chipResource.full_dehydrate(chipResource_bundle), 'application/json')
    else:
        chipResource_bundle = None
        chipResource_serialize_json = json.dumps(None)
    return chipResource_serialize_json


def _review_plan(request, pk):
    per = PlannedExperimentResource()
    pe = per.obj_get(pk=pk)
    per_bundle = per.build_bundle(obj=pe, request=request)
    pe_json = per.serialize(None, per.full_dehydrate(per_bundle), 'application/json')

    rt_json = _get_runtype_json(request, pe.runType)
    dna_json = _get_dnabarcode_json(request, pe.barcodeId)
    chipType_json = _get_chiptype_json(request, pe.chipType)

    return render_to_response("rundb/plan/modal_review_plan.html", {
                              "plan": pe,
                              "selectedPlanTemplate": pe_json,
                              "selectedRunType": rt_json,
                              "selectedBarcodes": dna_json,
                              "view": 'template' if 'template' in request.path else 'Planned Run',
                              "selectedChip": chipType_json
                              })


@login_required
def review_plan_template(request, _id):
    """
    Review plan template contents
    """
    return _review_plan(request, _id)


@login_required
def review_plan_run(request, _id):
    """
    Review plan contents
    """
    return _review_plan(request, _id)


@login_required
def edit_plan_template(request, template_id):
    """
    Edit plan template in template wizard
    """

    context = _plan_template_helper(request, template_id, True, "Edit")
    #logger.debug("TIMING create_plan_from_template B4 if planplugins in planTemplate.selectedPlugins.keys()...");
    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context)


@login_required
def edit_plan_run(request, _id):
    """
    Edit plan in template wizard
    """

    context = _plan_template_helper(request, _id, False, "EditPlan")

    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context)


@login_required
def copy_plan_run(request, _id):
    """
    Copy plan in template wizard
    """

    context = _plan_template_helper(request, _id, False, "CopyPlan")

    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context)


def _plan_template_helper(request, _id, isForTemplate, intent):
    data = _get_allApplProduct_data(isForTemplate)
    planTemplate = get_object_or_404(PlannedExperiment, pk=_id)
    runType = get_object_or_404(RunType, runType=planTemplate.runType)

    chipTypeDetails = None
    if planTemplate.chipType:
        chipTypeDetails = get_object_or_404(Chip, name=planTemplate.chipType)

    selectedProjectNames = [selectedProject.name for selectedProject in list(planTemplate.projects.all())]
    logger.debug("views._plan_template_helper selectedProjectNames=%s" % selectedProjectNames)

    #default - assume no plugins selected
    for plugin in data['plugins']:
        plugin.selected = False
    # mark plugins selected if any
    if 'planplugins' in planTemplate.selectedPlugins.keys():
        selectedPluginsNames = [p['name'] for p in planTemplate.selectedPlugins['planplugins']]
        # retrieve plugin userInput configuration
        selectedUserInput = {}
        for p in planTemplate.selectedPlugins['planplugins']:
              selectedUserInput[p['name']] = json.dumps(p.get('userInput',None))
        
        for plugin in data['plugins']:
            plugin.selected = plugin.name in selectedPluginsNames
            if plugin.name in selectedUserInput.keys():
                plugin.userInput = selectedUserInput[plugin.name]

    #default - assume no uploaders selected
    for plugin in data['uploaders']:
        plugin.selected = False
    # mark uploaders selected if any
    if 'planuploaders' in planTemplate.selectedPlugins.keys():
        selectedPluginsNames = [p['name'] for p in planTemplate.selectedPlugins['planuploaders']]
        for plugin in data['uploaders']:
            plugin.selected = plugin.name in selectedPluginsNames

        # get IonReporter config selections if any
        for p in planTemplate.selectedPlugins['planuploaders']:
            if ('IonReporter' in p['name']) and 'userInput' in p.keys():
                data['irConfigSaved'] = json.dumps(p['userInput'])
                # figure out if this is IR1.0 or higher (TODO: use IR version#; it's not correct at this time, so using IR name)
                data['irConfigSaved_version'] = 1.0 if p['name'] == 'IonReporterUploader_V1_0' else 1.2


    #planTemplateData contains what are available for selection
    #and what each application product's characteristics and default selection
    ctxd = {
        "intent": intent,
        "planTemplateData": data,
        "selectedApplProductData": "",
        "selectedPlanTemplate": planTemplate,
        "selectedRunType": runType,
        "selectedProjectNames": selectedProjectNames,
        "selectedChipTypeDetails": chipTypeDetails
    }
    context = RequestContext(request, ctxd)
    return context


@login_required
def copy_plan_template(request, template_id):
    """
    Clone plan template in template wizard
    """
    context = _plan_template_helper(request, template_id, True, "Copy")
    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context)


@login_required
def create_plan_from_template(request, template_id):
    """
    Create a plan run from existing template via wizard
    """
    #logger.debug("TIMING START - create_plan_from_template...");
    context = _plan_template_helper(request, template_id, False, "Plan Run")
    #logger.debug("TIMING create_plan_from_template B4 if planplugins in planTemplate.selectedPlugins.keys()...");
    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context)

@login_required
def batch_plans_from_template(request, template_id):
    """
    To create multiple plans from an existing template    
    """
    
    planTemplate = get_object_or_404(PlannedExperiment, pk=template_id)

    #planTemplateData contains what are available for selection
    ctxd = {
        "selectedPlanTemplate": planTemplate
    }
    context = RequestContext(request, ctxd)
    return render_to_response("rundb/plan/modal_batch_planning.html", context_instance=context)

@login_required
def getCSV_for_batch_planning(request, templateId, count):
    """
    To create csv file for batch planning based on an existing template    
    """
    
    #logger.debug("ENTER views.getCSV_for_batch_planning() templateId=%s; count=%s;" %(templateId, count))
    
    response = http.HttpResponse(mimetype='text/csv')
    now = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    response['Content-Disposition'] = 'attachment; filename=batchPlanning_%s.csv' % now
    
    hdr, body = get_template_data_for_batch_planning(templateId)
    
    writer = csv.writer(response)
    writer.writerow(hdr)
    
    index = 0
    max = int(count)
    while (index < max):
        writer.writerow(body)
        index += 1

    return response

@login_required
def upload_plans_for_template(request):
    """
    Allow user to upload a csv file to create plans based on a previously selected template
    """

    ctxd = {}

    context = RequestContext(request, ctxd)
    return render_to_response("rundb/plan/modal_batch_planning_upload.html", context_instance=context)


@login_required
@transaction.commit_manually
def save_uploaded_plans_for_template(request):
    """add plans, with CSV validation"""
    logger.info(request)
        
    if request.method != 'POST':
        logger.exception(format_exc())
        transaction.rollback()
        return HttpResponse(json.dumps({"error": "Error, unsupported HTTP Request method (%s) for saving plan upload." % request.method}), mimetype="application/json")
               
    postedfile = request.FILES['postedfile']
    destination = tempfile.NamedTemporaryFile(delete=False)

    for chunk in postedfile.chunks():
        destination.write(chunk)
    postedfile.close()
    destination.close()

    #check to ensure it is not empty
    headerCheck = open(destination.name, "rU")
    firstCSV = []
    for firstRow in csv.reader(headerCheck):
        firstCSV.append(firstRow)            
        #logger.info("views.save_uploaded_plans_for_template() firstRow=%s;" %(firstRow))
        
    headerCheck.close()
    if not firstRow:
        os.unlink(destination.name)
        transaction.rollback()
        return HttpResponse(json.dumps({"status": "Error: batch planning file is empty"}), mimetype="text/html")
        
    index = 0
    plans = []
    rawPlanDataList = []
    failed = {}
    file = open(destination.name, "rU")
    reader = csv.DictReader(file)
    for index, row in enumerate(reader, start=1):
        errorMsg, planObj, rawPlanDict, isToSkipRow = validate_csv_plan(row)
        
        logger.info("views.save_uploaded_plans_for_template() index=%d; errorMsg=%s; planDict=%s" %(index, errorMsg, rawPlanDict))
        if errorMsg:
            logger.info("views.save_uploaded_plans_for_template() ERROR MESSAGE index=%d; errorMsg=%s; planDict=%s" %(index, errorMsg, rawPlanDict))

            failed[index] = errorMsg
            continue
        elif isToSkipRow:
            logger.info("views.save_uploaded_plans_for_template() SKIPPED ROW index=%d; row=%s" %(index, row))            
            continue
        else:
            plans.append(planObj)
            rawPlanDataList.append(rawPlanDict)

    destination.close()  # now close and remove the temp file
    os.unlink(destination.name)
    if index == 0:
        transaction.rollback()
        return HttpResponse(json.dumps({"status": "Error: There must be at least one plan! Please reload the page and try again with more plans."}), mimetype="text/html")

    if failed:
        r = {"status": "Plan validation failed. The plans have not been saved.", "failed": failed}
        logger.info("views.save_uploaded_plans_for_template() failed=%s" %(r))
       
        transaction.rollback()
        return HttpResponse(json.dumps(r), mimetype="text/html")

    #saving to db needs to be the last thing to happen
    try:
        index = 0
        for plan in plans:
            plan.save()
            
            planDict = rawPlanDataList[index]
            
            # add QCtype thresholds
            qcTypes = QCType.objects.all()
            for qcType in qcTypes:
                qc_threshold = planDict.get(qcType.qcName, '')
                if qc_threshold:
                    # get existing PlannedExperimentQC if any
                    plannedExpQcs = PlannedExperimentQC.objects.filter(plannedExperiment=plan.id, qcType=qcType.id)
                    if len(plannedExpQcs) > 0:
                        for plannedExpQc in plannedExpQcs:
                            plannedExpQc.threshold = qc_threshold
                            plannedExpQc.save()
                    else:
                        kwargs = {
                            'plannedExperiment': plan,
                            'qcType': qcType,
                            'threshold': qc_threshold
                        }
                        plannedExpQc = PlannedExperimentQC(**kwargs)
                        plannedExpQc.save()

            # add projects
            projectObjList = get_projects(request.user, planDict)
            for project in projectObjList:
                if project:
                    plan.projects.add(project)

             
            index += 1
    except:
        logger.exception(format_exc())
        transaction.rollback()
        return HttpResponse(json.dumps({"status": "Error saving plans to database!"}), mimetype="text/html")
        ##return HttpResponse(json.dumps({"error": "Internal error while trying to save the plan."}), mimetype="application/json")
    else:
        transaction.commit()            
        r = {"status": "Plans Uploaded! The plans will be listed on the planned run page.", "failed": failed}
        return HttpResponse(json.dumps(r), mimetype="text/html")


@login_required
def get_application_product_presets(request):
    data = _get_allApplProduct_data(True)
    json_serializer = serializers.get_serializer("json")()
    fields = data.keys()
    result = json_serializer.serialize(data, fields=fields)
    return HttpResponse(result, mimetype="application/json")


def _get_allApplProduct_data(isForTemplate):
    def pretty(d, indent=0):
        if d:
            for key, value in d.iteritems():
                logger.debug('\t' * indent + str(key))
                if isinstance(value, dict):
                    pretty(value, indent + 1)
                else:
                    logger.debug('\t' * (indent + 1) + str(value))

    data = _get_base_planTemplate_data(isForTemplate)

    runTypes = list(RunType.objects.all())
    for appl in runTypes:

        applType = appl.runType

        try:
            #we should only have 1 default per application. TODO: add logic to ApplProduct to ensure that
            defaultApplProduct = list(ApplProduct.objects.filter(isActive=True, isDefault=True, applType=appl))
            if defaultApplProduct[0]:

                applData = {}

                applData["runType"] = defaultApplProduct[0].applType
                applData["reference"] = defaultApplProduct[0].defaultGenomeRefName
                applData["targetBedFile"] = defaultApplProduct[0].defaultTargetRegionBedFileName
                applData["hotSpotBedFile"] = defaultApplProduct[0].defaultHotSpotRegionBedFileName
                applData["seqKit"] = defaultApplProduct[0].defaultSequencingKit
                applData["libKit"] = defaultApplProduct[0].defaultLibraryKit
                applData["peSeqKit"] = defaultApplProduct[0].defaultPairedEndSequencingKit
                applData["peLibKit"] = defaultApplProduct[0].defaultPairedEndLibraryKit
                applData["chipType"] = defaultApplProduct[0].defaultChipType

                if defaultApplProduct[0].defaultChipType:
                    applData["chipTypeDetails"] = get_object_or_404(Chip, name=defaultApplProduct[0].defaultChipType)
                else:
                    applData["chipTypeDetails"] = None

                applData["isPairedEndSupported"] = defaultApplProduct[0].isPairedEndSupported
                applData["isDefaultPairedEnd"] = defaultApplProduct[0].isDefaultPairedEnd
                applData['defaultVariantFrequency'] = defaultApplProduct[0].defaultVariantFrequency

                applData['flowCount'] = defaultApplProduct[0].defaultFlowCount
                applData['peAdapterKit'] = defaultApplProduct[0].defaultPairedEndAdapterKit
                applData['templateKit'] = defaultApplProduct[0].defaultTemplateKit
                applData['controlSeqKit'] = defaultApplProduct[0].defaultControlSeqKit

                #20120619-TODO-add compatible plugins, default plugins

                data[applType] = applData

#                if applType == 'AMPS':
#                    pretty(data[applType])
            else:
                data[applType] = 'none'
        except:
            data[applType] = 'none'

    return data


def _dict_IR_plugins_uploaders(isForTemplate):
    '''
    Returns a dict containing keys:
        irConfigSelection,
        irConfigSelection_1,
        plugins,
        uploaders,
    '''

    data = {}

    data['irConfigSelection_1'] = json.dumps(None)
    data['irConfigSelection'] = json.dumps(None)

    #based on features.EXPORT to determine if a plugin should be included in the Export tab
    uploaderNames = []
    pluginNames = []
    #selected=True + active=True = plugin ENABLED
    #since template creation/edit does not support IR configuration, we're going to skip going to the cloud to fetch IR
    #configuration selectable values just so to speed things up
    pluginCandidates = Plugin.objects.filter(selected=True, active=True)
    if isForTemplate:
        pluginCandidates = pluginCandidates.exclude(name__icontains="IonReporter")
        IRuploaders = list(Plugin.objects.filter(name__icontains="IonReporter", selected=True, active=True).order_by('name', '-version'))
    pluginCandidates = list(pluginCandidates.order_by('name', '-version'))

    # Issue bulk query for efficiency
    plugin_list = [(plugin.name, plugin.pluginscript(), {'plugin':plugin}) for plugin in pluginCandidates]
    from iondb.plugins.manager import pluginmanager
    pluginInfo = pluginmanager.get_plugininfo_list(plugin_list)

    # But don't use pluginInfo - query plugins individually via ORM
    for p in pluginCandidates:
        info = p.info()
        if info:
            infoName = info['name']
            if 'features' in info:
                #watch out: "Export" was changed to "export" recently!
                if ('export' in (feature.lower() for feature in info['features'])):
                    uploaderNames.append(infoName)
                else:
                    pluginNames.append(infoName)
            else:
                pluginNames.append(infoName)

            if infoName.lower() == 'IonReporterUploader_V1_0'.lower():
                if 'config' in info:
                    data['irConfigSelection_1'] = json.dumps(info['config'])
            elif ('IonReporterUploader'.lower() in infoName.lower()):
                if 'config' in info:
                    data['irConfigSelection'] = json.dumps(info['config'])

    #force querySet to be evaluated
    data['plugins'] = list(Plugin.objects.filter(selected=True, active=True).filter(name__in=pluginNames).order_by('name', '-version'))
    data['uploaders'] = list(Plugin.objects.filter(selected=True, active=True).filter(name__in=uploaderNames).order_by('name', '-version'))
    # add back IR plugins
    if isForTemplate:
        data['uploaders'] += IRuploaders
    return data


def _get_base_planTemplate_data(isForTemplate):
    data = {}

    data["runTypes"] = list(RunType.objects.all().order_by('nucleotideType', 'runType'))
    data["barcodes"] = list(dnaBarcode.objects.values('name').distinct().order_by('name'))

    ##barcodeKitNames = dnaBarcode.objects.values_list('name', flat=True).distinct().order_by('name')
    ##for barcodeKitName in barcodeKitNames:
    ##    data[barcodeKitName] = dnaBarcode.objects.filter(name=barcodeKitName).order_by('index')

    data["barcodeKitInfo"] = list(dnaBarcode.objects.all().order_by('name', 'index'))

    references = list(ReferenceGenome.objects.all().filter(index_version=settings.TMAP_VERSION))
    data["references"] = references
    data["referenceShortNames"] = [ref.short_name for ref in references]

    data.update(dict_bed_hotspot())

    data["seqKits"] = KitInfo.objects.filter(kitType='SequencingKit', isActive=True).order_by("name")
    data["libKits"] = KitInfo.objects.filter(kitType='LibraryKit', isActive=True).order_by("name")

    data["variantfrequencies"] = VariantFrequencies.objects.all().order_by("name")

    #the entry marked as the default will be on top of the list
    data["forwardLibKeys"] = LibraryKey.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'name')
    data["forward3Adapters"] = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'name')

    #pairedEnd does not have special forward library keys
    #for TS-4669: remove paired-end from wizard, if there are no active PE lib kits, do not prepare pe keys or adapters
    peLibKits = KitInfo.objects.filter(kitType='LibraryKit', runMode='pe', isActive=True)
    if (peLibKits.count() > 0):
        data["peForwardLibKeys"] = LibraryKey.objects.filter(direction='Forward').order_by('-isDefault', 'name')
        data["peForward3Adapters"] = ThreePrimeadapter.objects.filter(direction='Forward', runMode='pe').order_by('-isDefault', 'name')
        data["reverseLibKeys"] = LibraryKey.objects.filter(direction='Reverse').order_by('-isDefault', 'name')
        data["reverse3Adapters"] = ThreePrimeadapter.objects.filter(direction='Reverse').order_by('-isDefault', 'name')
    else:
        data["peForwardLibKeys"] = None
        data["peForward3Adapters"] = None
        data["reverseLibKeys"] = None
        data["reverse3Adapters"] = None

    #chip types
    data['chipTypes'] = list(Chip.objects.all().order_by('name'))
    #QC
    data['qcTypes'] = list(QCType.objects.all().order_by('qcName'))
    #project
    data['projects'] = list(Project.objects.filter(public=True).order_by('name'))

    #templating kit selection
    data["templateKits"] = KitInfo.objects.filter(kitType='TemplatingKit', isActive=True).order_by("name")
    #control sequence kit selection
    data["controlSeqKits"] = KitInfo.objects.filter(kitType='ControlSequenceKit', isActive=True).order_by("name")

    #pairedEnd library adapter selection
    #for TS-4669: remove paired-end from wizard, if there are no active PE seq kits, do not prepare pe keys or adapters
    if (peLibKits.count() > 0):
        data["pairedEndLibAdapters"] = KitInfo.objects.filter(kitType='AdapterKit', runMode="pe", isActive=True).order_by('name')
    else:
        data["pairedEndLibAdapters"] = None

    #samplePrep kits
    data["samplePrepKits"] = KitInfo.objects.filter(kitType='SamplePrepKit', isActive=True).order_by('name')

    data.update(_dict_IR_plugins_uploaders(isForTemplate))

    #to allow data entry for multiple non-barcoded samples at the plan wizard
    data['nonBarcodedSamples_irConfig_loopCounter'] = [i + 1 for i in range(20)]

    return data


@login_required
@transaction.commit_manually
def save_plan_or_template(request, planOid):
    """
    Saving new or edited plan/template to db (source: plan template wizard)
    Editing a planned run from having 1 sample to 2 samples will result in one edited planned run and one new planned run
    """
    def isReusable(submitIntent):
        return not (submitIntent == 'savePlan' or submitIntent == 'updatePlan')

    def isValidChars(value, validChars=r'^[a-zA-Z0-9-_\.\s\,]+$'):
        ''' Determines if value is valid: letters, numbers, spaces, dashes, underscores only '''
        return bool(re.compile(validChars).match(value))

    if request.method != 'POST':
        logger.exception(format_exc())
        return HttpResponse(json.dumps({"error": "Error, unsupported HTTP Request method (%s) for plan update." % request.method}), mimetype="application/json")

    # Process Inputs

    # pylint:disable=E1103
    json_data = simplejson.loads(request.raw_post_data)
    submitIntent = json_data.get('submitIntent', '')
    logger.debug('views.editplannedexperiment POST.raw_post_data... simplejson Data: "%s"' % json_data)
    logger.debug("views.editplannedexperiment submitIntent=%s" % submitIntent)
    # saving Template or Planned Run
    isReusable = isReusable(submitIntent)
    runModeValue = json_data.get('runMode', 'single')
    isPlanGroupValue = runModeValue == 'pe' and not isReusable
    libraryKeyValue = json_data.get('libraryKey', '')
    forward3primeAdapterValue = json_data.get('forward3primeAdapter', '')

    msgvalue = 'Run Plan' if not isReusable else 'Template'
    if runModeValue == 'pe':
        return HttpResponse(json.dumps({"error": "Error, paired-end plan is no longer supported. %s will not be saved." % (msgvalue)}), mimetype="application/html")
    
    planDisplayedNameValue = json_data.get('planDisplayedName', '').strip()
    noteValue = json_data.get('notes_workaround', '')

    # perform server-side validation to avoid things falling through the crack    
    if not planDisplayedNameValue:
        return HttpResponse(json.dumps({"error": "Error, please enter a %s Name." % (msgvalue)}), mimetype="application/html")

    if not isValidChars(planDisplayedNameValue):
        return HttpResponse(json.dumps({"error": "Error, %s Name should contain only numbers, letters, spaces, and the following: . - _" % (msgvalue)}), mimetype="application/html")

    if noteValue and not isValidChars(noteValue):
        return HttpResponse(json.dumps({"error": "Error, %s note should contain only numbers, letters, spaces, and the following: . - _" % (msgvalue)}), mimetype="application/html")

    # Projects
    projectObjList = get_projects(request.user, json_data)

    # IonReporterUploader configuration and samples
    selectedPlugins = json_data.get('selectedPlugins', {})
    IRconfigList = json_data.get('irConfigList', [])
    IRU_1_2_selected = False
    for uploader in selectedPlugins.get('planuploaders', []):
        if 'ionreporteruploader' in uploader['name'].lower() and uploader['name'] != 'IonReporterUploader_V1_0':
            IRU_1_2_selected = True
            samples_IRconfig = json_data.get('sample_irConfig', '')
            samples_IRconfig = ','.join(samples_IRconfig)

            #generate UUID for unique setIds
            id_uuid = {}
            setids = [ir['setid'] for ir in IRconfigList]
            for setid in set(setids):
                id_uuid[setid] = str(uuid.uuid4())
            for ir_config in IRconfigList:
                ir_config['setid'] += '__' + id_uuid[ir_config['setid']]

    # Samples

    barcodeIdValue = json_data.get('barcodeId', '')
    barcodedSamples = ''
    sampleValidationErrorMsg = ''

    # one Plan will be created per entry in sampleList
    # samples for barcoded Plan have a separate field (barcodedSamples)

    if isReusable:
        # samples entered only when saving planned run (not template)
        sampleList = ['']
    elif barcodeIdValue:
        # a barcode Set is selected
        sampleList = ['']
        bcSamplesValues = json_data.get('bcSamples_workaround', '')
        bcDictionary = {}
        bcId = ""
        for token in bcSamplesValues.split(","):
            if ((token.find("bcKey|")) == 0):
                bcId, bcId_str = token.split("|")[1:]
            else:
                sample = token.strip()
                if bcId and sample:
                    if not isValidChars(sample):
                        sampleValidationErrorMsg += sample + ', '

                    bcDictionary.setdefault(sample, {}).setdefault('barcodes',[]).append(bcId_str)
                bcId = ""

        barcodedSamples = simplejson.dumps(bcDictionary)
        logger.debug("views.editplannedexperiment after simplejson.dumps... barcodedSamples=%s;" % (barcodedSamples))

        if not bcDictionary:
            transaction.rollback()
            return HttpResponse(json.dumps({"error": "Error, please enter at least one barcode sample name."}), mimetype="application/html")

    else:
        # Non-barcoded samples
        sampleList = []
        if IRU_1_2_selected:
            samples = samples_IRconfig
        else:
            samples = json_data.get('samples_workaround', '')

        for sample in samples.split(','):
            if sample.strip():
                if not isValidChars(sample):
                    sampleValidationErrorMsg += sample + ', '
                else:
                    sampleList.append(sample)

        logger.debug("views.editplannedexperiment sampleList=%s " % (sampleList))
        
        if  len(sampleList) == 0:
            transaction.rollback()
            return HttpResponse(json.dumps({"error": "Error, please enter a sample name for the run plan."}), mimetype="application/html")

    # Samples validation
    if sampleValidationErrorMsg:
        message = "Error, sample name should contain only numbers, letters, spaces, and the following: . - _"
        message = message + ' <br>Please fix: ' + sampleValidationErrorMsg
        transaction.rollback()
        return HttpResponse(json.dumps({"error": message}), mimetype="application/html")

    selectedPluginsValue = json_data.get('selectedPlugins', [])

    # end processing input data

    # Edit/Create Plan(s)

    if int(planOid) == 0:
        edit_existing_plan = False
    else:
        edit_existing_plan = True

    for i, sample in enumerate(sampleList):
        logger.debug("...LOOP... views.editplannedexperiment SAMPLE=%s; isSystem=%s; isReusable=%s; isPlanGroup=%s "
                     % (sample.strip(), json_data["isSystem"], isReusable, isPlanGroupValue))

        # add IonReporter config values for each sample
        if len(IRconfigList) > 0:
            for uploader in selectedPluginsValue['planuploaders']:
                if 'ionreporter' in uploader['name'].lower():
                    if len(IRconfigList) > 1 and not barcodeIdValue:
                        uploader['userInput'] = [IRconfigList[i]]
                    else:
                        uploader['userInput'] = IRconfigList

        if len(sampleList) > 1:
            inputPlanDisplayedName = planDisplayedNameValue + '_' + sample.strip()
        else:
            inputPlanDisplayedName = planDisplayedNameValue

        kwargs = {
            'planDisplayedName': inputPlanDisplayedName,
            "planName": inputPlanDisplayedName.replace(' ', '_'),
            'chipType': json_data.get('chipType', ''),
            'usePreBeadfind': toBoolean(json_data['usePreBeadfind'], False),
            'usePostBeadfind': toBoolean(json_data['usePostBeadfind'], False),
            'flows': json_data.get('flows', None),
            'autoAnalyze': True,
            'preAnalysis': True,
            'runType': json_data['runType'],
            'library': json_data.get('library', ''),
            'notes': noteValue,
            'bedfile': json_data.get('bedfile', ''),
            'regionfile': json_data.get('regionfile', ''),
            'variantfrequency': json_data.get('variantfrequency', ''),
            'librarykitname': json_data.get('librarykitname', ''),
            'sequencekitname': json_data.get('sequencekitname', ''),
            'barcodeId': barcodeIdValue,
            'templatingKitName': json_data.get('templatekitname', ''),
            'controlSequencekitname': json_data.get('controlsequence', ''),
            'runMode': runModeValue,
            'isSystem': toBoolean(json_data['isSystem'], False),
            'isReusable': isReusable,
            'isPlanGroup': isPlanGroupValue,
            'sampleDisplayedName': sample.strip(),
            "sample": sample.strip().replace(' ', '_'),
            'username': request.user.username,
            'isFavorite': toBoolean(json_data.get('isFavorite', 'False'), False),
            'barcodedSamples': barcodedSamples,
            'libraryKey': libraryKeyValue,
            'forward3primeadapter': forward3primeAdapterValue,
            'reverselibrarykey': json_data.get('reverselibrarykey', ''),
            'reverse3primeadapter': json_data.get('reverse3primeAdapter', ''),
            'pairedEndLibraryAdapterName': json_data.get('pairedEndLibraryAdapterName', ''),
            'samplePrepKitName': json_data.get('samplePrepKitName', ''),
            'selectedPlugins': selectedPluginsValue
        }

        #if we're changing a plan from having 1 sample to say 2 samples, we need to UPDATE 1 plan and CREATE 1 plan!!
        try:
            if not edit_existing_plan:
                planTemplate = PlannedExperiment(**kwargs)
            else:
                planTemplate = PlannedExperiment.objects.get(pk=planOid)
                for key, value in kwargs.items():
                    setattr(planTemplate, key, value)
                edit_existing_plan = False

            planTemplate.save()

            # Update QCtype thresholds
            qcTypes = QCType.objects.all()
            for qcType in qcTypes:
                qc_threshold = json_data.get(qcType.qcName, '')
                if qc_threshold:
                    # get existing PlannedExperimentQC if any
                    plannedExpQcs = PlannedExperimentQC.objects.filter(plannedExperiment=planTemplate.id, qcType=qcType.id)
                    if len(plannedExpQcs) > 0:
                        for plannedExpQc in plannedExpQcs:
                            plannedExpQc.threshold = qc_threshold
                            plannedExpQc.save()
                    else:
                        kwargs = {
                            'plannedExperiment': planTemplate,
                            'qcType': qcType,
                            'threshold': qc_threshold
                        }
                        plannedExpQc = PlannedExperimentQC(**kwargs)
                        plannedExpQc.save()

            # add/remove projects
            if projectObjList:
                #TODO: refactor this logic to simplify using django orm
                projectNameList = [project.name for project in projectObjList]
                for currentProject in planTemplate.projects.all():
                    if currentProject.name not in projectNameList:
                        planTemplate.projects.remove(currentProject)
                for projectObj in projectObjList:
                    planTemplate.projects.add(projectObj)
            else:
                planTemplate.projects.clear()

        except:
            transaction.rollback()
            logger.exception(format_exc())
            return HttpResponse(json.dumps({"error": "Internal error while trying to save the plan."}), mimetype="application/json")
        else:
            transaction.commit()

    return HttpResponse(json.dumps({"status": "plan template updated successfully"}), mimetype="application/json")
