# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.contrib.auth.decorators import login_required
from django.template import RequestContext
from django.shortcuts import render_to_response, get_object_or_404,\
    get_list_or_404
from django.conf import settings
from django.db import transaction
from django.http import HttpResponse

from iondb.rundb import models
from iondb.rundb.models import PlannedExperiment, RunType, ApplProduct,\
    ReferenceGenome, Content, KitInfo, VariantFrequencies, dnaBarcode,\
    LibraryKey, ThreePrimeadapter, Chip, QCType, Project, Plugin,\
    PlannedExperimentQC
from iondb.rundb import tasks
from iondb.rundb.views import toBoolean

from traceback import format_exc
import json
import simplejson
import uuid

import logging
from django.core import serializers
from iondb.rundb.api import PlannedExperimentResource, RunTypeResource,\
    dnaBarcodeResource, ChipResource
import re
from django.core.urlresolvers import reverse

from iondb.rundb.plan import IRConfig_jsonBlock, IRConfig_v1_jsonBlock

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
    _type = 'plannedexperiment';
    planTemplates = get_list_or_404(PlannedExperiment, pk__in=pks)
    _typeDescription = "Template" if planTemplates[0].isReusable == True else "Planned Run"
    actions = []
    for pk in pks:
        actions.append(reverse('api_dispatch_detail', kwargs={'resource_name':_type, 'api_name':'v1', 'pk':int(pk)}))
    names = ', '.join([x.planName for x in planTemplates])
    ctx = RequestContext(request, { 
                                    "id":pks[0]
                                    , "name" : names
                                    , "ids": json.dumps(pks)
                                    , "names": names
                                    , "method":"DELETE"
                                    , 'methodDescription': 'Delete' 
                                    , "readonly":False
                                    , 'type': _typeDescription
                                    , 'action': actions[0]
                                    , 'actions' : json.dumps(actions)
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
    
    #workaround: to get around error at plan/template creation due to barcoded sample handling (for planRun & editPlanRun).
    #            I'm going to include the system default plan here even though this will
    #            not be used
    unusedPlanTemplate = ""
    try:
        unusedPlanTemplate = PlannedExperiment.objects.filter(isReusable=True, isSystem=True, isSystemDefault=True)[0]
    except IndexError:
        #if somehow there is no system default, we'll settle for a system template
        unusedPlanTemplate = PlannedExperiment.objects.filter(isReusable=True, isSystem=True)[0]
        
    ctxd = {
        "intent": intent,
        "planTemplateData" : data,
        "selectedPlanTemplate": unusedPlanTemplate
        
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
        
    if not ctxd.has_key("selectedApplProductData"):
        #logger.debug("GENERIC add_plan.. ")        
        ctxd["selectedApplProductData"] = data["GENS"]

    context = RequestContext(request, ctxd)

    #logger.debug("TIMING END - _add_plan for either plan or template...");

    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context) 

def _review_plan(request, id):       
    per = PlannedExperimentResource()
    pe = per.obj_get(pk = id)
    per_bundle = per.build_bundle(obj=pe, request=request)
    rtr = RunTypeResource()
    rt = rtr.obj_get(runType = pe.runType)
    rtr_bundle = rtr.build_bundle(obj=rt, request=request)
    
    dnar = dnaBarcodeResource()
    dna = dnar.obj_get_list(name = pe.barcodeId, request = request).order_by('index')
    dna_bundles = [dnar.build_bundle(obj=x, request=request) for x in dna]
    
    chipResource = ChipResource()

    chipResource_serialize_json = None
    if pe.chipType:
        chip = chipResource.obj_get(name = pe.chipType)
        chipResource_bundle = chipResource.build_bundle(obj = chip, request = request)
        chipResource_serialize_json = chipResource.serialize(None, chipResource.full_dehydrate(chipResource_bundle), 'application/json')
    else:
        chipResource_bundle = None 
        chipResource_serialize_json = json.dumps(None)      
                                       
    return render_to_response("rundb/plan/modal_review_plan.html", {
        # Other things here.
         "plan": pe
        ,"selectedPlanTemplate": per.serialize(None, per.full_dehydrate(per_bundle), 'application/json')
        , "selectedRunType": rtr.serialize(None, rtr.full_dehydrate(rtr_bundle), 'application/json')
        , "selectedBarcodes": rtr.serialize(None, [dnar.full_dehydrate(bundle) for bundle in dna_bundles], 'application/json')        
        , "view": 'template' if 'template' in request.path else 'Planned Run'
        , "selectedChip":  chipResource_serialize_json                                                                 
    })
    
        
@login_required
def review_plan_template(request, id):
    """
    Review plan template contents 
    """
    return _review_plan(request, id)


@login_required
def review_plan_run(request, id):
    """
    Review plan contents 
    """
    return _review_plan(request, id)

@login_required
def edit_plan_template(request, id):
    """
    Edit plan template in template wizard 
    """
    
    data = _get_allApplProduct_data(True)

    planTemplate = get_object_or_404(PlannedExperiment,pk=id)
    runType = get_object_or_404(RunType, runType = planTemplate.runType)

    chipTypeDetails = None
    if planTemplate.chipType:
        chipTypeDetails = get_object_or_404(Chip, name = planTemplate.chipType)

    selectedProjects = planTemplate.projects.all()
    selectedProjectNames = []
    
    for selectedProject in selectedProjects:
        selectedProjectNames.append(selectedProject.name)
        
    logger.debug("views.edit_plan_template selectedProjectNames=%s" % selectedProjectNames)
    
    # mark plugins selected if any
    if 'planplugins' in planTemplate.selectedPlugins.keys():
        selectedPluginsNames = [p['name'] for p in planTemplate.selectedPlugins['planplugins']]
        for plugin in data['plugins']:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False    
    else: 
        #if no plugins selected
        for plugin in data['plugins']:
            plugin.selected = False   
                
    # mark uploaders selected if any
    if 'planuploaders' in planTemplate.selectedPlugins.keys():
        selectedPluginsNames = [p['name'] for p in planTemplate.selectedPlugins['planuploaders']]
        for plugin in data['uploaders']:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False    

        for plugin in data["IRuploads"]:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False  
    else:
        #if no uploaders selected
        for plugin in data['uploaders']:
            plugin.selected = False    

        for plugin in data["IRuploads"]:        
            plugin.selected = False              

                            
    #planTemplateData contains what are available for selection
    #and what each application product's characteristics and default selection
    ctxd = {
            "intent": "Edit",
            "planTemplateData" : data,
            "selectedApplProductData" : "",
            "selectedPlanTemplate": planTemplate,
            "selectedRunType" : runType,
            "selectedProjectNames": selectedProjectNames,
            "selectedChipTypeDetails": chipTypeDetails            
            }


    context = RequestContext(request, ctxd)

    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context)    


@login_required
def edit_plan_run(request, id):
    """
    Edit plan in template wizard 
    """
    
    data = _get_allApplProduct_data(False)

    planTemplate = get_object_or_404(PlannedExperiment,pk=id)
    runType = get_object_or_404(RunType, runType = planTemplate.runType)

    chipTypeDetails = None
    if planTemplate.chipType:
        chipTypeDetails = get_object_or_404(Chip, name = planTemplate.chipType)
   
    selectedProjects = planTemplate.projects.all()
    selectedProjectNames = []
    
    for selectedProject in selectedProjects:
        selectedProjectNames.append(selectedProject.name)
        
    logger.debug("views.edit_plan_run selectedProjectNames=%s" % selectedProjectNames)
    
    # mark plugins selected if any
    if 'planplugins' in planTemplate.selectedPlugins.keys():
        selectedPluginsNames = [p['name'] for p in planTemplate.selectedPlugins['planplugins']]
        for plugin in data['plugins']:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False    
    else: 
        #if no plugins selected
        for plugin in data['plugins']:
            plugin.selected = False   
            
    # mark uploaders selected if any
    if 'planuploaders' in planTemplate.selectedPlugins.keys():
        selectedPluginsNames = [p['name'] for p in planTemplate.selectedPlugins['planuploaders']]
        for plugin in data['uploaders']:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False

        for plugin in data["IRuploads"]:
            if plugin.name in selectedPluginsNames:
                 plugin.selected = True
            else:
                plugin.selected = False  
                
        # get IonReporter config selections if any
        ## This should be querying for all plugins which have 'Features.EXPORT'... IonReporter only for now.
        for p in planTemplate.selectedPlugins['planuploaders']:
            if ('IonReporter' in p['name']) and 'userInput' in p.keys():
                data['irConfigSaved'] = json.dumps(p['userInput'])   
                # figure out if this is IR1.0 or higher (TODO: use IR version#; it's not correct at this time, so using IR name)
                if p['name'] == 'IonReporterUploader_V1_0':
                    data['irConfigSaved_version'] = 1.0
                else:
                    data['irConfigSaved_version'] = 1.2
    else:
        #if no uploaders selected
        for plugin in data['uploaders']:
            plugin.selected = False    

        for plugin in data["IRuploads"]:
            plugin.selected = False              
                
    #planTemplateData contains what are available for selection
    #and what each application product's characteristics and default selection
    ctxd = {
            "intent": "EditPlan",
            "planTemplateData" : data,
            "selectedApplProductData" : "",
            "selectedPlanTemplate": planTemplate,
            "selectedRunType" : runType,
            "selectedProjectNames": selectedProjectNames,
            "selectedChipTypeDetails": chipTypeDetails
            }


    context = RequestContext(request, ctxd)

    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context)    


@login_required
def copy_plan_template(request, id):
    """
    Clone plan template in template wizard 
    """
    
    data = _get_allApplProduct_data(True)
    

    planTemplate = get_object_or_404(PlannedExperiment,pk=id)
    runType = get_object_or_404(RunType, runType = planTemplate.runType)

    chipTypeDetails = None
    if planTemplate.chipType:
        chipTypeDetails = get_object_or_404(Chip, name = planTemplate.chipType)

    selectedProjects = planTemplate.projects.all()
    selectedProjectNames = []
    
    for selectedProject in selectedProjects:
        selectedProjectNames.append(selectedProject.name)
        
    logger.debug("views.copy_plan_template selectedProjectNames=%s" % selectedProjectNames)

    # mark plugins selected if any
    if 'planplugins' in planTemplate.selectedPlugins.keys():
        selectedPluginsNames = [p['name'] for p in planTemplate.selectedPlugins['planplugins']]
        for plugin in data['plugins']:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False
    else: 
        #if no plugins selected
        for plugin in data['plugins']:
            plugin.selected = False   
                
                            
    # mark uploaders selected if any
    if 'planuploaders' in planTemplate.selectedPlugins.keys():
        selectedPluginsNames = [p['name'] for p in planTemplate.selectedPlugins['planuploaders']]
        for plugin in data['uploaders']:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False    

        for plugin in data["IRuploads"]:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False                  
    else:
        #if no uploaders selected
        for plugin in data['uploaders']:
            plugin.selected = False    

        for plugin in data["IRuploads"]:            
            plugin.selected = False       
        
        
    #planTemplateData contains what are available for selection
    #and what each application product's characteristics and default selection
    ctxd = {
            "intent": "Copy",
            "planTemplateData" : data,
            "selectedApplProductData" : "",
            "selectedPlanTemplate": planTemplate,
            "selectedRunType" : runType,
            "selectedProjectNames": selectedProjectNames,
            "selectedChipTypeDetails": chipTypeDetails
            }
    context = RequestContext(request, ctxd)
    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context)    


@login_required
def create_planRunFromTemplate(request, template_id):
    """
    Create a plan run from existing template via wizard 
    """
    
    data = _get_allApplProduct_data(False)
    planTemplate = get_object_or_404(PlannedExperiment,pk=template_id)
    runType = get_object_or_404(RunType, runType = planTemplate.runType)

    chipTypeDetails = None
    if planTemplate.chipType:
        chipTypeDetails = get_object_or_404(Chip, name = planTemplate.chipType)

    selectedProjects = planTemplate.projects.all()
    selectedProjectNames = []
    
    for selectedProject in selectedProjects:
        selectedProjectNames.append(selectedProject.name)
            
    # mark plugins selected if any
    if 'planplugins' in planTemplate.selectedPlugins.keys():
        selectedPluginsNames = [p['name'] for p in planTemplate.selectedPlugins['planplugins']]
        for plugin in data['plugins']:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False    
    else: 
        #if no plugins selected
        for plugin in data['plugins']:
            plugin.selected = False   
                
    # mark uploaders selected if any
    if 'planuploaders' in planTemplate.selectedPlugins.keys():
        selectedPluginsNames = [p['name'] for p in planTemplate.selectedPlugins['planuploaders']]
        for plugin in data['uploaders']:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False        

        for plugin in data["IRuploads"]:
            if plugin.name in selectedPluginsNames:
                plugin.selected = True
            else:
                plugin.selected = False  
    else:
        #if no uploaders selected
        for plugin in data['uploaders']:
            plugin.selected = False    

        for plugin in data["IRuploads"]:            
            plugin.selected = False       
                
    #planTemplateData contains what are available for selection
    #and what each application product's characteristics and default selection
    ctxd = {
            "intent": "Plan Run",
            "planTemplateData" : data,
            "selectedApplProductData" : "",
            "selectedPlanTemplate": planTemplate,
            "selectedRunType" : runType,
            "selectedProjectNames": selectedProjectNames,
            "selectedChipTypeDetails": chipTypeDetails
            }
    context = RequestContext(request, ctxd)
    return render_to_response("rundb/plan/modal_plan_wizard.html", context_instance=context)    

@login_required
def get_application_product_presets(request):
    data = _get_allApplProduct_data(True);
    json_serializer = serializers.get_serializer("json")()
    fields = data.keys()
    result = json_serializer.serialize(data, fields = fields)
    return HttpResponse(result , mimetype="application/json")    
    
def _get_allApplProduct_data(isForTemplate):    
    data = _get_base_planTemplate_data(isForTemplate)
    
    for appl in RunType.objects.all():
        
        applType = appl.runType

        try:
            #we should only have 1 default per application. TODO: add logic to ApplProduct to ensure that
            defaultApplProduct = ApplProduct.objects.filter(isActive = True, isDefault = True, applType = appl)
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
                    applData["chipTypeDetails"] =  get_object_or_404(Chip, name = defaultApplProduct[0].defaultChipType)
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
                
                ##if applType == 'AMPS':
                  ##  pretty(data[applType])
            else:
                data[applType] = 'none'
        except:
            data[applType] = 'none'

    
    return data

def _get_base_planTemplate_data(isForTemplate):
    data = {}
    
    data["runTypes"] = RunType.objects.all().order_by("id")
    data["barcodes"] = dnaBarcode.objects.values('name').distinct().order_by('name')

    ##barcodeKitNames = dnaBarcode.objects.values_list('name', flat=True).distinct().order_by('name')
    ##for barcodeKitName in barcodeKitNames:        
    ##    data[barcodeKitName] = dnaBarcode.objects.filter(name=barcodeKitName).order_by('index')        

    data["barcodeKitInfo"] = dnaBarcode.objects.all().order_by('name', 'index') 
    
    references = ReferenceGenome.objects.all().filter(index_version=settings.TMAP_VERSION)
    data["references"] = references
    data["referenceShortNames"] = [ref.short_name for ref in references]
    
    allFiles = Content.objects.filter(publisher__name="BED",path__contains="/unmerged/detail/")
    bedFiles, hotspotFiles = [], []
    bedFileFullPaths, bedFilePaths, hotspotFullPaths, hotspotPaths = [], [], [], []
    for file in allFiles:
        if file.meta.get("hotspot", False):
            hotspotFiles.append(file)
            hotspotFullPaths.append(file.file)
            hotspotPaths.append(file.path)
        else:
            bedFiles.append(file)
            bedFileFullPaths.append(file.file)
            bedFilePaths.append(file.path)
            
    data["bedFiles"] = bedFiles
    data["hotspotFiles"] = hotspotFiles

    data["bedFileFullPaths"] = bedFileFullPaths
    data["bedFilePaths"] = bedFilePaths
    data["hotspotFullPaths"] = hotspotFullPaths
    data["hotspotPaths"] = hotspotPaths
    
    data["seqKits"] = KitInfo.objects.filter(kitType='SequencingKit', isActive=True).order_by("name")
    data["libKits"] = KitInfo.objects.filter(kitType='LibraryKit', isActive=True).order_by("name")
    
    data["variantfrequencies"] = VariantFrequencies.objects.all().order_by("name")

    #include all the active IonReporter versions with oldest version last:
    IRuploads = Plugin.objects.filter(name__icontains="IonReporter",selected=True,active=True).order_by('name', '-version')     
    data["IRuploads"] = IRuploads
        
    #the entry marked as the default will be on top of the list
    data["forwardLibKeys"] = LibraryKey.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'name')
    data["forward3Adapters"] = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'name')
    
    #pairedEnd does not have special forward library keys
    #for TS-4669: remove paired-end from wizard, if there are no active PE lib kits, do not prepare pe keys or adapters
    peLibKits = KitInfo.objects.filter(kitType='LibraryKit', runMode = 'pe', isActive=True)
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
    data['chipTypes'] = Chip.objects.all().order_by('name')
    #QC
    data['qcTypes'] = QCType.objects.all().order_by('qcName')
    #project
    data['projects'] = Project.objects.filter(public=True).order_by('name')

    #templating kit selection
    data["templateKits"] = KitInfo.objects.filter(kitType='TemplatingKit', isActive=True).order_by("name")    
    #control sequence kit selection
    data["controlSeqKits"] = KitInfo.objects.filter(kitType='ControlSequenceKit', isActive=True).order_by("name")

    #pairedEnd library adapter selection
    #for TS-4669: remove paired-end from wizard, if there are no active PE seq kits, do not prepare pe keys or adapters
    if (peLibKits.count() > 0):
        data["pairedEndLibAdapters"] = KitInfo.objects.filter(kitType = 'AdapterKit', runMode="pe", isActive=True).order_by('name')
    else:
        data["pairedEndLibAdapters"] = None
        
    #samplePrep kits
    data["samplePrepKits"] = KitInfo.objects.filter(kitType = 'SamplePrepKit', isActive=True).order_by('name')

    #TODO: pass a list of JSON objects for IR configuration selection
    ## Query IR plugin for config
    #irConfigSelection_list = []
    #if IRuploads:
    #   for irUpload in IRuploads:
    #      irConfigSelection = {}
    #        irinfo = IRupload.info()
    #                    
    #        if irInfo and 'config' in irInfo:                            
    #            logger.debug("LIST irUpload=%s, version=%s" % (irUpload.name, irUpload.version))
    #        
    #            irConfigSelection = IRinfo['config']
    #            logger.warn(irConfigSelection)
    #        else:
    #            # FIXME - remove this - fall back to sample fields until this is working
    #            irConfigSelection = IRConfig_jsonBlock.sample_relationship_fields()
    #        
    #        irConfigSelection_list.append(irConfigSelection)
    #        
    #    data['irConfigSelections'] = irConfigSelection_list
    #else:
    #    data['irConfigSelections'] = irConfigSelection_list
    
            
    #based on features.EXPORT to determine if a plugin should be included in the Export tab
    #note that IonReporter plugins are already included in data["IRuploads"] regardless whether they have the Features.EXPORT or not
    uploaderNames = []
    pluginNames = []
    
    #selected=True + active=True = plugin ENABLED  
    pluginCandidates = Plugin.objects.filter(selected=True,active=True).order_by('name', '-version')
    
    #since template creation/edit does not support IR configuration, we're going to skip going to the cloud to fetch IR 
    #configuration selectable values just so to speed things up
    if isForTemplate:
        pluginCandidates = Plugin.objects.filter(selected=True,active=True).exclude(name__icontains="IonReporter").order_by('name', '-version')

    data['irConfigSelection_1'] = json.dumps(None)
    data['irConfigSelection'] = json.dumps(None)

    plugin_list = [(plugin.name, plugin.pluginscript(), { 'plugin' : plugin}) for plugin in pluginCandidates]

    from iondb.plugins.manager import pluginmanager

    pluginInfo = pluginmanager.get_plugininfo_list(plugin_list)

    # FIX Munged Names
    for (name, script, context) in plugin_list:
        info = pluginInfo.get(script, None)
        if info and info['name'] != name:
            info['name'] = name

    if pluginInfo:
        for key, info in pluginInfo.iteritems():
            #logger.debug("TIMING plugin.info - LOOP key=%s, info=%s" %(key, info));

            if info:
                #for key2, info2 in info.iteritems():
                 #  logger.debug("plugin.info - LOOP key2=%s, info2=%s" %(key2, info2));

                if 'features' in info:
                    #if annotated with EXPORT but its name is not *IonReporter*
                    #watch out: "Export" was changed to "export" recently!
                    ##if ('Export' in info['features']):
                    if ('export' in (feature.lower() for feature in info['features'])):
                        if (info['name'].lower().find('ionreporter') == -1):
                            uploaderNames.append(info['name'])
                    else:
                        pluginNames.append(info['name'])                                       
                else:
                        pluginNames.append(info['name'])                                       
                    
                if (info['name'].lower() == 'IonReporterUploader_V1_0'.lower()):
                    #logger.debug("TIMING pluginCandidate - info[name]= IonReporterUploader_V1.0")
 
                    if 'config' in info:
                        irConfigSelection = json.dumps(info['config'])
                        ##irConfigSelection = info['config']
                        
                        data['irConfigSelection_1'] = irConfigSelection
                        #logger.warn(irConfigSelection)
                    else:
                        data['irConfigSelection_1'] = irConfigSelection
                else:
                    if ('IonReporterUploader'.lower() in info['name'].lower()):
                        #logger.debug("TIMING pluginCandidate - info[name] contains IonReporterUploader")
                        
                        if 'config' in info:
                            irConfigSelection = json.dumps(info['config'])
                            ##irConfigSelection = info['config']
                            
                            data['irConfigSelection'] = irConfigSelection
                            #logger.warn(irConfigSelection)
                        else:
                            data['irConfigSelection'] = irConfigSelection

            #else:               
                #logger.debug("TIMING NO plugin.info for LOOP key=%s," %(key));


    #note that IonReporter plugins are already included in data["IRuploads"]
    data['plugins'] = Plugin.objects.filter(selected=True,active=True).filter(name__in=pluginNames).order_by('name', '-version')
    data['uploaders'] = Plugin.objects.filter(selected=True,active=True).filter(name__in=uploaderNames).order_by('name', '-version')
    
    #to speed things up, skip fetching IRConnfig info since IR configuration is only available during plan creation/edits
    #CANNED DATA FOR TESTING ONLY
    #else:
    #    # FIXME - remove this - fall back to sample fields until this is working
    #    if (irUpload.name.lower() == 'IonReporterUploader_V1_0'.lower()):
    #        irConfigSelection = IRConfig_v1_jsonBlock.sample_relationship_fields()                
    #    else:                
    #        irConfigSelection = IRConfig_jsonBlock.sample_relationship_fields()
                
    
    #20120828-if to query plugin config individually
    ##Query IR plugin for config
    #irConfigSelection = {}
    
    #to speed things up, skip fetching IRConnfig info since IR configuration is only available during plan creation/edits
    #if isForTemplate:
    #    data['irConfigSelection_1'] = irConfigSelection           
    #    data['irConfigSelection'] = irConfigSelection
    #else:        
    #   if IRuploads:
    #      for irUpload in IRuploads:
    #          irInfo = irUpload.info(use_cache=False)
    #        
    #          #logger.warn("irUpload=%s, version=%s" % (irUpload.name, irUpload.version))            
    #          if irInfo and 'config' in irInfo:
    #              irConfigSelection = json.dumps(irInfo['config'])
    #              logger.warn(irConfigSelection)
    #
    #          #CANNED DATA FOR TESTING ONLY
    #          else:
    #              # FIXME - remove this - fall back to sample fields until this is working
    #              if (irUpload.name.lower() == 'IonReporterUploader_V1_0'.lower()):
    #                  irConfigSelection = IRConfig_v1_jsonBlock.sample_relationship_fields()                
    #              else:                
    #                  irConfigSelection = IRConfig_jsonBlock.sample_relationship_fields()
    #            
    #          if (irUpload.name.lower() == 'IonReporterUploader_V1_0'.lower()):
    #              data['irConfigSelection_1'] = irConfigSelection
    #          else:
    #              if (irUpload.name.lower() == 'IonReporterUploader'.lower()):
    #                  data['irConfigSelection'] = irConfigSelection            
    #
    #          irConfigSelection = {}
    #    else:
    #       data['irConfigSelection_1'] = irConfigSelection           
    #       data['irConfigSelection'] = irConfigSelection


    #to allow data entry for multiple non-barcoded samples at the plan wizard
    data['nonBarcodedSamples_irConfig_loopCounter'] = [i+1 for i in range(20)]
    
    return data


@login_required
@transaction.commit_manually
def save_plan_or_template(request):
        
    """saving a new plan template or planned run to db (source: plan template wizard) """
    
    if request.method == 'POST':
 
        statusMessage = ''

        json_data = simplejson.loads(request.raw_post_data)
        logger.debug('views.createplannedexperiment() POST.raw_post_data... simplejson Data: "%s"' % json_data)

        #optional input
        chipTypeValue = ''
        flowValue = None
        libraryValue = ''
        noteValue = ''
        bedFileValue = ''
        regionFileValue = ''
        libraryKitNameValue = ''
        sequenceKitNameValue = ''
        barcodeIdValue = ''
        isPlanGroupValue = 'False'
        isReverseValue = 'False'
        variantFrequencyValue = ''
        selectedPluginsValue = []
        uploaderValues = []
        isReusable = 'False'
        #projectNameList is temp for debug only
        projectNameList = []
        projectObjList = []
        newProjectNameList = []
        
        #for plan template, we want to have an empty list of samples
        #TS-4513 fix: do not add an empty sample name unless there is no user input at all
        sampleList = []
        isFavoriteValue = 'False'
        runModeValue = 'single'
        metaDataValue = ''
        barcodedSamples = ''
        
        templateKitNameValue = ''
        controlSeqKitNameValue = ''
 
        libraryKeyValue = ''
        forward3primeAdapterValue = ''
        reverselibrarykeyValue = ''
        reverse3primeAdapterValue = ''
        pairedEndLibraryAdapterNameValue = '' 
        samplePrepKitNameValue = ''
        
        planDisplayedNameValue = ''
        hasBcSamples = False
        isMultiplePlanRuns = False
        
        sampleValidationErrorMsg = ''
                                
        try:
            submitIntent = json_data['submitIntent']
            logger.debug("views.createplannedexperiment submitIntent=%s" % submitIntent)
                        
            if submitIntent == 'savePlan' or submitIntent == 'updatePlan':
                isReusable = 'False'
            else:
                isReusable = 'True'
        except KeyError:
            pass
                
        #server side validation to avoid things falling through the crack
        msgvalue = 'Run Plan' if not toBoolean(isReusable, False) else 'Template'
        msg = "Error, please enter a %s Name." % (msgvalue)            
        try:
            planDisplayedNameValue = json_data['planDisplayedName'].strip()
            if not planDisplayedNameValue:
                logger.exception(format_exc())
                return HttpResponse(json.dumps({"error":msg}) , mimetype="application/html") 
            else:
                #valid: letters, numbers, spaces, dashes, underscores
                validChars = '^[a-zA-Z0-9-_\.\s\,]+$'
                isOk = bool(re.compile(validChars).match(planDisplayedNameValue))
                if (not isOk):
                    transaction.rollback()
                    return HttpResponse(json.dumps({"error":"Error, %s Name should contain only numbers, letters, spaces, and the following: . - _" %(msgvalue)}) , mimetype="application/html")  
                
        except KeyError:
                logger.exception(format_exc())
                transaction.rollback()
                ##return HttpResponse(json.dumps({"status":"Error, plan name is missing."}) , mimetype="application/html")            
                return HttpResponse(json.dumps({"error":"Error, %s Name is missing." %(msgvalue)}) , mimetype="application/html")            

        try:
            chipTypeValue = json_data['chipType']
        except KeyError:
            pass
        
        try:
            flowValue = json_data['flows']
        except KeyError:
            pass
        
        try:
            libraryValue = json_data['library']
        except KeyError:
            pass
        
        try:
            noteValue = json_data['notes_workaround']

            if noteValue:
                #valid: letters, numbers, spaces, dashes, underscores
                validChars = '^[a-zA-Z0-9-_\.\s\,]+$'
                isOk = bool(re.compile(validChars).match(noteValue))
                if (not isOk):
                    transaction.rollback()
                    return HttpResponse(json.dumps({"error":"Error, %s note should contain only numbers, letters, spaces, and the following: . - _" %(msgvalue)}) , mimetype="application/html")  
            
        except KeyError:
            pass
        
        try:
            bedFileValue = json_data['bedfile']
        except KeyError:
            pass
        
        try:
            regionFileValue = json_data['regionfile']
        except KeyError:
            pass
        
        try:
            libraryKitNameValue = json_data['librarykitname']
        except KeyError:
            pass
        
        try:
            sequenceKitNameValue = json_data['sequencekitname']
        except KeyError:
            pass
      
        try:
            templateKitNameValue = json_data['templatekitname']
        except KeyError:
            pass
        
        try:
            controlSeqKitNameValue = json_data['controlsequence']
        except KeyError:
            pass
                       
        try:
            barcodeIdValue = json_data['barcodeId']
        except KeyError:
            pass

        try:
            variantFrequencyValue = json_data['variantfrequency']
        except KeyError:
            pass
            
        try:
            selectedPluginsValue = json_data['selectedPlugins']
        except KeyError:
            pass
            
        try:
            uploaderValues = json_data['uploaders']
        except KeyError:
            pass
                
        try:
            isFavoriteValue = json_data['isFavorite']
        except KeyError:
            pass
               
        #20120712-wip
        #try:
        #    metaDataRaw = json_data['metadata']
        #    logger.debug("views.createplannedexperiment metaDataRaw=%s " % (metaDataRaw))
        #    if metaDataRaw:
        #        metaDataValue = simplejson.dumps(metaDataRaw)
        #    
        #    logger.debug("views.createplannedexperiment metaDataValue=%s " % (metaDataValue))
        #except KeyError:
        #    pass
        #except:
        #    logger.exception(format_exc())            
        #    pass


        try:
            projectIdAndNameList = json_data['projects']
                        
            logger.debug("views.createplannedexperiment projectIdAndNameList=%s " % (projectIdAndNameList))

            #it is a string if 1 entry; list otherwise
            if (isinstance(projectIdAndNameList, basestring)):
                projectIdAndNameTokens = projectIdAndNameList.split('|')
                
                try:
                    projectId = projectIdAndNameTokens[0]
                    
                    projectObj = Project.objects.get(id = int(projectId))
                    projectNameList.append(projectObj.name)
                    logger.debug("views.createplannedexperiment STRING ADDING project=%s to projectObjectList" % (projectObj.name))
                    
                    projectObjList.append(projectObj)
                except Project.DoesNotExist:
                    #use case: if someone happens to have deleted the project 
                    logger.warn("views.createplannedexperiment projectId=%d is no longer in db" % (projectId))
                                
                logger.debug("views.createplannedexperiment projectNameList=%s " % (projectNameList))                  
            else:
                #if multiple projects, it is contained in a List

                for projectIdAndName in projectIdAndNameList:
                    projectIdAndNameTokens = projectIdAndName.split('|')
                
                    try:
                        projectId = projectIdAndNameTokens[0]
                                
                        projectObj = Project.objects.get(id = int(projectId))
                        projectNameList.append(projectObj.name)
                        logger.debug("views.createplannedexperiment LIST ADDING project=%s to projectObjectList" % (projectObj.name))
                    
                        projectObjList.append(projectObj)
                    except Project.DoesNotExist:
                        #use case: if someone happens to have deleted the project 
                        logger.warn("views.createplannedexperiment projectId=%d is no longer in db" % (projectId))
                        continue
                                
            logger.debug("views.createplannedexperiment projectNameList=%s " % (projectNameList))  
                            
        except KeyError:
            pass
        except:                    
            logger.exception(format_exc())
            return HttpResponse(json.dumps({"error":"Internal error processing projects while trying to save the plan."}) , mimetype="application/json")            
                  
        try:
            projectNames = json_data['newProjects']

            #newProjects with no user input will be posted as "newProjects":""
            if projectNames and len(projectNames.strip()) > 0:
                newProjectNameTokens = projectNames.strip().split(',')
                for newProjectName in newProjectNameTokens:
                    if len(newProjectName.strip()) > 0:
                        newProjectNameList.append(newProjectName.strip().replace(' ','_'))
                        
                logger.debug("views.createplannedexperiment newProjectNameList=%s " % (newProjectNameList))  
            else:
                newProjectNameList = None
                logger.debug("views.createplannedexperiment newProjectNameList is NONE!!! ")  

        except KeyError:
            pass

        if newProjectNameList:
            #use case: user enters duplicate project names as "add new projects"
            uniqueNewProjectNameList = []
            for newProjectName in newProjectNameList:
                if newProjectName.strip() not in uniqueNewProjectNameList:
                    uniqueNewProjectNameList.append(newProjectName.strip())
                    
                    projectNameList.append(newProjectName.strip())
                else:
                    logger.info("views.createplannedexperiment SKIP DUPLICATE project=%s" % (newProjectName.strip()))                    

            for newProjectName in newProjectNameList:
                try:
                    #use cases:              
                    #project is already in the db and user overlooks it in the list
                    #user has already select this project in the list but re-enters the project name as add new project
                    existingProject = Project.objects.filter(name = newProjectName.strip())[0]
                    projectObjList.append(existingProject)
                
                    logger.debug("views.createplannedexperiment got project=%s" % (newProjectName.strip()))
                except IndexError:
                    #use case: project is not yet in db 
                    user = models.User.objects.get(username = request.user) 
                    projectObj = Project.objects.create(name = newProjectName.strip(), creator = user)
                               
                    projectObjList.append(projectObj)
                                                
                    logger.debug("views.createplannedexperiment created project=%s" % (newProjectName.strip()))                                
                except:
                    logger.exception(format_exc())  
                    transaction.rollback()              
                    return HttpResponse(json.dumps({"error":"Internal error processing manually added projects while trying to save the plan."}) , mimetype="application/json")            
            
            
            
        try:
            barcodedSamples = json_data['bcSamples_workaround']
                                
            logger.debug("views.createplannedexperiment barcodedSamples=%s " % (barcodedSamples))
            

            bcDictionary = {}
            barcodeId= ""
            for token in barcodedSamples.split(","):
                if ((token.find("bcKey|")) == 0):
                    barcodeId = token[6:]
                else:
                    sample = token                               
                    if (barcodeId):
                        if (sample.strip()):
                            hasBcSamples = True

                        #valid: letters, numbers, spaces, dashes, underscores
                        validChars = '^[a-zA-Z0-9-_\.\s\,]+$'
                        isOk = bool(re.compile(validChars).match(sample))
                        if (not isOk):
                            sampleValidationErrorMsg + sample + ', '  
                            
                        value = {'sample': sample}
                        entry = {barcodeId: value}
                        bcDictionary.update(entry)
                        barcodeId = ""
                    
            logger.debug("views.createplannedexperiment bcDictionary=%s;" % (bcDictionary)) 

            #fix: if user changes the plan from barcoded to non-barcded after s/he has entered sample info for the plan, barcodedSamples won't be empty!!
            if barcodeIdValue and barcodedSamples:   
                barcodedSamples = simplejson.dumps(bcDictionary)
                #avoid erroneous alert for sampleList should not be null
                sampleList = ['']
            else:
                barcodedSamples = ''
                
            logger.debug("views.createplannedexperiment after simplejson.dumps... barcodedSamples=%s;" % (barcodedSamples))              

        except KeyError:
            pass
        except:
            logger.exception(format_exc())
            transaction.rollback()
            return HttpResponse(json.dumps({"error":"Internal error processing barcoded samples while trying to save the plan."}) , mimetype="application/json")            

        
        samples = json_data.get('samples_workaround','')
        
        # save IonReporter configuration     
        selectedPlugins = json_data.get('selectedPlugins',{})
        IRconfigList = json_data.get('irConfigList', [])        
        for uploader in selectedPlugins.get('planuploaders',[]):
            if 'ionreporteruploader' in uploader['name'].lower() and uploader['name'] != 'IonReporterUploader_V1_0':                
                samples = json_data.get('sample_irConfig','')
                samples = ','.join(samples)
                
                #generate UUID for unique setIds
                id_uuid = {}
                setids = [ir['setid'] for ir in IRconfigList]
                for setid in set(setids):
                    id_uuid[setid] = str(uuid.uuid4())       
                for ir_config in IRconfigList:              
                    ir_config['setid'] += '__' + id_uuid[ir_config['setid']]    
        
        # Non-barcoded samples        
        if not barcodeIdValue:                  
          for sample in samples.split(','):
              if sample.strip():
                  #valid: letters, numbers, spaces, dashes, underscores
                   validChars = '^[a-zA-Z0-9-_\.\s\,]+$'
                   isOk = bool(re.compile(validChars).match(sample))
                   if not isOk:
                       sampleValidationErrorMsg += sample + ', '
                   else:                        
                       sampleList.append(sample);
                    
        logger.debug("views.createplannedexperiment sampleList=%s " % (sampleList))         
        
        if sampleList == None or len(sampleList) == 0:
            if  toBoolean(isReusable, False):
                sampleList = ['']
            else:
                if not barcodeIdValue:
                    transaction.rollback()
                    return HttpResponse(json.dumps({"error":"Error, please enter a sample name for the run plan."}) , mimetype="application/html")
        else:
            if (not toBoolean(isReusable, False)):
                if len(sampleList) > 1:
                    isMultiplePlanRuns = True
            
        try:
            runModeValue = json_data['runMode']
        except KeyError:
            pass
                 
        if runModeValue == 'pe' and toBoolean(isReusable, False) == False:
            isPlanGroupValue = 'True'


        if runModeValue == 'single':
            try:
                libraryKeyValue = json_data['libraryKey']
                forward3primeAdapterValue = json_data['forward3primeAdapter']
            except KeyError:
                pass
        else:
            if runModeValue == 'pe':
                try:
                    libraryKeyValue = json_data['peForwardLibraryKey']
                    forward3primeAdapterValue = json_data['peForward3primeAdapter']
                    reverselibrarykeyValue = json_data['reverselibrarykey']
                    reverse3primeAdapterValue = json_data['reverse3primeAdapter']
                    pairedEndLibraryAdapterNameValue = json_data['pairedEndLibraryAdapterName']
                except KeyError:
                    pass 
        
        try:
            samplePrepKitNameValue = json_data['samplePrepKitName']
        except KeyError:
            pass
        
        if sampleValidationErrorMsg:
            message = "Error, sample name should contain only numbers, letters, spaces, and the following: . - _"
            message = message + ' Please fix: ' + sampleValidationErrorMsg
            transaction.rollback()
            return HttpResponse(json.dumps({"error": message}) , mimetype="application/html")  
        
        logger.debug("views.createplannedexperiment user=%s " % (request.user))

        try:
            if len(sampleList) > 0:
                for i,sample in enumerate(sampleList): 

                    logger.debug("...LOOP... views.createplannedexperiment SAMPLE=%s; isSystem=%s; isReusable=%s; isPlanGroup=%s " % (sample.strip(), json_data["isSystem"], isReusable, isPlanGroupValue))
                    
                    # add IonReporter config values for each sample
                    if len(IRconfigList) > 0:                        
                        for uploader in selectedPluginsValue['planuploaders']:
                            if 'ionreporter' in uploader['name'].lower():
                                if len(IRconfigList) > 1 and not (barcodeIdValue and barcodedSamples):
                                    uploader['userInput'] = [IRconfigList[i]]
                                else:
                                    uploader['userInput'] = IRconfigList
                    
                    inputPlanDisplayedName = planDisplayedNameValue
                    if isMultiplePlanRuns:
                        inputPlanDisplayedName = planDisplayedNameValue + '_' + sample.strip()
                        
                    kwargs = {
                        'planDisplayedName': inputPlanDisplayedName,  ##'0612 test15', ##request.POST.get('planDisplayedName'),
                        "planName": inputPlanDisplayedName.replace(' ', '_'),  ##request.POST.get['planDisplayedName'].replace(' ', '_'),
                        'chipType': chipTypeValue,
                        'usePreBeadfind': toBoolean(json_data['usePreBeadfind'], False),
                        'usePostBeadfind': toBoolean(json_data['usePostBeadfind'], False),
                        'flows': flowValue,
                        'autoAnalyze': True,
                        'preAnalysis': True,
                        'runType': json_data['runType'],
                        'library': libraryValue,
                        'notes' : noteValue,
                        'bedfile': bedFileValue,
                        'regionfile': regionFileValue,
                        'variantfrequency': variantFrequencyValue,
                        'librarykitname': libraryKitNameValue,
                        'sequencekitname': sequenceKitNameValue,
                        'templatingKitName' : templateKitNameValue, 
                        'controlSequencekitname' : controlSeqKitNameValue,
                        'barcodeId': barcodeIdValue,
                        'runMode': runModeValue,
                        'isSystem': toBoolean(json_data['isSystem'], False),
                        'isReusable': toBoolean(isReusable, False),
                        'isPlanGroup': toBoolean(isPlanGroupValue, False),
                        'isReverseRun': toBoolean(isReverseValue, False),
                        'sampleDisplayedName': sample.strip(),
                        "sample": sample.strip().replace(' ', '_'),                        
                        'username': request.user,
                        'isFavorite': toBoolean(isFavoriteValue, False),
                         #'metaData': metaDataValue,
                        'barcodedSamples': barcodedSamples,                                
                        'libraryKey': libraryKeyValue,
                        'forward3primeadapter': forward3primeAdapterValue,
                        'reverselibrarykey': reverselibrarykeyValue,
                        'reverse3primeadapter': reverse3primeAdapterValue,
                        'pairedEndLibraryAdapterName': pairedEndLibraryAdapterNameValue, 
                        'samplePrepKitName': samplePrepKitNameValue,
                        'selectedPlugins' : selectedPluginsValue 
                        
                        ##
                        ##20120709  File "/usr/lib/pymodules/python2.6/django/db/models/base.py", line 367, in __init__
                        ##raise TypeError("'%s' is an invalid keyword argument for this function" % kwargs.keys()[0])
                        ##TypeError: 'projects' is an invalid keyword argument for this function
                        ###'projects': projectNameList

                    }
                    planTemplate = PlannedExperiment(**kwargs)
                    planTemplate.save()
    
                    logger.debug("views.createplannedexperiment after save id=%d " % planTemplate.id)
        
                    qcTypes = QCType.objects.all()
                    for qcType in qcTypes:
                    
                        logger.debug("views.createplannedexperiment qcType.id=%d " % qcType.id)
                
                        try:
                            if json_data[qcType.qcName]:
                                kwargs = {
                                    'plannedExperiment': planTemplate,
                                    'qcType': qcType,
                                    'threshold': json_data[qcType.qcName], ##'50', ##request.POST.get('qcValues|37'),
                                }
                    
                                qcValue = PlannedExperimentQC(**kwargs)
                                qcValue.save()
                        
                                #logger.debug("views.createplannedexperiment qcType.qcName=%s; qcValue.id=%d " % (qcType.qcName, qcValue.id))

                                
                        except KeyError:
                            logger.debug("createplannedexperiment KeyError for qcType=%s" % qcType.qcName)
                                
                            # this is a workaround for the wizard so that even if only some of the QC thresholds have values, the UI will still be complete

                            kwargs = {
                                'plannedExperiment': planTemplate,
                                'qcType': qcType,
                                'threshold': qcType.minThreshold,
                                }
                    
                            qcValue = PlannedExperimentQC(**kwargs)
                            qcValue.save()
                        
                            logger.debug("views.createplannedexperiment MIN THRESHOLD qcType.qcName=%s; qcValue.id=%d " % (qcType.qcName, qcValue.id))
                                                 

                    for projectObj in projectObjList:
                        logger.debug("views.createplannedexperiment GOING TO associate project.name=%s; " % (projectObj.name))                            
            
                        planTemplate.projects.add(projectObj)                        
                        logger.debug("views.createplannedexperiment associated project.name=%s; " % (projectObj.name))                            
            
                    logger.info("views.createplannedexperiment GOING TO RETURN planTemplate.id=%d " % planTemplate.id)

                    statusMessage = "plan template created successfully"
            else:           
                logger.info("views.createplannedexperiment sampleList should not be null ")
                transaction.rollback()
                return HttpResponse(json.dumps({"error":"Internal error while trying to save the plan."}) , mimetype="application/json")
                
        except:
            transaction.rollback()
                    
            logger.exception(format_exc())
            return HttpResponse(json.dumps({"error":"Internal error while trying to save the plan."}) , mimetype="application/json")            
        else:
            transaction.commit()
                    
            return HttpResponse(json.dumps({"status": statusMessage}) , mimetype="application/json")
            
    else:
        logger.exception(format_exc())
        return HttpResponse(json.dumps({"error":"Error, unsupported http request for saving the plan."}) , mimetype="application/json")


@login_required
@transaction.commit_manually
def save_edited_plan_or_template(request, planOid): 
            
    """saving an edited plan template or planned run to db (source: plan template wizard) """           
    """20120625Note: we don't have requirement to edit a planned run yet. """
    """However, editing a planned run from having 1 sample to 2 samples will result in one edited planned run and one new planned run """

   
    if request.method == 'POST':
    
        statusMessage = ''

        json_data = simplejson.loads(request.raw_post_data)
        logger.debug('views.editplannedexperiment POST.raw_post_data... simplejson Data: "%s"' % json_data)

        #optional input
        chipTypeValue = ''
        flowValue = None
        libraryValue = ''
        noteValue = ''
        bedFileValue = ''
        regionFileValue = ''
        libraryKitNameValue = ''
        sequenceKitNameValue = ''
        barcodeIdValue = ''
        isPlanGroupValue = 'False'
        variantFrequencyValue = ''
        selectedPluginsValue = ['']
        uploaderValues = ['']
        isReusable = 'False'
 
        #for plan template, we want to have an empty list of samples
        #TS-4513 fix: do not add an empty sample name unless there is no user input at all
        sampleList = []

        isFavoriteValue = 'False'
        metaDataValue = ''
        #projectNameList is temp for debug only
        projectNameList = []
        projectObjList = []
        newProjectNameList = []
        
        isFavoriteValue = 'False'
        runModeValue = 'single'
        barcodedSamples = ''

        templateKitNameValue = ''
        controlSeqKitNameValue = ''
 
        libraryKeyValue = ''
        forward3primeAdapterValue = ''
        reverselibrarykeyValue = ''
        reverse3primeAdapterValue = ''
        pairedEndLibraryAdapterNameValue = ''
        samplePrepKitNameValue = ''
        
        planDisplayedNameValue = ''
        hasBcSamples = False
        isMultiplePlanRuns = False

        sampleValidationErrorMsg = ''
                
        try:
            submitIntent = json_data['submitIntent']
            logger.debug("views.editplannedexperiment submitIntent=%s" % submitIntent)
            
            if submitIntent == 'savePlan' or submitIntent == 'updatePlan':
                isReusable = 'False'
            else:
                isReusable = 'True'
        except KeyError:
            pass
                
        #server side validation to avoid things falling through the crack
        msgvalue = 'Run Plan' if not toBoolean(isReusable, False) else 'Template'
        msg = "Error, please enter a %s Name." % (msgvalue)            
        try:
            planDisplayedNameValue = json_data['planDisplayedName'].strip()
            if not planDisplayedNameValue:
                logger.exception(format_exc())
                ##return HttpResponse(json.dumps({"status":"Error, please enter a plan name."}) , mimetype="application/html")
                return HttpResponse(json.dumps({"error":msg}) , mimetype="application/html")            
            else:
                #valid: letters, numbers, spaces, dashes, underscores
                validChars = '^[a-zA-Z0-9-_\.\s\,]+$'
                isOk = bool(re.compile(validChars).match(planDisplayedNameValue))
                if (not isOk):
                    transaction.rollback()
                    return HttpResponse(json.dumps({"error":"Error, %s Name should contain only numbers, letters, spaces, and the following: . - _" %(msgvalue)}) , mimetype="application/html")  
                
        except KeyError:
                logger.exception(format_exc())
                transaction.rollback()
                ##return HttpResponse(json.dumps({"status":"Error, plan name is missing."}) , mimetype="application/html")            
                return HttpResponse(json.dumps({"error":"Error, %s Name is missing." %(msgvalue)}) , mimetype="application/html")   
                                                            
        try:
            chipTypeValue = json_data['chipType']
        except KeyError:
            pass
        
        try:
            flowValue = json_data['flows']
        except KeyError:
            pass
        try:
            libraryValue = json_data['library']
        except KeyError:
            pass
        
        try:
            noteValue = json_data['notes_workaround']
            
            if noteValue:
                #valid: letters, numbers, spaces, dashes, underscores
                validChars = '^[a-zA-Z0-9-_\.\s\,]+$'
                isOk = bool(re.compile(validChars).match(noteValue))
                if (not isOk):
                    transaction.rollback()
                    return HttpResponse(json.dumps({"error":"Error, %s note should contain only numbers, letters, spaces, and the following: . - _" %(msgvalue)}) , mimetype="application/html")  

        except KeyError:
            pass
        
        try:
            bedFileValue = json_data['bedfile']
        except KeyError:
            pass
        
        try:
            regionFileValue = json_data['regionfile']
        except KeyError:
            pass
        
        try:
            libraryKitNameValue = json_data['librarykitname']
        except KeyError:
            pass
        
        try:
            sequenceKitNameValue = json_data['sequencekitname']
        except KeyError:
            pass
     
        try:
            templateKitNameValue = json_data['templatekitname']
        except KeyError:
            pass
        
        try:
            controlSeqKitNameValue = json_data['controlsequence']
        except KeyError:
            pass
                       
        try:
            barcodeIdValue = json_data['barcodeId']
        except KeyError:
            pass
        
        try:
            variantFrequencyValue = json_data['variantfrequency']
        except KeyError:
            pass
            
        try:
            selectedPluginsValue = json_data['selectedPlugins']
        except KeyError:
            pass
            
        try:
            uploaderValues = json_data['uploaders']
        except KeyError:
            pass
                
        try:
            isFavoriteValue = json_data['isFavorite']
        except KeyError:
            pass
            
        #try:
        #    metaDataValue = json_data['metadata']
        #except KeyError:
        #    pass
                        
        try:
            runModeValue = json_data['runMode']
        except KeyError:
            pass
                 
        if runModeValue == 'pe' and toBoolean(isReusable, False) == False:
            isPlanGroupValue = 'True'
   
        if runModeValue == 'single':
            try:
                libraryKeyValue = json_data['libraryKey']
                forward3primeAdapterValue = json_data['forward3primeAdapter']
            except KeyError:
                pass
        else:
            if runModeValue == 'pe':
                try:
                    libraryKeyValue = json_data['peForwardLibraryKey']
                    forward3primeAdapterValue = json_data['peForward3primeAdapter']
                    reverselibrarykeyValue = json_data['reverselibrarykey']
                    reverse3primeAdapterValue = json_data['reverse3primeAdapter']
                    pairedEndLibraryAdapterNameValue = json_data['pairedEndLibraryAdapterName']
                except KeyError:
                    pass
                
        try:
            samplePrepKitNameValue = json_data['samplePrepKitName']
        except KeyError:
            pass
        
        try:
            projectIdAndNameList = json_data['projects']
                        
            logger.debug("views.editplannedexperiment projectIdAndNameList=%s " % (projectIdAndNameList))

            #it is a string if 1 entry; list otherwise
            if (isinstance(projectIdAndNameList, basestring)):
                projectIdAndNameTokens = projectIdAndNameList.split('|')
                
                try:
                    projectId = projectIdAndNameTokens[0]
                                
                    projectObj = Project.objects.get(id = int(projectId))
                    projectNameList.append(projectObj.name)
                    logger.debug("views.editplannedexperiment STRING ADDING project=%s to projectObjectList" % (projectObj.name))
                    
                    projectObjList.append(projectObj)
                except Project.DoesNotExist:
                    #use case: if someone happens to have deleted the project 
                    logger.warn("views.editplannedexperiment STRING views.editplannedexperiment projectId=%d is no longer in db" % (projectId))
                                
                logger.debug("views.editplannedexperiment projectNameList=%s " % (projectNameList))                  
            else:
                #if there multiple projects selected, they are contained in a list
                for projectIdAndName in projectIdAndNameList:
                    logger.debug("views.editplannedexperiment projectIdAndName=%s " % (projectIdAndName))
                    projectIdAndNameTokens = projectIdAndName.split('|')
                
                    try:
                        projectId = projectIdAndNameTokens[0]
                        logger.debug("views.editplannedexperiment projectId as String=%s; projectId as int=%d; " % (projectId, int(projectId)))
                                
                        projectObj = Project.objects.get(id = int(projectId))
                        projectNameList.append(projectObj.name)
                        logger.debug("views.editplannedexperiment LIST ADDING project=%s to projectObjectList" % (projectObj.name))
                    
                        projectObjList.append(projectObj)
                    except Project.DoesNotExist:
                        #use case: if someone happens to have deleted the project 
                        logger.warn("views.editplannedexperiment views.editplannedexperiment projectId=%d is no longer in db" % (projectId))
                        continue
                                
                logger.debug("views.editplannedexperiment projectNameList=%s " % (projectNameList))  
                            
        except KeyError:
            pass
        except:                    
            logger.exception(format_exc())
            transaction.rollback()
            return HttpResponse(json.dumps({"status":"error, exception at editplannedexperiment when processing projects"}) , mimetype="application/json")            
                  
        try:
            projectNames = json_data['newProjects']

            if projectNames and len(projectNames.strip()) > 0:
                newProjectNameTokens = projectNames.strip().split(',')
                for newProjectName in newProjectNameTokens:
                    if len(newProjectName.strip()) > 0:
                        newProjectNameList.append(newProjectName.strip().replace(' ','_'))
                logger.debug("views.editplannedexperiment newProjectNameList=%s " % (newProjectNameList))  
            else:
                newProjectNameList = None
                logger.debug("views.editplannedexperiment newProjectNameList is NONE!!!" )
                  
        except KeyError:
            pass

        if newProjectNameList:
            #use case: user enters duplicate project names as "add new projects"
            uniqueNewProjectNameList = []
            for newProjectName in newProjectNameList:
                if newProjectName.strip() not in uniqueNewProjectNameList:
                    uniqueNewProjectNameList.append(newProjectName.strip())
                    
                    projectNameList.append(newProjectName.strip())
                else:
                    logger.info("views.editplannedexperiment views.editplannedexperiment SKIP DUPLICATE project=%s" % (newProjectName.strip()))                    

            for newProjectName in newProjectNameList:
                try:
                    #use cases:              
                    #project is already in the db and user overlooks it in the list
                    #user has already select this project in the list but re-enters the project name as add new project
                    existingProject = Project.objects.filter(name = newProjectName.strip())[0]
                    projectObjList.append(existingProject)
                
                    logger.debug("views.editplannedexperiment views.editplannedexperiment got project=%s" % (newProjectName.strip()))

                except IndexError:
                    #use case: project is not yet in db 
                    user = models.User.objects.get(username = request.user) 
                    projectObj = Project.objects.create(name = newProjectName.strip(), creator = user)
                               
                    projectObjList.append(projectObj)
                                                
                    logger.debug("views.editplannedexperiment  views.editplannedexperiment created project=%s" % (newProjectName.strip()))
                except:
                    logger.exception(format_exc())                
                    return HttpResponse(json.dumps({"status":"error, exception at editplannedexperiment when processing manually entered projects"}) , mimetype="application/json")            
                        
        try:
            barcodedSamples = json_data['bcSamples_workaround']
                                
            logger.debug("views.createplannedexperiment barcodedSamples=%s " % (barcodedSamples))
            

            bcDictionary = {}
            barcodeId= ""
            for token in barcodedSamples.split(","):
                if ((token.find("bcKey|")) == 0):
                    barcodeId = token[6:]
                else:
                    sample = token                               
                    if (barcodeId):
                        if (sample.strip()):
                            hasBcSamples = True

                        #valid: letters, numbers, spaces, dashes, underscores
                        validChars = '^[a-zA-Z0-9-_\.\s\,]+$'
                        isOk = bool(re.compile(validChars).match(sample))
                        if (not isOk):
                            sampleValidationErrorMsg + sample + ', '  
                                                    
                        value = {'sample': sample}
                        entry = {barcodeId: value}
                        bcDictionary.update(entry)
                        barcodeId = ""
                    
            logger.debug("views.editplannedexperiment bcDictionary=%s;" % (bcDictionary)) 
             
            #fix: if user changes the plan from barcoded to non-barcded after s/he has entered sample info for the plan, barcodedSamples won't be empty!!
            if barcodeIdValue and barcodedSamples:   
                barcodedSamples = simplejson.dumps(bcDictionary)
                #avoid erroneous alert for sampleList should not be null
                sampleList = ['']
            else:
                barcodedSamples = ''

            logger.debug("views.editplannedexperiment after simplejson.dumps... barcodedSamples=%s;" % (barcodedSamples))              

        except KeyError:
            pass
        except:
            logger.exception(format_exc())
            transaction.rollback()
            return HttpResponse(json.dumps({"error":"Internal error processing barcoded samples while trying to update the plan."}) , mimetype="application/json")            

        
        samples = json_data.get('samples_workaround','')
        
        # save IonReporter configuration     
        selectedPlugins = json_data.get('selectedPlugins',{})
        IRconfigList = json_data.get('irConfigList', [])        
        for uploader in selectedPlugins.get('planuploaders',[]):
            if 'ionreporteruploader' in uploader['name'].lower() and uploader['name'] != 'IonReporterUploader_V1_0':
                samples = json_data.get('sample_irConfig','')
                samples = ','.join(samples)
            
                #generate UUID for unique setIds
                id_uuid = {}
                setids = [ir['setid'] for ir in IRconfigList]
                for setid in set(setids):
                    id_uuid[setid] = str(uuid.uuid4())       
                for ir_config in IRconfigList:              
                    ir_config['setid'] += '__' + id_uuid[ir_config['setid']]    
        
        # Non-barcoded samples        
        if not barcodeIdValue:                  
          for sample in samples.split(','):
              if sample.strip():
                  #valid: letters, numbers, spaces, dashes, underscores
                   validChars = '^[a-zA-Z0-9-_\.\s\,]+$'
                   isOk = bool(re.compile(validChars).match(sample))
                   if not isOk:
                       sampleValidationErrorMsg += sample + ', '
                   else:                        
                       sampleList.append(sample);
                    
        logger.debug("views.editplannedexperiment sampleList=%s " % (sampleList)) 
        
        
        if sampleList == None or len(sampleList) == 0:
            if  toBoolean(isReusable, False):
                sampleList = ['']
            else:
                if (not barcodeIdValue):
                    transaction.rollback()
                    return HttpResponse(json.dumps({"error":"Error, please enter a sample name for the run plan."}) , mimetype="application/html")      
        else:
            if (not toBoolean(isReusable, False)):
                if len(sampleList) > 1:
                    isMultiplePlanRuns = True

        if sampleValidationErrorMsg:
            message = "Error, sample name should contain only numbers, letters, spaces, and the following: . - _"
            message = message + ' Please fix: ' + sampleValidationErrorMsg
            transaction.rollback()
            return HttpResponse(json.dumps({"error": message}) , mimetype="application/html")  
            
        logger.debug("views.editplannedexperiment user=%s " % (request.user))

        if len(sampleList) > 0:
            isFirst = True
            
            for i,sample in enumerate(sampleList): 

                logger.debug("...LOOP... views.editplannedexperiment SAMPLE=%s; isSystem=%s; isReusable=%s; isPlanGroup=%s " % (sample.strip(), json_data["isSystem"], isReusable, isPlanGroupValue))
                
                # add IonReporter config values for each sample
                if len(IRconfigList) > 0:                        
                    for uploader in selectedPluginsValue['planuploaders']:
                        if 'ionreporter' in uploader['name'].lower():
                            if len(IRconfigList) > 1 and not (barcodeIdValue and barcodedSamples):
                                uploader['userInput'] = [IRconfigList[i]]
                            else:
                                uploader['userInput'] = IRconfigList
                try:
                    inputPlanDisplayedName = planDisplayedNameValue
                    if isMultiplePlanRuns:
                        inputPlanDisplayedName = planDisplayedNameValue + '_' + sample.strip()

                    #if we're changing a plan from having 1 sample to say 2 samples, we need to UPDATE 1 plan and CREATE 1 plan!!
                    if isFirst:
                        try:
                            planTemplate = PlannedExperiment.objects.get(pk=planOid)
                            planTemplate.planDisplayedName = inputPlanDisplayedName
                            planTemplate.planName = inputPlanDisplayedName.replace(' ', '_')
                            planTemplate.chipType = chipTypeValue
                            planTemplate.usePreBeadfind = toBoolean(json_data['usePreBeadfind'], False)
                            planTemplate.usePostBeadfind = toBoolean(json_data['usePostBeadfind'], False)
                            planTemplate.flows = flowValue
                            planTemplate.autoAnalyze = True
                            planTemplate.preAnalyze = True
                            planTemplate.runType = json_data['runType']
                            planTemplate.library = libraryValue
                            planTemplate.notes = noteValue
                            planTemplate.bedfile = bedFileValue
                            planTemplate.regionfile = regionFileValue
                            planTemplate.variantfrequency = variantFrequencyValue
                            planTemplate.librarykitname = libraryKitNameValue
                            planTemplate.sequencekitname = sequenceKitNameValue
                            planTemplate.barcodeId = barcodeIdValue
                            planTemplate.templatingKitName = templateKitNameValue 
                            planTemplate.controlSequencekitname = controlSeqKitNameValue
                            planTemplate.runMode = runModeValue
                            planTemplate.isSystem = toBoolean(json_data['isSystem'], False)
                            planTemplate.isReusable = toBoolean(isReusable, False)
                            planTemplate.isPlanGroup = toBoolean(isPlanGroupValue, False)
                            planTemplate.sampleDisplayedName = sample.strip()
                            planTemplate.sample = sample.strip().replace(' ', '_')
                            
                            planTemplate.username = request.user.username
                            planTemplate.isFavorite = toBoolean(isFavoriteValue, False)
                            #planTemplate.metaData = metaDataValue
                            planTemplate.barcodedSamples = barcodedSamples                               
                            planTemplate.libraryKey = libraryKeyValue
                            planTemplate.forward3primeadapter = forward3primeAdapterValue
                            planTemplate.reverselibrarykey = reverselibrarykeyValue
                            planTemplate.reverse3primeadapter = reverse3primeAdapterValue
                            planTemplate.pairedEndLibraryAdapterName = pairedEndLibraryAdapterNameValue 
                            planTemplate.samplePrepKitName = samplePrepKitNameValue
                            planTemplate.selectedPlugins = selectedPluginsValue
                            
                            planTemplate.save()
                            isFirst = False
                            
                            logger.debug("views.createplannedexperiment after save UPDATE id=%d " % planTemplate.id)
                            
                        except PlannedExperiment.DoesNotExist:                           
                            transaction.rollback()
                            
                            logger.exception(format_exc())
                            return HttpResponse(json.dumps({"error":"Internal error while trying to update the plan."}) , mimetype="application/json")
                        except:
                            transaction.rollback()
                                                        
                            logger.exception(format_exc())
                            return HttpResponse(json.dumps({"error":"Internal error while trying to update the plan."}) , mimetype="application/json")            
                                     
                            
                        #user could have updated the QC types!!!
                        qcTypes = QCType.objects.all()
                        for qcType in qcTypes:
                
                            try:
                                if json_data[qcType.qcName]:

                                    plannedExpQcs = PlannedExperimentQC.objects.filter(plannedExperiment = planTemplate.id, qcType = qcType.id)
                                        
                                    #there should be zero or 1 occurrence
                                    for plannedExpQc in plannedExpQcs:
                                        plannedExpQc.threshold = json_data[qcType.qcName]
                                                         
                                        plannedExpQc.save()
                                        
                                    if not plannedExpQcs:
                                        kwargs = {
                                                'plannedExperiment': planTemplate,
                                                'qcType': qcType,
                                                'threshold': json_data[qcType.qcName], ##'50', ##request.POST.get('qcValues|37'),
                                                }
                    
                                        qcValue = PlannedExperimentQC(**kwargs)
                                        qcValue.save()
                        
                                    
                            except KeyError:
                                # this is a workaround for the wizard so that even if only some of the QC thresholds have values, the UI will still be complete
                                plannedExpQcs = PlannedExperimentQC.objects.filter(plannedExperiment = planTemplate.id, qcType = qcType.id)
                                        
                                #there should be zero or 1 occurrence
                                for plannedExpQc in plannedExpQcs:
                                    plannedExpQc.threshold = qcType.minThreshold
                                    plannedExpQc.save()
                                        
                                if not plannedExpQcs:
                                    kwargs = {
                                        'plannedExperiment': planTemplate,
                                        'qcType': qcType,
                                        'threshold': qcType.minThreshold,
                                        }
                                    try:
                                        qcValue = PlannedExperimentQC(**kwargs)
                                        qcValue.save()
                                                                                                
                                    except:
                                        transaction.rollback()
                                                                    
                                        logger.exception(format_exc())
                                        return HttpResponse(json.dumps({"error":"Internal error processing monitor info while trying to update the plan."}) , mimetype="application/json")            
    
                        currentProjects = planTemplate.projects.all();
                        for currentProject in currentProjects:
                            if currentProject.name not in projectNameList:
                                logger.debug("views.editplannedexperiment GOING TO REMOVE association to project.name=%s; " % (currentProject.name))
                                planTemplate.projects.remove(currentProject)

                        #use case: user could have removed a plan/project association                            
                        for projectObj in projectObjList:
                            logger.debug("views.editplannedexperiment GOING TO associate project.name=%s; " % (projectObj.name))                            
            
                            planTemplate.projects.add(projectObj)                        
                            logger.debug("views.editplannedexperiment associated project.name=%s; " % (projectObj.name))                            

                        logger.debug("views.editplannedexperiment GOING TO RETURN planTemplate.id=%d " % planTemplate.id)

                        statusMessage = "plan template updated successfully"
                 
                    else:                    
                        kwargs = {
                              'planDisplayedName': inputPlanDisplayedName,  ##'0612 test15', ##request.POST.get('planDisplayedName'),
                              "planName": inputPlanDisplayedName.replace(' ', '_'),  ##request.POST.get['planDisplayedName'].replace(' ', '_'),
                              'chipType': chipTypeValue,
                              'usePreBeadfind': toBoolean(json_data['usePreBeadfind'], False),
                              'usePostBeadfind': toBoolean(json_data['usePostBeadfind'], False),
                              'flows': flowValue,
                              'autoAnalyze': True,
                              'preAnalysis': True,
                              'runType': json_data['runType'],
                              'library': libraryValue,
                              'notes' : noteValue,
                              'bedfile': bedFileValue,
                              'regionfile': regionFileValue,
                              'variantfrequency': variantFrequencyValue,
                              'librarykitname': libraryKitNameValue,
                              'sequencekitname': sequenceKitNameValue,
                              'barcodeId': barcodeIdValue,
                               'templatingKitName' : templateKitNameValue, 
                               'controlSequencekitname' : controlSeqKitNameValue,
                              'runMode': json_data['runMode'],
                              'isSystem': toBoolean(json_data['isSystem'], False),
                              'isReusable': toBoolean(isReusable, False),
                              'isPlanGroup': toBoolean(isPlanGroupValue, False),
                              'sampleDisplayedName': sample.strip(),
                              "sample": sample.strip().replace(' ', '_'),
                              'username': request.user,
                              'isFavorite': toBoolean(isFavoriteValue, False),
                              #'metaData': metaDataValue
                              "barcodedSamples": barcodedSamples,                                
                              'libraryKey': libraryKeyValue,
                              'forward3primeadapter': forward3primeAdapterValue,
                              'reverselibrarykey': reverselibrarykeyValue,
                              'reverse3primeadapter': reverse3primeAdapterValue,
                              'pairedEndLibraryAdapterName': pairedEndLibraryAdapterNameValue,
                              'samplePrepKitName': samplePrepKitNameValue,
                              'selectedPlugins' : selectedPluginsValue 
                         
                            }
                        
                        try:
                            planTemplate = PlannedExperiment(**kwargs)
                            planTemplate.save()
    
                            logger.debug("views.editplannedexperiment after NEW SAMPLE save id=%d " % planTemplate.id)
                        except:
                            transaction.rollback()
                            
                            logger.exception(format_exc())
                            return HttpResponse(json.dumps({"error": "Internal error while trying to update the plan."}) , mimetype="application/json")            
                            
                            
                        #user could have updated the QC types!!!
                        qcTypes = QCType.objects.all()
                        for qcType in qcTypes:
                
                            try:
                                if json_data[qcType.qcName]:

                                    kwargs = {
                                        'plannedExperiment': planTemplate,
                                        'qcType': qcType,
                                        'threshold': json_data[qcType.qcName], ##'50', ##request.POST.get('qcValues|37'),
                                        }
                    
                                    qcValue = PlannedExperimentQC(**kwargs)
                                    qcValue.save()
                                                             
                            except KeyError:
                                
                                # this is a workaround for the wizard so that even if only some of the QC thresholds have values, the UI will still be complete
                                kwargs = {
                                    'plannedExperiment': planTemplate,
                                    'qcType': qcType,
                                    'threshold': qcType.minThreshold,
                                    }
                                try:
                                    qcValue = PlannedExperimentQC(**kwargs)
                                    qcValue.save()
                        
                                    logger.info("views.editplannedexperiment NEW MIN THRESHOLD qcType.qcName=%s; qcValue.id=%d " % (qcType.qcName, qcValue.id))
                                                                        
                                except:
                                    transaction.rollback()
                                    
                                    logger.exception(format_exc())
                                    return HttpResponse(json.dumps({"error":"Internal error while trying to update the plan."}) , mimetype="application/json")            

                                
                            except:
                                transaction.rollback()
                            
                                logger.exception(format_exc())
                                return HttpResponse(json.dumps({"error":"Internal error while trying to update the plan."}) , mimetype="application/json")            
                                  
    
                        for projectObj in projectObjList:
                            logger.debug("views.editplannedexperiment GOING TO associate project.name=%s; " % (projectObj.name))                            
            
                            planTemplate.projects.add(projectObj)                        
                            logger.debug("views.editplannedexperiment associated project.name=%s; " % (projectObj.name))

                        logger.debug("views.editplannedexperiment GOING TO RETURN planTemplate.id=%d " % planTemplate.id)

                        statusMessage = "plan template edited successfully"
                
                except:
                    transaction.rollback()
                    
                    logger.exception(format_exc())
                    return HttpResponse(json.dumps({"error":"Internal error while trying to update the plan."}) , mimetype="application/json")            
                else:
                    transaction.commit()

            return HttpResponse(json.dumps({"status": statusMessage}) , mimetype="application/json")
        else:           
            logger.info("views.editplannedexperiment sampleList should not be null ")
            transaction.rollback()
            return HttpResponse(json.dumps({"error":"Internal error while trying to update the plan."}) , mimetype="application/json")
            
    else:
        logger.exception(format_exc())
        return HttpResponse(json.dumps({"error":"Error, unsupported http request for plan update."}) , mimetype="application/json")

