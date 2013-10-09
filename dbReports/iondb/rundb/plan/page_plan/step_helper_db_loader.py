# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import types
from iondb.rundb.models import PlannedExperiment, PlannedExperimentQC,\
    RunType, dnaBarcode, Plugin, ApplProduct, SampleSet, ThreePrimeadapter

from iondb.rundb.plan.page_plan.step_helper import StepHelper, StepHelperType
from iondb.rundb.plan.views_helper import getPlanDisplayedName

import json
from iondb.rundb.json_field import JSONDict
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.plugins_step_data import PluginFieldNames
from iondb.rundb.plan.page_plan.output_step_data import OutputFieldNames 
from iondb.rundb.plan.page_plan.barcode_by_sample_step_data import BarcodeBySampleFieldNames
from iondb.rundb.plan.page_plan.save_plan_by_sample_step_data import SavePlanBySampleFieldNames

import logging
logger = logging.getLogger(__name__)

class StepHelperDbLoader():
    
    def getStepHelperForRunType(self, run_type_id, step_helper_type=StepHelperType.CREATE_NEW_TEMPLATE):
        '''
            Creates a step helper for the specified runtype, this can be a plan or a template step helper.
        '''
        step_helper = StepHelper(sh_type=step_helper_type)
        ionReporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        
        runType = RunType.objects.get(pk=run_type_id)
        step_helper.parentName = runType.description
        
        application_step_data.savedFields['runType'] = runType.pk
        application_step_data.savedFields['applicationGroup'] = runType.applicationGroups.all()[0:1][0].pk
        application_step_data.updateSavedObjectsFromSavedFields()
        
        step_helper.update_dependant_steps(ionReporter_step_data)
        return step_helper

    def getStepHelperForNewTemplateBySample(self, run_type_id, step_helper_type=StepHelperType.CREATE_NEW_TEMPLATE_BY_SAMPLE):
        '''
            
        '''
        step_helper = StepHelper(sh_type=step_helper_type)
        ionReporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        
        runType = RunType.objects.get(pk=run_type_id)
        step_helper.parentName = runType.description
        
        application_step_data.savedFields['runType'] = runType.pk
        application_step_data.savedFields['applicationGroup'] = runType.applicationGroups.all()[0:1][0].pk
        application_step_data.updateSavedObjectsFromSavedFields()
        
        step_helper.update_dependant_steps(ionReporter_step_data)
        return step_helper
    
    def updateTemplateSpecificStepHelper(self, step_helper, planned_experiment):
        '''
            Updates the template specific step helper with template specific info from the planned experiment.
        '''
        save_template_step_data = step_helper.steps[StepNames.SAVE_TEMPLATE]
        
        planDisplayedName = getPlanDisplayedName(planned_experiment)
        
        if step_helper.sh_type == StepHelperType.COPY_TEMPLATE:
            save_template_step_data.savedFields['templateName'] = "Copy of " + planDisplayedName
        else:
            save_template_step_data.savedFields['templateName'] = planDisplayedName
        save_template_step_data.savedFields['setAsFavorite'] = planned_experiment.isFavorite
    
    def updatePlanSpecificStepHelper(self, step_helper, planned_experiment, set_template_name=False):
        '''
            Updates the plan specific step helper with plan specific info from the planned experiment.
            
            If the planned experiment is a template and you'd like the originating template name to show up
            in the save plan page pass in set_template_name=True
        '''

        planDisplayedName = getPlanDisplayedName(planned_experiment)

        if set_template_name:
            step_helper.steps[StepNames.SAVE_TEMPLATE].savedFields['templateName'] = planDisplayedName
        
        save_plan_step_data = step_helper.steps[StepNames.SAVE_PLAN]
        kits_step_data = step_helper.steps[StepNames.KITS]

        #Add a "copy of" if we're copying.
        if step_helper.isCopy():
            save_plan_step_data.savedFields['planName'] = "Copy of " + planDisplayedName
        else:
            save_plan_step_data.savedFields['planName'] = planDisplayedName
        
        save_plan_step_data.savedFields['note'] = planned_experiment.get_notes()
        
        if step_helper.isBarcoded():
            save_plan_step_data.prepopulatedFields['prevBarcodeId'] = kits_step_data.savedFields['barcodeId']
            save_plan_step_data.prepopulatedFields['barcodes'] = list(dnaBarcode.objects.filter(name=kits_step_data.savedFields['barcodeId']).order_by('name', 'index'))
            
            barcodeToIrValues = None
            #build a barcodeid to ir info dict
            for plugin_name, plugin_dict in planned_experiment.get_selectedPlugins().items():
                if 'IonReporter' in plugin_name:
                    try:
                        plugin = Plugin.objects.filter(name=plugin_name, active=True)[0]
                    except:
                        continue
                    barcodeToIrValues = {}
                    save_plan_step_data.prepopulatedFields['selectedIr'] = plugin
                    save_plan_step_data.prepopulatedFields['irConfigJson'] = json.dumps(plugin.userinputfields)
                    if PluginFieldNames.USER_INPUT in plugin_dict:
                        #Handle the old and the new style userinput in the plugin dictionary
                        if isinstance(plugin_dict[PluginFieldNames.USER_INPUT], dict)\
                            and "userInputInfo" in plugin_dict[PluginFieldNames.USER_INPUT]\
                            and len(plugin_dict[PluginFieldNames.USER_INPUT]["userInputInfo"]) > 0:
                            for valueDict in plugin_dict[PluginFieldNames.USER_INPUT]["userInputInfo"]:
                                barcodeToIrValues[valueDict['barcodeId']] = valueDict
                                save_plan_step_data.prepopulatedFields['setid_suffix'] = valueDict['setid'][valueDict['setid'].find('__'):]
                        elif isinstance(plugin_dict[PluginFieldNames.USER_INPUT], list):
                            for valueDict in plugin_dict[PluginFieldNames.USER_INPUT]:
                                barcodeToIrValues[valueDict['barcodeId']] = valueDict
                    
            
            #build a sample to barcode dict
            sample_to_barcode = planned_experiment.get_barcodedSamples()
            barcodeIdStrToSample = {}
            for sample, value in sample_to_barcode.items():
                if 'barcodeSampleInfo' in value:
                    barcodeSampleInfo = value['barcodeSampleInfo']
                    for barcode, sampleInfo in barcodeSampleInfo.items():
                        barcodeIdStrToSample[barcode] = {"sampleName": sample}
                        barcodeIdStrToSample[barcode].update(sampleInfo)
                else:
                    barcodeIdStrList = value['barcodes']
                    for barcodeIdStr in barcodeIdStrList:
                        barcodeIdStrToSample[barcodeIdStr] = {"sampleName": sample, "externalId": None, "description": None}

            #for each barcode populate the sample and ir info
            for barcode in save_plan_step_data.prepopulatedFields['barcodes']:
                if barcode.id_str in barcodeIdStrToSample:
                    save_plan_step_data.savedFields['barcodeSampleName%s' % str(barcode.pk)] = barcodeIdStrToSample[barcode.id_str]["sampleName"]
                    save_plan_step_data.savedFields['barcodeSampleExternalId%s' % str(barcode.pk)] = barcodeIdStrToSample[barcode.id_str]["externalId"]
                    save_plan_step_data.savedFields['barcodeSampleDescription%s' % str(barcode.pk)] = barcodeIdStrToSample[barcode.id_str]["description"]
                    if barcodeToIrValues:
                        save_plan_step_data.savedFields['irGender%s' % str(barcode.pk)] = barcodeToIrValues[barcode.id_str]['Gender']
                        save_plan_step_data.savedFields['irWorkflow%s' % str(barcode.pk)] = barcodeToIrValues[barcode.id_str]['Workflow']
                        #some older plans do not have "Relation" specified - could be due to work-in-progress development code
                        save_plan_step_data.savedFields['irRelation%s' % str(barcode.pk)] = barcodeToIrValues[barcode.id_str].get('Relation', "")
                        save_plan_step_data.savedFields['irRelationRole%s' % str(barcode.pk)] = barcodeToIrValues[barcode.id_str]['RelationRole']
                        save_plan_step_data.savedFields['irSetID%s' % str(barcode.pk)] = barcodeToIrValues[barcode.id_str]['setid'][0:barcodeToIrValues[barcode.id_str]['setid'].find('__')] if barcodeToIrValues[barcode.id_str]['setid'] else ""
                    else:
                        save_plan_step_data.savedFields['irGender%s' % str(barcode.pk)] = None
                        save_plan_step_data.savedFields['irWorkflow%s' % str(barcode.pk)] = None
                        save_plan_step_data.savedFields['irRelation%s' % str(barcode.pk)] = None
                        save_plan_step_data.savedFields['irRelationRole%s' % str(barcode.pk)] = None
                        save_plan_step_data.savedFields['irSetID%s' % str(barcode.pk)] = None
                else:
                    save_plan_step_data.savedFields['barcodeSampleName%s' % str(barcode.pk)] = None
                    save_plan_step_data.savedFields['barcodeSampleExternalId%s' % str(barcode.pk)] = None
                    save_plan_step_data.savedFields['barcodeSampleDescription%s' % str(barcode.pk)] = None
                    save_plan_step_data.savedFields['irGender%s' % str(barcode.pk)] = None
                    save_plan_step_data.savedFields['irWorkflow%s' % str(barcode.pk)] = None
                    save_plan_step_data.savedFields['irRelation%s' % str(barcode.pk)] = None
                    save_plan_step_data.savedFields['irRelationRole%s' % str(barcode.pk)] = None
                    save_plan_step_data.savedFields['irSetID%s' % str(barcode.pk)] = None
            
            #do not copy sampleTubeLabel since a sample tube is meant for 1 run only
            save_plan_step_data.savedFields['barcodeSampleTubeLabel'] = "" if step_helper.isCopy() else planned_experiment.sampleTubeLabel
        else:
            #when we load a non-barcoded run for editing/copying we know it will only have a single sample.
            save_plan_step_data.prepopulatedFields['prevBarcodeId'] = None
            
            save_plan_step_data.savedFields['sampleName1'] = planned_experiment.get_sampleDisplayedName()
            #do not copy sampleTubeLabel since a sample tube is meant for 1 run only
            save_plan_step_data.savedFields['tubeLabel1'] = "" if step_helper.isCopy() else planned_experiment.sampleTubeLabel
            save_plan_step_data.savedFields['sampleExternalId1'] = planned_experiment.get_sample_external_id()
            save_plan_step_data.savedFields['sampleDescription1'] = planned_experiment.get_sample_description()
            
            #if there was ir info we grab it
            for plugin_name, plugin_dict in planned_experiment.get_selectedPlugins().items():
                if 'IonReporter' in plugin_name:
                    try:
                        plugin = Plugin.objects.filter(name=plugin_name, active=True)[0]
                    except:
                        continue
                    save_plan_step_data.prepopulatedFields['selectedIr'] = plugin
                    save_plan_step_data.prepopulatedFields['irConfigJson'] = json.dumps(plugin.userinputfields)
                    sample_ir_values = None
                    if PluginFieldNames.USER_INPUT in plugin_dict:
                        #Handle the old and the new style userinput in the plugin dictionary
                        if isinstance(plugin_dict[PluginFieldNames.USER_INPUT], dict)\
                            and "userInputInfo" in plugin_dict[PluginFieldNames.USER_INPUT]\
                            and len(plugin_dict[PluginFieldNames.USER_INPUT]["userInputInfo"])>0:
                            sample_ir_values = plugin_dict[PluginFieldNames.USER_INPUT]["userInputInfo"][0]
                            save_plan_step_data.prepopulatedFields['setid_suffix'] = plugin_dict[PluginFieldNames.USER_INPUT]["userInputInfo"][0]['setid'][plugin_dict[PluginFieldNames.USER_INPUT]["userInputInfo"][0]['setid'].find('__'):]
                        elif isinstance(plugin_dict[PluginFieldNames.USER_INPUT], list)\
                            and len(plugin_dict[PluginFieldNames.USER_INPUT])>0:
                            sample_ir_values = plugin_dict[PluginFieldNames.USER_INPUT][0]
                    
                    if sample_ir_values:
                        save_plan_step_data.savedFields['irWorkflow1'] = sample_ir_values['Workflow']
                        save_plan_step_data.savedFields['irGender1'] = sample_ir_values['Gender']
                        save_plan_step_data.savedFields['irRelation1'] = sample_ir_values['Relation']
                        save_plan_step_data.savedFields['irRelationRole1'] = sample_ir_values['RelationRole']
                        save_plan_step_data.savedFields['irSetID1'] = str(sample_ir_values['setid'])[0:str(sample_ir_values['setid']).find('__')]

        save_plan_step_data.updateSavedObjectsFromSavedFields()

    def updateUniversalStepHelper(self, step_helper, planned_experiment):
        '''
            Update a step helper with info from planned experiment that applies to both plans and templates.
        '''
        # export_step_data = step_helper.steps[StepNames.EXPORT]
        ionreporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        kits_step_data = step_helper.steps[StepNames.KITS]
        reference_step_data = step_helper.steps[StepNames.REFERENCE]
        plugins_step_data = step_helper.steps[StepNames.PLUGINS]
        # if not step_helper.isPlanBySample():
        #     ionreporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        appl_product = None
        
        # application_step_data.updateFromStep(export_step_data)
        
        ionreporter_step_data.savedFields['sampleGrouping'] = planned_experiment.sampleGrouping if planned_experiment.sampleGrouping else None 
        
        #why use AMPS?
        selectedRunType = RunType.objects.filter(runType="GENS")[0:1][0]
        if selectedRunType.applicationGroups.all().count() > 0:
            application_step_data.savedFields['applicationGroup'] = selectedRunType.applicationGroups.all()[0:1][0].pk
        
        #only set the runtype if its still a valid one, otherwise keep the default that was set in the constructor.
        if RunType.objects.filter(runType=planned_experiment.runType).count() > 0:
            selectedRunType = RunType.objects.filter(runType=planned_experiment.runType)[0:1][0]
            if hasattr(planned_experiment, 'applicationGroup') and planned_experiment.applicationGroup:
                application_step_data.savedFields['applicationGroup'] = planned_experiment.applicationGroup.pk
            else:
                #if no application group is selected, pick the first one associated with the runType
                application_step_data.savedFields['applicationGroup'] = selectedRunType.applicationGroups.all()[0:1][0].pk 
                #logger.debug("step_helper_db_loader.updateUniversalStepHelper() planned_experiment.id=%d; PICKING applicationGroup.id=%d" %(planned_experiment.id, application_step_data.savedFields['applicationGroup'])) 
                    
        application_step_data.savedFields['runType'] = selectedRunType.pk
        application_step_data.savedObjects['runType'] = selectedRunType
        appl_product = ApplProduct.objects.get(applType__runType = selectedRunType.runType, isDefault = True, isActive = True, isVisible = True)

        logger.debug("step_helper_db_loader.updateUniversalStepHelper() planned_experiment.id=%d; applProduct.productCode=%s" %(planned_experiment.id, appl_product.productCode))
        application_step_data.savedObjects['applProduct'] = appl_product
        
        kits_step_data.savedFields['templatekitType'] = "OneTouch"
        if planned_experiment.is_ionChef():
            kits_step_data.savedFields['templatekitType'] = "IonChef"
        
        kits_step_data.savedFields['templatekitname'] = planned_experiment.templatingKitName
        kits_step_data.savedFields['controlsequence'] = planned_experiment.controlSequencekitname
        kits_step_data.savedFields['samplePreparationKit'] = planned_experiment.samplePrepKitName
        kits_step_data.savedFields['barcodeId'] = planned_experiment.get_barcodeId()
        chipType = planned_experiment.get_chipType()
        kits_step_data.savedFields['chipType'] = 'P1.1.17' if chipType == '900v2' else chipType
        kits_step_data.savedFields['flows'] = planned_experiment.get_flows()
        kits_step_data.savedFields['forward3primeAdapter'] = planned_experiment.get_forward3primeadapter()
        kits_step_data.savedFields['libraryKey'] = planned_experiment.get_libraryKey()
        kits_step_data.savedFields['librarykitname'] = planned_experiment.get_librarykitname()
        kits_step_data.savedFields['sequencekitname'] = planned_experiment.get_sequencekitname()
        kits_step_data.savedFields['isDuplicateReads'] = planned_experiment.is_duplicateReads()

        avalanche3PrimeAdapters = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single', chemistryType = 'avalanche').order_by('-isDefault', 'name')
        kits_step_data.savedFields['avalancheForward3PrimeAdapter'] = avalanche3PrimeAdapters[0].sequence
        if appl_product.defaultAvalancheTemplateKit:
            kits_step_data.savedFields['avalancheTemplateKitName'] = appl_product.defaultAvalancheTemplateKit.name
        if appl_product.defaultAvalancheSequencingKit:
            kits_step_data.savedFields['avalancheSequencekitname'] = appl_product.defaultAvalancheSequencingKit.name
        
        nonAvalanche3PrimeAdapters = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single').exclude(chemistryType = 'avalanche').order_by('-isDefault', 'name')
        kits_step_data.savedFields['nonAvalancheForward3PrimeAdapter'] = nonAvalanche3PrimeAdapters[0].sequence
        if appl_product.defaultTemplateKit:
            kits_step_data.savedFields['nonAvalancheTemplateKitName'] = appl_product.defaultTemplateKit.name
        if appl_product.defaultSequencingKit:
            kits_step_data.savedFields['nonAvalancheSequencekitname'] = appl_product.defaultSequencingKit.name
        
        reference_step_data.savedFields['targetBedFile'] = planned_experiment.get_bedfile()
        reference_step_data.savedFields["reference"] = planned_experiment.get_library()
        reference_step_data.savedFields['hotSpotBedFile'] = planned_experiment.get_regionfile()
        reference_step_data.prepopulatedFields['showHotSpotBed'] = False
        if appl_product and appl_product.isHotspotRegionBEDFileSuppported:
            reference_step_data.prepopulatedFields['showHotSpotBed'] = True
        
        plugins_step_data.savedFields[StepNames.PLUGINS] = []
        plugins = planned_experiment.get_selectedPlugins()
        pluginIds = []
        for plugin_name, plugin_dict in plugins.items():
            # find existing plugin by plugin_name (handles plugins that were reinstalled or uninstalled)
            try:
                plugin = Plugin.objects.filter(name=plugin_name, active=True)[0]
            except:
                continue
            if PluginFieldNames.EXPORT in plugin.pluginsettings.get(PluginFieldNames.FEATURES,[]):
                if not step_helper.isPlanBySample():
                    #ionreporter_step_data.savedFields['uploaders'].append(plugin.id)
                    pass
            else:
                pluginIds.append(plugin.id)
                plugins_step_data.savedFields[PluginFieldNames.PLUGIN_CONFIG % plugin.id] = json.dumps(plugin_dict.get(PluginFieldNames.USER_INPUT,''))
            
            if 'accountId' in plugin_dict:
                ionreporter_step_data.savedFields['irAccountId'] = plugin_dict.get('accountId')
                ionreporter_step_data.savedFields['irAccountName'] = plugin_dict.get('accountName')
                ionreporter_step_data.savedFields['irVersion'] = plugin_dict.get('version')
            elif PluginFieldNames.USER_INPUT in plugin_dict and 'accountId' in plugin_dict[PluginFieldNames.USER_INPUT]:
                ionreporter_step_data.savedFields['irAccountId'] = plugin_dict[PluginFieldNames.USER_INPUT].get('accountId')
                ionreporter_step_data.savedFields['irAccountName'] = plugin_dict[PluginFieldNames.USER_INPUT].get('accountName')
                
                if 'userconfigs' in plugin.config:
                    if 'ionadmin' in plugin.config.get('userconfigs'):
                        _list = plugin.config.get('userconfigs').get('ionadmin')
                        for l in _list:
                            if l.get('id') == ionreporter_step_data.savedFields['irAccountId']:
                                ionreporter_step_data.savedFields['irVersion'] = l.get('version')

        if 'IonReporterUploader' not in plugins:
            ionreporter_step_data.savedFields['irAccountId'] = '0'
            ionreporter_step_data.savedFields['irAccountName'] = 'None'

        
        step_helper.steps[StepNames.IONREPORTER].savedFields['irworkflow'] = planned_experiment.irworkflow
        plugins_step_data.savedFields[PluginFieldNames.PLUGIN_IDS] = ', '.join(str(v) for v in pluginIds)
        plugins_step_data.updateSavedObjectsFromSavedFields()
            
        logger.debug("PLUGINS ARE: %s" % str(plugins_step_data.savedFields[StepNames.PLUGINS]))
        
        qc_values = planned_experiment.qcValues.all()
        for qc_value in qc_values:
            step_helper.steps['Monitoring'].savedFields[qc_value.qcName] = PlannedExperimentQC.objects.get(plannedExperiment__pk=planned_experiment.pk,
                                                                                                           qcType__pk=qc_value.pk).threshold
        logger.debug("QCs ARE: %s" % str(step_helper.steps['Monitoring'].savedFields))
        
        
        step_helper.steps[StepNames.OUTPUT].savedFields[OutputFieldNames.PROJECTS] = []
        projects = planned_experiment.projects.all()
        for project in projects:
            step_helper.steps[StepNames.OUTPUT].savedFields[OutputFieldNames.PROJECTS].append(project.pk)

    def get_ir_fields_dict_from_user_input_info(self, user_input_info, sample_name, index):
        
        if sample_name == 'barcoded--Sample':
            if index >= len(user_input_info[0].keys()):
                return dict(
                    sample = "",
                    sampleDescription = "",
                    sampleExternalId = "",
                    barcodeId = "",
                    Gender = None,
                    RelationRole = None,
                    Workflow = None,
                    setid = None,
                )

            sample_name = user_input_info[0].keys()[index]
            barcode_id_str = user_input_info[0].get(sample_name).get('barcodeSampleInfo').keys()[0]
            sampleDescription = user_input_info[0].get(sample_name).get('barcodeSampleInfo').get(barcode_id_str).get('description')
            externalId = user_input_info[0].get(sample_name).get('barcodeSampleInfo').get(barcode_id_str).get('externalId')
            
            return dict(
                sample = sample_name,
                sampleDescription = sampleDescription,
                sampleExternalId = externalId,
                barcodeId = barcode_id_str,
                Gender = None,
                RelationRole = None,
                Workflow = None,
                setid = None,
            )
        else:
            return user_input_info[index]

    def updatePlanBySampleSpecificStepHelper(self, step_helper, planned_experiment, sampleset_id):
        """

        """
        barcoding_step = step_helper.steps[StepNames.BARCODE_BY_SAMPLE]
        save_plan_step = step_helper.steps[StepNames.SAVE_PLAN_BY_SAMPLE]
                        
        planDisplayedName = getPlanDisplayedName(planned_experiment)

        step_helper.steps[StepNames.IONREPORTER].savedFields['irworkflow'] = planned_experiment.irworkflow
        if step_helper.isCopy():
            save_plan_step.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME] = "Copy of " + planDisplayedName
        else:
            save_plan_step.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME] = planDisplayedName
        
        if sampleset_id:
            sampleset = SampleSet.objects.get(pk=sampleset_id)
            if sampleset.SampleGroupType_CV:
                step_helper.steps[StepNames.APPLICATION].savedFields['sampleGrouping'] = sampleset.SampleGroupType_CV.pk
        else:
            sampleset = planned_experiment.sampleSet

        save_plan_step.savedObjects[SavePlanBySampleFieldNames.SAMPLESET] = sampleset

        samplesetitems = list(sampleset.samples.all())

        all_barcodes = dnaBarcode.objects.values('name', 'id_str', 'sequence')
        barcodekits = set(map(lambda x : x['name'], all_barcodes))
        barcoding_step.prepopulatedFields['accountId'] = step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountId']
        barcoding_step.prepopulatedFields['all_barcodes'] = sorted(all_barcodes, key=lambda x : x['id_str'])
        barcoding_step.prepopulatedFields['irworkflow'] = planned_experiment.irworkflow
        barcoding_step.prepopulatedFields['pe_irworkflow'] = planned_experiment.irworkflow
        dnabarcodes_grouped = [
            dict(
                    name     = barcodekit, 
                    barcodes = sorted([
                                dict( id_str   = str(dnabarcode['id_str']), 
                                      sequence = str(dnabarcode['sequence'])) \
                                            for dnabarcode in all_barcodes if dnabarcode['name'] == barcodekit], key=lambda z: z['id_str'])) \
                                                for barcodekit in barcodekits]

        if step_helper.isBarcoded():
            # planned_dnabarcodes = dnaBarcode.objects.filter(name=planned_experiment.get_barcodeId()).values('id_str', 'sequence').order_by('id_str')
            planned_dnabarcodes = list(dnaBarcode.objects.filter(name=planned_experiment.get_barcodeId()).order_by('id_str'))
        else:
            planned_dnabarcodes = [None]

        barcoding_step.prepopulatedFields['samplesetitems'] = samplesetitems
        barcoding_step.prepopulatedFields[BarcodeBySampleFieldNames.PLANNED_DNABARCODES] = planned_dnabarcodes
        barcoding_step.prepopulatedFields['barcodekits'] = dnabarcodes_grouped
        barcoding_step.savedFields[BarcodeBySampleFieldNames.NUMBER_OF_CHIPS] = len(samplesetitems)
        barcoding_step.savedFields[BarcodeBySampleFieldNames.BARCODE_ID] = planned_experiment.get_barcodeId()
        
        selected_plugins = planned_experiment.get_selectedPlugins()
        if 'IonReporterUploader' in selected_plugins:

            try:
                user_input_info = selected_plugins['IonReporterUploader']['userInput']['userInputInfo']
            except:
                user_input_info = selected_plugins['IonReporterUploader']['userInput']
        else:
            
            user_input_info = [planned_experiment.get_barcodedSamples()]
            result = []
            if len(user_input_info) > 0 and len(user_input_info[0].keys()) > 0:
                for i in range(len(samplesetitems)):
                    result.append(self.get_ir_fields_dict_from_user_input_info(user_input_info, user_input_info[i].get('sample', "barcoded--Sample") if i < len(user_input_info) else user_input_info[0].get('sampleName', "barcoded--Sample"), i))
                user_input_info = sorted(result, key=lambda d: d.get('barcodeId'))

        existing_plan = not planned_experiment.isReusable
        if (step_helper.isEdit() or step_helper.isCopy()) and len(user_input_info[0]) > 0:
            if user_input_info[0].get('setid'):
                barcoding_step.prepopulatedFields['setid_suffix'] = user_input_info[0].get('setid')[user_input_info[0].get('setid').find('__'):]

        for i in range(100):
            barcoding_step.savedFields['%s%d' % (BarcodeBySampleFieldNames.NUMBER_OF_CHIPS, i)] = i + 1
            barcoding_step.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_NAME, i)] = None
            barcoding_step.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID, i)] = None
            barcoding_step.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_BARCODE, i)] = None
            barcoding_step.savedFields['%s%d' % (BarcodeBySampleFieldNames.WORKFLOW, i)] = None
            barcoding_step.savedFields['%s%d' % (BarcodeBySampleFieldNames.GENDER, i)] = None
            barcoding_step.savedFields['%s%d' % (BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION, i)] = None
            barcoding_step.savedFields['%s%d' % (BarcodeBySampleFieldNames.RELATION_ROLE, i)] = None
            barcoding_step.savedFields['%s%d' % ('relation', i)] = None
            barcoding_step.savedFields['%s%d' % (BarcodeBySampleFieldNames.SET_ID, i)] = None

        if step_helper.isBarcoded():

            for i in range(len(planned_dnabarcodes)):
                samplesetitem = samplesetitems[i] if i < len(samplesetitems) else None
                dnabarcode = planned_dnabarcodes[i] if i < len(planned_dnabarcodes) else None
                

                if existing_plan:
                    if i < len(user_input_info):
                        user_input_dict = self.get_ir_fields_dict_from_user_input_info(user_input_info, user_input_info[i].get('sample', "barcoded--Sample"), i)

                        barcoding_step.savedObjects[BarcodeBySampleFieldNames.BARCODE_TO_SAMPLE]['%s%d' % (user_input_dict.get('barcodeId'), i)] = {
                            BarcodeBySampleFieldNames.NUMBER_OF_CHIPS        : i + 1,
                            BarcodeBySampleFieldNames.SAMPLE_NAME            : user_input_dict.get('sample'),
                            BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION  : user_input_dict.get('sampleDescription'),
                            BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID     : user_input_dict.get('sampleExternalId'),
                            BarcodeBySampleFieldNames.GENDER         : user_input_dict.get('Gender'),
                            BarcodeBySampleFieldNames.RELATION_ROLE  : user_input_dict.get('RelationRole'),
                            BarcodeBySampleFieldNames.WORKFLOW  : user_input_dict.get('Workflow'),
                            BarcodeBySampleFieldNames.BARCODE_ID  : user_input_dict.get('barcodeId'),
                            BarcodeBySampleFieldNames.SET_ID  : user_input_dict.get('setid')[0:user_input_dict['setid'].find('__')] if user_input_dict.get('setid') else "",
                            'relation' : '',


                        }
                    else:
                        barcoding_step.savedObjects[BarcodeBySampleFieldNames.BARCODE_TO_SAMPLE]['%s%d' % (dnabarcode.id_str, i)] = {
                            BarcodeBySampleFieldNames.NUMBER_OF_CHIPS        : i + 1,
                            BarcodeBySampleFieldNames.SAMPLE_NAME            : '',
                            BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION  : '',
                            BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID     : '',
                            BarcodeBySampleFieldNames.GENDER         : '',
                            BarcodeBySampleFieldNames.RELATION_ROLE  : '',
                            BarcodeBySampleFieldNames.WORKFLOW  : '',
                            BarcodeBySampleFieldNames.BARCODE_ID  : dnabarcode.id_str,
                            BarcodeBySampleFieldNames.SET_ID  : 0,
                            'relation' : '',

                        }
                    
                else:
                    
                    if i < len(samplesetitems):
                        #create the saved objects dictionary for the dnabarcode to control
                        #the order of display of dnabarcode id_str's and sample item names and descriptions
                        barcoding_step.savedObjects[BarcodeBySampleFieldNames.BARCODE_TO_SAMPLE]['%s%d' % (dnabarcode.id_str, i)] = {
                            BarcodeBySampleFieldNames.NUMBER_OF_CHIPS : i + 1,
                            BarcodeBySampleFieldNames.SAMPLE_NAME            : samplesetitem.sample.displayedName,
                            BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION  : samplesetitem.sample.description,
                            BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID     : samplesetitem.sample.externalId,
                            BarcodeBySampleFieldNames.GENDER         : samplesetitem.gender,
                            BarcodeBySampleFieldNames.RELATION_ROLE  : samplesetitem.relationshipRole,
                            BarcodeBySampleFieldNames.WORKFLOW  : planned_experiment.irworkflow,
                            BarcodeBySampleFieldNames.BARCODE_ID  : dnabarcode.id_str,
                            BarcodeBySampleFieldNames.SET_ID  : samplesetitem.relationshipGroup,
                            'relation' : '',

                        }
                    else:
                        barcoding_step.savedObjects[BarcodeBySampleFieldNames.BARCODE_TO_SAMPLE]['%s%d' % (dnabarcode.id_str, i)] = {
                            BarcodeBySampleFieldNames.NUMBER_OF_CHIPS        : i + 1,
                            BarcodeBySampleFieldNames.SAMPLE_NAME            : '',
                            BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION  : '',
                            BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID     : '',
                            BarcodeBySampleFieldNames.GENDER         : '',
                            BarcodeBySampleFieldNames.RELATION_ROLE  : '',
                            BarcodeBySampleFieldNames.WORKFLOW  : '',
                            BarcodeBySampleFieldNames.BARCODE_ID  : dnabarcode.id_str,
                            BarcodeBySampleFieldNames.SET_ID  : 0,
                            'relation' : '',


                        }                        

            
                #create the saved objects dictionary for each sample set item which holds the dnabarcoding information
                #and will be used in the final save process
                if i < len(samplesetitems):
                    barcoding_step.savedObjects[BarcodeBySampleFieldNames.SAMPLE_TO_BARCODE][samplesetitem.sample.displayedName] = {
                        'barcodeSampleInfo' : { 
                                    dnabarcode.id_str : {
                                                    'externalId'   : samplesetitem.sample.externalId,
                                                    'description'  : samplesetitem.sample.description,

                                    }
                                },
                        'barcodes'          : [dnabarcode.id_str]
                        }
        else:

            for i in range(20):
                samplesetitem = samplesetitems[i] if i < len(samplesetitems) else None
                dnabarcode = planned_dnabarcodes[i] if i < len(planned_dnabarcodes) else None

                

            
                if existing_plan:
                    if i == planned_experiment.sampleSet_planIndex:
                        if step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountId'] == '0':
                            user_input_info = [dict()]

                        barcoding_step.savedObjects[BarcodeBySampleFieldNames.CHIP_TO_SAMPLE][i] = {
                            BarcodeBySampleFieldNames.NUMBER_OF_CHIPS : i + 1,
                            BarcodeBySampleFieldNames.SAMPLE_NAME            : planned_experiment.get_sampleDisplayedName(),
                            BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION  : planned_experiment.get_sample_description(),
                            BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID     : planned_experiment.get_sample_external_id(),
                            BarcodeBySampleFieldNames.GENDER         : user_input_info[0].get('Gender'),
                            BarcodeBySampleFieldNames.RELATION_ROLE  : user_input_info[0].get('RelationRole'),
                            BarcodeBySampleFieldNames.WORKFLOW  : user_input_info[0].get('Workflow'),
                            BarcodeBySampleFieldNames.SET_ID  : user_input_info[0].get('setid')[0:user_input_info[0]['setid'].find('__')] if user_input_info[0].get('setid') else "",
                            'relation' : '',
                        }
                else:
                    if i < len(samplesetitems):
                        barcoding_step.savedObjects[BarcodeBySampleFieldNames.CHIP_TO_SAMPLE][i] = {
                            BarcodeBySampleFieldNames.NUMBER_OF_CHIPS : i + 1,
                            BarcodeBySampleFieldNames.SAMPLE_NAME            : samplesetitem.sample.displayedName,
                            BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION  : samplesetitem.sample.description,
                            BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID     : samplesetitem.sample.externalId,
                            BarcodeBySampleFieldNames.GENDER         : samplesetitem.gender,
                            BarcodeBySampleFieldNames.RELATION_ROLE  : samplesetitem.relationshipRole,
                            BarcodeBySampleFieldNames.WORKFLOW  : planned_experiment.irworkflow,
                            BarcodeBySampleFieldNames.SET_ID  : samplesetitem.relationshipGroup,
                            'relation' : '',
                        }
                    elif step_helper.steps[StepNames.IONREPORTER].savedFields['irAccountId'] == '0':
                        barcoding_step.savedObjects[BarcodeBySampleFieldNames.CHIP_TO_SAMPLE][i] = {
                            BarcodeBySampleFieldNames.NUMBER_OF_CHIPS : i + 1,
                            BarcodeBySampleFieldNames.SAMPLE_NAME            : '',
                            BarcodeBySampleFieldNames.SAMPLE_DESCRIPTION  : '',
                            BarcodeBySampleFieldNames.SAMPLE_EXTERNAL_ID     : '',
                            BarcodeBySampleFieldNames.GENDER         : '',
                            BarcodeBySampleFieldNames.RELATION_ROLE  : '',
                            BarcodeBySampleFieldNames.WORKFLOW  : '',
                            'relation' : '',
                        }


    def getStepHelperForTemplatePlannedExperiment(self, pe_id, step_helper_type=StepHelperType.EDIT_TEMPLATE, sampleset_id=None):
        '''
            Get a step helper from a template planned experiment.
        '''
        step_helper = StepHelper(sh_type=step_helper_type, previous_template_id=pe_id)
        planned_experiment = PlannedExperiment.objects.get(pk=pe_id)
        if not planned_experiment.isReusable:
            raise ValueError("You must pass in a template id, not a plan id.")

        planDisplayedName = getPlanDisplayedName(planned_experiment)
        
        step_helper.parentName = planDisplayedName
        
        self.updateUniversalStepHelper(step_helper, planned_experiment)
        if step_helper.isPlan() and step_helper.isPlanBySample():
            self.updatePlanBySampleSpecificStepHelper(step_helper, planned_experiment, sampleset_id)
        elif step_helper.isPlan():
            self.updatePlanSpecificStepHelper(step_helper, planned_experiment, True)
        else:
            self.updateTemplateSpecificStepHelper(step_helper, planned_experiment)
        
        return step_helper
    
    def getStepHelperForPlanPlannedExperiment(self, pe_id, step_helper_type=StepHelperType.EDIT_PLAN):
        '''
            Get a plan step helper from a plan planned experiment.
        '''
        step_helper = StepHelper(sh_type=step_helper_type, previous_plan_id=pe_id)
        planned_experiment = PlannedExperiment.objects.get(pk=pe_id)

        planDisplayedName = getPlanDisplayedName(planned_experiment)

        step_helper.parentName = planDisplayedName
        
        if planned_experiment.isReusable:
            raise ValueError("You must pass in a plan id, not a template id.")
        
        self.updateUniversalStepHelper(step_helper, planned_experiment)
        if step_helper.isPlan() and step_helper.isPlanBySample():
            self.updatePlanBySampleSpecificStepHelper(step_helper, planned_experiment, planned_experiment.sampleSet.pk)
        elif step_helper.isPlan():
            self.updatePlanSpecificStepHelper(step_helper, planned_experiment)
        else:
            raise ValueError("Can not create templates from plans.")
        
        return step_helper
