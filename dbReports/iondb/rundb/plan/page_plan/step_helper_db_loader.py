# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from django.db.models import Q
from iondb.rundb.models import PlannedExperiment, PlannedExperimentQC,\
    RunType, dnaBarcode, Plugin, ApplProduct, SampleSet, ThreePrimeadapter, Chip, KitInfo, AnalysisArgs, ApplicationGroup

from iondb.rundb.plan.page_plan.step_helper import StepHelper

from iondb.rundb.plan.page_plan.step_helper_types import StepHelperType
from iondb.rundb.plan.views_helper import getPlanDisplayedName, getPlanBarcodeCount

import json
from iondb.rundb.json_field import JSONEncoder
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.kits_step_data import KitsFieldNames
from iondb.rundb.plan.page_plan.reference_step_data import ReferenceFieldNames
from iondb.rundb.plan.page_plan.plugins_step_data import PluginFieldNames
from iondb.rundb.plan.page_plan.output_step_data import OutputFieldNames
from iondb.rundb.plan.page_plan.barcode_by_sample_step_data import BarcodeBySampleFieldNames
from iondb.rundb.plan.page_plan.save_plan_by_sample_step_data import SavePlanBySampleFieldNames
from iondb.rundb.plan.page_plan.save_plan_step_data import ApplicationFieldNames
from iondb.rundb.plan.page_plan.save_template_step_data import SaveTemplateStepDataFieldNames
from iondb.rundb.plan.page_plan.save_plan_step_data import SavePlanFieldNames
from iondb.rundb.plan.page_plan.ionreporter_step_data import IonReporterFieldNames
from iondb.rundb.plan.page_plan.analysis_params_step_data import AnalysisParamsFieldNames

import logging

logger = logging.getLogger(__name__)


class StepHelperDbLoader():

    def getStepHelperForRunType(self, run_type_id, step_helper_type=StepHelperType.CREATE_NEW_TEMPLATE, applicationGroupName=None):
        '''
            Creates a step helper for the specified runtype, this can be a plan or a template step helper.
        '''
        # logger.debug("ENTER step_helper_db_loader.getStepHelperForRunType() run_type_id=%s; step_helper_type=%s" %(str(run_type_id), step_helper_type))

        step_helper = StepHelper(sh_type=step_helper_type)
        ionReporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        kits_step_data = step_helper.steps[StepNames.KITS]

        runType = RunType.objects.get(pk=run_type_id)
        applicationGroupObj = ApplicationGroup.objects.get(name=applicationGroupName) if applicationGroupName else None

        if applicationGroupObj:
            step_helper.parentName = applicationGroupObj.description
        else:
            step_helper.parentName = runType.description

        step_helper.isFromScratch = True
        step_helper.isParentSystem = False

        self._updateApplicationStepData(runType, step_helper, application_step_data, applicationGroupObj)
        self._updateKitsStepData(runType, step_helper, kits_step_data, applicationGroupObj)

        if step_helper.isPlan():
            save_plan_step_data = step_helper.steps[StepNames.SAVE_PLAN]
            self._updateSaveStepData(runType, step_helper, save_plan_step_data)

        return step_helper

    def _updateApplicationStepData(self, runTypeObj, step_helper, application_step_data, applicationGroupObj=None):

        application_step_data.savedFields[ApplicationFieldNames.RUN_TYPE] = runTypeObj.pk

        if applicationGroupObj:
            application_step_data.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME] = applicationGroupObj.name

            application_step_data.savedObjects[ApplicationFieldNames.APPL_PRODUCT] = ApplProduct.objects.get(isActive=True, isVisible=True,
                                                                                                             applType__runType=runTypeObj.runType, applicationGroup=applicationGroupObj)
            application_step_data.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS] = ApplProduct.objects.filter(isActive=True, isDefaultForInstrumentType=True,
                                                                                                             applType__runType=runTypeObj.runType, applicationGroup=applicationGroupObj)

            categorizedApplProducts = ApplProduct.objects.filter(isActive=True,
                                                          applType__runType=runTypeObj.runType, applicationGroup=applicationGroupObj).exclude(categories = "")

            if categorizedApplProducts:
                application_step_data.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS_CATEGORIZED] = categorizedApplProducts
            else:
                application_step_data.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS_CATEGORIZED] = None

        else:
            application_step_data.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME] = runTypeObj.applicationGroups.all()[0:1][0].name
            # application_step_data.savedObjects["runType"] = runTypeObj
            # application_step_data.savedObjects["applProduct"] = ApplProduct.objects.get(isActive=True, isDefault=True, isVisible=True,
            # applType__runType = runTypeObj.runType)

        application_step_data.updateSavedObjectsFromSavedFields()

        step_helper.update_dependent_steps(application_step_data)

    def _updateKitsStepData(self, runTypeObj, step_helper, kits_step_data, applicationGroupObj=None):
        kits_step_data.prepopulatedFields[KitsFieldNames.IS_CHIP_TYPE_REQUIRED] = step_helper.isPlan()

        if applicationGroupObj:
            applProduct = ApplProduct.objects.get(isActive=True, isVisible=True, applType__runType=runTypeObj.runType, applicationGroup=applicationGroupObj)
            if applProduct:
                kits_step_data.updateFieldsFromDefaults(applProduct)

    def _updateSaveStepData(self, runTypeObj, step_helper, save_plan_step_data):
        num_samples = 1
        if step_helper.isDualNucleotideTypeBySample():
            num_samples = 2
        if runTypeObj.runType == 'TAG_SEQUENCING':
            num_samples = 8
        save_plan_step_data.prepopulatedFields[SavePlanFieldNames.NUM_SAMPLES] = num_samples

    def _metaDataFromPlan(self, step_helper, planned_experiment):
        metaData = planned_experiment.metaData or {}
        if step_helper.isCreate() or step_helper.sh_type == StepHelperType.COPY_TEMPLATE:
            metaData["fromTemplate"] = planned_experiment.planName
            metaData["fromTemplateSource"] = "ION" if planned_experiment.isSystem else planned_experiment.username
        return metaData

    def getStepHelperForTemplateRunType(self, run_type_id, step_helper_type=StepHelperType.CREATE_NEW_TEMPLATE, template_id=-1):
        '''
            Creates a template step helper for the specified runty.
        '''
        # logger.debug("ENTER step_helper_db_loader.getStepHelperForRunType() run_type_id=%s" %(str(run_type_id)))

        step_helper = StepHelper(sh_type=step_helper_type, previous_template_id=template_id)
        ionReporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]

        runType = RunType.objects.get(pk=run_type_id)
        step_helper.parentName = runType.description

        step_helper.isParentSystem = False

        applicationGroupObj = None
        if (template_id > 0):
            planned_experiment = PlannedExperiment.objects.get(pk=template_id)
            if planned_experiment.applicationGroup:
                applicationGroupObj = planned_experiment.applicationGroup
        #if plan/template has applicationGroup, need to pass that along
        self._updateApplicationStepData(runType, step_helper, application_step_data, applicationGroupObj)

        kits_step_data = step_helper.steps[StepNames.KITS]
        self._updateKitsStepData(runType, step_helper, kits_step_data)
        return step_helper

    def getStepHelperForNewTemplateBySample(self, run_type_id, sampleset_id, step_helper_type=StepHelperType.CREATE_NEW_TEMPLATE_BY_SAMPLE):
        '''
            Start Plan by Sample creation with "Add new Template"
        '''
        step_helper = StepHelper(sh_type=step_helper_type)
        ionReporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        kits_step_data = step_helper.steps[StepNames.KITS]

        sampleset = SampleSet.objects.filter(pk__in=sampleset_id.split(','))[0]
        runType = RunType.objects.get(pk=run_type_id)
        step_helper.parentName = runType.description

        step_helper.isParentSystem = False

        ionReporter_step_data.savedFields['sampleset_id'] = sampleset_id
        self._updateApplicationStepData(runType, step_helper, application_step_data)

        barcodeSet = None
        for item in sampleset.samples.all().order_by("sample__displayedName"):
            if item.dnabarcode:
                barcodeSet = item.dnabarcode.name
                break
        kits_step_data.savedFields[KitsFieldNames.BARCODE_ID] = barcodeSet

        if sampleset.libraryPrepInstrument == 'chef':
            kits_step_data.savedFields[KitsFieldNames.TEMPLATE_KIT_TYPE] = KitsFieldNames.ION_CHEF

        if sampleset.libraryPrepKitName and sampleset.libraryPrepKitName != '0':
            kits_step_data.savedFields[KitsFieldNames.LIBRARY_KIT_NAME] = sampleset.libraryPrepKitName
        return step_helper

    def updateTemplateSpecificStepHelper(self, step_helper, planned_experiment):
        '''
            Updates the template specific step helper with template specific info from the planned experiment.
        '''
        # logger.debug("ENTER step_helper_db_loader.updateTemplateSpecificStepHelper()")

        if step_helper.isTemplateBySample():
            save_template_step_data = step_helper.steps[StepNames.SAVE_TEMPLATE_BY_SAMPLE]
        else:
            save_template_step_data = step_helper.steps[StepNames.SAVE_TEMPLATE]

        planDisplayedName = getPlanDisplayedName(planned_experiment)

        if step_helper.sh_type == StepHelperType.COPY_TEMPLATE:
            save_template_step_data.savedFields[SaveTemplateStepDataFieldNames.TEMPLATE_NAME] = "Copy of " + planDisplayedName
        else:
            save_template_step_data.savedFields[SaveTemplateStepDataFieldNames.TEMPLATE_NAME] = planDisplayedName

        save_template_step_data.savedFields[SaveTemplateStepDataFieldNames.SET_AS_FAVORITE] = planned_experiment.isFavorite

        save_template_step_data.savedFields[SaveTemplateStepDataFieldNames.NOTE] = planned_experiment.get_notes()

        LIMS_meta = planned_experiment.get_LIMS_meta()
        # logger.debug("step_helper_db_loader.updateTemplateSpecificStepHelper() type(LIMS_meta)=%s; LIMS_meta=%s" %(type(LIMS_meta), LIMS_meta))

        if (type(LIMS_meta) is list):
            # convert list to string
            save_template_step_data.savedFields[SaveTemplateStepDataFieldNames.LIMS_META] = ''.join(LIMS_meta)
        else:
            save_template_step_data.savedFields[SaveTemplateStepDataFieldNames.LIMS_META] = LIMS_meta
        # logger.debug("step_helper_db_loader.updateTemplateSpecificStepHelper() LIMS_META=%s" %(save_template_step_data.savedFields[SaveTemplateStepDataFieldNames.LIMS_META]))

        save_template_step_data.savedObjects[SaveTemplateStepDataFieldNames.META] = self._metaDataFromPlan(step_helper, planned_experiment)

    def updatePlanSpecificStepHelper(self, step_helper, planned_experiment, set_template_name=False):
        '''
            Updates the plan specific step helper with plan specific info from the planned experiment.

            If the planned experiment is a template and you'd like the originating template name to show up
            in the save plan page pass in set_template_name=True
        '''
        # logger.debug("ENTER step_helper_db_loader.updatePlanSpecificStepHelper()")

        planDisplayedName = getPlanDisplayedName(planned_experiment)

        if set_template_name:
            if step_helper.isTemplateBySample():
                step_helper.steps[StepNames.SAVE_TEMPLATE_BY_SAMPLE].savedFields[SaveTemplateStepDataFieldNames.TEMPLATE_NAME] = planDisplayedName
            else:
                step_helper.steps[StepNames.SAVE_TEMPLATE].savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME] = planDisplayedName

        save_plan_step_data = step_helper.steps[StepNames.SAVE_PLAN]

        # Add a "copy of" if we're copying.
        if step_helper.isCopy():
            save_plan_step_data.savedFields[SavePlanFieldNames.PLAN_NAME] = "Copy of " + planDisplayedName
        else:
            save_plan_step_data.savedFields[SavePlanFieldNames.PLAN_NAME] = planDisplayedName

        save_plan_step_data.savedFields[SavePlanFieldNames.NOTE] = planned_experiment.get_notes()

        LIMS_meta = planned_experiment.get_LIMS_meta()
        # logger.debug("step_helper_db_loader.updatePlanSpecificStepHelper() type(LIMS_meta)=%s; LIMS_meta=%s" %(type(LIMS_meta), LIMS_meta))

        if (type(LIMS_meta) is list):
            # convert list to string
            save_plan_step_data.savedFields[SavePlanFieldNames.LIMS_META] = ''.join(LIMS_meta)
        else:
            save_plan_step_data.savedFields[SavePlanFieldNames.LIMS_META] = LIMS_meta
        # logger.debug("step_helper_db_loader.updatePlanSpecificStepHelper() LIMS_META=%s" %(save_plan_step_data.savedFields[SavePlanFieldNames.LIMS_META]))

        save_plan_step_data.savedObjects[SavePlanFieldNames.META] = self._metaDataFromPlan(step_helper, planned_experiment)

        barcodeSet = planned_experiment.get_barcodeId()
        save_plan_step_data.savedFields[SavePlanFieldNames.BARCODE_SET] = barcodeSet
        if barcodeSet:
            # make sure barcodeSet is included in selection even if it's inactive
            if barcodeSet not in save_plan_step_data.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS]:
                available_dnaBarcodes = dnaBarcode.objects.filter(Q(active=True) | Q(name=barcodeSet))
                save_plan_step_data.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS] = list(available_dnaBarcodes.values_list('name', flat=True).distinct().order_by('name'))
                all_barcodes = {}
                for bc in available_dnaBarcodes.order_by('name', 'index').values('name', 'id_str', 'sequence'):
                    all_barcodes.setdefault(bc['name'], []).append(bc)
                save_plan_step_data.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_BARCODES] = json.dumps(all_barcodes)

        save_plan_step_data.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = planned_experiment.get_library()
        save_plan_step_data.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = planned_experiment.get_bedfile()
        save_plan_step_data.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = planned_experiment.get_regionfile()
        save_plan_step_data.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] = planned_experiment.runType
        save_plan_step_data.prepopulatedFields[SavePlanFieldNames.IR_WORKFLOW] = planned_experiment.irworkflow

        isOncoSameSample = False

        if (planned_experiment.runType == "AMPS_DNA_RNA"):
            sample_count = planned_experiment.get_sample_count()
            barcode_count = getPlanBarcodeCount(planned_experiment)
            isOncoSameSample = (sample_count * 2 == barcode_count)

        save_plan_step_data.savedFields[SavePlanFieldNames.ONCO_SAME_SAMPLE] = isOncoSameSample

        # logger.debug("step_helper_db_loader.updatePlanSpecificStepHelper isOncoSameSample=%s" %(isOncoSameSample))

        # add IonReporter parameters
        irInfo = self._getIRinfo(planned_experiment)
        if irInfo:
            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.SELECTED_IR] = irInfo[SavePlanFieldNames.SELECTED_IR]
            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.IR_CONFIG_JSON] = irInfo[SavePlanFieldNames.IR_CONFIG_JSON]
            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.SETID_SUFFIX] = irInfo.get(SavePlanFieldNames.SETID_SUFFIX)

            logger.debug("step_helper_db_loader.updatePlanSpecificStepHelper() irInfo=%s" % (irInfo))

            userInputInfo = irInfo.get("userInputInfo", [])
            iru_hasOncoData = False
            iru_hasPgsData = False
            if userInputInfo:
                for info in userInputInfo:
                    if info.get("cancerType", "") or info.get("cellularityPct", ""):
                        iru_hasOncoData = True
                    if info.get("biopsyDays", "") or info.get("coupleID", "") or info.get("embryoID", ""):
                        iru_hasPgsData = True
            if planned_experiment.categories:
                if "Oncomine" in planned_experiment.categories or "Onconet" in planned_experiment.categories:
                    iru_hasOncoData = True
            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.HAS_ONCO_DATA] = iru_hasOncoData
            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.HAS_PGS_DATA] = iru_hasPgsData

        samplesTable = self._getSamplesTable_from_plan(planned_experiment, step_helper, irInfo)
        if samplesTable:
            # if a plan is created from IR-enabled template, userInputInfo doesn't exist yet so need to add irWorkflow
            if planned_experiment.irworkflow and not (irInfo and irInfo['userInputInfo']):
                for sampleDict in samplesTable:
                    sampleDict['irWorkflow'] = planned_experiment.irworkflow

            save_plan_step_data.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = json.dumps(samplesTable)

        num_samples = len(samplesTable)
        if step_helper.isCreate():
            # initial number of samples for new plans, if greater than 1
            categories = planned_experiment.categories or ""
            if step_helper.isDualNucleotideTypeBySample():
                num_samples = 2

            if step_helper.isBarcoded():
                if "ocav2" in categories:
                    num_samples = 24
                elif "barcodes_" in categories:
                    num_samples = int([s.split('_') for s in categories.split(';') if 'barcodes_' in s][0][1])

        save_plan_step_data.prepopulatedFields[SavePlanFieldNames.NUM_SAMPLES] = num_samples

        if step_helper.isBarcoded():
            # do not copy sampleTubeLabel since a sample tube is meant for 1 run only
            save_plan_step_data.savedFields[SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL] = "" if step_helper.isCopy() else planned_experiment.sampleTubeLabel
            save_plan_step_data.savedFields[SavePlanFieldNames.CHIP_BARCODE_LABEL] = "" if step_helper.isCopy() else planned_experiment.get_chipBarcode()


    def updateUniversalStepHelper(self, step_helper, planned_experiment):
        '''
            Update a step helper with info from planned experiment that applies to both plans and templates.
        '''
        # logger.debug("ENTER step_helper_db_loader.updateUniversalStepHelper()")

        # export_step_data = step_helper.steps[StepNames.EXPORT]
        ionreporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        application_step_data = step_helper.steps[StepNames.APPLICATION]
        kits_step_data = step_helper.steps[StepNames.KITS]
        reference_step_data = step_helper.steps[StepNames.REFERENCE]
        plugins_step_data = step_helper.steps[StepNames.PLUGINS]
        analysisParams_step_data = step_helper.steps[StepNames.ANALYSIS_PARAMS]

        # if not step_helper.isPlanBySample():
        #     ionreporter_step_data = step_helper.steps[StepNames.IONREPORTER]
        appl_product = None

        # application_step_data.updateFromStep(export_step_data)

        self._updateUniversalStep_ionReporterData(step_helper, planned_experiment, ionreporter_step_data)

        appl_product = self._updateUniversalStep_applicationData(step_helper, planned_experiment, application_step_data)

        logger.debug("step_helper_db_loader.updateUniversalStepHelper() planned_experiment.id=%d; applProduct.productCode=%s" % (planned_experiment.id, appl_product.productCode))

        self._updateUniversalStep_kitData(step_helper, planned_experiment, appl_product, application_step_data, kits_step_data)

        self._updateUniversalStep_referenceData(step_helper, planned_experiment, appl_product, reference_step_data)

        self._updateUniversalStep_analysisParamsData(step_helper, planned_experiment, appl_product, application_step_data, kits_step_data, analysisParams_step_data)

        if step_helper.isEdit() or step_helper.isEditRun() or step_helper.isCopy():
            self._updateUniversalStep_applicationData_for_edit(step_helper, planned_experiment, application_step_data)

            # During plan editing, kits_step_data.updateFromStep() is executed before step_helper_db_loader.updateUniversalStepHelper().
            # This results in savedObjects[ApplicationFieldNames.APPL_PRODUCT] not getting set.
            # WORKAROUND: The following is a workaround to ensure prepopulatedFields are set for the Kits chevron
            self._updateUniversalStep_kitData_for_edit(step_helper, planned_experiment, appl_product, application_step_data, kits_step_data)

            self._updateUniversalStep_referenceData_for_edit(step_helper, planned_experiment, appl_product, application_step_data, reference_step_data)

        self._updateUniversalStep_pluginData_ionreporterData(step_helper, planned_experiment, appl_product, plugins_step_data, ionreporter_step_data)

        logger.debug("PLUGINS ARE: %s" % str(plugins_step_data.savedFields[StepNames.PLUGINS]))

        qc_values = planned_experiment.qcValues.all()
        if step_helper.isTemplate():
            if step_helper.isTemplateBySample():
                target_step = StepNames.SAVE_TEMPLATE_BY_SAMPLE
            else:
                target_step = StepNames.SAVE_TEMPLATE
        elif step_helper.isPlanBySample():
            target_step = StepNames.SAVE_PLAN_BY_SAMPLE
        else:
            target_step = StepNames.SAVE_PLAN

        for qc_value in qc_values:
            step_helper.steps[target_step].savedFields[qc_value.qcName] = PlannedExperimentQC.objects.get(plannedExperiment__pk=planned_experiment.pk, qcType__pk=qc_value.pk).threshold

        logger.debug("QCs ARE: %s" % str(step_helper.steps[target_step].savedFields))

        step_helper.steps[StepNames.OUTPUT].savedFields[OutputFieldNames.PROJECTS] = []
        projects = planned_experiment.projects.all()
        for project in projects:
            step_helper.steps[StepNames.OUTPUT].savedFields[OutputFieldNames.PROJECTS].append(project.pk)

    def _updateUniversalStep_ionReporterData(self, step_helper, planned_experiment, ionreporter_step_data):
        ionreporter_step_data.savedFields[IonReporterFieldNames.SAMPLE_GROUPING] = planned_experiment.sampleGrouping.pk if planned_experiment.sampleGrouping else None
        ionreporter_step_data.savedObjects[IonReporterFieldNames.SAMPLE_GROUPING] = planned_experiment.sampleGrouping if planned_experiment.sampleGrouping else None
        ionreporter_step_data.prepopulatedFields[IonReporterFieldNames.CATEGORIES] = planned_experiment.categories

    def _updateUniversalStep_applicationData(self, step_helper, planned_experiment, application_step_data):
        if RunType.objects.filter(runType=planned_experiment.runType).count() > 0:
            selectedRunType = RunType.objects.get(runType=planned_experiment.runType)
        else:
            selectedRunType = RunType.objects.get(runType="GENS")

        application_step_data.savedFields[ApplicationFieldNames.RUN_TYPE] = selectedRunType.pk
        application_step_data.savedObjects[ApplicationFieldNames.RUN_TYPE] = selectedRunType

        if planned_experiment.applicationGroup in selectedRunType.applicationGroups.all():
            selectedApplicationGroup = planned_experiment.applicationGroup
        else:
            # if no application group is selected, pick the first one associated with runType
            selectedApplicationGroup = selectedRunType.applicationGroups.first()

        if selectedApplicationGroup:
            application_step_data.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME] = selectedApplicationGroup.name

        try:
            appl_product = ApplProduct.objects.get(applType__runType=selectedRunType.runType, applicationGroup=selectedApplicationGroup, isDefault=True, isActive=True, isVisible=True)
        except:
            appl_product = ApplProduct.objects.get(applType__runType=selectedRunType.runType, isDefault=True, isActive=True, isVisible=True)

        # logger.debug("step_helper_db_loader._updateUniversalStep_applicationData() saving appl_product now!!")
        application_step_data.savedObjects[ApplicationFieldNames.APPL_PRODUCT] = appl_product
        
        application_step_data.prepopulatedFields[ApplicationFieldNames.CATEGORIES] = planned_experiment.categories
        # logger.debug(" step_helper_db_loader_updateUniversalStep_applicationData() helper.sh_type=%s   application_step_data.categories=%s" %(step_helper.sh_type, application_step_data.prepopulatedFields[ApplicationFieldNames.CATEGORIES]))

        step = step_helper.steps.get(StepNames.REFERENCE, "")
        if step:
            self._updateStep_with_applicationData(step, application_step_data)
            step.prepopulatedFields[ReferenceFieldNames.RUN_TYPE] = application_step_data.savedObjects[ApplicationFieldNames.RUN_TYPE].runType

        step = step_helper.steps.get(StepNames.ANALYSIS_PARAMS, "")
        if step:
            step.prepopulatedFields[AnalysisParamsFieldNames.RUN_TYPE] = application_step_data.savedObjects[ApplicationFieldNames.RUN_TYPE].runType
            step.prepopulatedFields[AnalysisParamsFieldNames.APPLICATION_GROUP_NAME] = application_step_data.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME]

        step = step_helper.steps.get(StepNames.BARCODE_BY_SAMPLE, "")
        if step:
            self._updateStep_with_applicationData(step, application_step_data)

        step = step_helper.steps.get(StepNames.SAVE_PLAN, "")
        if step:
            self._updateStep_with_applicationData(step, application_step_data)

        step = step_helper.steps.get(StepNames.SAVE_PLAN_BY_SAMPLE, "")
        if step:
            self._updateStep_with_applicationData(step, application_step_data)

        return appl_product

    def _updateStep_with_applicationData(self, step, application_step_data):
        if step and application_step_data:
            step.prepopulatedFields[SavePlanFieldNames.APPLICATION_GROUP_NAME] = application_step_data.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME]

    def _updateUniversalStep_applicationData_for_edit(self, step_helper, planned_experiment, application_step_data):
        application_step_data.prepopulatedFields[ApplicationFieldNames.PLAN_STATUS] = planned_experiment.planStatus
        
        applicationGroupObj = planned_experiment.applicationGroup if planned_experiment.applicationGroup else None

        categorizedApplProducts = None
        if (applicationGroupObj):
            categorizedApplProducts = ApplProduct.objects.filter(isActive=True,
                                                          applType__runType=planned_experiment.runType, applicationGroup=applicationGroupObj).exclude(categories = "")
        else:
            categorizedApplProducts = ApplProduct.objects.filter(isActive=True,
                                                          applType__runType=planned_experiment.runType).exclude(categories = "")
            
        if categorizedApplProducts:
            application_step_data.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS_CATEGORIZED] = categorizedApplProducts
        else:
            application_step_data.prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS_CATEGORIZED] = None


    def _updateUniversalStep_kitData(self, step_helper, planned_experiment, appl_product, application_step_data, kits_step_data):
        application_step_data.savedObjects[ApplicationFieldNames.APPL_PRODUCT] = appl_product

        kits_step_data.savedFields[KitsFieldNames.TEMPLATE_KIT_TYPE] = "OneTouch"
        if planned_experiment.is_ionChef():
            kits_step_data.savedFields[KitsFieldNames.TEMPLATE_KIT_TYPE] = "IonChef"

        kits_step_data.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME] = planned_experiment.templatingKitName
        kits_step_data.savedFields[KitsFieldNames.CONTROL_SEQUENCE] = planned_experiment.controlSequencekitname
        kits_step_data.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT] = planned_experiment.samplePrepKitName
        kits_step_data.savedFields[KitsFieldNames.BARCODE_ID] = planned_experiment.get_barcodeId()

        chipType = planned_experiment.get_chipType()
        kits_step_data.savedFields[KitsFieldNames.CHIP_TYPE] = '318' if chipType == '318v2' else chipType
        kits_step_data.prepopulatedFields[KitsFieldNames.IS_CHIP_TYPE_REQUIRED] = step_helper.isPlan()

        kits_step_data.savedFields[KitsFieldNames.FLOWS] = planned_experiment.get_flows()
        kits_step_data.savedFields[KitsFieldNames.LIBRARY_READ_LENGTH] = planned_experiment.libraryReadLength
        kits_step_data.savedFields[KitsFieldNames.READ_LENGTH] = planned_experiment.libraryReadLength
        kits_step_data.savedFields[KitsFieldNames.TEMPLATING_SIZE] = planned_experiment.templatingSize

        kits_step_data.savedFields[KitsFieldNames.FORWARD_3_PRIME_ADAPTER] = planned_experiment.get_forward3primeadapter()
        kits_step_data.savedFields[KitsFieldNames.FLOW_ORDER] = planned_experiment.experiment.flowsInOrder
        kits_step_data.savedFields[KitsFieldNames.LIBRARY_KEY] = planned_experiment.get_libraryKey()
        tfKey = planned_experiment.get_tfKey()
        if tfKey:
            kits_step_data.savedFields[KitsFieldNames.TF_KEY] = tfKey
        kits_step_data.savedFields[KitsFieldNames.LIBRARY_KIT_NAME] = planned_experiment.get_librarykitname()
        kits_step_data.savedFields[KitsFieldNames.SEQUENCE_KIT_NAME] = planned_experiment.get_sequencekitname()
        kits_step_data.savedFields[KitsFieldNames.IS_DUPLICATED_READS] = planned_experiment.is_duplicateReads()
        kits_step_data.savedFields[KitsFieldNames.BASE_RECALIBRATE] = planned_experiment.get_base_recalibration_mode()
        kits_step_data.savedFields[KitsFieldNames.REALIGN] = planned_experiment.do_realign()
        kits_step_data.savedFields[KitsFieldNames.SAMPLE_PREP_PROTOCOL] = planned_experiment.samplePrepProtocol
        
        kits_step_data.prepopulatedFields[KitsFieldNames.PLAN_CATEGORIES] = planned_experiment.categories or ''
        kits_step_data.prepopulatedFields[KitsFieldNames.IS_BARCODE_KIT_SELECTION_REQUIRED] = appl_product.isBarcodeKitSelectionRequired

    def _updateUniversalStep_kitData_for_edit(self, step_helper, planned_experiment, appl_product, application_step_data, kits_step_data):
        application_step_data.savedObjects[ApplicationFieldNames.APPL_PRODUCT] = appl_product

        # no chip type selection for sequenced run
        if step_helper.isEditRun():
            kits_step_data.prepopulatedFields[KitsFieldNames.CHIP_TYPES] = Chip.objects.filter(name=kits_step_data.savedFields[KitsFieldNames.CHIP_TYPE])

        available_dnaBarcodes = dnaBarcode.objects.filter(Q(active=True))

        # if editing a sequenced run old/obsolete chipType and kits must be included
        if step_helper.isEditRun() or step_helper.isEdit():
            kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] |= KitInfo.objects.filter(name=kits_step_data.savedFields[KitsFieldNames.CONTROL_SEQUENCE])
            kits_step_data.prepopulatedFields[KitsFieldNames.SAMPLE_PREP_KITS] |= KitInfo.objects.filter(name=kits_step_data.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT])
            kits_step_data.prepopulatedFields[KitsFieldNames.LIB_KITS] |= KitInfo.objects.filter(name=kits_step_data.savedFields[KitsFieldNames.LIBRARY_KIT_NAME])
            kits_step_data.prepopulatedFields[KitsFieldNames.SEQ_KITS] |= KitInfo.objects.filter(name=kits_step_data.savedFields[KitsFieldNames.SEQUENCE_KIT_NAME])

            savedtemplatekit = KitInfo.objects.filter(name=kits_step_data.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME])
            kits_step_data.prepopulatedFields[KitsFieldNames.TEMPLATE_KITS] |= savedtemplatekit
            oneTouchKits = kits_step_data.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ONE_TOUCH][KitsFieldNames.KIT_VALUES]
            ionChefKits = kits_step_data.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ION_CHEF][KitsFieldNames.KIT_VALUES]
            kits_step_data.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ONE_TOUCH][KitsFieldNames.KIT_VALUES] |= savedtemplatekit.filter(kitType__in=oneTouchKits.values_list('kitType', flat=True))
            kits_step_data.prepopulatedFields[KitsFieldNames.TEMPLATE_KIT_TYPES][KitsFieldNames.ION_CHEF][KitsFieldNames.KIT_VALUES] |= savedtemplatekit.filter(kitType__in=ionChefKits.values_list('kitType', flat=True))

            available_dnaBarcodes = dnaBarcode.objects.filter(Q(active=True) | Q(name=planned_experiment.get_barcodeId()))

        # if step_helper.isEdit():
        logger.debug("step_helper_db_loader._updateUniversalStep_kitData_for_edit() - isEdit - appl_product.barcodeKitSelectableType=%s" % (appl_product.barcodeKitSelectableType))
        if appl_product.applType.runType in ["AMPS", "AMPS_EXOME"]:
            kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in=["", "DNA", "AMPS_ANY"], isActive=True).order_by("name")
        elif appl_product.applType.runType in ["AMPS_RNA"]:
            kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in=["", "RNA", "AMPS_ANY"], isActive=True).order_by("name")
        elif appl_product.applType.runType in ["RNA"]:
            kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType="RNA", isActive=True).order_by("name")
        elif appl_product.applType.runType in ["AMPS_DNA_RNA"]:
            kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in=["", "DNA", "RNA", "AMPS_ANY"], isActive=True).order_by("name")
        else:
            kits_step_data.prepopulatedFields[KitsFieldNames.CONTROL_SEQ_KITS] = KitInfo.objects.filter(kitType='ControlSequenceKit', applicationType__in=["", "DNA"], isActive=True).order_by("name")

        kits_step_data.prepopulatedFields[KitsFieldNames.BARCODES] = list(available_dnaBarcodes.values('name').distinct().order_by('name'))
        kits_step_data.prepopulatedFields[KitsFieldNames.BARCODES_SUBSET] = \
            list(available_dnaBarcodes.filter(type__in=appl_product.barcodeKitSelectableTypes_list).values('name').distinct().order_by('name'))

    def _updateUniversalStep_referenceData(self, step_helper, planned_experiment, appl_product, reference_step_data):
        # logger.debug("ENTER step_helper_db_loader._updateUniversalStep_referenceData()...")

        reference_step_data.savedFields[ReferenceFieldNames.TARGET_BED_FILE] = planned_experiment.get_bedfile()
        reference_step_data.savedFields[ReferenceFieldNames.REFERENCE] = planned_experiment.get_library()
        reference_step_data.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] = planned_experiment.get_regionfile()

        sseBedFile = planned_experiment.get_sseBedFile()
        targetRegionBEDFile = reference_step_data.savedFields[ReferenceFieldNames.TARGET_BED_FILE]
        if sseBedFile and targetRegionBEDFile:
            reference_step_data.prepopulatedFields[ReferenceFieldNames.SSE_BED_FILE_DICT][targetRegionBEDFile.split('/')[-1]] = sseBedFile

        mixedTypeRNA_targetRegion = planned_experiment.get_mixedType_rna_bedfile()
        reference_step_data.savedFields[ReferenceFieldNames.MIXED_TYPE_RNA_TARGET_BED_FILE] = "" if mixedTypeRNA_targetRegion is None else mixedTypeRNA_targetRegion
        mixedTypeRNA_reference = planned_experiment.get_mixedType_rna_library()
        reference_step_data.savedFields[ReferenceFieldNames.MIXED_TYPE_RNA_REFERENCE] = "" if mixedTypeRNA_reference is None else mixedTypeRNA_reference
        mixedTypeRNA_hotSpot = planned_experiment.get_mixedType_rna_regionfile()
        reference_step_data.savedFields[ReferenceFieldNames.MIXED_TYPE_RNA_HOT_SPOT_BED_FILE] = "" if mixedTypeRNA_hotSpot is None else mixedTypeRNA_hotSpot

        reference_step_data.savedFields[ReferenceFieldNames.SAME_REF_INFO_PER_SAMPLE] = self._getIsSameRefInfoPerSample(step_helper, planned_experiment)

        logger.debug("step_helper_db_loader._updateUniversalStep_referenceData() REFERENCE savedFields=%s" % (reference_step_data.savedFields))
        logger.debug("step_helper_db_loader._updateUniversalStep_referenceData() REFERENCE appl_product=%s" % (appl_product))

        reference_step_data.prepopulatedFields[ReferenceFieldNames.SHOW_HOT_SPOT_BED] = True
        if appl_product and not appl_product.isHotspotRegionBEDFileSuppported:
            reference_step_data.prepopulatedFields[ReferenceFieldNames.SHOW_HOT_SPOT_BED] = False

        # if the plan or template has pre-selected reference info, it is possible that it is not found in db in this TS instance
        # a plan's or template's pre-selected reference info trumps applProducts default selection values!
        if reference_step_data.savedFields[ReferenceFieldNames.REFERENCE]:

            reference_step_data.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = True
            if reference_step_data.savedFields[ReferenceFieldNames.REFERENCE] in [ref.short_name for ref in reference_step_data.prepopulatedFields[ReferenceFieldNames.REFERENCES]]:
                reference_step_data.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = False
            else:
                logger.debug("at step_helper_db_loader.updateUniversalStepHelper() REFERENCE_MISSING saved reference=%s" % (reference_step_data.savedFields[ReferenceFieldNames.REFERENCE]));
        else:
            reference_step_data.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = False

        if reference_step_data.savedFields[ReferenceFieldNames.TARGET_BED_FILE]:

            reference_step_data.prepopulatedFields[ReferenceFieldNames.TARGET_BED_FILE_MISSING] = True
            if reference_step_data.savedFields[ReferenceFieldNames.TARGET_BED_FILE] in reference_step_data.file_dict[ReferenceFieldNames.BED_FILE_FULL_PATHS] or\
               reference_step_data.savedFields[ReferenceFieldNames.TARGET_BED_FILE] in reference_step_data.file_dict[ReferenceFieldNames.BED_FILE_PATHS]:
                reference_step_data.prepopulatedFields[ReferenceFieldNames.TARGET_BED_FILE_MISSING] = False
            else:
                logger.debug("at step_helper_db_loader.updateUniversalStepHelper() TARGET_BED_FILE_MISSING saved target=%s" % (reference_step_data.savedFields[ReferenceFieldNames.TARGET_BED_FILE]));
        else:
            reference_step_data.prepopulatedFields[ReferenceFieldNames.TARGET_BED_FILE_MISSING] = False

        if reference_step_data.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE]:

            reference_step_data.prepopulatedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE_MISSING] = True
            if reference_step_data.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] in reference_step_data.file_dict[ReferenceFieldNames.HOT_SPOT_FULL_PATHS] or\
               reference_step_data.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] in reference_step_data.file_dict[ReferenceFieldNames.HOT_SPOT_PATHS]:
                reference_step_data.prepopulatedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE_MISSING] = False
            else:
                logger.debug("at step_helper_db_loader.updateUniversalStepHelper() HOT_SPOT_BED_FILE_MISSING saved hotSpot=%s" % (reference_step_data.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE]));
                logger.debug("HOT_SPOT MISSING BED_FILE_FULL_PATHS=%s" % (reference_step_data.file_dict[ReferenceFieldNames.HOT_SPOT_FULL_PATHS]))
                logger.debug("HOT_SPOT MISSING BED_FILE_PATHS=%s" % (reference_step_data.file_dict[ReferenceFieldNames.HOT_SPOT_PATHS]))

        else:
            reference_step_data.prepopulatedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE_MISSING] = False

        stepHelper_type = step_helper.sh_type

        logger.debug("step_helper_db_loader._updateUniversalStep_referenceData() stepHelper_type=%s; reference_step_data.savedFields=%s" % (stepHelper_type, reference_step_data.savedFields))

        if stepHelper_type == StepHelperType.CREATE_NEW_PLAN_BY_SAMPLE or stepHelper_type == StepHelperType.EDIT_PLAN_BY_SAMPLE or stepHelper_type == StepHelperType.COPY_PLAN_BY_SAMPLE:
            barcoding_step = step_helper.steps[StepNames.BARCODE_BY_SAMPLE]
            save_plan_step = step_helper.steps[StepNames.SAVE_PLAN_BY_SAMPLE]

            barcoding_step.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = reference_step_data.savedFields.get(ReferenceFieldNames.REFERENCE, "")
            barcoding_step.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.TARGET_BED_FILE, "")
            barcoding_step.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.HOT_SPOT_BED_FILE, "")
            # logger.debug("step_helper_db_loader._updateUniversalStep_referenceData() stepHelper_type=%s; barcoding_step.savedFields=%s" %(stepHelper_type, barcoding_step.savedFields))
            # logger.debug("step_helper_db_loader._updateUniversalStep_referenceData() stepHelper_type=%s; step_helper=%s; barcoding_step=%s" %(stepHelper_type, step_helper, barcoding_step))

            save_plan_step.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = reference_step_data.savedFields.get(ReferenceFieldNames.REFERENCE, "")
            save_plan_step.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.TARGET_BED_FILE, "")
            save_plan_step.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.HOT_SPOT_BED_FILE, "")

            barcoding_step.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = reference_step_data
            save_plan_step.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = reference_step_data

        elif stepHelper_type == StepHelperType.CREATE_NEW_PLAN or stepHelper_type == StepHelperType.COPY_PLAN or stepHelper_type == StepHelperType.EDIT_PLAN or stepHelper_type == StepHelperType.EDIT_RUN:
            save_plan_step_data = step_helper.steps[StepNames.SAVE_PLAN]

            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = reference_step_data.savedFields.get(ReferenceFieldNames.REFERENCE, "")
            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.TARGET_BED_FILE, "")
            save_plan_step_data.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = reference_step_data.savedFields.get(ReferenceFieldNames.HOT_SPOT_BED_FILE, "")

            save_plan_step_data.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = reference_step_data

    def _updateUniversalStep_referenceData_for_edit(self, step_helper, planned_experiment, appl_product, application_step_data, reference_step_data):
        # logger.debug("_updateUniversalStep_referenceData_for_edit appl_product=%s" %(appl_product))

        reference_step_data.prepopulatedFields[ReferenceFieldNames.SHOW_HOT_SPOT_BED] = True
        reference_step_data.prepopulatedFields[ReferenceFieldNames.REQUIRE_TARGET_BED_FILE] = False
        if appl_product:
            reference_step_data.prepopulatedFields[ReferenceFieldNames.SHOW_HOT_SPOT_BED] = appl_product.isHotspotRegionBEDFileSuppported
            reference_step_data.prepopulatedFields[ReferenceFieldNames.REQUIRE_TARGET_BED_FILE] = appl_product.isTargetRegionBEDFileSelectionRequiredForRefSelection

        reference_step_data.prepopulatedFields[ReferenceFieldNames.PLAN_STATUS] = planned_experiment.planStatus

    def _updateUniversalStep_analysisParamsData(self, step_helper, planned_experiment, appl_product,  application_step_data, kits_step_data, analysisParams_step_data):
        self. _updateUniversalStep_analysisParamsData_basic(step_helper, planned_experiment, appl_product,  application_step_data, kits_step_data, analysisParams_step_data)

        # current chip selection
        chipType = planned_experiment.get_chipType()
        if Chip.objects.filter(name=chipType).count() == 0:
            chipType = chipType[:3]

        # there is no analysisArgs db definition for 318v2
        analysisParams_step_data.prepopulatedFields[AnalysisParamsFieldNames.CHIP_TYPE] = '318' if chipType == '318v2' else chipType

        logger.debug("step_helper_db_loader._updateUniversalStep_analysisParamsData() chipType=%s;" % (chipType))

        applicationGroupName = planned_experiment.applicationGroup.name if planned_experiment.applicationGroup else ""

        best_match_entry = AnalysisArgs.best_match(chipType, planned_experiment.get_sequencekitname(),
                                                   planned_experiment.templatingKitName, planned_experiment.get_librarykitname(), planned_experiment.samplePrepKitName, None, planned_experiment.runType, applicationGroupName)

        # system templates may not have any analysis args pre-selected
        doesPlanHaveCustomAnalysisArgs = self._doesPlanHaveCustomAnalysisArgs(planned_experiment)

        if doesPlanHaveCustomAnalysisArgs or not best_match_entry:
            current_selected_analysisArgs = planned_experiment.latest_eas.get_cmdline_args()
        else:
            current_selected_analysisArgs = best_match_entry.get_args()

        current_selected_analysisArgs.update({
            'description': AnalysisParamsFieldNames.AP_ENTRY_SELECTED_VALUE,
            'name': '',
            'custom_args': doesPlanHaveCustomAnalysisArgs
        })

        self. _updateUniversalStep_analysisParamsData_currentSelection(analysisParams_step_data, current_selected_analysisArgs)
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_CUSTOM] = "True" if doesPlanHaveCustomAnalysisArgs else "False"

    def _doesPlanHaveCustomAnalysisArgs(self, planned_experiment):
        latest_eas = planned_experiment.latest_eas
        if latest_eas and latest_eas.custom_args and latest_eas.have_args():
            return True
        else:
            return False

    def _updateUniversalStep_analysisParamsData_currentSelection(self, analysisParams_step_data, current_selected_analysisArgs):

        analysisParams_step_data.savedObjects[AnalysisParamsFieldNames.AP_ENTRY_SELECTED] = current_selected_analysisArgs
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_BEADFIND_SELECTED] = current_selected_analysisArgs['beadfindargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_ANALYSISARGS_SELECTED] = current_selected_analysisArgs['analysisargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_PREBASECALLER_SELECTED] = current_selected_analysisArgs['prebasecallerargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_CALIBRATE_SELECTED] = current_selected_analysisArgs['calibrateargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_BASECALLER_SELECTED] = current_selected_analysisArgs['basecallerargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_ALIGNMENT_SELECTED] = current_selected_analysisArgs['alignmentargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_IONSTATS_SELECTED] = current_selected_analysisArgs['ionstatsargs']

        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_BEADFIND_SELECTED] = current_selected_analysisArgs['thumbnailbeadfindargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_ANALYSISARGS_SELECTED] = current_selected_analysisArgs['thumbnailanalysisargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_PREBASECALLER_SELECTED] = current_selected_analysisArgs['prethumbnailbasecallerargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_CALIBRATE_SELECTED] = current_selected_analysisArgs['thumbnailcalibrateargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_BASECALLER_SELECTED] = current_selected_analysisArgs['thumbnailbasecallerargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_ALIGNMENT_SELECTED] = current_selected_analysisArgs['thumbnailalignmentargs']
        analysisParams_step_data.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_IONSTATS_SELECTED] = current_selected_analysisArgs['thumbnailionstatsargs']

    def _updateUniversalStep_analysisParamsData_basic(self, step_helper, planned_experiment, appl_product,  application_step_data, kits_step_data, analysisParams_step_data):
        # current chip selection
        chipType = planned_experiment.get_chipType()
        # current runType
        runType = planned_experiment.runType
        # current application group
        applicationGroup = planned_experiment.applicationGroup
        applicationGroupName = applicationGroup.name if applicationGroup else ""

        # return a list of entries
        possible_match_entries = AnalysisArgs.possible_matches(chipType, planned_experiment.get_sequencekitname(),
                                                               planned_experiment.templatingKitName, planned_experiment.get_librarykitname(), planned_experiment.samplePrepKitName, None, runType, applicationGroupName)
        best_match_entry = planned_experiment.get_default_cmdline_args_obj()

        for ap in possible_match_entries:
            if ap.name == best_match_entry.name:
                ap.name = AnalysisParamsFieldNames.AP_ENTRY_BEST_MATCH_PLAN_VALUE
                ap.best_match = True

        logger.debug("step_helper_db_loader._updateUniversalStep_analysisParamsData_basic() ANALYSIS_PARAMS possible_match_entries=%s" % (possible_match_entries))

        analysisParams_step_data.prepopulatedFields[AnalysisParamsFieldNames.AP_ENTRIES] = possible_match_entries
        analysisParams_step_data.prepopulatedFields[AnalysisParamsFieldNames.AP_DISPLAYED_NAMES] = [ap.description for ap in possible_match_entries]

        logger.debug("step_helper_db_loader._updateUniversalStep_analysisParamsData_basic() chipType=%s; runType=%s; applicationGroupName=%s" % (chipType, runType, applicationGroupName))

    def _getIsSameRefInfoPerSample(self, step_helper, planned_experiment):
        stepHelper_type = step_helper.sh_type

        if stepHelper_type in [StepHelperType.EDIT_PLAN_BY_SAMPLE, StepHelperType.COPY_PLAN_BY_SAMPLE, StepHelperType.COPY_PLAN, StepHelperType.EDIT_PLAN, StepHelperType.EDIT_RUN]:
            return planned_experiment.is_same_refInfo_as_defaults_per_sample()
        else:
            return True

    def _updateUniversalStep_pluginData_ionreporterData(self, step_helper, planned_experiment, appl_product, plugins_step_data, ionreporter_step_data):
        plugins_step_data.savedFields[StepNames.PLUGINS] = []
        plugins = planned_experiment.get_selectedPlugins()
        pluginIds = []
        for plugin_name, plugin_dict in plugins.items():
            # find existing plugin by plugin_name (handles plugins that were reinstalled or uninstalled)
            try:
                plugin = Plugin.objects.filter(name=plugin_name, active=True)[0]
            except:
                continue

            # we now need to show all non-IRU export plugins on the Plugins chevron
            if "ionreporter" in plugin_name.lower():
            # if PluginFieldNames.EXPORT in plugin.pluginsettings.get(PluginFieldNames.FEATURES,[]):
                if not step_helper.isPlanBySample():
                    # ionreporter_step_data.savedFields[IonReporterFieldNames.UPLOADERS].append(plugin.id)
                    pass
            else:
                pluginIds.append(plugin.id)
                plugins_step_data.savedFields[PluginFieldNames.PLUGIN_CONFIG % plugin.id] = json.dumps(plugin_dict.get(PluginFieldNames.USER_INPUT, ''), cls=JSONEncoder, separators=(',', ':'))

            if 'accountId' in plugin_dict:
                ionreporter_step_data.savedFields[IonReporterFieldNames.IR_ACCOUNT_ID] = plugin_dict.get('accountId')
                ionreporter_step_data.savedFields[IonReporterFieldNames.IR_ACCOUNT_NAME] = plugin_dict.get('accountName')
                ionreporter_step_data.savedFields[IonReporterFieldNames.IR_VERSION] = plugin_dict.get('version')
                ionreporter_step_data.savedFields[IonReporterFieldNames.IRU_UPLOAD_MODE] = plugin_dict[PluginFieldNames.USER_INPUT].get('iru_qc_option', 'no_check')
            elif PluginFieldNames.USER_INPUT in plugin_dict and 'accountId' in plugin_dict[PluginFieldNames.USER_INPUT]:
                ionreporter_step_data.savedFields[IonReporterFieldNames.IR_ACCOUNT_ID] = plugin_dict[PluginFieldNames.USER_INPUT].get('accountId')
                ionreporter_step_data.savedFields[IonReporterFieldNames.IR_ACCOUNT_NAME] = plugin_dict[PluginFieldNames.USER_INPUT].get('accountName')
                ionreporter_step_data.savedFields[IonReporterFieldNames.IRU_UPLOAD_MODE] = plugin_dict[PluginFieldNames.USER_INPUT].get('iru_qc_option')

                if 'userconfigs' in plugin.config:
                    if 'ionadmin' in plugin.config.get('userconfigs'):
                        _list = plugin.config.get('userconfigs').get('ionadmin')
                        for l in _list:
                            if l.get('id') == ionreporter_step_data.savedFields[IonReporterFieldNames.IR_ACCOUNT_ID]:
                                ionreporter_step_data.savedFields[IonReporterFieldNames.IR_VERSION] = l.get('version')

                # tag_isFactoryProvidedWorkflow is stored in userInputInfo list
                for info in plugin_dict[PluginFieldNames.USER_INPUT].get('userInputInfo', []):
                    if info['Workflow'] == planned_experiment.irworkflow:
                        step_helper.steps[StepNames.IONREPORTER].savedFields[IonReporterFieldNames.IR_ISFACTORY] = info.get('tag_isFactoryProvidedWorkflow')
                        break

        if 'IonReporterUploader' not in plugins:
            ionreporter_step_data.savedFields[IonReporterFieldNames.IR_ACCOUNT_ID] = '0'
            ionreporter_step_data.savedFields[IonReporterFieldNames.IR_ACCOUNT_NAME] = 'None'

        step_helper.steps[StepNames.IONREPORTER].savedFields[IonReporterFieldNames.IR_WORKFLOW] = planned_experiment.irworkflow
        plugins_step_data.savedFields[PluginFieldNames.PLUGIN_IDS] = ', '.join(str(v) for v in pluginIds)
        plugins_step_data.updateSavedObjectsFromSavedFields()

    def get_ir_fields_dict_from_user_input_info(self, user_input_info, sample_name, index):
        # logger.debug("ENTER step_helper_db_loader.get_ir_fields_dict_from_user_input_info()")

        if sample_name == 'barcoded--Sample':
            if index >= len(user_input_info[0].keys()):
                return dict(
                    sample="",
                    sampleDescription="",
                    sampleExternalId="",
                    barcodeId="",
                    Gender=None,
                    RelationRole=None,
                    Workflow=None,
                    setid=None,
                    cancerType=None,
                    cellularityPct=None,
                    biopsyDays=None,
                    coupleID=None,
                    embryoID=None
                )

            sample_name = user_input_info[0].keys()[index]

            # do not re-invent what has already been persisted in the JSON blob!
            barcodeSampleInfo = user_input_info[0].get(sample_name).get('barcodeSampleInfo', {})
            barcode_id_strs = sorted(user_input_info[0].get(sample_name).get('barcodeSampleInfo').keys())

            barcode_id_str = user_input_info[0].get(sample_name).get('barcodeSampleInfo').keys()[0]
            sampleDescription = user_input_info[0].get(sample_name).get('barcodeSampleInfo').get(barcode_id_str).get('description')
            externalId = user_input_info[0].get(sample_name).get('barcodeSampleInfo').get(barcode_id_str).get('externalId')

            return dict(
                sample=sample_name,
                sampleDescription=sampleDescription,
                sampleExternalId=externalId,
                barcodeSampleInfo=barcodeSampleInfo,
                barcode_id_strs=barcode_id_strs,
                Gender=None,
                RelationRole=None,
                Workflow=None,
                setid=None,
                cancerType=None,
                cellularityPct=None,
                biopsyDays=None,
                coupleID=None,
                embryoID=None
            )
        else:
            return user_input_info[index]

    def updatePlanBySampleSpecificStepHelper(self, step_helper, planned_experiment, sampleset_id=None):
        """

        """
        # logger.debug("ENTER step_helper_db_loader.updatePlanBySampleSpecificStepHelper() planned_experiment.id=%d; step_helper=%s" %(planned_experiment.id, step_helper))

        barcoding_step = step_helper.steps[StepNames.BARCODE_BY_SAMPLE]
        save_plan_step = step_helper.steps[StepNames.SAVE_PLAN_BY_SAMPLE]

        planDisplayedName = getPlanDisplayedName(planned_experiment)

        if step_helper.isCopy():
            save_plan_step.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME] = "Copy of " + planDisplayedName
        else:
            save_plan_step.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME] = planDisplayedName

        existing_plan = step_helper.isEdit() or step_helper.isCopy()

        barcoding_step.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] = planned_experiment.runType
        save_plan_step.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] = planned_experiment.runType

        isOncoSameSample = False

        if (planned_experiment.runType == "AMPS_DNA_RNA"):
            if existing_plan:
                sample_count = planned_experiment.get_sample_count()
                barcode_count = getPlanBarcodeCount(planned_experiment)

                isOncoSameSample = (sample_count * 2 == barcode_count)

        barcoding_step.savedFields[BarcodeBySampleFieldNames.ONCO_SAME_SAMPLE] = isOncoSameSample
        save_plan_step.savedFields[SavePlanFieldNames.ONCO_SAME_SAMPLE] = isOncoSameSample

        if sampleset_id:
            samplesets = SampleSet.objects.filter(pk__in=sampleset_id.split(','))
            if samplesets[0].SampleGroupType_CV:
                step_helper.steps[StepNames.APPLICATION].savedFields[ApplicationFieldNames.SAMPLE_GROUPING] = samplesets[0].SampleGroupType_CV.pk
        else:
            samplesets = planned_experiment.sampleSets.all()

        save_plan_step.savedObjects[SavePlanBySampleFieldNames.SAMPLESET] = samplesets

        sorted_sampleSetItems = []
        for sampleset in samplesets:
            sorted_sampleSetItems.extend(list(sampleset.samples.all().order_by("relationshipGroup", "nucleotideType", "sample__displayedName")))

        barcoding_step.prepopulatedFields[BarcodeBySampleFieldNames.SAMPLESET_ITEMS] = sorted_sampleSetItems
        barcoding_step.prepopulatedFields[BarcodeBySampleFieldNames.SHOW_SAMPLESET_INFO] = len(samplesets) > 1

        barcoding_step.savedFields[SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL] = planned_experiment.sampleTubeLabel
        save_plan_step.savedFields[SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL] = planned_experiment.sampleTubeLabel

        barcoding_step.savedFields[SavePlanFieldNames.CHIP_BARCODE_LABEL] = planned_experiment.get_chipBarcode()
        save_plan_step.savedFields[SavePlanFieldNames.CHIP_BARCODE_LABEL] = planned_experiment.get_chipBarcode()

        save_plan_step.savedFields[SavePlanFieldNames.NOTE] = planned_experiment.get_notes()

        LIMS_meta = planned_experiment.get_LIMS_meta()
        if (type(LIMS_meta) is list):
            # convert list to string
            save_plan_step.savedFields[SavePlanFieldNames.LIMS_META] = ''.join(LIMS_meta)
        else:
            save_plan_step.savedFields[SavePlanFieldNames.LIMS_META] = LIMS_meta

        save_plan_step.savedObjects[SavePlanFieldNames.META] = self._metaDataFromPlan(step_helper, planned_experiment)

        # Pick barcode set to use:
        #   1. Edit/Copy - get from plan
        #   2. Create - get from sampleSetItems or, if none, the barcode set selected in the plan template
        barcodeSet = planned_experiment.get_barcodeId()
        if not existing_plan:
            for item in sorted_sampleSetItems:
                if item.dnabarcode:
                    barcodeSet = item.dnabarcode.name
                    break
        barcoding_step.savedFields[SavePlanFieldNames.BARCODE_SET] = step_helper.steps[StepNames.KITS].savedFields[KitsFieldNames.BARCODE_ID] = barcodeSet

        if barcodeSet:
            # make sure barcodeSet is included in selection even if it's inactive
            if barcodeSet not in barcoding_step.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS]:
                available_dnaBarcodes = dnaBarcode.objects.filter(Q(active=True) | Q(name=barcodeSet))
                barcoding_step.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS] = list(available_dnaBarcodes.values_list('name', flat=True).distinct().order_by('name'))
                all_barcodes = {}
                for bc in available_dnaBarcodes.order_by('name', 'index').values('name', 'id_str', 'sequence'):
                    all_barcodes.setdefault(bc['name'], []).append(bc)
                barcoding_step.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_BARCODES] = json.dumps(all_barcodes)

        # IonReporter parameters
        irInfo = self._getIRinfo(planned_experiment)
        if irInfo:
            barcoding_step.prepopulatedFields[SavePlanFieldNames.SELECTED_IR] = irInfo['selectedIr']
            barcoding_step.prepopulatedFields[SavePlanFieldNames.SETID_SUFFIX] = irInfo.get('setid_suffix')

            userInputInfo = irInfo.get("userInputInfo", [])
            iru_hasOncoData = False
            iru_hasPgsData = False
            if userInputInfo:
                for info in userInputInfo:
                    if info.get("cancerType", "") or info.get("cellularityPct", ""):
                        iru_hasOncoData = True
                    if info.get("biopsyDays", "") or info.get("coupleID", "") or info.get("embryoID", ""):
                        iru_hasPgsData = True
                if planned_experiment.categories and ("Oncomine" in planned_experiment.categories or "Onconet" in planned_experiment.categories):
                    iru_hasOncoData = True
            barcoding_step.prepopulatedFields[BarcodeBySampleFieldNames.HAS_ONCO_DATA] = iru_hasOncoData
            barcoding_step.prepopulatedFields[BarcodeBySampleFieldNames.HAS_PGS_DATA] = iru_hasPgsData

        # TODO if irInfo is missing or this is a new plan creation, do that following (template could have IR pre-selected already!!!)
        if barcoding_step.sh_type == StepHelperType.CREATE_NEW_PLAN_BY_SAMPLE or not irInfo:
            sampleSetItem_hasPgsData = False
            sampleSetItem_hasOncoData = False
            for item in sorted_sampleSetItems:
                if item.cancerType or item.cellularityPct:
                    sampleSetItem_hasOncoData = True
                if item.biopsyDays or item.coupleId or item.embryoId:
                    sampleSetItem_hasPgsData = True

            barcoding_step.prepopulatedFields[BarcodeBySampleFieldNames.HAS_ONCO_DATA] = sampleSetItem_hasOncoData
            barcoding_step.prepopulatedFields[BarcodeBySampleFieldNames.HAS_PGS_DATA] = sampleSetItem_hasPgsData

        # Populate samples table
        if existing_plan:
            samplesTable = self._getSamplesTable_from_plan(planned_experiment, step_helper, irInfo)
        else:
            samplesTable = []
            for item in sorted_sampleSetItems:
                sampleDict = {
                    "barcodeId": item.dnabarcode.id_str if item.dnabarcode else '',
                    "sampleName": item.sample.displayedName,
                    "sampleExternalId": item.sample.externalId,
                    "sampleDescription": item.description,
                    "nucleotideType": item.nucleotideType.upper(),
                    "controlSequenceType": "",
                    "reference": "",
                    "targetRegionBedFile": "",
                    "hotSpotRegionBedFile": "",
                    "controlType": item.controlType,
                    "cancerType": "",
                    "cellularityPct": "",
                    "irWorkflow": planned_experiment.irworkflow,
                    "irGender": item.gender,
                    "irRelationRole": item.relationshipRole,
                    "irSetID": item.relationshipGroup,
                    "ircancerType": item.cancerType,
                    "ircellularityPct": item.cellularityPct,
                    "biopsyDays": "",
                    "coupleID": "",
                    "embryoID": "",
                    "irbiopsyDays": item.biopsyDays,
                    "ircoupleID": item.coupleId,
                    "irembryoID": item.embryoId,
                }

                # logger.debug("step_helper_db_loader.updatePlanBySampleSpecificStepHelper() sampleDict=%s" %(sampleDict))

                samplesTable.append(sampleDict)

        if samplesTable:
            barcoding_step.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST] = samplesTable
            barcoding_step.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = json.dumps(samplesTable)

        num_samples = len(samplesTable)
        if step_helper.isCreate():
            if step_helper.isDualNucleotideTypeBySample() and num_samples < 2:
                num_samples = 2
                barcoding_step.savedFields[BarcodeBySampleFieldNames.ONCO_SAME_SAMPLE] = True

        barcoding_step.prepopulatedFields[SavePlanFieldNames.NUM_SAMPLES] = num_samples


    def _getIRinfo(self, planned_experiment):
        # logger.debug("ENTER step_helper_db_loader._getIRinfo()")

        # get IonReporterUploader parameters, if any
        for plugin_name, plugin_dict in planned_experiment.get_selectedPlugins().items():
            if 'IonReporter' in plugin_name:
                try:
                    plugin = Plugin.objects.filter(name=plugin_name, active=True)[0]
                except:
                    continue

                irInfo = {
                    'selectedIr': plugin,
                    'irConfigJson': json.dumps(plugin.userinputfields),
                    'userInputInfo': None
                }
                if PluginFieldNames.USER_INPUT in plugin_dict:
                    # Handle the old and the new style userinput in the plugin dictionary
                    if isinstance(plugin_dict[PluginFieldNames.USER_INPUT], dict):
                        userInputInfo = plugin_dict[PluginFieldNames.USER_INPUT].get("userInputInfo", [])
                        if userInputInfo and len(userInputInfo) > 0:
                            irInfo['userInputInfo'] = userInputInfo
                            irInfo['setid_suffix'] = userInputInfo[0]['setid'][userInputInfo[0]['setid'].find('__'):]
                    elif isinstance(plugin_dict[PluginFieldNames.USER_INPUT], list) and len(plugin_dict[PluginFieldNames.USER_INPUT]) > 0:
                        irInfo['userInputInfo'] = plugin_dict[PluginFieldNames.USER_INPUT]

                return irInfo
        return None

    def _getSamplesTable_from_plan(self, planned_experiment, step_helper, irInfo=None):
        # logger.debug("ENTER step_helper_db_loader._getSamplesTable_from_plan() with step_helper.")

        samplesTable = []

        planNucleotideType = planned_experiment.get_default_nucleotideType()
        runType = planned_experiment.runType

        logger.debug("step_helper_db_loader._getSamplesTable_from_plan() planNucleotideType=%s; runType=%s" % (planNucleotideType, runType))

        if step_helper.isBarcoded():
            # build samples table from barcodedSamples
            sample_to_barcode = planned_experiment.get_barcodedSamples()
            barcodeSet = planned_experiment.get_barcodeId()
            barcode_order = list(dnaBarcode.objects.filter(name=barcodeSet).order_by('index').values_list('id_str', flat=True))
            multibarcode_samples = False

            # WORKAROUND FOR HUB: plan from HUB can have barcodeKit selected but with empty barcodedSamples JSON blob
            application_group_name = "" if not planned_experiment.applicationGroup else planned_experiment.applicationGroup.name
            # logger.debug("step_helper_db_loader._getSamplesTable_from_plan() application_group_name=%s" %(application_group_name))

            if not sample_to_barcode:
                # logger.debug("step_helper_db_loader._getSamplesTable_from_plan()")

                sampleInfo = None
                experiment = planned_experiment.experiment
                latest_eas = planned_experiment.latestEAS
                if experiment and experiment.samples.count() > 0:
                    sampleInfo = experiment.samples.values()[0]

                sampleDict = {"barcodeId": "",
                              "sampleName": sampleInfo['displayedName'] if sampleInfo else "",
                              "sampleExternalId": sampleInfo['externalId'] if sampleInfo else "",
                              "sampleDescription": sampleInfo['description'] if sampleInfo else "",
                              "nucleotideType": planNucleotideType,
                              "controlSequenceType": sampleInfo.get('controlSequenceType', "") if sampleInfo else None,
                              "reference": planned_experiment.get_library() if planned_experiment.get_library() else "",
                              "hotSpotRegionBedFile": planned_experiment.get_regionfile() if planned_experiment.get_regionfile() else "",
                              "targetRegionBedFile": planned_experiment.get_bedfile() if planned_experiment.get_bedfile() else "",
                              "orderKey": format(1, "05d")
                              }
                samplesTable.append(sampleDict)

                logger.debug("step_helper_db_loader._getSamplesTable_from_plan() NO existing barcodedSamples for plan.pk=%d; planName=%s; sampleDict=%s" % (planned_experiment.id, planned_experiment.planDisplayedName, sampleDict))

            else:
                for sample, value in sample_to_barcode.items():
                    if 'barcodeSampleInfo' in value:
                        multibarcode_samples = len(value['barcodeSampleInfo']) > 1

                        for barcode, sampleInfo in value['barcodeSampleInfo'].items():
                            sampleReference = sampleInfo.get("reference", "")
                            sampleHotSpotRegionBedFile = sampleInfo.get("hotSpotRegionBedFile", "")
                            sampleTargetRegionBedFile = sampleInfo.get("targetRegionBedFile", "")

                            if runType != "AMPS_DNA_RNA":
                                if not sampleReference and not step_helper.isReferenceBySample():
                                    if not sampleReference:
                                        sampleReference = planned_experiment.get_library()
                                    # else:
                                    #    logger.debug("step_helper_db_loader._getSamplesTable_from_plan() SKIP SETTING sampleReference to planReference")

                                    if not sampleHotSpotRegionBedFile:
                                        sampleHotSpotRegionBedFile = planned_experiment.get_regionfile()
                                    # else:
                                    #    logger.debug("step_helper_db_loader._getSamplesTable_from_plan() SKIP SETTING sampleHotSpotRegionBedFile")

                                    if not sampleTargetRegionBedFile:
                                        sampleTargetRegionBedFile = planned_experiment.get_bedfile()
                                    # else:
                                    #    logger.debug("step_helper_db_loader._getSamplesTable_from_plan() SKIP SETTING sampleTargetRegionBedFile")

                            order_counter = barcode_order.index(barcode)+1 if barcode in barcode_order else 0
                            sampleDict = {
                                "barcodeId": barcode,
                                "sampleName": sample,
                                "sampleExternalId": sampleInfo.get('externalId', ''),
                                "sampleDescription": sampleInfo.get('description', ''),
                                "nucleotideType": sampleInfo.get("nucleotideType", planNucleotideType),
                                "controlSequenceType": sampleInfo.get("controlSequenceType", ""),
                                "reference": sampleReference,
                                "hotSpotRegionBedFile": sampleHotSpotRegionBedFile,
                                "targetRegionBedFile": sampleTargetRegionBedFile,
                                "controlType": sampleInfo.get('controlType', ''),
                                "orderKey": format(order_counter, "05d")
                            }
                            samplesTable.append(sampleDict)
                            #logger.debug("step_helper_db_loader._getSamplesTable_from_plan() barcodeSampleInfo plan.pk=%d; planName=%s; sampleName=%s; sampleDict=%s" % (planned_experiment.id, planned_experiment.planDisplayedName, sample, sampleDict))

                    else:
                        multibarcode_samples = len(value.get('barcodes', [])) > 1

                        for barcode in value.get('barcodes', []):
                            order_counter = barcode_order.index(barcode)+1 if barcode in barcode_order else 0
                            sampleDict = {"barcodeId": barcode,
                                          "sampleName": sample,
                                          "sampleExternalId": None,
                                          "sampleDescription": None,
                                          "nucleotideType": planNucleotideType,
                                          "controlSequenceType": None,
                                          "reference": planned_experiment.get_library(),
                                          "hotSpotRegionBedFile": "" if planNucleotideType == "RNA" else planned_experiment.get_regionfile(),
                                          "targetRegionBedFile": "" if planNucleotideType == "RNA" else planned_experiment.get_bedfile(),
                                          "orderKey": format(order_counter, "05d")
                                          }
                            samplesTable.append(sampleDict)

            # add IR values
            if irInfo and irInfo['userInputInfo']:
                barcodeToIrValues = {}
                for irvalues in irInfo['userInputInfo']:
                    barcodeId = irvalues.get('barcodeId')
                    if barcodeId:
                        barcodeToIrValues[barcodeId] = irvalues

                for sampleDict in samplesTable:
                    for irkey, irvalue in barcodeToIrValues.get(sampleDict['barcodeId'], {}).items():
                        if irkey == 'Relation':
                            sampleDict['irRelationshipType'] = irvalue
                        elif irkey == 'setid':
                            setid = irvalue.split('__')[0]
                            sampleDict['irSetID'] = setid
                            if setid and setid.isdigit():
                                sampleDict['orderKey'] = "%05d_%s" % (int(setid), sampleDict['orderKey'])
                        else:
                            sampleDict['ir'+irkey] = irvalue

            # sort barcoded samples table
            samplesTable.sort(key=lambda item: item['orderKey'])
            # if same sample for dual nuc type want to order by the DNA/RNA sample pair
            if runType == "AMPS_DNA_RNA" and multibarcode_samples:
                samplesTable.sort(key=lambda item: (item['sampleName'], item['nucleotideType']) )

        else:
            # when we load a non-barcoded run for editing/copying we know it will only have a single sample.
            sampleTubeLabel = "" if step_helper.isCopy() else planned_experiment.sampleTubeLabel
            if sampleTubeLabel is None:
                sampleTubeLabel = ""
            # when we load a non-barcoded run for editing/copying we know it will only have a single chip barcode.
            chipBarcode = "" if step_helper.isCopy() else planned_experiment.get_chipBarcode()
            if chipBarcode is None:
                chipBarcode = ""

            sampleDict = {
                'sampleName':  planned_experiment.get_sampleDisplayedName(),
                'sampleExternalId': planned_experiment.get_sample_external_id(),
                'sampleDescription': planned_experiment.get_sample_description(),
                'tubeLabel':   sampleTubeLabel,
                'chipBarcode': chipBarcode,
                "nucleotideType": planNucleotideType,
                "orderKey": format(1, "05d")
            }

            # add IR values
            if irInfo and irInfo['userInputInfo']:
                for irkey, irvalue in irInfo['userInputInfo'][0].items():
                    if irkey == 'Relation':
                        sampleDict['irRelationshipType'] = irvalue
                    elif irkey == 'setid':
                        sampleDict['irSetID'] = irvalue.split('__')[0]
                    else:
                        sampleDict['ir'+irkey] = irvalue

            samplesTable = [sampleDict]

        return samplesTable

    def getStepHelperForTemplatePlannedExperiment(self, pe_id, step_helper_type=StepHelperType.EDIT_TEMPLATE, sampleset_id=None):
        '''
            Get a step helper from a template planned experiment.
        '''

        logger.debug("ENTER step_helper_db_loader.getStepHelperForTemplatePlannedExperiment() step_helper_type=%s; pe_id=%s" % (step_helper_type, str(pe_id)))

        planned_experiment = PlannedExperiment.objects.get(pk=pe_id)
        if not planned_experiment.isReusable:
            raise ValueError("You must pass in a template id, not a plan id.")

        runType = planned_experiment.runType
        if runType:
            runTypeObjs = RunType.objects.filter(runType=runType)
            if runTypeObjs.count > 0:
                # logger.debug("step_helper_db_loader.getStepHelperForTemplatePlannedExperiment() runType_id=%d" %(runTypeObjs[0].id))
                step_helper = self.getStepHelperForTemplateRunType(runTypeObjs[0].id, step_helper_type, pe_id)

            else:
                step_helper = StepHelper(sh_type=step_helper_type, previous_template_id=pe_id)

        else:
            step_helper = StepHelper(sh_type=step_helper_type, previous_template_id=pe_id)

        planDisplayedName = getPlanDisplayedName(planned_experiment)

        step_helper.parentName = planDisplayedName

        step_helper.isParentSystem = planned_experiment.isSystem

        self.updateUniversalStepHelper(step_helper, planned_experiment)

        if step_helper.isPlan() and step_helper.isPlanBySample():
            self.updatePlanBySampleSpecificStepHelper(step_helper, planned_experiment, sampleset_id)
        elif step_helper.isPlan():
            self.updatePlanSpecificStepHelper(step_helper, planned_experiment, True)
        else:
            self.updateTemplateSpecificStepHelper(step_helper, planned_experiment)

        self.generate_warnings(step_helper)

        return step_helper

    def getStepHelperForPlanPlannedExperiment(self, pe_id, step_helper_type=StepHelperType.EDIT_PLAN):
        '''
            Get a plan step helper from a plan planned experiment.
        '''
        logger.debug("ENTER step_helper_db_loader.getStepHelperForPlanPlannedExperiment() step_helper_type=%s; pe_id=%s" % (step_helper_type, str(pe_id)))

        planned_experiment = PlannedExperiment.objects.get(pk=pe_id)
        if step_helper_type == StepHelperType.EDIT_RUN:
            step_helper = StepHelper(sh_type=step_helper_type, previous_plan_id=pe_id, experiment_id=planned_experiment.experiment.id)
        else:
            step_helper = StepHelper(sh_type=step_helper_type, previous_plan_id=pe_id)

        planDisplayedName = getPlanDisplayedName(planned_experiment)

        step_helper.parentName = planDisplayedName

        if planned_experiment.isReusable:
            raise ValueError("You must pass in a plan id, not a template id.")

        step_helper.isParentSystem = planned_experiment.isSystem

        self.updateUniversalStepHelper(step_helper, planned_experiment)
        if step_helper.isPlan() and step_helper.isPlanBySample():
            self.updatePlanBySampleSpecificStepHelper(step_helper, planned_experiment)
        elif step_helper.isPlan():
            self.updatePlanSpecificStepHelper(step_helper, planned_experiment)
        else:
            raise ValueError("Can not create templates from plans.")

        self.generate_warnings(step_helper)

        return step_helper


    def generate_warnings(self, step_helper):
        ''' add step warnings if any selections are obsolete '''

        if step_helper.isEditRun():
            return

        msg_not_found = "Selected %s %s is not found"
        msg_inactive = "Selected %s %s is not active"

        kits_step_data = step_helper.steps[StepNames.KITS]
        check_kitInfo = [
            ('Library Kit', ['LibraryKit', 'LibraryPrepKit'], kits_step_data.savedFields[KitsFieldNames.LIBRARY_KIT_NAME]),
            ('Template Kit', ['TemplatingKit','IonChefPrepKit'], kits_step_data.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME]),
            ('Sequencing Kit', ['SequencingKit'], kits_step_data.savedFields[KitsFieldNames.SEQUENCE_KIT_NAME]),
            ('Control Sequence', ['ControlSequenceKit'], kits_step_data.savedFields[KitsFieldNames.CONTROL_SEQUENCE]),
            ('Sample Preparation Kit', ['SamplePrepKit'], kits_step_data.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT]),
        ]
        for display, types, kit in check_kitInfo:
            if kit:
                qs = KitInfo.objects.filter(name=kit, kitType__in=types)
                if qs:
                    if not qs[0].isActive:
                        kits_step_data.warnings.append(msg_inactive % (display, kit))
                else:
                    kits_step_data.warnings.append(msg_not_found % (display, kit))

        # barcode set
        barcodeKit = kits_step_data.savedFields[KitsFieldNames.BARCODE_ID]
        if barcodeKit:
            qs = dnaBarcode.objects.filter(name=barcodeKit)
            if qs:
                if not qs.filter(active=True):
                    kits_step_data.warnings.append(msg_inactive % ('Barcode Set', barcodeKit))
            else:
                kits_step_data.warnings.append(msg_not_found % ('Barcode Set', barcodeKit))

        # chip
        chip = kits_step_data.savedFields[KitsFieldNames.CHIP_TYPE]
        if chip:
            qs = Chip.objects.filter(name=chip)
            if qs:
                if not qs.filter(isActive=True):
                    kits_step_data.warnings.append(msg_inactive % ('Chip Type', chip))
            else:
                kits_step_data.warnings.append(msg_not_found % ('Chip Type', chip))
