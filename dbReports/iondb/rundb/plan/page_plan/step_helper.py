# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
'''
Created on May 21, 2013

@author: ionadmin
'''
import logging

from iondb.rundb.plan.page_plan.application_step_data import ApplicationStepData
from iondb.rundb.plan.page_plan.kits_step_data import KitsStepData, KitsFieldNames
from iondb.rundb.plan.page_plan.monitoring_step_data import MonitoringStepData
from iondb.rundb.plan.page_plan.reference_step_data import ReferenceStepData
from iondb.rundb.plan.page_plan.plugins_step_data import PluginsStepData
from iondb.rundb.plan.page_plan.output_step_data import OutputStepData
from iondb.rundb.plan.page_plan.ionreporter_step_data import IonreporterStepData
from iondb.rundb.plan.page_plan.save_template_step_data import SaveTemplateStepData
from iondb.rundb.plan.page_plan.save_plan_step_data import SavePlanStepData, SavePlanFieldNames
from iondb.rundb.plan.page_plan.barcode_by_sample_step_data import BarcodeBySampleStepData, BarcodeBySampleFieldNames
# from iondb.rundb.plan.page_plan.output_by_sample_step_data import OutputBySampleStepData
from iondb.rundb.plan.page_plan.save_plan_by_sample_step_data import SavePlanBySampleStepData
from iondb.rundb.plan.page_plan.save_template_by_sample_step_data import SaveTemplateBySampleStepData
from iondb.rundb.plan.page_plan.analysis_params_step_data import AnalysisParamsStepData

from iondb.rundb.plan.page_plan.step_names import StepNames

from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames

from iondb.rundb.plan.page_plan.step_helper_types import StepHelperType

from iondb.rundb.models import ApplicationGroup, PlannedExperiment

from iondb.rundb.plan.views_helper import isOCP_enabled

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from iondb.rundb.models import KitInfo, Chip

logger = logging.getLogger(__name__)


class StepHelper(object):

    '''
    Helper class for interacting with the plan/template creation steps.
    '''

    def __init__(self, sh_type=StepHelperType.CREATE_NEW_TEMPLATE, previous_template_id=-1, previous_plan_id=-1, experiment_id=-1):
        self.sh_type = sh_type
        self.previous_template_id = previous_template_id
        self.previous_plan_id = previous_plan_id
        self.experiment_id = experiment_id
        self.parentName = None
        self.isFromScratch = False
        # set to true if we are creating a plan or template from a system template
        self.isParentSystem = False

        if (sh_type in [StepHelperType.EDIT_PLAN, StepHelperType.EDIT_PLAN_BY_SAMPLE, StepHelperType.EDIT_RUN, StepHelperType.EDIT_TEMPLATE] and previous_template_id == -1 and previous_plan_id == -1):
            logger.error("step_helper - StepHelper.init() for EDIT should have an existing ID.")
            raise ValueError("You must pass in a plan id or a template id.")

        self.steps = OrderedDict()
        if sh_type == StepHelperType.CREATE_NEW_PLAN_BY_SAMPLE or sh_type == StepHelperType.EDIT_PLAN_BY_SAMPLE or sh_type == StepHelperType.COPY_PLAN_BY_SAMPLE:
            referenceStepData = ReferenceStepData(sh_type)
            barcodeBySampleStepData = BarcodeBySampleStepData(sh_type)
            analysisParamsStepData = AnalysisParamsStepData(sh_type)

            steps_list = [IonreporterStepData(sh_type), ApplicationStepData(sh_type), KitsStepData(sh_type),
                          referenceStepData,  analysisParamsStepData, PluginsStepData(sh_type), barcodeBySampleStepData,
                          OutputStepData(sh_type), SavePlanBySampleStepData(sh_type)]

            # some section can appear in multiple chevrons, key is the step name and value is the step_data object
            barcodeBySampleStepData.step_sections.update({StepNames.REFERENCE: referenceStepData})
            barcodeBySampleStepData.step_sections.update({StepNames.ANALYSIS_PARAMS: analysisParamsStepData})

        elif sh_type == StepHelperType.CREATE_NEW_TEMPLATE_BY_SAMPLE:
            referenceStepData = ReferenceStepData(sh_type)
            saveTemplateBySampleData = SaveTemplateBySampleStepData(sh_type)
            analysisParamsStepData = AnalysisParamsStepData(sh_type)

            steps_list = [IonreporterStepData(sh_type), ApplicationStepData(sh_type), KitsStepData(sh_type),
                          referenceStepData,  analysisParamsStepData, PluginsStepData(sh_type),
                          OutputStepData(sh_type), saveTemplateBySampleData]

            saveTemplateBySampleData.step_sections.update({StepNames.REFERENCE: referenceStepData})
            saveTemplateBySampleData.step_sections.update({StepNames.ANALYSIS_PARAMS: analysisParamsStepData})

        elif sh_type == StepHelperType.COPY_TEMPLATE or sh_type == StepHelperType.CREATE_NEW_TEMPLATE or sh_type == StepHelperType.EDIT_TEMPLATE:
            referenceStepData = ReferenceStepData(sh_type)
            saveTemplateStepData = SaveTemplateStepData(sh_type)
            analysisParamsStepData = AnalysisParamsStepData(sh_type)

            steps_list = [IonreporterStepData(sh_type), ApplicationStepData(sh_type), KitsStepData(sh_type),
                          referenceStepData, analysisParamsStepData, PluginsStepData(sh_type), OutputStepData(sh_type),
                          saveTemplateStepData]

            saveTemplateStepData.step_sections.update({StepNames.REFERENCE: referenceStepData})
            saveTemplateStepData.step_sections.update({StepNames.ANALYSIS_PARAMS: analysisParamsStepData})

        else:

            referenceStepData = ReferenceStepData(sh_type)
            # SaveTemplateStepData is needed for the last chevron during plan creation
            saveTemplateStepData = SaveTemplateStepData(sh_type)
            savePlanStepData = SavePlanStepData(sh_type)
            analysisParamsStepData = AnalysisParamsStepData(sh_type)

            steps_list = [IonreporterStepData(sh_type), ApplicationStepData(sh_type), KitsStepData(sh_type),
                          referenceStepData, analysisParamsStepData, PluginsStepData(sh_type), OutputStepData(sh_type),
                          saveTemplateStepData, savePlanStepData]

            savePlanStepData.step_sections.update({StepNames.REFERENCE: referenceStepData})            ###referenceStepData.sectionParentStep = savePlanStepData
            savePlanStepData.step_sections.update({StepNames.ANALYSIS_PARAMS: analysisParamsStepData})

        for step in steps_list:
            self.steps[step.getStepName()] = step

        self.update_dependent_steps(self.steps[StepNames.APPLICATION])

    def getStepDict(self):
        return self.steps

    def updateStepFromRequest(self, request, step_name):
        logger.debug("updateStepFromRequest... Updating %s with data from %s" % (step_name, str(request.POST)))
        if step_name in self.getStepDict():
            step = self.steps[step_name]
            retval = step.updateSavedFieldValuesFromRequest(request)
            if retval:
                step.updateSavedObjectsFromSavedFields()
                self.update_dependent_steps(step)
            return retval
        return False

    def update_dependent_steps(self, updated_step):
        '''
            Applies updates to all steps that depend on the updated step.
            If other steps depend on a step that got updated does that too.
        '''
        updated_steps = [updated_step]

        while updated_steps:
            updated_step = updated_steps[0]
            for dependent_step in self.steps.values():

                # if editing run post-sequencing, don't load defaults when application changes
                if self.isEditRun() and updated_step.getStepName() == StepNames.APPLICATION:
                    if updated_step.getStepName() in dependent_step._dependsOn:
                        dependent_step.alternateUpdateFromStep(updated_step)
                        updated_steps.append(dependent_step)
                        continue

                if updated_step.getStepName() in dependent_step._dependsOn:
                    dependent_step.updateFromStep(updated_step)
                    updated_steps.append(dependent_step)

                # need to update barcode Set here to avoid circular dependency
                if updated_step.getStepName() == StepNames.SAVE_PLAN:
                    self.steps[StepNames.KITS].savedFields[KitsFieldNames.BARCODE_ID] = updated_step.savedFields[SavePlanFieldNames.BARCODE_SET]

            updated_steps.remove(updated_step)

    def isPlan(self):
        return self.sh_type in StepHelperType.PLAN_TYPES

    def isPlanBySample(self):
        return self.sh_type in StepHelperType.PLAN_BY_SAMPLE_TYPES

    def isTemplate(self):
        return self.sh_type in StepHelperType.TEMPLATE_TYPES

    def isTemplateBySample(self):
        return self.sh_type == StepHelperType.CREATE_NEW_TEMPLATE_BY_SAMPLE

    def isBarcoded(self):

        if self.steps[StepNames.KITS].savedFields[KitsFieldNames.BARCODE_ID]:
            return True
        return False

    def isCreate(self):
        return self.sh_type in [StepHelperType.CREATE_NEW_PLAN, StepHelperType.CREATE_NEW_TEMPLATE, StepHelperType.CREATE_NEW_PLAN_BY_SAMPLE]

    def isEdit(self):
        return self.sh_type in [StepHelperType.EDIT_PLAN, StepHelperType.EDIT_TEMPLATE, StepHelperType.EDIT_PLAN_BY_SAMPLE]

    def isEditRun(self):
        return self.sh_type in [StepHelperType.EDIT_RUN]

    def isCopy(self):
        return self.sh_type in [StepHelperType.COPY_PLAN, StepHelperType.COPY_TEMPLATE, StepHelperType.COPY_PLAN_BY_SAMPLE]

    def isIonChef(self):
        selectedTemplateKit = self.steps[StepNames.KITS].savedFields[KitsFieldNames.TEMPLATE_KIT_NAME]
        isIonChef = False

        if (selectedTemplateKit):
            kits = KitInfo.objects.filter(name=selectedTemplateKit)
            if kits:
                isIonChef = kits[0].kitType == "IonChefPrepKit"

        return isIonChef

    def isProton(self):
        selectedChipType = self.steps[StepNames.KITS].savedFields[KitsFieldNames.CHIP_TYPE]
        isProton = False

        if (selectedChipType):
            chips = Chip.objects.filter(name=selectedChipType, instrumentType__iexact="proton")
            if chips:
                isProton = True

        return isProton

    def isTargetStepAfterOriginal(self, original_step_name, target_step_name):
        if original_step_name == StepNames.EXPORT:
            return True
        if original_step_name == StepNames.SAVE_TEMPLATE or original_step_name == StepNames.SAVE_PLAN \
                or original_step_name == StepNames.SAVE_TEMPLATE_BY_SAMPLE or original_step_name == StepNames.SAVE_PLAN_BY_SAMPLE:
            return False

        original_index = self.steps.keys().index(original_step_name)
        target_index = self.steps.keys().index(target_step_name)
        return target_index >= original_index

    def getApplProduct(self):
        return self.steps[StepNames.APPLICATION].savedObjects[ApplicationFieldNames.APPL_PRODUCT]

    def getApplProducts(self):
        """
        returns a collection of applProduct entries for the selected application and target technique
        """
        return self.steps[StepNames.APPLICATION].prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS]

    def getCategorizedApplProducts(self):
        """
        returns a collection of categorized applProduct entries for the selected application and target technique
        """
        return self.steps[StepNames.APPLICATION].prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS_CATEGORIZED]

    def getApplProductByInstrumentType(self, instrumentType):
        applProducts = self.getApplProducts()
        if applProducts:
            for applProduct in applProducts:
                 if applProduct.instrumentType.lower() == instrumentType.lower():
                     return applProduct
        return self.getApplProduct()

    def isToMandateTargetTechniqueToShow(self):
        """
        this is a workaround until applproduct supports application-group sepecific rules
        """
        if self.getApplicationGroupName():
            return True if self.getApplicationGroupName() in ["DNA + RNA"] else False;
        else:
            return True

#
#     def getApplProductByRunType(self, runTypeId):
#         applProducts = self.steps[StepNames.APPLICATION].prepopulatedFields[ApplicationFieldNames.APPL_PRODUCTS]
#
#         for applProduct in applProducts:
#             if applProduct.applType_id == runTypeId:
# logger.debug("step_helper.getApplProductByRunType() runTypeId=%s; going to return applProduct=%s" %(runTypeId, applProduct))
#                 return applProduct
#         return None

    def getRunTypeObject(self):
        # logger.debug("getRunTypeObject nucleotideType=%s" %(self.steps[StepNames.APPLICATION].savedObjects[ApplicationFieldNames.RUN_TYPE].nucleotideType))

        # save_plan_step_data.savedFields[SavePlanFieldNames.SAMPLES_TABLE]
        return self.steps[StepNames.APPLICATION].savedObjects[ApplicationFieldNames.RUN_TYPE]

    def getApplicationGroupName(self):
        return self.steps[StepNames.APPLICATION].savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME]

    def getApplicationCategoryDisplayedName(self):
        runType_obj = self.getRunTypeObject()
        categories = self.steps[StepNames.APPLICATION].prepopulatedFields[ApplicationFieldNames.CATEGORIES]
        if categories:
            if runType_obj:
                return PlannedExperiment.get_validatedApplicationCategoryDisplayedName(categories, runType_obj.runType)
            else:
                return PlannedExperiment.get_applicationCategoryDisplayedName(categories)
        else:
            return ""

    def isControlSeqTypeBySample(self):
        return self.getApplProduct().isControlSeqTypeBySampleSupported

    def isReferenceBySample(self):
        return self.getApplProduct().isReferenceBySampleSupported and self.getApplProduct().isReferenceSelectionSupported

    def isDualNucleotideTypeBySample(self):
        return self.getApplProduct().isDualNucleotideTypeBySampleSupported

    def isBarcodeKitSelectionRequired(self):
        return self.getApplProduct().isBarcodeKitSelectionRequired

    def isOCPEnabled(self):
        return isOCP_enabled()

    def isOCPApplicationGroup(self):
        return self.getApplicationGroupName() == "DNA + RNA"

    def getNucleotideTypeList(self):
        return ["", "DNA", "RNA"]

    def getIruQcUploadModeList(self):
        iruQcUploadModes = {"Review results after run completion, then upload to Ion Reporter *": "manual_check",\
                            "Automatically upload to Ion Reporter after run completion": "no_check"}
        return iruQcUploadModes

    def hasPgsData(self):
        if self.isTemplate():
            return False
        if self.isPlanBySample():
            step = self.steps.get(StepNames.BARCODE_BY_SAMPLE, None)
            if step:
                return step.prepopulatedFields[BarcodeBySampleFieldNames.HAS_PGS_DATA]
        else:
            step = self.steps.get(StepNames.SAVE_PLAN, None)
            if step:
                return step.prepopulatedFields[SavePlanFieldNames.HAS_PGS_DATA]
        return False

    def hasOncoData(self):
        if self.isTemplate():
            return False

        categories = self.steps[StepNames.APPLICATION].prepopulatedFields[ApplicationFieldNames.CATEGORIES]
        if "Oncomine" in categories or "Onconet" in categories:
            return True

        if self.isPlanBySample():
            step = self.steps.get(StepNames.BARCODE_BY_SAMPLE, None)
            if step:
                return step.prepopulatedFields[BarcodeBySampleFieldNames.HAS_ONCO_DATA]
        else:
            step = self.steps.get(StepNames.SAVE_PLAN, None)
            if step:
                return step.prepopulatedFields[SavePlanFieldNames.HAS_ONCO_DATA]
        return False

    def validateAll(self):
        for step_name, step in self.steps.items():
            # do not validate plan step if this is a template helper and vice versa
            if (self.isPlan() and step_name != StepNames.SAVE_TEMPLATE) or (self.isTemplate() and step_name != StepNames.SAVE_PLAN):
                step.validate()
                if step.hasErrors():
                    logger.debug("step_helper.validateAll() HAS ERRORS! step_name=%s" % (step_name))
                    return step_name

        return None

    def getChangedFields(self):
        changed = {}
        for step in self.steps.values():
            for key, values in step._changedFields.items():
                if (values[0] or values[1]) and (values[0] != values[1]) and str(values[0] != values[1]):
                    changed[key] = values
        return changed

    def isS5(self):
        selectedChipType = self.steps[StepNames.KITS].savedFields[KitsFieldNames.CHIP_TYPE]
        isS5 = False

        if (selectedChipType):
            chips = Chip.objects.filter(name=selectedChipType, instrumentType__iexact="s5")
            if chips:
                isS5 = True

        return isS5
