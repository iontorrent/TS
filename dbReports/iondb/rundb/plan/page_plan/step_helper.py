# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
'''
Created on May 21, 2013

@author: ionadmin
'''
import logging
from iondb.rundb.plan.page_plan.export_step_data import ExportStepData
from iondb.rundb.plan.page_plan.application_step_data import ApplicationStepData
from iondb.rundb.plan.page_plan.kits_step_data import KitsStepData
from iondb.rundb.plan.page_plan.monitoring_step_data import MonitoringStepData
from iondb.rundb.plan.page_plan.reference_step_data import ReferenceStepData
from iondb.rundb.plan.page_plan.plugins_step_data import PluginsStepData
from iondb.rundb.plan.page_plan.output_step_data import OutputStepData
from iondb.rundb.plan.page_plan.ionreporter_step_data import IonreporterStepData
from iondb.rundb.plan.page_plan.save_template_step_data import SaveTemplateStepData
from iondb.rundb.plan.page_plan.save_plan_step_data import SavePlanStepData
from iondb.rundb.plan.page_plan.barcode_by_sample_step_data import BarcodeBySampleStepData
#from iondb.rundb.plan.page_plan.output_by_sample_step_data import OutputBySampleStepData
from iondb.rundb.plan.page_plan.save_plan_by_sample_step_data import SavePlanBySampleStepData
from iondb.rundb.plan.page_plan.save_template_by_sample_step_data import SaveTemplateBySampleStepData
from iondb.rundb.plan.page_plan.step_names import StepNames
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from iondb.rundb.models import KitInfo
    
logger = logging.getLogger(__name__)

class StepHelperType():
    CREATE_NEW_TEMPLATE = "create_new_template"
    CREATE_NEW_TEMPLATE_BY_SAMPLE = "create_new_template_by_sample"

    EDIT_TEMPLATE = "edit_template"
    COPY_TEMPLATE = "copy_template"
    
    CREATE_NEW_PLAN = "create_new_plan"
    EDIT_PLAN = "edit_plan"
    COPY_PLAN = "copy_plan"

    CREATE_NEW_PLAN_BY_SAMPLE = "create_new_plan_by_sample"
    EDIT_PLAN_BY_SAMPLE = "edit_plan_by_sample"
    COPY_PLAN_BY_SAMPLE = "copy_plan_by_sample"

    EDIT_RUN = "edit_run"
    
    TEMPLATE_TYPES = [CREATE_NEW_TEMPLATE, EDIT_TEMPLATE, COPY_TEMPLATE, CREATE_NEW_TEMPLATE_BY_SAMPLE]
    PLAN_TYPES = [CREATE_NEW_PLAN, EDIT_PLAN, COPY_PLAN, EDIT_RUN,
                    CREATE_NEW_PLAN_BY_SAMPLE, EDIT_PLAN_BY_SAMPLE, COPY_PLAN_BY_SAMPLE]


class StepHelper(object):
    '''
    Helper class for interacting with the plan/template creation steps.
    '''

    def __init__(self, sh_type=StepHelperType.CREATE_NEW_TEMPLATE, previous_template_id=-1, previous_plan_id=-1):
        self.sh_type = sh_type
        self.previous_template_id = previous_template_id
        self.previous_plan_id = previous_plan_id
        self.parentName = None
        
        self.steps = OrderedDict()
        if sh_type == StepHelperType.CREATE_NEW_PLAN_BY_SAMPLE or sh_type == StepHelperType.EDIT_PLAN_BY_SAMPLE or sh_type == StepHelperType.COPY_PLAN_BY_SAMPLE:
            steps_list = [IonreporterStepData(), ApplicationStepData(), KitsStepData(), MonitoringStepData(),
                      ReferenceStepData(), PluginsStepData(), BarcodeBySampleStepData(), 
                      OutputStepData(), SavePlanBySampleStepData()]

        elif sh_type == StepHelperType.CREATE_NEW_TEMPLATE_BY_SAMPLE:
            steps_list = [IonreporterStepData(), ApplicationStepData(), KitsStepData(), MonitoringStepData(),
                      ReferenceStepData(), PluginsStepData(), 
                      OutputStepData(), SaveTemplateBySampleStepData()]            
        else:
            steps_list = [IonreporterStepData(), ApplicationStepData(), KitsStepData(), MonitoringStepData(),
                      ReferenceStepData(), PluginsStepData(), OutputStepData(),
                      SaveTemplateStepData(), SavePlanStepData()]    
        
        for step in steps_list:
            self.steps[step.getStepName()] = step
        self.update_dependant_steps(self.steps[StepNames.APPLICATION])
    
    def getStepDict(self):
        return self.steps
    
    def updateStepFromRequest(self, request, step_name):
        logger.debug("Updating %s with data from %s" % (step_name, str(request.POST)))
        if step_name in self.getStepDict():
            step = self.steps[step_name]
            retval = step.updateSavedFieldValuesFromRequest(request)
            if retval:
                step.updateSavedObjectsFromSavedFields()
                self.update_dependant_steps(step)
            return retval
        return False
    
    def update_dependant_steps(self, updated_step):
        '''
            Applies updates to all steps that depend on the updated step.
            If other steps depend on a step that got updated does that too.
        '''
        updated_steps = [updated_step]
        
        while updated_steps:
            updated_step = updated_steps[0]
            for dependant_step in self.steps.values():
                # if editing run post-sequencing, don't load defaults when application changes
                if self.isEditRun() and updated_step.getStepName() == StepNames.APPLICATION:
                    continue
                 
                if updated_step.getStepName() in dependant_step._dependsOn:
                    dependant_step.updateFromStep(updated_step)
                    updated_steps.append(dependant_step)
                    
            updated_steps.remove(updated_step)
    
    def isPlan(self):
        return self.sh_type in StepHelperType.PLAN_TYPES

    def isPlanBySample(self):
        return self.sh_type in (StepHelperType.CREATE_NEW_PLAN_BY_SAMPLE, StepHelperType.EDIT_PLAN_BY_SAMPLE, StepHelperType.COPY_PLAN_BY_SAMPLE)
    
    def isTemplate(self):
        return self.sh_type in StepHelperType.TEMPLATE_TYPES

    def isTemplateBySample(self):
        return self.sh_type == StepHelperType.CREATE_NEW_TEMPLATE_BY_SAMPLE
    
    def isBarcoded(self):

        if self.steps[StepNames.KITS].savedFields['barcodeId']:
            return True
        return False
    
    def isCreate(self):
        return self.sh_type in [StepHelperType.CREATE_NEW_PLAN, StepHelperType.CREATE_NEW_TEMPLATE]
    
    def isEdit(self):
        return self.sh_type in [StepHelperType.EDIT_PLAN, StepHelperType.EDIT_TEMPLATE, StepHelperType.EDIT_PLAN_BY_SAMPLE]
    
    def isEditRun(self):
        return self.sh_type in [StepHelperType.EDIT_RUN]
    
    def isCopy(self):
        return self.sh_type in [StepHelperType.COPY_PLAN, StepHelperType.COPY_TEMPLATE, StepHelperType.COPY_PLAN_BY_SAMPLE]
    
    def isIonChef(self):
        selectedTemplateKit = self.steps[StepNames.KITS].savedFields['templatekitname']
        isIonChef = False
        
        if (selectedTemplateKit):
            kits = KitInfo.objects.filter(name = selectedTemplateKit)
            if kits:
                isIonChef = kits[0].kitType == "IonChefPrepKit"
        
        return isIonChef
        
    def isTargetStepAfterOriginal(self, original_step_name, target_step_name):
        if original_step_name == StepNames.EXPORT:
            return True
        if original_step_name == StepNames.SAVE_TEMPLATE or original_step_name == StepNames.SAVE_PLAN \
        or original_step_name == StepNames.SAVE_TEMPLATE_BY_SAMPLE or original_step_name == StepNames.SAVE_PLAN_BY_SAMPLE:
            return False
        
        original_index = self.steps.keys().index(original_step_name)
        target_index = self.steps.keys().index(target_step_name)
        return target_index >= original_index
    
    def validateAll(self):
        for step_name, step in self.steps.items():
            # do not validate plan step if this is a template helper and vice versa
            if (self.isPlan() and step_name != StepNames.SAVE_TEMPLATE) or (self.isTemplate() and step_name != StepNames.SAVE_PLAN):
                step.validate()
                if step.hasErrors():
                    return step_name
        
        return None
