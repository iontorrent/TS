# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.plan_validator import validate_plan_name


class SaveTemplateStepData(AbstractStepData):

    def __init__(self):
        super(SaveTemplateStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_save_template.html'
        self.savedFields['templateName'] = None
        self.savedFields['setAsFavorite'] = None
        self.savedFields['note'] = None

    def validateField(self, field_name, new_field_value):
        if field_name == 'templateName':
            errors = validate_plan_name(new_field_value, 'Template Name')
            if errors:
                self.validationErrors[field_name] = errors
            else:
                self.validationErrors.pop(field_name, None)

    def getStepName(self):
        return StepNames.SAVE_TEMPLATE
    
    def updateSavedObjectsFromSavedFields(self):
        pass
