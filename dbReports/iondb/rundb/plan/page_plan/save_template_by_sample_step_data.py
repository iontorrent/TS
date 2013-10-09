# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.plan.views_helper import is_valid_chars, is_valid_length
from iondb.rundb.plan.page_plan.step_names import StepNames

MAX_LENGTH_PLAN_NAME = 512

class SaveTemplateBySampleStepData(AbstractStepData):

    def __init__(self):
        super(SaveTemplateBySampleStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_by_sample_save_plan.html'
        self.savedFields['templateName'] = None
        self.savedFields['setAsFavorite'] = None

    def getStepName(self):
        return StepNames.SAVE_TEMPLATE_BY_SAMPLE

    def validateField(self, field_name, new_field_value):
        if field_name == 'templateName':
            self.validationErrors[field_name] = None
            valid = True
            if not new_field_value:
                self.validationErrors[field_name] = 'Error, please enter a Template Name.'
                valid = False
            
            # if not is_valid_chars(new_field_value):
            #     self.validationErrors[field_name].append('Error, Template Name should contain only numbers, letters, spaces, and the following: . - _')
            #     valid = False
                
            # if not is_valid_length(new_field_value, MAX_LENGTH_PLAN_NAME):
            #     self.validationErrors[field_name].append('Error, Template Name length should be %s characters maximum. It is currently %s characters long.' % (str(MAX_LENGTH_PLAN_NAME), str(len(new_field_value))))
            #     valid = False
            if valid:
                self.validationErrors.pop(field_name, None)

    
    def updateSavedObjectsFromSavedFields(self):
        pass