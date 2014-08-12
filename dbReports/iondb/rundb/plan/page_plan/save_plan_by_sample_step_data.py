# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import dnaBarcode, Plugin
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.plan_validator import validate_plan_name
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import json
import uuid
import logging
logger = logging.getLogger(__name__)

class SavePlanBySampleFieldNames():

    TEMPLATE_NAME = 'templateName'
    SAMPLESET = 'sampleset'
    WARNING_MESSAGES = 'warning_messages'
    ERROR_MESSAGES = 'error_messages'
    NOTE = 'note'

class SavePlanBySampleStepData(AbstractStepData):

    def __init__(self):
        super(SavePlanBySampleStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_by_sample_save_plan.html'
        self.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME] = None
        self.savedFields[SavePlanBySampleFieldNames.ERROR_MESSAGES] = None
        self.savedFields[SavePlanBySampleFieldNames.WARNING_MESSAGES] = None
        self.savedFields[SavePlanBySampleFieldNames.NOTE] = None


    def getStepName(self):
        return StepNames.SAVE_PLAN_BY_SAMPLE

    def validateStep(self):
        new_field_value = self.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME]
        errors = validate_plan_name(new_field_value, 'Plan Name')
        if errors:
            self.validationErrors['invalidPlanName'] = errors
        else:
            self.validationErrors.pop('invalidPlanName', None)

    def updateSavedObjectsFromSavedFields(self):
        pass
