# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import dnaBarcode, Plugin
from iondb.rundb.plan.views_helper import is_valid_chars, is_valid_length,\
    is_invalid_leading_chars
from iondb.rundb.plan.page_plan.step_names import StepNames
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

class SavePlanBySampleStepData(AbstractStepData):

    def __init__(self):
        super(SavePlanBySampleStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_by_sample_save_plan.html'
        self.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME] = None
        self.savedFields[SavePlanBySampleFieldNames.ERROR_MESSAGES] = None
        self.savedFields[SavePlanBySampleFieldNames.WARNING_MESSAGES] = None


    def getStepName(self):
        return StepNames.SAVE_PLAN_BY_SAMPLE

    def validateStep(self):
        self.validationErrors['invalidPlanName'] = None

        if not self.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME]:
            self.validationErrors['invalidPlanName'] = "Please enter a plan name"

        if not self.validationErrors['invalidPlanName']:
            self.validationErrors.pop('invalidPlanName')

    def updateSavedObjectsFromSavedFields(self):
        pass