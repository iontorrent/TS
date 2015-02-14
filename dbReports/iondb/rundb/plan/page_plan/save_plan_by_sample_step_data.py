# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import dnaBarcode, Plugin, QCType
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.plan_validator import validate_plan_name, validate_notes, validate_QC
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

    LIMS_META = 'LIMS_meta'
    META = 'meta'

class MonitoringFieldNames():
    QC_TYPES = 'qcTypes'

    
class SavePlanBySampleStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(SavePlanBySampleStepData, self).__init__(sh_type)
        self.resourcePath = 'rundb/plan/page_plan/page_plan_by_sample_save_plan.html'
        self.savedFields = OrderedDict()
        
        self.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME] = None
        self.savedFields[SavePlanBySampleFieldNames.ERROR_MESSAGES] = None
        self.savedFields[SavePlanBySampleFieldNames.WARNING_MESSAGES] = None
        self.savedFields[SavePlanBySampleFieldNames.NOTE] = None

        self.savedFields[SavePlanBySampleFieldNames.LIMS_META] = None
        self.savedFields[SavePlanBySampleFieldNames.META] = {}
        
        
        self.sh_type = sh_type

        # Monitoring
        self.qcNames = []
        all_qc_types = list(QCType.objects.all().order_by('qcName'))
        self.prepopulatedFields[MonitoringFieldNames.QC_TYPES] = all_qc_types
        for qc_type in all_qc_types:
            self.savedFields[qc_type.qcName] = qc_type.defaultThreshold
            self.qcNames.append(qc_type.qcName)


    def getStepName(self):
        return StepNames.SAVE_PLAN_BY_SAMPLE

    def validateField(self, field_name, new_field_value):
        self.validationErrors.pop(field_name, None)

        if field_name == SavePlanBySampleFieldNames.NOTE:
            errors = validate_notes(new_field_value)
            if errors:
                self.validationErrors[field_name] = '\n'.join(errors)    
        
        '''
        All qc thresholds must be positive integers
        '''
        if field_name in self.qcNames:
            errors = validate_QC(new_field_value, field_name)
            if errors:
                self.validationErrors[field_name] = errors[0]
            else:
                self.validationErrors.pop(field_name, None)

    def validateStep(self):
        new_field_value = self.savedFields[SavePlanBySampleFieldNames.TEMPLATE_NAME]
        errors = validate_plan_name(new_field_value, 'Plan Name')
        if errors:
            self.validationErrors['invalidPlanName'] = errors
        else:
            self.validationErrors.pop('invalidPlanName', None)

    def updateSavedObjectsFromSavedFields(self):
        pass
