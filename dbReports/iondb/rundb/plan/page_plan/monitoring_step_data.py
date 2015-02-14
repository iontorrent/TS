# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import QCType
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.plan_validator import validate_QC

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

class MonitoringFieldNames():

    QC_TYPES = 'qcTypes'

class MonitoringStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(MonitoringStepData, self).__init__(sh_type)
        self.resourcePath = 'rundb/plan/page_plan/page_plan_monitoring.html'
        self.savedFields = OrderedDict()
        all_qc_types = list(QCType.objects.all().order_by('qcName'))
        self.prepopulatedFields[MonitoringFieldNames.QC_TYPES] = all_qc_types
        for qc_type in all_qc_types:
            self.savedFields[qc_type.qcName] = qc_type.defaultThreshold

        self.sh_type = sh_type
        
    def validateField(self, field_name, new_field_value):
        '''
        All qc thresholds must be positive integers
        '''
        errors = validate_QC(new_field_value, field_name)
        if errors:
            self.validationErrors[field_name] = errors[0]
        else:
            self.validationErrors.pop(field_name, None)

    def getStepName(self):
        return StepNames.MONITORING

    def updateSavedObjectsFromSavedFields(self):
        pass
    
    def updateFromStep(self, updated_step):
        pass
