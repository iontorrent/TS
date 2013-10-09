# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import QCType
from iondb.rundb.plan.page_plan.step_names import StepNames
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

class MonitoringFieldNames():

    QC_TYPES = 'qcTypes'

class MonitoringStepData(AbstractStepData):

    def __init__(self):
        super(MonitoringStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_monitoring.html'
        self.savedFields = OrderedDict()
        all_qc_types = list(QCType.objects.all().order_by('qcName'))
        self.prepopulatedFields[MonitoringFieldNames.QC_TYPES] = all_qc_types
        for qc_type in all_qc_types:
            self.savedFields[qc_type.qcName] = qc_type.defaultThreshold

    def validateField(self, field_name, new_field_value):
        '''
        All qc thresholds must be positive integers
        '''
        valid = False
        try:
            int_val = int(new_field_value)
            if int_val >= 0:
                valid = True
        except:
            pass
        if valid and field_name in self.validationErrors:
            self.validationErrors.pop(field_name, None)
        elif not valid:
            self.validationErrors[field_name] = "%s must be a positive integer." % field_name
        
        
        

    def getStepName(self):
        return StepNames.MONITORING

    def updateSavedObjectsFromSavedFields(self):
        pass
    
    def updateFromStep(self, updated_step):
        pass
