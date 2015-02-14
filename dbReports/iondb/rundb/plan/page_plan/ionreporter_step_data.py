# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import SampleGroupType_CV, Plugin
import logging
from iondb.rundb.plan.page_plan.step_names import StepNames
logger = logging.getLogger(__name__)

class IonReporterFieldNames():
    UPLOADERS = 'uploaders'
    SAMPLE_GROUPING = 'sampleGrouping'
    SAMPLE_GROUPINGS = 'sampleGroupings'
    IR_OPTIONS = 'irOptions'
    IR_ACCOUNT_ID = 'irAccountId'
    IR_ACCOUNT_NAME = 'irAccountName'
    IR_VERSION_NONE = 'None'
    IR_VERSION_16 = '1.6'
    IR_VERSION_40 = '4.0'
    IR_PLUGIN = 'IR_PLUGIN'
    IR_WORKFLOW = 'irworkflow'
    IR_VERSION = "irVersion"
    APPLICATION_TYPE = 'applicationType'


class IonreporterStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(IonreporterStepData, self).__init__(sh_type)
        self.resourcePath = 'rundb/plan/page_plan/page_plan_ionreporter.html'

        self.savedFields[IonReporterFieldNames.IR_WORKFLOW] = None
        self.savedFields[IonReporterFieldNames.IR_VERSION] = None

        self.savedFields[IonReporterFieldNames.SAMPLE_GROUPING] = None
        self.savedObjects[IonReporterFieldNames.SAMPLE_GROUPING] = None
        self.savedFields[IonReporterFieldNames.IR_WORKFLOW] = None
        self.prepopulatedFields[IonReporterFieldNames.SAMPLE_GROUPINGS] = SampleGroupType_CV.objects.filter(isActive=True).order_by('uid')

        self.savedFields[IonReporterFieldNames.IR_OPTIONS] = ''
        self.savedFields[IonReporterFieldNames.IR_ACCOUNT_ID] = '-1'
        self.savedFields[IonReporterFieldNames.IR_ACCOUNT_NAME] = 'None'
        self.savedFields[IonReporterFieldNames.APPLICATION_TYPE] = ''

        try:
            self.prepopulatedFields[IonReporterFieldNames.IR_PLUGIN] = Plugin.objects.get(name__iexact='IonReporterUploader', active=True)
        except Exception, e:
            self.prepopulatedFields[IonReporterFieldNames.IR_PLUGIN] = None

        self.sh_type = sh_type
        
    def getStepName(self):
        return StepNames.IONREPORTER
    
    def updateSavedObjectsFromSavedFields(self):
        if self.savedFields[IonReporterFieldNames.SAMPLE_GROUPING]:
            self.savedObjects[IonReporterFieldNames.SAMPLE_GROUPING] = SampleGroupType_CV.objects.get(pk=self.savedFields[IonReporterFieldNames.SAMPLE_GROUPING])
        if self.savedFields[IonReporterFieldNames.IR_WORKFLOW] and self.savedFields[IonReporterFieldNames.IR_WORKFLOW] == 'Upload Only':
            self.savedFields[IonReporterFieldNames.IR_WORKFLOW] = ''
    
    def updateFromStep(self, updated_step):
        pass

    def getIRU(self):
        if self.prepopulatedFields[IonReporterFieldNames.IR_PLUGIN]:
            return self.prepopulatedFields[IonReporterFieldNames.IR_PLUGIN]

