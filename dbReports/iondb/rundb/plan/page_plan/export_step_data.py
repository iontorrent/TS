# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.models import Plugin


class ExportFieldNames:

    IR_OPTIONS = "irOptions"
    IR_ACCOUNT_ID = "irAccountId"
    IR_ACCOUNT_NAME = "irAccountName"
    IR_VERSION_NONE = "None"
    IR_VERSION_16 = "1.6"
    IR_VERSION_40 = "4.0"
    IR_PLUGIN = "IR_PLUGIN"


class ExportStepData(AbstractStepData):

    """
    Holds the data needed by and saved into the export step.
    """

    def __init__(self, sh_type):
        super(ExportStepData, self).__init__(sh_type)
        self.resourcePath = "rundb/plan/page_plan/page_plan.html"
        self.savedFields[ExportFieldNames.IR_OPTIONS] = ""
        self.savedFields[ExportFieldNames.IR_ACCOUNT_ID] = "0"
        self.savedFields[ExportFieldNames.IR_ACCOUNT_NAME] = "None"
        try:
            self.prepopulatedFields[ExportFieldNames.IR_PLUGIN] = Plugin.objects.get(
                name__iexact="IonReporterUploader", active=True
            )
        except Exception as e:
            self.prepopulatedFields[ExportFieldNames.IR_PLUGIN] = None
        # self.prepopulatedFields[ExportFieldNames.IR_OPTIONS] = [ExportFieldNames.IR_VERSION_NONE, ExportFieldNames.IR_VERSION_16, ExportFieldNames.IR_VERSION_40]

        self.sh_type = sh_type

    def getStepName(self):
        return StepNames.EXPORT

    def updateSavedObjectsFromSavedFields(self):
        pass
