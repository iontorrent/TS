# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from django.conf import settings
from iondb.utils import validation

from iondb.rundb.models import AnalysisArgs

from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames
from iondb.rundb.plan.page_plan.kits_step_data import KitsFieldNames
from iondb.rundb.plan.page_plan.step_helper_types import StepHelperType


import logging
logger = logging.getLogger(__name__)


class AnalysisParamsFieldNames():

    # AP_DICT = "analysisParamsDict"
    AP_ENTRIES = "analysisParamsEntries"
    AP_DISPLAYED_NAMES = "analysisParamsNames"
    AP_ENTRY_SELECTED = "analysisParamsEntrySelected"
    AP_CUSTOM = "analysisParamsCustom"

    AP_ENTRY_PREVIOUSLY_SELECTED = "analysisParamsEntryPreviouslySelected"
    AP_ENTRY_SYSTEM_SELECTED = "analysisParamsEntrySystemSelected"

    AP_ENTRY_SELECTED_VALUE = "<Previous custom selection>"
    AP_ENTRY_PREVIOUSLY_SELECTED_VALUE = "<Selection before chip/kits selection change>"
    AP_ENTRY_BEST_MATCH_PLAN_VALUE = "System default for this plan"
    AP_ENTRY_BEST_MATCH_TEMPLATE_VALUE = "System default for this template"

    AP_BEADFIND_SELECTED = "beadFindSelected"
    AP_ANALYSISARGS_SELECTED = "analysisArgsSelected"
    AP_PREBASECALLER_SELECTED = "preBaseCallerSelected"
    AP_CALIBRATE_SELECTED = "calibrateSelected"
    AP_BASECALLER_SELECTED = "baseCallerSelected"
    AP_ALIGNMENT_SELECTED = "alignmentSelected"
    AP_IONSTATS_SELECTED = "ionStatsSelected"

    AP_THUMBNAIL_BEADFIND_SELECTED = "thumbnailBeadFindSelected"
    AP_THUMBNAIL_ANALYSISARGS_SELECTED = "thumbnailAnalysisArgsSelected"
    AP_THUMBNAIL_PREBASECALLER_SELECTED = "thumbnailPreBaseCallerSelected"
    AP_THUMBNAIL_CALIBRATE_SELECTED = "thumbnailCalibrateSelected"
    AP_THUMBNAIL_BASECALLER_SELECTED = "thumbnailBaseCallerSelected"
    AP_THUMBNAIL_ALIGNMENT_SELECTED = "thumbnailAlignmentSelected"
    AP_THUMBNAIL_IONSTATS_SELECTED = "thumbnailIonStatsSelected"

    APPL_PRODUCT = 'applProduct'
    RUN_TYPE = 'runType'
    APPLICATION_GROUP_NAME = "applicationGroupName"

    CHIP_TYPE = 'chipType'

    SAMPLE_PREPARATION_KIT = 'samplePreparationKit'
    LIBRARY_KIT_NAME = 'librarykitname'
    TEMPLATE_KIT_NAME = 'templatekitname'
    SEQUENCE_KIT_NAME = 'sequencekitname'


class AnalysisParamsStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(AnalysisParamsStepData, self).__init__(sh_type)
        self.resourcePath = 'rundb/plan/page_plan/page_plan_analysis_params.html'
        self._dependsOn.append(StepNames.APPLICATION)
        self._dependsOn.append(StepNames.KITS)

        analysisParamsEntries = list(AnalysisArgs.objects.all().filter(active=True))
        self.prepopulatedFields[AnalysisParamsFieldNames.AP_ENTRIES] = analysisParamsEntries
        self.prepopulatedFields[AnalysisParamsFieldNames.AP_DISPLAYED_NAMES] = [ap.description for ap in analysisParamsEntries]

        # self.prepopulatedFields[AnalysisParamsFieldNames.AP_ENTRY_BEST_MATCH] = None

        self.savedObjects[AnalysisParamsFieldNames.AP_ENTRY_SELECTED] = ""

        self.savedObjects[AnalysisParamsFieldNames.AP_ENTRY_PREVIOUSLY_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_BEADFIND_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_ANALYSISARGS_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_PREBASECALLER_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_CALIBRATE_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_BASECALLER_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_ALIGNMENT_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_IONSTATS_SELECTED] = ""

        self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_BEADFIND_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_ANALYSISARGS_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_PREBASECALLER_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_CALIBRATE_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_BASECALLER_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_ALIGNMENT_SELECTED] = ""
        self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_IONSTATS_SELECTED] = ""

        self.savedFields[AnalysisParamsFieldNames.AP_CUSTOM] = "False"

        self.prepopulatedFields[AnalysisParamsFieldNames.APPL_PRODUCT] = ""
        self.prepopulatedFields[AnalysisParamsFieldNames.RUN_TYPE] = ""
        self.prepopulatedFields[AnalysisParamsFieldNames.APPLICATION_GROUP_NAME] = ""

        self.prepopulatedFields[AnalysisParamsFieldNames.CHIP_TYPE] = ""

        self.prepopulatedFields[AnalysisParamsFieldNames.SAMPLE_PREPARATION_KIT] = ""
        self.prepopulatedFields[AnalysisParamsFieldNames.LIBRARY_KIT_NAME] = ""
        self.prepopulatedFields[AnalysisParamsFieldNames.TEMPLATE_KIT_NAME] = ""
        self.prepopulatedFields[AnalysisParamsFieldNames.SEQUENCE_KIT_NAME] = ""

        self.sh_type = sh_type

    def getStepName(self):
        return StepNames.ANALYSIS_PARAMS

    def updateSavedObjectsFromSavedFields(self):
        pass

    def validate(self):
        pass

    def hasErrors(self):
        """
        This step is a section of another step. It is crucial not to advertise having errors or
        user will be re-directed to analysis args' resourcePath for error correction.
        Let the parent step take care of the error broadcasting.
        """
        return False

    def validateField(self, field_name, new_field_value):
        pass

    def validateField_in_section(self, field_name, new_field_value):
        """
        field validation for a step that acts as a section to another step
        """
        logger.debug("at analysis_params_step_data.validateField_in_section field_name=%s; new_field_value=%s" % (field_name, new_field_value))

        pass

    def updateFromStep(self, updated_step):
        if updated_step.getStepName() not in self._dependsOn:
            return

        # reset best match if key attributes on other chevrons have changed
        needToRefreshSelectionList = False

        if updated_step.getStepName() == StepNames.APPLICATION and updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:

            applProduct = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]

            logger.debug("analysis_params_step_data - APPL CHEVRON...  applProduct %s; runType=%s; applicationGroupName=%s" \
                         % (applProduct.productCode, updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE].runType, updated_step.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME]))

            if self.prepopulatedFields[AnalysisParamsFieldNames.RUN_TYPE]  !=  updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE].runType or \
                    self.prepopulatedFields[AnalysisParamsFieldNames.APPLICATION_GROUP_NAME] != updated_step.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME]:
                needToRefreshSelectionList = True

            self.prepopulatedFields[AnalysisParamsFieldNames.APPL_PRODUCT] = ""
            self.prepopulatedFields[AnalysisParamsFieldNames.RUN_TYPE] = updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE].runType
            self.prepopulatedFields[AnalysisParamsFieldNames.APPLICATION_GROUP_NAME] = updated_step.savedFields[ApplicationFieldNames.APPLICATION_GROUP_NAME]

        elif updated_step.getStepName() == StepNames.KITS:

            if self.prepopulatedFields[AnalysisParamsFieldNames.CHIP_TYPE] != updated_step.savedFields[KitsFieldNames.CHIP_TYPE] or \
                self.prepopulatedFields[AnalysisParamsFieldNames.SAMPLE_PREPARATION_KIT] != updated_step.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT] or \
                self.prepopulatedFields[AnalysisParamsFieldNames.LIBRARY_KIT_NAME] != updated_step.savedFields[KitsFieldNames.LIBRARY_KIT_NAME] or \
                self.prepopulatedFields[AnalysisParamsFieldNames.TEMPLATE_KIT_NAME] != updated_step.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME] or \
                    self.prepopulatedFields[AnalysisParamsFieldNames.SEQUENCE_KIT_NAME] != updated_step.savedFields[KitsFieldNames.SEQUENCE_KIT_NAME]:

                needToRefreshSelectionList = True

            self.prepopulatedFields[AnalysisParamsFieldNames.CHIP_TYPE] = updated_step.savedFields[KitsFieldNames.CHIP_TYPE] if updated_step.savedFields[KitsFieldNames.CHIP_TYPE] else ""

            self.prepopulatedFields[AnalysisParamsFieldNames.SAMPLE_PREPARATION_KIT] = updated_step.savedFields[KitsFieldNames.SAMPLE_PREPARATION_KIT]
            self.prepopulatedFields[AnalysisParamsFieldNames.LIBRARY_KIT_NAME] = updated_step.savedFields[KitsFieldNames.LIBRARY_KIT_NAME]
            self.prepopulatedFields[AnalysisParamsFieldNames.TEMPLATE_KIT_NAME] = updated_step.savedFields[KitsFieldNames.TEMPLATE_KIT_NAME]
            self.prepopulatedFields[AnalysisParamsFieldNames.SEQUENCE_KIT_NAME] = updated_step.savedFields[KitsFieldNames.SEQUENCE_KIT_NAME]

        if needToRefreshSelectionList:
            self. _update_analysisParamsData_selection_list(self.prepopulatedFields[AnalysisParamsFieldNames.CHIP_TYPE],
                                                            self.prepopulatedFields[AnalysisParamsFieldNames.SEQUENCE_KIT_NAME],
                                                            self.prepopulatedFields[AnalysisParamsFieldNames.TEMPLATE_KIT_NAME],
                                                            self.prepopulatedFields[AnalysisParamsFieldNames.LIBRARY_KIT_NAME],
                                                            self.prepopulatedFields[AnalysisParamsFieldNames.SAMPLE_PREPARATION_KIT],
                                                            self.prepopulatedFields[AnalysisParamsFieldNames.RUN_TYPE],
                                                            self.prepopulatedFields[AnalysisParamsFieldNames.APPLICATION_GROUP_NAME]
                                                            )

    def _update_analysisParamsData_selection_list(self, chipType, sequenceKitName, templatingKitName, libraryKitName, samplePrepKitName, applicationType, applicationGroupName):

        possible_match_entries = AnalysisArgs.possible_matches(chipType, sequenceKitName, templatingKitName, libraryKitName, samplePrepKitName,
                                                               None, applicationType, applicationGroupName)

        logger.debug("_update_analysisParamsData_selection_list() applicationType=%s; applicationGroupName=%s" % (applicationType, applicationGroupName))
        best_match_entry = AnalysisArgs.best_match(chipType, sequenceKitName, templatingKitName, libraryKitName, samplePrepKitName, None, applicationType, applicationGroupName);

        if best_match_entry:

            isTemplate = self.sh_type in StepHelperType.TEMPLATE_TYPES

            for ap in possible_match_entries:
                if ap.name == best_match_entry.name:
                    ap.name = AnalysisParamsFieldNames.AP_ENTRY_BEST_MATCH_TEMPLATE_VALUE if isTemplate else AnalysisParamsFieldNames.AP_ENTRY_BEST_MATCH_PLAN_VALUE
                    ap.best_match = True

            if self.savedFields[AnalysisParamsFieldNames.AP_CUSTOM] == "False":
                previously_selected_analysisArgs = {
                    'description': AnalysisParamsFieldNames.AP_ENTRY_PREVIOUSLY_SELECTED_VALUE,
                    'name': '',
                    'beadfindargs': self.savedFields[AnalysisParamsFieldNames.AP_BEADFIND_SELECTED],
                    'analysisargs': self.savedFields[AnalysisParamsFieldNames.AP_ANALYSISARGS_SELECTED],
                    'prebasecallerargs': self.savedFields[AnalysisParamsFieldNames.AP_PREBASECALLER_SELECTED],
                    'calibrateargs': self.savedFields[AnalysisParamsFieldNames.AP_CALIBRATE_SELECTED],
                    'basecallerargs': self.savedFields[AnalysisParamsFieldNames.AP_BASECALLER_SELECTED],
                    'alignmentargs': self.savedFields[AnalysisParamsFieldNames.AP_ALIGNMENT_SELECTED],
                    'ionstatsargs': self.savedFields[AnalysisParamsFieldNames.AP_IONSTATS_SELECTED],

                    'thumbnailbeadfindargs': self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_BEADFIND_SELECTED],
                    'thumbnailanalysisargs': self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_ANALYSISARGS_SELECTED],
                    'prethumbnailbasecallerargs': self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_PREBASECALLER_SELECTED],
                    'thumbnailcalibrateargs': self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_CALIBRATE_SELECTED],
                    'thumbnailbasecallerargs': self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_BASECALLER_SELECTED],
                    'thumbnailalignmentargs': self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_ALIGNMENT_SELECTED],
                    'thumbnailionstatsargs': self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_IONSTATS_SELECTED],
                }

                self.savedObjects[AnalysisParamsFieldNames.AP_ENTRY_PREVIOUSLY_SELECTED] = previously_selected_analysisArgs

                best_match_entry.description = AnalysisParamsFieldNames.AP_ENTRY_SELECTED_VALUE
                self.savedObjects[AnalysisParamsFieldNames.AP_ENTRY_SELECTED] = best_match_entry

                self.savedFields[AnalysisParamsFieldNames.AP_BEADFIND_SELECTED] = best_match_entry.beadfindargs
                self.savedFields[AnalysisParamsFieldNames.AP_ANALYSISARGS_SELECTED] = best_match_entry.analysisargs
                self.savedFields[AnalysisParamsFieldNames.AP_PREBASECALLER_SELECTED] = best_match_entry.prebasecallerargs
                self.savedFields[AnalysisParamsFieldNames.AP_CALIBRATE_SELECTED] = best_match_entry.calibrateargs
                self.savedFields[AnalysisParamsFieldNames.AP_BASECALLER_SELECTED] = best_match_entry.basecallerargs
                self.savedFields[AnalysisParamsFieldNames.AP_ALIGNMENT_SELECTED] = best_match_entry.alignmentargs
                self.savedFields[AnalysisParamsFieldNames.AP_IONSTATS_SELECTED] = best_match_entry.ionstatsargs

                self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_BEADFIND_SELECTED] = best_match_entry.thumbnailbeadfindargs
                self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_ANALYSISARGS_SELECTED] = best_match_entry.thumbnailanalysisargs
                self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_PREBASECALLER_SELECTED] = best_match_entry.prethumbnailbasecallerargs
                self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_CALIBRATE_SELECTED] = best_match_entry. thumbnailcalibrateargs
                self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_BASECALLER_SELECTED] = best_match_entry.thumbnailbasecallerargs
                self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_ALIGNMENT_SELECTED] = best_match_entry.thumbnailalignmentargs
                self.savedFields[AnalysisParamsFieldNames.AP_THUMBNAIL_IONSTATS_SELECTED] = best_match_entry. thumbnailionstatsargs

        else:
            logger.debug("analysis_params_step_data._update_analysisParamsData_selection_list() BEST MATCH NOT FOUND!!! chipType=%s;" % (chipType))

        self.prepopulatedFields[AnalysisParamsFieldNames.AP_ENTRIES] = possible_match_entries
        self.prepopulatedFields[AnalysisParamsFieldNames.AP_DISPLAYED_NAMES] = [ap.description for ap in possible_match_entries]
