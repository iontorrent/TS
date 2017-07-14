# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from django.conf import settings
from iondb.rundb.models import ReferenceGenome
from iondb.utils import validation

from iondb.rundb.plan.views_helper import dict_bed_hotspot
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames
from iondb.rundb.plan.page_plan.step_helper_types import StepHelperType

from iondb.rundb.plan.plan_validator import validate_targetRegionBedFile_for_runType

import logging
logger = logging.getLogger(__name__)


class ReferenceFieldNames():

    BED_FILES = 'bedFiles'
    BED_FILE_FULL_PATHS = 'bedFileFullPaths'
    BED_FILE_PATHS = 'bedFilePaths'
    HOT_SPOT_BED_FILE = 'default_hotSpotBedFile'
    HOT_SPOT_BED_FILES = 'hotSpotBedFiles'
    HOT_SPOT_BED_FILE_MISSING = 'hotSpotBedFileMissing'
    HOT_SPOT_FILES = 'hotspotFiles'
    HOT_SPOT_FULL_PATHS = 'hotspotFullPaths'
    HOT_SPOT_PATHS = 'hotspotPaths'

    MIXED_TYPE_RNA_HOT_SPOT_BED_FILE = "mixedTypeRNA_hotSpotBedFile"
    MIXED_TYPE_RNA_REFERENCE = "mixedTypeRNA_reference"
    MIXED_TYPE_RNA_REFERENCE_MISSING = "mixedTypeRNA_referenceMissing"
    MIXED_TYPE_RNA_TARGET_BED_FILE = "mixedTypeRNA_targetBedFile"
    MIXED_TYPE_RNA_TARGET_BED_FILE_MISSING = "mixedTypeRNA_targetBedFileMissing"

    SSE_BED_FILE = "sseBedFile"
    SSE_BED_FILE_DICT = "sseBedFileDict"

    REFERENCE = 'default_reference'
    REFERENCES = 'references'
    REFERENCE_MISSING = 'referenceMissing'
    REFERENCE_SHORT_NAMES = 'referenceShortNames'
    REQUIRE_TARGET_BED_FILE = "requireTargetBedFile"

    SAME_REF_INFO_PER_SAMPLE = "isSameRefInfoPerSample"

    SHOW_HOT_SPOT_BED = 'showHotSpotBed'
    TARGET_BED_FILE = 'default_targetBedFile'
    TARGET_BED_FILES = 'targetBedFiles'
    TARGET_BED_FILE_MISSING = 'targetBedFileMissing'

    PLAN_STATUS = "planStatus"
    RUN_TYPE = "runType"
    APPLICATION_GROUP_NAME = "applicationGroupName"


class ReferenceStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(ReferenceStepData, self).__init__(sh_type)
        self.resourcePath = 'rundb/plan/page_plan/page_plan_reference.html'
        self._dependsOn = [StepNames.APPLICATION]
        references = list(ReferenceGenome.objects.all().filter(index_version=settings.TMAP_VERSION))
        self.file_dict = dict_bed_hotspot()
        self.prepopulatedFields[ReferenceFieldNames.REFERENCES] = references
        self.prepopulatedFields[ReferenceFieldNames.TARGET_BED_FILES] = self.file_dict[ReferenceFieldNames.BED_FILES]
        self.prepopulatedFields[ReferenceFieldNames.HOT_SPOT_BED_FILES] = self.file_dict[ReferenceFieldNames.HOT_SPOT_FILES]
        self.prepopulatedFields[ReferenceFieldNames.SHOW_HOT_SPOT_BED] = False
        self.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = False
        self.prepopulatedFields[ReferenceFieldNames.TARGET_BED_FILE_MISSING] = False
        self.prepopulatedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE_MISSING] = False
        self.savedFields[ReferenceFieldNames.REFERENCE] = ""
        self.savedFields[ReferenceFieldNames.TARGET_BED_FILE] = ""
        self.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] = ""
        self.prepopulatedFields[ReferenceFieldNames.REQUIRE_TARGET_BED_FILE] = False
        self.prepopulatedFields[ReferenceFieldNames.SSE_BED_FILE_DICT] = {}
        self.prepopulatedFields[ReferenceFieldNames.REFERENCE_SHORT_NAMES] = [ref.short_name for ref in references]

        self.savedFields[ReferenceFieldNames.MIXED_TYPE_RNA_REFERENCE] = ""
        self.savedFields[ReferenceFieldNames.MIXED_TYPE_RNA_TARGET_BED_FILE] = ""
        self.savedFields[ReferenceFieldNames.MIXED_TYPE_RNA_HOT_SPOT_BED_FILE] = ""
        self.savedFields[ReferenceFieldNames.SAME_REF_INFO_PER_SAMPLE] = True

        self.prepopulatedFields[ReferenceFieldNames.PLAN_STATUS] = ""

        self.prepopulatedFields[ReferenceFieldNames.RUN_TYPE] = ""
        self.prepopulatedFields[ReferenceFieldNames.APPLICATION_GROUP_NAME] = ""

        self.sh_type = sh_type

    def getStepName(self):
        return StepNames.REFERENCE

    def updateSavedObjectsFromSavedFields(self):
        pass

    def validate(self):
        pass

    def hasErrors(self):
        """
        Now that reference step is a section of another step. It is crucial not to advertise having errors or
        user will be re-directed to reference's resourcePath for error correction.
        Let the parent step take care of the error broadcasting.
        """
        return False

    # If the plan used to have a missing ref, check if the new ref is missing. If not, clear the missing flag.
    def validateField(self, field_name, new_field_value):
#         if field_name == ReferenceFieldNames.REFERENCE:
#             if new_field_value in [ref.short_name for ref in self.prepopulatedFields[ReferenceFieldNames.REFERENCES]]:
#                 self.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = False
#
#         if field_name == ReferenceFieldNames.TARGET_BED_FILE:
#             reference = self.savedFields[ReferenceFieldNames.REFERENCE]
#
#             if self.prepopulatedFields[ReferenceFieldNames.REQUIRE_TARGET_BED_FILE]:
#                 if reference:
#                     if validation.has_value(new_field_value):
#                         self.validationErrors.pop(field_name, None)
#                     else:
#                         self.validationErrors[field_name] = validation.required_error("Target Regions BED File")
#                 else:
#                     self.validationErrors.pop(field_name, None)

        pass

    def validateField_in_section(self, field_name, new_field_value):
        """
        field validation for a step that acts as a section to another step
        """
        # logger.debug("at validateField_in_section field_name=%s; new_field_value=%s" %(field_name, new_field_value))

        if field_name == ReferenceFieldNames.REFERENCE:
            if new_field_value and new_field_value not in [ref.short_name for ref in self.prepopulatedFields[ReferenceFieldNames.REFERENCES]]:
                self.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = True
                self.validationErrors[field_name] = "Reference Library not found: %s" % new_field_value
            else:
                self.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = False
                self.validationErrors.pop(field_name, None)

        # if the plan has been sequenced, do not enforce the target bed file to be selected

        if self.prepopulatedFields[ReferenceFieldNames.PLAN_STATUS] != "run":
            if field_name == ReferenceFieldNames.TARGET_BED_FILE:
                reference = self.savedFields[ReferenceFieldNames.REFERENCE]
                targetRegionBedFile = new_field_value

                runType = self.prepopulatedFields[ReferenceFieldNames.RUN_TYPE] if self.prepopulatedFields[ReferenceFieldNames.RUN_TYPE] else ""
                applicationGroupName = self.prepopulatedFields[ReferenceFieldNames.APPLICATION_GROUP_NAME]
                logger.debug(" validateField_in_section reference=%s; targetBed=%s; runType=%s; applicationGroupName=%s" % (reference, new_field_value, runType, applicationGroupName))

                errors = []
                errors = validate_targetRegionBedFile_for_runType(targetRegionBedFile, runType, reference, "", applicationGroupName, "Target Regions BED File")
                if errors:
                    self.validationErrors[field_name] = ''.join(errors)
                else:
                    self.validationErrors.pop(field_name, None)

    def updateFromStep(self, updated_step):
        if updated_step.getStepName() == StepNames.APPLICATION and updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:
            applProduct = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]

            logger.debug("Updating reference for applproduct %s; planStatus=%s" % (applProduct.productCode, self.prepopulatedFields[ReferenceFieldNames.PLAN_STATUS]))

            if self.sh_type in [StepHelperType.CREATE_NEW_TEMPLATE, StepHelperType.CREATE_NEW_TEMPLATE_BY_SAMPLE, StepHelperType.CREATE_NEW_PLAN, StepHelperType.CREATE_NEW_PLAN_BY_SAMPLE]:
                if updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultGenomeRefName:
                    self.savedFields[ReferenceFieldNames.REFERENCE] = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultGenomeRefName
                    self.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = True

                    if self.savedFields[ReferenceFieldNames.REFERENCE] in [ref.short_name for ref in self.prepopulatedFields[ReferenceFieldNames.REFERENCES]]:
                        self.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = False

                if updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultTargetRegionBedFileName:
                    self.savedFields[ReferenceFieldNames.TARGET_BED_FILE] = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultTargetRegionBedFileName
                    self.prepopulatedFields[ReferenceFieldNames.TARGET_BED_FILE_MISSING] = True

                    if self.savedFields[ReferenceFieldNames.TARGET_BED_FILE] in self.file_dict[ReferenceFieldNames.BED_FILE_FULL_PATHS] or\
                       self.savedFields[ReferenceFieldNames.TARGET_BED_FILE] in self.file_dict[ReferenceFieldNames.BED_FILE_PATHS]:
                        self.prepopulatedFields[ReferenceFieldNames.TARGET_BED_FILE_MISSING] = False

                if updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultHotSpotRegionBedFileName:
                    self.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultHotSpotRegionBedFileName
                    self.prepopulatedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE_MISSING] = True

                    if self.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] in self.file_dict[ReferenceFieldNames.HOT_SPOT_FULL_PATHS] or\
                       self.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] in self.file_dict[ReferenceFieldNames.HOT_SPOT_PATHS]:
                        self.prepopulatedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE_MISSING] = False

            if updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].isHotspotRegionBEDFileSuppported:
                self.prepopulatedFields[ReferenceFieldNames.SHOW_HOT_SPOT_BED] = True
            else:
                self.prepopulatedFields[ReferenceFieldNames.SHOW_HOT_SPOT_BED] = False

            if updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].isTargetRegionBEDFileSelectionRequiredForRefSelection:
                self.prepopulatedFields[ReferenceFieldNames.REQUIRE_TARGET_BED_FILE] = True
            else:
                self.prepopulatedFields[ReferenceFieldNames.REQUIRE_TARGET_BED_FILE] = False

            self.prepopulatedFields[ReferenceFieldNames.RUN_TYPE] = updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE].runType


    def get_sseBedFile(self, targetRegionBedFile):
        sse = ''
        if targetRegionBedFile and self.prepopulatedFields[ReferenceFieldNames.SSE_BED_FILE_DICT]:
            sse = self.prepopulatedFields[ReferenceFieldNames.SSE_BED_FILE_DICT].get(targetRegionBedFile.split('/')[-1], '')
        return sse

