# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from django.conf import settings
from iondb.rundb.models import ReferenceGenome
import logging
from iondb.rundb.plan.views_helper import dict_bed_hotspot
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames
logger = logging.getLogger(__name__)

class ReferenceFieldNames():

    REFERENCE = 'reference'
    REFERENCES = 'references'
    BED_FILES = 'bedFiles'
    TARGET_BED_FIELS = 'targetBedFiles'
    HOT_SPOT_FILES = 'hotspotFiles'
    HOT_SPOT_BED_FILE = 'hotSpotBedFile'
    HOT_SPOT_BED_FILES = 'hotSpotBedFiles'
    SHOW_HOT_SPOT_BED = 'showHotSpotBed'
    REFERENCE_MISSING = 'referenceMissing'
    TARGED_BED_FILE_MISSING = 'targetBedFileMissing'
    HOT_SPOT_BED_FILE_MISSING = 'hotSpotBedFileMissing'
    TARGET_BED_FILE = 'targetBedFile'
    REFERENCE_SHORT_NAMES = 'referenceShortNames'
    BED_FILE_FULL_PATHS = 'bedFileFullPaths'
    BED_FILE_PATHS = 'bedFilePaths'



class ReferenceStepData(AbstractStepData):

    def __init__(self):
        super(ReferenceStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_reference.html'
        self._dependsOn = [StepNames.APPLICATION]
        references = list(ReferenceGenome.objects.all().filter(index_version=settings.TMAP_VERSION))
        self.file_dict = dict_bed_hotspot()
        self.prepopulatedFields[ReferenceFieldNames.REFERENCES] = references
        self.prepopulatedFields[ReferenceFieldNames.TARGET_BED_FIELS] = self.file_dict[ReferenceFieldNames.BED_FILES]
        self.prepopulatedFields[ReferenceFieldNames.HOT_SPOT_BED_FILES] = self.file_dict[ReferenceFieldNames.HOT_SPOT_FILES]
        self.prepopulatedFields[ReferenceFieldNames.SHOW_HOT_SPOT_BED] = False
        self.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = False
        self.prepopulatedFields[ReferenceFieldNames.TARGED_BED_FILE_MISSING] = False
        self.prepopulatedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE_MISSING] = False
        self.savedFields[ReferenceFieldNames.REFERENCE] = None
        self.savedFields[ReferenceFieldNames.TARGET_BED_FILE] = None
        self.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] = None
        
        self.prepopulatedFields[ReferenceFieldNames.REFERENCE_SHORT_NAMES] = [ref.short_name for ref in references]


    def getStepName(self):
        return StepNames.REFERENCE

    def updateSavedObjectsFromSavedFields(self):
        pass
    
    def updateFromStep(self, updated_step):
        if updated_step.getStepName() == StepNames.APPLICATION and updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:
            applProduct = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]
            logger.debug("Updating reference for applproduct %s" % applProduct.productCode)
            
            if updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultGenomeRefName:
                self.savedFields[ReferenceFieldNames.REFERENCE] = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultGenomeRefName
                self.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = True
                if self.savedFields[ReferenceFieldNames.REFERENCE] in [ref.short_name for ref in self.prepopulatedFields[ReferenceFieldNames.REFERENCES]]:
                    self.prepopulatedFields[ReferenceFieldNames.REFERENCE_MISSING] = False
            
            if updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultTargetRegionBedFileName:
                self.savedFields[ReferenceFieldNames.TARGET_BED_FILE] = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultTargetRegionBedFileName
                self.prepopulatedFields[ReferenceFieldNames.TARGED_BED_FILE_MISSING] = True
                if self.savedFields[ReferenceFieldNames.TARGET_BED_FILE] in self.file_dict[ReferenceFieldNames.BED_FILE_FULL_PATHS] or\
                   self.savedFields[ReferenceFieldNames.TARGET_BED_FILE] in self.file_dict[ReferenceFieldNames.BED_FILE_PATHS]:
                    self.prepopulatedFields[ReferenceFieldNames.TARGED_BED_FILE_MISSING] = False
            
            if updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultHotSpotRegionBedFileName:
                self.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].defaultHotSpotRegionBedFileName
                self.prepopulatedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE_MISSING] = True
                if self.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] in self.file_dict[ReferenceFieldNames.BED_FILE_FULL_PATHS] or\
                   self.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] in self.file_dict[ReferenceFieldNames.BED_FILE_PATHS]:
                    self.prepopulatedFields["hotSpotBedFileMissing"] = False
            
            if updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT].isHotspotRegionBEDFileSuppported:
                self.prepopulatedFields[ReferenceFieldNames.SHOW_HOT_SPOT_BED] = True
            else:
                self.prepopulatedFields[ReferenceFieldNames.SHOW_HOT_SPOT_BED] = False
