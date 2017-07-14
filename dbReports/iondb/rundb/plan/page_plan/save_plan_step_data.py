# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from django.core.urlresolvers import reverse
from iondb.rundb.plan.page_plan.kits_step_data import KitsFieldNames
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import dnaBarcode, SampleAnnotation_CV, KitInfo, QCType
from iondb.utils import validation

from iondb.rundb.plan.page_plan.step_helper_types import StepHelperType
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames
from iondb.rundb.plan.page_plan.reference_step_data import ReferenceFieldNames
from iondb.rundb.plan.page_plan.ionreporter_step_data import IonReporterFieldNames
from iondb.rundb.plan.plan_validator import validate_plan_name, validate_notes, validate_sample_name, validate_sample_tube_label, \
    validate_barcode_sample_association, validate_QC, validate_targetRegionBedFile_for_runType, validate_chipBarcode

from iondb.utils.utils import convert

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

# from iondb.utils import toBoolean

import json
import logging
logger = logging.getLogger(__name__)

MAX_LENGTH_SAMPLE_DESCRIPTION = 1024
MAX_LENGTH_SAMPLE_NAME = 127


class SavePlanFieldNames():
    UPLOADERS = 'uploaders'
    SET_ID = 'setid'
    SETID_SUFFIX = 'setid_suffix'
    EXTERNAL_ID = 'externalId'
    WORKFLOW = 'Workflow'
    GENDER = 'Gender'
    CANCER_TYPE = "cancerType"
    CELLULARITY_PCT = "cellularityPct"
    BIOPSY_DAYS = "biopsyDays"
    COUPLE_ID = "coupleID"
    EMBRYO_ID = "embryoID"
    NUCLEOTIDE_TYPE = "NucleotideType"
    RELATIONSHIP_TYPE = 'Relation'
    RELATION_ROLE = 'RelationRole'
    PLAN_NAME = 'planName'
    SAMPLE = 'sample'
    NOTE = 'note'
    SELECTED_IR = 'selectedIr'
    IR_CONFIG_JSON = 'irConfigJson'
    BARCODE_SET = 'barcodeSet'
    BARCODE_SAMPLE_TUBE_LABEL = 'barcodeSampleTubeLabel'
    BARCODE_TO_SAMPLE = 'barcodeToSample'
    BARCODE_SETS = 'barcodeSets'
    BARCODE_SETS_BARCODES = 'barcodeSets_barcodes'
    SAMPLE_TO_BARCODE = 'sampleToBarcode'
    SAMPLE_EXTERNAL_ID = 'sampleExternalId'
    SAMPLE_NAME = 'sampleName'
    SAMPLE_DESCRIPTION = 'sampleDescription'
    TUBE_LABEL = 'tubeLabel'
    CHIP_BARCODE_LABEL = 'chipBarcodeLabel'
    CHIP_BARCODE = 'chipBarcode'

    IR_PLUGIN_ENTRIES = 'irPluginEntries'
    IR_GENDER = 'irGender'
    IR_CANCER_TYPE = "ircancerType"
    IR_CELLULARITY_PCT = "ircellularityPct"
    IR_BIOPSY_DAYS = "irbiopsyDays"
    IR_COUPLE_ID = "ircoupleID"
    IR_EMBRYO_ID = "irembryoID"

    IR_WORKFLOW = 'irWorkflow'
    IR_ISFACTORY = 'tag_isFactoryProvidedWorkflow'
    IR_DOWN = 'irDown'
    IR_RELATION_ROLE = 'irRelationRole'
    IR_RELATIONSHIP_TYPE = 'irRelationshipType'
    IR_APPLICATION_TYPE = 'ApplicationType'
    IR_SET_ID = 'irSetID'

    BAD_SAMPLE_NAME = 'bad_sample_name'
    BAD_SAMPLE_EXTERNAL_ID = 'bad_sample_external_id'
    BAD_SAMPLE_DESCRIPTION = 'bad_sample_description'
    BAD_TUBE_LABEL = 'bad_tube_label'
    BAD_CHIP_BARCODE = 'bad_chip_barcode'

    BARCODE_SAMPLE_NUCLEOTIDE_TYPE = "nucleotideType"
    BARCODE_SAMPLE_REFERENCE = "reference"
    BARCODE_SAMPLE_TARGET_REGION_BED_FILE = "targetRegionBedFile"
    BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE = "hotSpotRegionBedFile"
    BARCODE_SAMPLE_SSE_BED_FILE = "sseBedFile"
    BARCODE_SAMPLE_CONTROL_SEQ_TYPE = "controlSequenceType"
    BARCODE_SAMPLE_CONTROL_TYPE = "controlType"

    BARCODE_SAMPLE_INFO = 'barcodeSampleInfo'
    NO_SAMPLES = 'no_samples'
    DESCRIPTION = 'description'
    BAD_IR_SET_ID = 'badIrSetId'
    SAMPLE_ANNOTATIONS = 'sampleAnnotations'

    CONTROL_SEQ_TYPES = "controlSeqTypes"
    BARCODE_KIT_SELECTABLE_TYPE = "barcodeKitSelectableType"

    PLAN_REFERENCE = "plan_reference"
    PLAN_TARGET_REGION_BED_FILE = "plan_targetRegionBedFile"
    PLAN_HOTSPOT_REGION_BED_FILE = "plan_hotSpotRegionBedFile"
    SAMPLES_TABLE_LIST = "samplesTableList"
    SAMPLES_TABLE = "samplesTable"
    NUM_SAMPLES = "numberOfSamples"

    APPL_PRODUCT = "applProduct"

    RUN_TYPE = "runType"
    ONCO_SAME_SAMPLE = "isOncoSameSample"

    REFERENCE_STEP_HELPER = "referenceStepHelper"

    NO_BARCODE = 'no_barcode'
    BAD_BARCODES = 'bad_barcodes'

    APPLICATION_TYPE = "applicationType"
    FIRE_VALIDATION = "fireValidation"

    LIMS_META = 'LIMS_meta'
    META = 'meta'
    APPLICATION_GROUP_NAME = "applicationGroupName"
    HAS_PGS_DATA = "hasPgsData"
    HAS_ONCO_DATA = "hasOncoData"


class MonitoringFieldNames():
    QC_TYPES = 'qcTypes'


def update_ir_plugin_from_samples_table(samplesTable):
    # save IR fields for non-barcoded and barcoded plans
    userInputInfo = []
    for row in samplesTable:
        sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME, '').strip()
        if sample_name:
            ir_userinput_dict = {
                SavePlanFieldNames.SAMPLE: sample_name,
                SavePlanFieldNames.SAMPLE_NAME: sample_name.replace(' ', '_'),
                SavePlanFieldNames.SAMPLE_EXTERNAL_ID: row.get(SavePlanFieldNames.SAMPLE_EXTERNAL_ID, ''),
                SavePlanFieldNames.SAMPLE_DESCRIPTION: row.get(SavePlanFieldNames.SAMPLE_DESCRIPTION, ''),

                SavePlanFieldNames.WORKFLOW: row.get(SavePlanFieldNames.IR_WORKFLOW, ''),
                SavePlanFieldNames.IR_ISFACTORY: row.get(SavePlanFieldNames.IR_ISFACTORY),
                SavePlanFieldNames.RELATION_ROLE: row.get(SavePlanFieldNames.IR_RELATION_ROLE, ''),
                SavePlanFieldNames.RELATIONSHIP_TYPE: row.get(SavePlanFieldNames.IR_RELATIONSHIP_TYPE, ''),
                SavePlanFieldNames.SET_ID: row.get(SavePlanFieldNames.IR_SET_ID, ''),

                SavePlanFieldNames.GENDER: row.get(SavePlanFieldNames.IR_GENDER, ''),
                SavePlanFieldNames.NUCLEOTIDE_TYPE: row.get(SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE, ''),
                SavePlanFieldNames.CANCER_TYPE: row.get(SavePlanFieldNames.IR_CANCER_TYPE, ""),
                SavePlanFieldNames.CELLULARITY_PCT: row.get(SavePlanFieldNames.IR_CELLULARITY_PCT, ""),
                SavePlanFieldNames.BIOPSY_DAYS:  row.get(SavePlanFieldNames.IR_BIOPSY_DAYS, ""),
                SavePlanFieldNames.COUPLE_ID: row.get(SavePlanFieldNames.IR_COUPLE_ID,  ""),
                SavePlanFieldNames.EMBRYO_ID: row.get(SavePlanFieldNames.IR_EMBRYO_ID, ""),
                SavePlanFieldNames.IR_APPLICATION_TYPE: row.get(SavePlanFieldNames.IR_APPLICATION_TYPE, ''),
            }

            barcode_id = row.get('barcodeId')
            if barcode_id:
                ir_userinput_dict[KitsFieldNames.BARCODE_ID] = barcode_id

            userInputInfo.append(ir_userinput_dict)

    return userInputInfo


class SavePlanStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(SavePlanStepData, self).__init__(sh_type)
        self.resourcePath = 'rundb/plan/page_plan/page_plan_save_plan.html'
        self.prev_step_url = reverse("page_plan_output")
        self.next_step_url = reverse("page_plan_save")

        self.savedFields = OrderedDict()

        self.savedFields[SavePlanFieldNames.PLAN_NAME] = None
        self.savedFields[SavePlanFieldNames.NOTE] = None
        self.savedFields[SavePlanFieldNames.APPLICATION_TYPE] = ''
        self.savedFields[SavePlanFieldNames.IR_DOWN] = '0'

        self.savedFields[SavePlanFieldNames.BARCODE_SET] = ''

        self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = ""
        self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = ""
        self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = ""

        self.prepopulatedFields[SavePlanFieldNames.SELECTED_IR] = None
        self.prepopulatedFields[SavePlanFieldNames.IR_WORKFLOW] = None
        self.prepopulatedFields[SavePlanFieldNames.IR_ISFACTORY] = False
        self.prepopulatedFields[SavePlanFieldNames.IR_CONFIG_JSON] = None
        self.prepopulatedFields[SavePlanFieldNames.SAMPLE_ANNOTATIONS] = list(SampleAnnotation_CV.objects.all().order_by("annotationType", "iRValue"))
        self.savedFields[SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL] = None
        self.savedFields[SavePlanFieldNames.CHIP_BARCODE_LABEL] = None
        self.savedObjects[SavePlanFieldNames.BARCODE_TO_SAMPLE] = OrderedDict()

        self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS] = list(dnaBarcode.objects.filter(active=True).values_list('name', flat=True).distinct().order_by('name'))
        all_barcodes = {}
        for bc in dnaBarcode.objects.filter(active=True).order_by('name', 'index').values('name', 'id_str', 'sequence'):
            all_barcodes.setdefault(bc['name'], []).append(bc)
        self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_BARCODES] = json.dumps(all_barcodes)

        self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE] = OrderedDict()
        self.savedObjects[SavePlanFieldNames.IR_PLUGIN_ENTRIES] = []
        self.prepopulatedFields[SavePlanFieldNames.FIRE_VALIDATION] = "1"

        self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST] = [{"row": "1", 'sampleName': u'Sample 1'}]
        self.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = json.dumps(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST])
        self.prepopulatedFields[SavePlanFieldNames.NUM_SAMPLES] = 1

        self.savedFields[SavePlanFieldNames.ONCO_SAME_SAMPLE] = False

        # logger.debug("save_plan_step_data samplesTable=%s" %(self.savedFields[SavePlanFieldNames.SAMPLES_TABLE]))

        self.prepopulatedFields[SavePlanFieldNames.CONTROL_SEQ_TYPES] = KitInfo.objects.filter(kitType='ControlSequenceKitType', isActive=True).order_by("name")

        self.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER] = None

        self.savedObjects[SavePlanFieldNames.APPL_PRODUCT] = None

        self.savedFields[SavePlanFieldNames.LIMS_META] = None
        self.savedObjects[SavePlanFieldNames.META] = {}

        self.prepopulatedFields[SavePlanFieldNames.APPLICATION_GROUP_NAME] = ""

        self.prepopulatedFields[SavePlanFieldNames.HAS_ONCO_DATA] = False
        self.prepopulatedFields[SavePlanFieldNames.HAS_PGS_DATA] = False

        self._dependsOn.append(StepNames.IONREPORTER)
        self._dependsOn.append(StepNames.APPLICATION)
        self._dependsOn.append(StepNames.KITS)
        # self._dependsOn.append(StepNames.REFERENCE)

        self.sh_type = sh_type

        # Monitoring
        self.qcNames = []
        all_qc_types = list(QCType.objects.all().order_by('qcName'))
        self.prepopulatedFields[MonitoringFieldNames.QC_TYPES] = all_qc_types
        for qc_type in all_qc_types:
            self.savedFields[qc_type.qcName] = qc_type.defaultThreshold
            self.qcNames.append(qc_type.qcName)

    def getStepName(self):
        return StepNames.SAVE_PLAN

    def validateField(self, field_name, new_field_value):
        self.validationErrors.pop(field_name, None)

        # if the plan has been sequenced, do not enforce the target bed file to be selected
        planStatus = self.getDefaultSectionPrepopulatedFieldDict().get("planStatus", "")

        if field_name == SavePlanFieldNames.PLAN_NAME:
            errors = validate_plan_name(new_field_value, 'Plan Name')
            if errors:
                self.validationErrors[field_name] = '\n'.join(errors)
        elif field_name == SavePlanFieldNames.NOTE:
            errors = validate_notes(new_field_value)
            if errors:
                self.validationErrors[field_name] = '\n'.join(errors)
        elif field_name == SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL:
            errors = validate_sample_tube_label(new_field_value)
            if errors:
                self.validationErrors[field_name] = '\n'.join(errors)
        elif field_name == SavePlanFieldNames.CHIP_BARCODE_LABEL:
            errors = validate_chipBarcode(new_field_value)
            if errors:
                self.validationErrors[field_name] = '\n'.join(errors)

        elif field_name in self.qcNames:
            '''
            All qc thresholds must be positive integers
            '''
            errors = validate_QC(new_field_value, field_name)
            if errors:
                self.validationErrors[field_name] = errors[0]
            else:
                self.validationErrors.pop(field_name, None)

        elif field_name == SavePlanFieldNames.SAMPLES_TABLE:
            sample_table_list = json.loads(new_field_value)

            samples_errors = []

            applProduct = self.savedObjects[SavePlanFieldNames.APPL_PRODUCT]
            # applProduct object is not saved yet
            if applProduct:
                isTargetRegionSelectionRequired = applProduct.isTargetRegionBEDFileSelectionRequiredForRefSelection
            else:
                isTargetRegionSelectionRequired = False

            applicationGroupName = self.prepopulatedFields[SavePlanFieldNames.APPLICATION_GROUP_NAME]
            for row in sample_table_list:

                sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME, '').strip()
                if sample_name:
                    sample_nucleotideType = row.get(SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE, "")

                    sampleReference = row.get(SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE, "")
                    sampleTargetRegionBedFile = row.get(SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE, "")

                    runType = self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]

                    errors = []
                    # if the plan has been sequenced, do not enforce the target bed file to be selected
                    if planStatus != "run" and (self.sh_type not in StepHelperType.TEMPLATE_TYPES):
                        errors = validate_targetRegionBedFile_for_runType(sampleTargetRegionBedFile, runType, sampleReference, sample_nucleotideType, applicationGroupName, "Target Regions BED File for " + sample_name)

                    if errors:
                        samples_errors.append('\n'.join(errors))

                if samples_errors:
                    self.validationErrors[field_name] = '\n'.join(samples_errors)

    def validateStep(self):
        any_samples = False
        self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_NAME] = []
        self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID] = []
        self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION] = []
        self.validationErrors[SavePlanFieldNames.BAD_TUBE_LABEL] = []
        self.validationErrors[SavePlanFieldNames.BAD_CHIP_BARCODE] = []
        self.validationErrors[SavePlanFieldNames.BAD_IR_SET_ID] = []

        self.validationErrors.pop(SavePlanFieldNames.NO_BARCODE, None)
        self.validationErrors.pop(SavePlanFieldNames.BAD_BARCODES, None)

        barcodeSet = self.savedFields[SavePlanFieldNames.BARCODE_SET]
        selectedBarcodes = []

        samplesTable = json.loads(self.savedFields[SavePlanFieldNames.SAMPLES_TABLE])
        # logger.debug("save_plan_step_data - anySamples? samplesTable=%s" %(samplesTable))

        for row in samplesTable:
            sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME, '').strip()

            # logger.debug("save_plan_step_data - anySamples? sampleName=%s" %(sample_name))

            if sample_name:
                any_samples = True
                if validate_sample_name(sample_name):
                    self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_NAME].append(sample_name)

                external_id = row.get(SavePlanFieldNames.SAMPLE_EXTERNAL_ID, '')
                if external_id:
                    self.validate_field(external_id, self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID])

                description = row.get(SavePlanFieldNames.SAMPLE_DESCRIPTION, '')
                if description:
                    self.validate_field(description, self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION], False,
                                        MAX_LENGTH_SAMPLE_DESCRIPTION)

                ir_set_id = row.get('irSetId', '')
                if ir_set_id and not (str(ir_set_id).isdigit()):
                    self.validationErrors[SavePlanFieldNames.BAD_IR_SET_ID].append(ir_set_id)

                tube_label = row.get('tubeLabel', '')
                if validate_sample_tube_label(tube_label):
                    self.validationErrors[SavePlanFieldNames.BAD_TUBE_LABEL].append(tube_label)
                chip_barcode = row.get('chipBarcode', '')
                if validate_chipBarcode(chip_barcode):
                    self.validationErrors[SavePlanFieldNames.BAD_CHIP_BARCODE].append(chip_barcode)
                if barcodeSet:
                    selectedBarcodes.append(row.get('barcodeId'))

        if any_samples:
            self.validationErrors.pop(SavePlanFieldNames.NO_SAMPLES, None)
        else:
            self.validationErrors[SavePlanFieldNames.NO_SAMPLES] = "You must enter at least one sample"

        if not self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_NAME]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_SAMPLE_NAME, None)

        if not self.validationErrors[SavePlanFieldNames.BAD_TUBE_LABEL]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_TUBE_LABEL, None)
        if not self.validationErrors[SavePlanFieldNames.BAD_CHIP_BARCODE]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_CHIP_BARCODE, None)

        if not self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID, None)

        if not self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION, None)

        if not self.validationErrors[SavePlanFieldNames.BAD_IR_SET_ID]:
            self.validationErrors.pop(SavePlanFieldNames.BAD_IR_SET_ID, None)

        if barcodeSet:
            errors = validate_barcode_sample_association(selectedBarcodes, barcodeSet)

            myErrors = convert(errors)
            if myErrors.get("MISSING_BARCODE", ""):
                self.validationErrors[SavePlanFieldNames.NO_BARCODE] = myErrors.get("MISSING_BARCODE", "")
            if myErrors.get("DUPLICATE_BARCODE", ""):
                self.validationErrors[SavePlanFieldNames.BAD_BARCODES] = myErrors.get("DUPLICATE_BARCODE", "")

    def validate_field(self, value, bad_samples, validate_leading_chars=True, max_length=MAX_LENGTH_SAMPLE_NAME):
        exists = False
        if value:
            exists = True
            if not validation.is_valid_chars(value):
                bad_samples.append(value)

            if validate_leading_chars and value not in bad_samples and not validation.is_valid_leading_chars(value):
                bad_samples.append(value)

            if value not in bad_samples and not validation.is_valid_length(value, max_length):
                bad_samples.append(value)

        return exists

    def updateSavedObjectsFromSavedFields(self):
        self.prepopulatedFields["fireValidation"] = "0"

        for section, sectionObj in self.step_sections.items():
            sectionObj.updateSavedObjectsFromSavedFields()

            if section == StepNames.REFERENCE:
                self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = sectionObj.savedFields.get(ReferenceFieldNames.REFERENCE, "")
                self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = sectionObj.savedFields.get(ReferenceFieldNames.TARGET_BED_FILE, "")
                self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = sectionObj.savedFields.get(ReferenceFieldNames.HOT_SPOT_BED_FILE, "")
                # logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() REFERENCE reference.savedFields=%s" %(sectionObj.savedFields))

        self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST] = json.loads(self.savedFields[SavePlanFieldNames.SAMPLES_TABLE])
        self.prepopulatedFields[SavePlanFieldNames.NUM_SAMPLES] = len(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST])

        self.savedObjects[SavePlanFieldNames.IR_PLUGIN_ENTRIES] = update_ir_plugin_from_samples_table(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST])

        if self.savedFields[SavePlanFieldNames.BARCODE_SET]:
            planReference = self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]
            planHotSptRegionBedFile = self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE]
            planTargetRegionBedFile = self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE]

            logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() BARCODE_SET PLAN_REFERENCE=%s; TARGET_REGION=%s; HOTSPOT_REGION=%s;" % (planReference, planTargetRegionBedFile, planHotSptRegionBedFile))

            reference_step_helper = self.savedObjects[SavePlanFieldNames.REFERENCE_STEP_HELPER]
            self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE] = {}
            for row in self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]:

                logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() BARCODE_SET LOOP row=%s" % (row))

                sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME, '').strip()
                if sample_name:
                    id_str = row['barcodeId']

                    # update barcodedSamples dict
                    if sample_name not in self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE]:
                        self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name] = {
                            KitsFieldNames.BARCODES: [],
                            SavePlanFieldNames.BARCODE_SAMPLE_INFO: {}
                        }

                    sample_nucleotideType = row.get(SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE, "")

                    sampleReference = row.get(SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE, "")
                    sampleHotSpotRegionBedFile = row.get(SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE, "")
                    sampleTargetRegionBedFile = row.get(SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE, "")

                    runType = self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]

                    # logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() SETTING reference step helper runType=%s; sample_nucleotideType=%s; sampleReference=%s" %(runType, sample_nucleotideType, sampleReference))

                    if runType == "AMPS_DNA_RNA" and sample_nucleotideType == "DNA":
                        if reference_step_helper:
                            if sampleReference != planReference:
                                reference_step_helper.savedFields[ReferenceFieldNames.REFERENCE] = sampleReference
                                # logger.debug("save_plan_step_data DIFF SETTING reference step helper reference=%s" %(sampleReference))

                            if sampleHotSpotRegionBedFile != planHotSptRegionBedFile:
                                reference_step_helper.savedFields[ReferenceFieldNames.HOT_SPOT_BED_FILE] = sampleHotSpotRegionBedFile
                                # logger.debug("save_plan_step_data DIFF SETTING reference step helper hotSpot=%s" %(sampleHotSpotRegionBedFile))

                            if sampleTargetRegionBedFile != planTargetRegionBedFile:
                                reference_step_helper.savedFields[ReferenceFieldNames.TARGET_BED_FILE] = sampleTargetRegionBedFile
                                # logger.debug("save_plan_step_data DIFF SETTING reference step helper targetRegions=%s" %(sampleTargetRegionBedFile))

                    # cascade the reference and BED file info to sample if none specified at the sample level
                    if runType != "AMPS_DNA_RNA":
                        if not sampleReference and self.savedObjects[SavePlanFieldNames.APPL_PRODUCT] and not self.savedObjects[SavePlanFieldNames.APPL_PRODUCT].isReferenceBySampleSupported:
                            # logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() NOT REFERENCE_BY_SAMPLE GOING to set sampleReference to planReference... planReference=%s" %(planReference))

                            sampleReference = planReference
                            sampleHotSpotRegionBedFile = planHotSptRegionBedFile
                            sampleTargetRegionBedFile = planTargetRegionBedFile

                    sseBedFile = ""
                    if reference_step_helper:
                         sseBedFile = reference_step_helper.get_sseBedFile(sampleTargetRegionBedFile)

                    self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name][KitsFieldNames.BARCODES].append(id_str)
                    self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name][SavePlanFieldNames.BARCODE_SAMPLE_INFO][id_str] = \
                        {
                            SavePlanFieldNames.EXTERNAL_ID: row.get(SavePlanFieldNames.SAMPLE_EXTERNAL_ID, ''),
                            SavePlanFieldNames.DESCRIPTION: row.get(SavePlanFieldNames.SAMPLE_DESCRIPTION, ''),

                            SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE: sample_nucleotideType,

                            SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE: sampleReference,
                            SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE: sampleTargetRegionBedFile,
                            SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE: sampleHotSpotRegionBedFile,
                            SavePlanFieldNames.BARCODE_SAMPLE_SSE_BED_FILE: sseBedFile,

                            SavePlanFieldNames.BARCODE_SAMPLE_CONTROL_SEQ_TYPE: row.get(SavePlanFieldNames.BARCODE_SAMPLE_CONTROL_SEQ_TYPE, ""),
                            SavePlanFieldNames.BARCODE_SAMPLE_CONTROL_TYPE: row.get(SavePlanFieldNames.BARCODE_SAMPLE_CONTROL_TYPE, ""),
                        }

                    # logger.debug("save_plan_step_data.updateSavedObjectsFromSaveFields() sampleName=%s; id_str=%s; savedObjects=%s" %(sample_name, id_str, self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][sample_name][SavePlanFieldNames.BARCODE_SAMPLE_INFO][id_str]))
                    # logger.debug("save_plan_step_date.updateSavedObjectsFromSaveFields() savedObjects=%s" %(self.savedObjects));
                    # logger.debug("save_plan_step_date.updateSavedObjectsFromSaveFields() savedFields=%s" %(self.savedFields));

        # logger.debug("EXIT save_plan_step_date.updateSavedObjectsFromSaveFields() type(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST])=%s; self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]=%s" %(type(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]), self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]));

    def updateFromStep(self, updated_step):
        # logger.debug("ENTER save_plan_step_data.updateFromStep() updated_step.stepName=%s; self.savedFields=%s" %(updated_step.getStepName(), self.savedFields))

        if updated_step.getStepName() not in self._dependsOn:
            for sectionKey, sectionObj in self.step_sections.items():
                if sectionObj:
                    # logger.debug("save_plan_step_data.updateFromStep() sectionKey=%s" %(sectionKey))
                    for key in sectionObj.getCurrentSavedFieldDict().keys():
                        sectionObj.updateFromStep(updated_step)
            return

        if updated_step.getStepName() == StepNames.APPLICATION:

            if not updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:
                logger.debug("save_plan_step_data.updateFromStep() --- NO-OP --- APPLICATION APPL_PRODUCT IS NOT YET SET!!! ")
                return

            if updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE]:
                runTypeObj = updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE]

                # make sure hidden nucleotide type field is updated for new Application
                if self.prepopulatedFields.get(SavePlanFieldNames.RUN_TYPE) and self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] != runTypeObj.runType:
                    nucleotideType = runTypeObj.nucleotideType.upper()
                    if nucleotideType in ['DNA', 'RNA']:
                        for row in self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]:
                            row['nucleotideType'] = nucleotideType
                        self.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = json.dumps(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST])

                self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] = runTypeObj.runType
            else:
                self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] = ""

            # logger.debug("save_plan_step_data.updateFromStep() going to update RUNTYPE value=%s" %(self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]))

        if updated_step.getStepName() == StepNames.APPLICATION and updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:
            self.savedObjects[SavePlanFieldNames.APPL_PRODUCT] = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]

        if updated_step.getStepName() == StepNames.KITS:
            barcode_set = updated_step.savedFields[KitsFieldNames.BARCODE_ID]
            if str(barcode_set) != str(self.savedFields[SavePlanFieldNames.BARCODE_SET]):
                self.savedFields[SavePlanFieldNames.BARCODE_SET] = barcode_set
                if barcode_set:
                    barcodes = list(dnaBarcode.objects.filter(name=barcode_set).order_by('index'))
                    bc_count = min(len(barcodes), len(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]))
                    self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST] = self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST][:bc_count]
                    for i in range(bc_count):
                        self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST][i]['barcodeId'] = barcodes[i].id_str

                    self.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = json.dumps(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST])
                # if barcode kit selection changes, re-validate
                self.validateStep()

        if updated_step.getStepName() == StepNames.IONREPORTER:
            ir_account_id = updated_step.savedFields[IonReporterFieldNames.IR_ACCOUNT_ID]
            ir_workflow = updated_step.savedFields[IonReporterFieldNames.IR_WORKFLOW]
            ir_isfactory = updated_step.savedFields[IonReporterFieldNames.IR_ISFACTORY]

            if ir_account_id and ir_account_id != "0":
                if self.prepopulatedFields[SavePlanFieldNames.IR_WORKFLOW] != ir_workflow or self.prepopulatedFields[SavePlanFieldNames.IR_ISFACTORY] != ir_isfactory:

                    for row in self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]:
                        row[SavePlanFieldNames.IR_WORKFLOW] = ir_workflow
                        row[SavePlanFieldNames.IR_ISFACTORY] = ir_isfactory

                    self.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = json.dumps(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST])

            self.prepopulatedFields[SavePlanFieldNames.IR_WORKFLOW] = ir_workflow
            self.prepopulatedFields[SavePlanFieldNames.IR_ISFACTORY] = ir_isfactory
            self.prepopulatedFields[SavePlanFieldNames.SELECTED_IR] = ir_account_id

        for sectionKey, sectionObj in self.step_sections.items():
            if sectionObj:
                sectionObj.updateFromStep(updated_step)

        logger.debug("EXIT save_plan_step_data.updateFromStep() self.savedFields=%s" % (self.savedFields))

    def updateSavedFieldsForSamples_may_no_longer_needed(self):
        # logger.debug("ENTER save_plan_step_data.updateSavedFieldsForSamples() B4 type=%s; self.savedFields[samplesTable]=%s" %(type(self.savedFields[SavePlanFieldNames.SAMPLES_TABLE]), self.savedFields[SavePlanFieldNames.SAMPLES_TABLE]))
        # logger.debug("ENTER save_plan_step_data.updateSavedFieldsForSamples() B4 type=%s; self.savedObjects[samplesTableList]=%s" %(type(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]), self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]))

        # convert tuple to string
        planReference = str(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE])
        planHotSptRegionBedFile = str(self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE])
        planTargetRegionBedFile = str(self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE])

        # logger.debug("save_plan_step_data.updateSavedFieldsForSamples() type(planReference)=%s; planReference=%s" %(type(planReference), planReference))
        # logger.debug("save_plan_step_data.updateSavedFieldsForSamples() type(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE])=%s; self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]=%s" %(type(self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]), self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]))

        hasAnyChanges = False
        myTable = json.loads(self.savedFields[SavePlanFieldNames.SAMPLES_TABLE])
        # convert unicode to str
        myTable = convert(myTable)

        isCreate = self.sh_type in [StepHelperType.CREATE_NEW_PLAN, StepHelperType.CREATE_NEW_TEMPLATE]
        # logger.debug("save_plan_step_data.updateSavedFieldsForSamples() isCreate()=%s" %(isCreate))

        for index, row in enumerate(myTable):
            # logger.debug("save_plan_step_data.updateSavedFieldsForSamples() B4 CHANGES... BARCODE_SET LOOP row=%s" %(row))

            sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME, '').strip()
            if sample_name:

                sample_nucleotideType = row.get(SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE, "")

                sampleReference = row.get(SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE, "")
                sampleHotSpotRegionBedFile = row.get(SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE, "")
                sampleTargetRegionBedFile = row.get(SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE, "")

                runType = self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]

                if runType == "AMPS_DNA_RNA" and sample_nucleotideType == "RNA":
                    newSampleReference = sampleReference
                    newSampleHotspotRegionBedFile = sampleHotSpotRegionBedFile
                    newSampleTargetRegionBedFile = sampleTargetRegionBedFile

                else:
                    if not sampleReference and self.savedObjects[SavePlanFieldNames.APPL_PRODUCT] and not self.savedObjects[SavePlanFieldNames.APPL_PRODUCT].isReferenceBySampleSupported and isCreate:
                        logger.debug("save_plan_step_data.updateSavedFieldsForSamples() GOING to set sampleReference to planReference... planReference=%s" % (planReference))

                        sampleReference = planReference
                        sampleHotSpotRegionBedFile = planHotSptRegionBedFile
                        sampleTargetRegionBedFile = planTargetRegionBedFile
                    # else:
                    # logger.debug("save_plan_step_data.updateSavedFieldsForSamples() SKIP SETTING sampleReference to planReference... isCreate=%s" %(isCreate))

                    newSampleReference = planReference
                    newSampleHotspotRegionBedFile = planHotSptRegionBedFile
                    newSampleTargetRegionBedFile = planTargetRegionBedFile

                hasChanged = False
                if newSampleReference != sampleReference:
                    row[SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE] = newSampleReference
                    hasChanged = True
                if newSampleHotspotRegionBedFile != sampleHotSpotRegionBedFile:
                    row[SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE] = newSampleHotspotRegionBedFile
                    hasChanged = True
                if newSampleTargetRegionBedFile != sampleTargetRegionBedFile:
                    row[SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE] = newSampleTargetRegionBedFile
                    hasChanged = True

                if hasChanged:
                    myTable[index] = row
                    # logger.debug("save_plan_step_data.updateSavedFieldsForSamples() AFTER CHANGES  BARCODE_SET LOOP myTable[index]=%s" %(myTable[index]))
                    hasAnyChanges = True

        if hasAnyChanges:
            logger.debug("save_plan_step_data.updateSavedFieldsForSamples() hasAnyChanges AFTER CHANGES... type=%s; myTable=%s" % (type(myTable), myTable))

            # convert list with single quotes to str with double quotes. Then convert it to be unicode
            self.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = unicode(json.dumps(myTable))

            # logger.debug("save_plan_step_data.updateSavedFieldsForSamples() hasAnyChanges AFTER unicode(json.dumps)... type=%s; self.savedFields[samplesTable]=%s" %(type(self.savedFields[SavePlanFieldNames.SAMPLES_TABLE]), self.savedFields[SavePlanFieldNames.SAMPLES_TABLE]))

            self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST] = json.loads(self.savedFields[SavePlanFieldNames.SAMPLES_TABLE])
            logger.debug("save_plan_step_data.updateSavedFieldsForSamples() hasAnyChanges AFTER json.loads... type=%s; self.savedObjects[samplesTableList]=%s" % (type(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]), self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]))

    def getDefaultSection(self):
        """
        Sections are optional for a step.  Return the default section
        """
        if not self.step_sections:
            return None
        return self.step_sections.get(StepNames.REFERENCE, None)

    def getDefaultSectionSavedFieldDict(self):
        """
        Sections are optional for a step.  Return the savedFields dictionary of the default section if it exists.
        Otherwise, return an empty dictionary
        """
        default_value = {}
        if not self.step_sections:
            return default_value

        sectionObj = self.step_sections.get(StepNames.REFERENCE, None)
        if sectionObj:
            # logger.debug("save_plan_step_data.getDefaultSectionSavedFieldDict() sectionObj.savedFields=%s" %(sectionObj.savedFields))
            return sectionObj.savedFields
        else:
            return default_value

    def getDefaultSectionPrepopulatedFieldDict(self):
        """
        Sections are optional for a step.  Return the prepopuldatedFields dictionary of the default section if it exists.
        Otherwise, return an empty dictionary
        """
        default_value = {}
        if not self.step_sections:
            return default_value

        sectionObj = self.step_sections.get(StepNames.REFERENCE, None)
        if sectionObj:
            # logger.debug("save_plan_step_data.getDefaultSectionPrepopulatedFieldsFieldDict() sectionObj.prepopulatedFields=%s" %(sectionObj.prepopulatedFields))
            return sectionObj.prepopulatedFields
        else:
            return default_value
