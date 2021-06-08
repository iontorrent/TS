# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from django.utils.translation import ugettext as _
from django.core.urlresolvers import reverse
from iondb.rundb.plan.page_plan.kits_step_data import KitsFieldNames
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import (
    dnaBarcode,
    SampleAnnotation_CV,
    KitInfo,
    QCType,
    PlannedExperiment,
    Sample,
)
from iondb.rundb.labels import (
    Sample as _Sample,
    ModelsQcTypeToLabelsQcTypeAsDict,
    ModelsQcTypeToLabelsQcType,
)
from iondb.utils import validation

from iondb.rundb.plan.page_plan.step_helper_types import StepHelperType
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.application_step_data import ApplicationFieldNames
from iondb.rundb.plan.page_plan.reference_step_data import ReferenceFieldNames
from iondb.rundb.plan.page_plan.ionreporter_step_data import IonReporterFieldNames
from iondb.rundb.plan.plan_validator import (
    validate_plan_name,
    validate_notes,
    validate_sample_name,
    validate_sample_tube_label,
    validate_barcode_sample_association,
    validate_QC,
    validate_targetRegionBedFile_for_runType,
    validate_chipBarcode,
)
from iondb.utils.utils import convert
from collections import OrderedDict

# from iondb.utils import toBoolean

import json
import logging

logger = logging.getLogger(__name__)

MAX_LENGTH_SAMPLE_DESCRIPTION = 1024
MAX_LENGTH_SAMPLE_NAME = 127


class SavePlanFieldNames:
    UPLOADERS = "uploaders"
    SET_ID = "setid"
    SETID_SUFFIX = "setid_suffix"
    EXTERNAL_ID = "externalId"
    WORKFLOW = "Workflow"
    GENDER = "Gender"
    POPULATION = "Population"
    MOUSE_STRAINS = "mouseStrains"
    CANCER_TYPE = "cancerType"
    CELLULARITY_PCT = "cellularityPct"
    BIOPSY_DAYS = "biopsyDays"
    CELL_NUM = "cellNum"
    COUPLE_ID = "coupleID"
    EMBRYO_ID = "embryoID"
    BACTERIAL_MARKER_TYPE = "BacterialMarkerType"
    WITNESS = "Witness"
    NUCLEOTIDE_TYPE = "NucleotideType"
    RELATIONSHIP_TYPE = "Relation"
    RELATION_ROLE = "RelationRole"
    PLAN_NAME = "planName"
    SAMPLE = "sample"
    NOTE = "note"
    SAMPLE_COLLECTION_DATE = "SampleCollectionDate"
    SAMPLE_RECEIPT_DATE = "SampleReceiptDate"
    SELECTED_IR = "selectedIr"
    IR_CONFIG_JSON = "irConfigJson"
    BARCODE_SET = "barcodeSet"
    BARCODE_SAMPLE_TUBE_LABEL = "barcodeSampleTubeLabel"
    BARCODE_TO_SAMPLE = "barcodeToSample"
    BARCODE_SETS = "barcodeSets"
    BARCODE_SETS_BARCODES = "barcodeSets_barcodes"

    BARCODE_SETS_STATIC = "barcodeSets_static"
    END_BARCODE_SET = "endBarcodeSet"
    END_BARCODE_SETS = "endBarcodeSets"
    END_BARCODES_SUBSET = "endBarcodes_subset"
    END_BARCODE_SETS_BARCODES = "endBarcodeSets_barcodes"
    # note: barcodeKitName was named barcodeId in the plannedexperiment API for backward compatibility. It could be confusing
    # since UI uses barcodeId and endBarcodeId to represent selected barcode for a sample
    END_BARCODE_ID = "endBarcodeId"
    BARCODE_SAMPLE_BARCODE_ID_UI_KEY = "barcodeId"
    BARCODE_SAMPLE_END_BARCODE_ID_UI_KEY = "endBarcodeId"
    DUAL_BARCODES_DB_KEY = "dualBarcodes"

    SAMPLE_TO_BARCODE = "sampleToBarcode"
    SAMPLE_EXTERNAL_ID = "sampleExternalId"
    SAMPLE_NAME = "sampleName"
    SAMPLE_DESCRIPTION = "sampleDescription"
    TUBE_LABEL = "tubeLabel"
    CHIP_BARCODE_LABEL = "chipBarcodeLabel"
    CHIP_BARCODE = "chipBarcode"

    IR_SAMPLE_COLLECTION_DATE = "irSampleCollectionDate"
    IR_SAMPLE_RECEIPT_DATE = "irSampleReceiptDate"
    IR_PLUGIN_ENTRIES = "irPluginEntries"
    IR_GENDER = "irGender"
    IR_POPULATION = "irPopulation"
    IR_MOUSE_STRAINS = "irmouseStrains"
    IR_CANCER_TYPE = "ircancerType"
    IR_CELLULARITY_PCT = "ircellularityPct"
    IR_BIOPSY_DAYS = "irbiopsyDays"
    IR_CELL_NUM = "ircellNum"
    IR_COUPLE_ID = "ircoupleID"
    IR_EMBRYO_ID = "irembryoID"
    IR_BACTERIAL_MARKER_TYPE = "irBacterialMarkerType"
    IR_WITNESS = "irWitness"

    IR_WORKFLOW = "irWorkflow"
    IR_ISFACTORY = "tag_isFactoryProvidedWorkflow"
    IR_DOWN = "irDown"
    IR_RELATION_ROLE = "irRelationRole"
    IR_RELATIONSHIP_TYPE = "irRelationshipType"
    IR_APPLICATION_TYPE = "ApplicationType"
    IR_SET_ID = "irSetID"

    BAD_SAMPLE_NAME = "bad_sample_name"
    BAD_SAMPLE_EXTERNAL_ID = "bad_sample_external_id"
    BAD_SAMPLE_DESCRIPTION = "bad_sample_description"
    BAD_TUBE_LABEL = "bad_tube_label"
    BAD_CHIP_BARCODE = "bad_chip_barcode"

    BARCODE_SAMPLE_NUCLEOTIDE_TYPE = "nucleotideType"
    BARCODE_SAMPLE_REFERENCE = "reference"
    BARCODE_SAMPLE_TARGET_REGION_BED_FILE = "targetRegionBedFile"
    BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE = "hotSpotRegionBedFile"
    BARCODE_SAMPLE_SSE_BED_FILE = "sseBedFile"
    BARCODE_SAMPLE_CONTROL_SEQ_TYPE = "controlSequenceType"
    BARCODE_SAMPLE_CONTROL_TYPE = "controlType"
    BARCODE_SAMPLE_END_BARCODE_DB_KEY = "endBarcode"

    BARCODE_SAMPLE_INFO = "barcodeSampleInfo"
    NO_SAMPLES = "no_samples"
    DESCRIPTION = "description"
    BAD_IR_SET_ID = "badIrSetId"
    SAMPLE_ANNOTATIONS = "sampleAnnotations"

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

    NO_BARCODE = "no_barcode"
    BAD_BARCODES = "bad_barcodes"

    APPLICATION_TYPE = "applicationType"
    FIRE_VALIDATION = "fireValidation"

    LIMS_META = "LIMS_meta"
    META = "meta"
    APPLICATION_GROUP_NAME = "applicationGroupName"
    HAS_PGS_DATA = "hasPgsData"
    HAS_ONCO_DATA = "hasOncoData"


class MonitoringFieldNames:
    QC_TYPES = "qcTypes"


def update_ir_plugin_from_samples_table(samplesTable):
    # save IR fields for non-barcoded and barcoded plans
    userInputInfo = []
    for row in samplesTable:
        sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME, "").strip()
        if sample_name:
            ir_userinput_dict = {
                SavePlanFieldNames.SAMPLE: sample_name,
                SavePlanFieldNames.SAMPLE_NAME: sample_name.replace(" ", "_"),
                SavePlanFieldNames.SAMPLE_EXTERNAL_ID: row.get(
                    SavePlanFieldNames.SAMPLE_EXTERNAL_ID, ""
                ),
                SavePlanFieldNames.SAMPLE_DESCRIPTION: row.get(
                    SavePlanFieldNames.SAMPLE_DESCRIPTION, ""
                ),
                SavePlanFieldNames.SAMPLE_COLLECTION_DATE: row.get(
                    SavePlanFieldNames.IR_SAMPLE_COLLECTION_DATE, ""
                ),
                SavePlanFieldNames.SAMPLE_RECEIPT_DATE: row.get(
                    SavePlanFieldNames.IR_SAMPLE_RECEIPT_DATE, ""
                ),
                SavePlanFieldNames.WORKFLOW: row.get(
                    SavePlanFieldNames.IR_WORKFLOW, ""
                ),
                SavePlanFieldNames.IR_ISFACTORY: row.get(
                    SavePlanFieldNames.IR_ISFACTORY
                ),
                SavePlanFieldNames.RELATION_ROLE: row.get(
                    SavePlanFieldNames.IR_RELATION_ROLE, ""
                ),
                SavePlanFieldNames.RELATIONSHIP_TYPE: row.get(
                    SavePlanFieldNames.IR_RELATIONSHIP_TYPE, ""
                ),
                SavePlanFieldNames.SET_ID: row.get(SavePlanFieldNames.IR_SET_ID, ""),
                SavePlanFieldNames.GENDER: row.get(SavePlanFieldNames.IR_GENDER, ""),
                SavePlanFieldNames.POPULATION: row.get(
                    SavePlanFieldNames.IR_POPULATION, ""
                ),
                SavePlanFieldNames.MOUSE_STRAINS: row.get(
                    SavePlanFieldNames.IR_MOUSE_STRAINS, ""
                ),
                SavePlanFieldNames.NUCLEOTIDE_TYPE: row.get(
                    SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE, ""
                ),
                SavePlanFieldNames.CANCER_TYPE: row.get(
                    SavePlanFieldNames.IR_CANCER_TYPE, ""
                ),
                SavePlanFieldNames.CELLULARITY_PCT: row.get(
                    SavePlanFieldNames.IR_CELLULARITY_PCT, ""
                ),
                SavePlanFieldNames.BIOPSY_DAYS: row.get(
                    SavePlanFieldNames.IR_BIOPSY_DAYS, ""
                ),
                SavePlanFieldNames.CELL_NUM: row.get(
                    SavePlanFieldNames.IR_CELL_NUM, ""
                ),
                SavePlanFieldNames.COUPLE_ID: row.get(
                    SavePlanFieldNames.IR_COUPLE_ID, ""
                ),
                SavePlanFieldNames.EMBRYO_ID: row.get(
                    SavePlanFieldNames.IR_EMBRYO_ID, ""
                ),
                SavePlanFieldNames.BACTERIAL_MARKER_TYPE: row.get(SavePlanFieldNames.IR_BACTERIAL_MARKER_TYPE, ""),
                SavePlanFieldNames.WITNESS: row.get(SavePlanFieldNames.IR_WITNESS, ""),
                SavePlanFieldNames.IR_APPLICATION_TYPE: row.get(
                    SavePlanFieldNames.IR_APPLICATION_TYPE, ""
                ),
            }

            barcode_id = row.get(SavePlanFieldNames.BARCODE_SAMPLE_BARCODE_ID_UI_KEY)
            if barcode_id:
                ir_userinput_dict[KitsFieldNames.BARCODE_ID] = barcode_id
                endBarcode_id = row.get(
                    SavePlanFieldNames.BARCODE_SAMPLE_END_BARCODE_ID_UI_KEY
                )
                if endBarcode_id:
                    ir_userinput_dict[SavePlanFieldNames.END_BARCODE_ID] = endBarcode_id

            userInputInfo.append(ir_userinput_dict)

    return userInputInfo


class SavePlanStepData(AbstractStepData):
    def __init__(self, sh_type):
        super(SavePlanStepData, self).__init__(sh_type)
        self.resourcePath = "rundb/plan/page_plan/page_plan_save_plan.html"
        self.prev_step_url = reverse("page_plan_output")
        self.next_step_url = reverse("page_plan_save")

        self.savedFields = OrderedDict()

        self.savedFields[SavePlanFieldNames.PLAN_NAME] = None
        self.savedFields[SavePlanFieldNames.NOTE] = None
        self.savedFields[SavePlanFieldNames.APPLICATION_TYPE] = ""
        self.savedFields[SavePlanFieldNames.IR_DOWN] = "0"

        self.savedFields[SavePlanFieldNames.BARCODE_SET] = ""
        self.savedFields[SavePlanFieldNames.END_BARCODE_SET] = ""

        self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE] = ""
        self.prepopulatedFields[SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE] = ""
        self.prepopulatedFields[SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE] = ""

        self.prepopulatedFields[SavePlanFieldNames.SELECTED_IR] = None
        self.prepopulatedFields[SavePlanFieldNames.IR_WORKFLOW] = None
        self.prepopulatedFields[SavePlanFieldNames.IR_ISFACTORY] = False
        self.prepopulatedFields[SavePlanFieldNames.IR_CONFIG_JSON] = None
        self.prepopulatedFields[SavePlanFieldNames.SAMPLE_ANNOTATIONS] = list(
            SampleAnnotation_CV.objects.filter(isActive=True).order_by(
                "annotationType", "value"
            )
        )
        self.savedFields[SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL] = None
        self.savedFields[SavePlanFieldNames.CHIP_BARCODE_LABEL] = None
        self.savedObjects[SavePlanFieldNames.BARCODE_TO_SAMPLE] = OrderedDict()

        barcodeObjs_list = list(
            dnaBarcode.objects.filter(active=True)
            .values_list("name", flat=True)
            .distinct()
            .order_by("name")
        )
        self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS] = barcodeObjs_list
        all_barcodes = {}
        for bc in (
            dnaBarcode.objects.filter(active=True)
            .order_by("name", "index")
            .values("name", "id_str", "sequence")
        ):
            all_barcodes.setdefault(bc["name"], []).append(bc)
        self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_BARCODES] = json.dumps(
            all_barcodes
        )

        self.prepopulatedFields[SavePlanFieldNames.BARCODE_SETS_STATIC] = (
            dnaBarcode.objects.filter(active=True)
            .exclude(end_sequence="")
            .values_list("name", flat=True)
            .distinct()
        )
        self.prepopulatedFields[SavePlanFieldNames.END_BARCODE_SETS] = barcodeObjs_list
        self.prepopulatedFields[
            SavePlanFieldNames.END_BARCODE_SETS_BARCODES
        ] = json.dumps(all_barcodes)

        self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE] = OrderedDict()
        self.savedObjects[SavePlanFieldNames.IR_PLUGIN_ENTRIES] = []
        self.prepopulatedFields[SavePlanFieldNames.FIRE_VALIDATION] = "1"

        self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST] = [
            {"row": "1", "sampleName": u"Sample 1"}
        ]
        self.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = json.dumps(
            self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]
        )
        self.prepopulatedFields[SavePlanFieldNames.NUM_SAMPLES] = 1

        self.savedFields[SavePlanFieldNames.ONCO_SAME_SAMPLE] = False

        # logger.debug("save_plan_step_data samplesTable=%s" %(self.savedFields[SavePlanFieldNames.SAMPLES_TABLE]))

        self.prepopulatedFields[
            SavePlanFieldNames.CONTROL_SEQ_TYPES
        ] = KitInfo.objects.filter(
            kitType="ControlSequenceKitType", isActive=True
        ).order_by(
            "name"
        )

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
        self.ModelsQcTypeToLabelsQcTypeAsDict = ModelsQcTypeToLabelsQcTypeAsDict
        all_qc_types = list(QCType.objects.all().order_by("qcName"))
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
            errors = validate_plan_name(
                new_field_value,
                field_label=_("workflow.step.saveplan.fields.planName.label"),
            )  # 'Plan Name'
            if errors:
                self.validationErrors[field_name] = "\n".join(errors)
        elif field_name == SavePlanFieldNames.NOTE:
            errors = validate_notes(
                new_field_value,
                field_label=_("workflow.step.saveplan.fields.note.label"),
            )
            if errors:
                self.validationErrors[field_name] = "\n".join(errors)
        elif field_name == SavePlanFieldNames.BARCODE_SAMPLE_TUBE_LABEL:
            errors = validate_sample_tube_label(
                new_field_value,
                field_label=_(
                    "workflow.step.sample.fields.barcodeSampleTubeLabel.label"
                ),
            )
            if errors:
                self.validationErrors[field_name] = "\n".join(errors)
        elif field_name == SavePlanFieldNames.CHIP_BARCODE_LABEL:
            errors = validate_chipBarcode(
                new_field_value,
                field_label=_("workflow.step.saveplan.fields.chipBarcodeLabel.label"),
            )
            if errors:
                self.validationErrors[field_name] = "\n".join(errors)

        elif field_name in self.qcNames:
            """
            All qc thresholds must be positive integers
            """
            errors = validate_QC(
                new_field_value, ModelsQcTypeToLabelsQcType(field_name)
            )
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
                isTargetRegionSelectionRequired = (
                    applProduct.isTargetRegionBEDFileSelectionRequiredForRefSelection
                )
            else:
                isTargetRegionSelectionRequired = False

            applicationGroupName = self.prepopulatedFields[
                SavePlanFieldNames.APPLICATION_GROUP_NAME
            ]
            for row in sample_table_list:

                sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME, "").strip()
                if sample_name:
                    sample_nucleotideType = row.get(
                        SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE, ""
                    )

                    sampleReference = row.get(
                        SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE, ""
                    )
                    sampleTargetRegionBedFile = row.get(
                        SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE, ""
                    )

                    runType = self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]

                    errors = []
                    # if the plan has been sequenced, do not enforce the target bed file to be selected
                    isMainBEDFileValidated = (
                        "default_targetBedFile" in self.validationErrors
                    )
                    if (
                        not isMainBEDFileValidated
                        and planStatus != "run"
                        and (self.sh_type not in StepHelperType.TEMPLATE_TYPES)
                    ):
                        sampleTargetRegionBedFile_label = _(
                            "workflow.step.saveplan.bysample.fields.targetRegionBedFile.label"
                        ) % {"sampleName": sample_name}
                        errors = validate_targetRegionBedFile_for_runType(
                            sampleTargetRegionBedFile,
                            field_label=sampleTargetRegionBedFile_label,
                            runType=runType,
                            reference=sampleReference,
                            nucleotideType=sample_nucleotideType,
                            applicationGroupName=applicationGroupName,
                        )  # "Target Regions BED File for %(sampleName)s"

                    if errors:
                        samples_errors.append("\n".join(errors))

                if samples_errors:
                    self.validationErrors[field_name] = "\n".join(samples_errors)

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

        endBarcodeSet = self.savedFields[SavePlanFieldNames.END_BARCODE_SET]
        selectedEndBarcodes = []

        samplesTable = json.loads(self.savedFields[SavePlanFieldNames.SAMPLES_TABLE])
        # logger.debug("save_plan_step_data - anySamples? samplesTable=%s" %(samplesTable))

        for row in samplesTable:
            sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME, "").strip()

            # logger.debug("save_plan_step_data - anySamples? sampleName=%s" %(sample_name))

            if sample_name:
                any_samples = True
                if validate_sample_name(
                    sample_name, field_label=_Sample.displayedName.verbose_name
                ):
                    self.validationErrors[SavePlanFieldNames.BAD_SAMPLE_NAME].append(
                        sample_name
                    )

                external_id = row.get(SavePlanFieldNames.SAMPLE_EXTERNAL_ID, "")
                if external_id:
                    self.validate_field(
                        external_id,
                        self.validationErrors[
                            SavePlanFieldNames.BAD_SAMPLE_EXTERNAL_ID
                        ],
                    )

                description = row.get(SavePlanFieldNames.SAMPLE_DESCRIPTION, "")
                if description:
                    self.validate_field(
                        description,
                        self.validationErrors[
                            SavePlanFieldNames.BAD_SAMPLE_DESCRIPTION
                        ],
                        False,
                        MAX_LENGTH_SAMPLE_DESCRIPTION,
                    )

                ir_set_id = row.get("irSetId", "")
                if ir_set_id and not (str(ir_set_id).isdigit()):
                    self.validationErrors[SavePlanFieldNames.BAD_IR_SET_ID].append(
                        ir_set_id
                    )

                tube_label = row.get("tubeLabel", "")
                if validate_sample_tube_label(
                    tube_label,
                    field_label=_(
                        "workflow.step.saveplan.fields.barcodeSampleTubeLabel.label"
                    ),
                ):
                    self.validationErrors[SavePlanFieldNames.BAD_TUBE_LABEL].append(
                        tube_label
                    )
                chip_barcode = row.get("chipBarcode", "")
                if validate_chipBarcode(
                    chip_barcode,
                    field_label=_(
                        "workflow.step.saveplan.fields.chipBarcodeLabel.label"
                    ),
                ):
                    self.validationErrors[SavePlanFieldNames.BAD_CHIP_BARCODE].append(
                        chip_barcode
                    )
                if barcodeSet:
                    selectedBarcodes.append(
                        row.get(SavePlanFieldNames.BARCODE_SAMPLE_BARCODE_ID_UI_KEY)
                    )
                    if endBarcodeSet:
                        endBarcode_id_value = row.get(
                            SavePlanFieldNames.BARCODE_SAMPLE_END_BARCODE_ID_UI_KEY
                        )
                        if endBarcode_id_value:
                            selectedEndBarcodes.append(endBarcode_id_value)

        if any_samples:
            self.validationErrors.pop(SavePlanFieldNames.NO_SAMPLES, None)
        else:
            self.validationErrors[
                SavePlanFieldNames.NO_SAMPLES
            ] = validation.invalid_required_at_least_one(_Sample.verbose_name)

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

        # 20170928-TODO-WIP
        if barcodeSet:
            errors = validate_barcode_sample_association(selectedBarcodes, barcodeSet)

            myErrors = convert(errors)
            if myErrors.get("MISSING_BARCODE", ""):
                self.validationErrors[SavePlanFieldNames.NO_BARCODE] = myErrors.get(
                    "MISSING_BARCODE", ""
                )
            if myErrors.get("DUPLICATE_BARCODE", ""):
                self.validationErrors[SavePlanFieldNames.BAD_BARCODES] = myErrors.get(
                    "DUPLICATE_BARCODE", ""
                )

        if selectedEndBarcodes:
            applProduct = self.savedObjects[SavePlanFieldNames.APPL_PRODUCT]
            if applProduct and applProduct.dualBarcodingRule == "no_reuse":
                errors = validate_barcode_sample_association(
                    selectedEndBarcodes, endBarcodeSet, isEndBarcodeExists=True
                )

                myErrors = convert(errors)
                if myErrors.get("DUPLICATE_BARCODE", ""):
                    self.validationErrors[
                        SavePlanFieldNames.BAD_BARCODES
                    ] = myErrors.get("DUPLICATE_BARCODE", "")

    def validate_field(
        self,
        value,
        bad_samples,
        validate_leading_chars=True,
        max_length=MAX_LENGTH_SAMPLE_NAME,
    ):
        exists = False
        if value:
            exists = True
            if not validation.is_valid_chars(value):
                bad_samples.append(value)

            if (
                validate_leading_chars
                and value not in bad_samples
                and not validation.is_valid_leading_chars(value)
            ):
                bad_samples.append(value)

            if value not in bad_samples and not validation.is_valid_length(
                value, max_length
            ):
                bad_samples.append(value)

        return exists

    def updateSavedObjectsFromSavedFields(self):
        self.prepopulatedFields["fireValidation"] = "0"

        for section, sectionObj in list(self.step_sections.items()):
            sectionObj.updateSavedObjectsFromSavedFields()

            if section == StepNames.REFERENCE:
                self.prepopulatedFields[
                    SavePlanFieldNames.PLAN_REFERENCE
                ] = sectionObj.savedFields.get(ReferenceFieldNames.REFERENCE, "")
                self.prepopulatedFields[
                    SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE
                ] = sectionObj.savedFields.get(ReferenceFieldNames.TARGET_BED_FILE, "")
                self.prepopulatedFields[
                    SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE
                ] = sectionObj.savedFields.get(
                    ReferenceFieldNames.HOT_SPOT_BED_FILE, ""
                )
                # logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() REFERENCE reference.savedFields=%s" %(sectionObj.savedFields))

        self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST] = json.loads(
            self.savedFields[SavePlanFieldNames.SAMPLES_TABLE]
        )
        self.prepopulatedFields[SavePlanFieldNames.NUM_SAMPLES] = len(
            self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]
        )

        self.savedObjects[
            SavePlanFieldNames.IR_PLUGIN_ENTRIES
        ] = update_ir_plugin_from_samples_table(
            self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]
        )

        if self.savedFields[SavePlanFieldNames.BARCODE_SET]:
            planReference = self.prepopulatedFields[SavePlanFieldNames.PLAN_REFERENCE]
            planHotSptRegionBedFile = self.prepopulatedFields[
                SavePlanFieldNames.PLAN_HOTSPOT_REGION_BED_FILE
            ]
            planTargetRegionBedFile = self.prepopulatedFields[
                SavePlanFieldNames.PLAN_TARGET_REGION_BED_FILE
            ]

            logger.debug(
                "save_plan_step_data.updateSavedObjectsFromSavedFields() BARCODE_SET PLAN_REFERENCE=%s; TARGET_REGION=%s; HOTSPOT_REGION=%s;"
                % (planReference, planTargetRegionBedFile, planHotSptRegionBedFile)
            )

            reference_step_helper = self.savedObjects[
                SavePlanFieldNames.REFERENCE_STEP_HELPER
            ]

            endBarcodeKit = self.savedFields[SavePlanFieldNames.END_BARCODE_SET]

            self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE] = {}
            for row in self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]:

                logger.debug(
                    "save_plan_step_data.updateSavedObjectsFromSavedFields() BARCODE_SET LOOP row=%s"
                    % (row)
                )

                sample_name = row.get(SavePlanFieldNames.SAMPLE_NAME, "").strip()
                if sample_name:
                    id_str = row[SavePlanFieldNames.BARCODE_SAMPLE_BARCODE_ID_UI_KEY]
                    end_id_str = row[
                        SavePlanFieldNames.BARCODE_SAMPLE_END_BARCODE_ID_UI_KEY
                    ]

                    # update barcodedSamples dict
                    if (
                        sample_name
                        not in self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE]
                    ):
                        self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][
                            sample_name
                        ] = {
                            KitsFieldNames.BARCODES: [],
                            SavePlanFieldNames.BARCODE_SAMPLE_INFO: {},
                            SavePlanFieldNames.DUAL_BARCODES_DB_KEY: [],
                        }

                    sample_nucleotideType = row.get(
                        SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE, ""
                    )

                    sampleReference = row.get(
                        SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE, ""
                    )
                    sampleHotSpotRegionBedFile = row.get(
                        SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE, ""
                    )
                    sampleTargetRegionBedFile = row.get(
                        SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE, ""
                    )

                    runType = self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]

                    # logger.debug("save_plan_step_data.updateSavedObjectsFromSavedFields() SETTING reference step helper runType=%s; sample_nucleotideType=%s; sampleReference=%s" %(runType, sample_nucleotideType, sampleReference))

                    sseBedFile = ""
                    if reference_step_helper:
                        sseBedFile = reference_step_helper.get_sseBedFile(
                            sampleTargetRegionBedFile
                        )

                    self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][
                        sample_name
                    ][KitsFieldNames.BARCODES].append(id_str)
                    self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][
                        sample_name
                    ][SavePlanFieldNames.BARCODE_SAMPLE_INFO][id_str] = {
                        SavePlanFieldNames.EXTERNAL_ID: row.get(
                            SavePlanFieldNames.SAMPLE_EXTERNAL_ID, ""
                        ),
                        SavePlanFieldNames.DESCRIPTION: row.get(
                            SavePlanFieldNames.SAMPLE_DESCRIPTION, ""
                        ),
                        SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE: sample_nucleotideType,
                        SavePlanFieldNames.BARCODE_SAMPLE_REFERENCE: sampleReference,
                        SavePlanFieldNames.BARCODE_SAMPLE_TARGET_REGION_BED_FILE: sampleTargetRegionBedFile,
                        SavePlanFieldNames.BARCODE_SAMPLE_HOTSPOT_REGION_BED_FILE: sampleHotSpotRegionBedFile,
                        SavePlanFieldNames.BARCODE_SAMPLE_SSE_BED_FILE: sseBedFile,
                        SavePlanFieldNames.BARCODE_SAMPLE_END_BARCODE_DB_KEY: row.get(
                            SavePlanFieldNames.BARCODE_SAMPLE_END_BARCODE_ID_UI_KEY, ""
                        ),
                        SavePlanFieldNames.BARCODE_SAMPLE_CONTROL_SEQ_TYPE: row.get(
                            SavePlanFieldNames.BARCODE_SAMPLE_CONTROL_SEQ_TYPE, ""
                        ),
                        SavePlanFieldNames.BARCODE_SAMPLE_CONTROL_TYPE: row.get(
                            SavePlanFieldNames.BARCODE_SAMPLE_CONTROL_TYPE, ""
                        ),
                    }

                    if endBarcodeKit and id_str:
                        if end_id_str:
                            self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][
                                sample_name
                            ][SavePlanFieldNames.DUAL_BARCODES_DB_KEY].append(
                                id_str
                                + PlannedExperiment.get_dualBarcodes_delimiter()
                                + end_id_str
                            )
                        else:
                            self.savedObjects[SavePlanFieldNames.SAMPLE_TO_BARCODE][
                                sample_name
                            ][SavePlanFieldNames.DUAL_BARCODES_DB_KEY].append(id_str)

    def alternateUpdateFromStep(self, updated_step):
        """
        also runs if editing a plan post-sequencing and Application is updated
        """
        if updated_step.getStepName() == StepNames.APPLICATION:

            if updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE]:
                runTypeObj = updated_step.savedObjects[ApplicationFieldNames.RUN_TYPE]

                # make sure hidden nucleotide type field is updated for new Application
                if (
                    self.prepopulatedFields.get(SavePlanFieldNames.RUN_TYPE)
                    and self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE]
                    != runTypeObj.runType
                ):
                    nucleotideType = runTypeObj.nucleotideType.upper()
                    if nucleotideType in ["DNA", "RNA"]:
                        for row in self.savedObjects[
                            SavePlanFieldNames.SAMPLES_TABLE_LIST
                        ]:
                            row[
                                SavePlanFieldNames.BARCODE_SAMPLE_NUCLEOTIDE_TYPE
                            ] = nucleotideType
                        self.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = json.dumps(
                            self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]
                        )

                self.prepopulatedFields[
                    SavePlanFieldNames.RUN_TYPE
                ] = runTypeObj.runType
            else:
                self.prepopulatedFields[SavePlanFieldNames.RUN_TYPE] = ""

    def updateFromStep(self, updated_step):

        if updated_step.getStepName() not in self._dependsOn:
            for sectionKey, sectionObj in list(self.step_sections.items()):
                if sectionObj:
                    # logger.debug("save_plan_step_data.updateFromStep() sectionKey=%s" %(sectionKey))
                    for key in list(sectionObj.getCurrentSavedFieldDict().keys()):
                        sectionObj.updateFromStep(updated_step)
            return

        self.alternateUpdateFromStep(updated_step)

        if updated_step.getStepName() == StepNames.APPLICATION:

            if not updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]:
                logger.debug(
                    "save_plan_step_data.updateFromStep() --- NO-OP --- APPLICATION APPL_PRODUCT IS NOT YET SET!!! "
                )
                return

        if (
            updated_step.getStepName() == StepNames.APPLICATION
            and updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]
        ):
            self.savedObjects[
                SavePlanFieldNames.APPL_PRODUCT
            ] = updated_step.savedObjects[ApplicationFieldNames.APPL_PRODUCT]

        if updated_step.getStepName() == StepNames.KITS:
            barcode_set = updated_step.savedFields[KitsFieldNames.BARCODE_ID]
            if str(barcode_set) != str(
                self.savedFields[SavePlanFieldNames.BARCODE_SET]
            ):
                self.savedFields[SavePlanFieldNames.BARCODE_SET] = barcode_set
                if barcode_set:
                    barcodes = list(
                        dnaBarcode.objects.filter(name=barcode_set).order_by("index")
                    )
                    bc_count = min(
                        len(barcodes),
                        len(self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]),
                    )
                    self.savedObjects[
                        SavePlanFieldNames.SAMPLES_TABLE_LIST
                    ] = self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST][
                        :bc_count
                    ]
                    for i in range(bc_count):
                        self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST][i][
                            SavePlanFieldNames.BARCODE_SAMPLE_BARCODE_ID_UI_KEY
                        ] = barcodes[i].id_str

                    self.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = json.dumps(
                        self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]
                    )

                # if barcode kit selection changes, re-validate
                self.validateStep()

        if updated_step.getStepName() == StepNames.IONREPORTER:
            ir_account_id = updated_step.savedFields[
                IonReporterFieldNames.IR_ACCOUNT_ID
            ]
            ir_workflow = updated_step.savedFields[IonReporterFieldNames.IR_WORKFLOW]
            ir_isfactory = updated_step.savedFields[IonReporterFieldNames.IR_ISFACTORY]

            if ir_account_id and ir_account_id != "0":
                if (
                    self.prepopulatedFields[SavePlanFieldNames.IR_WORKFLOW]
                    != ir_workflow
                    or self.prepopulatedFields[SavePlanFieldNames.IR_ISFACTORY]
                    != ir_isfactory
                ):

                    for row in self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]:
                        row[SavePlanFieldNames.IR_WORKFLOW] = ir_workflow
                        row[SavePlanFieldNames.IR_ISFACTORY] = ir_isfactory

                    self.savedFields[SavePlanFieldNames.SAMPLES_TABLE] = json.dumps(
                        self.savedObjects[SavePlanFieldNames.SAMPLES_TABLE_LIST]
                    )

            self.prepopulatedFields[SavePlanFieldNames.IR_WORKFLOW] = ir_workflow
            self.prepopulatedFields[SavePlanFieldNames.IR_ISFACTORY] = ir_isfactory
            self.prepopulatedFields[SavePlanFieldNames.SELECTED_IR] = ir_account_id

        for sectionKey, sectionObj in list(self.step_sections.items()):
            if sectionObj:
                sectionObj.updateFromStep(updated_step)

        logger.debug(
            "EXIT save_plan_step_data.updateFromStep() self.savedFields=%s"
            % (self.savedFields)
        )

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
