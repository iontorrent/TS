# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.contrib.auth.models import User

from iondb.rundb.models import PlannedExperiment, Experiment, RunType, ApplProduct, \
    ReferenceGenome, Content, KitInfo, dnaBarcode, \
    LibraryKey, ThreePrimeadapter, Chip, QCType, Project, Plugin, \
    PlannedExperimentQC

from iondb.rundb.plan.views_helper import dict_bed_hotspot, get_IR_accounts_by_userName, get_default_or_first_IR_account_by_userName, get_internal_name_for_displayed_name, \
    get_ir_set_id, is_operation_supported_by_obj
from iondb.rundb.plan.plan_validator import validate_plan_name, validate_notes, validate_sample_name, validate_flows, \
    validate_QC, validate_projects, validate_sample_tube_label, validate_sample_id, validate_barcoded_sample_info, \
    validate_libraryReadLength, validate_templatingSize, validate_targetRegionBedFile_for_runType, validate_chipBarcode, \
    validate_reference, validate_sampleControlType
from iondb.rundb.plan.plan_csv_iru_validator import validate_iruConfig_process_userInputInfo, call_iru_validation_api
from traceback import format_exc

from plan_csv_writer import PlanCSVcolumns
import iondb.rundb.plan.views
import copy
import json
import os

import logging
logger = logging.getLogger(__name__)

import simplejson

KEY_SAMPLE_ID = "ID:"
KEY_SAMPLE_TYPE = "TYPE:"
KEY_SAMPLE_RNA_REF = "RNA REF:"
KEY_SAMPLE_REF = "REF:"
KEY_SAMPLE_TARGET = "TARGET:"
KEY_SAMPLE_RNA_TARGET = "RNA TARGET:"
KEY_SAMPLE_HOTSPOT = "HOTSPOT:"
KEY_SAMPLE_CONTROLTYPE = "CONTROL TYPE:"


class MyPlan:

    def __init__(self, selectedTemplate, selectedExperiment, selectedEAS, userName):
        if not selectedTemplate:
            self.planObj = None
            self.expObj = None
            self.easObj = None
        else:
            self.planObj = copy.copy(selectedTemplate)
            self.planObj.pk = None
            self.planObj.planGUID = None
            self.planObj.planShortID = None
            self.planObj.isReusable = False
            self.planObj.isSystem = False
            self.planObj.isSystemDefault = False
            self.planObj.expName = ""
            self.planObj.planName = ""
            self.planObj.planExecuted = False
            self.planObj.categories = selectedTemplate.categories
            self.planObj.origin = "csv"
            self.planObj.latestEAS = None

            metaData = selectedTemplate.metaData if selectedTemplate.metaData else {}
            metaData["fromTemplate"] = selectedTemplate.planName
            metaData["fromTemplateSource"] = "ION" if selectedTemplate.isSystem else selectedTemplate.username
            self.planObj.metaData = metaData

            if userName:
                self.planObj.username = userName

            self.expObj = copy.copy(selectedExperiment)
            self.expObj.pk = None
            self.expObj.unique = None
            self.expObj.plan = None

            # copy EAS
            self.easObj = copy.copy(selectedEAS)
            self.easObj.pk = None
            self.easObj.experiment = None
            self.easObj.isEditable = True
            self.easObj.reference = ""
            self.easObj.targetRegionBedFile = ""
            self.easObj.hotSpotRegionBedFile = ""

        self.sampleList = []
        self.sampleIdList = []
        self.nucleotideTypeList = []
        self.USERINPUT = {}

    def get_planObj(self):
        return self.planObj

    def get_expObj(self):
        return self.expObj

    def get_easObj(self):
        return self.easObj

    def get_sampleList(self):
        return self.sampleList

    def get_sampleIdList(self):
        return self.sampleIdList

    def get_nucleotideTypeList(self):
        return self.nucleotideTypeList

    def get_USERINPUT(self):
        return self.USERINPUT

def validate_csv_plan(csvPlanDict, username, single_file=True, samples_contents=None, httpHost=None):
    """ validate csv contents and convert user input to raw data to prepare for plan persistence
    returns: a collection of error messages if errors found, a dictionary of raw data values
    If single_file=False barcoded samples info is in separate csv files, must include samples_contents for each plan
    """

    logger.debug("ENTER plan_csv_validator.validate_csv_plan() csvPlanDict=%s; " % (csvPlanDict))

    failed = []
    rawPlanDict = {}
    planObj = None

    planDict = {}

    isToSkipRow = False

    selectedTemplate = None
    selectedExperiment = None
    selectedEAS = None

    # skip this row if no values found (will not prohibit the rest of the files from upload
    skipIfEmptyList = [PlanCSVcolumns.COLUMN_TEMPLATE_NAME]

    for skipIfEmpty in skipIfEmptyList:
        if skipIfEmpty in csvPlanDict:
            if not csvPlanDict[skipIfEmpty]:
                # required column is empty
                isToSkipRow = True

    if isToSkipRow:
        return failed, planDict, rawPlanDict, isToSkipRow

    # check if mandatory fields are present
    requiredList = [PlanCSVcolumns.COLUMN_TEMPLATE_NAME, PlanCSVcolumns.COLUMN_PLAN_NAME]
    if not single_file:
        requiredList.append(PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER)

    for required in requiredList:
        if required in csvPlanDict:
            if not csvPlanDict[required]:
                failed.append((required, "Required column is empty"))
        else:
            failed.append((required, "Required column is missing"))

    templateName = csvPlanDict.get(PlanCSVcolumns.COLUMN_TEMPLATE_NAME)

    if templateName:
        selectedTemplate, errorMsg = _get_template(templateName)

        if selectedTemplate:
            isSupported = is_operation_supported_by_obj(selectedTemplate)

            logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; isSupported=%s" % (selectedTemplate.id, selectedTemplate.planDisplayedName, isSupported))

            if (not isSupported):
                errorMsg = "Template name: " + templateName + " is not supported to create plans from"
                failed.append((PlanCSVcolumns.COLUMN_TEMPLATE_NAME, errorMsg))
                return failed, planDict, rawPlanDict, isToSkipRow

            selectedExperiment = selectedTemplate.experiment

            selectedEAS = selectedTemplate.latestEAS
            if not selectedEAS:
                logger.debug("plan_csv_validator.validate_csv_plan() NO latestEAS FOUND for selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s" % (selectedTemplate.id, selectedTemplate.planDisplayedName))
                selectedEAS = selectedTemplate.experiment.get_EAS()

            # logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; EAS.pk=%d" %(selectedTemplate.id, selectedTemplate.planDisplayedName, selectedEAS.id))

        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_TEMPLATE_NAME, errorMsg))
            return failed, planDict, rawPlanDict, isToSkipRow

        planObj = MyPlan(selectedTemplate, selectedExperiment, selectedEAS, username)

    else:
        return failed, planDict, rawPlanDict, isToSkipRow

    if PlanCSVcolumns.COLUMN_CHIP_BARCODE in csvPlanDict:
        errorMsg = _validate_chip_barcode(csvPlanDict.get(PlanCSVcolumns.COLUMN_CHIP_BARCODE), selectedTemplate, planObj)
        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_CHIP_BARCODE, errorMsg))
    elif PlanCSVcolumns.COLUMN_CHIP_BARCODE_V1 in csvPlanDict:
        errorMsg = _validate_chip_barcode(csvPlanDict.get(PlanCSVcolumns.COLUMN_CHIP_BARCODE_V1), selectedTemplate, planObj)
        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_CHIP_BARCODE_V1, errorMsg))

    logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; GOING TO validate Kit/chip combinations..." % (selectedTemplate.id, selectedTemplate.planDisplayedName))

    # check if deprecated header exists for chiptype
    chipType_header = PlanCSVcolumns.COLUMN_CHIP_TYPE if PlanCSVcolumns.COLUMN_CHIP_TYPE in csvPlanDict else PlanCSVcolumns.COLUMN_CHIP_TYPE_V1

    errorMsg = _validate_chip_type(csvPlanDict.get(chipType_header, None), selectedTemplate, planObj, selectedExperiment)

    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_CHIP_TYPE, errorMsg))
    else:
        errorMsg = _validate_sample_prep_kit(csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE_PREP_KIT), selectedTemplate, planObj)
        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_SAMPLE_PREP_KIT, errorMsg))

        errorMsg = _validate_lib_kit(csvPlanDict.get(PlanCSVcolumns.COLUMN_LIBRARY_KIT), selectedTemplate, planObj)
        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_LIBRARY_KIT, errorMsg))

        # check if deprecated header exists for templatingKitName
        templatingKit_header = PlanCSVcolumns.COLUMN_TEMPLATING_KIT if PlanCSVcolumns.COLUMN_TEMPLATING_KIT in csvPlanDict else PlanCSVcolumns.COLUMN_TEMPLATING_KIT_V1

        errorMsg = _validate_template_kit(csvPlanDict.get(templatingKit_header, None), selectedTemplate, planObj)
        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_TEMPLATING_KIT, errorMsg))

        errorMsg = _validate_control_seq_kit(csvPlanDict.get(PlanCSVcolumns.COLUMN_CONTROL_SEQ_KIT), selectedTemplate, planObj)
        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_CONTROL_SEQ_KIT, errorMsg))

        logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; GOING TO validate seq_kit..." % (selectedTemplate.id, selectedTemplate.planDisplayedName))

        errorMsg = _validate_seq_kit(csvPlanDict.get(PlanCSVcolumns.COLUMN_SEQ_KIT), selectedTemplate, planObj, selectedExperiment)
        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_SEQ_KIT, errorMsg))

    errorMsg = _validate_templatingSize(csvPlanDict.get(PlanCSVcolumns.COLUMN_TEMPLATING_SIZE), selectedTemplate, planObj)
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_TEMPLATING_SIZE, errorMsg))

    errorMsg = _validate_libraryReadLength(csvPlanDict.get(PlanCSVcolumns.COLUMN_LIBRARY_READ_LENGTH), selectedTemplate, planObj)
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_LIBRARY_READ_LENGTH, errorMsg))

    errorMsg = _validate_flows(csvPlanDict.get(PlanCSVcolumns.COLUMN_FLOW_COUNT), selectedTemplate, planObj)
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_FLOW_COUNT, errorMsg))

    errorMsg = _validate_sample_tube_label(csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE_TUBE_LABEL), selectedTemplate, planObj)

    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_SAMPLE_TUBE_LABEL, errorMsg))

    logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; GOING TO validate qc thresholds..." % (selectedTemplate.id, selectedTemplate.planDisplayedName))

    errorMsg, beadLoadQCValue = _validate_qc_pct(csvPlanDict.get(PlanCSVcolumns.COLUMN_BEAD_LOAD_PCT), selectedTemplate, planObj, "Bead loading")
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_BEAD_LOAD_PCT, errorMsg))

    rawPlanDict["Bead Loading (%)"] = beadLoadQCValue

    errorMsg, keySignalQCValue = _validate_qc_pct(csvPlanDict.get(PlanCSVcolumns.COLUMN_KEY_SIGNAL_PCT), selectedTemplate, planObj, "Key signal")
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_KEY_SIGNAL_PCT, errorMsg))

    rawPlanDict["Key Signal (1-100)"] = keySignalQCValue

    errorMsg, usableSeqQCValue = _validate_qc_pct(csvPlanDict.get(PlanCSVcolumns.COLUMN_USABLE_SEQ_PCT), selectedTemplate, planObj, "Usable sequence")
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_USABLE_SEQ_PCT, errorMsg))

    rawPlanDict["Usable Sequence (%)"] = usableSeqQCValue

    logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; GOING TO validate reference..." % (selectedTemplate.id, selectedTemplate.planDisplayedName))

    errorMsg = _validate_ref(csvPlanDict.get(PlanCSVcolumns.COLUMN_REF), selectedTemplate, planObj)
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_REF, errorMsg))

    errorMsg = _validate_target_bed(csvPlanDict.get(PlanCSVcolumns.COLUMN_TARGET_BED), selectedTemplate, planObj, csvPlanDict.get(PlanCSVcolumns.COLUMN_REF))
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_TARGET_BED, errorMsg))
    else:
        # sse file is tied to target region bed, we will blank it if targetRegionBedFile changed
        if planObj.get_easObj().sseBedFile and planObj.get_easObj().targetRegionBedFile != selectedEAS.targetRegionBedFile:
            planObj.get_easObj().sseBedFile = ""

    errorMsg = _validate_hotspot_bed(csvPlanDict.get(PlanCSVcolumns.COLUMN_HOTSPOT_BED), selectedTemplate, planObj, csvPlanDict.get(PlanCSVcolumns.COLUMN_REF))
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_HOTSPOT_BED, errorMsg))

    errorMsg, plugins = _validate_plugins(csvPlanDict.get(PlanCSVcolumns.COLUMN_PLUGINS), selectedTemplate, selectedEAS, planObj)
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_PLUGINS, errorMsg))

    logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; GOING TO validate projects..." % (selectedTemplate.id, selectedTemplate.planDisplayedName))

    errorMsg, projects = _validate_projects(csvPlanDict.get(PlanCSVcolumns.COLUMN_PROJECTS), selectedTemplate, planObj)
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_PROJECTS, errorMsg))

    rawPlanDict["newProjects"] = projects

    if 'Plan name (required)' not in dict(failed):
        errorMsg = _validate_plan_name(csvPlanDict.get(PlanCSVcolumns.COLUMN_PLAN_NAME), selectedTemplate, planObj)
        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_PLAN_NAME, errorMsg))

    errorMsg = _validate_notes(csvPlanDict.get(PlanCSVcolumns.COLUMN_NOTES), selectedTemplate, planObj)
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_NOTES, errorMsg))

    errorMsg = _validate_LIMS_data(csvPlanDict.get(PlanCSVcolumns.COLUMN_LIMS_DATA), selectedTemplate, planObj)
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_LIMS_DATA, errorMsg))

    barcodedSampleJson = None
    sampleDisplayedName = None
    sampleId = None

    logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; GOING TO validate barcodedSamples..." % (selectedTemplate.id, selectedTemplate.planDisplayedName))

    barcodeKitName = selectedEAS.barcodeKitName
    if barcodeKitName:
        csvfile = csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER)
        if single_file:
            errorMsg, barcodedSampleList = _parse_barcodedSamples_from_plan_csv(csvPlanDict, selectedTemplate, barcodeKitName, planObj)
        else:
            errorMsg, barcodedSampleList = _parse_barcodedSamples_from_sample_csv(samples_contents, barcodeKitName, csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER))
        if errorMsg:
            failed.append(("Sample CSV File Name: ", csvfile))
            failed.append(("Barcoded samples validation errors:", errorMsg))

        errorMsg, barcodedSampleJson = _validate_barcodedSamples(barcodedSampleList, selectedTemplate, planObj, selectedEAS)
        if errorMsg:
            failed.append(("Sample CSV File Name: ", csvfile))
            failed.append(("Barcoded samples validation errors:", errorMsg))
        else:
            planObj.get_sampleList().extend(barcodedSampleJson.keys())
    else:
        errorMsg, sampleDisplayedName = _validate_sample(csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE), selectedTemplate, planObj)
        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_SAMPLE, errorMsg))
        else:
            if sampleDisplayedName:
                planObj.get_sampleList().append(sampleDisplayedName)

        errorMsg, sampleId = _validate_sample_id(csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE_ID), selectedTemplate, planObj)
        if errorMsg:
            failed.append((PlanCSVcolumns.COLUMN_SAMPLE_ID, errorMsg))
        else:
            if sampleDisplayedName:
                planObj.get_sampleIdList().append(sampleId if sampleId else "")
                planObj.get_nucleotideTypeList().append("")

    logger.debug("plan_csv_validator.validate_csv_plan() selectedTemplate.pk=%d; selectedTemplate.planDisplayedName=%s; GOING TO validate IR..." % (selectedTemplate.id, selectedTemplate.planDisplayedName))

    errorMsg, uploaders, has_ir_v1_0 = _validate_export(csvPlanDict.get(PlanCSVcolumns.COLUMN_EXPORT), selectedTemplate, selectedEAS, planObj, sampleDisplayedName, sampleId, barcodedSampleJson, username)
    if errorMsg:
        failed.append((PlanCSVcolumns.COLUMN_EXPORT, errorMsg))

    # if uploaderJson:
    #   errorMsg = _validate_IR_workflow_v1_x(csvPlanDict.get(PlanCSVcolumns.COLUMN_IR_V1_X_WORKFLOW), selectedTemplate, planObj, uploaderJson)
    #   if errorMsg:
    #       failed.append((PlanCSVcolumns.COLUMN_IR_V1_X_WORKFLOW, errorMsg))

    try:
        if uploaders and has_ir_v1_0:
            errorMsg = _validate_IR_workflow_v1_0(csvPlanDict.get(PlanCSVcolumns.COLUMN_IR_V1_0_WORKFLOW), selectedTemplate, planObj, uploaders)
            if errorMsg:
                failed.append((PlanCSVcolumns.COLUMN_IR_V1_0_WORKFLOW, errorMsg))

        selectedPlugins = dict(plugins)
        selectedPlugins.update(uploaders)

    except:
        logger.exception(format_exc())
        errorMsg = "Internal error while processing selected plugins info. " + format_exc()
        failed.append(("selectedPlugins", errorMsg))

    #validate the iru configuration settings in the CSV plan upload
    if csvPlanDict.get(PlanCSVcolumns.COLUMN_IR_ACCOUNT):
        iru_validationErrors, selectedPlugins = validate_iruConfig_process_userInputInfo(csvPlanDict,
                                                                  username,
                                                                  samples_contents,
                                                                  planObj,
                                                                  httpHost,
                                                                  selectedPlugins = selectedPlugins)
        if iru_validationErrors:
            failed.append(("IRU validation errors:", iru_validationErrors))

        # this is validated inside the plan_csv_iru_validator
        irChevron_workflow = csvPlanDict.get(PlanCSVcolumns.COLUMN_SAMPLE_IR_WORKFLOW, "")
        if irChevron_workflow and not iru_validationErrors:
            planObj.get_planObj().irworkflow = irChevron_workflow

    rawPlanDict["selectedPlugins"] = selectedPlugins

    logger.debug("EXIT plan_csv_validator.validate_csv_plan() rawPlanDict=%s; " % (rawPlanDict))

    planObj.get_easObj().selectedPlugins = selectedPlugins


    planDict = {
        "plan": planObj.get_planObj(),
        "exp": planObj.get_expObj(),
        "eas": planObj.get_easObj(),
        "samples": planObj.get_sampleList(),
        "sampleIds":  planObj.get_sampleIdList(),
        "sampleNucleotideTypes": planObj.get_nucleotideTypeList()
        }

    return failed, planDict, rawPlanDict, isToSkipRow


def _get_template(templateName):
    templates = PlannedExperiment.objects.filter(planDisplayedName=templateName.strip(), isReusable=True).order_by("-date")

    if templates:
        # logger.debug("plan_csv_valiadtor._get_template() selectedTemplate=%d" %(templates[0].pk))

        return templates[0], None
    else:
        # original template name could have unicode (e.g., trademark)
        templates = PlannedExperiment.objects.filter(isReusable=True, isSystemDefault=False).order_by("-date")
        if templates:
            inputTemplateName = templateName.strip()
            for template in templates:
                templateDisplayedName = template.planDisplayedName.encode("ascii", "ignore")
                if templateDisplayedName == inputTemplateName:
                    return template, None

        logger.debug("plan_csv_validator._get_template() NO template found. ")
        return None, "Template name: " + templateName + " cannot be found to create plans from"

# Validate for supported kit/chip combination:
def _validate_kit_chip_combination(planObj, selectedTemplate, selectedKit):
    errorMsg = None
    selectedChip = None

    user_chipType = planObj.get_expObj().chipType
    try:
        selectedChip = Chip.objects.get(name__iexact=user_chipType, isActive=True)
    except Exception, Err:
        logger.debug("Error in plan.plan_csv_validation %s" % Err)
        errorMsg = Err

    # validate selected chip and kit instrument type combination is supported
    selectedChip_instType = selectedChip.instrumentType
    selectedKit_instType = selectedKit.instrumentType
    if selectedKit_instType:
        if selectedKit_instType not in selectedChip_instType:
            errorMsg = "specified Kit (%s) / Chip (%s) instrument type is not supported" % (selectedKit.name, user_chipType)
            return errorMsg

    # if instrument type is valid: validate if chip type of selected kit is in the supported list
    if selectedKit.chipTypes:
        selectedKit_chipTypes = selectedKit.chipTypes.split(";")
        if selectedKit_chipTypes:
            if user_chipType not in selectedKit_chipTypes:
                errorMsg = "specified Kit (%s) / Chip (%s) combination is not supported" % (selectedKit.name, user_chipType)
                return errorMsg

    # if instrument type and chip type are valid: validate if application type of selected kit is in the supported list
    template_runType = selectedTemplate.runType
    selectedKit_applicationType = selectedKit.applicationType
    if selectedKit_applicationType:
        if "AMPS_ANY" in selectedKit_applicationType:
            selectedKit_applicationType = ['AMPS', 'AMPS_DNA_RNA', 'AMPS_EXOME', 'AMPS_RNA']
        if template_runType not in selectedKit_applicationType:
            errorMsg = "specified Kit (%s) is not supported for the template (%s)" % (selectedKit.name, selectedTemplate.planDisplayedName)
            return errorMsg

    return errorMsg

def _validate_sample_prep_kit(input, selectedTemplate, planObj):
    """
    validate sample prep kit case-insensitively and ignore leading/trailing blanks in the input
    """

    errorMsg = None
    selectedKit = None

    if input:
        try:
            selectedKits = KitInfo.objects.filter(kitType="SamplePrepKit", isActive=True, description__iexact=input.strip())
            if not selectedKits:
                selectedKit = KitInfo.objects.filter(kitType="SamplePrepKit", isActive=True, name__iexact=input.strip())[0]
            else:
                selectedKit = selectedKits[0]

            errorMsg = _validate_kit_chip_combination(planObj, selectedTemplate, selectedKit)

            if not errorMsg:
                planObj.get_planObj().samplePrepKitName = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_planObj().samplePrepKtName = ""

    return errorMsg


def _validate_lib_kit(input, selectedTemplate, planObj):
    """
    validate library kit case-insensitively and ignore leading/trailing blanks in the input
    """

    errorMsg = None
    selectedKit = None

    if input:
        try:
            selectedKits = KitInfo.objects.filter(kitType__in=["LibraryKit", "LibraryPrepKit"], isActive=True, description__iexact=input.strip())
            if not selectedKits:
                selectedKit = KitInfo.objects.filter(kitType__in=["LibraryKit", "LibraryPrepKit"], isActive=True, name__iexact=input.strip())[0]
            else:
                selectedKit = selectedKits[0]

            errorMsg = _validate_kit_chip_combination(planObj, selectedTemplate, selectedKit)

            if not errorMsg:
                planObj.get_easObj().libraryKitName = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_easObj().libraryKitName = ""

    return errorMsg


def _validate_template_kit(input, selectedTemplate, planObj):
    """
    validate tempplating kit case-insensitively and ignore leading/trailing blanks in the input
    """

    errorMsg = None
    selectedKit = None

    if input:
        try:
            selectedKits = KitInfo.objects.filter(kitType__in=["TemplatingKit", "IonChefPrepKit"], isActive=True, description__iexact=input.strip())
            if not selectedKits:
                selectedKit = KitInfo.objects.filter(kitType__in=["TemplatingKit", "IonChefPrepKit"], isActive=True, name__iexact=input.strip())[0]
            else:
                selectedKit = selectedKits[0]

            errorMsg = _validate_kit_chip_combination(planObj, selectedTemplate, selectedKit)

            if not errorMsg:
                planObj.get_planObj().templatingKitName = selectedKit.name

                if selectedKit.kitType == "IonChefPrepKit":
                    planObj.get_planObj().planStatus = "pending"
                else:
                    planObj.get_planObj().planStatus = "planned"
        except:
            errorMsg = input + " not found."
    else:
        # planObj.get_planObj().templatingKitName = ""
        errorMsg = "Required column is empty. "

    return errorMsg


def _validate_control_seq_kit(input, selectedTemplate, planObj):
    """
    validate control sequencing kit case-insensitively and ignore leading/trailing blanks in the input
    """

    errorMsg = None
    selectedKit = None

    if input:
        try:
            selectedKits = KitInfo.objects.filter(kitType="ControlSequenceKit", isActive=True, description__iexact=input.strip())
            if not selectedKits:
                selectedKit = KitInfo.objects.filter(kitType="ControlSequenceKit", isActive=True, name__iexact=input.strip())[0]
            else:
                selectedKit = selectedKits[0]

            errorMsg = _validate_kit_chip_combination(planObj, selectedTemplate, selectedKit)
            if not errorMsg:
                planObj.get_planObj().controlSequencekitname = selectedKit.name
        except:
            errorMsg = input + " not found."
    else:
        planObj.get_planObj().controlSequencekitname = ""

    return errorMsg


def _validate_seq_kit(input, selectedTemplate, planObj, selectedExperiment):
    """
    validate sequencing kit case-insensitively and ignore leading/trailing blanks in the input
    """

    errorMsg = None
    selectedKit = None

    if input:
        try:
            selectedKits = KitInfo.objects.filter(kitType="SequencingKit", isActive=True, description__iexact=input.strip())
            if not selectedKits:
                selectedKit = KitInfo.objects.filter(kitType="SequencingKit", isActive=True, name__iexact=input.strip())[0]
            else:
                selectedKit = selectedKits[0]

            errorMsg = _validate_kit_chip_combination(planObj, selectedTemplate, selectedKit)

            if not errorMsg:
                planObj.get_expObj().sequencekitname = selectedKit.name

                if (selectedKit.name != selectedExperiment.sequencekitname):
                    flowOrder = selectedKit.defaultFlowOrder.flowOrder if selectedKit.defaultFlowOrder else ""
                    planObj.get_expObj().flowsInOrder = flowOrder
        except:
            logger.exception(format_exc())
            errorMsg = input + " not found."
    else:
        planObj.get_expObj().sequencekitname = ""

    return errorMsg


def _validate_chip_type(input, selectedTemplate, planObj, selectedExperiment):
    """
    validate chip type case-insensitively and ignore leading/trailing blanks in the input
    """
    errorMsg = None
    if input:
        try:
            selectedChips = Chip.objects.filter(description__iexact=input.strip(), isActive=True).order_by('-id')

            # if selected chipType is ambiguous, try to go with the template's. If that doesn't help, settle with the 1st one
            if len(selectedChips) == 1:
                planObj.get_expObj().chipType = selectedChips[0].name
            elif len(selectedChips) > 1:
                # template_chipType = selectedTemplate.get_chipType()
                template_chipType = selectedExperiment.chipType

                if template_chipType:
                    template_chipType_objs = Chip.objects.filter(name=template_chipType)

                    if template_chipType_objs:
                        template_chipType_obj = template_chipType_objs[0]
                        if template_chipType_obj.description == input.strip():
                            planObj.get_expObj().chipType = template_chipType_obj.name
                        else:
                            planObj.get_expObj().chipType = selectedChips[0].name
                else:
                    planObj.get_expObj().chipType = selectedChips[0].name

            else:
                errorMsg = input + " not found."
        except:
            logger.exception(format_exc())
            errorMsg = input + " not found."
    else:
        # error due to chip field is required
        errorMsg = "Required column is empty."

    return errorMsg


def _validate_templatingSize(input, selectedTemplate, planObj):
    """
    validate templating size value with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    if input:
        value = input.strip()
        errors = validate_templatingSize(value)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            planObj.get_planObj().templatingSize = value
    else:
        planObj.get_planObj().templatingSize = ""

    return errorMsg


def _validate_libraryReadLength(input, selectedTemplate, planObj):
    """
    validate library read length value with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    if input:
        value = input.strip()
        errors = validate_libraryReadLength(value)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            planObj.get_planObj().libraryReadLength = int(value)
    else:
        planObj.get_planObj().libraryReadLength = 0

    return errorMsg


def _validate_flows(input, selectedTemplate, planObj):
    """
    validate flow value with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    if input:
        value = input.strip()
        errors = validate_flows(value)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            planObj.get_expObj().flows = int(value)

    return errorMsg


def _validate_chip_barcode(input, selectedTemplate, planObj):
    """
    validate Chip Barcode value with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    if input:
        value = input.strip()
        errors = validate_chipBarcode(value)
        if errors:
            errorMsg = ' '.join(errors)
        else:
            planObj.get_expObj().chipBarcode = value
    return errorMsg


def _validate_sample_tube_label(input, selectedTemplate, planObj):
    """
    validate sample tube label with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    if input:
        # value = input.strip().lstrip("0")
        value = input.strip()
        errors = validate_sample_tube_label(value)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            planObj.get_planObj().sampleTubeLabel = value
    else:
        planObj.get_planObj().sampleTubeLabel = ""

    return errorMsg


def _validate_qc_pct(input, selectedTemplate, planObj, displayedName):
    """
    validate QC threshold with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    qcValue = None
    if input:
        value = input.strip()
        errors = validate_QC(value, displayedName)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            qcValue = int(value)
    else:
        qcValue = 1

    return errorMsg, qcValue


def _validate_ref(value, selectedTemplate, planObj):
    """
    validate genome reference case-insensitively with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    if value and value.strip().lower() != 'none':
        plan = planObj.get_planObj()
        applicationGroupName = plan.applicationGroup.name if plan.applicationGroup else ""
        errorMsg, ref_short_name = validate_reference(value, plan.runType, applicationGroupName)
        planObj.get_easObj().reference = ref_short_name
    else:
        planObj.get_easObj().reference = ""

    return errorMsg


def _validate_target_bed(input, selectedTemplate, planObj, selectedReference, isKeywordFound=True, isMixedTypeRNA=False):
    """
    validate target region BED file case-insensitively with leading/trailing blanks in the input ignored
    If value is blank, DO NOT use the template value to substitute
    For non-barcoded plan, isKeywordFound is set to True
    For barcodedPlan, if NO keyword is provided, use the template value to substitute
    For barcodedPlan, if keyword is provided but it is blank, DO NOT use the template value to substitute
    """
    errorMsg = None

    if not selectedTemplate.get_barcodeId():
        # if this is not a barcoded plan, delegate the full validation to the helper method
        runType = selectedTemplate.runType

        # logger.debug("_validate_target_bed GOING TO call _validate_sample_target_bed() input=%s; runType=%s; isKeywordFound=%s" %(input, runType, str(isKeywordFound)))
        applicationGroupName = selectedTemplate.applicationGroup.name if selectedTemplate.applicationGroup else ""
        targetBed_error, validated_targetBed = _validate_sample_target_bed(input, runType, selectedReference, applicationGroupName)
        if targetBed_error:
            errorMsg = targetBed_error
        else:
            errorMsg = ""

        planObj.get_easObj().targetRegionBedFile = validated_targetBed

    else:
        if input and input.strip().lower() != 'none':
            # for barcoded plan with user input
            bedFile = get_bedFile_for_reference(input, selectedReference, hotspot=False)
            if bedFile:
                planObj.get_easObj().targetRegionBedFile = bedFile
            else:
                errorMsg = input + " not found."
        else:
            planObj.get_easObj().targetRegionBedFile = ""

    logger.debug("plan_csv_validator._validate_target_bed() targetBed=%s" % (planObj.get_easObj().targetRegionBedFile))

    return errorMsg


def _validate_hotspot_bed(input, selectedTemplate, planObj, selectedReference):
    """
    validate hotSpot BED file case-insensitively with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    if input and input.strip().lower() != 'none':
        bedFile = get_bedFile_for_reference(input, selectedReference, hotspot=True)
        if bedFile:
            planObj.get_easObj().hotSpotRegionBedFile = bedFile
        else:
            logger.exception(format_exc())
            errorMsg = input + " not found."
    else:
        planObj.get_easObj().hotSpotRegionBedFile = ""

    logger.debug("plan_csv_validator._validate_hotspot_bed() targetBed=%s" % (planObj.get_easObj().hotSpotRegionBedFile))

    return errorMsg


def get_bedFile_for_reference(value, reference="", hotspot=False):
    ''' find BED file by full path, web path or filename
        optionally checks that reference matches, but will return found file even if the input reference is different
    '''
    bedFile = ""
    value = value.strip()
    if value:
        bedFileDict = dict_bed_hotspot()
        key = "hotspotFiles" if hotspot else "bedFiles"
        for bed in bedFileDict.get(key):
            if value == bed.file or value == bed.path or value == os.path.basename(bed.file):
                bedFile = bed.file
                if reference and reference == bed.meta.get("reference", ""):
                    # found the file, return
                    return bedFile

    return bedFile


def _validate_sample_target_bed(input, runType, sampleReference, applicationGroupName, nucleotideType=None):
    """
    validate target region BED file case-insensitively with leading/trailing blanks in the input ignored
    """
    bedFile = ""
    errorMsg = None
    if input and input.strip().lower() != 'none':
        if not sampleReference or sampleReference.strip().lower() == 'none':
            errorMsg = "Invalid/Missing reference"
            return errorMsg, bedFile

        bedFile = get_bedFile_for_reference(input, sampleReference, hotspot=False)
        if not bedFile:
            errorMsg = input + " not found."
            return errorMsg, bedFile

    # validate to ensure mandatory targetBedFile rule, if present, is satisfied
    errors = validate_targetRegionBedFile_for_runType(bedFile, runType, sampleReference, nucleotideType, applicationGroupName)
    if errors:
        errorMsg = '  '.join(errors)
        bedFile = ""

    logger.debug("EXIT _validate_sample_target_bed() input=%s; bedFile=%s" % (input, bedFile))
    return errorMsg, bedFile


def _validate_sample_hotspot_bed(input, sampleReference):
    """
    validate hotSpot BED file case-insensitively with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    bedFile = ""
    if input and input.strip().lower() != 'none':
        if not sampleReference or sampleReference.lower() == 'none':
            errorMsg = "Invalid/Missing reference"
            return errorMsg, bedFile
        bedFile = get_bedFile_for_reference(input, sampleReference, hotspot=True)
        if not bedFile:
            errorMsg = input + " not found."

    return errorMsg, bedFile


def validate_ref_bed_compatibility(input_reference, input_hotSpotBedFile, input_targetRegionBedFile):
    """
    validate if the validated existing bed files are compatible with the validated existing reference
    """
    # logger.debug("ENTER plan_csv_validator.validate_ref_bed_compatibility() input_reference=%s; input_hotSpotBedFile=%s; input_targetRegionBedFile=%s" %(input_reference, input_hotSpotBedFile, input_targetRegionBedFile))

    errorMsg = ""
    if input_reference:
        bedFileDict = dict_bed_hotspot()

        if input_hotSpotBedFile:
            isValidated = False
            for bedFile in bedFileDict.get("hotspotFiles"):
                if input_hotSpotBedFile == bedFile.file:
                    if input_reference != bedFile.meta.get("reference", ""):
                        logger.debug("plan_csv_validator.validate_ref_bed_compatibility() HOTSPOT reference=%s; meta_reference=%s" % (input_reference, bedFile.meta.get("reference", "")))
                        errorMsg = "HotSpot BED file is incompatible with selected reference: %s. " % input_reference
                    else:
                        isValidated = True
        if input_targetRegionBedFile:
            isValidated = False
            for bedFile in bedFileDict.get("bedFiles"):
                if input_targetRegionBedFile == bedFile.file:
                    if input_reference != bedFile.meta.get("reference", ""):
                        logger.debug("plan_csv_validator.validate_ref_bed_compatibility() TARGET REGION reference=%s; meta_reference=%s" % (input_reference, bedFile.meta.get("reference", "")))
                        errorMsg += "Target Regions BED file is incompatible with selected reference: %s." % input_reference
                    else:
                        isValidated = True
    else:
        if input_hotSpotBedFile or input_targetRegionBedFile:
            errorMsg = "Reference missing for the BED files selected."

    return errorMsg


def _validate_plugins(input, selectedTemplate, selectedEAS, planObj):
    """
    validate plugin case-insensitively with leading/trailing blanks in the input ignored
    """

    errorMsg = ""
    plugins = {}

    if input:
        for plugin in input.split(";"):
            if plugin:
                value = plugin.strip()
                try:
                    selectedPlugin = Plugin.objects.filter(name=value, selected=True, active=True)[0]

                    pluginUserInput = {}

                    template_selectedPlugins = selectedEAS.selectedPlugins

                    if selectedPlugin.name in template_selectedPlugins:
                        # logger.info("_validate_plugins() FOUND plugin in selectedTemplate....=%s" %(template_selectedPlugins[plugin.strip()]))
                        pluginUserInput = template_selectedPlugins[selectedPlugin.name]["userInput"]

                    pluginDict = {
                        "id": selectedPlugin.id,
                        "name": selectedPlugin.name,
                        "version": selectedPlugin.version,
                        "userInput": pluginUserInput,
                        "features": []
                    }

                    plugins[selectedPlugin.name] = pluginDict
                except:
                    logger.exception(format_exc())
                    errorMsg += plugin + " not found. "
    else:
        planObj.get_easObj().selectedPlugins = ""

    return errorMsg, plugins


def _validate_projects(input, selectedTemplate, planObj):
    """
    validate projects case-insensitively with leading/trailing blanks in the input ignored
    """

    errorMsg = None
    projects = ''

    if input:
        value = input.strip()
        errors, trimmed_projects = validate_projects(value, delim=";")
        if errors:
            errorMsg = '  '.join(errors)
        else:
            projects = trimmed_projects.replace(";", ",")

    return errorMsg, projects


def _validate_export(input, selectedTemplate, selectedEAS, planObj, sampleDisplayedName, sampleId, barcodedSampleJson, username):
    """
    validate export case-insensitively with leading/trailing blanks in the input ignored
    """
    errorMsg = ""

    plugins = {}
    has_ir_v1_0 = False

    try:
        if input:
            for plugin in input.split(";"):
                if plugin:
                    value = plugin.strip()
                    try:
                        # 20121212-TODO: can we query and filter by EXPORT feature fast?
                        selectedPlugin = Plugin.objects.filter(name=value, selected=True, active=True)[0]

                        if selectedPlugin.name == "IonReporterUploader_V1_0":
                            has_ir_v1_0 = True

                        if selectedEAS:
                            templateSelectedPlugins = selectedEAS.selectedPlugins
                            if selectedPlugin.name == "IonReporterUploader":
                                isToUseDefaultIRAccount = False

                                if "IonReporterUploader" in templateSelectedPlugins:
                                    templateIRUConfig = templateSelectedPlugins.get("IonReporterUploader", {})
                                    pluginDict = templateIRUConfig

                                    templateUserInput = pluginDict.get("userInput", "")
                                    if templateUserInput:
                                        accountId = templateUserInput.get("accountId", "")
                                        accountName = templateUserInput.get("accountName", "")

                                        userInputList = {
                                            "accountId": accountId,
                                            "accountName": accountName,
                                            "userInputInfo": _get_IR_userInputInfo_obj(selectedTemplate.irworkflow, sampleDisplayedName, sampleId, barcodedSampleJson)
                                        }
                                        pluginDict["userInput"] = userInputList

                                        plugins[selectedPlugin.name] = pluginDict

                                    else:
                                        isToUseDefaultIRAccount = True
                                else:
                                    isToUseDefaultIRAccount = True

                                if isToUseDefaultIRAccount:
                                    userIRConfig = get_default_or_first_IR_account_by_userName(username)
                                    if userIRConfig:
                                        pluginDict = {
                                            "id": selectedPlugin.id,
                                            "name": selectedPlugin.name,
                                            "version": selectedPlugin.version,
                                            "features": ['export']
                                        }

                                        userInputList = {
                                            "accountId": userIRConfig["id"],
                                            "accountName": userIRConfig["name"],
                                            "userInputInfo": _get_IR_userInputInfo_obj(selectedTemplate.irworkflow, sampleDisplayedName, sampleId, barcodedSampleJson)
                                        }
                                        pluginDict["userInput"] = userInputList

                                        plugins[selectedPlugin.name] = pluginDict
                    except:
                        logger.exception(format_exc())
                        errorMsg += plugin + " not found. "
        else:
            planObj.get_easObj().selectedPlugins = ""
    except:
        logger.exception(format_exc())
        errorMsg = "Internal error during IRU processing"

    return errorMsg, plugins, has_ir_v1_0


def _get_IR_userInputInfo_obj(selectedIrWorkflow, sampleDisplayedName, sampleId, barcodedSampleJson):

    userInputInfo_obj = []
    if sampleDisplayedName:
        userInputInfo_obj.append(_create_IR_sample_userInputInfo(selectedIrWorkflow, sampleDisplayedName, sampleId, ""))
    else:
        barcodedSamples = barcodedSampleJson.keys()
        for barcodedSample in barcodedSamples:
            value = barcodedSampleJson[barcodedSample]

            barcodeSampleInfo_list = value.get("barcodeSampleInfo", {})
            barcodes = value.get("barcodes", [])
            for barcode in barcodes:
                barcodeSampleInfo = barcodeSampleInfo_list.get(barcode, {})
                # logger.debug("_get_IR_userInputInfo_obj() barcodedSample=%s; sampleDisplayedName=%s; barcode=%s; barcodeSampleInfo=%s" %(barcodedSample, sampleDisplayedName, barcode, barcodeSampleInfo))

                userInputInfo_obj.append(_create_IR_barcoded_sample_userInputInfo(selectedIrWorkflow, barcodedSample, barcodeSampleInfo, barcode))

    return userInputInfo_obj


def _create_IR_sample_userInputInfo(selectedIrWorkflow, sampleDisplayedName, sampleId, sampleNucleotideType):
    # we don't have complete info to construct userInputInfo for IRU. Set workflow to blank
    irWorkflow = ""

    if sampleDisplayedName:
        info_dict = {
            "ApplicationType": "",
            "Gender": "",
            "Relation": "",
            "RelationRole": "",
            "Workflow": irWorkflow,
            "sample": sampleDisplayedName,
            "sampleDescription": "",
            "sampleExternalId": "" if sampleId == None else sampleId,
            "sampleName": get_internal_name_for_displayed_name(sampleDisplayedName),
            "NucleotideType": "" if sampleNucleotideType == None else sampleNucleotideType,
            "setid": get_ir_set_id()

        }
        return info_dict

    return {}


def _create_IR_barcoded_sample_userInputInfo(selectedIrWorkflow, sampleDisplayedName, barcodeSampleInfo, barcodeName):
    # we don't have complete info to construct userInputInfo for IRU. Set workflow to blank
    irWorkflow = ""

    if sampleDisplayedName and barcodeName:
        info_dict = {
            "ApplicationType": "",
            "Gender": "",
            "Relation": "",
            "RelationRole": "",
            "Workflow": irWorkflow,
            "barcodeId": barcodeName,
            "sample": sampleDisplayedName,
            "sampleDescription": "",
            "sampleExternalId": barcodeSampleInfo.get("externalId", ""),
            "sampleName": get_internal_name_for_displayed_name(sampleDisplayedName),
            "NucleotideType": barcodeSampleInfo.get("nucleotideType", ""),
            "setid": get_ir_set_id()

        }
        return info_dict

    return {}


def _validate_plan_name(input, selectedTemplate, planObj):
    """
    validate plan name with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    errors = validate_plan_name(input)
    if errors:
        errorMsg = '  '.join(errors)
    else:
        value = input.strip()
        planObj.get_planObj().planDisplayedName = value
        planObj.get_planObj().planName = value.replace(' ', '_')

    return errorMsg


def _validate_notes(input, selectedTemplate, planObj):
    """
    validate notes with leading/trailing blanks in the input ignored
    """
    errorMsg = None
    if input:
        errors = validate_notes(input)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            planObj.get_expObj().notes = input.strip()
    else:
        planObj.get_expObj().notes = ""

    return errorMsg


def _validate_LIMS_data(input, selectedTemplate, planObj):
    """
    No validation but LIMS data with leading/trailing blanks in the input will be trimmed off
    """
    errorMsg = None
    if input:
        data = input.strip()
        try:
            if planObj.get_planObj().metaData:
                logger.debug("plan_csv_validator._validator_LIMS_data() B4 planObj.get_planObj().metaData=%s" % (planObj.get_planObj().metaData))
            else:
                planObj.get_planObj().metaData = {}

            if len(planObj.get_planObj().metaData.get("LIMS", [])) == 0:
                planObj.get_planObj().metaData["LIMS"] = []

            planObj.get_planObj().metaData["LIMS"].append(data)

            logger.debug("EXIT plan_csv_validator._validator_LIMS_data() AFTER planObj.get_planObj().metaData=%s" % (planObj.get_planObj().metaData))

        except:
            logger.exception(format_exc())
            errorMsg = "Internal error during LIMS data processing"

#         self.metaData["Status"] = status
#         self.metaData["Date"] = "%s" % timezone.now()
#         self.metaData["Info"] = info
#         self.metaData["Comment"] = comment
#
# Try to read the Log entry, if it does not exist, create it
#         if len(self.metaData.get("Log",[])) == 0:
#             self.metaData["Log"] = []
#         self.metaData["Log"].append({"Status":self.metaData.get("Status"), "Date":self.metaData.get("Date"), "Info":self.metaData.get("Info"), "Comment":comment})
    return errorMsg


def _validate_sample(input, selectedTemplate, planObj):
    """
    validate sample name with leading/trailing blanks in the input ignored
    """

    errorMsg = None
    sampleDisplayedName = ""

    if not input:
        errorMsg = "Required column is empty"
    else:
        errors = validate_sample_name(input)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            sampleDisplayedName = input.strip()

    return errorMsg, sampleDisplayedName


def _validate_sample_id(input, selectedTemplate, planObj):
    """
    validate sample id with leading/trailing blanks in the input ignored
    """

    errorMsg = None
    sampleId = ""

    if input:
        errors = validate_sample_id(input)
        if errors:
            errorMsg = '  '.join(errors)
        else:
            sampleId = input.strip()

    return errorMsg, sampleId


def _parse_barcodedSamples_from_plan_csv(input, selectedTemplate, barcodeKitName, planObj):
    '''
    get barcodedSamples info from Plan CSV file
    '''
    barcodedSampleList = []
    errorMsgDict = {}
    errorMsg = None

    barcodes = list(dnaBarcode.objects.filter(name=barcodeKitName).values('id', 'id_str').order_by('id_str'))
    if len(barcodes) == 0:
        errorMsg = "Barcode " + barcodeKitName + " cannot be found. "
        return errorMsg, barcodedSampleList

    runType = selectedTemplate.runType
    applicationGroupName = selectedTemplate.applicationGroup.name if selectedTemplate.applicationGroup else ""

    try:
        for barcode in barcodes:
            barcodeName = barcode["id_str"]
            key = barcodeName + PlanCSVcolumns.COLUMN_BC_SAMPLE_KEY

            sampleDict = {
                'error_key': key,
                'barcode': barcodeName,
                'sampleName': "",
                'sampleExternalId': "",
                'sampleDescription': "",
                'nucleotideType': "",
                'reference': "",
                'targetRegionBedFile': "",
                'hotSpotRegionBedFile': "",
                'controlType': ""
            }

            foundSampleRefKeyword = False
            foundSampleTargetBedKeyword = False
            foundSampleRnaTargetBedKeyword = False
            foundSampleHotSpotBedKeyword = False

            sampleInfo = input.get(key, "").strip()
            if sampleInfo:
                for sampleToken in sampleInfo.split(";"):
                    sampleToken = sampleToken.strip()

                    if sampleToken:
                        if sampleToken.startswith(KEY_SAMPLE_ID):
                            sampleDict['sampleExternalId'] = sampleToken[3:].strip()
                        elif sampleToken.startswith(KEY_SAMPLE_TYPE):
                            sampleDict['nucleotideType'] = sampleToken[5:].strip().upper()
                        elif sampleToken.startswith(KEY_SAMPLE_RNA_REF):
                            sampleDict['reference'] = sampleToken[8:].strip()
                            foundSampleRefKeyword = True
                        elif sampleToken.startswith(KEY_SAMPLE_REF):
                            sampleDict['reference'] = sampleToken[4:].strip()
                            foundSampleRefKeyword = True
                        elif sampleToken.startswith(KEY_SAMPLE_TARGET):
                            sampleDict['targetRegionBedFile'] = sampleToken[7:].strip()
                            foundSampleTargetBedKeyword = True
                        elif sampleToken.startswith(KEY_SAMPLE_RNA_TARGET):
                            sampleDict['targetRegionBedFile'] = sampleToken[11:].strip()
                            foundSampleRnaTargetBedKeyword = True
                        elif sampleToken.startswith(KEY_SAMPLE_HOTSPOT):
                            sampleDict['hotSpotRegionBedFile'] = sampleToken[8:].strip()
                            foundSampleHotSpotBedKeyword = True
                        elif sampleToken.startswith(KEY_SAMPLE_CONTROLTYPE):
                            sampleDict['controlType'] = sampleToken[13:].strip()
                        else:
                            sampleDict['sampleName'] = sampleToken

                if not sampleDict['sampleName']:
                    errorMsgDict[key] = '  '.join(["Sample name is required "])
                else:
                    # Logic:
                    # If NO keyword is provided, use the template value to substitute
                    # If keyword is provided but it is blank, DO NOT use the template value to substitute
                    if runType == "AMPS_DNA_RNA" and sampleDict['nucleotideType'].upper() == "RNA":
                        if not foundSampleRefKeyword:
                            sampleDict['reference'] = planObj.get_easObj().mixedTypeRNA_reference
                        if not foundSampleRnaTargetBedKeyword:
                            sampleDict['targetRegionBedFile'] = planObj.get_easObj().mixedTypeRNA_targetRegionBedFile
                    else:
                        if not foundSampleRefKeyword:
                            sampleDict['reference'] = planObj.get_easObj().reference
                        if not foundSampleTargetBedKeyword:
                            sampleDict['targetRegionBedFile'] = planObj.get_easObj().targetRegionBedFile if runType != "RNA" else ""
                        if not foundSampleHotSpotBedKeyword:
                            sampleDict['hotSpotRegionBedFile'] = planObj.get_easObj().hotSpotRegionBedFile if runType not in ["RNA", "AMPS_RNA"] else ""

                    barcodedSampleList.append(sampleDict)
    except:
        logger.error(format_exc())
        errorMsg = "Internal error during barcoded sample processing"
        return errorMsg, []

    if not barcodedSampleList:
        errorMsg = "Required column is empty. At least one barcoded sample is required. "
    elif errorMsgDict:
        errorMsg = json.dumps(errorMsgDict)

    return errorMsg, barcodedSampleList


def _parse_barcodedSamples_from_sample_csv(samples_contents, barcodeKitName, csvFile):
    '''
    get barcodedSamples info from Samples CSV file
    '''
    if not samples_contents:
        errorMsg = "No barcoded sample contents found in %s" % csvFile
        return errorMsg, []

    try:
        barcodedSampleList = []
        errorMsgDict = {}
        errorMsg = None
        starting_index = 3
        for index, row in enumerate(samples_contents):
            sampleName = row.get(PlanCSVcolumns.COLUMN_SAMPLE_NAME, '').strip()
            if not sampleName:
                continue

            error_key = "%d" % (index + starting_index)
            barcodeName = row.get(PlanCSVcolumns.COLUMN_BARCODE, '').strip()
            if not barcodeName:
                errorMsgDict[error_key] = "Barcode is required"
            else:
                barcode = dnaBarcode.objects.filter(name=barcodeKitName, id_str=barcodeName)
                if not barcode:
                    errorMsgDict[error_key] = "Barcode %s not found in selected barcode set" % barcodeName

            sampleDict = {
                'error_key': error_key,
                'barcode': barcodeName,
                'sampleName': sampleName,
                'sampleExternalId': row.get(PlanCSVcolumns.COLUMN_SAMPLE_ID, '').strip(),
                'sampleDescription': row.get(PlanCSVcolumns.COLUMN_SAMPLE_DESCRIPTION, '').strip(),
                'nucleotideType': row.get(PlanCSVcolumns.COLUMN_NUCLEOTIDE_TYPE, '').strip(),
                'reference': row.get(PlanCSVcolumns.COLUMN_REF, '').strip(),
                'targetRegionBedFile': row.get(PlanCSVcolumns.COLUMN_TARGET_BED, '').strip(),
                'hotSpotRegionBedFile': row.get(PlanCSVcolumns.COLUMN_HOTSPOT_BED, '').strip(),
                'controlType': row.get(PlanCSVcolumns.COLUMN_SAMPLE_CONTROLTYPE, '').strip(),
            }
            barcodedSampleList.append(sampleDict)

    except:
        logger.error(format_exc())
        errorMsg = "Internal error during barcoded sample processing from %s" % csvFile
        return errorMsg, []

    if not barcodedSampleList:
        errorMsg = json.dumps({starting_index: "Required column is empty. At least one barcoded sample is required. "})
    elif errorMsgDict:
        errorMsg = json.dumps(errorMsgDict)

    return errorMsg, barcodedSampleList


def _validate_barcodedSamples(barcodedSampleList, selectedTemplate, planObj, selectedEAS):
    barcodedSampleJson = {}
    errorMsgDict = {}
    errorMsg = None

# 20131211 example:
# barcodedSamples": {
#
#    "s 1": {
#        "barcodeSampleInfo": {
#            "MuSeek_001": {
#                "description": "",
#                "externalId": ""
#            },
#            "MuSeek_011": {
#                "description": "",
#                "externalId": ""
#            }
#        },
#        "barcodes": [
#            "MuSeek_001"
#            "MuSeek_011"
#        ]
#    },
#    "s 2": {
#        "barcodeSampleInfo": {
#            "MuSeek_002": {
#                "description": "",
#                "externalId": ""
#            }
#        },
#        "barcodes": [
#            "MuSeek_002"
#        ]
#    }
#
#},

# 20140325 example for DNA + RNA:
# barcodedSamples": {
#
#    "s 1": {
#        "barcodeSampleInfo": {
#            "IonXpress_001": {
#                "controlSequenceType": "",
#                "description": "test desc",
#                "externalId": "ext 1",
#                "hotSpotRegionBedFile": "/results/uploads/BED/8/polio/unmerged/detail/polio_hotspot.bed",
#                "nucleotideType": "DNA",
#                "reference": "polio",
#                "targetRegionBedFile": "/results/uploads/BED/7/polio/unmerged/detail/polio.bed"
#            },
#            "IonXpress_002": {
#                "controlSequenceType": "",
#                "description": "test desc",
#                "externalId": "ext 1",
#                "hotSpotRegionBedFile": "",
#                "nucleotideType": "RNA",
#                "reference": "",
#                "targetRegionBedFile": ""
#            }
#        },
#        "barcodes": [
#            "IonXpress_001",
#            "IonXpress_002"
#        ]
#    }
#
#},

    runType = selectedTemplate.runType
    applicationGroupName = selectedTemplate.applicationGroup.name if selectedTemplate.applicationGroup else ""

    try:
        for row in barcodedSampleList:
            error_key = row["error_key"]
            barcodeName = row["barcode"]
            sampleName = row["sampleName"]
            sampleExternalId = row['sampleExternalId']
            sampleDescription = row['sampleDescription']
            nucleotideType = row['nucleotideType']
            sampleReference = row['reference']
            targetRegionBedFile = row['targetRegionBedFile']
            hotSpotRegionBedFile = row['hotSpotRegionBedFile']
            controlType = row['controlType']

            # validate reference and bedfiles
            errors, sampleReference, nucleotideType = validate_barcoded_sample_info(sampleName, sampleExternalId, nucleotideType, sampleReference, runType, applicationGroupName)

            sample_targetBed_error, targetRegionBedFile = _validate_sample_target_bed(targetRegionBedFile, runType, sampleReference, applicationGroupName, nucleotideType)
            if sample_targetBed_error:
                errors.extend([" Target BED File: ", sample_targetBed_error])
            sample_hotSpotBed_error, hotSpotRegionBedFile = _validate_sample_hotspot_bed(hotSpotRegionBedFile, sampleReference)
            if sample_hotSpotBed_error:
                errors.extend([" HotSpot BED File: ", sample_hotSpotBed_error])

            # validate reference and bed files compatibility
            error_ref_bedFiles = validate_ref_bed_compatibility(sampleReference, hotSpotRegionBedFile, targetRegionBedFile)
            if error_ref_bedFiles:
                errors.append(error_ref_bedFiles)

            # validate sample controlType
            error_controlType, controlType = validate_sampleControlType(controlType)
            if error_controlType:
                errors.extend(error_controlType)

            if errors:
                errorMsgDict[error_key] = '  '.join(errors)
            else:
                planObj.get_sampleList().append(sampleName)
                planObj.get_sampleIdList().append(sampleExternalId if sampleExternalId else "")

                # Generate barcodedSamples dict
                if sampleName not in barcodedSampleJson:
                    barcodedSampleData = {
                        "barcodes": [],
                        "barcodeSampleInfo": {}
                    }
                else:
                    barcodedSampleData = barcodedSampleJson[sampleName]

                barcodedSampleData["barcodes"].append(barcodeName)
                barcodedSampleData["barcodeSampleInfo"][barcodeName] = {
                    'externalId': sampleExternalId,
                    'description': sampleDescription,
                    'nucleotideType': nucleotideType,
                    'controlSequenceType': '',
                    'reference': sampleReference,
                    'targetRegionBedFile': targetRegionBedFile,
                    'hotSpotRegionBedFile': hotSpotRegionBedFile,
                    'sseBedFile': selectedEAS.sseBedFile if targetRegionBedFile == selectedEAS.targetRegionBedFile else '',
                    'controlType': controlType
                }

                barcodedSampleJson[sampleName] = dict(barcodedSampleData)
    except:
        logger.error(format_exc())
        errorMsg = "Internal error during barcoded sample processing"
        return errorMsg, barcodedSampleJson

    if errorMsgDict:
        errorMsg = json.dumps(errorMsgDict)

    planObj.get_easObj().barcodedSamples = barcodedSampleJson

    return errorMsg, barcodedSampleJson


def _validate_IR_workflow_v1_0(input, selectedTemplate, planObj, uploaderList):
    errorMsg = None

    if not input:
        return errorMsg

    irUploader = None
    for uploader in uploaderList:
        for key, value in uploader.items():

            if key == "name" and value == "IonReporterUploader_V1_0":
                irUploader = uploader

    for key, value in irUploader.items():
        if key == "userInput":
            userInputDict = value[0]
            userInputDict["Workflow"] = input.strip()

    # logger.info("EXIT _validate_IR_workflow_v1_0() workflow_v1_0 uploaderList=%s" %(uploaderList))
    return errorMsg

# def _validate_IR_workflow_v1_x(input, selectedTemplate, planObj, uploaderJson):
 #   return None

