# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb.models import Chip, LibraryKey, RunType, KitInfo, common_CV, ApplicationGroup, SampleGroupType_CV, dnaBarcode, ReferenceGenome, ApplProduct, PlannedExperiment, SampleAnnotation_CV
from iondb.utils import validation
from iondb.rundb.plan.views_helper import dict_bed_hotspot
import os
import logging
logger = logging.getLogger(__name__)


MAX_LENGTH_PLAN_NAME = 60
MAX_LENGTH_SAMPLE_NAME = 127
MAX_LENGTH_SAMPLE_ID = 127
MAX_LENGTH_PROJECT_NAME = 64
MAX_LENGTH_NOTES = 1024
MAX_LENGTH_SAMPLE_TUBE_LABEL = 512
MAX_FLOWS = 2000
MAX_QC_INT = 100
PROJECT_NAME_LENGTH = 64
MIN_LIBRARY_READ_LENGTH = 0
MAX_LIBRARY_READ_LENGTH = 1000

VALID_TEMPLATING_SIZES = ["200", "400"]


def validate_plan_name(value, displayedName='Plan Name'):
    errors = []
    if not validation.has_value(value):
        errors.append(validation.required_error(displayedName))

    if not validation.is_valid_chars(value):
        errors.append(validation.invalid_chars_error(displayedName))

    if not validation.is_valid_length(value, MAX_LENGTH_PLAN_NAME):
        errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_PLAN_NAME))

    return errors


def validate_notes(value, displayedName='Notes'):
    errors = []
    if not validation.is_valid_chars(value):
        errors.append(validation.invalid_chars_error(displayedName))

    if not validation.is_valid_length(value, MAX_LENGTH_NOTES):
        errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_NOTES))

    return errors


def validate_sample_name(value, displayedName='Sample Name'):
    errors = []
    if not validation.is_valid_chars(value):
        errors.append(validation.invalid_chars_error(displayedName))

    if not validation.is_valid_leading_chars(value):
        errors.append(validation.invalid_chars_error(displayedName))

    if not validation.is_valid_length(value, MAX_LENGTH_SAMPLE_NAME):
        errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_SAMPLE_NAME))

    return errors


def validate_barcoded_sample_info(sampleName, sampleId, nucleotideType, sampleReference, runType, applicationGroupName, displayedName='Barcoded Sample'):
    errors = []
    if not validation.is_valid_chars(sampleName):
        errors.append(validation.invalid_chars_error(displayedName))

    if not validation.is_valid_leading_chars(sampleName):
        errors.append(validation.invalid_chars_error(displayedName))

    if not validation.is_valid_length(sampleName, MAX_LENGTH_SAMPLE_NAME):
        errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_SAMPLE_NAME))

    sample_id_errors = validate_sample_id(sampleId)
    if sample_id_errors:
        errors.extend(sample_id_errors)

    nucleotideType_errors, sample_nucleotideType = validate_sample_nucleotideType(nucleotideType, runType, applicationGroupName)
    if (nucleotideType_errors):
        errors.extend(nucleotideType_errors)

    ref_displayedName = "Sample Reference" if nucleotideType != "RNA" else "RNA Sample Reference"
    ref_errors, ref_short_name = validate_reference(sampleReference, runType, applicationGroupName, displayedName=ref_displayedName)
    if (ref_errors):
        errors.extend(ref_errors)

    return errors, ref_short_name, sample_nucleotideType


def validate_sample_nucleotideType(value, runType, applicationGroupName, displayedName='Sample Nucleotide Type'):
    """
    validate nucleotide type case-insensitively with leading/trailing blanks in the input ignored
    """
    errors = []
    nucleotideType = ""
    try:
        runTypeObj = RunType.objects.filter(runType=runType)[0]
        value_from_runType = runTypeObj.nucleotideType.upper() if runTypeObj.nucleotideType else ""
    except:
        value_from_runType = ""

    if value:
        nucleotideType = value.strip().upper()
        if nucleotideType == "FUSIONS": nucleotideType = "RNA"

        if value_from_runType:
            if value_from_runType == "DNA_RNA":
                valid_values = ["DNA", "RNA", "Fusions"]
            else:
                valid_values = [value_from_runType]
        else:
            valid_values = ["DNA", "RNA"]

        if not validation.is_valid_keyword(nucleotideType, valid_values):
            errors.append(validation.invalid_keyword_error(displayedName, valid_values))
    else:
        # fill in nucleotideType from selected runType or applicationGroup
        if value_from_runType and value_from_runType != "DNA_RNA":
            nucleotideType = value_from_runType
        elif applicationGroupName in ["DNA", "RNA"]:
            nucleotideType = applicationGroupName

    return errors, nucleotideType


def validate_reference(value, runType, applicationGroupName, displayedName='Reference'):
    errors = []
    ref_short_name = ""
    value = value.strip() if value else ""

    if value:
        applProduct = ApplProduct.objects.filter(isActive=True, applType__runType=runType, applicationGroup__name=applicationGroupName) \
            or ApplProduct.objects.filter(isActive=True, applType__runType=runType)

        if applProduct and not applProduct[0].isReferenceSelectionSupported:
            errors.append(displayedName+" selection is not supported for this Application")
        else:
            selectedRefs = ReferenceGenome.objects.filter(name=value) or \
                ReferenceGenome.objects.filter(name__iexact=value) or \
                ReferenceGenome.objects.filter(short_name=value) or \
                ReferenceGenome.objects.filter(short_name__iexact=value)

            if selectedRefs:
                ref_short_name = selectedRefs[0].short_name
            else:
                errors.append(validation.invalid_not_found_error(displayedName, value))

    return errors, ref_short_name


def validate_reference_short_name(value, displayedName='Reference'):
    errors = []
    if validation.has_value(value):
        value = value.strip()
        reference = ReferenceGenome.objects.filter(short_name=value, enabled=True)
        if not reference.exists():
            generic_not_found_error = validation.invalid_not_found_error(displayedName, value)
            error_fixing_message = ". To import it, visit Settings > References > Import Preloaded Ion References"
            reference_error_message = generic_not_found_error.strip() + error_fixing_message
            errors.append(reference_error_message)
    return errors


def validate_sample_id(value, displayedName='Sample Id'):
    errors = []
    if not validation.is_valid_chars(value):
        errors.append(validation.invalid_chars_error(displayedName))

    if not validation.is_valid_length(value, MAX_LENGTH_SAMPLE_ID):
        errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_SAMPLE_ID))

    return errors


def validate_sample_tube_label(value, displayedName='Sample Tube Label'):
    errors = []

    if value:
        if not validation.is_valid_chars(value):
            errors.append(validation.invalid_chars_error(displayedName))

        if not validation.is_valid_length(value, MAX_LENGTH_SAMPLE_TUBE_LABEL):
            errors.append(validation.invalid_length_error(displayedName, MAX_LENGTH_SAMPLE_TUBE_LABEL))

    return errors


def validate_chip_type(value, displayedName='Chip Type'):
    errors = []
    if not value:
        errors.append(validation.required_error(displayedName))
    else:
        value = value.strip()
        chip = Chip.objects.filter(name=value)
        if not chip:
            chip = Chip.objects.filter(description=value)

        if not chip:
            errors.append('Chip %s not found' % value)

    return errors


def validate_flows(value, displayedName='Flows'):
    errors = []
    if type(value) != int and value.isdigit():
        value2 = int(value)
    else:
        value2 = value

    if type(value2) == int:
        if not validation.is_valid_uint(value2):
            errors.append(validation.invalid_uint(displayedName))
        elif value2 > MAX_FLOWS:
            errors.append(displayedName + ' must be a positive integer within range [1, 2000]')
    else:
        errors.append(displayedName + ' must be a positive integer within range [1, 2000]')

    return errors

def validate_library_key(value, displayedName='Library key'):
    errors = []
    selectedLibKey = None
    if not value:
        errors.append(validation.required_error(displayedName))
    else:
        try:
            selectedLibKey = LibraryKey.objects.get(name__iexact=value.strip())
        except LibraryKey.DoesNotExist:
            try:
                selectedLibKey = LibraryKey.objects.get(sequence__iexact=value.strip())
            except LibraryKey.DoesNotExist:
                logger.debug("plan_validator.validate_lib_key ...%s not found" % input)
                selectedLibKey = None

        if not selectedLibKey:
            errors.append('Library key %s not found' % value)

    return errors, selectedLibKey

def validate_libraryReadLength(value, displayedName='Library Read Length'):
    errors = []
    if type(value) != int and value.isdigit():
        value2 = int(value)
    else:
        value2 = value

    if type(value2) == int:
        if not validation.is_valid_uint_n_zero(value2):
            errors.append(validation.invalid_uint(displayedName))
        elif MIN_LIBRARY_READ_LENGTH < value2 > MAX_LIBRARY_READ_LENGTH:
            errors.append(displayedName + ' must be a positive integer within range [0, 1000]')
    else:
        errors.append(displayedName + ' must be a positive integer within range [0, 1000]')

    return errors


def validate_templatingSize(value, displayedName='Templating Size'):
    """
    validate templating size case-insensitively with leading/trailing blanks in the input ignored
    """
    errors = []
    input = ""
    valid_values = VALID_TEMPLATING_SIZES

    if value:
        input = value.strip().upper()

        if not validation.is_valid_keyword(input, valid_values):
            errors.append(validation.invalid_keyword_error(displayedName, valid_values))
    return errors


def validate_QC(value, displayedName):
    errors = []
    if not validation.is_valid_uint(value):
        errors.append(validation.invalid_uint(displayedName))
    elif int(value) > MAX_QC_INT:
        errors.append(displayedName + ' must be a positive whole number within range [1, 100)')

    return errors


def validate_projects(value, displayedName='Project Name', delim=','):
    """
    validate projects case-insensitively with leading/trailing blanks in the input ignored
    """

    errors = []
    trimmed_projects = ""
    if value:
        for project in value.split(delim):
            trimmed_project = project.strip()
            if trimmed_project:
                trimmed_projects = trimmed_projects + trimmed_project + delim

                if not validation.is_valid_chars(trimmed_project):
                    errors.append(validation.invalid_chars_error(displayedName))
                if not validation.is_valid_length(trimmed_project, PROJECT_NAME_LENGTH):
                    errors.append(validation.invalid_length_error(displayedName, PROJECT_NAME_LENGTH))
                if errors:
                    break

    return errors, trimmed_projects


def validate_barcode_kit_name(value, displayedName="Barcode Kit"):
    errors = []

    if validation.has_value(value):
        value = value.strip()
        kits = dnaBarcode.objects.filter(name=value)
        if not kits:
            errors.append("%s %s not found" % (displayedName, value))

    return errors


def validate_sequencing_kit_name(value, displayedName="Sequencing Kit"):
    errors = []

    if validation.has_value(value):
        value = value.strip()
        kit = KitInfo.objects.filter(kitType__in=["SequencingKit"], name=value)
        if not kit:
            kit = KitInfo.objects.filter(kitType__in=["SequencingKit"], description=value)

        if not kit:
            errors.append("%s %s not found" % (displayedName, value))

    return errors


def validate_plan_templating_kit_name(value, displayedName="Template Kit"):
    errors = []

    if not validation.has_value(value):
        errors.append(validation.required_error(displayedName))
    else:
        value = value.strip()
        kit = KitInfo.objects.filter(kitType__in=["TemplatingKit", "IonChefPrepKit"], name=value)
        if not kit:
            kit = KitInfo.objects.filter(kitType__in=["TemplatingKit", "IonChefPrepKit"], description=value)

        if not kit:
            errors.append("%s %s not found" % (displayedName, value))

    return errors

def _validate_spp_value_uid(value):
    common_CVs = common_CV.objects.filter(cv_type__in=["samplePrepProtocol"])

    for cv in common_CVs:
        isValid = True
        if ((value.lower() == cv.value.lower()) or
                (value.lower() == cv.uid.lower())):
            return isValid, cv
        else:
            isValid = False

    return isValid, value

def _validate_ssp_templatingKit(samplePrepValue, templatingKitName):
    # validate whether the samplePrepProtocal is supported for this templatingKit
    isValid = False
    kit = KitInfo.objects.get(kitType__in=["TemplatingKit", "IonChefPrepKit"], name=templatingKitName)
    for value in samplePrepValue.split(";"):
            if value in kit.categories:
                isValid = True
                break;
    return isValid

def validate_plan_samplePrepProtocol(value, templatingKitName, displayedName="sample Prep Protocol"):
    # validate if input matches the common_CV value or uid
    errors = []
    if value:
        value = value.strip()
        isValid, cv = _validate_spp_value_uid(value)
        if not isValid:
            errors.append("%s not found : %s is ignored" % (displayedName, value))
        if isValid:
            isValid = _validate_ssp_templatingKit(cv.categories, templatingKitName)
            if not isValid:
                errors.append("%s not supported for this templatingKit %s : %s is ignored  " % (displayedName, templatingKitName, value))
            value = cv.value
    return errors, value

def validate_runType(runType):
    errors = []
    if runType:
        runTypeObjs = RunType.objects.filter(runType__iexact=runType)
        if not runTypeObjs:
            validRunTypes = RunType.objects.values_list('runType', flat=True)
            validRunTypes = ', '.join(validRunTypes)
            errors.append("%s not a valid Run Type. Valid Run Types are %s" % (runType, validRunTypes))
    return errors


def validate_application_group_for_runType(value, runType, displayedName="Application Group"):
    errors = []

    if value:
        value = value.strip()
        applicationGroups = ApplicationGroup.objects.filter(name__iexact=value) | ApplicationGroup.objects.filter(description__iexact=value)
        if applicationGroups:
            applicationGroup = applicationGroups[0]
            if runType:
                runTypeObjs = RunType.objects.filter(runType__iexact=runType)
                if runTypeObjs:
                    associations = runTypeObjs[0].applicationGroups.filter(name=applicationGroup.name)
                    if not associations:
                        errors.append("%s %s not valid for Run Type %s" % (displayedName, value, runType))
            if not runType or not runTypeObjs:
                errors.append("Invalid/Missing RunType")
        else:
            errors.append("%s %s not found" % (displayedName, value))

    return errors


def validate_sample_grouping(value, displayedName="Sample Grouping"):
    errors = []

    if value:
        value = value.strip()

        groupings = SampleGroupType_CV.objects.filter(displayedName__iexact=value)
        if not groupings:
            errors.append("%s %s not found" % (displayedName, value))

    return errors


def validate_barcode_sample_association(selectedBarcodes, selectedBarcodeKit):
    errors = {"MISSING_BARCODE": "", "DUPLICATE_BARCODE": ""}

    if not selectedBarcodeKit:
        return errors

    prior_barcodes = []

    if not selectedBarcodes:
        errors["MISSING_BARCODE"] = "Please select a barcode for each sample"
    else:
        for selectedBarcode in selectedBarcodes:
            if selectedBarcode in prior_barcodes:
                # only include unique barcode selection error messages
                message = "Barcode %s selections have to be unique\n" % selectedBarcode

                value = errors["DUPLICATE_BARCODE"]
                if message not in value:
                    errors["DUPLICATE_BARCODE"] = errors["DUPLICATE_BARCODE"] + message
            else:
                prior_barcodes.append(selectedBarcode)

    # logger.debug("errors=%s" %(errors))

    return errors


def validate_targetRegionBedFile_for_runType(value, runType, reference, nucleotideType=None, applicationGroupName=None, displayedName="Target Regions BED File"):
    """
    validate targetRegionBedFile based on the selected reference and the plan's runType
    """
    errors = []
    value = value.strip() if value else ""

    logger.debug("plan_validator.validate_targetRegionBedFile_for_runType() value=%s; runType=%s; reference=%s; nucleotideType=%s; applicationGroupName=%s" % (value, runType, reference, nucleotideType, applicationGroupName))

    if reference:
        if runType:
            runType = runType.strip()
            applProducts = ApplProduct.objects.filter(isActive=True, applType__runType=runType, applicationGroup__name=applicationGroupName) \
                or ApplProduct.objects.filter(isActive=True, applType__runType=runType)
            if applProducts:
                applProduct = applProducts[0]
                if applProduct:
                    if validation.has_value(value) and not applProduct.isTargetRegionBEDFileSupported:
                        errors.append(displayedName+" selection is not supported for this Application")
                    else:
                        isRequired = applProduct.isTargetRegionBEDFileSelectionRequiredForRefSelection
                        if isRequired and not validation.has_value(value):
                            # skip for now
                            if runType == "AMPS_DNA_RNA" and nucleotideType and nucleotideType.upper() == "RNA":
                                logger.debug("plan_validator.validate_targetRegionBedFile_for_runType() ALLOW MISSING targetRegionBed for runType=%s; nucleotideType=%s" % (runType, nucleotideType))
                            elif runType == "AMPS_RNA" and applicationGroupName and applicationGroupName in ["DNA + RNA", "DNA and Fusions"]:
                                logger.debug("plan_validator.validate_targetRegionBedFile_for_runType() ALLOW MISSING targetRegionBed for runType=%s; applicationGroupName=%s" % (runType, applicationGroupName))
                            else:
                                errors.append("%s is required for this application" % (displayedName))
                        elif value:
                            if not os.path.isfile(value):
                                errors.append("Missing or Invalid %s - %s" % (displayedName, value))
            else:
                errors.append("%s Application %s not found" % (displayedName, runType))
        else:
            errors.append("%s Run type is missing" % (displayedName))

    # logger.debug("EXIT plan_validator.validate_targetRegionBedFile_for_runType() errors=%s" %(errors))

    return errors


def validate_reference_for_runType(value, runType, applicationGroupName, displayedName='Reference'):
    errors = []
    value = value.strip() if value else ""

    if value:
        applProduct = ApplProduct.objects.filter(isActive=True, applType__runType=runType, applicationGroup__name=applicationGroupName) \
            or ApplProduct.objects.filter(isActive=True, applType__runType=runType)

        if applProduct and not applProduct[0].isReferenceSelectionSupported:
            errors.append(displayedName+" selection is not supported for this Application")

    return errors


def validate_hotspot_bed(hotSpotRegionBedFile):
    """
    validate hotSpot BED file case-insensitively with leading/trailing blanks in the input ignored
    """
    errors = []
    if hotSpotRegionBedFile:
        bedFileDict = dict_bed_hotspot()
        value = hotSpotRegionBedFile.strip()

        isValidated = False
        for bedFile in bedFileDict.get("hotspotFiles"):
            if value == bedFile.file or value == bedFile.path:
                isValidated = True
        if not isValidated:
            errors.append("%s hotSpotRegionBedFile is missing" % (hotSpotRegionBedFile))

    logger.debug("plan_validator.validate_hotspot_bed() value=%s;" % (value))

    return errors


def validate_chipBarcode(chipBarcode):
    """
    validate chip barcide is alphanumberic
    """
    errors = []
    if chipBarcode:
        value = chipBarcode.strip()

        isValidated = False
        if (value.isalnum()) or (len(value) == 0):
            isValidated = True
        if not isValidated:
            errors.append("%s is invalid.  Chip ID can only contain letters and numbers." % (value))

    logger.debug("plan_validator.validate_chipBarcode() value=%s;" % (chipBarcode))

    return errors


def get_default_planStatus():
    defaultPlanStatus = [status[1] for status in PlannedExperiment.ALLOWED_PLAN_STATUS]
    defaultPlanStatus = [status.lower() for status in defaultPlanStatus]

    return defaultPlanStatus


def validate_planStatus(planStatus):
    """
    validate planStatus is in ALLOWED_PLAN_STATUS
    """
    errors = []

    if planStatus:
        planStatus = planStatus.strip()
        isValid = False
        defaultPlanStatus = get_default_planStatus()

        for status in defaultPlanStatus:
            if planStatus.lower() == status:
                isValid = True

        if not isValid:
            defaultPlanStatus_display = ', '.join([status for status in defaultPlanStatus])
            errors.append("The plan status(%s) is not valid. Default Values are: %s" % (planStatus, defaultPlanStatus_display))

    logger.debug("plan_validator.validate_planStatus() value=%s;" % (planStatus))

    return errors
    

def validate_sampleControlType(value, displayedName="Control Type"):
    errors = []
    value = value.strip().lower() if value else ''
    if value:
        if value == 'none':
            value = ''
        else:
            controlTypes = SampleAnnotation_CV.objects.filter(annotationType='controlType')
            controlType = controlTypes.filter(value__iexact=value) or controlTypes.filter(iRValue__iexact=value)
            if controlType:
                value = controlType[0].value
            else:
                errors.append("%s %s not found" % (displayedName, value))

    return errors, value


def validate_fusions_reference(value, runType, applicationGroupName, displayedName='Fusions Reference'):
    errors = []
    value = value.strip() if value else ""

    if value:
        applProduct = ApplProduct.objects.filter(isActive=True, applType__runType=runType, applicationGroup__name=applicationGroupName) \
            or ApplProduct.objects.filter(isActive=True, applType__runType=runType)
        # need to validate only if this is a dual nuc type plan
        if applProduct and applProduct[0].isDualNucleotideTypeBySampleSupported:
            errors = validate_reference_short_name(value, displayedName)

    return errors
