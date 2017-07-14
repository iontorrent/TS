# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb.models import Chip, LibraryKey, RunType, KitInfo, common_CV, ApplicationGroup, \
    SampleGroupType_CV, dnaBarcode, ReferenceGenome, ApplProduct, PlannedExperiment, \
    SampleAnnotation_CV, FlowOrder, Content
from iondb.utils import validation
from iondb.rundb.plan.views_helper import dict_bed_hotspot
import os
import re
import logging
from django.db.models import Q
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
    if value:
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


def validate_chip_type(value, displayedName='Chip Type', isNewPlan=None):
    errors = []
    warnings = []
    if not value:
        errors.append(validation.required_error(displayedName))
    else:
        value = value.strip()

        query_args = (Q(name__iexact=value) | Q(description__iexact=value),)

        chip = Chip.objects.filter(*query_args)

        if not chip:
            errors.append('Chip %s not found' % value)
        elif chip and not chip[0].isActive:
            if isNewPlan:
                errors.append('Chip %s not active' % value)
            else:
                warnings.append("Found inactive %s %s" % (displayedName, value))

    return errors, warnings


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


def validate_sequencing_kit_name(value, displayedName="Sequencing Kit", isNewPlan=None):
    errors = []
    warnings = []
    if validation.has_value(value):
        value = value.strip()

        query_kwargs = {"kitType__in": ["SequencingKit"]}
        query_args = (Q(name__iexact=value) | Q(description__iexact=value),)

        kit = KitInfo.objects.filter(*query_args, **query_kwargs)

        if not kit:
            errors.append("%s %s not found" % (displayedName, value))
        elif kit and not kit[0].isActive:
            if isNewPlan:
                errors.append("%s %s not active" % (displayedName, value))
            else:
                warnings.append("Found inactive %s %s" % (displayedName, value))

    return errors, warnings

def _validate_spp_value_uid(value):
    # validate if the user specified SPP matches the common_CV value or uid
    # if not, send an appropriate error message with the valid SPP values.
    common_CVs = common_CV.objects.filter(cv_type__in=["samplePrepProtocol"])

    for cv in common_CVs:
        isValid = True
        if ((value.lower() == cv.value.lower()) or
                (value.lower() == cv.uid.lower())):
            return (isValid, cv, common_CVs)
        else:
            isValid = False
    return (isValid, value, common_CVs)

def _validate_ssp_templatingKit(samplePrepValue, templatingKitName, cvLists):
    # validate whether the samplePrepProtocol is supported for this templatingKit
    isValid = False
    validSPP_tempKit = []
    allowedSpp = []
    try:
        # check if the user specified spp's category is in the Templating Kit categories lists
        # If not, send an appropriate error message with the valid samplePrep Protocol values
        # which is supported by the specific templatingKit
        kit = KitInfo.objects.get(kitType__in=["TemplatingKit", "IonChefPrepKit"], name=templatingKitName)
        allowedSpp = kit.categories
        for value in samplePrepValue.split(";"):
            if value in kit.categories:
                isValid = True
                return isValid, validSPP_tempKit
    except Exception,err:
        logger.debug("plan_validator._validate_ssp_templatingKit() : Error, %s" % err)
    if not isValid:
        try:
            if "samplePrepProtocol" in allowedSpp:
                for category in allowedSpp.split(";"):
                    if category == "samplePrepProtocol":
                        continue
                    if "sampleprep" in category.lower():
                        for cvlist in cvLists:
                            if category in cvlist.categories:
                                validSPP_tempKit.append(cvlist.value)
        except Exception,err:
            logger.debug("plan_validator._validate_ssp_templatingKit() : Error, %s" % err)
        return isValid, validSPP_tempKit

def validate_plan_samplePrepProtocol(value, templatingKitName, displayedName="sample Prep Protocol"):
    # validate if input matches the common_CV value or uid
    errors = []
    if value:
        value = value.strip()
        (isValid, cv, cvLists) = _validate_spp_value_uid(value)
        if not isValid:
            cvLists = [cvlist.value for cvlist in cvLists]
            cvLists.append("undefined")
            errors.append("%s is not valid. Valid sample prep protocols are: %s" % (value, ", ".join(cvLists)))
            logger.debug("plan_validator.validate_plan_samplePrepProtocol() : Error, %s is not valid. Valid sample kit protocols are: %s" % (value, cvLists))
        else:
            # valid spp but still validate if it is compatible with the specified templatingKit
            isValid, validSPP_tempKit = _validate_ssp_templatingKit(cv.categories, templatingKitName, cvLists)
            if not isValid:
                validSPP_tempKit.append("undefined")
                errors.append('%s not supported for the specified templatingKit %s. Valid sample prep protocols are: %s ' % (value, templatingKitName, ", ".join(validSPP_tempKit)))
                logger.debug("plan_validator.validate_plan_samplePrepProtocol() : Error,%s %s not supported for this templatingKit %s " % (displayedName, value, templatingKitName))
            value = cv.value
    return errors, value

def validate_plan_templating_kit_name(value, displayedName="Template Kit", isNewPlan=None):
    errors = []
    warnings = []

    if not validation.has_value(value):
        errors.append(validation.required_error(displayedName))
    else:
        value = value.strip()
        query_kwargs = {"kitType__in": ["TemplatingKit", "IonChefPrepKit"]}

        query_args = (Q(name=value) | Q(description=value), )
        kit = KitInfo.objects.filter(*query_args, **query_kwargs)

        if not kit:
            errors.append("%s %s not found" % (displayedName, value))
        elif kit and not kit[0].isActive:
            if isNewPlan:
                errors.append("%s %s not active" % (displayedName, value))
            else:
                warnings.append("Found inactive %s %s" % (displayedName, value))

    return errors, warnings


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
    validate chip barcode is alphanumberic
    """
    errors = []
    if chipBarcode:
        value = chipBarcode.strip()

        isValidated = False
        if (value.isalnum()) or (len(value) == 0):
            isValidated = True
        if not isValidated:
            errors.append("%s is invalid.  Chip barcode can only contain letters and numbers." % (value))

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

def validate_flowOrder(value, displayedName='Flow Order'):
    errors = []
    value = value.strip() if value else ''
    if value:
        try:
            selectedflowOrder = FlowOrder.objects.get(name__iexact=value)
            logger.debug("plan_validator.validate_flowOrder...%s: flow-order exists in the DB table" % value)
        except FlowOrder.DoesNotExist:
            try:
                selectedflowOrder = FlowOrder.objects.get(description__iexact=value)
                logger.debug("plan_validator.validate_flowOrder...%s: flow-order exists in the DB table" % value)
            except FlowOrder.DoesNotExist:
                try:
                    selectedflowOrder = FlowOrder.objects.get(flowOrder__iexact=value)
                    logger.debug("plan_validator.validate_flowOrder...%s: flow-order exists in the DB table" % value)
                except FlowOrder.DoesNotExist:
                    try:
                        logger.debug("plan_validator.validate_flowOrder...%s: arbitrary flow-order is specified by the user" % value)
                        flowOrderPattern = "[^ACTG]+"
                        if re.findall(flowOrderPattern, value):
                            selectedflowOrder = None
                        else:
                            selectedflowOrder = value
                    except:
                        logger.debug("plan_validator.validate_flowOrder ...%s not found" % value)

        if not selectedflowOrder:
            errors.append('%s %s is not valid' % (displayedName, value))
        elif selectedflowOrder and isinstance(selectedflowOrder, FlowOrder):
            selectedflowOrder = selectedflowOrder.flowOrder

    return errors, selectedflowOrder


def check_uploaded_files(referenceNames=[], bedfilePaths=[]):
    ''' checks if reference or BED files are missing '''
    missing_files = {}
    for reference in referenceNames:
        ref_err = validate_reference_short_name(reference)
        if ref_err:
            missing_files.setdefault('references',[]).append(reference)
    
    content = Content.objects.filter(publisher__name="BED")
    for bedfile in bedfilePaths:
        bedfile_err = validation.has_value(bedfile) and content.filter(path=bedfile).count() == 0
        if bedfile_err:
            missing_files.setdefault('bedfiles',[]).append(bedfile)

    return missing_files


# Validate for supported kit/chip combination when created via API
def validate_kit_chip_combination(bundle):
    errorMsg = None
    chipType = bundle.data.get("chipType",None)
    runType = bundle.data.get("runType",None)
    templatingKitName = bundle.data.get("templatingKitName",None)
    sequencekitname = bundle.data.get("sequencekitname",None)
    librarykitname = bundle.data.get("librarykitname",None)
    planExp_Kits = [templatingKitName, sequencekitname, librarykitname]

    # since these kits are already been validated, just concentrate on validating the chip/instrument/kits combination
    try:
        if chipType:
            query_args = (Q(name__iexact=chipType) | Q(description__iexact=chipType),)
            selectedChips = Chip.objects.filter(*query_args)

            if selectedChips:
                selectedChip = selectedChips[0]
                for kit in planExp_Kits:
                    if kit:
                        selectedKits = KitInfo.objects.filter(kitType__in=["TemplatingKit", "IonChefPrepKit", "SequencingKit", "LibraryKit", "LibraryPrepKit"], name__iexact=kit)
                        if selectedKits:
                            selectedKit = selectedKits[0]
                            # validate selected chip and kit instrument type combination is supported
                            selectedChip_instType = selectedChip.instrumentType
                            selectedKit_instType = selectedKit.instrumentType
                            if selectedKit_instType:
                                if selectedKit_instType not in selectedChip_instType:
                                    errorMsg = "specified Kit (%s) / Chip (%s) instrument type is not supported" % (selectedKit.name, chipType)
                                    return errorMsg

                            # if instrument type is valid: validate if chip type of selected kit is in the supported list
                            if selectedKit.chipTypes:
                                selectedKit_chipTypes = selectedKit.chipTypes.split(";")
                                if selectedKit_chipTypes:
                                    if chipType not in selectedKit_chipTypes:
                                        errorMsg = "specified Kit (%s) / Chip (%s) combination is not supported" % (selectedKit.name, chipType)
                                        return errorMsg

                            # if instrument type and chip type are valid: validate if application type of selected kit is in the supported list
                            selectedKit_applicationType = selectedKit.applicationType
                            if selectedKit_applicationType:
                                if "AMPS_ANY" in selectedKit_applicationType:
                                    selectedKit_applicationType = ['AMPS', 'AMPS_DNA_RNA', 'AMPS_EXOME', 'AMPS_RNA']
                                if runType not in selectedKit_applicationType:
                                    errorMsg = "specified Kit (%s) is not supported for %s (runType=%s)" % (selectedKit.name, selectedKit.get_applicationType_display(), runType)
                                    return errorMsg
    except Exception, Err:
        logger.debug("Error during plan creation %s" % str(Err))
        errorMsg = str(Err)

    return errorMsg
