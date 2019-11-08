# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from django.utils.translation import ugettext as _, ugettext_lazy
from iondb.rundb.models import (
    Chip,
    LibraryKey,
    RunType,
    KitInfo,
    common_CV,
    ApplicationGroup,
    SampleGroupType_CV,
    dnaBarcode,
    ReferenceGenome,
    ApplProduct,
    PlannedExperiment,
    SampleAnnotation_CV,
    FlowOrder,
    Content,
    Plugin,
    Sample,
)
from iondb.utils import validation
from iondb.rundb.labels import Experiment as _Experiment
from iondb.rundb.plan.views_helper import dict_bed_hotspot
import os
import re
import logging
from django.db.models import Q

from iondb.rundb.labels import PlanTemplate, ScientificApplication

logger = logging.getLogger(__name__)


MAX_LENGTH_PLAN_NAME = 60
MAX_LENGTH_SAMPLE_NAME = 127
MAX_LENGTH_SAMPLE_ID = 127
MAX_LENGTH_PROJECT_NAME = 64
MAX_LENGTH_NOTES = 1024
MAX_LENGTH_SAMPLE_TUBE_LABEL = 512
MIN_FLOWS = 1
MAX_FLOWS = 2000
MIN_QC_INT = 1
MAX_QC_INT = 100
PROJECT_NAME_LENGTH = 64
MIN_LIBRARY_READ_LENGTH = 0
MAX_LIBRARY_READ_LENGTH = 1000

# VALID_TEMPLATING_SIZES = ["200", "400"]


def validate_plan_name(value, field_label):
    errors = []
    if not validation.has_value(value):
        errors.append(validation.required_error(field_label))

    if not validation.is_valid_chars(value):
        errors.append(validation.invalid_chars_error(field_label))

    if not validation.is_valid_length(value, MAX_LENGTH_PLAN_NAME):
        errors.append(
            validation.invalid_length_error(field_label, MAX_LENGTH_PLAN_NAME, value)
        )

    return errors


def validate_notes(value, field_label=_Experiment.notes.verbose_name):
    errors = []
    if value:
        if not validation.is_valid_chars(value):
            errors.append(validation.invalid_chars_error(field_label))

        if not validation.is_valid_length(value, MAX_LENGTH_NOTES):
            errors.append(
                validation.invalid_length_error(field_label, MAX_LENGTH_NOTES, value)
            )

    return errors


def validate_sample_name(
    value,
    field_label,
    isTemplate=None,
    isTemplate_label=PlanTemplate.verbose_name,
    barcodeId=None,
    barcodeId_label=ugettext_lazy("workflow.step.kits.fields.barcodeId.label"),
):  # TODO: i18n
    errors = []
    if not value:
        if not isTemplate:
            errors.append(validation.required_error(field_label))
    else:
        if isTemplate:
            errors.append(
                validation.format(
                    ugettext_lazy("template.messages.validation.invalidsamples"),
                    {"name": isTemplate_label},
                )
            )  # "Invalid input. Sample information cannot be saved in the %(name)s"
        if barcodeId:
            errors.append(
                validation.format(
                    ugettext_lazy(
                        "plannedexperiment.messages.validation.nonbarcoded.barcodesetnotrequired"
                    ),
                    {"barcodeSetName": barcodeId_label, "barcodeSetValue": barcodeId},
                )
            )  # "Invalid input. %(barcodeSetName)s (%(barcodeSetValue)s) should not be provided for non barcoded plan"
        if not validation.is_valid_chars(value):
            errors.append(validation.invalid_chars_error(field_label))
        if not validation.is_valid_leading_chars(value):
            errors.append(validation.invalid_leading_chars(field_label))
        if not validation.is_valid_length(value, MAX_LENGTH_SAMPLE_NAME):
            errors.append(
                validation.invalid_length_error(
                    field_label, MAX_LENGTH_SAMPLE_NAME, value
                )
            )

    return errors


def validate_barcoded_sample_info(
    sampleName,
    sampleName_label,
    sampleExternalId,
    sampleExternalId_label,
    nucleotideType,
    nucleotideType_label,
    sampleReference,
    sampleReference_label,
    runType,
    applicationGroupName,
):
    errors = []
    if not validation.is_valid_chars(sampleName):
        errors.append(validation.invalid_chars_error(sampleName_label))

    if not validation.is_valid_leading_chars(sampleName):
        errors.append(validation.invalid_leading_chars(sampleName_label))

    if not validation.is_valid_length(sampleName, MAX_LENGTH_SAMPLE_NAME):
        errors.append(
            validation.invalid_length_error(
                sampleName_label, MAX_LENGTH_SAMPLE_NAME, sampleName
            )
        )

    sample_id_errors = validate_sample_id(
        sampleExternalId, field_label=sampleExternalId_label
    )
    if sample_id_errors:
        errors.extend(sample_id_errors)

    nucleotideType_errors, sample_nucleotideType = validate_sample_nucleotideType(
        nucleotideType, runType, applicationGroupName, field_label=nucleotideType_label
    )
    if nucleotideType_errors:
        errors.extend(nucleotideType_errors)

    ref_errors, ref_short_name = validate_reference(
        sampleReference,
        field_label=sampleReference_label,
        runType=runType,
        applicationGroupName=applicationGroupName,
        application_label=ScientificApplication.verbose_name,
    )
    if ref_errors:
        errors.extend(ref_errors)

    return errors, ref_short_name, sample_nucleotideType


def validate_sample_nucleotideType(value, runType, applicationGroupName, field_label):
    """
    validate nucleotide type case-insensitively with leading/trailing blanks in the input ignored
    """
    errors = []
    nucleotideType = ""
    try:
        runTypeObj = RunType.objects.filter(runType=runType)[0]
        value_from_runType = (
            runTypeObj.nucleotideType.upper() if runTypeObj.nucleotideType else ""
        )
    except Exception:
        value_from_runType = ""

    if value:
        nucleotideType = value.strip().upper()
        if nucleotideType == "FUSIONS":
            nucleotideType = "RNA"

        if value_from_runType:
            if value_from_runType == "DNA_RNA":
                valid_values = ["DNA", "RNA", "Fusions"]
            else:
                valid_values = [value_from_runType]
        else:
            valid_values = ["DNA", "RNA"]

        if not validation.is_valid_keyword(nucleotideType, valid_values):
            errors.append(validation.invalid_keyword_error(field_label, valid_values))
    else:
        # fill in nucleotideType from selected runType or applicationGroup
        if value_from_runType and value_from_runType != "DNA_RNA":
            nucleotideType = value_from_runType
        elif applicationGroupName in ["DNA", "RNA"]:
            nucleotideType = applicationGroupName

    return errors, nucleotideType


def validate_reference(
    value, field_label, runType, applicationGroupName, application_label
):
    errors = []
    ref_short_name = ""
    value = value.strip() if value else ""

    if value:
        applProduct = ApplProduct.objects.filter(
            isActive=True,
            applType__runType=runType,
            applicationGroup__name=applicationGroupName,
        ) or ApplProduct.objects.filter(isActive=True, applType__runType=runType)

        if applProduct and not applProduct[0].isReferenceSelectionSupported:
            errors.append(
                validation.invalid_invalid_value_related(
                    field_label, value, application_label
                )
            )
        else:
            selectedRefs = (
                ReferenceGenome.objects.filter(name=value)
                or ReferenceGenome.objects.filter(name__iexact=value)
                or ReferenceGenome.objects.filter(short_name=value)
                or ReferenceGenome.objects.filter(short_name__iexact=value)
            )

            if selectedRefs:
                ref_short_name = selectedRefs[0].short_name
            else:
                errors.append(validation.invalid_not_found_error(field_label, value))

    return errors, ref_short_name


def validate_reference_short_name(value, field_label):
    errors = []
    if validation.has_value(value):
        value = value.strip()
        reference = ReferenceGenome.objects.filter(short_name=value, enabled=True)
        if not reference.exists():
            generic_not_found_error = validation.invalid_not_found_error(
                field_label, value
            )
            error_fixing_message = ugettext_lazy(
                "references_genome_download.messages.help.import"
            )  # "To import it, visit Settings > References > Import Preloaded Ion References"
            reference_error_message = (
                generic_not_found_error.strip() + error_fixing_message.strip()
            )  # avoid lazy
            errors.append(reference_error_message)
    return errors


def validate_sample_id(value, field_label):
    errors = []
    if not validation.is_valid_chars(value):
        errors.append(validation.invalid_chars_error(field_label))

    if not validation.is_valid_length(value, MAX_LENGTH_SAMPLE_ID):
        errors.append(
            validation.invalid_length_error(field_label, MAX_LENGTH_SAMPLE_ID, value)
        )

    return errors


def validate_sample_tube_label(value, field_label):
    errors = []

    if value:
        if not validation.is_valid_chars(value):
            errors.append(validation.invalid_chars_error(field_label))

        if not validation.is_valid_length(value, MAX_LENGTH_SAMPLE_TUBE_LABEL):
            errors.append(
                validation.invalid_length_error(
                    field_label, MAX_LENGTH_SAMPLE_TUBE_LABEL, value
                )
            )

    return errors


def validate_chip_type(value, field_label, isNewPlan=None):
    errors = []
    warnings = []
    if not value:
        errors.append(validation.required_error(field_label))
    else:
        value = value.strip()

        query_args = (Q(name__iexact=value) | Q(description__iexact=value),)

        chip = Chip.objects.filter(*query_args)

        if chip:
            if not chip[0].isActive:
                if isNewPlan:
                    errors.append(
                        validation.invalid_not_active(
                            field_label, value, include_error_prefix=False
                        )
                    )  # 'Chip %s not active' % value
                else:
                    warnings.append(
                        validation.invalid_not_active(
                            field_label, value, include_error_prefix=False
                        )
                    )  # "Found inactive %s %s" % (field_label, value))
            else:
                if chip[0].getChipWarning:
                    warnings.append(chip[0].getChipWarning)
        else:
            errors.append(
                validation.invalid_invalid_value(field_label, value)
            )  # 'Chip %s not found' % value

    return errors, warnings


def validate_flows(value, field_label):
    errors = []
    if type(value) != int and value.isdigit():
        value2 = int(value)
    else:
        value2 = value

    if type(value2) == int:
        if not validation.is_valid_uint(value2):
            errors.append(validation.invalid_uint(field_label))
        elif value2 < MIN_FLOWS or value2 > MAX_FLOWS:
            errors.append(validation.invalid_range(field_label, MIN_FLOWS, MAX_FLOWS))
    else:
        errors.append(validation.invalid_range(field_label, MIN_FLOWS, MAX_FLOWS))

    return errors


def validate_library_key(value, field_label):
    errors = []
    selectedLibKey = None
    if not value:
        errors.append(validation.required_error(field_label))
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
            errors.append(
                validation.invalid_not_found_error(field_label, value)
            )  # 'Library key %s not found' % value

    return errors, selectedLibKey


def validate_libraryReadLength(value, field_label):
    errors = []
    if type(value) != int and value.isdigit():
        value2 = int(value)
    else:
        value2 = value

    if type(value2) == int:
        if not validation.is_valid_uint_n_zero(value2):
            errors.append(validation.invalid_uint(field_label))
        elif value2 < MIN_LIBRARY_READ_LENGTH:
            errors.append(
                validation.invalid_range(
                    field_label, MIN_LIBRARY_READ_LENGTH, MAX_LIBRARY_READ_LENGTH
                )
            )
        elif value2 > MAX_LIBRARY_READ_LENGTH:
            errors.append(
                validation.invalid_range(
                    field_label, MIN_LIBRARY_READ_LENGTH, MAX_LIBRARY_READ_LENGTH
                )
            )
    else:
        errors.append(
            validation.invalid_range(
                field_label, MIN_LIBRARY_READ_LENGTH, MAX_LIBRARY_READ_LENGTH
            )
        )

    return errors


'''
#templatingSize validation became obsolete, this field has been replaced by samplePrepProtocol
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
            errors.append(validation.invalid_keyword_error(field_label, valid_values))
    return errors
'''


def validate_QC(value, field_label):
    errors = []
    if not validation.is_valid_uint(value):
        errors.append(validation.invalid_uint(field_label))
    elif int(value) > MAX_QC_INT:
        errors.append(validation.invalid_range(field_label, MIN_QC_INT, MAX_QC_INT))

    return errors


def validate_projects(value, field_label, delim=","):
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
                    errors.append(validation.invalid_chars_error(field_label))
                if not validation.is_valid_length(trimmed_project, PROJECT_NAME_LENGTH):
                    errors.append(
                        validation.invalid_length_error(
                            field_label, PROJECT_NAME_LENGTH, trimmed_project
                        )
                    )
                if errors:
                    break

    return errors, trimmed_projects


def validate_barcode_kit_name(value, field_label):
    errors = []

    if validation.has_value(value):
        value = value.strip()
        kits = dnaBarcode.objects.filter(name=value)
        if not kits:
            errors.append(
                validation.invalid_not_found_error(field_label, value)
            )  # "%s %s not found" % (displayedName, value)

    return errors


def get_kitInfo_by_name_or_description(value, kitType=[]):
    value = value.strip()
    query_args = (Q(name__iexact=value) | Q(description__iexact=value),)
    query_kwargs = {"kitType__in": kitType} if kitType else {}
    kit = KitInfo.objects.filter(*query_args, **query_kwargs)

    return kit[0] if kit else None


def validate_optional_kit_name(value, kitType, field_label, isNewPlan=None):
    errors = []
    warnings = []
    if validation.has_value(value):
        kit = get_kitInfo_by_name_or_description(value, kitType)

        if not kit:
            errors.append(validation.invalid_not_found_error(field_label, value))
        elif kit and not kit.isActive:
            if isNewPlan:
                errors.append(validation.invalid_not_active(field_label, value))
            else:
                warnings.append(validation.invalid_not_active(field_label, value))

    return errors, warnings


def _validate_spp_value_uid(value):
    # validate if the user specified SPP matches the common_CV value or uid
    # if not, send an appropriate error message with the valid SPP values.
    common_CVs = common_CV.objects.filter(cv_type__in=["samplePrepProtocol"])

    for cv in common_CVs:
        isValid = True
        if (value.lower() == cv.value.lower()) or (value.lower() == cv.uid.lower()):
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
        kit = KitInfo.objects.get(
            kitType__in=["TemplatingKit", "IonChefPrepKit"], name=templatingKitName
        )
        allowedSpp = kit.categories
        for value in samplePrepValue.split(";"):
            if value in kit.categories:
                isValid = True
                return isValid, validSPP_tempKit
    except Exception as err:
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
        except Exception as err:
            logger.debug(
                "plan_validator._validate_ssp_templatingKit() : Error, %s" % err
            )
        return isValid, validSPP_tempKit


def validate_plan_samplePrepProtocol(
    value, templatingKitName, field_label, templatingKitName_label
):
    # validate if input matches the common_CV value or uid
    errors = []
    if value:
        value = value.strip()
        (isValid, cv, cvLists) = _validate_spp_value_uid(value)
        if not isValid:
            cvLists = [cvlist.value for cvlist in cvLists]
            cvLists.append("undefined")
            validation.invalid_choice(field_label, value, cvLists)
            errors.append(validation.invalid_choice(field_label, value, cvLists))
            logger.debug(
                "plan_validator.validate_plan_samplePrepProtocol() : Error, %s is not valid. Valid sample kit protocols are: %s"
                % (value, cvLists)
            )
        else:
            # valid spp but still validate if it is compatible with the specified templatingKit
            isValid, validSPP_tempKit = _validate_ssp_templatingKit(
                cv.categories, templatingKitName, cvLists
            )
            if not isValid:
                validSPP_tempKit.append("undefined")
                errors.append(
                    validation.invalid_choice_related_choice(
                        field_label,
                        value,
                        validSPP_tempKit,
                        templatingKitName_label,
                        templatingKitName,
                    )
                )  # '%s not supported for the specified templatingKit %s. Valid sample prep protocols are: %s ' % (value, templatingKitName, ", ".join(validSPP_tempKit))
                logger.debug(
                    "plan_validator.validate_plan_samplePrepProtocol() : Error,%s %s not supported for this templatingKit %s "
                    % (field_label, value, templatingKitName)
                )
            value = cv.value
    return errors, value


def validate_plan_templating_kit_name(value, field_label, isNewPlan=None):
    errors = []
    warnings = []

    if not validation.has_value(value):
        errors.append(validation.required_error(field_label))
    else:
        value = value.strip()
        query_kwargs = {"kitType__in": ["TemplatingKit", "IonChefPrepKit"]}

        query_args = (Q(name=value) | Q(description=value),)
        kit = KitInfo.objects.filter(*query_args, **query_kwargs)

        if not kit:
            errors.append(validation.invalid_not_found_error(field_label, value))
        elif kit and not kit[0].isActive:
            if isNewPlan:
                errors.append(validation.invalid_not_active(field_label, value))
            else:
                warnings.append(validation.invalid_not_active(field_label, value))

    return errors, warnings


def validate_runType(runType, field_label):
    errors = []
    if runType:
        runTypeObjs = RunType.objects.filter(runType__iexact=runType)
        if not runTypeObjs:
            validRunTypes = RunType.objects.values_list("runType", flat=True)
            errors.append(
                validation.invalid_choice(field_label, runType, validRunTypes)
            )
    return errors


def validate_application_group_for_runType(value, field_label, runType, runType_label):
    errors = []

    if value:
        value = value.strip()
        applicationGroups = ApplicationGroup.objects.filter(
            name__iexact=value
        ) | ApplicationGroup.objects.filter(description__iexact=value)
        if applicationGroups:
            applicationGroup = applicationGroups[0]
            if runType:
                runTypeObjs = RunType.objects.filter(runType__iexact=runType)
                if runTypeObjs:
                    associations = runTypeObjs[0].applicationGroups
                    if not associations.filter(name=applicationGroup.name):
                        errors.append(
                            validation.invalid_choice_related_choice(
                                field_label,
                                value,
                                associations.values_list("name", flat=True),
                                runType_label,
                                runType,
                            )
                        )
            if not runType:
                errors.append(validation.missing_error(runType_label))
            if not runTypeObjs:
                errors.append(
                    validation.invalid_not_found_error(runType_label, runType)
                )
        else:
            errors.append(validation.invalid_not_found_error(field_label, value))

    return errors


def validate_sample_grouping(value, field_label):
    errors = []

    if value:
        value = value.strip()

        groupings = SampleGroupType_CV.objects.filter(displayedName__iexact=value)
        if not groupings:
            errors.append(validation.invalid_not_found_error(field_label, value))

    return errors


def validate_barcode_sample_association(selectedBarcodes, selectedBarcodeKit):
    errors = {"MISSING_BARCODE": "", "DUPLICATE_BARCODE": ""}

    if not selectedBarcodeKit:
        return errors

    prior_barcodes = []

    if not selectedBarcodes:
        errors["MISSING_BARCODE"] = _(
            "workflow.step.sample.messages.validate.barcodepersample"
        )  # "Please select a barcode for each sample"
    else:
        dupBarcodes = validation.list_duplicates(selectedBarcodes)
        for selectedBarcode in dupBarcodes:
            # only include unique barcode selection error messages
            message = validation.format(
                _("workflow.step.sample.messages.validate.barcode.unique"),
                {"barcode": selectedBarcode},
            )  # "Barcode %s selections have to be unique\n" % selectedBarcode
            errors["DUPLICATE_BARCODE"] = errors["DUPLICATE_BARCODE"] + message + "\n"

    # logger.debug("errors=%s" %(errors))

    return errors


def validate_targetRegionBedFile_for_runType(
    value,
    field_label,
    runType,
    reference,
    nucleotideType=None,
    applicationGroupName=None,
    isPrimaryTargetRegion=True,
    barcodeId="",
    runType_label=ugettext_lazy("workflow.step.application.fields.runType.label"),
):
    """
    validate targetRegionBedFile based on the selected reference and the plan's runType
    """
    errors = []
    value = value.strip() if value else ""
    if value:
        missing_file = check_uploaded_files(bedfilePaths=[value])
        if missing_file:
            errors.append("%s : %s not found" % (field_label, value))
            logger.debug(
                "plan_validator.validate_targetRegionBedFile_for_run() SKIPS validation due to no targetRegion file exists in db. value=%s"
                % (value)
            )
        return errors

    logger.debug(
        "plan_validator.validate_targetRegionBedFile_for_runType() value=%s; runType=%s; reference=%s; nucleotideType=%s; applicationGroupName=%s"
        % (value, runType, reference, nucleotideType, applicationGroupName)
    )

    if not isPrimaryTargetRegion:
        logger.debug(
            "plan_validator.validate_targetRegionBedFile_for_run() SKIPS validation due to no validation rules for non-primary targetRegion. value=%s"
            % (value)
        )
        return errors

    if reference:
        if runType:
            runType = runType.strip()
            applProducts = ApplProduct.objects.filter(
                isActive=True,
                applType__runType=runType,
                applicationGroup__name=applicationGroupName,
            ) or ApplProduct.objects.filter(isActive=True, applType__runType=runType)
            if applProducts:
                applProduct = applProducts[0]
                if applProduct:
                    if (
                        validation.has_value(value)
                        and not applProduct.isTargetRegionBEDFileSupported
                    ):
                        errors.append(
                            validation.invalid_invalid_related(
                                field_label, ScientificApplication.verbose_name
                            )
                        )
                    else:
                        isRequired = (
                            applProduct.isTargetRegionBEDFileSelectionRequiredForRefSelection
                        )
                        if (
                            isRequired
                            and not validation.has_value(value)
                            and not barcodeId
                        ):
                            # skip for now
                            if (
                                runType in ["AMPS_DNA_RNA", "AMPS_HD_DNA_RNA"]
                                and nucleotideType
                                and nucleotideType.upper() == "RNA"
                            ):
                                logger.debug(
                                    "plan_validator.validate_targetRegionBedFile_for_runType() ALLOW MISSING targetRegionBed for runType=%s; nucleotideType=%s"
                                    % (runType, nucleotideType)
                                )
                            elif runType in ["AMPS_RNA", "AMPS_HD_RNA"]:
                                logger.debug(
                                    "plan_validator.validate_targetRegionBedFile_for_runType() ALLOW MISSING targetRegionBed for runType=%s; applicationGroupName=%s"
                                    % (runType, applicationGroupName)
                                )
                            else:
                                errors.append(
                                    validation.invalid_required_related(
                                        field_label, ScientificApplication.verbose_name
                                    )
                                )
                        elif value:
                            if not os.path.isfile(value):
                                errors.append(
                                    validation.invalid_invalid_value(field_label, value)
                                )
            else:
                errors.append(
                    validation.invalid_invalid_value_related(
                        runType_label, runType, ScientificApplication.verbose_name
                    )
                )
        else:
            errors.append(
                validation.invalid_required_related(
                    runType_label, ScientificApplication.verbose_name
                )
            )

    return errors


def validate_reference_for_runType(
    value, field_label, runType, applicationGroupName, application_label
):
    errors = []
    value = value.strip() if value else ""

    if value:
        applProduct = ApplProduct.objects.filter(
            isActive=True,
            applType__runType=runType,
            applicationGroup__name=applicationGroupName,
        ) or ApplProduct.objects.filter(isActive=True, applType__runType=runType)

        if applProduct and not applProduct[0].isReferenceSelectionSupported:
            errors.append(
                validation.invalid_invalid_value_related(
                    field_label, value, application_label
                )
            )

    return errors


def validate_hotspot_bed(hotSpotRegionBedFile, field_label):
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
            errors.append(
                validation.invalid_invalid_value(field_label, hotSpotRegionBedFile)
            )  # "%s hotSpotRegionBedFile is missing" % (hotSpotRegionBedFile))

    logger.debug("plan_validator.validate_hotspot_bed() value=%s;" % (value))

    return errors


def validate_chipBarcode(chipBarcode, field_label):
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
            errors.append(
                validation.invalid_alphanum(field_label, chipBarcode)
            )  # "%(value)s is invalid. %(displayedName)s can only contain letters and numbers." % {'value': value, 'displayedName': field_label}

    logger.debug("plan_validator.validate_chipBarcode() value=%s;" % (chipBarcode))

    return errors


def get_default_planStatus():
    defaultPlanStatus = [status[1] for status in PlannedExperiment.ALLOWED_PLAN_STATUS]
    defaultPlanStatus = [status.lower() for status in defaultPlanStatus]

    return defaultPlanStatus


def validate_planStatus(planStatus, field_label):
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
            errors.append(
                validation.invalid_choice(field_label, planStatus, defaultPlanStatus)
            )  # "The plan status(%s) is not valid. Default Values are: %s" % (planStatus, defaultPlanStatus_display))

    logger.debug("plan_validator.validate_planStatus() value=%s;" % (planStatus))

    return errors


def validate_sampleControlType(value, field_label):
    errors = []
    value = value.strip().lower() if value else ""
    if value:
        if value == "none":
            value = ""
        else:
            controlTypes = SampleAnnotation_CV.objects.filter(
                annotationType="controlType"
            )
            controlType = controlTypes.filter(
                value__iexact=value
            ) or controlTypes.filter(iRValue__iexact=value)
            if controlType:
                value = controlType[0].value
            else:
                choices = controlTypes.order_by("value").values_list("value", flat=True)
                errors.append(validation.invalid_choice(field_label, value, choices))

    return errors, value


def validate_16s_markers(value, field_label):
    errors = []
    if not value or value == "none":
        value = ""
    else:
        value = value.strip().lower() if value else ""
        annotations = SampleAnnotation_CV.objects.filter(
            annotationType="16s_markers"
        )
        annotationObj = annotations.filter(value__iexact=value) or annotations.filter(iRValue__iexact=value)
        if annotationObj:
            value = annotationObj[0].value
        else:
            choices = annotations.order_by("value").values_list("value", flat=True)
            errors.append(validation.invalid_choice(field_label, value, choices))

    return errors, value


def validate_reference_for_fusions(
    value, field_label, runType, applicationGroupName, application_label
):
    errors = []
    value = value.strip() if value else ""

    if value:
        applProduct = ApplProduct.objects.filter(
            isActive=True,
            applType__runType=runType,
            applicationGroup__name=applicationGroupName,
        ) or ApplProduct.objects.filter(isActive=True, applType__runType=runType)
        # need to validate only if this is a dual nuc type plan
        if applProduct and applProduct[0].isDualNucleotideTypeBySampleSupported:
            errors = validate_reference_short_name(
                value, field_label=field_label
            )  # TODO: Why does this only support the shortname?

    return errors


def validate_flowOrder(value, field_label):
    errors = []
    value = value.strip() if value else ""
    if value:
        try:
            selectedflowOrder = FlowOrder.objects.get(name__iexact=value)
            logger.debug(
                "plan_validator.validate_flowOrder...%s: flow-order exists in the DB table"
                % value
            )
        except FlowOrder.DoesNotExist:
            try:
                selectedflowOrder = FlowOrder.objects.get(description__iexact=value)
                logger.debug(
                    "plan_validator.validate_flowOrder...%s: flow-order exists in the DB table"
                    % value
                )
            except FlowOrder.DoesNotExist:
                try:
                    selectedflowOrder = FlowOrder.objects.get(flowOrder__iexact=value)
                    logger.debug(
                        "plan_validator.validate_flowOrder...%s: flow-order exists in the DB table"
                        % value
                    )
                except FlowOrder.DoesNotExist:
                    try:
                        logger.debug(
                            "plan_validator.validate_flowOrder...%s: arbitrary flow-order is specified by the user"
                            % value
                        )
                        flowOrderPattern = "[^ACTG]+"
                        if re.findall(flowOrderPattern, value):
                            selectedflowOrder = None
                        else:
                            selectedflowOrder = value
                    except Exception:
                        logger.debug(
                            "plan_validator.validate_flowOrder ...%s not found" % value
                        )

        if not selectedflowOrder:
            errors.append(
                validation.invalid_nucleotide(field_label, value)
            )  # '%s %s is not valid' % (field_label, value)
        elif selectedflowOrder and isinstance(selectedflowOrder, FlowOrder):
            selectedflowOrder = selectedflowOrder.flowOrder

    return errors, selectedflowOrder


def check_uploaded_files(
    referenceNames=[], bedfilePaths=[], referenceNames_label="Reference"
):
    """ checks if reference or BED files are missing
    referenceNames_label='Reference'
    """
    missing_files = {}
    for reference in referenceNames:
        ref_err = validate_reference_short_name(
            reference, field_label=referenceNames_label
        )
        if ref_err:
            missing_files.setdefault("references", []).append(reference)

    content = Content.objects.filter(publisher__name="BED")
    for bedfile in bedfilePaths:
        bedfile_err = validation.has_value(bedfile) and (
            content.filter(path=bedfile).count() == 0
            and content.filter(file=bedfile).count() == 0
        )
        if bedfile_err:
            missing_files.setdefault("bedfiles", []).append(bedfile)

    return missing_files


# Validate for supported kit/chip combination when created via API
def validate_kit_chip_combination(
    bundle,
    chipType_label="chipType",
    templatingKitName_label="templatingKitName",
    sequencekitname_label="sequencekitname",
    librarykitname_label="librarykitname",
    runType_label="runType",
):
    errorMsg = None
    chipType = bundle.data.get("chipType", None)
    runType = bundle.data.get("runType", None)
    templatingKitName = bundle.data.get("templatingKitName", None)
    sequencekitname = bundle.data.get("sequencekitname", None)
    librarykitname = bundle.data.get("librarykitname", None)
    planExp_Kits = [
        (templatingKitName, templatingKitName_label),
        (sequencekitname, sequencekitname_label),
        (librarykitname, librarykitname_label),
    ]

    # since these kits are already been validated, just concentrate on validating the chip/instrument/kits combination
    try:
        if chipType:
            query_args = (Q(name__iexact=chipType) | Q(description__iexact=chipType),)
            selectedChips = Chip.objects.filter(*query_args)

            if selectedChips:
                selectedChip = selectedChips[0]
                for kit_name, kit_label in planExp_Kits:
                    if not kit_name:
                        continue

                    selectedKitInfos = KitInfo.objects.filter(
                        kitType__in=[
                            "TemplatingKit",
                            "IonChefPrepKit",
                            "SequencingKit",
                            "LibraryKit",
                            "LibraryPrepKit",
                        ],
                        name__iexact=kit_name,
                    )
                    if not selectedKitInfos.count():
                        continue

                    selectedKitInfo = selectedKitInfos[0]
                    #
                    # validate selected chip and kit instrument type combination is supported
                    #
                    selectedChip_instType = selectedChip.instrumentType
                    selectedKit_instType = selectedKitInfo.instrumentType
                    if (
                        selectedKit_instType
                        and selectedKit_instType not in selectedChip_instType
                    ):  # BUG: 'proton;S5' not in 'S5' reports True... find alternate way to validate...
                        # The %(kitLabel)s (%(kitName)s) instrument type of %(kitInstrumentType)s is
                        # incompatible with the %(chipLabel)s (%(chipName)s) instrument type of
                        # %(chipInstrumentType)s. Specify a different %(kitLabel)s or %(chipLabel)s.
                        errorMsg = validation.format(
                            ugettext_lazy(
                                "plannedexperiment.messages.validate.invalid.kitandchip.instrumenttype"
                            ),
                            {
                                "kitLabel": kit_label,
                                "kitName": selectedKitInfo.name,
                                "kitInstrumentType": selectedKitInfo.get_instrument_types_list(),
                                "chipLabel": chipType_label,
                                "chipName": chipType,
                                "chipInstrumentType": selectedChip_instType,
                            },
                        )
                        return errorMsg

                    #
                    # if instrument type is valid: validate if chip type of selected kit is in the supported list
                    #
                    if selectedKitInfo.chipTypes:
                        selectedKit_chipTypes = selectedKitInfo.get_chip_types_list()
                        if (
                            selectedKit_chipTypes
                            and chipType not in selectedKit_chipTypes
                        ):
                            errorMsg = validation.invalid_choice_related_choice(
                                chipType_label,
                                chipType,
                                selectedKit_chipTypes,
                                kit_label,
                                selectedKitInfo.name,
                            )
                            return errorMsg
                    #
                    # if instrument type and chip type are valid: validate if application type of selected kit is in the supported list
                    #
                    if (
                        runType
                        and selectedKitInfo.applicationType
                        and runType not in selectedKitInfo.get_kit_application_list()
                    ):
                        errorMsg = validation.invalid_invalid_value_related_value(
                            kit_label,
                            selectedKitInfo.name
                            + "[%s]" % selectedKitInfo.get_applicationType_display(),
                            runType_label,
                            runType,
                        )
                        return errorMsg

    except Exception as Err:
        logger.debug("Error during plan creation %s" % str(Err))
        errorMsg = str(Err)

    return errorMsg


def validate_plugin_configurations(selected_plugins):
    """this will validate all of the plugins as part of the plan.  It returns a list of all of the validation error messages"""
    validation_messages = list()
    if not selected_plugins:
        return validation_messages
    for name, plugin_parameters in list(selected_plugins.items()):
        try:
            configuration = plugin_parameters.get("userInput", {}) or {}
            plugin_model = Plugin.objects.get(
                name=plugin_parameters["name"], active=True
            )
            if plugin_model.requires_configuration:
                validation_messages += Plugin.validate(
                    plugin_model.id, configuration
                )
        except Exception as exc:
            validation_messages += [str(exc)]
    return validation_messages
