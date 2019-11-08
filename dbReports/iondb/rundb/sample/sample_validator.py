# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from django.shortcuts import get_object_or_404
from django.utils.translation import ugettext_lazy

from iondb.rundb import models
import types
import datetime
import time
import logging

import re

from iondb.rundb.models import (
    SampleGroupType_CV,
    SampleAnnotation_CV,
    SampleSet,
    SampleSetItem,
    Sample,
    SampleAttribute,
    SampleAttributeValue,
)
from iondb.rundb.labels import (
    Sample as _Sample,
    SampleAttribute as _SampleAttribute,
    SampleSet as _SampleSet,
    SampleSetItem as _SampleSetItem,
)
from iondb.utils import validation
from iondb.rundb.plan.plan_validator import (
    validate_sampleControlType,
    validate_optional_kit_name,
)
from iondb.utils.verify_types import (
    RepresentsInt,
    RepresentsUnsignedInt,
    RepresentsUnsignedIntOrZero,
)

logger = logging.getLogger(__name__)

MAX_LENGTH_SAMPLE_NAME = 127
MAX_LENGTH_SAMPLE_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_EXTERNAL_ID = 127
MAX_LENGTH_SAMPLE_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_SET_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_SET_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_ATTRIBUTE_DISPLAYED_NAME = 127
MAX_LENGTH_SAMPLE_ATTRIBUTE_DESCRIPTION = 1024

MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE = SampleAttributeValue._meta.get_field(
    "value"
).max_length

MAX_LENGTH_PCR_PLATE_SERIAL_NUM = 64
MAX_LENGTH_SAMPLE_CELL_NUM = SampleSetItem._meta.get_field("cellNum").max_length
MAX_LENGTH_SAMPLE_COUPLE_ID = SampleSetItem._meta.get_field("coupleId").max_length
MAX_LENGTH_SAMPLE_EMBRYO_ID = SampleSetItem._meta.get_field("cellNum").max_length
SC_DATE_FORMAT = "YYYY/MM/DD"
ASHD_SAMPLESET_ITEM_LIMIT = 8

VALID_LIB_TYPES = ["amps_on_chef_v1", "amps_hd_on_chef_v1"] # some validations are applicable only for these types


def validate_sampleSet(queryDict, isNew=False, sampleset_label=_SampleSet.verbose_name):
    """
    validate the sampleSet input.
    returns a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length will not be validated since maxLength has been specified in the form.
    """

    logger.debug("sample_validator.validate_sampleset() queryDict=%s" % (queryDict))

    if not queryDict:
        return (
            False,
            validation.invalid_no_data(sampleset_label, include_error_prefix=True),
        )

    sampleSetName = queryDict.get("sampleSetName", "").strip()
    sampleSetDesc = queryDict.get("sampleSetDescription", "").strip()
    pcrPlateSerialNum = queryDict.get("pcrPlateSerialNum", "").strip()
    sampleGroupTypeName = queryDict.get("sampleGroupTypeName", "").strip()
    additionalCycles = queryDict.get("additionalCycles", "").strip()
    cyclingProtocols = queryDict.get("cyclingProtocols", "").strip()

    return validate_sampleSet_values(
        sampleSetName,
        sampleSetDesc,
        pcrPlateSerialNum,
        isNew,
        sampleGroupTypeName,
        additionalCycles,
        cyclingProtocols,
    )


def validate_sampleSet_values(
    sampleSetName,
    sampleSetDesc,
    pcrPlateSerialNum,
    isNew=False,
    sampleGroupTypeName=None,
    additionalCycles=None,
    cyclingProtocols=None,
):
    """
    validate the sampleSet input.
    returns a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length willl not be validated since maxLength has been specified in the form.
    """
    if not validation.has_value(sampleSetName):
        return (
            False,
            validation.required_error(
                ugettext_lazy("samplesets.fields.displayedName.label"),
                include_error_prefix=True,
            ),
        )  # "Sample set name"
    else:
        if not validation.is_valid_chars(sampleSetName):
            return (
                False,
                validation.invalid_chars_error(
                    ugettext_lazy("samplesets.fields.displayedName.label"),
                    include_error_prefix=True,
                ),
            )

        if not validation.is_valid_length(
            sampleSetName, MAX_LENGTH_SAMPLE_SET_DISPLAYED_NAME
        ):
            errorMessage = validation.invalid_length_error(
                ugettext_lazy("samplesets.fields.displayedName.label"),
                MAX_LENGTH_SAMPLE_SET_DISPLAYED_NAME,
                sampleSetName.strip(),
                include_error_prefix=True,
            )
            return False, errorMessage

        if isNew:
            # error if new sample set already exists
            existingSampleSets = SampleSet.objects.filter(displayedName=sampleSetName)
            if existingSampleSets:
                errorMessage = validation.invalid_entity_field_unique(
                    _SampleSet.verbose_name,
                    ugettext_lazy("samplesets.fields.displayedName.label"),
                    include_error_prefix=True,
                )
                return False, errorMessage

    if validation.has_value(sampleSetDesc):
        if not validation.is_valid_chars(sampleSetDesc):
            return (
                False,
                validation.invalid_chars_error(
                    ugettext_lazy("samplesets.fields.description.label"),
                    include_error_prefix=True,
                ),
            )

        if not validation.is_valid_length(
            sampleSetDesc, MAX_LENGTH_SAMPLE_SET_DESCRIPTION
        ):
            errorMessage = validation.invalid_length_error(
                ugettext_lazy("samplesets.fields.description.label"),
                MAX_LENGTH_SAMPLE_SET_DESCRIPTION,
                sampleSetDesc.strip(),
                include_error_prefix=True,
            )
            return False, errorMessage

    if validation.has_value(pcrPlateSerialNum):
        if not validation.is_valid_chars(pcrPlateSerialNum):
            return (
                False,
                validation.invalid_chars_error(
                    ugettext_lazy("samplesets.fields.pcrPlateSerialNum.label"),
                    include_error_prefix=True,
                ),
            )  # Sample set PCR plate serial number

        if not validation.is_valid_length(
            pcrPlateSerialNum, MAX_LENGTH_PCR_PLATE_SERIAL_NUM
        ):
            errorMessage = validation.invalid_length_error(
                ugettext_lazy("samplesets.fields.pcrPlateSerialNum.label"),
                MAX_LENGTH_PCR_PLATE_SERIAL_NUM,
                pcrPlateSerialNum.strip(),
                include_error_prefix=True,
            )  # Sample PCR plate serial number
            return False, errorMessage

    if validation.has_value(sampleGroupTypeName):
        isValid, errorMessage = validate_samplesetGroupType(
            sampleGroupTypeName, field_label="sampleGroupTypeName"
        )
        if not isValid:
            return isValid, errorMessage

    if validation.has_value(additionalCycles):

        isValid, errorMessage, cvValue = validate_additionalCycles(
            additionalCycles, field_label="additionalCycles"
        )
        if not isValid:
            return isValid, errorMessage

    if validation.has_value(cyclingProtocols):
        isValid, errorMessage, cvValue = validate_cyclingProtocols(
            cyclingProtocols, field_label="cyclingProtocols"
        )
        if not isValid:
            return isValid, errorMessage

    return True, None


def validate_barcoding_samplesetitems(
    samplesetitems,
    barcodeKit,
    barcode,
    samplesetitem_id,
    allPcrPlates=None,
    pcrPlateRow=None,
):
    """
        1) validate only one barcodeKit is used for all barcode items, no barcode seletion is ok
        2) validate barcode selected for current SampleSetItem (given by samplesetitem_id) is unique or None
        3) special logic when this function is used by Chef validation (pcrPlateRow is specified)

        If validating with a dict of samplesetitems, "pending_id" key must contain a unique id number.
    """
    if not barcodeKit and not barcode:
        return True, None

    for item in samplesetitems:
        if type(item) == types.DictType:
            id1 = item.get("pending_id")
            barcodeKit1 = item.get("barcodeKit", barcodeKit)
            barcode1 = item.get("barcode", None)
            item_name = item.get("name", None)
        else:
            id1 = item.pk
            barcode1 = item.dnabarcode.id_str if item.dnabarcode else None
            barcodeKit1 = item.dnabarcode.name if item.dnabarcode else None
            item_name = item.sample.name

        # skip self
        if id1 and samplesetitem_id and int(id1) == int(samplesetitem_id):
            continue

        # ensure only 1 barcode kit for the whole sample set
        if barcodeKit and barcodeKit1 and barcodeKit != barcodeKit1:
            if not pcrPlateRow:
                errorMessage = validation.format(
                    ugettext_lazy(
                        "samplesets.samplesetitem.messages.validate.barcodekit.notequal"
                    ),
                    include_error_prefix=True,
                )  # "Error, Only one barcode kit can be used for a sample set"
                return False, errorMessage

        # ensure only 1 barcode id_str per sample
        if barcode and barcode1 and barcode == barcode1:
            # TODO : change the localization key field to sample name
            # errorMessage = validation.format(ugettext_lazy('samplesets.samplesetitem.messages.validate.barcode.unique.withplateposition'), {'barcode': barcode, 'pcrPlateRow': item_name}, include_error_prefix=True)  # "Error, A barcode can be assigned to only one sample in the sample set. %s has been assigned to another sample at PCR plate position (%s)" % (barcode, item_pcrPlate)
            errorMessage = (
                "Error, A barcode can be assigned to only one sample in the sample set. %s has been assigned to another sample (%s)"
                % (barcode, item_name)
            )

            if pcrPlateRow:
                if item.pcrPlateRow not in allPcrPlates:
                    return False, errorMessage
                else:
                    return True, ""

            return False, errorMessage

    return True, None


"""
    If User select "Ampliseq HD on chef" -> 
        - validate for 4 samples in each Chef assay group i.e 4 samples in group1 and 4 samples in group2
        - make sure each assay group belongs to one category(nuc. type + pool type + sample source ) respectively
         - If validation is successfull, tube position and pcr plate assignment operation will be performed in views.save_input_samples_for_sampleset()
         - Detailed validation and design logic @ https://confluence.amer.thermo.com/display/TS/Tech+Design+proposal+-+Ampliseq+HD+on+chef+support
    isAmpliseqHD : "is Ampliseq or Ampliseq HD". This flag is used by API/CSV to parse all the items
"""


def validate_inconsistent_ampliseq_HD_category(
    samplesetitems, pending_sampleSetItem_list, sampleset=None, isAmpliseqHD=False
):
    isValid = True
    errorMessage = []
    categoryDict = {}
    no_of_partitionKey = None
    if (
        sampleset and "amps_hd_on_chef_v1" in sampleset.libraryPrepType.lower()
    ) or isAmpliseqHD:
        if not samplesetitems:
            samplesetitems = pending_sampleSetItem_list
        try:
            samplesetitems.reverse()
            for item in samplesetitems:
                if type(item) == types.DictType:
                    nucleotideType = item.get("nucleotideType", "")
                    sampleSource = item.get("sampleSource", "")
                    panelPoolType = item.get("panelPoolType", "")
                else:
                    nucleotideType = item.nucleotideType
                    sampleSource = item.sampleSource
                    panelPoolType = item.panelPoolType

                if not nucleotideType or not sampleSource or not panelPoolType:
                    errorMessage.append(
                        "Error, Missing values: NucleotideType, SampleSource and panelPoolType must be specified for AmpliseSeq HD on chef sample"
                    )
                    isValid = False
                category = "{0}_{1}_{2}".format(
                    nucleotideType, sampleSource, panelPoolType
                )
                if not categoryDict:
                    categoryDict[category] = {"Group 1": [item]}
                    continue
                else:
                    existingGroups = [
                        list(groups.keys()) for cat, groups in categoryDict.items()
                    ]
                    flattened_groups = [y for x in existingGroups for y in x]
                    if (
                        "Group 2" not in flattened_groups
                        and category not in categoryDict
                    ):
                        categoryDict[category] = {"Group 2": [item]}
                        continue

                if category not in categoryDict:
                    errorMessage.append(
                        "Error, Too many assay group categories (Nucleotide type + Sample source + Panel pool type) specified for AmpliSeq HD on Chef sample. Only 2 assay groups are allowed."
                    )
                    return False, errorMessage, categoryDict
                if "Group 1" in categoryDict[category]:
                    categoryDict[category]["Group 1"].append(item)
                else:
                    categoryDict[category]["Group 2"].append(item)

            if categoryDict:
                no_of_partitionKey = len(list(categoryDict.keys()))
        except Exception as Err:
            logger.debug(Err)
            errorMessage.append(Err)
        if no_of_partitionKey > 2:
            errorMessage.append(
                "Error, Too many assay group category (NucleotideType + Sample source + panel pool type) specified for AmpliSeq HD on Chef sample"
            )
            isValid = False
        if categoryDict:
            for partitionKey, groups in categoryDict.items():
                for key, value in groups.items():
                    if len(value) > 4:
                        errorMessage.append(
                            "Error, Only 4 samples are allowed in each Assay group categories (Nucleotide type + Sample source + Pool type)"
                        )
                        isValid = False

    return isValid, errorMessage, categoryDict


def validate_pcrPlate_position_samplesetitems(
    samplesetitems, pcrPlateRow, samplesetitem_id, sampleset=None
):
    """
        1) validate pcrPlateRow position selected for sampleSetItem is unique
        2) for Ampliseq on Chef pcrPlateRow value is required

        If validating with a dict of samplesetitems, "pending_id" key must contain a unique id number.
    """
    for item in samplesetitems:
        if (
            sampleset
            and not pcrPlateRow
            and "amps_on_chef" in sampleset.libraryPrepType.lower()
        ):
            errorMessage = validation.format(
                ugettext_lazy(
                    "samplesets.samplesetitem.messages.validate.pcrPlateRow.required"
                ),
                include_error_prefix=True,
            )  # "Error, A PCR plate position must be specified for AmpliSeq on Chef sample"
            return False, errorMessage

        if type(item) == types.DictType:
            pcrPlateRow1 = item.get("pcrPlateRow", "")
            id1 = item.get("pending_id")
        else:
            pcrPlateRow1 = item.pcrPlateRow
            id1 = item.pk

        # skip self
        if id1 and samplesetitem_id and int(id1) == int(samplesetitem_id):
            continue

        # ensure only 1 pcr plate position per sample
        if pcrPlateRow and pcrPlateRow1 and pcrPlateRow.lower() == pcrPlateRow1.lower():
            errorMessage = validation.format(
                ugettext_lazy(
                    "samplesets.samplesetitem.messages.validate.pcrPlateRow.unique"
                ),
                {"pcrPlateRow": pcrPlateRow},
                include_error_prefix=True,
            )  # "Error, A PCR plate position can only have one sample in it. Position %s has already been occupied by another sample" % (pcrPlateRow)
            return False, errorMessage

    return True, None


def validate_sample_for_sampleSet(
    queryDict, samplesetitem_label=_SampleSetItem.verbose_name
):
    """
    basic validation of sample attributes for sample set item creation/update
    return a boolean isValid and a list of text error messages
    """
    errors = []

    if not queryDict:
        return (
            False,
            [
                validation.invalid_no_data(
                    samplesetitem_label, include_error_prefix=True
                )
            ],
        )  # "Error, No sample data to validate."

    # validate sample fields

    sample_errors = validate_sample_data(queryDict)
    if sample_errors:
        errors.extend(sample_errors)

    # validate sampleset item fields

    isValid, errorMessage, _ = validate_nucleotideType(
        queryDict.get("nucleotideType", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage, _ = validate_sampleSource(
        queryDict.get("sampleSource", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage, _ = validate_panelPoolType(
        queryDict.get("panelPoolType", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage, _ = validate_sampleGender(
        queryDict.get("gender", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage, _ = validate_controlType(
        queryDict.get("controlType", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage, _ = validate_sampleGroupType(
        queryDict.get("relationshipRole", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage = validate_sampleGroup(queryDict.get("relationshipGroup", ""))
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage, _ = validate_cancerType(
        queryDict.get("cancerType", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage = validate_cellularityPct(queryDict.get("cellularityPct"))
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage = validate_sampleBiopsyDays(queryDict.get("biopsyDays", ""))
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage = validate_sampleCoupleId(
        queryDict.get("coupleId", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage = validate_sampleCellNum(queryDict.get("cellNum", "").strip())
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage = validate_sampleEmbryoId(
        queryDict.get("embryoId", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage, _ = validate_population(
        queryDict.get("population", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage, _ = validate_mouseStrains(
        queryDict.get("mouseStrains", "").strip()
    )
    if not isValid:
        errors.append(errorMessage)

    sampleCollectionDate = queryDict.get("sampleCollectionDate", "").strip()
    isValid, errorMessage = validate_sampleCollectionDate(sampleCollectionDate)
    if not isValid:
        errors.append(errorMessage)

    sampleReceiptDate = queryDict.get("sampleReceiptDate", "").strip()
    isValid, errorMessage = validate_sampleReceiptDate(
        sampleReceiptDate, sampleCollectionDate
    )
    if not isValid:
        errors.append(errorMessage)

    return len(errors) == 0, errors


def validate_sampleDisplayedName(
    sampleDisplayedName, field_label=_Sample.displayedName.verbose_name
):
    isValid, errorMessage = _validate_textValue_mandatory(
        sampleDisplayedName, field_label
    )

    if not isValid:
        return False, errorMessage

    isValid, errorMessage = _validate_textValue(sampleDisplayedName, field_label)

    if not isValid:
        return False, errorMessage

    isValid, errorMessage = _validate_textValue_leadingChars(
        sampleDisplayedName, field_label
    )

    if not isValid:
        return False, errorMessage

    if not validation.is_valid_length(
        sampleDisplayedName.strip(), MAX_LENGTH_SAMPLE_DISPLAYED_NAME
    ):
        errorMessage = validation.invalid_length_error(
            field_label,
            MAX_LENGTH_SAMPLE_DISPLAYED_NAME,
            sampleDisplayedName.strip(),
            include_error_prefix=True,
        )
        return False, errorMessage

    return True, None


def validate_sampleExternalId(
    sampleExternalId, field_label=_Sample.externalId.verbose_name
):
    isValid = False
    isValid, errorMessage = _validate_textValue(sampleExternalId, field_label)

    if not isValid:
        return False, errorMessage

    if not validation.is_valid_length(
        sampleExternalId.strip(), MAX_LENGTH_SAMPLE_EXTERNAL_ID
    ):
        errorMessage = validation.invalid_length_error(
            field_label,
            MAX_LENGTH_SAMPLE_EXTERNAL_ID,
            sampleExternalId.strip(),
            include_error_prefix=True,
        )
        return False, errorMessage

    return True, None


def validate_sampleDescription(
    sampleDescription, field_label=_Sample.description.verbose_name
):
    isValid = False
    if validation.has_value(sampleDescription):
        isValid, errorMessage = _validate_textValue(sampleDescription, field_label)
        if not isValid:
            return False, errorMessage

        if not validation.is_valid_length(
            sampleDescription.strip(), MAX_LENGTH_SAMPLE_DESCRIPTION
        ):
            errorMessage = validation.invalid_length_error(
                field_label,
                MAX_LENGTH_SAMPLE_DESCRIPTION,
                sampleDescription.strip(),
                include_error_prefix=True,
            )
            return False, errorMessage

    return True, None


def _validate_SampleAnnotation_CV(cvValue, cvAnnotationType, field_label):
    if not cvValue:
        return True, None, cvValue

    cvs = SampleAnnotation_CV.objects.filter(
        annotationType=cvAnnotationType, isActive=True
    )
    cvsMatching = cvs.filter(value__iexact=cvValue)

    isValid = False
    if cvsMatching.count() == 0:
        choices = cvs.order_by("value").values_list("value", flat=True)
        return (
            False,
            validation.invalid_choice(
                field_label, cvValue, choices, include_error_prefix=True
            ),
            cvValue,
        )

    return True, None, cvsMatching[0]


def validate_sampleSource(sampleSource, field_label="Sample Source"):
    return _validate_SampleAnnotation_CV(sampleSource, "sampleSource", field_label)


def validate_panelPoolType(panelPoolType, field_label="Panel Pool Type"):
    return _validate_SampleAnnotation_CV(panelPoolType, "panelPoolType", field_label)


def validate_additionalCycles(additionalCycles, field_label="Additional Cycles"):
    return _validate_SampleAnnotation_CV(
        additionalCycles, "additionalCycles", field_label
    )


def validate_cyclingProtocols(cyclingProtocols, field_label="Cycling Protocols"):
    return _validate_SampleAnnotation_CV(
        cyclingProtocols, "cyclingProtocols", field_label
    )


def validate_sampleGender(
    sampleGender,
    field_label=ugettext_lazy("samplesets.samplesetitem.fields.gender.label"),
):
    return _validate_SampleAnnotation_CV(sampleGender, "gender", field_label)


def validate_sampleGroupType(
    sampleGroupType,
    field_label=ugettext_lazy("samplesets.samplesetitem.fields.relationshipRole.label"),
):
    return _validate_SampleAnnotation_CV(
        sampleGroupType, "relationshipRole", field_label
    )


def validate_sampleGroup(
    sampleGroup,
    field_label=ugettext_lazy(
        "samplesets.samplesetitem.fields.relationshipGroup.label"
    ),
):
    if sampleGroup and not sampleGroup.isdigit():
        return False, validation.invalid_uint(field_label, include_error_prefix=True)

    return True, None


def validate_cancerType(
    cancerType,
    field_label=ugettext_lazy("samplesets.samplesetitem.fields.cancerType.label"),
):
    return _validate_SampleAnnotation_CV(cancerType, "cancerType", field_label)


def validate_population(population, field_label="Population"):
    return _validate_SampleAnnotation_CV(population, "population", field_label)


def validate_mouseStrains(mouseStrains, field_label="Mouse Strains"):
    return _validate_SampleAnnotation_CV(mouseStrains, "mouseStrains", field_label)


def validate_cellularityPct(
    cellularityPct,
    field_label=ugettext_lazy("samplesets.samplesetitem.fields.cellularityPct.label"),
):
    """
    check if input is a positive integer between 0 and 100 inclusively.
    """

    if cellularityPct:
        if RepresentsInt(cellularityPct):
            value = int(cellularityPct)
            if value < 0:
                return (
                    False,
                    validation.invalid_min_value(
                        field_label, 0, include_error_prefix=True
                    ),
                )
            elif value > 100:
                return (
                    False,
                    validation.invalid_max_value(
                        field_label, 100, include_error_prefix=True
                    ),
                )
            else:
                return True, None
        else:
            return (
                False,
                validation.invalid_uint(field_label, include_error_prefix=True),
            )

    return True, None


def validate_sampleCollectionDate(
    sampleCollectionDate,
    field_label=ugettext_lazy(
        "samplesets.samplesetitem.fields.sampleCollectionDate.label"
    ),
):
    if sampleCollectionDate:
        try:
            datetime.datetime.strptime(sampleCollectionDate, "%Y-%m-%d")
        except ValueError:
            return (
                False,
                validation.invalid_date_format(field_label, include_error_prefix=True),
            )

    return True, None


def validate_sampleReceiptDate(
    sampleReceiptDate,
    sampleCollectionDate,
    field_label=ugettext_lazy(
        "samplesets.samplesetitem.fields.sampleReceiptDate.label"
    ),
):
    if sampleReceiptDate:
        try:
            datetime.datetime.strptime(sampleReceiptDate, "%Y-%m-%d")
        except ValueError:
            return (
                False,
                validation.invalid_date_format(field_label, include_error_prefix=True),
            )
        if sampleCollectionDate:
            try:
                datetime.datetime.strptime(sampleCollectionDate, "%Y-%m-%d")
                sample_collectDate = time.strptime(sampleCollectionDate, "%Y-%m-%d")
                sample_receiptDate = time.strptime(sampleReceiptDate, "%Y-%m-%d")
                if sample_receiptDate and sample_receiptDate < sample_collectDate:
                    return (
                        False,
                        validation.invalid_receipt_date(
                            field_label, include_error_prefix=True
                        ),
                    )
            except ValueError:
                return (
                    False,
                    validation.invalid_receipt_date(
                        field_label, include_error_prefix=True
                    ),
                )

    return True, None


def validate_sampleBiopsyDays(
    sampleBiopsyDays,
    field_label=ugettext_lazy("samplesets.samplesetitem.fields.biopsyDays.label"),
):
    isValid, errorMessage = _validate_uintandzeroValue(sampleBiopsyDays, field_label)
    return isValid, errorMessage


def validate_sampleCellNum(
    sampleCellNum,
    field_label=ugettext_lazy("samplesets.samplesetitem.fields.cellNum.label"),
):
    return _validate_optional_text(
        sampleCellNum, MAX_LENGTH_SAMPLE_CELL_NUM, field_label
    )


def validate_sampleCoupleId(
    sampleCoupleId,
    field_label=ugettext_lazy("samplesets.samplesetitem.fields.coupleId.label"),
):
    return _validate_optional_text(
        sampleCoupleId, MAX_LENGTH_SAMPLE_COUPLE_ID, field_label
    )


def validate_sampleEmbryoId(
    sampleEmbryoId,
    field_label=ugettext_lazy("samplesets.samplesetitem.fields.embryoId.label"),
):
    return _validate_optional_text(
        sampleEmbryoId, MAX_LENGTH_SAMPLE_EMBRYO_ID, field_label
    )


def _validate_textValue_mandatory(value, displayedTerm):
    if not validation.has_value(value):
        return (
            False,
            validation.required_error(displayedTerm, include_error_prefix=True),
        )

    return True, None


def _validate_intValue(value, displayedTerm):
    if value and not RepresentsUnsignedInt(value):
        return (
            False,
            validation.invalid_uint(displayedTerm, include_error_prefix=True),
        )  # "Error, " + displayedTerm + ERROR_MSG_INVALID_DATATYPE
    return True, None


def _validate_uintandzeroValue(value, displayedTerm):
    if value and not RepresentsUnsignedIntOrZero(value):
        return (
            False,
            validation.invalid_uint_n_zero(displayedTerm, include_error_prefix=True),
        )  # "Error, " + displayedTerm + ERROR_MSG_INVALID_DATATYPE
    return True, None


def _validate_textValue(value, displayedTerm):
    if value and not validation.is_valid_chars(value):
        return (
            False,
            validation.invalid_chars_error(displayedTerm, include_error_prefix=True),
        )
    return True, None


def _validate_textValue_leadingChars(value, displayedTerm):
    if value and not validation.is_valid_leading_chars(value):
        return (
            False,
            validation.invalid_leading_chars(displayedTerm, include_error_prefix=True),
        )

    return True, None


def _validate_optional_text(value, maxLength, displayedTerm):
    isValid, errorMessage = _validate_textValue(value.strip(), displayedTerm)
    if not isValid:
        return False, errorMessage

    if not validation.is_valid_length(value.strip(), maxLength):
        errorMessage = validation.invalid_length_error(
            displayedTerm, maxLength, value.strip(), include_error_prefix=True
        )
        return False, errorMessage

    return True, None


def validate_sampleAttribute(
    attribute, value, sampleattribute_label=_SampleAttribute.verbose_name
):
    """
    validate the sample attribute value for the attribute of interest
    return a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length will not be validated since maxLength has been specified in the form.
    """

    isValid = False
    if not attribute:
        return (
            False,
            validation.invalid_no_data(
                sampleattribute_label, include_error_prefix=True
            ),
        )  # "Error, No sample attribute to validate."

    if not validation.has_value(value):
        if attribute.isMandatory:
            return (
                False,
                validation.required_error(
                    attribute.displayedName, include_error_prefix=True
                ),
            )
    else:
        aValue = value.strip()
        if attribute.dataType.dataType == "Text" and not validation.is_valid_chars(
            aValue
        ):
            return (
                False,
                validation.invalid_chars_error(
                    attribute.displayedName, include_error_prefix=True
                ),
            )
        if attribute.dataType.dataType == "Integer" and not aValue.isdigit():
            return (
                False,
                validation.invalid_int(
                    attribute.displayedName, include_error_prefix=True
                ),
            )
        if not validation.is_valid_chars(aValue):
            return (
                False,
                validation.invalid_chars_error(
                    attribute.displayedName, include_error_prefix=True
                ),
            )

        if not validation.is_valid_length(aValue, MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE):
            errorMessage = validation.invalid_length_error(
                attribute.displayedName,
                MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE,
                aValue,
                include_error_prefix=True,
            )
            return False, errorMessage

    isValid = True
    return isValid, None


def validate_sampleAttribute_mandatory_for_no_value(
    attribute, sampleattribute_label=_SampleAttribute.verbose_name
):
    if not attribute:
        return (
            False,
            validation.invalid_no_data(
                sampleattribute_label, include_error_prefix=True
            ),
        )  # "Error, No sample attribute to validate."

    if attribute.isMandatory:
        return (
            False,
            validation.required_error(
                attribute.displayedName, include_error_prefix=True
            ),
        )  # "Error, " + attribute.displayedName + " value is required."

    return True, None


def validate_sampleAttribute_definition(
    attributeName,
    attributeDescription,
    attribute_name_label=ugettext_lazy(
        "samplesets.sampleattributes.fields.displayedName.label"
    ),
    attribute_description_label=ugettext_lazy(
        "samplesets.sampleattributes.fields.description.label"
    ),
):
    """
    validate the sample attribute definition
    return a boolean isValid and a text string for error message, None if input passes validation
    Note: Input length will not be validated since maxLength has been specified in the form.
    """

    if not validation.has_value(attributeName):
        return (
            False,
            validation.required_error(attribute_name_label, include_error_prefix=True),
        )
    if not validation.is_valid_chars(attributeName.strip()):
        return (
            False,
            validation.invalid_chars_error(
                attribute_name_label, include_error_prefix=True
            ),
        )

    if not validation.is_valid_length(
        attributeName.strip(), MAX_LENGTH_SAMPLE_ATTRIBUTE_DISPLAYED_NAME
    ):
        errorMessage = validation.invalid_length_error(
            attribute_name_label,
            MAX_LENGTH_SAMPLE_ATTRIBUTE_DISPLAYED_NAME,
            attributeName.strip(),
            include_error_prefix=True,
        )
        return False, errorMessage

    if not validation.is_valid_chars(attributeDescription):
        return (
            False,
            validation.invalid_chars_error(
                attribute_description_label, include_error_prefix=True
            ),
        )

    if not validation.is_valid_length(
        attributeDescription.strip(), MAX_LENGTH_SAMPLE_ATTRIBUTE_DESCRIPTION
    ):
        errorMessage = validation.invalid_length_error(
            attribute_description_label,
            MAX_LENGTH_SAMPLE_ATTRIBUTE_DESCRIPTION,
            attributeDescription.strip(),
            include_error_prefix=True,
        )
        return False, errorMessage

    return True, None


def validate_barcodekit_and_id_str(
    barcodeKit,
    barcode_id_str,
    barcodeKit_label="Barcode Set",
    barcode_id_str_label="Barcode",
):
    item = ""
    errorMessage = ""

    # Trim off barcode and barcode kit leading and trailing spaces and update the log file if exists
    if (len(barcodeKit) - len(barcodeKit.lstrip())) or (
        len(barcodeKit) - len(barcodeKit.rstrip())
    ):
        logger.warning(
            "The BarcodeKitName(%s) contains Leading/Trailing spaces and got trimmed."
            % barcodeKit
        )

    if (len(barcode_id_str) - len(barcode_id_str.lstrip())) or (
        len(barcode_id_str) - len(barcode_id_str.rstrip())
    ):
        logger.warning(
            "The BarcodeName (%s) of BarcodeKitName(%s) contains Leading/Trailing spaces and got trimmed."
            % (barcode_id_str, barcodeKit)
        )

    barcodeKit = barcodeKit.strip()
    barcode_id_str = barcode_id_str.strip()

    if not barcodeKit and not barcode_id_str:
        return True, errorMessage, item

    # First validate that if the barcodeKit is entered then the id_str must also be entered
    if barcodeKit and not barcode_id_str:
        return (
            False,
            validation.required_error(barcode_id_str_label, include_error_prefix=True),
            "barcode_id_str",
        )  # "Error, Please enter a barcode item", 'barcode_id_str'
    # Next validate that if the id_str is entered the barcodeKit must also be entered
    elif barcode_id_str and not barcodeKit:
        return (
            False,
            validation.required_error(barcodeKit_label, include_error_prefix=True),
            "barcodeKit",
        )  # "Error, Please enter a Barcoding Kit", 'barcodeKit'
    # Next validate that the barcodeKit is spelled correctly
    dnabarcode = models.dnaBarcode.objects.filter(name__iexact=barcodeKit)
    if dnabarcode.count() == 0:
        return (
            False,
            validation.invalid_invalid_value(
                barcodeKit_label, barcodeKit, include_error_prefix=True
            ),
            "barcodeKit",
        )  # "Error, Invalid Barcodekit"
    # Next validate the that id_str is spelled correctly
    dnabarcode = models.dnaBarcode.objects.filter(id_str__iexact=barcode_id_str)
    if dnabarcode.count() == 0:
        return (
            False,
            validation.invalid_invalid_value(
                barcode_id_str_label, barcode_id_str, include_error_prefix=True
            ),
            "barcode_id_str",
        )  # "Error, Invalid barcode"
    # Next validate that the Barcodekit and barcode belong together
    dnabarcode = models.dnaBarcode.objects.filter(
        name__iexact=barcodeKit, id_str__iexact=barcode_id_str
    )
    if dnabarcode.count() != 1:
        return (
            False,
            validation.format(
                ugettext_lazy(
                    "samplesets.samplesetitem.messages.validate.barcodekit.invalid_choice"
                ),
                {
                    "barcode": barcode_id_str,
                    "barcodeSet": barcodeKit,
                    "barcodeSetLabel": barcodeKit_label,
                    "barcodeLabel": barcode_id_str_label,
                },
                include_error_prefix=True,
            ),
            "barcodeKit",
        )

    return True, errorMessage, item


def validate_nucleotideType(nucleotideType, field_label="Sample Nucleotide Type"):
    """
    validate nucleotide type case-insensitively with leading/trailing blanks in the input ignored
    """
    VALID_NUCLEOTIDE_TYPES = [Nuc[0] for Nuc in SampleSetItem.ALLOWED_NUCLEOTIDE_TYPES]
    error = ""
    input = ""

    if nucleotideType:
        input = nucleotideType.strip().lower()
        if input == "fusions":
            input = "rna"

        if not validation.is_valid_choice(input, VALID_NUCLEOTIDE_TYPES):
            return (
                False,
                validation.invalid_choice(
                    field_label, nucleotideType, VALID_NUCLEOTIDE_TYPES
                ),
                input,
            )

    return True, error, input


def validate_sample_data(queryDict, samplesetitem_label=_SampleSetItem.verbose_name):
    """
    validate Sample attributes
    return a list containing error messages
    """
    errors = []

    if not queryDict:
        return (
            False,
            validation.invalid_no_data(samplesetitem_label, include_error_prefix=True),
        )  # "Error, No sample data to validate."

    sampleName = queryDict.get("name", "")
    sampleDisplayedName = queryDict.get("displayedName", "")

    if not sampleName:
        sampleName = sampleDisplayedName.replace(" ", "_")
    if not sampleDisplayedName:
        sampleDisplayedName = sampleName

    isValid, errorMessage = validate_sampleName(sampleName)
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage = validate_sampleExternalId(queryDict.get("externalId", ""))
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage = validate_sampleDescription(queryDict.get("description", ""))
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage = validate_sampleStatus(queryDict.get("status", ""))
    if not isValid:
        errors.append(errorMessage)

    isValid, errorMessage = validate_for_same_sampleName_displayedName(
        sampleDisplayedName, sampleName
    )
    if not isValid:
        errors.append(errorMessage)

    return errors


def validate_for_same_sampleName_displayedName(sampleDisplayedName, sampleName):
    # Verify the sample name and displayed name are similar (allow spaces)
    tmpSampleDisplayedName = "_".join(sampleDisplayedName.split())
    if sampleName != sampleDisplayedName and sampleName != tmpSampleDisplayedName:
        return (
            False,
            "Sample name should match sample displayed name, with spaces allowed in the displayedName",
        )  # TODO: i18n

    return True, None


def validate_sampleStatus(sampleStatus, field_label=_Sample.status.verbose_name):
    allowed_status_list = [s[0] for s in Sample.ALLOWED_STATUS]

    if not validation.is_valid_choice(sampleStatus, allowed_status_list):
        return (
            False,
            validation.invalid_choice(field_label, sampleStatus, allowed_status_list),
        )

    return True, None


def validate_sampleName(sampleName, field_label=_Sample.displayedName.verbose_name):

    isValid, errorMessage = _validate_textValue_mandatory(sampleName, field_label)

    if not isValid:
        return isValid, errorMessage

    isValid, errorMessage = _validate_textValue(sampleName, field_label)
    if not isValid:
        return isValid, errorMessage

    isValid, errorMessage = _validate_textValue_leadingChars(sampleName, field_label)
    if not isValid:
        return isValid, errorMessage

    if not validation.is_valid_length(sampleName.strip(), MAX_LENGTH_SAMPLE_NAME):
        errorMessage = validation.invalid_length_error(
            field_label,
            MAX_LENGTH_SAMPLE_NAME,
            sampleName.strip(),
            include_error_prefix=True,
        )
        return isValid, errorMessage

    return True, None


def validate_pcrPlateRow(pcrPlateRow, field_label="PCR Plate Position"):
    """
    validate PCR plate row case-insensitively with leading/trailing blanks in the input ignored
    """
    isValid = True
    errors = []
    input = ""

    if pcrPlateRow:
        input = pcrPlateRow.strip().upper()
        validValues = [v[0] for v in SampleSetItem.ALLOWED_AMPLISEQ_PCR_PLATE_ROWS_V1]
        if not validation.is_valid_keyword(input, validValues):
            errors.append(validation.invalid_keyword_error(field_label, validValues))
            isValid = False

    return isValid, errors, input


def validate_pcrPlateCol(pcrPlateCol, field_label="PCR Plate Position"):
    """
    validate PCR plate row case-insensitively with leading/trailing blanks in the input ignored
    """
    isValid = True
    errors = []
    input = ""

    if pcrPlateCol:
        input = pcrPlateCol.strip().upper()
        validValues = [
            v[0] for v in SampleSetItem.ALLOWED_AMPLISEQ_PCR_PLATE_COLUMNS_V1
        ]
        if not validation.is_valid_keyword(input, validValues):
            errors.append(validation.invalid_keyword_error(field_label, validValues))
            isValid = False

    return isValid, errors, input


def validate_samplesetStatus(samplesetStatus, field_label="Status"):
    """
    validate samplesetStatus with leading/trailing blanks in the input ignored
    """
    isValid = True
    errors = []
    inputData = ""

    if samplesetStatus:
        inputData = samplesetStatus.strip().lower()
        validValues = [v[0] for v in SampleSet.ALLOWED_SAMPLESET_STATUS]
        if not validation.is_valid_keyword(inputData, validValues):
            errors.append(validation.invalid_keyword_error(field_label, validValues))
            errors = "".join(errors).replace("are ,", ":")
            isValid = False
            return isValid, errors, samplesetStatus

    isValid = True
    return isValid, None, None


def validate_libPrepType(
    libPrepType, field_label="Library Prep Type"
):  # TODO: Dead code?
    """
    validate libPrepType with leading/trailing blanks in the input ignored
    """
    isValid = True
    errors = []
    inputData = ""
    if libPrepType:
        inputData = libPrepType.strip()

        validValues = [v[0] for v in SampleSet.ALLOWED_LIBRARY_PREP_TYPES]
        if not validation.is_valid_keyword(inputData, validValues):
            errors.append(validation.invalid_keyword_error(field_label, validValues))
            errors = "".join(errors).replace("are ,", ":")
            isValid = False
            return isValid, errors, libPrepType

    isValid = True
    return isValid, None, None


def validate_sampleBarcodeMapping(queryDict, sampleSet):
    """
    validate sampleBarcodeMapping input sent via API
        - BarcodeKit, Barcode
        - SampleRow, SampleColumn
    """
    isValid = False
    errordict = {}
    pcrplateBarcodeQueryDict = queryDict.get("sampleBarcodeMapping", None)

    # Input JSON object from Chef to Update
    allBarcodeKits = [
        pcr_plate_barcode["sampleToBarcode"]["barcodeKit"]
        for pcr_plate_barcode in pcrplateBarcodeQueryDict
    ]
    allBarcodes = [
        pcr_plate_barcode["sampleToBarcode"]["barcode"]
        for pcr_plate_barcode in pcrplateBarcodeQueryDict
    ]
    allPcrPlates = [
        pcr_plate_barcode["sampleToBarcode"]["sampleRow"]
        for pcr_plate_barcode in pcrplateBarcodeQueryDict
    ]
    singleBarcode = allBarcodeKits and all(
        allBarcodeKits[0] == elem for elem in allBarcodeKits
    )

    # validate if same barcode is being used for multiple samples
    if len(allBarcodes) != len(set(allBarcodes)):
        dupBarcode = [x for x in allBarcodes if allBarcodes.count(x) >= 2]
        errordict = {
            "result": "1",
            "message": "Fail",
            "detailMessage": validation.format(
                ugettext_lazy(
                    "samplesets.samplesetitem.messages.validate.barcode.unique"
                ),
                {"barcode": ", ".join(dupBarcode)},
                include_error_prefix=True,
            ),  # "Error, A barcode can be assigned to only one sample in the sample set."
            "inputData": dupBarcode,
        }
        return isValid, errordict

    if not singleBarcode:
        errordict = {
            "result": "1",
            "message": "Fail",
            "detailMessage": validation.format(
                ugettext_lazy(
                    "samplesets.samplesetitem.messages.validate.barcodekit.notequal"
                ),
                include_error_prefix=True,
            ),  # "Error, Only one barcode Kit can be used for a sample set"
            "inputData": allBarcodeKits,
        }
        return isValid, errordict

    if pcrplateBarcodeQueryDict:
        sampleSetItems = sampleSet.samples.all()
        userPcrPlates = [item.pcrPlateRow for item in sampleSetItems]
        isValid, errorMessage = validate_user_chef_barcodeKit(
            sampleSetItems, allBarcodeKits, allPcrPlates
        )
        if not isValid:
            return isValid, errorMessage
        for pcr_plate_barcode in pcrplateBarcodeQueryDict:
            barcodeKit = pcr_plate_barcode["sampleToBarcode"]["barcodeKit"]
            barcode = pcr_plate_barcode["sampleToBarcode"]["barcode"]
            row = pcr_plate_barcode["sampleToBarcode"]["sampleRow"]

            # validate pcrPlate Row
            isValid, errormsg, inputData = validate_pcrPlateRow(row)
            if not isValid:
                errordict = {
                    "result": "1",
                    "message": "Fail",
                    "detailMessage": "".join(errormsg),
                    "inputData": inputData,
                }
                return isValid, errordict
            else:
                isValid = False

            # validate pcrPlate Column
            col = pcr_plate_barcode["sampleToBarcode"]["sampleColumn"]
            isValid, errormsg, inputData = validate_pcrPlateCol(col)
            if not isValid:
                errordict = {
                    "result": "1",
                    "message": "Fail",
                    "detailMessage": "".join(errormsg),
                    "inputData": inputData,
                }
                return isValid, errordict
            else:
                isValid = False

            # validate the specified barcode belongs to appropriate barcodeKit
            isValid, errormsg, items = validate_barcodekit_and_id_str(
                barcodeKit, barcode
            )
            if not isValid:
                errordict = {
                    "inputData": [barcodeKit, barcode],
                    "result": "1",
                    "message": "Fail",
                    "detailMessage": errormsg,
                }
                return isValid, errordict

            # Override the barcode and barcodeKit if User specified PCR Plate row and chef specified PCR Plate rows are similar
            # Validate if there is any pcrPlate Row mismatch between User and Chef Inputs for Data integrity
            mistmatch_PcrPlates = set(userPcrPlates) - set(allPcrPlates)
            if len(mistmatch_PcrPlates):
                isValid, errors = validate_barcoding_samplesetitems(
                    sampleSetItems,
                    barcodeKit,
                    barcode,
                    sampleSet.id,
                    allPcrPlates=allPcrPlates,
                    pcrPlateRow=row,
                )
            if not isValid:
                return isValid, errors

    isValid = True
    return isValid, None


def validate_user_chef_barcodeKit(samplesetitems, chef_barcodeKit, allPcrPlates):
    all_TS_barcodeKit = []
    userPcrPlates = []
    isValid = True
    errorMessage = ""

    for item in samplesetitems:
        barcodeKit_TS = item.dnabarcode.name if item.dnabarcode else None
        userPcrPlate = item.pcrPlateRow
        all_TS_barcodeKit.append(barcodeKit_TS)
        userPcrPlates.append(userPcrPlate)
    barcodeKit_mistmatch = set(chef_barcodeKit) - set(all_TS_barcodeKit)
    pcrPlate_mistmatch = set(userPcrPlates) - set(allPcrPlates)

    if len(pcrPlate_mistmatch) and len(barcodeKit_mistmatch):
        print(pcrPlate_mistmatch)
        print(barcodeKit_mistmatch)
        isValid = False
        errorMessage = validation.format(
            ugettext_lazy(
                "samplesets.samplesetitem.messages.validate.barcodekit.notequal"
            ),
            include_error_prefix=True,
        )  # "Error, Only one barcode kit can be used for a sample set"
        return isValid, errorMessage
    return isValid, errorMessage


def validate_sampleSets_for_planning(sampleSets):
    """ Validate multiple sampleSets are compatible to create a Plan from """
    errors = []
    items = SampleSetItem.objects.filter(sampleSet__in=sampleSets)
    if not items:
        errors.append(
            validation.invalid_required_at_least_one_child(
                _SampleSet.verbose_name, _Sample.verbose_name
            )
        )  # 'Sample Set must have at least one sample'
        return errors

    samples_w_barcodes = items.exclude(dnabarcode__isnull=True)
    barcodeKitNames = samples_w_barcodes.values_list(
        "dnabarcode__name", flat=True
    ).distinct()
    if len(barcodeKitNames) > 1:
        errors.append(
            validation.format(
                ugettext_lazy("samplesets.messages.validate.barcodekit.notequal"),
                {"barcodeSetNames": ", ".join(barcodeKitNames)},
            )
        )  # 'Selected Sample Sets have different Barcode Kits: %s.' % ', '.join(barcodeKitNames)
    elif len(barcodeKitNames) == 1:
        barcodes = {}
        for barcode, sample, setname in samples_w_barcodes.values_list(
            "dnabarcode__id_str", "sample__name", "sampleSet__displayedName"
        ):
            if barcode in barcodes:
                msg = validation.format(
                    ugettext_lazy(
                        "samplesets.samplesetitem.messages.validate.barcode.unique.detailed"
                    ),
                    {
                        "barcode": barcode,
                        "sampleName": sample,
                        "sampleSetName": setname,
                        "dup_sampleName": barcodes[barcode][0],
                        "dup_sampleSetName": barcodes[barcode][1],
                    },
                )  # 'Multiple samples are assigned to barcode %s: %s (%s), %s (%s)' % (barcode, sample, setname, barcodes[barcode][0], barcodes[barcode][1])
                errors.append(msg)
            else:
                barcodes[barcode] = (sample, setname)

    return errors


def validate_controlType(controlType, field_label="Control Type"):
    errors, controlType = validate_sampleControlType(controlType, field_label)
    if errors:
        return False, errors[0], ""
    else:
        return True, "", controlType


def validate_libraryPrepKitName(libraryPrepKitName, field_label="Library Prep Kit"):
    errors, _ = validate_optional_kit_name(
        libraryPrepKitName, ["LibraryPrepKit"], field_label
    )
    return errors


def validate_samplesetGroupType(groupType, field_label="Group Type"):
    if groupType:
        groupTypeValues = SampleGroupType_CV.objects.values_list(
            "displayedName", flat=True
        )
        if groupType not in groupTypeValues:
            return (
                False,
                validation.invalid_choice(
                    field_label, groupType, groupTypeValues, include_error_prefix=True
                ),
            )

    return True, None


# Validation check for max samples limit
def validate_sampleset_items_limit(pending_samplesetitems, sampleSet):
    ss_object = SampleSet.objects.get(pk=int(sampleSet[0]))
    # sample set items limit only apply to Ampliseq HD on chef
    if ss_object.libraryPrepType not in VALID_LIB_TYPES:
        return True, None

    available_samples_in_ss = len(ss_object.samples.all())
    pending_samples_to_add = len(pending_samplesetitems)
    total_samples = available_samples_in_ss + pending_samples_to_add
    if sampleSet and total_samples > ASHD_SAMPLESET_ITEM_LIMIT:
        errorMessage = (
            "Error: a Sample Set must contain only %s samples. There are %s specified currently."
            % (ASHD_SAMPLESET_ITEM_LIMIT, total_samples)
        )
        return False, errorMessage

    return True, None
