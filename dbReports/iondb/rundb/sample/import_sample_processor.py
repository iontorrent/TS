# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb import models
from django.utils.translation import ugettext_lazy
import datetime
from django.utils import timezone
import logging

from django.db import transaction

from iondb.rundb import models

from iondb.rundb.models import (
    Sample,
    SampleSet,
    SampleSetItem,
    SampleAttribute,
    SampleGroupType_CV,
    SampleAttributeDataType,
    SampleAttributeValue,
)

from django.conf import settings

from django.contrib.auth.models import User

import sample_validator
from iondb.utils import validation
from distutils.version import StrictVersion

from traceback import format_exc
from datetime import datetime
from django.shortcuts import get_object_or_404

logger = logging.getLogger(__name__)

COLUMN_SAMPLE_CSV_VERSION = "CSV Version (required)"
COLUMN_SAMPLE_EXT_ID = "Sample ID"
COLUMN_SAMPLE_NAME = "Sample Name (required)"
COLUMN_GENDER = "Gender"
COLUMN_GROUP_TYPE = "Type"
COLUMN_GROUP = "Group"
COLUMN_SAMPLE_DESCRIPTION = "Description"
COLUMN_BARCODE_KIT = "Barcodekit"
COLUMN_BARCODE = "Barcode"
COLUMN_CANCER_TYPE = "Cancer Type"
COLUMN_CELLULARITY_PCT = "Cellularity %"
COLUMN_NUCLEOTIDE_TYPE = "Nucleotide Type"
ALTERNATE_COLUMN_NUCLEOTIDE_TYPE = "DNA/RNA/Fusions"
COLUMN_PCR_PLATE_POSITION = "PCR Plate Position"
COLUMN_BIOPSY_DAYS = "Biopsy Days"
COLUMN_CELL_NUM = "Cell Number"
COLUMN_COUPLE_ID = "Couple ID"
COLUMN_EMBRYO_ID = "Embryo ID"
COLUMN_CONTROLTYPE = "Control Type"
COLUMN_SAMPLE_SOURCE = "Sample Source"
COLUMN_PANEL_POOL_TYPE = "Panel Pool Type"
COLUMN_SAMPLE_COLLECTION_DATE = "Sample Collection Date"
COLUMN_SAMPLE_RECEIPT_DATE = "Sample Receipt Date"
COLUMN_SAMPLE_POPULATION = "Population"
COLUMN_SAMPLE_MOUSE_STRAINS = "Mouse Strains"


def process_csv_sampleSet(csvSampleDict, request, user, sampleSet_ids):
    """ read csv contents and convert data to raw data to prepare for sample persistence
    returns: a collection of error messages if errors found, a dictionary of raw data values
    """

    logger.debug(
        "ENTER import_sample_processor.process_csv_sampleSet() csvSampleDict=%s; "
        % (csvSampleDict)
    )
    failed = []
    isToSkipRow = False

    # check if mandatory fields are present
    requiredColumns = ["sampleName", "sample"]
    isRequired = [
        csvSampleDict.get(requiredColumn, None) for requiredColumn in requiredColumns
    ]

    if not any(isRequired):
        failed.append(
            (COLUMN_SAMPLE_NAME, validation.required_error(COLUMN_SAMPLE_NAME))
        )
    sample, sampleSetItem, ssi_sid = _create_sampleSetItem(
        csvSampleDict, request, user, sampleSet_ids
    )
    siv_sid = _create_sampleAttributeValue(csvSampleDict, request, user, sample)

    return failed, sample, sampleSetItem, isToSkipRow, ssi_sid, siv_sid


def get_sampleSetItem_kwargs(csvSampleDict, user):
    sampleDisplayedName = csvSampleDict.get(COLUMN_SAMPLE_NAME, "").strip()
    sampleExtId = csvSampleDict.get(COLUMN_SAMPLE_EXT_ID, "").strip()
    sampleGender = csvSampleDict.get(COLUMN_GENDER, "").strip()
    sampleControlType = csvSampleDict.get(COLUMN_CONTROLTYPE, "").strip()
    sampleGroupType = csvSampleDict.get(COLUMN_GROUP_TYPE, None)
    sampleGroup = csvSampleDict.get(COLUMN_GROUP, "0").strip()
    if not sampleGroup:
        sampleGroup = "0"
    sampleDescription = csvSampleDict.get(COLUMN_SAMPLE_DESCRIPTION, "").strip()
    barcodeKit = csvSampleDict.get(COLUMN_BARCODE_KIT, "").strip()
    barcodeAssignment = csvSampleDict.get(COLUMN_BARCODE, "").strip()
    pcrPlateRow = csvSampleDict.get(COLUMN_PCR_PLATE_POSITION, "").strip()
    nucleotideType = (
        csvSampleDict.get(COLUMN_NUCLEOTIDE_TYPE, "").strip()
        or csvSampleDict.get(ALTERNATE_COLUMN_NUCLEOTIDE_TYPE, "").strip()
    )
    sampleSource = csvSampleDict.get(COLUMN_SAMPLE_SOURCE, "").strip()
    panelPoolType = csvSampleDict.get(COLUMN_PANEL_POOL_TYPE, "").strip()
    sampleCollectionDate = csvSampleDict.get(COLUMN_SAMPLE_COLLECTION_DATE, "").strip()
    sampleReceiptDate = csvSampleDict.get(COLUMN_SAMPLE_RECEIPT_DATE, "").strip()
    if sampleCollectionDate:
        sampleCollectionDate = datetime.strptime(
            str(sampleCollectionDate), "%Y-%m-%d"
        ).date()
    if sampleReceiptDate:
        sampleReceiptDate = datetime.strptime(str(sampleReceiptDate), "%Y-%m-%d").date()

    cancerType = csvSampleDict.get(COLUMN_CANCER_TYPE, "").strip()
    cellularityPct = csvSampleDict.get(COLUMN_CELLULARITY_PCT, None)
    if cellularityPct:
        cellularityPct = cellularityPct.strip()

    if not cellularityPct:
        cellularityPct = None

    biopsyDays = csvSampleDict.get(COLUMN_BIOPSY_DAYS, "0").strip()
    cellNum = csvSampleDict.get(COLUMN_CELL_NUM, "").strip()
    coupleId = csvSampleDict.get(COLUMN_COUPLE_ID, None)
    if coupleId:
        coupleId = coupleId.strip()
    embryoId = csvSampleDict.get(COLUMN_EMBRYO_ID, "").strip()
    population = csvSampleDict.get(COLUMN_SAMPLE_POPULATION, "").strip()
    mouseStrains = csvSampleDict.get(COLUMN_SAMPLE_MOUSE_STRAINS, "").strip()
    currentDateTime = timezone.now()  ##datetime.datetime.now()

    sampleName = sampleDisplayedName.replace(" ", "_")

    isValid, errorMessage, nucleotideType_internal_value = sample_validator.validate_nucleotideType(
        nucleotideType, field_label=COLUMN_NUCLEOTIDE_TYPE
    )

    # validation has been done already, this is just to get the official value
    isValid, errorMessage, gender_CV_value = sample_validator.validate_sampleGender(
        sampleGender, field_label=COLUMN_GENDER
    )
    isValid, errorMessage, role_CV_value = sample_validator.validate_sampleGroupType(
        sampleGroupType, field_label=COLUMN_GROUP_TYPE
    )
    isValid, errorMessage, controlType_CV_value = sample_validator.validate_controlType(
        sampleControlType, field_label=COLUMN_CONTROLTYPE
    )
    isValid, errorMessage, cancerType_CV_value = sample_validator.validate_cancerType(
        cancerType, field_label=COLUMN_CANCER_TYPE
    )
    isValid, errorMessage, pcrPlateRow_internal_value = sample_validator.validate_pcrPlateRow(pcrPlateRow, field_label=COLUMN_PCR_PLATE_POSITION)

    sampleSetItem_kwargs = {
        "sampleName": sampleName,
        "sampleDisplayedName": sampleDisplayedName,
        "sampleExtId": sampleExtId,
        "barcodeKit": barcodeKit,
        "barcodeAssignment": barcodeAssignment,
        "gender": gender_CV_value,
        "relationshipRole": role_CV_value,
        "relationshipGroup": sampleGroup,
        "cancerType": cancerType_CV_value,
        "pcrPlateRow": pcrPlateRow_internal_value,
        "nucleotideType": nucleotideType_internal_value,
        "sampleSource": sampleSource,
        "panelPoolType": panelPoolType,
        "cellularityPct": cellularityPct if cellularityPct else None,
        "biopsyDays": int(biopsyDays) if biopsyDays else 0,
        "cellNum": cellNum,
        "coupleId": coupleId,
        "embryoId": embryoId,
        "creator": user,
        "creationDate": currentDateTime,
        "lastModifiedUser": user,
        "lastModifiedDate": currentDateTime,
        "description": sampleDescription,
        "controlType": controlType_CV_value,
        "displayedName": sampleDisplayedName,
        "sampleStatus": "created",
        "sampleDescription": sampleDescription,
        "sampleCollectionDate": sampleCollectionDate or None,
        "sampleReceiptDate": sampleReceiptDate or None,
        "population": population,
        "mouseStrains": mouseStrains,
        "date": timezone.now(),
    }

    return sampleSetItem_kwargs


def _create_sampleSetItem(csvSampleDict, request, user, sampleSet_id):
    sampleDisplayedName = csvSampleDict.get("sampleDisplayedName", "")
    sampleExtId = csvSampleDict.get("sampleExtId", "")
    sampleGender = csvSampleDict.get("gender", "")
    sampleControlType = csvSampleDict.get("controlType", "")
    relationshipRole = csvSampleDict.get("relationshipRole", "")
    sampleGroup = csvSampleDict.get("sampleGroup", "") or 0
    sampleDescription = csvSampleDict.get("sampleDescription", "")
    barcodeKit = csvSampleDict.get("barcodeKit", "")
    barcodeAssignment = csvSampleDict.get("barcodeAssignment", "")
    nucleotideType = csvSampleDict.get("nucleotideType", "")
    cancerType = csvSampleDict.get("cancerType", "")
    cellularityPct = csvSampleDict.get("cellularityPct", None)
    pcrPlateRow = csvSampleDict.get("pcrPlateRow", "")
    biopsyDays = csvSampleDict.get("coupleId", "0")
    cellNum = csvSampleDict.get("cellNum", "")
    coupleId = csvSampleDict.get("coupleId", None)
    embryoId = csvSampleDict.get("embryoId", "")
    assayGroup = csvSampleDict.get("assayGroup", "")
    sampleSource = csvSampleDict.get("sampleSource", "")
    panelPoolType = csvSampleDict.get("panelPoolType", "")
    tubePosition = csvSampleDict.get("tubePosition", "")
    population = csvSampleDict.get("population", "")
    mouseStrains = csvSampleDict.get("mouseStrains", "")
    sampleCollectionDate = csvSampleDict.get("sampleCollectionDate", "")
    sampleReceiptDate = csvSampleDict.get("sampleReceiptDate", "")
    dnabarcodeID = csvSampleDict.get("dnabarcode", "")
    sampleName = sampleDisplayedName.replace(" ", "_")
    samplePK = csvSampleDict.get("sample", "")
    sample_kwargs = {
        "displayedName": sampleDisplayedName,
        "status": "created",
        "description": sampleDescription,
        "date": timezone.now(),
    }
    isCreated = None
    if samplePK:
        sample = Sample.objects.get(pk=samplePK)
    else:
        sample, isCreated = Sample.objects.get_or_create(
            name=sampleName, externalId=sampleExtId, defaults=sample_kwargs
        )

    if isCreated:
        logger.debug(
            "import_sample_processor._create_sampleSetItem() new sample created for sample=%s; id=%d"
            % (sampleDisplayedName, sample.id)
        )
    else:
        if sample.description != sampleDescription:
            sample.description = sampleDescription
            sample.save()

            logger.debug(
                "import_sample_processor._create_sampleSetItem() just updated sample description for sample=%s; id=%d"
                % (sampleDisplayedName, sample.id)
            )

    logger.debug(
        "import_sample_processor._create_sampleSetItem() going to create sampleSetItem for sample=%s; sampleSetId=%s in sampleSet_ids=%s"
        % (sampleDisplayedName, str(sampleSet_id), sampleSet_id)
    )

    currentDateTime = timezone.now()

    dnabarcode = None
    if barcodeKit and barcodeAssignment:
        dnabarcode = models.dnaBarcode.objects.get(
            name__iexact=barcodeKit, id_str__iexact=barcodeAssignment
        )
    elif dnabarcodeID:
        dnabarcode = models.dnaBarcode.objects.get(id=dnabarcodeID)

    pcrPlateColumn = "1" if pcrPlateRow else ""

    sampleSetItem_kwargs = {
        "gender": sampleGender,
        "relationshipRole": relationshipRole,
        "relationshipGroup": sampleGroup,
        "cancerType": cancerType,
        "cellularityPct": cellularityPct if cellularityPct else None,
        "biopsyDays": int(biopsyDays) if biopsyDays else 0,
        "cellNum": cellNum,
        "coupleId": coupleId,
        "embryoId": embryoId,
        "creator": user,
        "creationDate": currentDateTime,
        "lastModifiedUser": user,
        "lastModifiedDate": currentDateTime,
        "description": sampleDescription,
        "controlType": sampleControlType,
        "assayGroup": assayGroup,
        "sampleSource": sampleSource,
        "panelPoolType": panelPoolType,
        "tubePosition": tubePosition,
        "population": population,
        "mouseStrains": mouseStrains,
        "sampleCollectionDate": sampleCollectionDate,
        "sampleReceiptDate": sampleReceiptDate,
    }

    item_id = csvSampleDict.get("id", "")
    if item_id:
        sampleSetItem = get_object_or_404(SampleSetItem, pk=item_id)
        sampleSetItem_kwargs["pcrPlateRow"] = pcrPlateRow
        sampleSetItem_kwargs["dnabarcode"] = dnabarcode
        for field, value in sampleSetItem_kwargs.items():
            setattr(sampleSetItem, field, value)

        logger.debug(
            "import_sample_processor._create_sampleSetItem() sampleSetItem_kwargs=%s"
            % (sampleSetItem_kwargs)
        )
        sampleSetItem.save()
    else:
        sampleSetItem, isCreated = SampleSetItem.objects.get_or_create(
            sample=sample,
            sampleSet_id=sampleSet_id,
            description=sampleDescription,
            nucleotideType=nucleotideType,
            dnabarcode=dnabarcode,
            sampleSource=sampleSource,
            panelPoolType=panelPoolType,
            pcrPlateColumn=pcrPlateColumn,
            pcrPlateRow=pcrPlateRow,
            defaults=sampleSetItem_kwargs,
        )
    logger.debug(
        "import_sample_processor._create_sampleSetItem() after get_or_create isCreated=%s; sampleSetItem=%s; samplesetItem.id=%d"
        % (str(isCreated), sampleDisplayedName, sampleSetItem.id)
    )

    ssi_sid = transaction.savepoint()

    return sample, sampleSetItem, ssi_sid


def _create_sampleAttributeValue(csvSampleDict, request, user, sample):
    """
    save sample customer attribute value to db.

    """
    customAttributes = SampleAttribute.objects.filter(isActive=True)
    currentDateTime = timezone.now()  ##datetime.datetime.now()

    for attribute in customAttributes:
        newValue = None

        if attribute.displayedName not in list(csvSampleDict.keys()):

            # add mandatory custom attributes for an imported sample if user has not added it
            if attribute.isMandatory:
                if attribute.dataType and attribute.dataType.dataType == "Integer":
                    newValue = "0"
                else:
                    newValue = ""
        else:
            newValue = csvSampleDict.get(attribute.displayedName, "")

        if newValue is None:
            logger.debug(
                "import_sample_processor._create_sampleAttributeValue SKIPPING due to NO VALUE for attribute=%s;"
                % (attribute.displayedName)
            )
        else:
            logger.debug(
                "import_sample_processor._create_sampleAttributeValue going to get_or_create sample=%s; attribute=%s; value=%s"
                % (sample.displayedName, attribute.displayedName, newValue)
            )

            sampleAttributeValues = SampleAttributeValue.objects.filter(
                sample=sample, sampleAttribute=attribute
            )

            if sampleAttributeValues:
                sampleAttributeValue = sampleAttributeValues[0]

                # logger.debug("import_sample_processor._create_sampleAttributeValue ORIGINAL VALUE pk=%s; sample=%s; attribute=%s; orig value=%s" %(sampleAttributeValue.id, sample.displayedName, attribute.displayedName, sampleAttributeValue.value))

                # there should only be 1 attribute value for each sample/attribute pair if the old entry has value but the new import doesn't, do not override it.
                if newValue:
                    sampleAttributeValue_kwargs = {
                        "value": newValue,
                        "lastModifiedUser": user,
                        "lastModifiedDate": currentDateTime,
                    }

                    for field, value in sampleAttributeValue_kwargs.items():
                        setattr(sampleAttributeValue, field, value)

                    sampleAttributeValue.save()

                    # logger.debug("import_sample_processor._create_sampleAttributeValue UPDATED pk=%s; sample=%s; attribute=%s; newValue=%s" %(sampleAttributeValue.id, sample.displayedName, attribute.displayedName, newValue))

                else:
                    # logger.debug("import_sample_processor._create_sampleAttributeValue going to DELETE pk=%s; sample=%s; attribute=%s; newValue=%s" %(sampleAttributeValue.id, sample.displayedName, attribute.displayedName, newValue))

                    sampleAttributeValue.delete()
            else:
                # create a record only there is a value
                if newValue:
                    sampleAttributeValue_kwargs = {
                        "sample": sample,
                        "sampleAttribute": attribute,
                        "value": newValue,
                        "creator": user,
                        "creationDate": currentDateTime,
                        "lastModifiedUser": user,
                        "lastModifiedDate": currentDateTime,
                    }

                    sampleAttributeValue = SampleAttributeValue(
                        **sampleAttributeValue_kwargs
                    )
                    sampleAttributeValue.save()

                    logger.debug(
                        "import_sample_processor._create_sampleAttributeValue CREATED sampleAttributeValue.pk=%d; sample=%s; attribute=%s; newValue=%s"
                        % (
                            sampleAttributeValue.pk,
                            sample.displayedName,
                            attribute.displayedName,
                            newValue,
                        )
                    )
    siv_sid = transaction.savepoint()
    return siv_sid


def validate_csv_sample(csvSampleDict, request):
    """
    validate csv contents and convert user input to raw data to prepare for sample persistence
    returns: a collection of error messages if errors found and whether to skip the row
    """
    failed = []
    isToSkipRow = False
    isToAbort = False

    logger.debug(
        "ENTER import_sample_processor.validate_csv_sample() csvSampleDict=%s; "
        % (csvSampleDict)
    )
    try:

        sampleDisplayedName = csvSampleDict.get(COLUMN_SAMPLE_NAME, "").strip()
        sampleExtId = csvSampleDict.get(COLUMN_SAMPLE_EXT_ID, "").strip()
        sampleControlType = csvSampleDict.get(COLUMN_CONTROLTYPE, "").strip()
        sampleGender = csvSampleDict.get(COLUMN_GENDER, "").strip()
        sampleGroupType = csvSampleDict.get(COLUMN_GROUP_TYPE, "").strip()
        sampleGroup = csvSampleDict.get(COLUMN_GROUP, "").strip()
        if not sampleGroup:
            sampleGroup = None
        sampleDescription = csvSampleDict.get(COLUMN_SAMPLE_DESCRIPTION, "").strip()
        barcodeKit = csvSampleDict.get(COLUMN_BARCODE_KIT, "")
        barcodeAssignment = csvSampleDict.get(COLUMN_BARCODE, "")

        nucleotideType = csvSampleDict.get(COLUMN_NUCLEOTIDE_TYPE, "").strip()

        cancerType = csvSampleDict.get(COLUMN_CANCER_TYPE, "").strip()
        cellularityPct = csvSampleDict.get(COLUMN_CELLULARITY_PCT, None).strip()
        if not cellularityPct:
            cellularityPct = None
        pcrPlateRow = csvSampleDict.get(COLUMN_PCR_PLATE_POSITION, "").strip()

        biopsyDays = csvSampleDict.get(COLUMN_BIOPSY_DAYS, "0").strip()
        cellNum = csvSampleDict.get(COLUMN_CELL_NUM, "").strip()
        coupleId = csvSampleDict.get(COLUMN_COUPLE_ID, "").strip()
        embryoId = csvSampleDict.get(COLUMN_EMBRYO_ID, "").strip()
        sampleSource = csvSampleDict.get(COLUMN_SAMPLE_SOURCE, "").strip()
        panelPoolType = csvSampleDict.get(COLUMN_PANEL_POOL_TYPE, "").strip()
        sampleCollectionDate = csvSampleDict.get(
            COLUMN_SAMPLE_COLLECTION_DATE, ""
        ).strip()
        sampleReceiptDate = csvSampleDict.get(COLUMN_SAMPLE_RECEIPT_DATE, "").strip()
        population = csvSampleDict.get(COLUMN_SAMPLE_POPULATION, "").strip()
        mouseStrains = csvSampleDict.get(COLUMN_SAMPLE_MOUSE_STRAINS, "").strip()

        # Trim off barcode and barcode kit leading and trailing spaces and update the log file if exists
        if (len(barcodeKit) - len(barcodeKit.lstrip())) or (
            len(barcodeKit) - len(barcodeKit.rstrip())
        ):
            logger.warning(
                "The BarcodeKitName(%s) contains Leading/Trailing spaces and got trimmed."
                % barcodeKit
            )

        if (len(barcodeAssignment) - len(barcodeAssignment.lstrip())) or (
            len(barcodeAssignment) - len(barcodeAssignment.rstrip())
        ):
            logger.warning(
                "The BarcodeName (%s) of BarcodeKitName(%s) contains Leading/Trailing spaces and got trimmed."
                % (barcodeAssignment, barcodeKit)
            )

        barcodeKit = barcodeKit.strip()
        barcodeAssignment = barcodeAssignment.strip()

        # skip blank line
        hasAtLeastOneValue = bool([v for v in list(csvSampleDict.values()) if v != ""])
        if not hasAtLeastOneValue:
            isToSkipRow = True
            return failed, isToSkipRow, isToAbort

        isValid, errorMessage = sample_validator.validate_sampleDisplayedName(
            sampleDisplayedName, field_label=COLUMN_SAMPLE_NAME
        )
        if not isValid:
            failed.append((COLUMN_SAMPLE_NAME, errorMessage))

        isValid, errorMessage = sample_validator.validate_sampleExternalId(
            sampleExtId, field_label=COLUMN_SAMPLE_EXT_ID
        )
        if not isValid:
            failed.append((COLUMN_SAMPLE_EXT_ID, errorMessage))

        isValid, errorMessage = sample_validator.validate_sampleDescription(
            sampleDescription, field_label=COLUMN_SAMPLE_DESCRIPTION
        )
        if not isValid:
            failed.append((COLUMN_SAMPLE_DESCRIPTION, errorMessage))

        isValid, errorMessage, gender_CV_value = sample_validator.validate_sampleGender(
            sampleGender, field_label=COLUMN_GENDER
        )
        if not isValid:
            failed.append((COLUMN_GENDER, errorMessage))

        isValid, errorMessage, role_CV_value = sample_validator.validate_sampleGroupType(
            sampleGroupType, field_label=COLUMN_GROUP_TYPE
        )
        if not isValid:
            failed.append((COLUMN_GROUP_TYPE, errorMessage))

        if sampleGroup:
            isValid, errorMessage = sample_validator.validate_sampleGroup(
                sampleGroup, field_label=COLUMN_GROUP
            )
            if not isValid:
                failed.append((COLUMN_GROUP, errorMessage))

        if cancerType:
            isValid, errorMessage, cancerType_CV_value = sample_validator.validate_cancerType(
                cancerType, field_label=COLUMN_CANCER_TYPE
            )
            if not isValid:
                failed.append((COLUMN_CANCER_TYPE, errorMessage))

        if cellularityPct:
            isValid, errorMessage = sample_validator.validate_cellularityPct(
                cellularityPct, field_label=COLUMN_CELLULARITY_PCT
            )
            if not isValid:
                failed.append((COLUMN_CELLULARITY_PCT, errorMessage))

        if pcrPlateRow:
            isValid, errorMessage, pcrPlateRow_internal_value = sample_validator.validate_pcrPlateRow(pcrPlateRow,
                                                                                                      field_label=COLUMN_PCR_PLATE_POSITION)
            if not isValid:
                failed.append((COLUMN_PCR_PLATE_POSITION, errorMessage))

        if nucleotideType:
            isValid, errorMessage, nucleotideType_internal_value = sample_validator.validate_nucleotideType(
                nucleotideType, field_label=COLUMN_NUCLEOTIDE_TYPE
            )
            if not isValid:
                failed.append((COLUMN_NUCLEOTIDE_TYPE, errorMessage))

        if biopsyDays:
            isValid, errorMessage = sample_validator.validate_sampleBiopsyDays(
                biopsyDays, field_label=COLUMN_BIOPSY_DAYS
            )
            if not isValid:
                failed.append((COLUMN_BIOPSY_DAYS, errorMessage))

        if cellNum:
            isValid, errorMessage = sample_validator.validate_sampleCellNum(
                cellNum, field_label=COLUMN_CELL_NUM
            )
            if not isValid:
                failed.append((COLUMN_CELL_NUM, errorMessage))

        if coupleId:
            isValid, errorMessage = sample_validator.validate_sampleCoupleId(
                coupleId, field_label=COLUMN_COUPLE_ID
            )
            if not isValid:
                failed.append((COLUMN_COUPLE_ID, errorMessage))

        if embryoId:
            isValid, errorMessage = sample_validator.validate_sampleEmbryoId(
                embryoId, field_label=COLUMN_EMBRYO_ID
            )
            if not isValid:
                failed.append((COLUMN_EMBRYO_ID, errorMessage))

        if sampleControlType:
            isValid, errorMessage, controlType_CV_value = sample_validator.validate_controlType(
                sampleControlType, field_label=COLUMN_CONTROLTYPE
            )
            if not isValid:
                failed.append((COLUMN_CONTROLTYPE, errorMessage))
        if population:
            isValid, errorMessage, _ = sample_validator.validate_population(population)
            if not isValid:
                failed.append((COLUMN_SAMPLE_POPULATION, errorMessage))
        if mouseStrains:
            isValid, errorMessage, _ = sample_validator.validate_mouseStrains(
                mouseStrains
            )
            if not isValid:
                failed.append((COLUMN_SAMPLE_MOUSE_STRAINS, errorMessage))
        if sampleSource:
            isValid, errorMessage, _ = sample_validator.validate_sampleSource(
                sampleSource
            )
            if not isValid:
                failed.append((COLUMN_SAMPLE_SOURCE, errorMessage))
        if panelPoolType:
            isValid, errorMessage, _ = sample_validator.validate_panelPoolType(
                panelPoolType
            )
            if not isValid:
                failed.append((COLUMN_PANEL_POOL_TYPE, errorMessage))
        if sampleCollectionDate:
            isValid, errorMessage = sample_validator.validate_sampleCollectionDate(
                sampleCollectionDate
            )
            if not isValid:
                failed.append((COLUMN_SAMPLE_COLLECTION_DATE, errorMessage))
        if sampleReceiptDate:
            isValid, errorMessage = sample_validator.validate_sampleReceiptDate(
                sampleReceiptDate, sampleCollectionDate
            )
            if not isValid:
                failed.append((COLUMN_SAMPLE_RECEIPT_DATE, errorMessage))

        # NEW VALIDATION FOR BARCODEKIT AND BARCODE_ID_STR
        isValid, errorMessage, item = sample_validator.validate_barcodekit_and_id_str(
            barcodeKit,
            barcodeAssignment,
            barcodeKit_label=COLUMN_BARCODE_KIT,
            barcode_id_str_label=COLUMN_BARCODE,
        )  # TODO: i18n
        if not isValid:
            if item == "barcodeKit":
                failed.append((COLUMN_BARCODE_KIT, errorMessage))
            else:
                failed.append((COLUMN_BARCODE, errorMessage))
        # if not isValid:
        #     failed.append((COLUMN_BARCODE, errorMessage))

        # validate user-defined custom attributes
        failed_userDefined = _validate_csv_user_defined_attributes(
            csvSampleDict, request
        )
        failed.extend(failed_userDefined)

        logger.debug(
            "import_sample_processor.validate_csv_sample() failed=%s" % (failed)
        )

        return failed, isToSkipRow, isToAbort

    except Exception:
        logger.exception(format_exc())
        failed.append(
            (
                "File Contents",
                " the CSV file does not seem to have all the columns. Click the Sample File Format button for an example. ",
            )
        )  # TODO: i18n
        logger.debug(
            "import_sample_processor.validate_csv_sample() failed=%s" % (failed)
        )
        return failed, isToSkipRow, True


def _validate_csv_user_defined_attributes(csvSampleDict, request):
    failed = []

    customAttributes = SampleAttribute.objects.filter(isActive=True)

    for attribute in customAttributes:
        newValue = None

        if attribute.displayedName not in list(csvSampleDict.keys()):
            # add mandatory custom attributes for an imported sample if user has not added it
            if attribute.isMandatory:
                failed.append(
                    (
                        attribute.displayedName,
                        validation.required_error(
                            attribute.displayedName, include_error_prefix=True
                        ),
                    )
                )
        else:
            newValue = csvSampleDict.get(attribute.displayedName, "").strip()

        if not newValue:
            continue

        if attribute.dataType and attribute.dataType.dataType == "Integer":
            isValid, errorMessage = sample_validator._validate_intValue(
                newValue, attribute.displayedName
            )
            if not isValid:
                failed.append((attribute.displayedName, errorMessage))
        if attribute.dataType and attribute.dataType.dataType == "Text":
            if not validation.is_valid_length(
                newValue, sample_validator.MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE
            ):
                errorMessage = validation.invalid_length_error(
                    attribute.displayedName,
                    sample_validator.MAX_LENGTH_SAMPLE_ATTRIBUTE_VALUE,
                    newValue,
                    include_error_prefix=True,
                )
                failed.append((attribute.displayedName, errorMessage))

    logger.debug(
        "import_sample_processor._validate_csv_user_defined_attributes() failed=%s"
        % (failed)
    )

    return failed


def validate_barcodes_are_unique(row_list):
    barcodekits = []
    dnabarcodes = []
    msgs = dict()
    count = 0
    for row in row_list:
        count += 1
        barcodeKit = row.get(COLUMN_BARCODE_KIT, "").strip()
        barcode = row.get(COLUMN_BARCODE, "").strip()
        if barcodeKit and barcode:
            barcodekits.append(barcodeKit)
            if len(set(barcodekits)) > 1:
                msgs[count] = (
                    COLUMN_BARCODE_KIT,
                    validation.format(
                        ugettext_lazy(
                            "samplesets.samplesetitem.messages.validate.barcodekit.notequal"
                        ),
                        include_error_prefix=True,
                    ),
                )

            dnabarcode = models.dnaBarcode.objects.filter(
                name__iexact=barcodeKit, id_str__iexact=barcode
            )
            if dnabarcode.count() != 1:
                # choices = cvs.order_by('value').values_list('value', flat=True)
                # validation.invalid_choice(field_label, cvValue, choices, include_error_prefix=True)
                msgs[count] = (
                    COLUMN_BARCODE_KIT,
                    validation.format(
                        ugettext_lazy(
                            "samplesets.samplesetitem.messages.validate.barcodekit.invalid_choice"
                        ),
                        {
                            "barcode": barcode,
                            "barcodeSet": barcodeKit,
                            "barcodeSetLabel": COLUMN_BARCODE_KIT,
                            "barcodeLabel": COLUMN_BARCODE,
                        },
                        include_error_prefix=True,
                    ),
                )  # "Error, %s %s is an invalid barcodeKit and barcode combination" % (barcodeKit, barcode)

            if dnabarcode.count() > 0:
                if dnabarcode[0] in dnabarcodes:
                    msgs[count] = (
                        COLUMN_BARCODE,
                        validation.format(
                            ugettext_lazy(
                                "samplesets.samplesetitem.messages.validate.barcode.unique"
                            ),
                            {"barcode": barcode},
                            include_error_prefix=True,
                        ),
                    )  # "Error, A barcode can be assigned to only one sample in the sample set and %s has been assigned to another sample" % (barcode)
                else:
                    dnabarcodes.append(dnabarcode[0])

    return msgs


def validate_barcodes_for_existing_samples(row_list, sampleset_ids):
    msgs = dict()
    count = 0
    for row in row_list:
        count += 1
        barcode = row.get(COLUMN_BARCODE, "").strip()

        if barcode:
            dnabarcode = models.dnaBarcode.objects.filter(id_str__iexact=barcode)
            barcodeKit = dnabarcode[0].name
            row[COLUMN_BARCODE] = dnabarcode[0].id_str
        else:
            barcode = None
            barcodeKit = None

        for sampleset_id in sampleset_ids:

            sampleset = models.SampleSet.objects.get(pk=sampleset_id)
            samplesetitems = sampleset.samples.all()

            for item in samplesetitems:
                # first validate that all barcode kits are the same for all samples
                if item.dnabarcode:
                    barcodeKit1 = item.dnabarcode.name
                    barcode1 = item.dnabarcode.id_str
                else:
                    barcodeKit1 = None
                    barcode1 = None

                if barcodeKit and barcodeKit1 and barcodeKit != barcodeKit1:
                    msgs[count] = (
                        COLUMN_BARCODE,
                        validation.format(
                            ugettext_lazy(
                                "samplesets.samplesetitem.messages.validate.barcodekit.notequal"
                            ),
                            include_error_prefix=True,
                        ),
                    )

                # next validate that all barcodes are unique per each sample
                if barcode and barcode1 and barcode == barcode1:
                    msgs[count] = (
                        COLUMN_BARCODE,
                        validation.format(
                            ugettext_lazy(
                                "samplesets.samplesetitem.messages.validate.barcode.unique"
                            ),
                            {"barcode": barcode},
                            include_error_prefix=True,
                        ),
                    )

    return msgs


def validate_pcrPlateRow_for_existing_samples(row_list, sampleset_ids):
    """
    Validates for uniqueness of PCR plate position and mandatory input for AmpliSeqOnChef sample sets
    """

    msgs = dict()
    count = 0

    for row in row_list:
        count += 1
        pcrPlateRow = row.get(COLUMN_PCR_PLATE_POSITION, "").strip()

        for sampleset_id in sampleset_ids:

            sampleset = models.SampleSet.objects.get(pk=sampleset_id)
            samplesetitems = sampleset.samples.all()

            # validate when adding samples to pre-existing sampleset
            if samplesetitems.count():
                for item in samplesetitems:
                    pcrPlateRow1 = item.pcrPlateRow

                    item_id = item.pk

                    # ensure only 1 pcr plate position per sample
                    if (
                        pcrPlateRow
                        and pcrPlateRow1
                        and pcrPlateRow.upper() == pcrPlateRow1.upper()
                    ):
                        isValid = False
                        errorMessage = validation.format(
                            ugettext_lazy(
                                "samplesets.samplesetitem.messages.validate.pcrPlateRow.unique"
                            ),
                            message_params={"pcrPlateRow": pcrPlateRow.upper()},
                            include_error_prefix=True,
                        )
                        msgs[count] = (COLUMN_PCR_PLATE_POSITION, errorMessage)

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
                        )
                        msgs[count] = (COLUMN_PCR_PLATE_POSITION, errorMessage)
            else:
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
                    )
                    msgs[count] = (COLUMN_PCR_PLATE_POSITION, errorMessage)

    return msgs


def validate_pcrPlateRow_are_unique(row_list):
    positions = []
    msgs = dict()
    count = 0
    for row in row_list:
        count += 1
        pcrPlateRow = row.get(COLUMN_PCR_PLATE_POSITION, "").strip().upper()

        if pcrPlateRow:
            if pcrPlateRow in positions:
                msgs[count] = (
                    COLUMN_PCR_PLATE_POSITION,
                    validation.format(
                        ugettext_lazy(
                            "samplesets.samplesetitem.messages.validate.pcrPlateRow.unique"
                        ),
                        {"pcrPlateRow": pcrPlateRow},
                        include_error_prefix=True,
                    ),
                )
            else:
                positions.append(pcrPlateRow)

    return msgs


def get_sample_csv_version():
    systemCSV_version = settings.SAMPLE_CSV_VERSION
    return [COLUMN_SAMPLE_CSV_VERSION, systemCSV_version]
