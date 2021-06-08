# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import json

from django.template import RequestContext
from django.shortcuts import get_object_or_404

from django.utils import timezone
from traceback import format_exc

from django.utils.translation import ugettext_lazy
from collections import OrderedDict

from iondb.rundb.models import (
    Sample,
    SampleSet,
    SampleSetItem,
    SampleAttribute,
    SampleGroupType_CV,
    SampleAttributeDataType,
    SampleAttributeValue,
    SamplePrepData,
    dnaBarcode,
    SampleAnnotation_CV,
    common_CV,
    KitInfo, ChefPcrPlateconfig)

from django.contrib.auth.models import User
from iondb.rundb.sample import sample_validator
from iondb.utils import validation
from datetime import datetime
import types
import logging

logger = logging.getLogger(__name__)


def validate_for_existing_samples(
    pending_sampleSetItem_list, sampleSet_ids, isEdit_amp_sampleSet=None
):
    """ validate new samples as a group for each selected SampleSet
        return a boolean isValid and a list of text error messages

        Note: validate functions this calls must be able to handle a mixed type list:
            SampleSetItem for for existing samples and dictionary for new samples
    """
    errors = []
    if not pending_sampleSetItem_list and not isEdit_amp_sampleSet:
        return True, errors, {}, pending_sampleSetItem_list

    for sampleset_id in sampleSet_ids:
        # validate each new item against a complete set of samples in this sampleSet plus all pending samples
        sampleset = SampleSet.objects.get(pk=sampleset_id)
        samplesetitems = list(sampleset.samples.all())
        if pending_sampleSetItem_list:
            samplesetitems.extend(pending_sampleSetItem_list)

        isValid, errorMessage, categoryDict = sample_validator.validate_inconsistent_ampliseq_HD_category(
            samplesetitems, pending_sampleSetItem_list, sampleset
        )
        if not isValid:
            errors.extend(
                [sampleset.displayedName + ": " + err for err in errorMessage]
            )

        for pending_item in pending_sampleSetItem_list:
            # validate barcoding is consistent between multiple samples
            isValid, errorMessage = sample_validator.validate_barcoding_samplesetitems(
                samplesetitems,
                pending_item.get("barcodeKit"),
                pending_item.get("barcode"),
                pending_item.get("pending_id", ""),
            )

            if not isValid:
                errorMessage = sampleset.displayedName + ": " + errorMessage
                errors.append(errorMessage)

            """
            # validate PCR Plate position
            isValid, errorMessage = sample_validator.validate_pcrPlate_position_samplesetitems(samplesetitems,
                    pending_item.get('pcrPlateRow', ""), pending_item.get('pending_id', ""), sampleset)

            if not isValid:
                errorMessage = sampleset.displayedName + ": " + errorMessage
                errors.append(errorMessage)
            """
    return len(errors) == 0, errors, categoryDict, samplesetitems


def create_or_update_sampleSet(queryDict, user, sampleSet_id=None):
    sampleSetName = queryDict.get("sampleSetName", "").strip()
    if not sampleSetName and not sampleSet_id:
        return True, None, None

    kwargs = {
        "displayedName": sampleSetName,
        "description": queryDict.get("sampleSetDescription", "").strip(),
        "SampleGroupType_CV_id": queryDict.get("groupType"),
        "libraryPrepType": queryDict.get("libraryPrepType", "").strip(),
        "libraryPrepKitName": queryDict.get("libraryPrepKit", "").strip(),
        "pcrPlateSerialNum": queryDict.get("pcrPlateSerialNum", "").strip(),
        "libraryPrepProtocol": queryDict.get("libraryPrepProtocol", "").strip(),
        "additionalCycles": queryDict.get("additionalCycles", "").strip(),
        "libraryPrepInstrument": "chef" if "chef" in queryDict.get("libraryPrepType", "") else "",
        "lastModifiedUser": user,
        "lastModifiedDate": timezone.now(),
        "categories": "",
    }

    if kwargs["SampleGroupType_CV_id"] == "0" or kwargs["SampleGroupType_CV_id"] == 0:
        kwargs["SampleGroupType_CV_id"] = None

    # copy any categories to SampleSet
    if kwargs["libraryPrepProtocol"]:
        kwargs["categories"] = common_CV.objects.get(
            value=kwargs["libraryPrepProtocol"]
        ).categories

    if sampleSet_id:
        isNew = False
        sampleSet = get_object_or_404(SampleSet, pk=sampleSet_id)
        kwargs["status"] = sampleSet.status or "created"
        kwargs["libraryPrepInstrumentData"] = sampleSet.libraryPrepInstrumentData
    else:
        isNew = True
        sampleSet = SampleSet()
        kwargs["status"] = "created"
        kwargs["libraryPrepInstrumentData"] = None
        kwargs["creator"] = user

    # handle Ampliseq on Chef
    if kwargs["libraryPrepInstrument"] == "chef":
        if kwargs["status"] == "created":
            kwargs["status"] = "libPrep_pending"
        if not kwargs["libraryPrepInstrumentData"]:
            kwargs["libraryPrepInstrumentData"] = SamplePrepData.objects.create(
                samplePrepDataType="lib_prep"
            )
    else:
        if kwargs["status"] == "libPrep_pending":
            kwargs["status"] = "created"
        if kwargs["libraryPrepInstrumentData"]:
            kwargs["libraryPrepInstrumentData"].delete()

    # validate sampleSet parameters
    isValid, errorMessage = sample_validator.validate_sampleSet_values(
        kwargs["displayedName"],
        kwargs["description"],
        kwargs["pcrPlateSerialNum"],
        isNew,
        additionalCycles=kwargs["additionalCycles"],
        libraryPrepProtocol=kwargs["libraryPrepProtocol"],
        libraryPrepKit=kwargs["libraryPrepKitName"],
    )
    if errorMessage:
        return isValid, errorMessage, None

    # create SampleSet
    for field, value in kwargs.items():
        setattr(sampleSet, field, value)
    sampleSet.save()

    return True, None, sampleSet.id


def _create_or_update_sample_for_sampleSetItem(queryDict, user, sampleSetItem_id):
    """ create or update sample values when editing existing sampleSetItem
        this function also handles renaming
    """
    currentDateTime = timezone.now()

    sampleDisplayedName = queryDict.get("displayedName", "")
    sampleExternalId = queryDict.get("externalId", "")
    sampleDesc = queryDict.get("description", "")

    orig_sampleSetItem = get_object_or_404(SampleSetItem, pk=sampleSetItem_id)
    orig_sample = orig_sampleSetItem.sample

    if (
        orig_sample.displayedName == sampleDisplayedName
        and orig_sample.externalId == sampleExternalId.strip()
    ):

        new_sample = orig_sample

        if new_sample.description != sampleDesc:
            new_sample.description = sampleDesc
            new_sample.date = currentDateTime
            new_sample.save()
    else:
        # link the renamed sample to an existing sample if one is found. Otherwise, rename the sample only if the sample has not yet been planned.
        existingSamples = Sample.objects.filter(
            displayedName=sampleDisplayedName, externalId=sampleExternalId
        )

        canSampleBecomeOrphan = (
            orig_sample.sampleSets.count() < 2
        ) and orig_sample.experiments.count() == 0
        logger.info(
            "views_helper - _create_or_update_sample_for_sampleSetItem - #3 can sample becomes ORPHAN? orig_sample.id=%d; orig_sample.name=%s; canSampleBecomeOrphan=%s"
            % (orig_sample.id, orig_sample.displayedName, str(canSampleBecomeOrphan))
        )

        if existingSamples.count() > 0:
            orig_sample_id = orig_sample.id

            # by sample uniqueness rule, there should only be 1 existing sample max
            existingSample = existingSamples[0]
            existingSample.description = sampleDesc
            orig_sampleSetItem.sample = existingSample
            orig_sampleSetItem.lastModifiedUser = user
            orig_sampleSetItem.lastModifiedDate = currentDateTime
            orig_sampleSetItem.save()

            new_sample = existingSample

            logger.debug(
                "views_helper - _create_or_update_sample_for_sampleSetItem - #4 SWITCH TO EXISTING SAMPLE sampleSetItem.id=%d; existingSample.id=%d"
                % (orig_sampleSetItem.id, existingSample.id)
            )

            # cleanup if the replaced sample is not being used anywhere
            if canSampleBecomeOrphan:
                logger.debug(
                    "views_helper - _create_or_update_sample_for_sampleSetItem - #5 AFTER SWITCH orig_sample becomes ORPHAN! orig_sample.id=%d; orig_sample.name=%s"
                    % (orig_sample.id, orig_sample.displayedName)
                )
                orig_sample.delete()
            else:
                logger.debug(
                    "views_helper - _create_or_update_sample_for_sampleSetItem - #6 AFTER SWITCH orig_sample is still NOT ORPHAN YET! orig_sample.id=%d; orig_sample.name=%s sample.sampleSets.count=%d; sample.experiments.count=%d; "
                    % (
                        orig_sample.id,
                        orig_sample.displayedName,
                        orig_sample.sampleSets.count(),
                        orig_sample.experiments.count(),
                    )
                )
        else:
            name = sampleDisplayedName.replace(" ", "_")

            if canSampleBecomeOrphan:
                # update existing sample record
                sample_kwargs = {
                    "name": name,
                    "displayedName": sampleDisplayedName,
                    "description": sampleDesc,
                    "externalId": sampleExternalId,
                    "description": sampleDesc,
                    "date": currentDateTime,
                }
                for field, value in sample_kwargs.items():
                    setattr(orig_sample, field, value)

                orig_sample.save()

                logger.debug(
                    "views_helper - _create_or_update_sample_for_sampleSetItem - #7 RENAME SAMPLE sampleSetItem.id=%d; sample.id=%d"
                    % (orig_sampleSetItem.id, orig_sample.id)
                )
                new_sample = orig_sample
            else:
                # create a new sample record
                sample_kwargs = {
                    "displayedName": sampleDisplayedName,
                    "description": sampleDesc,
                    "status": "created",
                    "date": currentDateTime,
                }

                sample = Sample.objects.get_or_create(
                    name=name, externalId=sampleExternalId, defaults=sample_kwargs
                )[0]

                orig_sampleSetItem.sample = sample
                orig_sampleSetItem.save()

                logger.debug(
                    "views_helper - _create_or_update_sample_for_sampleSetItem - #8 CREATE NEW SAMPLE sampleSetItem.id=%d; sample.id=%d"
                    % (orig_sampleSetItem.id, sample.id)
                )
                new_sample = sample

    return new_sample


def _create_or_update_sample(queryDict):
    currentDateTime = timezone.now()
    sample_id = queryDict.get("sample", "")
    sampleName = queryDict.get("name", "")
    sampleDisplayedName = queryDict.get("displayedName", "")
    sampleDesc = queryDict.get("description", "")
    sampleExternalId = queryDict.get("externalId", "")

    if sample_id:
        try:
            sample = Sample.objects.get(pk=sample_id)
            return sample
        except Exception:
            logger.info("No samples exists:views_helper._create_or_update_sample()")

    if not sampleName:
        sampleName = "_".join(sampleDisplayedName.split())

    existingSamples = Sample.objects.filter(
        name=sampleName, externalId=sampleExternalId
    )

    if existingSamples.count() > 0:
        created = False
        sample = existingSamples[0]
        if sample.description != sampleDesc:
            sample.description = sampleDesc
            sample.date = currentDateTime
            sample.save()
    else:
        # create a new sample record
        sample_kwargs = {
            "displayedName": sampleDisplayedName,
            "description": sampleDesc,
            "status": "created",
            "date": currentDateTime,
        }
        sample, created = Sample.objects.get_or_create(
            name=sampleName, externalId=sampleExternalId, defaults=sample_kwargs
        )

    logger.debug(
        "_create_or_update_sample: sample.id=%d name=%s externalId=%s created=%s"
        % (sample.id, sample.name, sample.externalId, created)
    )

    return sample


def _update_input_samples_session_context(request, pending_sampleSetItem, isNew=True):
    logger.debug(
        "views_helper._update_input_samples_session_context pending_sampleSetItem=%s"
        % (pending_sampleSetItem)
    )

    _create_pending_session_if_needed(request)

    if isNew:
        request.session["input_samples"]["pending_sampleSetItem_list"].insert(
            0, pending_sampleSetItem
        )
    else:
        pendingList = request.session["input_samples"]["pending_sampleSetItem_list"]
        hasUpdated = False

        for index, item in enumerate(pendingList):
            if (
                item["pending_id"] == pending_sampleSetItem["pending_id"]
                and not hasUpdated
            ):
                pendingList[index] = pending_sampleSetItem
                hasUpdated = True

        if not hasUpdated:
            request.session["input_samples"]["pending_sampleSetItem_list"].insert(
                0, pending_sampleSetItem
            )

    request.session.modified = True


def _create_pending_session_if_needed(request):
    """
    return or create a session context for entering samples manually to create a sample set
    """
    if "input_samples" not in request.session:

        logger.debug(
            "views_helper._create_pending_session_if_needed() going to CREATE new request.session"
        )

        sampleSet_list = SampleSet.objects.all().order_by(
            "-lastModifiedDate", "displayedName"
        )
        sampleGroupType_list = list(
            SampleGroupType_CV.objects.filter(isActive=True).order_by("displayedName")
        )
        #        custom_sample_column_list = list(SampleAttribute.objects.filter(isActive = True).values_list('displayedName', flat=True).order_by('id'))

        pending_sampleSetItem_list = []

        request.session["input_samples"] = {}
        request.session["input_samples"][
            "pending_sampleSetItem_list"
        ] = pending_sampleSetItem_list

    else:
        logger.debug(
            "views_helper._create_pending_session_if_needed() ALREADY EXIST request.session[input_samples]=%s"
            % (request.session["input_samples"])
        )


def _handle_enter_samples_manually_request(request):
    _create_pending_session_if_needed(request)

    ctxd = _create_context_from_session(request)

    return ctxd


def _create_context_from_session(request):
    #    ctxd = request.session['input_samples'],

    custom_sample_column_list = list(
        SampleAttribute.objects.filter(isActive=True)
        .values_list("displayedName", flat=True)
        .order_by("id")
    )

    ctx = {
        "input_samples": request.session.get("input_samples", {}),
        "custom_sample_column_list": json.dumps(custom_sample_column_list),
    }

    context = RequestContext(request, ctx)

    return context


def parse_sample_kwargs_from_dict(queryDict):
    """ Converts request.POST to dict used by various SampleSet views
    """
    sampleDisplayedName = queryDict.get("sampleDisplayedName", "").strip()
    sampleName = "_".join(sampleDisplayedName.split())
    sampleExternalId = queryDict.get("sampleExternalId", "").strip()
    sampleDescription = queryDict.get("sampleDescription", "").strip()
    sampleCollectionDate = queryDict.get("sampleCollectionDate", "").strip()
    sampleReceiptDate = queryDict.get("sampleReceiptDate", "").strip()
    gender = queryDict.get("gender", "")
    population = queryDict.get("population", "")
    mouseStrains = queryDict.get("mouseStrains", "")
    relationshipRole = queryDict.get("relationshipRole", "")
    relationshipGroup = queryDict.get("relationshipGroup", "") or "0"

    selectedBarcodeKitName = queryDict.get("barcodeKit", "")
    selectedBarcode = queryDict.get("barcode", "")

    controlType = queryDict.get("controlType", "")
    cancerType = queryDict.get("cancerType", "")
    cellularityPct = queryDict.get("cellularityPct") or None
    sampleSource = queryDict.get("sampleSource", "").strip()
    panelPoolType = queryDict.get("panelPoolType", "").strip()
    nucleotideType = queryDict.get("nucleotideType", "").strip()
    if nucleotideType.lower() == "fusions":
        nucleotideType = "rna"

    pcrPlateRow = queryDict.get("pcrPlateRow", "").strip()
    pcrPlateColumn = "1" if pcrPlateRow else ""

    sampleBiopsyDays = queryDict.get("biopsyDays") or None
    sampleCellNum = queryDict.get("cellNum", "")
    sampleCoupleId = queryDict.get("coupleId", "")
    sampleEmbryoId = queryDict.get("embryoId", "")

    kwargs = {
        # sample kwargs
        "name": sampleName,
        "displayedName": sampleDisplayedName,
        "description": sampleDescription,
        "externalId": sampleExternalId,
        # sampleSetItem kwargs
        "sampleCollectionDate": sampleCollectionDate,
        "sampleReceiptDate": sampleReceiptDate,
        "gender": gender,
        "population": population,
        "mouseStrains": mouseStrains,
        "relationshipRole": relationshipRole,
        "relationshipGroup": relationshipGroup,
        "controlType": controlType,
        "cancerType": cancerType,
        "cellularityPct": cellularityPct,
        "nucleotideType": nucleotideType,
        "sampleSource": sampleSource,
        "panelPoolType": panelPoolType,
        "pcrPlateRow": pcrPlateRow,
        "pcrPlateColumn": pcrPlateColumn,
        "biopsyDays": sampleBiopsyDays,
        "cellNum": sampleCellNum,
        "coupleId": sampleCoupleId,
        "embryoId": sampleEmbryoId,
        # DNA barcode
        "barcodeKit": selectedBarcodeKitName,
        "barcode": selectedBarcode,
    }
    return kwargs


def _get_pending_sampleSetItem_id(request):
    return _get_pending_sampleSetItem_count(request) + 1


def _get_pending_sampleSetItem_count(request):
    _create_pending_session_if_needed(request)

    return len(request.session["input_samples"]["pending_sampleSetItem_list"])


def _get_pending_sampleSetItem_by_id(request, _id):

    if _id and "input_samples" in request.session:
        items = request.session["input_samples"]["pending_sampleSetItem_list"]

        for index, item in enumerate(
            request.session["input_samples"]["pending_sampleSetItem_list"]
        ):
            # logger.debug("views_helper._get_pending_sampleSetItem_by_id - item[pending_id]=%s; _id=%s" %(str(item['pending_id']), str(_id)))

            if str(item["pending_id"]) == str(_id):
                return item
        return _id
    else:
        return None


def _create_pending_sampleAttributes_for_sampleSetItem(request):

    sampleAttribute_list = SampleAttribute.objects.filter(isActive=True).order_by("id")

    pending_attributeValue_dict = {}

    new_attributeValue_dict = {}
    for attribute in sampleAttribute_list:
        value = request.POST.get("sampleAttribute|" + str(attribute.id), None)

        if value:
            isValid, errorMessage = sample_validator.validate_sampleAttribute(
                attribute, value.encode("utf8")
            )
            if not isValid:
                return isValid, errorMessage, None
        else:
            isValid, errorMessage = sample_validator.validate_sampleAttribute_mandatory_for_no_value(
                attribute
            )
            if not isValid:
                return isValid, errorMessage, None

        new_attributeValue_dict[attribute.id] = value.encode("utf8") if value else None

    # logger.debug("views_helper._create_pending_sampleAttributes_for_sampleSetItem#1 new_attributeValue_dict=%s" %(str(new_attributeValue_dict)))

    if new_attributeValue_dict:
        for key, newValue in list(new_attributeValue_dict.items()):
            sampleAttribute_objs = SampleAttribute.objects.filter(id=key)

            if sampleAttribute_objs.count() > 0:
                if newValue:
                    pending_attributeValue_dict[
                        sampleAttribute_objs[0].displayedName
                    ] = newValue

    return True, None, pending_attributeValue_dict


def _create_or_update_sampleAttributes_for_sampleSetItem(request, user, sample):
    queryDict = request.POST

    sampleAttribute_list = SampleAttribute.objects.filter(isActive=True).order_by("id")

    new_attributeValue_dict = {}
    for attribute in sampleAttribute_list:
        value = request.POST.get("sampleAttribute|" + str(attribute.id), None)

        if value:
            isValid, errorMessage = sample_validator.validate_sampleAttribute(
                attribute, value.encode("utf8")
            )
            if not isValid:
                return isValid, errorMessage
        else:
            isValid, errorMessage = sample_validator.validate_sampleAttribute_mandatory_for_no_value(
                attribute
            )
            if not isValid:
                return isValid, errorMessage

        new_attributeValue_dict[attribute.id] = value.encode("utf8") if value else None

    logger.debug(
        "views_helper._create_or_update_sampleAttributes_for_sampleSetItem #1 new_attributeValue_dict=%s"
        % (str(new_attributeValue_dict))
    )

    _create_or_update_sampleAttributes_for_sampleSetItem_with_values(
        request, user, sample, new_attributeValue_dict
    )

    return True, None


def _create_or_update_sampleAttributes_for_sampleSetItem_with_dict(
    request, user, sample, sampleAttribute_dict
):
    """
    sampleAttribute_dict has the attribute name be the key
    """

    logger.debug(
        "ENTER views_helper._create_or_update_sampleAttributes_for_sampleSetItem_with_dict - sampleAttribute_dict=%s"
        % (sampleAttribute_dict)
    )

    new_attributeValue_dict = {}

    if sampleAttribute_dict:
        attribute_objs = SampleAttribute.objects.all()
        for attribute_obj in attribute_objs:
            value = sampleAttribute_dict.get(attribute_obj.displayedName, "")
            if value:
                isValid, errorMessage = sample_validator.validate_sampleAttribute(
                    attribute_obj, value.encode("utf8")
                )
                if not isValid:
                    return isValid, errorMessage

                new_attributeValue_dict[attribute_obj.id] = value.encode("utf8")

        logger.debug(
            "views_helper._create_or_update_sampleAttributes_for_sampleSetItem_with_dict - new_attributeValue_dict=%s"
            % (new_attributeValue_dict)
        )

    _create_or_update_sampleAttributes_for_sampleSetItem_with_values(
        request, user, sample, new_attributeValue_dict
    )

    isValid = True
    return isValid, None


def _create_or_update_sampleAttributes_for_sampleSetItem_with_values(
    request, user, sample, new_attributeValue_dict
):
    if new_attributeValue_dict:
        currentDateTime = timezone.now()  ##datetime.datetime.now()

        # logger.debug("views_helper - ENTER new_attributeValue_dict=%s" %(new_attributeValue_dict))

        for key, newValue in list(new_attributeValue_dict.items()):
            sampleAttribute_objs = SampleAttribute.objects.filter(id=key)

            logger.debug(
                "views_helper._create_or_update_sampleAttributes_for_sampleSetItem_with_values() #3 sampleAttribute_objs.count=%d"
                % (sampleAttribute_objs.count())
            )

            if sampleAttribute_objs.count() > 0:
                if newValue:

                    attributeValue_kwargs = {
                        "value": newValue,
                        "creator": user,
                        "creationDate": currentDateTime,
                        "lastModifiedUser": user,
                        "lastModifiedDate": currentDateTime,
                    }
                    attributeValue, isCreated = SampleAttributeValue.objects.get_or_create(
                        sample=sample,
                        sampleAttribute=sampleAttribute_objs[0],
                        defaults=attributeValue_kwargs,
                    )

                    if not isCreated:
                        if attributeValue.value != newValue:
                            attributeValue.value = newValue
                            attributeValue.lastModifiedUser = user
                            attributeValue.lastModifiedDate = currentDateTime

                            attributeValue.save()
                            logger.debug(
                                "views_helper - _create_or_update_sampleAttributes_for_sampleSetItem_with_values - #4 UPDATED!! isCreated=%s attributeValue.id=%d; value=%s"
                                % (str(isCreated), attributeValue.id, newValue)
                            )
                    else:
                        logger.debug(
                            "views_helper - _create_or_update_sampleAttributes_for_sampleSetItem_with_values - #5 existing attributeValue!! attributeValue.id=%d; value=%s"
                            % (attributeValue.id, newValue)
                        )
                else:
                    existingAttributeValues = SampleAttributeValue.objects.filter(
                        sample=sample, sampleAttribute=sampleAttribute_objs[0]
                    )

                    logger.debug(
                        "views_helper._create_or_update_sampleAttributes_for_sampleSetItem_with_values() #6 existingAttributeValues.count=%d"
                        % (existingAttributeValues.count())
                    )

                    if existingAttributeValues.count() > 0:
                        existingAttributeValue = existingAttributeValues[0]
                        existingAttributeValue.value = newValue
                        existingAttributeValue.lastModifiedUser = user
                        existingAttributeValue.lastModifiedDate = currentDateTime

                        existingAttributeValue.save()
                        logger.debug(
                            "views_helper - _create_or_update_sampleAttributes_for_sampleSetItem_with_values - #7 UPDATED with None!! attributeValue.id=%d;"
                            % (attributeValue.id)
                        )


def _dict_fileds_values(queryDict):
    selectedBarcode = queryDict.get("barcode", "").strip()
    selectedBarcodeKitName = queryDict.get("barcodeKit", "").strip()
    dnabarcode = queryDict.get("dnabarcode", None)
    selectedDnaBarcode = None
    if selectedBarcodeKitName and selectedBarcode:
        selectedDnaBarcode = dnaBarcode.objects.get(
            name=selectedBarcodeKitName, id_str=str(selectedBarcode)
        )
    elif dnabarcode:
        selectedDnaBarcode = dnaBarcode.objects.get(id=dnabarcode)

    return {
        "gender": queryDict.get("gender", ""),
        "sampleCollectionDate": queryDict.get("sampleCollectionDate") or None,
        "sampleReceiptDate": queryDict.get("sampleReceiptDate") or None,
        "population": queryDict.get("population", ""),
        "mouseStrains": queryDict.get("mouseStrains", ""),
        "relationshipRole": queryDict.get("relationshipRole", ""),
        "relationshipGroup": queryDict.get("relationshipGroup", "") or "0",
        "description": queryDict.get("description", ""),
        "dnabarcode": selectedDnaBarcode,
        "controlType": queryDict.get("controlType", ""),
        "cancerType": queryDict.get("cancerType", ""),
        "cellularityPct": queryDict.get("cellularityPct") or None,
        "biopsyDays": queryDict.get("biopsyDays") or None,
        "cellNum": queryDict.get("cellNum", ""),
        "coupleId": queryDict.get("coupleId", ""),
        "embryoId": queryDict.get("embryoId", ""),
        "nucleotideType": queryDict.get("nucleotideType", "").strip(),
        "sampleSource": queryDict.get("sampleSource", "").strip(),
        "panelPoolType": queryDict.get("panelPoolType", "").strip(),
    }


def _get_sampleSetItem_kwargs(queryDict, sampleSet_id=None):
    sampleSetItem_kwargs = {}
    if type(queryDict) == types.DictType:
        sampleSetItem_kwargs = _dict_fileds_values(queryDict)

    nucleotideType = sampleSetItem_kwargs.get("nucleotideType")
    sampleCollectionDate = sampleSetItem_kwargs.get("sampleCollectionDate") or None
    sampleReceiptDate = sampleSetItem_kwargs.get("sampleReceiptDate") or None

    if queryDict.get("assayGroup"):
        sampleSetItem_kwargs["assayGroup"] = queryDict["assayGroup"]
    if queryDict.get("pcrPlateRow"):
        sampleSetItem_kwargs["pcrPlateRow"] = queryDict["pcrPlateRow"]
    if queryDict.get("tubePosition"):
        sampleSetItem_kwargs["tubePosition"] = queryDict["tubePosition"]

    if sampleSet_id:
        sampleSet = get_object_or_404(SampleSet, pk=sampleSet_id)

    if nucleotideType and nucleotideType.lower() == "fusions":
        sampleSetItem_kwargs["nucleotideType"] = "rna"
    if sampleCollectionDate:
        sampleSetItem_kwargs["sampleCollectionDate"] = datetime.strptime(
            str(sampleCollectionDate), "%Y-%m-%d"
        ).date()
    if sampleReceiptDate:
        sampleSetItem_kwargs["sampleReceiptDate"] = datetime.strptime(
            str(sampleReceiptDate), "%Y-%m-%d"
        ).date()

    sampleSetItem_kwargs[
        "pcrPlateColumn"
    ] = "1"  # since we assign PCR Plate row dynamically

    return sampleSetItem_kwargs


def _create_or_update_sampleSetItem(
    queryDict, user, item_id=None, sampleSet_id=None, sample=None
):
    # use case 1: if item_id is defined we are editing existing SampleSetItem
    # use case 2: if sampleSet_id is defined we are editing or adding a new SampleSetItem

    sampleSetItem_kwargs = _get_sampleSetItem_kwargs(queryDict, sampleSet_id)
    if item_id:
        sampleSetItem = get_object_or_404(SampleSetItem, pk=item_id)

    elif sampleSet_id:
        sampleSet = get_object_or_404(SampleSet, pk=sampleSet_id)
        try:
            sampleSetItem = SampleSetItem.objects.get(
                sample=sample,
                sampleSet_id=sampleSet_id,
                nucleotideType=sampleSetItem_kwargs["nucleotideType"],
                dnabarcode=sampleSetItem_kwargs["dnaBarcode"],
            )
        except Exception:
            sampleSetItem = SampleSetItem(
                sample=sample,
                sampleSet_id=sampleSet_id,
                nucleotideType=sampleSetItem_kwargs["nucleotideType"],
                creator=user,
            )
    else:
        raise Exception(
            "Unable to update sample: missing SampleSetItem id or SampleSet id"
        )

    for field, value in sampleSetItem_kwargs.items():
        setattr(sampleSetItem, field, value)

    logger.debug(
        "views_helper._create_or_update_sampleSetItem sampleSetItem_kwargs=%s"
        % (sampleSetItem_kwargs)
    )
    sampleSetItem.save()


def _get_nucleotideType_choices(sampleGroupType=""):
    nucleotideType_choices = []
    for internalValue, displayedValue in SampleSetItem.get_nucleotideType_choices():
        if internalValue == "rna" and "fusions" in sampleGroupType.lower():
            continue
        else:
            nucleotideType_choices.append((internalValue, displayedValue))

    if "fusions" in sampleGroupType.lower() or not sampleGroupType:
        nucleotideType_choices.append(("fusions", "Fusions"))

    return nucleotideType_choices


def _get_libraryPrepType_choices(request):
    choices_tuple = SampleSet.ALLOWED_LIBRARY_PREP_TYPES
    choices = OrderedDict()
    active_libraryPrep_list = [
        libPrepObj.value
        for libPrepObj in SampleAnnotation_CV.objects.filter(
            isActive=True, annotationType="libraryPrepType"
        ).order_by("value")
    ]
    for i, (internalValue, displayedValue) in enumerate(choices_tuple):
        if displayedValue in active_libraryPrep_list:
            choices[internalValue] = displayedValue
    return choices


def _get_libraryPrepProtocol_choices(request):
    libraryPrepProtocol_choices = [
        libraryPrepProtocol
        for libraryPrepProtocol in common_CV.objects.filter(
            isActive=True, cv_type="libraryPrepProtocol"
        ).order_by("id")
    ]

    return libraryPrepProtocol_choices


def _get_additionalCycles_choices(request):
    additionalCycles_choices = [
        additionalCycle
        for additionalCycle in SampleAnnotation_CV.objects.filter(
            isActive=True, annotationType="additionalCycles"
        ).order_by("id")
    ]

    return additionalCycles_choices


"""
- Ampliseq HD on chef support
- Refer the detailed design doc on how the tube position and pcr plate is being assigned 
 @ https://confluence.amer.thermo.com/display/TS/Tech+Design+proposal+-+Ampliseq+HD+on+chef+support
"""


def assign_tube_postion_pcr_plates(categoryDict):
    upated_pending_sampleSetItem_list = []
    pcrPlates = []
    all_groups = [group for key, group in categoryDict.items()]
    for group in all_groups:
        for group, items in group.items():
            for item in items:
                # Existing/persisted sample set item is model object when adding items to sampeset,so, type checking
                if type(item) == types.DictType:
                    panelPoolType = item.get("panelPoolType", "")
                    if item.get("pcrPlateRow"):
                        item["pcrPlateRow"] = ""
                    if group == "Group 1":
                        item["assayGroup"] = "group 1"
                        if panelPoolType == "Single Pool":
                            item["tubePosition"] = "A"
                        else:
                            item["tubePosition"] = "A,B"
                        if "A" not in pcrPlates:
                            item["pcrPlateRow"] = "A"
                        elif "B" not in pcrPlates:
                            item["pcrPlateRow"] = "B"
                        elif "C" not in pcrPlates:
                            item["pcrPlateRow"] = "C"
                        elif "D" not in pcrPlates:
                            item["pcrPlateRow"] = "D"
                    elif group == "Group 2":
                        item["assayGroup"] = "group 2"
                        if panelPoolType == "Single Pool":
                            item["tubePosition"] = "C"
                        else:
                            item["tubePosition"] = "C,D"
                        if "E" not in pcrPlates:
                            item["pcrPlateRow"] = "E"
                        elif "F" not in pcrPlates:
                            item["pcrPlateRow"] = "F"
                        elif "G" not in pcrPlates:
                            item["pcrPlateRow"] = "G"
                        elif "H" not in pcrPlates:
                            item["pcrPlateRow"] = "H"
                    pcrPlates.append(item["pcrPlateRow"])
                else:
                    panelPoolType = item.panelPoolType
                    if item.pcrPlateRow:
                        item.pcrPlateRow = ""
                    if group == "Group 1":
                        item.assayGroup = "group 1"
                        if panelPoolType == "Single Pool":
                            item.tubePosition = "A"
                        else:
                            item.tubePosition = "A,B"
                        if "A" not in pcrPlates:
                            item.pcrPlateRow = "A"
                        elif "B" not in pcrPlates:
                            item.pcrPlateRow = "B"
                        elif "C" not in pcrPlates:
                            item.pcrPlateRow = "C"
                        elif "D" not in pcrPlates:
                            item.pcrPlateRow = "D"
                    elif group == "Group 2":
                        item.assayGroup = "group 2"
                        if panelPoolType == "Single Pool":
                            item.tubePosition = "C"
                        else:  # two pool
                            item.tubePosition = "C,D"
                        if "E" not in pcrPlates:
                            item.pcrPlateRow = "E"
                        elif "F" not in pcrPlates:
                            item.pcrPlateRow = "F"
                        elif "G" not in pcrPlates:
                            item.pcrPlateRow = "G"
                        elif "H" not in pcrPlates:
                            item.pcrPlateRow = "H"
                    pcrPlates.append(item.pcrPlateRow)

                upated_pending_sampleSetItem_list.append(item)
    return upated_pending_sampleSetItem_list


def assign_pcr_plate_rows(parsedSamplesetitems):
    # Assign PCR plate for Ampliseq on Chef
    upated_pending_sampleSetItem_list = []
    pcrPlates = []
    for item in parsedSamplesetitems:
        # Existing/persisted sample set item is model object when adding items to sampeset,so, type checking
        if type(item) == types.DictType:
            if not item.get("pcrPlateRow", ""):
                if "A" not in pcrPlates:
                    item["pcrPlateRow"] = "A"
                elif "B" not in pcrPlates:
                    item["pcrPlateRow"] = "B"
                elif "C" not in pcrPlates:
                    item["pcrPlateRow"] = "C"
                elif "D" not in pcrPlates:
                    item["pcrPlateRow"] = "D"
                elif "E" not in pcrPlates:
                    item["pcrPlateRow"] = "E"
                elif "F" not in pcrPlates:
                    item["pcrPlateRow"] = "F"
                elif "G" not in pcrPlates:
                    item["pcrPlateRow"] = "G"
                elif "H" not in pcrPlates:
                    item["pcrPlateRow"] = "H"
            pcrPlates.append(item["pcrPlateRow"])
        else:
            if not item.pcrPlateRow:
                if "A" not in pcrPlates:
                    item.pcrPlateRow = "A"
                elif "B" not in pcrPlates:
                    item.pcrPlateRow = "B"
                elif "C" not in pcrPlates:
                    item.pcrPlateRow = "C"
                elif "D" not in pcrPlates:
                    item.pcrPlateRow = "D"
                elif "E" not in pcrPlates:
                    item.pcrPlateRow = "E"
                elif "F" not in pcrPlates:
                    item.pcrPlateRow = "F"
                elif "G" not in pcrPlates:
                    item.pcrPlateRow = "G"
                elif "H" not in pcrPlates:
                    item.pcrPlateRow = "H"
            pcrPlates.append(item.pcrPlateRow)
        upated_pending_sampleSetItem_list.append(item)

    return upated_pending_sampleSetItem_list


def getChefPcrPlateConfig(sampleSet):
    libraryPrepKitId = KitInfo.objects.filter(name=sampleSet.libraryPrepKitName)[0].id
    chefPcrPlateconfig = ChefPcrPlateconfig.objects.get(
        kit=libraryPrepKitId
    )
    return chefPcrPlateconfig.get_chefPlatesConfig(libraryPrepKitId)

def processMultiPoolPlanSupport(sampleSets):
    allPlatesMapping = []
    pool1PlanSampleSetItemIds = []
    pool2PlanSampleSetItemIds = []
    for sampleSet in sampleSets:
        plateMapping = {}
        plateConfiguration = getChefPcrPlateConfig(sampleSet)
        sampleSetItems = sampleSet.samples.all()
        for item in sampleSetItems:
            if item.pcrPlateRow in plateConfiguration['pool1PcrPlateRows']:
                pool1PlanSampleSetItemIds.append(item.id)
            else:
                pool2PlanSampleSetItemIds.append(item.id)
            if item.dnabarcode:
                barcode1 = item.dnabarcode.id_str
                for k in plateConfiguration.keys():
                    if barcode1 in plateConfiguration[k]:
                        if k in plateMapping:
                            plateMapping[k] += 1
                        else:
                            plateMapping[k] = 1
        allPlatesMapping.append(plateMapping)
    allSamples, errors, warning = sample_validator.validate_multi_pool_support_samples(allPlatesMapping, sampleSets)

    return {
        'all': allSamples,
        'pool1': ','.join(str(x) for x in pool1PlanSampleSetItemIds),
        'pool2': ','.join(str(x) for x in pool2PlanSampleSetItemIds),
        'errors': errors,
        'warning': warning
    }