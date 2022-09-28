# Copyright (C) 2018 Ion Torrent Systems, Inc. All Rights Reserved
import ast
import os
import json
import traceback

from django.conf.urls import url
from django.contrib.auth.models import User
from django.db import transaction
from django.db.models import Q
from django.utils.translation import ugettext_lazy
from django.utils.encoding import force_unicode
from tastypie import fields
from tastypie.exceptions import ImmediateHttpResponse
from tastypie.http import *
from tastypie.validation import Validation
from tastypie.utils import dict_strip_unicode_keys, trailing_slash

from tastypie.authorization import DjangoAuthorization
from iondb.rundb.authn import IonAuthentication

from iondb.rundb import models
from iondb.rundb import labels
from iondb.rundb.json_lazy import LazyJSONEncoder
from iondb.rundb.plan.plan_share import (
    prepare_for_copy,
    transfer_plan,
    update_transferred_plan,
)
from iondb.rundb.plan import plan_validator
from iondb.rundb.sample import sample_validator
from iondb.rundb.sample import views_helper
from iondb.rundb.api_custom import (
    ModelResource,
    SDKValidationError,
    JSONconvert,
    field_dict,
    getAPIexamples,
)
from iondb.utils import toBoolean, validation

import logging

logger = logging.getLogger(__name__)

# replace the tastypie CharField
fields.CharField.convert = JSONconvert


class PlannedExperimentValidation(Validation):
    def is_valid(self, bundle, request=None):
        if not bundle.data:
            return {"__all__": "Fatal Error, no bundle!"}  # TODO: i18n

        errors = {}
        barcodedSampleWarnings = []
        barcodedWarnings = []
        planExp_warnings = {}

        isNewPlan = bundle.data.get("isNewPlan", False)
        isTemplate = bundle.data.get("isReusable", False)
        runType = bundle.data.get("runType", bundle.obj.runType)
        runType_label_ = "runType"

        if "applicationGroupDisplayedName" in bundle.data:
            label_applicationGroupDisplayedName = "applicationGroupDisplayedName"
            applicationGroupName = bundle.data.get("applicationGroupDisplayedName", "")
        else:
            label_applicationGroupDisplayedName = ugettext_lazy(
                "workflow.step.application.fields.applicationGroup.label"
            )
            applicationGroupName = (
                bundle.obj.applicationGroup.description
                if bundle.obj.applicationGroup
                else ""
            )

        # validate required parameters

        value = bundle.data.get("planName", "")
        if value or isNewPlan:
            err = plan_validator.validate_plan_name(value, "planName")
            if err:
                errors["planName"] = err

        value = bundle.data.get("chipType", "")
        if value or (isNewPlan and not isTemplate):
            err, chipType_warning = plan_validator.validate_chip_type(
                value, field_label="chipType", isNewPlan=isNewPlan
            )
            if err:
                errors["chipType"] = err
            if chipType_warning:
                planExp_warnings["chipType"] = chipType_warning

        value = bundle.data.get("templatingKitName")
        if value or (isNewPlan and not isTemplate):
            err, templatingKit_warning = plan_validator.validate_plan_templating_kit_name(
                value, field_label="templatingKitName", isNewPlan=isNewPlan
            )
            if err:
                errors["templatingKitName"] = err
            if templatingKit_warning:
                planExp_warnings["templatingKitName"] = templatingKit_warning

        # validate the sample input data
        sampleValue = bundle.data.get("sample")
        barcodedSampleValue = bundle.data.get("barcodedSamples")

        if not sampleValue and not barcodedSampleValue and not isTemplate and isNewPlan:
            errors["sample"] = " ".join(
                [
                    validation.missing_error("sample"),
                    validation.invalid_required_at_least_one_polymorphic_type_value(
                        validation.Entity_EntityFieldName(
                            "sample",
                            "(%s)" % force_unicode(labels.NonBarcodedPlan.verbose_name),
                        ),
                        validation.Entity_EntityFieldName(
                            "barcodedSamples",
                            "(%s)" % force_unicode(labels.BarcodedPlan.verbose_name),
                        ),
                    ),
                ]
            )
        if sampleValue and barcodedSampleValue:
            errors["sample"] = " ".join(
                [
                    validation.provided("barcodedSamples"),
                    validation.invalid_required_at_least_one_polymorphic_type_value(
                        validation.Entity_EntityFieldName(
                            "sample",
                            "(%s)" % force_unicode(labels.NonBarcodedPlan.verbose_name),
                        ),
                        validation.Entity_EntityFieldName(
                            "barcodedSamples",
                            "(%s)" % force_unicode(labels.BarcodedPlan.verbose_name),
                        ),
                    ),
                ]
            )
            errors["barcodedSamples"] = " ".join(
                [
                    "sample",
                    validation.invalid_required_at_least_one_polymorphic_type_value(
                        validation.Entity_EntityFieldName(
                            "sample",
                            "(%s)" % force_unicode(labels.NonBarcodedPlan.verbose_name),
                        ),
                        validation.Entity_EntityFieldName(
                            "barcodedSamples",
                            "(%s)" % force_unicode(labels.BarcodedPlan.verbose_name),
                        ),
                    ),
                ]
            )
        if "sample" not in errors and sampleValue:
            err = plan_validator.validate_sample_name(
                sampleValue,
                field_label="sample",
                isTemplate=isTemplate,
                barcodeId=bundle.data.get("barcodeId"),
                barcodeId_label="barcodeId",
                isTemplate_label="Plan Template",
            )
            if err:
                errors["sample"] = err

        value = bundle.data.get("libraryKey", "")
        noGlobal_libraryKit = None
        if not value:
            gc = models.GlobalConfig.get()
            if not gc.default_library_key:
                noGlobal_libraryKit = True
        if value or noGlobal_libraryKit:
            err, selectedLibKey = plan_validator.validate_library_key(
                value, field_label="libraryKey"
            )
            if err:
                errors["libraryKey"] = err

        # barcode kit
        if "barcodeId" in bundle.data:
            barcodeId = bundle.data.get("barcodeId", "")
            barcodeId_label_ = "barcodeId"
        else:
            barcodeId = (
                bundle.obj.latestEAS.barcodeKitName if bundle.obj.latestEAS else ""
            )
            barcodeId_label_ = ugettext_lazy(
                "workflow.step.sample.fields.barcodeSet.label"
            )

        # validate optional parameters
        for key, value in list(bundle.data.items()):
            err = []
            key_specific_warning = []
            if key == "planStatus":
                err = plan_validator.validate_planStatus(
                    value, field_label="planStatus"
                )
            if key == "sampleTubeLabel":
                err = plan_validator.validate_sample_tube_label(value, field_label=key)
            if key == "barcodedSamples" and value and "barcodedSamples" not in errors:
                barcodedSamples_label_ = key
                if isTemplate:
                    errors["barcodedSamples"] = validation.format(
                        ugettext_lazy(
                            "template.messages.validation.invalidbarcodedsamples"
                        ),
                        {"name": labels.PlanTemplate.verbose_name},
                    )  # "Invalid input. Barcoded sample information cannot be saved in the plan template"  # TODO: i18n
                    continue
                barcodedSamples = (
                    json.loads(value) if isinstance(value, basestring) else value
                )

                for sample in barcodedSamples:
                    barcodedSample_label_ = '%s["%s"]' % (
                        barcodedSamples_label_,
                        sample,
                    )
                    barcodedSampleErrors = []
                    barcodedWarnings = []
                    barcodeSampleInfo = None
                    err.extend(
                        plan_validator.validate_sample_name(
                            sample, field_label=barcodedSample_label_
                        )
                    )
                    # verify each sample contains a 'barcodes' object
                    try:
                        get_barcodedSamples = barcodedSamples[sample]["barcodes"]
                    except (KeyError, TypeError) as e:
                        if isNewPlan:
                            err.append(
                                {
                                    sample: validation.schema_error_missing_attribute(
                                        barcodedSample_label_, e
                                    )
                                }
                            )
                        else:
                            barcodedSampleWarnings += [
                                {
                                    sample: validation.schema_error_missing_attribute(
                                        barcodedSample_label_, e
                                    )
                                }
                            ]
                        continue
                    get_barcodedSamples = [
                        x.encode("UTF8") for x in get_barcodedSamples
                    ]

                    try:
                        barcodeSampleInfo = barcodedSamples[sample]["barcodeSampleInfo"]
                        barcodeSampleInfo_label_ = (
                            '%s["barcodeSampleInfo"]' % barcodedSample_label_
                        )
                    except (KeyError, TypeError) as e:
                        if isNewPlan:
                            err.append(
                                {
                                    sample: validation.schema_error_missing_attribute(
                                        barcodedSample_label_, e
                                    )
                                }
                            )
                        else:
                            barcodedSampleWarnings += [
                                {
                                    sample: validation.schema_error_missing_attribute(
                                        barcodedSample_label_, e
                                    )
                                }
                            ]
                            barcodeSampleInfo = None

                    if barcodeSampleInfo:
                        for barcode in get_barcodedSamples:
                            if barcode:
                                barcode_label_ = '%s["%s"]' % (
                                    barcodeSampleInfo_label_,
                                    barcode,
                                )
                                reference = None
                                barcode_reference_label_ = '%s["reference"]' % (
                                    barcode_label_
                                )
                                eachBarcodeErr = []
                                warnings = []
                                isValid, errInvalidBarcode, item = sample_validator.validate_barcodekit_and_id_str(
                                    barcodeId,
                                    barcode,
                                    barcodeKit_label=barcodeId_label_,
                                    barcode_id_str_label=barcode_label_,
                                )
                                if not isValid:
                                    eachBarcodeErr.append(errInvalidBarcode)
                                else:
                                    if barcode not in list(barcodeSampleInfo.keys()):
                                        if isNewPlan:
                                            eachBarcodeErr.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, barcode
                                                )
                                            )
                                            barcodedSampleErrors += [
                                                {barcode: eachBarcodeErr}
                                            ]
                                        else:
                                            warnings.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, barcode
                                                )
                                            )
                                            barcodedWarnings += [{barcode: warnings}]
                                        continue

                                    try:
                                        reference = barcodeSampleInfo[barcode][
                                            "reference"
                                        ]
                                        if reference:
                                            err_isRefSupported = plan_validator.validate_reference_for_runType(
                                                reference,
                                                barcode_reference_label_,
                                                runType,
                                                applicationGroupName,
                                                labels.ScientificApplication.verbose_name,
                                            )
                                            if not err_isRefSupported:
                                                err_ref = plan_validator.validate_reference_short_name(
                                                    reference,
                                                    field_label=barcode_reference_label_,
                                                )
                                                if err_ref:
                                                    eachBarcodeErr.extend(
                                                        err_ref
                                                    )  # expects that err_ref is list of strings
                                            else:
                                                eachBarcodeErr.extend(
                                                    err_isRefSupported
                                                )  # expects that err_isRefSupported is list of strings
                                        else:
                                            warnings.append(
                                                ugettext_lazy(
                                                    "plannedexperiment.messages.validation.barcodedsample.noreference.de-novo"
                                                )
                                            )  # "Barcoded sample with no reference specified will be analyzed as de-novo."
                                    except KeyError as e:
                                        if isNewPlan:
                                            eachBarcodeErr.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, e
                                                )
                                            )
                                        else:
                                            warnings.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, e
                                                )
                                            )

                                    # validate targetRegionBedFile - Error if invalid / Warning if empty
                                    try:
                                        targetRegionBedFile = barcodeSampleInfo[
                                            barcode
                                        ]["targetRegionBedFile"]
                                        barcode_targetRegionBedFile_label_ = (
                                            '%s["targetRegionBedFile"]'
                                            % (barcode_label_)
                                        )
                                        nucleotideType = barcodeSampleInfo[barcode].get(
                                            "nucleotideType", ""
                                        )
                                        if reference:
                                            if not isTemplate:
                                                err_targetRegionBedFile = plan_validator.validate_targetRegionBedFile_for_runType(
                                                    targetRegionBedFile,
                                                    field_label=barcode_targetRegionBedFile_label_,
                                                    runType=runType,
                                                    reference=reference,
                                                    nucleotideType=nucleotideType,
                                                    applicationGroupName=applicationGroupName,
                                                )
                                                if err_targetRegionBedFile:
                                                    err_targetRegionBedFile = "".join(
                                                        err_targetRegionBedFile
                                                    )
                                                    eachBarcodeErr.append(
                                                        err_targetRegionBedFile
                                                    )
                                        elif targetRegionBedFile:
                                            eachBarcodeErr.append(
                                                validation.invalid_required_related_value(
                                                    barcode_reference_label_,
                                                    barcode_targetRegionBedFile_label_,
                                                    targetRegionBedFile,
                                                )
                                            )  # "Bed file exists but No Reference"
                                    except KeyError as e:
                                        if isNewPlan:
                                            eachBarcodeErr.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, e
                                                )
                                            )
                                        else:
                                            warnings.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, e
                                                )
                                            )
                                    except Exception as e:
                                        warnings.append(e)

                                    # validate hotSpotRegionBedFile - Error if invalid / Warning if empty
                                    try:
                                        hotSpotRegionBedFile = barcodedSamples[sample][
                                            "barcodeSampleInfo"
                                        ][barcode]["hotSpotRegionBedFile"]
                                        barcode_hotSpotRegionBedFile_label_ = (
                                            '%s["hotSpotRegionBedFile"]'
                                            % (barcode_label_)
                                        )
                                        if hotSpotRegionBedFile:
                                            if targetRegionBedFile and reference:
                                                err_hotSpotRegionBedFile = plan_validator.validate_hotspot_bed(
                                                    hotSpotRegionBedFile,
                                                    field_label=barcode_hotSpotRegionBedFile_label_,
                                                )
                                                if err_hotSpotRegionBedFile:
                                                    err_hotSpotRegionBedFile = "".join(
                                                        err_hotSpotRegionBedFile
                                                    )
                                                    eachBarcodeErr.append(
                                                        err_hotSpotRegionBedFile
                                                    )
                                            else:
                                                if not reference:
                                                    eachBarcodeErr.append(
                                                        validation.invalid_required_related_value(
                                                            barcode_reference_label_,
                                                            barcode_hotSpotRegionBedFile_label_,
                                                            value,
                                                        )
                                                    )  # "Hot spot exists but No Reference"
                                                if not targetRegionBedFile:
                                                    eachBarcodeErr.append(
                                                        validation.invalid_required_related_value(
                                                            barcode_targetRegionBedFile_label_,
                                                            barcode_hotSpotRegionBedFile_label_,
                                                            value,
                                                        )
                                                    )  # "Hot spot exists but Bed File"
                                    except KeyError as e:
                                        if isNewPlan:
                                            eachBarcodeErr.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, e
                                                )
                                            )
                                        else:
                                            warnings.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, e
                                                )
                                            )
                                    except Exception as e:
                                        warnings.append(e)

                                    # validate nucleotideType - Error if invalid / Warning if empty
                                    try:
                                        nucleotideType = barcodedSamples[sample][
                                            "barcodeSampleInfo"
                                        ][barcode]["nucleotideType"]
                                        barcode_nucleotideType_label_ = (
                                            '%s["nucleotideType"]' % (barcode_label_)
                                        )
                                        if nucleotideType:
                                            err_nucleotideType, sample_nucleotideType = plan_validator.validate_sample_nucleotideType(
                                                nucleotideType,
                                                runType,
                                                applicationGroupName,
                                                field_label=barcode_nucleotideType_label_,
                                            )
                                            if err_nucleotideType:
                                                err_nucleotideType = "".join(
                                                    err_nucleotideType
                                                )
                                                eachBarcodeErr.append(
                                                    err_nucleotideType
                                                )
                                            if not runType:
                                                warnings.append(
                                                    validation.invalid_empty_related_value(
                                                        "runType",
                                                        barcode_nucleotideType_label_,
                                                        nucleotideType,
                                                    )
                                                )
                                            if not applicationGroupName:
                                                warnings.append(
                                                    validation.invalid_empty_related_value(
                                                        label_applicationGroupDisplayedName,
                                                        barcode_nucleotideType_label_,
                                                        nucleotideType,
                                                    )
                                                )
                                        else:
                                            if isNewPlan:
                                                eachBarcodeErr.append(
                                                    validation.required_error(
                                                        barcode_nucleotideType_label_
                                                    )
                                                )
                                            else:
                                                warnings.append(
                                                    validation.invalid_empty(
                                                        barcode_nucleotideType_label_
                                                    )
                                                )
                                    except KeyError as e:
                                        if isNewPlan:
                                            eachBarcodeErr.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, e
                                                )
                                            )
                                        else:
                                            warnings.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, e
                                                )
                                            )
                                    except Exception as e:
                                        warnings.append(e)

                                    # validate controlType
                                    try:
                                        if "controlType" in barcodeSampleInfo[barcode]:
                                            barcode_controlType_label_ = '%s["%s"]' % (
                                                barcode_label_,
                                                key,
                                            )
                                            err_controltype, _ = plan_validator.validate_sampleControlType(
                                                barcodeSampleInfo[barcode][
                                                    "controlType"
                                                ],
                                                field_label=barcode_controlType_label_,
                                            )  # Control Type
                                            if err_controltype:
                                                eachBarcodeErr.append(
                                                    "".join(err_controltype)
                                                )
                                    except KeyError as e:
                                        if isNewPlan:
                                            eachBarcodeErr.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, e
                                                )
                                            )
                                        else:
                                            warnings.append(
                                                validation.schema_error_missing_attribute(
                                                    barcode_label_, e
                                                )
                                            )
                                    except Exception as e:
                                        warnings.append(e)

                                if eachBarcodeErr:
                                    barcodedSampleErrors += [{barcode: eachBarcodeErr}]
                                if warnings:
                                    barcodedWarnings += [{barcode: warnings}]

                    if barcodedSampleErrors:
                        err.append({sample: barcodedSampleErrors})
                    if barcodedWarnings:
                        barcodedSampleWarnings += [{sample: barcodedWarnings}]
                if barcodedSampleWarnings:
                    planExp_warnings[key] = barcodedSampleWarnings
            if key == "chipBarcode":
                err = plan_validator.validate_chipBarcode(
                    value, field_label=key
                )  # 'Chip Barcode'
            if key == "notes":
                err = plan_validator.validate_notes(value, field_label=key)
            if key == "flows" and value != "0":
                err = plan_validator.validate_flows(value, field_label=key)
            if key == "project" or key == "projects":
                projectNames = (
                    value if isinstance(value, basestring) else ",".join(value)
                )
                project_errors, trimmed_projectNames = plan_validator.validate_projects(
                    projectNames, field_label=key
                )
                err.extend(project_errors)
            if key == "barcodeId":
                err = plan_validator.validate_barcode_kit_name(value, field_label=key)
            if key == "sequencekitname":
                err, key_specific_warning = plan_validator.validate_optional_kit_name(
                    value,
                    kitType=["SequencingKit"],
                    field_label=key,
                    isNewPlan=isNewPlan,
                )
            if key == "librarykitname":
                err, key_specific_warning = plan_validator.validate_optional_kit_name(
                    value,
                    kitType=["LibraryKit", "LibraryPrepKit"],
                    field_label=key,
                    isNewPlan=isNewPlan,
                )
            if key == "runType":
                err = plan_validator.validate_runType(value, field_label=key)
            if key == "applicationGroupDisplayedName":
                err = plan_validator.validate_application_group_for_runType(
                    value, field_label=key, runType=runType, runType_label="runType"
                )
            if key == "sampleGroupingName":
                err = plan_validator.validate_sample_grouping(value, field_label=key)
            if key == "libraryReadLength":
                err = plan_validator.validate_libraryReadLength(value, field_label=key)
            if key == "templatingSize" and value:
                key_specific_warning = "This field is deprecated, use samplePrepProtocol (pcr200bp, pcr400bp) instead."
            if key == "samplePrepProtocol":
                err = bundle.data.get("errorInSamplePrepProtocol", "")
            if key == "bedfile":
                if "library" in bundle.data:
                    reference = bundle.data.get("library")
                else:
                    reference = (
                        bundle.obj.latestEAS.reference if bundle.obj.latestEAS else ""
                    )
                if reference and not isTemplate:
                    err = plan_validator.validate_targetRegionBedFile_for_runType(
                        value,
                        field_label=key,
                        runType=runType,
                        reference=reference,
                        nucleotideType="",
                        applicationGroupName=applicationGroupName,
                        barcodeId=barcodeId,
                    )
                if value and not reference:
                    err = "Bed file(%s) exists but No Reference" % (value)  # TODO: i18n
            if key == "flowsInOrder":
                err = bundle.data.get("errorInflowOrder", "")
            if key == "library":
                err = plan_validator.validate_reference_for_runType(
                    value,
                    key,
                    runType,
                    applicationGroupName,
                    labels.ScientificApplication.verbose_name,
                )
                if not err:
                    err = plan_validator.validate_reference_short_name(
                        value, field_label=key
                    )
            if key == "mixedTypeRNA_reference":
                # displayedName = "RNA Reference" if runType == "MIXED" else "Fusions Reference"
                err = plan_validator.validate_reference_for_fusions(
                    value,
                    field_label=key,
                    runType=runType,
                    applicationGroupName=applicationGroupName,
                    application_label=labels.ScientificApplication.verbose_name,
                )
            if key == "mixedTypeRNA_targetRegionBedFile":
                if "mixedTypeRNA_reference" in bundle.data:
                    reference = bundle.data.get("mixedTypeRNA_reference")
                else:
                    reference = (
                        bundle.obj.latestEAS.mixedTypeRNA_reference
                        if bundle.obj.latestEAS
                        else ""
                    )
                if reference and not isTemplate:
                    # displayedName = "RNA Target Regions BED File" if runType == "MIXED" else "Fusions Target Regions BED File"
                    err = plan_validator.validate_targetRegionBedFile_for_runType(
                        value,
                        field_label=key,
                        runType=runType,
                        reference=reference,
                        nucleotideType="",
                        applicationGroupName=applicationGroupName,
                        isPrimaryTargetRegion=False,
                    )
                if value and not reference:
                    err = validation.invalid_required_related_value(
                        "mixedTypeRNA_reference",
                        "mixedTypeRNA_targetRegionBedFile",
                        value,
                    )  # "Bed file(%s) exists but No Reference" % (value)

            if key == "selectedPlugins":
                plugin_config_validation = " | ".join(
                    plan_validator.validate_plugin_configurations(bundle.data.get(key))
                )
                if isTemplate:
                    key_specific_warning = plugin_config_validation
                else:
                    err = plugin_config_validation

            if err:
                errors[key] = err
            if key_specific_warning:
                planExp_warnings[key] = key_specific_warning

        if errors:
            planName = bundle.data.get("planName", "")
            logger.error(
                "plan validation errors for plan=%s: Errors=%s" % (planName, errors)
            )
            raise SDKValidationError(errors)

        # validate the kit_chips combination
        unSupportedKits_error = plan_validator.validate_kit_chip_combination(bundle)
        if unSupportedKits_error:
            planName = bundle.data.get("planName", "")
            logger.error(
                "plan validation errors for plan=%s: Errors=%s"
                % (planName, unSupportedKits_error)
            )
            raise SDKValidationError({"Error": unSupportedKits_error})

        if planExp_warnings:
            bundle.data["Warnings"] = {}
            for key, value in planExp_warnings.items():
                bundle.data["Warnings"][key] = value

        return errors


class PlannedExperimentDbResource(ModelResource):
    # Backwards support - single project field
    project = fields.CharField(readonly=True, blank=True)

    projects = fields.ToManyField(
        "iondb.rundb.api.ProjectResource", "projects", full=False, null=True, blank=True
    )
    qcValues = fields.ToManyField(
        "iondb.rundb.api.PlannedExperimentQCResource",
        "plannedexperimentqc_set",
        full=True,
        null=True,
        blank=True,
    )
    parentPlan = fields.CharField(blank=True, default=None)
    childPlans = fields.ListField(default=[])
    experiment = fields.ToOneField(
        "iondb.rundb.api.ExperimentResource",
        "experiment",
        full=False,
        null=True,
        blank=True,
    )

    sampleSets = fields.ToManyField(
        "iondb.rundb.api.SampleSetResource",
        "sampleSets",
        full=False,
        null=True,
        blank=True,
    )
    applicationGroup = fields.ToOneField(
        "iondb.rundb.api.ApplicationGroupResource",
        "applicationGroup",
        full=False,
        null=True,
        blank=True,
    )
    sampleGrouping = fields.ToOneField(
        "iondb.rundb.api.SampleGroupType_CVResource",
        "sampleGrouping",
        full=False,
        null=True,
        blank=True,
    )

    def hydrate_m2m(self, bundle):
        if "projects" in bundle.data or "project" in bundle.data:
            # Promote projects from names to something tastypie recognizes in hydrate_m2m
            projects_list = []
            for k in ["projects", "project"]:
                value = bundle.data.get(k, [])

                if isinstance(value, basestring):
                    value = [value]
                for p in value:
                    if p not in projects_list:
                        projects_list.append(p)

            project_objs = []
            if projects_list:
                user = getattr(bundle.request, "user", None)
                if user is not None and user.is_superuser:
                    username = bundle.data.get("username")
                    if username is not None:
                        try:
                            user = User.objects.get(username=username)
                        except User.DoesNotExist:
                            pass
                project_objs = models.Project.bulk_get_or_create(projects_list, user)
        else:
            project_objs = bundle.obj.projects.all()

        bundle.data["projects"] = project_objs
        bundle.data["project"] = None

        # hydrate SampleSets
        sampleSetDisplayedName = bundle.data.get("sampleSetDisplayedName", "")
        if sampleSetDisplayedName:
            sampleSets = models.SampleSet.objects.filter(
                displayedName__in=sampleSetDisplayedName.split(",")
            )
            if sampleSets:
                bundle.data["sampleSets"] = sampleSets
        else:
            bundle.data["sampleSets"] = bundle.obj.sampleSets.all()

        return super(PlannedExperimentDbResource, self).hydrate_m2m(bundle)

    def dehydrate_projects(self, bundle):
        """Return a list of project names rather than any specific objects"""
        # logger.debug("Dehydrating %s with projects %s", bundle.obj, projects_names)
        return [unicode(project.name) for project in bundle.obj.projects.all()]

    def dehydrate_project(self, bundle):
        """Return the first project name"""
        try:
            firstProject = unicode(bundle.obj.projects.all()[0].name)
        except IndexError:
            firstProject = ""

        return firstProject

    def build_filters(self, filters=None):
        if filters is None:
            filters = {}

        for key, val in filters.items():
            if "chipBarcode" in key and "experiment__chipBarcode" not in key:
                # redirect filtering to experiment.chipBarcode field
                filters[key.replace("chipBarcode", "experiment__chipBarcode")] = val
                del filters[key]

        filter_platform = filters.get("platform") or filters.get("instrument")
        if "platform" in filters:
            del filters["platform"]

        orm_filters = super(PlannedExperimentDbResource, self).build_filters(filters)
        if filter_platform:
            orm_filters.update(
                {
                    "custom_platform": (
                        Q(experiment__platform="")
                        | Q(experiment__platform__iexact=filter_platform)
                    )
                }
            )

        return orm_filters

    def apply_filters(self, request, filters):
        custom_query = filters.pop("custom_platform", None)
        base_object_list = super(PlannedExperimentDbResource, self).apply_filters(
            request, filters
        )

        if custom_query is not None:
            base_object_list = base_object_list.filter(custom_query)

        name_or_id = request.GET.get("name_or_id") or request.GET.get(
            "name_or_id__icontains"
        )
        if name_or_id is not None:
            qset = (
                Q(planName__icontains=name_or_id)
                | Q(planDisplayedName__icontains=name_or_id)
                | Q(planShortID__icontains=name_or_id)
            )
            base_object_list = base_object_list.filter(qset)

        combinedLibraryTubeLabel = request.GET.get("combinedLibraryTubeLabel", None)
        if combinedLibraryTubeLabel is not None:
            combinedLibraryTubeLabel = combinedLibraryTubeLabel.strip()
            qset = Q(
                sampleSets__combinedLibraryTubeLabel__icontains=combinedLibraryTubeLabel
            )
            base_object_list = base_object_list.filter(qset)

        return base_object_list.distinct()

    class Meta:
        queryset = (
            models.PlannedExperiment.objects.select_related("experiment", "latestEAS")
            .prefetch_related(
                "projects",
                "plannedexperimentqc_set",
                "plannedexperimentqc_set__qcType",
                "experiment__samples",
                "sampleSets",
            )
            .all()
        )

        transfer_allowed_methods = ["get", "post"]
        copy_allowed_methods = ["post"]
        create_allowed_methods = ["get", "post"]
        always_return_data = True

        # allow ordering and filtering by all fields
        field_list = models.PlannedExperiment._meta.get_all_field_names()
        ordering = field_list
        filtering = field_dict(field_list)

        authentication = IonAuthentication(ion_mesh_access=True)
        authorization = DjangoAuthorization()
        validation = PlannedExperimentValidation()


class PlannedExperimentResource(PlannedExperimentDbResource):
    autoAnalyze = fields.BooleanField()
    barcodedSamples = fields.CharField(blank=True, null=True, default="")
    barcodeId = fields.CharField(blank=True, null=True, default="")
    bedfile = fields.CharField(blank=True, default="")
    chipType = fields.CharField(default="")
    chipBarcode = fields.CharField(default="")
    endBarcodeKitName = fields.CharField(blank=True, null=True, default="")
    flows = fields.IntegerField(default=0)
    forward3primeadapter = fields.CharField(blank=True, null=True, default="")
    library = fields.CharField(blank=True, null=True, default="")
    libraryKey = fields.CharField(blank=True, default="")
    tfKey = fields.CharField(blank=True, default="")
    librarykitname = fields.CharField(blank=True, null=True, default="")
    notes = fields.CharField(blank=True, null=True, default="")
    regionfile = fields.CharField(blank=True, default="")
    reverse3primeadapter = fields.CharField(readonly=True, default="")
    reverselibrarykey = fields.CharField(readonly=True, default="")
    sample = fields.CharField(blank=True, null=True, default="")
    sampleDisplayedName = fields.CharField(blank=True, null=True, default="")
    selectedPlugins = fields.CharField(blank=True, null=True, default="")
    sequencekitname = fields.CharField(blank=True, null=True, default="")
    sseBedFile = fields.CharField(blank=True, default="")
    variantfrequency = fields.CharField(readonly=True, default="")
    isDuplicateReads = fields.BooleanField()
    base_recalibration_mode = fields.CharField(blank=True, null=True, default="")
    realign = fields.BooleanField()
    flowsInOrder = fields.CharField(blank=True, null=True, default="")

    # this is a comma-separated string if multiple sampleset names
    sampleSetDisplayedName = fields.CharField(readonly=True, blank=True, null=True)

    applicationCategoryDisplayedName = fields.CharField(
        readonly=True, blank=True, null=True
    )
    applicationGroupDisplayedName = fields.CharField(
        readonly=True, blank=True, null=True
    )
    sampleGroupingName = fields.CharField(readonly=True, blank=True, null=True)

    libraryPrepType = fields.CharField(readonly=True, blank=True, null=True, default="")
    libraryPrepTypeDisplayedName = fields.CharField(
        readonly=True, blank=True, null=True, default=""
    )

    platform = fields.CharField(blank=True, null=True)

    chefInfo = fields.DictField(default={})

    earlyDatFileDeletion = fields.BooleanField(readonly=True, default=False)

    mixedTypeRNA_reference = fields.CharField(blank=True, null=True, default="")
    mixedTypeRNA_targetRegionBedFile = fields.CharField(blank=True, default="")
    mixedTypeRNA_hotSpotRegionBedFile = fields.CharField(blank=True, default="")

    def prepend_urls(self):
        return [
            url(
                r"^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/transfer%s$"
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view("dispatch_transfer"),
                name="api_dispatch_transfer",
            ),
            url(
                r"^(?P<resource_name>%s)/copy/(?P<planGUID>\w[\w/-]*)%s$"
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view("dispatch_copy"),
                name="api_dispatch_copy",
            ),
            url(
                r"^(?P<resource_name>%s)/create%s$"
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view("dispatch_create"),
                name="api_dispatch_create",
            ),
        ]

    def dispatch_transfer(self, request, **kwargs):
        return self.dispatch("transfer", request, **kwargs)

    def get_transfer(self, request, **kwargs):
        # runs on destination TS to update plan-related objects and return any errors
        plan = self.get_object_list(request).get(pk=kwargs["pk"])
        status = update_transferred_plan(plan, request)

        return self.create_response(request, status)

    def post_transfer(self, request, **kwargs):
        # runs on origin TS to initiate plan transfer
        destination = request.POST.get("destination")
        if not destination:
            return HttpBadRequest(validation.required_error("destination"))

        plan = self.get_object_list(request).get(pk=kwargs["pk"])
        if plan is None:
            return HttpGone()

        try:
            # transfer plan to destination
            bundle = self.build_bundle(obj=plan, request=request)
            bundle = prepare_for_copy(self.full_dehydrate(bundle))
            serialized = self.serialize(None, bundle, "application/json")

            status = transfer_plan(plan, serialized, destination, request.user.username)

        except Exception as err:
            logger.error(
                "Error while attempting to transfer plan %s", traceback.format_exc()
            )
            return HttpBadRequest(err)

        return self.create_response(request, status)

    def dispatch_copy(self, request, **kwargs):
        return self.dispatch("copy", request, **kwargs)

    def post_copy(self, request, **kwargs):
        # copy plan objects
        plan = self.get_object_list(request).filter(planGUID=kwargs.get("planGUID"))
        if not plan:
            return HttpGone()

        try:
            bundle = self._copy_plan_bundle_from_obj(plan[0])
            bundle.data["origin"] = "copy"

            # allow request to overwrite parameters
            for key, value in list(request.POST.dict().items()):
                bundle.data[key] = value

            # create object
            bundle = self.obj_create(bundle)
            return self.create_response(
                request, self.full_dehydrate(bundle), response_class=HttpAccepted
            )

        except Exception as err:
            return HttpBadRequest(err)

    def _copy_plan_bundle_from_obj(self, planObj):
        bundle = self.build_bundle(obj=planObj)
        bundle = self.full_dehydrate(bundle)

        # modify data to create a new plan
        bundle.obj = None
        bundle.data.pop("experiment")
        bundle.data.pop("planGUID")
        bundle.data.pop("planShortID")
        for qc in bundle.data.pop("qcValues", []):
            bundle.data[qc.obj.qcType.qcName] = qc.obj.threshold

        bundle.data["planDisplayedName"] = "CopyOf_" + bundle.data["planName"]
        bundle.data["planStatus"] = "planned"
        bundle.data["planExecuted"] = False
        bundle.data["isSystem"] = False
        bundle.data["isSystemDefault"] = False
        # metaData is required for Dynamic Tech Param
        metaData = {
            "fromTemplate": bundle.data.get("planName"),
            "fromTemplateId":  bundle.data.pop("id"),
            "fromTemplateCategories": bundle.data.get("categories"),
            "fromTemplateChipType": bundle.data.get("chipType"),
            "fromTemplateSequenceKitname": bundle.data.get("sequencekitname"),
            "fromTemplateSource": "ION"
        }
        if bundle.data.get("metaData"):
            metaData.update(bundle.data.get("metaData"))
        bundle.data["metaData"] = metaData

        return bundle

    def dehydrate(self, bundle):
        try:
            experiment = bundle.obj.experiment
            bundle.data["autoAnalyze"] = experiment.autoAnalyze
            bundle.data["chipType"] = experiment.chipType
            bundle.data["chipBarcode"] = experiment.chipBarcode
            bundle.data["flows"] = experiment.flows
            bundle.data["flowsInOrder"] = experiment.flowsInOrder
            bundle.data["notes"] = experiment.notes
            bundle.data["sequencekitname"] = experiment.sequencekitname
            bundle.data["platform"] = experiment.platform

            # retrieve EAS parameters from specified result or latest object
            report_pk = bundle.request.GET.get("for_report")
            if report_pk:
                try:
                    latest_eas = experiment.results_set.get(pk=report_pk).eas
                except Exception:
                    raise ImmediateHttpResponse(
                        HttpBadRequest("Invalid report pk specified: %s" % report_pk)
                    )
            else:
                latest_eas = bundle.obj.latestEAS

            if not latest_eas:
                latest_eas = experiment.get_EAS()

            bundle.data["barcodedSamples"] = (
                latest_eas.barcodedSamples if latest_eas else ""
            )
            bundle.data["barcodeId"] = latest_eas.barcodeKitName if latest_eas else ""
            bundle.data["bedfile"] = (
                latest_eas.targetRegionBedFile if latest_eas else ""
            )
            bundle.data["endBarcodeKitName"] = (
                latest_eas.endBarcodeKitName if latest_eas else ""
            )
            bundle.data["forward3primeadapter"] = (
                latest_eas.threePrimeAdapter if latest_eas else ""
            )
            bundle.data["library"] = latest_eas.reference if latest_eas else ""
            bundle.data["libraryKey"] = latest_eas.libraryKey if latest_eas else ""
            bundle.data["tfKey"] = latest_eas.tfKey if latest_eas else ""
            bundle.data["librarykitname"] = (
                latest_eas.libraryKitName if latest_eas else ""
            )
            bundle.data["regionfile"] = (
                latest_eas.hotSpotRegionBedFile if latest_eas else ""
            )
            bundle.data["isDuplicateReads"] = (
                latest_eas.isDuplicateReads if latest_eas else False
            )
            bundle.data["base_recalibration_mode"] = (
                latest_eas.base_recalibration_mode if latest_eas else "no_recal"
            )
            bundle.data["realign"] = latest_eas.realign if latest_eas else False
            bundle.data["sseBedFile"] = latest_eas.sseBedFile if latest_eas else ""

            bundle.data["mixedTypeRNA_reference"] = (
                latest_eas.mixedTypeRNA_reference if latest_eas else ""
            )
            bundle.data["mixedTypeRNA_targetRegionBedFile"] = (
                latest_eas.mixedTypeRNA_targetRegionBedFile if latest_eas else ""
            )
            bundle.data["mixedTypeRNA_hotSpotRegionBedFile"] = (
                latest_eas.mixedTypeRNA_hotSpotRegionBedFile if latest_eas else ""
            )

            bundle.data["beadfindargs"] = latest_eas.beadfindargs if latest_eas else ""
            bundle.data["thumbnailbeadfindargs"] = (
                latest_eas.thumbnailbeadfindargs if latest_eas else ""
            )
            bundle.data["analysisargs"] = latest_eas.analysisargs if latest_eas else ""
            bundle.data["thumbnailanalysisargs"] = (
                latest_eas.thumbnailanalysisargs if latest_eas else ""
            )
            bundle.data["prebasecallerargs"] = (
                latest_eas.prebasecallerargs if latest_eas else ""
            )
            bundle.data["prethumbnailbasecallerargs"] = (
                latest_eas.prethumbnailbasecallerargs if latest_eas else ""
            )
            bundle.data["calibrateargs"] = (
                latest_eas.calibrateargs if latest_eas else ""
            )
            bundle.data["thumbnailcalibrateargs"] = (
                latest_eas.thumbnailcalibrateargs if latest_eas else ""
            )
            bundle.data["basecallerargs"] = (
                latest_eas.basecallerargs if latest_eas else ""
            )
            bundle.data["thumbnailbasecallerargs"] = (
                latest_eas.thumbnailbasecallerargs if latest_eas else ""
            )
            bundle.data["alignmentargs"] = (
                latest_eas.alignmentargs if latest_eas else ""
            )
            bundle.data["thumbnailalignmentargs"] = (
                latest_eas.thumbnailalignmentargs if latest_eas else ""
            )
            bundle.data["ionstatsargs"] = latest_eas.ionstatsargs if latest_eas else ""
            bundle.data["thumbnailionstatsargs"] = (
                latest_eas.thumbnailionstatsargs if latest_eas else ""
            )
            bundle.data["custom_args"] = latest_eas.custom_args if latest_eas else ""

            if latest_eas and latest_eas.barcodeKitName:
                bundle.data["sample"] = ""
                bundle.data["sampleDisplayedName"] = ""
            else:
                if experiment.samples.all():
                    bundle.data["sample"] = experiment.samples.all()[0].name
                    bundle.data["sampleDisplayedName"] = experiment.samples.all()[
                        0
                    ].displayedName
                else:
                    bundle.data["sample"] = ""
                    bundle.data["sampleDisplayedName"] = ""

            bundle.data["selectedPlugins"] = (
                latest_eas.selectedPlugins if latest_eas else ""
            )

            if latest_eas:
                selectedPlugins = latest_eas.selectedPlugins

                pluginInfo = selectedPlugins.get("variantCaller", {})

                if pluginInfo:
                    userInput = pluginInfo.get("userInput", {})
                    if userInput:
                        bundle.data["variantfrequency"] = userInput.get(
                            "variationtype", ""
                        )

            bundle.data["earlyDatFileDeletion"] = False
            if experiment.chipType:
                chip = bundle.obj.get_chipType()
                if chip:
                    chipObjs = models.Chip.objects.filter(name=chip)
                    if chipObjs:
                        if chipObjs[0].earlyDatFileDeletion == "1":
                            bundle.data["earlyDatFileDeletion"] = True

        except models.Experiment.DoesNotExist:
            logger.error(
                "Missing experiment for Plan %s(%s)"
                % (bundle.obj.planName, bundle.obj.pk)
            )

        sampleSets = bundle.obj.sampleSets.order_by("displayedName")
        bundle.data["sampleSetDisplayedName"] = ",".join(
            sampleSets.values_list("displayedName", flat=True)
        )

        for sampleset in sampleSets:
            if sampleset.libraryPrepType:
                bundle.data["libraryPrepType"] = sampleset.libraryPrepType
                bundle.data[
                    "libraryPrepTypeDisplayedName"
                ] = sampleset.get_libraryPrepType_display()


            if sampleset.combinedLibraryTubeLabel:
                libPoolId = bundle.obj.libraryPool
                if libPoolId:
                    bundle.data[
                        "combinedLibraryTubeLabel"
                    ] = sampleset.combinedLibraryTubeLabel.split(',')[int(libPoolId)-1]
                else:
                    bundle.data[
                        "combinedLibraryTubeLabel"
                    ] = sampleset.combinedLibraryTubeLabel
        applicationGroup = bundle.obj.applicationGroup
        bundle.data["applicationGroupDisplayedName"] = (
            applicationGroup.description if applicationGroup else ""
        )

        bundle.data[
            "applicationCategoryDisplayedName"
        ] = bundle.obj.get_applicationCategoryDisplayedName(bundle.obj.categories)

        sampleGrouping = bundle.obj.sampleGrouping
        bundle.data["sampleGroupingName"] = (
            sampleGrouping.displayedName if sampleGrouping else ""
        )

        # Chip and Manifold Tec Parameters from experiment
        bundle.data["chipTecDfltAmbient"] = experiment.chipTecDfltAmbient
        bundle.data["chipTecSlope"] = experiment.chipTecSlope
        bundle.data["chipTecMinThreshold"] = experiment.chipTecMinThreshold
        bundle.data["manTecDfltAmbient"] = experiment.manTecDfltAmbient
        bundle.data["manTecSlope"] = experiment.manTecSlope
        bundle.data["manTecMinThreshold"] = experiment.manTecMinThreshold

        bundle.data["metaData"] = experiment.metaData or {}


        # IonChef parameters from Experiment

        if experiment.chefInstrumentName:
            try:
                chefFields = [
                    f.name for f in experiment._meta.fields if f.name.startswith("chef")
                ]
                for field in chefFields:
                    bundle.data["chefInfo"][field] = getattr(experiment, field)
            except Exception:
                logger.error(
                    "Error getting Chef fields for Plan %s(%s)"
                    % (bundle.obj.planName, bundle.obj.pk)
                )

        return bundle

    def hydrate_autoAnalyze(self, bundle):
        if "autoAnalyze" in bundle.data:
            bundle.data["autoAnalyze"] = toBoolean(bundle.data["autoAnalyze"], True)
        return bundle

    def hydrate_barcodedSamples(self, bundle):
        barcodedSamples = bundle.data.get("barcodedSamples") or ""
        # soft validation, will not raise errors
        # valid barcodedSamples format is {'sample_XXX':{'barcodes':['IonXpress_001', ]}, }
        valid = True
        if barcodedSamples:
            # for both string and unicode
            if isinstance(barcodedSamples, basestring):
                # example: "barcodedSamples":"{'s1':{'barcodes': ['IonSet1_01']},'s2':
                # {'barcodes': ['IonSet1_02']},'s3':{'barcodes': ['IonSet1_03']}}"
                barcodedSamples = ast.literal_eval(barcodedSamples)

            try:
                barcoded_bedfiles = {}
                for k, v in list(barcodedSamples.items()):
                    if isinstance(v["barcodes"], list):
                        for bc in v["barcodes"]:
                            if not isinstance(bc, basestring):
                                logger.debug(
                                    "api.PlannedExperiment.hydrate_barcodedSamples() - INVALID bc - NOT an str - bc=%s"
                                    % (bc)
                                )
                                valid = False
                    else:
                        logger.debug(
                            "api.PlannedExperiment.hydrate_barcodedSamples() -  INVALID v[barcodes] - NOT a list!!! v[barcodes]=%s"
                            % (v["barcodes"])
                        )
                        valid = False

                    if "endBarcodes" in v and isinstance(v["endBarcodes"], list):
                        for bc in v["endBarcodes"]:
                            if not isinstance(bc, basestring):
                                logger.debug(
                                    "api.PlannedExperiment.hydrate_barcodedSamples() - INVALID bc - NOT an str - bc=%s"
                                    % (bc)
                                )
                                valid = False
                    else:
                        if "endBarcodes" in v:
                            logger.debug(
                                "api.PlannedExperiment.hydrate_barcodedSamples() -  INVALID v[endBarcodes] - NOT a list!!! v[endBarcodes]=%s"
                                % (v["endBarcodes"])
                            )
                            valid = False

                    if "dualBarcodes" in v and isinstance(v["dualBarcodes"], list):
                        for bc in v["dualBarcodes"]:
                            if not isinstance(bc, basestring):
                                logger.debug(
                                    "api.PlannedExperiment.hydrate_barcodedSamples() - INVALID bc - NOT an str - bc=%s"
                                    % (bc)
                                )
                                valid = False
                    else:
                        if "dualBarcodes" in v:
                            logger.debug(
                                "api.PlannedExperiment.hydrate_barcodedSamples() -  INVALID v[dualBarcodes] - NOT a list!!! v[dualBarcodes]=%s"
                                % (v["dualBarcodes"])
                            )
                            valid = False

                    if isinstance(v.get("barcodeSampleInfo"), dict):
                        for barcode in list(v["barcodeSampleInfo"].values()):
                            reference = barcode.get("reference") or bundle.data.get(
                                "reference", ""
                            )
                            # this also handles the sse/svb bed files
                            for target_or_hotspot in [
                                "targetRegionBedFile",
                                "hotSpotRegionBedFile",
                                "sseBedFile"
                            ]:
                                bedfile = barcode.get(target_or_hotspot)
                                if bedfile:
                                    found_key = "%s__%s" % (
                                        os.path.basename(bedfile),
                                        reference,
                                    )
                                    if found_key not in barcoded_bedfiles:
                                        barcoded_bedfiles[
                                            found_key
                                        ] = self._get_bedfile_path(bedfile, reference)
                                    if barcoded_bedfiles[found_key]:
                                        barcode[target_or_hotspot] = barcoded_bedfiles[
                                            found_key
                                        ]

                            nucleotideType = barcode.get("nucleotideType", "")
                            if nucleotideType and nucleotideType.lower() == "fusions":
                                barcode["nucleotideType"] = "RNA"

            except Exception:
                logger.error(traceback.format_exc())
                valid = False

            # validate for malformed JSON "barcodedSamples" - raise error if it is New Plan
            bundle.data["barcodedSamples"] = barcodedSamples
        return bundle

    def hydrate_barcodeId(self, bundle):
        if bundle.data.get("barcodeId") and bundle.data["barcodeId"].lower() == "none":
            bundle.data["barcodeId"] = ""
        if "barcodeId" in bundle.data:
            bundle.data["barcodeKitName"] = bundle.data["barcodeId"]
        return bundle

    def hydrate_endBarcodeKitName(self, bundle):
        if (
            bundle.data.get("endBarcodeKitName")
            and bundle.data["endBarcodeKitName"].lower() == "none"
        ):
            bundle.data["endBarcodeKitName"] = ""
        return bundle

    def hydrate_planStatus(self, bundle):
        planStatus = bundle.data.get("planStatus")
        if planStatus:
            defaultPlanStatus = plan_validator.get_default_planStatus()
            for status in defaultPlanStatus:
                if planStatus.lower() == status and planStatus != status:
                    bundle.data["planStatus"] = status
        return bundle

    def hydrate_bedfile(self, bundle):
        bedfile = bundle.data.get("bedfile")
        if bedfile:
            if bedfile.lower() == "none":
                bundle.data["bedfile"] = ""
            else:
                bedfile_path = self._get_bedfile_path(
                    bedfile, bundle.data.get("reference", "")
                )
                if bedfile_path:
                    bundle.data["bedfile"] = bedfile_path

        if "bedfile" in bundle.data:
            bundle.data["targetRegionBedFile"] = bundle.data["bedfile"]

        return bundle

    def hydrate_chipType(self, bundle):
        chipType = bundle.data.get("chipType", None)
        if chipType:
            if bundle.data["chipType"].lower() == "none":
                bundle.data["chipType"] = ""
            else:
                # persist appropriate name when description or lowercase chipName is provided
                chip = models.Chip.objects.filter(
                    Q(name__iexact=chipType) | Q(description__iexact=chipType)
                )
                if chip:
                    bundle.data["chipType"] = chip[0].name

        return bundle

    def hydrate_forward3primeadapter(self, bundle):
        if "forward3primeadapter" in bundle.data:
            bundle.data["threePrimeAdapter"] = bundle.data["forward3primeadapter"]
        return bundle

    def hydrate_library(self, bundle):
        if bundle.data.get("library") and bundle.data["library"].lower() == "none":
            bundle.data["library"] = ""
        if "library" in bundle.data:
            bundle.data["reference"] = bundle.data["library"]
        return bundle

    def hydrate_librarykitname(self, bundle):
        if (
            bundle.data.get("librarykitname")
            and bundle.data["librarykitname"].lower() == "none"
        ):
            bundle.data["librarykitname"] = ""
        if "librarykitname" in bundle.data:
            bundle.data["libraryKitName"] = bundle.data["librarykitname"]
        return bundle

    def hydrate_libraryKey(self, bundle):
        libraryKey = bundle.data.get("libraryKey", None)
        if libraryKey:
            error, selectedLibKey = plan_validator.validate_library_key(
                libraryKey, field_label="libraryKey"
            )
            if not error:
                bundle.data["libraryKey"] = selectedLibKey.sequence
        else:
            gc = models.GlobalConfig.get()
            bundle.data["libraryKey"] = gc.default_library_key
        return bundle

    def hydrate_flowsInOrder(self, bundle):
        if "flowsInOrder" in bundle.data:
            flowsInOrder = bundle.data.get("flowsInOrder")
            if flowsInOrder:
                # get flowsInOrder sequence if name, description is specified and persist in the database
                # Error out if not in A,C,G,T and blank(system default)
                error, selectedflowOrder = plan_validator.validate_flowOrder(
                    flowsInOrder, "flowsInOrder"
                )
                if not error:
                    bundle.data["flowsInOrder"] = selectedflowOrder
                else:
                    bundle.data["errorInflowOrder"] = error
        elif "sequencekitname" in bundle.data and not bundle.obj.pk:
            input = bundle.data.get("sequencekitname")
            if input:
                input = input.strip()
                selectedKits = models.KitInfo.objects.filter(
                    Q(kitType="SequencingKit")
                    & Q(isActive=True)
                    & Q(description__iexact=input)
                    | Q(name__iexact=input)
                )

                if selectedKits:
                    selectedKit = selectedKits[0]
                    if selectedKit.defaultFlowOrder:
                        bundle.data[
                            "flowsInOrder"
                        ] = selectedKit.defaultFlowOrder.flowOrder
        return bundle

    def hydrate_samplePrepProtocol(self, bundle):
        if "samplePrepProtocol" in bundle.data:
            samplePrepProtocol = bundle.data.get("samplePrepProtocol")
            if samplePrepProtocol:
                # Get the valid spp and persist if no error, Ex: if uuid is given, only spp value should be persisted.
                # Allowed values are samplePrepProtool value, UUID and blank(system default)
                templatingKitName = bundle.obj.templatingKitName
                error, selectedSamplePrepProtocol = plan_validator.validate_plan_samplePrepProtocol(
                    samplePrepProtocol,
                    templatingKitName,
                    field_label="samplePrepProtocol",
                    templatingKitName_label="templatingKitName",
                )
                if not error:
                    bundle.data["samplePrepProtocol"] = selectedSamplePrepProtocol
                else:
                    bundle.data["errorInSamplePrepProtocol"] = error

        return bundle

    def hydrate_regionfile(self, bundle):
        bedfile = bundle.data.get("regionfile")
        if bedfile:
            if bedfile.lower() == "none":
                bundle.data["regionfile"] = ""
            else:
                bedfile_path = self._get_bedfile_path(
                    bedfile, bundle.data.get("reference", "")
                )
                if bedfile_path:
                    bundle.data["regionfile"] = bedfile_path

        if "regionfile" in bundle.data:
            bundle.data["hotSpotRegionBedFile"] = bundle.data["regionfile"]

        return bundle

    def hydrate_sseBedFile(self, bundle):
        bedfile = bundle.data.get("sseBedFile")
        if bedfile:
            if bedfile.lower() == "none":
                bundle.data["sseBedFile"] = ""
            else:
                bedfile_path = self._get_bedfile_path(
                    bedfile, bundle.data.get("reference", "")
                )
                if bedfile_path:
                    bundle.data["sseBedFile"] = bedfile_path

        return bundle

    def hydrate_mixedTypeRNA_reference(self, bundle):
        if (
            bundle.data.get("mixedTypeRNA_reference")
            and bundle.data["mixedTypeRNA_reference"].lower() == "none"
        ):
            bundle.data["mixedTypeRNA_reference"] = ""
        if "mixedTypeRNA_reference" in bundle.data:
            bundle.data["mixedTypeRNA_reference"] = bundle.data[
                "mixedTypeRNA_reference"
            ]
        return bundle

    def hydrate_mixedTypeRNA_targetRegionBedFile(self, bundle):
        bedfile = bundle.data.get("mixedTypeRNA_targetRegionBedFile")
        if bedfile:
            if bedfile.lower() == "none":
                bundle.data["mixedTypeRNA_targetRegionBedFile"] = ""
            else:
                bedfile_path = self._get_bedfile_path(
                    bedfile, bundle.data.get("mixedTypeRNA_reference", "")
                )
                if bedfile_path:
                    bundle.data["mixedTypeRNA_targetRegionBedFile"] = bedfile_path

        if "mixedTypeRNA_targetRegionBedFile" in bundle.data:
            bundle.data["mixedTypeRNA_targetRegionBedFile"] = bundle.data[
                "mixedTypeRNA_targetRegionBedFile"
            ]

        return bundle

    def hydrate_mixedTypeRNA_hotSpotRegionBedFile(self, bundle):
        bedfile = bundle.data.get("mixedTypeRNA_hotSpotRegionBedFile")
        if bedfile:
            if bedfile.lower() == "none":
                bundle.data["mixedTypeRNA_hotSpotRegionBedFile"] = ""
            else:
                bedfile_path = self._get_bedfile_path(
                    bedfile, bundle.data.get("mixedTypeRNA_reference", "")
                )
                if bedfile_path:
                    bundle.data["mixedTypeRNA_hotSpotRegionBedFile"] = bedfile_path

        if "mixedTypeRNA_hotSpotRegionBedFile" in bundle.data:
            bundle.data["mixedTypeRNA_hotSpotRegionBedFile"] = bundle.data[
                "mixedTypeRNA_hotSpotRegionBedFile"
            ]

        return bundle

    def hydrate_selectedPlugins(self, bundle):
        selectedPlugins = bundle.data.get("selectedPlugins", "")
        # soft validation, will not raise errors
        # valid selectedPlugins format is {'plugin_XXX': {'name': 'plugin_XXX', 'userInput': {} }, }
        valid = True
        if selectedPlugins:
            try:
                for k, v in list(selectedPlugins.items()):
                    name = v["name"]
                    userInput = v.get("userInput", {})
                    if not isinstance(userInput, (dict, list, basestring)):
                        valid = False
            except Exception:
                valid = False

            bundle.data["selectedPlugins"] = selectedPlugins if valid else ""
        return bundle

    def hydrate_sequencekitname(self, bundle):
        if (
            bundle.data.get("sequencekitname")
            and bundle.data["sequencekitname"].lower() == "none"
        ):
            bundle.data["sequencekitname"] = ""

        sequencekitname = bundle.data.get("sequencekitname", None)

        if sequencekitname:
            value = sequencekitname.strip()
            # persist only the name when description is provided
            kit = models.KitInfo.objects.filter(
                Q(kitType__in=["SequencingKit"]) & Q(name__iexact=value)
                | Q(description__iexact=value)
            )
            if kit:
                bundle.data["sequencekitname"] = kit[0].name

        return bundle

    def hydrate_templatingKitName(self, bundle):
        if (
            bundle.data.get("templatingKitName")
            and bundle.data["templatingKitName"].lower() == "none"
        ):
            bundle.data["templatingKitName"] = ""

        templatingKitName = bundle.data.get("templatingKitName", None)

        if templatingKitName:
            value = templatingKitName.strip()
            # persist only the name when description is provided
            kit = models.KitInfo.objects.filter(
                Q(kitType__in=["TemplatingKit", "IonChefPrepKit"])
                & Q(name__iexact=value)
                | Q(description__iexact=value)
            )
            if kit:
                bundle.data["templatingKitName"] = kit[0].name

        return bundle

    def hydrate_isDuplicateReads(self, bundle):
        if "isDuplicateReads" in bundle.data:
            bundle.data["isDuplicateReads"] = toBoolean(
                bundle.data["isDuplicateReads"], False
            )
        return bundle

    def hydrate_realign(self, bundle):
        if "realign" in bundle.data:
            bundle.data["realign"] = toBoolean(bundle.data["realign"], False)
        return bundle

    def hydrate_metaData(self, bundle):
        # logger.debug("api.PlannedExperimentResource.hydrate_metaData()
        # metaData=%s" %(bundle.data.get('metaData', "")))
        plan_metaData = bundle.data.get("metaData", "")
        valid = True
        if plan_metaData:
            try:
                if isinstance(plan_metaData, basestring):
                    plan_metaData_dict = ast.literal_eval(plan_metaData)
                    plan_metaData = json.dumps(plan_metaData_dict, cls=LazyJSONEncoder)
            except Exception:
                logger.error(traceback.format_exc())
                valid = False
                raise SDKValidationError(
                    "Error: Invalid JSON value for field metaData=%s in plannedExperiment with planName=%s"
                    % (plan_metaData, bundle.data.get("planName", ""))
                )  # TODO: i18n

            bundle.data["metaData"] = plan_metaData if valid else ""
        return bundle

    def hydrate_platform(self, bundle):
        if "platform" in bundle.data:
            platform = bundle.data["platform"].upper()
            if platform and platform.lower() == "none":
                platform = ""
            bundle.data["platform"] = platform
        return bundle

    def hydrate(self, bundle):
        # boolean handling for API posting
        for key in [
            "planExecuted",
            "isReverseRun",
            "isReusable",
            "isFavorite",
            "isSystem",
            "isSystemDefault",
            "isPlanGroup",
        ]:
            if key in bundle.data:
                bundle.data[key] = toBoolean(bundle.data[key], False)

        for key in ["preAnalysis", "usePreBeadfind", "usePostBeadfind"]:
            if key in bundle.data:
                bundle.data[key] = toBoolean(bundle.data[key], True)

        # run plan created on TS should not have post-run-bead find enabled for S5 and proton
        if "usePostBeadfind" not in bundle.data:
            bundleChipType = bundle.data.get("chipType")
            if bundleChipType:
                isPostbead_disable_chips = models.Chip.objects.filter(
                    name=bundleChipType, instrumentType__in=["proton", "S5"]
                )
                if isPostbead_disable_chips:
                    bundle.data["usePostBeadfind"] = False

        applicationGroupDisplayedName = bundle.data.get(
            "applicationGroupDisplayedName", ""
        )

        if applicationGroupDisplayedName:
            applicationGroups = models.ApplicationGroup.objects.filter(
                description__iexact=applicationGroupDisplayedName.strip()
            )
            if applicationGroups:
                bundle.data["applicationGroup"] = applicationGroups[0]

        sampleGroupingName = bundle.data.get("sampleGroupingName", "")

        if sampleGroupingName:
            sampleGroupings = models.SampleGroupType_CV.objects.filter(
                displayedName__iexact=sampleGroupingName.strip()
            )
            if sampleGroupings:
                bundle.data["sampleGrouping"] = sampleGroupings[0]

        # strip leading zeros of sampleTubeLabel
        if (
            bundle.data.get("planStatus", "") != "run"
            or not bundle.data["planExecuted"]
        ):
            sampleTubeLabel = bundle.data.get("sampleTubeLabel", "")
            if sampleTubeLabel:
                # sampleTubeLabel = sampleTubeLabel.strip().lstrip("0")
                sampleTubeLabel = sampleTubeLabel.strip()
                bundle.data["sampleTubeLabel"] = sampleTubeLabel

        # update Fusions BED file path, if specified
        rna_bedfile = bundle.data.get("mixedTypeRNA_targetRegionBedFile")
        rna_reference = bundle.data.get("mixedTypeRNA_reference")
        if rna_bedfile and rna_reference:
            rna_bedfile_path = self._get_bedfile_path(rna_bedfile, rna_reference)
            if rna_bedfile_path:
                bundle.data["mixedTypeRNA_targetRegionBedFile"] = rna_bedfile_path

        return bundle

    def _get_bedfile_path(self, bedfile, reference):
        bedfile_path = ""
        if os.path.exists(bedfile):
            bedfile_path = bedfile
        else:
            name = os.path.basename(bedfile)
            path = "/%s/unmerged/detail/%s" % (reference, name)
            content = models.Content.objects.filter(publisher__name="BED", path=path)
            if content:
                bedfile_path = content[0].file
        return bedfile_path

    def _isNewPlan(self, bundle, **kwargs):
        isNewPlan = True
        if bundle.obj and bundle.obj.pk:
            isNewPlan = False
        elif bundle.data.get("id") or kwargs.get("pk"):
            isNewPlan = False

        return isNewPlan

    def _include_default_selected_plugins(self, bundle, **kwargs):
        include_plugin_default_selection = bundle.data.get(
            "include_plugin_default_selection", True
        )
        default_selected_plugins = None
        if not include_plugin_default_selection:
            return

        default_selected_plugins = models.Plugin.objects.filter(
            active=True, selected=True, defaultSelected=True
        ).order_by("name")

        if not default_selected_plugins:
            return

        try:
            isNewPlan = bundle.data["isNewPlan"]
        except Exception:
            isNewPlan = False

        try:
            isSystem = bundle.data["isSystem"]
        except Exception:
            isSystem = False

        if isSystem or not isNewPlan:
            return

        # if include_plugin_default_selection and isNewPlan and NOT isSystem, then
        # add in plugins to selectedPlugins.  Watch for duplicates!!!
        selectedPlugins = bundle.data.get("selectedPlugins") or {}

        for default_plugin in default_selected_plugins:
            if default_plugin.name not in list(selectedPlugins.keys()):
                selectedPlugins[default_plugin.name] = {
                    "id": default_plugin.id,
                    "name": default_plugin.name,
                    "version": default_plugin.version,
                    "userInput": {},
                    "features": [],
                }
        bundle.data["selectedPlugins"] = selectedPlugins
        return bundle

    def obj_create(self, bundle, request=None, **kwargs):
        """
        A ORM-specific implementation of ``obj_create``.
        """
        bundle.obj = self._meta.object_class()

        for key, value in list(kwargs.items()):
            setattr(bundle.obj, key, value)

        bundle = self.full_hydrate(bundle)
        bundle.data["isNewPlan"] = self._isNewPlan(bundle, **kwargs)
        if bundle.data["isNewPlan"] and not bundle.obj.origin:
            bundle.obj.origin = "api"

        logger.debug(
            "PlannedExperimentResource.obj_create()...bundle.data=%s" % bundle.data
        )
        if bundle.obj.origin not in ["transfer", "ampliseq.com"]:
            # ignore the default plugin selection during plan transfer and ampliseq.com panel import
            # transfer plugins only if selected in origin
            self._include_default_selected_plugins(bundle, **kwargs)

        # validate plan bundle
        self.is_valid(bundle)

        # Save FKs just in case.
        self.save_related(bundle)

        try:
            with transaction.commit_on_success():
                bundle.obj.save()
                bundle.obj.save_plannedExperiment_association(
                    bundle.data.pop("isNewPlan"), **bundle.data
                )
                bundle.obj.update_plan_qcValues(**bundle.data)
        except Exception:
            logger.error("Failed PlannedExperimentResource.obj_create()")
            logger.error(traceback.format_exc())
            return HttpBadRequest()

        # Now pick up the M2M bits.
        m2m_bundle = self.hydrate_m2m(bundle)
        self.save_m2m(m2m_bundle)
        return bundle

    def obj_update(self, bundle, **kwargs):
        logger.debug(
            "PlannedExperimentResource.obj_update() bundle.data=%s" % bundle.data
        )

        # log changes for plan history
        bundle.obj.update_changed_fields_for_plan_history(bundle.data, bundle.obj)

        bundle = super(PlannedExperimentResource, self).obj_update(bundle, **kwargs)
        bundle.obj.save_plannedExperiment_association(False, **bundle.data)

        bundle.obj.save_plan_history_log()
        return bundle

    def dispatch_create(self, request, **kwargs):
        return self.dispatch("create", request, **kwargs)

    def post_create(self, request, **kwargs):
        """
            Custom endpoint /plannedexperiment/create/ to create Plan from existing Template and optionally Sample Set
        """

        def _get_template(templateName):
            # return Plan Template, must exist and name has to be unique
            if not templateName:
                raise SDKValidationError(
                    {"templateName": validation.required_error("templateName")}
                )

            plans = models.PlannedExperiment.objects.filter(
                Q(planDisplayedName=templateName) | Q(planName=templateName)
            )
            if not plans:
                raise SDKValidationError(
                    {
                        "templateName": validation.invalid_not_found_error(
                            "templateName", templateName
                        )
                    }
                )
            elif plans.count() > 1:
                # limit to templates only
                plans = plans.filter(isReusable=True)
                if plans.count() > 1:
                    raise SDKValidationError(
                        {
                            "templateName": "Found %d objects with name=%s. Please rename your Plan Template and try again"
                            % (template.count(), templateName)
                        }
                    )
            else:
                return plans[0]

        def _get_barcodedSamples(samplesList, default_nuctype):
            # create barcodedSamples from payload or SampleSet samples
            errors = []
            processed_barcodes = []
            try:
                for sample in samplesList:
                    sampleName = sample["sampleName"].replace(" ", "_")
                    barcode = sample.get("dnabarcode")

                    if not barcode:
                        errors.append(
                            validation.invalid_required_related(
                                "dnabarcode", sampleName
                            )
                        )
                        continue
                    elif barcode in processed_barcodes:
                        errors.append(
                            "multiple samples assigned to barcode: %s" % barcode
                        )
                        continue
                    else:
                        processed_barcodes.append(barcode)

                    # add sample to barcodedSamples
                    if sampleName not in barcodedSamples:
                        barcodedSamples[sampleName] = {
                            "barcodes": [],
                            "barcodeSampleInfo": {},
                        }

                    barcodedSamples[sampleName]["barcodes"].append(barcode)
                    barcodedSamples[sampleName]["barcodeSampleInfo"][barcode] = {
                        "controlType": sample.get("controlType", ""),
                        "description": sample.get("description")
                        or sample.get("sampleDescription", ""),
                        "externalId": sample.get("externalId")
                        or sample.get("sampleExternalId", ""),
                        "nucleotideType": models.SampleSetItem.nuctype_for_planning(
                            sample.get("nucleotideType")
                        )
                        or default_nuctype,
                        "reference": sample.get("reference", ""),
                        "targetRegionBedFile": sample.get("targetRegionBedFile", ""),
                        "hotSpotRegionBedFile": sample.get("hotSpotRegionBedFile", ""),
                    }

            except (TypeError, KeyError):
                errors.append("failed to parse samples info - incorrect format")
            except Exception as err:
                errors.append(str(err))

            if errors:
                raise SDKValidationError({"samples": " | ".join(errors)})

            return barcodedSamples

        deserialized = self.deserialize(
            request,
            request.body,
            format=request.META.get("CONTENT_TYPE") or "application/json",
        )
        data = dict_strip_unicode_keys(deserialized)

        # crate a copy of specified Template
        templateObj = _get_template(data.pop("templateName", ""))

        bundle = self._copy_plan_bundle_from_obj(templateObj)
        bundle.data["planDisplayedName"] = data.pop("planName", "")
        bundle.data["planName"] = bundle.data["planDisplayedName"].replace(" ", "_")
        bundle.data["isReusable"] = False
        bundle.data["origin"] = "api"
        bundle.data.pop(
            "planStatus"
        )  # will be "planned" or "pending" based on Templating Kit

        # request payload can overwrite parameters
        for key, value in list(data.items()):
            if key in bundle.data:
                bundle.data[key] = value

        # get samples info, accept barcodedSamples or samplesList
        barcodedSamples = data.get("barcodedSamples") or {}
        payload_samples_key = "barcodedSamples"
        default_nuctype = templateObj.get_default_nucleotideType()
        isDynamicTecParamsModified, errMsg = plan_validator.get_dynamicTecParams(templateObj, data)
        if isDynamicTecParamsModified:
            raise SDKValidationError(
                {
                    "Manifold Tec Params": errMsg
                }
            )
        if data.get("samplesList"):
            if barcodedSamples:
                raise SDKValidationError(
                    {
                        "samples": "Please use barcodedSamples or samplesList to specify Samples, but not both"
                    }
                )
            else:
                barcodedSamples = _get_barcodedSamples(
                    data["samplesList"], default_nuctype
                )
                payload_samples_key = "samplesList"

        # process SampleSet, if specified
        sampleSetName = data.get("sampleSetName")
        libraryPoolId = data.get("libraryPool")
        if sampleSetName:
            try:
                sampleSet = models.SampleSet.objects.get(displayedName=sampleSetName)
            except models.SampleSet.DoesNotExist:
                raise SDKValidationError(
                    {"sampleSetName": "sampleSetName %s does not exist" % sampleSetName}
                )
            except models.SampleSet.MultipleObjectsReturned:
                raise SDKValidationError(
                    "More than one object is found for sampleSetName: %s"
                    % sampleSetName
                )

            if sampleSet.status in ["libPrep_pending", "voided"]:
                raise SDKValidationError(
                    {
                        "sampleSetName": "sampleSetName: %s is not ready for planning (status: %s)"
                        % (sampleSetName, sampleSet.status)
                    }
                )

            if sampleSet.samples.count() == 0:
                raise SDKValidationError(
                    {"samples": "No samples in SampleSet: %s" % sampleSetName}
                )

            # set sample group from payload if present and not null, then sample set, and then finally template
            if data.get("sampleGroupingName", ""):
                # sampleGroupingName is part of payload and the value is not empty or null
                sampleGroupingName = data.get("sampleGroupingName")
            elif sampleSet.SampleGroupType_CV:
                # check if it is defined in sample set
                sampleGroupingName = sampleSet.SampleGroupType_CV.displayedName
            elif templateObj.sampleGrouping:
                # check if it is defined in template
                sampleGroupingName = templateObj.sampleGrouping.displayedName
            else:
                sampleGroupingName = ""

            # add SampleSet to Plan data
            bundle.data["sampleSetDisplayedName"] = sampleSet.displayedName
            bundle.data["sampleGroupingName"] = sampleGroupingName
            bundle.data["librarykitname"] = sampleSet.libraryPrepKitName
            bundle.data["libraryPool"] = libraryPoolId

            if not barcodedSamples:
                # get samples from SampleSet
                sampleset_samples = []
                sampleset_barcodeKit = ""
                libraryPoolPlanData = []
                if libraryPoolId:
                    libPool = "pool"+str(libraryPoolId)
                    multiPoolPlanData = views_helper.processMultiPoolPlanSupport([sampleSet])
                    if libPool in multiPoolPlanData.keys():
                        if multiPoolPlanData[libPool]:
                            libraryPoolPlanData = multiPoolPlanData[libPool].split(",")
                    else:
                        raise SDKValidationError(
                            {"libraryPool": "Invalid library pool id. Valid values are 1 or 2. Libraries should be either Pool: 1(A-D) or 2(E-H): %s" % sampleSetName}
                        )
                for item in sampleSet.samples.all():
                    if libraryPoolPlanData and str(item.id) not in libraryPoolPlanData:
                        continue
                    sampleset_samples.append(
                        {
                            "sampleName": item.sample.name,
                            "dnabarcode": item.dnabarcode.id_str
                            if item.dnabarcode
                            else "",
                            "controlType": item.controlType,
                            "description": item.description,
                            "externalId": item.sample.externalId,
                            "nucleotideType": item.nucleotideType,
                            # add reference and BED files
                            "reference": bundle.data["library"],
                            "targetRegionBedFile": bundle.data["bedfile"],
                            "hotSpotRegionBedFile": bundle.data["regionfile"],
                        }
                    )
                    if item.dnabarcode:
                        sampleset_barcodeKit = item.dnabarcode.name
                barcodedSamples = _get_barcodedSamples(
                    sampleset_samples, default_nuctype
                )

                if not bundle.data["barcodeId"]:
                    bundle.data["barcodeId"] = sampleset_barcodeKit

            else:
                # validate specified samples are compatible with SampleSet
                errors = []
                warnings = []
                available_samples = sampleSet.samples.values_list(
                    "sample__name", flat=True
                )
                for sampleName, info in list(barcodedSamples.items()):
                    if sampleName not in available_samples:
                        errors.append(
                            validation.invalid_choice(
                                "sample", sampleName, available_samples
                            )
                        )

                if errors:
                    raise SDKValidationError({"samples": " | ".join(errors)})

        bundle.data["barcodedSamples"] = barcodedSamples

        if templateObj.experiment.chipTecDfltAmbient or templateObj.experiment.manTecDfltAmbient:
            bundle.data["chipTecDfltAmbient"] = templateObj.experiment.chipTecDfltAmbient
            bundle.data["chipTecSlope"] = templateObj.experiment.chipTecSlope
            bundle.data["chipTecMinThreshold"] = templateObj.experiment.chipTecMinThreshold
            bundle.data["manTecDfltAmbient"] = templateObj.experiment.manTecDfltAmbient
            bundle.data["manTecSlope"] = templateObj.experiment.manTecSlope
            bundle.data["manTecMinThreshold"] = templateObj.experiment.manTecMinThreshold

        # validate all fields and create the plan
        bundle = self.obj_create(bundle)
        return self.create_response(
            request, self.full_dehydrate(bundle), response_class=HttpAccepted
        )

    def get_create(self, request, **kwargs):
        # return payload examples for plannedexperiment/create/ API endpoint
        examples = getAPIexamples("/api/v1/plannedexperiment/create/")
        return self.create_response(request, examples)
