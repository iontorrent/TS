# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from django.utils.translation import ugettext as _, ugettext_lazy, ugettext
from django import http

from django.contrib.auth.decorators import login_required
from django.template import RequestContext
from django.shortcuts import render_to_response, get_object_or_404, get_list_or_404
from django.conf import settings
from django.db import transaction
from django.http import (
    HttpResponse,
    HttpResponseRedirect,
    Http404,
    HttpResponseServerError,
)

from traceback import format_exc
import json
import simplejson
import datetime
from django.utils import timezone

import logging

from django.core import serializers
from django.core.urlresolvers import reverse
from django.db.models import Q
from django.db.models import Count
from iondb.rundb.json_lazy import LazyJSONEncoder, LazyDjangoJSONEncoder

from iondb.utils import utils, validation, i18n_errors
from django.forms.models import model_to_dict
import os
from distutils.version import StrictVersion
import string
import traceback
import tempfile
import csv
import types
from django.core.exceptions import ValidationError

from iondb.rundb.models import (
    Sample,
    SampleSet,
    SampleSetItem,
    SampleAttribute,
    SampleGroupType_CV,
    SampleAnnotation_CV,
    SampleAttributeDataType,
    SampleAttributeValue,
    PlannedExperiment,
    dnaBarcode,
    GlobalConfig,
    SamplePrepData,
    KitInfo,
    common_CV,
    ChefPcrPlateconfig)
from iondb.rundb.labels import (
    Sample as _Sample,
    SampleAttribute as _SampleAttribute,
    SampleSet as _SampleSet,
    SampleSetItem as _SampleSetItem,
)

# from iondb.rundb.api import SampleSetItemInfoResource

from django.contrib.auth.models import User
from django.conf import settings
from django.views.generic.detail import DetailView

import import_sample_processor
import views_helper
import sample_validator

from iondb.utils import toBoolean
from iondb.utils.unicode_csv import is_ascii
from iondb.utils.validation import SeparatedValuesBuilder

logger = logging.getLogger(__name__)


def clear_samplesetitem_session(request):
    if request.session.get("input_samples", None):
        request.session["input_samples"].pop("pending_sampleSetItem_list", None)
        request.session.pop("input_samples", None)
    return HttpResponse("Manually Entered Sample session has been cleared")


def _get_sample_groupType_CV_list(request):
    sample_groupType_CV_list = None
    isSupported = GlobalConfig.get().enable_compendia_OCP

    if isSupported:
        sample_groupType_CV_list = SampleGroupType_CV.objects.all().order_by(
            "displayedName"
        )
    else:
        sample_groupType_CV_list = (
            SampleGroupType_CV.objects.all()
            .exclude(displayedName="DNA_RNA")
            .order_by("displayedName")
        )

    return sample_groupType_CV_list


def _get_sampleSet_list(request):
    """
    Returns a list of sample sets to which we can still add samples
    """
    sampleSet_list = None
    isSupported = GlobalConfig.get().enable_compendia_OCP

    if isSupported:
        sampleSet_list = SampleSet.objects.all().order_by(
            "-lastModifiedDate", "displayedName"
        )
    else:
        sampleSet_list = (
            SampleSet.objects.all()
            .exclude(SampleGroupType_CV__displayedName="DNA_RNA")
            .order_by("-lastModifiedDate", "displayedName")
        )

    # exclude sample sets that are of amps_on_chef_v1 AND already have 8 samples
    annotated_list = sampleSet_list.exclude(
        status__in=["voided", "libPrep_reserved"]
    ).annotate(Count("samples"))
    exclude_id_list = annotated_list.values_list("id", flat=True).filter(
        libraryPrepType="amps_on_chef_v1", samples__count=8
    )
    available_sampleSet_list = annotated_list.exclude(pk__in=exclude_id_list)
    logger.debug(
        "_get_sampleSet_list() sampleSet_list.count=%d; available_sampleSet_list=%d"
        % (annotated_list.count(), available_sampleSet_list.count())
    )

    return available_sampleSet_list


def _get_all_userTemplates(request):
    isSupported = GlobalConfig.get().enable_compendia_OCP

    all_templates = None
    if isSupported:
        all_templates = PlannedExperiment.objects.filter(
            isReusable=True, isSystem=False
        ).order_by("applicationGroup", "sampleGrouping", "-date", "planDisplayedName")

    else:
        all_templates = (
            PlannedExperiment.objects.filter(isReusable=True, isSystem=False)
            .exclude(sampleGrouping__displayedName="DNA_RNA")
            .order_by(
                "applicationGroup", "sampleGrouping", "-date", "planDisplayedName"
            )
        )

    return all_templates


def _get_all_systemTemplates(request):
    isSupported = GlobalConfig.get().enable_compendia_OCP

    all_templates = None
    if isSupported:
        all_templates = PlannedExperiment.objects.filter(
            isReusable=True, isSystem=True, isSystemDefault=False
        ).order_by("applicationGroup", "sampleGrouping", "planDisplayedName")

    else:
        all_templates = (
            PlannedExperiment.objects.filter(
                isReusable=True, isSystem=True, isSystemDefault=False
            )
            .exclude(sampleGrouping__displayedName="DNA_RNA")
            .order_by("applicationGroup", "sampleGrouping", "planDisplayedName")
        )

    return all_templates


def get_custom_sample_column_list(request):
    custom_sample_column_list = list(
        SampleAttribute.objects.filter(isActive=True)
        .values_list("displayedName", flat=True)
        .order_by("id")
    )
    ctxd = {"custom_sample_column_list": simplejson.dumps(custom_sample_column_list)}

    ctx = RequestContext(request, ctxd)

    return ctx


def show_samplesets(request):
    """
    show the sample sets home page
    """
    ctx = get_custom_sample_column_list(request)

    return render_to_response(
        "rundb/sample/samplesets.html", context_instance=ctx, mimetype="text/html"
    )


def show_sample_attributes(request):
    """
    show the user-defined sample attribute home page
    """

    ctxd = {}

    ctx = RequestContext(request, ctxd)
    return render_to_response(
        "rundb/sample/sampleattributes.html", context_instance=ctx, mimetype="text/html"
    )


def download_samplefile_format(request):
    """
    download sample file format
    """

    response = HttpResponse(mimetype="text/csv")
    now = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    response["Content-Disposition"] = (
        "attachment; filename=sample_file_format_%s.csv" % now
    )

    sample_csv_version = import_sample_processor.get_sample_csv_version()

    hdr = [
        import_sample_processor.COLUMN_SAMPLE_NAME,
        import_sample_processor.COLUMN_SAMPLE_EXT_ID,
        import_sample_processor.COLUMN_CONTROLTYPE,
        import_sample_processor.COLUMN_PCR_PLATE_POSITION,
        import_sample_processor.COLUMN_BARCODE_KIT,
        import_sample_processor.COLUMN_BARCODE,
        import_sample_processor.COLUMN_GENDER,
        import_sample_processor.COLUMN_GROUP_TYPE,
        import_sample_processor.COLUMN_GROUP,
        import_sample_processor.COLUMN_SAMPLE_DESCRIPTION,
        import_sample_processor.COLUMN_SAMPLE_COLLECTION_DATE,
        import_sample_processor.COLUMN_SAMPLE_RECEIPT_DATE,
        import_sample_processor.COLUMN_NUCLEOTIDE_TYPE,
        import_sample_processor.COLUMN_SAMPLE_SOURCE,
        import_sample_processor.COLUMN_PANEL_POOL_TYPE,
        import_sample_processor.COLUMN_CANCER_TYPE,
        import_sample_processor.COLUMN_CELLULARITY_PCT,
        import_sample_processor.COLUMN_BIOPSY_DAYS,
        import_sample_processor.COLUMN_CELL_NUM,
        import_sample_processor.COLUMN_COUPLE_ID,
        import_sample_processor.COLUMN_EMBRYO_ID,
        import_sample_processor.COLUMN_SAMPLE_POPULATION,
        import_sample_processor.COLUMN_SAMPLE_MOUSE_STRAINS,
    ]

    customAttributes = (
        SampleAttribute.objects.all().exclude(isActive=False).order_by("displayedName")
    )
    for customAttribute in customAttributes:
        hdr.append(customAttribute)

    writer = csv.writer(response)
    writer.writerow(sample_csv_version)
    writer.writerow(hdr)

    return response


def show_import_samplesetitems(request):
    """
    show the page to import samples from file for sample set creation
    """
    ctxd = {}
    sampleSet_list = _get_sampleSet_list(request)
    sampleGroupType_list = list(_get_sample_groupType_CV_list(request))
    libraryPrepType_choices = views_helper._get_libraryPrepType_choices(request)
    libraryPrepKits = KitInfo.objects.filter(kitType="LibraryPrepKit", isActive=True)
    libraryPrepProtocol_choices = views_helper._get_libraryPrepProtocol_choices(request)
    additionalCycles_choices = views_helper._get_additionalCycles_choices(request)

    ctxd = {
        "sampleSet_list": sampleSet_list,
        "sampleGroupType_list": sampleGroupType_list,
        "libraryPrepType_choices": libraryPrepType_choices,
        "libraryPrepKits": libraryPrepKits,
        "libraryPrepProtocol_choices": libraryPrepProtocol_choices,
        "additionalCycles_choices": additionalCycles_choices,
        "rowErrorsFormat": ugettext_lazy(
            "global.messages.validation.format.row.errors"
        ),  # "'Row %(n)s contained errors:'
        "columnErrorsFormat": ugettext_lazy(
            "global.messages.validation.format.column.errors"
        ),  # '<strong>%(columnName)s</strong> column : %(columnErrors)s'
        "fieldErrorsFormat": ugettext_lazy(
            "global.messages.validation.format.field.errors"
        ),  # '%(fieldName)s contained errors:'s
    }

    ctx = RequestContext(request, ctxd)
    return render_to_response(
        "rundb/sample/import_samples.html", context_instance=ctx, mimetype="text/html"
    )


def show_edit_sampleset(request, _id=None):
    """
    show the sample set add/edit page
    """
    ctxd = {}
    if _id:
        sampleSet = get_object_or_404(SampleSet, pk=_id)
        intent = "edit"
        editable = sampleSet.status in ["", "created", "libPrep_pending"]
        # Allow user to use advanced edit page for any major updates,
        #   since PCR plate and tube position is assigned to items dynamically
        isAmpliseq_or_HD = sampleSet.libraryPrepType in [
            "amps_on_chef_v1",
            "amps_hd_on_chef_v1",
        ]
    else:
        sampleSet = None
        intent = "add"
        editable = True

    sampleGroupType_list = _get_sample_groupType_CV_list(request)
    libraryPrepType_choices = views_helper._get_libraryPrepType_choices(request)
    libraryPrepProtocol_choices = views_helper._get_libraryPrepProtocol_choices(request)
    additionalCycles_choices = views_helper._get_additionalCycles_choices(request)

    if editable:
        libraryPrepKits = KitInfo.objects.filter(
            kitType="LibraryPrepKit", isActive=True
        )
    else:
        libraryPrepKits = KitInfo.objects.filter(name=sampleSet.libraryPrepKitName)

    ctxd = {
        "sampleSet": sampleSet,
        "sampleGroupType_list": sampleGroupType_list,
        "libraryPrepType_choices": libraryPrepType_choices,
        "libraryPrepProtocol_choices": libraryPrepProtocol_choices,
        "additionalCycles_choices": additionalCycles_choices,
        "libraryPrepKits": libraryPrepKits,
        "intent": intent,
        "editable": editable,
        "isAmpliseq_or_HD": isAmpliseq_or_HD,
        "form": {
            "title": _("samplesets.addedit.title.edit")
            if "edit" == intent
            else _("samplesets.addedit.title.add"),
            "action": "/sample/sampleset/edited/"
            if "edit" == intent
            else "/sample/sampleset/added/",
            "fields": {
                "groupType": {
                    "editable": editable,
                    "title": _("samplesets.addedit.fields.groupType.tooltip")
                    % {"status": sampleSet.status}
                    if editable
                    else _("samplesets.addedit.fields.groupType.tooltip.disabled")
                    % {"status": sampleSet.status},
                    # 'Group Type cannot be changed since this sample set status is {{sampleSet.status}}'
                    "title_amp_hd": "Group Type cannot be changed here. Click the Advanced Edit button to update."
                    if isAmpliseq_or_HD
                    else "",
                },
                "libraryPrepType": {
                    "editable": editable,
                    "title": _("samplesets.addedit.fields.libraryPrepType.tooltip")
                    % {"status": sampleSet.status}
                    if editable
                    else _("samplesets.addedit.fields.libraryPrepType.tooltip.disabled")
                    % {"status": sampleSet.status},
                    # 'Library Prep Type cannot be changed since this sample set status is {{sampleSet.status}}'
                    "title_amp_hd": "Library Prep Type cannot be changed here. Click the Advanced Edit button to update."
                    if isAmpliseq_or_HD
                    else "",
                },
                "libraryPrepKit": {
                    "editable": editable,
                    "title": _("samplesets.addedit.fields.libraryPrepKit.tooltip")
                    % {"status": sampleSet.status}
                    if editable
                    else _("samplesets.addedit.fields.libraryPrepKit.tooltip.disabled")
                    % {"status": sampleSet.status},
                    # 'Library Prep Kit cannot be changed since this sample set status is {{sampleSet.status}}'
                    "title_amp_hd": "Library Prep Kit cannot be changed here. Click the Advanced Edit button to update."
                    if isAmpliseq_or_HD
                    else "",
                },
            },
        },
    }
    ctx = RequestContext(request, ctxd)
    logger.info("ctx %s", ctxd)
    return render_to_response(
        "rundb/sample/modal_add_sampleset.html",
        context_instance=ctx,
        mimetype="text/html",
    )

def show_plan_run(request, ids):
    """
    show the plan run popup
    """
    warnings = []
    multiLibraryPoolPlanData = {}
    sampleset_ids = ids.split(",")
    sampleSets = SampleSet.objects.filter(pk__in=sampleset_ids)
    if len(sampleSets) < 1:
        raise Http404(
            validation.invalid_not_found_error(_SampleSet.verbose_name, sampleset_ids)
        )
    isMultiPoolSupport = sampleSets.filter(categories__contains="multiPoolSupport").values_list("categories", flat=True)
    # Validate for  multiPoolSupport protocol for all the selected sampleSets
    if isMultiPoolSupport and len(sampleSets) != isMultiPoolSupport.count():
        libraryPrepProtocolsDisplayed = common_CV.objects.filter(categories__in=isMultiPoolSupport).values_list(
            "displayedValue", flat=True
        )
        msg = validation.format(
            ugettext_lazy("samplesets.messages.sampleset_plan_run.validationerrors"),
            {
                "sampleSetNames": ", ".join(
                    sampleSets.values_list("displayedName", flat=True)
                )
            },
        )
        return HttpResponseServerError("%s<br>Missing %s protocol in one of the Sampleset" % (msg, ", ".join(libraryPrepProtocolsDisplayed)))

    if isMultiPoolSupport:
        multiLibraryPoolPlanData = views_helper.processMultiPoolPlanSupport(sampleSets)
        if multiLibraryPoolPlanData:
            if multiLibraryPoolPlanData.get('errors', None):
                msg = validation.format(
                    ugettext_lazy("samplesets.messages.sampleset_plan_run.validationerrors"),
                    {
                        "sampleSetNames": ", ".join(
                            sampleSets.values_list("displayedName", flat=True)
                        )
                    },
                )  # "Cannot Plan Run from %s<br>" % ', '.join(sampleSets.values_list('displayedName', flat=True))
                return HttpResponseServerError("%s<br>%s" % (msg, "<br>".join(multiLibraryPoolPlanData['errors'])))

    # validate
    errors = sample_validator.validate_sampleSets_for_planning(sampleSets)
    if errors:
        msg = validation.format(
            ugettext_lazy("samplesets.messages.sampleset_plan_run.validationerrors"),
            {
                "sampleSetNames": ", ".join(
                    sampleSets.values_list("displayedName", flat=True)
                )
            },
        )  # "Cannot Plan Run from %s<br>" % ', '.join(sampleSets.values_list('displayedName', flat=True))
        return HttpResponseServerError("%s<br>%s" % (msg, "<br>".join(errors)))

    # multiple sample group types are allowed, with a warning
    sampleGroupTypes = (
        sampleSets.filter(SampleGroupType_CV__isnull=False)
        .values_list("SampleGroupType_CV__displayedName", flat=True)
        .distinct()
    )
    if len(sampleGroupTypes) > 1:
        warnings.append(
            validation.format(
                ugettext_lazy(
                    "samplesets.messages.sampleset_plan_run.warnings.SampleGroupType_CV.multiple"
                ),
                {"sampleGroupTypes": ", ".join(sampleGroupTypes)},
                include_warning_prefix=True,
            )
        )  # 'Warning: multiple Group Types selected: %s' % ', '.join(sampleGroupTypes))

    # categories to filter available Templates
    sampleSetCategories = []
    categories = sampleSets.exclude(categories="").values_list("categories", flat=True).distinct()
    if categories:
        for category in categories:
            sampleSetCategories.extend(category.split(";"))

    all_templates = _get_all_userTemplates(request)
    all_templates_params = list(
        all_templates.values(
            "pk", "planDisplayedName", "sampleGrouping__displayedName", "categories",
        )
    )

    # we want to display the system templates last
    all_systemTemplates = _get_all_systemTemplates(request)
    all_systemTemplates_params = list(
        all_systemTemplates.values(
            "pk", "planDisplayedName", "sampleGrouping__displayedName", "categories",
        )
    )

    ctxd = {
        "sampleSet_ids": ids,
        "sampleGroupTypes": sampleGroupTypes,
        "sampleSetCategories": sampleSetCategories,
        "template_params": all_templates_params + all_systemTemplates_params,
        "warnings": warnings,
        "multiLibraryPoolPlanData": multiLibraryPoolPlanData
    }
    logger.debug("show_plan_run Contenxt.dicts = %s", ctxd)
    ctx = RequestContext(request, ctxd)
    return render_to_response(
        "rundb/sample/modal_plan_run.html", context_instance=ctx, mimetype="text/html"
    )


def save_sampleset(request):
    """
    create or edit a new sample set (with no contents)
    """
    if request.method == "POST":
        queryDict = request.POST.dict()
        logger.debug("save_sampleset POST save_sampleset queryDict=%s" % queryDict)

        sampleSet_id = queryDict.get("id", None)
        try:
            isValid, errorMessage, _ = views_helper.create_or_update_sampleSet(queryDict, request.user, sampleSet_id)
            if errorMessage:
                return HttpResponse(
                    json.dumps([errorMessage], cls=LazyJSONEncoder),
                    mimetype="application/json",
                )
            else:
                return HttpResponse("true")
        except:
            logger.error(format_exc())
            message = i18n_errors.fatal_internalerror_during_save(
                _SampleSet.verbose_name
            )  # "Cannot save sample set to database. "
            if settings.DEBUG:
                message += format_exc()
            return HttpResponse(
                json.dumps([message], cls=LazyJSONEncoder),
                mimetype="application/json",
            )

    else:
        return HttpResponseRedirect("/sample/")


@transaction.commit_manually
def save_samplesetitem(request):
    """
    create or edit a new sample set item
    """

    def rollback_and_return_error(errorMessage):
        if not isinstance(errorMessage, list):
            errorMessage = [errorMessage]
        transaction.rollback()
        return HttpResponse(
            json.dumps(errorMessage, cls=LazyJSONEncoder), mimetype="application/json"
        )

    if request.method == "POST":
        queryDict = request.POST.dict()
        intent = queryDict.get("intent")
        logger.debug(
            "POST %s save_input_samples_for_sampleset queryDict=%s"
            % (intent, queryDict)
        )

        sampleSetItem_dict = views_helper.parse_sample_kwargs_from_dict(queryDict)

        # validate sampleSetItem parameters
        isValid, errorMessage = sample_validator.validate_sample_for_sampleSet(
            sampleSetItem_dict
        )
        if not isValid:
            return rollback_and_return_error(errorMessage)

        # next validate sampleSetItems as a group
        samplesetitem_id = queryDict.get("id")
        samplesetitems = None
        if samplesetitem_id:
            item = SampleSetItem.objects.get(pk=samplesetitem_id)
            samplesetitems = item.sampleSet.samples.all()
            item_id = samplesetitem_id
            sampleSet = item.sampleSet
        elif "pending_sampleSetItem_list" in request.session.get("input_samples", {}):
            samplesetitems = request.session["input_samples"][
                "pending_sampleSetItem_list"
            ]
            item_id = queryDict.get("pending_id")
            sampleSet = None

        if samplesetitems:
            # validate barcoding is consistent between multiple samples
            barcodeKit = queryDict.get("barcodeKit")
            barcode = queryDict.get("barcode")
            isValid, errorMessage = sample_validator.validate_barcoding_samplesetitems(
                samplesetitems, barcodeKit, barcode, item_id
            )
            if not isValid:
                return rollback_and_return_error(errorMessage)

            # validate PCR Plate position
            pcrPlateRow = queryDict.get('pcrPlateRow', "")
            isValid, errorMessage = sample_validator.validate_pcrPlate_position_samplesetitems(samplesetitems, pcrPlateRow, item_id, sampleSet)
            if not isValid:
                return rollback_and_return_error(errorMessage)

        try:
            if intent == "add":
                logger.info("views.save_samplesetitem - TODO!!! - unsupported for now")

            elif intent == "edit":
                sampleSetItem_id = queryDict.get("id")
                new_sample = views_helper._create_or_update_sample_for_sampleSetItem(
                    sampleSetItem_dict, request.user, sampleSetItem_id
                )

                # process custom sample attributes, if any
                isValid, errorMessage = views_helper._create_or_update_sampleAttributes_for_sampleSetItem(
                    request, request.user, new_sample
                )
                if not isValid:
                    return rollback_and_return_error(errorMessage)

                views_helper._create_or_update_sampleSetItem(
                    sampleSetItem_dict, request.user, sampleSetItem_id, None, new_sample
                )

            elif intent == "add_pending" or intent == "edit_pending":
                # process custom sample attributes, if any
                isValid, errorMessage, sampleAttributes_dict = views_helper._create_pending_sampleAttributes_for_sampleSetItem(
                    request
                )
                if errorMessage:
                    return isValid, errorMessage, sampleAttributes_dict

                sampleSetItem_pendingId = queryDict.get("pending_id")
                if not sampleSetItem_pendingId:
                    sampleSetItem_pendingId = views_helper._get_pending_sampleSetItem_id(
                        request
                    )

                # create sampleSetItem dict
                sampleSetItem_dict["pending_id"] = int(sampleSetItem_pendingId)
                sampleSetItem_dict["attribute_dict"] = sampleAttributes_dict

                isNew = intent == "add_pending"
                views_helper._update_input_samples_session_context(
                    request, sampleSetItem_dict, isNew
                )

            transaction.commit()
            return HttpResponse("true")
        except Exception:
            logger.error(format_exc())
            errorMessage = "Error saving sample"
            if settings.DEBUG:
                errorMessage += format_exc()
            return rollback_and_return_error(errorMessage)
    else:
        return HttpResponseRedirect("/sample/")


@transaction.commit_manually
def save_input_samples_for_sampleset(request):
    """
    create or update SampleSet with manually entered samples
    """

    def rollback_and_return_error(errorMessage):
        if not isinstance(errorMessage, list):
            errorMessage = [errorMessage]
        transaction.rollback()
        return HttpResponse(
            json.dumps(errorMessage, cls=LazyJSONEncoder), mimetype="application/json"
        )

    if request.method == "POST":
        queryDict = request.POST.dict()
        if (
            "input_samples" not in request.session
            and "edit_amp_sampleSet" not in queryDict
        ):
            errorMessage = (
                "No manually entered samples found to create a sample set."
            )  # TODO: i18n
            return rollback_and_return_error(errorMessage)

        sampleSet_ids = request.POST.getlist("sampleset", [])
        logger.debug(
            "POST save_input_samples_for_sampleset queryDict=%s, samplesets=%s"
            % (queryDict, sampleSet_ids)
        )

        if "pending_sampleSetItem_list" in request.session.get("input_samples", {}):
            pending_sampleSetItem_list = request.session["input_samples"][
                "pending_sampleSetItem_list"
            ]
        else:
            pending_sampleSetItem_list = []

        try:
            # edit or create new sample set, if any
            isValid, errorMessage, new_sampleSet_id = views_helper.create_or_update_sampleSet(
                queryDict, request.user, queryDict.get("id")
            )
            if not isValid:
                return rollback_and_return_error(errorMessage)

            if new_sampleSet_id and new_sampleSet_id not in sampleSet_ids:
                sampleSet_ids.append(new_sampleSet_id)

            # must select at least one sampleSet to process
            if not sampleSet_ids:
                transaction.rollback()
                return HttpResponse(
                    json.dumps(
                        [
                            validation.required_error(
                                ugettext(
                                    "samplesets.input_samples.save.fields.sampleset.label"
                                ),
                                include_error_prefix=True,
                            )
                        ],
                        cls=LazyJSONEncoder,
                    ),
                    mimetype="application/json",
                )  # "Error, Please select a sample set or add a new sample set first."

            # validate for Ampliseq HD on chef and assign PCR plate and tube position automatically
            isValid, errorMessage = sample_validator.validate_sampleset_items_limit(
                pending_sampleSetItem_list, sampleSet_ids
            )
            if not isValid:
                return rollback_and_return_error(errorMessage)

            # validate new and existing sample set items as a group
            isEdit_amp_sampleSet = "edit_amp_sampleSet" in queryDict
            isValid, errorMessage, categoryDict, parsedSamplesetitems = views_helper.validate_for_existing_samples(
                pending_sampleSetItem_list, sampleSet_ids, isEdit_amp_sampleSet
            )
            if not isValid:
                return rollback_and_return_error(errorMessage)

            if categoryDict:
                pending_sampleSetItem_list = views_helper.assign_tube_postion_pcr_plates(
                    categoryDict
                )
            """ TS-17723, TS-17910:Allow user to manually assign the PCR plate for Ampliseq on Chef and manual libPrep Type
            else:
                pending_sampleSetItem_list = views_helper.assign_pcr_plate_rows(
                    parsedSamplesetitems
                )
            """
            # create SampleSetItems from pending list
            for pending_sampleSetItem_dict in pending_sampleSetItem_list:
                new_sample = None
                if type(pending_sampleSetItem_dict) != types.DictType:  #
                    pending_sampleSetItem_dict = model_to_dict(
                        pending_sampleSetItem_dict
                    )

                new_sample = views_helper._create_or_update_sample(
                    pending_sampleSetItem_dict
                )
                sampleAttribute_dict = (
                    pending_sampleSetItem_dict.get("attribute_dict") or {}
                )
                isValid, errorMessage = views_helper._create_or_update_sampleAttributes_for_sampleSetItem_with_dict(
                    request, request.user, new_sample, sampleAttribute_dict
                )
                if not isValid:
                    return rollback_and_return_error(errorMessage)

                itemID = pending_sampleSetItem_dict.get("id", None)
                for sampleSet_id in sampleSet_ids:
                    views_helper._create_or_update_sampleSetItem(
                        pending_sampleSetItem_dict,
                        request.user,
                        itemID,
                        sampleSet_id,
                        new_sample,
                    )

            clear_samplesetitem_session(request)

            transaction.commit()
            return HttpResponse("true")
        except Exception:
            logger.exception(format_exc())

            transaction.rollback()
            # return HttpResponse(json.dumps({"status": "Error saving manually entered sample set info to database. " + format_exc()}), mimetype="text/html")

            errorMessage = ugettext_lazy(
                "samplesets.input_samples.save.error"
            )  # "Error saving manually entered sample set info to database. "
            if settings.DEBUG:
                errorMessage += format_exc()
            return rollback_and_return_error(errorMessage)
    else:
        return HttpResponseRedirect("/sample/")


def clear_input_samples_for_sampleset(request):
    clear_samplesetitem_session(request)
    return HttpResponseRedirect("/sample/samplesetitem/input/")


"""
 Get the saved samplesetitem for editing during sample set edit
 Sample set and items need to be validated for Ampliseq HD on chef 
"""


def get_persisted_input_samples_data(request, setID=None):
    sampleset = SampleSet.objects.get(pk=setID)
    samplesetitems = list(sampleset.samples.all())
    custom_sample_column_list = list(
        SampleAttribute.objects.filter(isActive=True).order_by("id")
    )
    items_dataDict = []

    for item in samplesetitems:
        itemDict = model_to_dict(item)
        itemDict["displayedName"] = item.sample.displayedName
        itemDict["externalId"] = item.sample.externalId
        itemDict["description"] = item.sample.description
        itemDict["pending_id"] = item.id

        dnabarcode = item.dnabarcode
        if dnabarcode:
            itemDict["barcode"] = dnabarcode.id_str
            itemDict["barcodeKit"] = dnabarcode.name

        # Get custom sample attribute for the persisted items
        for cutom_sample_attrbute in custom_sample_column_list:
            attributeName = cutom_sample_attrbute.displayedName
            attribute = SampleAttributeValue.objects.filter(
                sample=item.sample.id, sampleAttribute=cutom_sample_attrbute
            )
            if attribute:
                itemDict["attribute_dict"] = {attributeName: attribute[0].value}
            else:
                itemDict["attribute_dict"] = {attributeName: None}
        items_dataDict.append(itemDict)

    # provided option to add new item while editing sample set
    if "input_samples" in request.session:
        pending_sampleSetItem_list = request.session["input_samples"].get(
            "pending_sampleSetItem_list"
        )
        for pending_item in pending_sampleSetItem_list:
            items_dataDict.append(pending_item)

    data = {}
    data["meta"] = {}
    data["meta"]["total_count"] = views_helper._get_pending_sampleSetItem_count(
        request
    ) + len(samplesetitems)
    data["objects"] = items_dataDict

    json_data = json.dumps(data, cls=LazyDjangoJSONEncoder)

    logger.debug("views.get_persisted_input_samples_data json_data=%s" % (json_data))

    return HttpResponse(json_data, mimetype="application/json")


def get_input_samples_data(request):
    data = {}
    data["meta"] = {}
    data["meta"]["total_count"] = views_helper._get_pending_sampleSetItem_count(request)
    data["objects"] = request.session["input_samples"]["pending_sampleSetItem_list"]

    json_data = json.dumps(data, cls=LazyJSONEncoder)

    logger.debug("views.get_input_samples_data json_data=%s" % (json_data))

    return HttpResponse(json_data, mimetype="application/json")
    # return HttpResponse(json_data, mimetype="text/html")


@transaction.commit_manually
def save_import_samplesetitems(request):
    """
    save the imported samples from file for sample set creation
    """
    ERROR_MSG_SAMPLE_IMPORT_VALIDATION = ugettext_lazy(
        "import_samples.messages.failure"
    )  # "Import Samples validation failed. The samples have not been imported. Please correct the errors and try again or choose a different sample file to import. "

    def _fail(_status, _failed=None, isError=True, mimetype="application/json"):
        # helper method to clean up and return HttpResponse with error messages
        transaction.rollback()

        json_body = {"status": _status, "error": isError}
        if _failed:
            json_body["failed"] = _failed
        logger.info("views.save_import_samplesetitems() error=%s" % json_body)
        return HttpResponse(
            json.dumps(json_body, cls=LazyJSONEncoder), mimetype=mimetype
        )

    def _success(_status, _failed=None, mimetype="application/json"):
        transaction.commit()
        json_body = {"status": _status, "error": False}
        if _failed:
            json_body["failed"] = _failed
            json_body["error"] = True
        return HttpResponse(
            json.dumps(json_body, cls=LazyJSONEncoder), mimetype=mimetype
        )

    if request.method != "POST":
        logger.exception(format_exc())
        transaction.rollback()
        return _fail(status=i18n_errors.fatal_unsupported_http_method(request.method))

    postedfile = request.FILES["postedfile"]
    destination = tempfile.NamedTemporaryFile(delete=False)

    for chunk in postedfile.chunks():
        destination.write(chunk)
    postedfile.close()
    destination.close()

    non_ascii_files = []
    # open read bytes and detect if file contains non-ascii characters
    with open(destination.name, "rU") as _tmp:
        if not is_ascii(_tmp.read()):  # if not ascii
            non_ascii_files.append(postedfile.name)  # add uploaded file name

    if len(non_ascii_files) > 0:
        # "Only ASCII characters are supported. The following files contain non-ASCII characters: %(files)s."
        _error_msg = validation.format(
            ugettext("import_samples.messages.file_contains_non_ascii_characters"),
            {"files": SeparatedValuesBuilder().build(non_ascii_files)},
        )
        os.unlink(destination.name)
        return _fail(_status=_error_msg)

    # check to ensure it is not empty
    headerCheck = open(destination.name, "rU")
    firstCSV = []
    for firstRow in csv.reader(headerCheck):
        firstCSV.append(firstRow)
        # logger.info("views.save_import_samplesetitems() firstRow=%s;" %(firstRow))

    headerCheck.close()
    if not firstCSV:
        os.unlink(destination.name)
        return _fail(_status=validation.invalid_empty(postedfile))

    index = 0
    errorMsg = []
    samples = []
    rawSampleDataList = []
    sampleSetItemList = []
    failed = {}
    file = open(destination.name, "rU")
    csv_version_row = csv.reader(
        file
    ).next()  # skip the csv template version header and proceed
    reader = csv.DictReader(file)

    userName = request.user.username
    user = User.objects.get(username=userName)

    # Validate the sample CSV template version
    csv_version_header = import_sample_processor.get_sample_csv_version()[0]
    errorMsg, isToSkipRow, isToAbort = utils.validate_csv_template_version(
        headerName=csv_version_header,
        isSampleCSV=True,
        firstRow=csv_version_row,
        SampleCSVTemplateLabel=ugettext("import_samples.fields.file.label"),
    )

    if isToAbort:
        csv_version_index = 1
        failed[csv_version_index] = errorMsg
        return _fail(_status=ERROR_MSG_SAMPLE_IMPORT_VALIDATION, _failed=failed)

    try:
        startOffset = 2
        for index, row in enumerate(reader, start=startOffset):
            rowNumber = index + 1
            logger.debug(
                "LOOP views.save_import_samples_for_sampleset() validate_csv_sample...index=%d; row=%s"
                % (index, row)
            )
            errorMsg, isToSkipRow, isToAbort = import_sample_processor.validate_csv_sample(
                row, request
            )
            if errorMsg:
                if isToAbort:
                    failed[ugettext("import_samples.fields.file.label")] = errorMsg
                else:
                    failed[rowNumber] = errorMsg
            elif isToSkipRow:
                logger.debug(
                    "views.save_import_samples_for_sampleset() SKIPPED ROW index=%d; rowNumber=%d; row=%s"
                    % (index, rowNumber, row)
                )
                continue
            else:
                sampleSetItemList.append(
                    import_sample_processor.get_sampleSetItem_kwargs(row, user)
                )
                rawSampleDataList.append(row)

            if isToAbort:
                return _fail(_status=ERROR_MSG_SAMPLE_IMPORT_VALIDATION, _failed=failed)

        # now validate that all barcode kit are the same and that each combo of barcode kit and barcode id_str is unique
        errorMsg = import_sample_processor.validate_barcodes_are_unique(
            rawSampleDataList
        )
        if errorMsg:
            for k, v in list(errorMsg.items()):
                failed[k] = [v]

        queryDict = request.POST.dict()
        if "new_sampleSet_libraryPrepType" in queryDict:
            libraryPrepType = queryDict["new_sampleSet_libraryPrepType"]
        if "libraryPrepType" in queryDict:
            libraryPrepType = queryDict["libraryPrepType"]

        if "amps_hd_on_chef_v1" not in libraryPrepType:
            errorMsg = import_sample_processor.validate_pcrPlateRow_are_unique(rawSampleDataList)
            if errorMsg:
                for k, v in errorMsg.items():
                    failed[k] = [v]

        if StrictVersion(str(float(csv_version_row[1]))) < StrictVersion("2.0"):
            errorMsg = import_sample_processor.validate_pcrPlateRow_are_unique(
                rawSampleDataList
            )
            if errorMsg:
                for k, v in list(errorMsg.items()):
                    failed[k] = [v]
        logger.info(
            "views.save_import_samples_for_sampleset() len(rawSampleDataList)=%d"
            % (len(rawSampleDataList))
        )
    except:
        logger.exception(format_exc())

        message = i18n_errors.fatal_internalerror_during_processing(postedfile)
        if settings.DEBUG:
            message += format_exc()
        return _fail(_status=message)
    finally:
        if not file.closed:
            file.close()
        if not destination.closed:
            destination.close()  # now close and remove the temp file
        os.unlink(destination.name)  # remove the tempfile

    if not rawSampleDataList:
        failed[ugettext("import_samples.fields.file.label")] = [
            validation.invalid_required_at_least_one(_Sample.verbose_name)
        ]

    if failed:
        return _fail(_status=ERROR_MSG_SAMPLE_IMPORT_VALIDATION, _failed=failed)

    try:
        # validate new sampleSet entry before proceeding further
        queryDict = request.POST.dict()
        sampleSet_ids = request.POST.getlist("sampleset", [])

        isValid, errorMessage, new_sampleSet_id = views_helper.create_or_update_sampleSet(
            queryDict, user
        )
        if not isValid:
            failed[ugettext("import_samples.fields.sampleset.label")] = [errorMessage]
            return _fail(_status=ERROR_MSG_SAMPLE_IMPORT_VALIDATION, _failed=failed)

        if new_sampleSet_id:
            sampleSet_ids.append(new_sampleSet_id)

        errorMsg = import_sample_processor.validate_barcodes_for_existing_samples(
            rawSampleDataList, sampleSet_ids
        )
        if errorMsg:
            for k, v in list(errorMsg.items()):
                failed[k] = [v]

        errorMsg = import_sample_processor.validate_pcrPlateRow_for_existing_samples(rawSampleDataList, sampleSet_ids)
        if errorMsg:
            for k, v in errorMsg.items():
                failed[k] = [v]

        if StrictVersion(str(float(csv_version_row[1]))) < StrictVersion("2.0"):
            errorMsg = import_sample_processor.validate_pcrPlateRow_for_existing_samples(
                rawSampleDataList, sampleSet_ids
            )
            if errorMsg:
                for k, v in list(errorMsg.items()):
                    failed[k] = [v]
        if len(sampleSet_ids) == 0:
            failed[ugettext("import_samples.fields.sampleset.label")] = [
                validation.invalid_required_at_least_one(_SampleSet.verbose_name)
            ]
            return _fail(_status=ERROR_MSG_SAMPLE_IMPORT_VALIDATION, _failed=failed)

        for sampleSetId in sampleSet_ids:
            isValid, errorMessage = sample_validator.validate_sampleset_items_limit(
                sampleSetItemList, [sampleSetId]
            )
            if not isValid:
                failed[ugettext("import_samples.fields.sampleset.label")] = [
                    errorMessage
                ]
                return _fail(_status=ERROR_MSG_SAMPLE_IMPORT_VALIDATION, _failed=failed)
            sampleset = SampleSet.objects.get(pk=sampleSetId)
            samplesetitems = list(sampleset.samples.all())

            if sampleSetItemList:
                samplesetitems.extend(sampleSetItemList)
            isValid, errorMessage, categoryDict = sample_validator.validate_inconsistent_ampliseq_HD_category(
                samplesetitems, None, sampleset
            )
            if not isValid:
                failed[ugettext("import_samples.fields.sampleset.label")] = [
                    errorMessage
                ]
                return _fail(_status=ERROR_MSG_SAMPLE_IMPORT_VALIDATION, _failed=failed)
                # errors.extend([sampleset.displayedName + ": " + err for err in errorMessage])

            if categoryDict:
                sampleSetItemList = views_helper.assign_tube_postion_pcr_plates(
                    categoryDict
                )
            """
            CSV Import: TS-17723, TS-17910:Allow user to manually assign the PCR plate for Ampliseq on Chef and manual libPrep Type
            else:
                sampleSetItemList = views_helper.assign_pcr_plate_rows(samplesetitems)
            """
            if index > 0:
                index_process = index
                for sampleData in sampleSetItemList:
                    if type(sampleData) != types.DictType:  #
                        sampleData = model_to_dict(sampleData)
                    index_process += 1

                    logger.debug(
                        "LOOP views.save_import_samples_for_sampleset() process_csv_sampleSet...index_process=%d; sampleData=%s"
                        % (index_process, sampleData)
                    )
                    errorMsg, sample, sampleSetItem, isToSkipRow, ssi_sid, siv_sid = import_sample_processor.process_csv_sampleSet(
                        sampleData, request, user, sampleSetId
                    )

                    if errorMsg:
                        failed[index_process] = errorMsg
    except Exception:
        logger.exception(format_exc())
        message = i18n_errors.fatal_internalerror_during_save(
            ugettext_lazy("import_samples.title")
        )
        if settings.DEBUG:
            message += format_exc()
        return _fail(_status=message)

    if failed:
        return _fail(_status=ERROR_MSG_SAMPLE_IMPORT_VALIDATION, _failed=failed)

    return _success(
        _status=ugettext_lazy("import_samples.messages.success"), _failed=failed
    )  # "Import Samples completed successfully! The sample set will be listed on the sample set page.""


def show_input_samplesetitems(request):
    """
    show the page to allow user to input samples for sample set creation
    """
    ctx = views_helper._handle_enter_samples_manually_request(request)
    ctxd = get_sampleset_meta_data(request)
    ctx.update(ctxd)
    return render_to_response(
        "rundb/sample/input_samples.html", context_instance=ctx, mimetype="text/html"
    )


def show_samplesetitem_modal(
    request, intent, sampleSetItem=None, pending_sampleSetItem=None, isDetailEdit=False
):
    sample_groupType_CV_list = _get_sample_groupType_CV_list(request)
    sample_role_CV_list = SampleAnnotation_CV.objects.filter(
        isActive=True, annotationType="relationshipRole"
    ).order_by("value")
    controlType_CV_list = SampleAnnotation_CV.objects.filter(
        isActive=True, annotationType="controlType"
    ).order_by("value")
    gender_CV_list = SampleAnnotation_CV.objects.filter(
        isActive=True, annotationType="gender"
    ).order_by("value")
    cancerType_CV_list = SampleAnnotation_CV.objects.filter(
        isActive=True, annotationType="cancerType"
    ).order_by("value")
    population_CV_list = SampleAnnotation_CV.objects.filter(
        isActive=True, annotationType="population"
    ).order_by("value")
    sampleSource_CV_list = SampleAnnotation_CV.objects.filter(
        isActive=True, annotationType="sampleSource"
    ).order_by("value")
    panelPoolType_CV_list = SampleAnnotation_CV.objects.filter(
        isActive=True, annotationType="panelPoolType"
    ).order_by("value")
    mouseStrains_CV_list = SampleAnnotation_CV.objects.filter(
        isActive=True, annotationType="mouseStrains"
    ).order_by("value")
    sampleAttribute_list = SampleAttribute.objects.filter(isActive=True).order_by("id")
    sampleAttributeValue_list = []
    selectedBarcodeKit = None
    sampleGroupTypeName = ""
    pcrPlateRow_choices = SampleSetItem.ALLOWED_AMPLISEQ_PCR_PLATE_ROWS_V1

    if intent == "edit":
        selectedGroupType = sampleSetItem.sampleSet.SampleGroupType_CV
        if selectedGroupType:
            # if sample grouping is selected, try to limit to whatever relationship roles are compatible.  But if none is compatible, include all
            filtered_sample_role_CV_list = SampleAnnotation_CV.objects.filter(
                sampleGroupType_CV=selectedGroupType, annotationType="relationshipRole"
            ).order_by("value")
            if filtered_sample_role_CV_list:
                sample_role_CV_list = filtered_sample_role_CV_list

            sampleGroupTypeName = selectedGroupType.displayedName
            if (
                sampleSetItem.nucleotideType == "rna"
                and "Fusions" in selectedGroupType.displayedName
            ):
                sampleSetItem.nucleotideType = "fusions"

        sampleAttributeValue_list = SampleAttributeValue.objects.filter(
            sample_id=sampleSetItem.sample
        )
        selectedBarcodeKit = (
            sampleSetItem.dnabarcode.name if sampleSetItem.dnabarcode else None
        )

    available_dnaBarcodes = dnaBarcode.objects.filter(
        Q(active=True) | Q(name=selectedBarcodeKit)
    )
    barcodeKits = list(available_dnaBarcodes.values("name").distinct().order_by("name"))
    barcodeInfo = list(available_dnaBarcodes.order_by("name", "index"))
    nucleotideType_choices = views_helper._get_nucleotideType_choices(
        sampleGroupTypeName
    )
    isAmpliseqHD = False
    if sampleSetItem:
        isAmpliseqHD = "amps_hd_on_chef_v1" in sampleSetItem.sampleSet.libraryPrepType
        if isDetailEdit:
            isAmpliseqHD = False
    ctxd = {
        "sampleSetItem": sampleSetItem,
        "pending_sampleSetItem": pending_sampleSetItem,
        "sample_groupType_CV_list": sample_groupType_CV_list,
        "sample_role_CV_list": sample_role_CV_list,
        "controlType_CV_list": controlType_CV_list,
        "gender_CV_list": gender_CV_list,
        "cancerType_CV_list": cancerType_CV_list,
        "population_CV_list": population_CV_list,
        "mouseStrains_CV_list": mouseStrains_CV_list,
        "sampleSource_CV_list": sampleSource_CV_list,
        "panelPoolType_CV_list": panelPoolType_CV_list,
        "sampleAttribute_list": sampleAttribute_list,
        "sampleAttributeValue_list": sampleAttributeValue_list,
        "barcodeKits": barcodeKits,
        "barcodeInfo": barcodeInfo,
        "nucleotideType_choices": nucleotideType_choices,
        "pcrPlateRow_choices": pcrPlateRow_choices,
        "intent": intent,
        "isAmpliseqHD": isAmpliseqHD,
    }

    ctx = RequestContext(request, ctxd)
    return render_to_response(
        "rundb/sample/modal_add_samplesetitem.html",
        context_instance=ctx,
        mimetype="text/html",
    )


def show_add_pending_samplesetitem(request):
    """
    show the sample set input page
    """
    return show_samplesetitem_modal(request, "add_pending")


def input_samples_advanced_edit(request, setID=None):
    ctx = views_helper._handle_enter_samples_manually_request(request)
    ctxd = get_sampleset_meta_data(request)
    ctxd["edit_amp_sampleSet"] = get_object_or_404(SampleSet, pk=setID)
    ctx.update(ctxd)

    return render_to_response(
        "rundb/sample/input_samples.html", context_instance=ctx, mimetype="text/html"
    )


def show_edit_pending_samplesetitem(request, pending_sampleSetItemId):
    """
    show the sample set edit page
    """

    logger.debug(
        "views.show_edit_pending_samplesetitem pending_sampleSetItemId=%s; "
        % (str(pending_sampleSetItemId))
    )

    pending_sampleSetItem = views_helper._get_pending_sampleSetItem_by_id(
        request, pending_sampleSetItemId
    )

    if pending_sampleSetItem is None:
        ctxd = {
            "title": _("samplesets.samplesetitem.edit.title"),
            "errormsg": _("samplesets.samplesetitem.edit.pending.messages.invalidid")
            % {"pendingSampleSetItemId": pending_sampleSetItemId},
            "cancel": _("global.action.modal.cancel"),
        }
        ctx = RequestContext(request, ctxd)
        return render_to_response(
            "rundb/common/modal_error_message.html", context_instance=ctx
        )
    return show_samplesetitem_modal(
        request, "edit_pending", pending_sampleSetItem=pending_sampleSetItem
    )


def remove_pending_samplesetitem(request, _id, fromDetailPage=False):
    """
    remove the selected pending sampleSetItem from the session context
    """

    logger.debug("ENTER views.remove_pending_samplesetitem() id=%s" % (str(_id)))

    if "input_samples" in request.session:
        items = request.session["input_samples"]["pending_sampleSetItem_list"]
        index = 0
        for item in items:
            if item.get("pending_id", -99) == int(_id):
                # logger.debug("FOUND views.delete_pending_samplesetitem() id=%s; index=%d" %(str(_id), index))

                del request.session["input_samples"]["pending_sampleSetItem_list"][
                    index
                ]
                request.session.modified = True
                if fromDetailPage:
                    return True
                return HttpResponseRedirect("/sample/samplesetitem/input/")
            else:
                index += 1

    if fromDetailPage:
        return False

    return HttpResponseRedirect("/sample/samplesetitem/input/")


def get_sampleset_meta_data(request):
    """
    show the page to allow user to assign input samples to a sample set and trigger save
    """
    ctxd = {}
    sampleSet_list = _get_sampleSet_list(request)
    sampleGroupType_list = list(_get_sample_groupType_CV_list(request))
    libraryPrepType_choices = views_helper._get_libraryPrepType_choices(request)
    libraryPrepKits = KitInfo.objects.filter(kitType="LibraryPrepKit", isActive=True)
    libraryPrepProtocol_choices = views_helper._get_libraryPrepProtocol_choices(request)
    additionalCycles_choices = views_helper._get_additionalCycles_choices(request)
    ctxd = {
        "sampleSet_list": sampleSet_list,
        "sampleGroupType_list": sampleGroupType_list,
        "libraryPrepType_choices": libraryPrepType_choices,
        "libraryPrepKits": libraryPrepKits,
        "libraryPrepProtocol_choices": libraryPrepProtocol_choices,
        "additionalCycles_choices": additionalCycles_choices,
    }
    return ctxd


def show_save_input_samples_for_sampleset(request):
    """
    show the page to allow user to assign input samples to a sample set and trigger save
    """
    ctxd = {}
    sampleSet_list = _get_sampleSet_list(request)
    sampleGroupType_list = list(_get_sample_groupType_CV_list(request))
    libraryPrepType_choices = views_helper._get_libraryPrepType_choices(request)
    libraryPrepKits = KitInfo.objects.filter(kitType="LibraryPrepKit", isActive=True)

    ctxd = {
        "sampleSet_list": sampleSet_list,
        "sampleGroupType_list": sampleGroupType_list,
        "libraryPrepType_choices": libraryPrepType_choices,
        "libraryPrepKits": libraryPrepKits,
    }

    ctx = RequestContext(request, ctxd)
    return render_to_response(
        "rundb/sample/modal_save_samplesetitems.html",
        context_instance=ctx,
        mimetype="text/html",
    )


def show_add_sample_attribute(request):
    """
    show the page to add a custom sample attribute
    """
    ctxd = {}
    attr_type_list = SampleAttributeDataType.objects.filter(isActive=True).order_by(
        "dataType"
    )
    ctxd = {
        "sample_attribute": None,
        "attribute_type_list": attr_type_list,
        "intent": "add",
    }
    ctx = RequestContext(request, ctxd)

    return render_to_response(
        "rundb/sample/modal_add_sample_attribute.html",
        context_instance=ctx,
        mimetype="text/html",
    )


def show_edit_sample_attribute(request, _id):
    """
    show the page to edit a custom sample attribute
    """

    ctxd = {}
    sample_attribute = get_object_or_404(SampleAttribute, pk=_id)
    attr_type_list = SampleAttributeDataType.objects.filter(isActive=True).order_by(
        "dataType"
    )

    ctxd = {
        "sample_attribute": sample_attribute,
        "attribute_type_list": attr_type_list,
        "intent": "edit",
    }
    ctx = RequestContext(request, ctxd)
    return render_to_response(
        "rundb/sample/modal_add_sample_attribute.html",
        context_instance=ctx,
        mimetype="text/html",
    )


def toggle_visibility_sample_attribute(request, _id):
    """
    show or hide a custom sample attribute
    """
    ctxd = {}
    sample_attribute = get_object_or_404(SampleAttribute, pk=_id)

    toggle_showHide = False if (sample_attribute.isActive) else True
    sample_attribute.isActive = toggle_showHide

    userName = request.user.username
    user = User.objects.get(username=userName)
    currentDateTime = timezone.now()  ##datetime.datetime.now()
    sample_attribute.lastModifiedUser = user
    sample_attribute.lastModifiedDate = currentDateTime

    sample_attribute.save()

    return HttpResponseRedirect("/sample/sampleattribute/")


def delete_sample_attribute(request, _id):
    """
    delete the selected custom sample attribute
    """

    _type = "sampleattribute"

    sampleAttribute = get_object_or_404(SampleAttribute, pk=_id)

    sampleAttributeSamples_count = sampleAttribute.samples.count()
    isPlural = sampleAttributeSamples_count > 0

    _ids = [_id]
    actions = [
        reverse(
            "api_dispatch_detail",
            kwargs={"resource_name": _type, "api_name": "v1", "pk": int(_id)},
        )
    ]

    title = _("samplesets.sampleattribute.modal_confirm_delete.title.singular")
    confirmmsg = _(
        "samplesets.sampleattribute.modal_confirm_delete.messages.confirmmsg.singular"
    ) % {"sampleAttributeName": sampleAttribute.displayedName, "sampleAttributeId": _id}
    if isPlural:
        title = _("samplesets.sampleattribute.modal_confirm_delete.title.plural") % {
            "sampleAttributeSamplesCount": sampleAttributeSamples_count
        }
        confirmmsg = _(
            "samplesets.sampleattribute.modal_confirm_delete.messages.confirmmsg.plural"
        ) % {
            "sampleAttributeSamplesCount": sampleAttributeSamples_count,
            "sampleAttributeName": sampleAttribute.displayedName,
            "sampleAttributeId": u", ".join(map(str, _ids)),
        }

    ctx = RequestContext(
        request,
        {
            "ids": json.dumps(_ids),
            "method": "DELETE",
            "readonly": False,
            "action": actions[0],
            "actions": json.dumps(actions),
            "items": None,
            "isMultiple": len(_ids) > 1,
            "i18n": {
                "title": title,
                "confirmmsg": confirmmsg,
                "submit": _(
                    "samplesets.sampleattribute.modal_confirm_delete.action.submit"
                ),
                "cancel": _(
                    "samplesets.sampleattribute.modal_confirm_delete.action.cancel"
                ),
            },
        },
    )

    return render_to_response(
        "rundb/common/modal_confirm_delete.html", context_instance=ctx
    )


def delete_sampleset(request, _id):
    """
    delete the selected sample set
    """

    _type = "sampleset"

    sampleSet = get_object_or_404(SampleSet, pk=_id)

    sampleSetItems = SampleSetItem.objects.filter(sampleSet=sampleSet)

    sampleSetItems_count = sampleSetItems.count()

    isPlural = sampleSetItems_count > 0

    title = _("samplesets.modal_confirm_delete.title.singular")
    confirmmsg = _("samplesets.modal_confirm_delete.messages.confirmmsg.singular") % {
        "sampleSetId": _id,
        "sampleSetName": sampleSet.displayedName,
    }
    if isPlural:
        title = _("samplesets.modal_confirm_delete.title.plural") % {
            "sampleSetItemsCount": sampleSetItems_count
        }
        confirmmsg = _("samplesets.modal_confirm_delete.messages.confirmmsg.plural") % {
            "sampleSetItemsCount": sampleSetItems_count,
            "sampleSetName": sampleSet.displayedName,
            "sampleSetId": u", ".join(map(str, [_id])),
        }

    # Perform Validation checks
    plans = PlannedExperiment.objects.filter(sampleSets=sampleSet)
    if plans:
        # >> > sum([True, True, False, False, False, True])
        # 3
        planTemplatesCount = sum(
            [p.isReusable for p in plans]
        )  # number of Plan Template using Sample Set
        plannedRunsCount = len(plans) - planTemplatesCount
        validationmsg = _(
            "samplesets.messages.delete.usedbyplantemplatesorplannedruns"
        ) % {
            "planTemplatesCount": planTemplatesCount,
            "plannedRunsCount": plannedRunsCount,
            "sampleSetName": sampleSet.displayedName,
        }

        # return HttpResponse(json.dumps({"error": msg}, cls=LazyJSONEncoder), mimetype="text/html")

        ctxd = {
            "validationError": True,
            "i18n": {
                "title": title,
                "validationmsg": validationmsg,
                "cancel": _("samplesets.modal_confirm_delete.action.cancel"),
            },
        }
        ctx = RequestContext(request, ctxd)
        return render_to_response(
            "rundb/common/modal_confirm_delete.html", context_instance=ctx
        )

    actions = []
    actions.append(
        reverse(
            "api_dispatch_detail",
            kwargs={"resource_name": _type, "api_name": "v1", "pk": int(_id)},
        )
    )
    # need to delete placeholder samplePrepData if any
    instrumentData_pk = None
    if sampleSet.libraryPrepInstrumentData:
        instrumentData_pk = sampleSet.libraryPrepInstrumentData.pk
        instrumentData_resource = "sampleprepdata"
        actions.append(
            reverse(
                "api_dispatch_detail",
                kwargs={
                    "resource_name": instrumentData_resource,
                    "api_name": "v1",
                    "pk": int(instrumentData_pk),
                },
            )
        )

    _ids = [_id, int(instrumentData_pk)] if instrumentData_pk else [_id]

    isMultiple = len(_ids) > 1

    ctxd = {
        "ids": json.dumps(_ids),
        "method": "DELETE",
        "readonly": False,
        "action": actions[0],
        "actions": json.dumps(actions),
        "items": None,
        "isMultiple": isMultiple,
        "i18n": {
            "title": title,
            "confirmmsg": confirmmsg,
            "submit": _("samplesets.modal_confirm_delete.action.submit"),
            "cancel": _("samplesets.modal_confirm_delete.action.cancel"),
        },
    }
    ctx = RequestContext(request, ctxd)

    return render_to_response(
        "rundb/common/modal_confirm_delete.html", context_instance=ctx
    )


def remove_samplesetitem(request, _id):
    """
    remove the sample associated with the sample set
    """

    _type = "samplesetitem"
    try:
        if remove_pending_samplesetitem(request, _id, fromDetailPage=True):
            return HttpResponse("true")
    except Exception:
        logger.debug(
            "Going to remove the perssited sample set item. Confirm before deleting"
        )

    sampleSetItem = get_object_or_404(SampleSetItem, pk=_id)

    # if the sampleset has been run, do not allow sample to be removed
    sample = sampleSetItem.sample
    sampleSet = sampleSetItem.sampleSet

    logger.debug(
        "views.remove_samplesetitem - sampleSetItem.id=%s; name=%s; sampleSet.id=%s"
        % (str(_id), sample.displayedName, str(sampleSet.id))
    )

    title = _("samplesets.samplesetitem.modal_confirm_delete.title.singular") % {
        "sampleSetName": sampleSet.displayedName
    }

    # Perform Validation check
    plans = PlannedExperiment.objects.filter(sampleSets=sampleSet)
    if plans:
        planTemplatesCount = sum(
            [p.isReusable for p in plans]
        )  # number of Plan Template using Sample Set
        plannedRunsCount = len(plans) - planTemplatesCount
        validationmsg = _(
            "samplesets.samplesetitem.messages.delete.usedbyplantemplatesorplannedruns"
        ) % {
            "planTemplatesCount": planTemplatesCount,
            "plannedRunsCount": plannedRunsCount,
            "sampleSetName": sampleSet.displayedName,
            "sampleName": sample.displayedName,
        }

        ctxd = {
            "validationError": True,
            "i18n": {
                "title": title,
                "validationmsg": validationmsg,
                "cancel": _(
                    "samplesets.samplesetitem.modal_confirm_delete.action.cancel"
                ),
            },
        }
        ctx = RequestContext(request, ctxd)
        return render_to_response(
            "rundb/common/modal_confirm_delete.html", context_instance=ctx
        )
    else:
        # _typeDescription = "Sample from the sample set %(sampleSetName)s" % {'sampleSetName': sampleSet.displayedName}
        ctx = RequestContext(
            request,
            {
                "id": _id,
                "ids": json.dumps([_id]),
                "names": sample.displayedName,
                "method": "DELETE",
                "readonly": False,
                "action": reverse(
                    "api_dispatch_detail",
                    kwargs={"resource_name": _type, "api_name": "v1", "pk": int(_id)},
                ),
                "actions": json.dumps([]),
                "items": None,
                "isMultiple": False,
                "i18n": {
                    "title": title,
                    "confirmmsg": _(
                        "samplesets.samplesetitem.modal_confirm_delete.messages.confirmmsg"
                    )
                    % {
                        "sampleId": _id,
                        "sampleName": sample.displayedName,
                        "sampleSetName": sampleSet.displayedName,
                    },
                    "submit": _(
                        "samplesets.samplesetitem.modal_confirm_delete.action.submit"
                    ),
                    "cancel": _(
                        "samplesets.samplesetitem.modal_confirm_delete.action.cancel"
                    ),
                },
            },
        )

    return render_to_response(
        "rundb/common/modal_confirm_delete.html", context_instance=ctx
    )


def save_sample_attribute(request):
    """
    save sample attribute
    """

    if request.method == "POST":
        queryDict = request.POST

        logger.debug("views.save_sample_attribute POST queryDict=%s; " % (queryDict))

        intent = queryDict.get("intent", None)

        sampleAttribute_id = queryDict.get("id", None)
        attribute_type_id = queryDict.get("attributeType", None)
        if attribute_type_id == "0":
            attribute_type_id = None

        attribute_name = queryDict.get("sampleAttributeName", None)
        attribute_description = queryDict.get("attributeDescription", None)

        if not attribute_name or not attribute_name.strip():
            # return HttpResponse(json.dumps({"status": "Error: Attribute name is required"}, cls=LazyJSONEncoder), mimetype="text/html")
            return HttpResponse(
                json.dumps(
                    [
                        validation.required_error(
                            ugettext_lazy(
                                "samplesets.sampleattributes.fields.displayedName.label"
                            ),
                            include_error_prefix=True,
                        )
                    ],
                    cls=LazyJSONEncoder,
                ),
                mimetype="application/json",
            )

        try:
            sample_attribute_type = SampleAttributeDataType.objects.get(
                id=attribute_type_id
            )
        except Exception:
            # return HttpResponse(json.dumps({"status": "Error: Attribute type is required"}), mimetype="text/html")
            return HttpResponse(
                json.dumps(
                    [
                        validation.required_error(
                            ugettext_lazy(
                                "samplesets.sampleattributes.fields.dataType.dataType.label"
                            ),
                            include_error_prefix=True,
                        )
                    ],
                    cls=LazyJSONEncoder,
                ),
                mimetype="application/json",
            )

        is_mandatory = toBoolean(queryDict.get("is_mandatory", False))

        # TODO: validation (including checking the status again!!
        isValid = True
        if isValid:
            try:
                userName = request.user.username
                user = User.objects.get(username=userName)
                currentDateTime = timezone.now()  ##datetime.datetime.now()

                underscored_attribute_name = str(
                    attribute_name.strip().replace(" ", "_")
                ).lower()

                isValid, errorMessage = sample_validator.validate_sampleAttribute_definition(
                    underscored_attribute_name, attribute_description.strip()
                )
                if errorMessage:
                    # return HttpResponse(errorMessage, mimetype="text/html")
                    return HttpResponse(
                        json.dumps([errorMessage], cls=LazyJSONEncoder),
                        mimetype="application/json",
                    )

                if intent == "add":
                    new_sample_attribute = SampleAttribute(
                        displayedName=underscored_attribute_name,
                        dataType=sample_attribute_type,
                        description=attribute_description.strip(),
                        isMandatory=is_mandatory,
                        isActive=True,
                        creator=user,
                        creationDate=currentDateTime,
                        lastModifiedUser=user,
                        lastModifiedDate=currentDateTime,
                    )

                    new_sample_attribute.save()
                else:

                    orig_sampleAttribute = get_object_or_404(
                        SampleAttribute, pk=sampleAttribute_id
                    )

                    if (
                        orig_sampleAttribute.displayedName == underscored_attribute_name
                        and str(orig_sampleAttribute.dataType_id) == attribute_type_id
                        and orig_sampleAttribute.description
                        == attribute_description.strip()
                        and orig_sampleAttribute.isMandatory == is_mandatory
                    ):

                        logger.debug(
                            "views.save_sample_attribute() - NO UPDATE NEEDED!! sampleAttribute.id=%d"
                            % (orig_sampleAttribute.id)
                        )

                    else:
                        sampleAttribute_kwargs = {
                            "displayedName": underscored_attribute_name,
                            "description": attribute_description.strip(),
                            "dataType_id": attribute_type_id,
                            "isMandatory": is_mandatory,
                            "lastModifiedUser": user,
                            "lastModifiedDate": currentDateTime,
                        }
                        for field, value in sampleAttribute_kwargs.items():
                            setattr(orig_sampleAttribute, field, value)

                        orig_sampleAttribute.save()
                        logger.debug(
                            "views.save_sample_attribute - UPDATED sampleAttribute.id=%d"
                            % (orig_sampleAttribute.id)
                        )
            except Exception:
                logger.exception(format_exc())

                # return HttpResponse(json.dumps({"status": "Error: Cannot save user-defined sample attribute to database"}, cls=LazyJSONEncoder), mimetype="text/html")
                message = i18n_errors.fatal_internalerror_during_save(
                    _SampleAttribute.verbose_name
                )
                if settings.DEBUG:
                    message += format_exc()
                return HttpResponse(
                    json.dumps([message], cls=LazyJSONEncoder),
                    mimetype="application/json",
                )
            return HttpResponse("true")
        else:
            # errors = form._errors.setdefault(forms.forms.NON_FIELD_ERRORS, forms.util.ErrorList())
            # return HttpResponse(errors)

            # return HttpResponse(json.dumps({"status": "Could not save sample attribute due to validation errors"}, cls=LazyJSONEncoder), mimetype="text/html")
            return HttpResponse(
                json.dumps(
                    [
                        i18n_errors.validationerrors_cannot_save(
                            _SampleAttribute.verbose_name
                        )
                    ],
                    cls=LazyJSONEncoder,
                ),
                mimetype="application/json",
            )
    else:
        # return HttpResponse(json.dumps({"status": "Error: Request method to save user-defined sample attribute is invalid"}, cls=LazyJSONEncoder), mimetype="text/html")
        return HttpResponse(
            json.dumps(
                [i18n_errors.fatal_unsupported_http_method(request.method)],
                cls=LazyJSONEncoder,
            ),
            mimetype="application/json",
        )


def show_detailed_edit_sample_for_sampleset(request, sampleSetItemId):
    """
    show the sample edit page
    """
    logger.debug(
        "views.show_edit_sample_for_sampleset sampleSetItemId=%s; "
        % (str(sampleSetItemId))
    )
    try:
        sampleSetItem = SampleSetItem.objects.get(pk=sampleSetItemId)
        return show_samplesetitem_modal(
            request, "edit", sampleSetItem=sampleSetItem, isDetailEdit=True
        )
    except Exception:
        logger.debug(
            "Edit pending sampleset item views.show_edit_sample_for_sampleset sampleSetItemId=%s; "
            % (str(sampleSetItemId))
        )
        pending_samplesetItem = sampleSetItemId
        return show_edit_pending_samplesetitem(request, pending_samplesetItem)


def show_edit_sample_for_sampleset(request, sampleSetItemId):
    """
    show the sample edit page
    """
    logger.debug(
        "views.show_edit_sample_for_sampleset sampleSetItemId=%s; "
        % (str(sampleSetItemId))
    )
    sampleSetItem = get_object_or_404(SampleSetItem, pk=sampleSetItemId)
    return show_samplesetitem_modal(request, "edit", sampleSetItem=sampleSetItem)


class LibraryPrepDetailView(DetailView):
    model = SamplePrepData
    template_name = "rundb/sample/modal_libraryprep_detail.html"
    context_object_name = "data"


def library_prep_summary(request, pk):
    sampleSet = get_object_or_404(SampleSet, pk=pk)
    if sampleSet.libraryPrepInstrumentData:
        return LibraryPrepDetailView.as_view()(
            request, pk=sampleSet.libraryPrepInstrumentData.pk
        )
    else:
        return render_to_response(
            "rundb/sample/modal_libraryprep_detail.html",
            context_instance=RequestContext(request),
        )
