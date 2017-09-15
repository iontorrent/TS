# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from django import http

from django.contrib.auth.decorators import login_required
from django.template import RequestContext
from django.shortcuts import render_to_response, get_object_or_404, \
    get_list_or_404
from django.conf import settings
from django.db import transaction
from django.http import HttpResponse, HttpResponseRedirect

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
from iondb.utils import utils

import os
import string
import traceback
import tempfile
import csv

from django.core.exceptions import ValidationError

from iondb.rundb.models import Sample, SampleSet, SampleSetItem, SampleAttribute, SampleGroupType_CV,  \
    SampleAnnotation_CV, SampleAttributeDataType, SampleAttributeValue, PlannedExperiment, dnaBarcode, GlobalConfig, \
    SamplePrepData, KitInfo

# from iondb.rundb.api import SampleSetItemInfoResource

from django.contrib.auth.models import User
from django.conf import settings
from django.views.generic.detail import DetailView

import import_sample_processor
import views_helper
import sample_validator

from iondb.utils import toBoolean


logger = logging.getLogger(__name__)


ERROR_MSG_SAMPLE_IMPORT_VALIDATION = "Error: Sample set validation failed. "


def clear_samplesetitem_session(request):
    if request.session.get("input_samples", None):
        request.session['input_samples'].pop('pending_sampleSetItem_list', None)
        request.session.pop('input_samples', None)
    return HttpResponse("Manually Entered Sample session has been cleared")


def _get_sample_groupType_CV_list(request):
    sample_groupType_CV_list = None
    isSupported = GlobalConfig.get().enable_compendia_OCP

    if (isSupported):
        sample_groupType_CV_list = SampleGroupType_CV.objects.all().order_by("displayedName")
    else:
        sample_groupType_CV_list = SampleGroupType_CV.objects.all().exclude(displayedName="DNA_RNA").order_by("displayedName")

    return sample_groupType_CV_list


def _get_sampleSet_list(request):
    """
    Returns a list of sample sets to which we can still add samples
    """
    sampleSet_list = None
    isSupported = GlobalConfig.get().enable_compendia_OCP

    if (isSupported):
        sampleSet_list = SampleSet.objects.all().order_by("-lastModifiedDate", "displayedName")
    else:
        sampleSet_list = SampleSet.objects.all().exclude(SampleGroupType_CV__displayedName="DNA_RNA").order_by("-lastModifiedDate", "displayedName")

    # exclude sample sets that are of amps_on_chef_v1 AND already have 8 samples
    annotated_list = sampleSet_list.exclude(status__in=["voided", "libPrep_reserved"]).annotate(Count("samples"))
    exclude_id_list = annotated_list.values_list("id", flat=True).filter(libraryPrepType="amps_on_chef_v1", samples__count=8)
    available_sampleSet_list = annotated_list.exclude(pk__in=exclude_id_list)
    logger.debug("_get_sampleSet_list() sampleSet_list.count=%d; available_sampleSet_list=%d" % (annotated_list.count(), available_sampleSet_list.count()))

    return available_sampleSet_list


def _get_all_userTemplates(request):
    isSupported = GlobalConfig.get().enable_compendia_OCP

    all_templates = None
    if isSupported:
        all_templates = PlannedExperiment.objects.filter(isReusable=True,
                                                         isSystem=False).order_by('applicationGroup', 'sampleGrouping', '-date', 'planDisplayedName')

    else:
        all_templates = PlannedExperiment.objects.filter(isReusable=True,
                                                         isSystem=False).exclude(sampleGrouping__displayedName="DNA_RNA").order_by('applicationGroup', 'sampleGrouping', '-date', 'planDisplayedName')

    return all_templates


def _get_all_systemTemplates(request):
    isSupported = GlobalConfig.get().enable_compendia_OCP

    all_templates = None
    if isSupported:
        all_templates = PlannedExperiment.objects.filter(isReusable=True,
                                                         isSystem=True, isSystemDefault=False).order_by('applicationGroup', 'sampleGrouping', 'planDisplayedName')

    else:
        all_templates = PlannedExperiment.objects.filter(isReusable=True,
                                                         isSystem=True, isSystemDefault=False).exclude(sampleGrouping__displayedName="DNA_RNA").order_by('applicationGroup', 'sampleGrouping', 'planDisplayedName')

    return all_templates


@login_required
def show_samplesets(request):
    """
    show the sample sets home page
    """

    ctxd = {}

    # custom_sample_column_objs = SampleAttribute.objects.filter(isActive = True).order_by('id')
    custom_sample_column_list = list(SampleAttribute.objects.filter(isActive=True).values_list('displayedName', flat=True).order_by('id'))
    # custom_sample_column_list = SampleAttribute.objects.filter(isActive = True).order_by('id')

    sample_groupType_CV_list = _get_sample_groupType_CV_list(request)

    sample_role_CV_list = SampleAnnotation_CV.objects.filter(annotationType="relationshipRole").order_by('value')

    ctxd = {
        # 'custom_sample_column_objs' : custom_sample_column_objs,
        'custom_sample_column_list': simplejson.dumps(custom_sample_column_list),
        # 'custom_sample_column_list' : custom_sample_column_list,
        'sample_groupType_CV_list': sample_groupType_CV_list,
        'sample_role_CV_list': sample_role_CV_list
        }

    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/sample/samplesets.html", context_instance=ctx, mimetype="text/html")


@login_required
def show_sample_attributes(request):
    """
    show the user-defined sample attribute home page
    """

    ctxd = {}

    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/sample/sampleattributes.html", context_instance=ctx, mimetype="text/html")


@login_required
def download_samplefile_format(request):
    """
    download sample file format
    """

    response = http.HttpResponse(mimetype='text/csv')
    now = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    response['Content-Disposition'] = 'attachment; filename=sample_file_format_%s.csv' % now

    sample_csv_version = import_sample_processor.get_sample_csv_version()

    hdr = [import_sample_processor.COLUMN_SAMPLE_NAME, import_sample_processor.COLUMN_SAMPLE_EXT_ID, import_sample_processor.COLUMN_CONTROLTYPE,
           import_sample_processor.COLUMN_PCR_PLATE_POSITION, import_sample_processor.COLUMN_BARCODE_KIT, import_sample_processor.COLUMN_BARCODE,
           import_sample_processor.COLUMN_GENDER, import_sample_processor.COLUMN_GROUP_TYPE, import_sample_processor.COLUMN_GROUP,
           import_sample_processor.COLUMN_SAMPLE_DESCRIPTION, import_sample_processor.COLUMN_NUCLEOTIDE_TYPE, import_sample_processor.COLUMN_CANCER_TYPE,
           import_sample_processor.COLUMN_CELLULARITY_PCT, import_sample_processor.COLUMN_BIOPSY_DAYS, import_sample_processor.COLUMN_CELL_NUM,
           import_sample_processor.COLUMN_COUPLE_ID, import_sample_processor.COLUMN_EMBRYO_ID
           ]

    customAttributes = SampleAttribute.objects.all().exclude(isActive=False).order_by("displayedName")
    for customAttribute in customAttributes:
        hdr.append(customAttribute)

    writer = csv.writer(response)
    writer.writerow(sample_csv_version)
    writer.writerow(hdr)

    return response


@login_required
def show_import_samplesetitems(request):
    """
    show the page to import samples from file for sample set creation
    """
    ctxd = {}
    sampleSet_list = _get_sampleSet_list(request)
    sampleGroupType_list = list(_get_sample_groupType_CV_list(request))
    libraryPrepType_choices = views_helper._get_libraryPrepType_choices(request)
    libraryPrepKits = KitInfo.objects.filter(kitType='LibraryPrepKit', isActive=True)

    ctxd = {
        'sampleSet_list': sampleSet_list,
        'sampleGroupType_list': sampleGroupType_list,
        'libraryPrepType_choices': libraryPrepType_choices,
        'libraryPrepKits': libraryPrepKits,
        }

    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/sample/import_samples.html", context_instance=ctx, mimetype="text/html")


@login_required
def show_edit_sampleset(request, _id=None):
    """
    show the sample set add/edit page
    """
    ctxd = {}
    if _id:
        sampleSet = get_object_or_404(SampleSet, pk=_id)
        intent = "edit"
        editable = sampleSet.status in ['', 'created', 'libPrep_pending']
    else:
        sampleSet = None
        intent = "add"
        editable = True

    sampleGroupType_list = _get_sample_groupType_CV_list(request)
    libraryPrepType_choices = views_helper._get_libraryPrepType_choices(request)

    if editable:
        libraryPrepKits = KitInfo.objects.filter(kitType='LibraryPrepKit', isActive=True)
    else:
        libraryPrepKits = KitInfo.objects.filter(name=sampleSet.libraryPrepKitName)

    ctxd = {
        'sampleSet': sampleSet,
        'sampleGroupType_list': sampleGroupType_list,
        'libraryPrepType_choices': libraryPrepType_choices,
        'libraryPrepKits': libraryPrepKits,
        'intent': intent,
        'editable': editable
        }
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/sample/modal_add_sampleset.html", context_instance=ctx, mimetype="text/html")


@login_required
def show_plan_run(request, ids):
    """
    show the plan run popup
    """
    warnings = []
    sampleset_ids = ids.split(',')
    sampleSets = SampleSet.objects.filter(pk__in=sampleset_ids)
    if len(sampleSets) < 1:
        raise http.Http404("SampleSet not found")

    # validate
    errors = sample_validator.validate_sampleSets_for_planning(sampleSets)
    if errors:
        msg = "Cannot Plan Run from %s<br>" % ', '.join(sampleSets.values_list('displayedName', flat=True))
        return http.HttpResponseServerError(msg + '<br>'.join(errors))

    # multiple sample group types are allowed, with a warning
    sampleGroupTypes = sampleSets.filter(SampleGroupType_CV__isnull=False).values_list('SampleGroupType_CV__displayedName', flat=True).distinct()
    if len(sampleGroupTypes) > 1:
        warnings.append('Warning: multiple Group Types selected: %s' % ', '.join(sampleGroupTypes))

    all_templates = _get_all_userTemplates(request)
    all_templates_params = list(all_templates.values('pk', 'planDisplayedName', 'sampleGrouping__displayedName'))

    # we want to display the system templates last
    all_systemTemplates = _get_all_systemTemplates(request)
    all_systemTemplates_params = list(all_systemTemplates.values('pk', 'planDisplayedName', 'sampleGrouping__displayedName'))

    ctxd = {
        'sampleSet_ids': ids,
        'sampleGroupTypes': sampleGroupTypes,
        'template_params': all_templates_params + all_systemTemplates_params,
        'warnings': warnings,
        }

    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/sample/modal_plan_run.html", context_instance=ctx, mimetype="text/html")


@login_required
def save_sampleset(request):
    """
    create or edit a new sample set (with no contents)
    """

    if request.method == "POST":
        intent = request.POST.get('intent', None)

        # TODO: validation (including checking the status again!!
        queryDict = request.POST
        isValid, errorMessage = sample_validator.validate_sampleSet(queryDict);

        if errorMessage:
            # return HttpResponse(errorMessage, mimetype="text/html")
            return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

        if isValid:
            userName = request.user.username
            user = User.objects.get(username=userName)

            logger.debug("views.save_sampleset POST save_sampleset queryDict=%s" % (queryDict))

#            logger.debug("save_sampleset sampleSetName=%s" %(queryDict.get("sampleSetName", "")))
#            logger.debug("save_sampleset sampleSetDesc=%s" %(queryDict.get("sampleSetDescription", "")))
#            logger.debug("save_sampleset_sampleSet_groupType=%s" %(queryDict.get("groupType", None)))
#            logger.debug("save_sampleset id=%s" %(queryDict.get("id", None)))

            sampleSetName = queryDict.get("sampleSetName", "").strip()
            sampleSetDesc = queryDict.get("sampleSetDescription", "").strip()

            sampleSet_groupType_id = queryDict.get("groupType", None)
            if sampleSet_groupType_id == "0":
                sampleSet_groupType_id = None

            libraryPrepType = queryDict.get("libraryPrepType", "").strip()
            if libraryPrepType and "chef" in libraryPrepType.lower():
                libraryPrepInstrument = "chef"
            else:
                libraryPrepInstrument = ""

            libraryPrepKitName = queryDict.get("libraryPrepKit", "").strip()
            pcrPlateSerialNum = queryDict.get("pcrPlateSerialNum", "").strip()

            sampleSet_id = queryDict.get("id", None)
            currentDateTime = timezone.now()  #datetime.datetime.now()

            try:
                if intent == "add":
                    sampleSetStatus = "created"
                    if sampleSet.libraryPrepInstrument == "chef":
                        libraryPrepInstrumentData_obj = models.SamplePrepData.objects.create(samplePrepDataType="lib_prep")
                        sampleSetStatus = "libPrep_pending"
                    else:
                        libraryPrepInstrumentData_obj = None

                    sampleSet_kwargs = {
                        'displayedName': sampleSetName,
                        'description': sampleSetDesc,
                        'status': sampleSetStatus,
                        'SampleGroupType_CV_id': sampleSet_groupType_id,
                        'libraryPrepType': libraryPrepType,
                        'libraryPrepKitName': libraryPrepKitName,
                        'pcrPlateSerialNum': pcrPlateSerialNum,
                        'libraryPrepInstrument': libraryPrepInstrument,
                        'libraryPrepInstrumentData': libraryPrepInstrumentData_obj,
                        'creator': user,
                        'creationDate': currentDateTime,
                        'lastModifiedUser': user,
                        'lastModifiedDate': currentDateTime
                        }

                    sampleSet = SampleSet(**sampleSet_kwargs)
                    sampleSet.save()

                    logger.debug("views - save_sampleset - ADDED sampleSet.id=%d" % (sampleSet.id))
                else:
                    orig_sampleSet = get_object_or_404(SampleSet, pk=sampleSet_id)

                    if (orig_sampleSet.displayedName == sampleSetName and
                        orig_sampleSet.description == sampleSetDesc and
                        orig_sampleSet.SampleGroupType_CV and str(orig_sampleSet.SampleGroupType_CV.id) == sampleSet_groupType_id and
                        orig_sampleSet.libraryPrepType == libraryPrepType and
                        orig_sampleSet.libraryPrepKitName == libraryPrepKitName and
                            orig_sampleSet.pcrPlateSerialNum == pcrPlateSerialNum):

                        logger.debug("views.save_sampleset() - NO UPDATE NEEDED!! sampleSet.id=%d" % (orig_sampleSet.id))
                    else:
                        sampleSetStatus = orig_sampleSet.status

                        libraryPrepInstrumentData_obj = orig_sampleSet.libraryPrepInstrumentData
                        # clean up the associated object if the sample set used to be "amps_on_chef" and now is not
                        if libraryPrepInstrumentData_obj and not libraryPrepType:
                            logger.debug("views - GOING TO DELETE orig_sampleSet orig_sampleSet.libraryPrepInstrumentData.id=%d" % (orig_sampleSet.libraryPrepInstrumentData.id))
                            libraryPrepInstrumentData_obj.delete()
                            if sampleSetStatus == "libPrep_pending":
                                sampleSetStatus = "created"
                        elif libraryPrepType and not libraryPrepInstrumentData_obj:
                            libraryPrepInstrumentData_obj = SamplePrepData.objects.create(samplePrepDataType="lib_prep")
                            logger.debug("views - orig_sampleSet.id=%d; GOING TO ADD libraryPrepInstrumentData_obj.id=%d" % (orig_sampleSet.id, libraryPrepInstrumentData_obj.id))
                            if sampleSetStatus == "created":
                                sampleSetStatus = "libPrep_pending"

                        sampleSet_kwargs = {
                            'displayedName': sampleSetName,
                            'description': sampleSetDesc,
                            'SampleGroupType_CV_id': sampleSet_groupType_id,
                            'libraryPrepType': libraryPrepType,
                            'libraryPrepKitName': libraryPrepKitName,
                            'pcrPlateSerialNum': pcrPlateSerialNum,
                            'libraryPrepInstrument': libraryPrepInstrument,
                            'libraryPrepInstrumentData':  libraryPrepInstrumentData_obj,
                            'status': sampleSetStatus,
                            'lastModifiedUser': user,
                            'lastModifiedDate': currentDateTime
                            }
                        for field, value in sampleSet_kwargs.iteritems():
                            setattr(orig_sampleSet, field, value)

                        orig_sampleSet.save()
                        logger.debug("views.save_sampleset - UPDATED sampleSet.id=%d" % (orig_sampleSet.id))

                return HttpResponse("true")
            except:
                logger.exception(format_exc())

                # return HttpResponse(json.dumps({"status": "Error saving sample set info to database!"}), mimetype="text/html")
                message = "Cannot save sample set to database. "
                if settings.DEBUG:
                    message += format_exc()
                return HttpResponse(json.dumps([message]), mimetype="application/json")
        else:
            return HttpResponse(json.dumps(["Error, Cannot save sample set due to validation errors."]), mimetype="application/json")

    else:
        return HttpResponseRedirect("/sample/")


@login_required
@transaction.commit_manually
def save_samplesetitem(request):
    """
    create or edit a new sample set item
    """

    if request.method == "POST":
        intent = request.POST.get('intent', None)

        logger.debug("at views.save_samplesetitem() intent=%s" % (intent))
        # json_data = simplejson.loads(request.body)
        raw_data = request.body
        logger.debug('views.save_samplesetitem POST.body... body: "%s"' % raw_data)
        logger.debug('views.save_samplesetitem request.session: "%s"' % request.session)

        # TODO: validation (including checking the status)

        queryDict = request.POST

        isValid, errorMessage = sample_validator.validate_sample_for_sampleSet(queryDict)

        if errorMessage:
            # return HttpResponse(errorMessage, mimetype="text/html")

            transaction.rollback()
            return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

        logger.debug("views.save_samplesetitem() B4 validate_barcoding queryDict=%s" % (queryDict))

        isValid, errorMessage = sample_validator.validate_sample_pgx_attributes_for_sampleSet(queryDict)

        if errorMessage:
            transaction.rollback()
            return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

        try:
            isValid, errorMessage = sample_validator.validate_barcoding(request, queryDict)
        except:
            logger.exception(format_exc())
            transaction.rollback()
            return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

        if errorMessage:
            transaction.rollback()
            return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

        try:
            sampleSetItemDescription = queryDict.get("sampleDescription", "").strip()
            isValid, errorMessage = sample_validator.validate_sampleDescription(sampleSetItemDescription)
        except:
            logger.exception(format_exc())
            transaction.rollback()
            return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

        if errorMessage:
            transaction.rollback()
            return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

        try:
            isValid, errorMessage = sample_validator.validate_pcrPlate_position(request, queryDict)
        except:
            logger.exception(format_exc())
            transaction.rollback()
            return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

        if errorMessage:
            transaction.rollback()
            return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

        if isValid:
            userName = request.user.username
            user = User.objects.get(username=userName)
            currentDateTime = timezone.now()  ##datetime.datetime.now()

            logger.info("POST save_samplesetitem queryDict=%s" % (queryDict))

            sampleSetItem_id = queryDict.get("id", None)

            new_sample = None

            try:
                if intent == "add":
                    logger.info("views.save_samplesetitem - TODO!!! - unsupported for now")
                elif intent == "edit":
                    new_sample = views_helper._create_or_update_sample_for_sampleSetItem(request, user)

                    isValid, errorMessage = views_helper._create_or_update_sampleAttributes_for_sampleSetItem(request, user, new_sample)

                    if not isValid:
                        transaction.rollback()
                        # return HttpResponse(errorMessage,  mimetype="text/html")
                        return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

                    isValid, errorMessage = views_helper._create_or_update_sampleSetItem(request, user, new_sample)

                    if not isValid:
                        transaction.rollback()
                        # return HttpResponse(errorMessage,  mimetype="text/html")
                        return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

                elif intent == "add_pending":

                    isValid, errorMessage, pending_sampleSetItem = views_helper._create_pending_sampleSetItem_dict(request, userName, currentDateTime)

                    if errorMessage:
                        transaction.rollback()
                        # return HttpResponse(errorMessage, mimetype="text/html")
                        return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

                    views_helper._update_input_samples_session_context(request, pending_sampleSetItem)

                    transaction.commit()

                    return HttpResponse("true")

                elif intent == "edit_pending":
                    isValid, errorMessage, pending_sampleSetItem = views_helper._update_pending_sampleSetItem_dict(request, userName, currentDateTime)

                    if errorMessage:
                        transaction.rollback()
                        # return HttpResponse(errorMessage, mimetype="text/html")
                        return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

                    views_helper._update_input_samples_session_context(request, pending_sampleSetItem, False)

                    transaction.commit()

                    return HttpResponse("true")

                transaction.commit()
                return HttpResponse("true")
            except:
                logger.exception(format_exc())
                transaction.rollback()

                # return HttpResponse(json.dumps({"status": "Error saving sample set info to database!"}), mimetype="text/html")
                message = "Cannot save sample changes. "
                if settings.DEBUG:
                    message += format_exc()
                return HttpResponse(json.dumps([message]), mimetype="application/json")
        else:
            # errors = form._errors.setdefault(forms.forms.NON_FIELD_ERRORS, forms.util.ErrorList())
            # return HttpResponse(errors)
            logger.info("views.save_samplesetitem - INVALID - FAILED!!")
            transaction.rollback()

            # return HttpResponse(json.dumps({"status": "Could not save sample set due to validation errors"}), mimetype="text/html")
            return HttpResponse(json.dumps(["Cannot save sample changes due to validation errors."]), mimetype="application/json")
    else:
        return HttpResponseRedirect("/sample/")


@login_required
@transaction.commit_manually
def save_input_samples_for_sampleset(request):
    """
    create a new sample set item with manually entered samples
    """

    if request.method == "POST":
        # json_data = simplejson.loads(request.body)
        raw_data = request.body
        logger.debug('views.save_input_samples_for_sampleset POST.body... body: "%s"' % raw_data)
        logger.debug('views.save_input_samples_for_sampleset request.session: "%s"' % request.session)

        # TODO: validation
        # logic:
        # 1) if no input samples, nothing to do
        # 2) validate input samples
        # 3) validate input sample set
        # 4) validate sample does not exist inside the sample set yet
        # 5) if valid, get or create sample sets
        # 6) for each input sample,
        #    6.a) create or update sample
        #    6.b) create or update sample attributes
        #    6.c) create or update sample set item

        if "input_samples" not in request.session:
            transaction.rollback()
            return HttpResponse(json.dumps(["No manually entered samples found to create a sample set."]), mimetype="application/json")

        userName = request.user.username
        user = User.objects.get(username=userName)
        currentDateTime = timezone.now()  ##datetime.datetime.now()

        queryDict = request.POST
        logger.info("POST save_input_samples_for_sampleset queryDict=%s" % (queryDict))

            # 1) get or create sample sets
        try:
            # create sampleSets only if we have at least one good sample to process
            isValid, errorMessage, sampleSet_ids = views_helper._get_or_create_sampleSets(request, user)

            if not isValid:

                transaction.rollback()
                # return HttpResponse(errorMessage, mimetype="text/html")
                return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

            isValid, errorMessage = views_helper.validate_for_existing_samples(request, sampleSet_ids)

            if not isValid:

                transaction.rollback()
                # return HttpResponse(errorMessage, mimetype="text/html")
                return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

            if not sampleSet_ids:
                transaction.rollback()
                return HttpResponse(json.dumps(["Error, Please select a sample set or add a new sample set first."]), mimetype="application/json")

            # a list of dictionaries
            pending_sampleSetItem_list = request.session['input_samples']['pending_sampleSetItem_list']
            for pending_sampleSetItem in pending_sampleSetItem_list:

                sampleDisplayedName = pending_sampleSetItem.get("displayedName", "").strip()
                sampleExternalId = pending_sampleSetItem.get("externalId", "").strip()
                sampleDesc = pending_sampleSetItem.get("description", "").strip()
                sampleControlType = pending_sampleSetItem.get("controlType", "")

                sampleAttribute_dict = pending_sampleSetItem.get("attribute_dict") or {}

                sampleGender = pending_sampleSetItem .get("gender", "")
                sampleRelationshipRole = pending_sampleSetItem.get("relationshipRole", "")
                sampleRelationshipGroup = pending_sampleSetItem.get("relationshipGroup", "")

                sampleCancerType = pending_sampleSetItem.get("cancerType", "")

                sampleCellularityPct = pending_sampleSetItem.get("cellularityPct", None)
                if sampleCellularityPct == "":
                    sampleCellularityPct = None

                selectedBarcodeKit = pending_sampleSetItem.get("barcodeKit", "")
                selectedBarcode = pending_sampleSetItem.get("barcode", "")

                sampleNucleotideType = pending_sampleSetItem.get("nucleotideType", "")
                pcrPlateRow = pending_sampleSetItem.get("pcrPlateRow", "")

                sampleBiopsyDays = pending_sampleSetItem.get("biopsyDays", "0")
                if not sampleBiopsyDays:
                    sampleBiopsyDays = "0"
                sampleCellNum = pending_sampleSetItem.get("cellNum", "")
                sampleCoupleId = pending_sampleSetItem.get("coupleId", "")
                sampleEmbryoId = pending_sampleSetItem.get("embryoId", "")

                new_sample = views_helper._create_or_update_sample_for_sampleSetItem_with_values(request, user, sampleDisplayedName, sampleExternalId, sampleDesc, selectedBarcodeKit, selectedBarcode)

                isValid, errorMessage = views_helper._create_or_update_sampleAttributes_for_sampleSetItem_with_dict(request, user, new_sample, sampleAttribute_dict)
                if not isValid:
                    transaction.rollback()
                    # return HttpResponse(errorMessage, mimetype="text/html")
                    return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

                views_helper._create_or_update_pending_sampleSetItem(request, user, sampleSet_ids, new_sample, sampleGender, sampleRelationshipRole, sampleRelationshipGroup, sampleControlType,\
                        selectedBarcodeKit, selectedBarcode, sampleCancerType, sampleCellularityPct, sampleNucleotideType,
                        pcrPlateRow, sampleBiopsyDays, sampleCellNum, sampleCoupleId, sampleEmbryoId, sampleDesc)

            clear_samplesetitem_session(request)

            transaction.commit()
            return HttpResponse("true")
        except:
            logger.exception(format_exc())

            transaction.rollback()
            # return HttpResponse(json.dumps({"status": "Error saving manually entered sample set info to database. " + format_exc()}), mimetype="text/html")

            errorMessage = "Error saving manually entered sample set info to database. "
            if settings.DEBUG:
                errorMessage += format_exc()
            return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")
    else:
        return HttpResponseRedirect("/sample/")


@login_required
def clear_input_samples_for_sampleset(request):
    clear_samplesetitem_session(request)
    return HttpResponseRedirect("/sample/samplesetitem/input/")


@login_required
def get_input_samples_data(request):
    data = {}
    data["meta"] = {}
    data["meta"]["total_count"] = views_helper._get_pending_sampleSetItem_count(request)
    data["objects"] = request.session['input_samples']['pending_sampleSetItem_list']

    json_data = json.dumps(data)

    logger.debug("views.get_input_samples_data json_data=%s" % (json_data))

    return HttpResponse(json_data, mimetype='application/json')
    # return HttpResponse(json_data, mimetype="text/html")


@login_required
@transaction.commit_manually
def save_import_samplesetitems(request):
    """
    save the imported samples from file for sample set creation
    """
    logger.info(request)

    if request.method != 'POST':
        logger.exception(format_exc())
        transaction.rollback()
        return HttpResponse(json.dumps({"error": "Error, unsupported HTTP Request method (%s) for saving sample upload." % request.method}), mimetype="application/json")

    postedfile = request.FILES['postedfile']
    destination = tempfile.NamedTemporaryFile(delete=False)

    for chunk in postedfile.chunks():
        destination.write(chunk)
    postedfile.close()
    destination.close()

    # check to ensure it is not empty
    headerCheck = open(destination.name, "rU")
    firstCSV = []
    for firstRow in csv.reader(headerCheck):
        firstCSV.append(firstRow)
        # logger.info("views.save_import_samplesetitems() firstRow=%s;" %(firstRow))

    headerCheck.close()
    if not firstRow:
        os.unlink(destination.name)

        transaction.rollback()
        return HttpResponse(json.dumps({"status": "Error: sample file is empty"}), mimetype="text/html")

    index = 0
    row_count = 0
    errorMsg = []
    samples = []
    rawSampleDataList = []
    failed = {}
    file = open(destination.name, "rU")
    csv_version_row = csv.reader(file).next()  # skip the csv template version header and proceed
    reader = csv.DictReader(file)

    userName = request.user.username
    user = User.objects.get(username=userName)

    # Validate the sample CSV template version
    csv_version_header = import_sample_processor.get_sample_csv_version()[0]
    errorMsg, isToSkipRow, isToAbort = utils.validate_csv_template_version(headerName=csv_version_header, isSampleCSV=True, firstRow=csv_version_row)

    if isToAbort:
        csv_version_index = 1
        failed[csv_version_index] = errorMsg
        r = {"status": ERROR_MSG_SAMPLE_IMPORT_VALIDATION, "failed": failed}
        logger.info("views.save_import_samples_for_sampleset() failed=%s" % (r))
        transaction.rollback()
        return HttpResponse(json.dumps(r), mimetype="text/html")

    try:
        for index, row in enumerate(reader, start=2):
            logger.debug("LOOP views.save_import_samples_for_sampleset() validate_csv_sample...index=%d; row=%s" % (index, row))
            errorMsg, isToSkipRow, isToAbort = import_sample_processor.validate_csv_sample(row, request)
            if errorMsg:
                if isToAbort:
                    failed["File"] = errorMsg
                else:
                    failed[index] = errorMsg
            elif isToSkipRow:
                logger.debug("views.save_import_samples_for_sampleset() SKIPPED ROW index=%d; row=%s" % (index, row))
                continue
            else:
                rawSampleDataList.append(row)
                row_count += 1

            if isToAbort:
                r = {"status": ERROR_MSG_SAMPLE_IMPORT_VALIDATION, "failed": failed}
                logger.info("views.save_import_samples_for_sampleset() failed=%s" % (r))

                transaction.rollback()
                return HttpResponse(json.dumps(r), mimetype="text/html")

        # now validate that all barcode kit are the same and that each combo of barcode kit and barcode id_str is unique
        errorMsg = import_sample_processor.validate_barcodes_are_unique(rawSampleDataList)
        if errorMsg:
            for k, v in errorMsg.items():
                failed[k] = [v]
            # return HttpResponse(json.dumps({"status": ERROR_MSG_SAMPLE_IMPORT_VALIDATION, "failed" : {"Sample Set" : [errorMessage]}}), mimetype="text/html")

        errorMsg = import_sample_processor.validate_pcrPlateRow_are_unique(rawSampleDataList)
        if errorMsg:
            for k, v in errorMsg.items():
                failed[k] = [v]

        logger.info("views.save_import_samples_for_sampleset() row_count=%d" % (row_count))

        destination.close()  # now close and remove the temp file
        os.unlink(destination.name)
    except:
        logger.exception(format_exc())

        transaction.rollback()
        message = "Error saving sample set info to database. "
        if settings.DEBUG:
            message += format_exc()
        return HttpResponse(json.dumps({"status": message}), mimetype="text/html")

    if failed:
        r = {"status": ERROR_MSG_SAMPLE_IMPORT_VALIDATION, "failed": failed}
        logger.info("views.save_import_samples_for_sampleset() failed=%s" % (r))

        transaction.rollback()
        return HttpResponse(json.dumps(r), mimetype="text/html")

    if row_count > 0:
        try:
            # validate new sampleSet entry before proceeding further
            isValid, errorMessage, sampleSet_ids = views_helper._get_or_create_sampleSets(request, user)

            if not isValid:
                msgList = []
                msgList.append(errorMessage)
                failed["Sample Set"] = msgList

                r = {"status": ERROR_MSG_SAMPLE_IMPORT_VALIDATION, "failed": failed}

                logger.info("views.save_import_samples_for_sampleset() failed=%s" % (r))

                transaction.rollback()
                return HttpResponse(json.dumps(r), mimetype="text/html")

            errorMsg = import_sample_processor.validate_barcodes_for_existing_samples(rawSampleDataList, sampleSet_ids)
            if errorMsg:
                for k, v in errorMsg.items():
                    failed[k] = [v]

            errorMsg = import_sample_processor.validate_pcrPlateRow_for_existing_samples(rawSampleDataList, sampleSet_ids)
            if errorMsg:
                for k, v in errorMsg.items():
                    failed[k] = [v]

            if len(sampleSet_ids) == 0:
                msgList = []
                msgList.append("Error: There must be at least one valid sample set. Please select or input a sample set. ")

                failed["Sample Set"] = msgList

                r = {"status": ERROR_MSG_SAMPLE_IMPORT_VALIDATION, "failed": failed}
                transaction.rollback()
                return HttpResponse(json.dumps(r), mimetype="text/html")

            if (index > 0):
                index_process = index
                for sampleData in rawSampleDataList:
                    index_process += 1

                    logger.debug("LOOP views.save_import_samples_for_sampleset() process_csv_sampleSet...index_process=%d; sampleData=%s" % (index_process, sampleData))
                    errorMsg, sample, sampleSetItem, isToSkipRow, ssi_sid, siv_sid = import_sample_processor.process_csv_sampleSet(sampleData, request, user, sampleSet_ids)
                    if errorMsg:
                        failed[index_process] = errorMsg
        except:
            logger.exception(format_exc())

            transaction.rollback()

            message = "Error saving sample set info to database. "
            if settings.DEBUG:
                message += format_exc()

            return HttpResponse(json.dumps({"status": message}), mimetype="text/html")

        if failed:
            r = {"status": "Error: Sample set validation failed. The sample set info has not been saved.", "failed": failed}
            logger.info("views.save_import_samples_for_sampleset() failed=%s" % (r))

            transaction.rollback()
            return HttpResponse(json.dumps(r), mimetype="text/html")
        else:
            transaction.commit()

            r = {"status": "Samples Uploaded! The sample set will be listed on the sample set page.", "failed": failed}
            return HttpResponse(json.dumps(r), mimetype="text/html")

    else:
        logger.debug("EXITING views.save_import_samples_for_sampleset() row_count=%d" % (row_count))

        transaction.rollback()
        return HttpResponse(json.dumps({"status": "Error: There must be at least one valid sample. Please correct the errors or import another sample file."}), mimetype="text/html")


@login_required
def show_input_samplesetitems(request):
    """
    show the page to allow user to input samples for sample set creation
    """

    ctx = views_helper._handle_enter_samples_manually_request(request)
    return render_to_response("rundb/sample/input_samples.html", context_instance=ctx, mimetype="text/html")


def show_samplesetitem_modal(request, intent, sampleSetItem=None, pending_sampleSetItem=None):

    sample_groupType_CV_list = _get_sample_groupType_CV_list(request)
    sample_role_CV_list = SampleAnnotation_CV.objects.filter(isActive=True, annotationType="relationshipRole").order_by('value')
    controlType_CV_list = SampleAnnotation_CV.objects.filter(isActive=True, annotationType="controlType").order_by('value')
    gender_CV_list = SampleAnnotation_CV.objects.filter(isActive=True, annotationType="gender").order_by('value')
    cancerType_CV_list = SampleAnnotation_CV.objects.filter(isActive=True, annotationType="cancerType").order_by('value')
    sampleAttribute_list = SampleAttribute.objects.filter(isActive=True).order_by('id')
    sampleAttributeValue_list = []
    selectedBarcodeKit = None
    sampleGroupTypeName = ""

    pcrPlateRow_choices = views_helper._get_pcrPlateRow_choices(request)

    if intent == "edit":
        selectedGroupType = sampleSetItem.sampleSet.SampleGroupType_CV
        if selectedGroupType:
            # if sample grouping is selected, try to limit to whatever relationship roles are compatible.  But if none is compatible, include all
            filtered_sample_role_CV_list = SampleAnnotation_CV.objects.filter(sampleGroupType_CV=selectedGroupType, annotationType="relationshipRole").order_by('value')
            if filtered_sample_role_CV_list:
                sample_role_CV_list = filtered_sample_role_CV_list

            sampleGroupTypeName = selectedGroupType.displayedName
            if sampleSetItem.nucleotideType == "rna" and "Fusions" in selectedGroupType.displayedName:
                sampleSetItem.nucleotideType = "fusions"

        sampleAttributeValue_list = SampleAttributeValue.objects.filter(sample_id=sampleSetItem.sample)
        selectedBarcodeKit = sampleSetItem.dnabarcode.name if sampleSetItem.dnabarcode else None

    available_dnaBarcodes = dnaBarcode.objects.filter(Q(active=True) | Q(name=selectedBarcodeKit))
    barcodeKits = list(available_dnaBarcodes.values('name').distinct().order_by('name'))
    barcodeInfo = list(available_dnaBarcodes.order_by('name', 'index'))
    nucleotideType_choices = views_helper._get_nucleotideType_choices(sampleGroupTypeName)

    ctxd = {
        'sampleSetItem': sampleSetItem,
        'pending_sampleSetItem': pending_sampleSetItem,
        'sample_groupType_CV_list': sample_groupType_CV_list,
        'sample_role_CV_list': sample_role_CV_list,
        'controlType_CV_list': controlType_CV_list,
        'gender_CV_list': gender_CV_list,
        'cancerType_CV_list': cancerType_CV_list,
        'sampleAttribute_list': sampleAttribute_list,
        'sampleAttributeValue_list': sampleAttributeValue_list,
        'barcodeKits': barcodeKits,
        'barcodeInfo': barcodeInfo,
        'nucleotideType_choices': nucleotideType_choices,
        'pcrPlateRow_choices': pcrPlateRow_choices,
        'intent': intent
    }

    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/sample/modal_add_samplesetitem.html", context_instance=ctx, mimetype="text/html")


def show_add_pending_samplesetitem(request):
    """
    show the sample set input page
    """
    return show_samplesetitem_modal(request, "add_pending")


@login_required
def show_edit_pending_samplesetitem(request, pending_sampleSetItemId):
    """
    show the sample set edit page
    """

    logger.debug("views.show_edit_pending_samplesetitem pending_sampleSetItemId=%s; " % (str(pending_sampleSetItemId)))

    pending_sampleSetItem = views_helper._get_pending_sampleSetItem_by_id(request, pending_sampleSetItemId)
    if pending_sampleSetItem is None:
        msg = "Error, The selected sample is no longer available for this session."
        return HttpResponse(msg, mimetype="text/html")

    return show_samplesetitem_modal(request, "edit_pending", pending_sampleSetItem=pending_sampleSetItem)


@login_required
def remove_pending_samplesetitem(request, _id):
    """
    remove the selected pending sampleSetItem from the session context
    """

    logger.debug("ENTER views.remove_pending_samplesetitem() id=%s" % (str(_id)))

    if "input_samples" in request.session:
        items = request.session["input_samples"]["pending_sampleSetItem_list"]
        index = 0
        for item in items:
            if (item.get("pending_id", -99) == int(_id)):
                # logger.debug("FOUND views.delete_pending_samplesetitem() id=%s; index=%d" %(str(_id), index))

                del request.session["input_samples"]["pending_sampleSetItem_list"][index]
                request.session.modified = True
                return HttpResponseRedirect("/sample/samplesetitem/input/")
            else:
                index += 1

    logger.debug("views_helper._update_input_samples_session_context AFTER REMOVE session_contents=%s" % (request.session["input_samples"]))

    return HttpResponseRedirect("/sample/samplesetitem/input/")


@login_required
def show_save_input_samples_for_sampleset(request):
    """
    show the page to allow user to assign input samples to a sample set and trigger save
    """
    ctxd = {}
    sampleSet_list = _get_sampleSet_list(request)
    sampleGroupType_list = list(_get_sample_groupType_CV_list(request))
    libraryPrepType_choices = views_helper._get_libraryPrepType_choices(request)
    libraryPrepKits = KitInfo.objects.filter(kitType='LibraryPrepKit', isActive=True)

    ctxd = {
        'sampleSet_list': sampleSet_list,
        'sampleGroupType_list': sampleGroupType_list,
        'libraryPrepType_choices': libraryPrepType_choices,
        'libraryPrepKits': libraryPrepKits,
        }

    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/sample/modal_save_samplesetitems.html", context_instance=ctx, mimetype="text/html")


@login_required
def show_add_sample_attribute(request):
    """
    show the page to add a custom sample attribute
    """
    ctxd = {}
    attr_type_list = SampleAttributeDataType.objects.filter(isActive=True).order_by("dataType")
    ctxd = {
        'sample_attribute': None,
        'attribute_type_list': attr_type_list,
        'intent': "add"
        }
    ctx = RequestContext(request, ctxd)

    return render_to_response("rundb/sample/modal_add_sample_attribute.html", context_instance=ctx, mimetype="text/html")


@login_required
def show_edit_sample_attribute(request, _id):
    """
    show the page to edit a custom sample attribute
    """

    ctxd = {}
    sample_attribute = get_object_or_404(SampleAttribute, pk=_id)
    attr_type_list = SampleAttributeDataType.objects.filter(isActive=True).order_by("dataType")

    ctxd = {
        'sample_attribute': sample_attribute,
        'attribute_type_list': attr_type_list,
        'intent': "edit"
        }
    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/sample/modal_add_sample_attribute.html", context_instance=ctx, mimetype="text/html")


@login_required
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


@login_required
def delete_sample_attribute(request, _id):
    """
    delete the selected custom sample attribute
    """

    _type = 'sampleattribute'

    sampleAttribute = get_object_or_404(SampleAttribute, pk=_id)

    sampleValue_count = sampleAttribute.samples.count()
    _typeDescription = "Sample Attribute and " + str(sampleValue_count) + " related Sample Attribute Value(s)" if sampleValue_count > 0 else "Sample Attribute"
    ctx = RequestContext(request, {
        "id": _id,
        "ids": json.dumps([_id]),
        "names": sampleAttribute.displayedName,
        "method": "DELETE",
        'methodDescription': 'Delete',
        "readonly": False,
        'type': _typeDescription,
        'action': reverse('api_dispatch_detail', kwargs={'resource_name': _type, 'api_name': 'v1', 'pk': int(_id)}),
        'actions': json.dumps([])
    })

    return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)


@login_required
def delete_sampleset(request, _id):
    """
    delete the selected sample set
    """

    _type = 'sampleset'

    sampleSet = get_object_or_404(SampleSet, pk=_id)

    sampleSetItems = SampleSetItem.objects.filter(sampleSet=sampleSet)

    sample_count = sampleSetItems.count()

    plans = PlannedExperiment.objects.filter(sampleSets=sampleSet)

    if plans:
        planCount = plans.count()
        msg = "Error, There are %d plans for this sample set. Sample set %s cannot be deleted." % (planCount, sampleSet.displayedName)

        # return HttpResponse(json.dumps({"error": msg}), mimetype="text/html")
        return HttpResponse(msg, mimetype="text/html")
    else:
        actions = []
        actions.append(reverse('api_dispatch_detail', kwargs={'resource_name': _type, 'api_name': 'v1', 'pk': int(_id)}))
        # need to delete placeholder samplePrepData if any
        instrumentData_pk = None
        if sampleSet.libraryPrepInstrumentData:
            instrumentData_pk = sampleSet.libraryPrepInstrumentData.pk
            instrumentData_resource = "sampleprepdata"
            actions.append(reverse('api_dispatch_detail', kwargs={'resource_name': instrumentData_resource, 'api_name': 'v1', 'pk': int(instrumentData_pk)}))

        _typeDescription = "Sample Set and " + str(sample_count) + " related Sample Association(s)" if sample_count > 0 else "Sample Set"
        ctx = RequestContext(request, {
            "id": _id,
            "ids": json.dumps([_id, int(instrumentData_pk)]) if instrumentData_pk else json.dumps([_id]),
            "names": sampleSet.displayedName,
            "method": "DELETE",
            'methodDescription': 'Delete',
            "readonly": False,
            'type': _typeDescription,
            'action': actions[0],
            'actions': json.dumps(actions)
        })

        return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)


@login_required
def remove_samplesetitem(request, _id):
    """
    remove the sample associated with the sample set
    """

    _type = 'samplesetitem'

    sampleSetItem = get_object_or_404(SampleSetItem, pk=_id)

    # if the sampleset has been run, do not allow sample to be removed
    sample = sampleSetItem.sample
    sampleSet = sampleSetItem.sampleSet

    logger.debug("views.remove_samplesetitem - sampleSetItem.id=%s; name=%s; sampleSet.id=%s" % (str(_id), sample.displayedName, str(sampleSet.id)))

    plans = PlannedExperiment.objects.filter(sampleSets=sampleSet)

    if plans:
        planCount = plans.count()
        msg = "Error, There are %d plans for this sample set. Sample: %s cannot be removed from the sample set." % (planCount, sample.displayedName)

        # return HttpResponse(json.dumps({"error": msg}), mimetype="text/html")
        return HttpResponse(msg, mimetype="text/html")
    else:

        _typeDescription = "Sample from the sample set %s" % (sampleSet.displayedName)
        ctx = RequestContext(request, {
            "id": _id,
            "ids": json.dumps([_id]),
            "names": sample.displayedName,
            "method": "DELETE",
            'methodDescription': 'Remove',
            "readonly": False,
            'type': _typeDescription,
            'action': reverse('api_dispatch_detail', kwargs={'resource_name': _type, 'api_name': 'v1', 'pk': int(_id)}),
            'actions': json.dumps([])
        })

    return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)


@login_required
def save_sample_attribute(request):
    """
    save sample attribute
    """

    if request.method == 'POST':
        queryDict = request.POST

        logger.debug("views.save_sample_attribute POST queryDict=%s; " % (queryDict))

        intent = queryDict.get('intent', None)

        sampleAttribute_id = queryDict.get("id", None)
        attribute_type_id = queryDict.get('attributeType', None)
        if attribute_type_id == "0":
            attribute_type_id = None

        attribute_name = queryDict.get('sampleAttributeName', None)
        attribute_description = queryDict.get('attributeDescription', None)

        if not attribute_name or not attribute_name.strip():
            # return HttpResponse(json.dumps({"status": "Error: Attribute name is required"}), mimetype="text/html")
            return HttpResponse(json.dumps(["Error, Attribute name is required."]), mimetype="application/json")

        try:
            sample_attribute_type = SampleAttributeDataType.objects.get(id=attribute_type_id)
        except:
            # return HttpResponse(json.dumps({"status": "Error: Attribute type is required"}), mimetype="text/html")
            return HttpResponse(json.dumps(["Error, Attribute type is required."]), mimetype="application/json")

        is_mandatory = toBoolean(queryDict.get('is_mandatory', False))

         # TODO: validation (including checking the status again!!
        isValid = True
        if isValid:
            try:
                userName = request.user.username
                user = User.objects.get(username=userName)
                currentDateTime = timezone.now()  ##datetime.datetime.now()

                underscored_attribute_name = str(attribute_name.strip().replace(' ', '_')).lower()

                isValid, errorMessage = sample_validator.validate_sampleAttribute_definition(underscored_attribute_name, attribute_description.strip())
                if errorMessage:
                    # return HttpResponse(errorMessage, mimetype="text/html")
                    return HttpResponse(json.dumps([errorMessage]), mimetype="application/json")

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
                        lastModifiedDate=currentDateTime
                    )

                    new_sample_attribute.save()
                else:

                    orig_sampleAttribute = get_object_or_404(SampleAttribute, pk=sampleAttribute_id)

                    if (orig_sampleAttribute.displayedName == underscored_attribute_name and
                        str(orig_sampleAttribute.dataType_id) == attribute_type_id and
                        orig_sampleAttribute.description == attribute_description.strip() and
                            orig_sampleAttribute.isMandatory == is_mandatory):

                        logger.debug("views.save_sample_attribute() - NO UPDATE NEEDED!! sampleAttribute.id=%d" % (orig_sampleAttribute.id))

                    else:
                        sampleAttribute_kwargs = {
                            'displayedName': underscored_attribute_name,
                            'description': attribute_description.strip(),
                            'dataType_id': attribute_type_id,
                            'isMandatory': is_mandatory,
                            'lastModifiedUser': user,
                            'lastModifiedDate': currentDateTime
                            }
                        for field, value in sampleAttribute_kwargs.iteritems():
                            setattr(orig_sampleAttribute, field, value)

                        orig_sampleAttribute.save()
                        logger.debug("views.save_sample_attribute - UPDATED sampleAttribute.id=%d" % (orig_sampleAttribute.id))
            except:
                logger.exception(format_exc())

                # return HttpResponse(json.dumps({"status": "Error: Cannot save user-defined sample attribute to database"}), mimetype="text/html")
                message = "Cannot save sample attribute to database. "
                if settings.DEBUG:
                    message += format_exc()
                return HttpResponse(json.dumps([message]), mimetype="application/json")
            return HttpResponse("true")
        else:
            # errors = form._errors.setdefault(forms.forms.NON_FIELD_ERRORS, forms.util.ErrorList())
            # return HttpResponse(errors)

            # return HttpResponse(json.dumps({"status": "Could not save sample attribute due to validation errors"}), mimetype="text/html")
            return HttpResponse(json.dumps(["Cannot save sample attribute due to validation errors."]), mimetype="application/json")
    else:
        # return HttpResponse(json.dumps({"status": "Error: Request method to save user-defined sample attribute is invalid"}), mimetype="text/html")
        return HttpResponse(json.dumps(["Request method for save_sample_attribute is invalid."]), mimetype="application/json")


@login_required
def show_edit_sample_for_sampleset(request, sampleSetItemId):
    """
    show the sample edit page
    """
    logger.debug("views.show_edit_sample_for_sampleset sampleSetItemId=%s; " % (str(sampleSetItemId)))
    sampleSetItem = get_object_or_404(SampleSetItem, pk=sampleSetItemId)

    return show_samplesetitem_modal(request, "edit", sampleSetItem=sampleSetItem)


class LibraryPrepDetailView(DetailView):
    model = SamplePrepData
    template_name = 'rundb/sample/modal_libraryprep_detail.html'
    context_object_name = 'data'


def library_prep_summary(request, pk):
    sampleSet = get_object_or_404(SampleSet, pk=pk)
    if sampleSet.libraryPrepInstrumentData:
        return LibraryPrepDetailView.as_view()(request, pk=sampleSet.libraryPrepInstrumentData.pk)
    else:
        return render_to_response("rundb/sample/modal_libraryprep_detail.html")
