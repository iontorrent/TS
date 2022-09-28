# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django import http
from datetime import datetime
from django.contrib.auth.decorators import login_required
from django.template import RequestContext
from django.shortcuts import (
    render_to_response,
    get_object_or_404,
    get_list_or_404,
    render,
)
from django.conf import settings
from django.db import transaction
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.utils import translation
from django.views.generic.detail import DetailView
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext as _, ugettext_lazy, ugettext
from iondb.utils.validation import SeparatedValuesBuilder

from iondb.rundb.json_lazy import LazyJSONEncoder, LazyDjangoJSONEncoder
from iondb.rundb.api import PlannedExperimentResource, SDKValidationError
from iondb.rundb.models import (
    PlannedExperiment,
    RunType,
    ApplProduct,
    IonMeshNode,
    ReferenceGenome,
    Content,
    KitInfo,
    dnaBarcode,
    PlanSession,
    LibraryKey,
    ThreePrimeadapter,
    Chip,
    QCType,
    Project,
    Plugin,
    PlannedExperimentQC,
    Sample,
    GlobalConfig,
    Message,
    Experiment,
    Results,
    EventLog,
    common_CV,
    SampleSet)
from django.db.models import Q
from distutils.version import StrictVersion, LooseVersion

from traceback import format_exc
import json
import simplejson
import uuid

import logging
from django.core.serializers.json import DjangoJSONEncoder

from django.core.urlresolvers import reverse

from iondb.rundb.plan.views_helper import (
    get_projects,
    dict_bed_hotspot,
    isOCP_enabled,
    is_operation_supported,
    getChipDisplayedNamePrimaryPrefix,
    getChipDisplayedNameSecondaryPrefix,
    getChipDisplayedVersion,
    get_template_categories,
)

from iondb.utils.utils import convert
from iondb.utils.prepopulated_planning import apply_prepopulated_values_to_step_helper

from iondb.rundb.plan.plan_csv_writer import (
    get_template_data_for_batch_planning,
    get_plan_csv_version,
    get_samples_data_for_batch_planning,
)
from iondb.rundb.plan.plan_csv_writer import (
    PlanCSVcolumns,
    get_template_data_for_export,
    export_template_keys,
)
from iondb.rundb.plan.plan_csv_validator import (
    validate_csv_plan,
    get_bedFile_for_reference,
    SAMPLE_CSV_FILE_NAME_,
    BARCODED_SAMPLES_VALIDATION_ERRORS_,
    IRU_VALIDATION_ERRORS_,
)
from iondb.rundb.plan import plan_validator
from iondb.utils import utils, toBoolean, validation
from iondb.utils.TaskLock import TaskLock

import os
import traceback
import tempfile
import csv
import io
import collections
import zipfile

from iondb.rundb.plan.page_plan.step_helper import StepHelper
from iondb.rundb.plan.page_plan.step_helper_types import StepHelperType
from iondb.rundb.plan.page_plan.step_helper_db_loader import StepHelperDbLoader
from iondb.rundb.plan.page_plan.step_helper_db_saver import StepHelperDbSaver
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.labels import ScientificApplication, ModelsQcTypeToLabelsQcType
from iondb.utils.unicode_csv import is_ascii

logger = logging.getLogger(__name__)

MAX_LENGTH_PLAN_NAME = 512
MAX_LENGTH_SAMPLE_NAME = 127
MAX_LENGTH_PROJECT_NAME = 64
MAX_LENGTH_NOTES = 1024
MAX_LENGTH_SAMPLE_TUBE_LABEL = 512


def get_ir_config(request):
    if request.is_ajax():
        try:
            iru_plugin = Plugin.objects.get(
                name__iexact="IonReporterUploader", active=True
            )
            account_id = request.POST.get("accountId")
            userconfigs = iru_plugin.config.get("userconfigs").get(
                request.user.username
            )
            if userconfigs:
                for config in userconfigs:
                    if config["id"] == account_id:
                        return HttpResponse(
                            simplejson.dumps(config, cls=LazyJSONEncoder),
                            content_type="application/javascript",
                        )
            else:
                return HttpResponse(
                    simplejson.dumps(
                        dict(error="Unable to find userconfigs in IRU plugin"),
                        cls=LazyJSONEncoder,
                    ),
                    content_type="application/javascript",
                )
        except Exception:
            return HttpResponse(
                simplejson.dumps(
                    dict(error="Could not find IRU Plugin"), cls=LazyJSONEncoder
                ),
                content_type="application/javascript",
            )
    return HttpResponse(
        simplejson.dumps(dict(error="Invalid view function call"), cls=LazyJSONEncoder),
        content_type="application/javascript",
    )


def reset_page_plan_session(request):
    request.session.pop("plan_step_helper", None)
    return HttpResponse("Page Plan Session Object has been reset")


def plan_templates(request):
    """
    plan template home page
    """
    s5_chips_plan_json_upload = (
        Chip.objects.filter(
            isActive=True,
            name__in=["510", "520", "530", "540", "550"],
            instrumentType="S5",
        )
        .values_list("name", flat=True)
        .order_by("name")
    )
    usernames = PlannedExperiment.objects.filter(
        isReusable=True, username__isnull=False
    ).values_list("username", flat=True)
    instruments = Chip.getInstrumentTypesForChips(include_undefined=False)
    ctxd = {
        "categories": get_template_categories(),
        "s5_chips": s5_chips_plan_json_upload,
        "instruments": instruments,
        "instruments_dict": dict(instruments),  # for easier use in template
        "projects": Project.objects.order_by("name").values_list("name", flat=True),
        "barcodes": dnaBarcode.objects.filter(active=True)
        .values_list("name", flat=True)
        .distinct()
        .order_by("name"),
        "references": ReferenceGenome.objects.filter(enabled=True)
        .values_list("short_name", flat=True)
        .order_by("short_name"),
        "usernames": [user for user in set(usernames) if user.strip()],
        "rowErrorsFormat": ugettext_lazy(
            "global.messages.validation.format.row.errors"
        ),  # "'<strong>Row %(n)s</strong> contained errors:'
        "columnErrorsFormat": ugettext_lazy(
            "global.messages.validation.format.column.errors"
        ),  # '<strong>%(columnName)s</strong> column : %(columnErrors)s'
        "fieldErrorsFormat": ugettext_lazy(
            "global.messages.validation.format.field.errors"
        ),  # '%(fieldName)s contained errors:'s
        "rowWarningsFormat": ugettext_lazy(
            "global.messages.validation.format.row.warnings"
        ),  # "'<strong>Row %(n)s</strong> contained warnings:'
        "columnWarningsFormat": ugettext_lazy(
            "global.messages.validation.format.column.warnings"
        ),  # '<strong>%(columnName)s</strong> column: %(columnErrors)s'
        "fieldWarningsFormat": ugettext_lazy(
            "global.messages.validation.format.field.warnings"
        ),  # '%(fieldName)s contained warnings: '
    }
    return render_to_response(
        "rundb/plan/plan_templates.html", context_instance=RequestContext(request, ctxd)
    )


def page_plan_edit_template(request, template_id):
    """ edit a template with id template_id """

    isSupported = is_operation_supported(template_id)
    if not isSupported:
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForTemplatePlannedExperiment(
        template_id, StepHelperType.EDIT_TEMPLATE
    )
    ctxd = handle_step_request(request, StepNames.SAVE_TEMPLATE, step_helper)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_copy_template(request, template_id):
    """ copy a template with id template_id """
    isSupported = is_operation_supported(template_id)
    if not isSupported:
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForTemplatePlannedExperiment(
        template_id, StepHelperType.COPY_TEMPLATE
    )
    ctxd = handle_step_request(request, StepNames.SAVE_TEMPLATE, step_helper)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_new_template(request, code=None):
    """ Create a new template from runtype code or from scratch """
    if code:
        #        if (code == "8"):
        #            isSupported = isOCP_enabled()
        #            if (not isSupported):
        #                return render_to_response("501.html")

        if code == "9":
            step_helper = StepHelperDbLoader().getStepHelperForRunType(
                run_type_id=_get_runtype_from_code("1").pk, applicationGroupName="PGx"
            )
        elif code == "11":
            step_helper = StepHelperDbLoader().getStepHelperForRunType(
                run_type_id=_get_runtype_from_code("1").pk, applicationGroupName="HID"
            )
        elif code == "12":
            step_helper = StepHelperDbLoader().getStepHelperForRunType(
                run_type_id=_get_runtype_from_code("5").pk,
                applicationGroupName="immune_repertoire",
            )
        elif code == "13":
            step_helper = StepHelperDbLoader().getStepHelperForRunType(
                run_type_id=_get_runtype_from_code("1").pk,
                applicationGroupName="mutation_load",
            )
        else:
            step_helper = StepHelperDbLoader().getStepHelperForRunType(
                _get_runtype_from_code(code).pk
            )
    else:
        step_helper = StepHelper()
    if settings.FEATURE_FLAGS.IONREPORTERUPLOADER:
        ctxd = handle_step_request(request, StepNames.IONREPORTER, step_helper)
    else:
        ctxd = handle_step_request(request, StepNames.APPLICATION, step_helper)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_new_template_by_sample_set_item(request, lib_pool_id, samplesetitem_ids):
    return page_plan_new_template_by_sample(request, None, lib_pool_id, samplesetitem_ids)

def page_plan_new_template_by_sample(request, sampleset_id=None, lib_pool_id=None, samplesetitem_ids=None):
    """ Create a new template by sample """

    step_helper = StepHelperDbLoader().getStepHelperForNewTemplateBySample(
        _get_runtype_from_code(0).pk, sampleset_id, lib_pool_id, samplesetitem_ids
    )
    if settings.FEATURE_FLAGS.IONREPORTERUPLOADER:
        ctxd = handle_step_request(request, StepNames.IONREPORTER, step_helper)
    else:
        ctxd = handle_step_request(request, StepNames.APPLICATION, step_helper)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_new_plan(request, template_id):
    """ create a new plan from a template with id template_id """

    isSupported = is_operation_supported(template_id)
    if not isSupported:
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForTemplatePlannedExperiment(
        template_id, StepHelperType.CREATE_NEW_PLAN
    )

    apply_prepopulated_values_to_step_helper(request, step_helper)

    next_step_name = redirect_step(StepNames.SAVE_PLAN, step_helper)
    ctxd = handle_step_request(request, next_step_name, step_helper)
    ctxd[
        "step"
    ].validationErrors.clear()  # Remove validation errors found during wizard initialization
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)

def page_plan_new_plan_by_sample_set_item(request, template_id, lib_pool_id, sampleset_item_id):
    return page_plan_new_plan_by_sample(request, template_id, None, lib_pool_id, sampleset_item_id)

def page_plan_new_plan_by_sample(request, template_id, sampleset_id, lib_pool_id=None, sampleset_item_id=None):
    """ create a new plan by sample from a template with id template_id """

    if int(template_id) == int(0):
        # we are creating a new experiment based on the sample set
        return HttpResponseRedirect(
            reverse("page_plan_new_template_by_sample", args=(sampleset_id,))
        )

    isSupported = is_operation_supported(template_id)
    if not isSupported:
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForTemplatePlannedExperiment(
        template_id, StepHelperType.CREATE_NEW_PLAN_BY_SAMPLE, sampleset_id=sampleset_id, lib_pool_id=lib_pool_id, sampleset_item_id=sampleset_item_id
    )
    apply_prepopulated_values_to_step_helper(request, step_helper)

    next_step_name = redirect_step(StepNames.BARCODE_BY_SAMPLE, step_helper)
    ctxd = handle_step_request(request, next_step_name, step_helper)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_new_plan_from_code(request, code):
    """ create a new plan from a runtype code """

    #    if (code == "8"):
    #        isSupported = isOCP_enabled()
    #        if (not isSupported):
    #            return render_to_response("501.html")

    if code == "9":
        step_helper = StepHelperDbLoader().getStepHelperForRunType(
            run_type_id=_get_runtype_from_code("1").pk,
            step_helper_type=StepHelperType.CREATE_NEW_PLAN,
            applicationGroupName="PGx",
        )
    elif code == "11":
        step_helper = StepHelperDbLoader().getStepHelperForRunType(
            run_type_id=_get_runtype_from_code("1").pk,
            step_helper_type=StepHelperType.CREATE_NEW_PLAN,
            applicationGroupName="HID",
        )
    elif code == "12":
        step_helper = StepHelperDbLoader().getStepHelperForRunType(
            run_type_id=_get_runtype_from_code("5").pk,
            step_helper_type=StepHelperType.CREATE_NEW_PLAN,
            applicationGroupName="immune_repertoire",
        )
    elif code == "13":
        step_helper = StepHelperDbLoader().getStepHelperForRunType(
            run_type_id=_get_runtype_from_code("1").pk,
            step_helper_type=StepHelperType.CREATE_NEW_PLAN,
            applicationGroupName="mutation_load",
        )
    elif code == "14":
        step_helper = StepHelperDbLoader().getStepHelperForRunType(
            run_type_id=_get_runtype_from_code("14").pk,
            step_helper_type=StepHelperType.CREATE_NEW_PLAN,
            applicationGroupName="DNA",
        )
    else:
        runType = _get_runtype_from_code(code)
        step_helper = StepHelperDbLoader().getStepHelperForRunType(
            runType.pk, StepHelperType.CREATE_NEW_PLAN
        )

    apply_prepopulated_values_to_step_helper(request, step_helper)
    if settings.FEATURE_FLAGS.IONREPORTERUPLOADER:
        ctxd = handle_step_request(request, StepNames.IONREPORTER, step_helper)
    else:
        ctxd = handle_step_request(request, StepNames.APPLICATION, step_helper)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_edit_plan_by_sample(request, plan_id):

    isSupported = is_operation_supported(plan_id)
    if not isSupported:
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForPlanPlannedExperiment(
        plan_id, StepHelperType.EDIT_PLAN_BY_SAMPLE
    )
    ctxd = handle_step_request(request, StepNames.BARCODE_BY_SAMPLE, step_helper)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_edit_plan(request, plan_id):
    """ edit plan with id plan_id """
    # first get the plan and check if it's plan by sample set

    isSupported = is_operation_supported(plan_id)
    if not isSupported:
        return render_to_response("501.html")

    plan = PlannedExperiment.objects.get(pk=plan_id)
    if plan.sampleSets.exists():
        return HttpResponseRedirect(
            reverse("page_plan_edit_plan_by_sample", args=(plan_id,))
        )

    step_helper = StepHelperDbLoader().getStepHelperForPlanPlannedExperiment(
        plan_id, StepHelperType.EDIT_PLAN
    )
    ctxd = handle_step_request(request, StepNames.SAVE_PLAN, step_helper)
    ctxd[
        "step"
    ].validationErrors.clear()  # Remove validation errors found during wizard initialization
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_edit_run(request, exp_id):
    """ retrieve and edit a plan for existing experiment """
    try:
        plan = PlannedExperiment.objects.get(experiment=exp_id)

        isSupported = is_operation_supported(plan.id)
        if not isSupported:
            return render_to_response("501.html")

    except PlannedExperiment.DoesNotExist:
        # experiment doesn't have a Plan - give it a copy of system default plan template
        exp = Experiment.objects.get(pk=exp_id)
        plan = PlannedExperiment.get_latest_plan_or_template_by_chipType(exp.chipType)
        if not plan:
            plan = PlannedExperiment.get_latest_plan_or_template_by_chipType()
        plan.pk = plan.planGUID = plan.planShortID = None
        plan.isReusable = plan.isSystem = plan.isSystemDefault = False
        plan.planName = plan.planDisplayedName = "CopyOfSystemDefault_" + exp.expName
        plan.planStatus = "run"
        plan.planExecuted = True
        plan.latestEAS = exp.get_EAS()
        plan.save()
        exp.plan = plan
        exp.save()

    step_helper = StepHelperDbLoader().getStepHelperForPlanPlannedExperiment(
        plan.id, StepHelperType.EDIT_RUN
    )
    request.session["return"] = request.META.get("HTTP_REFERER", "")
    ctxd = handle_step_request(request, StepNames.SAVE_PLAN, step_helper)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_copy_plan(request, plan_id):
    """ copy plan with id plan_id """

    isSupported = is_operation_supported(plan_id)
    if not isSupported:
        return render_to_response("501.html")

    plan = PlannedExperiment.objects.get(pk=plan_id)
    if plan.sampleSets.exists():
        return HttpResponseRedirect(
            reverse("page_plan_copy_plan_by_sample", args=(plan_id,))
        )
    step_helper = StepHelperDbLoader().getStepHelperForPlanPlannedExperiment(
        plan_id, StepHelperType.COPY_PLAN
    )

    next_step_name = redirect_step(StepNames.SAVE_PLAN, step_helper)
    ctxd = handle_step_request(request, next_step_name, step_helper)
    ctxd[
        "step"
    ].validationErrors.clear()  # Remove validation errors as this the user is just now seeing the plan.
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_copy_plan_by_sample(request, plan_id):

    isSupported = is_operation_supported(plan_id)
    if not isSupported:
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForPlanPlannedExperiment(
        plan_id, StepHelperType.COPY_PLAN_BY_SAMPLE
    )

    next_step_name = redirect_step(StepNames.BARCODE_BY_SAMPLE, step_helper)
    ctxd = handle_step_request(request, next_step_name, step_helper)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_export(request):
    ctxd = handle_step_request(request, StepNames.EXPORT)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_application(request):
    ctxd = handle_step_request(request, StepNames.APPLICATION)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_save_sample(request):
    ctxd = handle_step_request(request, StepNames.SAVE_PLAN_BY_SAMPLE)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_kits(request):
    ctxd = handle_step_request(request, StepNames.KITS)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_monitoring(request):
    # TODO: Remove Dead Monitoring Page Code
    ctxd = handle_step_request(request, StepNames.MONITORING)
    # return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)
    return HttpResponseRedirect(reverse("page_plan_reference"))


def page_plan_reference(request):
    ctxd = handle_step_request(request, StepNames.REFERENCE)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_plugins(request):
    ctxd = handle_step_request(request, StepNames.PLUGINS)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_output(request):
    ctxd = handle_step_request(request, StepNames.OUTPUT)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_ionreporter(request):
    ctxd = handle_step_request(request, StepNames.IONREPORTER)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_save_template(request):
    ctxd = handle_step_request(request, StepNames.SAVE_TEMPLATE)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_save_template_by_sample(request):
    ctxd = handle_step_request(request, StepNames.SAVE_TEMPLATE_BY_SAMPLE)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_save_plan(request):
    ctxd = handle_step_request(request, StepNames.SAVE_PLAN)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_by_sample_barcode(request):
    ctxd = handle_step_request(request, StepNames.BARCODE_BY_SAMPLE)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_by_sample_save_plan(request):
    ctxd = handle_step_request(request, StepNames.SAVE_PLAN_BY_SAMPLE)
    return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def page_plan_save(request, exp_id=None):
    """ you may only come here from the last plan/template step that has a save button. """

    # update the step_helper with the latest data from the save page
    ctxd = handle_step_request(request, "")
    if "session_error" in ctxd:
        return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)

    step_helper = ctxd["helper"]

    # 20141005-BEWARE: the reference step object held in ctxd[helper] does not seem to be the same as the ones
    # created in StepHelper!!!

    # make sure the step helper is valid
    first_error_step = step_helper.validateAll()
    if not first_error_step:
        try:
            # save the step helper
            step_helper_db_saver = StepHelperDbSaver()
            planTemplate = step_helper_db_saver.save(step_helper, request.user.username)
            update_plan_log(
                step_helper_db_saver.getSavedPlansList(),
                step_helper,
                request.user.username,
            )
            request.session[
                "created_plan_pks"
            ] = step_helper_db_saver.getSavedPlansList()
        except Exception:
            step_helper_db_saver = None
            logger.exception(format_exc())
            Message.error("There was an error saving your plan/template.")

        # this plan is done, delete its session
        plan_session_key = request.session["plan_session_key"]
        PlanSession.objects.filter(
            session_key=request.session.session_key, plan_key=plan_session_key
        ).delete()

        # return response if initiated from prepopulated session API
        if "post_planning_redirect_url" in request.session:
            response = HttpResponseRedirect(
                request.session["post_planning_redirect_url"]
                + "?created_plans="
                + json.dumps(
                    request.session.get("created_plan_pks", "[]"), cls=LazyJSONEncoder
                )
            )
            response["Created-Plans"] = json.dumps(
                request.session.get("created_plan_pks", "[]"), cls=LazyJSONEncoder
            )
            del request.session["post_planning_redirect_url"]
            return response

        # or redirect based on context
        if step_helper.isTemplateBySample():
            if (step_helper.steps[StepNames.IONREPORTER].savedFields[
                            "samplesetitem_ids"
                        ]):
                return HttpResponseRedirect(
                    reverse(
                        "page_plan_new_plan_by_sample_set_item",
                        args=(
                            planTemplate.pk,
                            step_helper.steps[StepNames.IONREPORTER].savedFields[
                                "libraryPool"
                            ],
                            step_helper.steps[StepNames.IONREPORTER].savedFields[
                                "samplesetitem_ids"
                            ],
                        ),
                    )
                )
            return HttpResponseRedirect(
                reverse(
                    "page_plan_new_plan_by_sample",
                    args=(
                        planTemplate.pk,
                        step_helper.steps[StepNames.IONREPORTER].savedFields[
                            "sampleset_id"
                        ],
                    ),
                )
            )
        elif step_helper.isEditRun():
            if exp_id:
                return HttpResponseRedirect(
                    reverse("report_analyze", kwargs={"exp_pk": exp_id, "report_pk": 0})
                )
            else:
                return HttpResponseRedirect(
                    request.session.pop("return")
                    if request.session.get("return")
                    else "/data"
                )
        elif step_helper.isPlan():
            return HttpResponseRedirect("/plan/planned")
        else:
            return HttpResponseRedirect("/plan/plan_templates/#recently_created")
    else:
        # tell the context which step to go to
        ctxd["step"] = step_helper.steps[first_error_step]

        # go to that step
        return render_to_response(ctxd["step"].resourcePath, context_instance=ctxd)


def handle_step_request(request, next_step_name, step_helper=None):

    current_step_name = request.POST.get("stepName", None)
    step_helper, plan_session = _update_plan_session(request, step_helper)

    if not step_helper:
        step_helper = StepHelper()
        ctxd = {
            "step": step_helper.steps.get(current_step_name)
            or list(step_helper.steps.values())[0],
            "session_error": validation.format(
                ugettext_lazy("workflow.messages.errors.internal.session_error"),
                include_error_prefix=True,
            ),  # 'Error: Unable to retrieve planning session'
        }
        return RequestContext(request, ctxd)

    # logger.debug("views.handle_step_request() current_step_name=%s; next_step_name=%s" %(current_step_name, next_step_name))
    application_step_data = step_helper.steps["Application"]
    applProduct = application_step_data.savedObjects["applProduct"]

    # update the step helper with the input from the step we came from
    updated = step_helper.updateStepFromRequest(request, current_step_name)
    # raise RuntimeError("what" + '%s' % updated)
    logger.debug("Step: %s, Updated: %s" % (current_step_name, str(updated)))
    if updated:
        request.session.modified = True

    if current_step_name and step_helper.steps[current_step_name].hasErrors():
        if step_helper.isTargetStepAfterOriginal(current_step_name, next_step_name):
            next_step_name = current_step_name
        elif step_helper.steps[current_step_name].hasStepSections():
            next_step_name = current_step_name

    plan_session.set_data(step_helper)
    ctxd = _create_context_from_session(request, next_step_name, step_helper)
    return ctxd


def redirect_step(next_step_name, step_helper):
    # redirects next step if warnings exist
    redirect = next_step_name
    for name, step in step_helper.steps.items():
        if len(step.warnings) > 0 or name == next_step_name:
            redirect = name
            break
    return redirect


def update_plan_log(saved_plans, step_helper, username):
    # Add log entry when a Plan is created or modified
    if not saved_plans or not step_helper.isPlan():
        return

    edited_plan_pk = saved_plans[0]
    created_plans = []
    if step_helper.isCreate() or step_helper.isCopy():
        created_plans = saved_plans
        edited_plan_pk = None
    elif len(saved_plans) > 1:
        created_plans = saved_plans[1:]

    for pk in created_plans:
        plan = PlannedExperiment.objects.get(pk=pk)
        msg = "Created Planned Run: %s (%s)" % (plan.planName, plan.pk)
        EventLog.objects.add_entry(plan, msg, username)

    if edited_plan_pk:
        plan = PlannedExperiment.objects.get(pk=edited_plan_pk)
        state = "Run" if step_helper.isEditRun() else "Planned Run"
        msg = "Updated %s: %s (%s)." % (state, plan.planName, plan.pk)
        EventLog.objects.add_entry(plan, msg, username)


def toggle_template_favorite(request, template_id):
    """
    toggle a template to set/unset the favorite tag with id template_id
    """
    plan_template = get_object_or_404(PlannedExperiment, pk=template_id)
    plan_template.isFavorite = False if (plan_template.isFavorite) else True
    plan_template.save()

    return HttpResponse()


def _get_runtype_from_code(code):
    codes = {
        "1": "AMPS",
        "2": "TARS",
        "3": "WGNM",
        "4": "RNA",
        "5": "AMPS_RNA",
        "6": "AMPS_EXOME",
        "7": "TARS_16S",
        "8": "AMPS_DNA_RNA",
        #'9': PGx - AMPS_DNA
        "10": "TAG_SEQUENCING",
        "11": "HID",
        "12": "AMPS_RNA",
        "14": "AMPS_HD_DNA",
    }
    product_code = codes.get(code, "GENS")
    return RunType.objects.get(runType=product_code)


def _create_context_from_session(request, next_step_name, step_helper):
    ctxd = {
        "plan_session_key": request.session["plan_session_key"],
        "helper": step_helper,
        "step": step_helper.steps.get(next_step_name),
    }
    return RequestContext(request, ctxd)


def _update_plan_session(request, step_helper):
    # get or create unique plan session id
    key = request.POST.get("plan_session_key")
    if not key and step_helper:
        key = "%s-%s" % (
            step_helper.sh_type,
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        )

    ps = None
    if key:
        request.session["plan_session_key"] = key
        ps, created = PlanSession.objects.get_or_create(
            session_key=request.session.session_key, plan_key=key
        )
        if not created:
            step_helper = ps.get_data()

    return step_helper, ps


def planned(request):
    ctx = RequestContext(request)

    if "created_plan_pks" in request.session:
        ctx["created_plan_pks"] = request.session["created_plan_pks"]
        del request.session["created_plan_pks"]

    ctx["planshare"] = IonMeshNode.objects.filter(active=True)
    ctx["runTypes"] = RunType.objects.filter(isActive=True)
    ctx["projects"] = Project.objects.order_by("name").values_list("name", flat=True)
    ctx["barcodes"] = (
        dnaBarcode.objects.filter(active=True)
        .values_list("name", flat=True)
        .distinct()
        .order_by("name")
    )
    ctx["references"] = (
        ReferenceGenome.objects.filter(enabled=True)
        .values_list("short_name", flat=True)
        .order_by("short_name")
    )

    return render_to_response("rundb/plan/planned.html", context_instance=ctx)


def delete_plan_template(request, pks=None):
    # TODO: See about pulling this out into a common methods
    pks = pks.split(",")
    _type = "plannedexperiment"
    planTemplates = get_list_or_404(PlannedExperiment, pk__in=pks)
    isPlanTemplate = planTemplates[0].isReusable is True
    actions = [
        reverse(
            "api_dispatch_detail",
            kwargs={"resource_name": _type, "api_name": "v1", "pk": int(pk)},
        )
        for pk in pks
    ]
    names = ", ".join([x.planName for x in planTemplates])

    isMultiple = len(pks) > 1
    isPlural = isMultiple  # same condition
    if isPlanTemplate:  # "Template"
        submit = _("template.modal_confirm_delete.action.submit")
        cancel = _("template.modal_confirm_delete.action.cancel")
        title = _(
            "template.modal_confirm_delete.title.singular"
        )  # Confirm Delete Template
        confirmmsg = _("template.modal_confirm_delete.messages.confirmmsg.singular") % {
            "templateId": u", ".join(pks),
            "templateName": names,
        }
        if isPlural:
            title = _(
                "template.modal_confirm_delete.title.plural"
            )  # Confirm Delete Templates
            confirmmsg = _("template.modal_confirm_delete.messages.confirmmsg.plural")
    else:  # "Planned Run"
        submit = _("plannedrun.modal_confirm_delete.action.submit")
        cancel = _("plannedrun.modal_confirm_delete.action.cancel")
        title = _("plannedrun.modal_confirm_delete.title.singular")  #
        confirmmsg = _(
            "plannedrun.modal_confirm_delete.messages.confirmmsg.singular"
        ) % {"plannedRunId": u", ".join(pks), "plannedRunName": names}
        if isPlural:
            title = _("plannedrun.modal_confirm_delete.title.plural")
            confirmmsg = _("plannedrun.modal_confirm_delete.messages.confirmmsg.plural")

    ctx = RequestContext(
        request,
        {
            "id": pks[0],
            "ids": json.dumps(pks, cls=LazyJSONEncoder),
            "method": "DELETE",
            "action": actions[0],
            "actions": json.dumps(actions, cls=LazyJSONEncoder),
            "items": [{"id": x.pk, "name": x.planName} for x in planTemplates],
            "isMultiple": isMultiple,
            "i18n": {
                "title": title,
                "confirmmsg": confirmmsg,
                "submit": _("samplesets.modal_confirm_delete.action.submit"),
                "cancel": _("samplesets.modal_confirm_delete.action.cancel"),
            },
        },
    )
    return render_to_response(
        "rundb/common/modal_confirm_delete.html", context_instance=ctx
    )


class PlanDetailView(DetailView):
    model = PlannedExperiment
    template_name = "rundb/plan/modal_plannedexperiment_detail.html"
    context_object_name = "plan"

    def get_context_data(self, **kwargs):
        plan = self.object

        # can Review from either Report or Planned Runs pages
        report_pk = self.kwargs.get("report_pk")
        if report_pk:
            result = Results.objects.get(pk=report_pk)
            eas = result.eas
            state = "Plan"
        else:
            eas = plan.latestEAS or plan.experiment.get_EAS()
            state = "Template" if plan.isReusable else "Planned Run"

        chipType = Chip.objects.filter(name=plan.get_chipType()).values_list(
            "description", flat=True
        )

        # generate context for page
        context = super(PlanDetailView, self).get_context_data(**kwargs)
        if report_pk:
            context["result"] = result
        context["state"] = state
        context["eas"] = eas
        context["runType"] = RunType.objects.filter(runType=plan.runType)

        if plan.runType:
            runType = str(plan.runType)
            context[
                "applicationCategoryDisplayedName"
            ] = PlannedExperiment.get_validatedApplicationCategoryDisplayedName(
                plan.categories, runType
            )
        else:
            context[
                "applicationCategoryDisplayedName"
            ] = PlannedExperiment.get_applicationCategoryDisplayedName(plan.categories)

        context["samplePrepKit"] = KitInfo.objects.filter(name=plan.samplePrepKitName)
        context["libraryKit"] = KitInfo.objects.filter(name=eas.libraryKitName)
        context["templatingKit"] = KitInfo.objects.filter(name=plan.templatingKitName)
        context["samplePrepProtocol"] = (
            common_CV.objects.filter(value=plan.samplePrepProtocol)
            if plan.samplePrepProtocol
            else ""
        )
        context["sequenceKit"] = KitInfo.objects.filter(
            name=plan.experiment.sequencekitname
        )
        context["controlSequencekit"] = KitInfo.objects.filter(
            name=plan.controlSequencekitname
        )
        context["chipTypePrefix"] = (
            getChipDisplayedNamePrimaryPrefix(chipType[0])
            if chipType
            else plan.experiment.chipType
        )
        context["chipTypeSecondaryPrefix"] = (
            getChipDisplayedNameSecondaryPrefix(chipType[0])
            if chipType
            else plan.experiment.chipType
        )
        context["chipTypeVersion"] = (
            getChipDisplayedVersion(chipType[0]) if chipType else ""
        )

        _chipType = Chip.objects.filter(name=plan.get_chipType()).order_by("-isActive")
        context["chipType"] = _chipType[0] if _chipType else None

        context["ampsOnChef_sampleSets"] = plan.sampleSets.filter(
            libraryPrepInstrumentData__isnull=False,
            libraryPrepType__contains="amps_on_chef",
        )
        context["combinedLibraryTubeLabel"] = plan.libraryPool

        for sampleSet in context["ampsOnChef_sampleSets"]:
            if plan.libraryPool:
                context["combinedLibraryTubeLabel"] = sampleSet.combinedLibraryTubeLabel.split(',')[int(plan.libraryPool) - 1]
            else:
                context["combinedLibraryTubeLabel"] = sampleSet.combinedLibraryTubeLabel

        context["thumbnail"] = True if report_pk and result.isThumbnail else False
        context["show_thumbnail"] = (
            True
            if state != "Plan" and (plan.experiment.getPlatform in ["s5", "proton"])
            else False
        )
        context["from_report"] = True if report_pk else False

        # display saved args if viewing from existing Report or if custom args, otherwise display current default args
        context["args"] = (
            eas.get_cmdline_args()
            if report_pk or eas.custom_args
            else plan.get_default_cmdline_args()
        )

        # IRU configuration
        iru_info = eas.selectedPlugins.get("IonReporterUploader", {}).get(
            "userInput", {}
        )
        iru_info = (
            iru_info["userInputInfo"] if "userInputInfo" in iru_info else iru_info
        )
        # map columns to be displayed to actual parameter names saved in userInputInfo
        # non-empty columns will appear in the order given
        iru_display_keys = (
            ("Cancer Type", "cancerType"),
            ("Cellularity %", "cellularityPct"),
            ("Workflow", "Workflow"),
            ("Relation", "RelationRole"),
            ("Gender", "Gender"),
            ("Set ID", "setid"),
        )
        context["iru_config"] = {}

        if iru_info:
            # unicode handling
            iru_info = convert(iru_info)
            # logger.debug("views.PlanDetailView - AFTER iru_info=%s" %(iru_info))

        for config_dict in iru_info:
            params = {}
            columns = []
            if isinstance(config_dict, dict):
                for column, key in iru_display_keys:
                    if key == "Workflow" and config_dict.get(key) == "":
                        config_dict["Workflow"] = "Upload Only"
                    if key in config_dict and config_dict[key]:
                        params[column] = config_dict[key]
                        columns.append(column)
                        if key == "setid":
                            params["Set ID"] = config_dict["setid"].split("_")[0]
            else:
                logger.debug(
                    "views.PlanDetailView - SKIPPED -  config_dict.type=%s"
                    % (type(config_dict))
                )

            if params:
                barcoded = config_dict.get("barcodeId") or "nobarcode"
                if barcoded in context["iru_config"]:
                    irworkflow = context["iru_config"][barcoded]["Workflow"]
                    multiWorkflowList = irworkflow if isinstance(irworkflow, list) else [str(irworkflow)]
                    multiWorkflowList.append(str(params["Workflow"]))
                    params["Workflow"] = multiWorkflowList
                context["iru_config"][barcoded] = params
                context["iru_columns"] = [
                    v[0] for v in iru_display_keys if v[0] in columns
                ]

        # QC thresholds
        context["qcValues"] = []
        for qcValue in plan.qcValues.all().order_by("qcName"):
            context["qcValues"].append(
                {
                    "qcName": qcValue.qcName,
                    "label": ModelsQcTypeToLabelsQcType(qcValue.qcName),
                    "threshold": PlannedExperimentQC.objects.get(
                        plannedExperiment=plan, qcType=qcValue
                    ).threshold,
                }
            )

        # plugins
        context["plugins"] = filter(
            lambda d: "export" not in d.get("features", []),
            list(eas.selectedPlugins.values()),
        )
        context["uploaders"] = filter(
            lambda d: "export" in d.get("features", []),
            list(eas.selectedPlugins.values()),
        )

        plugins = Plugin.objects.filter(
            name__in=list(eas.selectedPlugins.keys()), active=True
        )
        for plugin in context["plugins"]:
            if plugin.get("userInput"):
                pluginObj = plugins.filter(name=plugin["name"])
                if pluginObj:
                    plugin["isPlanConfig"] = pluginObj[0].isPlanConfig
                    plugin["userInputJSON"] = json.dumps(
                        plugin["userInput"], cls=LazyDjangoJSONEncoder
                    )
        context["plugins"].sort(key=lambda d: d["name"])

        # projects
        if report_pk:
            context["projects"] = result.projectNames()
        else:
            context["projects"] = ", ".join(
                [p.name for p in plan.projects.all().order_by("-modified")]
            )

        # barcodes
        if eas.barcodeKitName and eas.barcodedSamples:
            barcodedSamples = {}
            columns = []
            applicationGroup = (
                plan.applicationGroup.name if plan.applicationGroup else ""
            )
            bcsamples_display_keys = (
                (
                    _("reviewplan.fields.barcodedSamples.barcode.controlType.label"),
                    "controlType",
                ),  #'Control Type'
                (
                    _("reviewplan.fields.barcodedSamples.barcode.description.label"),
                    "description",
                ),  #'Sample Description'
                (
                    _("reviewplan.fields.barcodedSamples.barcode.externalId.label"),
                    "externalId",
                ),  #'Sample ID'
                (
                    _(
                        "reviewplan.fields.barcodedSamples.barcode.nucleotideType.label.DNA + RNA"
                    )
                    if applicationGroup == "DNA + RNA"
                    else _(
                        "reviewplan.fields.barcodedSamples.barcode.nucleotideType.label"
                    ),
                    "nucleotideType",
                ),  #'DNA/RNA' OR 'DNA/Fusions'
                (
                    _("reviewplan.fields.barcodedSamples.barcode.reference.label"),
                    "reference",
                ),  #'Reference'
                (
                    _(
                        "reviewplan.fields.barcodedSamples.barcode.targetRegionBedFile.label"
                    ),
                    "targetRegionBedFile",
                ),  #'Target Regions'
                (
                    _(
                        "reviewplan.fields.barcodedSamples.barcode.hotSpotRegionBedFile.label"
                    ),
                    "hotSpotRegionBedFile",
                ),  #'Hotspot Regions'
            )

            barcodes = []
            for sample, info in list(eas.barcodedSamples.items()):
                for bcId in info.get("barcodes", []):
                    barcodeKey = self._get_barcode_sample_barcode_key(plan, info, bcId)
                    bcId2 = barcodeKey
                    barcodes.append(barcodeKey)

                    barcodedSamples[bcId2] = {}
                    barcodedSamples[bcId2]["sample"] = sample

                    barcodeSampleInfo = info.get("barcodeSampleInfo", {}).get(bcId, {})
                    for column, key in bcsamples_display_keys:
                        if key in barcodeSampleInfo and barcodeSampleInfo[key]:
                            if column not in columns:
                                columns.append(column)

                            value = barcodeSampleInfo[key]
                            if key in ["targetRegionBedFile", "hotSpotRegionBedFile"]:
                                value = os.path.basename(barcodeSampleInfo[key])
                            elif (
                                key == "nucleotideType"
                                and value == "RNA"
                                and applicationGroup == "DNA + RNA"
                            ):
                                value = "Fusions"

                            barcodedSamples[bcId2][column] = value

            context["bcsamples_columns"] = [
                v[0] for v in bcsamples_display_keys if v[0] in columns
            ]
            context["barcodedSamples"] = barcodedSamples
            barcodes.sort()
            context["barcodes"] = barcodes

        # LIMS data
        if plan.metaData:
            data = plan.metaData.get("LIMS", "")
            if type(data) is list:
                # convert list to string
                context["LIMS_meta"] = "".join(data)
            else:
                # convert unicode to str
                context["LIMS_meta"] = convert(data)
        else:
            context["LIMS_meta"] = ""

        if plan.origin:
            planMeta = plan.origin.split("|")
            context["origin"] = planMeta[0].upper()
            if len(planMeta) > 1:
                context["tsVersion"] = planMeta[1]

        # Library Prep Protocol
        libPrepProtocols = plan.sampleSets.exclude(libraryPrepProtocol="Unspecified").values_list(
            "libraryPrepProtocol", flat=True
        )
        if len(libPrepProtocols) > 0:
            libraryPrepProtocolsDisplayed = common_CV.objects.filter(value__in=libPrepProtocols).values_list(
                "displayedValue", flat=True
            )
            context["libraryPrepProtocol"] = ", ".join(libraryPrepProtocolsDisplayed)

        # log
        history = EventLog.objects.for_model(plan).order_by("-created")
        for log in history:
            try:
                log.json_log = convert(json.loads(log.text))
            except Exception:
                pass
        context["event_log"] = history

        return context

    def _get_barcode_sample_barcode_key(
        self, plan, barcodedSampleItemInfo, startBarcode
    ):
        """
        If plan is a dualBarcoded plan, return the dualBarcode as the key. Otherwise, return the input barcodeId as the key.
        startBarcode has to be unique among all the startBarcodes used within a plan.

        barcodedSampleItemInfo - the barcodeSampleInfo JSON definition for a sample in experimentAnalysisSettings.barcodedSamples
        startBarcode - one start barcode specified in the the barcodes list for a sample in experimentAnalysisSettings.barcodedSamples
        """
        if not plan.get_endBarcodeKitName():
            return startBarcode
        dualBarcodes = barcodedSampleItemInfo.get("dualBarcodes", [])
        return (
            self._getDualBarcode_for_matching_startBarcode(dualBarcodes, startBarcode)
            if dualBarcodes
            else startBarcode
        )

    def _getDualBarcode_for_matching_startBarcode(self, dualBarcodes, startBarcode):
        """
        return the entry with matching startBarcode in a list of barcode pairs
        dualBarcodes is a list of dualBarcodes in the form of startBarcode--endBarcode
        e.g., IonXpress_015--IonSet1_15
        """
        if not startBarcode or not dualBarcodes:
            return startBarcode
        for dualBarcode in dualBarcodes:
            dualBarcodeTokens = dualBarcode.split(
                PlannedExperiment.get_dualBarcodes_delimiter()
            )
            if dualBarcodeTokens and dualBarcodeTokens[0] == startBarcode:
                return dualBarcode
        return startBarcode


def batch_plans_from_template(request, template_id):
    """
    To create multiple plans from an existing template
    """

    planTemplate = get_object_or_404(PlannedExperiment, pk=template_id)

    # planTemplateData contains what are available for selection
    ctxd = {
        "selectedPlanTemplate": planTemplate,
        "is_barcoded": True if planTemplate.get_barcodeId() else False,
    }
    context = RequestContext(request, ctxd)
    return render_to_response(
        "rundb/plan/modal_batch_planning.html", context_instance=context
    )


def getCSV_for_batch_planning(request, templateId, count, uploadtype="single"):
    """
    To create csv file for batch planning based on an existing template
    From Rel 5.4 onwards : The download option supports only column based CSV download for batch planning
    """

    def create_csv(output, version, hdr, body, isSampleFile=None):
        writer = csv.writer(output)
        writer.writerow(version)

        if isSampleFile:
            writer.writerow(hdr)
            for row in body:
                writer.writerow(row)
        else:
            colbased_plan_csv_headingRow = [PlanCSVcolumns.COLUMN_PLAN_HEADING_KEY]
            if len(body):
                for index, item in enumerate(body, start=1):
                    planValueHeading = "%s%d" % (
                        PlanCSVcolumns.COLUMN_PLAN_HEADING_VALUE,
                        index,
                    )
                    colbased_plan_csv_headingRow.append(planValueHeading)
            writer.writerow(colbased_plan_csv_headingRow)

            for head_index, item in enumerate(hdr):
                row = []
                row.append(item)
                for body_index, body_item in enumerate(body):
                    for eachColIndex, eachColItem in enumerate(body_item):
                        if eachColIndex == head_index:
                            row.append(body_item[eachColIndex])
                            break
                writer.writerow(row)

    single_file = uploadtype == "single"
    filename = "tsPlan_%s" % str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    batch_filename = filename.replace("tsPlan", "tsBatchPlanning")
    plan_csv_version = get_plan_csv_version()

    hdr, body = get_template_data_for_batch_planning(templateId, single_file)
    if single_file:
        response = http.HttpResponse(mimetype="text/csv")
        response["Content-Disposition"] = "attachment; filename=%s.csv" % batch_filename
        rows = int(count) * [body]
        create_csv(response, plan_csv_version, hdr, rows)
    else:
        response = HttpResponse(mimetype="application/zip")
        response["Content-Disposition"] = "attachment; filename=%s.zip" % batch_filename
        zip_file = zipfile.ZipFile(response, "w")

        rows = []
        samples_hdr, samples_body = get_samples_data_for_batch_planning(templateId)
        for n in range(int(count)):
            name = filename + "_plan_%02d_samples.csv" % (n + 1)
            body[hdr.index(PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER)] = name
            rows.append(list(body))
            # create and add samples CSV file
            csv_output = io.BytesIO()
            create_csv(
                csv_output,
                plan_csv_version,
                samples_hdr,
                samples_body,
                isSampleFile=True,
            )
            zip_file.writestr(name, csv_output.getvalue())

        # create and add plan CSV file
        csv_output = io.BytesIO()
        create_csv(csv_output, plan_csv_version, hdr, rows)
        zip_file.writestr(filename + "_master.csv", csv_output.getvalue())

    return response


def upload_plans_for_template(request):
    """
    Allow user to upload a csv file to create plans based on a previously selected template
    """

    ctxd = {
        "KEY_BARCODED_SAMPLES_VALIDATION_ERRORS": BARCODED_SAMPLES_VALIDATION_ERRORS_(),
        "KEY_SAMPLE_CSV_FILE_NAME": SAMPLE_CSV_FILE_NAME_,
        "KEY_IRU_VALIDATION_ERRORS": IRU_VALIDATION_ERRORS_(),
    }

    context = RequestContext(request, ctxd)
    return render_to_response(
        "rundb/plan/modal_batch_planning_upload.html", context_instance=context
    )


def _get_reader_for_v2(columnBasedReader):
    """
        Get the reader Dict for the column based CSV
          - csv module handles data row-wise
          - process the column based data set and return the dict
          - this function is called only for main plan csv file, not for the samples csv
    """
    rowBasedReader = []
    for colInput in columnBasedReader:
        rowBasedReader.append(tuple(colInput))

    # null values gets trimmed when no input in the last row
    # parse it and handle gracefully
    if len(rowBasedReader[-1]) != len(rowBasedReader[0]):
        missingelement = len(rowBasedReader[0]) - len(rowBasedReader[-1])
        lastElement = list(rowBasedReader[-1])
        lastElement.extend([""] * missingelement)
        rowBasedReader[-1] = tuple(lastElement)

    converted = zip(*rowBasedReader)
    headers = list(converted.pop(0))

    v2_reader = []
    for row in converted:
        processed_dictReader = {}
        planDetails = list(row)
        for rowIndex, rowValue in enumerate(planDetails):
            for headerIndex, headerItem in enumerate(headers):
                if rowIndex == headerIndex:
                    processed_dictReader[headerItem] = rowValue
                    break
        v2_reader.append(processed_dictReader)
    return v2_reader

@transaction.commit_manually
def save_uploaded_plans_for_template(request):
    """add plans, with CSV validation"""

    def _close_files(files, temp_file):
        for csvfile in list(files.values()):
            if not csvfile.closed:
                csvfile.close()
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        return {}, None

    def _fail(files={}, temp_file=None, status="", failed=None):
        # helper method to clean up and return HttpResponse with error messages
        transaction.rollback()
        _close_files(files, temp_file)
        error = {"status": status}
        if failed:
            error["failed"] = failed
        logger.info("views.save_uploaded_plans_for_template() error=%s" % (error))
        return HttpResponse(
            json.dumps(error, cls=LazyJSONEncoder), mimetype="text/html"
        )

    logger.debug("language=%s", translation.get_language())
    postedfile = request.FILES["postedfile"]
    destination = tempfile.NamedTemporaryFile(delete=False)

    for chunk in postedfile.chunks():
        destination.write(chunk)
    postedfile.close()
    destination.close()

    files = {}
    non_ascii_files = []
    if zipfile.is_zipfile(destination.name):
        myzip = zipfile.ZipFile(destination.name)
        for filename in myzip.namelist():
            if ("__MACOSX" in filename) or (myzip.getinfo(filename).file_size == 0):
                continue

            # open read bytes and detect if file contains non-ascii characters
            with myzip.open(filename, "rU") as _tmp:
                if not is_ascii(_tmp.read()):  # if not ascii
                    non_ascii_files.append(os.path.basename(filename))

            files[os.path.basename(filename)] = myzip.open(filename, "rU")
    else:
        files[destination.name] = open(destination.name, "rU")

        # open read bytes and detect if file contains non-ascii characters
        with open(destination.name, "rU") as _tmp:
            if not is_ascii(_tmp.read()):  # if not ascii
                non_ascii_files.append(postedfile.name)  # add uploaded file name

    if len(files) == 0:
        return _fail(
            temp_file=destination,
            status=ugettext_lazy("upload_plans_for_template.messages.error.fileempty"),
        )  # "Error: batch planning file is empty"

    if len(non_ascii_files) > 0:
        # "Only ASCII characters are supported. The following files contain non-ASCII characters: %(files)s."
        _error_msg = validation.format(
            ugettext(
                "upload_plans_for_template.messages.error.file_contains_non_ascii_characters"
            ),
            {"files": SeparatedValuesBuilder().build(non_ascii_files)},
        )
        _error = {"NON_ASCII_FILES": _error_msg}
        return _fail(temp_file=destination, status=_error_msg, failed=_error)
    single_file = len(files) == 1
    failed = {}
    isToAbort = False
    # validate files contents and CSV version
    csv_version_header = get_plan_csv_version()[0]
    for filename, csvfile in list(files.items()):
        err_name = postedfile.name if single_file else filename
        try:
            csv_version_row = csv.reader(csvfile).next()
            errorMsg, isToSkipRow, abortFile = utils.validate_csv_template_version(
                headerName=csv_version_header,
                isPlanCSV=True,
                firstRow=csv_version_row,
                PlanCSVTemplateLabel=ugettext(
                    "upload_plans_for_template.fields.postedfile.label"
                ),
            )
            if abortFile:
                isToAbort = True
                failed[err_name] = [errorMsg[0][1]]
        except Exception:
            logger.error(traceback.format_exc())
            failed[err_name] = [
                ugettext_lazy(
                    "upload_plans_for_template.messages.error.errorreadingfile"
                )
            ]  # "Error reading from file"
            isToAbort = True

    if isToAbort:
        return _fail(files, destination, "", failed)

    samples = {}
    input_plan_count = 0
    isRowBased = False
    csv_version = StrictVersion(str(float(csv_version_row[1])))

    if single_file:
        if csv_version >= "2.0":
            # Column based CSV version : 2.0
            columnBasedReader = csv.reader(list(files.values())[0])
            reader = _get_reader_for_v2(columnBasedReader)
            input_plan_count = len(reader)
        else:
            isRowBased = True
            # Row Based : Support backward compatibility for CSV version 1
            reader = csv.DictReader(list(files.values())[0])
    else:
        reader = None
        for filename, csvfile in list(files.items()):
            if csv_version >= "2.0":
                # Column based CSV version : 2.0
                test = csv.DictReader(csvfile)
                fieldnames = test.fieldnames
                if PlanCSVcolumns.COLUMN_PLAN_HEADING_KEY in fieldnames:
                    # process only main plan CSV Data set
                    columnBasedReader = io.TextIOWrapper(csvfile)
                    columnBasedReader = csv.reader(columnBasedReader)
                    test = _get_reader_for_v2(columnBasedReader)
                    input_plan_count = len(test)
                    fieldnames = list(test[0].keys())
            else:
                isRowBased = True
                test = csv.DictReader(csvfile)
                fieldnames = test.fieldnames
            if PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER in fieldnames:
                if reader:
                    return _fail(
                        files,
                        destination,
                        status=ugettext_lazy(
                            "upload_plans_for_template.messages.error.zip.multipleplansfiles"
                        ),
                    )  # "Error: multiple batch planning files found"
                else:
                    reader = test
            else:
                samples[filename] = list(test)

        if not reader:
            return _fail(
                files,
                destination,
                status=ugettext_lazy(
                    "upload_plans_for_template.messages.error.zip.missingplanfile"
                ),
            )  # "Error: batch planning file not found for barcoded CSV files upload"
        if not samples:
            return _fail(
                files,
                destination,
                status=ugettext_lazy(
                    "upload_plans_for_template.messages.error.zip.missingsamplefile"
                ),
            )  # "Error: batch planning no samples files found for barcoded ZIP upload"

    index = 0
    plans = []
    rawPlanDataList = []
    username = request.user.username if request.user else ""
    row_or_column = "Column" if csv_version >= "2.0" else "Row"

    # process and validate Plans
    for index, row in enumerate(reader, start=2):
        if isRowBased:
            if "Plan Parameters" in row:
                # this happens rarely, If user updates the csv version to old version and keeps it column based, then exit
                logger.error(
                    "views.save_uploaded_plans_for_template(): User has wrongly manipulated the uploaded CSV file"
                )
                errorMsg = ugettext_lazy(
                    "upload_plans_for_template.messages.error.unsupportedheaderinfo"
                )  # "Error: CSV file is invalid due to unsupported header info. Please download the latest CSV version and reload."
                return _fail(files, destination, status=errorMsg)
            input_plan_count += 1
        samples_contents = (
            samples.get(row[PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER])
            if not single_file
            else None
        )
        errorMsg, aPlanDict, rawPlanDict, isToSkipRow = validate_csv_plan(
            row, username, single_file, samples_contents, httpHost=request.get_host()
        )

        logger.info(
            "views.save_uploaded_plans_for_template() index=%d; errorMsg=%s; planDict=%s"
            % (index, errorMsg, rawPlanDict)
        )
        if errorMsg:
            failed["%s%d" % (row_or_column, index)] = errorMsg
            continue
        elif isToSkipRow:
            logger.info(
                "views.save_uploaded_plans_for_template() SKIPPED ROW index=%d; row=%s"
                % (index, row)
            )
            continue
        else:
            plans.append(aPlanDict)
            rawPlanDataList.append(rawPlanDict)

    # now close and remove the temp file
    files, destination = _close_files(files, destination)

    if index == 0:
        return _fail(
            files,
            destination,
            status=ugettext_lazy(
                "upload_plans_for_template.messages.error.planrequired"
            ),
        )  # "Error: There must be at least one plan! Please reload the page and try again with more plans."

    # get the total no. of failures
    if failed:
        total_error_count = 0
        for key, value in failed.items():
            errorDict = dict(value)
            if single_file:
                total_error_count += len(errorDict)
            else:
                for key, value in errorDict.items():
                    if SAMPLE_CSV_FILE_NAME_ in key:
                        continue
                    try:
                        total_error_count += len(json.loads(value))
                    except Exception as err:
                        if isinstance(value, basestring):
                            total_error_count += 1
                        else:
                            total_error_count += len(value)
                        logger.debug(err)

        # Sort the error plans irrespective of row and column
        errorPlansSorted = []
        for plan in failed:
            errorPlansSorted.append(plan)
        errorPlansSorted.sort(key=lambda x: "{0:0>10}".format(x).lower())

        failedOrderedDict = collections.OrderedDict()
        for errorPlan in errorPlansSorted:
            failedOrderedDict[errorPlan] = failed[errorPlan]

        r = {
            "status": ugettext_lazy(
                "upload_plans_for_template.messages.error.validationfailed"
            ),  # "Plan validation failed. The plans have not been saved."
            "failed": failedOrderedDict,
            "plansFailed": len(errorPlansSorted),
            "inputPlanCount": input_plan_count,
            "totalErrors": total_error_count,
            "singleCSV": single_file,
        }

        logger.info("views.save_uploaded_plans_for_template() failed=%s" % (r))

        transaction.rollback()
        return HttpResponse(json.dumps(r, cls=LazyJSONEncoder), mimetype="text/html")

    # saving to db needs to be the last thing to happen
    try:
        index = 0
        for planFamily in plans:
            plan = planFamily["plan"]
            plan.save()

            expObj = planFamily["exp"]
            expObj.plan = plan
            expObj.expName = plan.planGUID
            expObj.unique = plan.planGUID
            expObj.displayname = plan.planGUID
            expObj.save()

            easObj = planFamily["eas"]
            easObj.experiment = expObj
            easObj.isEditable = True
            easObj.save()

            # set default cmdline args - this is needed in case chip type or kits changed from the selected template's values
            if not easObj.custom_args:
                easObj.reset_args_to_default()

            # associate EAS to the plan
            plan.latestEAS = easObj
            plan.save()

            # saving/associating samples
            sampleDisplayedNames = planFamily["samples"]
            sampleNames = [name.replace(" ", "_") for name in sampleDisplayedNames]
            externalIds = planFamily["sampleIds"]
            for name, displayedName, externalId in zip(
                sampleNames, sampleDisplayedNames, externalIds
            ):
                sample_kwargs = {
                    "name": name,
                    "displayedName": displayedName,
                    "date": plan.date,
                    "status": plan.planStatus,
                    "externalId": externalId,
                }

                sample = Sample.objects.get_or_create(
                    name=name, externalId=externalId, defaults=sample_kwargs
                )[0]
                sample.experiments.add(expObj)
                sample.save()

            planDict = rawPlanDataList[index]

            # add QCtype thresholds
            qcTypes = QCType.objects.all()
            for qcType in qcTypes:
                qc_threshold = planDict.get(qcType.qcName, "")
                if qc_threshold:
                    # get existing PlannedExperimentQC if any
                    plannedExpQcs = PlannedExperimentQC.objects.filter(
                        plannedExperiment=plan.id, qcType=qcType.id
                    )
                    if len(plannedExpQcs) > 0:
                        for plannedExpQc in plannedExpQcs:
                            plannedExpQc.threshold = qc_threshold
                            plannedExpQc.save()
                    else:
                        kwargs = {
                            "plannedExperiment": plan,
                            "qcType": qcType,
                            "threshold": qc_threshold,
                        }
                        plannedExpQc = PlannedExperimentQC(**kwargs)
                        plannedExpQc.save()

            # add projects
            projectObjList = get_projects(request.user, planDict)
            for project in projectObjList:
                if project:
                    plan.projects.add(project)

            index += 1
    except:
        logger.error(format_exc())
        transaction.rollback()

        r = {
            "status": ugettext_lazy("upload_plans_for_template.messages.error.saving"),
            "failed": failed,
        }  # "Error saving plans to database. "
        return HttpResponse(json.dumps(r, cls=LazyJSONEncoder), mimetype="text/html")
    else:
        transaction.commit()

        warnings = {}
        for i, planDict in enumerate(rawPlanDataList):
            if planDict.get("warnings"):
                warnings["%s%d" % (row_or_column, i + 2)] = planDict["warnings"]

        r = {
            "status": ugettext_lazy("upload_plans_for_template.messages.success"),
            "failed": failed,
            "warnings": warnings,
        }  # "Plans Uploaded! The plans will be listed on the planned run page."
        return HttpResponse(json.dumps(r, cls=LazyJSONEncoder), mimetype="text/html")


def plan_transfer(request, pk, destination_server_name=None):
    plan = get_object_or_404(PlannedExperiment, pk=pk)
    ctxd = {
        "planName": plan.planDisplayedName,
        "destination_server_name": destination_server_name,
        "action": reverse(
            "api_dispatch_transfer",
            kwargs={
                "resource_name": "plannedexperiment",
                "api_name": "v1",
                "pk": int(pk),
            },
        ),
        "i18n": {
            "title": ugettext_lazy(
                "plan_transfer.title"
            ),  # 'Confirm Transfer of Planned Run',
            "confirmmsg": ugettext_lazy("plan_transfer.messages.confirmmsg")
            % {
                "destination_server_name": destination_server_name
            },  # 'Are you sure you want to move this Planned Run to Torrent Server <b>%(destination)s</b> ?' % {'destination': destination},
            "infomsg": ugettext_lazy(
                "plan_transfer.messages.infomsg"
            ),  # 'Transferred planned run will no longer be available on this Torrent Server.',
            "processingmsg": _(
                "plan_transfer.messages.processingmsg"
            ),  # 'Planned run transfer in progress, please wait ...',
            "submit": ugettext_lazy("plan_transfer.action.submit"),
            "cancel": ugettext_lazy("global.action.modal.cancel"),
            "close": ugettext_lazy("global.action.modal.close"),
            "failmsg": ugettext_lazy("plan_transfer.messages.failmsg")
            % {
                "destination_server_name": destination_server_name
            },  # 'ERROR creating Planned Run on %(destination)s' % {'destination': destination},
        },
    }
    return render_to_response(
        "rundb/plan/modal_plan_transfer.html",
        context_instance=RequestContext(request, ctxd),
    )


def page_plan_samples_table_keys(is_barcoded, include_IR=False):
    barcoded = (
        ("barcodeId", PlanCSVcolumns.COLUMN_BARCODE),
        ("controlType", PlanCSVcolumns.COLUMN_SAMPLE_CONTROLTYPE),
    )
    default = (
        ("sampleName", PlanCSVcolumns.COLUMN_SAMPLE_NAME),
        ("sampleExternalId", PlanCSVcolumns.COLUMN_SAMPLE_ID),
        ("sampleDescription", PlanCSVcolumns.COLUMN_SAMPLE_DESCRIPTION),
        ("nucleotideType", PlanCSVcolumns.COLUMN_NUCLEOTIDE_TYPE),
        ("reference", PlanCSVcolumns.COLUMN_REF),
        ("targetRegionBedFile", PlanCSVcolumns.COLUMN_TARGET_BED),
        ("hotSpotRegionBedFile", PlanCSVcolumns.COLUMN_HOTSPOT_BED),
    )
    non_barcoded = (
        ("tubeLabel", PlanCSVcolumns.COLUMN_SAMPLE_TUBE_LABEL),
        ("chipBarcode", PlanCSVcolumns.COLUMN_CHIP_BARCODE),
    )
    ir = (
        ("ircancerType", "Cancer Type"),
        ("ircellularityPct", "Cellularity %"),
        ("irbiopsyDays", "Biopsy Days"),
        ("ircellNum", "Cell Number"),
        ("ircoupleID", "Couple ID"),
        ("irembryoID", "Embryo ID"),
        ("irWorkflow", "IR Workflow"),
        ("irRelationRole", "IR Relation"),
        ("irGender", "IR Gender"),
        ("irPopulation", "IR Population"),
        ("irBacterialMarkerType", "Bacterial Marker Type"),
        ("irWitness", "Witness"),
        ("irSetID", "IR Set ID"),
    )

    if is_barcoded:
        keys = barcoded + default
    else:
        keys = default + non_barcoded

    if include_IR:
        keys += ir
    return keys


def page_plan_samples_table_alternate_keys():
    keys = (("nucleotideType", "nucleotideType (DNA/RNA)"),)
    return keys

def page_plan_save_samples_table(request):
    """ request from editing Plan page to save samples table from js to csv """
    response = http.HttpResponse(mimetype="text/csv")
    response["Content-Disposition"] = "attachment; filename=plan_samples_%s.csv" % str(
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )
    writer = csv.writer(response)

    plan_csv_version = get_plan_csv_version()
    writer.writerow(plan_csv_version)

    table = json.loads(request.POST.get("samplesTable"))
    has_IR = "irWorkflow" in table[0] and "irSetID" in table[0]
    is_barcoded = "barcodeId" in table[0]
    keys, header = zip(*page_plan_samples_table_keys(is_barcoded, has_IR))

    writer.writerow(header)
    for row in table:
        values = [getcsv_rowData(row, key) for key in keys ]
        writer.writerow(values)
    return response

def getcsv_rowData(row, key):
    multiWorkflowString = None
    if key == 'irWorkflow':
        if row.get('irMultipleWorkflowSelected'):
            multiWorkflowSelectedObj = row.get('irMultipleWorkflowSelected', [])
            if multiWorkflowSelectedObj and isinstance(multiWorkflowSelectedObj[0], dict):
                multiWorkflowSelectedList = [obj.get('workflow') for obj in multiWorkflowSelectedObj]
                multiWorkflowString = ", ".join(multiWorkflowSelectedList)

    if multiWorkflowString:
        return multiWorkflowString
    else:
        return row[key]

def page_plan_load_samples_table(request):
    """ load and validate samples csv file """
    csv_file = request.FILES["csv_file"]
    has_IR = request.POST.get("irSelected") == "1"
    barcodeSet = request.POST.get("barcodeSet")
    default_reference = request.POST.get("default_reference")
    default_targetBedFile = request.POST.get("default_targetBedFile")
    default_hotSpotBedFile = request.POST.get("default_hotSpotBedFile")
    runType = request.POST.get("runType_name")
    applicationGroup = request.POST.get("applicationGroupName")

    bedFileDict = dict_bed_hotspot()

    keys = (
        page_plan_samples_table_keys(barcodeSet, has_IR)
        + page_plan_samples_table_alternate_keys()
    )
    ret = {
        "same_ref_and_bedfiles": True,
        "samplesTable": [],
        "ordered_keys": zip(*keys)[0],
    }

    destination = tempfile.NamedTemporaryFile(delete=False)
    for chunk in csv_file.chunks():
        destination.write(chunk)
    destination.close()

    non_ascii_files = []
    # open read bytes and detect if file contains non-ascii characters
    with open(destination.name, "rU") as _tmp:
        if not is_ascii(_tmp.read()):  # if not ascii
            non_ascii_files.append(csv_file.name)  # add uploaded file name

    if len(non_ascii_files) > 0:
        os.unlink(destination.name)
        # "Only ASCII characters are supported. The following files contain non-ASCII characters: %(files)s."
        _error_msg = validation.format(
            ugettext(
                "workflow.step.sample.messages.load_samples_table.file_contains_non_ascii_characters"
            ),
            {"files": SeparatedValuesBuilder().build(non_ascii_files)},
        )
        return http.HttpResponseBadRequest(_error_msg)

    try:
        f = open(destination.name, "rU")

        # validate CSV version
        csv_version_header = get_plan_csv_version()[0]
        csv_version_row = csv.reader(f).next()
        errorMsg, isToSkipRow, abortFile = utils.validate_csv_template_version(
            headerName=csv_version_header, isPlanCSV=True, firstRow=csv_version_row
        )  # ignore translation, override error message using abortFile variable
        if abortFile:
            error = ugettext_lazy(
                "workflow.step.sample.messages.load_samples_table.invalidcsvversion"
            )  # "CSV Version is missing or not supported. Please use Save Samples Table to download latest CSV file format"
            return http.HttpResponseBadRequest(error)

        reader = csv.DictReader(f)
        error = ""
        for n, row in enumerate(reader):
            checkMultiWorkflow = row.get('IR Workflow', '')
            workflowSelectedList = []
            if checkMultiWorkflow:
                workflowSelectedList = checkMultiWorkflow.split(',')
                if len(workflowSelectedList) > 1:
                    row['IR Workflow'] = workflowSelectedList[0]
            processed_row = dict([(k, row[v]) for k, v in keys if v in row])
            if not processed_row:
                continue

            ret["samplesTable"].append(processed_row)
            if len(workflowSelectedList) > 1:
                workflowSelectedList = [workflow.strip() for workflow in workflowSelectedList]
                ret["samplesTable"][n]["irMultipleWorkflowSelected"] = workflowSelectedList

            # validation
            row_errors = []
            for key, value in list(processed_row.items()):
                if key == 'irMultipleWorkflowSelected':
                    continue
                if barcodeSet:
                    if (len(value) - len(value.lstrip())) or (
                        len(value) - len(value.rstrip())
                    ):
                        logger.warning(
                            "The BarcodeName (%s) of BarcodeSetName(%s) contains Leading/Trailing spaces and got trimmed."
                            % (value, barcodeSet)
                        )
                    value = value.strip()
                    # barcoded
                    if key == "barcodeId" and barcodeSet not in dnaBarcode.objects.filter(
                        id_str=value
                    ).values_list(
                        "name", flat=True
                    ):
                        row_errors.append(
                            validation.invalid_invalid_value_related_value(
                                PlanCSVcolumns.COLUMN_BARCODE,
                                value,
                                ugettext_lazy(
                                    "workflow.step.kits.fields.barcodeId.label"
                                ),
                                barcodeSet,
                            )
                        )

                else:
                    # non-barcoded
                    if key == "chipBarcode" and value:
                        row_errors.extend(
                            plan_validator.validate_chipBarcode(
                                value, field_label=PlanCSVcolumns.COLUMN_CHIP_BARCODE
                            )
                        )
                    if key == "tubeLabel" and value:
                        row_errors.extend(
                            plan_validator.validate_sample_tube_label(
                                value,
                                field_label=PlanCSVcolumns.COLUMN_SAMPLE_TUBE_LABEL,
                            )
                        )

                if key == "sampleName" and value:
                    row_errors.extend(
                        plan_validator.validate_sample_name(
                            value, field_label=PlanCSVcolumns.COLUMN_SAMPLE_NAME
                        )
                    )
                if key == "sampleExternalId" and value:
                    row_errors.extend(
                        plan_validator.validate_sample_id(
                            value, field_label=PlanCSVcolumns.COLUMN_SAMPLE_ID
                        )
                    )
                if key == "nucleotideType" and value:
                    nuctype_err, processed_row[
                        key
                    ] = plan_validator.validate_sample_nucleotideType(
                        value,
                        runType,
                        applicationGroup,
                        field_label=PlanCSVcolumns.COLUMN_NUCLEOTIDE_TYPE,
                    )
                    row_errors.extend(nuctype_err)

                if key == "reference" and value:
                    if value.strip().lower() == "none":
                        processed_row[key] = ""
                        continue
                    ref_err, processed_row[key] = plan_validator.validate_reference(
                        value,
                        field_label=PlanCSVcolumns.COLUMN_REF,
                        runType=runType,
                        applicationGroupName=applicationGroup,
                        application_label=ScientificApplication.verbose_name,
                    )
                    row_errors.extend(ref_err)
                    if default_reference != processed_row[key]:
                        ret["same_ref_and_bedfiles"] = False

                if key == "targetRegionBedFile" and value:
                    if value.strip().lower() == "none":
                        processed_row[key] = ""
                        continue
                    inputRef = processed_row.get("reference")
                    if not inputRef:
                        row_errors.append(
                            validation.invalid_required_related_value(
                                PlanCSVcolumns.COLUMN_REF,
                                PlanCSVcolumns.COLUMN_TARGET_BED,
                                value,
                            )
                        )  # 'Invalid/Missing reference for target regions BED file %s' % value
                    else:
                        validated = get_bedFile_for_reference(
                            value, inputRef, hotspot=False
                        )
                        if validated:
                            processed_row[key] = validated
                            if default_targetBedFile != processed_row[key]:
                                ret["same_ref_and_bedfiles"] = False
                        else:
                            row_errors.append(
                                validation.invalid_not_found_error(
                                    PlanCSVcolumns.COLUMN_TARGET_BED, value
                                )
                            )  # 'Target regions BED file not found for %s' % value

                if key == "hotSpotRegionBedFile" and value:
                    if value.strip().lower() == "none":
                        processed_row[key] = ""
                        continue
                    inputRef = processed_row.get("reference")
                    if not inputRef:
                        row_errors.append(
                            validation.invalid_required_related_value(
                                PlanCSVcolumns.COLUMN_REF,
                                PlanCSVcolumns.COLUMN_HOTSPOT_BED,
                                value,
                            )
                        )  # 'Invalid/Missing reference for hotspot regions BED file %s' % value
                    else:
                        validated = get_bedFile_for_reference(
                            value, inputRef, hotspot=True
                        )
                        if validated:
                            processed_row[key] = validated
                            if default_hotSpotBedFile != processed_row[key]:
                                ret["same_ref_and_bedfiles"] = False
                        else:
                            row_errors.append(
                                validation.invalid_not_found_error(
                                    PlanCSVcolumns.COLUMN_HOTSPOT_BED, value
                                )
                            )  # 'Hotspot regions BED file not found for %s' % value

                if key == "controlType" and value:
                    controltype_err, processed_row[
                        key
                    ] = plan_validator.validate_sampleControlType(
                        value, PlanCSVcolumns.COLUMN_SAMPLE_CONTROLTYPE
                    )
                    row_errors.extend(controltype_err)

            if row_errors:
                error += validation.row_errors(
                    n + 1, row_errors, rowErrors_separator=" ", ending="<br/>"
                )  # 'Error in row %i: %s<br>' % (n+1, ' '.join(row_errors))

            if (
                (
                    PlanCSVcolumns.COLUMN_REF in row
                    and row[PlanCSVcolumns.COLUMN_REF].strip() != default_reference
                )
                or (
                    PlanCSVcolumns.COLUMN_TARGET_BED in row
                    and row[PlanCSVcolumns.COLUMN_TARGET_BED].strip()
                    != default_targetBedFile
                )
                or (
                    PlanCSVcolumns.COLUMN_HOTSPOT_BED in row
                    and row[PlanCSVcolumns.COLUMN_HOTSPOT_BED].strip()
                    != default_hotSpotBedFile
                )
            ):
                ret["same_ref_and_bedfiles"] = False

        f.close()  # now close and remove the temp file
        os.unlink(destination.name)

        if len(ret["samplesTable"]) == 0:
            return http.HttpResponseBadRequest(
                validation.format(
                    ugettext_lazy(
                        "workflow.step.sample.messages.load_samples_table.unabletoparse"
                    ),
                    {"fileName": csv_file.name},
                )
            )  # 'Error: No rows could be parsed from %s' % csv_file.name
        elif error:
            return http.HttpResponseBadRequest(error)
        else:
            return http.HttpResponse(
                json.dumps(ret, cls=LazyJSONEncoder), mimetype="text/html"
            )

    except Exception as err:
        logger.error(format_exc())
        return http.HttpResponseServerError(repr(err))


def plan_template_export(request, templateId):
    """
    Return csv file for Template export
    """
    filename = "exported_template_%s" % str(
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )
    response = http.HttpResponse(mimetype="text/csv")
    response["Content-Disposition"] = "attachment; filename=%s.csv" % filename

    plan_csv_version = get_plan_csv_version()
    data = get_template_data_for_export(templateId)
    writer = csv.writer(response)
    writer.writerow(plan_csv_version)
    for row in data:
        writer.writerow(row)

    return response

def parse_tec_params(dtp):
    try:
        return float(dtp)
    except ValueError:
        return None

def get_template_meta(row):
    return {
        "fromTemplate": row.get("Template name (required)"),
        "fromTemplateCategories": row.get("Categories"),
        "fromTemplateChipType": row.get("Chip type (required)"),
        "fromTemplateSequenceKitname": row.get("Sequence kit name"),
        "fromTemplateSource": "ION"
        }

def convert_tec_params_to_float(row):
    ct = 'Chip Heater (Do not change)'
    cts = 'Chip Slope (Do not change)'
    ctmt = 'Chip Min Threshold (Do not change)'

    mt = 'Manifold Heater (Do not change)'
    mts = 'Manifold Slope (Do not change)'
    mtmt = 'Manifold Min Threshold (Do not change)'
    if ct in row or mt in row:
        if ct in row:
            row[ct] = parse_tec_params(row[ct])
        if cts in row:
            row[cts] = parse_tec_params(row[cts])
        if ctmt in row:
            row[ctmt] = parse_tec_params(row[ctmt])
        if mt in row:
            row[mt] = parse_tec_params(row[mt])
        if mts in row:
            row[mts] = parse_tec_params(row[mts])
        if mtmt in row:
            row[mtmt] = parse_tec_params(row[mtmt])

    return row

def plan_template_import(request):
    def _get_kit_name(value, kitTypes):
        kit = KitInfo.objects.filter(isActive=True, kitType__in=kitTypes).filter(
            Q(name=value) | Q(description=value)
        )
        return kit[0].name if kit else value

    csv_file = request.FILES["csv_file"]
    destination = tempfile.NamedTemporaryFile(delete=False)
    for chunk in csv_file.chunks():
        destination.write(chunk)
    destination.close()

    non_ascii_files = []
    # open read bytes and detect if file contains non-ascii characters
    with open(destination.name, "rU") as _tmp:
        if not is_ascii(_tmp.read()):  # if not ascii
            non_ascii_files.append(csv_file.name)  # add uploaded file name

    if len(non_ascii_files) > 0:
        os.unlink(destination.name)
        # "Only ASCII characters are supported. The following files contain non-ASCII characters: %(files)s."
        _error_msg = validation.format(
            ugettext(
                "template.messages.import_plan_template.file_contains_non_ascii_characters"
            ),
            {"files": SeparatedValuesBuilder().build(non_ascii_files)},
        )
        return http.HttpResponseBadRequest(_error_msg)

    try:
        f = open(destination.name, "rU")
        # validate CSV version
        csv_version_header = get_plan_csv_version()[0]
        csv_version_row = csv.reader(f).next()
        errorMsg, isToSkipRow, abortFile = utils.validate_csv_template_version(
            headerName=csv_version_header, isPlanCSV=True, firstRow=csv_version_row
        )  # ignore translation, override error message using abortFile variable
        if abortFile:
            error = ugettext_lazy(
                "template.messages.import_plan_template.invalidcsvversion"
            )  # "CSV Version is missing or not supported. Please download latest CSV file format"
            return http.HttpResponseBadRequest(error)

        csv_version = StrictVersion(str(float(csv_version_row[1])))
        if csv_version >= "2.2":
            # process column-based file
            csv_reader = csv.reader(f)
            reader = _get_reader_for_v2(csv_reader)
        else:
            # process row-based file
            reader = csv.DictReader(f)

        errors = {}
        warnings = {}
        plans = []
        for index, row in enumerate(reader):
            # skip blank rows
            if not any(v.strip() for v in list(row.values())):
                continue
            row = convert_tec_params_to_float(row)
            planDict = {}
            planDict["isNewPlan"] = True  # this will trigger validation
            planDict["isReusable"] = True
            planDict["username"] = request.user.username
            planDict["origin"] = "csv"

            custom_args = toBoolean(row.get(PlanCSVcolumns.CUSTOM_ARGS, False))
            planCsvKeys = export_template_keys(custom_args)
            for field, csvKey in list(planCsvKeys.items()):
                if csvKey in row:
                    if isinstance(row[csvKey], str):
                        value = row[csvKey].strip() if row[csvKey] else ""
                    else:
                        value = row[csvKey]
                    if csvKey == PlanCSVcolumns.TEMPLATE_NAME:
                        planDict["planName"] = value.replace(" ", "_")
                    elif csvKey == PlanCSVcolumns.RUNTYPE:
                        runType = RunType.objects.filter(
                            Q(runType=value)
                            | Q(alternate_name=value)
                            | Q(description=value)
                        )
                        if runType:
                            value = runType[0].runType
                    elif (
                        csvKey == PlanCSVcolumns.COLUMN_LIBRARY_READ_LENGTH
                        and not value
                    ):
                        value = 0
                    elif (
                        csvKey == PlanCSVcolumns.FLOW_ORDER
                        and value.lower() == "default"
                    ):
                        value = ""
                    elif csvKey == PlanCSVcolumns.COLUMN_CHIP_TYPE and value:
                        chip = Chip.objects.filter(isActive=True).filter(
                            Q(name=value) | Q(description=value)
                        )
                        if chip:
                            value = chip[0].name
                    elif csvKey == PlanCSVcolumns.COLUMN_SAMPLE_PREP_KIT:
                        value = _get_kit_name(value, ["SamplePrepKit"])
                    elif csvKey == PlanCSVcolumns.COLUMN_LIBRARY_KIT:
                        value = _get_kit_name(value, ["LibraryKit", "LibraryPrepKit"])
                    elif csvKey == PlanCSVcolumns.COLUMN_TEMPLATING_KIT:
                        value = _get_kit_name(
                            value, ["TemplatingKit", "IonChefPrepKit"]
                        )
                    elif csvKey == PlanCSVcolumns.COLUMN_SEQ_KIT:
                        value = _get_kit_name(value, ["SequencingKit"])
                    elif csvKey == PlanCSVcolumns.COLUMN_CONTROL_SEQ_KIT:
                        value = _get_kit_name(value, ["ControlSequenceKit"])
                    elif csvKey == PlanCSVcolumns.COLUMN_PLUGINS and value:
                        selectedPlugins = {}
                        plugins = value.split(";")
                        for plugin in Plugin.objects.filter(
                            name__in=plugins, active=True
                        ):
                            plugin_config = row.get(PlanCSVcolumns.COLUMN_PLUGIN_CONFIG % plugin.name, {})
                            if plugin_config:
                                try:
                                    plugin_config = json.loads(plugin_config)
                                except:
                                    warnings.setdefault("%d" % (index + 1), {})[
                                        PlanCSVcolumns.COLUMN_PLUGIN_CONFIG % plugin.name
                                    ] = validation.invalid_invalid_value(
                                        PlanCSVcolumns.COLUMN_PLUGIN_CONFIG % plugin.name,
                                        plugin_config
                                    )
                                    plugin_config = {}

                            selectedPlugins[plugin.name] = {
                                "id": plugin.id,
                                "name": plugin.name,
                                "version": plugin.version,
                                "userInput": plugin_config,
                            }
                        value = selectedPlugins
                        # add warning if missing plugins
                        missing = [p for p in plugins if p and p not in selectedPlugins]
                        if len(missing) > 0:
                            warnings.setdefault("%d" % (index + 1), {})[
                                PlanCSVcolumns.COLUMN_PLUGINS
                            ] = validation.invalid_not_found_error(
                                PlanCSVcolumns.COLUMN_PLUGINS, missing
                            )  # 'Plugin(s) not found: %s' % ', '.join(missing)
                    elif csvKey == PlanCSVcolumns.COLUMN_PROJECTS:
                        value = value.split(";")
                    elif csvKey == PlanCSVcolumns.COLUMN_LIMS_DATA:
                        value = {"LIMS": [value]}
                        value.update(get_template_meta(row))
                    elif csvKey == PlanCSVcolumns.CUSTOM_ARGS:
                        value = custom_args

                    planDict[field] = value

            plans.append((planDict, planCsvKeys))

        f.close()
        os.unlink(destination.name)

        if len(plans) == 0:
            return http.HttpResponseBadRequest(
                validation.format(
                    ugettext_lazy(
                        "template.messages.import_plan_template.unabletoparse"
                    ),
                    {"fileName": csv_file.name},
                )
            )  # 'Error: No rows could be parsed from %s' % csv_file.name)

        # now use PlannedExperiment API resource to validate and create Template
        res = PlannedExperimentResource()
        validated = []
        for index, planTuple in enumerate(plans):
            bundle = res.build_bundle(data=planTuple[0])
            planCsvKeys = planTuple[1]
            try:
                res.is_valid(res.full_hydrate(bundle))
            except SDKValidationError as err:
                errors["status"] = "failed"
                errors["status_msg"] = ugettext_lazy(
                    "plan_template_import.messages.error.validationfailed"
                )  # 'Template import failed validation.'
                sdkErrors = err.errors()

                #
                # Translate from the SDK attribute/property specific to the Plan CSV Column
                #
                invalidErrors = {}
                for key, sdkError in list(sdkErrors.items()):
                    if key in planCsvKeys:
                        planCsvColumnName = planCsvKeys[key]
                        if isinstance(sdkError, basestring):
                            invalidErrors[planCsvColumnName] = sdkError.replace(
                                key, planCsvColumnName
                            )
                        elif isinstance(sdkError, list):
                            for x in sdkError:
                                invalidErrors[planCsvColumnName] = x.replace(
                                    key, planCsvColumnName
                                )
                        else:
                            invalidErrors[planCsvColumnName] = sdkError

                errors.setdefault("msg", {})["%d" % (index + 1)] = invalidErrors
            else:
                validated.append(bundle)

        if errors:
            return http.HttpResponse(json.dumps(errors, cls=LazyJSONEncoder))
        else:
            for index, bundle in enumerate(validated):
                warning_row_key = "%d" % (index + 1)

                # add warning if BED files are missing
                targetRegionBedFile = bundle.data.get("targetRegionBedFile")
                if targetRegionBedFile and not os.path.exists(targetRegionBedFile):
                    warnings.setdefault(warning_row_key, {})[
                        PlanCSVcolumns.COLUMN_TARGET_BED
                    ] = validation.invalid_not_found_error(
                        PlanCSVcolumns.COLUMN_TARGET_BED, targetRegionBedFile
                    )
                    bundle.data["targetRegionBedFile"] = bundle.data["bedfile"] = ""

                hotSpotRegionBedFile = bundle.data.get("hotSpotRegionBedFile")
                if hotSpotRegionBedFile and not os.path.exists(hotSpotRegionBedFile):
                    warnings.setdefault(warning_row_key, {})[
                        PlanCSVcolumns.COLUMN_HOTSPOT_BED
                    ] = validation.invalid_not_found_error(
                        PlanCSVcolumns.COLUMN_HOTSPOT_BED, hotSpotRegionBedFile
                    )
                    bundle.data["hotSpotRegionBedFile"] = bundle.data["regionfile"] = ""

                fusionsTargetRegionBedFile = bundle.data.get(
                    "mixedTypeRNA_targetRegionBedFile"
                )
                if fusionsTargetRegionBedFile and not os.path.exists(
                    fusionsTargetRegionBedFile
                ):
                    warnings.setdefault(warning_row_key, {})[
                        PlanCSVcolumns.FUSIONS_TARGET_BED
                    ] = validation.invalid_not_found_error(
                        PlanCSVcolumns.FUSIONS_TARGET_BED, fusionsTargetRegionBedFile
                    )
                    bundle.data["mixedTypeRNA_targetRegionBedFile"] = ""

                if warning_row_key in warnings:
                    metaData = bundle.data.get("metaData") or {}
                    metaData["warning"] = warnings[warning_row_key]
                    bundle.data["metaData"] = metaData

                # create plan
                res.obj_create(bundle)

            if warnings:
                return http.HttpResponse(
                    json.dumps(
                        {
                            "status": "warning",
                            "status_msg": ugettext_lazy(
                                "plan_template_import.messages.success.withwarning"
                            ),  # "Template created with warnings."
                            "msg": warnings,
                        },
                        cls=LazyJSONEncoder,
                    )
                )
            else:
                return http.HttpResponse()

    except Exception as err:
        logger.error(format_exc())
        return http.HttpResponseServerError(repr(err))


def upload_and_install_files(request):
    " Install preloaded Ion reference and BED files"
    from iondb.rundb.configure.genomes import get_references, new_reference_genome
    from iondb.rundb.tasks import install_BED_files, release_tasklock

    try:
        reference_json = get_references()
    except Exception:
        reference_json = []

    def get_ref_info(reference):
        ret = [d for d in reference_json if d["meta"]["short_name"] == reference]
        return ret[0] if ret else {}

    if request.method == "GET":
        # check if requested files are available for install from Ion source
        # check if requested files are being processed already
        ctxd = {}
        ctxd["files"] = {"references": [], "bedfiles": [], "not_found": []}

        for reference in request.GET.getlist("references[]", []):
            if get_ref_info(reference):
                ctxd["files"]["references"].append(reference)
            else:
                ctxd["files"]["not_found"].append(reference)

        for bedfile in request.GET.getlist("bedfiles[]", []):
            reference = bedfile.split("/")[1]
            info = get_ref_info(reference)
            bedfile_available = False

            if info and info.get("bedfiles"):
                for available_bedfile in info["bedfiles"]:
                    if os.path.basename(bedfile) == os.path.basename(
                        available_bedfile["source"]
                    ):
                        ctxd["files"]["bedfiles"].append(bedfile)

            if not bedfile_available:
                ctxd["files"]["not_found"].append(bedfile)

        ctxd["available"] = (
            len(ctxd["files"]["references"]) > 0 or len(ctxd["files"]["bedfiles"]) > 0
        )
        ctxd["files_json"] = json.dumps(ctxd["files"], cls=LazyJSONEncoder)

        return render_to_response(
            "rundb/plan/modal_upload_and_install_files.html",
            context_instance=RequestContext(request, ctxd),
        )

    elif request.method == "POST":

        application = request.POST.get("application")
        parent_lock = TaskLock(application)
        if not parent_lock.lock():
            return http.HttpResponseServerError(
                "Previous install is in progress for %s" % application
            )  # TODO: i18n
        else:
            logger.debug(
                "upload_and_install_files: created install files lock for %s"
                % application
            )

        bedfiles_to_install = {}
        for bedfile in request.POST.getlist("bedfiles[]"):
            reference = bedfile.split("/")[1]
            bedfiles_to_install.setdefault(reference, []).append(
                os.path.basename(bedfile)
            )

        started = {"references": [], "bedfiles": [], "lock_ids": []}
        error = []

        # process references and any BED files that are dependent on them
        for reference in request.POST.getlist("references[]"):
            requested_bedfiles = bedfiles_to_install.pop(reference, None)

            info = get_ref_info(reference)
            if not info:
                error.append(
                    "Unable to retrieve reference information for %s" % reference
                )  # TODO: i18n
                continue

            tasklock_id = application + "_" + reference
            applock = TaskLock(tasklock_id)
            if not applock.lock():
                error.append(
                    "Previous install is in progress for %s %s"
                    % (application, reference)
                )  # TODO: i18n
                continue

            logger.debug(
                "upload_and_install_files: new reference install: %s" % reference
            )
            unlock_task = release_tasklock.subtask(
                (tasklock_id, application), immutable=True
            )
            callback = unlock_task
            bedfileList = []
            if requested_bedfiles:
                for available_bedfile in info["bedfiles"]:
                    if (
                        os.path.basename(available_bedfile["source"])
                        in requested_bedfiles
                    ):
                        bedfileList.append(available_bedfile)

                if bedfileList:
                    # task will install BED files after reference is done
                    callback = install_BED_files.subtask(
                        (bedfileList, request.user.username, unlock_task),
                        immutable=True,
                    )
                    logger.debug(
                        "upload_and_install_files: new bedfiles install for %s reference: %s"
                        % (reference, ",".join([b["source"] for b in bedfileList]))
                    )

            try:
                # create reference and start install
                new_reference_genome(
                    info["meta"],
                    info["url"],
                    reference_mask_filename=info.get("reference_mask"),
                    callback_task=callback,
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                error.append(repr(e))
                applock.unlock()
            else:
                started["references"].append(reference)
                started["lock_ids"].append(tasklock_id)
                if bedfileList:
                    started["bedfiles"].extend(
                        [os.path.basename(bed["source"]) for bed in bedfileList]
                    )

        # process BED files that have references already installed
        bedfileList = []
        for reference, requested_bedfiles in list(bedfiles_to_install.items()):
            info = get_ref_info(reference)
            for available_bedfile in info["bedfiles"]:
                if os.path.basename(available_bedfile["source"]) in requested_bedfiles:
                    bedfileList.append(available_bedfile)

        if bedfileList:
            tasklock_id = application + "_bedfiles"
            applock = TaskLock(tasklock_id)
            if not applock.lock():
                error.append(
                    "Previous install is in progress for %s BED files" % application,
                    reference,
                )  # TODO: i18n
            else:
                unlock_task = release_tasklock.subtask(
                    (tasklock_id, application), immutable=True
                )
                logger.debug(
                    "upload_and_install_files: new bedfiles install: %s"
                    % ", ".join([b["source"] for b in bedfileList])
                )
                try:
                    install_BED_files.delay(
                        bedfileList, request.user.username, callback=unlock_task
                    )
                except Exception as e:
                    logger.error(traceback.format_exc())
                    error.append(repr(e))
                    applock.unlock()
                else:
                    started["lock_ids"].append(tasklock_id)
                    started["bedfiles"].extend(
                        [os.path.basename(bed["source"]) for bed in bedfileList]
                    )

        # update parent lock
        if started["lock_ids"]:
            logger.debug(
                "Started reference and BED files install parent lock: %s, child locks: %s"
                % (application, ",".join(started["lock_ids"]))
            )
            parent_lock.update(started["lock_ids"])
        else:
            parent_lock.unlock()

        # show message
        msg = []
        if started["references"]:
            msg.append(
                "Started download and install Genome Reference(s): %s"
                % ", ".join(started["references"])
            )  # TODO: i18n
        if started["bedfiles"]:
            msg.append(
                "Started download and install BED File(s): %s"
                % ", ".join(started["bedfiles"])
            )  # TODO: i18n

        if msg:
            Message.info("<br>".join(msg), tags="install_" + application)
            logger.info(" ".join(msg))

        return HttpResponse(
            json.dumps(
                {"error": "<br>".join(error), "started": started}, cls=LazyJSONEncoder
            ),
            mimetype="application/json",
        )
