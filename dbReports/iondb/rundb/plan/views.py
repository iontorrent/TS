# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django import http
from datetime import datetime
from django.contrib.auth.decorators import login_required
from django.template import RequestContext
from django.shortcuts import render_to_response, get_object_or_404, \
    get_list_or_404, render
from django.conf import settings
from django.db import transaction
from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic.detail import DetailView
from django.core.exceptions import ValidationError

from iondb.rundb.api import PlannedExperimentResource
from iondb.rundb.models import PlannedExperiment, RunType, ApplProduct, SharedServer, \
    ReferenceGenome, Content, KitInfo, VariantFrequencies, dnaBarcode, \
    LibraryKey, ThreePrimeadapter, Chip, QCType, Project, Plugin, \
    PlannedExperimentQC, Sample, GlobalConfig, Message, Experiment, Results, EventLog, common_CV
from django.db.models import Q

from traceback import format_exc
import json
from urllib import unquote_plus
import simplejson
import uuid

import logging
from django.core import serializers
from django.core.serializers.json import DjangoJSONEncoder
from iondb.rundb.api import PlannedExperimentResource, RunTypeResource, \
    dnaBarcodeResource, ChipResource, ApplicationGroupResource, SampleGroupType_CVResource

from django.core.urlresolvers import reverse

from iondb.utils import toBoolean
from iondb.rundb.plan.views_helper import get_projects, dict_bed_hotspot, isOCP_enabled, is_operation_supported, getChipDisplayedNamePrimaryPrefix, \
    getChipDisplayedNameSecondaryPrefix, getChipDisplayedVersion, get_template_categories

from iondb.utils.validation import is_valid_chars, is_valid_leading_chars, is_valid_length
from iondb.utils.utils import convert
from iondb.utils.prepopulated_planning import apply_prepopulated_values_to_step_helper

from iondb.rundb.plan.plan_csv_writer import get_template_data_for_batch_planning, get_plan_csv_version, get_samples_data_for_batch_planning
from iondb.rundb.plan.plan_csv_writer import PlanCSVcolumns, get_template_data_for_export, export_template_keys
from iondb.rundb.plan.plan_csv_validator import validate_csv_plan, get_bedFile_for_reference
from iondb.rundb.plan import plan_validator
from iondb.utils import utils, toBoolean

import os
import string
import traceback
import tempfile
import csv
import io
import zipfile

from django.core.exceptions import ValidationError
from iondb.rundb.plan.page_plan.step_helper import StepHelper
from iondb.rundb.plan.page_plan.step_helper_types import StepHelperType
from iondb.rundb.plan.page_plan.step_helper_db_loader import StepHelperDbLoader
from iondb.rundb.plan.page_plan.step_helper_db_saver import StepHelperDbSaver
from iondb.rundb.plan.page_plan.step_names import StepNames

logger = logging.getLogger(__name__)

MAX_LENGTH_PLAN_NAME = 512
MAX_LENGTH_SAMPLE_NAME = 127
MAX_LENGTH_PROJECT_NAME = 64
MAX_LENGTH_NOTES = 1024
MAX_LENGTH_SAMPLE_TUBE_LABEL = 512

ERROR_MSG_INVALID_CHARS = " should contain only numbers, letters, spaces, and the following: . - _"
ERROR_MSG_INVALID_LENGTH = " length should be %s characters maximum. "
ERROR_MSG_INVALID_LEADING_CHARS = " should only start with numbers or letters. "


def get_ir_config(request):
    if request.is_ajax():
        try:
            iru_plugin = Plugin.objects.get(name__iexact='IonReporterUploader', active=True)
            account_id = request.POST.get('accountId')
            userconfigs = iru_plugin.config.get('userconfigs').get(request.user.username)
            if userconfigs:
                for config in userconfigs:
                    if config["id"] == account_id:
                        return HttpResponse(simplejson.dumps(config), content_type='application/javascript')
            else:
                return HttpResponse(simplejson.dumps(dict(error="Unable to find userconfigs in IRU plugin")), content_type='application/javascript')
        except:
            return HttpResponse(simplejson.dumps(dict(error="Could not find IRU Plugin")), content_type='application/javascript')
    return HttpResponse(simplejson.dumps(dict(error="Invalid view function call")), content_type='application/javascript')


def reset_page_plan_session(request):
    request.session.pop('plan_step_helper', None)
    return HttpResponse("Page Plan Session Object has been reset")


@login_required
def plan_templates(request):
    """
    plan template home page
    """
    ctxd = {
        'categories': get_template_categories()
    }
    return render_to_response("rundb/plan/plan_templates.html", context_instance=RequestContext(request, ctxd))


@login_required
def page_plan_edit_template(request, template_id):
    ''' edit a template with id template_id '''

    isSupported = is_operation_supported(template_id)
    if (not isSupported):
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForTemplatePlannedExperiment(template_id, StepHelperType.EDIT_TEMPLATE)
    ctxd = handle_step_request(request, 'Save_template', step_helper)
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_copy_template(request, template_id):
    ''' copy a template with id template_id '''
    isSupported = is_operation_supported(template_id)
    if (not isSupported):
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForTemplatePlannedExperiment(template_id, StepHelperType.COPY_TEMPLATE)
    ctxd = handle_step_request(request, 'Save_template', step_helper)
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_new_template(request, code=None):
    """ Create a new template from runtype code or from scratch """
    if code:
#        if (code == "8"):
#            isSupported = isOCP_enabled()
#            if (not isSupported):
#                return render_to_response("501.html")

        if (code == "9"):
            step_helper = StepHelperDbLoader().getStepHelperForRunType(run_type_id=_get_runtype_from_code("1").pk, applicationGroupName="PGx")
        elif (code == "11"):
            step_helper = StepHelperDbLoader().getStepHelperForRunType(run_type_id=_get_runtype_from_code("1").pk, applicationGroupName="HID")
        else:
            step_helper = StepHelperDbLoader().getStepHelperForRunType(_get_runtype_from_code(code).pk)
    else:
        step_helper = StepHelper()
    ctxd = handle_step_request(request, 'Ionreporter', step_helper)
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_new_template_by_sample(request, sampleset_id):
    """ Create a new template by sample """

    step_helper = StepHelperDbLoader().getStepHelperForNewTemplateBySample(_get_runtype_from_code(0).pk, sampleset_id)
    ctxd = handle_step_request(request, 'Ionreporter', step_helper)
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_new_plan(request, template_id):
    ''' create a new plan from a template with id template_id '''

    isSupported = is_operation_supported(template_id)
    if (not isSupported):
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForTemplatePlannedExperiment(template_id, StepHelperType.CREATE_NEW_PLAN)

    apply_prepopulated_values_to_step_helper(request, step_helper)

    ctxd = handle_step_request(request, 'Save_plan', step_helper)
    ctxd['step'].validationErrors.clear()  # Remove validation errors found during wizard initialization
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_new_plan_by_sample(request, template_id, sampleset_id):
    ''' create a new plan by sample from a template with id template_id '''

    if int(template_id) == int(0):
        # we are creating a new experiment based on the sample set
        return HttpResponseRedirect(reverse('page_plan_new_template_by_sample', args=(sampleset_id,)))

    isSupported = is_operation_supported(template_id)
    if (not isSupported):
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForTemplatePlannedExperiment(template_id,
                                                                                 StepHelperType.CREATE_NEW_PLAN_BY_SAMPLE,
                                                                                 sampleset_id=sampleset_id)
    apply_prepopulated_values_to_step_helper(request, step_helper)

    ctxd = handle_step_request(request, 'Barcode_by_sample', step_helper)
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_new_plan_from_code(request, code):
    ''' create a new plan from a runtype code '''

#    if (code == "8"):
#        isSupported = isOCP_enabled()
#        if (not isSupported):
#            return render_to_response("501.html")

    if (code == "9"):
        step_helper = StepHelperDbLoader().getStepHelperForRunType(run_type_id=_get_runtype_from_code("1").pk, step_helper_type=StepHelperType.CREATE_NEW_PLAN, applicationGroupName="PGx")
    elif (code == "11"):
        step_helper = StepHelperDbLoader().getStepHelperForRunType(run_type_id=_get_runtype_from_code("1").pk, step_helper_type=StepHelperType.CREATE_NEW_PLAN, applicationGroupName="HID")
    else:
        runType = _get_runtype_from_code(code)
        step_helper = StepHelperDbLoader().getStepHelperForRunType(runType.pk, StepHelperType.CREATE_NEW_PLAN)

    apply_prepopulated_values_to_step_helper(request, step_helper)
    ctxd = handle_step_request(request, 'Ionreporter', step_helper)
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_edit_plan_by_sample(request, plan_id):

    isSupported = is_operation_supported(plan_id)
    if (not isSupported):
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForPlanPlannedExperiment(plan_id, StepHelperType.EDIT_PLAN_BY_SAMPLE)
    ctxd = handle_step_request(request, 'Barcode_by_sample', step_helper)
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_edit_plan(request, plan_id):
    ''' edit plan with id plan_id '''
    # first get the plan and check if it's plan by sample set

    isSupported = is_operation_supported(plan_id)
    if (not isSupported):
        return render_to_response("501.html")

    plan = PlannedExperiment.objects.get(pk=plan_id)
    if plan.sampleSets.exists():
        return HttpResponseRedirect(reverse('page_plan_edit_plan_by_sample', args=(plan_id,)))

    step_helper = StepHelperDbLoader().getStepHelperForPlanPlannedExperiment(plan_id, StepHelperType.EDIT_PLAN)
    ctxd = handle_step_request(request, 'Save_plan', step_helper)
    ctxd['step'].validationErrors.clear()  # Remove validation errors found during wizard initialization
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_edit_run(request, exp_id):
    ''' retrieve and edit a plan for existing experiment '''
    try:
        plan = PlannedExperiment.objects.get(experiment=exp_id)

        isSupported = is_operation_supported(plan.id)
        if (not isSupported):
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
        plan.planStatus = 'run'
        plan.planExecuted = True
        plan.latestEAS = exp.get_EAS()
        plan.save()
        exp.plan = plan
        exp.save()

    step_helper = StepHelperDbLoader().getStepHelperForPlanPlannedExperiment(plan.id, StepHelperType.EDIT_RUN)
    request.session['return'] = request.META.get('HTTP_REFERER', '')
    ctxd = handle_step_request(request, 'Save_plan', step_helper)
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_copy_plan(request, plan_id):
    ''' copy plan with id plan_id '''

    isSupported = is_operation_supported(plan_id)
    if (not isSupported):
        return render_to_response("501.html")

    plan = PlannedExperiment.objects.get(pk=plan_id)
    if plan.sampleSets.exists():
        return HttpResponseRedirect(reverse('page_plan_copy_plan_by_sample', args=(plan_id,)))
    step_helper = StepHelperDbLoader().getStepHelperForPlanPlannedExperiment(plan_id, StepHelperType.COPY_PLAN)
    ctxd = handle_step_request(request, 'Save_plan', step_helper)
    ctxd['step'].validationErrors.clear()  # Remove validation errors as this the user is just now seeing the plan.
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_copy_plan_by_sample(request, plan_id):

    isSupported = is_operation_supported(plan_id)
    if (not isSupported):
        return render_to_response("501.html")

    step_helper = StepHelperDbLoader().getStepHelperForPlanPlannedExperiment(plan_id, StepHelperType.COPY_PLAN_BY_SAMPLE)
    ctxd = handle_step_request(request, 'Barcode_by_sample', step_helper)
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_export(request):
    ctxd = handle_step_request(request, 'Export')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_application(request):
    ctxd = handle_step_request(request, 'Application')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_save_sample(request):
    ctxd = handle_step_request(request, 'Save_plan_by_sample')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_kits(request):
    ctxd = handle_step_request(request, 'Kits')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_monitoring(request):
    # TODO: Remove Dead Monitoring Page Code
    ctxd = handle_step_request(request, 'Monitoring')
    # return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)
    return HttpResponseRedirect(reverse("page_plan_reference"))


@login_required
def page_plan_reference(request):
    ctxd = handle_step_request(request, 'Reference')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_plugins(request):
    ctxd = handle_step_request(request, 'Plugins')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_output(request):
    ctxd = handle_step_request(request, 'Output')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_ionreporter(request):
    ctxd = handle_step_request(request, 'Ionreporter')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_save_template(request):
    ctxd = handle_step_request(request, 'Save_template')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_save_template_by_sample(request):
    ctxd = handle_step_request(request, 'Save_template_by_sample')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_save_plan(request):
    ctxd = handle_step_request(request, 'Save_plan')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_by_sample_barcode(request):
    ctxd = handle_step_request(request, 'Barcode_by_sample')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_by_sample_save_plan(request):
    ctxd = handle_step_request(request, 'Save_plan_by_sample')
    return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


@login_required
def page_plan_save(request, exp_id=None):
    ''' you may only come here from the last plan/template step that has a save button. '''

    # update the step_helper with the latest data from the save page
    ctxd = handle_step_request(request, '')
    if 'session_error' in ctxd:
        return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)

    step_helper = ctxd['helper']

    # 20141005-BEWARE: the reference step object held in ctxd[helper] does not seem to be the same as the ones
    # created in StepHelper!!!

    # make sure the step helper is valid
    first_error_step = step_helper.validateAll()
    if not first_error_step:
        try:
            # save the step helper
            step_helper_db_saver = StepHelperDbSaver()
            planTemplate = step_helper_db_saver.save(step_helper, request.user.username)
            update_plan_log(step_helper_db_saver.getSavedPlansList(), step_helper, request.user.username)
            request.session["created_plan_pks"] = step_helper_db_saver.getSavedPlansList()
        except:
            step_helper_db_saver = None
            logger.exception(format_exc())
            Message.error("There was an error saving your plan/template.")

        # this plan is done, remove it from sessions
        request.session['plan_sessions'].pop(request.session['current_plan_key'])

        # return response if initiated from prepopulated session API
        if "post_planning_redirect_url" in request.session:
            response = HttpResponseRedirect(request.session['post_planning_redirect_url']+"?created_plans=" + json.dumps(request.session.get("created_plan_pks", "[]")))
            response['Created-Plans'] = json.dumps(request.session.get("created_plan_pks", "[]"))
            del request.session['post_planning_redirect_url']
            return response

        # or redirect based on context
        if step_helper.isTemplateBySample():
            return HttpResponseRedirect(reverse('page_plan_new_plan_by_sample', args=(planTemplate.pk, step_helper.steps['Ionreporter'].savedFields['sampleset_id'])))
        elif step_helper.isEditRun():
            if exp_id:
                return HttpResponseRedirect(reverse('report_analyze', kwargs={'exp_pk': exp_id, 'report_pk': 0}))
            else:
                return HttpResponseRedirect(request.session.pop('return') if request.session.get('return') else '/data')
        elif step_helper.isPlan():
            return HttpResponseRedirect('/plan/planned')
        else:
            return HttpResponseRedirect('/plan/plan_templates/#recently_created')
    else:
        # tell the context which step to go to
        ctxd['step'] = step_helper.steps[first_error_step]

        # go to that step
        return render_to_response(ctxd['step'].resourcePath, context_instance=ctxd)


def handle_step_request(request, next_step_name, step_helper=None):

    # find out which step we came from
    current_step_name = request.POST.get('stepName', None)
    step_helper = _update_plan_session(request, step_helper)

    if not step_helper:
        step_helper = StepHelper()
        ctxd = {
            'step': step_helper.steps.get(current_step_name) or step_helper.steps.values()[0],
            'session_error': 'This Planning session expired'
        }
        return RequestContext(request, ctxd)

    # logger.debug("views.handle_step_request() current_step_name=%s; next_step_name=%s" %(current_step_name, next_step_name))
    application_step_data = step_helper.steps['Application']
    applProduct = application_step_data.savedObjects['applProduct']

    # logger.debug("views.handle_step_request() applProduct=%s" %(applProduct))

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
            # logger.debug("views.handle_step_request() AFTER hasStepSections() next_step_name=%s" %(next_step_name))
        # else:
        #    logger.debug("views.handle_step_request() NO-OP next_step_name=%s" %(next_step_name))

        # logger.debug("EXIT views.handle_step_request() next_step_name=%s" %(next_step_name))

    ctxd = _create_context_from_session(request, next_step_name)
    return ctxd


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
        msg = 'Created Planned Run: %s (%s)' % (plan.planName, plan.pk)
        EventLog.objects.add_entry(plan, msg, username)

    if edited_plan_pk:
        plan = PlannedExperiment.objects.get(pk=edited_plan_pk)
        state = 'Run' if step_helper.isEditRun() else 'Planned Run'
        msg = 'Updated %s: %s (%s).' % (state, plan.planName, plan.pk)

        # log old/new values for changed fields
        changed = step_helper.getChangedFields()
        changed_msg = {}
        for key, values in changed.items():
            if key == 'pluginIds':
                # convert plugin ids to names
                plugins = step_helper.steps['Plugins'].prepopulatedFields['plugins']
                old = [plugin.name for plugin in plugins if values[0] and str(plugin.id) in values[0].replace(' ', '').split(',')]
                new = [plugin.name for plugin in plugins if values[1] and str(plugin.id) in values[1].replace(' ', '').split(',')]
                changed_msg['plugins'] = [', '.join(old), ', '.join(new)]
            elif key.startswith('plugin_config'):
                try:
                    old = json.loads(values[0])
                    new = json.loads(values[1])
                    if old != new:
                        changed_msg[key] = [json.dumps(old, indent=1), json.dumps(new, indent=1)]
                except:
                    changed_msg[key] = [values[0], values[1]]
            elif key == 'projects':
                # convert project ids to names
                projects = step_helper.steps['Output'].prepopulatedFields['projects']
                old = [project.name for project in projects if values[0] and (project.id in values[0] or str(project.id) in values[0])]
                new = [project.name for project in projects if values[1] and (project.id in values[1] or str(project.id) in values[1])]
                changed_msg[key] = [', '.join(old), ', '.join(new)]
            else:
                changed_msg[key] = [values[0], values[1]]

        if changed_msg:
            EventLog.objects.add_entry(plan, json.dumps(changed_msg), username)
            EventLog.objects.add_entry(plan, msg, username)


def toggle_template_favorite(request, template_id):
    '''
    toggle a template to set/unset the favorite tag with id template_id
    '''
    ctxd = {}

    plan_template = get_object_or_404(PlannedExperiment, pk=template_id)

    toggle_favorite = False if (plan_template.isFavorite) else True
    plan_template.isFavorite = toggle_favorite

    plan_template.save()

    ctx = RequestContext(request, ctxd)
    return render_to_response("rundb/plan/plan_templates.html", context_instance=ctx, mimetype="text/html")


def _get_runtype_from_code(code):
    codes = {
        '1': "AMPS",
        '2': "TARS",
        '3': "WGNM",
        '4': "RNA",
        '5': "AMPS_RNA",
        '6': "AMPS_EXOME",
        '7': "TARS_16S",
        '8': "AMPS_DNA_RNA",
        #'9': PGx - AMPS_DNA
        '10': "TAG_SEQUENCING",
        '11': "HID"
    }
    product_code = codes.get(code, "GENS")
    return RunType.objects.get(runType=product_code)


def _create_context_from_session(request, next_step_name):
    key = request.session['current_plan_key']
    step_helper = request.session['plan_sessions'][key]
    ctxd = {
        'plan_session_key': key,
        'helper': step_helper,
        'step': step_helper.steps.get(next_step_name)
    }
    return RequestContext(request, ctxd)


def _update_plan_session(request, step_helper):
    if 'plan_sessions' not in request.session:
        request.session['plan_sessions'] = {}
        # expire session after 3 hrs inactive
        request.session.set_expiry(10800)

    # find or generate unique plan session id
    key = request.POST.get('plan_session_key')
    if not key and step_helper:
        key = '%s-%s' % (step_helper.sh_type, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    request.session['current_plan_key'] = key

    if key not in request.session['plan_sessions']:
        request.session['plan_sessions'][key] = step_helper
    else:
        step_helper = request.session['plan_sessions'][ key ]

    return step_helper

@login_required
def planned(request):
    ctx = RequestContext(request)

    if "created_plan_pks" in request.session:
        ctx["created_plan_pks"] = request.session["created_plan_pks"]
        del request.session["created_plan_pks"]

    ctx["planshare"] = SharedServer.objects.filter(active=True)

    return render_to_response("rundb/plan/planned.html", context_instance=ctx)


@login_required
def plan_run_home(request):
    ctx = RequestContext(request)
    return render_to_response("rundb/plan/plan_run_home.html", context_instance=ctx)


@login_required
def delete_plan_template(request, pks=None):
    # TODO: See about pulling this out into a common methods
    pks = pks.split(',')
    _type = 'plannedexperiment'
    planTemplates = get_list_or_404(PlannedExperiment, pk__in=pks)
    _typeDescription = "Template" if planTemplates[0].isReusable is True else "Planned Run"
    actions = []
    for pk in pks:
        actions.append(reverse('api_dispatch_detail', kwargs={'resource_name': _type, 'api_name': 'v1', 'pk': int(pk)}))
    names = ', '.join([x.planName for x in planTemplates])
    ctx = RequestContext(request, {
        "id": pks[0], "name": names, "ids": json.dumps(pks), "names": names, "method": "DELETE", 'methodDescription': 'Delete', "readonly": False, 'type': _typeDescription, 'action': actions[0], 'actions': json.dumps(actions)
    })
    return render_to_response("rundb/common/modal_confirm_delete.html", context_instance=ctx)


# def  _handle_request_preProcess_failure():
# result = HttpResponse("error message here", content_type="text/plain")
# result.status_code = 417
# return result
def _get_runtype_json(request, runType):
    rtr = RunTypeResource()
    base_bundle = rtr.build_bundle(request=request)
    rt = rtr.obj_get(bundle=base_bundle, runType=runType)
    rtr_bundle = rtr.build_bundle(obj=rt, request=request)
    rt_json = rtr.serialize(None, rtr.full_dehydrate(rtr_bundle), 'application/json')
    return rt_json


def _get_dnabarcode_json(request, barcodeId):
    dnar = dnaBarcodeResource()
    base_bundle = dnar.build_bundle(request=request)
    dna = dnar.obj_get_list(bundle=base_bundle, name=barcodeId, request=request).order_by('index')
    dna_bundles = [dnar.build_bundle(obj=x, request=request) for x in dna]
    dna_json = dnar.serialize(None, [dnar.full_dehydrate(bundle) for bundle in dna_bundles], 'application/json')
    return dna_json


def _get_chiptype_json(request, chipType, plan):
    chipResource = ChipResource()
    chipResource_serialize_json = None
    if chipType:
        try:
            base_bundle = chipResource.build_bundle(request=request)
            try:
                chip = chipResource.obj_get(bundle=base_bundle, name=chipType)
            except Chip.DoesNotExist:
                chip = chipResource.obj_get(bundle=base_bundle, name=chipType[:3])

            chipResource_bundle = chipResource.build_bundle(obj=chip, request=request)
            chipResource_serialize_json = chipResource.serialize(None, chipResource.full_dehydrate(chipResource_bundle), 'application/json')
        except Chip.DoesNotExist:
            logger.error("views._get_chiptype_json() Plan.pk=%d; name=%s has invalid chip type=%s" % (plan.id, plan.planName, chipType))
            chipResource_bundle = None
            chipResource_serialize_json = json.dumps("INVALID")
    else:
        chipResource_bundle = None
        chipResource_serialize_json = json.dumps(None)
    return chipResource_serialize_json


def _get_stripped_chipType(chipType):
    '''
    some historical runs are found to have chipType is extra double quotes and/or backslash with quotes
    0047_ migration script should have fixed the extra double quotes problems. For defensive coding, double check here
    '''
    if chipType:
        return chipType.replace('\\', '').replace('"', '')
    else:
        return chipType


def _get_applicationGroup_json(request, applicationGroup_id):
    applicationGroupResource = ApplicationGroupResource()
    applicationGroupResource_serialize_json = None

    if applicationGroup_id:
        try:
            base_bundle = applicationGroupResource.build_bundle(request=request)

            applicationGroup = applicationGroupResource.obj_get(bundle=base_bundle, id=applicationGroup_id)

            applicationGroupResource_bundle = applicationGroupResource.build_bundle(obj=applicationGroup, request=request)
            applicationGroupResource_serialize_json = applicationGroupResource.serialize(None, applicationGroupResource.full_dehydrate(applicationGroupResource_bundle), 'application/json')
        except:
            applicationGroupResource_bundle = None
            applicationGroupResource_serialize_json = json.dumps(None)
    else:
        applicationGroupResource_bundle = None
        applicationGroupResource_serialize_json = json.dumps(None)
    return applicationGroupResource_serialize_json


def _get_sampleGrouping_json(request, sampleGroup_id):
    sampleGroupResource = SampleGroupType_CVResource()
    sampleGroupResource_serialize_json = None

    if sampleGroup_id:
        try:
            base_bundle = sampleGroupResource.build_bundle(request=request)

            sampleGroup = sampleGroupResource.obj_get(bundle=base_bundle, id=sampleGroup_id)

            sampleGroupResource_bundle = sampleGroupResource.build_bundle(obj=sampleGroup, request=request)
            sampleGroupResource_serialize_json = sampleGroupResource.serialize(None, sampleGroupResource.full_dehydrate(sampleGroupResource_bundle), 'application/json')
        except:
            sampleGroupResource_bundle = None
            sampleGroupResource_serialize_json = json.dumps(None)
    else:
        sampleGroupResource_bundle = None
        sampleGroupResource_serialize_json = json.dumps(None)
    return sampleGroupResource_serialize_json


class PlanDetailView(DetailView):
    model = PlannedExperiment
    template_name = 'rundb/plan/modal_plannedexperiment_detail.html'
    context_object_name = 'plan'

    def get_context_data(self, **kwargs):
        plan = self.object

        # can Review from either Report or Planned Runs pages
        report_pk = self.kwargs.get('report_pk')
        if report_pk:
            result = Results.objects.get(pk=report_pk)
            eas = result.eas
            state = 'Plan'
        else:
            eas = plan.latestEAS or plan.experiment.get_EAS()
            state = 'Template' if plan.isReusable else 'Planned Run'

        chipType = Chip.objects.filter(name=plan.get_chipType()).values_list('description', flat=True)

        # generate context for page
        context = super(PlanDetailView, self).get_context_data(**kwargs)
        context['extra_head'] = 'Report: %s' % result.resultsName if report_pk else ''
        context['state'] = state
        context['eas'] = eas
        context['runType'] = RunType.objects.filter(runType=plan.runType)
        context['samplePrepKit'] = KitInfo.objects.filter(name=plan.samplePrepKitName)
        context['libraryKit'] = KitInfo.objects.filter(name=eas.libraryKitName)
        context['templatingKit'] = KitInfo.objects.filter(name=plan.templatingKitName)
        context['samplePrepProtocol'] = common_CV.objects.filter(value=plan.samplePrepProtocol) if plan.samplePrepProtocol else "" 
        context['sequenceKit'] = KitInfo.objects.filter(name=plan.experiment.sequencekitname)
        context['controlSequencekit'] = KitInfo.objects.filter(name=plan.controlSequencekitname)
        context["chipTypePrefix"] = getChipDisplayedNamePrimaryPrefix(chipType[0]) if chipType else plan.experiment.chipType
        context["chipTypeSecondaryPrefix"] = getChipDisplayedNameSecondaryPrefix(chipType[0]) if chipType else plan.experiment.chipType
        context["chipTypeVersion"] = getChipDisplayedVersion(chipType[0]) if chipType else ""

        context["ampsOnChef_sampleSets"] = plan.sampleSets.filter(libraryPrepInstrumentData__isnull=False, libraryPrepType__contains="amps_on_chef")

        context['thumbnail'] = True if report_pk and result.isThumbnail else False
        context['show_thumbnail'] = True if state != 'Plan' and (plan.experiment.getPlatform in ['s5', 'proton']) else False
        context['from_report'] = True if report_pk else False

        # display saved args if viewing from existing Report or if custom args, otherwise display current default args
        context['args'] = eas.get_cmdline_args() if report_pk or eas.custom_args else plan.get_default_cmdline_args()

        # IRU configuration
        iru_info = eas.selectedPlugins.get('IonReporterUploader', {}).get('userInput', {})
        iru_info = iru_info['userInputInfo'] if 'userInputInfo' in iru_info else iru_info
        # map columns to be displayed to actual parameter names saved in userInputInfo
        # non-empty columns will appear in the order given
        iru_display_keys = (
            ('Cancer Type', 'cancerType'),
            ('Cellularity %', 'cellularityPct'),
            ('Workflow', 'Workflow'),
            ('Relation', 'RelationRole'),
            ('Gender', 'Gender'),
            ('Set ID', 'setid')
        )
        context['iru_config'] = {}

        if (iru_info):
            # unicode handling
            iru_info = convert(iru_info)
            # logger.debug("views.PlanDetailView - AFTER iru_info=%s" %(iru_info))

        for config_dict in iru_info:
            params = {}
            columns = []
            if (isinstance(config_dict, dict)):
                for column, key in iru_display_keys:
                    if key in config_dict:
                        params[column] = config_dict[key]
                        columns.append(column)
                        if key == 'setid':
                            params['Set ID'] = config_dict['setid'].split('_')[0]
                        if key == 'Workflow' and config_dict[key] == '':
                            params['Workflow'] = 'Upload Only'
            else:
                logger.debug("views.PlanDetailView - SKIPPED -  config_dict.type=%s" % (type(config_dict)))

            if params:
                barcoded = config_dict.get('barcodeId') or 'nobarcode'
                context['iru_config'][barcoded] = params
                context['iru_columns'] = columns

        # QC thresholds
        context['qcValues'] = []
        for qcValue in plan.qcValues.all().order_by('qcName'):
            context['qcValues'].append((qcValue.qcName, PlannedExperimentQC.objects.get(plannedExperiment=plan, qcType=qcValue).threshold))

        # plugins
        context['plugins'] = filter(lambda d: 'export' not in d.get('features', []), eas.selectedPlugins.values())
        context['uploaders'] = filter(lambda d: 'export' in d.get('features', []), eas.selectedPlugins.values())

        plugins = Plugin.objects.filter(name__in=eas.selectedPlugins.keys(), active=True)
        for plugin in context['plugins']:
            if plugin.get('userInput'):
                pluginObj = plugins.filter(name=plugin['name'])
                if pluginObj:
                    plugin['isPlanConfig'] = pluginObj[0].isPlanConfig
                    plugin['userInputJSON'] = json.dumps(plugin['userInput'], cls=DjangoJSONEncoder)
        context['plugins'].sort(key=lambda d: d['name'])

        # projects
        if report_pk:
            context['projects'] = result.projectNames()
        else:
            context['projects'] = ', '.join([p.name for p in plan.projects.all().order_by('-modified')])

        # barcodes
        if eas.barcodeKitName:
            barcodedSamples = {}
            columns = []
            applicationGroup = plan.applicationGroup.name if plan.applicationGroup else ""
            if eas.barcodedSamples:
                bcsamples_display_keys = (
                    ('Control Type', 'controlType'),
                    ('Sample ID', 'externalId'),
                    ('Sample Description', 'description'),
                    ('DNA/Fusions' if applicationGroup == 'DNA + RNA' else 'DNA/RNA', 'nucleotideType'),
                    ('Reference', 'reference'),
                    ('Target Regions', 'targetRegionBedFile'),
                    ('Hotspot Regions', 'hotSpotRegionBedFile'),
                )

                for sample, info in eas.barcodedSamples.items():
                    for bcId in info.get('barcodes', []):
                        barcodedSamples[bcId] = {}
                        barcodedSamples[bcId]['sample'] = sample

                        barcodeSampleInfo = info.get('barcodeSampleInfo', {}).get(bcId, {})
                        for column, key in bcsamples_display_keys:
                            if key in barcodeSampleInfo and barcodeSampleInfo[key]:
                                if column not in columns:
                                    columns.append(column)

                                value = barcodeSampleInfo[key]
                                if key in ['targetRegionBedFile', 'hotSpotRegionBedFile']:
                                    value = os.path.basename(barcodeSampleInfo[key])
                                elif key == "nucleotideType" and value == "RNA" and applicationGroup == 'DNA + RNA':
                                    value = "Fusions"

                                barcodedSamples[bcId][column] = value

            context['bcsamples_columns'] = columns
            context['barcodedSamples'] = barcodedSamples
            context['barcodes'] = dnaBarcode.objects.filter(name=eas.barcodeKitName, id_str__in=barcodedSamples.keys()).order_by('id_str')

        # LIMS data
        if plan.metaData:
            data = plan.metaData.get("LIMS", "")
            if (type(data) is list):
                # convert list to string
                context['LIMS_meta'] = ''.join(data)
            else:
                # convert unicode to str
                context['LIMS_meta'] = convert(data)
        else:
            context['LIMS_meta'] = ""

        if plan.origin:
            planMeta = plan.origin.split('|')
            context['origin'] = planMeta[0].upper()
            if len(planMeta) > 1:
                context['tsVersion'] = planMeta[1]

        # log
        history = EventLog.objects.for_model(plan)
        for log in history:
            try:
                log.json_log = json.loads(log.text)
            except:
                pass
        context['event_log'] = history

        return context


def _plan_template_helper(request, _id, isForTemplate, intent):
    data = _get_allApplProduct_data(isForTemplate)
    planTemplate = get_object_or_404(PlannedExperiment, pk=_id)
    runType = get_object_or_404(RunType, runType=planTemplate.runType)

    # add experiment attributes
    experiment = planTemplate.experiment
    exp_keys = [
        ('autoAnalyze', 'autoAnalyze'),
        ('flows', 'flows'),
        ('notes', 'notes'),
        ('sequencekitname', 'sequencekitname'),
    ]
    for plankey, key in exp_keys:
        setattr(planTemplate, plankey, getattr(experiment, key, ''))

    chipResource = ChipResource()
    base_bundle = chipResource.build_bundle(request=request)
    try:
        chip = chipResource.obj_get(bundle=base_bundle, name=experiment.chipType)
        if chip:
            setattr(planTemplate, "chipType", experiment.chipType)
    except Chip.DoesNotExist:
        try:
            chip = chipResource.obj_get(bundle=base_bundle, name=experiment.chipType[:3])
            if chip:
                setattr(planTemplate, "chipType", experiment.chipType[:3])
        except Chip.DoesNotExist:
            setattr(planTemplate, "chipType", experiment.chipType)

    setattr(planTemplate, "sample", experiment.get_sample())
    setattr(planTemplate, "sampleDisplayedName", experiment.get_sampleDisplayedName())

    setattr(planTemplate, "isIonChef", planTemplate.is_ionChef())

    # add EAS attributes
    if experiment:
        eas = experiment.get_EAS()
    else:
        eas = None

    eas_keys = [
        ('barcodedSamples', 'barcodedSamples'),
        ('barcodeId', 'barcodeKitName'),
        ('bedfile', 'targetRegionBedFile'),
        ('forward3primeadapter', 'threePrimeAdapter'),
        ('libraryKey', 'libraryKey'),
        ('librarykitname', 'libraryKitName'),
        ('regionfile', 'hotSpotRegionBedFile'),
        ('selectedPlugins', 'selectedPlugins'),
        ('isDuplicateReads', 'isDuplicateReads')
    ]
    for plankey, key in eas_keys:
        setattr(planTemplate, plankey, getattr(eas, key, ''))

    if eas:
        setattr(planTemplate, "library", eas.reference if eas.reference != "none" else '')
    else:
        setattr(planTemplate, "library", "")

    chipTypeDetails = None
    if planTemplate.chipType:
        chipTypeDetails = get_object_or_404(Chip, name=planTemplate.chipType)

    selectedProjectNames = [selectedProject.name for selectedProject in list(planTemplate.projects.all())]
    logger.debug("views._plan_template_helper selectedProjectNames=%s" % selectedProjectNames)

    # mark plugins selected if any
    for plugin in data['plugins']:
        if plugin.name in planTemplate.selectedPlugins.keys():
            plugin.selected = True
            plugin.userInput = json.dumps(planTemplate.selectedPlugins[plugin.name].get('userInput', None))
        else:
            plugin.selected = False

    # mark uploaders selected if any
    for plugin in data['uploaders']:
        if plugin.name in planTemplate.selectedPlugins.keys():
            plugin.selected = True

            # in case we're copying old plans that have dict {} data type for userInput. new format is list of dict [{}]
            userInput = planTemplate.selectedPlugins[plugin.name].get('userInput', None)

            plugin.userInput = json.dumps(userInput)

            if (isinstance(userInput, dict)):
                plugin.userInput = json.dumps([userInput])

            if 'IonReporter' in plugin.name:
                data['irConfigSaved'] = plugin.userInput
                data['irConfigSaved_version'] = 1.0 if plugin.name == 'IonReporterUploader_V1_0' else plugin.version

        else:
            plugin.selected = False

    globalConfig_isDuplicateReads = GlobalConfig.get().mark_duplicates

    # planTemplateData contains what are available for selection
    # and what each application product's characteristics and default selection
    ctxd = {
        "intent": intent,
        "planTemplateData": data,
        "selectedApplProductData": "",
        "selectedPlanTemplate": planTemplate,
        "selectedRunType": runType,
        "selectedProjectNames": selectedProjectNames,
        "selectedChipTypeDetails": chipTypeDetails,
        "globalConfig_isDuplicateReads": globalConfig_isDuplicateReads
    }
    context = RequestContext(request, ctxd)
    return context


@login_required
def batch_plans_from_template(request, template_id):
    """
    To create multiple plans from an existing template
    """

    planTemplate = get_object_or_404(PlannedExperiment, pk=template_id)

    # planTemplateData contains what are available for selection
    ctxd = {
        "selectedPlanTemplate": planTemplate,
        "is_barcoded": True if planTemplate.get_barcodeId() else False
    }
    context = RequestContext(request, ctxd)
    return render_to_response("rundb/plan/modal_batch_planning.html", context_instance=context)


@login_required
def getCSV_for_batch_planning(request, templateId, count, uploadtype="single"):
    """
    To create csv file for batch planning based on an existing template
    """
    def create_csv(output, version, hdr, body):
        writer = csv.writer(output)
        writer.writerow(version)
        writer.writerow(hdr)
        for row in body:
            writer.writerow(row)

    single_file = (uploadtype == 'single')
    batch_filename = 'batchPlanning_%s' % str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    plan_csv_version = get_plan_csv_version()

    hdr, body = get_template_data_for_batch_planning(templateId, single_file)
    if single_file:
        response = http.HttpResponse(mimetype='text/csv')
        response['Content-Disposition'] = 'attachment; filename=%s.csv' % batch_filename
        rows = int(count)*[body]
        create_csv(response, plan_csv_version, hdr, rows)
    else:
        response = HttpResponse(mimetype='application/zip')
        response['Content-Disposition'] = 'attachment; filename=%s.zip' % batch_filename
        zip_file = zipfile.ZipFile(response, 'w')

        rows = []
        samples_hdr, samples_body = get_samples_data_for_batch_planning(templateId)
        for n in range(int(count)):
            name = batch_filename + '_samples_%02d.csv' % (n+1)
            body[hdr.index(PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER)] = name
            rows.append(list(body))
            # create and add samples CSV file
            csv_output = io.BytesIO()
            create_csv(csv_output, plan_csv_version, samples_hdr, samples_body)
            zip_file.writestr(name, csv_output.getvalue())

        # create and add plan CSV file
        csv_output = io.BytesIO()
        create_csv(csv_output, plan_csv_version, hdr, rows)
        zip_file.writestr(batch_filename + ".csv", csv_output.getvalue())

    return response


@login_required
def upload_plans_for_template(request):
    """
    Allow user to upload a csv file to create plans based on a previously selected template
    """

    ctxd = {}

    context = RequestContext(request, ctxd)
    return render_to_response("rundb/plan/modal_batch_planning_upload.html", context_instance=context)


@login_required
@transaction.commit_manually
def save_uploaded_plans_for_template(request):
    """add plans, with CSV validation"""

    def _close_files(files, temp_file):
        for csvfile in files.values():
            if not csvfile.closed:
                csvfile.close()
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    def _fail(files={}, temp_file=None, status="", failed=""):
        # helper method to clean up and return HttpResponse with error messages
        transaction.rollback()
        _close_files(files, temp_file)
        error = {"status": status, "failed": failed}
        logger.info("views.save_uploaded_plans_for_template() error=%s" % (error))
        return HttpResponse(json.dumps(error), mimetype="text/html")

    postedfile = request.FILES['postedfile']
    destination = tempfile.NamedTemporaryFile(delete=False)

    for chunk in postedfile.chunks():
        destination.write(chunk)
    postedfile.close()
    destination.close()

    files = {}
    if zipfile.is_zipfile(destination.name):
        myzip = zipfile.ZipFile(destination.name)
        for filename in myzip.namelist():
            if ("__MACOSX" in filename) or (myzip.getinfo(filename).file_size == 0):
                continue
            files[os.path.basename(filename)] = myzip.open(filename, "rU")
    else:
        files[destination.name] = open(destination.name, "rU")

    if len(files) == 0:
        return _fail(temp_file=destination, status="Error: batch planning file is empty")

    single_file = len(files) == 1
    failed = {}
    isToAbort = False

    # validate files contents and CSV version
    csv_version_header = get_plan_csv_version()[0]
    for filename, csvfile in files.items():
        err_name = postedfile.name if single_file else filename
        try:
            csv_version_row = csv.reader(csvfile).next()
            errorMsg, isToSkipRow, abortFile = utils.validate_csv_template_version(headerName=csv_version_header, isPlanCSV=True, firstRow=csv_version_row)
            if abortFile:
                isToAbort = True
                failed[err_name] = [errorMsg[0][1]]
        except:
            logger.error(traceback.format_exc())
            failed[err_name] = ["Error reading from file"]
            isToAbort = True

    if isToAbort:
        return _fail(files, destination, "", failed)

    samples = {}
    if single_file:
        reader = csv.DictReader(files.values()[0])
    else:
        reader = None
        for filename, csvfile in files.items():
            test = csv.DictReader(csvfile)
            if PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER in test.fieldnames:
                if reader:
                    return _fail(files, destination, "Error: multiple batch planning files found")
                else:
                    reader = test
            else:
                samples[filename] = list(test)

        if not reader:
            return _fail(files, destination, "Error: batch planning file not found for barcoded CSV files upload")
        if not samples:
            return _fail(files, destination, "Error: batch planning no samples files found for barcoded ZIP upload")

    index = 0
    plans = []
    rawPlanDataList = []
    username = request.user.username if request.user else ""
    # process and validate Plans
    for index, row in enumerate(reader, start=2):

        samples_contents = samples.get(row[PlanCSVcolumns.COLUMN_SAMPLE_FILE_HEADER]) if not single_file else None
        errorMsg, aPlanDict, rawPlanDict, isToSkipRow = validate_csv_plan(row, username, single_file, samples_contents)

        logger.info("views.save_uploaded_plans_for_template() index=%d; errorMsg=%s; planDict=%s" % (index, errorMsg, rawPlanDict))
        if errorMsg:
            failed['Row %d' % (index+1)] = errorMsg
            continue
        elif isToSkipRow:
            logger.info("views.save_uploaded_plans_for_template() SKIPPED ROW index=%d; row=%s" % (index, row))
            continue
        else:
            plans.append(aPlanDict)
            rawPlanDataList.append(rawPlanDict)

    # now close and remove the temp file
    _close_files(files, destination)

    if index == 0:
        return HttpResponse(json.dumps({"status": "Error: There must be at least one plan! Please reload the page and try again with more plans."}), mimetype="text/html")

    if failed:
        r = {"status": "Plan validation failed. The plans have not been saved.", "failed": failed}
        logger.info("views.save_uploaded_plans_for_template() failed=%s" % (r))

        transaction.rollback()
        return HttpResponse(json.dumps(r), mimetype="text/html")

    # saving to db needs to be the last thing to happen
    try:
        index = 0
        for planFamily in plans:
            plan = planFamily['plan']
            plan.save()

            expObj = planFamily['exp']
            expObj.plan = plan
            expObj.expName = plan.planGUID
            expObj.unique = plan.planGUID
            expObj.displayname = plan.planGUID
            expObj.save()

            easObj = planFamily['eas']
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
            sampleDisplayedNames = planFamily['samples']
            sampleNames = [name.replace(' ', '_') for name in sampleDisplayedNames]
            externalIds = planFamily['sampleIds']
            for name, displayedName, externalId in zip(sampleNames,  sampleDisplayedNames, externalIds):
                sample_kwargs = {
                    'name': name,
                    'displayedName': displayedName,
                    'date': plan.date,
                    'status': plan.planStatus,
                    'externalId': externalId
                    }

                sample = Sample.objects.get_or_create(name=name, externalId=externalId, defaults=sample_kwargs)[0]
                sample.experiments.add(expObj)
                sample.save()

            planDict = rawPlanDataList[index]

            # add QCtype thresholds
            qcTypes = QCType.objects.all()
            for qcType in qcTypes:
                qc_threshold = planDict.get(qcType.qcName, '')
                if qc_threshold:
                    # get existing PlannedExperimentQC if any
                    plannedExpQcs = PlannedExperimentQC.objects.filter(plannedExperiment=plan.id, qcType=qcType.id)
                    if len(plannedExpQcs) > 0:
                        for plannedExpQc in plannedExpQcs:
                            plannedExpQc.threshold = qc_threshold
                            plannedExpQc.save()
                    else:
                        kwargs = {
                            'plannedExperiment': plan,
                            'qcType': qcType,
                            'threshold': qc_threshold
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

        r = {"status": "Error saving plans to database. ", "failed": failed}
        return HttpResponse(json.dumps(r), mimetype="text/html")
    else:
        # logger.info("views.save_uploaded_plans_for_template going to transaction.COMMIT")

        transaction.commit()
        r = {"status": "Plans Uploaded! The plans will be listed on the planned run page.", "failed": failed}
        return HttpResponse(json.dumps(r), mimetype="text/html")


@login_required
def get_application_product_presets(request):
    data = _get_allApplProduct_data(True)
    json_serializer = serializers.get_serializer("json")()
    fields = data.keys()
    result = json_serializer.serialize(data, fields=fields)
    return HttpResponse(result, mimetype="application/json")


def _get_allApplProduct_data(isForTemplate):
    def pretty(d, indent=0):
        if d:
            for key, value in d.iteritems():
                logger.debug('\t' * indent + str(key))
                if isinstance(value, dict):
                    pretty(value, indent + 1)
                else:
                    logger.debug('\t' * (indent + 1) + str(value))

    data = _get_base_planTemplate_data(isForTemplate)

    runTypes = list(RunType.objects.all())
    for appl in runTypes:

        applType = appl.runType

        try:
            # we should only have 1 default per application. TODO: add logic to ApplProduct to ensure that
            defaultApplProduct = list(ApplProduct.objects.filter(isActive=True, isDefault=True, applType=appl))
            if defaultApplProduct[0]:

                applData = {}

                applData["runType"] = defaultApplProduct[0].applType
                applData["reference"] = defaultApplProduct[0].defaultGenomeRefName
                applData["targetBedFile"] = defaultApplProduct[0].defaultTargetRegionBedFileName
                applData["hotSpotBedFile"] = defaultApplProduct[0].defaultHotSpotRegionBedFileName
                applData["seqKit"] = defaultApplProduct[0].defaultSequencingKit
                applData["libKit"] = defaultApplProduct[0].defaultLibraryKit
                applData["peSeqKit"] = defaultApplProduct[0].defaultPairedEndSequencingKit
                applData["peLibKit"] = defaultApplProduct[0].defaultPairedEndLibraryKit
                applData["chipType"] = defaultApplProduct[0].defaultChipType

                if defaultApplProduct[0].defaultChipType:
                    applData["chipTypeDetails"] = get_object_or_404(Chip, name=defaultApplProduct[0].defaultChipType)
                else:
                    applData["chipTypeDetails"] = None

                applData["isPairedEndSupported"] = defaultApplProduct[0].isPairedEndSupported
                applData["isDefaultPairedEnd"] = defaultApplProduct[0].isDefaultPairedEnd

                applData['flowCount'] = defaultApplProduct[0].defaultFlowCount
                applData['peAdapterKit'] = defaultApplProduct[0].defaultPairedEndAdapterKit
                applData['defaultOneTouchTemplateKit'] = defaultApplProduct[0].defaultTemplateKit
                applData['controlSeqKit'] = defaultApplProduct[0].defaultControlSeqKit
                applData['isHotspotRegionBEDFileSupported'] = defaultApplProduct[0].isHotspotRegionBEDFileSuppported

                applData['isDefaultBarcoded'] = defaultApplProduct[0].isDefaultBarcoded
                applData['defaultBarcodeKitName'] = defaultApplProduct[0].defaultBarcodeKitName

                applData['defaultIonChefKit'] = defaultApplProduct[0].defaultIonChefPrepKit
                applData['defaultSamplePrepKit'] = defaultApplProduct[0].defaultSamplePrepKit

                # 20120619-TODO-add compatible plugins, default plugins

                data[applType] = applData

#                if applType == 'AMPS':
#                    pretty(data[applType])
            else:
                data[applType] = 'none'
        except:
            data[applType] = 'none'

    return data


def _get_base_planTemplate_data(isForTemplate):
    data = {}

    # per requirement, we want to display Generic Sequencing as the last entry in the selection list
    data["runTypes"] = list(RunType.objects.all().exclude(runType="GENS").order_by('nucleotideType', 'runType'))
    data["secondaryRunTypes"] = list(RunType.objects.filter(runType="GENS"))

    data["barcodes"] = list(dnaBarcode.objects.values('name').distinct().order_by('name'))

    # barcodeKitNames = dnaBarcode.objects.values_list('name', flat=True).distinct().order_by('name')
    # for barcodeKitName in barcodeKitNames:
    # data[barcodeKitName] = dnaBarcode.objects.filter(name=barcodeKitName).order_by('index')

    # barcodeKitInfo not currently used now that we have removed ability to select barcode kit from the sample config page
    # data["barcodeKitInfo"] = list(dnaBarcode.objects.values('name', 'index', 'id_str', 'sequence').order_by('name', 'index'))

    references = list(ReferenceGenome.objects.all().filter(index_version=settings.TMAP_VERSION))
    data["references"] = references
    data["referenceShortNames"] = [ref.short_name for ref in references]

    data.update(dict_bed_hotspot())

    data["seqKits"] = KitInfo.objects.filter(kitType='SequencingKit', isActive=True).order_by("name")
    data["libKits"] = KitInfo.objects.filter(kitType='LibraryKit', isActive=True).order_by("name")

    data["variantfrequencies"] = VariantFrequencies.objects.all().order_by("name")

    # the entry marked as the default will be on top of the list
    data["forwardLibKeys"] = LibraryKey.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'name')
    data["forward3Adapters"] = ThreePrimeadapter.objects.filter(direction='Forward', runMode='single').order_by('-isDefault', 'name')

    # pairedEnd does not have special forward library keys
    # for TS-4669: remove paired-end from wizard, if there are no active PE lib kits, do not prepare pe keys or adapters
    peLibKits = KitInfo.objects.filter(kitType='LibraryKit', runMode='pe', isActive=True)
    if (peLibKits.count() > 0):
        data["peForwardLibKeys"] = LibraryKey.objects.filter(direction='Forward').order_by('-isDefault', 'name')
        data["peForward3Adapters"] = ThreePrimeadapter.objects.filter(direction='Forward', runMode='pe').order_by('-isDefault', 'name')
        data["reverseLibKeys"] = LibraryKey.objects.filter(direction='Reverse').order_by('-isDefault', 'name')
        data["reverse3Adapters"] = ThreePrimeadapter.objects.filter(direction='Reverse').order_by('-isDefault', 'name')
    else:
        data["peForwardLibKeys"] = None
        data["peForward3Adapters"] = None
        data["reverseLibKeys"] = None
        data["reverse3Adapters"] = None

    # chip types
    # note: customer-facing chip names are no longer unique
    data['chipTypes'] = list(Chip.objects.filter(isActive=True).order_by('description', 'name').distinct('description'))
    # QC
    data['qcTypes'] = list(QCType.objects.all().order_by('qcName'))
    # project
    data['projects'] = list(Project.objects.filter(public=True).order_by('name'))

    # templating kit selection
    data["templateKits"] = KitInfo.objects.filter(kitType='TemplatingKit', isActive=True).order_by("name")
    # control sequence kit selection
    data["controlSeqKits"] = KitInfo.objects.filter(kitType='ControlSequenceKit', isActive=True).order_by("name")

    # ionChef kit selection
    data["ionChefKits"] = KitInfo.objects.filter(kitType='IonChefPrepKit', isActive=True).order_by("name")

    # pairedEnd library adapter selection
    # for TS-4669: remove paired-end from wizard, if there are no active PE seq kits, do not prepare pe keys or adapters
    if (peLibKits.count() > 0):
        data["pairedEndLibAdapters"] = KitInfo.objects.filter(kitType='AdapterKit', runMode="pe", isActive=True).order_by('name')
    else:
        data["pairedEndLibAdapters"] = None

    # samplePrep kits
    data["samplePrepKits"] = KitInfo.objects.filter(kitType='SamplePrepKit', isActive=True).order_by('name')

    # to allow data entry for multiple non-barcoded samples at the plan wizard
    data['nonBarcodedSamples_irConfig_loopCounter'] = [i + 1 for i in range(20)]

    return data


@login_required
@transaction.commit_manually
def save_plan_or_template(request, planOid):
    """
    Saving new or edited plan/template to db (source: plan template wizard)
    Editing a planned run from having 1 sample to 2 samples will result in one edited planned run and one new planned run
    """
    def isReusable(submitIntent):
        return not (submitIntent == 'savePlan' or submitIntent == 'updatePlan')

    if request.method != 'POST':
        logger.exception(format_exc())
        return HttpResponse(json.dumps({"error": "Error, unsupported HTTP Request method (%s) for plan update." % request.method}), mimetype="application/json")

    # Process Inputs

    # pylint:disable=E1103
    json_data = simplejson.loads(request.body)
    submitIntent = json_data.get('submitIntent', '')
    logger.debug('views.save_plan_or_template POST.body... simplejson Data: "%s"' % json_data)
    logger.debug("views.save_plan_or_template submitIntent=%s" % submitIntent)
    # saving Template or Planned Run
    isReusable = isReusable(submitIntent)
    runModeValue = json_data.get('runMode', 'single')
    isPlanGroupValue = runModeValue == 'pe' and not isReusable
    libraryKeyValue = json_data.get('libraryKey', '')
    forward3primeAdapterValue = json_data.get('forward3primeAdapter', '')
    flowsInOrderValue = jaon_data.get("flowOrder", "")

    msgvalue = 'Run Plan' if not isReusable else 'Template'
    if runModeValue == 'pe':
        return HttpResponse(json.dumps({"error": "Error, paired-end plan is no longer supported. %s will not be saved." % (msgvalue)}), mimetype="application/html")

    planDisplayedNameValue = json_data.get('planDisplayedName', '').strip()
    noteValue = json_data.get('notes_workaround', '')

    # perform server-side validation to avoid things falling through the crack
    if not planDisplayedNameValue:
        return HttpResponse(json.dumps({"error": "Error, please enter a %s Name." % (msgvalue)}), mimetype="application/html")

    if not is_valid_chars(planDisplayedNameValue):
        return HttpResponse(json.dumps({"error": "Error, %s Name" % (msgvalue) + ERROR_MSG_INVALID_CHARS}), mimetype="application/html")

    if not is_valid_length(planDisplayedNameValue, MAX_LENGTH_PLAN_NAME):
        return HttpResponse(json.dumps({"error": "Error, %s Name" % (msgvalue) + ERROR_MSG_INVALID_LENGTH % (str(MAX_LENGTH_PLAN_NAME))}), mimetype="application/html")

    if noteValue:
        if not is_valid_chars(noteValue):
            return HttpResponse(json.dumps({"error": "Error, %s note" % (msgvalue) + ERROR_MSG_INVALID_CHARS}), mimetype="application/html")

        if not is_valid_length(noteValue, MAX_LENGTH_NOTES):
            return HttpResponse(json.dumps({"error": "Error, Note" + ERROR_MSG_INVALID_LENGTH % (str(MAX_LENGTH_NOTES))}), mimetype="application/html")

    # Projects
    projectObjList = get_projects(request.user, json_data)

    # IonReporterUploader configuration and samples
    selectedPlugins = json_data.get('selectedPlugins', {})
    IRconfigList = json_data.get('irConfigList', [])

    IRU_selected = False
    for uploader in selectedPlugins.values():
        if 'ionreporteruploader' in uploader['name'].lower() and uploader['name'] != 'IonReporterUploader_V1_0':
            IRU_selected = True

    if IRU_selected:
        samples_IRconfig = json_data.get('sample_irConfig', '')

        if samples_IRconfig:
            samples_IRconfig = ','.join(samples_IRconfig)

        # generate UUID for unique setIds
        id_uuid = {}
        setids = [ir.get('setid', "") for ir in IRconfigList]

        if setids:
            for setid in set(setids):
                if setid:
                    id_uuid[setid] = str(uuid.uuid4())
            for ir_config in IRconfigList:
                setid = ir_config.get('setid', '')

                if setid:
                    ir_config['setid'] += '__' + id_uuid[setid]

        if not samples_IRconfig:
            # if more than one IRU version is set to autorun and user does not explicitly select one,
            # gui shows workflow config for IRU v1.0
            samples_IRconfig = json_data.get('samples_workaround', '')

    # Samples
    barcodeIdValue = json_data.get('barcodeId', '')
    barcodedSamples = ''
    sampleValidationErrorMsg = ''
    sampleValidationErrorMsg_leadingChars = ''
    sampleValidationErrorMsg_length = ''

    sampleTubeLabelValidationErrorMsg_length = ''

    # one Plan will be created per entry in sampleList
    # samples for barcoded Plan have a separate field (barcodedSamples)

    sampleTubeLabelList = []

    if isReusable:
        # samples entered only when saving planned run (not template)
        sampleList = ['']
    elif barcodeIdValue:
        # a barcode Set is selected
        sampleTubeLabelValue = json_data.get('sampleTubeLabel', '')
        if sampleTubeLabelValue:
            if not is_valid_length(sampleTubeLabelValue, MAX_LENGTH_PLAN_NAME):
                transaction.rollback()
                return HttpResponse(json.dumps({"error": "Error, %s sample tube label" % (msgvalue) + ERROR_MSG_INVALID_LENGTH % (str(MAX_LENGTH_SAMPLE_TUBE_LABEL))}), mimetype="application/html")
            else:
                sampleTubeLabelList.append(sampleTubeLabelValue)

        sampleList = ['']
        bcSamplesValues = json_data.get('bcSamples_workaround', '')
        bcDictionary = {}
        bcId = ""
        for token in bcSamplesValues.split(","):
            if ((token.find("bcKey|")) == 0):
                bcId, bcId_str = token.split("|")[1:]
            else:
                sample = token.strip()
                if bcId and sample:
                    if not is_valid_chars(sample):
                        sampleValidationErrorMsg += sample + ', '
                    elif not is_valid_leading_chars(sample):
                        sampleValidationErrorMsg_leadingChars += sample + ", "
                    elif not is_valid_length(sample, MAX_LENGTH_SAMPLE_NAME):
                        sampleValidationErrorMsg_length += sample + ", "

                    bcDictionary.setdefault(sample, {}).setdefault('barcodes', []).append(bcId_str)
                bcId = ""

        barcodedSamples = simplejson.dumps(bcDictionary)
        logger.debug("views.save_plan_or_template after simplejson.dumps... barcodedSamples=%s;" % (barcodedSamples))

        if not bcDictionary:
            transaction.rollback()
            return HttpResponse(json.dumps({"error": "Error, please enter at least one barcode sample name."}), mimetype="application/html")

    else:
        # Non-barcoded samples
        sampleList = []
        sampleTubeLabels = ''

        if IRU_selected:
            samples = samples_IRconfig
        else:
            samples = json_data.get('samples_workaround', '')

        sampleTubeLabels = json_data.get('irSampleTubeLabels_workaround', '') if IRU_selected else json_data.get('nonIrSampleTubeLabels_workaround', '')

        logger.debug("views.save_plan_or_template json_data.get() IRU_selected=%s; sampleTubeLabels=%s " % (str(IRU_selected), sampleTubeLabels))

        sampleTubeLabelList = sampleTubeLabels.split(",")

        i = 0
        for sample in samples.split(','):
            if sample.strip():
                if not is_valid_chars(sample):
                    sampleValidationErrorMsg += sample + ', '
                elif not is_valid_leading_chars(sample):
                    sampleValidationErrorMsg_leadingChars += sample + ", "
                elif not is_valid_length(sample, MAX_LENGTH_SAMPLE_NAME):
                    sampleValidationErrorMsg_length += sample + ", "
                else:
                    sampleList.append(sample)

                    if (len(sampleTubeLabelList) > i):
                        sampleTubeLabel = sampleTubeLabelList[i]

                        if sampleTubeLabel.strip():
                            if not is_valid_length(sampleTubeLabel, MAX_LENGTH_SAMPLE_TUBE_LABEL):
                                sampleTubeLabelValidationErrorMsg_length += sampleTubeLabel + ", "
                    else:
                        sampleTubeLabelList.append("")
            i += 1

        if len(sampleList) == 0 and not sampleValidationErrorMsg and not sampleValidationErrorMsg_leadingChars and not sampleValidationErrorMsg_length:
            transaction.rollback()
            return HttpResponse(json.dumps({"error": "Error, please enter a sample name for the run plan."}), mimetype="application/html")

        logger.debug("views.save_plan_or_template sampleList=%s " % (sampleList))

        if sampleTubeLabelValidationErrorMsg_length:
            transaction.rollback()
            return HttpResponse(json.dumps({"error": "Error, sample tube label %s" % (sampleTubeLabelValidationErrorMsg_length) + ERROR_MSG_INVALID_LENGTH % (str(MAX_LENGTH_SAMPLE_TUBE_LABEL))}), mimetype="application/html")

    # Samples validation
    if sampleValidationErrorMsg or sampleValidationErrorMsg_leadingChars or sampleValidationErrorMsg_length:
        message = ""
        if sampleValidationErrorMsg:
            message = "Error, sample name" + ERROR_MSG_INVALID_CHARS
            message = message + ' <br>Please fix: ' + sampleValidationErrorMsg + '<br>'
        if sampleValidationErrorMsg_leadingChars:
            message = message + "Error, sample name" + ERROR_MSG_INVALID_LEADING_CHARS
            message = message + ' <br>Please fix: ' + sampleValidationErrorMsg_leadingChars + '<br>'
        if sampleValidationErrorMsg_length:
            message = message + "Error, sample name" + ERROR_MSG_INVALID_LENGTH % (str(MAX_LENGTH_SAMPLE_NAME))
            message = message + ' <br>Please fix: ' + sampleValidationErrorMsg_length

        transaction.rollback()
        return HttpResponse(json.dumps({"error": message}), mimetype="application/html")

    selectedPluginsValue = json_data.get('selectedPlugins', {})

    # end processing input data

    # Edit/Create Plan(s)

    if int(planOid) == 0:
        edit_existing_plan = False
    else:
        edit_existing_plan = True

    for i, sample in enumerate(sampleList):
        logger.debug("...LOOP... views.save_plan_or_template SAMPLE=%s; isSystem=%s; isReusable=%s; isPlanGroup=%s "
                     % (sample.strip(), json_data["isSystem"], isReusable, isPlanGroupValue))

        # add IonReporter config values for each sample
        if len(IRconfigList) > 0:
            for uploader in selectedPluginsValue.values():
                if 'ionreporteruploader' in uploader['name'].lower():
                    if len(IRconfigList) > 1 and not barcodeIdValue:
                        uploader['userInput'] = [IRconfigList[i]]
                    else:
                        uploader['userInput'] = IRconfigList

        if len(sampleList) > 1:
            inputPlanDisplayedName = planDisplayedNameValue + '_' + sample.strip()
        else:
            inputPlanDisplayedName = planDisplayedNameValue

        selectedTemplatingKit = json_data.get('templatekitname', '')
        samplePrepInstrumentType = json_data.get('samplePrepInstrumentType', '')
        if samplePrepInstrumentType == 'ionChef':
            selectedTemplatingKit = json_data.get('templatekitionchefname', '')

        sampleTubeLabel = ''
        if (sampleTubeLabelList and len(sampleTubeLabelList) >= i):
            sampleTubeLabel = sampleTubeLabelList[i].strip()

        # logger.debug("plan/views - i=%d; sample=%s, sampleTubeLabel=%s" %(i, sample, sampleTubeLabel))

        # PDD-TODO: remove the x_ prefix. the x_ prefix is just a reminder what the obsolete attributes to remove during the next phase
        kwargs = {
            'planDisplayedName': inputPlanDisplayedName,
            "planName": inputPlanDisplayedName.replace(' ', '_'),
            'usePreBeadfind': toBoolean(json_data['usePreBeadfind'], False),
            'usePostBeadfind': toBoolean(json_data['usePostBeadfind'], False),
            'preAnalysis': True,
            'runType': json_data['runType'],
            'templatingKitName': selectedTemplatingKit,
            'controlSequencekitname': json_data.get('controlsequence', ''),
            'runMode': runModeValue,
            'isSystem': toBoolean(json_data['isSystem'], False),
            'isReusable': isReusable,
            'isPlanGroup': isPlanGroupValue,
            'username': request.user.username,
            'isFavorite': toBoolean(json_data.get('isFavorite', 'False'), False),
            'pairedEndLibraryAdapterName': json_data.get('pairedEndLibraryAdapterName', ''),
            'samplePrepKitName': json_data.get('samplePrepKitName', ''),
            'planStatus': "planned",
            'sampleTubeLabel': sampleTubeLabel,

            'x_autoAnalyze': True,
            'x_barcodedSamples': barcodedSamples,
            'x_barcodeId': barcodeIdValue,
            'x_bedfile': json_data.get('bedfile', ''),
            'x_chipType': json_data.get('chipType', ''),
            'x_flows': json_data.get('flows', None),
            'x_forward3primeadapter': forward3primeAdapterValue,
            'x_flowsInOrder': flowsInOrderValue,
            # '_isReverseRun':  = self.isReverseRun
            'x_library': json_data.get('library', ''),
            'x_libraryKey': libraryKeyValue,
            'x_librarykitname': json_data.get('librarykitname', ''),
            'x_notes': noteValue,
            'x_regionfile': json_data.get('regionfile', ''),
            'x_sample': sample.strip().replace(' ', '_'),
            'x_sampleDisplayedName': sample.strip(),
            'x_selectedPlugins': selectedPluginsValue,
            'x_sequencekitname': json_data.get('sequencekitname', ''),
            'x_variantfrequency': json_data.get('variantfrequency', ''),
            'x_isDuplicateReads': json_data.get('isDuplicateReads', False)
        }

        planTemplate = None
        logger.debug("KWARGS ARE: %s" % str(kwargs))
        for key, value in sorted(kwargs.items()):
            logger.debug('KWARG %s: %s' % (str(key), str(value)))
        # if we're changing a plan from having 1 sample to say 2 samples, we need to UPDATE 1 plan and CREATE 1 plan!!
        try:
            if not edit_existing_plan:
                planTemplate, extra_kwargs = PlannedExperiment.objects.save_plan(-1, **kwargs)
            else:
                planTemplate, extra_kwargs = PlannedExperiment.objects.save_plan(planOid, **kwargs)

                edit_existing_plan = False

            # Update QCtype thresholds
            qcTypes = QCType.objects.all()
            for qcType in qcTypes:
                qc_threshold = json_data.get(qcType.qcName, '')
                if qc_threshold:
                    # get existing PlannedExperimentQC if any
                    plannedExpQcs = PlannedExperimentQC.objects.filter(plannedExperiment=planTemplate.id, qcType=qcType.id)
                    if len(plannedExpQcs) > 0:
                        for plannedExpQc in plannedExpQcs:
                            plannedExpQc.threshold = qc_threshold
                            plannedExpQc.save()
                    else:
                        kwargs = {
                            'plannedExperiment': planTemplate,
                            'qcType': qcType,
                            'threshold': qc_threshold
                        }
                        plannedExpQc = PlannedExperimentQC(**kwargs)
                        plannedExpQc.save()

            # add/remove projects
            if projectObjList:
                # TODO: refactor this logic to simplify using django orm
                projectNameList = [project.name for project in projectObjList]
                for currentProject in planTemplate.projects.all():
                    if currentProject.name not in projectNameList:
                        planTemplate.projects.remove(currentProject)
                for projectObj in projectObjList:
                    planTemplate.projects.add(projectObj)
            else:
                planTemplate.projects.clear()

        except ValidationError, err:
            transaction.rollback()
            logger.exception(format_exc())

            message = "Internal error while trying to save the plan. "
            for msg in err.messages:
                message += str(msg)
                message += " "

            return HttpResponse(json.dumps({"error": message}), mimetype="application/json")

        except Exception as excp:
            transaction.rollback()
            logger.exception(format_exc())

            message = "Internal error while trying to save the plan. %s" % (excp.message)
            return HttpResponse(json.dumps({"error": message}), mimetype="application/json")
        except:
            transaction.rollback()
            logger.exception(format_exc())
            return HttpResponse(json.dumps({"error": "Internal error while trying to save the plan."}), mimetype="application/json")
        else:
            transaction.commit()

    return HttpResponse(json.dumps({"status": "plan template updated successfully"}), mimetype="application/json")


@login_required
def plan_transfer(request, pk, destination=None):
    plan = get_object_or_404(PlannedExperiment, pk=pk)
    ctxd = {
        'planName': plan.planDisplayedName,
        'destination': destination,
        'action': reverse('api_dispatch_transfer', kwargs={'resource_name': 'plannedexperiment', 'api_name': 'v1', 'pk': int(pk)})
    }
    return render_to_response("rundb/plan/modal_plan_transfer.html", context_instance=RequestContext(request, ctxd))


def page_plan_samples_table_keys(is_barcoded, include_IR=False):
    barcoded = (
        ('barcodeId',           PlanCSVcolumns.COLUMN_BARCODE),
        ('controlType',         PlanCSVcolumns.COLUMN_SAMPLE_CONTROLTYPE),
    )
    default = (
        ('sampleName',          PlanCSVcolumns.COLUMN_SAMPLE_NAME),
        ('sampleExternalId',    PlanCSVcolumns.COLUMN_SAMPLE_ID),
        ('sampleDescription',   PlanCSVcolumns.COLUMN_SAMPLE_DESCRIPTION),
        ('nucleotideType',      PlanCSVcolumns.COLUMN_NUCLEOTIDE_TYPE),
        ('reference',           PlanCSVcolumns.COLUMN_REF),
        ('targetRegionBedFile', PlanCSVcolumns.COLUMN_TARGET_BED),
        ('hotSpotRegionBedFile', PlanCSVcolumns.COLUMN_HOTSPOT_BED),
    )
    non_barcoded = (
        ('tubeLabel',           PlanCSVcolumns.COLUMN_SAMPLE_TUBE_LABEL),
        ('chipBarcode',         PlanCSVcolumns.COLUMN_CHIP_BARCODE),
    )
    ir = (
        ('ircancerType',        'Cancer Type'),
        ('ircellularityPct',    'Cellularity %'),
        ('irbiopsyDays',        'Biopsy Days'),
        ('ircoupleID',          'Couple ID'),
        ('irembryoID',          'Embryo ID'),
        ('irWorkflow',          'IR Workflow'),
        ('irRelationRole',      'IR Relation'),
        ('irGender',            'IR Gender'),
        ('irSetID',             'IR Set ID'),
    )

    if is_barcoded:
        keys = barcoded + default
    else:
        keys = default + non_barcoded

    if include_IR: keys += ir
    return keys


def page_plan_samples_table_alternate_keys():
    keys = (
        ('nucleotideType',           'nucleotideType (DNA/RNA)'),
    )
    return keys


def page_plan_save_samples_table(request):
    ''' request from editing Plan page to save samples table from js to csv '''
    response = http.HttpResponse(mimetype='text/csv')
    response['Content-Disposition'] = 'attachment; filename=plan_samples_%s.csv' % str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    writer = csv.writer(response)

    plan_csv_version = get_plan_csv_version()
    writer.writerow(plan_csv_version)

    table = json.loads(request.POST.get('samplesTable'))
    has_IR = 'irWorkflow' in table[0]
    is_barcoded = 'barcodeId' in table[0]
    keys, header = zip(*page_plan_samples_table_keys(is_barcoded, has_IR))

    writer.writerow(header)
    for row in table:
        values = [row[key] for key in keys]
        writer.writerow(values)

    return response


def page_plan_load_samples_table(request):
    ''' load and validate samples csv file '''
    csv_file = request.FILES['csv_file']
    has_IR = request.POST.get('irSelected') == '1'
    barcodeSet = request.POST.get('barcodeSet')
    default_reference = request.POST.get('default_reference')
    default_targetBedFile = request.POST.get('default_targetBedFile')
    default_hotSpotBedFile = request.POST.get('default_hotSpotBedFile')
    runType = request.POST.get('runType_name')
    applicationGroup = request.POST.get('applicationGroupName')

    bedFileDict = dict_bed_hotspot()

    keys = page_plan_samples_table_keys(barcodeSet, has_IR) + page_plan_samples_table_alternate_keys()
    ret = {
        'same_ref_and_bedfiles': True,
        'samplesTable': [],
        'ordered_keys': zip(*keys)[0]
    }

    destination = tempfile.NamedTemporaryFile(delete=False)
    for chunk in csv_file.chunks():
        destination.write(chunk)
    destination.close()

    try:
        f = open(destination.name, 'rU')

        # validate CSV version
        csv_version_header = get_plan_csv_version()[0]
        csv_version_row = csv.reader(f).next()
        errorMsg, isToSkipRow, abortFile = utils.validate_csv_template_version(headerName=csv_version_header, isPlanCSV=True, firstRow=csv_version_row)
        if abortFile:
            error = "CSV Version is missing or not supported. Please use Save Samples Table to download latest CSV file format"
            return http.HttpResponseBadRequest(error)

        reader = csv.DictReader(f)
        error = ''
        for n, row in enumerate(reader):
            processed_row = dict([(k, row[v]) for k, v in keys if v in row])
            if not processed_row:
                continue

            ret['samplesTable'].append(processed_row)

            # validation
            row_errors = []
            for key, value in processed_row.items():
                if barcodeSet:
                    if ((len(value) - len(value.lstrip())) or
                            (len(value) - len(value.rstrip()))):
                        logger.warning("The BarcodeName (%s) of BarcodeSetName(%s) contains Leading/Trailing spaces and got trimmed." % (value, barcodeSet))
                    value = value.strip()
                    # barcoded
                    if key == 'barcodeId' and barcodeSet not in dnaBarcode.objects.filter(id_str=value).values_list('name', flat=True):
                        row_errors.append('Barcode %s is not part of selected %s Kit' % (value, barcodeSet))
                else:
                    # non-barcoded
                    if key == 'chipBarcode' and value:
                        row_errors.extend(plan_validator.validate_chipBarcode(value))
                    if key == 'tubeLabel' and value:
                        row_errors.extend(plan_validator.validate_sample_tube_label(value))

                if key == 'sampleName' and value:
                    row_errors.extend(plan_validator.validate_sample_name(value))
                if key == 'sampleExternalId' and value:
                    row_errors.extend(plan_validator.validate_sample_id(value))
                if key == 'nucleotideType' and value:
                    nuctype_err, processed_row[key] = plan_validator.validate_sample_nucleotideType(value, runType, applicationGroup)
                    row_errors.extend(nuctype_err)

                if key == 'reference' and value:
                    ref_err, processed_row[key] = plan_validator.validate_reference(value, runType, applicationGroup)
                    row_errors.extend(ref_err)
                    if default_reference != processed_row[key]:
                        ret['same_ref_and_bedfiles'] = False

                if key == 'targetRegionBedFile' and value:
                    validated = get_bedFile_for_reference(value, processed_row.get('reference'), hotspot=False)
                    if validated:
                        processed_row[key] = validated
                        if default_targetBedFile != processed_row[key]:
                            ret['same_ref_and_bedfiles'] = False
                    else:
                        row_errors.append('Target regions BED file not found for %s' % value)

                if key == 'hotSpotRegionBedFile' and value:
                    validated = get_bedFile_for_reference(value, processed_row.get('reference'), hotspot=True)
                    if validated:
                        processed_row[key] = validated
                        if default_hotSpotBedFile != processed_row[key]:
                            ret['same_ref_and_bedfiles'] = False
                    else:
                        row_errors.append('Hotspot regions BED file not found for %s' % value)

                if key == 'controlType' and value:
                    controltype_err, processed_row[key] = plan_validator.validate_sampleControlType(value)
                    row_errors.extend(controltype_err)

            if row_errors:
                error += 'Error in row %i: %s<br>' % (n+1, ' '.join(row_errors))

            if (PlanCSVcolumns.COLUMN_REF in row and row[PlanCSVcolumns.COLUMN_REF].strip() != default_reference ) \
                or (PlanCSVcolumns.COLUMN_TARGET_BED in row and row[PlanCSVcolumns.COLUMN_TARGET_BED].strip() != default_targetBedFile ) \
                    or (PlanCSVcolumns.COLUMN_HOTSPOT_BED in row and row[PlanCSVcolumns.COLUMN_HOTSPOT_BED].strip() != default_hotSpotBedFile):
                ret['same_ref_and_bedfiles'] = False

        f.close()  # now close and remove the temp file
        os.unlink(destination.name)

        if len(ret['samplesTable']) == 0:
            return http.HttpResponseBadRequest('Error: No rows could be parsed from %s' % csv_file.name)
        elif error:
            return http.HttpResponseBadRequest(error)
        else:
            return http.HttpResponse(json.dumps(ret), mimetype="text/html")

    except Exception as err:
        logger.error(format_exc())
        return http.HttpResponseServerError(repr(err))


@login_required
def plan_template_export(request, templateId):
    """
    Return csv file for Template export
    """
    filename = 'exported_template_%s' % str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    response = http.HttpResponse(mimetype='text/csv')
    response['Content-Disposition'] = 'attachment; filename=%s.csv' % filename

    plan_csv_version = get_plan_csv_version()
    data = get_template_data_for_export(templateId)
    writer = csv.writer(response)
    writer.writerow(plan_csv_version)
    for row in data:
        writer.writerow(row)

    return response


@login_required
def plan_template_import(request):
    def _get_kit_name(value, kitTypes):
        kit = KitInfo.objects.filter(isActive=True, kitType__in=kitTypes).filter(Q(name=value) | Q(description=value))
        return kit[0].name if kit else value
    
    csv_file = request.FILES['csv_file']
    destination = tempfile.NamedTemporaryFile(delete=False)
    for chunk in csv_file.chunks():
        destination.write(chunk)
    destination.close()
    try:
        f = open(destination.name, 'rU')
        # validate CSV version
        csv_version_header = get_plan_csv_version()[0]
        csv_version_row = csv.reader(f).next()
        errorMsg, isToSkipRow, abortFile = utils.validate_csv_template_version(headerName=csv_version_header, isPlanCSV=True, firstRow=csv_version_row)
        if abortFile:
            error = "CSV Version is missing or not supported. Please use Save Samples Table to download latest CSV file format"
            return http.HttpResponseBadRequest(error)

        reader = csv.DictReader(f)
        errors = {}
        warnings = {}
        plans = []
        for index, row in enumerate(reader):
            # skip blank rows
            if not any(v.strip() for v in row.values()):
                continue
            
            planDict = {}
            planDict['isNewPlan'] = True # this will trigger validation
            planDict['isReusable'] = True
            planDict['username'] = request.user.username
            planDict['origin'] = 'csv'

            custom_args = toBoolean(row.get(PlanCSVcolumns.CUSTOM_ARGS, False))
            for field, csvKey in export_template_keys(custom_args).items():
                if csvKey in row:
                    value = row[csvKey].strip() if row[csvKey] else ''
                    if csvKey == PlanCSVcolumns.TEMPLATE_NAME:
                        planDict['planName'] = value.replace(' ', '_')
                    elif csvKey == PlanCSVcolumns.RUNTYPE:
                        runType = RunType.objects.filter(Q(runType=value) | Q(alternate_name=value) | Q(description=value))
                        if runType:
                            value = runType[0].runType
                    elif csvKey == PlanCSVcolumns.COLUMN_LIBRARY_READ_LENGTH and not value:
                        value = 0
                    elif csvKey == PlanCSVcolumns.FLOW_ORDER and value.lower() == 'default':
                        value = ''
                    elif csvKey == PlanCSVcolumns.COLUMN_CHIP_TYPE and value:
                        chip = Chip.objects.filter(isActive=True).filter(Q(name=value) | Q(description=value))
                        if chip:
                            value = chip[0].name
                    elif csvKey == PlanCSVcolumns.COLUMN_SAMPLE_PREP_KIT:
                        value = _get_kit_name(value, ["SamplePrepKit"])
                    elif csvKey == PlanCSVcolumns.COLUMN_LIBRARY_KIT:
                        value = _get_kit_name(value, ["LibraryKit", "LibraryPrepKit"])
                    elif csvKey == PlanCSVcolumns.COLUMN_TEMPLATING_KIT:
                        value = _get_kit_name(value, ["TemplatingKit", "IonChefPrepKit"])
                    elif csvKey == PlanCSVcolumns.COLUMN_SEQ_KIT:
                        value = _get_kit_name(value, ["SequencingKit"])
                    elif csvKey == PlanCSVcolumns.COLUMN_CONTROL_SEQ_KIT:
                        value = _get_kit_name(value, ["ControlSequenceKit"])
                    elif csvKey == PlanCSVcolumns.COLUMN_PLUGINS and value:
                        selectedPlugins = {}
                        plugins = value.split(';')
                        for plugin in Plugin.objects.filter(name__in=plugins, active=True):
                            selectedPlugins[plugin.name] = {
                                "id": plugin.id,
                                "name": plugin.name,
                                "version": plugin.version,
                                "userInput": {},
                            }
                        value = selectedPlugins
                        # add warning if missing plugins
                        missing = [p for p in plugins if p and p not in selectedPlugins]
                        if len(missing) > 0:
                            warnings.setdefault('Row %d' % (index+1),[]).append('Plugin(s) not found: %s' % ', '.join(missing))
                    elif csvKey == PlanCSVcolumns.COLUMN_PROJECTS:
                        value = value.split(';')
                    elif csvKey == PlanCSVcolumns.COLUMN_LIMS_DATA:
                        value = { "LIMS": [value] }
                    elif csvKey == PlanCSVcolumns.CUSTOM_ARGS:
                        value = custom_args

                    planDict[field] = value

            plans.append(planDict)

        f.close()
        os.unlink(destination.name)

        if len(plans) == 0:
            return http.HttpResponseBadRequest('Error: No rows could be parsed from %s' % csv_file.name)

        # now use PlannedExperiment API resource to validate and create Template
        res = PlannedExperimentResource()
        validated = []
        for index, planDict in enumerate(plans):
            bundle = res.build_bundle(data=planDict)
            try:
                res.is_valid(res.full_hydrate(bundle))
            except ValidationError as err:
                errors['status'] = 'failed'
                errors['status_msg'] = 'Template import failed validation.'
                errors.setdefault('msg',{})['Row %d' % (index+1)] = json.loads(err.message).values()
            else:
                validated.append(bundle)

        if errors:
            return http.HttpResponse(json.dumps(errors))
        else:
            for index, bundle in enumerate(validated):
                warning_row_key = 'Row %d' % (index+1)

                # add warning if BED files are missing
                targetRegionBedFile =  bundle.data.get('targetRegionBedFile')
                if targetRegionBedFile and not os.path.exists(targetRegionBedFile):
                    warnings.setdefault(warning_row_key,[]).append('Target regions BED file not found: %s' % targetRegionBedFile)
                    bundle.data['targetRegionBedFile'] = bundle.data['bedfile'] = ''

                hotSpotRegionBedFile = bundle.data.get('hotSpotRegionBedFile')
                if hotSpotRegionBedFile and not os.path.exists(hotSpotRegionBedFile):
                    warnings.setdefault(warning_row_key,[]).append('Hotspot regions BED file not found: %s' % hotSpotRegionBedFile)
                    bundle.data['hotSpotRegionBedFile'] = bundle.data['regionfile'] = ''

                fusionsTargetRegionBedFile = bundle.data.get('mixedTypeRNA_targetRegionBedFile')
                if fusionsTargetRegionBedFile and not os.path.exists(fusionsTargetRegionBedFile):
                    warnings.setdefault(warning_row_key,[]).append('Fusions Target regions BED file not found: %s' % fusionsTargetRegionBedFile)
                    bundle.data['mixedTypeRNA_targetRegionBedFile'] = ''

                if warning_row_key in warnings:
                    metaData = bundle.data.get('metaData') or {}
                    metaData['warning'] = warnings[warning_row_key]
                    bundle.data['metaData'] = metaData

                # create plan
                res.obj_create(bundle)

            if warnings:
                return http.HttpResponse(json.dumps({
                    'status': 'warning',
                    'status_msg': "Template created with warnings.",
                    'msg': warnings
                }))
            else:
                return http.HttpResponse()

    except Exception as err:
        logger.error(format_exc())
        return http.HttpResponseServerError(repr(err))
