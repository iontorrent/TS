# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import os
import requests
import traceback
import logging
import json
import time

from django.conf import settings
from django.utils.encoding import force_unicode
from django.utils.translation import ugettext_lazy, ugettext
from django.core.urlresolvers import reverse

from iondb.rundb import labels
from iondb.rundb.json_lazy import LazyJSONEncoder, LazyDjangoJSONEncoder

from iondb.rundb.models import (
    IonMeshNode,
    PlannedExperiment,
    PlannedExperimentQC,
    QCType,
    dnaBarcode,
    User,
    Sample,
    SampleSet,
    SampleSetItem,
    Content,
    Plugin,
    EventLog,
    SamplePrepData,
)

from iondb.plugins.launch_utils import find_IRU_account
from ion import version as TS_VERSION
from iondb.utils.hostip import gethostname


logger = logging.getLogger(__name__)

debug = False


class Status:
    def __init__(self):
        self.msg = u""
        self.error = u""
        self.warning = u""

    def update(self, msg="", error="", warning=""):
        if msg:
            self.msg += u"<p>%s</p>" % msg
        if error:
            self.error += u"<p>%s</p>" % error
        if warning:
            self.warning += u"<p>%s</p>" % warning

    def to_dict(self):
        return {"msg": self.msg, "error": self.error, "warning": self.warning}

    def add_plan_history(self, plan, username):
        # update plan History log
        if self.msg:
            text = "Plan Transfer: \n" + self.msg.replace("<p>", "").replace(
                "</p>", "\n"
            )
            EventLog.objects.add_entry(plan, text, username)
        if self.error:
            text = "Plan Transfer Errors: \n" + self.error.replace("<p>", "").replace(
                "</p>", "\n"
            )
            EventLog.objects.add_entry(plan, text, username)


def parse_to_string(data, output=u""):
    if isinstance(data, dict):
        for value in list(data.values()):
            output = parse_to_string(value, output)
    elif isinstance(data, list):
        data.sort()
        for value in data:
            output = parse_to_string(value, output)
    elif isinstance(data, basestring) and data not in output:
        return output + " " + data

    return output


""" Plan share Destination TS functions """


def create_associated_objects(status, plan, obj_dict, user):

    if debug:
        starttime = time.time()

    # Samples
    if "samples" in obj_dict:
        try:
            for sample_dict in obj_dict["samples"]:
                sample, created = Sample.objects.get_or_create(
                    name=sample_dict["name"],
                    externalId=sample_dict["externalId"],
                    defaults=sample_dict,
                )
                sample.experiments.add(plan.experiment)

            status.update(
                msg=ugettext("plan_transfer.messages.postprocessing.entity.found")
                % {
                    "entityName": force_unicode(labels.Sample.verbose_name_plural),
                    "displayedName": ", ".join(
                        _sample["displayedName"] for _sample in obj_dict["samples"]
                    ),
                }
            )  # '....processed %(entityName)s: %(displayedName)s'
        except Exception as err:
            logger.error(
                "Error processing Samples for %s(%s)" % (plan.planName, plan.pk)
            )
            logger.error(traceback.format_exc())
            status.update(
                error=ugettext("plan_transfer.messages.postprocessing.entity.error")
                % {
                    "entityName": force_unicode(labels.Sample.verbose_name_plural),
                    "exceptionMsg": err,
                }
            )  # 'Error processing %(entityName)s: %(exceptionMsg)s'

    # SampleSet
    for set_dict in obj_dict["sampleSets"]:
        try:
            set_dict["creator_id"] = set_dict["lastModifiedUser_id"] = user.pk
            libraryPrep_dict = set_dict.pop("libraryPrepInstrumentData", "")
            samplesetitems = set_dict.pop("sampleSetItems", [])

            sampleSet, created = SampleSet.objects.get_or_create(
                displayedName=set_dict["displayedName"], defaults=set_dict
            )
            sampleSet.plans.add(plan)

            for setitem_dict in samplesetitems:
                sample = plan.experiment.samples.get(
                    name=setitem_dict.pop("sample__name")
                )
                if "dnabarcode" in setitem_dict:
                    barcode = setitem_dict.pop("dnabarcode")
                    dnabarcode = dnaBarcode.objects.filter(
                        name=barcode["name"], id_str=barcode["id_str"]
                    )
                    setitem_dict["dnabarcode"] = dnabarcode[0] if dnabarcode else None

                setitem_dict["creator_id"] = setitem_dict[
                    "lastModifiedUser_id"
                ] = user.pk
                item, created = SampleSetItem.objects.get_or_create(
                    sample=sample, sampleSet=sampleSet, defaults=setitem_dict
                )

            if libraryPrep_dict:
                if not sampleSet.libraryPrepInstrumentData:
                    sampleSet.libraryPrepInstrumentData = SamplePrepData.objects.create(
                        **libraryPrep_dict
                    )
                    sampleSet.save()
                else:
                    for field, value in list(libraryPrep_dict.items()):
                        setattr(sampleSet.libraryPrepInstrumentData, field, value)
                    sampleSet.libraryPrepInstrumentData.save()

            status.update(
                msg=ugettext("plan_transfer.messages.postprocessing.entity.found")
                % {
                    "entityName": force_unicode(labels.SampleSet.verbose_name),
                    "displayedName": sampleSet.displayedName,
                }
            )  # '....processed %(entityName)s: %(displayedName)s'
        except Exception as err:
            logger.error(
                "Error processing sampleSet for %s(%s)" % (plan.planName, plan.pk)
            )
            logger.error(traceback.format_exc())
            status.update(
                error=ugettext("plan_transfer.messages.postprocessing.entity.error")
                % {
                    "entityName": force_unicode(labels.SampleSet.verbose_name_plural),
                    "exceptionMsg": err,
                }
            )  # 'Error processing %(entityName)s: %(exceptionMsg)s'

    if debug:
        logger.debug(
            "%f s: Plan Transfer create_associated_objects" % (time.time() - starttime)
        )
        starttime = time.time()

    return True


def update_transferred_plan(plan, request):
    """ This function that runs on destination TS to update plan-related objects """
    if debug:
        starttime = time.time()

    # update plan History log
    log = "Transferred Planned Run: %s from %s." % (
        plan.planDisplayedName,
        plan.metaData.get("origin"),
    )  # DO NOT i18n
    EventLog.objects.add_entry(plan, log, request.user.username)

    obj_dict = json.loads(request.body)

    status = Status()
    eas = plan.latest_eas

    # create Samples, etc.
    create_associated_objects(status, plan, obj_dict, request.user)

    # Ion Reporter account id needs to be updated
    if "IonReporterUploader" in eas.selectedPlugins:
        accountId = None
        try:
            irserver = obj_dict["IR_account"]["server"]
            irversion = obj_dict["IR_account"]["version"]
            irtoken = obj_dict["IR_account"]["token"]
            irname = obj_dict["IR_account"]["name"]

            userconfigs = Plugin.objects.get(
                name="IonReporterUploader", active=True
            ).config["userconfigs"][plan.username]
            for config in userconfigs:
                if (
                    irserver == config["server"]
                    and irversion == config["version"]
                    and irtoken == config["token"]
                ):
                    accountId = config["id"]
                    accountName = config["name"]
                    break
        except Exception:
            pass

        if accountId:
            userInput = eas.selectedPlugins["IonReporterUploader"]["userInput"]
            userInput["accountId"] = accountId
            if irname != accountName:
                userInput["accountName"] = userInput["accountName"].replace(
                    irname, accountName
                )
            status.update(
                msg=ugettext(
                    "plan_transfer.messages.postprocessing.IonReporterUploader.accountfound"
                )
                % {"iru_accountname": userInput["accountName"]}
            )  # '....found IR account %(iru_accountname)s' )
        else:
            eas.selectedPlugins.pop("IonReporterUploader")
            status.update(
                error=ugettext(
                    "plan_transfer.messages.validation.postprocessing.IonReporterUploader.accountnotfound"
                )
            )  # 'Error: IonReporter account not found. Please add IR account on destination Torrent Server and update the Planned run.'

    eas.save()

    status.add_plan_history(plan, request.user.username)

    if debug:
        logger.debug(
            "%f s: Plan Transfer update_transferred_plan %s"
            % ((time.time() - starttime), plan.planDisplayedName)
        )

    return status.to_dict()


""" Plan share Origin TS functions """


def prepare_for_copy(bundle):

    # remove obj keys that need to be recreated
    bundle.data.pop("id")
    bundle.data.pop("experiment")
    bundle.data.pop("sampleSets")

    # qcValues
    qcValues = bundle.data.pop("qcValues", [])
    try:
        for qc in qcValues:
            bundle.data[qc.obj.qcType.qcName] = qc.obj.threshold
    except Exception:
        logger.error(traceback.format_exc())

    # indicate this plan's origin
    bundle.data["origin"] = "transfer"
    bundle.data["metaData"]["origin"] = gethostname()
    bundle.data["metaData"]["uri"] = bundle.data.pop("resource_uri")
    return bundle


def get_associated_objects_json(plan):
    """ Gather associated objects to send to destination """

    def get_obj_dict(obj):
        d = {}
        for field in obj._meta.fields:
            if field.get_internal_type() not in [
                "AutoField",
                "ForeignKey",
                "OneToOneField",
                "ManyToManyField",
            ]:
                d[field.name] = getattr(obj, field.name)
        return d

    obj_dict = {}

    # Samples
    samples = plan.experiment.samples.all()
    obj_dict["samples"] = []
    for sample in samples:
        d = get_obj_dict(sample)
        obj_dict["samples"].append(d)

    # SampleSet
    obj_dict["sampleSets"] = []
    for sampleSet in plan.sampleSets.all():
        sampleSet_dict = get_obj_dict(sampleSet)
        sampleSet_dict["SampleGroupType_CV_id"] = sampleSet.SampleGroupType_CV_id
        sampleSet_dict["libraryPrepInstrumentData"] = (
            get_obj_dict(sampleSet.libraryPrepInstrumentData)
            if sampleSet.libraryPrepInstrumentData
            else {}
        )

        sampleSet_dict["sampleSetItems"] = []
        for setitem in sampleSet.samples.filter(sample__in=samples):
            setitem_dict = get_obj_dict(setitem)
            setitem_dict["sample__name"] = setitem.sample.name
            if setitem.dnabarcode:
                setitem_dict["dnabarcode"] = {
                    "name": setitem.dnabarcode.name,
                    "id_str": setitem.dnabarcode.id_str,
                }
            sampleSet_dict["sampleSetItems"].append(setitem_dict)

        obj_dict["sampleSets"].append(sampleSet_dict)

    # Ion Reporter account
    eas = plan.latest_eas
    if "IonReporterUploader" in eas.selectedPlugins:
        try:
            accountId = (
                eas.selectedPlugins["IonReporterUploader"]
                .get("userInput", {})
                .get("accountId")
            )
            iru_config = Plugin.objects.get(
                name="IonReporterUploader", active=True
            ).config
            obj_dict["IR_account"] = find_IRU_account(iru_config, accountId)
        except Exception:
            logger.error(traceback.format_exc())

    return json.dumps(obj_dict, cls=LazyDjangoJSONEncoder)


def mark_plan_transferred(plan, location, username, status):
    # change local plan status and mark it executed, so it can no longer be edited or used for sequencing
    plan.planStatus = "transferred"
    plan.planExecuted = True
    plan.metaData = {
        "username": username,
        "date": time.strftime("%Y_%m_%d_%H_%M_%S"),
        "location": location,
        "msg": status.msg.replace("<p>", "").replace("</p>", " "),
        "error": status.error.replace("<p>", "").replace("</p>", " "),
    }
    plan.save()
    # also update status for experiment obj
    plan.experiment.status = "transferred"
    plan.experiment.save()


def check_for_existing_plan(plan, session, status):
    """ This handles "undo" action: plan was transferred by mistake and user is now trying to return it to original server.
        If plan is found and has status Transferred, delete it so it can be re-created.
    """
    r = session.get(session.api_url + "plannedexperiment/?planGUID=%s" % plan.planGUID)
    r.raise_for_status()

    exists = False
    ret = r.json()
    if len(ret["objects"]) > 0:
        exists = True
        remote_plan = ret["objects"][0]
        if remote_plan["planStatus"] == "transferred":
            # delete the plan so it can be re-transferred
            try:
                r = session.delete(
                    "http://%s%s" % (session.address, remote_plan["resource_uri"])
                )
                r.raise_for_status()
                exists = False
            except Exception as e:
                status.update(
                    error=_general_failure_msg(plan.planDisplayedName, session.server)
                )
                status.update(
                    error=ugettext(
                        "plan_transfer.messages.destination.alreadyexists.unabletodelete"
                    )
                    % {
                        "planDisplayedName": remote_plan["planDisplayedName"],
                        "errormsg": e,
                    }
                )  # 'Planned run %(planDisplayedName)s already exists and cannot be deleted. Details: %(errormsg)s '
        else:
            status.update(
                error=_general_failure_msg(plan.planDisplayedName, session.server)
            )
            status.update(
                error=ugettext("plan_transfer.messages.destination.alreadyexists")
                % {
                    "planDisplayedName": remote_plan["planDisplayedName"],
                    "planStatus": remote_plan["planStatus"],
                }
            )  # 'Planned run %(planDisplayedName)s already exists and has a status of %(planStatus)s. '

    return exists


def setup_plantransfersdksession(server_name):
    # get mesh server and set up authenticated session to use for requests
    try:
        mesh_node = IonMeshNode.objects.filter(active=True).get(name=server_name)
    except Exception:
        raise Exception(
            ugettext("plan_transfer.messages.failure.meshnode.notfound")
            % {"mesh_node_name": server_name}
        )  # 'Unable to obtain login credentials for destination Torrent Server (%(mesh_node_name)s). '

    try:
        s = mesh_node.SetupMeshSDKSession(apikey=mesh_node.apikey_remote)
        # convenient variables for communication
        s.api_url = "http://%s/rundb/api/v1/" % mesh_node.hostname
        s.address = mesh_node.hostname
        s.server = mesh_node.name
        r = s.get(s.api_url + "ionmeshnode/")
        r.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.TooManyRedirects):
        raise Exception(
            ugettext("plan_transfer.messages.failure.meshnode.unreachable")
            % {"mesh_node_name": s.server, "mesh_node_address": s.address}
        )  # 'Connection Error: Torrent Server %(mesh_node_name)s (%(mesh_node_address)s) is unreachable. '
    except requests.exceptions.HTTPError as e:
        if r.status_code == 401:
            msg = ugettext(
                "plan_transfer.messages.failure.meshnode.httperror.401"
            )  # 'Invalid user credentials.'
        else:
            msg = ugettext("plan_transfer.messages.failure.meshnode.httperror") % {
                "mesh_node_name": s.server,
                "mesh_node_address": s.address,
            }  # 'Unable to connect to Torrent Server %(mesh_node_name)s (%(mesh_node_address)s).'
            configure_mesh_link = '<a href="%s" target="_blank">%s</a>' % (
                reverse("configure_mesh"),
                ugettext("global.nav.menu.configure.menu.mesh.label"),
            )
        msg += ugettext(
            "plan_transfer.messages.failure.meshnode.httperror.verifyconfiguration"
        ) % {
            "configure_mesh_page": configure_mesh_link
        }  # '<br>Please visit <a href="/configure/mesh/" target="_blank">Ion Mesh</a> page to make sure remote server connection is established.'
        raise Exception(msg)
    except Exception as e:
        raise Exception(
            ugettext("plan_transfer.messages.failure.meshnode.genericerror")
            % {
                "mesh_node_name": s.server,
                "mesh_node_address": s.address,
                "exceptionMsg": e,
            }
        )  # 'Error: Unable to access Torrent Server %(mesh_node_name)s (%(mesh_node_address)s). Details: %(exceptionMsg)s'

    try:
        # get software version
        r = s.get(s.api_url + "torrentsuite/version")
        version = r.json()["meta_version"]
    except Exception:
        msg = ugettext("plan_transfer.messages.failure.meshnode.versionerror") % {
            "mesh_node_name": s.server,
            "mesh_node_address": s.address,
        }  # 'Error obtaining software version for Torrent Server %(mesh_node_name)s (%(mesh_node_address)s). '
        raise Exception(msg)

    return s, version


def _general_failure_msg(plan_displayed_name, destination_server_name):
    return ugettext("plan_transfer.messages.failure.msg") % {
        "planDisplayedName": plan_displayed_name,
        "destination_server_name": destination_server_name,
    }  # 'Unable to transfer plan %(planDisplayedName)s to Torrent Server %(destination_server_name)s. ' % (plan.planDisplayedName, server_name))


def transfer_plan(plan, serialized, destination_server_name, username):
    """ This function runs on origin TS to initiate plan transfer through the API """

    logger.debug(
        "Transfer plan resource, planName= %s, to %s"
        % (plan.planDisplayedName, destination_server_name)
    )

    status = Status()
    general_failure_msg = _general_failure_msg(
        plan.planDisplayedName, destination_server_name
    )

    # Make sure transfer is allowed for this plan status
    if plan.planStatus == "reserved" or plan.planStatus == "run":
        status.update(error=general_failure_msg)
        status.update(
            error=ugettext("plan_transfer.messages.validation.invalidstatus")
            % {
                "planDisplayedName": plan.planDisplayedName,
                "planStatus": plan.planStatus,
            }
        )  # 'Error: Planned run %(planDisplayedName)s has status of %(planStatus)s and cannot be transferred.'
        return status.to_dict()

    # set up communication
    ptsdk_session, version = setup_plantransfersdksession(destination_server_name)
    if version != TS_VERSION and not os.path.exists("/opt/ion/.ion-internal-server"):
        status.update(error=general_failure_msg)
        status.update(
            error=ugettext("plan_transfer.messages.validation.invalidversion")
            % {
                "source_server_version": TS_VERSION,
                "destination_server_name": destination_server_name,
                "destination_server_version": version,
            }
        )  # 'Error: Torrent Suite version %(source_server_version)s does not match destination Torrent Server %(destination_server_name)s software version %(destination_server_version)s. '
        return status.to_dict()

    if debug:
        starttime = time.time()

    # Check if a plan already exists on destination
    exists = check_for_existing_plan(plan, ptsdk_session, status)
    if exists:
        # plan exists and cannot be deleted, break early and return any errors
        return status.to_dict()

    if debug:
        logger.debug(
            "%f s: Plan Transfer check_for_existing_plan" % (time.time() - starttime)
        )
        starttime = time.time()

    # copy Plan/Experiment/EAS through plannedexperiment API
    r = ptsdk_session.post(
        ptsdk_session.api_url + "plannedexperiment/", data=serialized
    )
    response = r.json()
    # handle unsuccessful POST
    if not r.ok:
        try:
            status.update(
                error=general_failure_msg
            )  # 'Unable to transfer plan %s to Torrent Server %s.' % (plan.planDisplayedName, server_name))
            # parse validation errors
            errjson = response["error"]
            for k, v in list(errjson.items()):
                v_errmsg = parse_to_string(v)
                status.update(
                    error="%s %s"
                    % (ugettext("validation.messages.error_prefix"), v_errmsg)
                )
            return status.to_dict()
        except Exception:
            logger.error(
                "Error while attempting to transfer plan %s", traceback.format_exc()
            )
            r.raise_for_status()

    new_plan_url = r.headers["location"]

    if "Warnings" in response:
        try:
            parsed = []
            for key, warning in list(response["Warnings"].items()):
                parsed.append(key + ":" + parse_to_string(warning))
        except Exception:
            logger.error("Unable to parse warnings from API response")
            logger.error(traceback.format_exc())

        status.update(warning=" ".join(parsed))

    if debug:
        logger.debug(
            "%f s: Plan Transfer POST %s/plannedexperiment/"
            % (time.time() - starttime, ptsdk_session.api_url)
        )

    planlink = "http://%s/plan/planned/" % ptsdk_session.address
    status.update(
        msg=ugettext("plan_transfer.messages.successgmsg")
        % {
            "planDisplayedName": plan.planDisplayedName,
            "destination_plan_url": planlink,
            "destination_server_name": destination_server_name,
        }
    )  # 'Successfully created %(planDisplayedName)s on destination Torrent Server <a href="%(destination_plan_url)s" target="_blank">%(destination_server_name)s</a>' % (plan.planDisplayedName, planlink, server_name))

    # send get transfer request to destination TS, this will do postprocessing and return status/errors
    objJson = get_associated_objects_json(plan)

    if debug:
        starttime = time.time()

    postprocessing_url = new_plan_url + "transfer/"
    editlink = "http://%s/plan/page_plan_edit_plan/%s/" % (
        ptsdk_session.address,
        postprocessing_url.split("/")[-2],
    )
    edit_plan_page = '<a href="%s" target="_blank">%s</a>' % (
        editlink,
        ugettext("plannedruns.action.page_plan_edit_plan"),
    )
    try:
        r = ptsdk_session.get(postprocessing_url, data=objJson)
        r.raise_for_status()
        ret = r.json()
        errors = ret.get("error")
        if errors:
            # Translators: The errors discovered during Plan Transfer postprocessing will be displayed after this message.
            fix_with_errors = ugettext(
                "plan_transfer.messages.validation.postprocessing.data.incomplete"
            ) % {
                "planDisplayedName": plan.planDisplayedName,
                "edit_destination_plan_page": edit_plan_page,
            }  # 'The transferred Planned run %(planDisplayedName)s has incomplete data configuration after transfer. '
            fix_with_errors += ugettext(
                "plan_transfer.messages.validation.postprocessing.data.incomplete.fix.witherrors"
            ) % {
                "edit_destination_plan_page": edit_plan_page
            }  # 'Please click %(edit_plan_page)s to fix the following errors: '
            status.update(error=fix_with_errors)
        status.update(msg=ret.get("msg"), error=errors)
    except Exception as err:
        status.update(
            error=ugettext("plan_transfer.messages.validation.postprocessing.error")
            % {
                "mesh_node_name": ptsdk_session.server,
                "mesh_node_address": ptsdk_session.address,
                "exceptionMsg": err,
            }
        )  # 'Error: Unable to update plan on destination Torrent Server %(mesh_node_name)s (%(mesh_node_address)s). Details: %(exceptionMsg)s '
        fix_msg = ugettext(
            "plan_transfer.messages.validation.postprocessing.data.incomplete"
        ) % {
            "planDisplayedName": plan.planDisplayedName
        }  # 'The transferred Planned run %(planDisplayedName)s has incomplete data configuration after transfer. '
        fix_msg += ugettext(
            "plan_transfer.messages.validation.postprocessing.data.incomplete.fix"
        ) % {
            "edit_destination_plan_page": edit_plan_page
        }  # 'Please click %(edit_plan_page)s to complete the transfer. '
        status.update(error=fix_msg)
        logger.error("Failed to update plan for %s(%s)" % (plan.planName, plan.pk))
        logger.error(traceback.format_exc())

    if debug:
        logger.debug(
            "%f s: Plan Transfer GET %s"
            % (time.time() - starttime, new_plan_url + "transfer/")
        )

    # update local plan
    mark_plan_transferred(plan, new_plan_url, username, status)

    return status.to_dict()
