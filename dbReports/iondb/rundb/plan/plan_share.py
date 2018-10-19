# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import os
import requests
import traceback
import logging
import json
import time

from django.conf import settings
from iondb.rundb.models import IonMeshNode, PlannedExperiment, PlannedExperimentQC, QCType, dnaBarcode, User, \
    Sample, SampleSet, SampleSetItem, Content, Plugin, EventLog, SamplePrepData
from django.core.serializers.json import DjangoJSONEncoder

from iondb.plugins.launch_utils import find_IRU_account
from ion import version as TS_VERSION
from iondb.utils.hostip import gethostname

logger = logging.getLogger(__name__)

debug = False


class Status:

    def __init__(self):
        self.msg = ''
        self.error = ''

    def update(self, msg='', error=''):
        if msg:
            self.msg += '<p>%s</p>' % msg
        if error:
            self.error += '<p>%s</p>' % error

    def to_dict(self):
        return {'msg': self.msg, 'error': self.error}

    def add_plan_history(self, plan, username):
        # update plan History log
        if self.msg:
            text = 'Plan Transfer: \n' + self.msg.replace('<p>', '').replace('</p>', '\n')
            EventLog.objects.add_entry(plan, text, username)
        if self.error:
            text = 'Plan Transfer Errors: \n' + self.error.replace('<p>', '').replace('</p>', '\n')
            EventLog.objects.add_entry(plan, text, username)

def parse_to_string(data, output=""):
    if isinstance(data, dict):
        for value in data.values():
            output = parse_to_string(value, output)
    elif isinstance(data, list):
        data.sort()
        for value in data:
            output = parse_to_string(value, output)
    elif isinstance(data, basestring) and data not in output:
        return output + ' ' + data

    return output

''' Plan share Destination TS functions '''


def create_associated_objects(status, plan, obj_dict, user):

    if debug:
        starttime = time.time()

    # Samples
    if 'samples' in obj_dict:
        try:
            for sample_dict in obj_dict['samples']:
                sample, created = Sample.objects.get_or_create(name=sample_dict['name'], externalId=sample_dict['externalId'], defaults=sample_dict)
                sample.experiments.add(plan.experiment)

            status.update(msg='....processed Samples: '+', '.join(s['displayedName'] for s in obj_dict['samples']))
        except Exception as err:
            logger.error('Error processing Samples for %s(%s)' % (plan.planName, plan.pk))
            logger.error(traceback.format_exc())
            status.update(error='Error processing samples: %s' % err)

    # SampleSet
    for set_dict in obj_dict['sampleSets']:
        try:
            set_dict['creator_id'] = set_dict['lastModifiedUser_id'] = user.pk
            libraryPrep_dict = set_dict.pop('libraryPrepInstrumentData', '')
            samplesetitems = set_dict.pop('sampleSetItems', [])

            sampleSet, created = SampleSet.objects.get_or_create(displayedName=set_dict['displayedName'], defaults=set_dict)
            sampleSet.plans.add(plan)

            for setitem_dict in samplesetitems:
                sample = plan.experiment.samples.get(name=setitem_dict.pop('sample__name'))
                if 'dnabarcode' in setitem_dict:
                    barcode = setitem_dict.pop('dnabarcode')
                    dnabarcode = dnaBarcode.objects.filter(name=barcode['name'], id_str=barcode['id_str'])
                    setitem_dict['dnabarcode'] = dnabarcode[0] if dnabarcode else None

                setitem_dict['creator_id'] = setitem_dict['lastModifiedUser_id'] = user.pk
                item, created = SampleSetItem.objects.get_or_create(sample=sample, sampleSet=sampleSet, defaults=setitem_dict)

            if libraryPrep_dict:
                if not sampleSet.libraryPrepInstrumentData:
                    sampleSet.libraryPrepInstrumentData = SamplePrepData.objects.create(**libraryPrep_dict)
                    sampleSet.save()
                else:
                    for field, value in libraryPrep_dict.items():
                        setattr(sampleSet.libraryPrepInstrumentData, field, value)
                    sampleSet.libraryPrepInstrumentData.save()

            status.update(msg='....processed SampleSet: %s' % sampleSet.displayedName)
        except Exception as err:
            logger.error('Error processing sampleSet for %s(%s)' % (plan.planName, plan.pk))
            logger.error(traceback.format_exc())
            status.update(error='Error processing sampleSet: %s' % err)

    if debug:
        logger.debug('%f s: Plan Transfer create_associated_objects' % (time.time()-starttime))
        starttime = time.time()

    return True


def update_transferred_plan(plan, request):
    ''' This function that runs on destination TS to update plan-related objects '''
    if debug:
        starttime = time.time()

    # update plan History log
    log = 'Transferred Planned Run: %s from %s.' % (plan.planDisplayedName, plan.metaData.get('origin'))
    EventLog.objects.add_entry(plan, log, request.user.username)

    obj_dict = json.loads(request.body)

    status = Status()
    eas = plan.latest_eas

    # create Samples, etc.
    create_associated_objects(status, plan, obj_dict, request.user)

    # Ion Reporter account id needs to be updated
    if 'IonReporterUploader' in eas.selectedPlugins:
        accountId = None
        try:
            irserver = obj_dict['IR_account']['server']
            irversion = obj_dict['IR_account']['version']
            irtoken = obj_dict['IR_account']['token']
            irname = obj_dict['IR_account']['name']

            userconfigs = Plugin.objects.get(name='IonReporterUploader', active=True).config['userconfigs'][plan.username]
            for config in userconfigs:
                if irserver == config['server'] and irversion == config['version'] and irtoken == config['token']:
                    accountId = config['id']
                    accountName = config['name']
                    break
        except:
            pass

        if accountId:
            userInput = eas.selectedPlugins['IonReporterUploader']['userInput']
            userInput['accountId'] = accountId
            if irname != accountName:
                userInput['accountName'] = userInput['accountName'].replace(irname, accountName)
            status.update(msg='....found IR account %s' % userInput['accountName'])
        else:
            eas.selectedPlugins.pop('IonReporterUploader')
            status.update(error='Error: IonReporter account not found. Please add IR account on destination Server and update the Planned run.')

    eas.save()

    status.add_plan_history(plan, request.user.username)

    if debug:
        logger.debug('%f s: Plan Transfer update_transferred_plan %s' % ((time.time()-starttime), plan.planDisplayedName))

    return status.to_dict()


''' Plan share Origin TS functions '''


def prepare_for_copy(bundle):

    # remove obj keys that need to be recreated
    bundle.data.pop('id')
    bundle.data.pop('experiment')
    bundle.data.pop('sampleSets')

    # qcValues
    qcValues = bundle.data.pop('qcValues', [])
    try:
        for qc in qcValues:
            bundle.data[qc.obj.qcType.qcName] = qc.obj.threshold
    except:
        logger.error(traceback.format_exc())

    # indicate this plan's origin
    bundle.data['origin'] = 'transfer'
    bundle.data['metaData']['origin'] = gethostname()
    bundle.data['metaData']['uri'] = bundle.data.pop('resource_uri')
    return bundle


def get_associated_objects_json(plan):
    ''' Gather associated objects to send to destination '''
    def get_obj_dict(obj):
        d = {}
        for field in obj._meta.fields:
            if field.get_internal_type() not in ['AutoField', 'ForeignKey', 'OneToOneField', 'ManyToManyField']:
                d[field.name] = getattr(obj, field.name)
        return d

    obj_dict = {}

    # Samples
    samples = plan.experiment.samples.all()
    obj_dict['samples'] = []
    for sample in samples:
        d = get_obj_dict(sample)
        obj_dict['samples'].append(d)

    # SampleSet
    obj_dict['sampleSets'] = []
    for sampleSet in plan.sampleSets.all():
        sampleSet_dict = get_obj_dict(sampleSet)
        sampleSet_dict['SampleGroupType_CV_id'] = sampleSet.SampleGroupType_CV_id
        sampleSet_dict['libraryPrepInstrumentData'] = get_obj_dict(sampleSet.libraryPrepInstrumentData) if sampleSet.libraryPrepInstrumentData else {}

        sampleSet_dict['sampleSetItems'] = []
        for setitem in sampleSet.samples.filter(sample__in=samples):
            setitem_dict = get_obj_dict(setitem)
            setitem_dict['sample__name'] = setitem.sample.name
            if setitem.dnabarcode:
                setitem_dict['dnabarcode'] = {
                    'name': setitem.dnabarcode.name,
                    'id_str': setitem.dnabarcode.id_str
                }
            sampleSet_dict['sampleSetItems'].append(setitem_dict)

        obj_dict['sampleSets'].append(sampleSet_dict)

    # Ion Reporter account
    eas = plan.latest_eas
    if 'IonReporterUploader' in eas.selectedPlugins:
        try:
            accountId = eas.selectedPlugins['IonReporterUploader'].get('userInput', {}).get('accountId')
            iru_config = Plugin.objects.get(name='IonReporterUploader', active=True).config
            obj_dict['IR_account'] = find_IRU_account(iru_config, accountId)
        except:
            logger.error(traceback.format_exc())

    return json.dumps(obj_dict, cls=DjangoJSONEncoder)


def mark_plan_transferred(plan, location, username, status):
    # change local plan status and mark it executed, so it can no longer be edited or used for sequencing
    plan.planStatus = 'transferred'
    plan.planExecuted = True
    plan.metaData = {
        'username': username,
        'date': time.strftime("%Y_%m_%d_%H_%M_%S"),
        'location': location,
        'msg': status.msg.replace('<p>', '').replace('</p>', ' '),
        'error': status.error.replace('<p>', '').replace('</p>', ' ')
    }
    plan.save()
    # also update status for experiment obj
    plan.experiment.status = 'transferred'
    plan.experiment.save()


def check_for_existing_plan(plan, session, status):
    ''' This handles "undo" action: plan was transferred by mistake and user is now trying to return it to original server.
        If plan is found and has status Transferred, delete it so it can be re-created.
    '''
    r = session.get(session.api_url + 'plannedexperiment/?planGUID=%s' % plan.planGUID)
    r.raise_for_status()

    exists = False
    ret = r.json()
    if len(ret['objects']) > 0:
        exists = True
        remote_plan = ret['objects'][0]
        if remote_plan['planStatus'] == 'transferred':
            # delete the plan so it can be re-transferred
            try:
                r = session.delete('http://%s%s' % (session.address, remote_plan['resource_uri']))
                r.raise_for_status()
                exists = False
            except Exception as e:
                status.update(error='Error: Unable to transfer %s to Torrent Server %s' % (plan.planDisplayedName, session.server))
                status.update(error='Planned run %s already exists and cannot be deleted: %s' % (remote_plan['planDisplayedName'], e))
        else:
            status.update(error='Error: Unable to transfer %s to Torrent Server %s' % (plan.planDisplayedName, session.server))
            status.update(error='Planned run %s already exists and has status= %s' % (remote_plan['planDisplayedName'], remote_plan['planStatus']))

    return exists


def setup_session(server_name):
    # get mesh server and set up authenticated session to use for requests
    try:
        mesh_node = IonMeshNode.objects.filter(active=True).get(name=server_name)
    except:
        raise Exception('Unable to get login credentials for destination Torrent Server: %s' % server_name)

    try:
        s = requests.Session()
        # convenient variables for communication
        s.api_url = 'http://%s/rundb/api/v1/' % mesh_node.hostname
        s.address = mesh_node.hostname
        s.server = mesh_node.name
        # set up session
        s.params = {
            "api_key": mesh_node.apikey_remote,
            "system_id": settings.SYSTEM_UUID
        }
        r = s.get(s.api_url + 'ionmeshnode/')
        r.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.TooManyRedirects):
        raise Exception('Connection Error: Torrent Server %s (%s) is unreachable' % (s.server, s.address))
    except requests.exceptions.HTTPError as e:
        if r.status_code == 401:
            msg = 'Invalid user credentials.'
        else:
            msg = 'Unable to connect to Torrent Server %s (%s).' % (s.server, s.address)
        msg += '<br>Please visit <a href="/configure/mesh/" target="_blank">Ion Mesh</a> page to make sure remote server connection is established.'
        raise Exception(msg)
    except Exception as e:
        raise Exception('Error: Unable to access Torrent Server %s (%s): %s' % (s.server, s.address, e))

    try:
        # get software version
        r = s.get(s.api_url + 'torrentsuite/version')
        version = r.json()['meta_version']
    except:
        msg = 'Error getting software version for Torrent Server %s (%s). ' % (s.server, s.address)
        raise Exception(msg)

    return s, version


def transfer_plan(plan, serialized, server_name, username):
    ''' This function runs on origin TS to initiate plan transfer through the API '''

    logger.debug('Transfer plan resource, planName= %s, to %s' % (plan.planDisplayedName, server_name))

    status = Status()

    # Make sure transfer is allowed for this plan status
    if plan.planStatus == 'reserved' or plan.planStatus == 'run':
        status.update(error='Error: Planned run %s has status= %s and cannot be transferred.' % (plan.planDisplayedName, plan.planStatus))
        return status.to_dict()

    # set up communication
    session, version = setup_session(server_name)
    if version != TS_VERSION and not os.path.exists('/opt/ion/.ion-internal-server'):
        status.update(error='Unable to transfer plan: Torrent Suite version %s does not match %s software version %s.' % (TS_VERSION, server_name, version))
        return status.to_dict()

    if debug:
        starttime = time.time()

    # Check if a plan already exists on destination
    exists = check_for_existing_plan(plan, session, status)
    if exists:
        # plan exists and cannot be deleted, break early and return any errors
        return status.to_dict()

    if debug:
        logger.debug('%f s: Plan Transfer check_for_existing_plan' % (time.time()-starttime))
        starttime = time.time()

    # copy Plan/Experiment/EAS through plannedexperiment API
    r = session.post(session.api_url + 'plannedexperiment/', data=serialized)
    response = r.json()
    # handle unsuccessful POST
    if not r.ok:
        try:
            status.update(error='Unable to transfer plan %s to Torrent Server %s.' % (plan.planDisplayedName, server_name))
            # parse validation errors
            errjson = json.loads(response['error'][3:-2])
            for k, v in errjson.items():
                status.update(error='Error: %s' % (json.dumps(v)))
            return status.to_dict()
        except:
            r.raise_for_status()

    new_plan_url = r.headers['location']

    if 'Warnings' in response:
        try:
            parsed = []
            for key, warning in response['Warnings'].items():
                parsed.append(key + ':' + parse_to_string(warning))
        except:
            logger.error("Unable to parse warnings from API response")
            logger.error(traceback.format_exc())

        status.update(error= ' '.join(parsed))

    if debug:
        logger.debug('%f s: Plan Transfer POST %s/plannedexperiment/' % (time.time()-starttime, session.api_url))

    planlink = 'http://%s/plan/planned/' % session.address
    status.update(msg='Successfully created %s on Torrent Server <a href="%s" target="_blank">%s</a>' % (plan.planDisplayedName, planlink, server_name))

    # send get transfer request to destination TS, this will do postprocessing and return status/errors
    objJson = get_associated_objects_json(plan)

    if debug:
        starttime = time.time()

    try:
        r = session.get(new_plan_url+'transfer/', data=objJson)
        r.raise_for_status()
        ret = r.json()
        errors = ret.get('error')
        if errors:
            editlink = 'http://%s/plan/page_plan_edit_plan/%s/' % (session.address, new_plan_url.split('/')[-2])
            status.update(error='Planned run data is incomplete, please <a href="%s" target="_blank">Edit %s</a> to fix the following errors' % (editlink, plan.planDisplayedName))
        status.update(ret.get('msg'), errors)
    except Exception as err:
        status.update(error='Error: Unable to update plan on Torrent Server %s: %s' % (server_name, err))
        status.update(error='Planned run data is incomplete, please Edit %s on destination Server to complete the transfer' % plan.planDisplayedName)
        logger.error('Failed to update plan for %s(%s)' % (plan.planName, plan.pk))
        logger.error(traceback.format_exc())

    if debug:
        logger.debug('%f s: Plan Transfer GET %s' % (time.time()-starttime, new_plan_url+'transfer/'))

    # update local plan
    mark_plan_transferred(plan, new_plan_url, username, status)

    return status.to_dict()
