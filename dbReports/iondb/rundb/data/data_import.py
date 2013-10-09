#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import logging
import traceback
import json
import shutil
import time
import subprocess
from glob import glob
from celery import task
from iondb.bin.djangoinit import *
from iondb.rundb.models import Experiment, DMFileSet, ReportStorage, Location, Message, EventLog
from iondb.settings import RELVERSION
from django.core import serializers
from django.core.urlresolvers import reverse
from django.utils import timezone

from iondb.rundb.data import dmactions_types
from iondb.rundb.data.dmactions import _file_selector, _copy_to_dir

logger = logging.getLogger()

def update_Result(saved_objs):
    # Set report storage and link
    result = saved_objs['results']
    storage = ReportStorage.objects.filter(default=True)[0]
    location = Location.objects.all()[0]
    result.reportstorage = storage
    result.reportLink = os.path.join(storage.webServerPath, location.name, "%s_%03d" % (result.resultsName, result.pk), "")
    result.sffLink = os.path.join(result.reportLink, os.path.basename(result.sffLink))
    
    # update metrics FKs on Results
    result.analysismetrics = saved_objs['analysis metrics']
    result.libmetrics = saved_objs['lib metrics']
    result.qualitymetrics = saved_objs['quality metrics']

    # set status
    result.status = 'Importing'
    result.save()

def update_DMFileStats(result, data):
    dmfilestats = result.dmfilestat_set.all()
    # set state to Deleted, will be updated later when files are copied
    dmfilestats.update(action_state='DD')
    # set correct DMFileSet version
    for d in json.loads(data):
        if 'dmfileset' in d['model']:
            dmtype = d['fields']['type']
            version = d['fields']['version']
            if version != RELVERSION:
                try:
                    dmfileset = DMFileSet.objects.get(version=version, type=dmtype)
                    dmfilestats.filter(dmfileset__type=dmtype).update(dmfileset=dmfileset)
                except Exception as e:
                    pass

def create_obj(name, data, saved_objs):
    # strip object pk and replace relations and DateTimeField then save
    for deserialized_obj in serializers.deserialize('json', data):
        obj = deserialized_obj.object
        if name == obj._meta.verbose_name:
            obj.pk = None
            for field in obj._meta.fields:
                field_type = field.get_internal_type()

                if field_type == 'DateTimeField':
                    setattr(obj, field.name, timezone.now() )
                
                if field_type in ['ForeignKey', 'OneToOneField', 'ManyToManyField']:
                    relation_name = field.related.parent_model._meta.verbose_name
                    if relation_name in saved_objs.keys():
                        setattr(obj, field.name, saved_objs[relation_name] )
                    elif field.null:
                        setattr(obj, field.name, None )

            obj.save()
            logger.debug('[Data Import] SAVED %s pk=%s' % (name, obj.pk) )
            
            return obj
    else:
        logger.warning( '[Data Import] No object %s found' % name )

def load_serialized_json(json_path, log, create_result):
    # creates DB objects for Plan, Experiment, Result etc.
    # serialized.json file is created during DataManagement export/archive and contains objects to be imported
    # Note that DB objects must be saved in specific sequence for correct relations to be set up
    log.write('Creating database objects ...\n')
    
    with open(json_path) as f:
        data = f.read()
    
    create_sequence = []
    saved_objs = {}
    
    # skip creating experiment if it already exists
    for d in json.loads(data):
        if d['model'] == 'rundb.experiment':
            unique = d['fields']['unique']
            exp = Experiment.objects.filter(unique=unique)
            if exp:
                log.write('Found existing experiment %s.\n' % exp[0].expName)
                saved_objs['experiment'] = exp[0]
                saved_objs['planned experiment'] = exp[0].plan
            else:
                create_sequence += ['planned experiment', 'experiment']
    
    if create_result:
        create_sequence += ['experiment analysis settings', 'results', 'analysis metrics', 'lib metrics', 'quality metrics']
        # ?? want import plugin result if plugin of that name found on this TS
    else:
        create_sequence += ['experiment analysis settings']

    # create the records
    for name in create_sequence:
        try:
            obj = create_obj(name, data, saved_objs)
            saved_objs[name] = obj
        except:
            logger.error('[Data Import] Unable to create %s from %s.' % (name, json_path) )
            logger.error(traceback.format_exc())
            log.write('Failed to import %s.\n' % name)
    
    if create_result:
        # additional things to update for new Results
        try:
            update_Result(saved_objs)
            update_DMFileStats(saved_objs['results'], data)
        except:
            logger.error('[Data Import] Update Result failed.')
            logger.error(traceback.format_exc())
            log.write('Failed to update Result.\n')
    
    log.write('Creating database objects done.\n')
    return saved_objs


def copy_files_to_destination(source_dir, destination, dmfileset, log):
    
    to_process, to_keep = _file_selector(source_dir, dmfileset.include, dmfileset.exclude, [])
    logger.debug('[Data Import] Importing %d files from %s to %s' % (len(to_process), source_dir, destination) )
    log.write('Copy files to %s ...\n' % destination)
    for i, filepath in enumerate(to_process):
        log.write('%s.\n' % filepath)
        _copy_to_dir(filepath, source_dir, destination)

    if dmfileset.type == dmactions_types.OUT:
        # make sure we have plugin_out folder
        plugin_out = os.path.join(destination, 'plugin_out')
        if not os.path.isdir(plugin_out):
            oldmask = os.umask(0000)   #grant write permission to plugin user
            os.mkdir(plugin_out)
            os.umask(oldmask)
    
    log.write('Copy files done.\n')


@task(queue='transfer')
def data_import(name, selected, username, copy_all=False):
    ''' Data import main task.
        Selected dict contains categories to import and path to their serialized json
        Log file used to display progress on webpage.
        Don't copy Signal Processing or Basecalling Input files to local drive unless copy_all=True
    '''
    logfile = os.path.join('/tmp', 'di_log_%s_%f' % (name, time.time()) )
    
    importReport = dmactions_types.OUT in selected.keys()
    importBasecall = dmactions_types.BASE in selected.keys()
    importSigproc = dmactions_types.SIG in selected.keys()
    createReport = importReport or importBasecall # don't create Results objects if only sigproc imported
    
    if importReport:
        json_path = selected[dmactions_types.OUT]
    else:
        json_path = selected.values()[0]
    
    logger.info('[Data Import] (%s) Started import %s using %s.' % (name, ', '.join(selected.keys()), json_path) )
    log = open(logfile, 'w', 0)
    log.write('(%s) Import of selected categories started.\n' % name)
    
    msg_banner(name, selected.keys(), logfile, 'Started')
    
    # create DB records
    try:
        objs = load_serialized_json(json_path, log, createReport)
        result = objs.get('results', None)
        exp = objs['experiment']
    except:
        msg = traceback.format_exc()
        logger.error(msg)
        log.write(msg)
        msg_banner(name, selected.keys(), logfile, 'Error')

    # process files
    for category, path in selected.items():
        source_dir = os.path.dirname(path)
        log.write('Importing %s from %s.\n' % (category, source_dir ) )
        logger.debug('(%s) Importing %s from %s.' % (name, category, source_dir ) )
        
        if category == dmactions_types.OUT:
            destination = result.get_report_dir()
        elif category == dmactions_types.BASE and copy_all:
            destination = exp.expDir if 'onboard_results' in source_dir else result.get_report_dir()
        elif category == dmactions_types.SIG and copy_all:
            destination = exp.expDir
        else:
            destination = None

        if result:
            dmfilestat = result.get_filestat(category)
            dmfileset = dmfilestat.dmfileset
        else:
            dmfileset = DMFileSet.objects.get(version=RELVERSION, type=category)

        # copy files
        if destination:
            try:
                copy_files_to_destination(source_dir, destination, dmfileset, log)
            except:
                msg = traceback.format_exc()
                logger.error(msg)
                log.write(msg)
                msg_banner(name, selected.keys(), logfile, 'Error')
        
        # update dmfilestats
        if result:
            if destination:
                dmfilestat.action_state = 'L'
            else:
                # data files left on media, need to update dmfilestat to archived location
                dmfilestat.action_state = 'AD'
                dmfilestat.archivepath=source_dir
            dmfilestat.save()
    
    # finish up
    if result:
        EventLog.objects.add_entry(result, "Imported from %s." % os.path.dirname(json_path), username)
        result.status = 'Completed'
        result.save()
    log.write('FINISHED Import of selected categories.\n')
    log.close()
    
    msg_banner(name, selected.keys(), logfile, 'Completed', importReport, result, exp)
    
    # save the temp log file to destination
    if destination:
        shutil.copy(logfile, os.path.join(destination,'import_data.log') )

    logger.info('(%s) Done.' % name)    


def find_data_to_import(start_dir):
    musthave = [
        {
            'category': dmactions_types.SIG,
            'filename': 'acq_0000.dat',
            'search_paths': ['', 'block_X*_Y*']
        },
        {
            'category': dmactions_types.BASE,
            'filename': '1.wells',
            'search_paths': ['', 'sigproc_results', 'sigproc_results/block_X*_Y*', 'onboard_results/sigproc_results/block_X*_Y*']
        },
        {
            'category': dmactions_types.OUT,
            'filename': 'ion_params_00.json',
            'search_paths': ['']
        }
    ]

    found = {}
    
    cmd = ['find', start_dir, '-type', 'f', '-name', 'serialized_*.json']
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for match in p1.stdout.readlines():
        # for each serialized json, test for necessary files by category
        json_path = match.strip()
        root, filename = os.path.split(json_path)
        resultsName = filename.split('serialized_')[1].split('.json')[0]
        for test in musthave:
            for path in test['search_paths']:
                if glob(os.path.join(root, path, test['filename'])):
                    found.setdefault(resultsName, {}).update({ test['category']: json_path })
                    break
    found_results = [{'name':k,'categories':v} for k,v in found.items()]
    return sorted(found_results, key=lambda r:r['name'].lower())

def msg_banner(name, categories, logfile, status, importReport=None, result=None, exp=None):
    msg = '(%s) Import %s, %s' % (name,  ', '.join(categories), status)
    
    if status == 'Completed':
        if importReport:
            logurl = reverse('dm_log', args=(result.pk,))
            msg += " <a href='%s' data-toggle='modal' data-target='#modal_report_log'>View Report Log</a>" % (logurl)
            reporturl = reverse('report', args=(result.pk,))
            msg += " Imported Report: <a href='%s'>%s</a>'" % (reporturl, result.resultsName)
        else:
            msg += " Imported Run available for analysis: <a href='/data/'>%s</a>" % exp.expName
    
    Message.objects.filter(tags=logfile).delete()
    if status == 'Error':
        errlog = reverse('import_data_log', args=(logfile,))
        msg += " <a href='%s' data-toggle='modal' data-target='#modal_report_log'>Error Log</a>" % (errlog)
        Message.error(msg, tags=logfile)
    else:
        Message.info(msg, tags=logfile)
