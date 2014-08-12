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
import urllib
from glob import glob
from celery import task
from iondb.bin.djangoinit import *
from iondb.rundb.models import Experiment, DMFileSet, ReportStorage, Location, Message, EventLog, FileServer, Content
from iondb.settings import RELVERSION
from django.core import serializers
from django.core.urlresolvers import reverse
from django.utils import timezone
from django.db.models import get_model
from django.db import transaction

from iondb.rundb.data import dmactions_types
from iondb.rundb.data.dmactions import _file_selector, _copy_to_dir, get_walk_filelist

logger = logging.getLogger()

IMPORT_RIG_FOLDER = 'Imported'

def update_Result(saved_objs):
    # Set report storage and link
    result = saved_objs['results']
    storage = ReportStorage.objects.filter(default=True)[0]
    location = Location.getdefault()
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

def update_Plan(saved_objs):
    
    def find_bedfile(filename, reference):
        content_objs = Content.objects.filter(publisher__name="BED", path__contains="/unmerged/detail/") \
            .filter(path__contains=reference).filter(path__contains=filename)
        for filepath in content_objs.values_list('file', flat=True):
            if filename == os.path.basename(filepath):
                return filepath

        logger.debug('[Data Import] Update Plan: unable to find bedfile: %s for reference: %s' % (filename,reference))
        return ''
    
    eas = saved_objs['experiment analysis settings']
    
    # if any BED files were selected, attempt to find them and update location    
    if getattr(eas,'targetRegionBedFile','') and not os.path.exists(eas.targetRegionBedFile):
        eas.targetRegionBedFile = find_bedfile(os.path.basename(eas.targetRegionBedFile), eas.reference)
        eas.save()
    if getattr(eas,'hotSpotRegionBedFile','') and not os.path.exists(eas.hotSpotRegionBedFile):
        eas.hotSpotRegionBedFile = find_bedfile(os.path.basename(eas.hotSpotRegionBedFile), eas.reference)
        eas.save()

def update_DMFileStats(result, data):
    dmfilestats = result.dmfilestat_set.all()
    dmfilestats.update(action_state='IG')
    # set correct DMFileSet version
    for d in data:
        if 'dmfileset' in d['model']:
            dmtype = d['fields']['type']
            version = d['fields']['version']
            if version != RELVERSION:
                try:
                    dmfileset = DMFileSet.objects.get(version=version, type=dmtype)
                    dmfilestats.filter(dmfileset__type=dmtype).update(dmfileset=dmfileset)
                except Exception as e:
                    pass

def create_obj(model_name, data, saved_objs):
    # strip object pk and replace relations and DateTimeField then save
    app_model_string = 'rundb' + '.' + model_name.replace(' ', '')
    model = get_model(*app_model_string.split('.'))
    for d in data:
        if d['model'] == app_model_string:
            data_fields = d['fields']
            obj = model()
            for field in obj._meta.fields:
                field_type = field.get_internal_type()

                if field_type == 'DateTimeField':
                    setattr(obj, field.name, timezone.now() )
                elif field_type in ['ForeignKey', 'OneToOneField', 'ManyToManyField']:
                    relation_name = field.related.parent_model._meta.verbose_name
                    if relation_name in saved_objs.keys():
                        setattr(obj, field.name, saved_objs[relation_name] )
                    elif field.null:
                        setattr(obj, field.name, None )
                else:
                    if field.name in data_fields:
                        setattr(obj, field.name, data_fields[ field.name ] )

            obj.save()
            logger.debug('[Data Import] SAVED %s pk=%s' % (model_name, obj.pk) )

            return obj


def load_serialized_json(json_path, log, create_result):
    # creates DB objects for Plan, Experiment, Result etc.
    # serialized.json file is created during DataManagement export/archive and contains objects to be imported
    # Note that DB objects must be saved in specific sequence for correct relations to be set up
    log.write('Importing database objects ...\n')
    logger.debug('[Data Import] Creating database objects from %s.' % json_path)

    with open(json_path) as f:
        data = json.load(f)

    create_sequence = []
    saved_objs = {}

    # skip creating experiment if it already exists
    exp_fields = [d['fields'] for d in data if d['model'] == 'rundb.experiment'][0]
    unique = exp_fields['unique']
    unique_imported = os.path.join(_get_imported_path(), os.path.basename(exp_fields['expDir']) )
   
    exp = Experiment.objects.filter(unique__in=[unique,unique_imported])
    if exp:
        log.write('Found existing experiment %s (%s).\n' % (exp[0].expName, exp[0].pk))
        saved_objs['experiment'] = exp[0]
        saved_objs['planned experiment'] = exp[0].plan
    else:
        create_sequence += ['planned experiment', 'experiment']
        # import raw data into separate location
        if IMPORT_RIG_FOLDER:
            exp_fields['expDir'] = unique_imported
            exp_fields['unique'] = unique_imported
            exp_fields['pgmName'] = IMPORT_RIG_FOLDER
    
    # skip creating result if it already exists
    if create_result:
        result_fields = [d['fields'] for d in data if d['model'] == 'rundb.results'][0]
        result = saved_objs['experiment'].results_set.filter(runid=result_fields['runid']) if 'experiment' in saved_objs else None
        if result:
            log.write('Found existing result %s (%s).\n' % (result[0].resultsName, result[0].pk))
            saved_objs['results'] = result[0]
            saved_objs['experiment analysis settings'] = result[0].eas
        else:
            create_sequence += ['experiment analysis settings', 'results', 'analysis metrics', 'lib metrics', 'quality metrics']
    else:
        create_sequence += ['experiment analysis settings']

    # create the records
    log.write(' create objects: %s\n' % (', '.join(create_sequence) or 'None'))
    try:
        with transaction.commit_on_success():
            for model_name in create_sequence:
                obj = create_obj(model_name, data, saved_objs)
                saved_objs[model_name] = obj
        log.write('Database objects created.\n')
    except:
        logger.error('[Data Import] Failed creating database objects from %s.' % json_path )
        logger.error(traceback.format_exc())
        log.write('Failed to create database objects.\n')
        raise

    try:
        update_Plan(saved_objs)
        log.write('Updated Plan-related objects.\n')
    except:
        logger.error('[Data Import] Update Plan objects failed.')
        logger.error(traceback.format_exc())
        log.write('Failed to update Plan objects.\n')

    if 'results' in create_sequence:
        # additional things to update for new Results
        try:
            update_Result(saved_objs)
            update_DMFileStats(saved_objs['results'], data)
            log.write('Updated Result-related objects.\n')
        except:
            logger.error('[Data Import] Update Result failed.')
            logger.error(traceback.format_exc())
            log.write('Failed to update Result.\n')

    log.write('Importing database objects done.\n')
    logger.debug('[Data Import] Done creating database objects from %s.' % json_path)
    return saved_objs


def copy_files_to_destination(source_dir, destination, dmfileset, log):

    cached_file_list = get_walk_filelist([source_dir])
    to_process, to_keep = _file_selector(source_dir, dmfileset.include, dmfileset.exclude, [], cached=cached_file_list)
    logger.debug('[Data Import] Importing %d files from %s to %s' % (len(to_process), source_dir, destination) )
    log.write('Copy files to %s ...\n' % destination)
    for i, filepath in enumerate(to_process):
        log.write('%s.\n' % filepath)
        try:
            _copy_to_dir(filepath, source_dir, destination)
        except Exception as exception:
            logger.error(traceback.format_exc())
            log.write("%s" % exception)
        log.flush()

    if dmfileset.type == dmactions_types.OUT:
        # make sure we have plugin_out folder
        plugin_out = os.path.join(destination, 'plugin_out')
        if not os.path.isdir(plugin_out):
            oldmask = os.umask(0000)   #grant write permission to plugin user
            os.mkdir(plugin_out)
            os.umask(oldmask)

    if dmfileset.type == dmactions_types.BASE:
        # for onboard results need to make sigproc_results link
        if os.path.exists(os.path.join(destination, 'onboard_results')):
            os.symlink(os.path.join(destination, 'onboard_results', 'sigproc_results'),os.path.join(destination, 'sigproc_results'))
    
    log.write('Copy files done.\n')

def generate_report_pdf(source_dir, result, dmfilestat, log):
    # copy report pdf from archived if exists, otherwise create
    report_dir = result.get_report_dir()
    source_dir = source_dir.rstrip('/')
    pdf_filepath = os.path.join(report_dir, os.path.basename(report_dir)+'-full.pdf')

    logger.debug('[Data Import] Generating report pdf %s.' % pdf_filepath)
    log.write('Generating report pdf %s\n' % pdf_filepath)

    if os.path.exists(os.path.join(source_dir, os.path.basename(source_dir)+'-full.pdf')):
        _copy_to_dir(os.path.join(source_dir, os.path.basename(source_dir)+'-full.pdf'), source_dir, report_dir)
        os.rename(os.path.join(report_dir, os.path.basename(source_dir)+'-full.pdf'), pdf_filepath)
    elif os.path.exists(os.path.join(source_dir, 'report.pdf')):
        _copy_to_dir(os.path.join(source_dir, 'report.pdf'), source_dir, report_dir)
        os.rename(os.path.join(report_dir, 'report.pdf'), pdf_filepath)
    else:
        # set archivepath for get_report_dir to find files when generating pdf
        dmfilestat.archivepath = source_dir
        dmfilestat.save()
        # create report pdf via latex
        latex_filepath = os.path.join('/tmp', os.path.basename(report_dir)+'-full.tex' )
        url = "http://127.0.0.1/report/" + str(result.pk) + "/?latex=1"
        urllib.urlretrieve(url , latex_filepath)
        pdf = ["pdflatex", "-output-directory", "/tmp", "-interaction", "batchmode", latex_filepath]
        proc = subprocess.Popen(pdf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=source_dir)
        stdout, stderr = proc.communicate()
        if stderr:
            log.write('Error: '+ stderr)
        else:
            _copy_to_dir(os.path.join('/tmp', os.path.basename(report_dir)+'-full.pdf' ), '/tmp', report_dir)

    log.write('Generate report pdf done.\n')


@task(queue='transfer')
def data_import(name, selected, username, copy_data=False, copy_report=True):
    ''' Data import main task.
        Selected dict contains categories to import and path to their serialized json
        Log file can be used to display progress on webpage.
        Copy options:
            if copy_data=True copy Signal Processing or Basecalling Input files to local drive, otherwise mark these categories Archived
            if copy_report=True copy Output files to local drive, otherwise mark it Archived and copy/create report.pdf
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
    log.write('Selected: %s, copy data: %s, copy report: %s.\n' % (', '.join(selected.keys()), copy_data, copy_report) )

    msg_banner(name, selected.keys(), logfile, 'Started', username)

    # create DB records
    try:
        objs = load_serialized_json(json_path, log, createReport)
        result = objs.get('results', None)
        exp = objs['experiment']
    except:
        msg = traceback.format_exc()
        logger.error(msg)
        log.write(msg)
        log.close()
        msg_banner(name, selected.keys(), logfile, 'Error', username)
        return

    # process files
    for category, path in selected.items():
        source_dir = os.path.dirname(path)
        log.write('Importing %s from %s.\n' % (category, source_dir ) )
        logger.debug('(%s) Importing %s from %s.' % (name, category, source_dir ) )

        if category == dmactions_types.OUT and copy_report:
            destination = result.get_report_dir()
        elif category == dmactions_types.BASE and copy_data:
            destination = exp.expDir if 'onboard_results' in source_dir else result.get_report_dir()
        elif category == dmactions_types.SIG and copy_data:
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
                msg_banner(name, selected.keys(), logfile, 'Error', username)
        elif category == dmactions_types.OUT:
            generate_report_pdf(source_dir, result, dmfilestat, log)

        # update locations of imported files; DM state is Local if files copied, otherwise Archived
        if result:
            if destination:
                dmfilestat.action_state = 'L'
                dmfilestat.created = timezone.now()
            else:
                # data files left on media, need to update dmfilestat to archived location
                dmfilestat.action_state = 'AD'
                dmfilestat.archivepath=source_dir
            dmfilestat.save()
        elif not destination:
            # only Sigproc imported (no dmfilestats) and data files not copied
            exp.expDir = source_dir
            exp.save()

    # finish up
    if result:
        EventLog.objects.add_entry(result, "Imported from %s." % os.path.dirname(json_path), username)
        result.status = 'Completed'
        result.save()
        # any categories not imported will have state = Deleted
        result.dmfilestat_set.filter(action_state='IG').update(action_state='DD')

    log.write('FINISHED Import of selected categories.\n')
    log.close()

    msg_banner(name, selected.keys(), logfile, 'Completed', username, importReport, result, exp)

    # save the temp log file to destination
    save_log_path = ''
    if result:
        save_log_path = result.get_report_dir()
    elif copy_data:
        save_log_path = exp.expDir
    if save_log_path:
        shutil.copy(logfile, os.path.join(save_log_path,'import_data.log') )

    logger.info('[Data Import] (%s) Done.' % name)


def find_data_to_import(start_dir, maxdepth=7):
    
    def has_result(resultsName, json_path):
        # special case: serialized_R_*.json files created when run picked up by crawler don't have result objects
        if resultsName.startswith('R_'):
            with open(json_path, 'r') as f:
                for line in f:
                    if 'rundb.results' in line:
                        return True
                return False
        else:
            return True
    
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

    cmd = ['find', start_dir, '-maxdepth', str(maxdepth), '-type', 'f', '-name', 'serialized_*.json']
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for match in p1.stdout.readlines():
        # for each serialized json, test for necessary files by category
        json_path = match.strip()
        root, filename = os.path.split(json_path)
        resultsName = filename.replace('serialized_','',1).rstrip('.json')
        for test in musthave:
            if test['category'] != dmactions_types.SIG and not has_result(resultsName, json_path):
                continue
            for path in test['search_paths']:
                if glob(os.path.join(root, path, test['filename'])):
                    found.setdefault(resultsName, {}).update({ test['category']: json_path })
                    break
    found_results = [{'name':k,'categories':v} for k,v in found.items()]
    return sorted(found_results, key=lambda r:r['name'].lower())

def msg_banner(name, categories, logfile, status, username, importReport=None, result=None, exp=None):
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
        Message.error(msg, tags=logfile, route=username)
    else:
        Message.info(msg, tags=logfile, route=username)

def _get_imported_path():
    location = Location.getdefault()
    try:
        filesPrefix = FileServer.objects.get(name=location).filesPrefix
    except:
        filesPrefix = FileServer.objects.all()[0].filesPrefix
    return os.path.join(filesPrefix,IMPORT_RIG_FOLDER)
