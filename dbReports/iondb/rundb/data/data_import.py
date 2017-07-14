#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import
import os
import sys
import logging
import traceback
import json
import shutil
import time
import subprocess
import urllib
import errno
from glob import glob
from celery import task
from iondb.bin.djangoinit import *
from iondb.rundb.models import Experiment, DMFileStat, DMFileSet, ReportStorage, Location, Message, EventLog, FileServer, Content, Sample
from iondb.settings import RELVERSION
from django.core import serializers
from django.utils import timezone
from django.db.models import get_model
from django.db import transaction

from iondb.rundb.data import dmactions_types
from iondb.rundb.data import dm_utils
from iondb.rundb.data.dmactions import _copy_to_dir
from iondb.rundb.data.dm_utils import get_walk_filelist
from iondb.utils.files import getSpaceMB

logger = logging.getLogger()

IMPORT_RIG_FOLDER = 'Imported'


def update_Result(saved_objs):
    # Set report storage and link
    result = saved_objs['results']
    storage = ReportStorage.objects.filter(default=True)[0]
    location = Location.getdefault()
    result.reportstorage = storage
    result.reportLink = os.path.join(
        storage.webServerPath, location.name, "%s_%03d" % (result.resultsName, result.pk), "")

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

        logger.info('[Data Import] Update Plan: unable to find bedfile: %s for reference: %s' %
                    (filename, reference))
        return ''

    eas = saved_objs['experiment analysis settings']

    # if any BED files were selected, attempt to find them and update location
    if getattr(eas, 'targetRegionBedFile', '') and not os.path.exists(eas.targetRegionBedFile):
        eas.targetRegionBedFile = find_bedfile(os.path.basename(eas.targetRegionBedFile), eas.reference)
        eas.save()
    if getattr(eas, 'hotSpotRegionBedFile', '') and not os.path.exists(eas.hotSpotRegionBedFile):
        eas.hotSpotRegionBedFile = find_bedfile(os.path.basename(eas.hotSpotRegionBedFile), eas.reference)
        eas.save()


def update_or_create_Samples(data, saved_objs):
    samples = [d for d in data if d['model'] == 'rundb.sample']
    for sample_dict in samples:
        if sample_dict['fields'].get('name'):
            try:
                sample = Sample.objects.get(name=sample_dict['fields'][
                                            'name'], externalId=sample_dict['fields'].get('externalId', ''))
            except Sample.DoesNotExist:
                sample = create_obj('sample', [sample_dict], saved_objs)
            sample.experiments.add(saved_objs['experiment'])


def update_DMFileStats(result, data):
    dmfilestats = result.dmfilestat_set.all()
    dmfilestats.update(action_state='DD', diskspace=0)
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
                    setattr(obj, field.name, timezone.now())
                elif field_type in ['ForeignKey', 'OneToOneField', 'ManyToManyField']:
                    relation_name = field.related.parent_model._meta.verbose_name
                    if relation_name in saved_objs.keys():
                        setattr(obj, field.name, saved_objs[relation_name])
                    elif field.null:
                        setattr(obj, field.name, None)
                else:
                    if field.name in data_fields:
                        setattr(obj, field.name, data_fields[field.name])

            obj.save()
            logger.info('[Data Import] SAVED %s pk=%s' % (model_name, obj.pk))

            return obj


def load_serialized_json(json_path, create_result, log, add_warning):
    # creates DB objects for Plan, Experiment, Result etc.
    # serialized.json file is created during DataManagement export/archive and contains objects to be imported
    # Note that DB objects must be saved in specific sequence for correct relations to be set up
    log('Importing database objects ...')
    logger.info('[Data Import] Creating database objects from %s.' % json_path)

    with open(json_path) as f:
        data = json.load(f)

    create_sequence = []
    saved_objs = {}

    # skip creating experiment if it already exists
    exp_fields = [d['fields'] for d in data if d['model'] == 'rundb.experiment'][0]
    unique = exp_fields['unique']
    unique_imported = os.path.join(_get_imported_path(), os.path.basename(exp_fields['expDir']))

    exp = Experiment.objects.filter(unique__in=[unique, unique_imported])
    if exp:
        log('Found existing experiment %s (%s).' % (exp[0].expName, exp[0].pk))
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
        result = saved_objs['experiment'].results_set.filter(
            runid=result_fields['runid']) if 'experiment' in saved_objs else None
        if result:
            log('Found existing result %s (%s).' % (result[0].resultsName, result[0].pk))
            saved_objs['results'] = result[0]
            saved_objs['experiment analysis settings'] = result[0].eas
        else:
            create_sequence += ['experiment analysis settings', 'results',
                                'analysis metrics', 'lib metrics', 'quality metrics']
    else:
        create_sequence += ['experiment analysis settings']

    # create the records
    log(' create objects: %s' % (', '.join(create_sequence) or 'None'))
    try:
        with transaction.commit_on_success():
            for model_name in create_sequence:
                obj = create_obj(model_name, data, saved_objs)
                saved_objs[model_name] = obj
        log('Database objects created')
    except:
        logger.error('[Data Import] Failed creating database objects from %s.' % json_path)
        log('Failed to create database objects.')
        raise

    try:
        update_Plan(saved_objs)
        log('Updated Plan-related objects.')
    except:
        logger.error('[Data Import] Update Plan objects failed.')
        add_warning('Failed to update Plan objects')
        log(traceback.format_exc())

    if 'experiment' in create_sequence:
        try:
            update_or_create_Samples(data, saved_objs)
            log('Updated Sample objects.')
        except:
            logger.error('[Data Import] Update Sample objects failed.')
            add_warning('Failed to update Sample objects')
            log(traceback.format_exc())

    # additional things to update for new Results
    if 'results' in create_sequence:
        try:
            update_Result(saved_objs)
            update_DMFileStats(saved_objs['results'], data)
            log('Updated Result-related objects')
        except:
            logger.error('[Data Import] Update Result failed.')
            add_warning('Failed to update Result objects.')
            log(traceback.format_exc())

    log('Importing database objects done.', flush=True)
    logger.info('[Data Import] Done creating database objects from %s.' % json_path)
    return saved_objs


def copy_files_to_destination(source_dir, destination, dmfileset, cached_file_list, log, add_warning):

    to_process, to_keep = dm_utils._file_selector(
        source_dir, dmfileset.include, dmfileset.exclude, [], cached=cached_file_list)
    logger.info('[Data Import] Importing %d files from %s to %s' %
                (len(to_process), source_dir, destination))
    log('Copy files to destination: %d files, source=%s destination=%s' %
        (len(to_process), source_dir, destination))

    plugin_warnings = {}
    for i, filepath in enumerate(to_process):
        log('%s' % filepath, flush=True)
        try:
            _copy_to_dir(filepath, source_dir, destination)
        except Exception as e:
            # log and ignore errors from plugin files
            if 'plugin_out' in filepath:
                plugin_name = filepath.split('plugin_out/')[1].split('_out')[0]
                plugin_warnings[plugin_name] = plugin_warnings.get(plugin_name, 0) + 1
                log(traceback.format_exc())
            else:
                raise

    for plugin, count in plugin_warnings.items():
        add_warning('Unable to copy %d files for plugin %s' % (count, plugin))

    if dmfileset.type == dmactions_types.OUT:
        # make sure we have plugin_out folder
        plugin_out = os.path.join(destination, 'plugin_out')
        if not os.path.isdir(plugin_out):
            oldmask = os.umask(0000)  # grant write permission to plugin user
            os.mkdir(plugin_out)
            os.umask(oldmask)

        # remove pdf folder, it may have incorrect permissions
        pdf_dir = os.path.join(destination, 'pdf')
        if os.path.exists(pdf_dir):
            shutil.rmtree(pdf_dir, ignore_errors=True)

    # for onboard results need to create sigproc_results link
    if dmfileset.type == dmactions_types.BASE:
        if os.path.exists(os.path.join(destination, 'onboard_results')):
            os.symlink(os.path.join(destination, 'onboard_results', 'sigproc_results'),
                       os.path.join(destination, 'sigproc_results'))

    log('Copy files to destination %s done.' % dmfileset.type)


def generate_report_pdf(source_dir, result, dmfilestat, log, add_warning):
    # copy report pdf from archived if exists, otherwise create
    report_dir = result.get_report_dir()
    source_dir = source_dir.rstrip('/')
    pdf_filepath = os.path.join(report_dir, os.path.basename(report_dir) + '-full.pdf')

    logger.info('[Data Import] Generating report pdf %s.' % pdf_filepath)
    log('Generating report pdf %s' % pdf_filepath)

    if os.path.exists(os.path.join(source_dir, os.path.basename(source_dir) + '-full.pdf')):
        _copy_to_dir(
            os.path.join(source_dir, os.path.basename(source_dir) + '-full.pdf'), source_dir, report_dir)
        os.rename(os.path.join(report_dir, os.path.basename(source_dir) + '-full.pdf'), pdf_filepath)
    elif os.path.exists(os.path.join(source_dir, 'report.pdf')):
        _copy_to_dir(os.path.join(source_dir, 'report.pdf'), source_dir, report_dir)
        os.rename(os.path.join(report_dir, 'report.pdf'), pdf_filepath)
    else:
        # set archivepath for get_report_dir to find files when generating pdf
        dmfilestat.archivepath = source_dir
        dmfilestat.save()
        # create report pdf via latex
        latex_filepath = os.path.join('/tmp', os.path.basename(report_dir) + '-full.tex')
        url = "http://127.0.0.1/report/" + str(result.pk) + "/?latex=1"
        urllib.urlretrieve(url, latex_filepath)
        pdf = ["pdflatex", "-output-directory", "/tmp", "-interaction", "batchmode", latex_filepath]
        proc = subprocess.Popen(pdf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=source_dir)
        stdout, stderr = proc.communicate()
        if stderr:
            add_warning('Error creating report pdf: %s' % stderr)
        else:
            _copy_to_dir(os.path.join('/tmp', os.path.basename(report_dir) + '-full.pdf'), '/tmp', report_dir)

    log('Generate report pdf done.')


def get_diskspace(source_dir, dmfileset, cached_file_list, add_warning):
    try:
        to_process, to_keep = dm_utils._file_selector(
            source_dir, dmfileset.include, dmfileset.exclude, [], cached=cached_file_list)
        total_size = 0
        for path in to_process:
            if not os.path.islink(path):
                total_size += os.lstat(path)[6]
        diskspace = float(total_size) / (1024 * 1024)
    except:
        logger.error(traceback.format_exc())
        add_warning('Error calculating diskspace for %s' % dmfileset.type)
        diskspace = None

    return diskspace


class ImportData:

    def __init__(self, name, selected, username, copy_data, copy_report):
        self.name = name
        self.user = username
        self.result = None
        self.exp = None

        self.dmtypes = selected.keys()
        self.selected_str = ', '.join(selected.keys())
        self.createResult = (dmactions_types.OUT in selected) or (dmactions_types.BASE in selected)
        self.json_path = selected.get(dmactions_types.OUT) or selected.get(
            dmactions_types.BASE) or selected.values()[0]
        self.warnings = []

        self.tag = '%s_%f' % (name, time.time())
        self.tmp_log_path = os.path.join('/tmp', 'di_log_%s' % self.tag)
        self.logfile = None

        self.categories = []
        for dmtype, json_path in selected.items():
            if dmtype in dmactions_types.FILESET_TYPES:
                self.categories.append({
                    'dmtype': dmtype,
                    'src_path': os.path.dirname(json_path),
                    'dest_path': None,
                    'diskspace': 0,
                    'copy_files': copy_report if dmtype == dmactions_types.OUT else copy_data
                })

    def update_destinations(self, result, exp):
        # update destination paths for selected categories
        self.result = result
        self.exp = exp

        for category in self.categories:
            if category['copy_files']:
                dmtype = category['dmtype']
                if dmtype == dmactions_types.OUT:
                    category['dest_path'] = self.result.get_report_dir()
                elif dmtype == dmactions_types.BASE:
                    category['dest_path'] = self.exp.expDir if 'onboard_results' in category[
                        'src_path'] else self.result.get_report_dir()
                elif dmtype == dmactions_types.SIG:
                    category['dest_path'] = self.exp.expDir

    def update_diskspace(self, file_list):
        for category in self.categories:
            if category['copy_files']:
                dmtype = category['dmtype']
                dmfileset = self.result.get_filestat(
                    dmtype).dmfileset if self.result else DMFileSet.objects.get(version=RELVERSION, type=dmtype)
                category['diskspace'] = get_diskspace(
                    category['src_path'], dmfileset, file_list, self.add_warning)

    def start(self):
        self.logfile = open(self.tmp_log_path, 'w', 0)
        self.log('(%s) Import of selected categories started' % self.name)
        msg_banner('Started', self)

    def fail(self, err, trace):
        self.logfile.write('ERROR: ' + trace)
        self.logfile.close()
        msg_banner('ERROR', self, error_str=err)

        if self.result:
            EventLog.objects.add_entry(self.result, "Importing Error: %s %s." % (self.name, err), self.user)
            self.result.dmfilestat_set.filter(action_state='IG').update(action_state='E')
            self.result.status = "Importing Failed"
            self.result.save()
            copy_log_file(self.tmp_log_path, self.result.get_report_dir())

    def finish(self):
        self.log('FINISHED Import of selected categories.')
        self.logfile.close()

        # save log file to local destination
        save_log_dir = self.result.get_report_dir() if self.result else ''
        if not save_log_dir and self.categories[0]['copy_files']:
            save_log_dir = self.exp.expDir
        if save_log_dir and os.path.exists(save_log_dir):
            copy_log_file(self.tmp_log_path, save_log_dir)

        if self.warnings:
            msg_banner('Completed with %d warnings' % len(self.warnings), self)
        else:
            msg_banner('Completed', self)

        if self.result:
            txt = "Imported from %s" % self.json_path
            if self.warnings:
                txt += "<br>Warnings:<br>" + "<br>".join(self.warnings)
            EventLog.objects.add_entry(self.result, txt, self.user)

    def log(self, msg, flush=False):
        logger.debug(msg)
        self.logfile.write("[ %s ] %s\n" % (time.strftime('%X'), msg))
        if flush:
            self.logfile.flush()

    def add_warning(self, msg):
        self.warnings.append(msg)
        self.log('Warning: %s' % msg)


@task(queue='transfer')
def data_import(name, selected, username, copy_data=False, copy_report=True):
    ''' Data import main task.
        Selected dict contains categories to import and path to their serialized json
        Copy options:
            if copy_data=True copy Signal Processing and/or Basecalling Input files to local drive, otherwise mark these categories Archived
            if copy_report=True copy Output files to local drive, otherwise mark it Archived and copy/create report.pdf
    '''
    try:
        importing = ImportData(name, selected, username, copy_data, copy_report)
        importing.start()
        importing.log('Selected: %s, copy data: %s, copy report: %s.' %
                      (importing.selected_str, copy_data, copy_report))
        logger.info('[Data Import] (%s) Started import %s using %s, copy data: %s, copy report: %s.' %
                    (name, importing.selected_str, importing.json_path, copy_data, copy_report))
        process_import(importing, copy_data, copy_report)
        # finish up
        importing.finish()
        logger.info('[Data Import] (%s) Done.' % importing.name)
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(trace)
        importing.fail(str(e), trace)


def process_import(importing, copy_data, copy_report):
    # create DB records
    try:
        objs = load_serialized_json(
            importing.json_path, importing.createResult, importing.log, importing.add_warning)
        result = objs.get('results', None)
        exp = objs['experiment']
        importing.update_destinations(result, exp)
    except Exception as e:
        raise

    if result:
        dmfilestats_to_import = result.dmfilestat_set.filter(dmfileset__type__in=importing.dmtypes)
        # check if importing is allowed
        for dmfilestat in dmfilestats_to_import:
            if dmfilestat.action_state in ['AG', 'DG', 'EG', 'SA', 'SE', 'SD']:
                raise Exception("Cannot import %s when data is in process: %s" %
                                (dmfilestat.dmfileset.type, dmfilestat.get_action_state_display()))

        # set status
        dmfilestats_to_import.update(action_state='IG')
        result.status = 'Importing'
        result.save()
        EventLog.objects.add_entry(result, "Importing %s %s." %
                                   (importing.name, importing.selected_str), importing.user)

    # get list of files
    file_list = []
    if copy_data or copy_report:
        source_paths = set([c['src_path'] for c in importing.categories if c['copy_files']])
        file_list = get_walk_filelist(list(source_paths), list_dir=False, save_list=False)

    # calculate dmfilestat diskspace
    importing.update_diskspace(file_list)
    importing.log('Selected categories:' + json.dumps(importing.categories, indent=1))

    # destination validation
    try:
        validate_destination(importing.categories)
    except:
        raise

    # copy files to destination
    for category in importing.categories:
        dmtype = category['dmtype']
        source_dir = category['src_path']
        destination = category['dest_path']

        if result:
            dmfilestat = result.get_filestat(dmtype)
            dmfileset = dmfilestat.dmfileset
        else:
            dmfilestat = None
            dmfileset = DMFileSet.objects.get(version=RELVERSION, type=dmtype)

        # process files
        if category['copy_files']:
            importing.log('Start processing files for %s.' % dmtype)

            if not os.path.exists(source_dir):
                raise Exception("Source path %s does not exist, exiting." % source_dir)

            try:
                copy_files_to_destination(
                    source_dir, destination, dmfileset, file_list, importing.log, importing.add_warning)
            except:
                raise

        elif dmtype == dmactions_types.OUT:
            # special case: importing Report as Archived (copy_report=False)
            try:
                generate_report_pdf(source_dir, result, dmfilestat, importing.log, importing.add_warning)
            except:
                importing.add_warning('Failed to generate report pdf')
                importing.log(traceback.format_exc())

    # update database objects; DM state is Local if files copied, otherwise Archived
    importing.log('Updating location of imported files')
    if result:
        for category in importing.categories:
            dmfilestat = result.get_filestat(category['dmtype'])
            if category['copy_files']:
                dmfilestat.action_state = 'L'
                dmfilestat.created = timezone.now()
            else:
                # data files left on media, need to update dmfilestat to archived location
                dmfilestat.action_state = 'AD'
                dmfilestat.archivepath = category['src_path']
            dmfilestat.diskspace = category['diskspace']
            dmfilestat.save()

        result.status = 'Completed'
        result.save()

    elif dmactions_types.SIG in importing.dmtypes:
        if copy_data:
            # if any results exist for this data set, need to update their dmfilestats
            DMFileStat.objects.filter(
                dmfileset__type=dmactions_types.SIG, result__experiment=exp).update(action_state='L')
        else:
            # only Sigproc imported (no dmfilestats) and data files not copied
            exp.expDir = os.path.dirname(importing.json_path)
            exp.save()


def find_data_to_import(start_dir, maxdepth=7):

    def has_result(resultsName, json_path):
        # special case: serialized_R_*.json files created when run picked up by
        # crawler don't have result objects
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
            'search_paths': ['', 'block_X*_Y*', 'thumbnail']
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
        resultsName = filename.replace('serialized_', '', 1).split('.json')[0]
        for test in musthave:
            if test['category'] != dmactions_types.SIG and not has_result(resultsName, json_path):
                continue
            for path in test['search_paths']:
                if glob(os.path.join(root, path, test['filename'])):
                    found.setdefault(resultsName, {}).update({test['category']: json_path})
                    break
    found_results = [{'name': k, 'categories': v} for k, v in found.items()]
    return sorted(found_results, key=lambda r: r['name'].lower())


def msg_banner(status, importing, error_str=''):
    msg = '(%s) Import %s, %s.' % (importing.name, importing.selected_str, status)

    if 'Completed' in status:
        if importing.result:
            logurl = '/data/datamanagement/log/%s/' % importing.result.pk
            msg += " <a href='%s' data-toggle='modal' data-target='#modal_report_log'>View Report Log</a>" % (
                logurl)
        if dmactions_types.OUT in importing.dmtypes:
            reporturl = '/report/%s/' % importing.result.pk
            msg += " Imported Report: <a href='%s'>%s</a>'" % (reporturl, importing.result.resultsName)
        else:
            msg += " Imported Run available for analysis: <a href='/data/'>%s</a>" % importing.exp.expName

    Message.objects.filter(tags=importing.tag).delete()
    if status == 'ERROR':
        errlog = '/data/datamanagement/import_data_log/%s' % importing.logfile.name
        msg += " %s <a href='%s' data-toggle='modal' data-target='#modal_report_log'>Error Log</a>" % (
            error_str, errlog)
        Message.error(msg, tags=importing.tag, route=importing.user)
    else:
        Message.info(msg, tags=importing.tag, route=importing.user)


def _get_imported_path():
    location = Location.getdefault()
    try:
        filesPrefix = FileServer.objects.get(name=location).filesPrefix
    except:
        filesPrefix = FileServer.objects.all()[0].filesPrefix
    return os.path.join(filesPrefix, IMPORT_RIG_FOLDER)


def copy_log_file(tmp_path, dst):
    # copy import log file from temporary location
    save_log_path = os.path.join(dst, 'import_data.log')
    try:
        if os.path.exists(save_log_path):
            idx = 1
            while os.path.exists(os.path.join('%s.%d' % (save_log_path, idx))):
                idx += 1
            shutil.move(save_log_path, os.path.join('%s.%d' % (save_log_path, idx)))
        shutil.copy(tmp_path, save_log_path)
    except:
        logger.error(traceback.format_exc())


def validate_destination(categories):
    dest = {}
    for category in categories:
        if category['copy_files']:
            destination = category['dest_path']
            if destination:
                if os.path.normpath(destination) == os.path.normpath(category['src_path']):
                    raise Exception("%s destination is the same as source path: %s" %
                                    (category['dmtype'], destination))

                if category['diskspace']:
                    dest[destination] = dest.setdefault(destination, 0) + category['diskspace']
            else:
                raise Exception("Missing %s copy files destination" % category['dmtype'])

    for destination, diskspace in dest.items():
        # create destination
        try:
            os.makedirs(destination)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # check diskspace
        freespace = getSpaceMB(destination)
        if diskspace >= freespace:
            raise Exception("Not enough space to copy files at %s (required=%dMB, free=%dMB)" %
                            (destination, diskspace, freespace))
