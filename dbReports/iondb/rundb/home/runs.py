# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
"""
Module to generate dashboard Runs
"""
import os
import sys
import json
import pytz
import time
from datetime import timedelta, datetime
import logging
logger = logging.getLogger(__name__)

from django.db.models import Q
from django.conf import settings
from django.core.serializers.json import DjangoJSONEncoder

from iondb.anaserve.client import get_running_jobs
from iondb.rundb.models import Experiment, PlannedExperiment, SampleSet, Results
from iondb.rundb.home.definitions import *
from iondb.utils import utils

# time to limit active plugin re-run
ACTIVE_PLUGINS_LIMIT_HRS = 48


def get_uid_from_instance(obj):
    # the uid will be Experiment.pk when Experiment object exists
    # for SampleSet the uid will be 'sset_' + SampleSet.pk
    if isinstance(obj, Results):
        uid = '%d' % obj.experiment.pk
    elif isinstance(obj, Experiment):
        uid = '%d' % obj.pk
    elif isinstance(obj, SampleSet):
        uid = 'sset_%d' % obj.pk
    else:
        raise Exception('Invalid model: %s', obj.__class__)
        
    return uid


class Run:
    def __init__(self, stage, obj):
        self.uid =  get_uid_from_instance(obj)
        self.stage = stage # LIB_PREP, TEMPL_PREP, SEQUENCING, ANALYSIS, PLUGINS
        self.name = ""
        self.date = None
        self.last_updated = None
        self.state = "" # IN_PROGRESS, DONE, ERROR
        self.url = ""
        self.instrumentName = ""
        # strings for status and progress
        self.progress_string = ""
        self.status_string = ""
        self.error_string = ""
        # application icon
        self.runType = ""
        self.applicationCategoryDisplayedName = ""
        # samples
        self.sampleDisplayedName = ""
        self.barcodedSamples = []
        # thumbnail report
        self.thumb_url = ""
        self.thumb_error = ""

        if stage == ANALYSIS:
            self.create_analysis(obj)
        elif stage == SEQUENCING:
            self.create_sequencing(obj)
        elif stage == PLUGINS:
            self.create_plugins(obj)
        elif stage == TEMPL_PREP:
            self.create_template_prep(obj)
        elif stage == LIB_PREP:
            self.create_library_prep(obj)
        else:
            raise Exception('Invalid stage specified: ' + stage)

    def update_application_and_samples(self, plan):
        self.runType = plan.runType
        self.applicationCategoryDisplayedName = plan.get_composite_applCatDisplayedName()
        
        self.sampleDisplayedName = plan.get_sampleDisplayedName()
        # format barcoded samples for display
        barcodedSamples = plan.get_barcodedSamples()
        if barcodedSamples:
            samples_list = []
            for key, value in barcodedSamples.items():
                if value.get('dualBarcodes'):
                    barcode_str = ' '.join(sorted(value["dualBarcodes"]))
                else:
                    barcode_str = ' '.join(sorted(value["barcodes"])) if "barcodes" in value else ""

                samples_list.append({
                    "sampleName": key,
                    "barcodes": barcode_str
                })
            self.barcodedSamples = sorted(samples_list, key=lambda x:x['barcodes'] )

    def update_thumbnail(self, exp):
        results = exp.results_set.filter(resultsName__startswith="Auto_", metaData__contains='thumb')
        if results:
            thumb = results[0]
            self.thumb_url = "/report/%d/" % thumb.pk
            if "error" in thumb.status.lower() or thumb.status == "No Live Beads":
                self.thumb_error = thumb.status


    def create_plugins(self, result):
        # handle Plugins stage
        self.name = result.experiment.displayName
        self.date = result.timeStamp
        self.url = "/report/%d/" % result.pk
        self.instrumentName = result.experiment.pgmName
        # for simplicity use result.timeStamp as last updated time
        self.last_updated = result.timeStamp

        # get state of plugin jobs: will be in-progress if any plugins are still running
        pr_set = result.pluginresult_set.all()
        active_plugins = pr_set.filter(plugin_result_jobs__state__in= RUNNING_STATES).distinct()
        error_plugins = pr_set.filter(plugin_result_jobs__state= "Error").distinct()

        self.state = DONE
        
        n_total = pr_set.count()
        n_active = active_plugins.count()
        n_error = error_plugins.count()

        if n_active > 0:
            self.state = IN_PROGRESS
            self.progress_string = "%d of %d" % (n_total - n_active, n_total)

        if n_error > 0:
            self.state = ERROR if n_active == 0 else IN_PROGRESS
            self.error_string = "%d plugin%s in Error" % (n_error, "s" if n_error>1 else "")

        self.update_application_and_samples(result.experiment.plan)
        self.update_thumbnail(result.experiment)

    def create_analysis(self, result):
        # handle Analysis stage
        self.name = result.experiment.displayName
        self.date = result.timeStamp
        self.last_updated = result.timeStamp
        self.url = "/report/%d/" % result.pk
        self.instrumentName = result.experiment.pgmName

        if "error" in result.status.lower() or result.status == "TERMINATED":
            self.state = ERROR
            self.error_string = result.status
        elif "Complete" in result.status:
            self.state = DONE
        else:
            self.state = IN_PROGRESS

        self.update_application_and_samples(result.experiment.plan)
        self.update_thumbnail(result.experiment)

    def create_sequencing(self, exp):
        # handle Sequencing stage
        self.name = exp.displayName
        self.date = exp.date
        self.last_updated = exp.resultDate
        self.instrumentName = exp.pgmName

        if exp.in_progress():
            self.state = IN_PROGRESS
            self.progress_string = "%s/%s" % (exp.ftpStatus, exp.flows)
        elif exp.in_error():
            self.state = ERROR
            self.error_string = exp.ftpStatus
        else:
            self.state = DONE

        self.update_application_and_samples(exp.plan)
        self.update_thumbnail(exp)

    def create_template_prep(self, exp):
        # handle Template Prep stage
        self.name = exp.plan.planDisplayedName
        self.date = exp.chefStartTime
        self.last_updated = exp.chefLastUpdate
        self.instrumentName = exp.chefInstrumentName
        # if run aborted only last update date is available
        if not self.date:
            self.date = self.last_updated

        if exp.plan.planStatus == "voided":
            self.state = ERROR
            self.error_string = exp.chefStatus
        elif exp.chefProgress < 100:
            self.state = IN_PROGRESS
            self.progress_string = "%d%%" % exp.chefProgress
            self.status_string = exp.chefStatus
            estimated_end_time = utils.convert_seconds_to_datetime_string(exp.chefLastUpdate, exp.chefRemainingSeconds)
            if estimated_end_time:
                self.status_string += ' Until ' + estimated_end_time.strftime('%I:%M%p').lower()
        else:
            self.state = DONE

        self.update_application_and_samples(exp.plan)

    def create_library_prep(self, sampleSet):
        # handle Library Prep stage
        self.name = sampleSet.displayedName
        self.date = sampleSet.libraryPrepInstrumentData.startTime
        self.last_updated = sampleSet.libraryPrepInstrumentData.lastUpdate
        self.instrumentName = sampleSet.libraryPrepInstrumentData.instrumentName
        if not self.date:
            self.date = self.last_updated

        if sampleSet.status == "voided":
            self.state = ERROR
            self.error_string = sampleSet.libraryPrepInstrumentData.instrumentStatus
        elif sampleSet.libraryPrepInstrumentData.progress < 100:
            self.state = IN_PROGRESS
            self.progress_string = "%d%%" % sampleSet.libraryPrepInstrumentData.progress
            self.status_string = sampleSet.libraryPrepInstrumentData.instrumentStatus
            estimated_end_time = utils.convert_seconds_to_datetime_string(sampleSet.libraryPrepInstrumentData.lastUpdate,
                            sampleSet.libraryPrepInstrumentData.remainingSeconds)
            if estimated_end_time:
                self.status_string += ' Until ' + estimated_end_time.strftime('%I:%M %p').lower()
        else:
            self.state = DONE

        # application icon for Chef library prep - don't show, it's visually distinct from run type icons
        #self.runType = sampleSet.libraryPrepType
        #self.applicationCategoryDisplayedName = sampleSet.get_libraryPrepType_display()

        # samples
        sampleSetItems = sampleSet.samples.order_by('dnabarcode__id_str').values_list('sample__displayedName', 'dnabarcode__id_str')
        samples = {}
        if len(sampleSetItems) == 1 and not sampleSetItems[0][1]:
            self.sampleDisplayedName = sampleSetItems[0][0]
        elif len(sampleSetItems) > 0:
            for sampleName, barcode in sampleSetItems:
                if sampleName not in samples:
                        samples[sampleName] = {
                            "sampleName": sampleName,
                            "barcodes": barcode or ""
                        }
                elif barcode:
                    samples[sampleName]["barcodes"] += " " + barcode
            
            self.barcodedSamples = sorted( samples.values(), key=lambda x:(x['barcodes'],x['sampleName'].lower()) )


class Runs:
    """ Runs must be generated in order from last to first active stage,
        only new runs get added, runs with existing uids will NOT be replaced
    """
    def __init__(self):
        self.uids = []
        self.runs = []
        # get a list of active results from jobServer
        try:
            running = get_running_jobs(settings.JOBSERVER_HOST, settings.JOBSERVER_PORT)
            self.active_results_ids = [r[2] for r in running]
        except:
            raise Exception('Error: Unable to retrieve list of active analysis jobs')

    def contains(self, uid):
        return uid in self.uids

    def sort(self):
        self.runs.sort(key=lambda r:r.date, reverse=True)

    def add_run(self, run):
        if not self.contains(run.uid):
            self.uids.append(run.uid)
            self.runs.append(run)
    
    #def get_experiment_ids(self):
    #    return [uid for uid in self.uids if not uid.startswith('sset')]
    def get_runs(self):
        runs_list = []
        for run in self.runs:
            runs_list.append(run.__dict__)
        return runs_list

    def get_state(self):
        state = []
        for run in self.runs:
            state.append({
                "uid": run.uid,
                "name": run.name,
                "stage": run.stage,
                "state": run.state,
            })
        return state


def results_in_range(runs, timelimit):
    '''
    1) Results with Result.timeStamp within time range
    2) limit to non-thumbnail and auto-analysis only
    3) exclude active jobs - these are handled by add_active_analysis
    4) if Experiment is in progress or in error: stage = Sequencing
    5)  otherwise:
            if Analysis is done with "Error", stage = Analysis and Error
            if plugins exist stage = Plugins, otherwise stage = Analysis and Complete
    '''
    results = Results.objects.filter(timeStamp__gt=timelimit).exclude(pk__in=runs.active_results_ids).exclude(metaData__contains='thumb')
    # need better way to tell reanalysis vs auto run
    results = results.filter(resultsName__startswith="Auto_")

    for result in results:
        if result.experiment.in_progress() or result.experiment.in_error():
            r = Run(SEQUENCING, result.experiment)
            runs.add_run(r)
        else:
            if "error" in result.status.lower() or result.pluginresult_set.count() == 0:
                r = Run(ANALYSIS, result)
                runs.add_run(r)
            else:
                r = Run(PLUGINS, result)
                runs.add_run(r)

    return runs


def add_active_analysis(runs):
    '''
    1) get active Analysis jobs from jobServer
    2) limit to auto-analysis non-thumbnail results only
    3) if Experiment is in progress or in error: stage = Sequencing
    4)  otherwise: stage = Analysis
    '''
    results = Results.objects.filter(pk__in=runs.active_results_ids)
    # need better way to tell reanalysis vs auto run
    results = results.filter(resultsName__startswith="Auto_").exclude(metaData__contains='thumb')

    for result in results:
        uid = get_uid_from_instance(result)
        if not runs.contains(uid):
            if result.experiment.in_progress() or result.experiment.in_error():
                r = Run(SEQUENCING, result.experiment)
                runs.add_run(r)
            else:
                r = Run(ANALYSIS, result)
                runs.add_run(r)


def sequencing_in_range(runs, timelimit):
    '''
    1) Experiments with status = "run" and ftpStatus != "Complete"
    2) limit to Experiment.date within time range
    3) this function is called after all Results are added and adds only runs that have not already been picked up
    '''
    experiments = Experiment.objects.filter(status="run", date__gt=timelimit).exclude(ftpStatus="Complete")

    for exp in experiments:
        uid = get_uid_from_instance(exp)
        if not runs.contains(uid):
            r = Run(SEQUENCING, exp)
            runs.add_run(r)


def add_active_plugins(runs):
    '''
    1) Results with PluginResultJob.state in RUNNING_STATES and Result.timeStamp within ACTIVE_PLUGINS_LIMIT_HRS
        the timeStamp limit is needed because we don't distinquish manual vs pipeline plugin runs
    2) limit to non-thumbnail and auto-analysis only
    3) exclude active analysis jobs
    '''
    results = Results.objects.filter(pluginresult_set__plugin_result_jobs__state__in= RUNNING_STATES).distinct()
    # limit to recent results
    active_plugins_timelimit = datetime.now(pytz.UTC) - timedelta(hours=ACTIVE_PLUGINS_LIMIT_HRS)
    results = results.filter(timeStamp__gt=active_plugins_timelimit).exclude(pk__in=runs.active_results_ids)
    # need better way to tell reanalysis vs auto run
    results = results.filter(resultsName__startswith="Auto_")
    
    for result in results:
        uid = get_uid_from_instance(result)
        if "error" in result.status.lower() or result.isThumbnail or runs.contains(uid):
            continue
        else:
            r = Run(PLUGINS, result)
            runs.add_run(r)


def add_template_prep(runs, timelimit):
    '''
    1) Experiment where plan.planExecuted = False
    2) Experiment.chefLastUpdate is within time range
    '''
    experiments = Experiment.objects.filter(plan__planExecuted=False, chefLastUpdate__gte=timelimit)
    for exp in experiments:
        r = Run(TEMPL_PREP, exp)
        runs.add_run(r)


def add_library_prep(runs, timelimit):
    '''
    1) SampleSet where libraryPrepInstrumentData exists and 
    2) SampleSet.libraryPrepInstrumentData.lastUpdate is within time range
    3) if Plans were created from SampleSet, do not add Library Prep if all runs are already included
    '''

    sampleSets = SampleSet.objects.filter(libraryPrepInstrumentData__lastUpdate__gte=timelimit)
    for sampleSet in sampleSets:
        # many-to-many between SampleSet and Experiment, need to make sure we don't have duplicate rows
        experiment_ids = sampleSet.plans.values_list('experiment', flat=True)
        if experiment_ids and all(runs.contains(pk) for pk in experiment_ids):
            continue
        else:
            r = Run(LIB_PREP, sampleSet)
            runs.add_run(r)


def generate_runs(timelimit):
    runs = Runs()

    add_active_analysis(runs)
    add_active_plugins(runs)

    results_in_range(runs, timelimit)
    sequencing_in_range(runs, timelimit)

    add_template_prep(runs, timelimit)
    add_library_prep(runs, timelimit)

    try:
        runs.sort()
    except Exception as e:
        logger.error("Error sorting Runs: " + str(e))

    return runs


def get_runs_list(timelimit):
    # generates Runs and returns a list of dicts
    runs = generate_runs(timelimit)
    return runs.get_runs()

def get_runs_state(timelimit):
    # generates Runs and returns a list of dicts for a subset of Run fields
    runs = generate_runs(timelimit)
    return runs.get_state()
