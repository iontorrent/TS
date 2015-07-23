# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
"""
Views
=====

The ``views`` module contains all the Python functions which handle requests
to the Torrent PC Analysis Suite frontend. Each function is a Django "view,"
in that the Django system will forward HTTP requests to the view, which will
then return a Django HTTP response, which is then passed on to the user.

The views in this module serve several purposes:

* Starting new analyses.
* Finding and sorting experiments.
* Finding and sorting experiment analysis reports.
* Monitoring the status of background processes (such as the `crawler` or the
  job `server`.

Not all functions contained in ``views`` are actual Django views, only those
that take ``request`` as their first argument and appear in a ``urls`` module
are in fact Django views.
"""

import datetime
import csv
import os
import logging
import traceback
import json
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.core import serializers

#for sorting a list of lists by a key
from operator import itemgetter

from django import shortcuts, template
from django import http

os.environ['MPLCONFIGDIR'] = '/tmp'

from iondb.rundb import models
from iondb.utils import tables

from ion.utils.explogparser import load_log
from ion.utils.explogparser import parse_log
from ion.utils.explogparser import exp_kwargs
import copy

FILTERED = None # contains the last filter for the reports page for csv export

logger = logging.getLogger(__name__)


@login_required
def tf_csv(request):
    """Return a comma separated values list of all test fragment metrics."""
    #tbl = models.Results.to_pretty_table(models.Results.objects.all())
#    global FILTERED
    if FILTERED == None:
        tbl = models.Results.to_pretty_table(models.Results.objects.all())
    else:
        tbl = models.Results.to_pretty_table(FILTERED)
    ret = http.HttpResponse(tables.table2csv(tbl), mimetype='text/csv')
    now = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    ret['Content-Disposition'] = 'attachment; filename=metrics_%s.csv' % now
    return ret

def remove_experiment(request, page=None):
    """TODO: Blocked on modifying the database schema"""
    pass

@login_required
@csrf_exempt
def displayReport(request, pk):
    ctx = {}
    ctx = template.RequestContext(request, ctx)
    return shortcuts.render_to_response("rundb/reports/report.html", context_instance=ctx)


@login_required
def blank(request, **kwargs):
    """
    just render a blank template
    """
    return shortcuts.render_to_response("rundb/reports/30_default_report.html",
        {'tab':kwargs['tab']})

def getCSV(request):
    CSVstr = ""
    try:
        table = models.Results.to_pretty_table(models.Results.objects.all())
        for k in table:
            for val in k:
                CSVstr += '%s,' % val
            CSVstr = CSVstr[:(len(CSVstr)-1)] + '\n'
    except:
        logger.warn(traceback.format_exc())
    ret = http.HttpResponse(CSVstr, mimetype='text/csv')
    now = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    ret['Content-Disposition'] = 'attachment; filename=metrics_%s.csv' % now
    return ret

# ============================================================================
# Global configuration processing and helpers
# ============================================================================

@login_required
def PDFGen(request, pkR):
    from iondb.utils import makePDF
    pkR = pkR[:len(pkR)-4]
    return http.HttpResponse(makePDF.get_summary_pdf(pkR), mimetype="application/pdf")

@login_required
def PDFGenOld(request, pkR):
    from iondb.utils import makePDF
    pkR = pkR[:len(pkR)-4]
    return http.HttpResponse(makePDF.getOldPDF(pkR), mimetype="application/pdf")


@login_required
def viewLog(request, pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    try:
        log = []
        for datum in ret.metaData["Log"]:
            logList = []
            for dat in datum:
                logList.append("%s: %s"%(dat, datum[dat]))
            log.append(logList)
    except:
        log = [["no actions have been taken on this report."]]
    ctxd = {"log":log}
    context = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_reportLog.html",
                                        context_instance=context)



def barcodeData(filename, metric=None):
    """
    Read the barcode alignment summary file, parse it and return data for the API and graph.
    if no metric is given return all the data
    """
    dictList = []

    #FIX, we got bugs
    if not os.path.exists(filename):
        logger.error("Barcode data file does not exist: '%s'", filename)
        return False

    try:
        fh = open(filename, "rU")
        reader = csv.DictReader(fh)
        for row in reader:
            if metric:
                if metric in row:
                    try:
                        d = {"axis": int(row["Index"]),
                             "name": row["ID"],
                             "value" : float(row[metric]),
                             "sequence" : row["Sequence"],
                             "adapter": '',
                             "annotation" : '',
                        }
                        if "Adapter" in row:
                            d["adapter"] = row["Adapter"]
                        if "Adapter" in row and "Annotation" in row:
                            d["annotation"] = row["Annotation"]
                        elif "Annotation" in row:
                            # V1 had only Annotation column, but it was really adapter sequence
                            d["adapter"] = row["Annotation"]
                        dictList.append(d)
                    except (KeyError, ValueError) as e:
                        ## Could have truncated data!
                        logger.exception(row)
                else:
                    logger.error("Metric missing: '%s'", metric)
            else:
                del row[""] ## Delete empty string (from trailing comma)
                #return a list of dicts where each dict is one row
                dictList.append(row)
    except (IOError, csv.Error) as e:
        ## Could have truncated data!
        logger.exception(e)
    except:
        logger.exception()
        return False

    if not dictList:
        logger.warn("Empty Metric List")
        return False

    #now sort by the "axis" label
    if metric:
        dictList = sorted(dictList, key=itemgetter('axis'))
    else:
        dictList = sorted(dictList, key=itemgetter('Index'))

    return dictList


def pretty(d, indent=0):
    if d:
        for key, value in d.iteritems():
            logger.debug('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                logger.debug('\t' * (indent+1) + str(value))


def updateruninfo(request):
    '''Replaces crawler.py code
    Ensure this function/action is idempotent'''

    folder = request.POST.get('datapath')
    logger.debug("updateruninfo looking at %s" % folder)

    # parse explog.txt.
    LOG_BASENAME = "explog.txt"
    text = load_log(folder, LOG_BASENAME)
    if text is None:
        payload = "Cannot read %s" % (os.path.join(folder, LOG_BASENAME))
        logger.warn(payload)
        return http.HttpResponseServerError(payload)
    else:
        try:
            explog_parsed = parse_log(text)
        except:
            logger.warn("Error parsing %s, skipping %s" % (LOG_BASENAME, folder))
            logger.error(traceback.format_exc())
            return http.HttpResponseServerError('%s %s' % (folder, traceback.format_exc()))

        # Update experiment, plan, eas objects for dataset in this folder
        try:
            planObj, expObj, easObj = get_planned_exp_objects(explog_parsed, folder)
            update_exp_objects_from_log(explog_parsed, folder, planObj, expObj, easObj)
            save_serialized_json(folder, planObj, expObj, easObj)
        except:
            logger.error(traceback.format_exc())
            return http.HttpResponseServerError('%s %s' % (folder, traceback.format_exc()))

        return http.HttpResponse("%s database objects have been updated" % folder, mimetype='text/plain')


def get_planned_exp_objects(d, folder):
    '''
    Find pre-created Plan, Experiment and ExperimentAnalysisSettings objects.
    If not found, create from system default templates.
    '''
    planObj = None
    expObj = None
    easObj = None

    expName = d.get("experiment_name", '')

    # Check if User selected Plan on instrument: explog contains guid or shortId
    selectedPlanGUId = d.get("planned_run_guid", '')

    planShortId = d.get("planned_run_short_id", '')
    if (planShortId is None or len(planShortId) == 0):
        planShortId = d.get("pending_run_short_id", '')

    logger.debug("...planShortId=%s; selectedPlanGUId=%s" % (planShortId, selectedPlanGUId))

    # Find selected Plan
    if selectedPlanGUId:
        try:
            planObj = models.PlannedExperiment.objects.get(planGUID=selectedPlanGUId)
        except models.PlannedExperiment.DoesNotExist:
            logger.warn("No plan with GUId %s found in database " % selectedPlanGUId)
        except models.PlannedExperiment.MultipleObjectsReturned:
            logger.warn("Multiple plan with GUId %s found in database " % selectedPlanGUId)
    elif planShortId:
        # Warning: this is NOT guaranteed to be unique in db!
        try:
            planObj = models.PlannedExperiment.objects.filter(planShortID=planShortId, planExecuted=True).order_by("-date")[0]
        except IndexError:
            logger.warn("No plan with short id %s and planExecuted=True found in database " % planShortId)

    if planObj:
        try:
            expObj = planObj.experiment
            easObj = expObj.get_EAS()

            # Make sure this plan is not already used by an existing experiment!
            if expObj.status == 'run':
                logger.info('WARNING: Plan %s is already associated to an experiment, a copy will be created' % planObj.pk)
                planObj.pk = None
                planObj.latestEAS = None
                planObj.save()

                expObj.pk = None
                expObj.unique = folder
                expObj.plan = planObj
                expObj.repResult = None
                expObj.ftpStatus = ''
                expObj.save()

                easObj.pk = None
                easObj.experiment = expObj
                easObj.save()
                
                planObj.latestEAS = easObj
                planObj.save()
                
            #skip setting the sampleSet to run since a sampleSet can have multiple plans and sequencing runs associated with it
#             sampleSetObj = planObj.sampleSet
#             if sampleSetObj:
#                 logger.debug("crawler going to mark planObj.name=%s; sampleSet.id=%d as run" %(planObj.planDisplayedName, sampleSetObj.id))
# 
#                 sampleSetObj.status = "run"
#                 sampleSetObj.save()


        except:
            logger.warn(traceback.format_exc())
            logger.warn("Error in trying to retrieve experiment and eas for planName=%s, pk=%s" % (planObj.planName, planObj.pk))
    else:
    #if user does not use a plan for the run, fetch the system default plan template, and clone it for this run
        logger.warn("expName: %s not yet in database and needs a sys default plan" % expName)
        try:
            chipversion = d.get('chipversion','')
            if chipversion:
                explogChipType = chipversion
                if explogChipType.startswith('1.10'):
                    explogChipType = 'P1.1.17'
                elif explogChipType.startswith('1.20'):
                    explogChipType = 'P1.2.18'
            else:
                explogChipType = d.get('chiptype','')

            systemDefaultPlanTemplate = None

            if explogChipType:
                systemDefaultPlanTemplate = models.PlannedExperiment.get_latest_plan_or_template_by_chipType(explogChipType)

            if not systemDefaultPlanTemplate:
                logger.debug("Chip-specific system default plan template not found in database for chip=%s; experiment=%s" % (explogChipType, expName))
                systemDefaultPlanTemplate = models.PlannedExperiment.get_latest_plan_or_template_by_chipType()

            logger.debug("Use system default plan template=%s for chipType=%s in experiment=%s" % (systemDefaultPlanTemplate.planDisplayedName, explogChipType, expName))

            # copy Plan
            currentTime = datetime.datetime.now()

            planObj = copy.copy(systemDefaultPlanTemplate)
            planObj.pk = None
            planObj.planGUID = None
            planObj.planShortID = None
            planObj.isReusable = False
            planObj.isSystem = False
            planObj.isSystemDefault = False
            planObj.planName = "CopyOfSystemDefault_" + expName
            planObj.planDisplayedName = planObj.planName
            planObj.planStatus = 'run'
            planObj.planExecuted = True
            planObj.date = currentTime
            planObj.latestEAS = None
            planObj.save()

            # copy Experiment
            expObj = copy.copy(systemDefaultPlanTemplate.experiment)
            expObj.pk = None
            expObj.unique = folder
            expObj.plan = planObj
            expObj.chipType = explogChipType
            expObj.date = currentTime
            expObj.ftpStatus = ''
            expObj.save()

            # copy EAS
            easObj = systemDefaultPlanTemplate.experiment.get_EAS()
            easObj.pk = None
            easObj.experiment = expObj
            easObj.isEditable = True
            easObj.date = currentTime
            easObj.save()

            planObj.latestEAS = easObj
            planObj.save()

            logger.debug("cloned systemDefaultPlanTemplate: planObj.pk=%s; easObj.pk=%s; expObj.pk=%s" % (planObj.pk, easObj.pk, expObj.pk))

            #clone the qc thresholds as well
            qcValues = systemDefaultPlanTemplate.plannedexperimentqc_set.all()

            for qcValue in qcValues:
                qcObj = copy.copy(qcValue)

                qcObj.pk = None
                qcObj.plannedExperiment = planObj
                qcObj.save()

            logger.info("crawler (using template=%s) AFTER SAVING SYSTEM DEFAULT CLONE %s for experiment=%s;" % (systemDefaultPlanTemplate.planDisplayedName, planObj.planName, expName))

        except:
            logger.warn(traceback.format_exc())
            logger.warn("Error in trying to clone system default plan template for experiment=%s" % (expName))

    return planObj, expObj, easObj


def update_exp_objects_from_log(d, folder, planObj, expObj, easObj):
    """ Update plan, experiment, sample and experimentAnalysisSettings
        Returns the experiment object
    """

    if planObj:
        logger.info("Going to update db objects for plan=%s" % planObj.planName)

    kwargs = exp_kwargs(d, folder, logger)

    # PairedEnd runs no longer supported
    if kwargs['isReverseRun']:
        logger.warn("PAIRED-END is NO LONGER SUPPORTED. Skipping experiment %s" % expObj.expName)
        return None

    # get valid libraryKey (explog or entered during planning), if failed use default value
    libraryKey = kwargs['libraryKey']
    if libraryKey and models.LibraryKey.objects.filter(sequence=libraryKey, direction='Forward').exists():
        pass
    elif easObj.libraryKey and models.LibraryKey.objects.filter(sequence=easObj.libraryKey, direction='Forward').exists():
        kwargs['libraryKey'] = easObj.libraryKey
    else:
        try:
            defaultForwardLibraryKey = models.LibraryKey.objects.get(direction='Forward', isDefault=True)
            kwargs['libraryKey'] = defaultForwardLibraryKey.sequence
        except models.LibraryKey.DoesNotExist:
            logger.warn("No default forward library key in database for experiment %s" % expObj.expName)
            return None
        except models.LibraryKey.MultipleObjectsReturned:
            logger.warn("Multiple default forward library keys found in database for experiment %s" % expObj.expName)
            return None

    plan_log = {}

    # *** Update Experiment ***
    expObj.status = 'run'
    expObj.ftpStatus = ''
    expObj.runMode = planObj.runMode
    for key, value in kwargs.items():
        try:
            old = getattr(expObj, key)
        except AttributeError:
            # these are attributes of other objects
            continue

        if key == "sequencekitname" and not value:
            continue
        elif old != value:
            setattr(expObj, key, value)
            if key not in ['unique','expDir','log','date','rawdatastyle']:
                plan_log[key] = [old, value]

    expObj.save()
    logger.info("Updated experiment=%s, pk=%s, expDir=%s" % (expObj.expName, expObj.pk, expObj.expDir) )


    # *** Update Plan ***
    plan_keys = ['expName', 'runType', 'isReverseRun']
    for key in plan_keys:
        old = getattr(planObj, key)
        if old != kwargs[key]:
            setattr(planObj, key, kwargs[key])
            plan_log[key] = [old, kwargs[key] ] 

    planObj.planStatus = 'run'
    planObj.planExecuted = True # this should've been done by instrument already

    # add project from instrument, if any. Note: this will not create a project if doesn't exist
    projectName = kwargs['project']
    planProjects = list(planObj.projects.values_list('name', flat=True))
    if projectName and projectName not in planProjects:
        try:
            planObj.projects.add(models.Project.objects.get(name=projectName))
            plan_log['projects'] = [', '.join(planProjects), ', '.join(planProjects+[projectName]) ]
        except:
            logger.warn("Couldn't add project %s to %s plan: project does not exist" % (projectName, planObj.planName))

    planObj.save()
    logger.info("Updated plan=%s, pk=%s" % (planObj.planName, planObj.pk))


    # *** Update ExperimentAnalysisSettings ***
    eas_keys = ['barcodeKitName', 'reference', 'libraryKey']
    for key in eas_keys:
        old = getattr(easObj, key)
        if old != kwargs[key]:
            setattr(easObj, key, kwargs[key])
            plan_log[key] = [old, kwargs[key] ]

    #do not replace plan's EAS value if explog does not have a value for it
    eas_keys = ['libraryKitName', 'libraryKitBarcode']
    for key in eas_keys:
        old = getattr(easObj, key)
        if kwargs.get(key) and old != kwargs[key]:
            setattr(easObj, key, kwargs[key])
            plan_log[key] = [old, kwargs[key] ]

    easObj.status = 'run'
    easObj.save()
    logger.info("Updated EAS=%s" % easObj)

    # Refresh default cmdline args - this is needed in case chip type or kits changed from their planned values
    default_args = planObj.get_default_cmdline_args()
    for key, value in default_args.items():
        setattr(easObj, key, value)
    easObj.save()

    # *** Update samples associated with experiment***
    sampleCount = expObj.samples.all().count()
    sampleName = kwargs['sample']

    #if this is a barcoded run, user can't change any sample names on the instrument, only sample status update is needed
    if sampleName and not easObj.barcodeKitName:
        need_new_sample = False
        if sampleCount == 1:
            sample = expObj.samples.all()[0]
            # change existing sample name only if this sample has association to 1 experiment,
            # otherwise need to dissociate the original sample and add a new one
            if sample.name != sampleName:
                plan_log['sample'] = [sample.name, sampleName]
                sample_found = models.Sample.objects.filter(name = sampleName)
                if sample_found:
                    sample.experiments.remove(expObj)
                    sample_found[0].experiments.add(expObj)
                    logger.info("Replaced sample=%s; sample.id=%s" %(sampleName, sample_found[0].pk))
                elif sample.experiments.count() == 1:
                    sample.name = sampleName
                    sample.displayedName = sampleName
                    sample.save()
                    logger.info("Updated sample=%s; sample.id=%s" %(sampleName, sample.pk))
                else:
                    sample.experiments.remove(expObj)
                    need_new_sample = True

        if sampleCount == 0 or need_new_sample:
            sample_kwargs = {
                            'name' : sampleName,
                            'displayedName' : sampleName,
                            'date' : expObj.date,
                            }
            try:
                (sample, created) = models.Sample.objects.get_or_create(name=sampleName, defaults=sample_kwargs)
                sample.experiments.add(expObj)
                sample.save()
                logger.info("Added sample=%s; sample.id=%s" %(sampleName, sample.pk))
            except:
                logger.debug("Failed to add sample=%s to experiment=%s" %(sampleName, expObj.expName))
                logger.debug(traceback.format_exc())

    # update status for all samples
    for sample in expObj.samples.all():
        sample.status = expObj.status
        sample.save()
        
    # add log entry if any Plan/Exp/EAS parameters changed after updating from explog
    if plan_log:
        models.EventLog.objects.add_entry(planObj, json.dumps(plan_log), 'system')
        models.EventLog.objects.add_entry(planObj, 'Updated Planned Run from explog: %s (%s).' % (planObj.planName, planObj.pk), 'system')

    return expObj

def save_serialized_json(folder, planObj, expObj, easObj):
    # Saves a snapshot of plan, experiment and eas objects in a json file
    sfile = os.path.join(folder, "serialized_%s.json" % expObj.expName)
    expObj = models.Experiment.objects.get(pk=expObj.pk) # need to refresh obj to get correct DateTimeField value for expObj.date
    serialize_objs = [planObj, expObj, easObj]
    try:
        obj_json = serializers.serialize('json', serialize_objs, indent=2, use_natural_keys=True)
        with open(sfile,'wt') as f:
            f.write(obj_json)
    except:
        logger.error("Unable to save serialized.json for experiment %s(%d)" % (expObj.expName, expObj.pk))
        logger.error(traceback.format_exc())
