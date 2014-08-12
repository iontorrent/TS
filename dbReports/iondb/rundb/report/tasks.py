#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Tasks
=====

The ``tasks`` module contains Python functions which spawn Celery tasks 
in the background.

Not all functions contained in ``tasks`` are actual Celery tasks, only those
that have the  ``@task`` decorator.
"""

from celery import task
from celery.utils.log import get_task_logger

import simplejson
import os
import uuid
from os import path
import traceback
import logging
import iondb.anaserve.djangoinit
import iondb.rundb.models

__version__ = filter(str.isdigit, "$Revision: 74807 $")

logger = get_task_logger(__name__)

def numBarcodes(r):
    """Count the number of barcodes in this run that weren't filtered out of the views.py output """
    nBarcodes = 0;

    try:
        filename = os.path.join(r.get_report_dir(), 'basecaller_results', 'datasets_basecaller.json')
        with open(filename) as fp:
            basecallerDict = simplejson.load(fp)
            read_groups = basecallerDict['read_groups']
            for read_group in read_groups:
                try:
                    if not read_groups[read_group]['filtered']:
                        if "nomatch" not in read_groups[read_group]['barcode_name']: 
                            nBarcodes = nBarcodes + 1
                except:
                    pass
    except:
        pass
    return nBarcodes 

@task
def createRSMExperimentMetrics(resultId):
    """ Creates a file named TSExperiment-UUID.txt that contains metrics from the Results of an experiment."""
    logger.debug("createRSMExperimentMetrics begins for resultId" + str(resultId))
    try:
        r = iondb.rundb.models.Results.objects.get(id=resultId)
        if not r:
            return 1
        logger.debug("createRSMExperimentMetrics after results objects")

        # initialize metrics object
        metrics = [ ]

        # information from explog
        e = r.experiment

        try:
            v = e.chipType
            if v:
                metrics.append('chiptype:' + str(v))
        except:
            pass

        explog = e.log
        keys = [
                'serial_number',
                'run_number',
                'chipbarcode',
                'seqbarcode',
                'flows',
                'cycles',
                'gain',
                'noise',
                'cal_chip_high_low_inrange'
                ]

        for k in keys:
            try:
                v = explog.get(k)
                if v:
                    metrics.append(k + ':' + str(v))
            except:
                pass

        # information from libmetrics
        keys = [
                'sysSNR',
                'aveKeyCounts',
                'total_mapped_target_bases',
                'totalNumReads',
                'raw_accuracy',
                ]
        for k in keys:
            try:
                # there should be only one libmetrics in the set
                v = r.libmetrics_set.values()[0][k]
                if v:
                    metrics.append(k + ':' + str(v))
            except Exception as err:
                pass

        # information from quality metrics
        keys = [
                'q17_mean_read_length',
                'q20_mean_read_length',
                ]
        for k in keys:
            try:
                # there should be only one qualitymetrics in the set
                v = r.qualitymetrics_set.values()[0][k]
                if v:
                    metrics.append(k + ':' + str(v))
            except Exception as err:
                pass

        try:
            metrics.append("RunType:" + e.plan.runType)
        except Exception as err:
            pass

        eas = e.get_EAS()
        try:
            if (eas.targetRegionBedFile == ""):
                eas.targetRegionBedFile = "none"
            if (eas.hotSpotRegionBedFile == ""):
                eas.hotSpotRegionBedFile = "none"
            metrics.append("targetRegionBedFile:" + eas.targetRegionBedFile);
            metrics.append("hotSpotRegionBedFile:" + eas.hotSpotRegionBedFile);
        except Exception as err:
            pass

        try:
            #metrics.append("runPlanName:" + e.plan); not wanted 
            metrics.append("isBarcoded:" + str(e.isBarcoded()));
            nBarcodes = numBarcodes(r)
            metrics.append("numBarcodes:" + str(nBarcodes));
        except:
            pass

        try:
            # get the names of all Ion Chef kits
            ion_chef_kit_names = iondb.rundb.models.KitInfo.objects.filter(kitType="IonChefPrepKit").values_list('name', flat=True)

            # if the kit for this run is in the list of Ion Chef kits, then this run was an Ion Chef run.
            if e.plan.templatingKitName in ion_chef_kit_names:
                metrics.append("chef:y");
            else:
                metrics.append("chef:n");
        except:
            pass

        # report loading for this run (percent of addressable wells that contain ISPs)
        wells_with_isps = 0
        addressable_wells = 0
        keys = [
                'bead',
                'empty',
                'pinned',
                'ignored',
            ]
        for k in keys:
            try:
                v = r.analysismetrics_set.values()[0][k]
                if v:
                    addressable_wells = addressable_wells + v
                    if k == 'bead':
                        wells_with_isps = v
            except Exception as err:
                pass

        if (addressable_wells > 0):
            percent_loaded = 100.0 * float(wells_with_isps) / float(addressable_wells)
            pstr = 'loading:{0:.3}'.format(percent_loaded)
            metrics.append(pstr)

        # write out the metrics
        x = uuid.uuid1()
        fname = os.path.join("/var/spool/ion/",'TSexperiment-' + str(x) + '.txt')
        f = open(fname, 'w' )

        try:
            f.write("\n".join(metrics))
            f.write("\n")
        finally:
            f.close()
            os.chmod(fname, 0666)

        logger.debug("RSM createExperimentMetrics done for resultsId " + str(resultsId))
        return True, "RSM createExperimentMetrics"
    except:
        logger.error(traceback.format_exc())
        return False, traceback.format_exc()
