#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Tasks
=====

The ``tasks`` module contains Python functions which spawn Celery tasks
in the background.

Not all functions contained in ``tasks`` are actual Celery tasks, only those
that have the  ``@app.task`` decorator.
"""

from __future__ import absolute_import
from iondb.celery import app
from celery.utils.log import get_task_logger

import simplejson
import os

# uncomment for testing
# os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

import uuid
import traceback
import iondb.rundb.models

__version__ = filter(str.isdigit, "$Revision: 74807 $")


def num_barcodes(run):
    """Count barcodes in run not filtered out of the views.py output """
    nbarcodes = 0

    try:
        filename = os.path.join(
            run.get_report_dir(), "basecaller_results", "datasets_basecaller.json"
        )
        with open(filename) as inputfile:
            basecallerdict = simplejson.load(inputfile)
            read_groups = basecallerdict["read_groups"]
            for grp in read_groups:
                try:
                    if not read_groups[grp]["filtered"]:
                        if "nomatch" not in read_groups[grp]["barcode_name"]:
                            nbarcodes = nbarcodes + 1
                except KeyError:
                    pass
                except:
                    pass  # print traceback.format_exc()
    except IOError:
        pass
    except:
        pass  # print traceback.format_exc()

    return nbarcodes


@app.task
def createRSMExperimentMetrics(result_id):
    """Creates file TSExperiment-UUID.txt with metrics for an experiment."""
    logger = get_task_logger(__name__)
    logger.debug("createRSMExperimentMetrics begins for resultId=" + str(result_id))
    try:
        run = iondb.rundb.models.Results.objects.get(id=result_id)
        if not run:
            return 1
        logger.debug("createRSMExperimentMetrics after results objects")

        # initialize metrics object
        metrics = []

        # information from explog
        exp = run.experiment

        try:
            chiptype = exp.chipType
            if chiptype:
                metrics.append("chiptype:" + str(chiptype))
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if exp.sequencekitname:
                metrics.append("seqkit:" + str(exp.sequencekitname))
        except Exception:
            pass  # print traceback.format_exc()

        explog = exp.log
        keys = [
            "serial_number",
            "run_number",
            "chipbarcode",
            "seqbarcode",
            "flows",
            "cycles",
            "gain",
            "noise",
            "cal_chip_high_low_inrange",
        ]

        for key in keys:
            try:
                val = explog.get(key)
                if val:
                    metrics.append(key + ":" + str(val))
            except Exception:
                pass  # print traceback.format_exc()

        # information from libmetrics
        keys = [
            "sysSNR",
            "aveKeyCounts",
            "total_mapped_target_bases",
            "totalNumReads",
            "raw_accuracy",
        ]
        for key in keys:
            try:
                # there should be only one libmetrics in the set
                val = list(run.libmetrics_set.values())[0][key]
                if val:
                    metrics.append(key + ":" + str(val))
            except IndexError:
                pass
            except:
                pass  # print traceback.format_exc()

        # information from quality metrics
        keys = ["q17_mean_read_length", "q20_mean_read_length"]
        for key in keys:
            try:
                # there should be only one qualitymetrics in the set
                val = list(run.qualitymetrics_set.values())[0][key]
                if val:
                    metrics.append(key + ":" + str(val))
            except IndexError:
                pass
            except:
                pass  # print traceback.format_exc()

        try:
            if exp.plan.metaData is not None:
                key = "fromTemplate"
                if key in exp.plan.metaData:
                    metrics.append("fromTmpl:" + exp.plan.metaData[key])
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if exp.plan.metaData is not None:
                key = "fromTemplateSource"
                if key in exp.plan.metaData:
                    metrics.append("fromTmplSrc:" + exp.plan.metaData[key])
        except Exception:
            pass  # print traceback.format_exc()

        try:
            metrics.append("RunType:" + exp.plan.runType)
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if exp.plan.sampleTubeLabel is not None:
                if len(exp.plan.sampleTubeLabel) > 0:
                    metrics.append("sampTube:" + exp.plan.sampleTubeLabel)
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if len(exp.plan.templatingKitName) > 0:
                metrics.append("tplkit:" + exp.plan.templatingKitName)
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if exp.plan.libraryReadLength is None:
                metrics.append("libRdLn:none")
            else:
                metrics.append("libRdLn:" + str(exp.plan.libraryReadLength))
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if len(exp.plan.irworkflow) > 0:
                metrics.append("irworkflow:" + exp.plan.irworkflow)
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if len(exp.plan.applicationGroup.name) > 0:
                metrics.append("appGp:" + exp.plan.applicationGroup.name)
        except Exception:
            pass  # print traceback.format_exc()

        eas = exp.get_EAS()
        try:
            if eas.targetRegionBedFile == "":
                eas.targetRegionBedFile = "none"
            if eas.hotSpotRegionBedFile == "":
                eas.hotSpotRegionBedFile = "none"
            metrics.append("targetRegionBedFile:" + eas.targetRegionBedFile)
            metrics.append("hotSpotRegionBedFile:" + eas.hotSpotRegionBedFile)
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if len(eas.selectedPlugins) <= 0:
                metrics.append("pluginsUsed:none")
            else:
                plugins = ""
                for plugin in eas.selectedPlugins:
                    if len(plugins) > 0:
                        plugins = plugins + ","
                    plugins = plugins + str(plugin)

                metrics.append("pluginsUsed:" + plugins)
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if len(eas.reference) > 0:
                metrics.append("reflib:" + eas.reference)
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if len(eas.threePrimeAdapter) > 0:
                metrics.append("3pAdapter:" + eas.threePrimeAdapter)
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if len(eas.barcodeKitName) > 0:
                metrics.append("barcodeSet:" + eas.barcodeKitName)
        except Exception:
            pass  # print traceback.format_exc()

        try:
            ir_account_name = get_ir_account_name(eas)
            if ir_account_name:
                metrics.append("irCfgd:yes")
            else:
                metrics.append("irCfgd:no")
        except Exception:
            pass  # print traceback.format_exc()

        try:
            if eas.libraryKitName:
                metrics.append("libkit:" + str(eas.libraryKitName))
        except Exception:
            pass  # print traceback.format_exc()

        try:
            metrics.append("isBarcoded:" + str(exp.isBarcoded()))
        except Exception:
            pass  # print traceback.format_exc()

        try:
            nbarcodes = num_barcodes(run)
            metrics.append("numBarcodes:" + str(nbarcodes))
        except Exception:
            pass  # print traceback.format_exc()

        try:
            # get the names of all Ion Chef kits
            kits = iondb.rundb.models.KitInfo.objects.filter(
                kitType="IonChefPrepKit"
            ).values_list("name", flat=True)

            # if kit in list of Chef kits, then this run was an Ion Chef run.
            if exp.plan.templatingKitName in kits:
                metrics.append("chef:y")
            else:
                metrics.append("chef:n")
        except Exception:
            pass  # print traceback.format_exc()

        # report loading for run (% addressable wells that contain ISPs)
        wells_with_isps = 0
        addressable_wells = 0
        keys = ["bead", "empty", "pinned", "ignored"]
        for key in keys:
            try:
                val = list(run.analysismetrics_set.values())[0][key]
                if val:
                    addressable_wells = addressable_wells + val
                    if key == "bead":
                        wells_with_isps = val
            except IndexError:
                pass
            except:
                pass  # print traceback.format_exc()

        if addressable_wells > 0:
            pctload = 100.0 * float(wells_with_isps) / float(addressable_wells)
            pstr = "loading:{0:.3}".format(pctload)
            metrics.append(pstr)

        # write out the metrics
        runid = uuid.uuid1()
        fname = os.path.join("/var/spool/ion/", "TSexperiment-" + str(runid) + ".txt")
        # fname = os.path.join('/home/ionadmin/jt/',
        #                     'TSexperiment-' + str(result_id) + '.txt')
        outfile = open(fname, "w")

        try:
            outfile.write("\n".join(sorted(metrics)))
            outfile.write("\n")
        finally:
            outfile.close()
            os.chmod(fname, 0o0666)

        logger.debug("RSM createExperimentMetrics done resultId=" + str(result_id))
        return True, "RSM createExperimentMetrics"
    except Exception:
        logger.error(traceback.format_exc())
        return False, traceback.format_exc()


def get_ir_account_name(eas):
    """Return IR account name, or None if not found."""
    try:
        if eas.selectedPlugins:
            if eas.selectedPlugins.IonReporterUploader:
                if eas.selectedPlugins.IonReporterUploader.userInput:
                    inpt = eas.selectedPlugins.IonReporterUploader.userInput
                    if inpt.accountName:
                        if len(inpt.accountName) > 0:
                            return inpt.accountName
    except AttributeError:
        pass
    except:
        pass  # print traceback.format_exc()
    return None


# uncomment these lines for unit testing
# def main():
#    for run in iondb.rundb.models.Results.objects.all():
#        print 'id', run.id, run
#        createRSMExperimentMetrics(run.id)
#
#
# if __name__ == "__main__":
#    main()
