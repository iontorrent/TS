# Copyright (C) 2019 Ion Torrent Systems, Inc. All Rights Reserved
import json
import logging
from datetime import timedelta

from django.core import serializers
from django.utils import timezone

from iondb.rundb.models import Message, Results, PluginResult, Rig, Experiment
from iondb.utils.utils import service_status, ion_processes
from collections import Counter

logger = logging.getLogger(__name__)

IRU_PLUGIN_NAME = "IonReporterUploader"


def get_server_metrics():
    metrics = {}

    # system loads/uptime, i.e. uptime
    with open("/proc/uptime") as fp:
        metrics["systemUptime"] = fp.read().strip()
    with open("/proc/loadavg") as fp:
        metrics["systemLoadavg"] = fp.read().strip()

    # GPU errors: /var/spool/ion/gpuErrors
    metrics["systemGpuErrors"] = None
    try:
        with open("/var/spool/ion/gpuErrors") as fp:
            metrics["systemGpuErrors"] = json.load(fp)
    except Exception as e:
        logger.exception(
            "Could not read /var/spool/ion/gpuErrors when collecting metrics!"
        )

    # hardware product name: from /etc/torrentserver/tsconf.conf
    # biosversion: from /etc/torrentserver/tsconf.conf
    metrics["systemHardwareName"] = "Unknown"
    metrics["systemBiosVersion"] = "Unknown"
    with open("/etc/torrentserver/tsconf.conf") as fp:
        for line in fp:
            if line.startswith("hardwarename:"):
                metrics["systemHardwareName"] = line.split(":")[1].strip()

            elif line.startswith("biosversion:"):
                metrics["systemBiosVersion"] = line.split(":")[1].strip()

    # service status
    metrics["systemServices"] = service_status(ion_processes())

    # Torrent Server Errors
    metrics["messages"] = serializers.serialize("json", Message.objects.all())

    # total reports
    metrics["reportCount"] = Results.objects.count()

    # last report date
    metrics["lastReportDate"] = (
        Results.objects.order_by("-timeStamp")[0].timeStamp
        if Results.objects.count()
        else None
    )

    # total run experiments
    metrics["experimentCount"] = Experiment.objects.filter(status="run").count()

    # last eun experiment date
    metrics["lastExperimentDate"] = (
        Experiment.objects.filter(status="run").order_by("-date")[0].date
        if Experiment.objects.filter(status="run").count()
        else None
    )

    # rig counts
    metrics["rigTypes"] = dict(Counter([r.type for r in Rig.objects.all()]))

    return metrics


def get_report_metrics(hours=36):
    reports = []
    # Get any results that have finished recently
    results = Results.objects.filter(
        timeStamp__gte=timezone.now() - timedelta(hours=hours)
    ).filter(status__iregex="complete|error")
    for result in results:
        experiment = result.experiment
        plan = experiment.plan
        eas = result.eas

        reports.append(
            {
                "reportName": result.resultsName,
                "reportDate": result.timeStamp,
                "reportId": result.id,
                "reportStatus": result.status,
                "experimentName": experiment.expName,
                "experimentDate": experiment.date,
                "templateName": plan.metaData.get("fromTemplate"),
                "planOrigin": plan.origin,
                "researchApplication": plan.applicationGroup.name
                if plan.applicationGroup
                else None,
                "targetTechnique": plan.runType,
                "researchCategories": plan.categories,
                "selectedPlugins": eas.selectedPlugins.keys(),
                "libraryKit": eas.libraryKitName,
                "templateKit": plan.templatingKitName,
                "sequencingKit": experiment.sequencekitname,
                "samplePrepProtocol": plan.samplePrepProtocol,
                "barcodeKit": eas.barcodeKitName,
                "calibrationMode": eas.base_recalibration_mode,
                "chipType": experiment.chipType,
                "references": [eas.reference]
                if not eas.barcodeKitName
                else eas.barcoded_samples_reference_names,
                "bedFiles": [
                    eas.targetRegionBedFile,
                    eas.hotSpotRegionBedFile,
                    eas.sseBedFile,
                ]
                if not eas.barcodeKitName
                else eas.barcoded_samples_bed_files,
                "hasIru": IRU_PLUGIN_NAME in eas.selectedPlugins,
                "usesChef": plan.is_ionChef(),
                "usesSystemTemplate": plan.metaData.get("fromTemplateSource") == "ION",
                "useCustomArgs": eas.custom_args,
                "useCustomKitSettings": plan.isCustom_kitSettings,
            }
        )
    return {"reports": reports}


def get_plugin_metrics(hours=36):
    plugins = []
    # Get any plugins that have finished recently
    plugin_results = PluginResult.objects.filter(
        plugin_result_jobs__endtime__gte=timezone.now() - timedelta(hours=hours)
    )
    for plugin_result in plugin_results:
        recent_job = plugin_result.plugin_result_jobs.order_by("-starttime").first()
        plugins.append(
            {
                "pluginName": plugin_result.plugin.name,
                "pluginVersion": plugin_result.plugin.version,
                "pluginStatus": recent_job.state,
                "reportId": plugin_result.result.id,
                "pluginResultId": plugin_result.id,
                "pluginStart": plugin_result.starttime(),
                "pluginEnd": plugin_result.endtime(),
                "isIonSupported": plugin_result.plugin.isSupported,
            }
        )
    return {"plugins": plugins}
