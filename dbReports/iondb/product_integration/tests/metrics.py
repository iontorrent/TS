# Copyright (C) 2019 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from iondb.product_integration.metrics import (
    get_server_metrics,
    get_report_metrics,
    get_plugin_metrics,
)
from iondb.rundb.models import (
    PlannedExperiment,
    Experiment,
    Results,
    GlobalConfig,
    ExperimentAnalysisSettings,
    PluginResultJob,
    PluginResult,
    Plugin,
)
from django.contrib.auth.models import User
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)


class MetricsTest(TestCase):
    """Make sure we can gather metrics"""

    def setUp(self):
        GlobalConfig.objects.create(selected=True, site_name="Test Site")
        user = User.objects.create(username="ionadmin")
        plan = PlannedExperiment.objects.create(planName="Test Plan")
        experiment = Experiment.objects.create(
            plan=plan, date=timezone.now(), cycles=200, flows=300
        )
        eas = ExperimentAnalysisSettings.objects.create(
            experiment=experiment, barcodeKitName="Ion Dual Barcode Kit 1-96"
        )
        result = Results.objects.create(
            resultsName="Test Report",
            experiment=experiment,
            processedCycles=200,
            status="complete",
            processedflows=300,
            framesProcessed=300,
            eas=eas,
        )
        plugin = Plugin.objects.create(name="Test Plugin", version="2.3.4")
        plugin_result = PluginResult.objects.create(
            result=result, owner=user, plugin=plugin
        )
        plugin_result_job = PluginResultJob.objects.create(plugin_result=plugin_result, endtime=timezone.now())

    def test_get_server_metrics(self):
        metrics = get_server_metrics()
        self.assertTrue(type(metrics["messages"]), list)
        logger.debug(metrics)

    def test_get_report_metrics(self):
        metrics = get_report_metrics(hours=24 * 365 * 10)
        self.assertTrue(len(metrics["reports"]) == 1)
        self.assertEquals(metrics["reports"][0]["reportName"], "Test Report")
        self.assertEquals(metrics["reports"][0]["barcodeKit"], "Ion Dual Barcode Kit 1-96")
        logger.debug(metrics)

    def test_get_plugin_metrics(self):
        metrics = get_plugin_metrics(hours=24 * 365 * 10)
        self.assertEquals(metrics["plugins"][0]["pluginName"], "Test Plugin")
        logger.debug(metrics)
