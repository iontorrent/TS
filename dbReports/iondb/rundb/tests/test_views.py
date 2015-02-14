# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

from django.test import TestCase


class ReportContextCheck(TestCase):
    """Caveat: This is not really a unit test.  It's more of a regression test which requries a specific report be
    available in order to run
    """

    def test_report_context(self):
        target = set(self.client.get("/report/1").context.keys())
        standard = set(
            "ProtonResultBlock",
            "addressable_wells",
            "avg_coverage_depth_of_target",
            "barcodes",
            "barcodes_json",
            "basecaller",
            "bead_loading",
            "bead_loading_threshold",
            "beadfind",
            "beadsummary",
            "c",
            "datasets",
            "dmfilestat",
            "duplicate_metrics",
            "encoder",
            "error",
            "experiment",
            "genome_length",
            "globalconfig",
            "has_major_plugins",
            "ionstats_alignment",
            "isInternalServer",
            "key_signal_threshold",
            "latex",
            "major_plugins",
            "major_plugins_images",
            "noheader",
            "noplugins",
            "otherReports",
            "output_file_groups",
            "plan",
            "pluginList",
            "qcTypes",
            "qs",
            "raw_accuracy",
            "read_stats",
            "reference",
            "report",
            "report_extra_tables",
            "report_pk",
            "report_status",
            "request",
            "software_versions",
            "testfragments",
            "usable_sequence",
            "usable_sequence_threshold"
        )
        self.assertEqual(target, standard)
