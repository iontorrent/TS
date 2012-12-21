# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from mockito import mock, when, any, verify, contains
import iondb.backup.tasks
from iondb.backup.tasks import xmlrpclib
from django.contrib.auth.models import User
from iondb.rundb.tests.views.report.test_report_action import verifyMessage
import tempfile
import shutil
from iondb.rundb.models import Experiment, Results, ReportStorage
from datetime import datetime
from iondb.rundb.tests.models.test_results import ResultsTest, create_result
from iondb.backup import ion_archiveResult


class ReportActionTaskTest(TestCase):
    fixtures = ['iondb/rundb/tests/views/report/fixtures/globalconfig.json',
                'iondb/rundb/tests/models/fixtures/groups.json',
                'iondb/rundb/tests/models/fixtures/users.json']

    def setUp(self):
        self.ionadmin = User.objects.get(username='ionadmin')
        self.report = create_result(self)

    def test_export_report(self):
        _id = self.report.id
        username = self.ionadmin.username
        comment = "abc"
        proxy = mock()
        when(xmlrpclib).ServerProxy(any(), allow_none=any()).thenReturn(proxy)
        proxyResult = True
        when(proxy).export_report(any(), any()).thenReturn(proxyResult)

        # returns method results directly
        result = iondb.backup.tasks.export_report(username, _id, comment)

        self.assertEquals(result[0], proxyResult)
        verifyMessage(self, proxyResult, self.report.resultsName)
        verify(proxy).export_report(any(), contains(comment))

    def test_prune_report(self):
        _id = self.report.id
        username = self.ionadmin.username
        comment = "abc"
        proxy = mock()
        when(xmlrpclib).ServerProxy(any(), allow_none=any()).thenReturn(proxy)
        proxyResult = True
        when(proxy).prune_report(any(), any()).thenReturn(proxyResult)

        # returns method results directly
        result = iondb.backup.tasks.prune_report(username, _id, comment)

        self.assertEquals(result[0], proxyResult)
        verifyMessage(self, proxyResult, self.report.resultsName)
        verify(proxy).prune_report(any(), contains(comment))

    def test_archive_report(self):
        _id = self.report.id
        username = self.ionadmin.username
        comment = "abc"
        proxy = mock()
        when(xmlrpclib).ServerProxy(any(), allow_none=any()).thenReturn(proxy)
        proxyResult = True
        when(proxy).archive_report(any(), any()).thenReturn(proxyResult)

        # returns method results directly
        result = iondb.backup.tasks.archive_report(username, _id, comment)

        self.assertEquals(result[0], proxyResult)
        verifyMessage(self, proxyResult, self.report.resultsName)
        verify(proxy).archive_report(any(), contains(comment))

    def test_archive_report_using_delay(self):
        _id = self.report.id
        username = self.ionadmin.username
        comment = "abc"
        proxy = mock()
        when(xmlrpclib).ServerProxy(any(), allow_none=any()).thenReturn(proxy)
        proxyResult = True
        when(proxy).archive_report(any(), any()).thenReturn(proxyResult)

        #returns ASyncResult wrapping method invocation results
        result = iondb.backup.tasks.archive_report.delay(username, _id, comment)

        self.assertEquals(result.get()[0], proxyResult)
        verifyMessage(self, proxyResult, self.report.resultsName)
        verify(proxy).archive_report(any(), contains(comment))


class ArchiveReportTaskTest(ResultsTest):

    def test_sync_filesystem_and_db_report_state(self):
        result = self.test_is_archived_false()
        result2 = self.test_is_archived_false()
        result3 = self.test_is_archived_false()
        
        reports = Results.objects.exclude(reportStatus__in=ion_archiveResult.STATUS)
        self.assertEquals(3, reports.count())
        
        iondb.backup.tasks.sync_filesystem_and_db_report_state.delay()
        
        self.assertEqual(0, Results.objects.filter(reportStatus__in=ion_archiveResult.STATUS).count())

        self._mark_fs_archived(result3)
        iondb.backup.tasks.sync_filesystem_and_db_report_state.delay()
        self.assertEqual(1, Results.objects.filter(reportStatus__in=ion_archiveResult.STATUS).count())
        
        self._mark_fs_archived(result)
        iondb.backup.tasks.sync_filesystem_and_db_report_state.delay()
        self.assertEqual(2, Results.objects.filter(reportStatus__in=ion_archiveResult.STATUS).count())
        
        self._mark_fs_archived(result2)
        iondb.backup.tasks.sync_filesystem_and_db_report_state.delay()
        self.assertEqual(3, Results.objects.filter(reportStatus__in=ion_archiveResult.STATUS).count())

        reports = Results.objects.exclude(reportStatus__in=ion_archiveResult.STATUS)
        self.assertEquals(0, reports.count())

        result4 = self.test_is_archived_false()
        reports = Results.objects.exclude(reportStatus__in=ion_archiveResult.STATUS)
        self.assertEquals(1, reports.count())
        
        self._mark_fs_archived(result4)
        iondb.backup.tasks.sync_filesystem_and_db_report_state.delay()
        self.assertEqual(4, Results.objects.filter(reportStatus__in=ion_archiveResult.STATUS).count())
        
        