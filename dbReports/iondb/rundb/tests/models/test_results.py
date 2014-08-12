# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from iondb.rundb.models import Results, Experiment, ReportStorage
from django.contrib.auth.models import User
from datetime import datetime
import tempfile
import shutil
from os import path
import os

import logging
from iondb.utils import touch
import uuid
logger = logging.getLogger(__name__)

def create_result(self):
    exp = Experiment()
    exp.date = datetime.now()
    exp.cycles = 0
    exp.flows = 0
    exp.unique = uuid.uuid1()
    exp.save()
    result = Results()
    result.resultsName = 'foo'
    result.experiment = exp
    result.processedCycles = 0
    result.processedflows = 0
    result.framesProcessed = 0
    result.save()
    self.assertIsNotNone(result.id, 'Result id is None')
    return result

class ResultsTest(TestCase):
    
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        rs = ReportStorage()
        rs.dirPath = self.tempdir
        rs.default = True
        rs.name = 'HomeTest'
        rs.webServerPath = '/outputTest'
        rs.save()
        self.reportstorage = rs

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_save(self):
        result = create_result(self)
        return result

    def test_unicode(self):
        result = self.test_save()
        self.assertEquals(unicode(result), result.resultsName)

#retired_test
#    def test__findReportStorage_none(self):
#        result = self.test_save()
#        result.reportstorage = None
#        result.reportLink = path.join(self.reportstorage.webServerPath, self.reportstorage.name, result._basename(),'')
#        logger.debug(result.reportLink)
#        self.assertIsNone(result._findReportStorage(), 'Should be None')
#        return result

#retired_test
#    def test__findReportStorage_found(self):
#        result = self.test__findReportStorage_none()
#        resultFSDirectory = path.join(self.reportstorage.dirPath ,self.reportstorage.name, result._basename())
#        os.makedirs(resultFSDirectory)
#        self.assertTrue(os.path.exists(resultFSDirectory), 'Directory should exist')
#        
#        self.assertIsNotNone(result._findReportStorage(), 'Should not be None')
#        self.assertEqual(self.reportstorage.id, result._findReportStorage().id, 'Should be same')
#        return result

#retired_test
#    def test_get_report_dir_is_none(self):
#        result = self.test__findReportStorage_none()
#        result.reportstorage = None
#        self.assertIsNone(result.get_report_dir(), 'Expecting None')

#retired_test
#    def test_get_report_path_is_none(self):
#        result = self.test__findReportStorage_none()
#        result.reportstorage = None
#        self.assertIsNone(result.get_report_path(), 'Expecting None')
#        return result

#retired_test
#    def test_get_report_path_is_found(self):
#        result = self.test__findReportStorage_found()
#        result.reportstorage = None
#        self.assertIsNotNone(result.get_report_path(), 'Expecting None')
#        _expected = path.join(self.reportstorage.dirPath ,self.reportstorage.name, result._basename(), '')
#        self.assertEqual(_expected, result.get_report_path(), 'Expecting valid path %s' % _expected)
#        self.assertIsNotNone(result.reportstorage, 'Should have been set')
#        return result

#retired_test
#    def test_get_report_dir_is_found(self):
#        result = self.test__findReportStorage_found()
#        result.reportstorage = None
#        self.assertIsNotNone(result.get_report_dir(), 'Expecting None')
#        _expected = path.join(self.reportstorage.dirPath ,self.reportstorage.name, result._basename())
#        self.assertEqual(_expected, result.get_report_dir(), 'Expecting valid path %s' % _expected)
#        self.assertIsNotNone(result.reportstorage, 'Should have been set')

#retired_test
#    def test_is_archived_false(self):
#        result = self.test__findReportStorage_found()
#        self.assertFalse(result.is_archived(), 'Should be archivable (not archived yet)')
#        return result

    def _mark_fs_archived(self, result):
        for f in ['report.pdf', "%s.support.zip" % os.path.basename(result.get_report_dir())]:
            _file = path.join(result.get_report_dir(), f)
            touch(_file)

#retired_test
#    def test_is_archived_true(self):
#        result = self.test__findReportStorage_found()
#        self._mark_fs_archived(result)
#        self.assertTrue(result.is_archived(), 'Should be archived (not archived yet)')

#retired_test
#    def test_report_exist_none(self):
#        result = self.test_get_report_path_is_none()
#        self.assertFalse(result.report_exist())

#retired_test
#    def test_report_exist_false(self):
#        result = self.test_get_report_path_is_found()
#        shutil.rmtree(result.get_report_dir())
#        self.assertFalse(result.report_exist())

#retired_test
#    def test_report_exist_true(self):
#        result = self.test_get_report_path_is_found()
#        self.assertTrue(result.report_exist())