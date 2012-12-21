# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from iondb.rundb.models import ReportStorage
import tempfile
import shutil


class ReportStorageTest(TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.tempdir2 = tempfile.mkdtemp()
        self.tempdir3 = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)
        shutil.rmtree(self.tempdir2)
        shutil.rmtree(self.tempdir3)

    def test_save(self):
        rs = ReportStorage()
        rs.dirPath = self.tempdir
        rs.default = True
        rs.name = 'HomeTest'
        rs.webServerPath = '/outputTest'
        rs.save()
        self.assertIsNotNone(rs.id, 'ReportStorage id is None')
        return rs

    def test_save_method(self):
        rs = ReportStorage()
        rs.dirPath = self.tempdir
        rs.default = True
        rs.name = 'HomeTest'
        rs.webServerPath = '/outputTest'
        rs.save()

        rs = ReportStorage()
        rs.dirPath = self.tempdir2
        rs.default = True
        rs.name = 'HomeTest'
        rs.webServerPath = '/outputTest'
        rs.save()

        for rs in ReportStorage.objects.exclude(id=rs.id):
            self.assertFalse(rs.default, 'Should not be default report storage location, only one default location is allowed')

    def test_unicode(self):
        rs = self.test_save()
        self.assertEquals(unicode(rs), "%s (%s)" % (rs.name, rs.dirPath))
