# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from iondb.rundb.models import BackupConfig
import tempfile
import shutil
from iondb.rundb.tests.models.test_location import save_location

class BackupConfigModelTest(TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.location = save_location(self)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_save(self):
        bc = BackupConfig()
        bc.name = 'ArchiveTest'
        bc.online = True
        bc.location = self.location
        bc.backup_directory = ''
        bc.backup_threshold = 90
        bc.number_to_backup = 10
        bc.timeout = 60
        bc.bandwidth_limit = 0
        bc.status = '-'
        bc.comments = ''
        bc.save()
        self.assertIsNotNone(bc.id, 'BackupConfig id is None')
        return bc

    def test_unicode(self):
        rs = self.test_save()
        self.assertEquals(unicode(rs), rs.name)
