# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from django.test.utils import override_settings
import tempfile
import shutil
from iondb.rundb.models import GlobalConfig
import logging
import os.path
logger = logging.getLogger(__name__)

class PluginManagerTest(TestCase):
    fixtures = ['iondb/rundb/tests/views/report/fixtures/globalconfig.json']

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.assertEqual(1, len(GlobalConfig.objects.all().order_by('pk')))
        self.gc = GlobalConfig.objects.all().order_by('pk')[0]

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_init(self):
        with override_settings(PLUGIN_PATH = self.tempdir):
            from iondb.plugins.manager import PluginManager
            pm = PluginManager()
            self.assertEqual(pm.default_plugin_script, 'launch.sh')
            self.assertEqual(pm.pluginroot, os.path.normpath(self.tempdir))
            self.assertEqual(pm.infocache, {})
    
    def test_init_no_settings(self):
        with override_settings(PLUGIN_PATH = None):
            from iondb.plugins.manager import PluginManager
            pm = PluginManager()
            self.assertEqual(pm.default_plugin_script, 'launch.sh')
            self.assertEqual(pm.pluginroot, os.path.normpath(os.path.join("/results", self.gc.plugin_folder)))
            self.assertEqual(pm.infocache, {})