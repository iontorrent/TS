# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from iondb.rundb.models import Plugin
from iondb.rundb.configure.views import get_object_or_404
from mockito import when
from django.core.urlresolvers import reverse

class PluginViewTest(TestCase):
    def test_configure_plugins_plugin_refresh_404(self):
        _id = 0
        data = {}
        response = self.client.get('configure/plugins/%s/refresh/' % _id, data)
        self.assertEqual(response.status_code, 404)

    def test_configure_plugins_plugin_refresh(self):
        _id = 0
        plugin = Plugin()
        plugin.name = "Foo"
        #skip call to DB and return this Plugin stub instead
        when(iondb.rundb.configure.views).get_object_or_404(Plugin,_id).thenReturn(plugin)
        
        data = {}
        response = self.client.get('configure/plugins/%s/refresh/' % _id, data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['plugin'], plugin)
        self.assertEqual(response.context['action'], reverse('api_dispatch_info', kwargs={'resource_name': 'plugin', 'api_name': 'v1', 'pk': int(_id)}) + '?use_cache=false')
        self.assertEqual(response.context['method'], 'get')
        
        self.assertIn('Refresh Plugin %s Information' % plugin.name, response.content)