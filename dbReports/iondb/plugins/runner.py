# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import string
import os

from django.conf import settings

from django.template.loader import render_to_string
from django.utils.encoding import smart_str

import logging
import json

from iondb.rundb import models
from django.forms.models import model_to_dict
from iondb.rundb.json_field import JSONEncoder
from iondb.rundb.barcodedata import BarcodeSampleInfo

"""
This class is used by components which can access the database directly. Remote instances must use XMLRPC or REST API calls.

Only result, plugin, and plugin params are necessary to launch a plugin, all other context can be gathered from these values, and is done so here.


"""

class InvalidPlugin(Exception):
    def __init__(self, name, version, **kwargs):
        self.name = name
        self.version = version
        return super(InvalidPlugin, self).__init__(**kwargs)

class PluginRunner():
    def __init__(self, result=None, plugin=None, params=None):
        # Default Settings
        self.context = {
            'memory': '8G',
            'debug': settings.TEST_INSTALL,
            'version': 0,
            'command': [],
            'sge': [],
        }

        if result:
            self.result =  self.lookupResult(result)
        else:
            self.result = None
        if plugin:
            self.plugin = self.lookupPlugin(plugin)
        else:
            self.plugin = None

        self.params = params

    def lookupResult(self, result):
        if not result: return result
        if hasattr(result, 'pk'): return result
        ## lookup result object by pk
        try:
            r = models.Results.objects.get(pk=result)
        except models.Results.DoesNotExist:
            return None
        return r

    def setPlugin(self, plugin, version=None):
        # plugin instance

        # plugin name and version
        self.plugin = lookupPlugin(plugin, version)

        # plugin name with version suffix

        # plugin name - get latest version
        pass

    def lookupPlugin(self, plugin_name, plugin_version=None):
        """ Find a plugin by name and version. If plugin_version is none, find latest version """
        if plugin_version:
            # Exact match. Raises Plugin.DoesNotExist or Plugin.MultipleObjectsReturned
            plugin = models.Plugin.objects.get(name=plugin_name,version=plugin_version,active=True)
        else:
            qs = models.Plugin.objects.filter(name=plugin_name,active=True).exclude(path='')
            if not qs:
                raise models.Plugin.DoesNotExist()
            plugin = None
            ## Get Latest version of plugin - must iterate through all to do proper version comparison
            for p in qs:
                if (not plugin) or p.versionGreater(plugin):
                    plugin = p

        if not plugin:
            raise models.Plugin.DoesNotExist()
        return plugin

    def writePluginLauncher(self, pluginout, pluginname, content):
        """ Write expanded shell script for plugin job """
        pluginWrapperFile = os.path.join(pluginout, "ion_plugin_%s_launch.sh" % pluginname)
        with open(pluginWrapperFile, 'w') as f:
            f.write(content)
        os.chmod(pluginWrapperFile, 0775)
        return pluginWrapperFile

    def createPluginWrapper(self, launchFile, start_json):
        c = {}
        c.update(self.context)

        # Override config with plugin specific settings
        c.update(start_json['globalconfig'])
        c.update(start_json['pluginconfig'])
        c.update(start_json['runinfo'])
        c.update(start_json) # keep explicit values at original keys

        # Set essential plugin execution values to expected aliases
        c['pluginout']  = c["results_dir"]
        c['pluginpath'] = c["plugin_dir"]
        c['pluginname'] = c["plugin_name"]

        # If there's no memory request, fall back to globalconfig MEM_MAX
        if (not "memory" in c) and "MEM_MAX" in c:
            c["memory"] = c["MEM_MAX"]

        # Allow launch.sh to define some attributes...
        if launchFile:
            with open(launchFile, 'r') as launch:
                for line in launch:
                    # Messy.
                    if line.startswith("VERSION="):
                        # Fixme version is quoted...
                        c["version"] = line.split('=')[1].strip()

                    if line.startswith("#!"):
                        # Skip #! and special comments
                        pass
                    elif line.startswith("#$"):
                        # SGE Resources 
                        c["sge"].append(line)
                    else:
                        c["command"].append(line)
        # else - new launch class - comamnd passed in via start_json

        # Flatten arrays to strings
        c["sge"] = string.join(c["sge"], '')
        c["command"] = string.join(c["command"], '')

        # Create pluginWrapperFile from templateFile
        templateFile = "plugin/ion_plugin_wrapper.sh.tmpl"
        content = render_to_string(templateFile, c)
        return smart_str(content)

    def writePluginJson(self, start_json):
        output_start_json = {}
        output_start_json.update(start_json)
        if 'command' in output_start_json:
            del(output_start_json['command'])
        start_json_fname = os.path.join(start_json['runinfo']['results_dir'],'startplugin.json')
        with open(start_json_fname,"w") as fp:
            json.dump(output_start_json,fp,indent=2,cls=JSONEncoder)

        # Also write new barcodes.json
        try:
            if self.result:
                barcodeSampleInfo = BarcodeSampleInfo(self.result.id, self.result)
            else:
                barcodeSampleInfo = BarcodeSampleInfo(start_json['runinfo']['pk'])
            barcode_json_fname = os.path.join(start_json['runinfo']['results_dir'], 'barcodes.json')
            
            with open(barcode_json_fname, "w") as fp:
                json.dump(barcodeSampleInfo.data(start_json), fp, indent=2,cls=JSONEncoder)
            
        except IOError as e:
            logger = logging.getLogger(__name__)
            logger.error(e)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.exception("Failed to write barcodes.json --- " + str(e))

        return start_json_fname

