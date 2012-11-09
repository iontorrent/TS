# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import string
import os

from django.conf import settings

from django.template.loader import render_to_string
from django.utils.encoding import smart_str

import logging
import json

from iondb.plugins.config import PluginConfig
# From pipeline package
from ion.plugin.remote import callPluginXMLRPC

from iondb.rundb import models
from django.forms.models import model_to_dict


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
            json.dump(output_start_json,fp,indent=2)
        return

def launch_plugin(result, plugin, params={}):
    #launcher = PluginRunner(result=result, plugin=plugin, params=params)

    # FIXME  -- PluginConfig

    plugin_output_dir = os.path.join(result.get_report_dir(),"plugin_out", plugin_orm.name + "_out")

    # Mirror path structure defined in pipeline scripts
    report_root = result.get_report_dir()
    sigproc_results = os.path.join(report_root, 'sigproc_results')
    basecaller_results = os.path.join(report_root, 'basecaller_results')
    alignment_results = os.path.join(report_root, 'alignment_results')

    # Fallback to 2.2 single folder layout
    if not os.path.exists(sigproc_results):
        sigproc_results = report_root
    if not os.path.exists(basecaller_results):
        basecaller_results = report_root
    if not os.path.exists(alignment_results):
        alignment_results = report_root

    env={
        'pathToRaw':result.experiment.unique,
        'report_root_dir':result.get_report_dir(),
        'analysis_dir': sigproc_results,
        'basecaller_dir': basecaller_results,
        'alignment_dir': alignment_results,
        'libraryKey':result.experiment.libraryKey,
        'results_dir' : plugin_output_dir,
        'net_location' : hostname,
        'testfrag_key':'ATCG',
    }

    # if thumbnail
    is_thumbnail = result.metaData.get("thumb",False)
    if is_thumbnail:
        env['pathToRaw'] =  os.path.join(env['pathToRaw'],'thumbnail')

    plugindata = model_to_dict(plugin)

    start_json = make_plugin_json(env,plugindata,result.pk,"plugin_out",url_root)

    # Override plugin config with instance config
    start_json["pluginconfig"].update(params)

    # Set Started status before launching to avoid race condition
    # Set here so it appears immediately in refreshed plugin status list
    (pluginresult, created) = result.pluginresult_set.get_or_create(plugin=plugin)
    pluginresult.prepare(config=start_json) ## Set Pending state
    pluginresult.save()

    # Created necessary folders
    if not os.path.exists(start_json['runinfo']['results_dir']):
        os.makedirs(start_json['runinfo']['results_dir'])
    if not os.path.exists(pluginresult.path()):
        os.makedirs(pluginresult.path())

    # Create individual launch script from template and plugin launch.sh
    (launch, iscompat) = pluginmanager.find_pluginscript(plugin_path, plugin_name)
    if not launch or not os.path.exists(launch):
        logger.error("Analysis: %s. Path to plugin script: '%s' Does Not Exist!" % (analysis_name,launch))

    launcher = PluginRunner()
    if isCompat:
        launchScript = launcher.createPluginWrapper(launch, start_json)
    else:
        start_json.update({'command': ["python '%s'" % launch]})
        launchScript = launcher.createPluginWrapper(None, start_json)
    launchWrapper = launcher.writePluginLauncher(pluginDir, plugin_name, launchScript)
    launcher.writePluginJson(start_json)

    ## TODO Launch SGE task via DRMAA
    ret = callPluginXMLRPC(start_json)

    if ret < 0:
        logger.error('Unable to launch plugin: %s', plugin_orm.name) # See ionPlugin.log for details
        pluginresult.complete(state='Error')
        pluginresult.save()

    # Percolating through queue *should* take longer it takes to update status...
    pluginresult = result.pluginresult_set.get(pk=pluginresult.pk)
    if pluginresult.state == 'Pending':
        pluginresult.state = "Queued"
    pluginresult.jobid = ret
    pluginresult.save()

    return

def launch_plugin_byname(result, plugin_name, plugin_version=None):
    plugin = lookup_plugin(plugin_name, plugin_version)
    ret = launch_plugin(result, plugin)
    return ret

