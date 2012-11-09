# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
import traceback
import json

from django.conf import settings

# Circular import
import iondb.rundb.models as models


## Legacy Config Nodes

def get_runinfo(env,plugin,primary_key,basefolder,plugin_out,url_root):
    dict={
            "raw_data_dir":env['pathToRaw'],
            "report_root_dir":env['report_root_dir'],
            "analysis_dir":env['analysis_dir'],
            "basecaller_dir":env['basecaller_dir'],
            "alignment_dir":env['alignment_dir'],
            "library_key":env['libraryKey'],
            "testfrag_key":env['testfrag_key'],
            "results_dir":os.path.join(env['report_root_dir'],basefolder,plugin_out),
            "net_location":env['net_location'],
            "url_root":url_root,
            "plugin_dir":plugin['path'], # compat
            "plugin_name":plugin['name'], # compat
            "plugin": plugin, # name,version,id,pluginresult_id,path
            "pk":primary_key
        }
    return dict

def get_pluginconfig(plugin):

    if hasattr(plugin, 'items'):
        # dictionary?
        pass
    elif hasattr(plugin, 'id'):
        #plugin object
        pass

    pluginjson = os.path.join(plugin['path'],'pluginconfig.json')
    d={}
    if os.path.exists( pluginjson ):
       f = open(pluginjson, 'r')
       try:
           d = json.load(f)
       finally:
           f.close()
    return d

def get_globalconfig():
    dict={
        "debug":0,
        "MEM_MAX":"15G"
    }
    return dict

def make_plugin_json(env,plugin,primary_key,basefolder,url_root):
    json_obj={
        "runinfo":get_runinfo(env,plugin,primary_key,basefolder,plugin['name']+"_out",url_root),
        "pluginconfig":get_pluginconfig(plugin),
        "globalconfig":get_globalconfig(),
        "plan":{},
    }
    if "plan" in env:
        json_obj["plan"] = env["plan"]
    if DEBUG:
        print json.dumps(json_obj,indent=2)
    return json_obj


class BaseModuleConfig(object):
    def __init__(self, config={}, **kwargs):
        pass


class PluginConfig(BaseModuleConfig):
    """ Class to generate startplugin.json from pluginresult """
    def __init__(self, pluginresultid, pluginconfig={}, **kwargs):
        super(PluginConfig,self).__init__(config=pluginconfig, **kwargs)
        self.legacyenv={}
        # Populate from PluginResultID
        self.pluginresult = models.PluginResult.objects.get(id=pluginresultid)
        self.plugin = self.pluginresult.plugin
        self.result = self.pluginresult.result
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def startpluginjson(self):
        pass

    def getLegacyEnv(self):
        # FIXME - fill in values
        #https://iontorrent.jira.com/wiki/display/TS/Plugin+json+file+format
        # Most comes from parent ion_params_00.json

        # Mirror path structure defined in pipeline scripts
        report_root = self.result.get_report_dir()
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

        # TODO PluginResult.path()
        plugin_output_dir = os.path.join(report_root,"plugin_out", plugin_orm.name + "_out")


        #get the hostname try to get the name from global config first
        sge_master = str(socket.getfqdn())
        gc = models.GlobalConfig.objects.all()[0]
        if gc.web_root:
            net_location = gc.web_root
        else:
            #if a hostname was not found in globalconfig.webroot then use what the system reports
            net_location = 'http://%s' % sge_master

        report = str(result.reportLink)
        reportList = report.split("/")[:-1]
        reportUrl = ("/").join(reportList)

        # URL ROOT is a relative path, no hostname,
        # as there is no canonical hostname for users visiting reports
        url_root = reportUrl
        api_url = net_location + "/rundb/api/"

        d = {
            'pathToRaw': self.result.experiment.unique,
            'report_root_dir': report_root,
            'library_key': self.result.libraryKey,
            'testfrag_key': self.result.testfragKey or 'ATCG',
            'plan': self.result.plan,

            'report_root_dir': report_root,
            'analysis_dir': report_root,

            'sigproc_dir': sigproc_results,
            'basecaller_dir': basecaller_results,
            'alignment_dir': alignment_results,

            'results_dir' : plugin_output_dir,

            'net_location' : net_location,
            'master_node': sge_master,
            'api_url': api_url,

            'tmap_version': settings.TMAP_VERSION,
        }
        return d

    def __repr__(self):

        env = self.getLegacyEnv()

        return { 'pluginResultID': getattr(self, 'pluginresultid', None),
                 # Legacy fields
                 "runinfo":get_runinfo(env,plugin,primary_key,basefolder,plugin['name']+"_out",url_root),
                 "pluginconfig": get_pluginconfig(plugin),
                 "globalconfig": get_globalconfig(),
                 "plan": env.get('plan', {}),
                 }

config=PluginConfig

