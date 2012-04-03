#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import os
import traceback
import json

DEBUG = False

def get_runinfo(env,plugin,primary_key,basefolder,plugin_out,url_root):
    dict={
            "raw_data_dir":env['pathToRaw'],
            "report_root_dir":env['report_root_dir'],
            "analysis_dir":env['analysis_dir'],
            "sigproc_dir":env['sigproc_dir'],
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

if __name__ == '__main__':
    '''Library of functions to handle json structures used in Plugin launching'''
    print 'python/ion/utils/plugin_json.py'
    analysis_results = "."
    basecaller_results = "."
    alignment_results = "."
    env={}
    env['pathToRaw']='/results/PGM_test/cropped_CB1-42'
    env['libraryKey']='TCAG'
    env['report_root_dir']=os.getcwd()
    env['analysis_dir']=os.getcwd()
    env['sigproc_dir']=os.path.join(env['report_root_dir'],sigproc_results)
    env['basecaller_dir']=os.path.join(env['report_root_dir'],basecaller_results)
    env['alignment_dir']=os.path.join(env['report_root_dir'],alignment_results)
    env['testfrag_key']='ATCG'

    #todo
    env['net_location']='???'

    plugin={}
    plugin['name']='test_plugin'
    plugin['path']='/results/plugins/test_plugin'
    pk=7
    jsonobj=make_plugin_json(env,plugin,pk,'plugin_out',"http://localhost")
