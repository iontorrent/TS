#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import os
import traceback
import json
from iondb.rundb.models import Chip

DEBUG = False

def get_runinfo(env,plugin,primary_key,basefolder,plugin_out,url_root):
    # Compat shim for new 2.2 values
    if 'report_root_dir' not in env:
        env['report_root_dir']=env['analysis_dir']
    if 'analysis_dir' not in env:
        env['analysis_dir']=env['report_root_dir']

    sigproc_dir = os.path.join(env['report_root_dir'],env['SIGPROC_RESULTS'])
    basecaller_dir = os.path.join(env['report_root_dir'],env['BASECALLER_RESULTS'])
    alignment_dir = os.path.join(env['report_root_dir'],env['ALIGNMENT_RESULTS'])
    
    # Fallback to 2.2 single folder layout
    if not os.path.exists(sigproc_dir):
        sigproc_dir = env['report_root_dir']
    if not os.path.exists(basecaller_dir):
        basecaller_dir = env['report_root_dir']
    if not os.path.exists(alignment_dir):
        alignment_dir = env['report_root_dir']

    # replace with Proton block folders for plugins running at blocklevel
    raw_data_dir = env['pathToRaw']
    if 'runlevel' in env.keys() and env['runlevel'] == 'block':
        if 'blockId' in env.keys() and env['blockId']:
            sigproc_dir = os.path.join(sigproc_dir, 'block_'+env['blockId'])
            basecaller_dir = os.path.join(basecaller_dir, 'block_'+env['blockId'])
            alignment_dir = os.path.join(alignment_dir, 'block_'+env['blockId'])
            raw_data_dir = os.path.join(env['pathToRaw'], 'block_'+env['blockId'])

    dict={
            "raw_data_dir":raw_data_dir,
            "report_root_dir":env['report_root_dir'],
            "analysis_dir":env['analysis_dir'],
            "sigproc_dir":sigproc_dir,
            "basecaller_dir":basecaller_dir,
            "alignment_dir":alignment_dir,
            "library_key":env['libraryKey'],
            "testfrag_key":env['tfKey'],
            "results_dir":os.path.join(env['report_root_dir'],basefolder,plugin_out),
            "net_location":env['net_location'],
            "url_root":url_root,
            "api_url": 'http://%s/rundb/api' % env['master_node'],
            "plugin_dir":plugin['path'], # compat
            "plugin_name":plugin['name'], # compat
            "plugin": plugin, # name,version,id,pluginresult_id,path
            "pk":primary_key,
            "tmap_version":env['tmap_version'],
            "library": env.get('libraryName'),
            "chipType": env.get('chipType',''),
            "barcodeId": env.get('barcodeId',''),
            "username": env.get('username','')
        }
    try:
        chipDescription = Chip.objects.filter(name=env.get('chipType','')).values_list('description',flat=True)[0]
    except:
        chipDescription = ''
    dict['chipDescription'] = chipDescription
        
    return dict
    
def get_runplugin(env):
    d = {
        "run_type": env.get('report_type','unknown'),
        "runlevel": env.get('runlevel', ''),
        "blockId":  env.get('blockId', '') if env.get('runlevel', 'none') == 'block' else '',
        "block_dirs": env.get('block_dirs', ["."]),
        "numBlocks": len(env.get('block_dirs', ["."]))
    }             
    return d  
      
def get_expmeta(env):
    from datetime import datetime
    exp_json = json.loads(env.get('exp_json','{}'))
    
    # compatibility fallback: expMeta.dat
    if not exp_json:        
        expmeta_file = os.path.join(env['report_root_dir'],'expMeta.dat')
        try:
          with open(expmeta_file, 'r') as f:
              expmeta = dict(line.replace(' ','').strip().split('=') for line in f)
              exp_json['expName'] = expmeta['RunName']
              exp_json['date'] = expmeta['RunDate']
              exp_json['flows'] = expmeta['RunFlows']
              exp_json['sample'] = expmeta['Sample']
              exp_json['instrument'] = expmeta['PGM']
              exp_json['chipType'] = expmeta['ChipType']
              exp_json['notes'] = expmeta['Notes']
        except:
          pass

    ion_params_path = os.path.join(env['report_root_dir'],'ion_params_00.json')
    d = {
        "run_name": exp_json.get('expName'),
        "run_date": exp_json.get('date'),
        "run_flows": exp_json.get('flows'),
        "instrument": exp_json.get('pgmName'),
        "chiptype": exp_json.get('chipType'),
        "chipBarcode": exp_json.get('chipBarcode') if exp_json.get('chipBarcode') else json.loads(exp_json.get('log','{}')).get('chip_efuse'),
        "notes": exp_json.get('notes'),
        
        "barcodeId": env.get('barcodeId'),
        "results_name": env.get('resultsName'),
        "flowOrder": env.get('flowOrder'),
        "project": env.get('project'),
        "runid": env.get('runID',''),
        "sample": env.get('sample'),
        "analysis_date": str(datetime.date(datetime.fromtimestamp(os.path.getmtime(ion_params_path)))),
    }
    return d

def get_pluginconfig(plugin):    
    d = plugin.get('pluginconfig',{})
    if not d:
        # 2.2 Compat
        pluginjson = os.path.join(plugin['path'],'pluginconfig.json')
        if not os.path.exists( pluginjson ):
            return {}
        with open(pluginjson, 'r') as f:
           d = json.load(f)
    return d

def get_globalconfig():
    dict={
        "debug":0,
        "MEM_MAX":"15G"
    }
    return dict

def get_plan(env):
    # re-create "plan" section from Plan, Experiment and ExperimentAnalysisSettings attributes
    plan = env.get('plan',{})
    if not plan:
        return {}
    exp = json.loads(env.get('exp_json','{}'))
    eas = env.get('experimentAnalysisSettings', {})
    d = {
        "barcodeId": eas.get('barcodeKitName',''),
        "barcodedSamples": eas.get('barcodedSamples',''),
        "bedfile": eas.get('targetRegionBedFile',''),
        "librarykitname": eas.get('libraryKitName',''),
        "regionfile": eas.get('hotSpotRegionBedFile'),
        "threePrimeAdapter": eas.get('threePrimeAdapter'),
        
        "controlSequencekitname": plan.get('controlSequencekitname',''),
        "planName": plan.get('planName'),
        "reverse_primer": plan.get('reverse_primer'),
        "runMode": plan.get('runMode'),
        "runType": plan.get('runType'),
        "samplePrepKitName": plan.get('samplePrepKitName'),
        "templatingKitName": plan.get('templatingKitName',''),
        "username": plan.get('username'),
        
        "sampleGrouping": plan.get("sampleGrouping"),
        "sampleTubeLabel": plan.get("sampleTubeLabel"),
        "sampleSet_name": plan.get("sampleSet_name"),
        "sampleSet_planIndex": plan.get("sampleSet_planIndex"),
        "sampleSet_planTotal": plan.get("sampleSet_planTotal"),
        "sampleSet_uid": plan.get("sampleSet_uid"),
        
        "sequencekitname": exp.get('sequencekitname',''), 
    }
    
    # compatibility: pre-EAS these attributes were part of Plan
    if not eas:
        old_keys = ["barcodeId", "barcodedSamples", "bedfile", "librarykitname", "regionfile", "forward3primeadapter"]
        for key in old_keys:
            d[key] = plan.get(key)        
    
    return d

def make_plugin_json(env,plugin,primary_key,basefolder,url_root):
    json_obj={
        "runinfo":get_runinfo(env,plugin,primary_key,basefolder,plugin['name']+"_out",url_root),
        "runplugin":get_runplugin(env),
        "expmeta": get_expmeta(env),
        "pluginconfig":get_pluginconfig(plugin),
        "globalconfig":get_globalconfig(),
        "plan":get_plan(env),
    }
    # IonReporterUploader_V1_0 compatibility shim
    if plugin["name"] == "IonReporterUploader_V1_0" and plugin.get("userInput",""):
        json_obj["plan"]["irworkflow"] = plugin["userInput"][0].get("Workflow")
        
    if DEBUG:
        print json.dumps(json_obj,indent=2)
    return json_obj

if __name__ == '__main__':
    '''Library of functions to handle json structures used in Plugin launching'''
    print 'python/ion/utils/plugin_json.py'
    sigproc_results = "sigproc_results"
    basecaller_results = "basecaller_results"
    alignment_results = "./"#"alignment_results"
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
    env['net_location']='http://localhost/' # Constructing internal URLS for plugin communication. Absolute hostname, resolvable by compute
    url_root = "/" # Externally Resolvable URL. Preferably without hardcoded hostnames

    plugin={'name': 'test_plugin',
            'path': '/results/plugins/test_plugin',
            'version': '0.1-pre',
           }
    pk=7
    jsonobj=make_plugin_json(env,plugin,pk,'plugin_out',url_root)
