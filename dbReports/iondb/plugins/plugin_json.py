#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import os
import traceback
import json
from ion.utils.explogparser import getparameter, getparameter_minimal
from ion.plugin.constants import RunLevel

from iondb.rundb.models import Chip, Results

def get_runinfo(ion_params, primary_key, report_dir, plugin, plugin_out_dir, net_location, url_root, username, runlevel, blockId):

    raw_data_dir = ion_params['pathToRaw']
    
    sigproc_dir = os.path.normpath(os.path.join(report_dir, ion_params['SIGPROC_RESULTS']))
    basecaller_dir = os.path.normpath(os.path.join(report_dir, ion_params['BASECALLER_RESULTS']))
    alignment_dir = os.path.normpath(os.path.join(report_dir, ion_params['ALIGNMENT_RESULTS']))
    
    # Fallback to 2.2 single folder layout
    if not os.path.exists(sigproc_dir):
        sigproc_dir = report_dir
    if not os.path.exists(basecaller_dir):
        basecaller_dir = report_dir
    if not os.path.exists(alignment_dir):
        alignment_dir = report_dir

    # replace with Proton block folders for plugins running at blocklevel
    if runlevel == RunLevel.BLOCK and blockId:
        block = 'block_'+ blockId
        sigproc_dir = os.path.join(sigproc_dir, block)
        basecaller_dir = os.path.join(basecaller_dir, block)
        alignment_dir = os.path.join(alignment_dir, block)
        raw_data_dir = os.path.join(ion_params['pathToRaw'], block)
        plugin_out_dir = os.path.join(report_dir, block, plugin_out_dir.replace(report_dir,'').lstrip('/') )
    
    #TS-5227: provide separate folder to store plugin 'post' runlevel output
    if runlevel == RunLevel.POST:
        plugin_out_dir = os.path.join(plugin_out_dir, 'post')

    d = {
        "raw_data_dir": raw_data_dir,
        "report_root_dir": report_dir,
        "analysis_dir": report_dir, # compat
        "sigproc_dir": sigproc_dir,
        "basecaller_dir": basecaller_dir,
        "alignment_dir": alignment_dir,
        "library_key": ion_params.get('libraryKey',''),
        "testfrag_key": ion_params.get('tfKey',''),
        "results_dir": plugin_out_dir,
        "net_location": net_location,
        "url_root": url_root,
        "api_url": net_location + '/rundb/api',
        "plugin_dir": plugin['path'], # compat
        "plugin_name": plugin['name'], # compat
        "plugin": plugin, # name,version,id,path
        "pk": primary_key, # Result PK
        "tmap_version": ion_params.get('tmap_version',''),
        "library": ion_params.get('referenceName',''),
        "chipType": ion_params.get('chipType',''),
        "barcodeId": ion_params.get('barcodeId',''),
        "username": username,
    }
    
    try:
        chipDescription = Chip.objects.filter(name=ion_params.get('chipType','')).values_list('description',flat=True)[0]
    except:
        chipDescription = ''
    d['chipDescription'] = chipDescription
        
    return d
    
def get_runplugin(ion_params, runlevel, blockId, block_dirs):
    d = {
        "run_type": ion_params.get('report_type','unknown'),
        "runlevel": runlevel,
        "blockId":  blockId if runlevel == RunLevel.BLOCK else '',
        "block_dirs": block_dirs,
        "numBlocks": len(block_dirs)
    }             
    return d  
      
def get_expmeta(ion_params, report_dir):
    from datetime import datetime
    exp_json = json.loads(ion_params.get('exp_json','{}'))
    
    # compatibility fallback: expMeta.dat
    if not exp_json:        
        expmeta_file = os.path.join(report_dir,'expMeta.dat')
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

    ion_params_path = os.path.join(report_dir,'ion_params_00.json')
    d = {
        "run_name": exp_json.get('expName'),
        "run_date": exp_json.get('date'),
        "run_flows": exp_json.get('flows'),
        "instrument": exp_json.get('pgmName'),
        "chiptype": exp_json.get('chipType'),
        "chipBarcode": exp_json.get('chipBarcode') if exp_json.get('chipBarcode') else json.loads(exp_json.get('log','{}')).get('chip_efuse'),
        "notes": exp_json.get('notes'),
        
        "barcodeId": ion_params.get('barcodeId'),
        "results_name": ion_params.get('resultsName'),
        "flowOrder": ion_params.get('flowOrder'),
        "project": ion_params.get('project'),
        "runid": ion_params.get('runID',''),
        "sample": ion_params.get('sample'),
        "analysis_date": str(datetime.date(datetime.fromtimestamp(os.path.getmtime(ion_params_path)))),
    }
    return d

def get_pluginconfig(plugin, instance_pluginconfig):
    # Return first non-empty configuration in the following order:
    #   manual launch: instance pluginconfig, instance.html
    #   global pluginconfig: plugin db record, config.html
    #   planned parameters: userInput, plan.html
    pluginconfig = instance_pluginconfig or plugin.get('pluginconfig',{}) or plugin.get('userInput', {}) or {}

    if not pluginconfig:
        # 2.2 Compat
        pluginjson = os.path.join(plugin['path'],'pluginconfig.json')
        if os.path.exists( pluginjson ):
            with open(pluginjson, 'r') as f:
                pluginconfig = json.load(f)
    if not pluginconfig:
        pluginconfig = {}
    return pluginconfig

def get_globalconfig():
    dict={
        "debug":0,
        "MEM_MAX":"15G"
    }
    return dict

def get_plan(ion_params):
    # re-create "plan" section from Plan, Experiment and ExperimentAnalysisSettings attributes
    plan = ion_params.get('plan',{})
    if not plan:
        return {}
    exp = json.loads(ion_params.get('exp_json','{}'))
    eas = ion_params.get('experimentAnalysisSettings', {})
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

def get_datamanagement(pk, result=None):
    if not result:
        result = Results.objects.get(pk=pk)
    # provide status of files: True means files are available
    from iondb.rundb.data import dmactions_types as dmtypes
    retval = {}
    for dmtype in dmtypes.FILESET_TYPES:
        retval[dmtype] = not(result.get_filestat(dmtype).isdisposed())
    return retval


def make_plugin_json(primary_key, report_dir, plugin, plugin_out_dir, net_location, url_root, username,
                    runlevel=RunLevel.DEFAULT, blockId='', block_dirs=["."], instance_config={}):
    try:
        ion_params,warn = getparameter(os.path.join(report_dir,'ion_params_00.json'))
    except:
        ion_params = getparameter_minimal(os.path.join(report_dir,'ion_params_00.json'))
        
    json_obj={
        "runinfo":get_runinfo(ion_params, primary_key, report_dir, plugin, plugin_out_dir, net_location, url_root, username, runlevel, blockId),
        "runplugin":get_runplugin(ion_params, runlevel, blockId, block_dirs),
        "expmeta": get_expmeta(ion_params, report_dir),
        "pluginconfig":get_pluginconfig(plugin, instance_config),
        "globalconfig":get_globalconfig(),
        "plan":get_plan(ion_params),
        "sampleinfo": ion_params.get("sampleInfo",{}),
        "datamanagement": get_datamanagement(primary_key),
    }
    # IonReporterUploader_V1_0 compatibility shim
    if plugin["name"] == "IonReporterUploader_V1_0" and plugin.get("userInput",""):
        json_obj["plan"]["irworkflow"] = plugin["userInput"][0].get("Workflow")

    return json_obj
