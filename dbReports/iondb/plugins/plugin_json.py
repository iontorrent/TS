#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import os
import traceback
import json
from ion.utils.explogparser import getparameter, getparameter_minimal
from ion.plugin.constants import RunLevel

from iondb.rundb.models import Chip, Results, RunType

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
        "pk": int(primary_key), # Result PK
        "tmap_version": ion_params.get('tmap_version',''),
        "library": ion_params.get('referenceName',''),
        "chipType": ion_params.get('chipType',''),
        "barcodeId": ion_params.get('barcodeId',''),
        "username": username,
        "platform": ion_params.get('platform',''),
        "systemType": ion_params.get('systemType','')
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

def get_chefSummary(ion_params, report_dir):
    chefSummary = {}
    exp_json = ion_params.get('exp_json', {})
    isChefRun = exp_json.get('chefInstrumentName', None)

    if isChefRun:
        chefSummary = {
            "chefReagentID": exp_json.get('chefReagentID'),
            "chefSolutionsPart": exp_json.get('chefSolutionsPart'),
            "chefLotNumber": exp_json.get('chefLotNumber'),
            "chefChipExpiration1": exp_json.get('chefChipExpiration1'),
            "chefChipExpiration2": exp_json.get('chefChipExpiration2'),
            "chefChipType1": exp_json.get('chefChipType1'),
            "chefChipType2": exp_json.get('chefChipType2'),
            "chefKitType": exp_json.get('chefKitType'),
            "chefManufactureDate": exp_json.get('chefManufactureDate'),
            "chefMessage": exp_json.get('chefMessage'),
            "chefPackageVer": exp_json.get('chefPackageVer'),
            "chefProgress": exp_json.get('chefProgress'),
            "chefReagentID": exp_json.get('chefReagentID'),
            "chefReagentsExpiration": exp_json.get('chefReagentsExpiration'),
            "chefReagentsLot": exp_json.get('chefReagentsLot'),
            "chefReagentsPart": exp_json.get('chefReagentsPart'),
            "chefSamplePos": exp_json.get('chefSamplePos'),
            "chefScriptVersion": exp_json.get('chefScriptVersion'),
            "chefSolutionsExpiration": exp_json.get('chefSolutionsExpiration'),
            "chefSolutionsLot": exp_json.get('chefSolutionsLot'),
            "chefSolutionsPart": exp_json.get('chefSolutionsPart'),
            "chefStatus": exp_json.get('chefStatus'),
            "chefTipRackBarcode": exp_json.get('chefTipRackBarcode'),
            "chefExtraInfo_1": exp_json.get('chefExtraInfo_1'),
            "chefExtraInfo_2": exp_json.get('chefExtraInfo_2'),
            "chefLogPath": exp_json.get('chefLogPath'),
            "chefLastUpdate": exp_json.get('chefLastUpdate'),
            "chefInstrumentName": exp_json.get('chefInstrumentName'),
        }

    return chefSummary

def get_expmeta(ion_params, report_dir):
    from datetime import datetime
    exp_json = ion_params.get('exp_json',{})
    
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
    
    chipBarcode = exp_json.get('chipBarcode')
    if not chipBarcode:
        # try to get from log
        log = exp_json.get('log',{})
        if log and not isinstance(log, dict): log = json.loads(log)
        chipBarcode = log.get('chip_efuse')

    runType = ion_params['plan']['runType']
    
    d = {
        "run_name": exp_json.get('expName'),
        "run_date": exp_json.get('date'),
        "run_flows": exp_json.get('flows'),
        "instrument": exp_json.get('pgmName'),
        "chiptype": exp_json.get('chipType'),
        "chipBarcode": chipBarcode,
        "notes": exp_json.get('notes'),
        "barcodeId": ion_params.get('barcodeId'),
        "results_name": ion_params.get('resultsName'),
        "flowOrder": ion_params.get('flowOrder'),
        "project": ion_params.get('project'),
        "runid": ion_params.get('runID',''),
        "sample": ion_params.get('sample'),
        "output_file_name_stem": exp_json.get('expName') + "_" + ion_params.get('resultsName'),
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

    runTypeDescription = ''
    try:
        runType = RunType.objects.get(runType=plan.get('runType'))
        runTypeDescription = runType.description
    except Exception:
        pass

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
        "runTypeDescription": runTypeDescription,
        "samplePrepKitName": plan.get('samplePrepKitName'),
        "templatingKitName": plan.get('templatingKitName',''),
        "username": plan.get('username'),
        
        "sampleGrouping": plan.get("sampleGrouping"),
        "sampleTubeLabel": plan.get("sampleTubeLabel"),
        
        "sequencekitname": ion_params.get('exp_json',{}).get('sequencekitname',''),
        "sseBedFile": eas.get('sseBedFile',''),
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
    try:
        for dmtype in dmtypes.FILESET_TYPES:
            retval[dmtype] = not(result.get_filestat(dmtype).isdisposed())
    except:
        retval = {}
    return retval


def make_plugin_json(primary_key, report_dir, plugin, plugin_out_dir, net_location, url_root, username, runlevel=RunLevel.DEFAULT, blockId='', block_dirs=["."], instance_config={}):
    try:
        ion_params,warn = getparameter(os.path.join(report_dir,'ion_params_00.json'))
    except:
        ion_params = getparameter_minimal(os.path.join(report_dir,'ion_params_00.json'))

    json_obj={
        "runinfo":get_runinfo(ion_params, primary_key, report_dir, plugin, plugin_out_dir, net_location, url_root, username, runlevel, blockId),
        "runplugin":get_runplugin(ion_params, runlevel, blockId, block_dirs),
        "expmeta": get_expmeta(ion_params, report_dir),
        "chefSummary": get_chefSummary(ion_params, report_dir),
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
