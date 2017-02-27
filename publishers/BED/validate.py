#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import subprocess
import json
import traceback
import argparse
import zipfile
from pprint import pprint
from decimal import Decimal
import sys
import call_api as api
from iondb.bin.djangoinit import *
from iondb.rundb import models
from iondb.rundb.plan import ampliseq
from iondb.utils import files as file_utils
import shutil


def register(upload_id, base_path, file, meta):
    full_path = os.path.join(base_path, file)
    reg = "/%s" % file
    pub_uid = "/rundb/api/v1/publisher/BED/"
    upload_uid = "/rundb/api/v1/contentupload/%d/" % upload_id
    api.post("content", publisher=pub_uid, meta=json.dumps(meta),
             file=full_path, path=reg, contentupload=upload_uid)


def register_bed_file(upload_id, base_path, meta, bed_name):
    # Register files to Publisher
    register(upload_id, base_path, meta["reference"]+"/unmerged/plain/"+bed_name, meta)
    register(upload_id, base_path, meta["reference"]+"/unmerged/detail/"+bed_name, meta)
    register(upload_id, base_path, meta["reference"]+"/merged/plain/"+bed_name, meta)
    register(upload_id, base_path, meta["reference"]+"/merged/detail/"+bed_name, meta)


def is_BED_encrypted(meta):
    if 'key' in meta.get('pre_process_files'):
        return True
    return False


def validate(upload_id, base_path, meta, bed_file, bed_type):
    print("Validating %s file: %s" % (bed_type, bed_file))
    print
    path_end = '/'+meta["reference"]+"/unmerged/detail/"+bed_file
    data, response, raw = api.get("content", publisher_name='BED', format='json', path__endswith=path_end)

    if int(data['meta']['total_count']) > 0:
        if meta['is_ampliseq']:
            return data['objects'][0]['file']
        #api.post('log', upload='/rundb/api/v1/contentupload/%s/' % str(upload_id),
        #         text='Error: The file %s already exists. Please rename your file.'%bed_file)
        print 'ERROR: The file %s already exists. Please rename your file.' % bed_file
        sys.exit(1)

    result_UD_dir = os.path.join(base_path, meta['reference'], 'unmerged', 'detail')
    result_UP_dir = os.path.join(base_path, meta['reference'], 'unmerged', 'plain')
    result_MD_dir = os.path.join(base_path, meta['reference'], 'merged', 'detail')
    result_MP_dir = os.path.join(base_path, meta['reference'], 'merged', 'plain')
    if not os.path.exists(result_UD_dir):
        os.makedirs(result_UD_dir)
    if not os.path.exists(result_UP_dir):
        os.makedirs(result_UP_dir)
    if not os.path.exists(result_MD_dir):
        os.makedirs(result_MD_dir)
    if not os.path.exists(result_MP_dir):
        os.makedirs(result_MP_dir)

    #output_log = os.path.join(base_path, bed_file+'.log')
    output_json = os.path.join(base_path, bed_file+'.json')

    cmd = '/usr/local/bin/tvcutils validate_bed'
    cmd += '  --reference /results/referenceLibrary/tmap-f3/%s/%s.fasta' % (
        meta['reference'], meta['reference'])
    if bed_type == 'target regions BED':
        cmd += '  --target-regions-bed "%s"' % os.path.join(base_path,        bed_file)
    elif bed_type == 'hotspots BED':
        cmd += '  --hotspots-bed "%s"' % os.path.join(base_path,        bed_file)

    cmd += '  --unmerged-detail-bed "%s"' % os.path.join(result_UD_dir,    bed_file)
    cmd += '  --unmerged-plain-bed "%s"' % os.path.join(result_UP_dir,    bed_file)
    cmd += '  --merged-detail-bed "%s"' % os.path.join(result_MD_dir,    bed_file)
    cmd += '  --merged-plain-bed "%s"' % os.path.join(result_MP_dir,    bed_file)
    #cmd +=      '  --validation-log "%s"'       % output_log
    cmd += '  --meta-json "%s"' % output_json
    #print cmd
    p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print p.communicate()[0]
    #sys.stderr.write("=== -.- ===\n")
    #print stdout
    #print stderr

    #if os.path.exists(output_log):
    #  for line in open(output_log):
    #      api.post('log', upload='/rundb/api/v1/contentupload/%s/' % str(upload_id), text=line.strip())

    if os.path.exists(output_json):
        with open(output_json) as json_file:
            meta.update(json.load(json_file))

    if p.returncode != 0:
        sys.exit(p.returncode)

    return None

def decorate_S5_instruments(instrument_type):
    if instrument_type in ['520', '521', '530', '540']:
        instrument_type = "S5 Chip: " + instrument_type.upper()
    return instrument_type

def plan_json(meta, upload_id, primary_path, secondary_path):
    run_type = meta['design']['plan'].get('runType', None)
    plan_name = meta["design"]["design_name"].encode("ascii", "ignore")
    applicationGroupDescription = meta['design']['plan'].get('applicationGroup', None)
    application_product_id = meta['design']['plan'].get('application_product_id', None)
    available_choice = meta['design']['plan'].get("available_choice", None)
    run_type_model = models.RunType.objects.get(runType=run_type)

    message = { "E001" : "The Imported Panel is not supported for the selected Instrument Type : {0}",
                "E002" : "The supported Instrument Types for this panel {0}" }
    app_group_name = ""
    decoratedInstType = None
    if applicationGroupDescription:
        app_group = models.ApplicationGroup.objects.get(description=applicationGroupDescription)
        app_group_name = app_group.description
    else:
        app_group = run_type_model.applicationGroups.filter(isActive=True).order_by("id")[0]
        app_group_name = app_group.name
    # "choice": "None" will be in the JSON from 3.6 schema imports
    chip_type = ""
    if run_type == "AMPS_EXOME":
        chip_type = "P1.1.17"
    instrument_type = meta.get("choice", "None")
    if instrument_type == "None":
        if run_type == "AMPS_EXOME":
            instrument_type = "proton"
        else:
            instrument_type = "pgm"

    #HACK
    if instrument_type == 'p1' or instrument_type.lower() == 'proton':
        chip_type = "P1.1.17"
        instrument_type = "proton"
    elif instrument_type in ['520', '521', '530', '540']:
        decoratedInstType = decorate_S5_instruments(instrument_type)
        chip_type = instrument_type
        instrument_type = "s5"
    print("plan_json processing plan_name=%s; run_type=%s; instrument_type=%s" %
          (plan_name, run_type, instrument_type))

    # Access the appl product obj directly using the application_product_id in plan.json
    if application_product_id:
        app = models.ApplProduct.objects.get(id=application_product_id)
        if app.instrumentType != instrument_type:
            if decoratedInstType in available_choice:
                print "Application Product ID does not match the Instrument Type. Please check"
            else:
                if not decoratedInstType:
                    decoratedInstType = instrument_type.upper()
                print message["E001"].format(decoratedInstType)
                print message["E002"].format(available_choice)
            sys.exit(1)
    else:
        try:
            if instrument_type == "proton" and plan_name.endswith("_Hi-Q"):
                app = models.ApplProduct.objects.get(
                    applType__runType=run_type, isActive=True, productName__contains="_Hi-Q", instrumentType=instrument_type)
            else:
                app = models.ApplProduct.objects.filter(
                    applType__runType=run_type, isActive=True, instrumentType=instrument_type, applicationGroup=app_group)
                if app.count() > 0:
                    app = app[0]
                else:
                    app = models.ApplProduct.objects.filter(
                        applType__runType=run_type, isActive=True, isDefault=True, instrumentType=instrument_type)[0]

        except:
            if decoratedInstType in available_choice:
                traceback.print_exc()
            else:
                if not decoratedInstType:
                    decoratedInstType = instrument_type.upper()
                print message["E001"].format(decoratedInstType)
                print message["E002"].format(available_choice)
            sys.exit(1)

    if app.applicationGroup:
        if app.applicationGroup.description == "Pharmacogenomics" and instrument_type == "pgm":
            chip_type = app.defaultChipType


    plugin_details = meta["design"]["plan"].get("selectedPlugins", {});
    alignmentargs_override = None
    if "variantCaller" in plugin_details and "userInput" in plugin_details["variantCaller"]:
        try:
            if "meta" not in plugin_details["variantCaller"]["userInput"]:
                plugin_details["variantCaller"]["userInput"]["meta"] = {}
            plugin_details["variantCaller"]["userInput"]["meta"]["built_in"] = True;
            plugin_details["variantCaller"]["userInput"]["meta"]["compatibility"] = {
                "panel": "/rundb/api/v1/contentupload/"+str(upload_id)+"/"};
            if "configuration" not in plugin_details["variantCaller"]["userInput"]["meta"]:
                plugin_details["variantCaller"]["userInput"]["meta"]["configuration"] = ""
            if plugin_details["variantCaller"]["userInput"]["meta"]["configuration"] == "custom":
                plugin_details["variantCaller"]["userInput"]["meta"]["configuration"] = ""
            if "ts_version" not in plugin_details["variantCaller"]["userInput"]["meta"]:
                plugin_details["variantCaller"]["userInput"]["meta"]["ts_version"] = "5.2"
            if "name" not in plugin_details["variantCaller"]["userInput"]["meta"]:
                plugin_details["variantCaller"]["userInput"]["meta"][
                    "name"] = "Panel-optimized - " + meta["design"]["design_name"]
            if "repository_id" not in plugin_details["variantCaller"]["userInput"]["meta"]:
                plugin_details["variantCaller"]["userInput"]["meta"]["repository_id"] = ""
            if "tooltip" not in plugin_details["variantCaller"]["userInput"]["meta"]:
                plugin_details["variantCaller"]["userInput"]["meta"][
                    "tooltip"] = "Panel-optimized parameters from AmpliSeq.com"
            plugin_details["variantCaller"]["userInput"]["meta"]["user_selections"] = {
                "chip": "pgm", "frequency": "germline", "library": "ampliseq", "panel": "/rundb/api/v1/contentupload/"+str(upload_id)+"/"}
            if instrument_type == "proton":
                plugin_details["variantCaller"]["userInput"]["meta"]["user_selections"]["chip"] = "proton_p1"
            if chip_type in ['520', '521', '530', '540']:
                plugin_details["variantCaller"]["userInput"]["meta"]["user_selections"]["chip"] = chip_type
            if "tmapargs" in plugin_details["variantCaller"]["userInput"]["meta"]:
                alignmentargs_override = plugin_details["variantCaller"]["userInput"]["meta"]["tmapargs"]
        except:
            print "WARNING while generating plan entry"
            traceback.print_exc()

    defaultTemplateKit = app.defaultTemplateKit and app.defaultTemplateKit.name
    if not defaultTemplateKit:
        defaultTemplateKit = app.defaultIonChefPrepKit and app.defaultIonChefPrepKit.name

    #print(plan_name)
    #TS-12518
    defaultFlowOrder = app.defaultFlowOrder
    if not defaultFlowOrder:
        defaultFlowOrder = ""

    plan_stub = {
        "adapter": None,
        "applicationGroupDisplayedName": app_group_name,
        "autoAnalyze": True,
        "autoName": None,
        # Set if isBarcoded
        "barcodeId": app.defaultBarcodeKitName,
        "barcodedSamples": {},
        "bedfile": primary_path,
        "regionfile": secondary_path,
        "chipBarcode": None,
        "chipType": chip_type,
        "controlSequencekitname": "",
        "cycles": None,
        "date": "2012-11-21T04:59:11.000877+00:00",
        "expName": "",
        "flows": app.defaultFlowCount,
        "flowsInOrder": defaultFlowOrder and defaultFlowOrder.flowOrder,
        "forward3primeadapter": "ATCACCGACTGCCCATAGAGAGGCTGAGAC",
        "platform": instrument_type,
        "irworkflow": "",
        "isFavorite": False,
        "isPlanGroup": False,
        "isReusable": True,
        "isReverseRun": False,
        "isSystem": False,
        "isSystemDefault": False,
        "libkit": None,
        "library": meta["reference"],
        "libraryKey": "TCAG",
        # Kit
        "librarykitname":  app.defaultLibraryKit and app.defaultLibraryKit.name,
        "metaData": {},
        "notes": "",
        "origin": "ampliseq.com",
        "pairedEndLibraryAdapterName": "",
        "parentPlan": None,
        "planDisplayedName": plan_name,
        "planExecuted": False,
        "planExecutedDate": None,
        "planName": plan_name,
        "planPGM": None,
        "planStatus": "",
        "preAnalysis": True,
        "reverse3primeadapter": None,
        "reverse_primer": None,
        "reverselibrarykey": None,
        "runMode": "single",
        "runType": meta['design']['plan']['runType'],
        "runname": None,
        "sample": "",
        "sampleDisplayedName": "",
        "samplePrepKitName": app.defaultSamplePrepKit and app.defaultSamplePrepKit.name,
        "selectedPlugins": plugin_details,
        "seqKitBarcode": None,
        # Kit
        "sequencekitname": app.defaultSequencingKit and app.defaultSequencingKit.name,
        "storageHost": None,
        "storage_options": "A",
        # Kit
        "templatingKitName": defaultTemplateKit,
        "usePostBeadfind": True,
        "usePreBeadfind": True,
        "username": "ionuser",
    }
    return plan_stub, alignmentargs_override


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('upload_id', type=int)
    parse.add_argument('path')
    parse.add_argument('upload_file')
    parse.add_argument('meta_file')

    try:
        args = parse.parse_args()
    except IOError as err:
        print("ERROR: Input file error: %s" % err)
        parse.print_help()
        sys.exit(1)

    with open(args.meta_file) as f:
        meta = json.load(f, parse_float=Decimal)

    files = meta.get('pre_process_files')

    target_regions_bed = None
    hotspots_bed = None
    meta['is_ampliseq'] = False

    if len(files) == 1 and files[0].endswith('.bed') and meta.get('hotspot', False) == False:
        target_regions_bed = os.path.basename(files[0])
        meta['is_ampliseq'] = False
        print "Content:        Target regions file in BED format"
        print

    elif len(files) == 1 and files[0].endswith('.bed') and meta.get('hotspot', False) == True:
        hotspots_bed = os.path.basename(files[0])
        meta['is_ampliseq'] = False
        print "Content:        Hotspots file in BED format"
        print

    elif len(files) == 1 and files[0].endswith('.vcf') and meta.get('hotspot', False) == True:
        print "Content:        Hotspots file in VCF format"
        print
        print "Converting hotspot VCF file to BED: %s" % files[0]
        print

        hotspots_bed = os.path.basename(files[0]) + '.bed'
        convert_command = '/usr/local/bin/tvcutils prepare_hotspots'
        convert_command += '  --input-vcf %s' % os.path.join(args.path, os.path.basename(files[0]))
        convert_command += '  --output-bed %s' % os.path.join(args.path, hotspots_bed)
        convert_command += '  --reference /results/referenceLibrary/tmap-f3/%s/%s.fasta' % (
            meta["reference"], meta["reference"])
        convert_command += '  --filter-bypass on'

        p = subprocess.Popen(convert_command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
        print p.communicate()[0]
        if p.returncode != 0:
            sys.exit(p.returncode)

        #process = subprocess.Popen(convert_command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
        #for line in process.communicate()[0].splitlines():
        # api.post('log', upload='/rundb/api/v1/contentupload/%s/' %
        # str(args.upload_id), text=line.strip())
        meta['is_ampliseq'] = False

    elif "plan.json" in files:
        print "Content:        AmpliSeq ZIP"
        print
        meta['is_ampliseq'] = True
        plan_data = json.load(open(os.path.join(args.path, "plan.json")))
        version, design, meta = ampliseq.handle_versioned_plans(plan_data, meta, args.path)

        meta['design'] = design

        try:
            target_regions_bed = design['plan']['designed_bed']
            hotspots_bed = design['plan']['hotspot_bed']
            if not meta.get("reference", None):
                meta['reference'] = design['genome']
            if 'design_name' in plan_data:
                meta['description'] = design['design_name']
            api.update_meta(meta, args)
        except KeyError as err:
            api.patch("contentupload", args.upload_id, status="Error: malformed AmpliSeq archive")
            print "ERROR: Malformed AmpliSeq archive: missing json key "+str(err)
            sys.exit(1)
            #api.post('log', upload='/rundb/api/v1/contentupload/%s/' % str(args.upload_id), text="Malformed AmpliSeq archive: missing json key "+str(err))
            #raise

        if target_regions_bed and target_regions_bed not in files:
            api.patch("contentupload", args.upload_id, status="Error: malformed AmpliSeq archive")
            print "ERROR: Target region file %s not present in AmpliSeq archive" % target_regions_bed
            sys.exit(1)
            #api.post('log', upload='/rundb/api/v1/contentupload/%s/' % str(args.upload_id),
            #         text="Malformed AmpliSeq archive: Target region file %s not present in AmpliSeq archive" % target_regions_bed)
            #raise ValueError("Target region file %s not present in AmpliSeq archive" % target_regions_bed)

        if hotspots_bed and hotspots_bed not in files:
            api.patch("contentupload", args.upload_id, status="Error: malformed AmpliSeq archive")
            print "ERROR: Hotspots file %s not present in AmpliSeq archive" % target_regions_bed
            sys.exit(1)
            #api.post('log', upload='/rundb/api/v1/contentupload/%s/' % str(args.upload_id),
            #         text="Malformed AmpliSeq archive: Hotspots file %s not present in AmpliSeq archive" % target_regions_bed)
            #raise ValueError("Hotspots file %s not present in AmpliSeq archive" % target_regions_bed)

    else:
        api.patch("contentupload", args.upload_id, status="Error: Unrecognized upload type.")
        print
        print "ERROR: Unrecognized upload type. Upload must be either a valid Ampliseq ZIP or contain a single BED or VCF file."
        sys.exit(1)

    ''' === Validate and Register === '''
    primary_path = None
    secondary_path = None

    if is_BED_encrypted(meta):
        if target_regions_bed:
            meta['design']['plan']['designed_bed'] = ''
        if hotspots_bed:
            meta['design']['plan']['hotspot_bed'] = ''
        primary_path = ""
        secondary_path = ""
    else:
        if target_regions_bed:
            primary_path = validate(args.upload_id, args.path, meta, target_regions_bed, 'target regions BED')
        if hotspots_bed:
            secondary_path = validate(args.upload_id, args.path, meta, hotspots_bed, 'hotspots BED')

        meta["hotspot"] = False
        if target_regions_bed and not primary_path:
            register_bed_file(args.upload_id, args.path, meta, target_regions_bed)
        if hotspots_bed:
            meta["hotspot"] = True
            if not secondary_path:
                register_bed_file(args.upload_id, args.path, meta, hotspots_bed)

    if meta['is_ampliseq']:
        try:
            if not (is_BED_encrypted(meta)):
                if target_regions_bed and not primary_path:
                    primary_path = os.path.join(
                        args.path, meta["reference"]+"/unmerged/detail/"+target_regions_bed)
                if hotspots_bed and not secondary_path:
                    secondary_path = os.path.join(
                        args.path, meta["reference"]+"/unmerged/detail/"+hotspots_bed)
            else:
                run_type = meta['design']['plan'].get('runType', None)
                if run_type and (run_type == "AMPS_RNA"):
                    meta['reference'] = None
            plan_prototype, alignmentargs_override = plan_json(
                meta, args.upload_id, primary_path, secondary_path)
            success, response, content = api.post("plannedexperiment", **plan_prototype)

            if not success:
                api.patch("contentupload", args.upload_id, status="Error: unable to create TS Plan")
                err_content = json.loads(content)
                error_message_array = []
                if 'error' in err_content:
                    error_json = json.loads(str(err_content['error'][3:-2]))
                    for k in error_json:
                        for j in range(len(error_json[k])):
                            err_message = str(error_json[k][j])
                            err_message = err_message.replace('&gt;', '>')
                            error_message_array.append(err_message)
                error_messages = ','.join(error_message_array)
                raise Exception(error_messages)
            if alignmentargs_override:
                content_dict = json.loads(content)
                api.patch("plannedexperiment", content_dict[
                          "id"], alignmentargs=alignmentargs_override, thumbnailalignmentargs=alignmentargs_override)
        except Exception as err:
            print("ERROR: Could not create plan from this zip: %s." % err)
            raise

    api.update_meta(meta, args)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        traceback.print_exc()
        sys.exit(1)
