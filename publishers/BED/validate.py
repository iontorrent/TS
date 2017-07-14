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
from iondb.rundb.plan import ampliseq

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

    return os.path.join(base_path, meta["reference"]+"/unmerged/detail/" + bed_name)


def is_BED_encrypted(meta):
    is_BED_encrypted = None
    if 'key' in meta.get('pre_process_files'):
        is_BED_encrypted = "True"

    return is_BED_encrypted

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
    elif bed_type == 'SSE BED':
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
    sse_bed = None
    meta['is_ampliseq'] = False

    if len(files) == 1 and files[0].endswith('.bed') and meta.get('hotspot', False) == False:
        target_regions_bed = os.path.basename(files[0])
        print "Content:        Target regions file in BED format"
        print

    elif len(files) == 1 and files[0].endswith('.bed') and meta.get('hotspot', False) == True:
        hotspots_bed = os.path.basename(files[0])
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
    elif "plan.json" in files:
        # Call the validation script from ampliseq. validate reference, target bed and hotspot
        print "Content:        AmpliSeq ZIP\n"

        meta['is_ampliseq'] = True
        isRefInstallInProgress, meta = ampliseq.validate_ampliSeq_bundle(meta, args)
        api.update_meta(meta, args)
        """
        If reference mentioned in the plan.json (get the info from "genome_reference") is not installed in the TS,
            - wait for the ref to be installed
            - the subtask finish_me will be called at the end of the reference install
            - process to restart validation of the upload
        """
        if isRefInstallInProgress:
            return

        target_regions_bed = meta['design']['plan'].get('designed_bed','')
        hotspots_bed = meta['design']['plan'].get('hotspot_bed','')
        sse_bed = meta['design']['plan'].get('sse_bed','')
    else:
        api.patch("contentupload", args.upload_id, status="Error: Unrecognized upload type.")
        print
        print "ERROR: Unrecognized upload type. Upload must be either a valid Ampliseq ZIP or contain a single BED or VCF file."
        sys.exit(1)

    ''' === Validate and Register === '''
    target_regions_bed_path = ""
    hotspots_bed_path = ""
    sse_bed_path = ""

    isBED_Encrypted = is_BED_encrypted(meta)
    if target_regions_bed:
        if isBED_Encrypted:
            meta['design']['plan']['designed_bed'] = ''
        else:
            target_regions_bed_path = validate(args.upload_id, args.path, meta, target_regions_bed, 'target regions BED')
            if not target_regions_bed_path:
                meta["hotspot"] = False
                target_regions_bed_path = register_bed_file(args.upload_id, args.path, meta, target_regions_bed)
    
    if hotspots_bed:
        if isBED_Encrypted:
            meta['design']['plan']['hotspot_bed'] = ''
        else:
            hotspots_bed_path = validate(args.upload_id, args.path, meta, hotspots_bed, 'hotspots BED')
            if not hotspots_bed_path:
                meta["hotspot"] = True
                hotspots_bed_path = register_bed_file(args.upload_id, args.path, meta, hotspots_bed)

    if sse_bed:
        if isBED_Encrypted:
            meta['design']['plan']['sse_bed'] = ''
        else:
            sse_bed_path = validate(args.upload_id, args.path, meta, sse_bed, 'SSE BED')
            if not sse_bed_path:
                meta["hotspot"] = False
                meta["sse"] = True
                meta["sse_target_region_file"] = target_regions_bed_path
                sse_bed_path = register_bed_file(args.upload_id, args.path, meta, sse_bed)

    if meta['is_ampliseq']:
        if isBED_Encrypted:
            run_type = meta['design']['plan'].get('runType', '')
            if run_type == "AMPS_RNA":
                meta['reference'] = None
                api.update_meta(meta, args)

        # parse,process and convert the ampliseq plan.json to TS supported plan and post
        success, isUploadFailed, errMsg = ampliseq.convert_AS_to_TS_plan_and_post(
                                                            meta,
                                                            args,
                                                            target_regions_bed_path,
                                                            hotspots_bed_path,
                                                            sse_bed_path
                                                        )
        if isUploadFailed:
            print("ERROR: Could not create plan from this zip: %s." % errMsg)
            raise
    else:
        api.update_meta(meta, args)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        traceback.print_exc()
        sys.exit(1)
