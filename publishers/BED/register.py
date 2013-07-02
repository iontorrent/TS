#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Author: Daniel Cuevas

import call_api as api
import sys
import json
import os.path
from iondb.bin.djangoinit import *
from iondb.rundb import models

# This is the API UID for this publisher
pub_uid = "/rundb/api/v1/publisher/BED/"

# Usage function
def usage():
    print """
submitBeds.py <ID> <DIRECTORY> <UPLOAD_BED> <META_FILE>
Required args:
    <ID>              : Process ID number
    <DIRECTORY>       : Given directory for Publisher
    <UPLOAD_BED>      : BED file to validate
    <META_FILE>       : Metadata JSON file
    """
    sys.exit(1)

# Check input is correct
if len(sys.argv) != 5:
    usage()

id = str(sys.argv[1])
directory = str(sys.argv[2])
bedFile = str(sys.argv[3])
metaFile = str(sys.argv[4])

upload_uid = "/rundb/api/v1/contentupload/%s/" % id

# Open up metaFile to get ref name for pathname purposes
try:
    fh = open(metaFile, "r")
except IOError:
    print >> sys.stderr, "Couldn't open " + metaFile
    sys.exit(1)
line = fh.read().rstrip()
meta = json.loads(line)
ref = meta["reference"]

def register(file, meta):
    full_path = os.path.join(directory, file)
    reg = "/%s" % file
    api.post("content", publisher=pub_uid, meta=meta, file=full_path, path=reg, contentupload=upload_uid)


def register_bed_file(bedFName, meta):
    # File paths to call into publisher
    pbed = ref+"/unmerged/plain/"+bedFName
    dbed = ref+"/unmerged/detail/"+bedFName
    mpbed = ref+"/merged/plain/"+bedFName
    mdbed = ref+"/merged/detail/"+bedFName
    # Register files to Publisher
    register(pbed, meta)
    register(dbed, meta)
    register(mpbed, meta)
    register(mdbed, meta)

def plan_json(meta):
    primary_path = os.path.join(directory, ref+"/unmerged/detail/"+meta['primary_bed'])
    if meta['secondary_bed'] is not None:
        secondary_path = os.path.join(directory, ref+"/unmerged/detail/"+meta['secondary_bed'])
    else:
        secondary_path = None
    run_type = meta['design']['plan'].get('runType', None)
    app = models.ApplProduct.objects.get(applType__runType=run_type)
    plan_stub = {
       "adapter": None,
       "autoAnalyze": True,
       "autoName": None,
       # Set if isBarcoded 
       "barcodeId": app.defaultBarcodeKitName,
       "barcodedSamples": {},
       "bedfile": primary_path,
       "regionfile": secondary_path,
       "chipBarcode": None,
       "chipType": "",
       "controlSequencekitname": "",
       "cycles": None,
       "date": "2012-11-21T04:59:11.000877+00:00",
       "expName": "",
       "flows": app.defaultFlowCount,
       "flowsInOrder": None,
       "forward3primeadapter": "ATCACCGACTGCCCATAGAGAGGCTGAGAC",
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
       "pairedEndLibraryAdapterName": "",
       "parentPlan": None,
       "planDisplayedName": meta["design"]["design_name"],
       "planExecuted": False,
       "planExecutedDate": None,
       "planName": meta["design"]["design_name"],
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
       "selectedPlugins": meta["design"]["plan"].get("selectedPlugins", {}),
       "seqKitBarcode": None,
       # Kit
       "sequencekitname": app.defaultSequencingKit and app.defaultSequencingKit.name,
       "storageHost": None,
       "storage_options": "A",
       # Kit
       "templatingKitName":  app.defaultTemplateKit and app.defaultTemplateKit.name,
       "usePostBeadfind": True,
       "usePreBeadfind": True,
       "username": "ionadmin",
    }
    return plan_stub

if meta['is_ampliseq']:
    print("I can't believe it's not Ampliseq!")
    print("Primary: %s" % meta['primary_bed'])
    print("Secondary: %s" % meta['secondary_bed'])
    meta["hotspot"] = False
    register_bed_file(meta['primary_bed'], json.dumps(meta))
    if meta['secondary_bed'] is not None:
        meta["hotspot"] = True
        register_bed_file(meta['secondary_bed'], json.dumps(meta))
    try:
      plan_prototype = plan_json(meta)
      api.post("plannedexperiment", **plan_prototype)
    except Exception as err:
      print("Could not create plan from this zip: %s" % err)
    sys.exit()
elif bedFile.endswith('.vcf'):
    # Get bed file name without directory path
    bed_file_name = bedFile.split("/").pop()
    register_bed_file(bed_file_name+'.bed', line)
else:
    # Get bed file name without directory path
    bed_file_name = bedFile.split("/").pop()
    register_bed_file(bed_file_name, line)
