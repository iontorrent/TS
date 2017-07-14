#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
 - This is used to validate the reference and hotspot bed files when importing AmpliSeq Panels.
 - If reference is not installed on the TS machine, its starts downloading the reference and install the panels.
"""
from iondb.bin.djangoinit import *
from iondb.rundb import models
from iondb.rundb.configure.genomes import new_reference_genome
from iondb.rundb.publishers import call_api, run_pub_scripts

api = call_api()

def set_checkpoint(meta, args):
    # Set a sentinel in the meta data to determine whether or not
    # we've already run before and done this work.
    meta['reentry_stage'] = 'validate'
    api.update_meta(meta, args)

def validate_reference(meta, args, reference):
    """Check and install the needed reference genome"""
    print("Checking reference")
    # If we have a genome reference, check to see if it's installed
    if not reference:
        return False
    try:
        print reference
        url = reference.get('uri')
        ref_hash = reference.get('files_md5sum', {}).get('fasta')
        short_name = reference.get('short_name')
        name = reference.get('name')
        notes = reference.get('notes', "AmpliSeq Import")
        print("Got various fields")

    except KeyError as err:
        # If the key does not exist, that's fine, but it can't exist and be corrupt
        print("Corrupt genome_reference entry: {0}".format(err))
        sys.exit(1)

    # The identity_hash matching the files_md5sum.fasta hash determines whether
    # or not the genome is installed
    print("Checking reference " + ref_hash)
    if not models.ReferenceGenome.objects.filter(identity_hash=ref_hash).exists():
        reference_args = {
            'identity_hash': ref_hash,
            'name': name,
            'notes': notes,
            'short_name': short_name,
            'source': url,
            'index_version': "tmap-f3"
        }

        pub = models.Publisher.objects.get(name='BED')
        upload = models.ContentUpload.objects.get(pk=args.upload_id)
        # This is a celery subtask that will run the publisher scripts on this upload again
        finish_me = run_pub_scripts.si(pub, upload)
        print("About to set check point")
        set_checkpoint(meta, args)
        print("check point set")
        # With a status starting with "Waiting" the framework will stop
        # after pre_processing, before validate.
        upload.status = "Waiting on reference"
        upload.save()

        # the subtask finish_me will be called at the end of the reference install
        # process to restart validation of the upload
        new_reference_genome(reference_args, url=url, callback_task=finish_me)
        print("Started reference download")
        return upload.status
    print("The requested Reference from ampliSeq Bundle is already installed in the System. So Exiting....")
    return False

def validate_bed_files(meta, args):
    files = meta["pre_process_files"]
    target_regions_bed = meta['design']['plan'].get('designed_bed','')
    hotspots_bed = meta['design']['plan'].get('hotspot_bed','')
    sse_bed = meta['design']['plan'].get('sse_bed','')

    if target_regions_bed and target_regions_bed not in files:
        api.patch("contentupload", args.upload_id, status="Error: malformed AmpliSeq archive")
        print "ERROR: Target region file %s not present in AmpliSeq archive" % target_regions_bed
        sys.exit(1)

    if hotspots_bed and hotspots_bed not in files:
        api.patch("contentupload", args.upload_id, status="Error: malformed AmpliSeq archive")
        print "ERROR: Hotspots file %s not present in AmpliSeq archive" % hotspots_bed
        sys.exit(1)

    if sse_bed and sse_bed not in files:
        api.patch("contentupload", args.upload_id, status="Error: malformed AmpliSeq archive")
        print "ERROR: SSE file %s not present in AmpliSeq archive" % sse_bed
        sys.exit(1)

    if sse_bed and not target_regions_bed:
        api.patch("contentupload", args.upload_id, status="Error: malformed AmpliSeq archive")
        print "ERROR: Missing associated Target region file with SSE file %s in AmpliSeq archive" % sse_bed
        sys.exit(1)