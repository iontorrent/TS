#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import argparse
from decimal import Decimal
import json
import os
import os.path
import subprocess
import sys

from iondb.bin.djangoinit import *
from iondb.rundb import models
from iondb.rundb.configure.genomes import new_reference_genome
from iondb.rundb.publishers import run_pub_scripts
from iondb.utils import files as file_utils
from iondb.rundb.plan import ampliseq
import zipfile
import call_api as api


def check_stage_pre(upload_id, meta):
    reentry = meta.get('reentry_stage', None)
    if reentry and reentry != 'pre_process':
        # Success, we're past pre_processing already and do not need to run further
        sys.exit(0)


def set_checkpoint(meta, args):
    # Set a sentinel in the meta data to determine whether or not
    # we've already run before and done this work.
    meta['reentry_stage'] = 'validate'
    api.update_meta(meta, args)


def check_reference(meta, args):
    """Check and install the needed reference genome"""
    print("Checking reference")
    plan_data = json.load(open(os.path.join(args.path, "plan.json")))
    version, design, meta = ampliseq.handle_versioned_plans(plan_data, meta, arg_path=args.path)
    print("Got versioned stuff")
    # If we have a genome reference, check to see if it's installed
    reference = design.get('genome_reference', None)
    print(reference)
    if not reference:
        return False
    try:
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
        print("About t set check point")
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
        return True
    print("exiting in shame")
    return False


def deal_with_upload(meta, args):
    print("Dealing with the upload file")
    is_zip = zipfile.is_zipfile(args.upload_file)
    if is_zip:
        files = file_utils.unzip_archive(args.path, args.upload_file)
        print "Compressed:     Yes (zip)"
    elif args.upload_file.endswith('.gz'):
        print "Compressed:     Yes (gzip)"
        files = [os.path.basename(args.upload_file[:-3])]
        cmd = 'gzip -dc %s > %s ' % (args.upload_file, os.path.join(args.path, files[0]))
        p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
        print p.communicate()[0]
        if p.returncode != 0:
            sys.exit(p.returncode)

        subprocess.call(cmd, shell=True)
    else:
        print "Compressed:     No"
        files = [args.upload_file]
    meta['pre_process_files'] = files
    api.update_meta(meta, args)
    return files


def main(args):
    print("Pre Processing {0}".format(args.upload_file))
    with open(args.meta_file) as f:
        meta = json.load(f, parse_float=Decimal)

    check_stage_pre(args.upload_id, meta)
    files = deal_with_upload(meta, args)
    if "plan.json" in files:
        check_reference(meta, args)


if __name__ == '__main__':
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

    main(args)
