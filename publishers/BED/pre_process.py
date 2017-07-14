#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import argparse
from decimal import Decimal
import json
import os.path
import subprocess

from iondb.bin.djangoinit import *
from iondb.utils import files as file_utils
import zipfile
import call_api as api

def check_stage_pre(upload_id, meta):
    reentry = meta.get('reentry_stage', None)
    if reentry and reentry != 'pre_process':
        # Success, we're past pre_processing already and do not need to run further
        sys.exit(0)

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
