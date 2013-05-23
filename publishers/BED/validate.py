#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import subprocess
import sys
import json
import traceback
import argparse
import os.path
from pprint import pprint
import os


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('upload_id', type=int)
    parse.add_argument('path')
    parse.add_argument('upload_file')
    parse.add_argument('meta_file')
    
    try:
        args = parse.parse_args()
    except IOError as err:
        print("Input file error: %s" % err)
        parse.print_help()
        sys.exit(1)


    def validate(bed_file):
        print("Validating %s" % bed_file)
        cmd = [
            "./bed_validation.pl", 
            str(args.upload_id),
            args.path,
            os.path.join(args.path, bed_file),
            args.meta_file
        ]
        p = subprocess.Popen(cmd)
        p.communicate()
        if p.returncode != 0:
            sys.exit(p.returncode)
        
    meta = json.load(open(args.meta_file))
    upload_uid = "/rundb/api/v1/contentupload/%d/" % args.upload_id

    if meta['is_ampliseq']:
        print("I can't believe it's not Ampliseq!")
        validate(meta['primary_bed'])
        if meta['secondary_bed'] is not None:
            validate(meta['secondary_bed'])
    else:
        validate(args.upload_file)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
