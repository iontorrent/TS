#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import argparse
import json
import os
import os.path
import sys
import traceback
import zipfile
import call_api as api
from pprint import pprint
from iondb.rundb.plan import ampliseq

def get_common_prefix(files):
    """For a list of files, a common path prefix and a list file names with
    the prefix removed.

    Returns a tuple (prefix, relative_files):
        prefix: Longest common path to all files in the input. If input is a
                single file, contains full file directory.  Empty string is
                returned of there's no common prefix.
        relative_files: String containing the relative paths of files, skipping
                        the common prefix.
    """
    # Handle empty input
    if not files or not any(files):
        return '', []
    # find the common prefix in the directory names.
    directories = [os.path.dirname(f) for f in files]
    prefix = os.path.commonprefix(directories)
    start = len(prefix)
    if all(f[start] == "/" for f in files):
        start += 1
    relative_files = [f[start:] for f in files]
    return prefix, relative_files


def valid_files(files):
    black_list = [lambda f: "__MACOSX" in f]
    absolute_paths = [os.path.isabs(d) for d in files]
    if any(absolute_paths) and not all(absolute_paths):
        raise ValueError("Archive contains a mix of absolute and relative paths.")
    return [f for f in files if not any(reject(f) for reject in black_list)]


def make_relative_directories(root, files):
    directories = [os.path.dirname(f) for f in files]
    for directory in directories:
        path = os.path.join(root, directory)
        if not os.path.exists(path):
            os.makedirs(path)


def unzip_archive(root, data):
    zip_file = zipfile.ZipFile(data, 'r')
    namelist = zip_file.namelist()
    namelist = valid_files(namelist)
    prefix, files = get_common_prefix(namelist)
    make_relative_directories(root, files)
    out_names = [(n, f) for n, f in zip(namelist, files) if
                                                    os.path.basename(f) != '']
    for key, out_name in out_names:
        if os.path.basename(out_name) != "":
            full_path = os.path.join(root, out_name)
            contents = zip_file.open(key, 'r')
            try:
                output_file = open(full_path, 'wb')
                output_file.write(contents.read())
                output_file.close()
            except IOError as err:
                print("For zip's '%s', could not open '%s'" % (key, full_path))
    return [f for n, f in out_names]


def pre_process():
    parse = argparse.ArgumentParser()
    parse.add_argument('upload_id', type=int)
    parse.add_argument('path')
    parse.add_argument('upload_file')
    parse.add_argument('meta_file', type=argparse.FileType('r+'))
    
    try:
        args = parse.parse_args()
    except IOError as err:
        print("Input file error: %s" % err)
        parse.print_help()
        sys.exit(1)

    meta = json.load(args.meta_file)
    meta.update({
        "is_ampliseq": None,
        "primary_bed": None,
        "hotspot_bed": None
    })

    is_zip = zipfile.is_zipfile(args.upload_file)
    if is_zip:
        files = unzip_archive(args.path, args.upload_file)
    else:
        files = [args.upload_file]
    
    if len(files) == 1 and files[0].endswith('.bed'):
        meta['is_ampliseq'] = False
        meta['primary_bed'] = files[0]
    elif "plan.json" in files:
        print("Found ampliseq")
        meta['is_ampliseq'] = True
        plan_data = json.load(open(os.path.join(args.path, "plan.json")))
        version, design = ampliseq.handle_versioned_plans(plan_data)
        meta['design'] = design
        plan = design['plan']
        try:
            meta['primary_bed'] = plan['designed_bed']
            meta['secondary_bed'] = plan['hotspot_bed']
            if 'reference' not in meta:
                meta['reference'] = plan['genome'].lower()
        except KeyError as err:
            api.patch("contentupload", args.upload_id, status="Error: malformed AmpliSeq archive")
            raise
        print(meta)
    else:
        raise ValueError("Upload must be either valid Ampliseq export or contain a single BED file.")

    args.meta_file.truncate(0)
    args.meta_file.seek(0)
    json.dump(meta, args.meta_file)
    api.patch("contentupload", args.upload_id, meta=meta)


if __name__ == '__main__':
    try:
        pre_process()
    except Exception as err:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
