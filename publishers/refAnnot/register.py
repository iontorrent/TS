#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import json
import requests

api_url = "http://localhost/rundb/api/v1/content/register/%s/"
headers = {"Content-type": "application/json"}

def register(files, meta):
    print '....register files: %s' % files
    session = requests.Session()
    session.headers.update(headers)

    url = api_url % upload_id

    for my_file in files:
        payload = {
            'file': my_file,
            'path': '/' + meta['reference'] + '/' + os.path.basename(my_file),
            'meta': meta
        }
        r = session.post(url, data = json.dumps(payload), headers=headers)
        r.raise_for_status()
        print '....registered %s' % payload['path']


def main():
    try:
        with open(meta_file) as f:
            meta = json.load(f)
            print '....meta: %s' % json.dumps(meta,indent=1)

        files = meta.get('validate_files') or [upload_file]
        if len(files) < 1:
            raise Exception('No files to process, exiting.')
        else:
            register(files, meta)
        
    except Exception as e:
        print 'Error: %s' % e
        sys.exit(1)


if __name__ == '__main__':

    args = sys.argv
        
    try:
        upload_id = args[1]
        path = args[2]
        upload_file = args[3]
        meta_file = args[4]
    except Exception as err:
        print("ERROR: Input file error: %s" % err)
        parse.print_help()
        sys.exit(1)

    print 'Started register'
        
    main()
