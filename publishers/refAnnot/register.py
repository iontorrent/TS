#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import json
import requests
from iondb.rundb import publisher_types

api_url = "http://localhost/rundb/api/v1/content/register/%s/"
headers = {"Content-type": "application/json"}
supported_filetypes = ['gff', 'gff3', 'gtf', 'gtf2', 'gtf3', 'fasta']


def register(files, meta):
    print '....register files: %s' % files
    session = requests.Session()
    session.headers.update(headers)

    url = api_url % upload_id
    for my_file in files:
        """
        annot_type will be used to distinguish auxiliary reference type
        from annotation, which is the default type for this publisher.
        """
        type_field = publisher_types.ANNOTATION
        if meta.get('annot_type', '') == publisher_types.AUXILIARYREFERENCE:
            type_field = publisher_types.AUXILIARYREFERENCE

        """
        derive file_type from URL. If later release supports compressed
        format with different file ext. then file_type should be part
        of the meta. file_type should be stored in upper cases.
        """
        if not meta.get('file_type', None):
            file_type = meta.get('url', '').split('.').pop()
            if not file_type or file_type.lower() not in supported_filetypes:
                file_type = ''
            meta['file_type'] = file_type.upper()

        payload = {
            'file': my_file,
            'path': '/' + meta['reference'] + '/' + os.path.basename(my_file),
            'meta': meta,
            'type': type_field,
            'extra': meta['reference'],
            'application_tags': meta['application_tags'],
            'description': meta['description']
        }
        r = session.post(url, data=json.dumps(payload), headers=headers)
        r.raise_for_status()
        print '....registered %s' % payload['path']


def main():
    try:
        with open(meta_file) as f:
            meta = json.load(f)
            print '....meta: %s' % json.dumps(meta, indent=1)

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
