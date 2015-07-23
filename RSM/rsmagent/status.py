#!/usr/bin/python
# Copyright (C) 2013, 2014 Ion Torrent Systems, Inc. All Rights Reserved
"""Get status of various processes in TS"""

from __future__ import print_function
import sys
import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

from iondb.rundb.configure.views import process_set


def main():
    """Get status of various processes in TS"""
    processes = process_set()

    for process in processes:
        service_name, service_status = process
        if service_status:
            status_string = 'Running'
        else:
            status_string = 'Down'

        print(service_name, '|', status_string, file=sys.stderr)

if __name__ == "__main__":
    main()
