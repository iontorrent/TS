#!/usr/bin/env python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
'''
    Finds datasets where the diskspace usage has not been set in the database.
    Allows setting the diskspace usage for those datasets.
'''
import sys
import argparse
import traceback
from iondb.bin import djangoinit
from iondb.rundb import models
from iondb.rundb.data.dmactions_types import FILESET_TYPES
from iondb.rundb.data.dmfilestat_utils import update_diskspace
DEBUG = True


def main(cl_args):
    '''Main function'''
    for dmtype in FILESET_TYPES:
        sys.stdout.write("%s\n" % dmtype)
        stats = models.DMFileStat.objects.filter(dmfileset__type=dmtype) \
            .filter(action_state__in=['L', 'S', 'N', 'A']) \
            .filter(diskspace=None) \
            .order_by('created')
        num_datasets = 0
        for index, item in enumerate([stat for stat in stats if stat.diskspace is None]):
            if index == 0:
                sys.stdout.write("Missing disk space:\n")
            num_datasets = index
            if cl_args.repair:
                sys.stdout.write("\t%s..." % (item.result.resultsName))
                try:
                    if DEBUG:
                        sys.stdout.write("\nWe would run update_diskspace() here, if not in debug mode")
                    else:
                        sys.stdout.write("%d\n" % update_diskspace(item))
                except:
                    print traceback.format_exc()
            else:
                sys.stdout.write("\t%s\n" % item.result.resultsName)
        if num_datasets == 0:
            sys.stdout.write("...No datasets found\n")
    sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''

Identify and repair datasets without disk usage value recorded.

''')
    parser.add_argument('--list',
                        dest='list',
                        action='store_true',
                        default=False,
                        help='List all datasets missing disk usage')
    parser.add_argument('--repair',
                        dest='repair',
                        action='store_true',
                        default=False,
                        help='Calculate disk usage for datasets missing disk usage value')

    myargs = parser.parse_args()

    # If no arguments given, print usage and exit
    if any([myargs.list, myargs.repair]):
        main(myargs)
    else:
        parser.print_usage()
        sys.exit(0)
