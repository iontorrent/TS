#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
'''
There is a single cached.filelist file in the report directory.  All four file categories
use this file.
'''
import os
import sys
import argparse
from cPickle import Unpickler

from iondb.bin import djangoinit
from iondb.rundb import models
from iondb.rundb.data import dmactions_types as dmtypes

try:
    from dm_utils import get_walk_filelist, _file_selector
except:
    from iondb.rundb.data.dm_utils import get_walk_filelist, _file_selector
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned


def validate_report(reportname,
                    print_files=False,
                    write_file=False):
    '''Validate cached.filelist for the given report'''
    try:
        result = models.Results.objects.get(resultsName=reportname)
    except ObjectDoesNotExist:
        print "No result object was found for '%s'" % reportname
        return None
    except MultipleObjectsReturned:
        print "Multiple objects were found for '%s'" % reportname
        return None

    validate_result(result, print_files=print_files, write_file=write_file)


def validate_result(result,
                    filter_plugins=True,
                    print_files=False,
                    write_file=False):
    '''Validate cached.filelist for the given result'''

    path_to_report_dir = result.get_report_dir()
    path_to_file = os.path.join(path_to_report_dir, 'cached.filelist')
    print "validating %s" % path_to_file

    if result.isThumbnail:
        print("cannot validate: thumbnail")
        return None

#    for this_type in dmtypes.FILESET_TYPES:
#        dmfs = result.get_filestat(this_type)
#        if dmfs.action_state != 'L':
#            print "cannot validate: not local"
#            return None
    dmfs = result.get_filestat(dmtypes.SIG)

    # Get the cached filelist from cached.filelist file
    try:
        with open(path_to_file, 'rb') as fhandle:
            pickle = Unpickler(fhandle)
            cached_filelist = pickle.load()
    except IOError as ioerror:
        print "%s" % ioerror
        return None

    # Get a list of files on the filesystem currently
    dirs = [dmfs.result.get_report_dir(), dmfs.result.experiment.expDir]
    if write_file:
        current_fs_filelist = get_walk_filelist(dirs, list_dir=os.getcwd(), save_list=True)
    else:
        current_fs_filelist = get_walk_filelist(dirs)

    # Ignore plugin_out directories
    if filter_plugins:
        current_fs_filelist = [filename for filename in current_fs_filelist if not '/plugin_out' in filename]

    # Ignore the cached.filelist file
    current_fs_filelist = [filename for filename in current_fs_filelist if not 'cached.filelist' in filename]
    # Ignore the status.txt file
    current_fs_filelist = [filename for filename in current_fs_filelist if not 'status.txt' in filename]
    # Ignore the serialized_*.json file
    current_fs_filelist = [filename for filename in current_fs_filelist if not 'serialized_' in filename]

    # See if there are differences
    # leftovers = list(set(cached_filelist) - set(current_fs_filelist))
    # N.B. This difference here will tell us, "Of the files in the filesystem right now, how many are NOT in the cached.filelist file"
    # Even if the cached.filelist contains more files than are currently on the filesystem.
    # I am thinking this means we do not care if any action_state is not 'L'.  It doesn't matter because we are looking for deficient
    # cached.filelist.
    leftovers = list(set(current_fs_filelist) - set(cached_filelist))
    if print_files:
        for i, item in enumerate(leftovers):
            if not i: print "FILES MISSING FROM CACHED.FILELIST:"
            print item
    else:
        if len(leftovers) > 0:
            print "FILES MISSING FROM CACHED.FILELIST: %d" % len(leftovers)
    print "- %s\n" % ("Not valid" if len(leftovers) > 0 else "Valid")

    return None


def main(max_items=0, print_files=False):
    for i, result in enumerate(models.Results.objects.all().order_by('-timeStamp'), 1):
        sys.stdout.write("%05d: " % i)

        validate_result(result, print_files=print_files)
        if max_items:
            if i == max_items:
                print "Max items: %d reached" % i
                return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
dm_cached_fl_validator
CLI tool to validate existing cached.filelist files
''')
    parser.add_argument('--report',
                        default=None,
                        help='Name of a Report to validate')
    parser.add_argument('--all',
                        dest='all_reports',
                        action='store_true',
                        default=False,
                        help='Evaluate all Reports which are still local')
    parser.add_argument('--max_items',
                        default=0,
                        help='Limit number of items to check')
    parser.add_argument('--list',
                        action='store_true',
                        default=False,
                        help='Print name of missing files')
    parser.add_argument('--output',
                        action='store_true',
                        default=False,
                        help='Write the cached.filelist to local directory.  Only works with --report option.')
    args = parser.parse_args()

    # If no arguments given, print usage and exit
    if len(sys.argv) == 1:
        parser.print_usage()
        sys.exit(0)

    if args.report:
        # evaluate a single report
        sys.exit(validate_report(args.report,
                                 print_files=args.list,
                                 write_file=args.output))
    else:
        # evalute all reports marked Local
        sys.exit(main(max_items=int(args.max_items), print_files=args.list))
