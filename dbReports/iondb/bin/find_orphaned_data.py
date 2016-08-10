#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
'''
CLI tool to find and report on files that should no longer be found in the file system
according to the status of the DMFileStat objects in the database
'''
import os
import sys
import json
# import time
import argparse
import traceback

from iondb.bin import djangoinit
from iondb.rundb import models
from iondb.rundb.data import dmactions_types as dmtypes
from iondb.rundb.data.dm_utils import get_walk_filelist, _file_selector
from iondb.rundb.data.dmactions import _get_keeper_list

# Global progress bar indicator variable
progress = ['|',
            '/',
            '-',
            '\\',
            '|',
            '/',
            '-',
            '\\']


def search_for_files(dmfilestats, reset, report):
    '''Look for files for the given DM category still in the filesystem.
    This is the long-lived function so we enable ctrl-c interrupt to
    exit the loop and still write the log file.
    '''
    try:
        print ("Ctrl-C to exit")
        tracking = []
        num_dmfs = len(dmfilestats)
        for i, dmfs in enumerate(dmfilestats):
            sys.stdout.write("\r%05d/%05d %s" % (i + 1, num_dmfs, progress[i % 7]))
            sys.stdout.flush()
            to_process = []
            to_keep = []
            # For each dmfilestat object, check if files still exist in filesystem
            # 1. Do not rely on cache.filelist
            dirs = [dmfs.result.get_report_dir(), dmfs.result.experiment.expDir]
            for start_dir in [dir for dir in dirs if os.path.isdir(dir)]:
                tmp_process, tmp_keep = _file_selector(start_dir,
                                                       dmfs.dmfileset.include,
                                                       dmfs.dmfileset.exclude,
                                                       _get_keeper_list(dmfs, 'delete'),
                                                       dmfs.result.isThumbnail,
                                                       False,
                                                       cached=get_walk_filelist(dirs))
                to_process += tmp_process
                to_keep += tmp_keep

            orphans = list(set(to_process) - set(to_keep))
            logs = models.EventLog.objects.for_model(dmfs.result)
            # We only want to track those datasets with lots of files displaced.
            if len(orphans) > 10:
                # if dmfs.action_state in ['DD', 'AD']:   # Is it marked Deleted?
                if dmfs.action_state in ['DD']:   # Is it marked Deleted?
                    print "\nReport: %s" % (dmfs.result.resultsName)
                    print "Report Directory: %s" % dmfs.result.get_report_dir()
                    print "Status: %s" % 'Deleted' if dmfs.action_state == 'DD' else 'Archived'
                    print "Category: %s" % dmfs.dmfileset.type
                    print "Raw Data Directory: %s" % dmfs.result.experiment.expDir
                    print "No. files: %d" % len(orphans)
                    print "Action Date: %s" % logs[len(logs) - 1].created
                    print "Action Log: %s" % logs[len(logs) - 1].text
                    tracking.append({'report': dmfs.result.resultsName,
                                     'report_dir': dmfs.result.get_report_dir(),
                                     'state': 'Deleted' if dmfs.action_state == 'DD' else 'Archived',
                                     'rawdatadir': dmfs.result.experiment.expDir,
                                     'num_files': len(orphans),
                                     'reset': reset,
                                     'action_state': dmfs.action_state,
                                     'action_date': '%s' % logs[len(logs) - 1].created,
                                     'action_text': logs[len(logs) - 1].text})
                    if reset:
                        try:
                            print "Deleting the cached.filelist file"
                            cachefilename = os.path.join(dmfs.result.get_report_dir(), "cached.filelist")
                            if os.path.exists(cachefilename):
                                # os.unlink(cachefilename)
                                os.rename(cachefilename, cachefilename + ".hide")
                        except OSError:
                            print traceback.format_exc()
                        dmfs.action_state = "L" if dmfs.action_state == 'DD' else "SA"
                        dmfs.save()
                        print "Reset to %s: %s" % (dmfs.action_state, dmfs.result.resultsName)

                    if not report:
                        for entry in orphans:
                            print entry
            elif len(orphans) > 0:
                if not report:
                    print "\rLeft-overs Report: %s" % dmfs.result.resultsName
                    for entry in orphans:
                        print entry

        sys.stdout.write("\n ")
    except (KeyboardInterrupt):
        pass
    except:
        print traceback.format_exc()
    finally:
        return tracking


def dump_to_file(filename, stuff):
    '''Write log file'''
    rjson = json.dumps(stuff, indent=4)
    fileh = open(filename, 'w')
    print >> fileh, rjson
    fileh.close()


def main(my_dmtypes, report_filename, reset=False, partition=None, report=False):
    '''Main function'''

    print ("\nSearching these categories:")
    for dmtype in my_dmtypes:
        print ("\t%s" % dmtype)

    dmfilestats = models.DMFileStat.objects.filter(dmfileset__type__in=my_dmtypes).order_by('-created')

    print ("\nSearching for datasets with action_state 'DD'")
    dmfilestats = dmfilestats.filter(action_state__in=['DD'])

    # Filter experiment directory by raw data partition instead of using all
    if partition:
        print ("\nFiltering for datasets with raw data on partition: %s" % partition)
        dmfilestats = dmfilestats.filter(result__experiment__expDir__startswith=partition)

    # Terminal output
    print ("\nTotal DMFileStat Objects (deleted data): %d" % len(dmfilestats))
    print ("Looking for rogue files now...")

    # Search for files per dmfilestat
    process_results = search_for_files(dmfilestats, reset, report)
    dump_to_file(report_filename, process_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
find_orphaned_data
CLI tool to find and report on files that should no longer be found in the file system
according to the status of the DMFileStat objects in the database
''')
    parser.add_argument('--SIG',
                        action='store_true',
                        default=False,
                        help='Only Signal Processing Input files')
    parser.add_argument('--BASE',
                        action='store_true',
                        default=False,
                        help='Only Basecalling Input files')
    parser.add_argument('--OUT',
                        action='store_true',
                        default=False,
                        help='Only Output files')
    parser.add_argument('--INTR',
                        action='store_true',
                        default=False,
                        help='Only Intermediate files')
    parser.add_argument('--all',
                        dest='all_dmtypes',
                        action='store_true',
                        default=False,
                        help='All file categories (SIG, BASE, OUT, INTR)')
    parser.add_argument('-o',
                        dest='output',
                        default=os.path.join(os.getcwd(), 'orphaned_data.report'),
                        help='specify output filename (Default: orphaned_data.report)')
    parser.add_argument('--reset',
                        dest='reset_flag',
                        action='store_true',
                        default=False,
                        help='Objects that still have files are reset to Local')
    parser.add_argument('--partition',
                        default=None,
                        help='Limit Results to this partition for experiment directory.')
    parser.add_argument('--report',
                        action='store_true',
                        default=False,
                        help='Report on problem datasets, minimal debug printout')

    args = parser.parse_args()

    # If no arguments given, print usage and exit
    if len(sys.argv) == 1:
        parser.print_usage()
        sys.exit(0)

    # What File Categories do we examine?
    sel_dmtypes = []
    if args.SIG:
        sel_dmtypes.append(dmtypes.SIG)
    if args.BASE:
        sel_dmtypes.append(dmtypes.BASE)
    if args.OUT:
        sel_dmtypes.append(dmtypes.OUT)
    if args.INTR:
        sel_dmtypes.append(dmtypes.INTR)

    if args.all_dmtypes:
        sel_dmtypes = dmtypes.FILESET_TYPES

    if args.partition:
        if not os.path.exists(args.partition):
            print "Error: %s is not a valid path." % args.partition
            sys.exit(1)

    sys.exit(main(sel_dmtypes, args.output, reset=args.reset_flag,
             partition=args.partition, report=args.report))
