#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved 
'''Shows Data Management information for a Report'''
import sys
import argparse

from iondb.bin import djangoinit
from iondb.rundb import models
from iondb.rundb.data import dmactions_types as dmtypes


def main(report_name):
    result = models.Results.objects.get(resultsName=report_name)
    
    print ("")
    print("Report Dir: %s" % result.get_report_dir())
    print("      Date: %s" % result.timeStamp)
    print ("")
    print("SigProcDir: %s " % result.experiment.expDir)
    print("      Date: %s" % result.experiment.date)
    
    print ("")
    for dmtype in dmtypes.FILESET_TYPES:
        dmfs = result.get_filestat(dmtype)
        sys.stdout.write("%24s - %2s" % (dmtype, dmfs.action_state))
        if dmfs.getpreserved():
            sys.stdout.write(" - Keep")
        if dmfs.archivepath is not None:
            sys.stdout.write(" - Archived: %s" % dmfs.archivepath)
        sys.stdout.write("\n")
        dmfs = None

    # DM Logs
    print("\nData Management Log Entries")
    logs = models.EventLog.objects.for_model(result)
    for log in logs:
        print("Date: %s" % log.created)
        print("Entry: %s" % log.text)
        print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
Shows Data Management information for a Report
''')
    parser.add_argument('--report',
                        help = 'Name of Report (not the report directory, not the experiment name)')
    
    args = parser.parse_args()
    if args.report == None or len(sys.argv) == 1:
        parser.print_usage()
        sys.exit(0)
        
    sys.exit(main(args.report))