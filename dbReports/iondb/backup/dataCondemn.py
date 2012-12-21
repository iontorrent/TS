#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#
# Mark Raw Datasets with Delete designation
# parameterized time threshold
#
# Select the oldest Experiments that have not been Backed-Up
# Change their designation to "Delete"
#
import iondb.bin.djangoinit
from iondb.rundb import models
import datetime
import sys

if __name__ == '__main__':
    defaultdays = 365
    import argparse
    parser = argparse.ArgumentParser(description="Select the oldest Experiments that have not been Backed-Up and Change their storage designation to \"Delete\"")
    parser.add_argument('--simulate', default=False, action='store_true', help="Don't make any changes to the database")
    parser.add_argument('--days', type=int, default=defaultdays, help="How many days old data needs to be. (default=%d)" % defaultdays)

    # If no arguments, print help and exit.  Otherwise, the script is active and makes changes to the database
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse command line
    args = parser.parse_args()

    # Control whether we make any changes to database entries
    make_changes = not(args.simulate)
    if make_changes:
        print "Script will edit database entries"
    else:
        print "No changes will be made to database entries"

    # Calculate threshold date
    threshold_days = args.days
    time_threshold = datetime.datetime.now() - datetime.timedelta(threshold_days)

    print "Reading database Experiment Table..."
    # Get all Experiment objects in database
    expList = models.Experiment.objects.all()
    print ("Total Experiments: %d" % len(expList))

    # Exclude experiments already backed-up
    expList = expList.exclude(expName__in=models.Backup.objects.all().values('backupName'))
    print ("Active Experiments: %d (Experiments not yet archived)" % len(expList))

    # Select experiments with datetime earlier than the threshold
    # by excluding experiments with datetime newer than cutoff date
    expList = expList.exclude(date__gte=time_threshold)
    print "Experiments older than %d days: %d" % (threshold_days, len(expList))

    # Sort by date
    expList = expList.order_by('date')

    for exp in expList:
        sys.stdout.write("%s %s %s " % (exp.date, exp.expName, exp.storage_options))
        if make_changes:
            exp.storage_options = 'D'
            exp.save()
            sys.stdout.write("set to %s\n" % 'D')
        else:
            sys.stdout.write("\n")
