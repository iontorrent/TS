#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
#
# Identify experiments which are marked as Deleted, according to the old Backup
# object, but in fact still have their files on the filesystem.
#
import os
import re
import argparse
import shutil
import iondb.bin.djangoinit
from iondb.rundb import models

def main(delete=False):
    cnt = 0
    total_MB = 0
    re_pgm_data = re.compile(r'.*acq_\d+\.dat')
    re_proton_data = re.compile(r'.*X\d+_Y\d+.*')

    # Get the list of Backup objects
    # Get the subset of deleted objects
    backups = models.Backup.objects.filter(isBackedUp = False)

    # For every object, check for directory (and data) still existing and print
    for item in backups:
        # Related experiment object contains filepath to data directory.
        test_dir = item.experiment.expDir
        if os.path.exists(test_dir):
            for s in os.listdir(test_dir):
                if re_pgm_data.match(s) or re_proton_data.match(s):
                    cnt += 1
                    print "(%d) Exists: %s" % (cnt,test_dir)
                    print " Deleted on: %s" % item.backupDate
                    print " Planned: %s" % item.experiment.storage_options
                    print " Marked: %s" % item.experiment.user_ack
                    print " Size: %d MB" % item.experiment.diskusage
                    total_MB += item.experiment.diskusage
                    if delete and re_pgm_data.match(s):
                        print "We will now delete this directory: %s" % test_dir
                        try:
                            shutil.rmtree(test_dir)
                        except Exception as e:
                            print e
                    break

    if delete:
        print "Recovered disk space: %0.3f GB" % float(total_MB/1024.0)
    else:
        print "Recoverable disk space: %0.3f GB" % float(total_MB/1024.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finds files on filesystem that database considers deleted")
    parser.add_argument('--delete',
                        help='Deletes the directory, only if its PGM.',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    main(delete=args.delete)