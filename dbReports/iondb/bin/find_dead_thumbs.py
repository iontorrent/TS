#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import os
import re
import sys
import argparse
import shutil
import traceback
import iondb.bin.djangoinit
from iondb.rundb import models
from iondb.utils.files import getdiskusage


def main(delete=False):

    # DBase objects of sigproc datasets that are deleted.
    dir_list = models.DMFileStat.objects.filter(action_state__in=['DD'],dmfileset__type='Signal Processing Input').values_list('result_id__experiment_id__expDir',flat=True)
    print ("Number of sigproc datasets, deleted: %d" % (len(dir_list)))
    total_disk_space_mb = 0
    exclude_list = []
    for target_dir in dir_list:
        try:
            if not (target_dir in exclude_list) and os.path.isdir(target_dir):
            #if ("Corvette" in target_dir) and not (target_dir in exclude_list) and os.path.isdir(target_dir):
                exclude_list.append(target_dir)
                sys.stdout.write("%s " % target_dir)
                sys.stdout.write("exists ")
                size_mb = getdiskusage(target_dir)
                sys.stdout.write("Size: %d MB " % (size_mb))

                total_disk_space_mb += size_mb
                if delete:
                    if os.path.isdir(os.path.join(target_dir, 'thumbnail')):
                        # Recursively remove thumbnail directory
                        shutil.rmtree(os.path.join(target_dir, 'thumbnail'))
                        sys.stdout.write("thumbnail deleted ")

                    # Here determine if any other files can be deleted
                    # If onboard_results exists, leave it alone.
                    if os.path.isdir(os.path.join(target_dir, 'onboard_results')):
                        # Do nothing
                        pass
                    else:
                        # Delete everything else
                        if not os.path.isdir(os.path.join(target_dir, 'sigproc_results')):
                            shutil.rmtree(target_dir)
                            sys.stdout.write("rawdata dir deleted\n")
                            continue

                    # Here remove empty rawdata directory
                    if len(os.listdir(target_dir)) == 0:
                        shutil.rmtree(target_dir)
                        sys.stdout.write("rawdata dir deleted ")

                sys.stdout.write("\n")
        except:
            traceback.print_exc()

        else:
            pass

    print ("Total disk usage (MB): %d" % (total_disk_space_mb))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finds thumbnail datafiles on filesystem of datasets that have been deleted")
    parser.add_argument('--delete',
                        help='Deletes the thumbnail files',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    main(delete=args.delete)