#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

# Script will search Reports database for all Reports with a Status that is 
# NOT Success (or whatever a valid complete report status is)
#
# This identifies all Reports which could be deleted from database and deleted
# from the filesystem

import os
import sys
import commands
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from iondb.rundb import models

if __name__ == '__main__':

# Status field can be
#	Checksum Error
#	PGM Operation Error
#	TERMINATED
#	ERROR
#	Completed
#	Started

    results = models.Results.objects.all()
    count = 0
    totalds = 0
    for result in results:
        if not 'Completed' in result.status and not 'Started' in result.status:
            print result.status
            path = result.get_report_dir()
            if not os.path.isdir (path):
                print "Dir %s does not exist" % path
            else: 
                try:
                    diskusage = commands.getstatusoutput("du -sk %s 2>/dev/null" % path)
                    sum = int(diskusage[1].split()[0])
                    print ("%dkb %s") % (sum,path)
                    count += 1
                    totalds += sum
                except:
                    print ("error getting disk space")
    print "Total broken reports: %d" % count
    print "%d kb" % totalds
    
