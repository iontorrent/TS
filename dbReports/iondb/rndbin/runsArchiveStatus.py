#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from iondb.rundb import models
    
bk = models.BackupConfig.objects.all().order_by('pk')
print "Number of BackupConfig objs %d" % len(bk)
# populate dictionary with experiments ready to be archived

# Get all experiments in database sorted by date
allexperiments = models.Experiment.objects.all().order_by('date')
print "Total Number of Experiments %d" % len(allexperiments)

# Filter out all experiments marked Keep

keepers = allexperiments.exclude(storage_options='D').exclude(expName__in = models.Backup.objects.all().values('backupName'))
deletable = allexperiments.filter(storage_options='D').exclude(expName__in = models.Backup.objects.all().values('backupName'))

print "Currently Marked Keep or Archive %d" % len(keepers)
# Filter out experiments marked 'U' or 'D'
#experiments = experiments.exclude(user_ack='U').exclude(user_ack='D')
#print "...Ready for processing %d" % len(experiments)

for e in keepers:
    project = e.log['project']
    print "%60s\t\t%s" % (e.expName,project)
print ""
print "Currently Marked Delete %d" % len(deletable)
print ""
for e in deletable:
    project = e.log['project']
    print "%60s\t\t%s" % (e.expName,project)
    
