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
experiments = models.Experiment.objects.all().order_by('date')
print "Number of Experiments %d" % len(experiments)
# Filter out all experiments marked Keep
experiments = experiments.exclude(storage_options='KI').exclude(expName__in = models.Backup.objects.all().values('backupName'))
print "...Not Keep, Not already backedup %d" % len(experiments)
# Filter out experiments marked 'U' or 'D'
#experiments = experiments.exclude(user_ack='U').exclude(user_ack='D')
#print "...Ready for processing %d" % len(experiments)

for e in experiments:
    print "%60s\t\t%s" % (e.expName,e.user_ack)
    
