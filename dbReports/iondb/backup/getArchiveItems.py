# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

#!/usr/bin/env python

import os
from os import path
import sys
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from iondb.rundb import models
import commands
from iondb.backup.archiveExp import Experiment

def get_servers():
        fileservers = models.FileServer.objects.all()
        ret = []
        for fs in fileservers:
            if path.exists(fs.filesPrefix):
                ret.append(fs)
        return ret
    
if __name__ == '__main__':
    bk = models.BackupConfig.objects.all().order_by('pk')
    #TODO: populate dictionary with experiments ready to be archived
    # Get all experiments in database sorted by date
    experiments = models.Experiment.objects.all().order_by('date')
    print "Valid: %d" % experiments.count()
    # Filter out all experiments marked Keep
    experiments = experiments.exclude(storage_options='KI').exclude(expName__in = models.Backup.objects.all().values('backupName'))
    print "Valid: %d" % experiments.count()
    # Filter out experiments marked 'U' or 'D'
    experiments = experiments.exclude(user_ack='U').exclude(user_ack='D')
    print "Valid: %d" % experiments.count()
    
    #TODO: make dictionary, one array per file server of archiveExperiment objects
    status = False
    to_archive = {}
    servers =  get_servers()
    for fs in servers:
        print "For server %s" % fs.filesPrefix
        print type(experiments)
        explist = []
        for exp in experiments:
            print "\t%s" % exp.expName
            if fs.filesPrefix in exp.expDir:
                location = models.Rig.objects.get(name=exp.pgmName).location
                E = Experiment(exp,
                               str(exp.expName),
                               str(exp.date),
                               str(exp.star),
                               str(exp.storage_options),
                               str(exp.user_ack),
                               str(exp.expDir),
                               location,
                               exp.pk)
                explist.append(E)
                #TODO: Set this based on whether there are runs to backup
                status = True
        to_archive[fs.filesPrefix] = explist