#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#
# Prints list of Experiments not yet archived or deleted with storage usage and date.

import os
from os import path
import sys

import iondb.bin.djangoinit
from iondb.rundb import models
import commands

def get_size(start, progress):
    total_size = 0
    i = 0
    #faster algorithm?
    for dirpath, dirnames, filenames in os.walk(start):
        for f in filenames:
            try:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
                if i >= 10000 and progress:
                    print "\n    Progress update: ~%s Gb" %(total_size/(1024*1024*1024)) + "\n    Current directory: %s" %dirpath
                    i = 0
                i+=1
            except:
                pass
    if progress:
        print "\n    Finished %s" %start + ", total size: %s bytes" %total_size + " = ~%s Gb\n" %(total_size/(1024*1024*1024))
    return total_size
    
if __name__ == '__main__':
    # Get all experiments in database sorted by date
    experiments = models.Experiment.objects.all().order_by('date')
    
    # Filter out all experiments already archived or deleted
    experiments = experiments.exclude(expName__in = models.Backup.objects.all().values('backupName'))
    
    # List all experiments marked Keep
    print "Experiments marked KEEP"
    total = 0
    for e in experiments.exclude(storage_options='A').exclude(storage_options='D'):
        size = get_size(e.expDir, False)
        total += size
        print "%7.2f Gb - %s - %-80s" % (float(size)/1024/1024/1024,str(e.date).split()[0],e.expDir)
    print "==========\n%7.2f Gb TOTAL KEEP\n" % (float(total)/1024/1024/1024)
    
    # List all experiments marked Archive
    print "Experiments marked ARCHIVE"
    total = 0
    for e in experiments.exclude(storage_options='KI').exclude(storage_options='D'):
        size = get_size(e.expDir, False)
        total += size
        print "%7.2f Gb - %s - %-80s" % (float(size)/1024/1024/1024,str(e.date).split()[0],e.expDir)
    print "==========\n%7.2f Gb TOTAL ARCHIVE\n" % (float(total)/1024/1024/1024)
        
    # List all experiments marked Delete
    print "Experiments marked DELETE"
    total = 0
    for e in experiments.exclude(storage_options='KI').exclude(storage_options='A'):
        size = get_size(e.expDir, False)
        total += size
        print "%7.2f Gb - %s - %-80s" % (float(size)/1024/1024/1024,str(e.date).split()[0],e.expDir)
    print "==========\n%7.2f Gb TOTAL DELETE\n" % (float(total)/1024/1024/1024)
