#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from iondb.rundb import models
import commands

# Notes:
#
#   * - initial idea was to use explog.txt as a way to identify raw dataset dirs.
#       however, that is not going to work since the ionCrawler is using that to find
#       new runs.
#
#   * - Next idea is to just flag all the directories which are not registered
#       as Rigs in the database.  That will identify all non-Run data.
#
def valid_fileservers():
    # get all fileservers from dbase
    fileservers = models.FileServer.objects.all()
    validfileservers = []
    for fileserver in fileservers:
        if (os.path.isdir(fileserver.filesPrefix)):
            validfileservers.append(fileserver)
    return validfileservers

def disk_usage(path):
    import commands
    total = commands.getstatusoutput("du -ms %s 2>/dev/null" % path)
    if total[0] != 0:
        print "Error calling system command"
        print total
        return 0
    else:
        data = total[1]
    
    return int(data.split()[0])

def find_explog_files(path):
    #For given path, search for explog.txt files - this is our test for raw dataset
    #Optimization: we know that raw dataset directories are found within PGM subdirectory
    #we can limit find search to maxdepth of 3.
    print "Looking for explog.txt files..."
    files = []
    total = commands.getstatusoutput("find %s -maxdepth 3 -type f -name explog.txt" % path)
    if total[0] != 0:
        print "Error calling system command"
        print total
        return files
    else:
        #print "DEBUG: %s" % total[1]
        for entry in total[1].split():
            files.append(entry)

    return files

def parse_runname(filename):
    runname = ''
    total = commands.getstatusoutput('grep "^Experiment Name:" %s' % filename)
    if total[0] != 0:
        print "Error calling system command"
        print total
        return runname
    else:
        #print "DEBUG: %s" % total[1]
        runname = total[1].split(":")[1]
    return runname

if __name__ == '__main__':
    # Find orphan Experiments
    # Orphan Experiments are raw datasets that are not in the database
    # For each file server, for each PGM name, for each directory
    #      Test for corresponding entry in database
    # get all Experiment records from database.  we will hit database only once.
    experiments = models.Experiment.objects.all()
    print "Number of records: %d" % experiments.count()
    
    # get all file server records from database.
    validfileservers = valid_fileservers()
    
    # get all Rigs records from database
    rigs = models.Rig.objects.all()
    rignames = [rig.name for rig in rigs] 
    for fs in validfileservers:
        print "Checking %s" % fs.filesPrefix
        #NOTE: This assumes there are only valid Runs in the PGM directories.
        # get list of files in top level directory
        filelist = os.listdir(fs.filesPrefix)
        #print filelist
        # remove Rigs from list of files
        paredlist = [file for file in filelist if file not in rignames]
        #print paredlist
        # Get disc usage for remaining files
        totaldu = 0
        for file in paredlist:
            thisfile = disk_usage(os.path.join(fs.filesPrefix,file))
            print "%d\t%s" % (thisfile,file)
            totaldu += thisfile
        print "----------------"
        print "%d Mbytes\tTOTAL" % totaldu
                              