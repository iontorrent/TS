#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
'''Tool to show a disk's unusable space due to either Runs that are marked Keep, or non-Run data.'''

import os
from django.core.exceptions import ObjectDoesNotExist

import iondb.bin.djangoinit
from iondb.rundb import models


def disk_space(path):
    '''
    Returns used and available disk capacity in megabytes
    '''
    import commands
    total = commands.getstatusoutput("df -mP %s 2>/dev/null|tail -1" % path)
    data = total[1]
    capacity = int(data.split()[1])
    used = int(data.split()[2])
    available = int(data.split()[3])
    #kbytes = used + available

    # DEBUG PRINTOUT
    #print "Disk Capacity %s" % path
    #print "---------"
    #print str(kbytes) + " KBytes"
    #print str(int(kbytes) / 1024 / 1024) + " GB"
    return used, available, capacity


def disk_usage(path):
    import commands
    total = commands.getstatusoutput("du -ms %s 2>/dev/null" % path)
    if total[0] != 0:
        return 0
    else:
        data = total[1]

    return int(data.split()[0])


def valid_fileservers():
    # get all fileservers from dbase
    fileservers = models.FileServer.objects.all()
    validfileservers = []
    for fileserver in fileservers:
        if (os.path.isdir(fileserver.filesPrefix)):
            validfileservers.append(fileserver)
    return validfileservers


def valid_experiments(path):
    '''Returns list of experiments that have not been deleted or archived'''
    validExperiments = []
    # Get all experiment objects
    allExperiments = models.Experiment.objects.all()
    #print "Number of experiments %d" % len(allExperiments)
    # Filter out experiments that are not on this server
    thisServer = []
    for experiment in allExperiments:
        if path in experiment.expDir:
            thisServer.append(experiment)
    # Filter out experiments that are deleted or archived
    for experiment in thisServer:
        try:
            bk = models.Backup.objects.get(backupName=experiment.expName)
        except ObjectDoesNotExist:
            validExperiments.append(experiment)
    return validExperiments


def diskSpaceForRuns(experiments):
    keepSpace = 0
    otherSpace = 0
    for experiment in experiments:
        usedSpace = disk_usage(experiment.expDir)
        if experiment.storage_options == "KI":
            #TODO: get disk space usage for this Run
            keepSpace += usedSpace
        else:
            #TODO: get disk space usage for this Run
            otherSpace += usedSpace

    return keepSpace, otherSpace


def print_report():
    validFS = valid_fileservers()
    print "Number of servers: %d" % len(validFS)
    for FS in validFS:
        print FS.filesPrefix
        # total disk capacity
        # total data space occupied on server
        usedDiskSpace, available, diskCapacity = disk_space(FS.filesPrefix)
        # usable capacity is less than total capacity due to reserved superuser space (5%)
        usableCapacity = available + usedDiskSpace
        # list of Runs on this server
        validExperiments = valid_experiments(FS.filesPrefix)
        print "Number of valid experiments %d" % len(validExperiments)

        # wicked-heavy system resource call right here...use at own risk
        #print "du value for disk usage: %d" % disk_usage(FS.filesPrefix)

        # disk space occupied by Runs marked Keep and all other runs
        keepRuns, otherRuns = diskSpaceForRuns(validExperiments)

        # total space occupied by Runs
        runsDiskSpace = keepRuns + otherRuns

        # total non-Run data
        nonRunData = usedDiskSpace - runsDiskSpace

        # unavailable disk space - sum of non-Run data and Keep Runs space
        # or, total used space less non-Keep Runs - space that will never be recovered by archiving or deleting
        unavailable = usedDiskSpace - otherRuns

        # Print the key information: disk space lost to us
        print "Total Disk Space (Free + Used):            %12d MBytes %5.1f%%" % (usableCapacity, 100.0 * float(usableCapacity) / float(usableCapacity))
        print "Total Free Disk Space:                     %12d MBytes %5.1f%%" % (available, 100.0 * float(available) / float(usableCapacity))
        print "Total Used Disk Space:                     %12d MBytes %5.1f%%" % (usedDiskSpace, 100.0 * float(usedDiskSpace) / float(usableCapacity))
        print "Check Free + Used = %d" % (available + usedDiskSpace)
        print ""
        print "Total Runs Disk Space:                     %12d MBytes %5.1f%%" % (runsDiskSpace, 100.0 * float(runsDiskSpace) / float(usableCapacity))
        print "Total Volatile Runs Disk Space:            %12d MBytes %5.1f%%" % (otherRuns, 100.0 * float(otherRuns) / float(usableCapacity))
        print "Total Keep Runs Disk Space:                %12d MBytes %5.1f%%" % (keepRuns, 100.0 * float(keepRuns) / float(usableCapacity))
        print "Total Non-Run Space:                       %12d MBytes %5.1f%%" % (nonRunData, 100.0 * float(nonRunData) / float(usableCapacity))
        print "Total Disk Space Unavailable for Run Data: %12d MBytes %5.1f%%" % (unavailable, 100.0 * float(unavailable) / float(usableCapacity))
        print ""
    return

if __name__ == "__main__":
    # Print the way this is calculated
    print "Disk space unavailable for Run data is calculated by"
    print "Taking total used space and subtracting space used by volatile"
    print "Runs: Those Runs marked for Archive and Delete"
    print "This is space that will never be recovered via the ionArchive"
    print "daemon's archiving and deleting process"
    print ""
    print_report()
