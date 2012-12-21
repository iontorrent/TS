#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
"""
discUsage
=========

The discUsage module will determine what percentage of the disc space is being used by raw data, analyses, and other results data.

Note that right now, it looks in /results/PGM_test for rig data. This may not be accurate for all machines.
"""

import os
from os import path
import sys
import commands
import datetime
import statvfs
import shutil
from socket import gethostname

import iondb.bin.djangoinit
from iondb.rundb import models
from django import shortcuts


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
                    print "\n    Progress update: ~%s Mb" % (total_size / (1024 * 1024))
                    i = 0
                i += 1
            except:
                pass
    print "Finished, total size: %s bytes" % total_size
    return total_size

if __name__ == '__main__':
    rigSpace = 0
    freeSpace = 0
    totalSpace = 0

    allServers = models.FileServer.objects.all()
    validServers = []
    for server in allServers:
        if (os.path.isdir(server.filesPrefix)):
            validServers.append(server)
            f = os.statvfs(server.filesPrefix)
            freeSpace += f[statvfs.F_BAVAIL] * f[statvfs.F_BSIZE]
            totalSpace += f[statvfs.F_BLOCKS] * f[statvfs.F_BSIZE]
    validServerPGMCombos = []
    allPGM = models.Rig.objects.all()
    for PGM in allPGM:
        for server in validServers:
            testpath = os.path.join(server.filesPrefix, PGM.name)
            if os.path.isdir(testpath):
                space = get_size(testpath, True)
                rigSpace += space
                print "\nPGM %s " % PGM.name + "on server %s " % server.filesPrefix + "contains %s bytes" % space + "= %sMb" % (space / (1024 * 1024))

    analysisSpace = 0
    #(this is 'other space' without analysis or rig space removed)
    storageList = models.ReportStorage.objects.all()
    for storage in storageList:
        analysisSpace += get_size(storage.dirPath, True)

    #otherSpace = get_size('/results', True) - rigSpace - analysisSpace
    otherSpace = totalSpace - analysisSpace - rigSpace - freeSpace

    print "\nanalysis space: %s " % analysisSpace + " = %sMb" % (analysisSpace / (1024 * 1024))
    print "\ntotal rig space: %s " % rigSpace + "= %sMb" % (rigSpace / (1024 * 1024))
    print "\nother space: %s " % otherSpace + " = %sMb" % (otherSpace / (1024 * 1024))
    print "\ntotal space used: %s " % (rigSpace + analysisSpace + otherSpace) + " = %sMb" % ((rigSpace + analysisSpace + otherSpace) / (1024 * 1024))
    print "\nfree space: %s" % freeSpace + " = %sMb" % (freeSpace / (1024 * 1024))
    print "\ntotal space on disk: %s" % (totalSpace) + " = %sMb" % ((totalSpace) / (1024 * 1024))
