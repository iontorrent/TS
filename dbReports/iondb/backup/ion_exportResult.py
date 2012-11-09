#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import commands
import statvfs
import datetime
import shutil
import traceback

import iondb.bin.djangoinit
from iondb.rundb import models
from django import shortcuts

def getSpace(drive_path):
    total_space = 0
    s = os.statvfs(drive_path) 
    freebytes = s[statvfs.F_BSIZE] * s[statvfs.F_BAVAIL]
    return freebytes

def getSize(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path, followlinks=True):
        for f in filenames:
            try:
                os.readlink(os.path.join(dirpath, f))
            except:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return total_size

def exportReportShort(pkR, comment, _logger):
    result = shortcuts.get_object_or_404(models.Results, pk=pkR)
    bkL = models.dm_reports.objects.all().order_by('pk').reverse()
    bk = bkL[0]
    retName = result.resultsName
    if result.status != "Completed":
        raise Exception("analysis is not completed. Try again once the analysis has completed")
        
    if bk.location != (None or 'None' or '' or ' ' or '/'):
        exportReport(pkR, os.path.join(bk.location, "exportedReports"), comment, _logger)
    else:
        raise Exception("backup media not properly set up.")
        
def exportReport(pkR, dest, comment, _logger):
    
    logger = _logger
    
    result = shortcuts.get_object_or_404(models.Results, pk=pkR)
    oldDir = result.get_report_dir()
    reportFolder = os.path.basename(oldDir)

    # This is the directory for the result.  Exported reports will create subdirs in it
    newReportDir = os.path.join(dest,reportFolder)
    if not os.path.isdir(newReportDir):
        os.makedirs(newReportDir, mode=0777)
        logger.debug("Created dir: %s" % newReportDir)
    
    dt = datetime.datetime.now()
    dtForm = '%d_%02d_%02d_%02d_%02d_%02d'%(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    newDir = os.path.join(dest,reportFolder,dtForm)
    
    if getSize(oldDir) >= getSpace(newReportDir):
        raise Exception("Not enough space to export report '%s' at %s" % (result.resultsName,newDir))
    
    result.updateMetaData("Exporting", "Exporting from %s to %s."%(oldDir,newDir), 0, comment, logger = logger)
    
    try:
        # We desire to copy all files pointed to by symlinks; do not want just symlinks in the archive location
        shutil.copytree(oldDir, newDir, symlinks=False)
        result.updateMetaData("Exported", "Export successful.", 0, comment, logger = logger)
    except Exception as inst:
        raise Exception("shutil module failed to copy. Error type: %s"%inst)
