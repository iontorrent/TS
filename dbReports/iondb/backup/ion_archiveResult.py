#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
from os import path
import sys
import commands
import statvfs
import datetime
import shutil
import traceback

import iondb.bin.djangoinit
from iondb.rundb import models
from django import shortcuts
from iondb.backup import makePDF
from ion.utils import makeCSA


def removeDirContents(folder_path):
    for file_object in os.listdir(folder_path):
        file_object_path = os.path.join(folder_path, file_object)
        if os.path.isfile(file_object_path):
            os.unlink(file_object_path)
        elif os.path.islink(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)
        
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

def getSpace(drive_path):
    total_space = 0
    s = os.statvfs(drive_path) 
    freebytes = s[statvfs.F_BSIZE] * s[statvfs.F_BAVAIL]
    return freebytes

def archiveReportShort(pkR, comment, _logger):
    result = shortcuts.get_object_or_404(models.Results, pk=pkR)
    bkL = models.dm_reports.objects.all().order_by('pk').reverse()
    bk = bkL[0]
    retName = result.resultsName
    if result.status != "Completed":
        raise Exception("analysis is not completed. Try again once the analysis has completed")
        
    if bk.location != (None or 'None' or '' or ' ' or '/'):
        archiveReport(pkR, os.path.join(bk.location,"archivedReports"), comment, _logger)
    else:
        raise Exception("backup media not properly set up.")
        
def archiveReport(pkR, dest, comment, _logger):
    
    logger = _logger

    result = shortcuts.get_object_or_404(models.Results, pk=pkR)
    oldDir = result.get_report_dir()    
    reportFolder = os.path.basename(oldDir)
    rawDataDir = result.experiment.expDir
    
    # This is the directory for the result.  Exported reports will create subdirs in it
    newReportDir = os.path.join(dest,reportFolder)
    if not os.path.isdir(newReportDir):
        os.makedirs(newReportDir, mode=0777)
        logger.debug("Created dir: %s" % newReportDir)
    
    # make the Report PDF file now. backupPDF.pdf
    try:
        logger.debug("Making PDF of the report")
        makePDF.makePDF(pkR)
        logger.debug("Made PDF of the report")
    except:
        # TODO: Do we continue?  OR do we abort?
        logger.exception(traceback.format_exc())
        raise
    
    # make a customer support archive file
    try:
        logger.debug("Making CSA")
        csaFullPath = makeCSA.makeCSA(oldDir,rawDataDir)
        logger.debug("Made CSA")
    except:
        # TODO: Do we continue?  OR do we abort?
        logger.exception(traceback.format_exc())
        raise
    
    dt = datetime.datetime.now()
    dtForm = '%d_%02d_%02d_%02d_%02d_%02d'%(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    newDir = os.path.join(dest,reportFolder,dtForm)
    
    if getSize(oldDir) >= getSpace(newReportDir):
        raise Exception("Not enough space to archive report '%s' at %s" % (result.resultsName,newReportDir))
    
    result.updateMetaData("Archiving", "Archiving report '%s' to %s" % (result.resultsName,newDir), 0, comment, logger = logger)
    # copy the entire directory from oldDir to the archive destination.
    try:
        # We desire to copy all files pointed to by symlinks; do not want just symlinks in the archive location
        shutil.copytree(oldDir, newDir, symlinks=False)
        result.updateMetaData("Archiving", "Directory copy successful", 0, comment, logger = logger)
    except:
        logger.exception(traceback.format_exc())
        raise
    
    ##check the sizes of the old directory and the destination.
    total_size1 = getSize(oldDir)
    #total_size2 = getSize(newDir)
    #logger.debug('size old dir: %d size new dir: %d'%(total_size1, total_size2))
    
    ##if the sizes are identical, that means that all of the files were copied. If so, delete the old directory.
    #logger.debug('sizecheck complete, preparing to delete if sizes match')
    #if total_size1 == total_size2:
    try:
        #TODO:Enable a debug mode where this action can be skipped
        # Remove the contents of the results directory
        removeDirContents (oldDir)
        
        # Copy the backupPDF.pdf file
        shutil.copyfile(os.path.join(newDir,'backupPDF.pdf'),os.path.join(oldDir,'backupPDF.pdf'))
        os.chmod(os.path.join(oldDir,'backupPDF.pdf'),0777)
        
        # Copy the customer support archive file from the archive location back to results directory
        shutil.copyfile(os.path.join(newDir,os.path.basename(csaFullPath)),csaFullPath)
        os.chmod(csaFullPath,0777)
        
        result.updateMetaData("Archived", "Finished archiving.", total_size1, comment, logger = logger)
    except:
        logger.exception(traceback.format_exc())
        raise Exception ("Archive created but there was an error in cleaning up source data.  See /var/log/ion/reportsLog.log.")
        
    #else:
    #    logger.debug('copy unsuccessful, deletion will not occur...cleaning up failed archive...')
    #    shutil.rmtree(newDir)
    #    raise Exception("directory sizes do not match. Archive creation unsuccessful.")

