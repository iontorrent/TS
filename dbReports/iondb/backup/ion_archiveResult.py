#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import statvfs
import datetime
import shutil
import traceback

import iondb.bin.djangoinit
from iondb.rundb import models
from django import shortcuts
from iondb.utils import makePDF
from ion.utils import makeCSA
from iondb.rundb.tasks import removeDirContents

ARCHIVING = "Archiving"
ARCHIVED = "Archived"
STATUS = [ARCHIVED, ARCHIVING]
arc_directory = [
    'archivedReports',
    'archivedReports_'
]


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
    s = os.statvfs(drive_path)
    freebytes = s[statvfs.F_BSIZE] * s[statvfs.F_BAVAIL]
    return freebytes


def archiveReportShort(pkR, comment, _logger):
    result = shortcuts.get_object_or_404(models.Results, pk=pkR)
    bk = models.dm_reports.get()
    if result.status != "Completed":
        raise Exception("analysis is not completed. Try again once the analysis has completed")

    if bk.location != (None or 'None' or '' or ' ' or '/'):
        
        # Test for valid drive location and reset if invalid.
        if not os.path.isdir(bk.location):
            invalid_location = bk.location
            bk.location = 'None'
            bk.save()
            raise Exception("backup media is not available: %s" % invalid_location)
            
        # Test for writeable directory
        # Try to create default "archivedReports", then try "archivedReports_".  Previous versions
        # of code created first directory with root ownership.
        for directory in arc_directory:
            arc_dir = os.path.join(bk.location, directory)
            
            if not os.path.isdir(arc_dir):
                # Make the directory
                old_mask = os.umask(0000)
                os.makedirs(arc_dir)
                os.umask(old_mask)
                
            if os.access(arc_dir, os.W_OK|os.X_OK):
                break
            else:
                arc_dir = None
                    
        if arc_dir:
            archiveReport(pkR, arc_dir, comment, _logger)
        else:
            raise Exception("Permission denied writing to backup media: %s" % bk.location)
            
    else:
        raise Exception("backup media not configured.")


def archiveReport(pkR, dest, comment, _logger):

    logger = _logger

    result = shortcuts.get_object_or_404(models.Results, pk=pkR)
    oldDir = result.get_report_dir()
    reportFolder = os.path.basename(oldDir)
    rawDataDir = result.experiment.expDir

    # This is the directory for the result.  Exported reports will create subdirs in it
    newReportDir = os.path.join(dest, reportFolder)
    if not os.path.isdir(newReportDir):
        old_umask = os.umask(0000)
        os.makedirs(newReportDir)
        os.umask(old_umask)
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
        csaFullPath = makeCSA.makeCSA(oldDir, rawDataDir)
        logger.debug("Made CSA")
    except:
        # TODO: Do we continue?  OR do we abort?
        logger.exception(traceback.format_exc())
        raise

    dt = datetime.datetime.now()
    dtForm = '%d_%02d_%02d_%02d_%02d_%02d' % (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    newDir = os.path.join(dest, reportFolder, dtForm)

    if getSize(oldDir) >= getSpace(newReportDir):
        raise Exception("Not enough space to archive report '%s' at %s" % (result.resultsName, newReportDir))

    result.updateMetaData(ARCHIVING, "Archiving report '%s' to %s" % (result.resultsName, newDir), 0, comment, logger=logger)
    # copy the entire directory from oldDir to the archive destination.
    try:
        # We desire to copy all files pointed to by symlinks; do not want just symlinks in the archive location
        shutil.copytree(oldDir, newDir, symlinks=False)
        result.updateMetaData(ARCHIVING, "Directory copy successful", 0, comment, logger=logger)
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
        # Remove the contents of the results directory
        # N.B. Execute as celery task for the sole reason of getting it executed with root permissions
        dir_del = removeDirContents.delay(oldDir)
        dir_del.get()
    except:
        logger.exception(traceback.format_exc())
        raise Exception("Archive created but there was an error in cleaning up source data.  See /var/log/ion/reportsLog.log.")

    for pdffile in ['backupPDF.pdf','report.pdf','plugins.pdf']:
        try:
            # Copy any pdf files
            shutil.copyfile(os.path.join(newDir, pdffile), os.path.join(oldDir, pdffile))
            os.chmod(os.path.join(oldDir, pdffile), 0777)
        except:
            logger.exception(traceback.format_exc())
    
    try:
        # Copy the customer support archive file from the archive location back to results directory
        shutil.copyfile(os.path.join(newDir, os.path.basename(csaFullPath)), csaFullPath)
        os.chmod(csaFullPath, 0777)

        result.updateMetaData(ARCHIVED, "Finished archiving.", total_size1, comment, logger=logger)
    except:
        logger.exception(traceback.format_exc())

    #else:
    #    logger.debug('copy unsuccessful, deletion will not occur...cleaning up failed archive...')
    #    shutil.rmtree(newDir)
    #    raise Exception("directory sizes do not match. Archive creation unsuccessful.")
