#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import statvfs
import datetime
import shutil
import traceback

import iondb.bin.djangoinit
from iondb.rundb import models
from iondb.utils import files
from django import shortcuts

arc_directory = [
    'exportedReports',
    'exportedReports_'
]

def getSpace(drive_path):
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
    try:
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
            # Try to create default "exportedReports", then try "exportedReports_".  Previous versions
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
                exportReport(pkR, arc_dir, comment, _logger)
            else:
                raise Exception("Permission denied writing to backup media: %s" % bk.location)

        else:
            raise Exception("backup media not configured.")
    except:
        _logger.exception(traceback.format_exc())
        raise
        return False
    else:
        return True

def exportReport(pkR, dest, comment, _logger):

    logger = _logger

    result = shortcuts.get_object_or_404(models.Results, pk=pkR)
    oldDir = result.get_report_dir()
    reportFolder = os.path.basename(oldDir)
    newReportDir = os.path.join(dest, reportFolder)
    dt = datetime.datetime.now()
    dtForm = '%d_%02d_%02d_%02d_%02d_%02d' % (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    newDir = os.path.join(newReportDir, dtForm)

    if getSize(oldDir) >= getSpace(dest):
        raise Exception("Not enough space to export report '%s' at %s" % (result.resultsName, dest))

    # This is the directory for the result.  Exported reports will create subdirs in it
    if not os.path.isdir(newReportDir):
        old_umask = os.umask(0000)
        os.makedirs(newReportDir)
        os.umask(old_umask)
        logger.debug("Created dir: %s" % newReportDir)

    result.updateMetaData("Exporting", "Exporting from %s to %s." % (oldDir, newDir), 0, comment, logger=logger)

    files.test_sigproc_infinite_regression(oldDir)

    try:
        # We desire to copy all files pointed to by symlinks; do not want just symlinks in the archive location
        #TODO: Change UMASK to get 775 permissions (default gets us 755)
        shutil.copytree(oldDir, newDir, symlinks=False)
        result.updateMetaData("Exported", "Export successful.", 0, comment, logger=logger)
    except Exception as inst:
        raise Exception("shutil module failed to copy. Error type: %s" % inst)
