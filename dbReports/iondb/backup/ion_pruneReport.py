#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
import fnmatch
import traceback
from django import shortcuts

import iondb.bin.djangoinit
from iondb.rundb import models


def pruneReport(pkR, comment, _logger):

    logger = _logger

    try:
        # There should only ever be one object
        bk = models.dm_reports.get()
    except:
        logger.info("problem getting dm_reports object")
        raise

    # Determine if this pruning request is from autoAction.  If so, we only log to the Report's log
    # if there were files removed.  Otherwise, the Report's log will fill with a daily entry
    enable_logging = True
    if "auto-action" in comment:
        enable_logging = False
        
    # Get the selected prune group stored in PruneLevel.  This seems to be a string value of the id number of the prune group selected.
    # We should just use the name of the prune group instead.  Make sure we store the selected prune group name in this field instead.
    # bk.pruneLevel
    try:
        pruneGroup = models.dm_prune_group.objects.get(name=bk.pruneLevel)
        #TODO: handle empty string bk.pruneLevel
    except:
        logger.exception(traceback.format_exc())
        raise

    #get the full path to report directory
    res = shortcuts.get_object_or_404(models.Results, pk=pkR)
    reportDir = res.get_report_dir()

    # Get the file selection patterns (rules) from the pruneGroup object.  The pruneGroup contains a list of pk of the rules to use.
    pruneRules = []
    for element in pruneGroup.ruleNums.split(','):
        if len(element) > 0:
            num = int(element)
            obj = models.dm_prune_field.objects.get(pk=num)
            pruneRules.append(str(obj.rule))

    if enable_logging:
        res.updateMetaData("Pruning", "Pruning using prune group %s: %s" % (bk.pruneLevel, pruneRules), 0, comment, logger=logger)

    # Get the list of files which match the patterns
    # There are two stages to files selection.  First, find all the files to include, then find all the files to exclude
    toDel = []
    # For every file in the directory and recursively in subdirectories
    # Note that the default behavior of os.walk is to NOT follow symlinks that are directories.  We like this behavior because we do
    # not want to follow symlinks to other result directories and delete files there.  Case of an re-analysis report that links to the
    # original sigproc results directory.
    for dirpath, dirnames, filenames in os.walk(reportDir, followlinks=False):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            for rule in pruneRules:
                if rule.strip()[0] != '!':
                    # This is an inclusion rule
                    if fnmatch.fnmatch(filename, rule):
                        toDel.append(filepath)

                else:
                    # This is an exclusion rule, skip
                    pass

    for rule in pruneRules:
        if rule.strip()[0] == '!':
            rule = rule.strip()[1:].strip()
            rule = '*' + rule
            # This is an exclusion rule.  Apply this rule to the list of files matching the inclusion rules
            # remove the files matching the exclusion rule
            for filepath in toDel:
                if fnmatch.fnmatch(filepath, rule):
                    # Remove from list
                    toDel.remove(filepath)

    #
    # Removal of files action finally
    #
    totalSize = 0
    numFilesDel = 0
    errFiles = []
    for j, path in enumerate(toDel, start=1):
        try:
            logger.debug("%d.Removing %s" % (j, path))
            totalSize += os.path.getsize(path)
            os.remove(path)
            numFilesDel += 1
        except Exception as inst:
            logger.exception("Error removing %s. %s" % (path, inst))
            errFiles.append(path)

    #
    # Log the pruning results
    #
    if len(toDel) > 0 and len(errFiles) == 0:
        status = "Pruned"
        info = "Pruning completed.  %7.2f KB deleted" % float(totalSize / 1024)
        comment = "%d files deleted" % numFilesDel
        # Even when enable_logging is False, we want this message to be logged.
        res.updateMetaData(status, info, totalSize, comment, logger=logger)

    elif len(toDel) > 0 and (len(errFiles) == len(toDel)):
        # All files had an error being removed
        raise Exception("All %d files failed to be removed.  See /var/log/ion/reportsLog.log." % len(errFiles))

    elif len(errFiles) > 0 and len(errFiles) < len(toDel):
        # Some files had an error being removed
        raise Exception("%d of %d files failed to be removed.  See /var/log/ion/reportsLog.log." % (len(errFiles), len(toDel)))

    else:
        # No files were found to remove
        status = "Pruned"
        info = "Pruning completed.  No valid files to remove"
        comment = "%d files deleted" % numFilesDel
        if enable_logging:
            res.updateMetaData(status, info, totalSize, comment, logger=logger)


def getSymlinkList(pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    retDir = ret.get_report_dir()
    symLinkList = []
    linkCheckList = ["/1.wells", "/rawlib.sff"]
    resList = models.Results.objects.all()
    for r in resList:
        #get the full path to r's directory
        rDir = r.get_report_dir()
        for f in linkCheckList:
            rFileDir = rDir + f
            try:
                os.readlink(rFileDir)
                #check whether the symlink points to the report about to be deleted.
                if os.readlink(rFileDir) == (retDir + f):
                    for s in linkCheckList:
                        if f[1:] == s:
                            symLinkList.append(r)
            except:
                pass

    return symLinkList
