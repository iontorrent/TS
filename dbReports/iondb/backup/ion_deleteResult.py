#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
from os import path
import sys
import commands
import statvfs
import datetime

import iondb.bin.djangoinit
from iondb.rundb import models
from django import shortcuts

import json
from iondb.rundb import json_field

'''
Args: (when run from terminal)
--X: X = pk of report to delete.

This is a script to delete a report.
It will permanently delete a report from the database and filesystem.
However, it will fail if any other reports link to its files.
'''

def deleteReport(pkR, comment, _logger):
    logName = "/tmp/delLog.txt"
    file = open(logName, "w")
    file.write("delete begun...")
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    #get the full path to ret's directory
    retDir = getOldDir(ret)
    #get a list of all results.
    resList = models.Results.objects.all()
    file.write('database entries obtained...')
    
    total_size1 = 0
    for dirpath, dirnames, filenames in os.walk(retDir):
        for f in filenames:
            try:
                fp = os.path.join(dirpath, f)
                total_size1 += os.path.getsize(fp)
            except:
                pass
    
    fail = False
    file.write('beginning symlink check; warning, loop...')
    for r in resList:
        #get the full path to r's directory
        rDir = getOldDir(r)
        file.write('old directory found...')
        #list of files to check for symlinks; use the format, '/<file>.<extension>'
        fileList = ["/1.wells", "basecaller_results/rawlib.sff"]
        file.write('fileList set...')
        for f in fileList:
            rFileDir = rDir + f
            try:
                file.write('checking for symlink...')
                os.readlink(rFileDir)
                file.write('symlink found, checking for effect on ret...')
                #check whether the symlink points to the report about to be deleted.
                if os.readlink(rFileDir) == (oldDir + f):
                    #if it does, don't delete; a more sophisticated action can be added later.
                    fail = True
                    file.write('symlink links to ret; fail...')
            except:
                #print "no symlink detected."
                pass
    if fail:
        raise Exception("symlinks point to this report: cannot delete.")
    file.write('symlinks checked...')
    if not fail:
        file.write('not fail confirmed...')
        models.update_metaData(ret, "Deleting", "Deleting report with pk %s" %pkR, total_size1, comment)
        try:
            file.write('beginning deletion...')
            models.Results.objects.all().filter(pk=pkR).delete()
            file.write('delete success.')
        except Exception as inst:
            file.write("error: delete failure.")
            models.update_metaData(ret, "Failure", "ERROR: failure in results deletion step. Error type: %s"%inst, 0, "")
    else:
        file.write("error: symlinks detected.")
        models.update_metaData(ret, "Failure", "ERROR: symlinks detected drawing from this report. Cannot delete.", 0, "")

def getOldDir(ret):
    oldDir = ret.get_report_dir()
    return oldDir

if __name__ == '__main__':
    pk = ''
    
    if(len(sys.argv) > 1):
        for arg in sys.argv:
            pk = arg[2:]
    
    deleteReport(pk, "Run from terminal.")