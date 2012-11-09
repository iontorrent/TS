#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
from os import path
import sys
import commands
import statvfs
import datetime
import shutil

import iondb.bin.djangoinit
from iondb.rundb import models
from django import shortcuts

import json
from iondb.rundb import json_field

def archiveReport(pkR, dest):
    #first, define the result and get its path. Also update the destination to include the report name.
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    oldDir = getOldDir(ret)
    print '\nArchiving report from ', oldDir
    slashList = []
    i = 0
    for ch in oldDir:
        if ch == '/':
            slashList.append(i)
        i+=1
    reportFolder = oldDir[slashList[len(slashList)-1]+1:]
    if dest[len(dest)-1] == '/':
        newDir = dest + reportFolder
    else:
        newDir = dest + '/' + reportFolder
    print '\nto ', newDir
    #next, use shutil to copy the entire directory from oldDir to the archive destination.
    shutil.copytree(oldDir, newDir)
    
    #check the sizes of the old directory and the destination.
    total_size1 = 0
    for dirpath, dirnames, filenames in os.walk(oldDir):
        for f in filenames:
            try:
                fp = os.path.join(dirpath, f)
                total_size1 += os.path.getsize(fp)
            except:
                pass
    
    total_size2 = 0
    for dirpath, dirnames, filenames in os.walk(newDir):
        for f in filenames:
            try:
                fp = os.path.join(dirpath, f)
                total_size2 += os.path.getsize(fp)
            except:
                pass
    
    #if the sizes are identical, that means that all of the files were copied. If so, delete the old directory.
    if total_size1 == total_size2:
        shutil.rmtree(oldDir)
        models.update_metaData(ret, "Archived", "Finished archiving.")
    
def getOldDir(ret):
    #This is a lot of effort to go through for a full path - is there a better way?
    tempDir = "%s"%ret.reportstorage
    oldDir1 = ''
    start = False
    for ch in tempDir:
        if start:
            oldDir1 += ch
        else:
            if ch == '(':
                start = True
    oldDir1 = oldDir1[:len(oldDir1)-1]
    
    tempDir = ret.reportLink
    oldDir2 = '/'
    start = 0
    for ch in tempDir:
        if start >= 2:
            oldDir2+=ch
        if ch == '/':
            start+=1
    oldDir2 = oldDir2[:len(oldDir2)-1]
    
    oldDir = oldDir1 + oldDir2
    return oldDir
    
if __name__ == '__main__':
    pk = ''
    dir = ''
    if(len(sys.argv) > 1):
        for arg in sys.argv:
            if arg[:4] == '--pk':
                pk = arg[4:]
            elif arg[:5] == '--dir':
                dir = arg[5:]
    archiveReport(pk, dir)