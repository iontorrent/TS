#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import sys
import os
import tarfile
from django import http, shortcuts, template
from iondb.rundb import models
import fnmatch
import string
import commands

'''tf = tarfile.open(path)
ti = tf.extract('InitLog.txt', path='/tmp/tempLog/')
f.write('%s'%open('/tmp/tempLog/InitLog.txt', 'r').readlines())'''

def showpage(request):
    
    site_name = models.GlobalConfig.objects.all().order_by('id')[0].site_name
    
    #TODO: search all File Servers for a "Chips" directory
    fileservers = models.FileServer.objects.all()
    files = {}
    locList = []
    for server in fileservers:
        directory = os.path.join(server.filesPrefix,'Chips')
        if os.path.isdir(directory):
            files[server.name] = []
            listoffiles = os.listdir(directory)
            listoffiles.sort()
            listoffiles.reverse()
            for file in listoffiles:
                if (fnmatch.fnmatch(file, "*AutoPHFail*") or fnmatch.fnmatch(file, "*AutoPHPass*")) and not fnmatch.fnmatch(file, '*.zip'):
                    fileLoc = string.split(file, '_')[0]
                    if not [fileLoc, server.name] in locList:
                        locList.append([fileLoc, server.name])
                    tf = tarfile.open('%s/%s'%(directory, file))
                    ti = tf.extract('InitLog.txt', path='/tmp/tempLogChips/')
                    logLines = []
                    for line in open('/tmp/tempLogChips/InitLog.txt', 'r').readlines():
                        logLines.append(line)
                    output = commands.getstatusoutput('rm -rf /tmp/tempLogChips')
                    if fnmatch.fnmatch(file, "*AutoPHFail*"):
                        passVar = 'F'
                    elif fnmatch.fnmatch(file, "*AutoPHPass*"):
                        passVar = 'T'
                    files[server.name].append([string.split(file, '.')[0], fileLoc, '%s/%s'%(directory, file), logLines, passVar])
                elif fnmatch.fnmatch(file, "*AutoPHPass*") and not fnmatch.fnmatch(file, '*.zip'):
                    #create list of files that passed.
                    pass
    
    ctxd = {
        "error_state":0,
        "locations_list":locList,
        "base_site_name": site_name,
        "files":files,
        }
    ctx = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/ion_chips.html", context_instance=ctx)
