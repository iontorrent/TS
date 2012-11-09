#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import sys
import os
import tarfile
from django import http, shortcuts, template
from iondb.rundb import models
import fnmatch
import string

def showpage(request):
    
    site_name = models.GlobalConfig.get().site_name
    
    # search all File Servers for a "Chips" directory
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
                if fnmatch.fnmatch(file, "*AutoPH*.bz2"):
                    instName = string.split(file, '_')[0]
                    if not [instName, server.name] in locList:
                        locList.append([instName, server.name])
                    
                    if fnmatch.fnmatch(file, "*AutoPHFail*"):
                        passVar = 'F'
                    elif fnmatch.fnmatch(file, "*AutoPHPass*"):
                        passVar = 'T'
                        
                    files[server.name].append([string.split(file, '.')[0], instName, os.path.join(directory, file), passVar])
    
    ctxd = {
        "error_state":0,
        "locations_list":locList,
        "base_site_name": site_name,
        "files":files,
        }
    ctx = template.RequestContext(request, ctxd)
    return shortcuts.render_to_response("rundb/configure/ion_chips.html", context_instance=ctx)
