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

def deleteReport(pkR):
    ret = shortcuts.get_object_or_404(models.Results, pk=pkR)
    ret.delete()

if __name__ == '__main__':
    pk = ''
    
    if(len(sys.argv) > 1):
        for arg in sys.argv:
            pk = arg[2:]
            
    deleteReport(pk)