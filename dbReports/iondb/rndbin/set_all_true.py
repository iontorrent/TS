#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os
from os import path
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django.db import models
from iondb.rundb import models


def setTrue():
    temp = models.Template.objects.all()
    print temp
    for t in temp:
        print t
        t.isofficial = True
        t.save()
        print t.isofficial


def change_webpath(newpath):
    temp = models.Results.objects.all()
    for result in temp:
        explode = str(result.reportLink).split('/')
        print explode
        post = '/'.join(explode[3:])
        print path.join(newpath,post)
        

if __name__=="__main__":
    #setTrue()
    change_webpath('http://ionwest.iontorrent.com')
