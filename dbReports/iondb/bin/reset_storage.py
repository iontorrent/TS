#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *
from django import db
from django.db import transaction
import sys
import os
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

from django.db import models
from iondb.rundb import models
from socket import gethostname

exp = models.Experiment.objects.all()

for e in exp:
    e.storage_options = 'A'
    e.save()
