#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *
import sys
import os
from iondb.rundb import models
from socket import gethostname

exp = models.Experiment.objects.all()

for e in exp:
    e.storage_options = 'A'
    e.save()
