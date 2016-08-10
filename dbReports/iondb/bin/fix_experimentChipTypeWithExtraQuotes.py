#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from djangoinit import *

import sys
import os
from iondb.rundb import models


''' remove extra quotes or backslash in Experiment.chipType '''

chipTypes = models.Experiment.objects.all().values_list('chipType', flat=True).distinct('chipType')

for chipType in filter(lambda x: '\\' in x, chipTypes):
    badChars = '\\'
    clean_chipType = chipType.replace(badChars, '')

    exps = models.Experiment.objects.all().filter(chipType=chipType)
    print("FIX-1: Going to fix %d experiments by replacing chipType %s with %s" %
          (exps.count(), badChars, clean_chipType))

    models.Experiment.objects.filter(chipType=chipType).update(chipType=clean_chipType)


chipTypes = models.Experiment.objects.all().values_list('chipType', flat=True).distinct('chipType')

for chipType in filter(lambda x: '"' in x, chipTypes):
    badChars = '"'
    clean_chipType = chipType.replace(badChars, '')

    exps = models.Experiment.objects.all().filter(chipType=chipType)
    print("FIX-2: Going to fix %d experiments by replacing chipType %s with %s" %
          (exps.count(), badChars, clean_chipType))

    models.Experiment.objects.filter(chipType=chipType).update(chipType=clean_chipType)
