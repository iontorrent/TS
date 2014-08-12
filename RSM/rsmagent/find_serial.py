# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import sys
import os
import datetime
from os import path
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django.db import models
from iondb.rundb import models
import re



if __name__=="__main__":

    # select all rig objects, loop through one at a time and if/when we find a rig with a serial number, dump it and exit
    rigs = models.Rig.objects.all()
    for rig in rigs:
        if rig.serial is not None and rig.serial != "":
            print '%s' % (rig.serial)
            # break


