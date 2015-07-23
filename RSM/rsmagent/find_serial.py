# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
"""Get list of attached instruments, to be sent to Axeda"""
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django.db import models
from iondb.rundb import models


def main():
    """Get list of attached instruments, to be sent to Axeda"""

    # select all rig objects, loop through one at a time and if/when we find
    # a rig with a serial number, dump it
    rigs = models.Rig.objects.all()
    for rig in rigs:
        if rig.serial is not None and rig.serial != "":
            print '%s' % (rig.serial)

if __name__ == "__main__":
    main()
