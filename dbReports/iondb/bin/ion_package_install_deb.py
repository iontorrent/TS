#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
This script will be executed in order to install a plugin from a deb file but must be run at root level permission
"""
import apt.debfile
import os
import sys

# turn off complex traceback stuff
sys.tracebacklimit = 0
miscDeb = None

# check for root level permissions
if os.geteuid() != 0:
    sys.exit(
        "You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

# look for the file name
if len(sys.argv) < 2:
    sys.exit("There are no plugin packages indicated for install.")

# make sure the misc package is not installed via plugin install page.
if len(sys.argv) == 3:
    miscDeb = sys.argv[2]
    if miscDeb != "miscDeb":
        sys.exit("There are no miscellaneous packages indicated for install.")

filename = sys.argv[1]

# get the information about this package
debFile = apt.debfile.DebPackage(filename=filename)

if not miscDeb and debFile['section'] != 'ion-plugin':
        sys.exit("The package is not part of the ion-plugin section. - " + debFile['section'])

if miscDeb and debFile['section'] not in ['ion-instrument-updates', 'ion-plugin']:
    sys.exit("The package is not part of the offcycle miscellaneous ion package. - " + debFile['section'])

# check for conflicts
if not debFile.check_conflicts():
    sys.exit("There are conflicts with this package. " + ", ".join(debFile.conflicts))

# run the install process
try:
    os.environ['DEBIAN_FRONTEND'] = 'noninteractive'
    debFile.install()
except Exception as err:
    sys.exit(str(err))
