#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
This module will check to see if there is a new version of tsvm available
"""

import apt
import json
import os
import sys

# turn off complex traceback stuff
sys.tracebacklimit = 0

# check for root level permissions
if os.geteuid() != 0:
    sys.exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

# update the aptitude cache
apt_cache = apt.Cache()
apt_cache.update()
apt_cache.open()

# get the available version and current state
state = 'Not Available'
msg = 'No ion-tsvm packages are available for installation.'
if apt_cache.has_key('ion-tsvm'):
    pkg = apt_cache['ion-tsvm']
    msg = 'New ion-tsvm package version %s is available' % pkg.candidate if pkg.is_upgradable else 'ion-tsvm is up to date.'
    state = 'Upgradable' if pkg.is_upgradable else 'Not Upgradable'

# write to stdout
print(json.dumps({'state': state, 'msg': msg}))