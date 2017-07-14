#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
This will install or upgrade the tsvm package
"""

import json
import os
import sys

TSVM_PACKAGE = 'ion-tsvm'

# turn off complex traceback stuff
sys.tracebacklimit = 0

# check for root level permissions
if os.geteuid() != 0:
    sys.exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

# NOTE: Must be run with root privileges
response_obj = {'action': 'update'}
try:
    import pkg_resources
    from distutils.version import LooseVersion
    import apt
    apt_cache = apt.Cache()
    apt_cache.update()
    apt_cache.open()

    if apt_cache.has_key(TSVM_PACKAGE):
        pkg = apt_cache[TSVM_PACKAGE]

        if bool(pkg.is_installed):
            pkg.mark_upgrade()
        else:
            pkg.mark_install()

        apt_cache.commit()
        response_obj.update({'state': 'Success', 'msg': "ion-tsvm updated"})
except Exception as e:
    response_obj.update({'state': 'Error', 'msg': str(e)})

print(json.dumps(response_obj))
