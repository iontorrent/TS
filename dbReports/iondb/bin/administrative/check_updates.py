#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
This script will be for checking for updates via the TSconfig and checking for the presense of USB updater
"""

import os
import subprocess
import sys
from distutils.sysconfig import get_python_lib
from iondb.utils.files import rename_extension
from iondb.utils.usb_check import getUSBinstallerpath, change_list_files

# turn off complex traceback stuff
sys.tracebacklimit = 0

# check for root level permissions
if os.geteuid() != 0:
    sys.exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

try:
    path = getUSBinstallerpath()
    if path:
        change_list_files(path)
    if not path:
        rename_extension('etc/apt/', '.USBinstaller', '')
        if os.path.isfile('/etc/apt/sources.list.d/usb.list'):
            os.remove('/etc/apt/sources.list.d/usb.list')

    print(subprocess.check_output([os.path.join(get_python_lib(), 'ion_tsconfig/TSconfig.py'), '--poll']))

except Exception as err:
    rename_extension('etc/apt/', '.USBinstaller', '')
    if os.path.isfile('/etc/apt/sources.list.d/usb.list'):
        os.remove('/etc/apt/sources.list.d/usb.list')
    raise
