#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
This script will lock or unlock the version of the ion software by manipulating the apt source path.
"""
from ion import version
import os
import subprocess
import sys

# turn off complex traceback stuff
sys.tracebacklimit = 0

# check for root level permissions
if os.geteuid() != 0:
    sys.exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

# look for the file name
if len(sys.argv) < 2:
    sys.exit("Enable flag is required")

# parse the first argument to a bool
enable = sys.argv[1].lower() == 'true'

# get distribution string
os_codename = 'trusty'
with open('/etc/lsb-release', 'r') as fp:
    for line in fp.readlines():
        if line.startswith('DISTRIB_CODENAME'):
            os_codename = line.split('=')[1].strip()

if enable:
    find_string = "updates\/software.*%s\/" % os_codename
    replace_string = "updates\/software\/archive\/%s %s\/" % (version, os_codename)
else:
    find_string = "updates\/software\/archive\/%s.*" % version
    replace_string = "updates\/software %s\/" % os_codename
sed_string = "s/%s/%s/g" % (find_string, replace_string)

# Possible locations of Ion Apt repository strings:
#     /etc/apt/sources.list
#     /etc/apt/sources.list.d/*.list
#
filepaths = [os.path.join("/etc/apt/sources.list.d", x) for x in os.listdir("/etc/apt/sources.list.d") if os.path.splitext(x)[1] == '.list']
filepaths.append("/etc/apt/sources.list")
for filepath in filepaths:
    print("Looking in %s" % filepath)
    cmd = ["sed", "-i", sed_string, filepath]

    stdout = ''
    stderr = ''
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode:
            sys.stderr.write(stderr)
    except:
        sys.stderr.write(stderr)
