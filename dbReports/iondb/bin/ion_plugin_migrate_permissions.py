#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
# this script is used to migrate the permissions of unsupported plugins to make sure they have the correct permissions set

import os
import sys
import subprocess
from pwd import getpwnam
from grp import getgrnam

# turn off complex traceback stuff
sys.tracebacklimit = 0

# check for root level permissions
if os.geteuid() != 0:
    sys.exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

# check the correct number of arguments
if len(sys.argv) < 2:
    sys.exit("There are no plugin packages indicated for permissions migration.")

# check to make sure the directory is present
pluginName = sys.argv[1]
plugin_root = os.path.join('/results/plugins', pluginName)
if not os.path.exists(plugin_root):
    sys.exit("The plugin path does not exist for " + pluginName)

# find the launch script
pathToShellScript = os.path.join(plugin_root, 'launch.sh')
pathToPythonScript = os.path.join(plugin_root, pluginName + '.py')
script = pathToPythonScript if os.path.exists(pathToPythonScript) else pathToShellScript

# get the package name
packageName = ''
try:
    packageName = subprocess.check_output(['/usr/bin/dpkg', '-S', script]).split(':')[0]
except:
    pass

if packageName:
    sys.exit("We cannot change the permissions on a file under package management of the package " + packageName)

# get the www-data user and walk the
www_data_uid = getpwnam('www-data').pw_uid
www_data_gid = getgrnam('www-data').gr_gid
os.chown(plugin_root, www_data_uid, www_data_gid)
for root, dirs, files in os.walk(plugin_root):
    for cur_dir in dirs:
        os.chown(os.path.join(root, cur_dir), www_data_uid, www_data_gid)
    for cur_file in files:
        os.chown(os.path.join(root, cur_file), www_data_uid, www_data_gid)

