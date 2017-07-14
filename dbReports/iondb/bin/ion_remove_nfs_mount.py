#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
This module will remove an nfs mount

This will require one argument which is the mount point
"""

import os
import shutil
import sys
import subprocess


# turn off complex traceback stuff
sys.tracebacklimit = 0

# check for root level permissions
if os.geteuid() != 0:
    sys.exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

if len(sys.argv) < 2:
    sys.exit("Not enough arguments..")

# get the arguments
mountpoint = sys.argv[1]

filename = "/usr/share/ion-tsconfig/ansible/group_vars/all_local"
tempfile = "/tmp/all_local.temp"
# Delete the line matching the pattern
with open(tempfile, "w") as out, open(filename, "r") as fh:
    for line in fh:
        if mountpoint not in line:
            out.write(line)
shutil.move(tempfile, filename)

# Which hosts file to use
hosts_file = '/usr/share/ion-tsconfig/ansible/torrentsuite_hosts'
if os.path.isfile(hosts_file + '_local'):
    hosts_file += '_local'

# Run ansible-playbook in a subshell since running it from python celery task is broken
cmd = ["ansible-playbook", "-i",  hosts_file, 'nfs_client.yml', "--sudo"]
p = subprocess.Popen(cmd, cwd="/usr/share/ion-tsconfig/ansible", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = p.communicate()
sys.stdout.write(stdout)
sys.stderr.write(stderr)