#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
This module will install a new nfs mount

This module will require three arguments
1) Server name: The host or ip address of the nas server
2) Share name: The name of the share to be mounted
3) Mount point: The place on the local file system to mount it.
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

if len(sys.argv) < 4:
    sys.exit("Not enough arguments..")

# get the arguments
servername = sys.argv[1]
sharename = sys.argv[2]
mountpoint = sys.argv[3]

# create all_local from all
filename = "/usr/share/ion-tsconfig/ansible/group_vars/all_local"
tempfile_path = "/usr/share/ion-tsconfig/ansible/group_vars/all_local.temp"
if not os.path.isfile(filename):
    shutil.copy2("/usr/share/ion-tsconfig/ansible/group_vars/all", filename)
# Line to add - spaces are important!
new_nas = "  - { name: %s, directory: %s, mountpoint: %s, options: %s }" % (servername, sharename, mountpoint, "defaults")
# Check if line exists.  This is a small file so we can do this cheat
with open(filename, 'r') as fh:
    if new_nas in fh.read():
        print("This line: '%s'. Already exists in all_local." % new_nas)
    else:
        fh.seek(0)
        # Insert line after pattern ^nas_mounts:
        with open(tempfile_path, "w") as out:
            for line in fh:
                out.write(line)
                if 'nas_mounts:' in line:
                    out.write(new_nas + '\n')
        shutil.move(tempfile_path, filename)

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