#!/usr/bin/env python
# Copyright (C) 2020 Ion Torrent Systems, Inc. All Rights Reserved

from ion import version
import os
import subprocess
import sys

# turn off complex traceback stuff
sys.tracebacklimit = 0

# check for root level permissions
if os.geteuid() != 0:
    sys.exit(
        "You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting."
    )

# look for the file name
if len(sys.argv) < 2:
    sys.exit("enableGeneStudioRepo flag is required")

switchRepo = sys.argv[1].lower()
# Current repo stays intact for PGM/Proton

# get distribution string
os_codename = "trusty"
with open("/etc/lsb-release", "r") as fp:
    for line in fp.readlines():
        if line.startswith("DISTRIB_CODENAME"):
            os_codename = line.split("=")[1].strip()

if switchRepo == "s5_only":
    find_string = "updates\/software\/ %s\/" % os_codename
    replace_string = "updates\/software\/ %s\/" % (os_codename + '-' + "genestudio")
if switchRepo == "pgm_or_proton_only":
    find_string = "updates\/software\/ %s-%s\/" % (os_codename, 'genestudio')
    replace_string = "updates\/software\/ %s\/" % (os_codename)

sed_string = "/^[[:space:]]*#/!s/%s/%s/g" % (find_string, replace_string) # skip commented line
# Possible locations of Ion Apt repository strings:
#     /etc/apt/sources.list
#     /etc/apt/sources.list.d/*.list
#
filepaths = [
    os.path.join("/etc/apt/sources.list.d", x)
    for x in os.listdir("/etc/apt/sources.list.d")
    if os.path.splitext(x)[1] == ".list" and x != "iontorrent-offcycle.list"
]
filepaths.append("/etc/apt/sources.list")
for filepath in filepaths:
    cmd = ["sed", "-i", sed_string, filepath]

    stdout = ""
    stderr = ""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode:
            sys.stderr.write(stderr)
    except Exception:
        sys.stderr.write(stderr)
try:
    from ion_tsconfig import TSconfigPath
    subprocess.check_output([os.path.join(TSconfigPath, "TSconfig.py"), "--poll"])
except Exception as err:
    raise

print "Switching Repo to %s successful. Please proceed with the software upgrade. (TSconfig -s)" % sys.argv[1]