#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
This script will be executed in order to install a plugin from a deb file but must be run at root level permission
"""
import apt.debfile
import os
import sys
import platform

_, _, DISTRO_CODENAME = platform.linux_distribution()

SUPPORTED_DISTROS = ["trusty", "bionic"]

# turn off complex traceback stuff
sys.tracebacklimit = 0
miscDeb = None

# check for root level permissions
if os.geteuid() != 0:
    sys.exit(
        "You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting."
    )

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

deb_version = debFile["version"]
has_distro_in_version = any([(distro in deb_version) for distro in SUPPORTED_DISTROS])
# only check for distro compatibility if specified in the package version
if has_distro_in_version and DISTRO_CODENAME not in deb_version:
    msg = "The package has distro specified in version ({ver}), which is not compatible with current system ({cur})"
    sys.exit(msg.format(ver=deb_version, cur=DISTRO_CODENAME))

if not miscDeb and debFile["section"] != "ion-plugin":
    sys.exit(
        "The package is not part of the ion-plugin section. - " + debFile["section"]
    )

if miscDeb and debFile["section"] not in ["ion-instrument-updates", "ion-plugin"]:
    sys.exit(
        "The package is not part of the offcycle miscellaneous ion package. - "
        + debFile["section"]
    )

# check for conflicts
if not debFile.check_conflicts():
    sys.exit("There are conflicts with this package. " + ", ".join(debFile.conflicts))

# run the install process
try:
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"
    debFile.install()
except Exception as err:
    sys.exit(str(err))
