#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

"""
This module will install or upgrade ion torrent plugins

All the arguments given to this will be used as the name of the aptitude plugins to install
This can also remove plugins by adding a "=remove" suffix to each package name
"""

import apt
import os
import sys


def validateDependencyUpgrades(candidate, cache):
    """
    This helper method will validate that all of the packages will not force undesirable upgrades
    :param candidate: The python-apt package object to be marked for upgrade
    :param cache: The cache of packages
    :return: A list of all of the illegal package names which would be upgraded
    """

    illegal = list()
    for package in cache.get_changes():

        # this is the package which we are intending to update, it gets a free pass
        if package.name == candidate.source_name:
            continue

        # check that it is not a naughty naughty illegal package
        if package.name.startswith("ion-"):
            illegal.append(package.name)

    # return the list
    return illegal


# turn off complex traceback stuff
sys.tracebacklimit = 0

# check for root level permissions
if os.geteuid() != 0:
    sys.exit(
        "You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting."
    )

if len(sys.argv) < 2:
    sys.exit("There are no plugin packages indicated for install or upgrade.")

# get a list of all of the packages available
pluginPackageNames = sys.argv[1:]

# the current cache for the system
cache = apt.Cache()

# iterate over all of the package names
for pluginPackageName in pluginPackageNames:

    pluginPackageVersion = ""
    if "=" in pluginPackageName:
        pluginPackageName, pluginPackageVersion = pluginPackageName.split("=")

    # check to make sure this is a known package
    if not cache.has_key(pluginPackageName):
        sys.exit("There is no package named " + pluginPackageName)

    # do a sanity check here to make sure we are not trying to upgrade a non-plugin package
    if not pluginPackageName.startswith("ion-plugin-"):
        sys.exit("The package " + pluginPackageName + " is not a plugin package.")

    package = cache[pluginPackageName]

    # are we looking for a particular version
    installVersion = None
    if pluginPackageVersion and pluginPackageVersion != "remove":
        installVersion = package.versions.get(pluginPackageVersion)
        if not installVersion:
            sys.exit(
                "The version "
                + pluginPackageVersion
                + " cannot be found for the package "
                + pluginPackageName
                + "."
            )
        package.candidate = installVersion

    # check to see if the package is available for upgrade or install
    if pluginPackageVersion == "remove" and package.is_installed:
        package.mark_delete()
    elif not package.is_installed or installVersion:
        illegalPackages = validateDependencyUpgrades(package.candidate, cache)
        if len(illegalPackages) > 0:
            sys.exit(
                "This package cannot be upgraded because the following packages would be upgraded: "
                + str(illegalPackages)
            )
        package.mark_install()
    elif package.is_upgradable:
        illegalPackages = validateDependencyUpgrades(package.candidate, cache)
        if len(illegalPackages) > 0:
            sys.exit(
                "This package cannot be upgraded because the following packages would be upgraded: "
                + str(illegalPackages)
            )
        package.mark_upgrade()
    else:
        sys.exit(
            "The package " + pluginPackageName + " does not have any updates available."
        )

# commit the changes!
try:
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"
    cache.commit()
except Exception as err:
    sys.exit(str(err))
