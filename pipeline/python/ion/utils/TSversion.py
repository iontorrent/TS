# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import socket
from ion import version
import traceback
import subprocess
import sys
import os
from django.utils.datastructures import SortedDict
def findVersions():
    """
    find the version of the packages
    """
    packages =  ["ion-analysis",
                 "ion-dbreports",
                 "ion-docs",
                 "ion-gpu",
                 "ion-pgmupdates",
                 "ion-protonupdates",
                 "ion-chefupdates",
                 "ion-plugins",
                 "ion-referencelibrary",
                 "ion-rsmts",
                 "ion-sampledata",
                 "ion-tsconfig",
                 "ion-tsups",
                 "ion-publishers",
                 "ion-onetouchupdater",
                 "ion-pipeline",
                 "ion-usbmount",
                 "tmap",
                 "ion-torrentr"]

    packages = sorted(packages)

    ret = SortedDict()
    for package in packages:
        #command for version checking
        com = "dpkg -l %s | grep ^ii | awk '{print $3}'" % package
        try:
            a = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            #just get the version number
            tolist = a.stdout.readlines()[0].strip()
            ret[package] = tolist
        except:
#            traceback.print_exc()
            pass

    from ion import version
    meta_version = version
#    print ret
#    print meta_version

    return ret, meta_version
    
def findUpdates():
    """
    find package versions installed and candidate for updates webpage
    """
    packages =  ["ion-analysis",
                 "ion-dbreports",
                 "ion-docs",
                 "ion-gpu",
                 "ion-pgmupdates",
                 "ion-protonupdates",
                 "ion-chefupdates",
                 "ion-plugins",
                 "ion-referencelibrary",
                 "ion-rsmts",
                 "ion-sampledata",
                 "ion-tsconfig",
                 "ion-tsups",
                 "ion-publishers",
                 "ion-onetouchupdater",
                 "ion-pipeline",
                 "ion-usbmount",
                 "tmap",
                 "ion-torrentr"]
    
    packages = sorted(packages)

    ret = SortedDict()
    for package in packages:
        com = "apt-cache policy %s | grep 'Installed\|Candidate'" % package
        try:
            a = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout = a.stdout.readlines()
            installed = stdout[0].split()[1]
            if installed != '(none)':
              ret[package] = (installed, stdout[1].split()[1])
        except:
            pass

    from ion import version
    meta_version = version

    return ret, meta_version
