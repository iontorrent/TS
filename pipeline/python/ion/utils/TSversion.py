# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import socket
from ion import version
import traceback
import subprocess
import sys
import os
from django.utils.datastructures import SortedDict

# Note this list is duplicated in TSconfig.py
packages = [
    'ion-analysis',
    'ion-dbreports',
    'ion-docs',
    'ion-gpu',
    'ion-pipeline',
    'ion-publishers',
    'ion-referencelibrary',
    'ion-rsmts',
    'ion-sampledata',
    'ion-torrentpy',
    'ion-torrentr',
    'ion-tsconfig',
    ]

offcycle_packages = [
    'ion-plugins',
    'ion-chefupdates',
    'ion-onetouchupdater',
    'ion-pgmupdates',
    'ion-protonupdates',
    'ion-s5updates',
]


def findVersions():
    """
    find the version of the packages
    """
    ret = SortedDict()
    for package in packages:
        # command for version checking
        com = "dpkg -l %s | grep ^ii | awk '{print $3}'" % package
        try:
            a = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # just get the version number
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


def offcycleVersions():
    ret = SortedDict()
    for package in offcycle_packages:
        # command for version checking
        com = "dpkg -l %s | grep ^ii | awk '{print $3}'" % package
        try:
            a = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # just get the version number
            tolist = a.stdout.readlines()[0].strip()
            ret[package] = tolist
        except:
            pass

    return ret


def findUpdates():
    """
    find package versions installed and candidate for updates webpage
    """
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


def findOSversion():
    """
    find OS info
    """
    com = "cat /etc/lsb-release"
    try:
        a = subprocess.Popen(com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = a.stdout.read().splitlines()
        ret = dict(line[8:].split('=') for line in stdout if line.startswith('DISTRIB_'))
    except:
        ret = {}
    return ret
