#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import fnmatch


def getUSBinstallerpath():
    for path, dirs, files in os.walk("/media"):
        for directory in dirs:
            if directory.startswith("TS"):
                if os.path.exists(os.path.join(path, directory, "runme")):
                    return os.path.join(path, directory)
    return ""


def change_list_files(path):
    # if there is a USB, then get all the listfiles and rename them
    # loops through all subdirectories of the given directory
    # returns only the files that have the extension ".list"
    list_files = [os.path.join(dirpath, f)
                  for dirpath, _, files in os.walk(os.path.join('/etc/apt'))
                  for f in fnmatch.filter(files, '*.list')]
    # this ensures that the installed packages will come from only the usb.
    for filename in list_files:
        if filename != 'usb.list':  # dont rename if it is usb.list
            newfile = os.path.join(filename + '.USBinstaller')
            os.rename(filename, newfile)
    value = 'deb file:' + path + ' updates/\n'
    with open('/etc/apt/sources.list.d/usb.list', 'w') as f:
        f.write(value)
