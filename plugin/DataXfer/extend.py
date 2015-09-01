#!/usr/bin/python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

import json
import os
import subprocess
import re

pluginName = 'DataXfer'
pluginDir = ""

networkFS = ["nfs"]
localFS = ["ext4","ext3","xfs","ntfs","exfat"]
supportedFS = ",".join(localFS + networkFS)

def test(bucket):
	return bucket

def runProcess(exe):
    p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')

def runProcessAndReturnLastLine(exe):
    p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return  p.stdout.readlines()[-1]

def backupDevices(bucket):
    devices =""
    cmd = "mount -l -t " + supportedFS
    for line in runProcess(cmd.split()):
        line_arr = line.split()
        folder = line_arr[2]
        fstype = line_arr[4]
        perms = line_arr[5]
        if perms.find('w') != -1:
            use = True
            if fstype in localFS:
                m = re.match('^(/media|/mnt)', folder)
                if not m:
                    use = False
            if use:
                cmd2 = "df -h %s " % folder
                df = runProcessAndReturnLastLine(cmd2.split())
                avail = df.split()[2]
                devices = devices + "<OPTION VALUE=\"" + folder + "\">" + folder + " (" + avail + " free, " + fstype + ")</option>"

    return devices
