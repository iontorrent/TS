# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import subprocess


class Devices:
    def __init__(self, path, type, blocks, used, avail, capac, mounted):
        self.name = mounted.strip().split("/")[-1].strip()
        self.path = mounted
        self.type = type
        self.blocks = blocks
        self.used = used
        self.avail = avail
        self.capac = capac
        self.mounted = mounted

    def get_name(self):
        return self.name

    def get_free_space(self):
        return float(100 - int(self.capac.split('%')[0]))

    def get_path(self):
        return self.mounted

    def get_available(self):
        return self.avail

    def get_type(self):
        return self.type


def disk_report():
    report = {}  # dictionary, {'deviceName': [type,1024-blocks,Used,Aval,Capac,MountedOn]}
    #If df fails after 2 seconds kill the process
    p = subprocess.Popen("ion_timeout.sh 2 df -TP", shell=True,
                         stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()

    dat = [i.strip().split(' ') for i in stdout.splitlines(True)][1:]
    for i in dat:
        key = i[0]
        report[key] = []
        for j in i:
            if j != '' and j != key:
                report[key].append(j)
    devices = []
    for k, v in report.iteritems():
        type = v[0]
        blocks = v[1]
        used = v[2]
        avail = v[3]
        capac = v[4]
        mounted = v[5]
        devices.append(Devices(k, type, blocks, used, avail, capac, mounted))
    return devices


def to_media(devArr):
    ret = []
    for i in devArr:
        path = i.get_path()
        type = i.get_type()
        # Report Data Management requires an ext3/4 filesystem or nfs (anything that supports symbolic links actually)
        #if 'media' in path and ('ext' in type or 'nfs' in type):
        #if 'nfs' in type or ('/media' in path) or ('/mnt' in path):
        if 'nfs' in type or path.startswith('/media') or path.startswith('/mnt'):
            try:
                if os.path.exists(os.path.join(path, '.not_an_archive')):
                    continue
            except:
                continue
            ret.append((path, path))
    return ret
