# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os, sys
import subprocess

class Devices:
    def __init__(self, path, blocks, used, avail, capac, mounted):
        self.name = mounted.strip().split("/")[-1].strip()
        self.path = mounted
        self.blocks = blocks
        self.used = used
        self.avail = avail
        self.capac = capac
        self.mounted = mounted
    
    def get_name(self):
        return self.name

    def get_free_space(self):
        return float(100-int(self.capac.split('%')[0]))
    
    def get_path(self):
        return self.mounted     
    
    def get_available(self):
        return self.avail

def disk_report():  
    report = {} # dictionary, {'deviceName': [1024-blocks,Used,Aval,Capac,MountedOn]}
    #If df fails after 2 seconds kill the process
    p = subprocess.Popen("ion_timeout.sh 2 df -P", shell=True,  
                         stdout=subprocess.PIPE)  
    stdout, stderr = p.communicate()
    
    dat = [i.strip().split(' ') for i in stdout.splitlines(True)][1:]
    for i in dat:
        key = i[0]
        report[key]=[]
        for j in i:
            if j != '' and j != key:
                report[key].append(j)      
    devices = []
    for k,v in report.iteritems():
        blocks = v[0]
        used = v[1]
        avail = v[2]
        capac = v[3]
        mounted = v[4]
        devices.append(Devices(k,blocks,used,avail,capac,mounted))
    return devices

def to_media(devArr):
    ret = []
    for i in devArr:
        path = i.get_path()
        if 'media' in path:
            ret.append((path,path))
    return ret
