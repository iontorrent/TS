#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

from pynvml import *
nvmlInit()
print("Driver Version: %s" % nvmlSystemGetDriverVersion())

deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("Device %s: %s" % (i, nvmlDeviceGetName(handle)))

    memory_info = nvmlDeviceGetMemoryInfo(handle)
    print("Device %s: Total memory: %s" % (i,memory_info.total/1024/1024))
    print("Device %s: Free memory: %s" % (i,memory_info.free/1024/1024))
    print("Device %s: Used memory: %s" % (i,memory_info.used/1024/1024))

    util = nvmlDeviceGetUtilizationRates(handle)
    print("Device %s: GPU Utilization: %s%%" % (i,util.gpu))
    print("Device %s: Memory Utilization: %s%%" % (i,util.memory))

nvmlShutdown()
