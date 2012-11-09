#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import iondb.bin.djangoinit
from iondb.rundb import models

import socket
rigs = models.Rig.objects.all().order_by('pk')
for rig in rigs:
    try:
        IP = ((socket.getaddrinfo(rig.name,None)[0])[4])[0]
        print "%-20s %s" % (rig.name,IP)
    except:
        print "%-20s invalid" % rig.name
