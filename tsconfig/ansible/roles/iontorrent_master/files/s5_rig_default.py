#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
# Edit the default Rig record in Rigs table for S5 custom parameters

from iondb.bin import djangoinit
from iondb.rundb import models

if __name__ == '__main__':
    resultobj = models.Rig.objects.get(name='default')
    resultobj.ftpserver = '192.168.122.249'
    resultobj.ftpusername = 'ionguest'
    resultobj.ftppassword = 'ionguest'
    resultobj.updatehome = '192.168.122.249'
    resultobj.save()
