#!/usr/bin/python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
import sys
from iondb.bin import djangoinit
from iondb.rundb.models import Cruncher, Location
NODENAME = sys.argv[1]
newnode, created = Cruncher.objects.get_or_create(name=NODENAME, location=Location.getdefault())
if created:
    sys.stdout.write('%s created' % NODENAME)
else:
    sys.stdout.write('%s exists' % NODENAME)
