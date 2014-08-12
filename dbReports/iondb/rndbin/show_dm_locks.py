#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.bin import djangoinit
#import iondb.settings as settings
#from iondb.utils.TaskLock import TaskLock
from iondb.rundb.models import FileServer, ReportStorage
from iondb.rundb.data import data_management as dm
from iondb.rundb.data import dmactions_types
from django.core.cache import cache

print "Shows Data Management jobs with cache locks"

objects=["manage_manual_action"]
file_servers = FileServer.objects.all().order_by('pk').values()
report_storages = ReportStorage.objects.all().order_by('pk').values()
for partition in dm._partitions(file_servers, report_storages):
    for types in dmactions_types.FILESET_TYPES:
        foo = "%s_%s" %(hex(partition['devid']),dm.slugify(types))
        objects.append(foo)

print ("%-30s %14s" % (30 * '=', 14 * '='))
print ("%-30s %s" % ('Name of lock', 'Name of Report'))
print ("%-30s %14s" % (30 * '=', 14 * '='))

for lockid in sorted(list(set(objects))):
    thislock = cache.get(lockid)
    print ("%-30s %s" % (lockid, thislock))
