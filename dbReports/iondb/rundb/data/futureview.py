#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import pytz
import traceback
from iondb.bin import djangoinit
from iondb.rundb import models
from datetime import timedelta, datetime
from iondb.rundb.data.dmactions_types import FILESET_TYPES
from iondb.rundb.data.dm_utils import update_diskspace
import iondb.settings as settings

THRESHOLD_DAYS = None
BACKFILL = False

whole_enchillada_exceeds = 0
whole_enchillada_safe = 0
for type in FILESET_TYPES:
    stats = models.DMFileStat.objects.filter(dmfileset__type=type).filter(action_state__in=['L','S','N','A']).order_by('created')
    # Get age threshold for this category
    dmfileset = models.DMFileSet.objects.filter(version=settings.RELVERSION).filter(type=type)
    if not THRESHOLD_DAYS: AGE_THRESH_DAYS = dmfileset[0].auto_trigger_age
    else: AGE_THRESH_DAYS = THRESHOLD_DAYS
    threshdate = datetime.now(pytz.UTC) - timedelta(days=AGE_THRESH_DAYS)
    exceed_thresh_stats = stats.filter(created__lt=threshdate)
    within_thresh_stats = stats.filter(created__gte=threshdate)
    howmuchexceeds = 0
    howmuchsafe = 0
    unknownexceeds = 0
    unknownsafe = 0
    for item in exceed_thresh_stats:
        if item.diskspace != None:
            howmuchexceeds += item.diskspace
        else:
            if BACKFILL:
                print "getting %s diskspace for %s" % (type, item.result.resultsName)
                try:
                    howmuchexceeds += update_diskspace(item)
                except:
                    print traceback.format_exc()
            else:
                unknownexceeds += 1

    for item in within_thresh_stats:
        if item.diskspace != None:
            howmuchsafe += item.diskspace
        else:
            '''
            if BACKFILL:
                print "getting diskspace for %s" % (item.result.resultsName)
                try:
                    update_diskspace(item)
                    howmuchsafe += item.diskspace
                except:
                    print traceback.format_exc()
            else:
                unknown += 1
            '''
            unknownsafe += 1

    print ""
    print "%23s: %d" % (type,stats.count())
    print "%23s: %d days" % ("Age threshold",AGE_THRESH_DAYS)
    print "%23s: %9d MB (%d)" % ("Over-threshold",howmuchexceeds,exceed_thresh_stats.count())
    print "%23s: %9d MB (%d)" % ("Under-threshold",howmuchsafe,within_thresh_stats.count())
    print "%23s: %d" % ("Unknown Exceeding",unknownexceeds)
    print "%23s: %d" % ("Unknown Safe",unknownsafe)
    whole_enchillada_exceeds += howmuchexceeds
    whole_enchillada_safe += howmuchsafe

print "%s" % ('='*46)
print "%23s: %9d MB" % ("Total Available to Free",whole_enchillada_exceeds)
print "%23s: %9d MB" % ("Total Protected",whole_enchillada_safe)
