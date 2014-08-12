#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('cleanup_orphan_plans')
#log.setLevel(logging.DEBUG)

import iondb.bin.djangoinit
from iondb.rundb import models

if __name__ == '__main__':
    count_orphan = 0
    count_good = 0
    count_total = 0

    log.info("Querying for bogus plans. Please Wait...")
    qs = models.PlannedExperiment.objects.filter(planExecuted=True, planName='Copy of system default', experiment=None)
    expected = qs.count()
    for p in qs:
        if (count_total % 10000) == 0:
            log.info("Reviewed %d/%d (%d%%) plans... good/bad = %d/%d", count_total, expected, (count_total/expected) * 100, count_good, count_orphan)
        count_total += 1
        exp = p.experiment_set.all()
        if exp.exists():
            count_good += 1
            #for e in exp:
            #    log.debug("Plan %d [%s] - Experiment: %s", p.id, p.planShortID, e)
            #break
        else:
            count_orphan += 1
            log.debug("Orphan Plan %d: %s - %s", p.id, p.planShortID, p.date)
            #p.delete()

    if expected:
        log.info("Reviewed %d/%d (%d%%) plans... good/bad = %d/%d", count_total, expected, (count_total/expected) * 100, count_good, count_orphan)
    else:
        log.info("All clean. Nothing to cleanup")

