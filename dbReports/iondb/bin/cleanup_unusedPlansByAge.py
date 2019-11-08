#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cleanup_unusedPlansByAge")
# log.setLevel(logging.DEBUG)

import iondb.bin.djangoinit
from iondb.rundb import models

import pytz
from datetime import datetime, timedelta


class Intents:
    List, Delete, Mark_as_executed = range(3)


# To retire plans older than n days from present, you can
# mark the plan as executed or deleted it from database entirely.
# Default is to list all the plans matching the default criteria (created more than 30 days ago)
# that will be affected by this script

if __name__ == "__main__":

    INTENT = Intents.List
    AGE_IN_DAYS = 30

    count_toDelete = 0

    oldest = datetime.now(pytz.utc) - timedelta(days=AGE_IN_DAYS)
    log.info(
        "Query for unused plans older than %d days for deletion or be marked as Executed"
        % AGE_IN_DAYS
    )

    qs = models.PlannedExperiment.objects.filter(
        planExecuted=False, isReusable=False, date__lte=oldest
    )

    if qs.count() > 0:
        log.info("Plans to be deleted or marked as executed...")

        for plan in qs:
            count_toDelete += 1
            log.info(
                "%d: id=%d: shortId=%s; name=%s; date=%s"
                % (
                    count_toDelete,
                    plan.id,
                    plan.planShortID,
                    plan.planDisplayedName,
                    plan.date,
                )
            )

            if INTENT == Intents.Delete:
                rc = plan.delete()

        if INTENT == Intents.Mark_as_executed:
            rc = qs.update(planExecuted=True)
            log.info(
                "%d unused plans have been marked as executed. Return code=%s"
                % (count_toDelete, str(rc))
            )
        elif INTENT == Intents.Delete:
            log.info("%d unused plans have been deleted" % (count_toDelete))
        else:
            log.info(
                "Please review plan list above before deleting or marking them as executed. Total count=%d"
                % (count_toDelete)
            )

    else:
        log.info("No unused plans. Nothing to cleanup.")
