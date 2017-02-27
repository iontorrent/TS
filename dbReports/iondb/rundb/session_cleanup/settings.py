# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import
from celery.schedules import crontab

weekly_schedule = {
    'task': 'iondb.rundb.session_cleanup.tasks.cleanup',
    'schedule': crontab(hour=0, minute=0, day_of_week="sunday"),
}

nightly_schedule = {
    'task': 'iondb.rundb.session_cleanup.tasks.cleanup',
    'schedule': crontab(hour=0, minute=0),
}

everyminute_schedule = {
    'task': 'iondb.rundb.session_cleanup.tasks.cleanup',
    'schedule': crontab(),
}

hourly_schedule = {
    'task': 'iondb.rundb.session_cleanup.tasks.cleanup',
    'schedule': crontab(hour='*/4', minute=0),
}
