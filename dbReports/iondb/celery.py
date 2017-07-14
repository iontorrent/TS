# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from celery import Celery
from iondb.rundb.session_cleanup.settings import hourly_schedule

app = Celery('iondb')

app.conf.update(
    BROKER_URL="amqp://ion:ionadmin@localhost:5672/ion",
    BROKER_CONNECT_TIMEOUT=12,

    # Try to reconnect forever if disconnected
    # BROKER_CONNECTION_RETRY=True, # default
    BROKER_CONNECTION_MAX_RETRIES=None,  # retry forever

    # Disable heartbeats
    BROKER_HEARTBEAT=0,

    CELERY_DEFAULT_QUEUE="w1",

    # Not amqp. amqp backend only allows results to be fetched once
    # some tasks query result multiple times.
    CELERY_RESULT_BACKEND='cache+memcached://127.0.0.1:11211/',

    # Avoid indefinite hangs by forcing results to expire after 30 minutes
    CELERY_TASK_RESULT_EXPIRES=1800,

    # Tracking rate limits is expensive, not used
    CELERY_DISABLE_RATE_LIMITS=True,

    # Do not mark a queued task as executed until after it completes or
    # raises an exception
    CELERY_ACKS_LATE=True,

    CELERY_ACCEPT_CONTENT=['pickle', 'json'],
    CELERY_TASK_SERIALIZER='pickle',
    CELERY_RESULT_SERIALIZER='pickle',

    CELERY_TIMEZONE='UTC',
    CELERY_ENABLE_UTC=True,

    # Each worker will only take one task at a time from the queue
    # rather than the default of grabbing several.  This is acceptable because
    # we have relatively few tasks and the amqp connection overhead is not high.
    CELERYD_PREFETCH_MULTIPLIER=1,
    # Allow tasks the generous run-time of six hours before they're killed.
    CELERYD_TASK_TIME_LIMIT=21600,
    # Restart celery each time to ensure imported plugin data is fresh
    CELERYD_MAX_TASKS_PER_CHILD=1,

    CELERY_IMPORTS=(
        "iondb.rundb.tasks",
        "iondb.rundb.publishers",
        "iondb.plugins.tasks",
        "iondb.rundb.data.tasks",
        "iondb.rundb.session_cleanup.tasks",
        "iondb.rundb.data.data_management",
        "iondb.rundb.data.data_import",
        "iondb.rundb.tsvm",
        "iondb.rundb.data.data_export",
        "iondb.rundb.configure.cluster_info",
    ),

    CELERYBEAT_SCHEDULE = {
        'session_cleanup': hourly_schedule
    }
)

# a test task added to this module for task testing


@app.task(bind=True)
def debug_task(self):
    print('Request: {0!r}'.format(self.request))
