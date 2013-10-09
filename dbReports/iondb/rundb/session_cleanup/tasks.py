# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from celery.task import task
from django.conf import settings
from django.contrib.sessions.models import Session
from django.core.cache import cache
from django.utils.importlib import import_module


import datetime
import logging
logger = logging.getLogger(__name__)


@task(queue="periodic")
def cleanup():
    engine = import_module(settings.SESSION_ENGINE)
    SessionStore = engine.SessionStore

    expired_sessions = Session.objects.filter(expire_date__lte=datetime.datetime.now())

    logger.info('cleaning up %s expired sessions' % len(expired_sessions))
    for session in expired_sessions:
        store = SessionStore(session.session_key)
        store.delete()
