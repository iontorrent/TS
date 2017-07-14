# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.celery import app
from django.conf import settings
from django.contrib.sessions.models import Session
from django.core.cache import cache
from django.utils.importlib import import_module
from iondb.rundb.models import PlanSession

import datetime
import logging
logger = logging.getLogger(__name__)


@app.task(queue="periodic", ignore_result = True)
def cleanup():
    engine = import_module(settings.SESSION_ENGINE)
    SessionStore = engine.SessionStore

    expired_sessions = Session.objects.filter(expire_date__lte=datetime.datetime.now())

    logger.info('cleaning up %s expired sessions' % len(expired_sessions))
    for session in expired_sessions:
        store = SessionStore(session.session_key)
        store.delete()

    expired_plan_sessions = PlanSession.objects.filter(expire_date__lte=datetime.datetime.now())
    logger.info('cleaning up %s expired plan sessions' % expired_plan_sessions.count())
    expired_plan_sessions.delete()