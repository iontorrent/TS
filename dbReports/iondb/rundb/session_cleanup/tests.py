# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.conf import settings
from django.core.cache import cache
from django.test import TestCase
from django.utils.importlib import import_module
from iondb.rundb.session_cleanup.tasks import cleanup


import datetime


class CleanupTest(TestCase):
    def test_session_cleanup(self):
        """
        Tests that sessions are deleted by the task
        """
        engine = import_module(settings.SESSION_ENGINE)
        SessionStore = engine.SessionStore

        now = datetime.datetime.now()
        last_week = now - datetime.timedelta(days=7)
        stores = []
        unexpired_stores = []
        expired_stores = []

        # create unexpired sessions
        for i in range(20):
            store = SessionStore()
            store.save()
            stores.append(store)

        for store in stores:
            self.assertEquals(store.exists(store.session_key), True, 'Session store could not be created')

        unexpired_stores = stores[:10]
        expired_stores = stores[10:]

        # expire some sessions
        for store in expired_stores:
            store.set_expiry(last_week)
            store.save()

        cleanup()

        for store in unexpired_stores:
            self.assertEquals(store.exists(store.session_key), True, 'Unexpired store was deleted by cleanup')

        for store in expired_stores:
            self.assertEquals(store.exists(store.session_key), False, 'Expired store was not deleted by cleanup')
