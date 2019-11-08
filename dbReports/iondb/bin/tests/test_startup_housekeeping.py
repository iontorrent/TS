# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
from django.test import TestCase
from django.core.exceptions import MiddlewareNotUsed
from django.test.utils import override_settings
from mockito import mock, when, verify, contains, any
from iondb.rundb.models import Message, UserProfile
from django.contrib.auth.models import User
from django.conf import settings

import logging

logger = logging.getLogger(__name__)


class StartupHousekeepingWithFixturesTest(TestCase):
    fixtures = [
        "iondb/rundb/tests/models/fixtures/groups.json",
        "iondb/rundb/tests/models/fixtures/users.json",
    ]

    def setUp(self):
        self.lab_contact = User.objects.get(username="lab_contact")
        self.it_contact = User.objects.get(username="it_contact")

    def test_bother_the_user(self):
        #        UserProfile.objects.create(user=self.lab_contact, name="Lab Contact")
        #        UserProfile.objects.create(user=self.it_contact, name="IT Contact")
        logger.debug(
            "Found %s UserProfile.objects.count()" % UserProfile.objects.count()
        )

        self.assertEqual(
            0, Message.objects.count(), "Messages should not exist before call"
        )
        from iondb.bin.startup_housekeeping import bother_the_user

        bother_the_user()
        self.assertEqual(
            0, Message.objects.count(), "Messages should not exist after call"
        )


EMPTY = {}

SINGLE = {("Result", "save"): ["http://localhost/rundb/demo_consumer/result_save"]}


class StartupHousekeepingTest(TestCase):
    @override_settings(EVENTAPI_CONSUMERS=EMPTY)
    def test_init(self):
        from iondb.bin.startup_housekeeping import StartupHousekeeping

        self.assertRaisesMessage(
            MiddlewareNotUsed, "End of startup.", callable_obj=StartupHousekeeping
        )

    @override_settings(EVENTAPI_CONSUMERS=SINGLE)
    def test_events_from_settings_some(self):
        """TODO: determine what to test"""
        pass
        """
        Skipping to avoid registering events that cause other tests to bomb
        ======================================================================
        ERROR: test_auto_exempt (iondb.rundb.tests.views.report.test_report_action.ReportActionTest)
        ----------------------------------------------------------------------
        Traceback (most recent call last):
          File "/home/ionadmin/TS_/dbReports/iondb/rundb/tests/views/report/test_report_action.py", line 102, in test_auto_exempt
            result.save()
          File "/usr/lib/pymodules/python2.6/django/db/models/base.py", line 463, in save
            self.save_base(using=using, force_insert=force_insert, force_update=force_update)
          File "/usr/lib/pymodules/python2.6/django/db/models/base.py", line 565, in save_base
            created=(not record_exists), raw=raw, using=using)
          File "/usr/lib/pymodules/python2.6/django/dispatch/dispatcher.py", line 172, in send
            response = receiver(signal=self, sender=sender, **named)
          File "/home/ionadmin/TS_/dbReports/iondb/rundb/events.py", line 64, in handler
            model_event_post(domain, name, sender, instance, producer, consumer)
          File "/home/ionadmin/TS_/dbReports/iondb/rundb/events.py", line 44, in model_event_post
            data = producer(sender_model, instance)
          File "/home/ionadmin/TS_/dbReports/iondb/rundb/events.py", line 96, in producer
            dry = resource.full_dehydrate(bundle)
          File "/usr/lib/pymodules/python2.6/tastypie/resources.py", line 715, in full_dehydrate
            bundle.data[field_name] = field_object.dehydrate(bundle)
          File "/usr/lib/pymodules/python2.6/tastypie/fields.py", line 634, in dehydrate
            raise ApiFieldError("The model '%r' has an empty attribute '%s' and doesn't allow a null value." % (previous_obj, attr))
        ApiFieldError: The model '<Results: >' has an empty attribute 'reportstorage' and doesn't allow a null value.
        -------------------- >> begin captured logging << --------------------
        iondb.rundb.models: WARNING: Problem with /opt/ion/iondb/templates/rundb/php_base.html: [Errno 13] Permission denied: '/opt/ion/iondb/templates/rundb/php_base.html'
        --------------------- >> end captured logging << ---------------------
        
        """

    #        from iondb.bin.startup_housekeeping import events, events_from_settings
    #        when(events).register_events(any(wanted_type=dict()))
    #        events_from_settings()
    #        # TODO: determine how to verify method invoked
    #        verify(events).register_events(any(wanted_type=dict()))

    @override_settings(EVENTAPI_CONSUMERS=EMPTY)
    def test_events_from_settings_empty(self):
        """TODO: determine what to test"""
        from iondb.bin.startup_housekeeping import events, events_from_settings

        when(events).register_events(any(wanted_type=dict()))
        events_from_settings()
        verify(events).register_events(any(wanted_type=dict()))

    def test_expire_messages(self):
        msg = Message.objects.create(expires="startup")
        from iondb.bin.startup_housekeeping import expire_messages

        self.assertEqual(1, Message.objects.count(), "Message not created")
        expire_messages()
        self.assertEqual(0, Message.objects.count(), "Message not removed")

    def test_bother_the_user(self):
        self.assertEqual(
            0, Message.objects.count(), "Messages should not exist before call"
        )
        from iondb.bin.startup_housekeeping import bother_the_user

        bother_the_user()
        self.assertEqual(
            1, Message.objects.count(), "Messages should not exist after call"
        )
        msg = Message.objects.latest("time")
        self.assertEqual(
            """Please supply some customer support contact info.
<br/><a href="/rundb/config">Add contact information.</a>""",
            msg.body,
            "expecting different Message.body",
        )
