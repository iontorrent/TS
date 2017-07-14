# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


import django.core.exceptions
from iondb.rundb import models
from iondb.rundb import events
from django.conf import settings
import logging


logger = logging.getLogger(__name__)


def expire_messages():
    """Delete messages with an expiration value of "startup".  This is
    a rare class of messages which should hang around from some time
    until the server is restarted and never be seen again.
    """
    expired = models.Message.objects.filter(expires="startup")
    if len(expired):
        logger.info("Deleting %d messages on startup" % len(expired))
    expired.delete()


def bother_the_user():
    """This aggregates everything that bothers the user about stuff
    """
    desired_contacts = ["lab_contact", "it_contact"]
    info = []
    for profile in models.UserProfile.objects.filter(user__username__in=desired_contacts):
        info.extend([profile.phone_number, profile.user.email])
    if not any(info):
        models.Message.info("""Please supply some customer support contact info.
<br/><a href="/configure/configure">Add contact information.</a>""")


def events_from_settings():
    try:
        event_consumers = getattr(settings, 'EVENTAPI_CONSUMERS', {})
        events.register_events(event_consumers)
    except Exception as err:
        logger.exception("Error during event consumer registration")


class StartupHousekeeping(object):

    def __init__(self):
        """This class is instatiated when the Django process is started, once
        and only once.  The MiddlewareNotUsed exception at the end removes this
        'middleware' before any actual requests would go through it.
        """
        expire_messages()
        bother_the_user()
        events_from_settings()

        raise django.core.exceptions.MiddlewareNotUsed("End of startup.")
