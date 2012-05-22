# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


import django.core.exceptions
from iondb.rundb import models
import logging


logger = logging.getLogger(__name__)


def expire_messages():
    """Delete messages with an expiration value of "startup".  This is
    a rare class of messages which should hang around from some time
    until the server is restarted and never be seen again.
    """
    expired = models.Message.objects.filter(expires="startup")
    logger.debug("Deleting %d messages on startup" % len(expired))
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
<br/><a href="/rundb/config">Add contact information.</a>""")


class StartupHousekeeping(object):

    def __init__(self):
        """This class is instatiated when the Django process is started, once
        and only once.  The MiddlewareNotUsed exception at the end removes this
        'middleware' before any actual requests would go through it.
        """
        expire_messages()
        bother_the_user()

        raise django.core.exceptions.MiddlewareNotUsed("End of startup.")
