#!/usr/bin/python
# Copyright (C) 2013, 2014 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

from iondb.rundb import models

with open('/var/spool/ion/ContactInfo.txt', 'w') as f:

    desired_contacts = ["lab_contact", "it_contact"]
    for profile in models.UserProfile.objects.filter(user__username__in=desired_contacts):
        values = [
                  profile.title, 
                  profile.name,
                  profile.user.email, 
                  profile.phone_number
                 ]
        f.write("\t".join(values) + "\n")

