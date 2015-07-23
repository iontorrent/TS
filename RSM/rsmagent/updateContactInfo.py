#!/usr/bin/python
# Copyright (C) 2013, 2014 Ion Torrent Systems, Inc. All Rights Reserved
"""Retrieve contact info from TS and write to text file for agent to read"""
import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

from iondb.rundb import models


def main():
    """Retrieve contact info from TS and write to text file for agent"""
    with open('/var/spool/ion/ContactInfo.txt', 'w') as file_in:

        desired_contacts = ["lab_contact", "it_contact"]
        for profile in models.UserProfile.objects.filter(user__username__in=
                                                         desired_contacts):
            values = [profile.title,
                      profile.name,
                      profile.user.email,
                      profile.phone_number]

            file_in.write("\t".join(values) + "\n")

if __name__ == "__main__":
    main()
