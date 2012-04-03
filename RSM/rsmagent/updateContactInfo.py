#!/usr/bin/python

from django.core.management import setup_environ

import sys
import os
import datetime

from os import path
sys.path.append('/opt/ion/')
sys.path.append("/opt/ion/iondb/")
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

import settings
setup_environ(settings)

from django.db import models
from iondb.rundb import models
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist

import re



if __name__=="__main__":

    # save results in same directory as where this scrips resides - better way?
    f = open( os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),'ContactInfo.txt'), 'w' )

    user_list = ["lab_contact", "it_contact"]
    for user in User.objects.filter(username__in=user_list):
        try:
            profile = user.get_profile()
            values = [
                      profile.title, 
                      profile.name,
                      user.email, 
                      profile.phone_number
                     ]
            f.write("\t".join(values) + "\n")
        except ObjectDoesNotExist:
            continue
        
