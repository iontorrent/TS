#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import djangoinit
from django.contrib.auth.models import User
import psycopg2
import traceback
from django.db import connection
from django.core import management

def superUserExists():
    '''Tests database for existence of a superuser'''
    try:
        users = User.objects.all()
        if users:
            for user in users:
                if user.is_superuser:
                    return True
    except:
        # Typically get here if iondb database or User object does not exist
        return False
    
    return False

def createSuperUser():
    '''Creates initial superuser by calling ./manage.py syncdb --noinput
    If there is a file initial_data.json, it will be loaded.  If that file
    contains superuser element, that will create a superuser.
    NOTE: If superuser of the same name exists, it will be overwritten.'''
    connection.close()
    os.chdir('/opt/ion/iondb') ## FIXME - should be set outside script
    management.call_command("syncdb", interactive=False)

    management.call_command("migrate", app="tastypie") # Adding superuser requires tastypie_apikey
    management.call_command("migrate", app="rundb", target="0001") # Adding superuser requires rundb_userprofile
    management.call_command("loaddata", "ionadmin_superuser.json", raw=True)

    # Many rundb migrations require superuser...
    management.call_command("migrate")

if __name__=="__main__":
    if superUserExists():
        print "Super user exists."
        sys.exit(0)
    else:
        print "We need a superuser"
        createSuperUser()
        if superUserExists():
            print "Super user created"
            sys.exit(0)
        else:
            print "Failed to create superuser!"
            sys.exit(0)

