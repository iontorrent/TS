#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django.contrib.auth.models import User
import psycopg2
import traceback
from django.db import connection

def writeInitialDataFile():
    '''Writes initial_data.json file with superuser info.
    Subsequent call to ./manage.py syncdb --noinput will load this file and
    result in the django.contrib.auth module being initialized.
    '''
    import json
    fp = open('initial_data.json', 'wb')
    json.dump([
  {
    "pk": 1, 
    "model": "auth.user", 
    "fields": {
      "username": "ionadmin", 
      "first_name": "", 
      "last_name": "", 
      "is_active": True, 
      "is_superuser": True, 
      "is_staff": True, 
      "last_login": "2011-05-05 15:45:15", 
      "groups": [], 
      "user_permissions": [], 
      "password": "sha1$d0981$fd85e583cfa0d586e3f93093f8bf6040b3ddc7dc", 
      "email": "ionadmin@iontorrent.com", 
      "date_joined": "2011-05-03 14:37:38"
    }
  }
],fp,indent=True)
    fp.close()
    return None
    
    
    
def superUserExists():
    '''Tests database for existence of a superuser'''
    try:
        users = User.objects.all()
        if users:
            for user in users:
                if user.is_superuser:
                    return True
    except psycopg2.DatabaseError:
        # Typically get here if iondb database does not exist
        #print traceback.format_exc()
        return False
    except:
        print traceback.format_exc()
        return False

    return False

def createSuperUser():
    '''Creates initial superuser by calling ./manage.py syncdb --noinput
    If there is a file initial_data.json, it will be loaded.  If that file
    contains superuser element, that will create a superuser.
    NOTE: If superuser of the same name exists, it will be overwritten.'''
    connection.close()
    os.chdir('/opt/ion/iondb')
    writeInitialDataFile()
    os.system('python ./manage.py syncdb --noinput')
    os.system('rm initial_data.json')
    # The following will only work once the auth system is initialized
    #try:
    #    print "here"
    #    user = User(username='ionadmin',email='ionadmin@iontorrent.com')
    #    print "there"
    #    user.set_password('ionadmin')
    #    user.is_staff = True
    #    user.is_superuser = True
    #    user.save()
    #    return user
    #except psycopg2.DatabaseError:
    #    print traceback.format_exc()
    #    return None

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
            sys.exit(1)

