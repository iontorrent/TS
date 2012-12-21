#!/usr/bin/env python
import os, sys

ionpath = '/opt/ion'
if ionpath not in sys.path:
    sys.path.append(ionpath)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iondb.settings")
os.environ["CELERY_LOADER"] = "django"

# This application object is used by any WSGI server configured to use this
# file. This includes Django's development server, if the WSGI_APPLICATION
# setting points here.
from django.core.wsgi import get_wsgi_application
_application = get_wsgi_application()

def application(environ, start_response):
    """ Relocate app to serve just subdirectories - rundb and admin, and anything WSGIScriptAliased here.
        Converts: '/rundb' and '/reports/' to '/' and '/rundb/reports/',
        so django has full path to match urls.py
    """
    environ['PATH_INFO'] = environ['SCRIPT_NAME'] + environ['PATH_INFO']
    environ['SCRIPT_NAME'] = '' ## NB: Without this, got /rundb/rundb/.../ expansions
    return _application(environ, start_response)

##======================================================================
# Authentication component embedded here. Uses django Users database

## NOTE: Django 1.5 replace with django.contrib.auth.handlers.modwsgi
# from django.contrib.auth.handlers.modwsgi import check_password

from django.contrib.auth.models import User
from django import db

def check_password(environ, username, password):
    """
    Authenticates against Django's auth database
    """

    db.reset_queries()

    try:
        # verify the user exists
        try:
            user = User.objects.get(username=username, is_active=True)
        except User.DoesNotExist:
            return None

        # verify the password for the given user
        if user.check_password(password):
            return True
        else:
            return False
    finally:
        db.close_connection()

