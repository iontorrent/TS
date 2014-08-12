#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
import os, sys
#import site
#from distutils.sysconfig import get_python_version

ionpath = '/opt/ion'
if ionpath not in sys.path:
    sys.path.append(ionpath)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iondb.settings")
os.environ["CELERY_LOADER"] = "django"

## Preload Django.
# -- avoids delay for lazy loading at each thread restart
import django.core.management
utility = django.core.management.ManagementUtility()
command = utility.fetch_command('runserver')
command.validate()

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

from django.contrib.auth.handlers.modwsgi import check_password

