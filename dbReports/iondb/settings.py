# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# Django settings for iondb project.

from os import path
import sys
import socket
import djcelery
djcelery.setup_loader()

HOSTNAME = socket.gethostname()
TEST_INSTALL= False
LOCALPATH = path.dirname(__file__)
AUTO_START = False
DEBUG = False
TEMPLATE_DEBUG = DEBUG

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'iondb',
        'USER': 'ion',
        'PASSWORD': 'ion',
        'HOST': '',
        'PORT': ''
    }
}

QMASTERHOST = 'localhost'
SGEQUEUENAME = 'all.q'

# Django Celery config
BROKER_TRANSPORT = 'djkombu.transport.DatabaseTransport'
CELERYD_LOG_FILE = "/var/log/ion/celery.log"


# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = 'America/New_York'

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

SITE_ID = 1

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# Absolute path to the directory that holds media.
# Example: "/home/media/media.lawrence.com/"
MEDIA_ROOT = ((TEST_INSTALL and path.join(LOCALPATH, "media"))
              or '/opt/ion/iondb/media/')

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash if there is a path component (optional in other cases).
# Examples: "http://media.lawrence.com", "http://example.com/media/"
MEDIA_URL = ''

# URL prefix for admin media -- CSS, JavaScript and images. Make sure to use a
# trailing slash.
# Examples: "http://foo.com/media/", "/media/".
ADMIN_MEDIA_PREFIX = '/media/'

# Make this unique, and don't share it with anybody.
SECRET_KEY = 'mlnpl3nkj5(iu!517y%pr=gbcyi=d$^la)px-u_&i#u8hn0o@$'

# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.load_template_source',
)

MIDDLEWARE_CLASSES = (
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
)

ROOT_URLCONF = 'iondb.urls'

TEMPLATE_DIRS = ((TEST_INSTALL and path.join(LOCALPATH, "templates")) or
                 "/opt/ion/iondb/templates",
                 "/usr/share/pyshared/django/contrib/admindocs/templates/",
                 "/results/publishers/",
)

TEMPLATE_CONTEXT_PROCESSORS = ('django.contrib.auth.context_processors.auth',
                               'iondb.rundb.views.base_context_processor',
                                )

INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.admin',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.admindocs',
    'django.contrib.sites',
    'django.contrib.humanize',
    'iondb.rundb',
    'tastypie',
    'djkombu',
    'djcelery',
)

# This is not to be the full path to the module, just project.model_name
AUTH_PROFILE_MODULE = 'rundb.UserProfile'

CELERY_IMPORTS = (
    "iondb.rundb.tasks",
    "iondb.rundb.publishers",
)

# Allow tasks the generous run-time of one hour before they're killed.
CELERYD_TASK_TIME_LIMIT=3600

if path.exists("/opt/ion/.computenode"):
    # This is the standard way to disable logging in Django.
    # If you need to disable logging outside of this script, before you load
    # this settings script, do the following:
    #     from django.conf import global_settings
    #     global_settings.LOGGING_CONFIG = None
    # This will be inherited by this script, and logging will never be started
    LOGGING_CONFIG = None

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/ion/django.log',
            'maxBytes': 1024 * 1024 * 5, # 5 MB
            'backupCount': 5,
            'formatter': 'standard',
        },
    },
    'loggers': {
        # The logger name '' indicates the root of all loggers in Django, so
        # logs from any application in this project would go here.
        '': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True
        },
        # When DEBUG is True, django.db will log every SQL query.  That is too
        # much stuff that we don't normally need, so it's logged elsewhere.
        'django.db': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
    }
}

# handle email
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True

# set the root URL, for setting URLs in email
ROOT_URL = "http://updates.iontorrent.com/"

CRAWLER_PORT = 10001
CRAWLER_PERIOD = 60

ANALYSIS_ROOT = "/opt/ion/iondb/anaserve"

JOBSERVER_PORT = 10000
IARCHIVE_PORT = 10002
IPLUGIN_PORT = 9191

# SGE settings - all you need to run SGE
SGE_ROOT = "/var/lib/gridengine"
SGE_CELL = "iontorrent"
SGE_CLUSTER_NAME = "p6444"
SGE_QMASTER_PORT = 6444
SGE_EXECD_PORT = 6445
SGE_ENABLED = True
DRMAA_LIBRARY_PATH = "/usr/lib/libdrmaa.so.1.0"

ALTERNATIVE = False

TMAP_DIR = '/results/referenceLibrary/tmap-f2/'
TMAP_VERSION = 'tmap-f2'
TEMP_PATH = "/results/referenceLibrary/temp/"
PLUGIN_PATH = "/results/plugins/"

FILE_UPLOAD_MAX_MEMORY_SIZE = 114857600

sys.path.append('/etc')

# import from the local settings file
try:
    from local_settings import *
except ImportError:
    pass
    
# import from the cluster settings file
try:
    from torrentserver.cluster_settings import *
except ImportError:
    pass
