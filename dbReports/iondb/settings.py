# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# Django settings for iondb project.

from os import path
import socket
from bin import dj_config
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

# Django Celery config
CELERYD_LOG_FILE = "/var/log/ion/celery.log"
BROKER_HOST = "localhost"
BROKER_PORT = 5672
BROKER_USER = "ion"
BROKER_PASSWORD = "ionadmin"
BROKER_VHOST = "ion"

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = None

# When support for time zones is enabled, Django stores date and time
# information in UTC in the database, uses time-zone-aware datetime
# objects internally, and translates them to the end user's time zone
# in templates and forms.
USE_TZ = True

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
#ADMIN_MEDIA_PREFIX = '/media/'
STATIC_URL = '/media/'
STATIC_ROOT = '/var/www/media'

# Make this unique, and don't share it with anybody.
SECRET_KEY = 'mlnpl3nkj5(iu!517y%pr=gbcyi=d$^la)px-u_&i#u8hn0o@$'

# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    ('django.template.loaders.cached.Loader', (
        'django.template.loaders.filesystem.Loader',
        'django.template.loaders.app_directories.Loader',
    )),
)

MIDDLEWARE_CLASSES = (
    'iondb.rundb.middleware.ChangeRequestMethodMiddleware',
    'django.middleware.common.CommonMiddleware',
    'iondb.rundb.middleware.DeleteSessionOnLogoutMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.auth.middleware.RemoteUserMiddleware',
    'iondb.rundb.middleware.LocalhostAuthMiddleware',
    'iondb.bin.startup_housekeeping.StartupHousekeeping'
)

ROOT_URLCONF = 'iondb.urls'

TEMPLATE_DIRS = ((TEST_INSTALL and path.join(LOCALPATH, "templates")) or
                 "/opt/ion/iondb/templates",
                 "/usr/share/pyshared/django/contrib/admindocs/templates/",
                 "/results/publishers/",
)

TEMPLATE_CONTEXT_PROCESSORS = (
    'django.contrib.auth.context_processors.auth',
    'django.core.context_processors.request',
    'django.core.context_processors.media',
    'django.core.context_processors.static',
    'django.contrib.messages.context_processors.messages',
    'iondb.rundb.views.base_context_processor',
    'iondb.rundb.views.message_binding_processor',
    'django.contrib.messages.context_processors.messages',
)

INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.admin',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.admindocs',
    'django.contrib.sites',
    'django.contrib.staticfiles',
    'django.contrib.humanize',
    'django.contrib.messages',
    'iondb.rundb',
    'tastypie',
    'djcelery',
    'south',
)

# This is not to be the full path to the module, just project.model_name
AUTH_PROFILE_MODULE = 'rundb.UserProfile'

# Allow internal or apache based authentication
AUTHENTICATION_BACKENDS = (
        'django.contrib.auth.backends.RemoteUserBackend',
        'django.contrib.auth.backends.ModelBackend',
)

LOGIN_URL="/login/"
LOGIN_REDIRECT_URL="/data/"
# Whether to expire the session when the user closes his or her browser. 
# See "Browser-length sessions vs. persistent sessions", https://docs.djangoproject.com/en/dev/topics/http/sessions/#browser-length-sessions-vs-persistent-sessions
SESSION_EXPIRE_AT_BROWSER_CLOSE=True

CELERY_IMPORTS = (
    "iondb.rundb.tasks",
    "iondb.rundb.publishers",
    "iondb.plugins.tasks",
)

# Allow tasks the generous run-time of six hours before they're killed.
CELERYD_TASK_TIME_LIMIT=21600
# Plugin tasks drop privileges to ionian, so need to start new worker each time
CELERYD_MAX_TASKS_PER_CHILD=1

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
        #'mail_admins': {
        #    'level': 'ERROR',
        #         'filters': ['require_debug_false'],
        #         'class': 'django.utils.log.AdminEmailHandler'
        #}
    },
    'filters': {
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse'
        }
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
EXTERNAL_IP_URL = "http://icanhazip.com"

CRAWLER_PORT = 10001
CRAWLER_PERIOD = 60

ANALYSIS_ROOT = "/opt/ion/iondb/anaserve"

JOBSERVER_HOST = HOSTNAME
JOBSERVER_PORT = 10000
IARCHIVE_PORT = 10002
DM_LOGGER_PORT = 10003
IPLUGIN_HOST = HOSTNAME
IPLUGIN_PORT = 9191

# SGE settings - all you need to run SGE
SGE_ROOT = "/var/lib/gridengine"
SGE_CELL = "iontorrent"
SGE_CLUSTER_NAME = "p6444"
SGE_QMASTER_PORT = 6444
SGE_EXECD_PORT = 6445
SGE_ENABLED = True
DRMAA_LIBRARY_PATH = "/usr/lib/libdrmaa.so.1.0"

TMAP_VERSION = dj_config.get_tmap_version()
TMAP_DIR = '/results/referenceLibrary/%s/' % TMAP_VERSION
TEMP_PATH = "/results/referenceLibrary/temp/"
PLUGIN_PATH = "/results/plugins/"

FILE_UPLOAD_MAX_MEMORY_SIZE = 114857600

# Define Plugin Warehouse URLs as tuples of (Name, User URL, API URL)
PLUGIN_WAREHOUSE = [
    (
     'Torrent Browser Plugin Store',
     'http://lifetech-it.hosted.jivesoftware.com/community/products/torrent_browser_plugin_store',
     'http://torrentcircuit.iontorrent.com/warehouse/'
    )
]

try:
    # this file is generated and placed into /opt/ion/iondb/ion_dbreports_version.py by the CMake process and .deb pkg installation
    import version #@UnresolvedImport
    SVNREVISION = version.IonVersionGetSvnRev()
    VERSION = 'v' + '.'.join([version.IonVersionGetMajor(), version.IonVersionGetMinor(), version.IonVersionGetRelease(), version.IonVersionGetSvnRev()]) 
except:
    SVNREVISION=''
    VERSION=''
    pass

# import from the local settings file
try:
    from local_settings import *
except ImportError:
    pass
