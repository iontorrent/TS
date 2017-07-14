# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
# Django settings for iondb project.

from os import path
import socket
import json
from iondb.bin import dj_config
from django.core import urlresolvers
import subprocess
import sys

HOSTNAME = socket.gethostname()
TEST_INSTALL = False
LOCALPATH = path.abspath(path.dirname(__file__))
AUTO_START = False

# root directory on the file system of the running instance of Torrent Suite
TS_ROOT = path.realpath(path.join(__file__, '..', '..'))

DEBUG = False
JS_EXTRA = False
TEMPLATE_DEBUG = DEBUG

# This is the URL for the AmpliSeq website
AMPLISEQ_URL = "https://ampliseq.com/"
NEWS_FEED = "http://updates.iontorrent.com/news/ion_news_feed.xml"

# Set ADMINS in local_settings
ADMINS = ()
MANAGERS = ADMINS

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

if 'test' in sys.argv:
    DATABASES['default'] = {'ENGINE': 'django.db.backends.sqlite3'}

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
# If you set this to False, Django will not use timezone-aware datetimes.
USE_TZ = True

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

SITE_ID = 1

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = False

# If you set this to False, Django will not format dates, numbers and
# calendars according to the current locale.
USE_L10N = False

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/home/media/media.lawrence.com/media/"
MEDIA_ROOT = ''

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash.
# Examples: "http://media.lawrence.com/media/", "http://example.com/media/"
MEDIA_URL = ''

# Absolute path to the directory static files should be collected to.
# Don't put anything in this directory yourself; store your static files
# in apps' "static/" subdirectories and in STATICFILES_DIRS.
# Example: "/home/media/media.lawrence.com/static/"
STATIC_ROOT = '/var/www/site_media'

# URL prefix for static files.
# Example: "http://media.lawrence.com/static/"
STATIC_URL = '/site_media/'

# Additional locations of static files
STATICFILES_DIRS = (
    path.join(LOCALPATH, "media"),
)

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = (
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
    #    'django.contrib.staticfiles.finders.DefaultStorageFinder',
)
# Generate static paths with cache busting md5 names
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.CachedStaticFilesStorage'

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
    #'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'iondb.rundb.login.middleware.BasicAuthMiddleware',
    'iondb.rundb.middleware.LocalhostAuthMiddleware',
    'django.contrib.auth.middleware.RemoteUserMiddleware',
    'iondb.bin.startup_housekeeping.StartupHousekeeping'
)

ROOT_URLCONF = 'iondb.urls'
WSGI_APPLICATION = 'iondb.wsgi.application'

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
    'iondb.rundb.context_processors.base_context_processor',
    'iondb.rundb.context_processors.message_binding_processor',
    'iondb.rundb.context_processors.add_help_urls_processor',
)

INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.admin',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.admindocs',
    'django.contrib.sites',
    'django.contrib.staticfiles',
    'django.contrib.humanize',
    'django.contrib.messages',
    'iondb.ftpserver',
    'iondb.rundb',
    'iondb.security',
    'tastypie',
    'south',
)

# This is not to be the full path to the module, just project.model_name
# AUTH_PROFILE_MODULE = 'rundb.UserProfile' ## deprecated in 1.5

# Allow internal or apache based authentication
AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.RemoteUserBackend',
    'django.contrib.auth.backends.ModelBackend',
)

IONAUTH_ALLOW_REST_GET = False

# Only allow json in API. Disable xml, csv, plist, html.
TASTYPIE_DEFAULT_FORMATS = ['json', 'jsonp']

LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/data/"
# Whether to expire the session when the user closes his or her browser.
# See "Browser-length sessions vs. persistent sessions", https://docs.djangoproject.com/en/dev/topics/http/sessions/#browser-length-sessions-vs-persistent-sessions
SESSION_EXPIRE_AT_BROWSER_CLOSE = True

# Plans use objects not compatible with django 1.6 default json serializer
SESSION_SERIALIZER = "django.contrib.sessions.serializers.PickleSerializer"

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
        'short': {
            'format': '[%(asctime)s] [%(levelname)s] %(message)s'
        },
        'uniqueid': {
            'format': '[%(asctime)s] [%(levelname)s] [%(logid)s] %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'class': 'logging.handlers.WatchedFileHandler',
            'filename': '/var/log/ion/django.log',
            'formatter': 'standard',
        },
        'data_management': {
            'class': 'logging.handlers.WatchedFileHandler',
            'filename': '/var/log/ion/data_management.log',
            'formatter': 'uniqueid',
        },
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
            'level': 'INFO',  # python default is WARN for root logger
            'propagate': True
        },
        'iondb': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'ion': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        # When DEBUG is True, django.db will log every SQL query.  That is too
        # much stuff that we don't normally need, so it's logged elsewhere.
        'django.db': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
        'data_management': {
            'handlers': ['data_management'],
            'level': 'INFO',
            'propagate': False,
            'disable_existing_loggers': False
        },
    }
}

# handle email
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True

# set the root URL, for setting URLs in email
ROOT_URL = "http://ionupdates.com/"
EXTERNAL_IP_URL = "http://icanhazip.com"

CRAWLER_PORT = 10001
CRAWLER_PERIOD = 60

ANALYSIS_ROOT = "/opt/ion/iondb/anaserve"

JOBSERVER_HOST = HOSTNAME
JOBSERVER_PORT = 10000

# the settings for the xmlrpc server connection to the plugins daemon
IPLUGIN_HOST = HOSTNAME
IPLUGIN_PORT = 9191
IPLUGIN_STR = "http://%s:%s" % (IPLUGIN_HOST, IPLUGIN_PORT)

DEFAULT_NO_PROXY = "localhost,127.0.0.1,127.0.1.1,::1"

# SGE settings - all you need to run SGE
SGE_ROOT = "/var/lib/gridengine"
SGE_CELL = "iontorrent"
SGE_CLUSTER_NAME = "p6444"
SGE_QMASTER_PORT = 6444
SGE_EXECD_PORT = 6445
SGE_ENABLED = True
DRMAA_LIBRARY_PATH = "/usr/lib/libdrmaa.so"

try:
    TMAP_VERSION = dj_config.get_tmap_version()
except:
    TMAP_VERSION = 'tmap-f3'
TMAP_DIR = '/results/referenceLibrary/%s/' % TMAP_VERSION
TMAP_DISABLED_DIR = '/results/referenceLibrary/disabled/%s/' % TMAP_VERSION
TEMP_PATH = "/results/referenceLibrary/temp/"
PLUGIN_PATH = "/results/plugins/"

FILE_UPLOAD_MAX_MEMORY_SIZE = 114857600

# This is defined here for sharing and debugging purpuses only
# Remove before launch
# Identify a Model and an event, which can be 'create', 'save', 'delete'
EVENTAPI_CONSUMERS = {}
# EVENTAPI_CONSUMERS =  {
#    ('Result', 'save'): [
#        'http://localhost/rundb/demo_consumer/result_save',
#     ],
#    ('Experiment', 'create'): [
#        'http://localhost/rundb/demo_consumer/generic'
#     ],
#    ('Plan', 'delete'): [
#        'http://localhost/rundb/demo_consumer/plan_delete'
#     ],
#}

# The AWS settings enable and configure Amazon S3 upload
# Override them in local_settings.py
AWS_ACCESS_KEY = None
AWS_SECRET_KEY = None
AWS_BUCKET_NAME = None

SUPPORT_AUTH_URL = "https://support.iontorrent.com/asdf_authenticate"
SUPPORT_UPLOAD_URL = "https://support.iontorrent.com/asdf_upload"

REFERENCE_LIST_URL = "http://ionupdates.com/reference_downloads/references_list.json"

PRODUCT_UPDATE_BASEURL = "http://ionupdates.com/"
PRODUCT_UPDATE_PATH = "products/main.json"
EULA_TEXT_URL = "products/LICENSE.txt"

PLAN_CSV_VERSION = "2.0"
SUPPORTED_PLAN_CSV_VERSION = ["1.0","2.0"]
SAMPLE_CSV_VERSION = "1.0"
SUPPORTED_SAMPLE_CSV_VERSION = ["1.0"]

ABSOLUTE_URL_OVERRIDES = {
    'auth.user': lambda u: urlresolvers.reverse('configure_account'),
}

try:
    # this file is generated and placed into /opt/ion/iondb/ion_dbreports_version.py by the CMake process and .deb pkg installation
    import iondb.version as version  # @UnresolvedImport
    GITHASH = version.IonVersionGetGitHash()
    VERSION = 'v' + '.'.join([version.IonVersionGetMajor(), version.IonVersionGetMinor(), version.IonVersionGetRelease(), version.IonVersionGetGitHash()])
    RELVERSION = '.'.join([version.IonVersionGetMajor(), version.IonVersionGetMinor()])
except:
    GITHASH = ''
    VERSION = ''
    RELVERSION = ''
    pass

# TEST_RUNNER = "iondb.test_runner.IonTestSuiteRunner"

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.memcached.MemcachedCache',
        'LOCATION': '127.0.0.1:11211'
    },
    'file': {
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
        'LOCATION': '/var/spool/ion',
    }
}
NOSE_ARGS = ['--nocapture', '--nologcapture', ]

# Fetch uuid for this system. Used for TS mesh.
SYSTEM_UUID = None
try:
    SYSTEM_UUID = subprocess.check_output(['sudo', '-n', 'dmidecode', '-s', 'system-uuid']).strip()
except Exception as e:
    pass

# Load context sensitive help map. Installed by ion-docs package.
# JSON file contains an object with TS url patterns as keys and help system GUIDs as values.
HELP_URL_MAP = {}
try:
    with open("/var/www/ion-docs/help-url-map.json") as fp:
        HELP_URL_MAP = json.load(fp)
except Exception as e:
    pass

ALLOWED_HOSTS = ['*']

DEBUG_APPS = None
DEBUG_MIDDLE = None

# if the instance is a vm in a S5 we will need to masquerade address for the publicly facing IP address
FTPSERVER_MASQUERADE_ADDRESS = dj_config.get_s5_ip_addr() if dj_config.is_s5_tsvm() else None
FTPSERVER_HOST = "0.0.0.0"
FTPSERVER_PORT = 8021
FTPSERVER_DAEMONIZE = False
FTPSERVER_PASSIVE_PORTS = "20000-20100"

# OPTIONAL: this is the pattern to be used by the python-apt method "get_changelog" (https://apt.alioth.debian.org/python-apt-doc/library/apt.package.html)
# this should be where all of the changelogs are held for the plugin sets, should be setup in local_settings
# PLUGIN_CHANGELOG_URL = "file:///var/cache/apt/localrepo/changelogs/%(src_pkg)s_%(src_ver)s.changelog"

# Invoke this option in the local settings if you wish to process barcode information which would otherwise be filtered
# out from the barcodes.json file.  If not included this will default to false.
# PLUGINS_INCLUDE_FILTERED_BARCODES = True

# import from the local settings file
try:
    from iondb.local_settings import *
    # add debug apps if they are defined in local_settings.py
    if DEBUG_APPS:
        INSTALLED_APPS += DEBUG_APPS
    if DEBUG_MIDDLE:
        MIDDLEWARE_CLASSES += DEBUG_MIDDLE

except ImportError:
    pass

