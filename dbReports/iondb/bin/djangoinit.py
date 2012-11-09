# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
#import mimeparse
#import tastypie

# put two directories up on the path to allow "from iondb." imports
if os.path.dirname(__file__) == '':
    sys.path.insert(0,os.path.abspath('../../'))
else:
    _dn = os.path.dirname
    sys.path.insert(0,_dn(_dn(_dn(__file__))))
    sys.path.insert(0,_dn(__file__))
# set the django settings module in order to import from django
os.environ['DJANGO_SETTINGS_MODULE'] = "iondb.settings"

# Settings is emotionally complicated, when you import it, nothing happens, but
# when you either read or write any of it's variables, it will configure and
# load Django's logger as defined in settings.
from django.conf import global_settings
# Settings inherits the properties of global_settings which is a much simpler
# beast.  By settings LOGGING_CONFIG to None here, settings will not act on
# it's logging configuration when we read other stuff from it.
global_settings.LOGGING_CONFIG=None

from django.conf import settings

sys.path.append(os.path.dirname(settings.LOCALPATH))
