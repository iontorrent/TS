#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

# you need to call this script with sudo

import os
os.environ['DJANGO_SETTINGS_MODULE']="iondb.settings"

from iondb.rundb.models import *

# disable autorun for all plugins
Plugin.objects.all().update(autorun=False)

#only enable certain plugins
myplugin=Plugin.objects.get(name="1_swtest",active = True)
myplugin.autorun = True
myplugin.selected = True
myplugin.save()

