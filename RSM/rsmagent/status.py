#!/usr/bin/python
# Copyright (C) 2013, 2014 Ion Torrent Systems, Inc. All Rights Reserved

from __future__ import print_function
import sys
import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

from iondb.rundb.configure.views import process_set

processes = process_set()

for process in processes:
	serviceName, serviceStatus = process
	if serviceStatus:
		statusString = 'Running'
	else:
		statusString = 'Down'

	print(serviceName, '|', statusString, file=sys.stderr)
	
