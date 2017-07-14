#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import traceback
try:
    import json
except:
    import simplejson as json
from subprocess import *
from ion.plugin import *
from django.utils.datastructures import SortedDict

class variantCaller(IonPlugin):
    version = '5.4.0.46'
    envDict = dict(os.environ)

    def variantCall(self):
        # With only one line, this one is easy to convert.
        pluginRun = Popen(['%s/variant_caller_plugin.py'%self.envDict['DIRNAME'], '--install-dir', '%s'%self.envDict['DIRNAME'], '--output-dir', '%s'%self.envDict['TSP_FILEPATH_PLUGIN_DIR'], '--output-url', '%s'%self.envDict['TSP_URLPATH_PLUGIN_DIR'], '--report-dir', '%s'%self.envDict['ANALYSIS_DIR']], stdout=PIPE, env=self.envDict)
        print 'output: %s'%pluginRun.communicate()[0]

    def launch(self, data=None):
        # Run the plugin.
        print 'running the python plugin.'
        self.variantCall()

        # Exit gracefully.
        sys.exit(0)

    def output(self):
        pass

    def report(self):
        pass

    def metric(self):
        pass

if __name__ == "__main__":
    PluginCLI(variantCaller())
