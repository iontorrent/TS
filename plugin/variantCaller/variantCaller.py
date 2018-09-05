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
    version = '5.8.0.21'
    envDict = dict(os.environ)
    runtypes = [RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE]
    requires_configuration = True # The user can not run the plugin w/o clicking the configuration button.

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
    
    def custom_validation(self, configuration, run_mode):
        """
        run_mode takes values from "manual" or "Automatic"
        """
        errors = []
        if run_mode.lower() != 'manual' and configuration: # Empty configuration is handled by requires_configuration
            for key in ['meta', 'torrent_variant_caller', 'long_indel_assembler', 'freebayes']:
                if key not in configuration:
                    errors.append('The key "%s" is missing in the plugin configuration.' %key)
            # Note that the parameters weren't be checked when the user clicked "Save Changes" button when configuring the plugin in the plan or template.
            # Call valid parameters just like I manually start the plugin through the browser.
            file_dir = os.path.dirname(__file__) # self.envDict['DIRNAME'] doesn't work in the plan
            if file_dir not in sys.path:
                sys.path.append(file_dir)
            try:
                from extend import are_parameters_valid
                validate_results = are_parameters_valid({"request_get": {"base_set": configuration}})
                errors += validate_results['param_error_msg']
            except:
                pass
        return errors

if __name__ == "__main__":
    PluginCLI(variantCaller())
