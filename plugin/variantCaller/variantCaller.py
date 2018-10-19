#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import traceback
import subprocess
from ion.plugin import *

class variantCaller(IonPlugin):
    version = '5.10.1.19'
    envDict = dict(os.environ)
    runtypes = [RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE]
    requires_configuration = True # The user can not run the plugin w/o clicking the configuration button (exception: TS-16890).
    __doc__ = 'Torrent Variant Caller.\nPlease get more information by visiting \"http://tools.thermofisher.com/content/sfs/manuals/MAN0014668_Torrent_Suite_RUO_Help.pdf\"'

    def variantCall(self):
        # With only one line, this one is easy to convert.
        arg_list = [os.path.join(self.envDict['DIRNAME'], 'variant_caller_plugin.py'), '--install-dir', self.envDict['DIRNAME'], '--output-dir', self.envDict['TSP_FILEPATH_PLUGIN_DIR'], '--output-url', self.envDict['TSP_URLPATH_PLUGIN_DIR'], '--report-dir', self.envDict['ANALYSIS_DIR']]
        command_line = ' '.join(arg_list)
        print command_line
        return subprocess.call(command_line, shell=True)

    def launch(self, data=None):
        # Run the plugin.
        print 'running the python plugin.'
        exit_code = self.variantCall()
        sys.exit(exit_code)

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
            # TS-16890
            builtin_config = configuration.get('meta', {}).get('configuration', None)
            if (builtin_config is not None) and configuration == {'meta': {'configuration': builtin_config}}:
                return errors
            # Check the existence of the keys.
            for key in ['meta', 'torrent_variant_caller', 'long_indel_assembler', 'freebayes']:
                if key not in configuration:
                    errors.append('The key "%s" is missing in the plugin configuration.' %key)
            if errors:
                return errors
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
