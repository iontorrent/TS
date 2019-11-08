#!/usr/bin/python
# Copyright (C) 2019 Thermo Fisher Scientific, Inc. All Rights Reserved.

import os
import sys
from subprocess import *
from ion.plugin import *

class molecularCoverageAnalysis(IonPlugin):
  '''Molecular Coverage Analysis. (Ion R&D)'''
  version = "5.12.0.23"
  major_block = True
  runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
  runlevels = [ RunLevel.DEFAULT ]
  requires_configuration = True
  __doc__ = 'This plugin generates statistics, downloadable data files and interactive visualization of molecular coverage over targeted regions of the reference genome.'

  def launch(self,data=None):
    plugin_cmd = [
        os.path.join(os.environ['DIRNAME'], 'molecularCoverageAnalysis_plugin.py'), 
        '-V', self.version, 
        '-d',
        #'-B', os.environ['TSP_FILEPATH_BAM'],
        '-P', os.environ['TSP_RUN_NAME'],
        #'-Q', os.environ['TSP_URLPATH_GENOME_FASTA'], 
        '-R', os.environ['TSP_FILEPATH_GENOME_FASTA'],
        #'-U', os.environ['TSP_URLPATH_PLUGIN_DIR'], 
        os.path.join(os.environ['TSP_FILEPATH_PLUGIN_DIR'], 'startplugin.json'),
        os.path.join(os.environ['TSP_FILEPATH_PLUGIN_DIR'], 'barcodes.json')
    ]
    print(' '.join(plugin_cmd))
    plugin = Popen(plugin_cmd, stdout=PIPE, shell=False )
    plugin.communicate()
    sys.exit(plugin.poll())
    
  def custom_validation(self, configuration, run_mode):
    errors = []
    if run_mode.lower() != 'manual' and configuration:
      builtin_config = configuration.get('meta', {}).get('configuration', None)
      if (builtin_config is not None) and configuration == {'meta': {'configuration': builtin_config}}:
        # It is TS builtin configuration
        return errors
    
      plan_lib_type = str(configuration.get('planLibType', ''))
      if (plan_lib_type not in ['', 'TAG_SEQUENCING']) and ('AMPS_HD' not in plan_lib_type):
        errors.append('The plugin does not support the library type "%s", since no molecular tag is used.' %plan_lib_type)
        errors.append('Please un-select the plugin.')
    return errors


if __name__ == "__main__":
  PluginCLI(molecularCoverageAnalysis())

