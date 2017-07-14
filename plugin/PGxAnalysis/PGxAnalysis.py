#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import traceback
from subprocess import *
from ion.plugin import *

class PGxAnalysis(IonPlugin):
  '''Run the PGX Analysis pipeline.'''
  version = '5.4.0.1'
  #major_block = True
  allow_autorun = False
  runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
  runlevels = [ RunLevel.LAST ]
  depends = ['coverageAnalysis', 'variantCaller']
  
  def launch(self, data=None):
    plugin = Popen([
        'python',
        '%s/PGxAnalysis_plugin.py' % os.environ['DIRNAME'], '-V', self.version,'-d', '-k',
        '-B', os.environ['TSP_FILEPATH_BAM'], '-P', os.environ['TSP_FILEPATH_OUTPUT_STEM'],
        '-R', os.environ['TSP_FILEPATH_GENOME_FASTA'], '-U', os.environ['TSP_URLPATH_PLUGIN_DIR'],
        '%s/startplugin.json' % os.environ['TSP_FILEPATH_PLUGIN_DIR']
      ], stdout=PIPE, shell=False )
    # re-direct STDOUT using plugin.communicate()[0]
    plugin.communicate()
    sys.exit(plugin.poll())
  
  def output(self):
    pass
  
  def report(self):
    pass
  
  def metrics(self):
    pass

if __name__ == "__main__":
  PluginCLI(PGxAnalysis())

