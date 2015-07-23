#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import traceback
from subprocess import *
from ion.plugin import *

class RNASeqAnalysis(IonPlugin):
  '''Run the RNASeq pipeline.'''
  version = "4.6.0.8"
  major_block = True
  allow_autorun = True
  runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
  runlevels = [ RunLevel.DEFAULT ]
  
  def launch(self, data=None):
    plugin = Popen([
        '%s/RNASeqAnalysis_plugin.py' % os.environ['DIRNAME'], '-V', self.version, '-d',
        '-B', os.environ['TSP_FILEPATH_UNMAPPED_BAM'], '-P', os.environ['TSP_FILEPATH_OUTPUT_STEM'],
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
  PluginCLI()

