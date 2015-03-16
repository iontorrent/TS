#!/usr/bin/python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
from subprocess import *
from ion.plugin import *

class coverageAnalysis(IonPlugin):
  '''Genome and Targeted Re-sequencing Coverage Analysis. (Ion supprted)'''
  version = "4.4.0.20"
  major_block = True
  runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
  runlevels = [ RunLevel.DEFAULT ]

  def launch(self,data=None):
    plugin = Popen([
        '%s/coverageAnalysis_plugin.py' % os.environ['DIRNAME'], '-V', self.version, '-d',
        '-B', os.environ['TSP_FILEPATH_BAM'], '-P', os.environ['TSP_FILEPATH_OUTPUT_STEM'],
        '-Q', os.environ['TSP_URLPATH_GENOME_FASTA'], '-R', os.environ['TSP_FILEPATH_GENOME_FASTA'],
        '-U', os.environ['TSP_URLPATH_PLUGIN_DIR'], '%s/startplugin.json' % os.environ['TSP_FILEPATH_PLUGIN_DIR']
      ], stdout=PIPE, shell=False )
    plugin.communicate()
    sys.exit(plugin.poll())


if __name__ == "__main__":
  PluginCLI()

