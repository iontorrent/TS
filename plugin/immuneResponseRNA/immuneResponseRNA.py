#!/usr/bin/python
# Copyright (C) 2016 Thermo Fisher Scientific. All Rights Reserved

import os
import sys
from subprocess import *
from ion.plugin import *

class immuneResponseRNA(IonPlugin):
  '''Whole Transciptome AmpliSeq-RNA Analysis. (Ion supprted)'''
  version = '5.4.0.0'
  major_block = True
  runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
  runlevels = [ RunLevel.DEFAULT ]

  def launch(self,data=None):
    plugin = Popen([
        '%s/immuneResponseRNA_plugin.py' % os.environ['DIRNAME'], '-V', self.version, '-d', '-s',
        '-B', os.environ['TSP_FILEPATH_BAM'], '-P', os.environ['TSP_ANALYSIS_NAME'],
        '-R', os.environ['TSP_FILEPATH_GENOME_FASTA'], '-U', os.environ['TSP_URLPATH_PLUGIN_DIR'],
        '%s/startplugin.json' % os.environ['TSP_FILEPATH_PLUGIN_DIR'],
        '%s/barcodes.json' % os.environ['TSP_FILEPATH_PLUGIN_DIR']
      ], stdout=PIPE, shell=False )
    plugin.communicate()
    sys.exit(plugin.poll())


if __name__ == "__main__":
  PluginCLI()

