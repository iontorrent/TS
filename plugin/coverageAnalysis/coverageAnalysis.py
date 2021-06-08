#!/usr/bin/python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
from subprocess import *
from ion.plugin import *

class coverageAnalysis(IonPlugin):
  '''Genome and Targeted Re-sequencing Coverage Analysis. (Ion supported)'''
  version = '5.16.0.4'
  major_block = True
  runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
  runlevels = [ RunLevel.DEFAULT ]

  def launch(self):
    isDx = os.environ.get("CHIP_LEVEL_ANALYSIS_PATH", '')
    plugin = Popen([
        '%s/coverageAnalysis_plugin.py' % os.environ['DIRNAME'], '-V', self.version, '-i', isDx, '-d',
        'startplugin.json', 'barcodes.json'
      ], stdout=PIPE, shell=False )
    plugin.communicate()
    sys.exit(plugin.poll())


if __name__ == "__main__":
  PluginCLI()

