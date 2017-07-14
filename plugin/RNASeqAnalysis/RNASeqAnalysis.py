#!/usr/bin/env python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
from subprocess import *
from ion.plugin import *

class RNASeqAnalysis(IonPlugin):
  '''Run the RNASeq pipeline.'''
  version = '5.4.0.1'
  major_block = True
  allow_autorun = True
  runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
  runlevels = [ RunLevel.DEFAULT ]
  
  def launch(self):
    #self.call( 'RNASeqAnalysis_plugin.py', '-V', self.version, '-d', 'startplugin.json', 'barcodes.json' )
    plugin = Popen([
        '%s/RNASeqAnalysis_plugin.py' % os.environ['DIRNAME'], '-V', self.version, '-d',
        'startplugin.json', 'barcodes.json' ], stdout=PIPE, shell=False )
    plugin.communicate()
    sys.exit(plugin.poll())
  
if __name__ == "__main__":
  PluginCLI()

