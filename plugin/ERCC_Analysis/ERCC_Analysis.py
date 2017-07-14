#!/usr/bin/python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
from subprocess import *
from ion.plugin import *

class ERCC_Analysis(IonPlugin):
  '''Examine relative abundance of ERCC sequence reads. (Ion supported)'''
  version = '5.4.0.0'
  runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
  runlevels = [ RunLevel.DEFAULT ]

  def launch(self,data=None):
    plugin = Popen([
        '%s/ERCC_Analysis_plugin.py' % os.environ['DIRNAME'], '-V', self.version, '-d',
        'startplugin.json', 'barcodes.json'
      ], stdout=PIPE, shell=False )
    plugin.communicate()
    sys.exit(plugin.poll())


if __name__ == "__main__":
  PluginCLI()

