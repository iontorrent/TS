#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
from subprocess import *
from ion.plugin import *

class ampliSeqRNA(IonPlugin):
  '''Whole Transciptome AmpliSeq-RNA Analysis. (Ion supprted)'''
  version = '5.4.0.1'
  major_block = True
  runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
  runlevels = [ RunLevel.DEFAULT ]

  def launch(self,data=None):
    plugin = Popen([
        '%s/ampliSeqRNA_plugin.py' % os.environ['DIRNAME'], '-V', self.version, '-d',
        'startplugin.json', 'barcodes.json'
      ], stdout=PIPE, shell=False )
    plugin.communicate()
    sys.exit(plugin.poll())


if __name__ == "__main__":
  PluginCLI()

