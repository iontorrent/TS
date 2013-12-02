#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import traceback
import simplejson as json
from subprocess import *
from ion.plugin import *
from django.utils.datastructures import SortedDict


class Alignment(IonPlugin):
  version = "4.0-r%s" % filter(str.isdigit,"$Revision: 77189 $")
  allow_autorun = False
  envDict = dict(os.environ)

  
  def analyze(self):
    # Instead of (for example) 'self.envDict['TSP_LIBRARY']', you can also use 'self.context['environ_tsp']['TSP_LIBRARY']'.
    alignRead = Popen(['%s/alignment_plugin.py'%self.envDict['DIRNAME'], 'startplugin.json', self.envDict['TSP_LIBRARY'], self.envDict['TSP_FILEPATH_UNMAPPED_BAM'], self.envDict['TSP_FILEPATH_BAM']], stdout=PIPE, env=self.envDict)
    
    # Write the results to a file.
    alignOut = open('%s/Alignment_API_output.txt'%self.envDict['TSP_FILEPATH_PLUGIN_DIR'], 'w')
    alignOut.write(alignRead.communicate()[0])
    alignOut.close()
    
    # Copy files.
    cpCmd = Popen(['cp', '-r', '%s/Alignment_block.php'%self.envDict['DIRNAME'], self.envDict['TSP_FILEPATH_PLUGIN_DIR']], stdout=PIPE, env=self.envDict)
    cpCmd.communicate()
    cpCmd = Popen(['cp', '-r', '%s/library'%self.envDict['DIRNAME'], self.envDict['TSP_FILEPATH_PLUGIN_DIR']], stdout=PIPE, env=self.envDict)
    cpCmd.communicate()
    
    # Exit cleanly.
    sys.exit(0)

  
  def launch(self, data=None):
    # Run the script.
    self.analyze()
  
  def output(self):
    pass
  
  def report(self):
    pass
  
  def metrics(self):
    pass

if __name__ == "__main__":
  PluginCLI(Alignment())

