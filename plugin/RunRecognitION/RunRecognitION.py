#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import simplejson as json
import traceback
from subprocess import *
from ion.plugin import *
from django.utils.datastructures import SortedDict

class RunRecognitION(IonPlugin):
	version = "4.2-r%s" % filter(str.isdigit,"$Revision$")	

	envDict = dict(os.environ)
	
	def runRunRecognitION(self):
		# Remove old files.
		Popen(['rm', '-f', '%s/%s_block.html'%(self.envDict['RESULTS_DIR'], self.envDict['PLUGINNAME'])], stdout=PIPE, env=self.envDict)
		Popen(['rm', '-f', '%s/leaderboard.html'%self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)

		# Run the plugin file.
		pluginRun = Popen(['python', '%s/run_recognition_plugin.py'%self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)
		print pluginRun.communicate()[0]
		
		# Remove startplugin data
		#Popen(['rm', '%s/startplugin.json'%self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)
	
	def launch(self, data=None):
		
		# Run the plugin.
		self.runRunRecognitION()
		
		# Exit gracefully.
		sys.exit(0)
	
	def output(self):
		pass
	
	def metric(self):
		pass
	
	def report(self):
		pass

if __name__ == "__main__":
	PluginCLI(RunRecognitION())
