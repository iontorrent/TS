#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import simplejson as json
from subprocess import *
from ion.plugin import *
from django.utils.datastructures import SortedDict

class FilterDuplicates(IonPlugin):
 	version = "4.0-r%s" % filter(str.isdigit,"$Revision: 70791 $")	

	envDict = dict(os.environ)
	
	def runFilter(self):
		# Making environment variables from json with an external script isn't feasible, because python won't read env changes made by its children. Fortunately, it looks like it's also unnecessary because we have the environment dictionary handy.
		runVar = Popen(['/bin/bash', '-c', 'python %s/filterDuplicates_plugin.py -a %s -o %s -m Results.json'%(self.envDict['DIRNAME'], self.envDict['ANALYSIS_DIR'], self.envDict['RESULTS_DIR'])], stdout=PIPE, env=self.envDict)
		print runVar.communicate()[0]
		
		cpCmd = Popen(['cp', 'FilteredBam_block.html', 'FilterDuplicates.html'])
		cpCmd.communicate()
	
	def launch(self, data=None):
		
		# Run the plugin.
		self.runFilter()
		
		# Exit gracefully.
		sys.exit(0)
	
	def output(self):
		pass
	
	def report(self):
		pass
	
	def metric(self):
		pass
	
if __name__ == "__main__":
	PluginCLI(FilterDuplicates())
