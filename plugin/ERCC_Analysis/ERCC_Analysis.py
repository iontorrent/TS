#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import simplejson as json
from subprocess import *
from ion.plugin import *
from django.utils.datastructures import SortedDict

class ERCC_Analysis(IonPlugin):
	"""ERCC_Analysis Plugin for use with ERCC RNA Spike-In Control Mixes"""
	version = "4.0-r%s" % filter(str.isdigit,"$Revision: 72040 $")
	envDict = dict(os.environ)
	
	def ERCCRun(self):
		self.envDict['OUTFILE'] = '%s/%s.html'%(self.envDict['RESULTS_DIR'], self.envDict['PLUGINNAME'])
		self.envDict['REFERENCE'] = '%s/ERCC92/ERCC92.fasta'%self.envDict['PLUGIN_PATH']
		if (not os.path.isfile(self.envDict['REFERENCE'])):
			print 'ERROR: No fasta reference available.'
			sys.exit(1)
		else:
			print 'Reference is %s'%self.envDict['REFERENCE']
		
		# Check whether it's a barcode run.
		if (os.path.isfile(self.envDict['TSP_FILEPATH_BARCODE_TXT'])):
			print 'barcoded run\n'
			self.envDict['INPUT_BAM'] = '%s/%s'%(self.envDict['RESULTS_DIR'], self.envDict.get('PLUGINCONFIG__BARCODE', ""))
			self.envDict['BARCODING_USED'] = 'Y'
		else:
			print 'non-barcoded run\n'
			self.envDict['INPUT_BAM'] = self.envDict['TSP_FILEPATH_BAM']
			self.envDict['BARCODING_USED'] = 'N'
		
		# Preprocess.
		preOut = Popen(['python', '%s/code/preproc_fastq.py'%self.envDict['DIRNAME'], self.envDict['INPUT_BAM'], self.envDict['BARCODING_USED'], self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)
		print preOut.communicate()[0]
		
		# Call tmap to create the sam file.
		if (os.path.isfile('%s/filtered.fastq'%self.envDict['RESULTS_DIR'])):
			tmap_out = Popen(['tmap', 'mapall', '-f', self.envDict['REFERENCE'], '-r', '%s/filtered.fastq'%self.envDict['RESULTS_DIR'], '-a', '1', '-g', '0', '-n', '8', 'stage1', 'map1', '--seed-length', '18', 'stage2', 'map2', 'map3', '--seed-length', '18'], stdout=PIPE, env=self.envDict)
			sam_write = open('%s/tmap.sam'%self.envDict['RESULTS_DIR'], 'w')
			sam_write.write(tmap_out.communicate()[0])
			sam_write.close()
			self.envDict['FASTQ_EXISTS'] = 'Y'
		else:
			self.envDict['FASTQ_EXISTS'] = 'N'
		
		# Create the html report.
		htmlRun = Popen(['python', '%s/ERCC_analysis_plugin.py'%self.envDict['DIRNAME'], self.envDict['RESULTS_DIR'], self.envDict['ANALYSIS_DIR'], self.envDict['URL_ROOT'], self.envDict['PLUGINNAME'], self.envDict.get('PLUGINCONFIG__MINRSQUARED', ""), self.envDict['DIRNAME'], self.envDict.get('PLUGINCONFIG__MINCOUNTS', ""), self.envDict.get('PLUGINCONFIG__ERCCPOOL', ""), self.envDict['FASTQ_EXISTS'], self.envDict.get('PLUGINCONFIG__BARCODE', "")], stdout=PIPE, env=self.envDict)
		htmlOut = open(self.envDict['OUTFILE'], 'w')
		htmlOut.write(htmlRun.communicate()[0])
		htmlOut.close()
		
		# Delete filtered .bam in RESULTS_DIR, and also tmap.sam.
		if (os.path.isfile('%s/filtered.fastq'%self.envDict['RESULTS_DIR'])):
			Popen(['rm', '%s/filtered.fastq'%self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)
			# Uncomment if not all filtered fastq's are deleted.
			#Popen(['rm', '%s/filtered*.fastq'%self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)
		if (os.path.isfile('%s/tmap.sam'%self.envDict['RESULTS_DIR'])):
			Popen(['rm', '%s/tmap.sam'%self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)
			
		
	def launch(self, data=None):
		self.envDict['OUTFILE'] = '%s/%s.html'%(self.envDict['RESULTS_DIR'], self.envDict['PLUGINNAME'])
		self.envDict['REFERENCE'] = '%s/ERCC92/ERCC92.fasta'%self.envDict['PLUGIN_PATH']
		
		# Run the plugin.
		self.ERCCRun()
		
		# Exit gracefully.
		sys.exit(0)
		
	def output(self):
		pass
	
	def report(self):
		pass
	
	def metric(self):
		pass

if __name__ == "__main__":
	PluginCLI(ERCC_Analysis())
