#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import simplejson as json
import re
from subprocess import *
from ion.plugin import *
from django.utils.datastructures import SortedDict

class ERCC_Analysis(IonPlugin):
	"""ERCC_Analysis Plugin for use with ERCC RNA Spike-In Control Mixes"""
	version = "4.2-r%s" % filter(str.isdigit,"87667")
	envDict = dict(os.environ)
	# print 'Environment:\n\t' + '\n\t'.join([ '{}: {}'.format(x,y) for x,y in envDict.items() ])
	
	def ERCCRun(self):

		self.envDict['REFERENCE'] = '%s/ERCC92/ERCC92.fasta'%self.envDict['PLUGIN_PATH']
		if (not os.path.isfile(self.envDict['REFERENCE'])):
			print 'ERROR: No fasta reference available.'
			sys.exit(1)
		else:
			print 'Reference is %s'%self.envDict['REFERENCE']
		
		
		input_bams = []
		mapped_bams = []
		out_dirs = []
		barcodes = []

		# Check whether it's a barcode run.
		self.envDict['BARCODING_USED'] = 'N'
		if (os.path.isfile(self.envDict['TSP_FILEPATH_BARCODE_TXT'])):
			print 'barcoded run\n'

			os.system('rm -f %s/Barcodes.html'%self.envDict['RESULTS_DIR'])

			fd = open(self.envDict['TSP_FILEPATH_BARCODE_TXT'], 'r')
			for line in fd:
			        if line.startswith('barcode'):
			        	values = line.split(",")
			        	barcode = values[1]
					if ( self.envDict.get('PLUGINCONFIG__BARCODE', "")!="" and re.search(',%s,'%barcode,',%s,'%self.envDict['PLUGINCONFIG__BARCODE'].strip())==None ) or not os.path.isfile('%s/%s_rawlib.bam'%(self.envDict['ANALYSIS_DIR'],barcode)) and not os.path.isfile('%s/%s_rawlib.basecaller.bam'%(self.envDict['BASECALLER_DIR'],barcode)):
						continue
					print 'To be processed: %s/%s_rawlib.bam'%(self.envDict['ANALYSIS_DIR'],barcode)
					input_bams.append('%s/%s'%(self.envDict['RESULTS_DIR'],barcode))
					mapped_bams.append('%s/%s_rawlib.bam'%(self.envDict['ANALYSIS_DIR'],barcode))
					out_dirs.append('%s/%s'%(self.envDict['RESULTS_DIR'],barcode))
					barcodes.append(barcode)
					os.system('mkdir -p %s/%s'%(self.envDict['RESULTS_DIR'],barcode))
			fd.close()

			self.envDict['BARCODING_USED'] = 'Y'
		
		if (os.path.isfile(self.envDict['TSP_FILEPATH_BAM']) and self.envDict.get('PLUGINCONFIG__BARCODE', "")==""):
			input_bams.append( self.envDict['TSP_FILEPATH_BAM'] )
			mapped_bams.append( self.envDict['TSP_FILEPATH_BAM'] )
			out_dirs.append( self.envDict['RESULTS_DIR'] )

		for i in range(len(input_bams)):

			mapped_bam = mapped_bams[i]
			input_bam = input_bams[i]
			out_dir = out_dirs[i]
			out_file = '%s/%s.html'%(out_dir, self.envDict['PLUGINNAME'])

			if (out_dir == self.envDict['RESULTS_DIR'] ):
				self.envDict['BARCODING_USED'] = 'N'

			self.envDict['FASTQ_EXISTS'] = 'Y'

			ERCC_count = 0
			if (os.path.isfile(mapped_bam)):
				os.system('samtools idxstats %s | cut -f1 | grep "^ERCC-" > idxstats.txt'%mapped_bam)
				idxstats = open("idxstats.txt","r")
				for line in idxstats:
					ERCC_count = ERCC_count+1

			if (ERCC_count>0):
				os.system('samtools view -h %s > %s/tmap.sam'%(mapped_bam,self.envDict['RESULTS_DIR']))

			else:

				# Preprocess.
				preOut = Popen(['python', '%s/code/preproc_fastq.py'%self.envDict['DIRNAME'], input_bam, self.envDict['BARCODING_USED'], self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)
				print preOut.communicate()[0]


				# Call tmap to create the sam file.
				if (os.path.isfile('%s/filtered.fastq'%self.envDict['RESULTS_DIR'])):
					tmap_out = Popen(['tmap', 'mapall', '-f', self.envDict['REFERENCE'], '-r', '%s/filtered.fastq'%self.envDict['RESULTS_DIR'], '-a', '1', '-g', '0', '-n', '8', 'stage1', 'map1', '--seed-length', '18', 'stage2', 'map2', 'map3', '--seed-length', '18'], stdout=PIPE, env=self.envDict)
					sam_write = open('%s/tmap.sam'%self.envDict['RESULTS_DIR'], 'w')
					sam_write.write(tmap_out.communicate()[0])
					sam_write.close()
				else:
					self.envDict['FASTQ_EXISTS'] = 'N'
			
			# Create the html report.
			htmlRun = Popen(['python', '%s/ERCC_analysis_plugin.py'%self.envDict['DIRNAME'], self.envDict['RESULTS_DIR'], self.envDict['ANALYSIS_DIR'], self.envDict['URL_ROOT'], self.envDict['PLUGINNAME'], self.envDict.get('PLUGINCONFIG__MINRSQUARED', ""), self.envDict['DIRNAME'], self.envDict.get('PLUGINCONFIG__MINCOUNTS', ""), self.envDict.get('PLUGINCONFIG__ERCCPOOL', ""), self.envDict['FASTQ_EXISTS'], self.envDict.get('PLUGINCONFIG__BARCODE', "")], stdout=PIPE, env=self.envDict)
			htmlOut = open(out_file, 'w')
			htmlOut.write(htmlRun.communicate()[0])
			htmlOut.close()
			
			if out_dir != self.envDict['RESULTS_DIR']:
				if len(out_dirs)>1:
					fd = open('%s/Barcodes.html'%self.envDict['RESULTS_DIR'],'a')
					fd.write('<a target=_blank href=%s/%s.html>%s</a>:<br>'%(barcodes[i],self.envDict['PLUGINNAME'],barcodes[i]))
					fd.close()
					os.system('cd %s; cat summary*block.html >> Barcodes.html; mv summary*block.html *.png *.dat *.counts %s 2> /dev/null'%(self.envDict['RESULTS_DIR'], out_dir))
				else:
					os.system('mv %s %s 2> /dev/null'%(out_file,self.envDict['RESULTS_DIR']))
		
			# Delete filtered .bam in RESULTS_DIR, and also tmap.sam.
			if (os.path.isfile('%s/filtered.fastq'%self.envDict['RESULTS_DIR'])):
				Popen(['rm', '%s/filtered.fastq'%self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)
			# Uncomment if not all filtered fastq's are deleted.
				#Popen(['rm', '%s/filtered*.fastq'%self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)
			if (os.path.isfile('%s/tmap.sam'%self.envDict['RESULTS_DIR'])):
				Popen(['rm', '%s/tmap.sam'%self.envDict['RESULTS_DIR']], stdout=PIPE, env=self.envDict)
			
		
	def launch(self, data=None):
		
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
