#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import fnmatch
import json
import os
import sys
import glob
from ion.plugin import *
from subprocess import *
from django.utils.datastructures import SortedDict
import zipfile


class FileExporter(IonPlugin):
	version = "4.0-r%s" % filter(str.isdigit,"$Revision: 73226 $")
	runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
	runlevels = [ RunLevel.DEFAULT ]
	
	envDict = dict(os.environ)
	json_dat = {}
	renameString = ""
	barcodeNames = []
	isBarcodedRun = False

	# Method to locate files matching a certain pattern within a directory.
	# Was used for VC file finding, but has been depreciated since those locations are predicted rather than found. (The VC plugin takes far longer to run, so its directory structure cannot be relied on at runtime.)
	def locate(self, pattern, root):
		for path, dirs, files in os.walk(os.path.abspath(root)):
			#for filename in fnmatch.filter(files, pattern):
			for fileName in files:
				#sys.stderr.write('LOC: %s\n'%fileName)
				if fileName == pattern:
					yield os.path.join(path, fileName)
	
	# Method to rename and symlink the .bam and .bai files.
	def bamRename(self, bamFileList):
		for fileName in bamFileList:
			fileName = fileName.replace('./', '')
			if self.isBarcodedRun:
				# look for matching barcode key
				barSample = ''
				for testBarcode in self.barcodeNames:
					if testBarcode in fileName:
						barSample = testBarcode

				# build our new filename and move this file
				if barSample == '': # not expected?  use file name as-is
					destName = self.OUTPUT_DIR + '/' + self.renameString.replace('@BARINFO@', '') + '.bam'
				else:
					destName = self.OUTPUT_DIR + '/' + self.renameString.replace('@BARINFO@', barSample) + '.bam'
			else:
				destName = self.OUTPUT_DIR + '/' + self.renameString + '.bam'

			# And, link.
			print 'LINKING: %s --> %s'%(fileName, destName)
			linkCmd = Popen(['ln', '-sf', fileName, destName], stdout=PIPE, env=self.envDict)
			linkOut, linkErr = linkCmd.communicate()
			fileName = fileName.replace('.bam', '.bam.bai')
			destName = destName.replace('.bam', '.bam.bai')
			baiLink = Popen(['ln', '-sf', fileName, destName], stdout=PIPE, env=self.envDict)
			baiOut, baiErr = baiLink.communicate()
			print 'OUT: %s\nERR: %s'%(linkOut, linkErr)
			print 'BAIOUT: %s\nBAIERR: %s'%(baiOut, baiErr)


	def moveRenameFiles(self, suffix):
		print 'DEBUG: barcoded run: %s' % self.isBarcodedRun

		# rename them as we move them into the output directory

		# loop through all fastq files
		#for fileName in glob.glob('%s/*.fastq' % self.envDict['TSP_FILEPATH_PLUGIN_DIR']):
		for fileName in glob.glob('*.%s' % suffix):
			print 'DEBUG: checking %s file: %s' % (suffix, fileName)
			if self.isBarcodedRun:
				# look for matching barcode key
				barSample = ''
				for testBarcode in self.barcodeNames:
					if testBarcode in fileName:
						barSample = testBarcode

				# build our new filename and move this file
				if barSample == '': # not expected?  use file name as-is
					destName = self.OUTPUT_DIR + '/' + self.renameString.replace('@BARINFO@', '') + '.' + suffix
				else:
					destName = self.OUTPUT_DIR + '/' + self.renameString.replace('@BARINFO@', barSample) + '.' + suffix
			else:
				destName = self.OUTPUT_DIR + '/' + self.renameString + '.' + suffix

			print 'moving %s to %s' % (fileName, destName)
			os.rename(fileName, destName)


	def launch(self, data=None):
		# Define output directory.
		self.OUTPUT_DIR = os.path.join(self.envDict['ANALYSIS_DIR'], 'plugin_out', 'downloads')
		if not os.path.isdir(self.OUTPUT_DIR):
			Popen(['mkdir', self.OUTPUT_DIR], stdout=PIPE, env=self.envDict)
		
		try:
			with open('startplugin.json', 'r') as fh:
				self.json_dat = json.load(fh)
		except:
			print 'Error reading plugin json.'
		
		# Remove old html files if necessary.
		try:
			rmCmd = Popen(['rm', '%s/plugin_out/FileExporter_out/FileExporter_block.html'%self.envDict['ANALYSIS_DIR']], stdout=PIPE)
			rmOut, rmErr = rmCmd.communicate()
			rmCmd = Popen(['rm', 'FileExporter_block.html'%self.envDict['ANALYSIS_DIR']], stdout=PIPE)
			rmOut, rmErr = rmCmd.communicate()
		except:
			pass
		
		try:
			htmlOut = open('%s/plugin_out/FileExporter_out/FileExporter_block.html'%self.envDict['ANALYSIS_DIR'], 'a+')
		except:
			htmlOut = open('FileExporter_block.html', 'w')
		htmlOut.write('<html><body>\n')
		# htmlOut.write('<b>PLUGINCONFIG:</b> %s<br/><br/>\n'%self.json_dat['pluginconfig'])

		if os.path.isfile(self.envDict['TSP_FILEPATH_BARCODE_TXT']):
			self.isBarcodedRun = True

		# Parse pluginconfig json.
		try:
			delim = self.json_dat['pluginconfig']['delimiter_select']
			selections = self.json_dat['pluginconfig']['select_dialog']
			sffCreate = False
			fastqCreate = False
			vcCreate = False
			zipCreate = False
			try:
				temp = self.json_dat['pluginconfig']['sffCreate']
				if (temp == 'on'):
					sffCreate = True
			except:
				print 'Logged: no SFF creation.'
			try:
				temp = self.json_dat['pluginconfig']['fastqCreate']
				if (temp == 'on'):
					fastqCreate = True
			except:
				print 'Logged: no FASTQ creation.'
			try:
				temp = self.json_dat['pluginconfig']['vcCreate']
				if (temp == 'on'):
					vcCreate = True
			except:
				print 'Logged: no VC linking.'
			try:
				temp = self.json_dat['pluginconfig']['zipCreate']
				if (temp == 'on'):
					zipCreate = True
			except:
				print 'Logged: no ZIP creation.'
		except:
			print 'Warning: plugin does not appear to be configured, will default to run name with fastq zipped'
			#sys.exit(0)
			selections = ['TSP_RUN_NAME']
			delim = '.'
			fastqCreate = True
			zipCreate = True
		
		try:
			self.runlevel = self.json_dat['runplugin']['runlevel']
		except:
			self.runlevel = ""
			print 'No run level detected.'

		# DEBUG: Print barcoded sampleID data.
		samples = self.json_dat['plan']['barcodedSamples']
		print 'SAMPLEID DATA: %s'%samples
		
		htmlOut.write('<b>Create SFF?</b> %s<br/>\n'%sffCreate)
		htmlOut.write('<b>Create FASTQ?</b> %s<br/>\n'%fastqCreate)
		htmlOut.write('<b>Link Variants?</b> %s<br/>\n'%vcCreate)
		htmlOut.write('<b>Create ZIP?</b> %s<br/>\n'%zipCreate)
		
		# Remove empty values.
		if not isinstance(selections, unicode):
			selections[:] = [entry for entry in selections if entry != '']
		elif selections != u'':
			selections = [selections]
		else:
			print 'Warning: No options selected, will use default TSP_RUN_NAME'
			selections = ['TSP_RUN_NAME']
		
		# Get appropriate values.
		for i in range(len(selections)):
			# Use an arbitrary value that nobody will ever use otherwise, so they're easy to replace.
			# '@' is an invalid character, right? Maybe not, actually...
			if (selections[i] == 'OPT_BARCODE'):
				if (os.path.isfile(self.envDict['TSP_FILEPATH_BARCODE_TXT'])):
					selections[i] = '@BARINFO@'
				else:
					selections[i] = ''
			else:
				selections[i] = self.envDict[selections[i]]
				selections[i] = selections[i].replace('\\', '') # no idea why, but some chips look like \314R\ with backslashes?

		# Take care of case where barcode info is not provided in barcoded run.
		if not '@BARINFO@' in selections and os.path.isfile(self.envDict['TSP_FILEPATH_BARCODE_TXT']):
			selections = ['@BARINFO@'] + selections

		# Get bam filenames.
		with open(os.path.join(self.json_dat['runinfo']['basecaller_dir'], 'datasets_basecaller.json'), 'r') as f:
			json_basecaller = json.load(f)

		bamPaths = []
		bams = []
		for datum in json_basecaller['datasets']:
			tempPath = os.path.join(self.json_dat['runinfo']['alignment_dir'], datum['file_prefix']+'.bam')
			if os.path.exists(tempPath):
				bamPaths.append(tempPath)
				if datum['dataset_name'][datum['dataset_name'].rfind('/')+1:] != 'No_barcode_match' and '/' in datum['dataset_name']:
					bams.append(datum['dataset_name'])

		# get the list of 'valid' barcodes or samples (could be either depending on whether user altered names with run planning
		# and sort of hacky, but extract this from the BAM file names we just got above
		for bamFileName in bamPaths:
			barcodeName = bamFileName.split('/')[-1] # get the last part, just the name with no path (probably can use os method here too)
			barcodeName = barcodeName.split('_rawlib')[0] # get just the barcode part of the name
			print 'BARCODE FOUND: %s' % barcodeName
			self.barcodeNames.append(barcodeName)

		# log basic info for debug purposes
		print 'PLUGINCONFIG:'
		print '----------------------------------------------'
		print 'DELIMETER: "%s"'%delim
		htmlOut.write('<b>DELIMITER:</b> "%s"<br/>\n<b>SELECTIONS:</b><br/>\n'%delim)
		print 'SELECTIONS:'
		for sel in selections:
			print '  %s'%sel
			htmlOut.write('\t%s<br/>\n'%sel)
		print '----------------------------------------------'
		
		# Produce string to rename to.
		self.renameString = ""
		# Avoid delimiting anywhere but in between the arguments.
		doDelim = False
		for sel in selections:
			if sel != '':
				if doDelim:
					self.renameString += delim
				else:
					doDelim = True
				self.renameString += '%s'%sel
		
		# Perform bam symlink(s).
		self.bamRename(bamPaths)


		# Create fastq file(s) if requested.
		if (fastqCreate):
			# create FASTQ file(s)
			fromDir = '%s/FastqCreator.py'%self.envDict['DIRNAME']
			toDir = self.envDict['TSP_FILEPATH_PLUGIN_DIR']
			print 'cp: from %s\n to %s\n'%(fromDir, toDir)
			fqCmd = Popen(['cp', fromDir, toDir], stdout=PIPE, env=self.envDict)
			fqOut, fqErr = fqCmd.communicate()
			print 'exec: %s/FastqCreator.py'%toDir
			FastqCmd = Popen(['python', 'FastqCreator.py'], stdout=PIPE, env=self.envDict)
			FastqOut, FastqErr = FastqCmd.communicate()
			#print 'Fastq out: %s\nFastq err: %s'%(FastqOut, FastqErr)
			print 'mv: fastq -> specified format.'

			self.moveRenameFiles('fastq')


		# Create sff file(s) if requested.
		if (sffCreate):
			fromDir = '%s/SFFCreator.py'%self.envDict['DIRNAME']
			toDir = self.envDict['TSP_FILEPATH_PLUGIN_DIR']
			print 'cp: from %s\n to %s\n'%(fromDir, toDir)
			sfCmd = Popen(['cp', fromDir, toDir], stdout=PIPE, env=self.envDict)
			sfOut, sfErr = sfCmd.communicate()
			print 'exec: %s/SFFCreator.py'%toDir
			SFFCmd = Popen(['python', 'SFFCreator.py'], stdout=PIPE, env=self.envDict)
			SFFOut, SFFErr = SFFCmd.communicate()
			#print 'SFF out: %s\nSFF err: %s'%(SFFOut, SFFErr)
			print 'mv: sff -> specified format.'

			self.moveRenameFiles('sff')


		# Link to variants if requested.
		vcNew = ''
		vcNames = []
		if (vcCreate):
			'''#vcFiles = self.locate('TSVC_variants.vcf', '%s/plugin_out/variantCaller_out'%self.envDict['ANALYSIS_DIR'])
			vcFiles = []
			if bams != []:
				for bam in bams:
					vcFiles.append('%s/plugin_out/variantCaller_out/%s/TSVC_variants.vcf'%(self.envDict['ANALYSIS_DIR'], bam[bam.rfind('/')+1:]))
			else:
				vcFiles.append('%s/plugin_out/variantCaller_out/TSVC_variants.vcf'%self.envDict['ANALYSIS_DIR'])'''
			vcNew = self.renameString
			#vcNew = vcNew.replace(self.envDict['TSP_SAMPLE'], '')
			if bams != []:
				for bam in bams:
					vcFile = '%s/plugin_out/variantCaller_out/%s/TSVC_variants.vcf'%(self.envDict['ANALYSIS_DIR'], bam[bam.rfind('/'):])
					vcName = vcNew
					vcName = vcName.replace('@BARINFO@', bam[bam.rfind('/')+1:])
					#vcName = vcName.replace('@BARINFO@', '')
					if bam[:bam.rfind('/')] == 'None':
						vcName = vcName.replace(self.envDict['TSP_SAMPLE'], '')
					else:
						vcName = vcName.replace(self.envDict['TSP_SAMPLE'], bam[:bam.rfind('/')])
					vcName = vcName.replace('%s%s'%(self.json_dat['pluginconfig']['delimiter_select'], self.json_dat['pluginconfig']['delimiter_select']), self.json_dat['pluginconfig']['delimiter_select'])
					if vcName[0] == self.json_dat['pluginconfig']['delimiter_select']:
						vcName = vcName[1:]
					if vcName[len(vcName)-1] == self.json_dat['pluginconfig']['delimiter_select']:
						vcName = vcName[:len(vcName)-1]
					lnCmd = Popen(['ln', '-sf', vcFile, os.path.join('%s'%self.envDict['ANALYSIS_DIR'], 'plugin_out', 'downloads', '%s.vcf'%vcName)], stdout=PIPE)
					lnOut, lnErr = lnCmd.communicate()
					vcNames.append(vcName)
					sys.stderr.write('vcf appended: %s\n'%vcName)
			else:
				vcFile = '%s/plugin_out/variantCaller_out/TSVC_variants.vcf'%(self.envDict['ANALYSIS_DIR'])
				vcName = vcNew
				vcName = vcName.replace('@BARINFO@', '')
				vcName = vcName.replace(self.envDict['TSP_SAMPLE'], '')
				vcName = vcName.replace('%s%s'%(self.json_dat['pluginconfig']['delimiter_select'], self.json_dat['pluginconfig']['delimiter_select']), self.json_dat['pluginconfig']['delimiter_select'])
				if vcName[0] == self.json_dat['pluginconfig']['delimiter_select']:
					vcName = vcName[1:]
				if vcName[len(vcName)-1] == self.json_dat['pluginconfig']['delimiter_select']:
					vcName = vcName[:len(vcName)-1]
				lnCmd = Popen(['ln', '-sf', vcFile, os.path.join('%s'%self.envDict['ANALYSIS_DIR'], 'plugin_out', 'downloads', '%s.vcf'%vcName)], stdout=PIPE)
				lnOut, lnErr = lnCmd.communicate()
				vcNames.append(vcName)
				sys.stderr.write('vcf appended: %s\n'%vcName)
			
		
		#htmlOut.write('<br/><b>Files created: </b><a href="/report/%s/getZip">Download link</a><br/>'%self.envDict['RUNINFO__PK'])
		htmlOut.write('<b/><b>Output Files:</b></br>')

		webRootPathParts = self.envDict['ANALYSIS_DIR'].split('/')
		try:
			webRoot = webRootPathParts[-3] + '/' + webRootPathParts[-2] + '/' + webRootPathParts[-1]
		except:
			webRoot = self.envDict['ANALYSIS_DIR'].replace('/results/analysis/', '')
		print 'WebRoot: %s' % webRoot

		# Create zip files.
		if (zipCreate):
			print 'Starting zip.'
			zipSubdir = self.renameString
			if self.isBarcodedRun:
				removeThisPart = '@BARINFO@' + delim
				zipSubdir = self.renameString.replace(removeThisPart, '')
			zipFileName = zipSubdir + '.zip'
			downloads = zipfile.ZipFile(zipFileName, "w", zipfile.ZIP_DEFLATED, True) # note we are enabling zip64 extensions here
			#for fileName in glob.glob(self.envDict['ANALYSIS_DIR'] + '/plugin_out/downloads/*'):
			for fileName in os.listdir(self.envDict['ANALYSIS_DIR'] + '/plugin_out/downloads'):
				print 'ZIP: Adding file: %s' % fileName
				if os.path.isfile(self.envDict['ANALYSIS_DIR'] + '/plugin_out/downloads/' + fileName): # need to make sure sym links exist
					downloads.write(self.envDict['ANALYSIS_DIR'] + '/plugin_out/downloads/' + fileName, zipSubdir + '/' + fileName, zipfile.ZIP_DEFLATED)
			downloads.close()
			print 'Finished zipping.'

			#htmlOut.write('<a href="/report/%s/getZip">Zipped Output</a>'%self.envDict['RUNINFO__PK'])
			htmlOut.write('<a href="%s">Zipped Output</a>'%zipFileName)

			#if (vcCreate):
				#for vcName in vcNames:
					#htmlOut.write('<br/><a href="%s.vcf">%s.vcf</a>'%(vcName, vcName))
			
			# Remove leftovers.
			rmCmd = Popen(['rm', '-f', '%s/plugin_out/downloads/*'%self.envDict['ANALYSIS_DIR']], stdout=PIPE, env=self.envDict)
			rmOut, rmErr = rmCmd.communicate()
		
		# Create file links.
		else:
			htmlOut.write("<!--LINKS-->")
			for datum in sorted(os.listdir("%s/plugin_out/downloads"%self.envDict['ANALYSIS_DIR'])):
				htmlOut.write('<br/><a href="../downloads/%s">%s</a>'%(datum, datum))
			htmlOut.write("<!--ENDLINKS-->")
		
		htmlOut.close()		
		return True

if __name__ == "__main__":
	PluginCLI(FileExporter())
