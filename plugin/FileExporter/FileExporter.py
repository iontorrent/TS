#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import glob
import json
import os
import tarfile
import urllib2
import zipfile

from ion.plugin import *
from subprocess import *

class FileExporter(IonPlugin):
        version = '5.0.0.0'
	runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
	runlevels = [ RunLevel.DEFAULT ]
	
	envDict = dict(os.environ)
	json_dat = {}
	json_barcodes = {}
	renameString = ""
	barcodeNames = []
	sampleNameLookup = {} # dictionary allowing us to get the sample name associated with a particular barcode name
	isBarcodedRun = False
	variantCallerPath = "variantCaller_out" # path to the most recently run variant caller plugin, TS 4.4 and newer enables multiple instances, so this path may change
	delim = '.'
	selections= ""
	variantExists = False
	downloadDir =''

	# set defaults for user options
	sffCreate = False
	fastqCreate = False
	vcfCreate = False
	xlsCreate = False
	bamCreate = False
	
	zipSFF = False
	zipFASTQ = False
	zipBAM = False
	zipVCF = False
	zipXLS = False
	
	wantTar = True


	# Method to rename and symlink the .bam and .bai files.
	def bamRename(self):
		for barcode in self.barcodeNames:
			fileName = self.json_dat['runinfo']['report_root_dir'] + '/' + self.json_barcodes[barcode]['bam']
			if self.isBarcodedRun:
				# build our new filename and move this file
				finalName = self.renameString.replace('@BARINFO@', barcode)
				if not self.sampleNameLookup[barcode]: #if there is no sample
					finalName = finalName.replace('@SAMPLEID@'+ self.delim, '')
				else:
					finalName = finalName.replace('@SAMPLEID@', self.sampleNameLookup[barcode])
				destName = self.PLUGIN_DIR + '/' + finalName + '.bam'
				downloadsName = self.downloadDir + '/' + finalName + '.bam'
			else:
				destName = self.PLUGIN_DIR + '/' + self.renameString + '.bam'
				downloadsName = self.downloadDir + '/' + self.renameString + '.bam'

			if os.path.isfile(fileName):
				# And, link.
				print 'DEBUG: LINKING: %s --> %s'%(fileName, destName)
				linkCmd = Popen(['ln', '-sf', fileName, destName], stdout=PIPE, env=self.envDict)
				linkOut, linkErr = linkCmd.communicate()
				print 'DEBUG: OUT: %s\nERR: %s'%(linkOut, linkErr)

				
				linkCmd = Popen(['ln', '-sf', fileName, downloadsName], stdout=PIPE, env=self.envDict)
				linkOut, linkErr = linkCmd.communicate()
			else:
				print "Error: %s file does not exist" % fileName
				
			fileName = fileName.replace('.bam', '.bam.bai')
			destName = destName.replace('.bam', '.bam.bai')

			if os.path.isfile(fileName):
				baiLink = Popen(['ln', '-sf', fileName, destName], stdout=PIPE, env=self.envDict)
				baiOut, baiErr = baiLink.communicate()
				print 'DEBUG: BAIOUT: %s\nBAIERR: %s'%(baiOut, baiErr)

				downloadsName = downloadsName.replace('.bam', '.bam.bai')
				linkCmd = Popen(['ln', '-sf', fileName, downloadsName], stdout=PIPE, env=self.envDict)
				linkOut, linkErr = linkCmd.communicate()
			else:
				print "Error: %s file does not exist" % fileName

	# Method to rename and symlink the .vcf files.
	def vcfRename(self):
		for barcode in self.barcodeNames:
			if self.isBarcodedRun:
				# build our new filename and move this file
				finalName = self.renameString.replace('@BARINFO@', barcode)
				if not self.sampleNameLookup[barcode]: #if there is no sample
					finalName = finalName.replace('@SAMPLEID@'+ self.delim, '')
				else:
					finalName = finalName.replace('@SAMPLEID@', self.sampleNameLookup[barcode])
				destName = self.PLUGIN_DIR + '/' + finalName + '.vcf'
				downloadsName = self.downloadDir + '/' + finalName + '.vcf'
				srcName = '%s/plugin_out/%s/%s/TSVC_variants.vcf' % (self.json_dat['runinfo']['analysis_dir'], self.variantCallerPath, barcode)

			else:
				destName = self.PLUGIN_DIR + '/' + self.renameString + '.vcf'
				srcName = '%s/plugin_out/%s/TSVC_variants.vcf' % (self.json_dat['runinfo']['analysis_dir'], self.variantCallerPath)
				downloadsName = self.downloadDir + '/' + self.renameString + '.vcf'

			if os.path.isfile(srcName):
				# And, link.
				print 'LINKING: %s --> %s'%(srcName, destName)
				linkCmd = Popen(['ln', '-sf', srcName, destName], stdout=PIPE, env=self.envDict)
				linkOut, linkErr = linkCmd.communicate()
				print 'OUT: %s\nERR: %s'%(linkOut, linkErr)

				linkCmd = Popen(['ln', '-sf', srcName, downloadsName], stdout=PIPE, env=self.envDict)
				linkOut, linkErr = linkCmd.communicate()
			else:
				print "Error: %s file does not exist" % srcName


	# Method to rename and symlink the .xls files.
	def xlsRename(self):
		for barcode in self.barcodeNames:
			if self.isBarcodedRun:

				# build our new filename and move this file
				finalName = self.renameString.replace('@BARINFO@', barcode)
				if not self.sampleNameLookup[barcode]: #if there is no sample
					finalName = finalName.replace('@SAMPLEID@'+ self.delim, '')
				else:
					finalName = finalName.replace('@SAMPLEID@', self.sampleNameLookup[barcode])
				destNameAlleles = self.PLUGIN_DIR + '/' + finalName + self.delim + 'alleles' + '.xls'
				downloadsNameAlleles = self.downloadDir + '/' + finalName + self.delim + 'alleles' + '.xls'
				destNameVariants = self.PLUGIN_DIR + '/' + finalName + self.delim + 'variants' + '.xls'
				downloadsNameVariants = self.downloadDir + '/' + finalName + self.delim + 'variants' + '.xls'
				
				srcNameAlleles = '%s/plugin_out/%s/%s/alleles.xls' % (self.json_dat['runinfo']['analysis_dir'], self.variantCallerPath, barcode)
				srcNameVariants = '%s/plugin_out/%s/%s/variants.xls' % (self.json_dat['runinfo']['analysis_dir'], self.variantCallerPath, barcode)

			else:
				destNameAlleles = self.PLUGIN_DIR + '/' + self.renameString + self.delim + 'alleles' + '.xls'
				downloadsNameAlleles = self.downloadDir + '/' + self.renameString + self.delim + 'alleles' + '.xls'
				destNameVariants = self.PLUGIN_DIR + '/' + self.renameString + self.delim + 'variants' + '.xls'
				downloadsNameVariants = self.downloadDir + '/' + self.renameString + self.delim + 'variants' + '.xls'

				srcNameAlleles = '%s/plugin_out/%s/alleles.xls' % (self.json_dat['runinfo']['analysis_dir'], self.variantCallerPath)
				srcNameVariants = '%s/plugin_out/%s/variants.xls' % (self.json_dat['runinfo']['analysis_dir'], self.variantCallerPath)

			if os.path.isfile(srcNameAlleles):
				# And link alleles file
				print 'LINKING: %s --> %s'%(srcNameAlleles, destNameAlleles)
				linkCmd = Popen(['ln', '-sf', srcNameAlleles, destNameAlleles], stdout=PIPE, env=self.envDict)
				linkOut, linkErr = linkCmd.communicate()
				print 'OUT: %s\nERR: %s'%(linkOut, linkErr)

			
				linkCmd = Popen(['ln', '-sf', srcNameAlleles, downloadsNameAlleles], stdout=PIPE, env=self.envDict)
				linkOut, linkErr = linkCmd.communicate()
			else:
				print "Error: %s file does not exist" % srcNameAlleles
			
			if os.path.isfile(srcNameVariants):
				# And link variants file
				print 'LINKING: %s --> %s'%(srcNameVariants, destNameVariants)
				linkCmd = Popen(['ln', '-sf', srcNameVariants, destNameVariants], stdout=PIPE, env=self.envDict)
				linkOut, linkErr = linkCmd.communicate()
				print 'OUT: %s\nERR: %s'%(linkOut, linkErr)

			
				linkCmd = Popen(['ln', '-sf', srcNameVariants, downloadsNameVariants], stdout=PIPE, env=self.envDict)
				linkOut, linkErr = linkCmd.communicate()
			else:
				print "Error: %s file does not exist" % srcNameVariants

	def moveRenameFiles(self, suffix):
		# rename them as we move them into the output directory
		# loop through all fastq/sff files
		destName =''
		for fileName in glob.glob('*.%s' % suffix):

			print 'DEBUG: checking %s file: %s' % (suffix, fileName)
			if self.isBarcodedRun:
				for barcode in self.barcodeNames:
					if barcode in fileName:
						finalName = self.renameString.replace('@BARINFO@', barcode)
						if not self.sampleNameLookup[barcode]: #if there is no sample
							finalName = finalName.replace('@SAMPLEID@'+ self.delim, '')
						else:
							finalName = finalName.replace('@SAMPLEID@', self.sampleNameLookup[barcode])
						destName = self.PLUGIN_DIR + '/' + finalName + '.' + suffix

			else:
				destName = self.PLUGIN_DIR + '/' + self.renameString + '.' + suffix

			
			if destName and fileName:
				print 'moving %s to %s' % (fileName, destName)
				os.rename(fileName, destName)
			else:
				os.remove(fileName)
				print 'Logged ERROR: File %s was not renamed' % fileName


	def createFiles(self, type):
		file = 'FastqCreator.py'
		if (type == "sff"):
			file = 'SFFCreator.py'

		fromDir = '%s/%s' % (self.json_dat['runinfo']['plugin']['path'], file)
		toDir = self.PLUGIN_DIR
		print 'cp: from %s\n to %s\n'%(fromDir, toDir)
		Cmd = Popen(['cp', fromDir, toDir], stdout=PIPE, env=self.envDict)
		Out, Err = Cmd.communicate()
		print 'exec: %s/%s' % (toDir, file)
		Cmd = Popen(['python', file], stdout=PIPE, env=self.envDict)
		Out, Err = Cmd.communicate()
		print 'DEBUG: Create file out: %s\nError: %s'%(Out, Err)
		print 'mv: %s -> specified format.' % type

		self.moveRenameFiles(type)

	def compressFiles(self, CompressType, zipSubdir, htmlOut):
		htmlFiles = '<div id="files">'
		for fileName in os.listdir(self.PLUGIN_DIR):
			if (self.zipBAM and (fileName.endswith('.bam') or fileName.endswith('.bai') )) or (self.zipSFF and fileName.endswith('.sff')) or (self.zipFASTQ and fileName.endswith('.fastq')) or (self.zipVCF and fileName.endswith('.vcf')) or (self.zipXLS and fileName.endswith('.xls')):
				print 'Adding file: %s' % fileName
				fullPathAndFileName = self.PLUGIN_DIR + '/' + fileName
				if os.path.isfile(fullPathAndFileName): # need to make sure sym links exist
					if self.wantTar: 
						storedName = zipSubdir + '/' + fileName
						CompressType.add(fullPathAndFileName, arcname=storedName)
					else: 
						if os.path.getsize(fullPathAndFileName) > 1024*1024*1024*2: # store large files (>2G), don't try and compress
							CompressType.write(fullPathAndFileName, zipSubdir + '/' + fileName, zipfile.ZIP_STORED)

						else:
							CompressType.write(fullPathAndFileName, zipSubdir + '/' + fileName, zipfile.ZIP_DEFLATED)
					htmlFiles = htmlFiles + '<span><pre>' + fileName + '</pre></span>\n'

			elif (fileName.endswith('.bam') or fileName.endswith('.bai') or fileName.endswith('.sff') or fileName.endswith('.fastq') or fileName.endswith('.vcf') or fileName.endswith('.xls')):
				htmlOut.write('<a href="%s/%s" download>%s</a><br>\n'%(self.envDict['TSP_URLPATH_PLUGIN_DIR'],fileName, fileName))

		htmlFiles = htmlFiles + "</div>\n"
		return htmlFiles
	def launch(self, data=None):
		try:
			with open('startplugin.json', 'r') as fh:
				self.json_dat = json.load(fh)
		except:
			print 'Error reading plugin json.'
		# Define output directory.
		self.PLUGIN_DIR = self.json_dat['runinfo']['results_dir'] 
		self.downloadDir = os.path.join(self.json_dat['runinfo']['analysis_dir'], 'plugin_out', 'downloads')
		if not os.path.isdir(self.downloadDir):
			Popen(['mkdir', self.downloadDir], stdout=PIPE, env=self.envDict)

		print "DEBUG: PLUGIN_DIR in launch %s" % self.PLUGIN_DIR

		htmlOut = open('FileExporter_block.html', 'w')
		htmlOut.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n')
		htmlOut.write('<script type="text/javascript" src="/site_media/resources/jquery/jquery-1.8.2.js">\n</script>\n')
		htmlOut.write('<script>\n$(function() {\n\t$("#files").hide();\n\t$("#params").hide();\n')
		
		htmlOut.write('\t$("#showParams\").click(function() {\n\t\t$("#params").slideToggle("toggle");\n')
		htmlOut.write('\t\tif ($("#showParams").html() == "Show Parameters"){\n\t\t\t$("#showParams").html("Hide Parameters");\n\t\t}\n')
		htmlOut.write('\t\telse{\n\t\t\t$("#showParams").html("Show Parameters");\n\t\t}\n\t});\n\n')
		
		htmlOut.write('\t$("#showFiles\").click(function() {\n\t\t$("#files").slideToggle("toggle");\n')
		htmlOut.write('\t\tif ($("#showFiles").html() == "Show Contents of Compressed File"){\n\t\t\t$("#showFiles").html("Hide Contents of Compressed File");\n\t\t}\n')
		htmlOut.write('\t\telse{\n\t\t\t$("#showFiles").html("Show Contents of Compressed File");\n\t\t}\n\t});\n});\n')
		htmlOut.write('\n</script>\n</head>\n<html>\n<body>\n')


		try:
			with open('barcodes.json', 'r') as fh:
				self.json_barcodes = json.load(fh)
			if len( self.json_barcodes) > 1 : #non-barcoded runs have 1 barcode called "no match"
				self.isBarcodedRun = True
			else:
				self.isBarcodedRun = False
			self.barcodeNames = self.json_barcodes.keys()
		except:
			print 'Error reading barcodes json.'
			
			
		print "DEBUG: isBarcodedRun in launch: %s" % self.isBarcodedRun

		# Parse pluginconfig json.
		try:
			#get all the button information 
			self.delim = self.json_dat['pluginconfig']['delimiter_select']
			self.selections = self.json_dat['pluginconfig']['select_dialog']

			try:
				temp = self.json_dat['pluginconfig']['bamCreate']
				if (temp == 'on'):
					self.bamCreate = True	
			except:
				print 'Logged: no BAM creation.'

			try:
				temp = self.json_dat['pluginconfig']['sffCreate']
				if (temp == 'on'):
					self.sffCreate = True	
			except:
				print 'Logged: no SFF creation.'

			try:
				temp = self.json_dat['pluginconfig']['fastqCreate']
				if (temp == 'on'):
					self.fastqCreate = True
			except:
				print 'Logged: no FASTQ creation.'

			try:
				temp = self.json_dat['pluginconfig']['vcfCreate']
				if (temp == 'on'):
					self.vcfCreate = True
			except:
				print 'Logged: no VCF linking.'

			try:
				temp = self.json_dat['pluginconfig']['xlsCreate']
				if (temp == 'on'):
					self.xlsCreate = True
			except:
				print 'Logged: no XLS included.'

			try:
				temp = self.json_dat['pluginconfig']['zipSFF']
				if (temp == 'on'):
					self.zipSFF = True
			except:
				print 'Logged: no ZIP SFF'

			try:
				temp = self.json_dat['pluginconfig']['zipFASTQ']
				if (temp == 'on'):
					self.zipFASTQ = True
			except:
				print 'Logged: no ZIP FASTQ'

			try:
				temp = self.json_dat['pluginconfig']['zipBAM']
				if (temp == 'on'):
					self.zipBAM = True
			except:
				print 'Logged: no ZIP BAM'

			try:
				temp = self.json_dat['pluginconfig']['zipVCF']
				if (temp == 'on'):
					self.zipVCF = True
			except:
				print 'Logged: no ZIP VCF'

			try:
				temp = self.json_dat['pluginconfig']['zipXLS']
				if (temp == 'on'):
					self.zipXLS = True
			except:
				print 'Logged: no ZIP XLS'

			try:
				temp = self.json_dat['pluginconfig']['compressedType']
				if (temp == 'zip'):
					self.wantTar = False
				if (temp == 'tar'):
					self.wantTar = True
			except:
				print 'Logged: no compress type selected-  tar.bz2 assumed.'

		except:
			print 'Warning: plugin does not appear to be configured, will default to run name with fastq zipped'
			#sys.exit(0)
			self.delim = '.'
			self.selections = ['run_name']
			self.fastqCreate = True
			self.zipFASTQ = True
		
		try:
			self.runlevel = self.json_dat['runplugin']['runlevel']
			print "DEBUG: runlevel: %s " % self.runlevel
		except:
			self.runlevel = ""
			print 'No run level detected.'
		print "DEBUG: selections: %s, TYPE: %s" % (self.selections, type(self.selections))

		if not isinstance(self.selections, unicode):
			self.selections[:] = [entry for entry in self.selections if entry != '']

		# Get appropriate values.
		for i in range(len(self.selections)):
			# Use an arbitrary value that nobody will ever use otherwise, so they're easy to replace.
			# '@' is an invalid character, right? Maybe not, actually...
			if self.selections[i] == 'barcodename':
				if self.isBarcodedRun:
					self.selections[i] = '@BARINFO@'
				else:
					self.selections[i] = ""
			elif (self.selections[i] == 'sample'):
				if self.isBarcodedRun:
					self.selections[i] = '@SAMPLEID@'
				else:
					self.selections[i] = self.json_dat['expmeta'][self.selections[i]] # user may have provided a sample name to the single sample so just replace now
			else:
				try:
					self.selections[i] = self.json_dat['expmeta'][self.selections[i]]
				except:
					if self.selections[i] == "Enter Text Here":
						self.selections[i] = ""

				
		self.selections[:] = [entry for entry in self.selections if entry != '']
		print "DEBUG: selections: %s, TYPE: %s, %s" % (self.selections, type(self.selections), not self.selections)
		if not self.selections:
			print 'Warning: No options selected, will use default run_name'
			self.selections = [self.json_dat['expmeta']['run_name']]

		# Take care of case where barcode info is not provided in barcoded run.
		if not '@BARINFO@' in self.selections and self.isBarcodedRun:
			self.selections = ['@BARINFO@'] + self.selections
		if '@SAMPLEID@' in self.selections and len(self.selections) == 1:
			self.selections .append([self.json_dat['expmeta']['run_name']])
		elif '@BARINFO@' in self.selections and len(self.selections) == 1:
			self.selections .append(self.json_dat['expmeta']['run_name'])
		elif '@SAMPLEID@' in self.selections and '@BARINFO@' in self.selections and len(self.selections) == 2:
			self.selections .append(self.json_dat['expmeta']['run_name'])
			
		# get the actual path to the variant caller plugin
		if self.vcfCreate or self.xlsCreate:
			try:
				api_url = self.json_dat['runinfo']['api_url'] + '/v1/pluginresult/?format=json&plugin__name=variantCaller&result=' + str(self.json_dat['runinfo']['pk'])
				api_key = self.json_dat['runinfo']['api_key']
				pluginNumber = self.json_dat['runinfo'].get('pluginresult') or self.json_dat['runinfo'].get('plugin',{}).get('pluginresult')
				print "DEBUG: api_url: %s" % api_url
				print "DEBUG: api_key: %s" % api_key 
				if pluginNumber is not None and api_key is not None:
					api_url = api_url + '&pluginresult=%s&api_key=%s' % (pluginNumber, api_key)
					print 'Using pluginresult %s with API key: %s' % (pluginNumber, api_key)
					print "%s" % api_url
				else:
					print 'No API key available'
				d = json.loads(urllib2.urlopen(api_url).read())

				self.variantCallerPath = d['objects'][0]['path']
				print 'DEBUG: variantCallerPath %s' % self.variantCallerPath
				self.variantCallerPath = self.variantCallerPath.split('/')[-1]
				print 'INFO: using variant caller path: %s' % self.variantCallerPath
				self.variantExists = True

			except:
				print 'ERROR!  Failed to get variant caller path. Will not generate variant files.'

		if self.isBarcodedRun:
			for barcode in self.barcodeNames:
				self.sampleNameLookup[barcode] = self.json_barcodes[barcode]['sample']
				print 'DEBUG: sampleNameLookup with barcode %s is %s' % (barcode, self.sampleNameLookup[barcode])
				if self.sampleNameLookup[barcode] == 'none':
					self.sampleNameLookup[barcode] = ''
				self.sampleNameLookup[''] = '' # allows us to easily handle case where barcode might not have been found

		# log basic info for debug purposes
		print 'PLUGINCONFIG:'
		print '----------------------------------------------'
		print 'DELIMETER: "%s"'%self.delim
		print 'SELECTIONS:'
		for sel in self.selections:
			if sel != '':
				print '  %s'%sel
				self.renameString += sel
				self.renameString += self.delim 

		print '----------------------------------------------'
		#remove the last delimiter
		if self.renameString.endswith(self.delim):
			self.renameString = self.renameString.rsplit(self.delim ,1)[0]

		print 'BASE RENAME STRING: %s' % self.renameString


		# Create fastq file(s) if requested.
		if (self.fastqCreate):
			 self.createFiles('fastq')

		# Create sff file(s) if requested.
		if (self.sffCreate):
			self.createFiles('sff')

		# Link to TVC files if requested.
		if (self.bamCreate):
			self.bamRename()
		if self.variantExists:
			if (self.vcfCreate):
				self.vcfRename()
			if (self.xlsCreate):
				self.xlsRename()
		else:
			print "DEBUG: Variant Caller was not run. VCF and XLS not renamed (this message appears even if the user did not specify VCF or XLS"

		htmlOut.write('<b>Output Files:</b><br>\n')
		# Create compressed files (note that we create html links to them if we are not adding to the compressed file)


		print "DEBUG: Enter if? %s, %s, %s, %s, %s" % (self.zipBAM, self.zipSFF, self.zipFASTQ, self.zipVCF, self.zipXLS)
		if (self.zipBAM or self.zipSFF or self.zipFASTQ or self.zipVCF or self.zipXLS):
			print 'Starting write to compressed file'
			print "DEBUG: wantTar= %s" % self.wantTar
			zipSubdir = self.renameString
			if self.isBarcodedRun:
				zipSubdir = self.renameString.replace('@BARINFO@'+ self.delim , '')
				zipSubdir = zipSubdir.replace(self.delim +'@BARINFO@' , '')
				zipSubdir = zipSubdir.replace('@BARINFO@' , '')
				print "subdir = %s" % zipSubdir
				zipSubdir = zipSubdir.replace('@SAMPLEID@'+ self.delim , '')
				zipSubdir = zipSubdir.replace(self.delim +'@SAMPLEID@', '')
				zipSubdir = zipSubdir.replace('@SAMPLEID@', '')
				print "subdir = %s" % zipSubdir

			if self.wantTar:
				compressedFileName = zipSubdir + '.tar.bz2'
				tarName = tarfile.open(compressedFileName, "w:bz2")
				tarName.dereference = True
				htmlFiles = self.compressFiles(tarName, zipSubdir, htmlOut)
				tarName.close()
			else:
				compressedFileName = zipSubdir + '.zip'
				zipName = zipfile.ZipFile(compressedFileName, "w", zipfile.ZIP_DEFLATED, True) # note we are enabling zip64 extensions here
				htmlFiles = self.compressFiles(zipName, zipSubdir, htmlOut)
				zipName.close()
			print 'Finished writing to compressed file.'

			htmlOut.write('<a href="%s" download><b>Compressed Files</b></a><br>\n'%compressedFileName)
			htmlOut.write('<div style="height:30px"><button type="button" class="btn" id="showFiles">Show Contents of Compressed File</button></div>\n')
			htmlOut.write(htmlFiles)

		
		# Create file links.
		else:
			for datum in sorted(os.listdir(self.PLUGIN_DIR)):
				print "DEBUG: File: %s" % datum
				if(datum.endswith('.bam') or datum.endswith('.bai') or datum.endswith('.sff') or datum.endswith('.fastq') or datum.endswith('.vcf') or datum.endswith('.xls')):
					#at this time the download attribute with the filename specification is only supported by a few browsers
					# if at some point it is supported fully, the process will be much easier and the download link itsself can be used to rename the file
					htmlOut.write('<a href="%s/%s" download>%s</a><br>\n'%(self.envDict['TSP_URLPATH_PLUGIN_DIR'], datum, datum)) 
		htmlOut.write('<div style="height:50px"><button type="button" class="btn" id="showParams">Show Parameters</button>\n<div id="params">\n')
		htmlOut.write("<h3>Selected Parameters:</h3>\n<pre>Include BAM: %s &nbsp;Archive: %s\n" % (self.bamCreate, self.zipBAM))
		htmlOut.write("Include VCF: %s &nbsp;Archive: %s\n" % (self.vcfCreate, self.zipVCF))
		htmlOut.write("Include XLS: %s &nbsp;Archive: %s\n" % (self.xlsCreate, self.zipXLS))
		htmlOut.write("Include SFF: %s &nbsp;Archive: %s\n" % (self.sffCreate, self.zipSFF))
		htmlOut.write("Include FASTQ: %s Archive: %s\n</pre>\n" % (self.fastqCreate, self.zipFASTQ))

		htmlOut.write('<h3>Naming Parameters:</h3>\n<pre>Delimiter: "%s"</pre>\n'%self.delim)
		for sel in self.selections:
			if sel != '':
				htmlOut.write('<pre>"%s"</pre>\n'%sel)
		htmlOut.write('</div>')
		htmlOut.close()		
		return True

if __name__ == "__main__":
	PluginCLI(FileExporter())