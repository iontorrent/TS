#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import simplejson as json
from subprocess import *
from ion.plugin import *
from django.utils.datastructures import SortedDict
import parse_barcodedSampleNames

class coverageAnalysis(IonPlugin):
	'''Genome and Targeted Re-sequencing Coverage Analysis. (Ion supprted)'''
        version = "4.0-r%s" % filter(str.isdigit,"$Revision: 77897 $") 
	major_block = True
	runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]
	runlevels = [ RunLevel.DEFAULT ]
	
	envDict = dict(os.environ)
	
	def startAnalysis(self):
		# Enable coverageAnalysisLite mode? (No per-target coverage analysis)
		self.envDict['NOTARGETANALYSIS'] = '0'
		# Enable TargetSeq target coverage based on mean base read depth per target? (0 => Classic)
		self.envDict['TARGETCOVBYBASES'] = '1'
		# Dev/debug options; these should be '0' for production.
		self.envDict['PLUGIN_DEV_FULL_LOG'] = '0'
		self.envDict['CONTINUE_AFTER_BARCODE_ERROR'] = '1'
		
		# There are going to be a lot of definitions. A lot. For reference:
		# str[str.rfind('/')+1:] = str | sed -e 's/^.*\///'
		# str[:str.rfind('.')] = str | sed -e 's/\.[^.]*$//' (I think; note to self, if it doesn't work use find instead of rfind.)
		# str | sed -e 's/_/ /g' should just be str.replace('_', ' '), right?
		# and finally, str | sed -e 's/<X>//' should just remove <X>.
		self.envDict['PLUGIN_BAM_FILE'] = self.envDict['TSP_FILEPATH_BAM'][self.envDict['TSP_FILEPATH_BAM'].rfind('/')+1:]
		self.envDict['PLUGIN_BAM_NAME'] = self.envDict['PLUGIN_BAM_FILE'][:self.envDict['PLUGIN_BAM_FILE'].rfind('.')]
		self.envDict['PLUGIN_RUN_NAME'] = self.envDict['TSP_FILEPATH_OUTPUT_STEM']
		self.envDict['REFERENCE'] = self.envDict['TSP_FILEPATH_GENOME_FASTA']
		self.envDict['PLUGIN_SAMPLE_NAMES'] = parse_barcodedSampleNames.sampleNames(self.envDict['TSP_FILEPATH_PLUGIN_DIR']+'/startplugin.json')
		
		# Check for by-pass PUI.
		#65<</>>
		# check if PLUGINCONFIG__LIBRARYTYPE_ID was not set, and if not we set up with reasonable defaults
		if not self.envDict.get('PLUGINCONFIG__LIBRARYTYPE_ID'):
			# 'IFS' is used to determine delimiters in bash
			self.envDict['OLD_IFS'] = self.envDict.get('IFS', "")
			self.envDict['IFS'] = ';'
			# Get plan info w/ pars_plan.py, set library type & target regions.
			parseOut = Popen(['%s/parse_plan.py'%self.envDict['DIRNAME'], '%s/startplugin.json'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']], stdout=PIPE, env=self.envDict)
			parseRead = parseOut.communicate()[0].split(self.envDict['IFS'])
			self.envDict['IFS'] = self.envDict['OLD_IFS']
			self.envDict['PLUGINCONFIG__LIBRARYTYPE'] = parseRead[0]
			self.envDict['PLUGINCONFIG__TARGETREGIONS'] = parseRead[1].replace('\n', '')
			self.envDict['IFS'] = self.envDict['OLD_IFS']
			# GDM: These are defaults for GUI parameters that are not set by the PLAN
			self.envDict['PLUGINCONFIG__SAMPLEID'] = 'No'
			self.envDict['PLUGINCONFIG__TRIMREADS'] = 'No'
			self.envDict['PLUGINCONFIG__PADTARGETS'] = '0'
			self.envDict['PLUGINCONFIG__UNIQUEMAPS'] = 'No'
			self.envDict['PLUGINCONFIG__NONDUPLICATES'] = 'Yes'
			self.envDict['PLUGINCONFIG__BARCODETARGETREGIONS'] = ""
			# GDM: if library type not defined then probably there was no plan
			if (self.envDict['PLUGINCONFIG__LIBRARYTYPE'] == ""):
				# Remove any old results json.
				Popen(['rm', '-f', '%s/results.json'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']], stdout=PIPE, env=self.envDict)
				self.envDict['HTML'] = '%s/%s.html'%(self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['PLUGINNAME'])
				# Sometimes writing to a file from python and bash at the same time doesn't work; be sure to check this part.
				htmlOut = open(self.envDict['HTML'], 'w')
				htmlOut.write('<html><body>')
				if (os.path.isfile('%s/html/logo.sh'%self.envDict['DIRNAME'])):
					Popen(['/bin/bash', '-c', 'source %s/html/logo.sh; print_html_logo'%self.envDict['DIRNAME']], stdout=PIPE, env=self.envDict)
				htmlOut.write('<h3><center>%s</center></h3>'%self.envDict['PLUGIN_RUN_NAME'])
				htmlOut.write('<br/><h2 style="text-align:center;color:red">*** Automatic analysis was not performed. ***</h2>')
				htmlOut.write('<br/><h3 style="text-align:center">(Requires an associated Plan to specify the Run Type.)')
				htmlOut.write('</body></html>')
				# Exit gracefully.
				htmlOut.close()
				sys.exit(0)
			elif (self.envDict['PLUGINCONFIG__LIBRARYTYPE'] == "ampliseq"):
				self.envDict['PLUGINCONFIG__LIBRARYTYPE_ID'] = 'Ion AmpliSeq'
				self.envDict['PLUGINCONFIG__NONDUPLICATES'] = 'No'
			elif (self.envDict['PLUGINCONFIG__LIBRARYTYPE'] == "ampliseq-rna"):
				self.envDict['PLUGINCONFIG__LIBRARYTYPE_ID'] = 'Ion AmpliSeq RNA'
				self.envDict['PLUGINCONFIG__NONDUPLICATES'] = 'No'
			elif (self.envDict['PLUGINCONFIG__LIBRARYTYPE'] == "ampliseq-exome"):
				self.envDict['PLUGINCONFIG__LIBRARYTYPE_ID'] = 'Ion AmpliSeq Exome'
				self.envDict['PLUGINCONFIG__NONDUPLICATES'] = 'No'
			elif (self.envDict['PLUGINCONFIG__LIBRARYTYPE'] == 'targetseq'):
				self.envDict['PLUGINCONFIG__LIBRARYTYPE_ID'] = 'Ion TargetSeq'
			elif (self.envDict['PLUGINCONFIG__LIBRARYTYPE'] == 'wholegenome'):
				self.envDict['PLUGINCONFIG__LIBRARYTYPE_ID'] = 'Whole Genome'
			else:
				sys.stderr.write('ERROR: Unexpected library type: %s'%self.envDict['PLUGINCONFIG__LIBRARYTYPE'])
				sys.exit(0)
			if (self.envDict['PLUGINCONFIG__TARGETREGIONS'] != ""):
				# (should be equivalent to sed -e 's^.*\///' and 's/\.bed$//')
				self.envDict['PLUGINCONFIG__TARGETREGIONS_ID'] = self.envDict['PLUGINCONFIG__TARGETREGIONS'][self.envDict['PLUGINCONFIG__TARGETREGIONS'].rfind('/')+1:].replace('.bed', '')
		else: # PLUGINCONFIG__LIBRARYTYPE_ID existed, do some cleanups to make sure args are well formatted
			# Grab PUI parameters.
			# replace('_', ' ') should = sed 's/_/ /g'.i
			# GDM: This was originally done to circumvent a bug with spaces in passed values, that is probably fixed by now.
			self.envDict['PLUGINCONFIG__LIBRARYTYPE_ID'] = self.envDict.get('PLUGINCONFIG__LIBRARYTYPE_ID', "").replace('_', ' ')
			self.envDict['PLUGINCONFIG__TARGETREGIONS_ID'] = self.envDict.get('PLUGINCONFIG__TARGETREGIONS_ID', "").replace('_', ' ')
			self.envDict['PLUGINCONFIG__LIBRARYTYPE'] = self.envDict.get('PLUGINCONFIG__LIBRARYTYPE', "")
			self.envDict['PLUGINCONFIG__TARGETREGIONS'] = self.envDict.get('PLUGINCONFIG__TARGETREGIONS', "")
			self.envDict['PLUGINCONFIG__PADTARGETS'] = self.envDict.get('PLUGINCONFIG__PADTARGETS', "0")

			if self.envDict.get('PLUGINCONFIG__SAMPLEID'):
				self.envDict['PLUGINCONFIG__SAMPLEID'] = 'Yes'
			else:
				self.envDict['PLUGINCONFIG__SAMPLEID'] = 'No'

			if self.envDict.get('PLUGINCONFIG__TRIMREADS'):
				self.envDict['PLUGINCONFIG__TRIMREADS'] = 'Yes'
			else:
				self.envDict['PLUGINCONFIG__TRIMREADS'] = 'No'

			if self.envDict.get('PLUGINCONFIG__UNIQUEMAPS'):
				self.envDict['PLUGINCONFIG__UNIQUEMAPS'] = 'Yes'
			else:
				self.envDict['PLUGINCONFIG__UNIQUEMAPS'] = 'No'

			if self.envDict.get('PLUGINCONFIG__NONDUPLICATES'):
				self.envDict['PLUGINCONFIG__NONDUPLICATES'] = 'Yes'
			else:
				self.envDict['PLUGINCONFIG__NONDUPLICATES'] = 'No'
			
		# Customize analysis options based on library type.
		self.envDict['PLUGIN_DETAIL_TARGETS'] = self.envDict.get('PLUGINCONFIG__TARGETREGIONS', "")
		if (self.envDict['PLUGIN_DETAIL_TARGETS'] == 'none'):
			self.envDict['PLUGIN_DETAIL_TARGETS'] = ''
		self.envDict['PLUGIN_RUNTYPE'] = self.envDict.get('PLUGINCONFIG__LIBRARYTYPE', "")
		self.envDict['PLUGIN_TARGETS'] = self.envDict['PLUGIN_DETAIL_TARGETS'].replace('/unmerged/', '/merged/')
		self.envDict['PLUGIN_ANNOFIELDS'] = '-f 4,8'
		self.envDict['PLUGIN_READCOV'] = 'e2e'
		self.envDict['AMPOPT'] = ''
		if (self.envDict['PLUGIN_RUNTYPE'] == 'ampliseq'):
			self.envDict['AMPOPT'] = '-a'
		elif (self.envDict['PLUGIN_RUNTYPE'] == 'ampliseq-exome'):
			self.envDict['AMPOPT'] = '-a'
		elif (self.envDict['PLUGIN_RUNTYPE'] == 'ampliseq-rna'):
			self.envDict['AMPOPT'] = '-r'
		else:
			self.envDict['PLUGIN_DETAIL_TARGETS'] = self.envDict['PLUGIN_DETAIL_TARGETS'].replace('unmerged', 'merged')
			if (self.envDict['PLUGIN_RUNTYPE'] == 'targetseq'):
				self.envDict['AMPOPT'] = '-w'
		self.envDict['PLUGIN_SAMPLEID'] = self.envDict['PLUGINCONFIG__SAMPLEID']
		self.envDict['PLUGIN_TRIMREADS'] = self.envDict['PLUGINCONFIG__TRIMREADS']
		self.envDict['PLUGIN_PADSIZE'] = self.envDict['PLUGINCONFIG__PADTARGETS']
		self.envDict['PLUGIN_UMAPS'] = self.envDict['PLUGINCONFIG__UNIQUEMAPS']
		self.envDict['PLUGIN_NONDUPS'] = self.envDict['PLUGINCONFIG__NONDUPLICATES']
		self.envDict['PLUGIN_TRGSID'] = self.envDict['PLUGIN_TARGETS'][self.envDict['PLUGIN_TARGETS'].rfind('/')+1:]
		self.envDict['PLUGIN_TRGSID'] = self.envDict['PLUGIN_TRGSID'][:self.envDict['PLUGIN_TRGSID'].rfind('.')]
		
		self.envDict['PLUGIN_USE_TARGETS'] = '0'
		if (self.envDict['PLUGIN_TARGETS'] != ''):
			self.envDict['PLUGIN_USE_TARGETS'] = '1'
		self.envDict['PLUGIN_BC_TARGETS'] = self.envDict.get('PLUGINCONFIG__BARCODETARGETREGIONS', "")
		self.envDict['PLUGIN_MULTIBED'] = 'No'
		if (self.envDict['PLUGIN_BC_TARGETS'] != ""):
			self.envDict['PLUGIN_MULTIBED'] = 'Yes'
			self.envDict['PLUGIN_USE_TARGETS'] = '1'
		self.envDict['BC_MAPPED_BED'] = ""
		self.envDict['PLUGIN_CHECKBC'] = '1'
		
		# Absolute plugin path to fixed sampleID panel.
		self.envDict['PLUGIN_ROOT'] = os.path.dirname(self.envDict['DIRNAME'])
		self.envDict['PLUGIN_SAMPLEID_REGIONS'] = ""
		if (self.envDict['PLUGIN_SAMPLEID'] == 'Yes'):
			self.envDict['PLUGIN_SAMPLEID_REGIONS'] = '%s/sampleID/targets/KIDDAME_sampleID_regions.bed'%self.envDict['PLUGIN_ROOT']
			if (not os.path.isfile(self.envDict['PLUGIN_SAMPLEID_REGIONS'])):
				sys.stderr.write('\nWARNING: Cannot find sampleID regions file at %s.\nProceding without SampleID Tracking.\n' % 
					self.envDict['PLUGIN_SAMPLEID_REGIONS'])
				self.envDict['PLUGIN_SAMPLEID_REGIONS'] = ""
				self.envDict['PLUGIN_SAMPLEID'] = 'No'
		
		# Report on user/plan option selection and processed options.
		printCmd = Popen(['/bin/bash', '-c', 'source %s/functions/common.sh; print_options'%self.envDict['DIRNAME']], stdout=PIPE, env=self.envDict)
		printCmd.communicate()
		
		# GDM: Dont what the fuss is all about but kept the new code to be safe, minus the debug info. and moving real log output to STDERR
		T_BARCODE_TARGET_MAP = []
		T_BARCODE_TARGET_MAP_KEYS = []
		T_BARCODE_TARGET_MAP_VALS = []
		if (self.envDict['PLUGIN_BC_TARGETS'] != ""):
			# Equivalent of read -a with IFS=';'
			T_BARCODE_TARGET_MAP = self.envDict['PLUGIN_BC_TARGETS'].split(';')
			sys.stderr.write('  Barcoded Targets:\n')
			for T_BCTRGMAP in T_BARCODE_TARGET_MAP:
				if (T_BCTRGMAP != ""):
					T_BCKEY = T_BCTRGMAP.split('=')[0]
					T_BCVAL = T_BCTRGMAP.split('=')[1]
					sys.stderr.write('    %s -> %s\n'%(T_BCKEY, T_BCVAL))
					T_BARCODE_TARGET_MAP_KEYS.append(T_BCKEY)
					T_BARCODE_TARGET_MAP_VALS.append(T_BCVAL)
		# Format it for interpretation by bash. These strings are what the script will get.
		self.envDict['T_BARCODE_TARGET_MAP_KEYS'] = ' '.join(T_BARCODE_TARGET_MAP_KEYS)
		self.envDict['T_BARCODE_TARGET_MAP_VALS'] = ' '.join(T_BARCODE_TARGET_MAP_VALS)
		
		# Define file names, etc.
		self.envDict['LIFECHART'] = '%s/lifechart'%self.envDict['DIRNAME']
		self.envDict['PLUGIN_OUT_COVERAGE_HTML'] = 'COVERAGE_html'
		self.envDict['BARCODES_LIST'] = '%s/barcodeList.txt'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']
		self.envDict['SCRIPTSDIR'] = '%s/scripts'%self.envDict['DIRNAME']
		self.envDict['JSON_RESULTS'] = '%s/results.json'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']
		self.envDict['HTML_RESULTS'] = '%s.html'%self.envDict['PLUGINNAME']
		self.envDict['HTML_BLOCK'] = '%s_block.html'%self.envDict['PLUGINNAME']
		self.envDict['HTML_ROWSUMS'] = '%s_rowsum'%self.envDict['PLUGINNAME']
		self.envDict['PLUGIN_OUT_FILELINKS'] = 'filelinks.xls'
		
		# Help text, etc., for filtering messages.
		self.envDict['HTML_TORRENT_WRAPPER'] = '1'
		self.envDict['PLUGIN_FILTER_READS'] = '0'
		self.envDict['PLUGIN_INFO_FILTERED'] = 'Coverage statistics for uniquely mapped non-duplicate reads.'
		if (self.envDict['PLUGIN_UMAPS'] == 'Yes'):
			self.envDict['PLUGIN_FILTER_READS'] = '1'
			if (self.envDict['PLUGIN_NONDUPS'] == 'No'):
				self.envDict['PLUGIN_INFO_FILTERED'] = 'Coverage statistics for uniquely mapped reads.'
		elif (self.envDict['PLUGIN_NONDUPS'] == 'Yes'):	
			self.envDict['PLUGIN_FILTER_READS'] = '1'
			self.envDict['PLUGIN_INFO_FILTERED'] = 'Coverage statistics for non-duplicate reads.'
		self.envDict['PLUGIN_INFO_ALLREADS'] = 'Coverage statistics for all (unfiltered) aligned reads.'
		
		# Definition of fields and customization in barcode summary table.
		BC_COL_TITLE = []
		BC_COL_HELP = []
		BC_COL_TITLE.append('Mapped Reads')
		BC_COL_HELP.append('Number of reads that were mapped to the full reference genome.')
		BC_COL_TITLE.append('On Target')
		BC_COL_HELP.append('Percentage of mapped reads that were aligned over a target region.')
		BC_COL_TITLE.append('SampleID')
		BC_COL_HELP.append('The percentage of filtered reads mapped to any targeted region used for sample identification.')
		BC_COL_TITLE.append('Mean Depth')
		BC_COL_HELP.append('Mean average target base read depth, including non-covered target bases.')
		BC_COL_TITLE.append('Uniformity')
		BC_COL_HELP.append('Percentage of target bases covered by at least 0.2x the average base read depth.')
		
		# Move fields according to options.
		self.envDict['COV_PAGE_WIDTH'] = '900px'
		self.envDict['BC_SUM_ROWS'] = '5'
		if (self.envDict['AMPOPT'] == '-r'):
			# no mean depth/uniformity columns.
			if (self.envDict['PLUGIN_SAMPLEID'] == 'Yes'):
				self.envDict['BC_SUM_ROWS'] = '3'
			else:
				self.envDict['BC_SUM_ROWS'] = '2'
		elif (self.envDict['PLUGIN_DETAIL_TARGETS'] == "" and self.envDict['PLUGIN_BC_TARGETS'] == ""):
			if (self.envDict['PLUGIN_SAMPLEID'] == 'Yes'):
				# Remove on target
				self.envDict['BC_SUM_ROWS'] = '4'
				BC_COL_TITLE[1] = BC_COL_TITLE[2]
				BC_COL_HELP[1] = BC_COL_HELP[2]
				BC_COL_TITLE[2] = BC_COL_TITLE[3]
				BC_COL_HELP[2] = BC_COL_HELP[3]
				BC_COL_TITLE[3] = BC_COL_TITLE[4]
				BC_COL_HELP[3] = BC_COL_HELP[4]
			else:
				# Remove on target & sampleID
				self.envDict['BC_SUM_ROWS'] = '3'
				BC_COL_TITLE[1] = BC_COL_TITLE[3]
				BC_COL_HELP[1] = BC_COL_HELP[3]
				BC_COL_TITLE[2] = BC_COL_TITLE[4]
				BC_COL_HELP[2] = BC_COL_HELP[4]
		elif (self.envDict['PLUGIN_SAMPLEID'] == 'No'):
			# Remove sampleID
			self.envDict['BC_SUM_ROWS'] = '4'
			BC_COL_TITLE[2] = BC_COL_TITLE[3]
			BC_COL_HELP[2] = BC_COL_HELP[3]
			BC_COL_TITLE[3] = BC_COL_TITLE[4]
			BC_COL_HELP[3] = BC_COL_HELP[4]
		
		# Store in the environment for scripts to interpret.
		for i in range(int(self.envDict['BC_SUM_ROWS'])):
			self.envDict['BCT%s'%i] = BC_COL_TITLE[i]
			self.envDict['BCH%s'%i] = BC_COL_HELP[i]
		
		self.envDict['FILTOPTS'] = ""
		if (self.envDict['PLUGIN_FILTER_READS'] == '1'):
			self.envDict['BC_TITLE_INFO'] = 'Coverage summary statistics for filtered aligned barcoded reads.'
			if (self.envDict['PLUGIN_NONDUPS'] == 'Yes'):
				self.envDict['FILTOPTS'] = '%s -d'%self.envDict['FILTOPTS']
			if (self.envDict['PLUGIN_UMAPS'] == 'Yes'):
				self.envDict['FILTOPTS'] = '%s -u'%self.envDict['FILTOPTS']
		else:
			self.envDict['BC_TITLE_INFO'] = 'Coverage summary statistics for all (un-filtered) aligned barcoded reads.'

		# Tag customization options (e.g. Lite & Classic)
		if (self.envDict['NOTARGETANALYSIS'] == '1'):
			self.envDict['FILTOPTS'] = '%s -b'%self.envDict['FILTOPTS']
		if (self.envDict['TARGETCOVBYBASES'] == '1'):
			self.envDict['FILTOPTS'] = '%s -c'%self.envDict['FILTOPTS']

		# Set up log options.
		self.envDict['LOGOPT'] = ''
		if (int(self.envDict['PLUGIN_DEV_FULL_LOG']) > 0):
			self.envDict['LOGOPT'] = '-l'
		
		# Direct PLUGIN_TRMREADS to detect trimp option.
		self.envDict['TRIMOPT'] = ''
		if (self.envDict['PLUGIN_TRIMREADS'] == 'Yes'):
			self.envDict['TRIMOPT'] = '-t'
		
		# Sourcing is done in each individual Popen command, since Popen is incapable of saving a shell environment and passing sourced shell scripts is dissimilar to passing environment variables.
		
		# Start processing data.
		# Remove previous results.
		Popen(['rm', '-f', '%s/%s'%(self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['HTML_RESULTS']), '%s/%s'%(self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['HTML_BLOCK']), self.envDict['JSON_RESULTS']], stdout=PIPE, env=self.envDict)
		Popen(['rm', '-f', self.envDict['PLUGIN_OUT_COVERAGE_HTML']], stdout=PIPE, env=self.envDict)
		Popen(['rm', '-f', '%s/*.stats.cov.txt'%self.envDict['TSP_FILEPATH_PLUGIN_DIR'], '%s/*.xls'%self.envDict['TSP_FILEPATH_PLUGIN_DIR'], '%s/*.png'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']], stdout=PIPE, env=self.envDict)
		Popen(['rm', '-f', '%s/*.bam*'%self.envDict['TSP_FILEPATH_PLUGIN_DIR'], '%s/*.bed*'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']], stdout=PIPE, env=self.envDict)
		Popen(['rm', '-rf', '%s/static_links'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']], stdout=PIPE, env=self.envDict)
		
		# Local copy of stored barcode list file.
		if (not os.path.isfile(self.envDict['TSP_FILEPATH_BARCODE_TXT'])):
			self.envDict['PLUGIN_CHECKBC'] = '0'
		if (self.envDict['PLUGIN_CHECKBC'] == '1'):
			# Use the old barcode list. If this doesn't work, try assignment instead of file writing.
			sortOut = Popen(['sort', '-t', ' ', '-k', '2n,2', self.envDict['TSP_FILEPATH_BARCODE_TXT']], stdout=PIPE, env=self.envDict)
			barWrite = open(self.envDict['BARCODES_LIST'], 'w')
			barWrite.write(sortOut.communicate()[0])
			barWrite.close()
		
		# Create links to files required for barcode report summary table.
		lnCmd = Popen(['ln', '-sf', '%s/js'%self.envDict['DIRNAME'], self.envDict['TSP_FILEPATH_PLUGIN_DIR']], stdout=PIPE, env=self.envDict)
		lnCmd.communicate()
		lnCmd = Popen(['ln', '-sf', '%s/css'%self.envDict['DIRNAME'], self.envDict['TSP_FILEPATH_PLUGIN_DIR']], stdout=PIPE, env=self.envDict)
		lnCmd.communicate()
		
		# Create padded targets file.
		padOut = Popen(['/bin/bash', '-c', 'source %s/functions/common.sh; create_padded_targets "%s" "%s" "%s"'%(self.envDict['DIRNAME'], self.envDict['PLUGIN_TARGETS'], self.envDict['PLUGIN_PADSIZE'], self.envDict['TSP_FILEPATH_PLUGIN_DIR'])], stdout=PIPE, env=self.envDict)
		padRead = padOut.communicate()[0]
		# Read CREATE_PADDED_TARGETS back from the script.
		self.envDict['PLUGIN_EFF_TARGETS'] = padRead[padRead.find('RVAL:')+5:padRead.find(':END')]
		self.envDict['PADDED_TARGETS'] = ""

		# Create GC annotated BED file for read-to-target assignment.
		gcOut = Popen(['/bin/bash', '-c', 'source %s/functions/common.sh; gc_annotate_bed "%s" "%s"'%(self.envDict['DIRNAME'], self.envDict['PLUGIN_DETAIL_TARGETS'], self.envDict['TSP_FILEPATH_PLUGIN_DIR'])], stdout=PIPE, env=self.envDict)
		gcRead = gcOut.communicate()[0]
		# Read result back from the bash file.
		self.envDict['GCANNOBED'] = gcRead[gcRead.find('RVAL:')+5:gcRead.find(':END')]
		
		# Check for barcodes.
		if (int(self.envDict['PLUGIN_CHECKBC']) == 1):
			barOut = Popen(['/bin/bash', '-c', 'source %s/functions/barcode.sh; source %s/functions/common.sh; source %s/functions/endJavascript.sh; source %s/functions/fileLinks.sh; source %s/functions/head.sh; source %s/functions/footer.sh; source %s/functions/logo.sh; barcode'%(self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'])], stdout=PIPE, env=self.envDict)
			barStd, barErr = barOut.communicate()
		else:
			# Write a front page for non-barcode run.
			self.envDict['HTML'] = '%s/%s'%(self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['HTML_RESULTS'])
			statOut = Popen(['/bin/bash', '-c', 'source %s/functions/common.sh; source %s/functions/head.sh; source %s/functions/logo.sh; source %s/functions/footer.sh; source %s/functions/endJavascript.sh; write_html_header %s %s; display_static_progress %s; write_html_footer %s'%(self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['HTML'], 15, self.envDict['HTML'], self.envDict['HTML'])], stdout=PIPE, env=self.envDict)
			statOut.communicate()
			
			if (os.path.isfile(self.envDict['TSP_FILEPATH_BAM'])):
				self.envDict['TESTBAM'] = self.envDict['TSP_FILEPATH_BAM']
			else:
				self.envDict['TESTBAM'] = '%s/%s.bam'%(self.envDict['ANALYSIS_DIR'], self.envDict['PLUGIN_RUN_NAME'])
			
			lnCmd = Popen(['ln', '-sf', self.envDict['TESTBAM'], '%s.bam'%self.envDict['PLUGIN_RUN_NAME']], stdout=PIPE, env=self.envDict)
			lnCmd.communicate()
			lnCmd = Popen(['ln', '-sf', '%s.bai'%self.envDict['TESTBAM'], '%s.bam.bai'%self.envDict['PLUGIN_RUN_NAME']], stdout=PIPE, env=self.envDict)
			lnCmd.communicate()
			
			# Run on single bam.
			sampleName = self.envDict['TSP_SAMPLE']

			self.envDict['RT'] = '0'
			retOut = Popen(['/bin/bash', '-c', '"%s/run_coverage_analysis.sh" %s %s %s %s -N "%s" -D "%s" -A "%s" -B "%s" -C "%s" -P "%s" -p "%s" -Q "%s" -S "%s" -L "%s" "%s" "%s.bam"'%(self.envDict['SCRIPTSDIR'], self.envDict['LOGOPT'], self.envDict['FILTOPTS'], self.envDict['AMPOPT'], self.envDict['TRIMOPT'], sampleName, self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['GCANNOBED'], self.envDict['PLUGIN_EFF_TARGETS'], self.envDict['PLUGIN_TRGSID'], self.envDict['PADDED_TARGETS'], self.envDict['PLUGIN_PADSIZE'], self.envDict['HTML_BLOCK'], self.envDict['PLUGIN_SAMPLEID_REGIONS'], self.envDict['TSP_LIBRARY'], self.envDict['REFERENCE'], self.envDict['PLUGIN_RUN_NAME'])], stdout=PIPE, env=self.envDict)
			retOut.communicate()
			self.envDict['RT'] = '%s'%retOut.returncode
			if (self.envDict['RT'] != '0'):
				Popen(['/bin/bash', '-c', 'source %s/functions/common.sh; source %s/functions/head.sh; source %s/functions/footer.sh; source %s/functions/logo.sh; write_html_header %s write_html_error %s; write_html_footer %s'%(self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['HTML'], self.envDict['HTML'], self.envDict['HTML'])], stdout=PIPE, env=self.envDict)
				sys.exit(1)
			# Collect results for detail html report and clean up.
			resOut = Popen(['/bin/bash', '-c', 'source %s/functions/common.sh; source %s/functions/logo.sh; source %s/functions/footer.sh; source %s/functions/fileLinks.sh; write_html_results "%s" "%s" . "%s.bam" "%s" "%s"'%(self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['PLUGIN_RUN_NAME'], self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['PLUGIN_RUN_NAME'], sampleName, self.envDict['PLUGIN_DETAIL_TARGETS'])], stdout=PIPE, env=self.envDict)
			# Get PLUGIN_OUT_STATSFILE value.
			resRead, resErr = resOut.communicate()
			self.envDict['PLUGIN_OUT_STATSFILE'] = resRead[resRead.find('RVAL:')+5:resRead.find(':END')]
			jsonFoot = Popen(['/bin/bash', '-c', 'source %s/functions/common.sh; write_json_header; write_json_inner "%s" "%s" "" 2; write_json_footer 0'%(self.envDict['DIRNAME'], self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['PLUGIN_OUT_STATSFILE'])], stdout=PIPE, env=self.envDict)
			jsonFoot.communicate()
		
		# Remove after successful completion.
		Popen(['rm', '-f', '%s/header'%self.envDict['TSP_FILEPATH_PLUGIN_DIR'], '%s/footer'%self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['PADDED_TARGETS'], self.envDict['BARCODES_LIST']], stdout=PIPE, env=self.envDict)
		sys.stderr.write('Completed with statistics output to results.json\n')
	
	def launch(self, data=None):
		# Start the plugin.
		self.startAnalysis()
		return True

	def output(self):
		pass

	def report(self):
		pass

	def metric(self):
		pass

# For debugging purposes.
if __name__ == "__main__":
	PluginCLI(coverageAnalysis())
