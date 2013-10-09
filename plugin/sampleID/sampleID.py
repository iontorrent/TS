#!/usr/bin/python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import traceback
import simplejson as json
from subprocess import *
from ion.plugin import *
from django.utils.datastructures import SortedDict
import parse_barcodedSampleNames

class sampleID(IonPlugin):
	version = "4.0-r%s" % filter(str.isdigit,"$Revision: 73765 $")
	major_block = True
	runtypes = [ RunType.FULLCHIP, RunType.THUMB, RunType.COMPOSITE ]

	envDict = dict(os.environ)
	
	# HTML layout helpers.
	BC_COL_TITLE = []
	BC_COL_TITLE.append('Sample ID')
	BC_COL_TITLE.append('Reads On-Target')
	BC_COL_TITLE.append('Read Depth')
	BC_COL_TITLE.append('20x Coverage')
	BC_COL_TITLE.append('100x Coverage')
	BC_COL_HELP = []
	BC_COL_HELP.append('Sample identification code, based on gender and homozygous or heterozugous cells made for reads at target loci.')
	BC_COL_HELP.append('Number of mapped reads that were aligned over gender and sample identification regions.')
	BC_COL_HELP.append('Average target base read depth over all sample identification loci.')
	BC_COL_HELP.append('Percentage of all sample identification loci that were covered to at least 20x read depth.')
	BC_COL_HELP.append('Percentage of all sample identification loci that were covered to at least 100x read depth.')
	
	def write_html_results(self, rid, odir, ourl, bfile, sampleName='None'):
		self.envDict['RUNID'] = rid
		self.envDict['OUTDIR'] = odir
		self.envDict['OUTURL'] = ourl
		self.envDict['BAMFILE'] = bfile
		
		# Create softlinks to js/css folders and php scripts.
		Popen(['ln', '-sf', '%s/slickgrid'%self.envDict['DIRNAME'], self.envDict['OUTDIR']], stdout=PIPE, env=self.envDict)
		Popen(['ln', '-sf', '%s/lifegrid'%self.envDict['DIRNAME'], self.envDict['OUTDIR']], stdout=PIPE, env=self.envDict)
		Popen(['ln', '-sf', '%s/scripts/igv.php3'%self.envDict['DIRNAME'], self.envDict['OUTDIR']], stdout=PIPE, env=self.envDict)
		
		# Link bam/bed files from plugin dir, and create local URL names for the fileLinks table.
		self.envDict['PLUGIN_OUT_BAMFILE'] = self.envDict['BAMFILE'][self.envDict['BAMFILE'].rfind('/')+1:]
		self.envDict['PLUGIN_OUT_BAIFILE'] = '%s.bai'%self.envDict['PLUGIN_OUT_BAMFILE']
		if (self.envDict['BAMFILE'] is not None and self.envDict['BAMFILE'] != ""):
			# Create hard links if it's a combineAlignment file, symlinks otherwise.
			try:
				if (self.envDict['PLUGINCONFIG__MERGEDBAM'] is not None and self.envDict['PLUGINCONFIG__MERGEDBAM'] != ""):
					Popen(['ln', '-f', self.envDict['BAMFILE'], '%s/%s'%(self.envDict['OUTDIR'], self.envDict['PLUGIN_OUT_BAMFILE'])], stdout=PIPE, env=self.envDict)
					Popen(['ln', '-f', '%s.bai'%self.envDict['BAMFILE'], '%s/%s'%(self.envDict['OUTDIR'], self.envDict['PLUGIN_OUT_BAIFILE'])], stdout=PIPE, env=self.envDict)
				else:
					someVar = 2/0
			except:
				Popen(['ln', '-sf', self.envDict['BAMFILE'], '%s/%s'%(self.envDict['OUTDIR'], self.envDict['PLUGIN_OUT_BAMFILE'])], stdout=PIPE, env=self.envDict)
				Popen(['ln', '-sf', '%s.bai'%self.envDict['BAMFILE'], '%s/%s'%(self.envDict['OUTDIR'], self.envDict['PLUGIN_OUT_BAIFILE'])], stdout=PIPE, env=self.envDict)

		if (self.envDict['OUTDIR'] != self.envDict['TSP_FILEPATH_PLUGIN_DIR']):
			if (self.envDict['INPUT_BED_FILE'] is not None and self.envDict['INPUT_BED_FILE'] != ""):
				Popen(['ln', '-sf', '%s/%s'%(self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['PLUGIN_OUT_BEDFILE'])], stdout=PIPE, env=self.envDict)
			if (self.envDict['INPUT_SNP_BED_FILE'] is not None and self.envDict['INPUT_SNP_BED_FILE'] != ""):
				Popen(['ln', '-sf', '%s/%s'%(self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['PLUGIN_OUT_LOCI_BEDFILE'])], stdout=PIPE, env=self.envDict)
		
		# Create the html report page.
		sys.stderr.write( 'Generating report...\n' )
		self.envDict['HTMLOUT'] = '%s/%s'%(self.envDict['OUTDIR'], self.envDict['HTML_RESULTS'])
		covCmd = Popen(['/bin/bash', '-c', 'source %s/html/common.sh; source %s/html/logo.sh; write_page_header %s/SNPID.head.html %s'%(self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['LIFEGRIDDIR'], self.envDict['HTMLOUT'])], stdout=PIPE, env=self.envDict)
		covOut, covErr = covCmd.communicate()
		
		if ( sampleName != "" and sampleName != "None" ):
			with open(self.envDict['HTMLOUT'], "a") as myfile:
				myfile.write("<h3><center>Sample Name: %s</center></h3>\n" % sampleName)

		covCmd = Popen(['/bin/bash', '-c', 'source %s/html/common.sh; copy_coverage_html'%self.envDict['DIRNAME']], stdout=PIPE, env=self.envDict)
		covOut, covErr = covCmd.communicate()

		# Create a partial report and convert to .pdf
		self.envDict['HTMLTMP'] = '%s_.html'%self.envDict['HTMLOUT']
		writeCmd = Popen(['/bin/bash', '-c', 'source %s/html/common.sh; write_partial_header'%self.envDict['DIRNAME']], stdout=PIPE, env=self.envDict)
		writeOut, writeErr = writeCmd.communicate()

		self.envDict['PLUGIN_OUT_PDFFILE'] = '%s.pdf'%self.envDict['PLUGIN_OUT_BAMFILE'][:self.envDict['PLUGIN_OUT_BAMFILE'].rfind('.')]
		try:
			# This should make the .pdf
			PDFCMD = Popen(['%s/wkhtmltopdf-amd64'%self.envDict['BINDIR'], '--load-error-handling', 'ignore', '--no-background', self.envDict['HTMLTMP'], '%s/%s'%(self.envDict['OUTDIR'], self.envDict['PLUGIN_OUT_PDFFILE'])], stdout=PIPE, stderr=PIPE, env=self.envDict)
			writeOut, writeErr = PDFCMD.communicate()
		except:
			# If if does fail, say so.
			sys.stderr.write('ERROR: PDF not written.')
		
		Popen(['rm', '-f', self.envDict['HTMLTMP']], stdout=PIPE, env=self.envDict)
		
		# Add in the full file links to the report.
		writeCmd = Popen(['/bin/bash', '-c', 'source %s/html/common.sh; source %s/html/fileLinks.sh; source %s/html/footer.sh; write_file_links %s %s; write_page_footer %s'%(self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['OUTDIR'], self.envDict['PLUGIN_OUT_FILELINKS'], self.envDict['HTMLOUT'])], stdout=PIPE, env=self.envDict)
		writeOut, writeErr = writeCmd.communicate()
		
		# Remove temporary files.
		Popen(['rm', '-f', '%s/%s'%(self.envDict['OUTDIR'], self.envDict['PLUGIN_OUT_COVERAGE_HTML'])], stdout=PIPE, env=self.envDict)
		return 0
		
	def sampleID(self):
		# Dev/debug options. Should be 0 for production mode.
		self.envDict['PLUGIN_DEV_KEEP_INTERMEDIATE_FILES'] = '0'
		self.envDict['PLUGIN_DEV_FULL_LOG'] = '0'
		self.envDict['CONTINUE_AFTER_BARCODE_ERROR'] = '1'
		self.envDict['ENABLE_HOTSPOT_LEFT_ALIGNMENT'] = '0'
		
		# Setup environment.
		# (In case this winds up as example code) To demonstrate, this looks up every value in envDict whenever it needs them. In many cases, it will be faster to assign variables rather than looking up values in envDict every time, but if you do that make sure that changes made to it are also written to envDict. Python dictionary lookup time is...wow, O(1)? Really? Okay, maybe using vars wouldn't be too much faster, although I guess that it's on the order of the number of elements in the environment, not the number of times variables are retrieved...
		self.envDict['INPUT_BED_FILE'] = 'targets/KIDDAME_sampleID_regions.bed'
		self.envDict['INPUT_SNP_BED_FILE'] = 'targets/KIDDAME_sampleID_loci.bed'
		self.envDict['REFERENCE'] = self.envDict['TSP_FILEPATH_GENOME_FASTA']
		self.envDict['REFERENCE_FAI'] = '%s.fai'%self.envDict['REFERENCE']
		self.envDict['PLUGIN_SAMPLE_NAMES'] = parse_barcodedSampleNames.sampleNames(self.envDict['TSP_FILEPATH_PLUGIN_DIR']+'/startplugin.json')
		
		self.envDict['ERROUT'] = '2> /dev/null'
		self.envDict['LOGOPT'] = ''
		if self.envDict['PLUGIN_DEV_FULL_LOG'] != '0':
			self.envDict['ERROUT'] = ''
			self.envDict['LOGOPT'] = '-l'
		
		# Define many, many environment variables.
		self.envDict['PLUGIN_BAM_FILE'] = self.envDict['TSP_FILEPATH_BAM'][self.envDict['TSP_FILEPATH_BAM'].rfind('/')+1:]
		self.envDict['PLUGIN_BAM_NAME'] = self.envDict['PLUGIN_BAM_FILE'][:self.envDict['PLUGIN_BAM_FILE'].rfind('.')]
		self.envDict['PLUGIN_RUN_NAME'] = self.envDict['TSP_RUN_NAME']
		
		self.envDict['PLUGIN_HS_ALIGN_DIR'] = '%s/hs_align'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']
		self.envDict['PLUGIN_HS_ALIGN_BED'] = '%s/hs_align.bed'%self.envDict['PLUGIN_HS_ALIGN_DIR']
		
		self.envDict['PLUGIN_OUT_READ_STATS'] = 'read_stats.txt'
		self.envDict['PLUGIN_OUT_TARGET_STATS'] = 'on_target_stats.txt'
		self.envDict['PLUGIN_OUT_LOCI_STATS'] = 'on_loci_stats.txt'
		
		self.envDict['PLUGIN_OUT_COV_RAW'] = 'allele_counts.txt'
		self.envDict['PLUGIN_OUT_COV'] = 'allele_counts.xls'
		
		self.envDict['PLUGIN_OUT_COVERAGE_HTML'] = 'COVERAGE_html'
		self.envDict['PLUGIN_OUT_FILELINKS'] = 'flielinks.xls'
		
		self.envDict['PLUGIN_CHROM_X_TARGETS'] = '%s/CHROMX.bed'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']
		self.envDict['PLUGIN_CHROM_Y_TARGETS'] = '%s/CHROMY.bed'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']
		self.envDict['PLUGIN_CHROM_NO_Y_TARGETS'] = '%s/TARGETNOY.bed'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']
		
		self.envDict['BARCODES_LIST'] = '%s/barcodeList.txt'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']
		self.envDict['SCRIPTSDIR'] = '%s/scripts'%self.envDict['DIRNAME']
		self.envDict['BINDIR'] = '%s/bin'%self.envDict['DIRNAME']
		self.envDict['JSON_RESULTS'] = '%s/results.json'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']
		self.envDict['HTML_RESULTS'] = '%s.html'%self.envDict['PLUGINNAME']
		self.envDict['HTML_BLOCK'] = '%s_block.html'%self.envDict['PLUGINNAME']
		self.envDict['HTML_ROWSUMS'] = '%s_rowsum'%self.envDict['PLUGINNAME']
		self.envDict['HTML_TORRENT_WRAPPER'] = '1'
		
		self.envDict['PLUGIN_OUT_HAPLOCODE'] = '%s/haplocode'%self.envDict['TSP_FILEPATH_PLUGIN_DIR']
		
		self.envDict['INPUT_BED_FILE'] = '%s/%s'%(self.envDict['DIRNAME'], self.envDict['INPUT_BED_FILE'])
		self.envDict['INPUT_SNP_BED_FILE'] = '%s/%s'%(self.envDict['DIRNAME'], self.envDict['INPUT_SNP_BED_FILE'])
		
		self.envDict['BC_HAVE_LOCI'] = '0'
		if (self.envDict['INPUT_SNP_BED_FILE'] is not (None or "")):
			self.envDict['BC_HAVE_LOCI'] = '1'
		
		self.envDict['BC_ERROR'] = '0'
		
		# Local copy of sorted barcode list file.
		self.envDict['PLUGIN_CHECKBC'] = '0'
		if (os.path.isfile(self.envDict['TSP_FILEPATH_BARCODE_TXT'])):
			self.envDict['PLUGIN_CHECKBC'] = '1'
			#barList = Popen(['sort', '-t', "' '", '-k', '2n,2', self.envDict['TSP_FILEPATH_BARCODE_TXT']], stdout=PIPE, env=self.envDict)
			barList = Popen(['/bin/bash', '-c', 'sort -t \' \' -k 2n,2 %s'%self.envDict['TSP_FILEPATH_BARCODE_TXT']], stdout=PIPE, env=self.envDict)
			barListOut = open(self.envDict['BARCODES_LIST'], 'w')
			barListOut.write(barList.communicate()[0])
			barListOut.close()
		
		# Remove previous results so that they aren't displayed.
		# (This variable just shows up so often...)
		T_plugin_dir = self.envDict['TSP_FILEPATH_PLUGIN_DIR']
		Popen(['rm', '-f', '%s/%s'%(T_plugin_dir, self.envDict['HTML_RESULTS']), '%s/%s'%(T_plugin_dir, '%s'%self.envDict['HTML_BLOCK']), '%s'%self.envDict['JSON_RESULTS'], '%s/*.bed'%T_plugin_dir], stdout=PIPE, env=self.envDict)
		Popen(['rm', '-rf', '%s/*.bam'%T_plugin_dir, '%s/dibayes*'%T_plugin_dir, self.envDict['PLUGIN_HS_ALIGN_DIR']], stdout=PIPE, env=self.envDict)
		Popen(['rm', '-f', '%s/hotspot*'%T_plugin_dir, '%s/variant*'%T_plugin_dir, '%s/allele*'%T_plugin_dir], stdout=PIPE, env=self.envDict)
		Popen(['rm', '-f', '%s/*_stats.txt'%T_plugin_dir, '%s/*.xls'%T_plugin_dir, '%s/*.log'%T_plugin_dir], stdout=PIPE, env=self.envDict)
		
		# Link bed files locally.
		self.envDict['PLUGIN_OUT_BEDFILE'] = self.envDict['INPUT_BED_FILE'][self.envDict['INPUT_BED_FILE'].rfind('/')+1:]
		self.envDict['PLUGIN_OUT_LOCI_BEDFILE'] = self.envDict['INPUT_SNP_BED_FILE'][self.envDict['INPUT_SNP_BED_FILE'].rfind('/')+1:]
		if (self.envDict['INPUT_BED_FILE'] is not None and self.envDict['INPUT_BED_FILE'] != ""):
			Popen(['ln', '-sf', self.envDict['INPUT_BED_FILE'], '%s/%s'%(T_plugin_dir, self.envDict['PLUGIN_OUT_BEDFILE'])], stdout=PIPE, env=self.envDict)
		if (self.envDict['INPUT_SNP_BED_FILE'] is not None and self.envDict['INPUT_SNP_BED_FILE'] != ""):
			Popen(['ln', '-sf', self.envDict['INPUT_SNP_BED_FILE'], '%s/%s'%(T_plugin_dir, self.envDict['PLUGIN_OUT_LOCI_BEDFILE'])], stdout=PIPE, env=self.envDict)
		
		# Process HotSpot BED file for left-alignment.
		if ((self.envDict['INPUT_SNP_BED_FILE'] is not None and self.envDict['INPUT_SNP_BED_FILE'] != "") and int(self.envDict['ENABLE_HOTSPOT_LEFT_ALIGNMENT']) == 1):
			Popen(['mkdir', '-p', self.envDict['PLUGIN_HS_ALIGN_DIR']], stdout=PIPE, env=self.envDict)
			Popen(['ln', '-sf', '%s/%s'%(T_plugin_dir, self.envDict['PLUGIN_OUT_LOCI_BEDFILE'])], stdout=PIPE, env=self.envDict)
			javaCMD = ['java', '-jar', '-Xmx1500m', '%s/LeftAlignBed.jar'%self.envDict['DIRNAME'], '%s/%s'%(self.envDict['PLUGIN_HS_ALIGN_DIR'], self.envDict['PLUGIN_OUT_LOCI_BEDFILE']), self.fenvDict['PLUGIN_HS_ALIGN_BED'], '%s/GenomeAnalysisTK.jar'%self.envDict['DIRNAME'], self.envDict['REFERENCE']]
			# If java fails because it lacks virtual memory, that error won't be trapped.
			try:
				javaOut = Popen(javaCMD, stdout=PIPE, env=self.envDict)
				alignWrite = open('%s/LeftAlignBed.log'%self.envDict['PLUGIN_HS_ALIGN_DIR'], 'w')
				alignWrite.write(javaOut.communicate()[0])
				alignWrite.close()
			except:
				self.envDict['PLUGIN_HS_ALIGN_BED'] = '%s/%s'%(T_plugin_dir, self.envDict['PLUGIN_OUT_LOCI_BEDFILE'])
			
		else:
			self.envDict['PLUGIN_HS_ALIGN_BED'] = '%s/%s'%(T_plugin_dir, self.envDict['PLUGIN_OUT_LOCI_BEDFILE'])

		# Create temporary BED file w/ X/Y ID targets.
		if (self.envDict['INPUT_BED_FILE'] is not (None or "")):
			# While awk does look interesting, I'm still converting it and sed to python.
			bedRead = open(self.envDict['INPUT_BED_FILE'], 'r')
			chromX = open(self.envDict['PLUGIN_CHROM_X_TARGETS'], 'w')
			chromY = open(self.envDict['PLUGIN_CHROM_Y_TARGETS'], 'w')
			noY = open(self.envDict['PLUGIN_CHROM_NO_Y_TARGETS'], 'w')
			tempLine = bedRead.readline()
			while (tempLine is not None and tempLine != ""):
				if (tempLine[0:4] == 'chrX'):
					chromX.write('%s\n'%tempLine)
				if (tempLine[0:4] == 'chrY'):
					chromY.write('%s\n'%tempLine)
				if (tempLine[0:4] != 'chrY'):
					noY.write('%s\n'%tempLine)
				tempLine = bedRead.readline()
			bedRead.close()
			chromX.close()
			chromY.close()
			noY.close()
		
		# Make links to js/css.
		Popen(['ln', '-sf', '%s/js'%self.envDict['DIRNAME']], stdout=PIPE, env=self.envDict)
		Popen(['ln', '-sf', '%s/css'%self.envDict['DIRNAME']], stdout=PIPE, env=self.envDict)
		
		# Run for barcodes or single page, whichever is appropriate.
		if (int(self.envDict['PLUGIN_CHECKBC']) == 1):
			# Export BC_TITLE/HELP vars; they're arrays, so we can't directly set them in the environment.
			for i in range(5):
				self.envDict['BCT%s'%i] = '%s'%self.BC_COL_TITLE[i]
				self.envDict['BCH%s'%i] = '%s'%self.BC_COL_HELP[i]
			barOut = Popen(['/bin/bash', '-c', 'source %s/html/barcode.sh; source %s/html/common.sh; source %s/html/head.sh; source %s/html/footer.sh; source %s/html/endJavascript.sh; source %s/html/fileLinks.sh; source %s/html/logo.sh; barcode'%(self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'])], stdout=PIPE, env=self.envDict)
			#barOut = Popen(['/bin/bash', '-c'], stdout=PIPE, env=self.envDict)
			barStd, barErr = barOut.communicate()
		else:
			# Run a whole bunch of shell scripts. Notice the semicolons, they're equivalent to '&&' in shell. Also, commands that are baked into bash (like source) seem to like being in one string. I'm not sure why; maybe 'bash -c' only reads the first arg it comes to.
			self.envDict['HTML'] = '%s/%s'%(T_plugin_dir, self.envDict['HTML_RESULTS'])
			# Maybe I should break this command up into several smaller ones...I just don't want to source everything all over again; the subprocess shells evaporate immediately after they finish without saving anything about their environment.
			# Still, more than three commands at once is just gratuitous.
			writeCmd = Popen(['/bin/bash', '-c', 'source %s/html/common.sh; source %s/html/head.sh; source %s/html/logo.sh; source %s/html/footer.sh; source %s/html/endJavascript.sh; write_html_header %s 15; display_static_progress %s; write_html_footer %s'%(self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['DIRNAME'], self.envDict['HTML'], self.envDict['HTML'], self.envDict['HTML'])], stdout=PIPE, env=self.envDict)
			writeOut, writeErr = writeCmd.communicate()

			# Run on single bam.
			sampleName = self.envDict['TSP_SAMPLE']

			callCmd = Popen(['%s/call_variants.sh'%self.envDict['SCRIPTSDIR'], self.envDict['PLUGIN_RUN_NAME'], self.envDict['TSP_FILEPATH_PLUGIN_DIR'], '.', self.envDict['TSP_FILEPATH_BAM']], stdout=PIPE, stderr=PIPE, env=self.envDict)
			writeOut, writeErr = callCmd.communicate()
			if (callCmd.returncode != 0):
				sys.stderr.write( 'Error occured in call_variants.sh:\n' )
				sys.stderr.write( writeErr )
				sys.exit(1)
			# Collect results for the detailed report.
			self.write_html_results(self.envDict['PLUGIN_RUN_NAME'], self.envDict['TSP_FILEPATH_PLUGIN_DIR'], '.', self.envDict['TSP_FILEPATH_BAM'],sampleName)
			
			# Write json output.
			tarCmd = Popen(['/bin/bash', '-c', 'source %s/html/common.sh; write_json_header 0; write_json_inner %s %s mapped_reads 2; write_json_comma; write_json_inner %s %s target_coverage 2'%(self.envDict['DIRNAME'], self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['PLUGIN_OUT_READ_STATS'], self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['PLUGIN_OUT_TARGET_STATS'])], stdout=PIPE, env=self.envDict)
			tarOut, tarErr = tarCmd.communicate()
			if (self.envDict['BC_HAVE_LOCI'] != '0'):
				hotCmd = Popen(['/bin/bash', '-c', 'source %s/html/common.sh; write_json_comma; write_json_inner %s %s hotspot_coverage 2'%(self.envDict['DIRNAME'], self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['PLUGIN_OUT_LOCI_STATS'])], stdout=PIPE, env=self.envDict)
				hotOut, hotErr = hotCmd.communicate()
			footCmd = Popen(['/bin/bash', '-c', 'source %s/html/common.sh; write_json_footer'%self.envDict['DIRNAME']], stdout=PIPE, env=self.envDict)
			footOut, footErr = footCmd.communicate()
			
			# Remove extra files.
			if (self.envDict['PLUGIN_DEV_KEEP_INTERMEDIATE_FILES'] == '0'):
				Popen(['rm', '-f', '%s/*_stats.txt %s/%s'%(self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['TSP_FILEPATH_PLUGIN_DIR'], self.envDict['HTML_ROWSUMS'])], stdout=PIPE, env=self.envDict)
		# Remove temporary files.
		Popen(['rm', '-f', self.envDict['PLUGIN_HS_ALIGN_DIR']], stdout=PIPE, env=self.envDict)
		Popen(['rm', '-f', self.envDict['BARCODES_LIST'], self.envDict['PLUGIN_CHROM_X_TARGETS'], self.envDict['PLUGIN_CHROM_Y_TARGETS'], self.envDict['PLUGIN_CHROM_NO_Y_TARGETS'], self.envDict['PLUGIN_OUT_HAPLOCODE']], stdout=PIPE, env=self.envDict)
			
	
	def launch(self, data=None):
		self.envDict['LIFEGRIDDIR'] = '%s/lifegrid'%self.envDict['DIRNAME']
		self.envDict['BC_SUM_ROWS'] = '5'
		self.envDict['COV_PAGE_WIDTH'] = '900px'
		self.envDict['BC_TITLE_INFO'] = 'Sample ID sequence and marker coverage summary statistics for barcoded aligned reads.'
		
		# Get json data from the default file provided.
		with open('startplugin.json', 'r') as fh:
			json_dat = json.load(fh)
		
		# Start the actual plugin.
		self.sampleID()
		
		# Exit gracefully.
		sys.exit(0)
	
	def output(self):
		pass
	
	def report(self):
		pass
	
	def metrics(self):
		pass
	
if __name__ == "__main__":
	PluginCLI(sampleID())
