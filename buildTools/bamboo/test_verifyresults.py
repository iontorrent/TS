#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import unittest
import sys, os, string
import subprocess
import glob
import time
from os import path

# get the current build number from the env var
buildnum = os.environ["BUILDNUM"]

# find the analysis output directories
pathname=""
for pathname in glob.glob("/results/analysis/output/Home/Batch_CAR-194-Cropped_*_Build_"+buildnum+"*"):
   path_car194=pathname
next

pathname=""
for pathname in glob.glob("/results/analysis/output/Home/Batch_B16-440-Cropped_*_Build_"+buildnum+"*"):
   path_b16440=pathname
next

pathname=""
for pathname in glob.glob("/results/analysis/output/Home/Batch_B10-33_Cropped_*_Build_"+buildnum+"*"):
   path_b1033=pathname
next

pathname=""
for pathname in glob.glob("/results/analysis/output/Home/Batch_B10-IonXpress_Cropped_*_Build_"+buildnum+"*"):
   path_b10x=pathname
next



class VerifyData_B10_IonXpress_Cropped(unittest.TestCase):

    def test_CheckForCoreFiles(self):
    	for (dir_path, dir_dirs, dir_files) in os.walk(path_b10x):
        	for name in dir_files:
		# suppress this false positive
            	    if ("QScore" in name):
                	continue
            	    self.assertFalse("core" in name,"***Core file found:"+path_b10x+"/"+name)

    def test_CheckFileList_Barcode(self):
        # iterate through a list of files, report on which ones were missing
        errorstatus=False
	errortext=" "
        list1=["barcodeMask.bin",
	       "barcodeFilter.txt",
	       "barcodeList.txt",
	       "alignment_barcode_summary.csv",
	       "bc_files/IonXpress_016_rawlib.sff",
	       "bc_files/IonXpress_016_rawlib.fastq",
	       "bc_files/alignment_nomatch.summary",
	       "bc_files/alignment_IonXpress_009.summary",
	       "bc_files/alignTable_IonXpress_013.txt",
	       "bc_files/IonXpress_001_rawlib.bam",
	       "bc_files/IonXpress_002_rawlib.bam.bai",
	       "bc_files/nomatch_rawlib.bam.bai",
 	       "bc_files/Default.sam.parsed",
	       "bc_files/Q10.histo.dat",
	       "bc_files/alignmentQC_out_IonXpress_013.txt"]
        for filename in list1:
                if ((os.path.exists(path_b10x+"/"+filename))==False):
			errorstatus=True
			errortext = errortext+","+filename
                        print "error: File "+filename+" was not found"
        next
        self.assertFalse(errorstatus,"Barcode output files are missing from the report directory"+errortext)

    def test_CheckReportLog(self):
	# test fails if bad words are found in reportlog	
	errorstatus=False        
	errortext=" "
	filename="ReportLog.html"	
	try:
            f = open(path_b10x+"/"+filename)
            for line in f:
            	lline=line.lower()
             	# look for bad words 
              	if("fail" in lline or "exception" in lline or "error" in lline
                	or "core dumped" in lline or "no such file" in lline or "permission" in lline
                        or "not a directory" in lline or "illegal" in lline or "traceback" in lline
                        or "bad exit" in lline or "aborted" in lline or "corrupt" in lline):	
			errorstatus=True
			errortext=errortext+" "+line
            f.close()
        except:
            self.assertTrue(False,"***ERROR: no output file was found for"+filename+" ***")
	self.assertFalse(errorstatus,"Errors found in "+filename+": "+errortext)

    def test_CheckDrmaaStderr(self):
	# test fails if bad words are found in drmaa_stderr_block.txt
	# TODO: test for duplicate TFs in the database	
	errorstatus=False        
	errortext=" "
	filename="drmaa_stderr_block.txt"	
	try:
            f = open(path_b10x+"/"+filename)
            for line in f:
            	lline=line.lower()
             	# look for bad words 
              	if("fail" in lline or "exception" in lline
                	or "core dumped" in lline or "no such file" in lline or "permission" in lline
                        or "not a directory" in lline or "illegal" in lline or "traceback" in lline
                        or "bad exit" in lline or "aborted" in lline or "corrupt" in lline):	
			errorstatus=True
			errortext=errortext+" "+line
            f.close()
        except:
            self.assertTrue(False,"***ERROR: no output file was found for"+filename+" ***")
	self.assertFalse(errorstatus,"Errors found in "+filename+": "+errortext)


    def test_Verify_uploadStatus(self):
        # iterate through list of strings, report on which ones were missing
        errorstatus=False
	errortext=" "
        list1=["Updating Analysis",
		"Adding TF Metrics",
		"Adding Analysis Metrics",
		"Adding Library Metrics",
		"Adding Quality Metrics"]
        for mystring in list1:
                if (subprocess.call("cat "+path_b10x+"/uploadStatus | grep '"+mystring+"'", shell=True)!=0):
			errorstatus=True
			errortext = errortext+","+mystring
                        print "error: upload status failed "+mystring
        next
        self.assertFalse(errorstatus,"upload status is missing something: "+errortext)




class VerifyBaseCallerArgs_CAR_194_Cropped(unittest.TestCase):
# check that the arguments provided to the BaseCaller are correct

    def test_BaseCallerArgs_trimadapter(self):
     	self.assertTrue(subprocess.call("cat "+path_car194+"/drmaa_stdout_block.txt | grep BaseCaller | grep 'trim-adapter ATCACCGACTGCCCATAGAGAGGCTGAGAC'", shell=True)==0,"basecaller trim adapter sequence not found?")

    def test_BaseCallerArgs_librarykey(self):
     	self.assertTrue(subprocess.call("cat "+path_car194+"/drmaa_stdout_block.txt | grep BaseCaller | grep 'librarykey=TCAG'", shell=True)==0,"basecaller library key sequence not found?")

    def test_BaseCallerArgs_tfkey(self):
     	self.assertTrue(subprocess.call("cat "+path_car194+"/drmaa_stdout_block.txt | grep BaseCaller | grep 'tfkey=ATCG'", shell=True)==0,"basecaller tfkey sequence not found?")

    def test_BaseCallerArgs_floworder(self):
     	self.assertTrue(subprocess.call("cat "+path_car194+"/drmaa_stdout_block.txt | grep BaseCaller | grep 'flow-order TACGTACGTCTGAGCATCGATCGATGTACAGC'", shell=True)==0,"basecaller floworder sequence not found?")

    def test_BaseCallerArgs_trimqualcutoff(self):
     	self.assertTrue(subprocess.call("cat "+path_car194+"/drmaa_stdout_block.txt | grep BaseCaller | grep 'trim-qual-cutoff 9'", shell=True)==0,"basecaller trim-qual-cutoff is incorrect?")

    def test_BaseCallerArgs_trimqualwindowsize(self):
     	self.assertTrue(subprocess.call("cat "+path_car194+"/drmaa_stdout_block.txt | grep BaseCaller | grep 'trim-qual-window-size 30'", shell=True)==0,"basecaller trimqualwindowsize is incorrect?")

    def test_BaseCallerArgs_trimadaptercutoff(self):
     	self.assertTrue(subprocess.call("cat "+path_car194+"/drmaa_stdout_block.txt | grep BaseCaller | grep 'trim-adapter-cutoff 16'", shell=True)==0,"basecaller trimadaptercutoff is incorrect?")

    def test_BaseCallerArgs_trimminreadlen(self):
     	self.assertTrue(subprocess.call("cat "+path_car194+"/drmaa_stdout_block.txt | grep BaseCaller | grep 'trim-min-read-len 5'", shell=True)==0,"basecaller trimminreadlen is incorrect?")



class VerifyData_CAR_194_Cropped(unittest.TestCase):

    def test_CheckForCoreFiles(self):
    	for (dir_path, dir_dirs, dir_files) in os.walk(path_car194):
        	for name in dir_files:
		# suppress this false positive
            	    if ("QScore" in name):
                	continue
            	    self.assertFalse("core" in name,"***Core file found:"+path_car194+"/"+name)

    def test_CheckReportLog(self):
	# test fails if bad words are found in reportlog	
	errorstatus=False        
	errortext=" "
	filename="ReportLog.html"	
	try:
            f = open(path_car194+"/"+filename)
            for line in f:
            	lline=line.lower()
             	# look for bad words 
              	if("fail" in lline or "exception" in lline or "error" in lline
                	or "core dumped" in lline or "no such file" in lline or "permission" in lline
                        or "not a directory" in lline or "illegal" in lline or "traceback" in lline
                        or "bad exit" in lline or "aborted" in lline or "corrupt" in lline):	
			errorstatus=True
			errortext=errortext+" "+line
            f.close()
        except:
            self.assertTrue(False,"***ERROR: no output file was found for"+filename+" ***")
	self.assertFalse(errorstatus,"Errors found in "+filename+": "+errortext)


    def test_CheckDrmaaStderr(self):
	# test fails if bad words are found in drmaa_stderr_block.txt
	# TODO: test for duplicate TFs in the database	
	errorstatus=False        
	errortext=" "
	filename="drmaa_stderr_block.txt"	
	try:
            f = open(path_car194+"/"+filename)
            for line in f:
            	lline=line.lower()
             	# look for bad words 
              	if("fail" in lline or "exception" in lline
                	or "core dumped" in lline or "no such file" in lline or "permission" in lline
                        or "not a directory" in lline or "illegal" in lline or "traceback" in lline
                        or "bad exit" in lline or "aborted" in lline or "corrupt" in lline):	
			errorstatus=True
			errortext=errortext+" "+line
            f.close()
        except:
            self.assertTrue(False,"***ERROR: no output file was found for"+filename+" ***")
	self.assertFalse(errorstatus,"Errors found in "+filename+": "+errortext)


    def test_CheckAnalysisLibraryKey(self):
     	self.assertTrue(subprocess.call("cat "+path_car194+"/drmaa_stdout_block.txt | grep 'Analysis command:' | grep 'librarykey=TCAG '", shell=True)==0,"Analysis library key sequence not found?")        

 

    def test_Verify_uploadStatus(self):
        # iterate through list of strings, report on which ones were missing
        errorstatus=False
	errortext=" "
        list1=["Updating Analysis",
		"Adding TF Metrics",
		"Adding Analysis Metrics",
		"Adding Library Metrics",
		"Adding Quality Metrics"]
        for mystring in list1:
                if (subprocess.call("cat "+path_car194+"/uploadStatus | grep '"+mystring+"'", shell=True)!=0):
			errorstatus=True
			errortext = errortext+","+mystring
                        print "error: upload status failed "+mystring
        next
        self.assertFalse(errorstatus,"upload status is missing something: "+errortext)



    def test_CheckFileList_Initialization(self):
        # iterate through a list of files, report on which ones were missing
        errorstatus=False
	errortext=" "
        list1=["primary.key",
		"manifest.txt",
		"ion_analysis_00.py",
		"expMeta.dat",
		"DefaultTFs.conf",
		"sysinfo.txt",
		"report_layout.json",
		"ReportConfiguration.txt",
		"parsefiles.php",
		"Default_Report.php",
		"job_list.txt",
		"ion_params_00.json"]
        for filename in list1:
                if ((os.path.exists(path_car194+"/"+filename))==False):
			errorstatus=True
			errortext = errortext+","+filename
                        print "error: File "+filename+" was not found"
        next
        self.assertFalse(errorstatus,"Output files are missing from the report directory"+errortext)

    def test_CheckFileList_Beadfind(self):
        # iterate through a list of files, report on which ones were missing
	# 3/27 removed debug files: separator.step-diff.txt,separator.reference_bf_t0.txt,separator.reference_t0.txt,separator.mix-model.txt,separator.avg-traces.txt,avgNukeTraceOld_TCAG.txt,avgNukeTraceOld_ATCG.txt,separator.summary.txt,separator.outlier-trace.txt,separator.outlier-ref.txt,separator.outlier-bg.txt,separator.mask.bin,separator.h5
        errorstatus=False
	errortext=" "
	list1=["separator.bftraces.txt",
		"separator.trace.txt",
		"separator.region-avg-traces.txt",
		"avgNukeTrace_TCAG.txt",
		"avgNukeTrace_ATCG.txt",
		"NucStep",
		"dcOffset"]
        for filename in list1:
                if ((os.path.exists(os.path.join(path_car194,"sigproc_results",filename)))==False):
			errorstatus=True
			errortext = errortext+","+filename
                        print "error: File "+filename+" was not found"
        next
        self.assertFalse(errorstatus,"Output files are missing from the report directory"+errortext)

    def test_CheckFileList_BkgModel(self):
        # iterate through a list of files, report on which ones were missing
        errorstatus=False
	errortext=" "
	# 3/22/12 removed BkgModel.json
        # 3/27/12 removed MaskBead.mask
	list1=["avgNukeTrace_TCAG.txt",
		"avgNukeTrace_ATCG.txt",
		"pinsPerFlow.txt",
		"BkgModelRegionData.0260.txt",
		"analysis.bfmask.bin",
		"BkgModelInitVals.0260.txt",
		"BkgModelEmptyTraceData.0260.txt",
		"BkgModelEmphasisData.0260.txt",
		"BkgModelDarkMatterData.0260.txt",
		"BkgModelBeadDcData.0260.txt",
		"BkgModelBeadData.0260.txt",
		"processParameters.txt",
		"pinsPerFlow.txt",
		"Bead_density_raw.png",
		"Bead_density_contour.png",
		"bfmask.stats",
		"analysis_return_code.txt",
		"1.wells"]
        for filename in list1:
                if ((os.path.exists(os.path.join(path_car194,"sigproc_results",filename)))==False):
			errorstatus=True
			errortext = errortext+","+filename
                        print "error: File "+filename+" was not found"
        next
        self.assertFalse(errorstatus,"Output files are missing from the report directory"+errortext)

    def test_CheckFileList_BaseCaller(self):
        # iterate through a list of files, report on which ones were missing
        errorstatus=False
	errortext=" "
	#   "basecaller_ppf_ssq.txt",
	#   3/25 removed processParameters_tmp_BaseCaller.txt
	list1=["readLen.txt",
		"rawtf.sff",
		"rawtf.bam",
		"rawlib.sff",
		"rawlib.fastq",
		"quality.summary",
		"keypass.summary",
		"DefaultTFs.fasta.fai",
		"DefaultTFs.fasta",
		"bfmask.bin",
		"readLenHisto.png",
		"BaseCaller.json",
		"TFStats.json",
		"TF.sam.parsed",
		"TF.alignTable.txt",
		"TF.alignment.summary",
		"raw_peak_signal",
		"iontrace_Test_Fragment.png",
		"unfiltered",
		"iontrace_Library.png",
		"Q7.histo.dat",
		"Q47.histo.dat",
		"Q20.histo.dat",
		"Q17.histo.dat",
		"Q10.histo.dat"]
        for filename in list1:
                if ((os.path.exists(os.path.join(path_car194,"basecaller_results",filename)))==False):
			errorstatus=True
			errortext = errortext+","+filename
                        print "error: File "+filename+" was not found"
        next
        self.assertFalse(errorstatus,"Output files are missing from the report directory"+errortext)

    def test_CheckFileList_Alignment(self):
        # iterate through a list of files, report on which ones were missing
        errorstatus=False
	errortext=" "
	list1=["Default.sam.parsed",
		"blockstatus.txt",
		"alignTable.txt",
		"alignment.summary",
		"alignmentQC_out.txt",
		"Filtered_Alignments_Q20.png",
		"Filtered_Alignments_Q47.png",
		"Filtered_Alignments_Q17.png",
		"Filtered_Alignments_Q10.png",
		"rawlib.bam.bai",
		"rawlib.bam",
		"version.txt"]
        for filename in list1:
                if ((os.path.exists(path_car194+"/"+filename))==False):
			errorstatus=True
			errortext = errortext+","+filename
                        print "error: File "+filename+" was not found"
        next
        self.assertFalse(errorstatus,"Output files are missing from the report directory"+errortext)

    def test_CheckFileList_Completion(self):
        # iterate through a list of files, report on which ones were missing
        errorstatus=False
	errortext=" "
	list1=["drmaa_stdout_block.txt",
		"drmaa_stderr_block.txt",
		"progress.txt",
		"uploadStatus",
		"ReportLog.html",
		"plugin_out",
		"pgm_logs.zip",
		"explog_final.txt",
		"drmaa_stdout.txt",
                "status.txt"]
        for filename in list1:
                if ((os.path.exists(path_car194+"/"+filename))==False):
			errorstatus=True
			errortext = errortext+","+filename
                        print "error: File "+filename+" was not found"
        next
        self.assertFalse(errorstatus,"Output files are missing from the report directory"+errortext)


if __name__ == "__main__":

    try:
        import xmlrunner
        test_runner = xmlrunner.XMLTestRunner(stream=sys.stdout,output='test-reports')

    except ImportError:
        test_runner = unittest.TextTestRunner(stream=sys.stdout)

    unittest.main(testRunner=test_runner)

