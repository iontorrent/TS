#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import unittest
import sys
import os
import subprocess
import glob
import time

# get the current build number from the env var
buildnum = os.environ["BUILDNUM"]


class SystemTest_CAR_194_Cropped(unittest.TestCase):
# tests run in alphabetical order, forcing them to run sequentially, this is bad "unit test" design...


    # test to verify that calling the launch script works
    def test_1_StartAnalysis(self):
        self.assertTrue(subprocess.call("sudo /opt/ion/iondb/bin/startanalysis_batch.py test_CAR-194-Cropped", shell=True)==0,"Batch launch script did not work")
    
    # test to wait for json file to appear in the output location
    def test_2_AnalysisStarted(self):
        # /results/analysis/output/Home/Batch_CAR-194-Cropped_18_Build_112_048
        global path_car194
	path_car194=""        

	while (path_car194==""):
    	   for pathname in glob.glob("/results/analysis/output/Home/Batch_CAR-194-Cropped_*_Build_"+buildnum+"*"):
       	      path_car194=pathname
    	   next
           print "waiting for report directory"
	   time.sleep(10)

	while (os.path.exists(path_car194+"/ion_params_00.json")==False):
	   print "waiting for ion_params_00.json"
	   time.sleep(10)

	self.assertTrue(True)   

    # wait for uploadStatus to appear, indicating the core pipeline is complete (but not plugins)
    def test_3_AnalysisCompleted(self):
        while (os.path.exists(path_car194+"/uploadStatus")==False):
	   print "waiting for uploadStatus"
           time.sleep(10)

        self.assertTrue(True)

    # wait for status.txt 
    def test_4_statustxt(self):
        while (os.path.exists(path_car194+"/status.txt")==False):
	   print "waiting for status.txt"
           time.sleep(10)

        self.assertTrue(True)



class SystemTest_B10_ionxpress_Cropped(unittest.TestCase):
# tests run in alphabetical order, forcing them to run sequentially, this is bad "unit test" design...

    # test to verify that calling the launch script works
    def test_1_StartAnalysis(self):
        self.assertTrue(subprocess.call("sudo /opt/ion/iondb/bin/startanalysis_batch.py test_B10-IonXpress_Cropped", shell=True)==0,"Batch launch script did not work")
    
    # test to wait for json file to appear in the output location
    def test_2_AnalysisStarted(self):
        global path_b10x
        path_b10x=""        
	while (path_b10x==""):
    	   for pathname in glob.glob("/results/analysis/output/Home/Batch_B10-IonXpress_Cropped_*_Build_"+buildnum+"*"):
       	      path_b10x=pathname
    	   next
	   print "waiting for report directory"
	   time.sleep(10)
	while (os.path.exists(path_b10x+"/ion_params_00.json")==False):
	   print "waiting for ion_params_00.json"
	   time.sleep(10)

	self.assertTrue(0==0)   

    # wait for uploadStatus to appear, indicating the core pipeline is complete (but not plugins)
    def test_3_AnalysisCompleted(self):
        while (os.path.exists(path_b10x+"/uploadStatus")==False):
	   print "waiting for uploadStatus"
           time.sleep(10)

        self.assertTrue(0==0)

    # wait for status.txt 
    def test_4_statustxt(self):
        while (os.path.exists(path_b10x+"/status.txt")==False):
	   print "waiting for status.txt"
           time.sleep(10)

        self.assertTrue(True)


if __name__ == "__main__":

    try:
        import xmlrunner
        test_runner = xmlrunner.XMLTestRunner(stream=sys.stdout,output='test-reports')

    except ImportError:
        test_runner = unittest.TextTestRunner(stream=sys.stdout)

    unittest.main(testRunner=test_runner)

