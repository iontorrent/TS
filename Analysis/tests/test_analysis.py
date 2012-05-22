#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import unittest
import sys
import os
import subprocess

class Analysis_BasicTests(unittest.TestCase):

    
    def test_SanityCheck(self):
	# a very simple test to see if Analysis responds when called at the command line
        self.assertTrue(subprocess.call("Analysis | grep 'Command line = Analysis'", shell=True)==0,"Analysis failed to return info to stdout... compilation error?")

    def test_ReadDat_test1(self):
	# test if readDat returns the correct data for single col/row indices
	subprocess.check_call("readDat /results/PGM/B16-440-cropped/acq_0001.dat --col 0 --row 0 | tail -n 3 | md5sum > ~/readDat_test1.md5", shell=True)
        self.assertTrue(subprocess.call("diff ~/readDat_test1.md5 Analysis/tests/readDat_test1_baseline.md5", shell=True)==0,"readDat output did not match the baseline")

    def test_ReadDat_test2(self):
	# test if readDat returns the correct data for multiple col/row indices
	subprocess.check_call("readDat /results/PGM2/test_B10-33_cropped/acq_0005.dat --col 0,3 --row 5,2 | tail -n 3 | md5sum > ~/readDat_test2.md5", shell=True)
        self.assertTrue(subprocess.call("diff ~/readDat_test2.md5 Analysis/tests/readDat_test2_baseline.md5", shell=True)==0,"readDat output did not match the baseline")


class Basecaller_BasicTests(unittest.TestCase):

    def test_BasecallerSanityCheck(self):
	# a very simple test to see if Analysis responds when called at the command line
        self.assertTrue(subprocess.call("BaseCaller | grep 'Command line = BaseCaller'", shell=True)==0,"Analysis failed to return info to stdout... compilation error?")


if __name__ == "__main__":
    try:
        import xmlrunner
        test_runner = xmlrunner.XMLTestRunner(stream=sys.stdout,output='test-reports')

    except ImportError:
        test_runner = unittest.TextTestRunner(stream=sys.stdout)

    unittest.main(testRunner=test_runner)

