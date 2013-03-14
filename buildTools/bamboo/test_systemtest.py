#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import unittest
import sys
import os
import subprocess

class TestCheckProcesses(unittest.TestCase):

    def test_CheckProcessCrawler(self):
        processname = 'crawler.py'
        self.assertTrue(subprocess.call("ps ax | grep -v grep | grep "+processname, shell=True)==0,"Crawler process is not running")

    def test_CheckProcessJobServer(self):
        processname = 'serve.py'
        self.assertTrue(subprocess.call("ps ax | grep -v grep | grep "+processname, shell=True)==0,"JobServer process is not running")

    def test_CheckProcessArchive(self):
        processname = 'backup.py'
        self.assertTrue(subprocess.call("ps ax | grep -v grep | grep "+processname, shell=True)==0,"Archive process is not running")

    def test_CheckProcessPlugin(self):
        processname = 'ionPlugin.py'
        self.assertTrue(subprocess.call("ps ax | grep -v grep | grep "+processname, shell=True)==0,"Plugin process is not running")


class SystemTestDbreports(unittest.TestCase):

    # test the about link of the plugin, currently failing
    def test_PluginAboutLink(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/plugininput/21/?about=true|grep http_code=200", shell=True)==0,"Plugin About page is not working")

    # test case for TS-2868
    def test_SystemInfoPage(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/info|grep http_code=200", shell=True)==0,"info page is not working")



class VerifyInstallation(unittest.TestCase):


    def CheckPackageStatus(self):
        # iterate through a list of packagtes, check their status 
        errorstatus=False
        errortext=" "
        list1=["ion-dbreports",
		"ion-analysis",
		"ion-pipeline",
		"ion-alignment",
		"ion-gpu",
		"ion-publishers",
		"ion-torrentr",
		"ion-tsconfig",
		"ion-rndplugins",
		"ion-plugins",
                "tmap"]
        for packagename in list1:
                if (subprocess.call("dpkg -s "+packagename+" | grep Status | grep installed")==1):
                        errorstatus=True
                        errortext = errortext+","+packagename
                        print "error: Package "+packagename+" was not installed"
        next
        self.assertFalse(errorstatus,"Packages are not installed: "+errortext)




if __name__ == "__main__":
    try:
        import xmlrunner
        test_runner = xmlrunner.XMLTestRunner(stream=sys.stdout,output='test-reports')

    except ImportError:
        test_runner = unittest.TextTestRunner(stream=sys.stdout)

    unittest.main(testRunner=test_runner)

