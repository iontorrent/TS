#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import unittest
import sys
import os
import subprocess


class TestCheckWebPages(unittest.TestCase):

    def test_ReferencesTab(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/references|grep http_code=200", shell=True)==0,"Browser Error: References tab failed to open")

    def test_AboutTab(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/about|grep http_code=200", shell=True)==0,"Browser Error: About tab failed to open")

    def test_PlanningTab(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/planning|grep http_code=200", shell=True)==0,"Browser Error: Planning tab failed to open")

    def test_RunsTab(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb|grep http_code=200", shell=True)==0,"Browser Error: Runs tab failed to open")

    def test_ServicesTab(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/servers|grep http_code=200", shell=True)==0,"Browser Error: Services tab failed to open")

    def test_ConfigTab(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/config|grep http_code=200", shell=True)==0,"Browser Error: Config tab failed to open")

    def test_ReportsTab(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/reports|grep http_code=200", shell=True)==0,"Browser Error: Reports tab failed to open")

    def test_AdminPage(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/admin|grep http_code=200", shell=True)==0,"Browser Error: Admin tab failed to open")

    def test_Mobile_RunsPage(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/mobile/runs/|grep http_code=200", shell=True)==0,"Browser Error: Mobile run page failed to open")

    def test_Mobile_ReportPage(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/mobile/report/33|grep http_code=200", shell=True)==0,"Browser Error: Mobile report failed to open")




class TestCheckAPI(unittest.TestCase):

    def test_TestJsonSanity(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/api/v1/results/?format=json|grep http_code=200", shell=True)==0,"API call failed")

    def test_CheckForTraceback(self):
        self.assertFalse(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/rundb/api/v1/results/?format=json|grep Traceback", shell=True)==0 ,"API call returned a Traceback error")

    def test_Django_SearchPage(self):
        self.assertTrue(subprocess.call("curl -L --write-out 'http_code=%{http_code}\n' --user ionadmin:ionadmin http://localhost/admin/rundb/results/?q=B14-40|grep http_code=200", shell=True)==0,"Browser Error: Django search failed")


if __name__ == "__main__":
    try:
        # Write JUnit compatible XML test reports if possible
        import xmlrunner
        test_runner = xmlrunner.XMLTestRunner(stream=sys.stdout,output='test-reports')
    
    except ImportError:
        test_runner = unittest.TextTestRunner(stream=sys.stdout)

    unittest.main(testRunner=test_runner)
