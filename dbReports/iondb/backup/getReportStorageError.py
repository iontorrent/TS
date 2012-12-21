#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
#
#   This script will search the database Reports table entries for instances of
#   Reports that do nto have a ReportStorage object set.
#
import os
import sys
import subprocess

import iondb.bin.djangoinit
from iondb.rundb import models

if __name__ == '__main__':
    Reports = models.Results.objects.all().order_by('pk')
    for Report in Reports:
        if Report.reportstorage is None:
            print "* * * Report Storage is undefined!* * *"
            print ("Report Name: %s" % Report.resultsName)
            print ("Date: %s" % Report.timeStamp)

            reportpath = os.path.join("/results/analysis/output/Home", "%s_%03d" % (Report.resultsName, int(Report.pk)))
            if os.path.isdir(reportpath):
                print "Report Path exists: %s" % reportpath
                reportfile = os.path.join(reportpath, "Default_Report.php")
                if os.path.isfile(reportfile):
                    print "Report File exists: %s" % reportfile
                else:
                    print "Report File does not exist: %s" % reportfile
                    print "Either delete this entry from database, or set the ReportStorage for this run manually"
            else:
                print "Report Path does not exist: %s" % reportpath
            print ""
