#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
'''Print a list of Runs showing the ftpStatus field'''

import sys
import os
sys.path.append("/opt/ion/iondb")
from iondb.bin import djangoinit
from iondb.rundb import models
from django.db.models import Q

def main():
    '''main function'''
    
    #dbase query for list of Raw Data Experiments
    runs = models.Experiment.objects.all().order_by('date')
    totalRuns = len(runs)
    
    # Exclude Experiments marked Complete
    runs = runs.exclude(ftpStatus='Complete')
    transferringRuns = len(runs)
    
    print "Total Experiments: %5d" % totalRuns
    print "Transferring:      %5d" % transferringRuns
    
if __name__ == '__main__':
    sys.exit(main())