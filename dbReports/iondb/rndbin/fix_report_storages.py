#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import time
import datetime
from socket import gethostname
from django.core.exceptions import ObjectDoesNotExist

sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from iondb.rundb import models, views

def findReportStorage(self):
    '''Tries to determine the correct ReportStorage object by testing for
    a valid filesystem path using the reportLink path with the ReportStorage
    dirPath value.
    '''
    #List the reportStorage objects by newest objects first.
    #I am assuming that we want to use newest objects preferentially over older
    #which is what you might do if you are migrating from one disk to another
    storages = models.ReportStorage.objects.all().order_by('id').reverse()
    for storage in storages:
    
        loc = self.server_and_location()
    	if not loc:
            print "Try default location"
            locs = models.Location.objects.all().order_by('id')
            for loc in locs:
            	mypath = os.path.join(storage.dirPath, loc.name, self.resultsName+'_'+str(self.pk).zfill(3))
            	print "Trying: %s" % mypath
                if os.path.exists(mypath):
            		return storage
            print "there was an error getting loc"
            return None
            
        mypath = os.path.join(storage.dirPath, loc.name, self.resultsName+'_'+str(self.pk).zfill(3))
        if os.path.exists(mypath):
            return storage
    return None
    
if __name__ == '__main__':
    '''
    runs thru the database table of Results objects and determines if the
    current Report Storage object points to a valid filepath.  If it does not
    point to valid filepath, it searches all the Report Storage objects for
    a valid filepath and resets the Result's object Report Storage object.
    '''
    # Get all Results objects
    reports = models.Results.objects.all().order_by('id')

    for idx, report in enumerate(reports):
        #
        # Repair missing or incorrect ReportStorage object associated with Result
        #
        reportStore = findReportStorage(report)
        if reportStore is None:
            print "Crap!  Could not find this path: %s" % report.resultsName
            print "%s" % report.sffLink
        else:
            if report.reportstorage is None:
                print "%d added report storage object" % idx
                report.reportstorage = reportStore
                report.save()
            elif reportStore.dirPath != report.reportstorage.dirPath:
                print "%d corrected report storage object" % idx
                print "%s" % (report.reportstorage.dirPath)
                print "%s" % (reportStore.dirPath)
                report.reportstorage = reportStore
                report.save()
                print "SAVED A CHANGE"
            else:
                #print "report storage is okay"
                pass
            #
            # Repair incorrect report links
            #
            loc = models.Location.objects.get(defaultlocation=True)
            mypath = os.path.join(reportStore.webServerPath, loc.name, report.resultsName+'_'+str(report.pk))
            
            report.sffLink = os.path.join(mypath,os.path.basename(report.sffLink))
            report.fastqLink = os.path.join(mypath,os.path.basename(report.fastqLink))
            report.reportLink = os.path.join(mypath,os.path.basename(report.reportLink))
            report.tfSffLink = os.path.join(mypath,os.path.basename(report.tfSffLink))
            report.tfFastq = os.path.join(mypath,os.path.basename(report.tfFastq))
            #print report.sffLink
            #report.log =  os.path.join(mypath,os.path.basename(report.log))
            report.save();
            
    print "Total objects %d" % len(reports)
