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

if __name__ == '__main__':
    #TODO: get all Reports sorted oldest to newest
    reports = models.Results.objects.all().order_by('id')
    
    #TODO: check each report link field for http://ioneast.iontorrent.com string
    for report in reports:
        #print 'Checking: %s' % report.resultsName
        if 'http://' in report.fastqLink:
            print "Erroneous entry: %s" % report.fastqLink
            #TODO: and remove that string from link
            #report.sffLink = report.sffLink.replace('http://ioneast.iontorrent.com','')
            #report.fastqLink = report.fastqLink.replace('http://ioneast.iontorrent.com','')
            #report.reportLink = report.reportLink.replace('http://ioneast.iontorrent.com','')
            #report.tfSffLink = report.tfSffLink.replace('http://ioneast.iontorrent.com','')
            #report.tfFastq = report.tfFastq.replace('http://ioneast.iontorrent.com','')
            #report.log = report.log.replace('http://ioneast.iontorrent.com','')  #this textfield, not char field
            #report.save()
            #print '\tFixed!'
    
