#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
import sys
import time
import datetime
from socket import gethostname
from django.core.exceptions import ObjectDoesNotExist

import iondb.bin.djangoinit
from iondb.rundb import models, views

def disk_space(path):
    '''
    Returns used and available disk capacity in kilobytes
    '''
    import commands
    total = commands.getstatusoutput("df -k %s 2>/dev/null" % path)
    used = int(total[1].split()[9])
    available = int (total[1].split()[10])
    kbytes = used + available
    
    # DEBUG PRINTOUT
    #print "Disk Capacity %s" % path
    #print "---------"
    #print str(kbytes) + " KBytes"
    #print str(int(kbytes) / 1024 / 1024) + " GB"
    
    return used, available

#def used_disk_space(path):
#    '''
#    path argument is a list of paths to check disk usage
#    Return sum of all disk usage
#    
#    Warning: the du system command can take a long time to return when run
#    on a large filesystem.  Don't know what to do about that yet.
#    '''
#    import commands
#    def du (path):
#        diskusage = commands.getstatusoutput("du -sk %s 2>/dev/null" % path)
#        sum = int(diskusage[1].split()[0])
#        return sum
#    
#    kbytes = 0
#    for dir in path:
#        if os.path.isdir (dir):
#            kbytes += du (dir)
#        
#    # DEBUG PRINTOUT
#    #print "Disk Size %s" % path
#    #print "---------"
#    #print str(kbytes) + " KBytes"
#    #print str(int(kbytes) / 1024 / 1024) + " GB"
#    
#    return kbytes

def getpercents(tds, uds, fds):
    '''
    Calculate a nice percentage that equals 100%
    '''
    # Used disk space
    percentuds = 100 * (float(uds)/float(tds))
    # Free disk space
    percentfds = 100 * (float(fds)/float(tds))
    
    return percentuds, percentfds

def raw_data_storage_report():
    '''
    Summarizes state of Raw Data on the current server based on contents
    of the database
    '''
    # Get list of experiments from dbase
    exps = models.Experiment.objects.all()
    
    # Filter list to include only experiments not yet backed-up or archived
    # This might take a while on large dbase?
    experiments = []
    for experiment in exps:
        try:
            bk = models.Backup.objects.get(backupName=experiment.expName)
        except ObjectDoesNotExist:
            experiments.append(experiment)
    
    # Get all backup objects to get number of runs already backed-up or archived
    backups = models.Backup.objects.all()
    
    # get backup versus deleted detail
    bknumarchived = backups.filter(isBackedUp__exact = True).count()
    bknumdeleted = backups.filter(isBackedUp__exact = False).count()
    
    # Make sublists based on storageOption (defined experiment.STORAGE_CHOICES)
    storageOptionsList = []
    for crap in experiments[0].STORAGE_CHOICES:
        storageOptionsList.append(crap)
    
    # Make a dictionary containing three dictionaries, 1 for each storage option
    list = []
    for option in storageOptionsList:
        list.append ([option[0],[]])
    storeDict = dict(list)
    
    # Get total disk usage for each sublist
    for option in storageOptionsList:
        for experiment in experiments:
            if option[0] == experiment.storage_options:
                storeDict[option[0]].append (experiment.expName)
    
    # get backupConfig object
    bk = models.BackupConfig.get()
    
    # Get number of runs in Grace Period (3 days currently; check backup.py)
    timenow = time.localtime(time.time())
    cntGP = 0
    for experiment in experiments:
        diff = time.mktime(timenow) - time.mktime(datetime.date.timetuple(experiment.date))
        if diff < (bk.grace_period * 3600): # convert hours to seconds
            cntGP += 1
    
    print ""
    print "Raw Data Storage Report"
    print ""
    print "Runs Total               :         %5d" % exps.count()
    print "Runs Deleted             :         %5d" % bknumdeleted
    print "Runs Archived            :         %5d" % bknumarchived
    print "Runs Live                :         %5d" % (exps.count() -  bknumdeleted - bknumarchived)
    for option in storageOptionsList:
        print "Runs to %-17s:         %5d" % (option[1], int(len(storeDict[option[0]])))
    print "Runs in Grace Period     :         %5d" % cntGP
    print ""
    
def file_server_storage_report():
    '''
    Reports storage allocation only using the df system command.
    Anything needing du is not returned because du takes forever on large
    filesystems.  Large being the typical NAS of ~19TB.
    '''
    #--------------------------------------------------------------------------
    #   Filesystem tools
    #--------------------------------------------------------------------------
    # Get disk usage for
    # Unused disk space on /results partition                           : Unused
    # Used space in /results/analysis/output                            : Reports space
    # Used space in /results/<pgm directories>                          : Raw Data space
    # Used space in NOT previous locations, "other" disk space usage    : Other space
    
    # get all fileservers from dbase
    fileservers = models.FileServer.objects.all()
    validfileservers = []
    for fileserver in fileservers:
        if (os.path.isdir(fileserver.filesPrefix)):
            validfileservers.append(fileserver)
    
    # get list of PGM names/directories from dbase
    # List is filtered for valid directories
    pgm_list = []
    pgms = models.Rig.objects.all()
    for pgm in pgms:
        for fileserver in validfileservers:
            testpath = os.path.join(fileserver.filesPrefix,pgm.name)
            if os.path.isdir(testpath):
                pgm_list.append (testpath)
                
    # get all report storages from dbase
    #reportstorages = models.ReportStorage.objects.all()
    
    for fileserver in validfileservers:
        uds, fds = disk_space(fileserver.filesPrefix)
        tds = uds + fds
        percentuds, percentfds = getpercents(tds, uds, fds)
        #uds_reports = 0
        #for reportstorage in reportstorages:
        #    testpath = os.path.join(reportstorage.dirPath,fileserver.location.name)
        #    if os.path.isdir(testpath):
        #        uds_reports += used_disk_space ([testpath])
        #uds_rawdata = used_disk_space (pgm_list)
        #uds_other = tds - fds - uds_reports - uds_rawdata
    
        # Print Report
        print ""
        print "Disk Space Allocation Report: %s" % fileserver.filesPrefix
        print ""
        #print "Disk Usage by Datasets:      %12d KBytes" % uds_rawdata
        #print "Disk Usage by Reports :      %12d KBytes" % uds_reports
        #print "Disk Usage by Other   :      %12d KBytes" % uds_other
        print "Total Disk Space      :      %12d KBytes" % tds
        print "Used Disk Space       :      %12d KBytes %.1f%%" % (uds, percentuds)
        print "Free Disk Space       :      %12d KBytes %.1f%%" % (fds, percentfds)
        print ""
 
if __name__ == '__main__':
    '''
    Prints report containing file system usage data and dataset archiving
    disposition
    
    File paths are retrieved from the database
    
    fileservers - locations where raw data are stored
    reportstorages - locations where Reports are stored
    '''
    
    # Get date time stamp
    t = datetime.datetime.utcnow()
    epoch = time.mktime(t.timetuple())
    now = datetime.datetime.fromtimestamp(epoch)
    
    # Get site name
    sitename = models.GlobalConfig.objects.all().order_by('id')[0].site_name
    
    # print header
    print "Date %s" % now
    print "Site %s" % sitename
    
    raw_data_storage_report()
    
    file_server_storage_report()
    
