#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
from os import path
import sys
import time
import datetime
import statvfs
from socket import gethostname
from django.core.exceptions import ObjectDoesNotExist

import iondb.bin.djangoinit
from iondb.rundb import models

enable_progress_update = False
enable_long_report = False

def server_and_location(experiment):
    try:
        loc = models.Rig.objects.get(name=experiment.pgmName).location
    except:
        return None
    #server = models.FileServer.objects.filter(location=loc)
    return loc

def disk_space(path):
    '''
    Returns used and available disk capacity in gigabytes
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
    
    # Return gigabytes instead
    used = used / 1024 / 1024
    available = available / 1024 / 1024
    
    return used, available

def get_size(start, progress):
    total_size = 0
    i = 0
    #faster algorithm?
    for dirpath, dirnames, filenames in os.walk(start):
        for f in filenames:
            try:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
                if i >= 10000 and progress:
                    print "\n    Progress update: ~%s Gb" %(total_size/(1024*1024*1024)) + "\n    Current directory: %s" %dirpath
                    i = 0
                i+=1
            except:
                pass
    if progress:
        print "\n    Finished %s" %start + ", total size: %s bytes" %total_size + " = ~%s Gb\n" %(total_size/(1024*1024*1024))
    return total_size

def write_simple_report():
    rigSpace = 0
    freeSpace = 0
    totalSpace = 0
    
    allServers = models.FileServer.objects.all()
    validServers = []
    for server in allServers:
        if (os.path.isdir(server.filesPrefix)):
            validServers.append(server)
            #the following 3 lines may not be needed, since df calculates free/total space elsewhere.
            f = os.statvfs(server.filesPrefix)
            freeSpace += f[statvfs.F_BAVAIL]*f[statvfs.F_BSIZE]
            totalSpace += f[statvfs.F_BLOCKS]*f[statvfs.F_BSIZE]
    validServerPGMCombos = []
    allPGM = models.Rig.objects.all()
    for PGM in allPGM:
        for server in validServers:
            testpath = os.path.join(server.filesPrefix,PGM.name)
            if os.path.isdir(testpath):
                space = get_size(testpath, enable_progress_update)
                rigSpace += space
                if enable_progress_update:
                    print "\nPGM %s " %PGM.name + "on server %s " %server.filesPrefix + "contains %s bytes" %space + " = ~%sGb" %(space/(1024*1024*1024))
    
    analysisSpace = 0
    #(this is 'other space' without analysis or rig space removed)
    storageList = models.ReportStorage.objects.all()
    for storage in storageList:
        analysisSpace += get_size(storage.dirPath, enable_progress_update)
    
    
    #otherSpace = get_size('/results', True) - rigSpace - analysisSpace
    otherSpace = totalSpace - analysisSpace - rigSpace - freeSpace
    
    '''
    print "\nanalysis space: %s " %analysisSpace + " = %sGb" %(analysisSpace/(1024*1024*1024))
    print "\ntotal rig space: %s " %rigSpace + " = %sGb" %(rigSpace/(1024*1024*1024))
    print "\nother space: %s " %otherSpace + " = %sGb" %(otherSpace/(1024*1024*1024))
    print "\ntotal space used: %s " %(rigSpace+analysisSpace+otherSpace) + " = %sGb" %((rigSpace+analysisSpace+otherSpace)/(1024*1024*1024))
    print "\nfree space: %s" %freeSpace + " = %sGb" %(freeSpace/(1024*1024*1024))
    print "\ntotal space on disk: %s" %(totalSpace) + " = %sGb" %((totalSpace)/(1024*1024*1024))
    '''
    importantValues = [rigSpace, analysisSpace, otherSpace, freeSpace, totalSpace]
    return importantValues

def used_disk_space(path):
    '''
    path argument is a list of paths to check disk usage
    Return sum of all disk usage
    
    Warning: the du system command can take a long time to return when run
    on a large filesystem.  Don't know what to do about that yet.
    '''
    import commands
    def du (path):
        diskusage = commands.getstatusoutput("du -sk %s 2>/dev/null" % path)
        sum = int(diskusage[1].split()[0])
        return sum
    
    kbytes = 0
    for dir in path:
        if os.path.isdir (dir):
            kbytes += du (dir)
        
    # DEBUG PRINTOUT
    #print "Disk Size %s" % path
    #print "---------"
    #print str(kbytes) + " KBytes"
    #print str(int(kbytes) / 1024 / 1024) + " GB"
    
    # Return gigabytes instead
    kbytes = kbytes / 1024 / 1024
    
    return kbytes

def remote_server(path):
    import commands
    # strip trailing delimiter
    if path[-1:] == "/":
        path = path[0:-1]

    ret = commands.getstatusoutput("mount -l | grep \" %s \"" % path )
    if len(ret[1]) != 0:
        mountpoint = ret[1].split(':')[0]
    else:
        mountpoint = path
    return mountpoint

def getpercents(tds, uds, fds):
    '''
    Calculate a nice percentage that equals 100%
    '''
    # Used disk space
    percentuds = 100 * (float(uds)/float(tds))
    # Free disk space
    percentfds = 100 * (float(fds)/float(tds))
    
    return percentuds, percentfds

def raw_data_storage_report(exps):
    '''
    Summarizes state of Raw Data on the current server based on contents
    of the database
    '''
    
    # Filter list to include only experiments not yet backed-up or archived
    # This might take a while on large dbase?
    expDoesNotExist = False
    experiments = []
    for experiment in exps:
        try:
            bk = models.Backup.objects.get(backupName=experiment.expName)
        except ObjectDoesNotExist:
            experiments.append(experiment)
            expDoesNotExist = True
    
    # Get all backup objects to get number of runs already backed-up or archived
    backups = models.Backup.objects.all()
    
    # get backup versus deleted detail
    bknumarchived = backups.filter(isBackedUp__exact = True).count()
    bknumdeleted = backups.filter(isBackedUp__exact = False).count()
    
    # Make sublists based on storageOption (defined experiment.STORAGE_CHOICES)
    storageOptionsList = []
    if expDoesNotExist:
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
    
    # Get number of runs in Grace Period (3 days currently; check backup.py)
    timenow = time.localtime(time.time())
    cntGP = 0
    for experiment in experiments:
        diff = time.mktime(timenow) - time.mktime(datetime.date.timetuple(experiment.date))
        if diff < 84600*3: #from backup.py: build_exp_list function
            cntGP += 1

    report = []
    report.append('Raw Data Storage Report\n')
    report.append('Runs Total               :         %5d\n' % exps.count())
    report.append('Runs Deleted             :         %5d\n' % bknumdeleted)
    report.append('Runs Archived            :         %5d\n' % bknumarchived)
    report.append('Runs Live                :         %5d\n' % (exps.count() -  bknumdeleted - bknumarchived))
    for option in storageOptionsList:
        report.append("Runs to %-17s:         %5d\n" % (option[1], int(len(storeDict[option[0]]))))
    report.append("Runs in Grace Period     :         %5d\n" % cntGP)
    return report
    
def file_server_storage_report(exps):
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
            
    # get all report storages from dbase
    reportstorages = models.ReportStorage.objects.all()
    
    # get list of PGM names/directories from dbase
    # List is filtered for valid directories
    pgm_list = []
    pgms = models.Rig.objects.all()
    for pgm in pgms:
        for fileserver in validfileservers:
            testpath = os.path.join(fileserver.filesPrefix,pgm.name)
            if os.path.isdir(testpath):
                pgm_list.append (testpath)
    report = []
    for fileserver in validfileservers:
        uds, fds = disk_space(fileserver.filesPrefix)
        tds = uds + fds
        percentuds, percentfds = getpercents(tds, uds, fds)
        remotesrv = remote_server(fileserver.filesPrefix)
        #uds_reports = 0
        #for reportstorage in reportstorages:
        #    testpath = os.path.join(reportstorage.dirPath,fileserver.location.name)
        #    if os.path.isdir(testpath):
        #        uds_reports += used_disk_space ([testpath])
        #uds_rawdata = used_disk_space (pgm_list)
        #uds_other = tds - fds - uds_reports - uds_rawdata
        
        if enable_long_report:
            #vals will be a 5-element array containing the calculated disk usage values.
            vals = write_simple_report()
            uds_rawdata = vals[0]
            uds_reports = vals[1]
            uds_other = vals[2]
            #it looks like free/total space are already calculated by df, but freeSpace = vals[3] and totalSpace = vals[4] if they're needed. 
    
        # Print Report
        report.append("\n")
        report.append("Disk Space Allocation Report: %s (%s)\n" % (fileserver.filesPrefix, remotesrv))
        report.append("")
        if enable_long_report:
            print "Disk Usage by Datasets:      %12d GBytes" % (uds_rawdata/(1024*1024*1024))
            print "Disk Usage by Reports :      %12d GBytes" % (uds_reports/(1024*1024*1024))
            print "Disk Usage by Other   :      %12d GBytes" % (uds_other/(1024*1024*1024))
        report.append("Total Disk Space         :      %6d GBytes\n" % tds)
        report.append("Used Disk Space          :      %6d GBytes %.1f%%\n" % (uds, percentuds))
        report.append("Free Disk Space          :      %6d GBytes %.1f%%\n" % (fds, percentfds))
        report.append("\n")
    return report

def isMounted(server):
    import commands
    server = server.split(".")[0]
    ret = commands.getstatusoutput("mount -l | grep \"^%s\"" % server )
    if len(ret[1]) != 0:
        return True
    else:
        return False
        
def pgm_ftpserver_report(exps):
    '''Displays each PGM in database and the file server it writes to'''
    #Get all PGMs
    pgms = models.Rig.objects.all()
    
    # Make a dictionary of ftpservers and an array of PGM names associated with each
    serverDict = {}
    for pgm in pgms:
        ftpservername = pgm.ftpserver
        if ftpservername in serverDict:
            serverDict[ftpservername].append(pgm.name)
        else:
            serverDict[ftpservername] = [pgm.name]
    
    report = []
    report.append("\nFile servers and PGMs writing to them:\n")
    for server in sorted(serverDict):
        report.append("\n%s: %s\n" % (server,  "" if isMounted(server) else "(not mounted)"))
        for pgm in serverDict[server]:
            report.append("\t%s\n" % pgm)
    return report

class Experiment:
    def __init__(self, exp, name, date, star, storage_options, dir, location, pk):
        self.prettyname = exp.pretty_print()
        self.name = name
        self.date = date
        self.star = star
        self.store_opt = storage_options
        self.dir = dir
        self.location = location
        self.pk = pk

    def get_exp_path(self):
        return self.dir

    def is_starred(self):
        def to_bool(value):
            if value == 'True':
                return True
            else:
                return False
        return to_bool(self.star)

    def get_location(self):
        return self.location

    def get_folder_size(self):
        dir = self.dir
        dir_size = 0
        for (path, dirs, files) in os.walk(self.dir):
            for file in files:
                filename = os.path.join(path, file)
                dir_size += os.path.getsize(filename)
        return dir_size

    def get_exp_name(self):
        return self.name

    def get_pk(self):
        return self.pk

    def get_storage_option(self):
        return self.store_opt
    
    def get_prefix(self):
        '''This returns the first directory in the path'''
        temp = self.dir.split('/')[1]
        return '/'+temp

def dataset_disposition_report():
    exp = models.Experiment.objects.all().order_by('date')
     # local time from which to measure time difference
    timenow = time.localtime(time.time())

    # get all fileservers from dbase
    fileservers = models.FileServer.objects.all()
    validfileservers = []
    for fileserver in fileservers:
        if (os.path.isdir(fileserver.filesPrefix)):
            validfileservers.append(fileserver)
            
    cur_loc = models.Location.objects.all()[0]
    
    for server in validfileservers:
        serverPath = server.filesPrefix
        num = 30
        grace_period = 72
        experiments = []
        
        for e in exp:
            if len(experiments) < num: # only want to loop until we have the correct number
                
                location = server_and_location(e)
                if location == None:
                    continue
                
                diff = time.mktime(timenow) - time.mktime(datetime.datetime.timetuple(e.date))
                # grace_period units are hours
                if diff < (grace_period * 3600):    #convert hours to seconds
                    continue
                
                experiment = Experiment(e,
                                        str(e.expName),
                                        str(e.date),
                                        str(e.star),
                                        str(e.storage_options), 
                                        str(e.expDir),
                                        location,
                                        e.pk)
                            
                try:
                    # don't add anything to the list if its already archived
                    bk = models.Backup.objects.get(backupName=experiment.get_exp_name())
                    print "ALREADY ARCHIVED: %s" % experiment.get_exp_name()
                    continue 
                except:
                    # check that the path exists, and double check that its not marked to 'Keep'
                    
                    if not path.islink(experiment.get_exp_path()) and \
                            path.exists(experiment.get_exp_path()) and \
                            experiment.get_storage_option() != 'KI' and \
                            experiment.location==cur_loc and \
                            serverPath in experiment.dir:
                            #path.samefile(experiment.get_prefix(),serverPath):
                        experiments.append(experiment)
                    
                    if path.islink(experiment.get_exp_path()):
                        print "This is a link: %s" % experiment.get_exp_path()
                    if path.exists(experiment.get_exp_path()):
                        print "This path exists: %s" % experiment.get_exp_path()
                    #if not path.samefile(experiment.get_prefix(),serverPath):
                    if serverPath in experiment.dir:
                        print "experiment prefix and server are not the same %s %s" %(experiment.get_prefix(),serverPath)
                        print "experiment.dir = %s" % experiment.dir
                    if experiment.location==cur_loc:
                        print "current location matches"            
            else:
                pass
        print "Number of experiments %d" % len(experiments)
        for e in experiments:
            print e.name
            
def storage_report():
    # Get list of experiments from dbase
    exps = models.Experiment.objects.all()
    report_text = raw_data_storage_report(exps)
    report_text.extend(file_server_storage_report(exps))
    report_text.extend(pgm_ftpserver_report(exps))
    return report_text
    
def print_storage_report():
    print "FREAK"
    # Get date time stamp
    t = datetime.datetime.utcnow()
    epoch = time.mktime(t.timetuple())
    now = datetime.datetime.fromtimestamp(epoch)
    
    # Get site name
    sitename = models.GlobalConfig.objects.all().order_by('id')[0].site_name
    
    # print header
    print "Date %s" % now
    print "Site %s" % sitename
   
    # Get list of experiments from dbase
    exps = models.Experiment.objects.all()
    
    report_text = raw_data_storage_report(exps)
    for line in report_text:
        print line

    report_text = file_server_storage_report(exps)
    for line in report_text:
        print line
    
    #TODO: this is messy and output needs to be cleaned up
    #dataset_disposition_report()
    
    report_text = pgm_ftpserver_report(exps)
    print report_text

if __name__ == '__main__':
    '''
    Prints report containing file system usage data and dataset archiving
    disposition
    
    File paths are retrieved from the database
    
    fileservers - locations where raw data are stored
    reportstorages - locations where Reports are stored
    '''
    
    run = True
    
    if(len(sys.argv) > 1):
        for arg in sys.argv:
            if arg == "--prog-update":
                enable_progress_update = True
            elif arg == "--long-report":
                enable_long_report = True
            elif arg == "--help" or arg == "-h":
                print "\nCommands:\n\n    --prog-update:    Show progress updates. Useful for large file systems, only applies if --long-report is also sent." \
                +"\n\n    --long-report:    Enable long report. This will show how much used space is taken up by raw data, reports, and other data. It will take a long time on large file systems."
                run = False
    if run:
        for line in storage_report():
            sys.stdout.write(line)

