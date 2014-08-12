#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import os
import socket
import argparse
from iondb.bin import djangoinit
from iondb.rundb.models import DMFileStat, Results
from iondb.rundb.data import dmactions_types as dm

class DMRealityCheck:
    """Give me the scoop on the DM system"""
    #Class variables
    archive_locations = []
    completed_only = False
    outfilename = "testcase.csv"
    
    #Class methods
    def set_outputfilename(self, _outfilename):
        self.outfilename = _outfilename

    
    def set_completed_only(self, _state):
        self.completed_only = _state

    
    def set_archive_location(self, _list_in):
        self.archive_locations += _list_in
        print "We will search the following locations: %s" % self.archive_locations
        
    
    def get_all_dmfilestats(self, report):
        dmfilestats = report.dmfilestat_set.all()
        return dmfilestats
    
    
    def get_all_reports(self):
        reports = Results.objects.all().order_by('timeStamp')
        return reports
    
            
    def do_some_test(self, dmfilestat):
        def sigproc_exists(_path):
            mypath = _path
            if not mypath:
                return False
            if os.path.isdir(mypath):
                if 'acq_0000.dat' in os.listdir(mypath):
                    return True
                mypath = os.path.join(mypath, 'thumbnail')
                if os.path.isdir(mypath):
                    if 'acq_0000.dat' in os.listdir(mypath):
                        return True
            return False
        
        
        def basein_exists(_path):
            if not _path:
                return False
            pgmpath = os.path.join(_path, 'sigproc_results')
            if os.path.isdir(pgmpath):
                # PGM use case
                if 'analysis.bfmask.stats' in os.listdir(pgmpath):
                    return True
            else:
                # Proton use case
                #onboard_results/sigproc_results/block_*/1.wells
                protonpath = os.path.join(_path, 'onboard_results', 'sigproc_results')
                if os.path.isdir(protonpath):
                    for root, dirs, files in os.walk(protonpath, topdown=True):
                        for name in files:
                            if "1.wells" == name:
                                return True
            return False
        
        
        def report_exists(_path):
            if not _path:
                return False
            if os.path.isdir(_path):
                if 'ion_params_00.json' in os.listdir(_path):
                    return True
            return False
        
        
        def intr_exists(_path):
            if not _path:
                return False
            mypath = os.path.join(_path)
            if os.path.isdir(mypath):
                if 'job_list.json' in os.listdir(mypath):
                    return True
            return False
    
        #======================================================================
        # Signal Processing Input 
        #======================================================================
        if dmfilestat.dmfileset.type == dm.SIG:
            if dmfilestat.action_state == 'L':
                if sigproc_exists(dmfilestat.result.experiment.expDir):
                    return True # Yes, its marked Local and files are in primary storage
                #Search archive locations for this dataset
                for arc_location in self.archive_locations:
                    expDir = os.path.basename(dmfilestat.result.experiment.expDir)
                    if sigproc_exists(os.path.join(arc_location, expDir)):
                        return 'Archived'   # Its marked Local but files are in archive storage
                return 'Not found'
            
            elif dmfilestat.action_state == 'AD':
                if sigproc_exists(dmfilestat.archivepath):
                    return True # Yes, its marked Archive and files are in archive storage
                #Search primary storage for this dataset
                if sigproc_exists(dmfilestat.result.experiment.expDir):
                    return "Local"
                return "Not found"
            
            elif dmfilestat.action_state == 'DD':
                if not(sigproc_exists(dmfilestat.result.experiment.expDir)):
                    #Search archive locations for this dataset
                    for arc_location in self.archive_locations:
                        expDir = os.path.basename(dmfilestat.result.experiment.expDir)
                        if sigproc_exists(os.path.join(arc_location, expDir)):
                            return 'Archived'   # Its marked Local but files are in archive storage
                return True # Yes, its marked Delete and files are not in primary storage or archive storage
                
            else:
                return dmfilestat.action_state
            
        #======================================================================
        # Basecalling Input
        #======================================================================
        elif dmfilestat.dmfileset.type == dm.BASE:
            if dmfilestat.action_state == 'L':
                if basein_exists(dmfilestat.result.get_report_dir()):
                    return True # Yes, its marked Local and files are in primary storage
                #Search archive locations for this dataset
                for arc_location in self.archive_locations:
                    resultDir = os.path.basename(dmfilestat.result.get_report_dir())
                    if basein_exists(os.path.join(arc_location, 'archivedReports', resultDir)):
                        return 'Archived'   # Its marked Local but files are in archive storage
                return 'Not found'
            
            elif dmfilestat.action_state == 'AD':
                if basein_exists(dmfilestat.archivepath):
                    return True # Yes, its marked Archive and files are in archive storage
                #Search primary storage for this dataset
                if basein_exists(dmfilestat.result.get_report_dir()):
                    return "Local"
                return "Not found"
            
            elif dmfilestat.action_state == 'DD':
                if not(basein_exists(dmfilestat.result.get_report_dir())):
                    #Search archive locations for this dataset
                    for arc_location in self.archive_locations:
                        resultDir = os.path.basename(dmfilestat.result.get_report_dir())
                        if basein_exists(os.path.join(arc_location, 'archivedReports', resultDir)):
                            return 'Archived'   # Its marked Local but files are in archive storage
                return True # Yes, its marked Delete and files are not in primary storage or archive storage
            
            else:
                return dmfilestat.action_state
        
        #======================================================================
        # Output Files
        #======================================================================
        elif dmfilestat.dmfileset.type == dm.OUT:
            if dmfilestat.action_state == 'L':
                if report_exists(dmfilestat.result.get_report_dir()):
                    return True # Yes, its marked Local and files are in primary storage
                #Search archive locations for this dataset
                for arc_location in self.archive_locations:
                    resultDir = os.path.basename(dmfilestat.result.get_report_dir())
                    if report_exists(os.path.join(arc_location, 'archivedReports', resultDir)):
                        return 'Archived'   # Its marked Local but files are in archive storage
                return 'Not found'
            
            elif dmfilestat.action_state == 'AD':
                if report_exists(dmfilestat.archivepath):
                    return True # Yes, its marked Archive and files are in archive storage
                #Search primary storage for this dataset
                if report_exists(dmfilestat.result.get_report_dir()):
                    return "Local"
                return "Not found"
            
            elif dmfilestat.action_state == 'DD':
                if not(report_exists(dmfilestat.result.get_report_dir())):
                    #Search archive locations for this dataset
                    for arc_location in self.archive_locations:
                        resultDir = os.path.basename(dmfilestat.result.get_report_dir())
                        if report_exists(os.path.join(arc_location, 'archivedReports', resultDir)):
                            return 'Archived'   # Its marked Deleted but files are in archive storage
                return True # Yes, its marked Delete and files are not in primary storage or archive storage
            
            else:
                return dmfilestat.action_state
            
        #======================================================================
        # Intermediate Files
        #======================================================================
        elif dmfilestat.dmfileset.type == dm.INTR:
            if dmfilestat.action_state == 'L':
                if intr_exists(dmfilestat.result.get_report_dir()):
                    return True # Yes, its marked Local and files are in primary storage
                #Search archive locations for this dataset
                for arc_location in self.archive_locations:
                    resultDir = os.path.basename(dmfilestat.result.get_report_dir())
                    if intr_exists(os.path.join(arc_location, 'archivedReports', resultDir)):
                        return 'Archived'   # Its marked Local but files are in archive storage
                return 'Not found'
            
            elif dmfilestat.action_state == 'AD':
                if intr_exists(dmfilestat.archivepath):
                    return True # Yes, its marked Archive and files are in archive storage
                #Search primary storage for this dataset
                if intr_exists(dmfilestat.result.get_report_dir()):
                    return "Local"
                return "Not found"
            
            elif dmfilestat.action_state == 'DD':
                if not(intr_exists(dmfilestat.result.get_report_dir())):
                    #Search archive locations for this dataset
                    for arc_location in self.archive_locations:
                        resultDir = os.path.basename(dmfilestat.result.get_report_dir())
                        if intr_exists(os.path.join(arc_location, 'archivedReports', resultDir)):
                            return 'Archived'   # Its marked Deleted but files are in archive storage
                return True # Yes, its marked Delete and files are not in primary storage or archive storage
            else:
                return dmfilestat.action_state
    
    
    def main(self):
        # List of DMFileStat objects
        # List by Reports, each file category: dbase action_status, filesystem verification
        fout = open(self.outfilename, "w")
        fout.write("Report Name,Status,SigProc,Verify,BaseIn,Verify,Report,Verify,Intr,Verify,Version,Raw Data Directory,Report Directory\n")
        print "%50sStatus\tsigproc\t\tbasein\t\treport\t\tintr" % ""
        err_reports = 0
        for count, report in enumerate(self.get_all_reports(),start=1):
            if self.completed_only:
                if report.status != "Completed":
                    err_reports += 1
                    continue
            msg_str = '(%d)' % (count)
            csv_data = "%s" % (report.resultsName)
            msg_str += ("%40s\t") % report.resultsName
            msg_str += ("%10s\t") % report.status
            csv_data += ",%s" % (report.status)
            for dmfilestat in self.get_all_dmfilestats(report):
                #msg_str += ("%s\t") % dmfilestat.dmfileset.type
                msg_str += ("%8s\t") % ([act[1] for act in DMFileStat.ACT_STATES if act[0] == dmfilestat.action_state][0])
            print msg_str
            dmfilestats = self.get_all_dmfilestats(report)
            for dmtype in dm.FILESET_TYPES:
                dmfilestat = [dmfilestat for dmfilestat in dmfilestats if dmfilestat.dmfileset.type == dmtype][0]
                csv_data = csv_data+",%s" % ([act[1] for act in DMFileStat.ACT_STATES if act[0] == dmfilestat.action_state][0])
                verify = self.do_some_test(dmfilestat)
                csv_data = csv_data+",%s" % verify
            #Hack - use final dmfilestat to grab the version.
            csv_data = csv_data+",%s" % (dmfilestat.dmfileset.version)
            # Record raw data directory
            csv_data = csv_data+",%s" % dmfilestat.result.experiment.expDir
            # Record report directory
            csv_data = csv_data+",%s" % dmfilestat.result.get_report_dir()
            if fout:
                fout.write(csv_data+"\n")
        fout.close()
        
        # Dump statistics to screen
        if err_reports:
            print ("%d out of %d Reports did not complete" % (err_reports, count))
        else:
            print ("%d Reports" % (count))        

if __name__ == '__main__':
    """
    Generates a report on the dbase status of dm file sets versus actual disposition of files in the filesystem
    Note the tests are fragile and can be confused with incomplete analyses.
    """
    parser = argparse.ArgumentParser(description="Generate a report of Data Management integrity")
    parser.add_argument('--archives',
                        help='Define archive location. comma delimited list')
    parser.add_argument('--completed',
                        action="store_true",
                        default=False,
                        help='Only include Reports that completed successfully')

    args = parser.parse_args()
    foo = DMRealityCheck()
    if args.archives:
        foo.set_archive_location(args.archives.split(','))
    foo.set_completed_only(args.completed)
    foo.set_outputfilename("dm_reality_%s.csv" % socket.gethostname())
    foo.main()