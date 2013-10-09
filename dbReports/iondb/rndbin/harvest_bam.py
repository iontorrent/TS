#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

# Purpose: Given a TS Project name, get BAM files associated with each member Report.

import re
import sys
import shutil
import argparse
from datetime import datetime
import traceback

from iondb.bin.djangoinit import *
from iondb.rundb import models

logfilename=''

def main(project_names, destination):

    if destination == None:
        print "no destination specified"
    else:
        if not os.path.exists(destination):
            raise Exception("destination is invalid: %s" % destination)

        else:
            print "Copy location: %s" % destination

    project_space_total = 0 # record bytes required to copy Project files

    for project_name in project_names:

        logfilename = "prjlog__%s.csv" % project_name
        make_log_entry("Date: "+datetime.now().strftime("%Y-%m-%d"))
        make_log_entry("Project: %s" % project_name)
        make_log_entry("Project,Result,Version,BAM,BAI,Copied")
        print "Processing Project: %s" % project_name

        for result in get_desired_results(project_name):
            #print "Result Name: %s" % result.resultsName
            bampath,baipath = get_desired_filepaths(result)
            analysisVersion = get_analysis_version(result)
            if bampath is None or baipath is None:
                print "Error. BAM and/or BAI is missing"
                make_log_entry("%s,%s,%s,%s,%s" % (project_name,result.resultsName,analysisVersion,bampath,baipath))
                continue

            src_size = os.stat(bampath).st_size
            src_size += os.stat(baipath).st_size
            project_space_total += src_size

            if destination:
                dst_size = getSpaceB(destination)
                if dst_size > src_size:

                    dest_dir = os.path.join(destination,re.sub(r'\W+','-',project_name))
                    if not os.path.isdir(dest_dir):
                        os.makedirs(dest_dir)

                    print "Copying %d bytes to available %d bytes" %(src_size,dst_size)
                    faster = True
                    if faster:
                        try:
                            status = os.system("rsync -tLv %s %s" % (bampath, dest_dir))
                            if status:
                                print(sys.exc_info()[0])

                            status = os.system("rsync -tLv %s %s" % (baipath, dest_dir))
                            if status:
                                print(sys.exc_info()[0])

                        except Exception, err:
                            print(sys.exc_info()[0])
                            print(err)
                    else:
                        try:
                            shutil.copy2(bampath,dest_dir)
                            shutil.copy2(baipath,dest_dir)
                        except:
                            traceback.print_exc()
                    make_log_entry("%s,%s,%s,%s,%s,%s" % (project_name,result.resultsName,analysisVersion,bampath,baipath,dest_dir))
                else:
                    print "Error: Not enough disk space to complete"
            else:
                make_log_entry("%s,%s,%s,%s,%s" % (project_name,result.resultsName,analysisVersion,bampath,baipath))

    print "\nProject Requires %0.3f MB" % (float(project_space_total)/(1024*1024))


def make_log_entry(entry):
    with open(logfilename, "ab") as tracker:
        tracker.write(str(entry)+"\n")


def getSpaceB(drive_path):
    import statvfs
    s = os.statvfs(drive_path)
    freebytes = s[statvfs.F_BSIZE] * s[statvfs.F_BAVAIL]
    return freebytes


def get_desired_filepaths(result):
    loc = result.server_and_location()
    full_path = result.web_root_path(loc)
    bam_path = result.bamLink()
    full_path = os.path.join(full_path,'download_links',os.path.basename(bam_path))
    if os.path.exists(full_path):
        #print "Target BAM: %s" % full_path
        if os.path.exists(full_path+'.bai'):
            #print "Target BAI: %s" % full_path+'.bai'
            pass
        else:
            #print "ERROR.  Could not find %s" % full_path+'.bai'
            return (None,None)
    else:
        #print "ERROR.  Could not find %s" % full_path
        return (None,None)

    return (full_path,full_path+'.bai')


def get_analysis_version(report):
    def getElementContaining(theList, text):
        try:
            for item in theList.split(','):
                elements = item.split(':')
                if text == elements[0]:
                    return elements[1]
        except:
            pass
        return ''
    versionString = getElementContaining(report.analysisVersion, 'an')
    return versionString


def get_desired_results(project_name):
    '''
    Returns list of Results objects.
    Criteria is to get the most recent Result object for a given Project's list
    of results.  Since some Experiments will have more than one Result object, we
    need to get the most recent Result for each experiment.
    '''
    # Get the named Project object
    project = models.Project.objects.get(name=project_name)
    # Get all Result objects associated with this Project object
    results = models.Results.objects.filter(projects__name=project_name).order_by('-timeStamp')
    #print "%s has %d result objects" % (project.name,results.count())
    # Get all Experiment objects associated with list of Results objects
    experiment_names = []
    for result in results:
        if result.experiment.expName not in experiment_names:
            experiment_names.append(result.experiment.expName)
    #print "List of Experiments:"
    for experiment_name in experiment_names:
        try:
            #TODO: !!!!Probably missing something here - hacky
            exp_obj = models.Experiment.objects.filter(expName=experiment_name)[0]
            the_result = exp_obj.results_set.all().order_by("-timeStamp")[0]
            #print " %s - %s" % (experiment_name,the_result.resultsName)
            yield the_result
        except:
            print "Did not find a result object!!! for experiment %s" % (exp_obj.expName)
            pass

def print_project_list():
    project_list = []
    projects = models.Project.objects.all()
    print "\nAvailable Project Names:"
    for project in projects:
        print " %s" % project.name
        project_list.append(project.name)
    print ""
    return project_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve list of BAM files associated with given Project')
    parser.add_argument('projects',nargs='*',help="name of Project")
    parser.add_argument('-l','--list_projects',action='store_true',default=False,help='Lists available Project names')
    parser.add_argument('-d','--destination',default=None,help='Destination directory to copy to')

    args = vars(parser.parse_args())
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(0)

    if args['list_projects']:
        project_list = print_project_list()
        sys.exit(0)

    logfilename = "prjlog__%s.csv" % args['projects'][0]
    sys.exit(main(args['projects'],args['destination']))