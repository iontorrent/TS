#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import os
import re
import sys
import time
import logging
import traceback
import subprocess
from celery.utils.log import get_task_logger

from iondb.bin import djangoinit
from iondb.rundb.models import Results, DMFileStat, DMFileSet
from iondb.rundb.data.dmactions_types import FILESET_TYPES

ARCHIVE = 'archive'
EXPORT = 'export'
DELETE = 'delete'
TEST = 'test'

logger = logging.getLogger('test_diskusage')
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
rothandle = logging.handlers.RotatingFileHandler("./test_logger", maxBytes=1024 * 1024 * 10, backupCount=5)
rothandle.setFormatter(fmt)
logger.addHandler(rothandle)
logger.propagate = True


def get_walk_filelist(dirs, save_dir=None):
    USE_OS_WALK=True
    USE_FILE_CACHE=True
    
    '''
    Purpose of the function is to generate a list of all files rooted in the given directories.
    Since the os.walk is an expensive operation on large filesystems, we can store the file list
    in a text file in the report directory.  Once a report is analyzed, the only potential changes
    to the file list will be in the plugin_out directory.  Thus, we always update the contents of
    the file list for that directory.
    '''
    import pickle
    import stat
    
    def dump_the_list(filepath, thelist):
        '''
        Write a file to report directory containing cached file list.
        Needs to have same uid/gid as directory with 0x666 permissions
        '''
        if filepath == "":
            return
        
        uid = os.stat(os.path.dirname(filepath)).st_uid
        gid = os.stat(os.path.dirname(filepath)).st_gid
        
        mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH  # 0o666
        umask_original = os.umask(0)
        
        with os.fdopen(os.open(filepath, os.O_WRONLY | os.O_CREAT, mode), 'w') as fh:
            pickle.dump(thelist, fh)
        os.chown(filepath, uid, gid)
        os.umask(umask_original)
        
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    logger.debug("%s USE_OS_WALK: %s USE_FILE_CACHE: %s" %(sys._getframe().f_code.co_name, USE_OS_WALK, USE_FILE_CACHE))
    
    starttime = time.time()
    thelist = []
    if save_dir:
        cachefile = os.path.join(save_dir, "cached.filelist")
    else:
        cachefile = ""
    
    if USE_FILE_CACHE and os.path.isfile(cachefile):
        # if file of cached list exists, read it
        with open(cachefile, "r") as fh:
            thelist = pickle.load(fh)
            
        #Remove all plugin_out contents from the list
        thelist = [item for item in thelist if 'plugin_out' not in item]
        
        #Update the plugins directory contents now.
        for item in [dir for dir in dirs if os.path.isdir(dir)]:
            this_dir = os.path.join(item, 'plugin_out')
            if os.path.isdir(this_dir):
                os.chdir(this_dir)
                if USE_OS_WALK:
                    for root, dirs, files in os.walk('./',topdown=True):
                        for j in dirs + files:
                            this_file = os.path.join(this_dir, root.replace('./',''), j)
                            if not os.path.isdir(this_file): # exclude directories
                                thelist.append(this_file)
                else:
                    cmd = 'find'
                    dirname = './'
                    proc = subprocess.Popen([cmd, dirname], stdout=subprocess.PIPE)
                    filename = proc.stdout.readline()
                    while filename != '':
                        this_file = os.path.join(item, this_dir.replace('./',''), filename.rstrip('\n'))
                        if not os.path.isdir(this_file): # exclude directories
                            thelist.append(this_file)
                        filename = proc.stdout.readline()
                    proc.communicate()
    else:
        # else, generate a list and save it
        for item in [dir for dir in dirs if os.path.isdir(dir)]:
            os.chdir(item)
            if USE_OS_WALK:
                for root, dirs, files in os.walk('./',topdown=True):
                    for j in dirs + files:
                        this_file = os.path.join(item, root.replace('./',''), j)
                        if not os.path.isdir(this_file): # exclude directories
                            thelist.append(this_file)
            else:
                cmd = 'find'
                dirname = './'
                proc = subprocess.Popen([cmd, dirname], stdout=subprocess.PIPE)
                filename = proc.stdout.readline()
                while filename != '':
                    this_file = os.path.join(item.replace('./',''), filename.rstrip('\n'))
                    if not os.path.isdir(this_file): # exclude directories
                        thelist.append(this_file)
                    filename = proc.stdout.readline()
                proc.communicate()
            
    endtime = time.time()    
    logger.info("%s: %f seconds" % (sys._getframe().f_code.co_name,(endtime-starttime)))
    
    if save_dir:
        dump_the_list(cachefile, thelist)
    
    return thelist


# OLD VERSION
def _file_selector_old(start_dir, ipatterns, epatterns, kpatterns, isThumbnail=False, add_linked_sigproc=False):
    '''Returns list of files found in directory which match the list of
    patterns to include and which do not match any patterns in the list
    of patterns to exclude.  Also returns files matching keep patterns in
    separate list.
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    starttime = time.time()  # debugging time of execution
    to_include = []
    to_exclude = []
    to_keep = []

    if not ipatterns: ipatterns = []
    if not epatterns: epatterns = []
    if not kpatterns: kpatterns = []

    #find files matching include filters from start_dir
    for root, dirs, files in os.walk(start_dir,topdown=True):
        if isThumbnail and 'onboard_results' in root:
            continue
            
        for pattern in ipatterns:
            filter = re.compile(r'(%s/)(%s)' % (start_dir,pattern)) #NOTE: use of start_dir, not root here
            for filename in files:
                remfile = os.path.join(root, filename)
                match = filter.match(remfile)
                if match:
                    to_include.append(remfile)

        #find files matching keep filters from start_dir
        for pattern in kpatterns:
            filter = re.compile(r'(%s/)(%s)' % (start_dir,pattern))
            for filename in files:
                kfile = os.path.join(root, filename)
                match = filter.match(kfile)
                if match:
                    to_keep.append(kfile)

        #export Basecalling Input: include linked sigproc_results of from-wells reports
        if add_linked_sigproc:
            sigproc_path = os.path.join(root, 'sigproc_results')
            real_start_dir = ''
            if 'sigproc_results' in dirs and os.path.islink(sigproc_path) and ('onboard_results' not in os.path.realpath(sigproc_path)):
                for sigproc_root, sigproc_dirs, sigproc_files in os.walk(os.path.realpath(sigproc_path),topdown=True):
                    if not real_start_dir: real_start_dir = os.path.dirname(sigproc_root)
                    for pattern in ipatterns:
                        filter = re.compile(r'(%s/)(%s)' % (real_start_dir,pattern))
                        for filename in sigproc_files:
                            testfile = os.path.join(sigproc_root, filename)
                            match = filter.match(testfile)
                            if match:
                                to_include.append(testfile.replace(real_start_dir, start_dir))
                                to_keep.append(testfile.replace(real_start_dir, start_dir))
                                logger.debug("HIE! %s" % testfile.replace(real_start_dir, start_dir))

    #find files matching exclude filters from include list
    for pattern in epatterns:
        filter = re.compile(r'(%s/)(%s)' % (start_dir,pattern))
        for filename in to_include:
            match = filter.match(filename)
            if match:
                to_exclude.append(filename)

    selected = list(set(to_include) - set(to_exclude))
    endtime = time.time()
    logger.info("%s(): %f seconds" % (sys._getframe().f_code.co_name,(endtime - starttime)))
    return selected, to_keep


# NEW VERSION
def _file_selector_new(start_dir, ipatterns, epatterns, kpatterns, isThumbnail=False, add_linked_sigproc=False, cached = None):
    '''Returns list of files found in directory which match the list of
    patterns to include and which do not match any patterns in the list
    of patterns to exclude.  Also returns files matching keep patterns in
    separate list.
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    starttime = time.time()  # debugging time of execution
    to_include = []
    to_exclude = []
    to_keep = []

    if not ipatterns: ipatterns = []
    if not epatterns: epatterns = []
    if not kpatterns: kpatterns = []            
        
    #find files matching include filters from start_dir
    for filepath in cached:
        # A thumbnail report should not associate with onboard, fullchip files
        if isThumbnail and 'onboard_results' in filepath:
            continue
            
        for pattern in ipatterns:
            filter = re.compile(r'(%s/)(%s)' % (start_dir,pattern)) #NOTE: use of start_dir, not root here
            match = filter.match(filepath)
            if match:
                to_include.append(filepath)

        #find files matching keep filters from start_dir
        for pattern in kpatterns:
            filter = re.compile(r'(%s/)(%s)' % (start_dir,pattern))
            match = filter.match(filepath)
            if match:
                to_keep.append(filepath)
        
        #export Basecalling Input: include linked sigproc_results of from-wells reports
        if 'sigproc_results' in filepath and 'onboard_results' not in os.path.realpath(filepath):
            real_start_dir = filepath
            while 1:
                real_start_dir, folder = os.path.split(real_start_dir)
                if folder == 'sigproc_results':
                    break
            for pattern in ipatterns:
                filter = re.compile(r'(%s/)(%s)' % (real_start_dir,pattern))
                match = filter.match(filepath)
                if match:
                    to_include.append(filepath)
                    to_keep.append(filepath)

    #find files matching exclude filters from include list
    for pattern in epatterns:
        filter = re.compile(r'(%s/)(%s)' % (start_dir,pattern))
        for filename in to_include:
            match = filter.match(filename)
            if match:
                to_exclude.append(filename)

    selected = list(set(to_include) - set(to_exclude))
    endtime = time.time()
    logger.info("%s(): %f seconds" % (sys._getframe().f_code.co_name,(endtime - starttime)))
    return selected, to_keep


def _get_keeper_list(dmfilestat, action):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    if action == EXPORT:
        kpatterns = []
    else:
        kpatterns = []
        #Are there entries in dmfilestat.dmfileset.keepwith?
        #logger.debug("FILES IN KEEPWITH FIELD")
        #logger.debug(dmfilestat.dmfileset.keepwith)
        for type, patterns in dmfilestat.dmfileset.keepwith.iteritems():
            #Are the types specified in dmfilestat.dmfileset.keepwith still local?
            if not dmfilestat.result.dmfilestat_set.get(dmfileset__type=type).isdisposed():
                #add patterns to kpatterns
                kpatterns.extend(patterns)
    logger.debug("Keep Patterns are %s" % kpatterns)
    return kpatterns


def update_diskspace(dmfilestat, cached = None):
    '''Warning: This can be a long-lived task on large partitions'''
    logger = get_task_logger('test_diskusage')  #xtra
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    logger.propagate = True # xtra
    try:
        # search both results directory and raw data directory
        search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]
        logger.info(search_dirs)
        total_size = 0

        #Determine if this file type is eligible to use a keep list
        kpatterns = _get_keeper_list(dmfilestat, '')

        #Create a list of files eligible to process
        isThumbnail = dmfilestat.result.metaData.get('thumb')==1
        for start_dir in search_dirs:            
            to_process = []
            if os.path.isdir(start_dir):
                logger.info("Start dir: %s" % start_dir)    # xtra
                if cached:
                    logger.debug("Running the NEW file_selector")
                    to_process, to_keep = _file_selector_new(start_dir,
                                                         dmfilestat.dmfileset.include,
                                                         dmfilestat.dmfileset.exclude,
                                                         kpatterns,
                                                         isThumbnail,
                                                         cached = cached)
                else:
                    logger.debug("Running the OLD file_selector")
                    add_linked_sigproc=False if (dmfilestat.dmfileset.type=='Intermediate Files') else True
                    to_process, to_keep = _file_selector_old(start_dir,
                                                         dmfilestat.dmfileset.include,
                                                         dmfilestat.dmfileset.exclude,
                                                         kpatterns,
                                                         isThumbnail,
                                                         add_linked_sigproc=add_linked_sigproc)
                #process files in list
                starttime = time.time()
                for j, path in enumerate(to_process, start=1):
                    try:
                        #logger.debug("%d %s %s" % (j, 'diskspace', path))
                        if not os.path.islink(path):
                            total_size += os.lstat(path)[6]

                    except Exception as inst:
                        errmsg = "Error processing %s" % (inst)
                        logger.error(errmsg)
                endtime = time.time()
                logger.info("Loop time: %f seconds" % (endtime-starttime))
        diskspace = float(total_size)/(1024*1024)
        # DEBUGGING ONLY
#        for item in to_process:
#            logger.debug("to_process:%s" % item)
#        for item in to_keep:
#            logger.debug("to_keep:%s" % item)
            
    except:
        logger.exception(traceback.format_exc())
        diskspace = 0

    dmfilestat.diskspace = diskspace
    dmfilestat.save()
    return diskspace
     
    
def main(resultpk):
    try:
        result = Results.objects.get(pk=resultpk)
        search_dirs = [result.get_report_dir(), result.experiment.expDir]
        if True:
            logger.debug("Testing the NEW code")
            cached_file_list = get_walk_filelist(search_dirs, save_dir=result.get_report_dir())
        else:
            logger.debug("Testing the OLD code")
            cached_file_list = None
        
        for category in FILESET_TYPES:
            logger.info("TYPE: %s" % (category))
            dmfilestat = result.get_filestat(category)
            update_diskspace(dmfilestat, cached=cached_file_list)
    except:
        raise

if __name__ == '__main__':
    resultpk = int(sys.argv[1])
    logger.info("Resultpk = %d" % (resultpk))
    sys.exit(main(resultpk))
