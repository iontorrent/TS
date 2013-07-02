#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import os
import re
import sys
import shutil
import errno
import tempfile
import traceback
import urllib
import subprocess
from datetime import datetime
from iondb.utils.files import getSpaceMB, getSpaceKB
from iondb.utils import makePDF
from ion.utils import makeCSA
from iondb.rundb.models import EventLog, DMFileStat
from iondb.utils.TaskLock import TaskLock
from celery.task import task
from celery.utils.log import get_task_logger
from iondb.rundb.data import dmactions_types
from iondb.rundb.models import DMFileStat, Results
from iondb.utils.files import getdiskusage
from iondb.rundb.data import exceptions as DMExceptions
from iondb.rundb.data.project_msg_banner import project_msg_banner

#Send logging to data_management log file
logger = get_task_logger('data_management')

ARCHIVE='archive'
EXPORT='export'
DELETE='delete'
TEST='test'


def delete(user, user_comment, dmfilestat, lockfile, msg_banner, confirmed=False):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    msg = "Deleting %s - %s Using v.%s" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName,dmfilestat.dmfileset.version)
    logger.info(msg)
    _update_related_objects(user, user_comment, dmfilestat, DELETE, msg)
    try:
        action_validation(dmfilestat, DELETE, confirmed)
    except:
        raise
    try:
        if dmfilestat.dmfileset.type == dmactions_types.SIG:
            _process_fileset_task(dmfilestat, DELETE, user, user_comment, lockfile, msg_banner)
        else:
            if dmfilestat.dmfileset.type == dmactions_types.OUT:
                _create_archival_files(dmfilestat)
            _process_fileset_task(dmfilestat, DELETE, user, user_comment, lockfile, msg_banner)

    except:
        dmfilestat.setactionstate('E')
        raise


def export(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory=None):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    msg = "Exporting %s - %s Using v.%s" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName,dmfilestat.dmfileset.version)
    logger.info(msg)
    _update_related_objects(user, user_comment, dmfilestat, EXPORT, msg)
    try:
        destination_validation(dmfilestat, backup_directory)
        _create_destination(dmfilestat, EXPORT, dmfilestat.dmfileset.type, backup_directory)
        _create_archival_files(dmfilestat)
    except:
        raise
    try:
        _process_fileset_task(dmfilestat, EXPORT, user, user_comment, lockfile, msg_banner)
    except:
        dmfilestat.setactionstate('E')
        raise


def archive(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory=None, confirmed=False):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    msg = "Archiving %s - %s Using v.%s" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName,dmfilestat.dmfileset.version)
    logger.info(msg)
    _update_related_objects(user, user_comment, dmfilestat, ARCHIVE, msg)
    try:
        action_validation(dmfilestat, ARCHIVE, confirmed)
        destination_validation(dmfilestat, backup_directory)
        _create_destination(dmfilestat, ARCHIVE, dmfilestat.dmfileset.type, backup_directory)
    except:
        raise

    try:
        if dmfilestat.dmfileset.type == dmactions_types.SIG:
            _process_fileset_task(dmfilestat, ARCHIVE, user, user_comment, lockfile, msg_banner)
        else:
            if dmfilestat.dmfileset.type == dmactions_types.OUT:
                _create_archival_files(dmfilestat)
            _process_fileset_task(dmfilestat, ARCHIVE, user, user_comment, lockfile, msg_banner)

    except:
        dmfilestat.setactionstate('E')
        raise


def test(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory=None):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    msg = "Testing %s - %s Using v.%s" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName,dmfilestat.dmfileset.version)
    logger.info(msg)
    try:
        _process_fileset_task(dmfilestat, TEST, user, user_comment, lockfile, msg_banner)
    except:
        raise


def update_diskspace(dmfilestat):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    try:
        # search both results directory and raw data directory
        search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]

        total_size = 0

        #Determine if this file type is eligible to use a keep list
        kpatterns = _get_keeper_list(dmfilestat, '')

        #Create a list of files eligible to process
        for start_dir in search_dirs:
            to_process = []
            if os.path.isdir(start_dir):
                to_process, to_keep = _file_selector(start_dir,
                                            dmfilestat.dmfileset.include,
                                            dmfilestat.dmfileset.exclude,
                                            kpatterns)

                #process files in list
                for j, path in enumerate(to_process, start=1):
                    try:
                        #logger.debug("%d %s %s" % (j, 'diskspace', path))
                        if not os.path.islink(path):
                            total_size += os.lstat(path)[6]

                    except Exception as inst:
                        errmsg = "Error processing %s" % (inst)
                        logger.error(errmsg)

        diskspace = float(total_size)/(1024*1024)
    except:
        logger.exception(traceback.format_exc())
        diskspace = 0

    dmfilestat.diskspace = diskspace
    dmfilestat.save()
    return diskspace


def destination_validation(dmfilestat, backup_directory=None, manual_action=False):
    '''
    Tests to validate destination directory:
    Does destination directory exist.
    Is there sufficient disk space.
    Write permissions.
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    if backup_directory in [None, 'None', '']:
        backup_directory = dmfilestat.dmfileset.backup_directory

    # check for valid destination
    try:
        if backup_directory in [None, 'None', '']:
            raise DMExceptions.MediaNotSet("Backup media for %s is not configured. Please use Data Management Configuration page." % dmfilestat.dmfileset.type)
        if not os.path.isdir(backup_directory):
            raise DMExceptions.MediaNotAvailable("Backup media for %s is not available: %s" % (dmfilestat.dmfileset.type, backup_directory))
    except Exception as e:
        logger.error("%s" % e)
        raise

    # check for sufficient disk space (units: kilobytes)
    if dmfilestat.diskspace is None:
        diskspace = update_diskspace(dmfilestat)
    else:
        diskspace = dmfilestat.diskspace

    try:
        freespace = getSpaceMB(backup_directory)
        pending = 0
        if manual_action:
            # add up all objects that will be processed before this one
            exp_ids = []
            for obj in DMFileStat.objects.filter(action_state__in=['AG','EG','SA','SE']):
                if obj.archivepath and not os.path.normpath(obj.archivepath).startswith(os.path.normpath(backup_directory)):
                    continue
                elif not os.path.normpath(obj.dmfileset.backup_directory).startswith(os.path.normpath(backup_directory)):
                    continue
                if obj.dmfileset.type == dmactions_types.SIG:
                    if obj.result.experiment_id not in exp_ids:
                        exp_ids.append(obj.result.experiment_id)
                        pending += obj.diskspace
                else:
                    pending += obj.diskspace

            logger.debug("Required %dMB, pending %dMB, free %dMB" % (diskspace, pending, freespace))

        if diskspace >= freespace:
            raise DMExceptions.InsufficientDiskSpace("Not enough space to write files at %s (free=%dMB)" % (backup_directory,freespace))
        elif diskspace >= (freespace - pending):
            raise DMExceptions.InsufficientDiskSpace("Not enough space to write files at %s (free=%dMB, pending=%dMB)" % (backup_directory,freespace, pending))
    except Exception as e:
        logger.debug("%s" % str(e))
        raise

    # check for write permission.  NOTE: this function is called by apache2 user while the action function
    # will be executed within a celery task which takes the uid/gid of celeryd process - in our case that is currently root.
    # This test is too restrictive.
    try:
        tempfile.TemporaryFile(dir=backup_directory).close()
    except Exception as e:
        if e.errno == errno.EPERM or errno.EACCES: # Operation not permitted
            errmsg = "Insufficient write permission in %s" % backup_directory
        else:
            errmsg = e
        logger.error(errmsg)
        raise DMExceptions.FilePermission(errmsg)

    # check for existence of source directory - needed first to create destination folder
    # and, of course, as source of files to copy.  but we check here instead of failing halfway
    # thru the setup.  This shows inconsistency between dmfilestat action_status and filesystem.
    if dmfilestat.dmfileset.type == dmactions_types.SIG:
        src_dir = dmfilestat.result.experiment.expDir
    else:
        src_dir = dmfilestat.result.get_report_dir()
    if not os.path.exists(src_dir):
        raise DMExceptions.SrcDirDoesNotExist(src_dir)


def action_validation(dmfilestat, action, confirmed=False):
    '''
    Tests to validate that this fileset can be acted upon.
    One test is to ensure that fileset files are not currently in use.
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)

    # Any non-blank string in dmfilestat.files_in_use fails the validation.
    # Manual actions must call data_management.update_files_in_use directly prior to validating for up-to-date info.
    if dmfilestat.files_in_use:
        errmsg = "%s files are currently being used by %s." % (dmfilestat.dmfileset.type, dmfilestat.files_in_use)
        logger.info(errmsg)    #For developers only, more informative error string
        errmsg = " Cannot complete request because the files to be deleted/archived are currently in use."
        raise DMExceptions.FilesInUse(errmsg)

    # Do not allow delete action when the filestat preserve flag is set
    if action == DELETE and dmfilestat.getpreserved():
        errmsg = "Unable to delete files: %s currently marked as Keep." % dmfilestat.dmfileset.type
        logger.warn(errmsg)
        raise DMExceptions.FilesMarkedKeep(errmsg)

    # Fail delete basecaller input if the files are being linked by any from-wells results
    # Manual action is allowed to proceed after user confirms it
    if action == DELETE and dmfilestat.dmfileset.type==dmactions_types.BASE and not confirmed:
        related = get_related_dmfilestats(dmfilestat)
        #make sure to exclude current dmfilestat in case itself is linked
        if related is not None: related = related.exclude(pk=dmfilestat.pk)
        if related is not None and related.count() > 0:
            errmsg = "Basecalling Input files are used by reanalysis started from BaseCalling: %s" % \
                ', '.join(related.values_list('result__resultsName',flat=True))
            logger.error(errmsg)
            raise DMExceptions.BaseInputLinked(errmsg)

def get_related_dmfilestats(dmfilestat):
    # returns queryset containing related dmfilestats or None
    related = None
    if dmfilestat.dmfileset.type == dmactions_types.SIG:
        related = DMFileStat.objects.filter(
            dmfileset__type=dmactions_types.SIG,
            result__experiment=dmfilestat.result.experiment
        )
    elif dmfilestat.dmfileset.type == dmactions_types.BASE:
        # find any newer results that have linked Basecaller Input (sigproc_results folder)
        related_results = Results.objects.filter(experiment=dmfilestat.result.experiment)
        sigproc_path = os.path.join(dmfilestat.result.get_report_dir(), 'sigproc_results')
        if related_results and os.path.exists(sigproc_path):
            linked = []
            for result in related_results:
                testpath = os.path.join(result.get_report_dir(), 'sigproc_results')
                try:
                    if os.path.islink(testpath) and os.path.samefile(sigproc_path, testpath):
                        linked.append(result.pk)
                except:
                    pass
            if len(linked) > 0:
                related = DMFileStat.objects.filter(
                    dmfileset__type=dmactions_types.BASE,
                    result__pk__in=linked
                )
    return related

def _create_archival_files(dmfilestat):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    try:
        makePDF.makePDF(dmfilestat.result_id)
        csaFullPath = makeCSA.makeCSA(dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir)
    except:
        logger.error("Could not create Report PDF or CSA")
        raise


def _create_destination(dmfilestat, action, filesettype, backup_directory=None):
    '''
    Create directory in destination directory to write files
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    if backup_directory is None:
        backup_directory = dmfilestat.dmfileset.backup_directory

    try:
        if filesettype == dmactions_types.SIG:
            src_dir = dmfilestat.result.experiment.expDir
            dest_dir = os.path.basename(src_dir)
            dmfilestat.archivepath = os.path.join(backup_directory,dest_dir)
        else:
            src_dir = dmfilestat.result.get_report_dir()
            dest_dir = os.path.basename(src_dir)
            if action == ARCHIVE:
                dmfilestat.archivepath = os.path.join(backup_directory,
                                                      'archivedReports',
                                                      dest_dir)
            elif action == EXPORT:
                dmfilestat.archivepath = os.path.join(backup_directory,
                                                      'exportedReports',
                                                      dest_dir)

        if not os.path.isdir(dmfilestat.archivepath):
            src_st = os.stat(src_dir)
            old_umask = os.umask(0000)
            os.makedirs(dmfilestat.archivepath)
            try:
                os.chown(dmfilestat.archivepath, src_st.st_uid, src_st.st_gid)
                os.utime(dmfilestat.archivepath, (src_st.st_atime, src_st.st_mtime))
            except Exception as e:
                if e.errno == errno.EPERM:
                    pass
                else:
                    raise
            os.umask(old_umask)
            logger.debug("Created dir: %s" % dmfilestat.archivepath)
    except:
        raise


def _file_selector(start_dir, ipatterns, epatterns, kpatterns, add_linked_sigproc=False):
    '''Returns list of files found in directory which match the list of
    patterns to include and which do not match any patterns in the list
    of patterns to exclude.  Also returns files matching keep patterns in
    separate list.
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)

    to_include = []
    to_exclude = []
    to_keep = []

    if not ipatterns: ipatterns = []
    if not epatterns: epatterns = []
    if not kpatterns: kpatterns = []

    #find files matching include filters from start_dir
    for root, dirs, files in os.walk(start_dir,topdown=True):
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

    #find files matching exclude filters from include list
    for pattern in epatterns:
        filter = re.compile(r'(%s/)(%s)' % (start_dir,pattern))
        for filename in to_include:
            match = filter.match(filename)
            if match:
                to_exclude.append(filename)

    for item in to_include:
        logger.debug("Inc: %s" % item)

    for item in to_keep:
        logger.debug("Keep: %s" % item)

    for item in to_exclude:
        logger.debug("Excl: %s" % item)

    selected = list(set(to_include) - set(to_exclude))

    return selected, to_keep


def _copy_to_dir(filepath,_start_dir,_destination):
    '''
    filepath is absolute path to the file to copy
    start_dir is directory to root the copy in the _destination dir
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    src = filepath
    dst = filepath.replace(_start_dir,'')
    dst = dst[1:] if dst[0] == '/' else dst
    dst = os.path.join(_destination,dst)
    try:
        os.makedirs(os.path.dirname(dst))
        src_dir_stat = os.stat(os.path.dirname(filepath))
        os.chown(os.path.dirname(dst),src_dir_stat.st_uid,src_dir_stat.st_gid)
    except OSError as exception:
        if exception.errno == errno.EEXIST or exception.errno == errno.EPERM:
            pass
        else:
            raise

    faster = True
    if faster:
        cmd = ["rsync","-tL",src,dst]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            logger.error(stderr)

    else:
        try:
        # copy file preserving permissions and owner and group
            shutil.copy2(src,dst)
            os.chown(dst,os.stat(src).st_uid,os.stat(src).st_gid)
        except (shutil.Error,OSError) as e:
            if e.errno == errno.EPERM:
                pass
            else:
                # When file is the same, its okay
                logger.warn("%s" % e)
        except:
            raise
    return True



def _process(filepath, action, destination, _start_dir, to_keep):
    '''
    Two basic operations: copy file, delete file.
    DELETE ACTION: delete file
    EXPORT ACTION: copy file
    ARCHIVE ACTION: copy file && delete file
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)

    if action == DELETE:
        try:
            return _file_removal(filepath, to_keep)
        except:
            raise
    elif action == EXPORT:
        try:
            return _copy_to_dir(filepath,_start_dir, destination)
        except:
            raise
    elif action == ARCHIVE:
        try:
            _copy_to_dir(filepath,_start_dir, destination)
            return _file_removal(filepath, to_keep)
        except:
            raise
    elif action == TEST:
        try:
            return _print_selected(filepath, to_keep)
        except:
            raise


def _process_fileset_task(dmfilestat, action, user, user_comment, lockfile, msg_banner):
    '''
    This function generates a list of files to process, then hands the list to a recursive
    celery task function.  The recursion continues until the list is empty.  The calling
    function exits immediately.
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)

    if dmfilestat.isdisposed():
        errmsg = "The %s for %s are deleted/archived" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName)
        logger.warn(errmsg)
        raise Exception(errmsg)
    else:
        if action == ARCHIVE:
            dmfilestat.setactionstate('AG')
        elif action == DELETE:
            dmfilestat.setactionstate('DG')
        elif action == EXPORT:
            dmfilestat.setactionstate('EG')

    # search both results directory and raw data directory
    search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]

    #Determine if this file type is eligible to use a keep list
    kpatterns = _get_keeper_list(dmfilestat, action)

    #Create a list of files eligible to process
    list_of_file_dict = []
    add_linked_sigproc=False if (action==DELETE or dmfilestat.dmfileset.type==dmactions_types.INTR) else True
    for start_dir in search_dirs:
        logger.debug("Searching: %s" % start_dir)
        if os.path.isdir(start_dir):
            to_process, to_keep = _file_selector(start_dir,
                                        dmfilestat.dmfileset.include,
                                        dmfilestat.dmfileset.exclude,
                                        kpatterns,
                                        add_linked_sigproc)
            logger.info("%d files to process at %s" % (len(list(set(to_process) - set(to_keep))),start_dir))
            list_of_file_dict.append(
                {
                    'pk':dmfilestat.id,
                    'action':action,
                    'archivepath':dmfilestat.archivepath,
                    'start_dir':start_dir,
                    'to_process':to_process,
                    'to_keep':to_keep,
                    'total_cnt':len(list(set(to_process) - set(to_keep))),
                    'processed_cnt':0,
                    'total_size':0,
                    'user':user,
                    'user_comment':user_comment,
                    'lockfile':lockfile,
                    'msg_banner':msg_banner,
                }
            )

    pfilename = set_action_param_file(list_of_file_dict)

    # Call the recursive celery task function to process the list
    celery_result = _process_task.delay(pfilename)

    return

@task(queue='periodic')
def _process_task(pfilename):
    '''
    Recursive celery task
    '''
    from datetime import datetime
    from datetime import timedelta
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    logger.debug("Task ID: %s" % _process_task.request.id)


    #catch all unhandled exceptions and clean up
    try:
        list_of_file_dict = get_action_param_file(pfilename)
        os.unlink(pfilename)

        dmfilestat = DMFileStat.objects.get(id=list_of_file_dict[0]['pk'])
        terminate = True   # flag to indicate recursion termination
        total_processed = 0

        start_time = datetime.now()
        max_time_delta = timedelta(seconds=10)

        # list_of_file_dict contains zero, one, or two dictionary variables to iterate over.
        for q,dict in enumerate(list_of_file_dict):
            # The dictionary contains an element named 'to_process' which is a list variable to iterate over
            logger.debug("%d, start_dir: %s" % (q,dict['start_dir']))

            while (datetime.now() - start_time) < max_time_delta:
                # If there are no files left to process, (all to_process lists are empty), the recursion ends
                if len(dict['to_process']) > 0:
                    terminate = False

                    try:
                        # process one file and remove entry from the list
                        path = dict['to_process'].pop(0)

                        j = dict['processed_cnt'] + 1

                        this_file_size = 0
                        if not os.path.islink(path):
                            this_file_size = os.lstat(path)[6]

                        if _process(path, dict['action'], dict['archivepath'], dict['start_dir'], dict['to_keep']):
                            dict['processed_cnt'] = j
                            dict['total_size'] += this_file_size
                            logger.info("%04d/%04d %s %10d %s" % (j, dict['total_cnt'], dict['action'], dict['total_size'], path))

                    except (OSError,IOError) as e:
                        #IOError: [Errno 28] No space left on device:
                        if e.errno == errno.ENOSPC:
                            raise
                        elif e.errno == errno.ENOENT:
                            logger.warn("%04d No longer exists %s" % (j,path))
                            continue
                    except:
                        errmsg = "%04d/%04d %s %10d %s" % (j, dict['total_cnt'], dict['action'], dict['total_size'], path)
                        logger.error(errmsg)
                        logger.error(traceback.format_exc())

                    if not dict['action'] in [EXPORT,TEST] and dmfilestat.dmfileset.del_empty_dir:
                        dir = os.path.dirname(path)
                        try:
                            if len(os.listdir(dir)) == 0:
                                if not "plugin_out" in dir:
                                    try:
                                        os.rmdir(dir)
                                        logger.debug("Removed empty directory: %s" % dir)
                                    except Exception as e:
                                        logger.warn("rmdir [%d] %s: %s" % (e.errno,e.strerror,dir))
                        except OSError as e:
                            if e.errno == errno.ENOENT:
                                logger.warn("del_empty_dir Does not exist %s" % (path))
                                continue
                            else:
                                raise e
                else:
                    break

            # only expect to execute this line when no files to process
            total_processed += dict['total_size']
    except:
        dmfilestat.setactionstate('E')
        logger.error("DM Action failure on %s for %s report." % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName))
        logger.error("This %s action will need to be manually completed." % (dict['action']))
        logger.error("The following is the exception error:\n"+traceback.format_exc())
        EventLog.objects.add_entry(dmfilestat.result,"%s - %s" % (dmfilestat.dmfileset.type, msg),username='dm_agent')
        if dict['lockfile']:
            applock = TaskLock(dict['lockfile'])
            applock.unlock()
        return

    #logger.debug("Sleep for 1")
    #import time
    #time.sleep(1)
    if terminate:
        try:
            # No more files to process.  Do the clean up.
            dmfilestat.diskspace = float(total_processed)/(1024*1024)
            dmfilestat.save()
            logger.info("%0.1f MB %s processed" % (dmfilestat.diskspace, dmfilestat.dmfileset.type))
            if dict['action'] in [ARCHIVE, DELETE]:
                _emptydir_delete(dmfilestat)

            _action_complete_update(dict['user'], dict['user_comment'], dmfilestat, dict['action'])

            # pop up a message banner
            if dict['msg_banner']:
                dmfileset = dmfilestat.dmfileset
                project_msg = {}
                msg_dict = {}
                msg_dict[dmfileset.type] = "Success"
                project_msg[dmfilestat.result_id] = msg_dict
                project_msg_banner('', project_msg, dict['action'])

            if dict['lockfile']:
                applock = TaskLock(dict['lockfile'])
                applock.unlock()
        except:
            logger.exception(traceback.format_exc())
    else:
        # Launch next task
        try:
            pfilename = set_action_param_file(list_of_file_dict)
            celery_result = _process_task.delay(pfilename)
        except:
            logger.error(traceback.format_exc())

    return


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


def _file_removal(filepath, to_keep):
    '''Conditional removal.  Only remove if the filepath is not in the to_keep list'''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    #if filepath matches any kpattern, do not remove
    if filepath not in to_keep:
        try:
            os.remove(filepath)
        except OSError as e:
            if e.errno == errno.EISDIR:
                #TODO: need to handle a cascading exception here
                os.rmdir(filepath)
            elif e.errno == errno.ENOENT:   # no such file or directory
                pass
            else:
                raise e
    else:
        logger.debug("NOT REMOVING: %s" % filepath)
        return False
    return True


def _print_selected(filepath, to_keep):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    if filepath not in to_keep:
        logger.debug("Selected: %s" % filepath)
        return True
    return False


def _emptydir_delete(dmfilestat):
    '''
    Search for empty directories and delete
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    # search both results directory and raw data directory
    search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]
    for start_dir in search_dirs:
        for root, dirs, files in os.walk(start_dir,topdown=True):
            for name in dirs:
                filepath = os.path.join(root,name)
                try:
                    emptyDir = True if len(os.listdir(filepath)) == 0 else False
                except:
                    continue
                if emptyDir:
                    logger.debug("Removing Directory: %s" % filepath)
                    #this fancy bit is to not trip on a soft link (TS-6600)
                    try:
                        os.rmdir(filepath)
                    except OSError as e:
                        if e.errno == errno.ENOTDIR:
                            try:
                                os.unlink(filepath)
                            except:
                                pass
                        elif e.errno == errno.ENOENT:   # no such file or directory
                            pass
                        else:
                            raise e


def _update_diskusage(dmfilestat):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    if dmfilestat.dmfileset.type == dmactions_types.SIG:
        exp = dmfilestat.result.experiment
        used = getdiskusage(exp.expDir)
        exp.diskusage = int(used) if used != None else 0
        exp.save()
    else:
        result = dmfilestat.result
        used = getdiskusage(result.get_report_dir())
        result.diskusage = int(used) if used != None else 0
        result.save()


def _update_related_objects(user, user_comment, dmfilestat, action, msg, action_state=None):
    '''When category is signal processing, make sure that all related results must be updated'''
    logger.info("Function: %s()" % sys._getframe().f_code.co_name)

    if action_state is None: action_state = dmfilestat.action_state

    if action != EXPORT:
        diskspace = update_diskspace(dmfilestat)

    if dmfilestat.dmfileset.type == dmactions_types.SIG:
        # check that experiment.ftpStatus field is now set to complete.
        dmfilestat.result.experiment.ftpStatus = "Complete"
        dmfilestat.result.experiment.save()
    else:
        dmfilestat.action_state = action_state
        dmfilestat.save()

    related = get_related_dmfilestats(dmfilestat)
    if related is not None:
        related.update(action_state = action_state, archivepath=dmfilestat.archivepath)
        if dmfilestat.dmfileset.type == dmactions_types.SIG:
            related.update(diskspace=dmfilestat.diskspace)

    # add log entry
    msg = msg+"<br>User Comment: %s" % (user_comment)
    add_eventlog(dmfilestat, msg, user)

    # update diskusage on Experiment or Results if data was deleted/moved
    if action == ARCHIVE or action == DELETE:
        _update_diskusage(dmfilestat)


def _action_complete_update(user, user_comment, dmfilestat, action):
    logger.info("Function: %s()" % sys._getframe().f_code.co_name)

    if action == ARCHIVE:
        action_state = 'AD'
        msg = "%0.1f MB %s archived to %s." % (dmfilestat.diskspace, dmfilestat.dmfileset.type, dmfilestat.archivepath)
    elif action == DELETE:
        action_state = 'DD'
        msg = "%0.1f MB %s deleted." % (dmfilestat.diskspace, dmfilestat.dmfileset.type)
    elif action == EXPORT:
        action_state = 'L'
        msg = "%0.1f MB %s exported to %s." % (dmfilestat.diskspace, dmfilestat.dmfileset.type, dmfilestat.archivepath)
        dmfilestat.archivepath = None
        dmfilestat.save()
    elif action == TEST:
        return

    _update_related_objects(user, user_comment, dmfilestat, action, msg, action_state)


def set_action_pending(user, user_comment, action, dmfilestat, backup_directory):
    if not dmfilestat.in_process():
        if action == ARCHIVE:
            dmfilestat.action_state = 'SA'
        elif action == EXPORT:
            dmfilestat.action_state = 'SE'
        msg = "%s for %s" % (dmfilestat.get_action_state_display(), dmfilestat.dmfileset.type)
        # temporarily store manual destination selection in archivepath
        if backup_directory is not None:
            dmfilestat.archivepath = backup_directory
            msg += " to %s" % backup_directory
        else:
            dmfilestat.archivepath = None
        dmfilestat.save()
        msg+= ".<br>User Comment: %s" % user_comment

        if dmfilestat.dmfileset.type == dmactions_types.SIG:
            # update all realated dmfilestats
            exp_id = dmfilestat.result.experiment_id
            DMFileStat.objects.filter(dmfileset__type=dmactions_types.SIG, result__experiment__id=exp_id) \
                .update(action_state = dmfilestat.action_state, archivepath=dmfilestat.archivepath)

        add_eventlog(dmfilestat, msg, user)
        return "scheduled"
    else:
        return dmfilestat.get_action_state_display()


def add_eventlog(dmfilestat, msg, username):
    if dmfilestat.dmfileset.type == dmactions_types.SIG:
        exp_id = dmfilestat.result.experiment_id
        for result in Results.objects.filter(experiment_id=exp_id):
            EventLog.objects.add_entry(result, msg, username=username)
    else:
        EventLog.objects.add_entry(dmfilestat.result, msg, username=username)


def _create_symlink(dmfilestat):
    '''
    If directory exists, check if its empty and remove it
    If directory does not exist, create link to archive location
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    if os.path.isdir(dmfilestat.result.experiment.expDir):
        if len(os.listdir(dmfilestat.result.experiment.expDir)) == 0:
            logger.debug("expDir exists and is empty: %s" % dmfilestat.result.experiment.expDir)
            os.rmdir(dmfilestat.result.experiment.expDir)
            os.symlink(dmfilestat.archivepath,dmfilestat.result.experiment.expDir)
            logger.debug("symlink created: %s" % dmfilestat.result.experiment.expDir)
        else:
            logger.debug("no symlink, expDir is not empty: %s" % dmfilestat.result.experiment.expDir)
    else:
        logger.debug("expDir no longer exists: %s" % dmfilestat.result.experiment.expDir)
        os.symlink(dmfilestat.archivepath,dmfilestat.result.experiment.expDir)
        logger.debug("symlink created: %s" % dmfilestat.result.experiment.expDir)


def slugify(something):
    '''convert whitespace to hyphen and lower case everything'''
    return re.sub(r'\W+','-',something.lower())


def set_action_param_file(list_of_dict_files):
    '''
    Argument is dictionary to be pickled.  Return value is name of file.
    '''
    from cPickle import Pickler
    import tempfile
    action = list_of_dict_files[0].get('action','unk')
    fileh = tempfile.NamedTemporaryFile(dir='/tmp',delete=False,mode='w+b',prefix=action)
    #fileh = open(fileh.name,'wb')
    pickle = Pickler(fileh)
    pickle.dump(list_of_dict_files)
    fileh.close()
    return fileh.name


def get_action_param_file(pfilename):
    '''
    Argument is name of file to unpickle.  Return value is dictionary value.
    '''
    from cPickle import Unpickler
    fileh = open(pfilename,'rb')
    pickle = Unpickler(fileh)
    list_of_dict_files = pickle.load()
    fileh.close()
    return list_of_dict_files
