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
from celery.utils.log import get_task_logger
from iondb.rundb.data import dmactions_types
from iondb.rundb.models import DMFileStat, Results
from iondb.rundb.tasks import getdiskusage
from iondb.rundb.data import exceptions as DMExceptions

#Send logging to data_management log file
logger = get_task_logger('data_management')

ARCHIVE='archive'
EXPORT='export'
DELETE='delete'
TEST='donothingjustlog'


def delete(user, user_comment, dmfilestat):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    msg = "Deleting %s - %s Using v.%s" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName,dmfilestat.dmfileset.version)
    logger.info(msg)
    _update_related_objects(user, user_comment, dmfilestat, DELETE, msg)
    try:
        action_validation(dmfilestat, DELETE)
    except:
        raise
    try:
        if dmfilestat.dmfileset.type == dmactions_types.SIG:
            _process_fileset(dmfilestat, DELETE)
        else:
            if dmfilestat.dmfileset.type == dmactions_types.OUT:
                _create_archival_files(dmfilestat)
            _process_fileset(dmfilestat, DELETE)

        _emptydir_delete(dmfilestat)
        _action_complete_update(user, user_comment, dmfilestat, DELETE)
    except:
        dmfilestat.setactionstate('E')
        raise


def export(user, user_comment, dmfilestat, backup_directory=None):
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
        _process_fileset(dmfilestat, EXPORT)
        _action_complete_update(user, user_comment, dmfilestat, EXPORT)
    except:
        dmfilestat.setactionstate('E')
        raise


def archive(user, user_comment, dmfilestat, backup_directory=None):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    msg = "Archiving %s - %s Using v.%s" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName,dmfilestat.dmfileset.version)
    logger.info(msg)
    _update_related_objects(user, user_comment, dmfilestat, ARCHIVE, msg)
    try:
        action_validation(dmfilestat, ARCHIVE)
        destination_validation(dmfilestat, backup_directory)
        _create_destination(dmfilestat, ARCHIVE, dmfilestat.dmfileset.type, backup_directory)
    except:
        raise

    try:
        if dmfilestat.dmfileset.type == dmactions_types.SIG:
            _process_fileset(dmfilestat, ARCHIVE)
            #no longer need symlink from raw data to backup location to re-analyze
            #_create_symlink(dmfilestat)
        else:
            if dmfilestat.dmfileset.type == dmactions_types.OUT:
                _create_archival_files(dmfilestat)
            _process_fileset(dmfilestat, ARCHIVE)

        _emptydir_delete(dmfilestat)
        _action_complete_update(user, user_comment, dmfilestat, ARCHIVE)
    except:
        dmfilestat.setactionstate('E')
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
        if not os.path.exists(src_dir):
            raise DMExceptions.SrcDirDoesNotExist(src_dir)


def action_validation(dmfilestat, action):
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


def _file_selector(start_dir, ipatterns, epatterns, kpatterns):
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

    #find files matching keepwith filters from start_dir
    for root, dirs, files in os.walk(start_dir,topdown=True):
        #find files matching keep filters from start_dir
        for pattern in kpatterns:
            filter = re.compile(r'(%s/)(%s)' % (start_dir,pattern))
            for filename in files:
                kfile = os.path.join(root, filename)
                match = filter.match(kfile)
                if match:
                    to_keep.append(kfile)

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
            _file_removal(filepath, to_keep)
        except:
            raise
    elif action == EXPORT:
        try:
            _copy_to_dir(filepath,_start_dir, destination)
        except:
            raise
    elif action == ARCHIVE:
        try:
            _copy_to_dir(filepath,_start_dir, destination)
            _file_removal(filepath, to_keep)
        except:
            raise
    elif action == TEST:
        try:
            _print_selected(filepath, to_keep)
        except:
            raise


def _process_fileset(dmfilestat, action):
    '''All exceptions need to be handled here.  action_status needs to be
    updated correctly.
    To run in TEST mode - do nothing but print the selected files,
    python -c "import sys; import iondb.bin.djangoinit; from iondb.rundb import models; from iondb.rundb.data import dmactions; sys.stdout = open('stdout.txt', 'w');dmactions._process_fileset(models.DMFileStat.objects.get(id=5264),dmactions.TEST);"
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

    total_size = 0
    num_files_processed = 0

    #Determine if this file type is eligible to use a keep list
    kpatterns = _get_keeper_list(dmfilestat, action)

    #Create a list of files eligible to process
    for start_dir in search_dirs:
        logger.debug("Searching: %s" % start_dir)
        to_process = []
        if os.path.isdir(start_dir):
            to_process, to_keep = _file_selector(start_dir,
                                        dmfilestat.dmfileset.include,
                                        dmfilestat.dmfileset.exclude,
                                        kpatterns)
            logger.debug("%d files to process" % len(to_process))
            #process files in list
            for j, path in enumerate(to_process, start=1):

                try:
                    if not os.path.islink(path):
                        total_size += os.lstat(path)[6]

                    _process(path, action, dmfilestat.archivepath, start_dir, to_keep)
                    logger.debug("%04d %s %10d %s" % (j, action, total_size, path))
                    num_files_processed += 1

                except IOError as e:
                    #IOError: [Errno 28] No space left on device:
                    if e.errno == errno.ENOSPC:
                        raise
                    elif e.errno == errno.ENOENT:
                        logger.warn("%04d No longer exists %s" % (j,path))
                        continue
                except:
                    errmsg = "%04d %s %10d %s" % (j, action, total_size, path)
                    logger.error(errmsg)
                    logger.error(traceback.format_exc())

                if not action in [EXPORT,TEST] and dmfilestat.dmfileset.del_empty_dir:
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

            # Delete start_dir if it is empty: this is targeted towards raw data dirs
            if os.path.isdir(start_dir):
                if len(os.listdir(start_dir)) == 0:
                    os.rmdir(start_dir)
        else:
            logger.warn( "Directory '%s' does not exist" % start_dir)

    dmfilestat.diskspace = float(total_size)/(1024*1024)

    logger.info("%0.1f MB %s processed" % (dmfilestat.diskspace, dmfilestat.dmfileset.type))


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
    return


def _print_selected(filepath, to_keep):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)
    if filepath not in to_keep:
        logger.info("Selected: %s" % filepath)
    return


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
                if len(os.listdir(filepath)) == 0:
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
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)

    if action_state is None: action_state = dmfilestat.action_state

    if action != EXPORT:
        diskspace = update_diskspace(dmfilestat)

    if dmfilestat.dmfileset.type == dmactions_types.SIG:
        exp_id = dmfilestat.result.experiment_id

        DMFileStat.objects.filter(dmfileset__type=dmactions_types.SIG, result__experiment__id=exp_id) \
            .update(action_state = action_state, archivepath=dmfilestat.archivepath, diskspace=dmfilestat.diskspace)

        # check that experiment.ftpStatus field is now set to complete.
        dmfilestat.result.experiment.ftpStatus = "Complete"
        dmfilestat.result.experiment.save()
    else:
        dmfilestat.action_state = action_state
        dmfilestat.save()

    # add log entry
    msg = msg+"<br>User Comment: %s" % (user_comment)
    add_eventlog(dmfilestat, msg, user)

    # update diskusage on Experiment or Results if data was deleted/moved
    if action == ARCHIVE or action == DELETE:
        _update_diskusage(dmfilestat)


def _action_complete_update(user, user_comment, dmfilestat, action):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name)

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
