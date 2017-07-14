#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from __future__ import absolute_import
import os
import sys
import time
import shutil
import errno
import traceback
import subprocess
from iondb.utils.files import getSpaceMB, is_mounted
from iondb.utils import makePDF
from ion.utils import makeCSA
from iondb.utils.TaskLock import TaskLock
from celery.task import task
from celery.utils.log import get_task_logger
from iondb.rundb.data import dmactions_types
from iondb.rundb.models import DMFileStat, DMFileSet, Results, EventLog
from iondb.rundb.data import exceptions as DMExceptions
from iondb.rundb.data.project_msg_banner import project_msg_banner
from iondb.rundb.data import dm_utils
from iondb.rundb.data.dmfilestat_utils import update_diskspace
from django.core import serializers

# Send logging to data_management log file
logger = get_task_logger('data_management')
logid = {'logid': "%s" % ('dmactions')}

ARCHIVE = 'archive'
EXPORT = 'export'
DELETE = 'delete'
TEST = 'test'


def delete(user, user_comment, dmfilestat, lockfile, msg_banner, confirmed=False):
    '''DM Action which deletes files'''
    logid = {'logid': "%s" % (lockfile)}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    msg = "Deleting %s - %s Using v.%s" % (
        dmfilestat.dmfileset.type, dmfilestat.result.resultsName, dmfilestat.dmfileset.version)
    logger.info(msg, extra=logid)
    _update_related_objects(user, user_comment, dmfilestat, DELETE, msg)
    try:
        action_validation(dmfilestat, DELETE, confirmed)
    except:
        raise
    try:
        if dmfilestat.dmfileset.type == dmactions_types.OUT:
            _create_archival_files(dmfilestat)
        filelistdict = _get_file_list_dict(dmfilestat, DELETE, user, user_comment, msg_banner)
        locallockid = "%s_%s" % (dmfilestat.result.resultsName, dm_utils.slugify(dmfilestat.dmfileset.type))
        set_action_state(dmfilestat, 'DG', DELETE)
        _process_fileset_task(filelistdict, lockfile, locallockid)
    except:
        set_action_state(dmfilestat, 'E', DELETE)
        raise


def export(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory=None):
    '''DM Action which copies files'''
    logid = {'logid': "%s" % (lockfile)}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    msg = "Exporting %s - %s Using v.%s" % (
        dmfilestat.dmfileset.type, dmfilestat.result.resultsName, dmfilestat.dmfileset.version)
    logger.info(msg, extra=logid)
    _update_related_objects(user, user_comment, dmfilestat, EXPORT, msg)
    try:
        destination_validation(dmfilestat, backup_directory)
        _create_destination(dmfilestat, EXPORT, dmfilestat.dmfileset.type, backup_directory)
    except:
        raise
    try:
        if dmfilestat.dmfileset.type == dmactions_types.OUT:
            _create_archival_files(dmfilestat)
        filelistdict = _get_file_list_dict(dmfilestat, EXPORT, user, user_comment, msg_banner)
        locallockid = "%s_%s" % (dmfilestat.result.resultsName, dm_utils.slugify(dmfilestat.dmfileset.type))
        set_action_state(dmfilestat, 'EG', EXPORT)
        _process_fileset_task(filelistdict, lockfile, locallockid)
        prepare_for_data_import(dmfilestat)
    except:
        set_action_state(dmfilestat, 'E', EXPORT)
        raise


def archive(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory=None, confirmed=False):
    '''DM Action which copies files then deletes them'''
    logid = {'logid': "%s" % (lockfile)}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    msg = "Archiving %s - %s Using v.%s" % (
        dmfilestat.dmfileset.type, dmfilestat.result.resultsName, dmfilestat.dmfileset.version)
    logger.info(msg, extra=logid)
    _update_related_objects(user, user_comment, dmfilestat, ARCHIVE, msg)
    try:
        action_validation(dmfilestat, ARCHIVE, confirmed)
        destination_validation(dmfilestat, backup_directory)
        _create_destination(dmfilestat, ARCHIVE, dmfilestat.dmfileset.type, backup_directory)
    except:
        raise

    try:
        if dmfilestat.dmfileset.type == dmactions_types.OUT:
            _create_archival_files(dmfilestat)
        filelistdict = _get_file_list_dict(dmfilestat, ARCHIVE, user, user_comment, msg_banner)
        locallockid = "%s_%s" % (dmfilestat.result.resultsName, dm_utils.slugify(dmfilestat.dmfileset.type))
        set_action_state(dmfilestat, 'AG', ARCHIVE)
        _process_fileset_task(filelistdict, lockfile, locallockid)
        prepare_for_data_import(dmfilestat)
    except:
        set_action_state(dmfilestat, 'E', ARCHIVE)
        raise


def test(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory=None):
    '''DM Action which prints to log file only.  Used for testing only'''
    logid = {'logid': "%s" % (lockfile)}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    msg = "Testing %s - %s Using v.%s" % (
        dmfilestat.dmfileset.type, dmfilestat.result.resultsName, dmfilestat.dmfileset.version)
    logger.info(msg, extra=logid)
    _update_related_objects(user, user_comment, dmfilestat, TEST, msg)
    try:
        filelistdict = _get_file_list_dict(dmfilestat, TEST, user, user_comment, msg_banner)
        locallockid = "%s_%s" % (dmfilestat.result.resultsName, dm_utils.slugify(dmfilestat.dmfileset.type))
        _process_fileset_task(filelistdict, lockfile, locallockid)
    except:
        raise


def destination_validation(dmfilestat, backup_directory=None, manual_action=False):
    '''
    Tests to validate destination directory:
    Does destination directory exist.
    Is there sufficient disk space.
    Write permissions.
    '''
    def _skipdiskspacecheck(directory):
        '''
        The hidden file .no_size_check should be placed into the root directory of the scratch drive mounted on the local
        system for the tape drive system.
        '''
        if os.path.exists(os.path.join(directory, ".no_size_check")):
            logger.info("%s: Exists: %s" %
                        (sys._getframe().f_code.co_name, os.path.join(directory, ".no_size_check")), extra=logid)
            return True
        else:
            logger.info("%s: Not Found: %s" %
                        (sys._getframe().f_code.co_name, os.path.join(directory, ".no_size_check")), extra=logid)
            return False

    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    if backup_directory in [None, 'None', '']:
        backup_directory = dmfilestat.dmfileset.backup_directory

    # check for valid destination
    try:
        if backup_directory in [None, 'None', '', '/']:
            raise DMExceptions.MediaNotSet(
                "Backup media for %s is not configured. Please use Data Management Configuration page." % dmfilestat.dmfileset.type)

        if not os.path.isdir(backup_directory):
            raise DMExceptions.MediaNotAvailable(
                "Backup media for %s is not a directory: %s" % (dmfilestat.dmfileset.type, backup_directory))

        # check if destination is external filesystem
        # ie, catch the error of writing to a mountpoint which is unmounted.
        # Use the tool 'mountpoint' which returns 0 if its mountpoint and mounted
        # If its a subdirectory to a mountpoint, then the isdir() test will fail when not mounted.
        if not is_mounted(backup_directory):
            raise DMExceptions.MediaNotAvailable(
                "Backup media for %s is not mounted: %s" % (dmfilestat.dmfileset.type, backup_directory))

    except Exception as e:
        logger.error("%s" % e, extra=logid)
        raise

    # check for sufficient disk space (units: kilobytes)
    if _skipdiskspacecheck(backup_directory):
        logger.warn("%s - skipping destination disk space check" %
                    (sys._getframe().f_code.co_name), extra=logid)
        pass
    else:
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
                for obj in DMFileStat.objects.filter(action_state__in=['AG', 'EG', 'SA', 'SE']):
                    try:
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
                    except:
                        pass

                logger.debug("Required %dMB, pending %dMB, free %dMB" %
                             (diskspace, pending, freespace), extra=logid)

            if diskspace >= freespace:
                raise DMExceptions.InsufficientDiskSpace(
                    "Not enough space to write files at %s (free=%dMB)" % (backup_directory, freespace))
            elif diskspace >= (freespace - pending):
                raise DMExceptions.InsufficientDiskSpace(
                    "Not enough space to write files at %s (free=%dMB, pending=%dMB)" % (backup_directory, freespace, pending))
        except Exception as e:
            logger.debug("%s" % str(e), extra=logid)
            raise

    # check for write permission.
    cmd = ['sudo', '/opt/ion/iondb/bin/sudo_utils.py', 'check_write_permission', backup_directory]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderrdata = process.communicate()
    except Exception as err:
        logger.debug("Sub Process execution failed %s" % err, extra=logid)

    if process.returncode:
        raise DMExceptions.FilePermission(stderrdata)

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
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    # Any non-blank string in dmfilestat.files_in_use fails the validation.
    # Manual actions must call data_management.update_files_in_use directly
    # prior to validating for up-to-date info.
    if dmfilestat.files_in_use:
        errmsg = "%s files are currently being used by %s." % (
            dmfilestat.dmfileset.type, dmfilestat.files_in_use)
        logger.info(errmsg, extra=logid)  # For developers only, more informative error string
        errmsg = " Cannot complete request because the files to be deleted/archived are currently in use."
        raise DMExceptions.FilesInUse(errmsg)

    # Do not allow delete action when the filestat preserve flag is set
    if action == DELETE and dmfilestat.getpreserved():
        errmsg = "Unable to delete files: %s currently marked as Keep." % dmfilestat.dmfileset.type
        logger.warn(errmsg, extra=logid)
        raise DMExceptions.FilesMarkedKeep(errmsg)

    # Fail delete basecaller input if the files are being linked by any from-wells results
    # Manual action is allowed to proceed after user confirms it
    if action == DELETE and dmfilestat.dmfileset.type == dmactions_types.BASE and not confirmed:
        related = get_related_dmfilestats(dmfilestat)
        # make sure to exclude current dmfilestat in case itself is linked
        if related is not None: related = related.exclude(pk=dmfilestat.pk)
        if related is not None and related.count() > 0:
            errmsg = "Basecalling Input files are used by reanalysis started from BaseCalling: %s" % \
                ', '.join(related.values_list('result__resultsName', flat=True))
            logger.error(errmsg, extra=logid)
            raise DMExceptions.BaseInputLinked(errmsg)


def get_related_dmfilestats(dmfilestat):
    '''Returns related dmfilestats:
        Signal Processing category - all SIG dmfilestats for this experiment
        Basecalling Input catagory - BASE dmfilestats that have started from-basecalling for this result
    '''
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
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    # This check is needed for cases where src dir has been manually deleted
    report_dir = dmfilestat.result.get_report_dir()
    if not os.path.exists(report_dir):
        raise DMExceptions.SrcDirDoesNotExist(report_dir)

    try:
        makePDF.write_summary_pdf(dmfilestat.result_id)
    except:
        logger.error("Could not create Report PDF", extra=logid)
        raise

    try:
        makeCSA.makeCSA(dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir)
    except:
        logger.error("Could not create CSA", extra=logid)
        raise


def _create_destination(dmfilestat, action, filesettype, backup_directory=None):
    '''
    Create directory in destination directory to write files
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    if backup_directory is None:
        backup_directory = dmfilestat.dmfileset.backup_directory

    try:
        if filesettype == dmactions_types.SIG:
            src_dir = dmfilestat.result.experiment.expDir
            dest_dir = os.path.basename(src_dir)
            dmfilestat.archivepath = os.path.join(backup_directory, dest_dir)
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

        try:
            os.makedirs(dmfilestat.archivepath)
            logger.debug("Created dir: %s" % dmfilestat.archivepath, extra=logid)
        except Exception as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        src_st = os.stat(src_dir)
        old_umask = os.umask(0000)

        try:
            os.chown(dmfilestat.archivepath, src_st.st_uid, src_st.st_gid)
            os.utime(dmfilestat.archivepath, (src_st.st_atime, src_st.st_mtime))
        except Exception as e:
            if e.errno in [errno.EPERM, errno.EACCES]:
                pass
            else:
                raise
        os.umask(old_umask)
    except:
        raise


def _copy_to_dir(filepath, _start_dir, _destination):
    '''
    filepath is absolute path to the file to copy
    start_dir is directory to root the copy in the _destination dir
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    src = filepath
    dst = filepath.replace(_start_dir, '')
    dst = dst[1:] if dst[0] == '/' else dst
    dst = os.path.join(_destination, dst)

    # Check that remote mount is still mounted
    # We create the root destination directory once.  If it no longer is available, the mount has disappeared
    # and we should abort.
    orig_dir = os.getcwd()
    try:
        os.chdir(_destination)
    except:
        raise DMExceptions.MediaNotAvailable(
            "%s is no longer available. Check your remote mounts" % _destination)

    try:
        # Work in the local directory only.  Creating subdirectories if needed
        try:
            local_dir = dst.replace(_destination, '')
            local_dir = local_dir[1:] if local_dir.startswith('/') else local_dir
            local_dir = os.path.join("./", local_dir)
            os.makedirs(os.path.dirname(local_dir))

        except OSError as exception:
            if exception.errno in [errno.EEXIST, errno.EPERM, errno.EACCES]:
                pass
            else:
                # Unknown error, bail out
                raise

        # Trying to preserve symbolic link within local directory tree
        if os.path.islink(filepath) and (os.path.basename(_start_dir) in filepath):
            try:
                link = os.readlink(filepath)
                os.symlink(link, dst)
                return True
            except Exception as e:
                if e.errno == errno.EEXIST:
                    # Target exists so leave it alone
                    os.chdir(orig_dir)
                    return True
                elif e.errno in [errno.EOPNOTSUPP, errno.EACCES]:
                    # Cannot create a link so continue to copy instead below
                    pass
                else:
                    # Unknown error, bail out
                    raise

        # Catch broken links and do not copy them
        try:
            os.stat(filepath)
        except OSError as e:
            if e.errno == errno.ENOENT:
                logger.info("Broken link not copied: %s" % filepath, extra=logid)
                return True
            else:
                raise

        # Calling rsync command in a shell
        cmd = ["rsync", "--times", "--copy-links", "--owner", "--group", src, dst]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            errordump = stderr.splitlines()[0].split(":")[2]
            errordetail = stderr.splitlines()[0].split(":")[1]
            errstr = errordump.split('(')[0].strip()
            errnum = int(errordump.split('(')[1].strip(')'))
            raise DMExceptions.RsyncError(errstr + " " + errordetail, errnum)

        return True
    except:
        raise
    finally:
        os.chdir(orig_dir)


def _process(filepath, action, destination, _start_dir, to_keep):
    '''
    Two basic operations: copy file, delete file.
    DELETE ACTION: delete file
    EXPORT ACTION: copy file
    ARCHIVE ACTION: copy file && delete file
    '''
    # logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    if action == DELETE:
        try:
            return _file_removal(filepath, to_keep)
        except:
            raise
    elif action == EXPORT:
        try:
            return _copy_to_dir(filepath, _start_dir, destination)
        except:
            raise
    elif action == ARCHIVE:
        """It is possible to copy the file, but fail to remove the file.
        We allow this situation b/c there are cases where a single file
        may not be removed."""
        try:
            _copy_to_dir(filepath, _start_dir, destination)
            return _file_removal(filepath, to_keep)
        except:
            raise
    elif action == TEST:
        try:
            return _print_selected(filepath, to_keep)
        except:
            raise


def _get_file_list_dict(dmfilestat, action, user, user_comment, msg_banner):
    '''
    This function generates a list of files to process.
    '''
    logid = {'logid': "%s" % ('dmactions')}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    if dmfilestat.isdeleted():
        errmsg = "The %s for %s are deleted" % (dmfilestat.dmfileset.type, dmfilestat.result.resultsName)
        logger.warn(errmsg, extra=logid)
        raise Exception(errmsg)
    elif dmfilestat.isarchived():
        if not os.path.exists(dmfilestat.archivepath):
            errmsg = "Cannot access backup location %s" % dmfilestat.archivepath
            logger.warn(errmsg, extra=logid)
            raise Exception(errmsg)
        else:
            # search archived directory
            search_dirs = [dmfilestat.archivepath]
    else:
        # search both results directory and raw data directory
        search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]

    # List of all files associated with the report
    cached_file_list = dm_utils.get_walk_filelist(search_dirs, list_dir=dmfilestat.result.get_report_dir())

    # Determine if this file type is eligible to use a keep list
    kpatterns = _get_keeper_list(dmfilestat, action)

    # Create a list of files eligible to process
    list_of_file_dict = []
    is_thumbnail = dmfilestat.result.isThumbnail
    add_linked_sigproc = False if (
        action == DELETE or dmfilestat.dmfileset.type == dmactions_types.INTR) else True
    for start_dir in search_dirs:
        logger.debug("Searching: %s" % start_dir, extra=logid)
        to_process = []
        to_keep = []
        if os.path.isdir(start_dir):
            to_process, to_keep = dm_utils._file_selector(start_dir,
                                                          dmfilestat.dmfileset.include,
                                                          dmfilestat.dmfileset.exclude,
                                                          kpatterns,
                                                          is_thumbnail,
                                                          add_linked_sigproc,
                                                          cached=cached_file_list)
        logger.info("%d files to process at %s" %
                    (len(list(set(to_process) - set(to_keep))), start_dir), extra=logid)
        list_of_file_dict.append(
            {
                'pk': dmfilestat.id,
                'action': action,
                'archivepath': dmfilestat.archivepath,
                'start_dir': start_dir,
                'to_process': to_process,
                'to_keep': to_keep,
                'total_cnt': len(list(set(to_process) - set(to_keep))),
                'processed_cnt': 0,
                'total_size': 0,
                'user': user,
                'user_comment': user_comment,
                'lockfile': '',
                'msg_banner': msg_banner,
            }
        )
    return list_of_file_dict


def _process_fileset_task(list_of_file_dict, lockfile, lock_id):
    '''
    Hands the list to a recursive celery task function.
    The recursion continues until the list is empty.  The calling
    function exits immediately.
    '''
    logid = {'logid': "%s" % (lockfile)}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    # Create a lock file here to block any other actions on this report (see TS-8411)
    locallock = TaskLock(lock_id, timeout=60)  # short timeout in case lock release code doesn't get called

    if not(locallock.lock()):
        logger.warn("lock file exists: %s(%s)" % (lock_id, locallock.get()), extra=logid)
        # Release the task lock
        try:
            applock = TaskLock(lockfile)
            applock.unlock()
        except:
            logger.error(traceback.format_exc(), extra=logid)
        return

    logger.info("lock file created: %s(%s)" % (lock_id, locallock.get()), extra=logid)

    # lockfile variable stored in list_of_file_dict
    for entry in list_of_file_dict:
        entry['lockfile'] = lockfile

    try:
        pfilename = set_action_param_var(list_of_file_dict)

        # Call the recursive celery task function to process the list
        _process_task.delay(pfilename)

    except:
        logger.error("We got an error here, _process_fileset_task", extra=logid)
        raise
    finally:
        if locallock:
            locallock.unlock()

    return


@task(queue='dmprocess', ignore_result=True)
def _process_task(pfilename):
    '''
    Recursive celery task.

    To trigger an orphaned task:
    python -c "from iondb.bin import djangoinit; from iondb.rundb.data import dmactions; dmactions._process_task.(<filename>)"
    where <filename> is full path to the data file found in /var/spool/ion
    '''
    logid = {'logid': "%s" % ('dmactions')}
    from datetime import datetime
    from datetime import timedelta
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    logger.debug("Task ID: %s" % _process_task.request.id, extra=logid)

    # catch all unhandled exceptions and clean up
    try:
        try:
            list_of_file_dict = get_action_param_var(pfilename)
        except Exception as e:
            logger.error("Error accessing file: %s.  Cannot continue the DM action!" %
                         (pfilename), extra=logid)
            # parse the filename to extract the dmfilestat pk and retrieve dmfilestat object.
            dmfilestat_pk = os.path.basename(pfilename).split("_")[1]
            dmfilestat = DMFileStat.objects.get(id=dmfilestat_pk)
            raise e

        dmfilestat = DMFileStat.objects.get(id=list_of_file_dict[0]['pk'])
        terminate = True   # flag to indicate recursion termination
        total_processed = 0
        fstatus = "Success"

        start_time = datetime.now()
        max_time_delta = timedelta(seconds=10)

        # list_of_file_dict contains zero, one, or two dictionary variables to iterate over.
        for d_cnt, mydict in enumerate(list_of_file_dict):
            logid = {'logid': "%s" % (mydict.get('lockfile', '_process_task'))}

            # The dictionary contains an element named 'to_process' which is a list variable to iterate over
            logger.debug("%d, start_dir: %s" % (d_cnt, mydict['start_dir']), extra=logid)
            logger.info("%6d %s %s" %
                        (len(mydict['to_process']), dmfilestat.dmfileset.type, dmfilestat.result.resultsName), extra=logid)

            while (datetime.now() - start_time) < max_time_delta:
                # If there are no files left to process, (all to_process lists are empty), the recursion ends
                if len(mydict['to_process']) > 0:
                    terminate = False

                    try:
                        # process one file and remove entry from the list
                        path = mydict['to_process'].pop(0)

                        j = mydict['processed_cnt'] + 1

                        this_file_size = 0
                        if not os.path.islink(path):
                            this_file_size = os.lstat(path)[6]

                        if _process(path, mydict['action'], mydict['archivepath'], mydict['start_dir'], mydict['to_keep']):
                            mydict['processed_cnt'] = j
                            mydict['total_size'] += this_file_size
                            logger.debug("%04d/%04d %s %10d %s" % (
                                j, mydict['total_cnt'], mydict['action'], mydict['total_size'], path), extra=logid)

                    except (OSError, IOError) as e:
                        # IOError: [Errno 28] No space left on device:
                        if e.errno == errno.ENOSPC:
                            raise
                        elif e.errno == errno.ENOENT or e.errno == errno.ESTALE:
                            logger.warn("%04d No longer exists %s" % (j, path), extra=logid)
                            continue
                        else:
                            raise
                    except (DMExceptions.RsyncError, DMExceptions.MediaNotAvailable):
                        raise
                    except:
                        errmsg = "%04d/%04d %s %10d %s" % (
                            j, mydict['total_cnt'], mydict['action'], mydict['total_size'], path)
                        logger.error(errmsg, extra=logid)
                        logger.error(traceback.format_exc(), extra=logid)

                    if not mydict['action'] in [EXPORT, TEST] and dmfilestat.dmfileset.del_empty_dir:
                        thisdir = os.path.dirname(path)
                        try:
                            if len(os.listdir(thisdir)) == 0:
                                if not "plugin_out" in thisdir:
                                    try:
                                        os.rmdir(thisdir)
                                        logger.debug("Removed empty directory: %s" % thisdir, extra=logid)
                                    except Exception as e:
                                        logger.warn("rmdir [%d] %s: %s" % (
                                            e.errno, e.strerror, thisdir), extra=logid)
                        except OSError as e:
                            if e.errno == errno.ENOENT:
                                logger.warn("del_empty_dir Does not exist %s" % (path), extra=logid)
                                continue
                            else:
                                raise e
                else:
                    break

            # only expect to execute this line when no files to process
            total_processed += mydict['total_size']

    except Exception as e:
        fstatus = "Error"
        terminate = True
        dmfilestat.setactionstate('E')
        logger.error("DM Action failure on %s for %s report." %
                     (dmfilestat.dmfileset.type, dmfilestat.result.resultsName), extra=logid)
        logger.error("This %s action will need to be manually completed." % (mydict['action']), extra=logid)
        logger.error("The following is the exception error:\n" + traceback.format_exc(), extra=logid)
        EventLog.objects.add_entry(
            dmfilestat.result, "%s - %s. Action not completed.  User intervention required." % (fstatus, e), username='dm_agent')

        # Release the task lock
        try:
            applock = TaskLock(mydict['lockfile'])
            applock.unlock()
        except:
            logger.error(traceback.format_exc(), extra=logid)

        # Do the user notification
        try:
            # pop up a message banner
            if mydict['msg_banner']:
                dmfileset = dmfilestat.dmfileset
                project_msg = {}
                msg_dict = {}
                msg_dict[dmfileset.type] = fstatus
                project_msg[dmfilestat.result_id] = msg_dict
                project_msg_banner('', project_msg, mydict['action'])
        except:
            logger.error(traceback.format_exc(), extra=logid)

        # ====================================================================
        # Exit function here on error
        # ====================================================================
        return

    # Remove the data file here, no earlier.  In case the task is clobbered, celery
    # will relaunch the task, access the data file and continue the action.
    try:
        os.unlink(pfilename)
    except:
        pass

    if not terminate:
        # ====================================================================
        # Launch next task
        # ====================================================================
        try:
            mydict.get('action', 'unk')
            pfilename = set_action_param_var(list_of_file_dict)
            _process_task.delay(pfilename)
        except:
            logger.error(traceback.format_exc(), extra=logid)

    else:
        # ====================================================================
        # No more files to process.  Clean up and exit.
        # ====================================================================
        try:
            dmfilestat.diskspace = float(total_processed) / (1024 * 1024)
            dmfilestat.save()
            logger.info("%0.1f MB %s processed" %
                        (dmfilestat.diskspace, dmfilestat.dmfileset.type), extra=logid)
            if mydict['action'] in [ARCHIVE, DELETE]:
                _brokenlinks_delete([dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir])
                _emptydir_delete([dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir])
        except:
            logger.error(traceback.format_exc(), extra=logid)

        # Do the user notification
        try:
            _action_complete_update(mydict['user'], mydict['user_comment'], dmfilestat, mydict['action'])

            # pop up a message banner
            if mydict['msg_banner']:
                dmfileset = dmfilestat.dmfileset
                project_msg = {}
                msg_dict = {}
                msg_dict[dmfileset.type] = fstatus
                project_msg[dmfilestat.result_id] = msg_dict
                project_msg_banner(mydict['user'], project_msg, mydict['action'])
        except:
            logger.error(traceback.format_exc(), extra=logid)

        # Release the task lock
        try:
            applock = TaskLock(mydict['lockfile'])
            applock.unlock()
        except:
            logger.error(traceback.format_exc(), extra=logid)

    return


def _get_keeper_list(dmfilestat, action):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    if action == EXPORT:
        kpatterns = []
    else:
        kpatterns = []
        # Are there entries in dmfilestat.dmfileset.keepwith?
        # logger.debug("FILES IN KEEPWITH FIELD", extra=logid)
        # logger.debug(dmfilestat.dmfileset.keepwith, extra=logid)
        for settype, patterns in dmfilestat.dmfileset.keepwith.iteritems():
            # Are the types specified in dmfilestat.dmfileset.keepwith still local?
            if not dmfilestat.result.dmfilestat_set.get(dmfileset__type=settype).isdisposed():
                # add patterns to kpatterns
                kpatterns.extend(patterns)
    logger.debug("Keep Patterns are %s" % kpatterns, extra=logid)
    return kpatterns


def _file_removal(filepath, to_keep):
    '''Conditional removal.  Only remove if the filepath is not in the to_keep list'''
    # logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    starttime = time.time()
    # if filepath matches any kpattern, do not remove
    if filepath not in to_keep:
        try:
            os.remove(filepath)
        except OSError as e:
            if e.errno == errno.EISDIR:
                # TODO: need to handle a cascading exception here
                os.rmdir(filepath)
            elif e.errno == errno.ENOENT:   # no such file or directory
                pass
            elif e.errno == errno.EACCES:   # permission denied
                logger.info(e, extra=logid)
                return False
            else:
                raise e
    else:
        logger.debug("NOT REMOVING: %s" % filepath, extra=logid)
        return False

    endtime = time.time()
    if (endtime - starttime) > 1.0:
        logger.warn("%s: %f seconds" % (sys._getframe().f_code.co_name, (endtime - starttime)), extra=logid)
    return True


def _print_selected(filepath, to_keep):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    if filepath not in to_keep:
        logger.info("Selected: %s" % filepath, extra=logid)
        return True
    return False


def _emptydir_delete(search_dirs):
    '''
    Search for empty directories and delete
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    for start_dir in search_dirs:
        for root, dirs, files in os.walk(start_dir, topdown=True):
            for name in dirs:
                filepath = os.path.join(root, name)
                try:
                    empty_dir = True if len(os.listdir(filepath)) == 0 else False
                except:
                    continue
                if empty_dir and name != "plugin_out":
                    logger.debug("Removing Empty Directory: %s" % filepath, extra=logid)
                    # this fancy bit is to not trip on a soft link (TS-6600)
                    try:
                        os.rmdir(filepath)
                    except OSError as e:
                        if e.errno == errno.ENOTDIR:
                            try:
                                os.unlink(filepath)
                            except:
                                logger.warn(e, extra=logid)
                        elif e.errno == errno.ENOENT:   # no such file or directory
                            pass
                        else:
                            logger.warn(e, extra=logid)


def _brokenlinks_delete(search_dirs):
    '''
    Delete broken links
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    for start_dir in search_dirs:
        for root, dirs, files in os.walk(start_dir, topdown=True):
            for name in files:
                filepath = os.path.realpath(os.path.join(root, name))
                if os.path.islink(filepath):
                    try:
                        os.stat(filepath)
                    except OSError, e:
                        if e.errno == errno.ENOENT:
                        # If os.stat fails, then its a broken link at this point
                            try:
                                os.unlink(filepath)
                                logger.info("Broken symlink removed: %s" % filepath, extra=logid)
                            except OSError as e:
                                logger.warn(e, extra=logid)


def set_action_state(dmfilestat, action_state, action=''):
    '''Rules for setting action state for Basecalling Input Files with linked sigproc_results:
        action = DELETE or action = EXPORT applies to current report only
        action = ARCHIVE applies to all linked results
    '''
    if dmfilestat.dmfileset.type == dmactions_types.BASE and action == ARCHIVE:
        related = get_related_dmfilestats(dmfilestat)
        if related is not None:
            related.update(action_state=action_state)

    dmfilestat.setactionstate(action_state)


def _update_related_objects(user, user_comment, dmfilestat, action, msg, action_state=None):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    # Change the action_state field, if it is provided
    if action_state is not None:
        set_action_state(dmfilestat, action_state, action)

    # update related dmfilestats
    related = get_related_dmfilestats(dmfilestat)
    if related is not None:
        related.update(archivepath=dmfilestat.archivepath)
        if dmfilestat.dmfileset.type == dmactions_types.SIG:
            related.update(diskspace=dmfilestat.diskspace)

    # add log entry
    msg = msg + "<br>User Comment: %s" % (user_comment)
    add_eventlog(dmfilestat, msg, user)

    # Double check the ftpStatus field is properly set
    if dmfilestat.dmfileset.type == dmactions_types.SIG:
        # check that experiment.ftpStatus field is now set to complete.
        dmfilestat.result.experiment.ftpStatus = "Complete"
        dmfilestat.result.experiment.save()


def _update_diskspace_and_diskusage(dmfilestat):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    diskspace = update_diskspace(dmfilestat)

    # update diskusage on Experiment or Results object if data was deleted/moved
    # See data/tasks.update_dmfilestat_diskusage() which also updates Exp & Results diskusage fields
    if dmfilestat.dmfileset.type == dmactions_types.SIG:
        dmfilestat.result.experiment.diskusage = diskspace if diskspace != None else 0
        dmfilestat.result.experiment.save()
    else:
        result = dmfilestat.result
        disk_total = 0
        mylist = [
            result.get_filestat(dmactions_types.BASE).diskspace,
            result.get_filestat(dmactions_types.OUT).diskspace,
            result.get_filestat(dmactions_types.INTR).diskspace
        ]
        for partial in mylist:
            disk_total += int(partial) if partial != None else 0
        result.diskusage = disk_total
        result.save()


def _action_complete_update(user, user_comment, dmfilestat, action):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    if action == ARCHIVE:
        action_state = 'AD'
        msg = "%0.1f MB %s archived to %s." % (
            dmfilestat.diskspace, dmfilestat.dmfileset.type, dmfilestat.archivepath)
    elif action == DELETE:
        action_state = 'DD'
        msg = "%0.1f MB %s deleted." % (dmfilestat.diskspace, dmfilestat.dmfileset.type)
    elif action == EXPORT:
        action_state = 'L'
        msg = "%0.1f MB %s exported to %s." % (
            dmfilestat.diskspace, dmfilestat.dmfileset.type, dmfilestat.archivepath)
        dmfilestat.archivepath = None
        dmfilestat.save()
    elif action == TEST:
        return

    if action != EXPORT:
        _update_diskspace_and_diskusage(dmfilestat)

    _update_related_objects(user, user_comment, dmfilestat, action, msg, action_state)


def set_action_pending(user, user_comment, action, dmfilestat, backup_directory):
    '''Internal function to set dmfilestat action state'''
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
        # save username and comment to be used when action is run
        dmfilestat.user_comment = {'user': user, 'user_comment': user_comment}
        dmfilestat.save()
        msg += ".<br>User Comment: %s" % user_comment

        if dmfilestat.dmfileset.type == dmactions_types.SIG:
            # update all realated dmfilestats
            exp_id = dmfilestat.result.experiment_id
            DMFileStat.objects.filter(dmfileset__type=dmactions_types.SIG, result__experiment__id=exp_id) \
                .update(action_state=dmfilestat.action_state, archivepath=dmfilestat.archivepath, user_comment=dmfilestat.user_comment)

        add_eventlog(dmfilestat, msg, user)
        return "scheduled"
    else:
        return dmfilestat.get_action_state_display()


def add_eventlog(dmfilestat, msg, username):
    '''Adds an event entry log to database'''
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
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)
    if os.path.isdir(dmfilestat.result.experiment.expDir):
        if len(os.listdir(dmfilestat.result.experiment.expDir)) == 0:
            logger.debug("expDir exists and is empty: %s" % dmfilestat.result.experiment.expDir, extra=logid)
            os.rmdir(dmfilestat.result.experiment.expDir)
            os.symlink(dmfilestat.archivepath, dmfilestat.result.experiment.expDir)
            logger.debug("symlink created: %s" % dmfilestat.result.experiment.expDir, extra=logid)
        else:
            logger.debug("no symlink, expDir is not empty: %s" %
                         dmfilestat.result.experiment.expDir, extra=logid)
    else:
        logger.debug("expDir no longer exists: %s" % dmfilestat.result.experiment.expDir, extra=logid)
        os.symlink(dmfilestat.archivepath, dmfilestat.result.experiment.expDir)
        logger.debug("symlink created: %s" % dmfilestat.result.experiment.expDir, extra=logid)


def set_action_param_var(list_of_dict_files):
    '''
    Argument is dictionary to be pickled.  Return value is name of file.
    '''
    from cPickle import Pickler
    import tempfile
    # Format name to include pk of dmfilestat object - in case we lose the data file itself
    # /var/spool/ion/<action>_<pk>_<randomstring>
    store_dir = '/var/spool/ion'
    prefix = "%s_%d_" % (list_of_dict_files[0]['action'], list_of_dict_files[0]['pk'])
    with tempfile.NamedTemporaryFile(dir=store_dir, delete=False, mode='w+b', prefix=prefix) as fileh:
        pickle = Pickler(fileh)
        pickle.dump(list_of_dict_files)
    return fileh.name
    '''
    TODO: Store the variable in the task lock.
    '''


def get_action_param_var(pfilename):
    '''
    Argument is name of file to unpickle.  Return value is dictionary value.
    '''
    from cPickle import Unpickler
    with open(pfilename, 'rb') as fileh:
        pickle = Unpickler(fileh)
        list_of_dict_files = pickle.load()
    return list_of_dict_files
    '''
    TODO: Get the variable from the task lock.
    '''


def prepare_for_data_import(dmfilestat):
    ''' Additional files are added here that will be used if data is later Imported '''
    write_serialized_json(dmfilestat.result, dmfilestat.archivepath)

    # for convenience save the pdf files needed for importing report db object without copy files
    if dmfilestat.dmfileset.type == dmactions_types.OUT:
        report_path = dmfilestat.result.get_report_dir()
        pdf_files = [
            'report.pdf',
            'plugins.pdf',
            os.path.basename(report_path) + '-full.pdf'
        ]
        for pdf_file in pdf_files:
            filepath = os.path.join(report_path, pdf_file)
            if os.path.exists(filepath):
                shutil.copy2(filepath, dmfilestat.archivepath)


def write_serialized_json(result, destination):
    '''Writes serialized database objects to disk file'''
    sfile = os.path.join(destination, "serialized_%s.json" % result.resultsName)
    # skip if already exists
    if os.path.exists(sfile):
        return

    serialize_objs = [result, result.experiment]
    for obj in [result.experiment.plan, result.eas, result.analysismetrics, result.libmetrics, result.qualitymetrics]:
        if obj:
            serialize_objs.append(obj)
    serialize_objs += list(result.experiment.samples.all())
    serialize_objs += list(result.pluginresult_set.all())

    try:
        with open(sfile, 'wt') as fileh:
            obj_json = serializers.serialize('json', serialize_objs, indent=2, use_natural_keys=True)

            dmfilesets = DMFileSet.objects.filter(dmfilestat__result=result)
            obj_json = obj_json.rstrip(' ]\n') + ','
            obj_json += serializers.serialize(
                'json', dmfilesets, indent=2, fields=('type', 'version')).lstrip('[')

            fileh.write(obj_json)

    except:
        logger.error("Unable to save serialized.json for %s(%d)" %
                     (result.resultsName, result.pk), extra=logid)
        logger.error(traceback.format_exc(), extra=logid)


def get_file_list(dmfilestat):
    """Return list of files selected by this DMFileStat record and list of files to not process.
    There are some cases in which the list of selected files contains files which should not be
    processed.  Those are in the to_keep list."""
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra=logid)

    to_process = []
    to_keep = []
    try:
        # search both results directory and raw data directory
        search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]

        cached_file_list = dm_utils.get_walk_filelist(
            search_dirs, list_dir=dmfilestat.result.get_report_dir())
    except:
        # If this function has an error, this file set should be marked 'E'
        dmfilestat.setactionstate('E')
        logger.error(traceback.format_exc(), extra=logid)
        return (to_process, to_keep)

    try:
        # Determine if this file type is eligible to use a keep list
        kpatterns = _get_keeper_list(dmfilestat, '')

        # Create a list of files eligible to process
        is_thumbnail = dmfilestat.result.isThumbnail
        for start_dir in search_dirs:
            if os.path.isdir(start_dir):
                tmp_process, tmp_keep = dm_utils._file_selector(start_dir,
                                                                dmfilestat.dmfileset.include,
                                                                dmfilestat.dmfileset.exclude,
                                                                kpatterns,
                                                                is_thumbnail,
                                                                cached=cached_file_list)
                to_process += tmp_process
                to_keep += tmp_keep
            else:
                logger.error(traceback.format_exc(), extra=logid)
    except:
        logger.error(traceback.format_exc(), extra=logid)

    return (to_process, to_keep)
