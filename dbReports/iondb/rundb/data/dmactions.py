#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import os
import re
import sys
import time
import shutil
import errno
import tempfile
import traceback
import urllib
import subprocess
from datetime import datetime
from iondb.utils.files import getSpaceMB, getSpaceKB, ismountpoint
from iondb.utils import makePDF
from ion.utils import makeCSA
from iondb.utils.TaskLock import TaskLock
from celery.task import task
from celery.utils.log import get_task_logger
from iondb.rundb.data import dmactions_types
from iondb.rundb.models import DMFileStat, DMFileSet, Results, EventLog
from iondb.utils.files import getdiskusage
from iondb.rundb.data import exceptions as DMExceptions
from iondb.rundb.data.project_msg_banner import project_msg_banner
from django.core import serializers

#Send logging to data_management log file
logger = get_task_logger('data_management')
d = {'logid':"%s" % ('dmactions')}

ARCHIVE = 'archive'
EXPORT = 'export'
DELETE = 'delete'
TEST = 'test'


def delete(user, user_comment, dmfilestat, lockfile, msg_banner, confirmed=False):
    d = {'logid':"%s" % (lockfile)}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    msg = "Deleting %s - %s Using v.%s" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName,dmfilestat.dmfileset.version)
    logger.info(msg, extra = d)
    _update_related_objects(user, user_comment, dmfilestat, DELETE, msg)
    try:
        action_validation(dmfilestat, DELETE, confirmed)
    except:
        raise
    try:
        if dmfilestat.dmfileset.type == dmactions_types.OUT:
            _create_archival_files(dmfilestat)
        _process_fileset_task(dmfilestat, DELETE, user, user_comment, lockfile, msg_banner)

    except:
        dmfilestat.setactionstate('E')
        raise


def export(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory=None):
    d = {'logid':"%s" % (lockfile)}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    msg = "Exporting %s - %s Using v.%s" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName,dmfilestat.dmfileset.version)
    logger.info(msg, extra = d)
    _update_related_objects(user, user_comment, dmfilestat, EXPORT, msg)
    try:
        destination_validation(dmfilestat, backup_directory)
        _create_destination(dmfilestat, EXPORT, dmfilestat.dmfileset.type, backup_directory)
    except:
        raise
    try:
        if dmfilestat.dmfileset.type == dmactions_types.OUT:
            _create_archival_files(dmfilestat)
        _process_fileset_task(dmfilestat, EXPORT, user, user_comment, lockfile, msg_banner)
        prepare_for_data_import(dmfilestat)
    except:
        dmfilestat.setactionstate('E')
        raise


def archive(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory=None, confirmed=False):
    d = {'logid':"%s" % (lockfile)}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    msg = "Archiving %s - %s Using v.%s" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName,dmfilestat.dmfileset.version)
    logger.info(msg, extra = d)
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
        _process_fileset_task(dmfilestat, ARCHIVE, user, user_comment, lockfile, msg_banner)
        prepare_for_data_import(dmfilestat)
    except:
        dmfilestat.setactionstate('E')
        raise


def test(user, user_comment, dmfilestat, lockfile, msg_banner, backup_directory=None):
    d = {'logid':"%s" % (lockfile)}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    msg = "Testing %s - %s Using v.%s" % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName,dmfilestat.dmfileset.version)
    logger.info(msg, extra = d)
    try:
        _process_fileset_task(dmfilestat, TEST, user, user_comment, lockfile, msg_banner)
    except:
        raise


def update_diskspace(dmfilestat, cached = None):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    try:
        # search both results directory and raw data directory
        search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]

        if not cached:
            cached = get_walk_filelist(search_dirs, list_dir=dmfilestat.result.get_report_dir())

        total_size = 0

        #Create a list of files eligible to process
        isThumbnail = dmfilestat.result.isThumbnail
        for start_dir in search_dirs:
            to_process = []
            if os.path.isdir(start_dir):
                logger.debug("Start dir: %s" % start_dir, extra = d)    # xtra
                to_process, to_keep = _file_selector(start_dir,
                                                     dmfilestat.dmfileset.include,
                                                     dmfilestat.dmfileset.exclude,
                                                     [],
                                                     isThumbnail,
                                                     add_linked_sigproc=True,
                                                     cached = cached)

                #process files in list
                #starttime = time.time()
                for j, path in enumerate(to_process, start=1):
                    try:
                        #logger.debug("%d %s %s" % (j, 'diskspace', path), extra = d)
                        if not os.path.islink(path):
                            total_size += os.lstat(path)[6]

                    except Exception as inst:
                        errmsg = "Error processing %s" % (inst)
                        logger.error(errmsg, extra = d)
                #endtime = time.time()
                #logger.info("Loop time: %f seconds" % (endtime-starttime), extra = d)

        diskspace = float(total_size)/(1024*1024)
    except:
        logger.error(traceback.format_exc(), extra = d)
        diskspace = None

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
    def _skipdiskspacecheck(directory):
        '''
        The hidden file .no_size_check should be placed into the root directory of the scratch drive mounted on the local
        system for the tape drive system.
        '''
        if os.path.exists(os.path.join(directory, ".no_size_check")):
            logger.info("%s: Exists: %s" % (sys._getframe().f_code.co_name, os.path.join(directory, ".no_size_check")))
            return True
        else:
            logger.info("%s: Not Found: %s" % (sys._getframe().f_code.co_name, os.path.join(directory, ".no_size_check")))
            return False

    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    if backup_directory in [None, 'None', '']:
        backup_directory = dmfilestat.dmfileset.backup_directory

    # check for valid destination
    try:
        if backup_directory in [None, 'None', '', '/']:
            raise DMExceptions.MediaNotSet("Backup media for %s is not configured. Please use Data Management Configuration page." % dmfilestat.dmfileset.type)

        if not os.path.isdir(backup_directory):
            raise DMExceptions.MediaNotAvailable("Backup media for %s is not available: %s" % (dmfilestat.dmfileset.type, backup_directory))

        # check if destination is external filesystem
        # ie, catch the error of writing to a mountpoint which is unmounted.
        # Use the tool 'mountpoint' which returns 0 if its mountpoint and mounted
        # If its a subdirectory to a mountpoint, then the isdir() test will fail when not mounted.
        if ismountpoint(backup_directory):
            raise DMExceptions.MediaNotAvailable("Backup media for %s is not available: %s" % (dmfilestat.dmfileset.type, backup_directory))

    except Exception as e:
        logger.error("%s" % e, extra = d)
        raise

    # check for sufficient disk space (units: kilobytes)
    if _skipdiskspacecheck(backup_directory):
        logger.warn("%s - skipping destination disk space check" % (sys._getframe().f_code.co_name), extra = d)
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
                for obj in DMFileStat.objects.filter(action_state__in=['AG','EG','SA','SE']):
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

                logger.debug("Required %dMB, pending %dMB, free %dMB" % (diskspace, pending, freespace), extra = d)

            if diskspace >= freespace:
                raise DMExceptions.InsufficientDiskSpace("Not enough space to write files at %s (free=%dMB)" % (backup_directory,freespace))
            elif diskspace >= (freespace - pending):
                raise DMExceptions.InsufficientDiskSpace("Not enough space to write files at %s (free=%dMB, pending=%dMB)" % (backup_directory,freespace, pending))
        except Exception as e:
            logger.debug("%s" % str(e), extra = d)
            raise

    # check for write permission.  NOTE: this function is called by apache2 user while the action function
    # will be executed within a celery task which takes the uid/gid of celeryd process - in our case that is currently root.
    # This test is too restrictive.
    try:
        foo = tempfile.NamedTemporaryFile(dir=backup_directory)
        foo.close()
    except Exception as e:
        if e.errno in [errno.EPERM, errno.EACCES]: # Operation not permitted
            errmsg = "Insufficient write permission in %s" % backup_directory
        else:
            errmsg = e
        logger.error(errmsg, extra = d)
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
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)

    # Any non-blank string in dmfilestat.files_in_use fails the validation.
    # Manual actions must call data_management.update_files_in_use directly prior to validating for up-to-date info.
    if dmfilestat.files_in_use:
        errmsg = "%s files are currently being used by %s." % (dmfilestat.dmfileset.type, dmfilestat.files_in_use)
        logger.info(errmsg, extra = d)    #For developers only, more informative error string
        errmsg = " Cannot complete request because the files to be deleted/archived are currently in use."
        raise DMExceptions.FilesInUse(errmsg)

    # Do not allow delete action when the filestat preserve flag is set
    if action == DELETE and dmfilestat.getpreserved():
        errmsg = "Unable to delete files: %s currently marked as Keep." % dmfilestat.dmfileset.type
        logger.warn(errmsg, extra = d)
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
            logger.error(errmsg, extra = d)
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
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)

    # This check is needed for cases where src dir has been manually deleted
    report_dir = dmfilestat.result.get_report_dir()
    if not os.path.exists(report_dir):
        raise DMExceptions.SrcDirDoesNotExist(report_dir)

    try:
        makePDF.write_summary_pdf(dmfilestat.result_id)
    except:
        logger.error("Could not create Report PDF", extra = d)
        raise

    try:
        #TS-7385.  When executed by celery task, the files created are owned by root.root.
        #This hack sets the ownership to the same as the report directory.
        pdf_files = [
            'report.pdf',
            'plugins.pdf',
            'backupPDF.pdf',
            os.path.basename(dmfilestat.result.get_report_dir())+'-full.pdf',
            os.path.basename(dmfilestat.result.get_report_dir())+'.support.zip'
            ]
        report_path = dmfilestat.result.get_report_dir()
        uid = os.stat(report_path).st_uid
        gid = os.stat(report_path).st_gid
        for pdf_file in pdf_files:
            if os.path.exists(os.path.join(report_path,pdf_file)):
                os.chown(os.path.join(report_path,pdf_file), uid, gid)
    except:
        logger.warn("Something failed while changing ownership of pdf or support zip file", extra = d)
        logger.debug(traceback.format_exc(), extra = d)

    try:
        csaFullPath = makeCSA.makeCSA(dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir)
    except:
        logger.error("Could not create CSA", extra = d)
        raise


def _create_destination(dmfilestat, action, filesettype, backup_directory=None):
    '''
    Create directory in destination directory to write files
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
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

        try:
            os.makedirs(dmfilestat.archivepath)
            logger.debug("Created dir: %s" % dmfilestat.archivepath, extra = d)
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


def _file_selector(start_dir, ipatterns, epatterns, kpatterns, isThumbnail=False, add_linked_sigproc=False, cached = None):
    '''Returns list of files found in directory which match the list of
    patterns to include and which do not match any patterns in the list
    of patterns to exclude.  Also returns files matching keep patterns in
    separate list.
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    starttime = time.time()  # debugging time of execution
    to_include = []
    to_exclude = []
    to_keep = []

    if not ipatterns: ipatterns = []
    if not epatterns: epatterns = []
    if not kpatterns: kpatterns = []

    exclude_sigproc_folder = False
    if not add_linked_sigproc and os.path.islink(os.path.join(start_dir, 'sigproc_results')):
        exclude_sigproc_folder = True

    #find files matching include filters from start_dir
    #for root, dirs, files in os.walk(start_dir,topdown=True):
    for filepath in cached:
        if isThumbnail and 'onboard_results' in filepath:
            continue
        if exclude_sigproc_folder and 'sigproc_results' in filepath:
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

    #find files matching exclude filters from include list
    for pattern in epatterns:
        filter = re.compile(r'(%s/)(%s)' % (start_dir,pattern))
        for filename in to_include:
            match = filter.match(filename)
            if match:
                to_exclude.append(filename)

    selected = list(set(to_include) - set(to_exclude))
    endtime = time.time()
    logger.info("%s(): %f seconds" % (sys._getframe().f_code.co_name,(endtime - starttime)), extra = d)
    return selected, to_keep


def _copy_to_dir(filepath,_start_dir,_destination):
    '''
    filepath is absolute path to the file to copy
    start_dir is directory to root the copy in the _destination dir
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    src = filepath
    dst = filepath.replace(_start_dir,'')
    dst = dst[1:] if dst[0] == '/' else dst
    dst = os.path.join(_destination,dst)
    try:
        os.makedirs(os.path.dirname(dst))
        src_dir_stat = os.stat(os.path.dirname(filepath))
        os.chown(os.path.dirname(dst),src_dir_stat.st_uid,src_dir_stat.st_gid)
    except OSError as exception:
        if exception.errno in [errno.EEXIST, errno.EPERM, errno.EACCES]:
            pass
        else:
            raise

    if os.path.islink(filepath) and os.path.realpath(filepath).startswith(_start_dir):
        try:
            link = os.readlink(filepath)
            os.symlink(link, dst)
        except Exception as e:
            if e.errno != errno.EEXIST:
                raise
        return True

    faster = True
    if faster:
        cmd = ["rsync","-tL",src,dst]
        # Preserve symlinks with directory tree, copy unsafe links, preserve times and permissions
        # Cannot preserve ownership because of problems on NFS mounts.
        #cmd = ["rsync","--links","--copy-unsafe-links","-pt",src,dst]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            errordump = stderr.splitlines()[0].split(":")[2]
            errordetail = stderr.splitlines()[0].split(":")[1]
            errstr = errordump.split('(')[0].strip()
            errnum = int(errordump.split('(')[1].strip(')'))
            raise DMExceptions.RsyncError(errstr+" "+errordetail,errnum)

    else:
        try:
        # copy file preserving permissions and owner and group
            shutil.copy2(src,dst)
            os.chown(dst,os.stat(src).st_uid,os.stat(src).st_gid)
        except (shutil.Error,OSError) as e:
            if e.errno in [errno.EPERM, errno.EACCES]:
                pass
            else:
                # When file is the same, its okay
                logger.warn("%s" % e, extra = d)
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
    #logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)

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
        """It is possible to copy the file, but fail to remove the file.
        We allow this situation b/c there are cases where a single file
        may not be removed."""
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
    d = {'logid':"%s" % (lockfile)}
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)

    if dmfilestat.isdeleted():
        errmsg = "The %s for %s are deleted" % (dmfilestat.dmfileset.type, dmfilestat.result.resultsName)
        logger.warn(errmsg, extra = d)
        raise Exception(errmsg)
    elif dmfilestat.isarchived():
        if not os.path.exists(dmfilestat.archivepath):
            errmsg = "Cannot access backup location %s" % dmfilestat.archivepath
            logger.warn(errmsg, extra = d)
            raise Exception(errmsg)
        else:
            # search archived directory
            search_dirs = [dmfilestat.archivepath]
    else:
        # search both results directory and raw data directory
        search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]

    # Create a lock file here to block any other actions on this report (see TS-8411)
    lock_id = "%s_%s" % (dmfilestat.result.resultsName, slugify(dmfilestat.dmfileset.type))
    applock = TaskLock(lock_id, timeout=60) # short timeout in case lock release code doesn't get called

    if not(applock.lock()):
            logger.info("lock file exists: %s(%s)" % (lock_id, applock.get()), extra = d)
            return

    logger.info("lock file created: %s(%s)" % (lock_id, applock.get()), extra = d)

    if action == ARCHIVE:
        dmfilestat.setactionstate('AG')
    elif action == DELETE:
        dmfilestat.setactionstate('DG')
    elif action == EXPORT:
        dmfilestat.setactionstate('EG')

    # List of all files associated with the report
    cached_file_list = get_walk_filelist(search_dirs, list_dir=dmfilestat.result.get_report_dir())

    #Determine if this file type is eligible to use a keep list
    kpatterns = _get_keeper_list(dmfilestat, action)

    #Create a list of files eligible to process
    list_of_file_dict = []
    isThumbnail = dmfilestat.result.isThumbnail
    add_linked_sigproc=False if (action==DELETE or dmfilestat.dmfileset.type==dmactions_types.INTR) else True
    for start_dir in search_dirs:
        logger.debug("Searching: %s" % start_dir, extra = d)
        if os.path.isdir(start_dir):
            to_process, to_keep = _file_selector(start_dir,
                                                 dmfilestat.dmfileset.include,
                                                 dmfilestat.dmfileset.exclude,
                                                 kpatterns,
                                                 isThumbnail,
                                                 add_linked_sigproc,
                                                 cached=cached_file_list)
            logger.info("%d files to process at %s" % (len(list(set(to_process) - set(to_keep))),start_dir), extra = d)
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

    try:
        pfilename = set_action_param_var(list_of_file_dict)

        # Call the recursive celery task function to process the list
        celery_result = _process_task.delay(pfilename)
    except:
        logger.error("We got an error here, _process_fileset_task", extra = d)
        raise
    finally:
        if applock:
            applock.unlock()

    return

@task(queue='periodic')
def _process_task(pfilename):
    '''
    Recursive celery task.

    To trigger an orphaned task:
    python -c "from iondb.bin import djangoinit; from iondb.rundb.data import dmactions; dmactions._process_task.(<filename>)"
    where <filename> is full path to the data file found in /var/spool/ion
    '''
    d = {'logid':"%s" % ('dmactions')}
    from datetime import datetime
    from datetime import timedelta
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    logger.debug("Task ID: %s" % _process_task.request.id, extra = d)

    #catch all unhandled exceptions and clean up
    try:
        try:
            list_of_file_dict = get_action_param_var(pfilename)
        except Exception as e:
            logger.error("Error accessing file: %s.  Cannot continue the DM action!" % (pfilename), extra = d)
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
        for q,dict in enumerate(list_of_file_dict):
            d = {'logid':"%s" % (dict.get('lockfile','_process_task'))}

            # The dictionary contains an element named 'to_process' which is a list variable to iterate over
            logger.debug("%d, start_dir: %s" % (q,dict['start_dir']), extra = d)
            logger.info("%6d %s %s" %(len(dict['to_process']),dmfilestat.dmfileset.type,dmfilestat.result.resultsName), extra = d)

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
                            logger.debug("%04d/%04d %s %10d %s" % (j, dict['total_cnt'], dict['action'], dict['total_size'], path), extra = d)

                    except (OSError,IOError,DMExceptions.RsyncError) as e:
                        #IOError: [Errno 28] No space left on device:
                        if e.errno == errno.ENOSPC:
                            raise
                        elif e.errno == errno.ENOENT or e.errno == errno.ESTALE:
                            logger.warn("%04d No longer exists %s" % (j,path), extra = d)
                            continue
                        else:
                            raise
                    except:
                        errmsg = "%04d/%04d %s %10d %s" % (j, dict['total_cnt'], dict['action'], dict['total_size'], path)
                        logger.error(errmsg, extra = d)
                        logger.error(traceback.format_exc(), extra = d)

                    if not dict['action'] in [EXPORT,TEST] and dmfilestat.dmfileset.del_empty_dir:
                        dir = os.path.dirname(path)
                        try:
                            if len(os.listdir(dir)) == 0:
                                if not "plugin_out" in dir:
                                    try:
                                        os.rmdir(dir)
                                        logger.debug("Removed empty directory: %s" % dir, extra = d)
                                    except Exception as e:
                                        logger.warn("rmdir [%d] %s: %s" % (e.errno,e.strerror,dir), extra = d)
                        except OSError as e:
                            if e.errno == errno.ENOENT:
                                logger.warn("del_empty_dir Does not exist %s" % (path), extra = d)
                                continue
                            else:
                                raise e
                else:
                    break

            # only expect to execute this line when no files to process
            total_processed += dict['total_size']

    except Exception as e:
        fstatus = "Error"
        terminate = True
        dmfilestat.setactionstate('E')
        logger.error("DM Action failure on %s for %s report." % (dmfilestat.dmfileset.type,dmfilestat.result.resultsName), extra = d)
        logger.error("This %s action will need to be manually completed." % (dict['action']), extra = d)
        logger.error("The following is the exception error:\n"+traceback.format_exc(), extra = d)
        EventLog.objects.add_entry(dmfilestat.result,"%s - %s. Action not completed.  User intervention required." % (fstatus, e),username='dm_agent')

        # Release the task lock
        try:
            if dict['lockfile']:
                applock = TaskLock(dict['lockfile'])
                applock.unlock()
        except:
            logger.error(traceback.format_exc(), extra = d)

        # Do the user notification
        try:
            # pop up a message banner
            if dict['msg_banner']:
                dmfileset = dmfilestat.dmfileset
                project_msg = {}
                msg_dict = {}
                msg_dict[dmfileset.type] = fstatus
                project_msg[dmfilestat.result_id] = msg_dict
                project_msg_banner('', project_msg, dict['action'])
        except:
            logger.error(traceback.format_exc(), extra = d)

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
            action = dict.get('action','unk')
            pfilename = set_action_param_var(list_of_file_dict)
            celery_result = _process_task.delay(pfilename)
        except:
            logger.error(traceback.format_exc(), extra = d)

    else:
        # ====================================================================
        # No more files to process.  Clean up and exit.
        # ====================================================================
        try:
            dmfilestat.diskspace = float(total_processed)/(1024*1024)
            dmfilestat.save()
            logger.info("%0.1f MB %s processed" % (dmfilestat.diskspace, dmfilestat.dmfileset.type), extra = d)
            if dict['action'] in [ARCHIVE, DELETE]:
                _brokenlinks_delete([dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir])
                _emptydir_delete([dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir])
        except:
            logger.error(traceback.format_exc(), extra = d)

        # Do the user notification
        try:
            _action_complete_update(dict['user'], dict['user_comment'], dmfilestat, dict['action'])

            # pop up a message banner
            if dict['msg_banner']:
                dmfileset = dmfilestat.dmfileset
                project_msg = {}
                msg_dict = {}
                msg_dict[dmfileset.type] = fstatus
                project_msg[dmfilestat.result_id] = msg_dict
                project_msg_banner(dict['user'], project_msg, dict['action'])
        except:
            logger.error(traceback.format_exc(), extra = d)

        # Release the task lock
        try:
            if dict['lockfile']:
                applock = TaskLock(dict['lockfile'])
                applock.unlock()
        except:
            logger.error(traceback.format_exc(), extra = d)

    return


def _get_keeper_list(dmfilestat, action):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    if action == EXPORT:
        kpatterns = []
    else:
        kpatterns = []
        #Are there entries in dmfilestat.dmfileset.keepwith?
        #logger.debug("FILES IN KEEPWITH FIELD", extra = d)
        #logger.debug(dmfilestat.dmfileset.keepwith, extra = d)
        for type, patterns in dmfilestat.dmfileset.keepwith.iteritems():
            #Are the types specified in dmfilestat.dmfileset.keepwith still local?
            if not dmfilestat.result.dmfilestat_set.get(dmfileset__type=type).isdisposed():
                #add patterns to kpatterns
                kpatterns.extend(patterns)
    logger.debug("Keep Patterns are %s" % kpatterns, extra = d)
    return kpatterns


def _file_removal(filepath, to_keep):
    '''Conditional removal.  Only remove if the filepath is not in the to_keep list'''
    #logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)

    starttime = time.time()
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
            elif e.errno == errno.EACCES:   # permission denied
                logger.info(e, extra = d)
                return False
            else:
                raise e
    else:
        logger.debug("NOT REMOVING: %s" % filepath, extra = d)
        return False

    endtime = time.time()
    if (endtime-starttime) > 1.0:
        logger.warn("%s: %f seconds" % (sys._getframe().f_code.co_name,(endtime-starttime)), extra = d)
    return True


def _print_selected(filepath, to_keep):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    if filepath not in to_keep:
        logger.info("Selected: %s" % filepath, extra = d)
        return True
    return False


def _emptydir_delete(search_dirs):
    '''
    Search for empty directories and delete
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    for start_dir in search_dirs:
        for root, dirs, files in os.walk(start_dir,topdown=True):
            for name in dirs:
                filepath = os.path.join(root,name)
                try:
                    emptyDir = True if len(os.listdir(filepath)) == 0 else False
                except:
                    continue
                if emptyDir and name != "plugin_out":
                    logger.debug("Removing Empty Directory: %s" % filepath, extra = d)
                    #this fancy bit is to not trip on a soft link (TS-6600)
                    try:
                        os.rmdir(filepath)
                    except OSError as e:
                        if e.errno == errno.ENOTDIR:
                            try:
                                os.unlink(filepath)
                            except:
                                logger.warn(e, extra = d)
                        elif e.errno == errno.ENOENT:   # no such file or directory
                            pass
                        else:
                            logger.warn(e, extra = d)


def _brokenlinks_delete(search_dirs):
    '''
    Delete broken links
    '''
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    for start_dir in search_dirs:
        for root, dirs, files in os.walk(start_dir,topdown=True):
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
                                logger.info ("Broken symlink removed: %s" % filepath, extra = d)
                            except OSError as e:
                                logger.warn(e, extra = d)

def _update_diskusage(dmfilestat):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
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
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)

    if action_state is None: action_state = dmfilestat.action_state

#    if action != EXPORT:
#        diskspace = update_diskspace(dmfilestat)

    if dmfilestat.dmfileset.type == dmactions_types.SIG:
        # check that experiment.ftpStatus field is now set to complete.
        dmfilestat.result.experiment.ftpStatus = "Complete"
        dmfilestat.result.experiment.save()
#    else:
#        dmfilestat.action_state = action_state
#        dmfilestat.save()
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

    if action != EXPORT:
        diskspace = update_diskspace(dmfilestat)


def _action_complete_update(user, user_comment, dmfilestat, action):
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)

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
        # save username and comment to be used when action is run
        dmfilestat.user_comment = {'user': user, 'user_comment':user_comment }
        dmfilestat.save()
        msg+= ".<br>User Comment: %s" % user_comment

        if dmfilestat.dmfileset.type == dmactions_types.SIG:
            # update all realated dmfilestats
            exp_id = dmfilestat.result.experiment_id
            DMFileStat.objects.filter(dmfileset__type=dmactions_types.SIG, result__experiment__id=exp_id) \
                .update(action_state = dmfilestat.action_state, archivepath=dmfilestat.archivepath, user_comment=dmfilestat.user_comment)

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
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)
    if os.path.isdir(dmfilestat.result.experiment.expDir):
        if len(os.listdir(dmfilestat.result.experiment.expDir)) == 0:
            logger.debug("expDir exists and is empty: %s" % dmfilestat.result.experiment.expDir, extra = d)
            os.rmdir(dmfilestat.result.experiment.expDir)
            os.symlink(dmfilestat.archivepath,dmfilestat.result.experiment.expDir)
            logger.debug("symlink created: %s" % dmfilestat.result.experiment.expDir, extra = d)
        else:
            logger.debug("no symlink, expDir is not empty: %s" % dmfilestat.result.experiment.expDir, extra = d)
    else:
        logger.debug("expDir no longer exists: %s" % dmfilestat.result.experiment.expDir, extra = d)
        os.symlink(dmfilestat.archivepath,dmfilestat.result.experiment.expDir)
        logger.debug("symlink created: %s" % dmfilestat.result.experiment.expDir, extra = d)


def slugify(something):
    '''convert whitespace to hyphen and lower case everything'''
    return re.sub(r'\W+','-',something.lower())


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
    with tempfile.NamedTemporaryFile(dir=store_dir,delete=False,mode='w+b', prefix=prefix) as fileh:
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
    with open(pfilename,'rb') as fileh:
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
            os.path.basename(report_path)+'-full.pdf'
        ]
        for pdf_file in pdf_files:
            filepath = os.path.join(report_path,pdf_file)
            if os.path.exists(filepath):
                shutil.copy2(filepath, dmfilestat.archivepath)


def write_serialized_json(result, destination):
    sfile = os.path.join(destination, "serialized_%s.json" % result.resultsName)
    #skip if already exists
    if os.path.exists(sfile):
        return

    serialize_objs = [result, result.experiment]
    for obj in [result.experiment.plan, result.eas, result.analysismetrics, result.libmetrics, result.qualitymetrics]:
        if obj:
            serialize_objs.append(obj)
    serialize_objs += list(result.pluginresult_set.all())

    try:
        with open(sfile,'wt') as f:
            obj_json = serializers.serialize('json', serialize_objs, indent=2, use_natural_keys=True)

            dmfilesets = DMFileSet.objects.filter(dmfilestat__result = result)
            obj_json = obj_json.rstrip(' ]\n') + ','
            obj_json += serializers.serialize('json', dmfilesets, indent=2, fields=('type','version')).lstrip('[')

            f.write(obj_json)

        # set file permissions
        try:
            stat = os.stat(destination)
            os.chown(sfile,stat.st_uid,stat.st_gid)
        except:
            pass

    except:
        logger.error("Unable to save serialized.json for %s(%d)" % (result.resultsName, result.pk) , extra = d)
        logger.error(traceback.format_exc(), extra = d)


def get_file_list(dmfilestat):
    """Return list of files selected by this DMFileStat record and list of files to not process.
    There are some cases in which the list of selected files contains files which should not be
    processed.  Those are in the to_keep list."""
    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)

    to_process = []
    to_keep = []
    try:
        # search both results directory and raw data directory
        search_dirs = [dmfilestat.result.get_report_dir(), dmfilestat.result.experiment.expDir]

        cached_file_list = get_walk_filelist(search_dirs, list_dir=dmfilestat.result.get_report_dir())

        #Determine if this file type is eligible to use a keep list
        kpatterns = _get_keeper_list(dmfilestat, '')

        #Create a list of files eligible to process
        isThumbnail = dmfilestat.result.isThumbnail
        for start_dir in search_dirs:
            if os.path.isdir(start_dir):
                tmp_process, tmp_keep = _file_selector(start_dir,
                                                     dmfilestat.dmfileset.include,
                                                     dmfilestat.dmfileset.exclude,
                                                     kpatterns,
                                                     isThumbnail,
                                                     cached=cached_file_list)
                to_process += tmp_process
                to_keep += tmp_keep
            else:
                logger.error(traceback.format_exc(), extra = d)
    except:
        logger.error(traceback.format_exc(), extra = d)
    return (to_process, to_keep)


def get_walk_filelist(dirs, list_dir=None, save_list=False):
    '''
    Purpose of the function is to generate a list of all files rooted in the given directories.
    Since the os.walk is an expensive operation on large filesystems, we can store the file list
    in a text file in the report directory.  Once a report is analyzed, the only potential changes
    to the file list will be in the plugin_out directory.  Thus, we always update the contents of
    the file list for that directory.
    '''
    import pickle
    import stat

    USE_OS_WALK=True
    USE_FILE_CACHE=True

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

    logger.debug("Function: %s()" % sys._getframe().f_code.co_name, extra = d)

    starttime = time.time()
    thelist = []
    if list_dir:
        cachefile = os.path.join(list_dir, "cached.filelist")
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
                for root, dirs, files in os.walk('./',topdown=True):
                    for j in dirs + files:
                        this_file = os.path.join(this_dir, root.replace('./',''), j)
                        if not os.path.isdir(this_file): # exclude directories
                            thelist.append(this_file)
    else:
        # else, generate a list
        for item in [dir for dir in dirs if os.path.isdir(dir)]:
            os.chdir(item)
            for root, dirs, files in os.walk('./',topdown=True):
                for j in dirs + files:
                    this_file = os.path.join(item, root.replace('./',''), j)
                    if not os.path.isdir(this_file): # exclude directories
                        thelist.append(this_file)
                    elif j == 'sigproc_results':
                        # add files from linked sigproc_results folder, except proton onboard_results files
                        if os.path.islink(this_file) and 'onboard_results' not in os.path.realpath(this_file):
                            for root2, dirs2, files2 in os.walk(this_file):
                                thelist += [os.path.join(root2,file) for file in files2]

    endtime = time.time()
    logger.info("%s: %f seconds" % (sys._getframe().f_code.co_name,(endtime-starttime)), extra = d)

    if save_list:
        try:
            dump_the_list(cachefile, thelist)
        except:
            logger.error(traceback.format_exc(), extra = d)

    return thelist
