# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import datetime
import json
import os
from os import path
import shutil
import socket
import sys
import threading
import time
import re
import traceback
import fnmatch
import logging
import logging.handlers
from twisted.internet import task
from twisted.internet import reactor
from twisted.web import xmlrpc, server

import iondb.bin.djangoinit
from iondb.bin.djangoinit import *
from django import shortcuts
from django.conf import settings
from iondb.rundb import models
from iondb.backup import devices
from django.core import mail
from django.db import connection
from iondb.backup.archiveExp import Experiment

settings.EMAIL_HOST = 'localhost'
settings.EMAIL_PORT = 25
settings.EMAIL_USE_TLS = False

#Globals
gl_experiments = {}
last_exp_size = 0

__version__ = filter(str.isdigit, "$Revision: 49091 $")

# TODO: these are defined in crawler.py as well.  Needs to be consolidated.
RUN_STATUS_COMPLETE = "Complete"
RUN_STATUS_MISSING = "Missing File(s)"
RUN_STATUS_ABORT = "User Aborted"
RUN_STATUS_SYS_CRIT = "Lost Chip Connection"

PGM_INSTRUMENT = 'PGM'
PROTON_INSTRUMENT = 'PROTON'

class Email():

    def __init__(self):
        self.sent = False
        self.date_sent = None

    def send_drive_warning(self, recipient):
        if not self.sent or datetime.datetime.now() > self.date_sent + datetime.timedelta(days=1):
            if recipient != '':
                mail.send_mail('Archive Full on %s' % socket.gethostname(),
                               'The archive drive on %s is full or missing.  Please replace it.' % socket.gethostname(),
                               'donotreply@iontorrent.com', (recipient,))
                self.date_sent = datetime.datetime.now()
                self.sent = True

    def reset(self):
        self.sent = False


def notify(log, experiments, recipient):
    # helper function to update user_ack field

    def updateExpModelUserAck(e):
        exp = models.Experiment.objects.get(pk=e.pk)
        exp.user_ack = e.user_ack
        exp.save()
        return

    # Check for blank email
    # TODO: should check for valid email address
    if recipient is None or recipient == "":
        return

    try:
        site_name = models.GlobalConfig.get().site_name
    except:
        site_name = "Torrent Server"

    hname=socket.gethostname()
    
    subject_line = 'Torrent Server Archiver Action Request'
    reply_to = 'donotreply@iontorrent.com'
    message = 'From: %s (%s)\n' % (site_name, hname)
    message += '\n'
    message += 'Results drive capacity threshold has been reached.\n'
    message += 'Raw datasets have been identified for removal.\n'
    message += 'Please go to Services Page and acknowledge so that removal can proceed.\n'
    message += 'Removal will not occur without this acknowledgement.\n'
    message += '\n'
    message += 'The following experiments are selected for Deletion:'
    message += "\n"
    count = 0
    for e in experiments:
        if e.user_ack == 'S':
            message += "- %s\n" % e.name
            count += 1
            # set to Notified so that only one email gets sent
            e.user_ack = 'N'
            updateExpModelUserAck(e)

    # Send the email only if there are runs that have not triggered a notification
    if count > 0:
        mail.send_mail(subject_line, message, reply_to, [recipient])
        log.info("Notification email sent for user acknowledgement")


def build_exp_list(num, grace_period, serverPath, removeOnly, log, autoArchiveAck):  # removeOnly needs to be implemented
    '''
    Build a list of experiments to archive or delete

    Filter out grace period runs based on start time, runs marked Keep

    TODO: Filter out runs whose FTP status indicates they are still transferring

    Return n oldest experiments to archive or delete

    Remove only mode is important feature to protect against the condition wherein the archive volume is not available
    and there are runs marked for deletion.  Without remove only mode, the list of runs to process could get filled with
    Archive runs, and the Delete Runs would never get processed.
    
    When auto-acknowledge is disabled (the default) and an archive volume is configured, there can be a situation where
    the list of experiments is filled with Delete and the Archive experiments are not included and thus never processed -
    if the user fails to manually acknowledge deletion.
    
    But also, the remove_experiments function handles the Delete first because those are faster and free space quicky while
    archive can take a long time copying.
    '''
    exp = models.Experiment.objects.all().order_by('date')
    log.debug("Total Experiments: %d" % len(exp))
    if removeOnly:
        exp = exp.filter(storage_options='D').exclude(expName__in=models.Backup.objects.all().values('backupName'))
        log.debug("Deletable %d" % len(exp))
    else:
        exp = exp.exclude(storage_options='KI').exclude(expName__in=models.Backup.objects.all().values('backupName'))
        log.debug("Deletable or Archivable %d" % len(exp))
    # backupConfig

    # local time from which to measure time difference
    timenow = time.localtime(time.time())

    experiments = []
    for e in exp:
        log.debug('Experiment date %s' % str(e.date))
        if len(experiments) < num:  # only want to loop until we have the correct number

            if not os.path.isdir(e.expDir):
                #Create an entry in the Backup db.
                kwargs = {"experiment": e,
                          "backupName": e.expName,
                          # This is True when the data has been archived.  Since its missing, it hasn't been archived
                          "isBackedUp": False,
                          "backupDate": datetime.datetime.now(),
                          "backupPath": "DELETED"}
                ret = models.Backup(**kwargs)
                ret.save()
                log.info("Raw Data missing: %s.  Creating Backup object" %
                         e.expDir)
                continue

            # Instead of using the experiment date field, which is set to when the experiment was performed
            # use the file timestamp of the first (or last?) .dat file when calculating grace period
            diff = time.mktime(timenow) - time.mktime(datetime.datetime.timetuple(e.date))
            log.debug('Time since experiment was run %d' % diff)

            # grace_period units are hours
            if diff < (grace_period * 3600):  # convert hours to seconds
                log.info('Within grace period: %s' % e.expName)
                continue
# TS-2736 Do not archive/delete Runs that are still FTP transferring.
# What about runs that are stuck forever?  Without this test, those types of runs eventually get archived or deleted
# with this test in place, they will stay on server forever.
#            # If run is still transferring, skip
#            if e.ftpStatus.strip() == RUN_STATUS_COMPLETE or \
#               e.ftpStatus.strip() == RUN_STATUS_ABORT or \
#               e.ftpStatus.strip() == RUN_STATUS_MISSING or \
#               e.ftpStatus.strip() == RUN_STATUS_SYS_CRIT:
#                pass
#            else:
#                logger.errors.debug("Skip this one, still transferring: %s" % e.expName)
#                continue

            experiment = Experiment(e,
                                    str(e.expName),
                                    str(e.date),
                                    str(e.star),
                                    str(e.storage_options),
                                    str(e.user_ack),
                                    str(e.expDir),
                                    e.pk,
                                    str(e.rawdatastyle),
                                    str(e.diskusage) if e.diskusage is not None else "Unknown")

            try:
                # don't add anything to the list if its already archived
                bk = models.Backup.objects.get(
                    backupName=experiment.get_exp_name())
                log.debug('This has been archived')
                continue
            except:
                # check that the path exists, and double check that its not marked to 'Keep'
                if not path.islink(experiment.get_exp_path()) and \
                        path.exists(experiment.get_exp_path()) and \
                        experiment.get_storage_option() != 'KI' and \
                        experiment.dir.startswith(serverPath):
                    # change user ack status from unset to Selected: triggers an email notification
                    if e.user_ack == 'U':
                        e.user_ack = 'S'
                        e.save()
                        experiment.user_ack = 'S'
                    # Runs marked Archive are automatically handled so set them as Acknowledged:
                    if experiment.get_storage_option() == 'A':
                        e.user_ack = 'A'
                        e.save()
                        experiment.user_ack = 'A'
                    # If global settings are set to auto-acknowledge, then set user_ack as acknowledged
                    if autoArchiveAck:
                        e.user_ack = 'A'
                        e.save()
                        experiment.user_ack = 'A'

                    experiments.append(experiment)
                #DEBUG MODE: show why the experiment is or is not valid for archiving
                log.debug("Path: %s" % experiment.get_exp_path())
                log.debug("Does path exist? %s" % ('yes' if path.exists(
                    experiment.get_exp_path()) else 'no'))
                log.debug("Is path a link? %s" % ('yes' if path.islink(
                    experiment.get_exp_path()) else 'no'))
                log.debug("Is storage option not Keep? %s" % ('yes' if experiment.get_storage_option() != 'KI' else 'no'))
                log.debug("Is %s in %s? %s" % (serverPath, experiment.dir,
                          'yes' if serverPath in experiment.dir else 'no'))
                log.debug("User Ack is set to %s" % experiment.user_ack)
        else:
            log.debug("Number of experiments: %d" % len(experiments))
            return experiments
    log.debug("Number of experiments: %d" % len(experiments))
    return experiments


def get_server_full_space(log):
    try:
        fileservers = models.FileServer.objects.all()
        num = len(fileservers)
    except:
        log.error(traceback.print_exc())
    ret = []
    for fs in fileservers:
        if path.exists(fs.filesPrefix):
            ret.append((fs.filesPrefix, fs.percentfull))
    return ret


def add_to_db(exp, path, archived):
    # Update dbase Experiment object's user_ack field
    try:
        e = models.Experiment.objects.get(pk=exp.get_pk())
        e.user_ack = 'D'
        e.save()
        # Create dbase Backup object for this Experiment
        kwargs = {"experiment": e,
                  "backupName": exp.name,
                  "isBackedUp": archived,
                  "backupDate": datetime.datetime.now(),
                  "backupPath": path}
        ret = models.Backup(**kwargs)
        ret.save()
    except:
        raise
    return


def dispose_experiments(log, experiments, backupDrive, number_to_backup, backupFreeSpace, bwLimit, backup_pk, email, ADMIN_EMAIL):
    global last_exp_size
    # Flag for debugging: set to True to not really delete or archive
    JUST_TESTING = False
    updateNeeded = False
    #Hack: reorder items in list to have Delete experiments first, followed by Archive
    deleteable = [exp for exp in experiments if exp.get_storage_option() == 'D']
    log.debug("Deleteable: %d" % len(deleteable))
    archiveable = [exp for exp in experiments if exp.get_storage_option() == 'A']
    log.debug("Archiveable: %d" % len(archiveable))
    experiments = []
    experiments.extend(deleteable)
    experiments.extend(archiveable)
    isConf, params = get_params(log)
    log.info("Processing Datasets")
    for exp in experiments:
        expDir = exp.get_exp_path()
        log.info('Inspecting %s' % expDir)
        log.info('Storage option is %s' % exp.get_storage_option())
        log.info('Has user acknowledged? %s (%s)' % ("Yes" if str(exp.user_ack) == 'A' else "No", exp.user_ack))
        if str(exp.user_ack) != 'A':
            # Skip this experiment if user has not acknowledged action
            # user will acknowledge on the Services Tab
            continue

        copycomplete = False
        removecomplete = False
        try:
            exp_size = exp.get_folder_size()
        except:
            log.error(traceback.format_exc())
            exp_size = 0
        last_exp_size = exp_size
        if backupDrive is not None and backupFreeSpace is not None:
            if exp.get_storage_option() == 'A' or exp.get_storage_option() == 'PROD':
                if backupFreeSpace > exp_size:
                    email.reset()
                    try:
                        set_status(log, 'Copying %s' % expDir)
                        if not JUST_TESTING:
                            copycomplete = copyrawdata(log, expDir, backupDrive, bwLimit)
                        else:
                            log.info("TESTING MODE is ON.  No data was actually copied")
                            copycomplete = True
                    except:
                        log.error("Failed to copy directory %s" % expDir)
                else:
                    log.warn(
                        "Archive Drive is full or missing, please replace")
                    set_status(log, 'Archive drive is full or missing, please replace.')
                    email.send_drive_warning(ADMIN_EMAIL)
        else:
            log.info("No Archive drive")

        if exp.get_storage_option() == 'D' or copycomplete:
            try:
                set_status(log, 'Removing %s' % expDir)
                log.info('Removing %s' % expDir)
                if not JUST_TESTING:
                    inst_type = PROTON_INSTRUMENT if 'tiled' in exp.rawdatastyle else PGM_INSTRUMENT
                    removecomplete = remove_raw_data(log, expDir, inst_type)
                else:
                    log.info("TESTING MODE is ON.  Directory was not actually removed")
                    removecomplete = True
                #TODO: Update the user_ack flag for this experiment
            except:
                log.error("Failed to remove directory %s" % expDir)
                log.error(traceback.format_exc())

        if copycomplete and removecomplete and not JUST_TESTING:
            linkPath = path.join(backupDrive, expDir.strip().split("/")[-1])
            log.info("Linking %s to %s" % (linkPath, expDir))
            set_status(log, 'Linking %s to %s' % (linkPath, expDir))
            #os.umask(0002)
            try:
                # Proton dataset deletion omits onboard_results folder - delete exp dir here
                if inst_type == PROTON_INSTRUMENT:
                    remove_dir(log, expDir)
                os.symlink(linkPath, expDir)
                os.chmod(linkPath, 0775)
            except:
                log.error(traceback.format_exc())
                add_to_db(exp, linkPath, True)

        elif removecomplete and not JUST_TESTING:
            add_to_db(exp, 'DELETED', False)
        else:
            log.debug("copycomplete returned: %s" % copycomplete)
            log.debug("removecomplete returned: %s" % removecomplete)
        # Record a return value that will trigger a FS check
        if removecomplete:
            updateNeeded = True

    return updateNeeded


def copyrawdata(log, expDir, backupDir, bwLimit):
    def to_bool(number):
        return True if number == 0 else False
    
    log.info("Copying %s to %s " % (expDir, backupDir))
    rawPath = path.join(expDir)
    backup = path.join(backupDir)
    bwLimitS = "--bwlimit=%s" % bwLimit
    if path.exists(backup):
        try:
            status = os.system(
                "rsync -rpt %s %s %s" % (rawPath, backup, bwLimitS))
            if status:
                log.error(sys.exc_info()[0])

            return to_bool(status)

        except Exception, err:
            log.error(sys.exc_info()[0])
            log.error(err)
            return False
    else:
        log.warn("No path to Archive Drive")


def remove_dir(log, expDir):
    try:
        shutil.rmtree(expDir)
        if not path.exists(expDir):  # double check the folder is gone
            return True
        else:
            return False
    except:
        exc = traceback.format_exc()
        log.error(exc)
        return False


def remove_raw_data(log, expDir, inst_type):
    '''
    For PGM, delete the entire raw data directory
    For Proton, delete all but onboard_results directory
    '''
    
    # if its PGM data, delete the entire folder
    if inst_type == PGM_INSTRUMENT:
        log.debug("Deleting the entire folder")
        return remove_dir(log, expDir)

    # if its PROTON data, delete all subfolders except 'onboard_results'
    success = True
    log.debug("Deleting all but onboard_results/ in %s" % expDir)
    for name in os.listdir(expDir):
        if fnmatch.fnmatch(name, 'onboard_results'):
            continue
        elif os.path.isdir(os.path.join(expDir, name)):
            if remove_dir(log, os.path.join(expDir, name)) is False:
                success = False
        else:
            os.unlink(os.path.join(expDir, name))

    return success


def get_params(log):
    try:
        bk = models.BackupConfig.get()
        dict = {'NUMBER_TO_BACKUP': bk.number_to_backup,
                'TIME_OUT': bk.timeout,
                'BACKUP_DRIVE_PATH': bk.backup_directory,
                'BACKUP_THRESHOLD': bk.backup_threshold,
                'BANDWIDTH_LIMIT': bk.bandwidth_limit,
                'GRACE_PERIOD': bk.grace_period,
                'PK': bk.pk,
                'EMAIL': bk.email,
                'enabled': bk.online,
                'keepTN': False,  # We'll want to create a boolean in backupconfig object.
                }
        configured = True
    except:
        log.error(traceback.format_exc())
        configured = False
        dict = {'enabled': False}
    finally:
        return configured, dict


def json_out(log, data, fileout):
    try:
        f = open(fileout, 'w')
        f.write(json.dumps(data))
    except:
        log.error('failed to open file %s' % fileout)
    f.close()


def set_status(log, status):
    try:
        bk = models.BackupConfig.get()
        bk.status = status
        bk.save()
    except:
        log.error('No configured backup objects found in database')


def loopScheduler(log):
    global gl_experiments
    """Run the loop functions repeatedly."""
    LOOP = 6     # Wait this many seconds before executing loop again
    timeBefore = 0     # initialize to zero will force update at startup
    DELAY = 300   # default if not in params

    def checkFSTimer(delay, timeBefore, timeNow):
        diff = timeNow - timeBefore
        #print "DEBUG:\nNow:%d\nBefore:%d\ndiff:%d" % (timeNow,timeBefore,diff)
        if timeBefore == 0:
            return True
        if diff >= delay:
            return True
        else:
            return False

    while True:
        connection.close()  # Close any db connection to force new one.
        try:
            isConf, params = get_params(log)
            if isConf:
                DELAY = params['TIME_OUT']  # how often to check filesystem
            # The FS check is run less frequently than loop to conserve system
            # resources.
            timeNow = time.mktime(time.localtime(time.time()))
            if checkFSTimer(DELAY, timeBefore, timeNow):
                log.debug("Its been %d seconds since the last FS check" %
                          (timeNow - timeBefore))
                update_experiment_list(log, isConf, params)
                timeBefore = time.mktime(time.localtime(time.time()))
            else:
                pass
                # just update status of existing list
                #gl_experiments = refreshVals(gl_experiments)
            # The backup code is called frequently in order to be responsive to
            # user's acknowledge action on the webpage.
            updateNeeded = removeRuns(log, isConf, params)
            if updateNeeded:
                update_experiment_list(log, isConf, params)
                timeBefore = time.mktime(time.localtime(time.time()))
        except:
            log.error(traceback.format_exc())

        time.sleep(LOOP)


def update_experiment_list(log, isConf, params):
    '''Produces a list of candidate experiments to delete or archive'''
    global gl_experiments
    gl_experiments = {}  # clear the list of Runs to process
    try:
        autoArchiveAck = GetAutoArchiveAck_GC()
        if autoArchiveAck:
            log.info("User acknowledgement to archive is not required")

        if isConf and params['enabled']:
            NUMBER_TO_BACKUP = params['NUMBER_TO_BACKUP']

            PERCENT_FULL_BEFORE_BACKUP = params['BACKUP_THRESHOLD']
            ADMIN_EMAIL = params['EMAIL']
            GRACE_PERIOD = params['GRACE_PERIOD']
            backupFreeSpace, bkpPerFree, removeOnly = get_archive_report(log, params)
            serverFullSpace = get_server_full_space(log)

            if bkpPerFree:
                # when bkpPerFree is undefined, there is no backup drive mounted
                log.info("Archive Drive Free Space = %0.2f" % bkpPerFree)

            log.info("Backup trigger percentage: %0.2f %% full" %
                     PERCENT_FULL_BEFORE_BACKUP)

            for serverPath, percentFull in serverFullSpace:

                log.info("Results Drive %s Used Space = %0.2f" %
                         (serverPath, percentFull))

                if percentFull > PERCENT_FULL_BEFORE_BACKUP:
                    log.info("Building list of experiments")
                    experiments = build_exp_list(NUMBER_TO_BACKUP, GRACE_PERIOD, serverPath, removeOnly, log, autoArchiveAck)
                    gl_experiments[serverPath] = experiments

                    notify(log, experiments, ADMIN_EMAIL)
                else:
                    log.info("No experiments for processing because threshold not reached")

            if len(serverFullSpace) == 0:
                log.warn("No fileservers configured")

        else:
            log.warn("Archiving is disabled or not configured")
    except:
        log.error(traceback.format_exc())
        set_status(log, "ERROR")
    return


def removeRuns(log, isConf, params):
    '''Delete or archive the experiments'''
    #
    # Classic design runs thru each file server in sequence, runs thru each experiment in sequence
    #
    # A few improvements:
    #     - For each server, first delete any deletable runs, then start archiving
    #       This still can cause trouble with servers at end of list not getting serviced
    #     - Spawn backup in separate threads, one per server.  Then each server will get
    #       runs deleted quickly, but archiving will now be in parallel - trouble for archive volume?
    #
    global gl_experiments
    email = Email()
    updateNeeded = False
    try:
        set_status(log, "Reviewing System")
        NUMBER_TO_BACKUP = params['NUMBER_TO_BACKUP']
        PERCENT_FULL_BEFORE_BACKUP = params['BACKUP_THRESHOLD']
        ADMIN_EMAIL = params['EMAIL']
        BACKUP_DRIVE_PATH = params['BACKUP_DRIVE_PATH']
        BANDWIDTH_LIMIT = params['BANDWIDTH_LIMIT']
        BACKUP_PK = params['PK']

        backupFreeSpace, bkpPerFree, removeOnly = get_archive_report(log, params)

        serverFullSpace = get_server_full_space(log)
        for serverPath, percentFull in serverFullSpace:

            if percentFull > PERCENT_FULL_BEFORE_BACKUP:
                log.info("%s : %.2f %% full" % (serverPath, percentFull))
                set_status(log, "Backing Up")
                if len(gl_experiments) == 0:
                    log.info("No datasets to process")
                else:
                    updateNeeded = dispose_experiments(
                        log, gl_experiments[serverPath], BACKUP_DRIVE_PATH,
                        NUMBER_TO_BACKUP, backupFreeSpace,
                        BANDWIDTH_LIMIT, BACKUP_PK, email, ADMIN_EMAIL)
        set_status(log, "Idle")
    except:
        log.error(traceback.format_exc())
        set_status(log, "ERROR")

    return updateNeeded


def GetAutoArchiveAck_GC():
    try:
        gc = models.GlobalConfig.get()
        # If auto_archive_ack is True, then autoArchiveAck is True
        autoAck = gc.auto_archive_ack
    except:
        return False
    return autoAck


def _cleanup():
    bk = models.BackupConfig.get()
    bk.status = 'Off'
    bk.save()
    sys.exit(0)


def get_archive_report(log, params):
    backupFreeSpace = None
    bkpPerFree = None
    dev = devices.disk_report()
    removeOnly = False
    if params['BACKUP_DRIVE_PATH'] != 'None':
        for d in dev:
            if d.get_path() == params['BACKUP_DRIVE_PATH']:
                bkpPerFree = d.get_free_space()
                backupFreeSpace = int(d.get_available() * 1024)
                if backupFreeSpace < last_exp_size:
                    removeOnly = True
                    log.warn(
                        'Archive drive is full, entering Remove Only Mode.')
                return backupFreeSpace, bkpPerFree, removeOnly
    else:
        removeOnly = True
        log.debug('No archive drive, entering Remove Only Mode.')
    return backupFreeSpace, bkpPerFree, removeOnly


def refreshVals(experiments):
    for key in experiments.keys():
        for exp in experiments[key]:
            new_exp = models.Experiment.objects.get(pk=exp.pk)
            exp.storage_opt = str(new_exp.storage_options)
            exp.user_ack = str(new_exp.user_ack)
    return experiments


class Status(xmlrpc.XMLRPC):
    """Allow remote access to this server"""
    def __init__(self, _log, _reportLogger):
        xmlrpc.XMLRPC.__init__(self)
        self.log = _log  # ionArchive logfile: /var/log/ion/iarchive.log
        self.start_time = datetime.datetime.now()
        self.dmLog = _reportLogger  # Data Management logfile: /var/log/ion/reportsLog.log

    # N.B. This is no longer used to get the list of experiments
    # views.py=>db_backup() function queries dbase directly
    def xmlrpc_next_to_archive(self):
        retdict = {}
        return retdict

    def xmlrpc_user_ack(self):
        '''Called when archiving acknowledged state is changed for each experiment'''
        # We use this function to wake up the daemon and start the backup process
        # Typically, we expect about 10 calls to this function at a time
        global gl_experiments
        try:
            self.log.debug("Got an acknowledge xmlrpc from %s" %
                           self.request.getClientIP())
        except:
            self.log.error(traceback.format_exc())
        gl_experiments = refreshVals(gl_experiments)
        return True

    def render(self, request):
        self.request = request
        return xmlrpc.XMLRPC.render(self, request)

    def xmlrpc_status_check(self):
        """Return the amount of time the ionArchive service has been running."""
        try:
            diff = datetime.datetime.now() - self.start_time
            seconds = float(diff.days * 24 * 3600)
            seconds += diff.seconds
            seconds += float(diff.microseconds) / 1000000.0
            self.log.debug("Uptime called - %d (s)" % seconds)
        except:
            self.log.error(traceback.format_exc())
            seconds = 0
        return seconds

    def xmlrpc_remove_only_mode(self):
        '''Returns boolean indicating remove_only mode.
        If remove_only is True, Runs designated to be archived will not be processed.'''
        removeOnly = True
        try:
            isConf, params = get_params(self.log)
            if isConf:
                backupFreeSpace, bkpPerFree, removeOnly = get_archive_report(self.log, params)
        except:
            self.log.error(traceback.format_exc())
        return removeOnly

    def xmlrpc_archive_report(self, pk, comment):
        from iondb.backup import ion_archiveResult
        result = shortcuts.get_object_or_404(models.Results, pk=pk)
        if result.reportStatus in ion_archiveResult.STATUS:
            return True
        try:
            self.log.debug('xmlrpc_archive_report: %s' % (result.resultsName))
            ion_archiveResult.archiveReportShort(pk, comment, self.dmLog)
            self.log.debug('xmlrpc_archive_report: completed.')
            return True
        except Exception as inst:
            result.updateMetaData("Failure", "Error: %s" % inst, 0, comment)
            self.log.error('xmlrpc_archive_report: %s' % inst)
            return False

    def xmlrpc_export_report(self, pk, comment):
        from iondb.backup import ion_exportResult, ion_archiveResult
        result = shortcuts.get_object_or_404(models.Results, pk=pk)
        if result.reportStatus in ion_archiveResult.STATUS:
            return False
        try:
            self.log.debug('xmlrpc_export_report: %s' % (result.resultsName))
            ion_exportResult.exportReportShort(pk, comment, self.dmLog)
            self.log.debug('xmlrpc_export_report: completed.')
            return True
        except Exception as inst:
            result.updateMetaData("Failure", "Error: %s" % inst, 0, comment)
            self.log.error('xmlrpc_export_report: %s' % inst)
            return False

    def xmlrpc_prune_report(self, pk, comment):
        from iondb.backup import ion_pruneReport, ion_archiveResult
        result = shortcuts.get_object_or_404(models.Results, pk=pk)
        if result.reportStatus in ion_archiveResult.STATUS:
            return False
        try:
            self.log.debug('xmlrpc_prune_report: %s' % (result.resultsName))
            ion_pruneReport.pruneReport(pk, comment, self.dmLog)
            self.log.debug('xmlrpc_prune_report: completed.')
            return True
        except Exception as inst:
            result.updateMetaData("Failure", "Error: %s" % inst, 0, comment)
            self.log.error('xmlrpc_prune_report: %s' % inst)
            return False

    def xmlrpc_delete_report(self, pk, comment):
        self.log.debug('xmlrpc_delete_report: REPORT DELETE IS DISABLED! pk=%s comment=%s' % (pk, comment))
        return False


def checkThread(thread, log, _reactor):
    '''Checks thread for aliveness.
    If a valid reactor object is passed, the reactor will be stopped
    thus stopping the daemon entirely.'''
    thread.join(1)
    if not thread.is_alive():
        log.critical("loop thread is dead")
        if _reactor.running:
            log.critical("ionArchive daemon is exiting")
            _reactor.stop()

#Setting up a socket based logging receiver
#http://docs.python.org/howto/logging-cookbook.html#network-logging
import pickle
#import logging
#import logging.handlers
import SocketServer
import struct


class LogRecordStreamHandler(SocketServer.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)


class LogRecordSocketReceiver(SocketServer.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver suitable for testing.
    """

    allow_reuse_address = 1

    def __init__(self, host='localhost',
                 #port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 port=settings.DM_LOGGER_PORT,
                 handler=LogRecordStreamHandler):
        SocketServer.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = "reportLogger"

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()],
                                       [], [],
                                       self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort


def main():
    # Setup log file logging
    filename = '/var/log/ion/iarchive.log'
    archlog = logging.getLogger('archlog')
    archlog.propagate = False
    archlog.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(
        filename, maxBytes=1024 * 1024 * 10, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    archlog.addHandler(handler)

    #handle keyboard interrupt; others
    import signal
    signal.signal(signal.SIGTERM, _cleanup)

    #set status in database Backup model
    set_status(archlog, 'Service Started')

    archlog.info('ionArchive Started Ver: %s' % __version__)

    # start loopScheduler which runs 'loop' periodically in a thread
    loopfunc = lambda: loopScheduler(archlog)
    lthread = threading.Thread(target=loopfunc)
    lthread.setDaemon(True)
    lthread.start()

    # check thread health periodically and exit if thread is dead
    # pass in reactor object below to kill process when thread dies
    l = task.LoopingCall(checkThread, lthread, archlog, reactor)
    l.start(30.0)  # call every 30 second

    # define a logging handler for the reportsLog.log file (Data Management Logger)
    filename = '/var/log/ion/reportsLog.log'
    reportLogger = logging.getLogger('reportLogger')
    reportLogger.propagate = False
    reportLogger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler = logging.handlers.RotatingFileHandler(
        filename, maxBytes=1024 * 1024 * 10, backupCount=5)
    handler.setFormatter(formatter)
    reportLogger.addHandler(handler)

    # set up listener for log events, in a thread
    tcpserver = LogRecordSocketReceiver()
    print('About to start TCP server...')
    tfunc = lambda: tcpserver.serve_until_stopped()
    logThread = threading.Thread(target=tfunc)
    logThread.setDaemon(True)
    logThread.start()

    # start the xml-rpc server
    print('About to start reactor...')
    r = Status(archlog, reportLogger)
    reactor.listenTCP(settings.IARCHIVE_PORT, server.Site(r))
    reactor.run()

if __name__ == "__main__":
    sys.exit(main())
