# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import datetime
try:
    import json
except ImportError:
    import simplejson as json
import os
from os import path
import shutil
import socket
import sys
import threading
import time
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from django import db
from django.conf import settings
from twisted.internet import task
from twisted.internet import reactor
from twisted.web import xmlrpc,server
from djangoinit import *
from iondb.rundb import models, views
import logging
import logging.handlers
from iondb.backup import devices
from django.core import mail
import traceback
from django.db import connection, transaction
from iondb.backup.archiveExp import Experiment

settings.EMAIL_HOST = 'localhost'
settings.EMAIL_PORT = 25
settings.EMAIL_USE_TLS = False
last_exp_size = 0

#Global
gl_experiments = {}

__version__ = filter(str.isdigit, "$Revision: 21186 $")


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

def notify (log, experiments, recipient):
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
    
    subject_line = 'Torrent Server Archiver Action Request'
    reply_to = 'donotreply@iontorrent.com'
    message = """
Results drive capacity threshold has been reached.
Raw datasets have been identified for removal.
Please go to Services Tab and acknowledge so that removal can proceed.
Removal will not occur without this acknowledgement.

The following experiments are selected for Deletion:"""

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
        mail.send_mail(subject_line,message,reply_to,[recipient])
        log.info("Notification email sent for user acknowledgement")
    
def build_exp_list(cur_loc, num, grace_period, serverPath, removeOnly, log, autoArchiveAck): #removeOnly needs to be implemented
    '''
    Build a list of experiments to archive or delete
    
    Filter out grace period runs based on start time, runs marked Keep
    
    Return n oldest experiments to archive or delete
    
    Remove only mode is important feature to protect against the condition wherein the archive volume is not available
    and there are runs marked for deletion.  Without remove only mode, the list of runs to process could get filled with
    Archive runs, and the Delete Runs would never get processed.
    '''
    debug = True
    exp = models.Experiment.objects.all().order_by('date')
    log.debug ("Total Experiments: %d" % len(exp))
    if removeOnly:
        exp = exp.filter(storage_options='D').exclude(expName__in = models.Backup.objects.all().values('backupName'))
        log.debug ("Deletable %d" % len(exp))
    else:
        exp = exp.exclude(storage_options='KI').exclude(expName__in = models.Backup.objects.all().values('backupName'))
        log.debug ("Deletable or Archivable %d" % len(exp))
    # backupConfig
    
    # local time from which to measure time difference
    timenow = time.localtime(time.time())

    experiments = []
    for e in exp:
        if debug:
            log.debug('Experiment date %s' % str(e.date))
        if len(experiments) < num: # only want to loop until we have the correct number
            
            location = server_and_location(e)
            if location == None:
                continue
            
            diff = time.mktime(timenow) - time.mktime(datetime.datetime.timetuple(e.date))
            if debug:
                log.debug ('Time diff is %d' % diff)
            # grace_period units are hours
            if diff < (grace_period * 3600):    #convert hours to seconds
                continue
            
            experiment = Experiment(e,
                                    str(e.expName),
                                    str(e.date),
                                    str(e.star),
                                    str(e.storage_options),
                                    str(e.user_ack),
                                    str(e.expDir),
                                    location,
                                    e.pk)
                        
            try:
                # don't add anything to the list if its already archived
                bk = models.Backup.objects.get(backupName=experiment.get_exp_name())
                if debug:
                    log.debug('This has been archived')
                continue 
            except:
                # check that the path exists, and double check that its not marked to 'Keep'
                if not path.islink(experiment.get_exp_path()) and \
                        path.exists(experiment.get_exp_path()) and \
                        experiment.location==cur_loc and \
                        experiment.get_storage_option() != 'KI' and \
                        serverPath in experiment.dir:
                    # change user ack status from unset to Selected: triggers an email notification
                    if e.user_ack == 'U':
                        e.user_ack = 'S'
                        e.save()
                        experiment.user_ack = 'S'
                    # Only runs marked Delete should require acknowledge; Archive should go automatically
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
                if debug:
                    log.debug("Path: %s" % experiment.get_exp_path())
                    log.debug("Does path exist? %s" % ('yes' if path.exists(experiment.get_exp_path()) else 'no'))
                    log.debug("Is path a link? %s" % ('yes' if path.islink(experiment.get_exp_path()) else 'no'))
                    log.debug("Is exp location same as loc? %s" % ('yes' if experiment.location==cur_loc else 'no'))
                    log.debug("Is storage option not Keep? %s" % ('yes' if  experiment.get_storage_option() != 'KI' else 'no'))
                    log.debug("Is %s in %s? %s" % (serverPath,experiment.dir,'yes' if serverPath in experiment.dir else 'no'))
                    log.debug("User Ack is set to %s" % experiment.user_ack)
        else:
            log.debug("Number of experiments: %d" % len(experiments))
            return experiments
    log.debug("Number of experiments: %d" % len(experiments))
    return experiments

def server_and_location(experiment):
    try:
        loc = models.Rig.objects.get(name=experiment.pgmName).location
    except:
        return None
    #server = models.FileServer.objects.filter(location=loc)
    return loc

def get_server_full_space(cur_loc):
    fileservers = models.FileServer.objects.filter(location=cur_loc)
    ret = []
    for fs in fileservers:
        if path.exists(fs.filesPrefix):
            ret.append(free_percent(fs.filesPrefix))
    return ret

def free_percent(path):
    resDir = os.statvfs(path)
    totalSpace = resDir.f_blocks
    freeSpace = resDir.f_bavail
    return (path,100-(float(freeSpace)/float(totalSpace)*100))

def add_to_db(exp, path, archived):
    e = models.Experiment.objects.get(pk=exp.get_pk())
    kwargs = {"experiment": e,
              "backupName": exp.name, 
              "isBackedUp": archived,
              "backupDate": datetime.datetime.now(),
              "backupPath": path
              }
    ret = models.Backup(**kwargs)
    ret.save()
    return ret

def backup(log, experiments, backupDrive, number_to_backup, backupFreeSpace, bwLimit, backup_pk, email, ADMIN_EMAIL):
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
    log.info("Running Backup")
    for exp in experiments:
        expDir = exp.get_exp_path()
        pk = exp.get_pk()
        log.info('Currently inspecting %s' % expDir)
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
        last_exp_size = exp_size
        if backupDrive != None and backupFreeSpace != None:
            if exp.get_storage_option() == 'A' or exp.get_storage_option() == 'PROD':
                if backupFreeSpace > exp_size:
                    email.reset()
                    try:
                        set_status(log, 'Copying %s' % expDir)
                        if not JUST_TESTING:
                            copycomplete = copy(log, expDir, backupDrive, bwLimit)
                        else:
                            log.info ("TESTING MODE is ON.  No data was actually copied")
                            copycomplete = True
                    except:
                        log.error("Failed to copy directory %s" % expDir)
                else:
                    log.info("Backup Drive is full or missing, please replace")
                    set_status(log, 'Backup drive is full or missing, please replace.')
                    email.send_drive_warning(ADMIN_EMAIL)
        else:
            log.info("No backup drive")
            
        if exp.get_storage_option() == 'D' or copycomplete:
            try:
                set_status(log, 'Removing %s' % expDir)
                log.info('Removing %s' % expDir)
                if not JUST_TESTING:
                    removecomplete = remove_dir(log,expDir)
                else:
                    log.info ("TESTING MODE is ON.  Directory was not actually removed")
                    removecomplete = True
                #TODO: Update the user_ack flag for this experiment
            except:
                log.error("Failed to remove directory %s" % expDir)
                
        if copycomplete and removecomplete and not JUST_TESTING:
            linkPath = path.join(backupDrive, expDir.strip().split("/")[-1])
            log.info("Linking %s to %s" % (linkPath, expDir))
            set_status(log,'Linking %s to %s' % (linkPath, expDir))
            os.umask(0002)
            try:
                os.symlink(linkPath, expDir)
            except:
                log.error(traceback.format_exc())
            add_to_db(exp, linkPath, True)
        elif removecomplete and not JUST_TESTING:
            add_to_db(exp, 'DELETED', False)
        else:
            log.info("copycomplete returned: %s" % copycomplete)
            log.info("removecomplete returned: %s" % removecomplete)
        # Record a return value that will trigger a FS check
        if removecomplete:
            updateNeeded = True
    return updateNeeded

def copy(log, expDir, backupDir, bwLimit):
    def to_bool(int):
        if int == 0:
            return True
        else:
            return False
    log.info("Copying %s to %s " % (expDir, backupDir)) 
    rawPath = path.join(expDir)
    backup = path.join(backupDir)
    bwLimitS = "--bwlimit=%s" % bwLimit
    if path.exists(backup):
        try:
            return to_bool(os.system("rsync -r %s %s %s" % (rawPath,backup,bwLimitS)))
        except Exception, err:
            log.error(sys.exc_info()[0])
            log.error(err)
            return False
    else:
        log.info("No path to Backup Drive")
        
def remove_dir(log,expDir):
    try:
        shutil.rmtree(expDir)
        if not path.exists(expDir): # double check the folder is gone
            return True
        else:
            return False
    except:
        exc = traceback.format_exc()
        log.error(exc)
        return False
    
def get_params(cur_loc,log):
    try:
        bk = models.BackupConfig.objects.get(location=cur_loc)
        return True,{'NUMBER_TO_BACKUP':bk.number_to_backup,
                     'TIME_OUT':bk.timeout,
                     'BACKUP_DRIVE_PATH':bk.backup_directory,
                     'BACKUP_THRESHOLD':bk.backup_threshold,
                     'LOCATION':bk.location,
                     'BANDWIDTH_LIMIT':bk.bandwidth_limit,
                     'GRACE_PERIOD':bk.grace_period,
                     'PK':bk.pk,
                     'EMAIL':bk.email,
                     'enabled':bk.online
                     }
    except:
        log.error(traceback.format_exc())
        return False, {'enabled':False}

def json_out(log, data, fileout):
    try:
        f = open(fileout,'w')
        f.write(json.dumps(data))
    except:
        log.error('failed to open file %s' % fileout)
    f.close()

def set_status(log,status):
    try:
        bk = models.BackupConfig.objects.all()[0]
        bk.status = status
        bk.save()
    except:
        log.error('No configured backups found')

def get_current_location(ip_address,log):
    try:
        cur_loc = models.Location.objects.all()[0]
    except OperationalError:
        log.error(traceback.format_exc())
    except:
        log.error('No locations configured, please configure at least one location')
        return
    return cur_loc

def loopScheduler(log):
    global gl_experiments 
    """Run the loop functions repeatedly."""
    LOOP        = 6     # Wait this many seconds before executing loop again
    timeBefore  = 0     # initialize to zero will force update at startup
    DELAY       = 300   # default if not in params
    
    def checkFSTimer(delay,timeBefore,timeNow):
        diff = timeNow - timeBefore
        #print "DEBUG:\nNow:%d\nBefore:%d\ndiff:%d" % (timeNow,timeBefore,diff)
        if timeBefore == 0:
            return True
        if diff >= delay:
            return True
        else:
            return False
        
    IP_ADDRESS = socket.gethostbyname(socket.gethostname())
    cur_loc = get_current_location(IP_ADDRESS,log)
    while True:
        connection.close()  # Close any db connection to force new one.
        try:
            isConf, params = get_params(cur_loc,log)
            if isConf:
                DELAY = params['TIME_OUT']  # how often to check filesystem
            #loop(log, cur_loc, isConf, params)
            # The FS check is run less frequently than loop to conserve system
            # resources.
            timeNow = time.mktime(time.localtime(time.time()))
            if checkFSTimer(DELAY,timeBefore,timeNow):
                log.debug("Its been %d seconds since the last FS check" % (timeNow - timeBefore))
                update_experiment_list(log, cur_loc, isConf, params)
                timeBefore = time.mktime(time.localtime(time.time()))
            else:
                pass
                # just update status of existing list
                #gl_experiments = refreshVals(gl_experiments)
            # The backup code is called frequently in order to be responsive to
            # user's acknowledge action on the webpage.
            updateNeeded = removeRuns(log, cur_loc, isConf, params)
            if updateNeeded:
                update_experiment_list(log, cur_loc, isConf, params)
                timeBefore = time.mktime(time.localtime(time.time()))
        except:
            log.error(traceback.format_exc())
        transaction.commit_unless_managed()
        #log.info ("Sleeping for %d seconds" % LOOP)
        time.sleep(LOOP)
        
def update_experiment_list(log, cur_loc, isConf, params):
    global gl_experiments
    pk = None
    gl_experiments = {} # clear the list of Runs to process
    try:
        IP_ADDRESS = socket.gethostbyname(socket.gethostname())
        log.debug("Backup server running on %s" % IP_ADDRESS)
        autoArchiveAck = GetAutoArchiveAck_GC()
        if autoArchiveAck:
            log.info("User acknowledgement to archive is not required")
        
        if isConf and params['enabled']:
            NUMBER_TO_BACKUP = params['NUMBER_TO_BACKUP']
            
            BACKUP_DRIVE_PATH = params['BACKUP_DRIVE_PATH']
            PERCENT_FULL_BEFORE_BACKUP = params['BACKUP_THRESHOLD']
            BACKUP_PK = params['PK']
            ADMIN_EMAIL = params['EMAIL']
            pk = BACKUP_PK
            GRACE_PERIOD = params['GRACE_PERIOD']
            backupFreeSpace, bkpPerFree, removeOnly = get_archive_report(log,params)
            serverFullSpace = get_server_full_space(cur_loc)
                
            if bkpPerFree:
                # when bkpPerFree is undefined, there is no backup drive mounted
                log.info("Backup Drive Free Space = %0.2f" % bkpPerFree)

            log.info ("Backup trigger percentage: %0.2f %% full" % PERCENT_FULL_BEFORE_BACKUP)
            
            for serverPath, percentFull in serverFullSpace:
                
                log.info("Results Drive %s Used Space = %0.2f" % (serverPath,percentFull))
                
                if percentFull > PERCENT_FULL_BEFORE_BACKUP:
                    experiments = build_exp_list(cur_loc, NUMBER_TO_BACKUP, GRACE_PERIOD, serverPath, removeOnly, log, autoArchiveAck)
                    gl_experiments[serverPath] = experiments
                
                    notify(log,experiments,ADMIN_EMAIL)
                else:
                    log.info("No experiments for processing because threshold not reached")
            
            if len(serverFullSpace) == 0:
                log.debug ("No fileservers configured")
                
        else:
            log.info("No backups configured, or they're disabled")
    except:
        log.error(traceback.format_exc())
        set_status(log,"ERROR")
    return

def removeRuns(log, cur_loc, isConf, params):
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
    pk = None
    email = Email()
    updateNeeded = False
    try:
        NUMBER_TO_BACKUP            = params['NUMBER_TO_BACKUP']
        PERCENT_FULL_BEFORE_BACKUP  = params['BACKUP_THRESHOLD']
        ADMIN_EMAIL                 = params['EMAIL']
        BACKUP_DRIVE_PATH           = params['BACKUP_DRIVE_PATH']
        BANDWIDTH_LIMIT             = params['BANDWIDTH_LIMIT']
        BACKUP_PK                   = params['PK']
        
        backupFreeSpace, bkpPerFree, removeOnly = get_archive_report(log,params)
        
        serverFullSpace = get_server_full_space(cur_loc)
        for serverPath, percentFull in serverFullSpace:
            
            if percentFull > PERCENT_FULL_BEFORE_BACKUP:
                log.info("%s : %.2f %% full" % (serverPath, percentFull))
                set_status(log, "Backing Up")
                if len(gl_experiments) == 0:
                    log.info("No datasets are valid to process")
                else:
                    updateNeeded = backup(log, gl_experiments[serverPath], BACKUP_DRIVE_PATH, 
                           NUMBER_TO_BACKUP, backupFreeSpace, 
                           BANDWIDTH_LIMIT, BACKUP_PK, email, ADMIN_EMAIL)
        #set_status(log,"Idle")
    except:
        log.error(traceback.format_exc())
        set_status(log,"ERROR")
    
    return updateNeeded

def GetAutoArchiveAck_GC():
    try:
        gc = models.GlobalConfig.objects.all()[0]
        # If auto_archive_ack is True, then autoArchiveAck is True
        autoAck = gc.auto_archive_ack
    except:
        return False
    return autoAck

#def loop(log, cur_loc, isConf, params):
#    global gl_experiments
#    pk = None
#    email = Email()
#    try:
#        IP_ADDRESS = socket.gethostbyname(socket.gethostname())
#        log.debug("Backup server running on %s" % IP_ADDRESS)
#        autoArchiveAck = GetAutoArchiveAck_GC()
#        if autoArchiveAck:
#            log.info("User acknowledgement to archive is not required")
#        
#        if isConf and params['enabled']:
#            NUMBER_TO_BACKUP = params['NUMBER_TO_BACKUP']
#            
#            BACKUP_DRIVE_PATH = params['BACKUP_DRIVE_PATH']
#            PERCENT_FULL_BEFORE_BACKUP = params['BACKUP_THRESHOLD']
#            BANDWIDTH_LIMIT = params['BANDWIDTH_LIMIT']
#            BACKUP_PK = params['PK']
#            ADMIN_EMAIL = params['EMAIL']
#            pk = BACKUP_PK
#            GRACE_PERIOD = params['GRACE_PERIOD']
#            backupFreeSpace, bkpPerFree, removeOnly = get_archive_report(log,params)
#            serverFullSpace = get_server_full_space(cur_loc)
#            log.info ("Backup trigger percentage: %0.2f %% full" % PERCENT_FULL_BEFORE_BACKUP)
#            for serverPath, percentFull in serverFullSpace:
#                
#                experiments = build_exp_list(cur_loc, NUMBER_TO_BACKUP, GRACE_PERIOD, serverPath, removeOnly, log, autoArchiveAck)
#                gl_experiments[serverPath] = experiments
#                
#                if percentFull > PERCENT_FULL_BEFORE_BACKUP:
#                    log.info("%s : %.2f %% full" % (serverPath, percentFull))
#                    set_status(log, "Backing Up")
#                    if len(experiments) == 0:
#                        log.info("No datasets are valid to process")
#                    notify(log,experiments,ADMIN_EMAIL)
#                    backup(log, experiments, BACKUP_DRIVE_PATH, 
#                           NUMBER_TO_BACKUP, backupFreeSpace, 
#                           BANDWIDTH_LIMIT, BACKUP_PK, email, ADMIN_EMAIL)
#                else:
#                    #percentFree = 100-percentFull
#                    #log.info("Results Drive %s Free Space = %0.2f" % (serverPath,percentFree))
#                    log.info("Results Drive %s Used Space = %0.2f" % (serverPath,percentFull))
#                    if bkpPerFree:
#                        # when bkpPerFree is undefined, there is no backup drive mounted
#                        log.info("Backup Drive Free Space = %0.2f" % bkpPerFree)
#            
#            if len(serverFullSpace) == 0:
#                log.debug ("No fileservers configured")
#                
#        else:
#            log.info("No backups configured, or they're disabled")
#        db.reset_queries()
#        if pk != None:
#            set_status(log,"Idle")
#
#    except KeyboardInterrupt:
#        # when do we ever execute this code?
#        set_status(log,"Off")
#    except:
#        log.error(traceback.format_exc())
#        set_status(log,"ERROR")
        
def _cleanup(signum, frame):
    bk = models.BackupConfig.objects.all()[0]
    bk.status = 'Off'
    bk.save()
    sys.exit(0)

def get_archive_report(log,params):
    backupFreeSpace = None
    bkpPerFree = None
    dev = devices.disk_report()
    removeOnly = False
    if params['BACKUP_DRIVE_PATH'] != 'None':
        for d in dev:
            if d.get_path() == params['BACKUP_DRIVE_PATH']:
                bkpPerFree = d.get_free_space()
                backupFreeSpace = int(d.get_available()*1024)
                if backupFreeSpace < last_exp_size:
                    removeOnly=True
                    log.info('Archive drive is full, entering Remove Only Mode.')
                return backupFreeSpace, bkpPerFree, removeOnly
    else:
        removeOnly = True
        log.info('No archive drive, entering Remove Only Mode.')
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
    def __init__(self,log):
        xmlrpc.XMLRPC.__init__(self)
        self.log = log
        
    def xmlrpc_next_to_archive(self):
        global gl_experiments
        '''Returns the meta information of all the file servers currently in the database'''
        IP_ADDRESS = socket.gethostbyname(socket.gethostname())
        cur_loc = get_current_location(IP_ADDRESS,self.log)
        isConf, params = get_params(cur_loc,self.log)
        retdict = {}
        if isConf:
            # Need to update the field values of the experiments in the list:
            # in case the storage option has been changed
            gl_experiments = refreshVals(gl_experiments)
            retdict = gl_experiments
        return retdict
    
    def xmlrpc_user_ack(self):
        '''Called when archiving acknowledged state is changed for each experiment'''
        # We use this function to wake up the daemon and start the backup process
        # Typically, we expect about 10 calls to this function at a time
        global gl_experiments
        self.log.debug("Got an acknowledge xmlrpc")
        gl_experiments = refreshVals(gl_experiments)
        return True
    
def checkThread (thread,log,reactor):
    '''Checks thread for aliveness.
    If a valid reactor object is passed, the reactor will be stopped
    thus stopping the daemon entirely.'''
    thread.join(1)
    if not thread.is_alive():
        log.critical("loop thread is dead")
        if reactor.running:
            log.critical("ionArchive daemon is exiting")
            reactor.stop()

def main(argv):
    # Setup log file logging
    try:
        log='/var/log/ion/iarchive.log'
        infile = open(log, 'a')
        infile.close()
    except:
    	print traceback.format_exc()
    	log = '/var/log/iarchive.log'
    archlog = logging.getLogger('archlog')
    archlog.propagate = False
    #archlog.setLevel(logging.DEBUG)
    archlog.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(log, maxBytes=1024*1024*10, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    archlog.addHandler(handler)

    #handle keyboard interrupt; others
    import signal
    signal.signal(signal.SIGTERM,_cleanup)
    
    #set status in database Backup model
    set_status(archlog,'Service Started')
    
    archlog.info('ionArchive Started Ver: %s' % __version__)
    
    # start loopScheduler which runs 'loop' periodically in a thread
    loopfunc = lambda: loopScheduler(archlog)
    lthread = threading.Thread(target=loopfunc)
    lthread.setDaemon(True)
    lthread.start()
    
    # check thread health periodically and exit if thread is dead
    # pass in reactor object below to kill process when thread dies
    l = task.LoopingCall(checkThread,lthread,archlog,reactor)
    l.start(30.0) # call every 30 second
    
    # start the xml-rpc server
    r = Status(archlog)
    reactor.listenTCP(settings.IARCHIVE_PORT,server.Site(r))
    reactor.run()

if __name__=="__main__":
    sys.exit(main(sys.argv))

        
