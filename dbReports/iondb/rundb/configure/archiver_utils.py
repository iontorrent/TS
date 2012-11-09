# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import xmlrpclib
import traceback
import socket
import logging
from iondb.rundb import models
from django.conf import settings
from iondb.backup.archiveExp import Experiment

logger = logging.getLogger(__name__)

def get_servers():
    fileservers = models.FileServer.objects.all()
    ret = []
    for fs in fileservers:
        if os.path.exists(fs.filesPrefix):
            ret.append(fs)
    return ret
    
def explist(bk):
    
    # make dictionary, one array per file server of archiveExperiment objects
    to_archive = {}
    fs_stats = {}
    
    try:
        # populate dictionary with experiments ready to be archived
        # Get all experiments in database sorted by date
        experiments = models.Experiment.objects.all().order_by('date')
        # Ignore experiments already archived/deleted
        experiments = experiments.exclude(expName__in = models.Backup.objects.all().values('backupName'))
    except:
        logger.error(traceback.print_exc())
        raise

    servers = get_servers()
    for fs in servers:
        # Statistics for this server
        fs_stats[fs.filesPrefix] = fs.percentfull
        
        # only generate list of experiments if Archiving is enabled
        if bk.online:
            # Experiments for this server
            explist = []
            for exp in experiments:
                if fs.filesPrefix in exp.expDir:
                    E = Experiment(exp,
                                   str(exp.expName),
                                   str(exp.date),
                                   str(exp.star),
                                   str(exp.storage_options),
                                   str(exp.user_ack),
                                   str(exp.expDir),
                                   exp.pk,
                                   str(exp.rawdatastyle))
                    explist.append(E)
                # limit number in list to configured limit
                #if len(explist) >= bk[0].number_to_backup:
                #    break
            to_archive[fs.filesPrefix] = explist
        
    return to_archive, fs_stats
    
def areyourunning():
    uptime = 0.0
    try:
        astat = xmlrpclib.ServerProxy("http://127.0.0.1:%d" % settings.IARCHIVE_PORT)
        logger.debug("Sending xmplrpc status_check")
        uptime = astat.status_check()
        daemon_status = True
    except (socket.error, xmlrpclib.Fault):
        logger.warn (traceback.format_exc())
        daemon_status = False
    except:
        logger.exception(traceback.format_exc())
        daemon_status = False
    logger.debug("Status of ionArchiver %s" % str(daemon_status))
    return daemon_status
