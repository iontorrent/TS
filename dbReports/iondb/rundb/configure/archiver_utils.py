# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import xmlrpclib
import traceback
import socket
import logging
from iondb.rundb.models import Experiment, Backup, FileServer
from django.conf import settings
from iondb.backup.archiveExp import Experiment as ArchiveExperiment

logger = logging.getLogger(__name__)


def get_servers():
    fileservers = FileServer.objects.all()
    ret = []
    for fs in fileservers:
        if os.path.exists(fs.filesPrefix):
            ret.append(fs)
    return ret


def exp_list(backup_object):
    # make dictionary, one array per file server of archiveExperiment objects
    to_archive = {}
    server_stats = {}

    try:
        # populate dictionary with experiments ready to be archived
        # Get all experiments in database sorted by date
        experiments = Experiment.objects.all().order_by('date')
        # Ignore experiments already archived/deleted
        experiments = experiments.exclude(expName__in=Backup.objects.all().values('backupName'))
    except:
        logger.error(traceback.print_exc())
        raise

    servers = get_servers()
    for server in servers:
        # Statistics for this server
        server_stats[server.filesPrefix] = server.percentfull

        # only generate list of experiments if Archiving is enabled
        if backup_object.online:
            # Experiments for this server
            explist = []
            for exp in experiments:
                if exp.expDir.startswith(server.filesPrefix):
                    archive_experiment = ArchiveExperiment(exp,
                                                           str(exp.expName),
                                                           exp.date.strftime("%Y-%m-%d"),
                                                           str(exp.star),
                                                           str(exp.storage_options),
                                                           str(exp.user_ack),
                                                           str(exp.expDir),
                                                           exp.pk,
                                                           str(exp.rawdatastyle),
                                                           str(exp.diskusage) if exp.diskusage is not None else "Unknown",
                                                           )
                    explist.append(archive_experiment)
                # limit number in list to configured limit
                #if len(explist) >= backup_object[0].number_to_backup:
                #    break
            to_archive[server.filesPrefix] = explist

    return to_archive, server_stats


def areyourunning():
    try:
        astat = xmlrpclib.ServerProxy("http://127.0.0.1:%d" % settings.IARCHIVE_PORT)
        logger.debug("Sending xmplrpc status_check")
        astat.status_check()
        daemon_status = True
    except (socket.error, xmlrpclib.Fault):
        logger.warn(traceback.format_exc())
        daemon_status = False
    except:
        logger.exception(traceback.format_exc())
        daemon_status = False
    logger.debug("Status of ionArchiver %s" % str(daemon_status))
    return daemon_status
