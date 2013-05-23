# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import xmlrpclib
import traceback
import socket
import logging
import math
from iondb.rundb.models import Experiment, Backup, FileServer, Results, ReportStorage
from django.conf import settings
from iondb.rundb.data.archiveExp import Experiment as ArchiveExperiment
from iondb.utils.files import disk_attributes

logger = logging.getLogger(__name__)


def get_servers():
    fileservers = FileServer.objects.all()
    ret = []
    for fs in fileservers:
        if os.path.exists(fs.filesPrefix):
            ret.append(fs)
    return ret

# --- old services Data Management functions TODO: cleanup
def exp_list(backup_object):
    # make dictionary, one array per file server of archiveExperiment objects
    def rawdata_used(experiments, serverprefix):
        '''
        Sums diskusage field for given experiment objects
        diskusage stores MB
        '''
        tot_disk_used = 0
        experiments = experiments.filter(expDir__startswith=serverprefix).values_list('diskusage', flat=True)
        for diskusage in experiments:
            tot_disk_used += diskusage if diskusage is not None else 0
        return tot_disk_used

    def reportstorage_used(results, serverprefix):
        '''
        Sums diskusage field for given results objects
        diskusage stores MB
        '''
        tot_disk_used = 0
        reportstorage = ReportStorage.objects.filter(dirPath__startswith=serverprefix)

        results = results.filter(reportstorage=reportstorage).values_list('diskusage', flat=True)
        for result in results:
            tot_disk_used += result if result is not None else 0
        return tot_disk_used

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
    results = Results.objects.all().order_by('pk')

    storage_stats = [0, 0, 0]   # [ Keep, Archive, Delete]

    servers = get_servers()
    for server in servers:
        # Statistics for this server
        #TODO: store these in database??
        total, availSpace, freeSpace, bsize = disk_attributes(server.filesPrefix)    # bytes
        total_gb = float(total*bsize)/(1024*1024*1024)
        avail_gb = float(availSpace*bsize)/(1024*1024*1024)
        free_gb = float(freeSpace*bsize)/(1024*1024*1024)
        rawused = float(rawdata_used(experiments, server.filesPrefix))/1024 #gbytes
        reportsused = float(reportstorage_used(results, server.filesPrefix))/1024   #gbytes
        reserved = free_gb - avail_gb
        other = total_gb - (reserved + avail_gb + rawused + reportsused)    #gbytes
        server_stats[server.filesPrefix] = {
            'percentfull': server.percentfull,
            'disksize': total_gb,
            'diskfree': avail_gb,
            'rawused': rawused,
            'reportsused': reportsused,
            'other': other,
        }

        # only generate list of experiments if Archiving is enabled
        if backup_object.online:
            # Experiments for this server
            explist = []
            experiments_fs = experiments.filter(expDir__startswith=server.filesPrefix)
            for exp in experiments_fs:
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
                if 'KI' in exp.storage_options:
                    storage_stats[0] += 1
                elif 'A' in exp.storage_options:
                    storage_stats[1] += 1
                elif 'D' in exp.storage_options:
                    storage_stats[2] += 1

            to_archive[server.filesPrefix] = explist

    return to_archive, server_stats, storage_stats

def disk_usage_stats():
    stats = {}

    def diskspace_used(objects):
        '''
        Sums diskusage field for given Experiment or Result objects
        diskusage stores MB
        '''
        tot_disk_used = 0
        diskusage_list = objects.values_list('diskusage', flat=True)
        for diskusage in diskusage_list:
            tot_disk_used += diskusage if diskusage is not None else 0
        return tot_disk_used

    servers = get_servers()

    for server in servers:
        # Statistics for this location
        errormsg = None
        try:
            total, availSpace, freeSpace, bsize = disk_attributes(server.filesPrefix)    # bytes
            total_gb = float(total*bsize)/(1024*1024*1024)
            avail_gb = float(availSpace*bsize)/(1024*1024*1024)
            free_gb = float(freeSpace*bsize)/(1024*1024*1024)
            percentfull = 100-(float(availSpace)/float(total)*100) if total > 0 else 0
            reserved = free_gb - avail_gb
        except:
            errormsg = "Error accessing %s filesystem." % server.filesPrefix
            percentfull = '0'
            total_gb = ''
            avail_gb = ''
            free_gb = ''
            reserved = ''
            other = ''
            logger.error(traceback.format_exc())

        # for experiment data:
        experiments = Experiment.objects.filter(expDir__startswith=server.filesPrefix)
        rawused = float(diskspace_used(experiments))/1024 #gbytes
        # for results data:
        results = Results.objects.filter(reportstorage__dirPath__startswith=server.filesPrefix)
        reportsused = float(diskspace_used(results))/1024   #gbytes

        other = total_gb - (reserved + avail_gb + rawused + reportsused)    #gbytes



        stats[server.filesPrefix] = {
            'statusmsg':errormsg,
            'percentfull': percentfull,
            'disksize': total_gb,
            'diskfree': avail_gb,
            'rawused': rawused,
            'reportsused': reportsused,
            'other': other,
        }

    return stats
