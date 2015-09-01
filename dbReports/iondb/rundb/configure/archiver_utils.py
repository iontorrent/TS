# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import traceback
import logging
from iondb.rundb.models import Experiment, FileServer, Results
from iondb.utils.files import disk_attributes
from iondb.rundb.data.dmfilestat_utils import get_keepers_diskspace

logger = logging.getLogger(__name__)


def get_servers():
    fileservers = FileServer.objects.all()
    ret = []
    for fs in fileservers:
        if os.path.exists(fs.filesPrefix):
            ret.append(fs)
    return ret


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

        # get space used by data marked Keep
        keeper_used = get_keepers_diskspace(server.filesPrefix)
        keeper_used = float(sum(keeper_used.values()))/1024     #gbytes
        percentkeep = 100*(keeper_used/total_gb if total_gb and total_gb > 0 else 0)

        stats[server.filesPrefix] = {
            'statusmsg':errormsg,
            'percentfull': percentfull,
            'disksize': total_gb,
            'diskfree': avail_gb,
            'keeper_used': keeper_used,
            'percentkeep': percentkeep,
        }

    return stats
