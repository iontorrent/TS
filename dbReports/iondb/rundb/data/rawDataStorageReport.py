#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
'''
Prints report containing file system usage data and dataset archiving
disposition
'''
import os
import sys

import iondb.bin.djangoinit
from iondb.rundb import models
import iondb.settings


def disk_space(mypath):
    '''
    Returns used and available disk capacity in gigabytes
    '''
    import commands
    total = commands.getstatusoutput("df -k %s 2>/dev/null" % mypath)
    used = int(total[1].split()[9])
    available = int(total[1].split()[10])

    # DEBUG PRINTOUT
    # kbytes = used + available
    # print "Disk Capacity %s" % path
    # print "---------"
    # print str(kbytes) + " KBytes"
    # print str(int(kbytes) / 1024 / 1024) + " GB"

    # Return gigabytes instead
    used = used / 1024 / 1024
    available = available / 1024 / 1024

    return used, available


def remote_server(mypath):
    '''Returns the name of the server exporting the shared directory'''
    import commands
    # strip trailing delimiter
    if mypath[-1:] == "/":
        mypath = mypath[0:-1]

    ret = commands.getstatusoutput("mount -l | grep \" %s \"" % mypath)
    if len(ret[1]) != 0:
        mountpoint = ret[1].split(':')[0]
    else:
        mountpoint = mypath
    return mountpoint


def getpercents(tds, uds, fds):
    '''
    Calculate a nice percentage that equals 100%
    '''
    # Used disk space
    percentuds = 100 * (float(uds) / float(tds))
    # Free disk space
    percentfds = 100 * (float(fds) / float(tds))

    return percentuds, percentfds


def category_stats():
    '''
    Statistics by Fileset categories
    '''
    from iondb.rundb.data import dmactions_types

    report = []
    report.append("File Category Storage Statistics\n")
    report.append("")
    report.append("%-30s : %9s%9s%9s%9s%9s\n" %
                  ("File Category Group", 'Total', 'Local', 'Archived', 'Deleted', 'Error'))

    for settype in [dmactions_types.SIG, dmactions_types.BASE, dmactions_types.OUT, dmactions_types.INTR]:
        dmfilestats = models.DMFileStat.objects.filter(dmfileset__type=settype)
        list_L = dmfilestats.filter(action_state='L').count()
        list_S = dmfilestats.filter(action_state='S').count()
        list_N = dmfilestats.filter(action_state='N').count()
        list_AG = dmfilestats.filter(action_state='AG').count()
        list_DG = dmfilestats.filter(action_state='DG').count()
        # list_EG = dmfilestats.filter(action_state='EG').count()
        list_AD = dmfilestats.filter(action_state='AD').count()
        list_DD = dmfilestats.filter(action_state='DD').count()
        list_E = dmfilestats.filter(action_state='E').count()
        report.append("%-30s : %9d%9d%9d%9d%9d\n" % (settype,
                                                     dmfilestats.count(),
                                                     list_L + list_S + list_N,
                                                     list_AD + list_AG,
                                                     list_DD + list_DG,
                                                     list_E))

    return report


def file_server_storage_report():
    '''
    Reports storage allocation only using the df system command.
    Anything needing du is not returned because du takes forever on large
    filesystems.  Large being the typical NAS of ~19TB.
    '''
    #--------------------------------------------------------------------------
    #   Filesystem tools
    #--------------------------------------------------------------------------
    # Get disk usage for
    # Unused disk space on /results partition                           : Unused
    # Used space in /results/analysis/output                            : Reports space
    # Used space in /results/<pgm directories>                          : Raw Data space
    # Used space in NOT previous locations, "other" disk space usage    : Other space

    # get all fileservers from dbase
    fileservers = models.FileServer.objects.all()
    validfileservers = []
    for fileserver in fileservers:
        if os.path.isdir(fileserver.filesPrefix):
            validfileservers.append(fileserver)

    # get list of sequencer names/directories from dbase
    # List is filtered for valid directories
    pgm_list = []
    pgms = models.Rig.objects.all()
    for pgm in pgms:
        for fileserver in validfileservers:
            testpath = os.path.join(fileserver.filesPrefix, pgm.name)
            if os.path.isdir(testpath):
                pgm_list.append(testpath)
    report = []
    for fileserver in validfileservers:
        uds, fds = disk_space(fileserver.filesPrefix)
        tds = uds + fds
        percentuds, percentfds = getpercents(tds, uds, fds)
        remotesrv = remote_server(fileserver.filesPrefix)

        # Print Report
        report.append("\n")
        report.append("Disk Space Allocation Report: %s (%s)\n" % (fileserver.filesPrefix, remotesrv))
        report.append("")
        report.append("Total Disk Space         :      %6d GBytes\n" % tds)
        report.append("Used Disk Space          :      %6d GBytes %.1f%%\n" % (uds, percentuds))
        report.append("Free Disk Space          :      %6d GBytes %.1f%%\n" % (fds, percentfds))
        report.append("\n")
    return report


def isMounted(server):
    import commands
    server = server.split(".")[0]
    ret = commands.getstatusoutput("mount -l | grep \"^%s\"" % server)
    if len(ret[1]) != 0:
        return True
    else:
        return False


def pgm_ftpserver_report():
    '''Displays each sequencer in database and the file server it writes to'''
    # Get all sequencers
    pgms = models.Rig.objects.all()

    # Make a dictionary of ftpservers and an array of sequencer names associated with each
    serverDict = {}
    for pgm in pgms:
        ftpservername = pgm.ftpserver
        if ftpservername in serverDict:
            serverDict[ftpservername].append(pgm.name)
        else:
            serverDict[ftpservername] = [pgm.name]

    report = []
    report.append("\nFile servers and sequencers writing to them:\n")
    for server in sorted(serverDict):
        report.append("\n%s: %s\n" % (server, "" if isMounted(server) else "(not mounted)"))
        for pgm in serverDict[server]:
            report.append("\t%s\n" % pgm)
    return report


def dm_default_settings():
    auto_action_set = models.DMFileSet.AUTO_ACTION
    dmfilesets = models.DMFileSet.objects.filter(version=iondb.settings.RELVERSION)
    report = []
    report.append("\n")
    report.append("File Category Default Auto-Action Settings\n")
    report.append("%-8s - %-28s %11s %10s %10s %s\n" %
                  ("State", "File Category", "Age Thresh.", "Disk Thresh.", "Action", "Location"))
    for dmfileset in dmfilesets:
        for action in auto_action_set:
            if dmfileset.auto_action in action:
                this_action = action[1]
        report.append(
            "%-8s - %-30s %4d days %4.0f%% %17s %s\n" % ("Enabled" if dmfileset.enabled else "Disabled",
                                                         dmfileset.type,
                                                         dmfileset.auto_trigger_age,
                                                         dmfileset.auto_trigger_usage,
                                                         this_action,
                                                         dmfileset.backup_directory))
    return report


def storage_report():
    report_text = category_stats()
    report_text.extend(dm_default_settings())
    report_text.extend(file_server_storage_report())
    report_text.extend(pgm_ftpserver_report())
    return report_text


if __name__ == '__main__':
    for line in storage_report():
        sys.stdout.write(line)
