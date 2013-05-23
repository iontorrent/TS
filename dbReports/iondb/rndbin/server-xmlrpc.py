# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
#
#   This code is largely copied from backup.py (Feb 17, 2011) but isolates the xmlrpc server
#   code.  Its used to test performance of that server with nothing else interfering,
#   like the logging code or the threaded loop code.
#
from twisted.web import xmlrpc, server
import socket
import sys
import os
from os import path

import iondb.bin.djangoinit
from iondb.rundb import models
from iondb.utils import devices
from django import db

last_exp_size = 0

def get_current_location(ip_address):
    try:
        cur_loc = models.Location.objects.all()[0]
    except:
        #log.error('No locations configured, please configure at least one location')
        return
    return cur_loc

def get_params(cur_loc):
    try:
        bk = models.BackupConfig.objects.get(location=cur_loc)
        return True,{'NUMBER_TO_BACKUP':bk.number_to_backup,
                     'TIME_OUT':bk.timeout,
                     'BACKUP_DRIVE_PATH':bk.backup_directory,
                     'BACKUP_THRESHOLD':bk.backup_threshold,
                     'LOCATION':bk.location,
                     'BANDWIDTH_LIMIT':bk.bandwidth_limit,
                     'PK':bk.pk,
                     'EMAIL':bk.email,
                     'enabled':bk.online
                     }
    except:
        return False, {'enabled':False}

def get_archive_report(params):
    backupFreeSpace = None
    bkpPerFree = ''
    dev = devices.disk_report()
    removeOnly = False
    if params['BACKUP_DRIVE_PATH'] != 'None':
        for d in dev:
            if d.get_path() == params['BACKUP_DRIVE_PATH']:
                bkpPerFree = d.get_free_space()
                backupFreeSpace = int(d.get_available()*1024)
                if backupFreeSpace < last_exp_size:
                    removeOnly=True
                    #log.info('Archive drive is full, entering Remove Only Mode.')
                return backupFreeSpace, bkpPerFree, removeOnly
    else:
        removeOnly = True
        #log.info('No archive drive, entering Remove Only Mode.')
    return backupFreeSpace, bkpPerFree, removeOnly

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

def build_exp_list(cur_loc, num, serverPath, removeOnly): #removeOnly needs to be implemented
    '''My comments are here'''
    exp = models.Experiment.objects.all().order_by('date')
    if removeOnly:
        exp = exp.filter(storage_options='D').exclude(expName__in = models.Backup.objects.all().values('backupName'))
    else:
        exp = exp.exclude(storage_options='KI').exclude(expName__in = models.Backup.objects.all().values('backupName'))

    experiments = []
    for e in exp:
        if len(experiments) < num: # only want to loop until we have the correct number
            location = server_and_location(e)
            if location == None:
                continue
            experiment = Experiment(e,
                                    str(e.expName),
                                    str(e.date),
                                    str(e.star),
                                    str(e.storage_options),
                                    str(e.expDir),
                                    location,
                                    e.pk)

            try:
                # don't add anything to the list if its already archived
                bk = models.Backup.objects.get(backupName=experiment.get_exp_name())
                continue
            except:
                # check that the path exists, and double check that its not marked to 'Keep'
                if not path.islink(experiment.get_exp_path()) and \
                        path.exists(experiment.get_exp_path()) and \
                        experiment.location==cur_loc and \
                        experiment.get_storage_option() != 'KI' and \
                        path.samefile(experiment.get_prefix(),serverPath):
                    experiments.append(experiment)
        else:
            return experiments
    return experiments

def server_and_location(experiment):
    try:
        loc = models.Rig.objects.get(name=experiment.pgmName).location
    except:
        return None
    #server = models.FileServer.objects.filter(location=loc)
    return loc

class Experiment:
    def __init__(self, exp, name, date, star, storage_options, dir, location, pk):
        self.prettyname = exp.pretty_print()
        self.name = name
        self.date = date
        self.star = star
        self.store_opt = storage_options
        self.dir = dir
        self.location = location
        self.pk = pk

    def get_exp_path(self):
        return self.dir

    def is_starred(self):
        def to_bool(value):
            if value == 'True':
                return True
            else:
                return False
        return to_bool(self.star)

    def get_location(self):
        return self.location

    def get_folder_size(self):
        dir = self.dir
        dir_size = 0
        for (path, dirs, files) in os.walk(self.dir):
            for file in files:
                filename = os.path.join(path, file)
                dir_size += os.path.getsize(filename)
        return dir_size

    def get_exp_name(self):
        return self.name

    def get_pk(self):
        return self.pk

    def get_storage_option(self):
        return self.store_opt

    def get_prefix(self):
        '''This returns the first directory in the path'''
        temp = self.dir.split('/')[1]
        return '/'+temp


class Status(xmlrpc.XMLRPC):
    """Allow remote access to this server"""

    def xmlrpc_next_to_archive(self):
        '''Returns the meta information of all the file servers currently in the database'''
        #IP_ADDRESS = socket.gethostbyname(socket.gethostname())
        cur_loc = get_current_location('')
        isConf, params = get_params(cur_loc)
        retdict = {}
        if isConf:
            backupFreeSpace, bkpPerFree, removeOnly = get_archive_report(params)
            serverFullSpace = get_server_full_space(cur_loc)
            for serverpath, space in serverFullSpace:
                retdict[serverpath] = build_exp_list(cur_loc, params['NUMBER_TO_BACKUP'], serverpath, removeOnly)

        db.reset_queries()
        return retdict

class Example(xmlrpc.XMLRPC):
    """An example object to be published."""

    def xmlrpc_next_to_archive(self):
        '''Ion's function called from Service Tab'''
        return "FOOBAR"

    def xmlrpc_echo(self, x):
        """
        Return all passed args.
        """
        return x

    def xmlrpc_add(self, a, b):
        """
        Return sum of arguments.
        """
        return a + b

    def xmlrpc_fault(self):
        """
        Raise a Fault indicating that the procedure should not be used.
        """
        raise xmlrpc.Fault(123, "The fault procedure is faulty.")

if __name__ == '__main__':
    from twisted.internet import reactor
    #r = Example()
    r = Status()
    reactor.listenTCP(10012, server.Site(r))
    reactor.run()
