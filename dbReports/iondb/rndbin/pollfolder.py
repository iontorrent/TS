# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os, time, sys, subprocess, traceback
sys.path.append(os.path.split(os.getcwd())[0])
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from rundb.models import *
#from rundb.runtoanalysis import *
from django.db import models
import datetime

def getLog(dir):
    fname = dir + '/' + 'explog.txt'
    if (os.path.isfile(fname)):
        infile = open(fname)
        ret = infile.read()
        infile.close()
    else:
        ret = None
    return ret

def parseLog(logText):
    return dict([(k.strip(),v.strip()) for k,v in \
            filter(lambda ele: len(ele) == 2,
                map(lambda line: line.split(':'),
                    filter(lambda line: len(line) > 1,
                        logText.split('\n'))))])
   
def handleLog(fields):
    bad = False
    try:
        #m = Motherboard(serial=fields['Board serial'], version=fields['Board version'])
        #dc = DataCollect(version=fields['Datacollect version'])
        #lv = LiveView(version=fields['LiveView version'])
        #fp = FPGA(version=fields['FPGA version'])
        isFresh = lambda ele: not len(ele._default_manager.filter(pk=ele.pk))
        map(lambda ele: isFresh(ele) and ele.save(), [m,dc,lv,fp])
    except:
        bad = True
    return bad    
    
def extractNumFiles(log):
    try:
        ret = int(dict([(s.strip().split(':')[0], s.strip().split(':')[1]) \
                        for s in str(log).split('\n') if len(s) > 2])['Number of files'])
    except: ret = 0
    return ret
    
def pollFolders():
    makeName = lambda path: '/'.join(path.split('/')[-2:])
    servers = FileServer.objects.all()
    numSaved = 0
    for server in servers:
        print server
        try:
            location = server.location
            rigs = Rig.objects.filter(location=location)
            #basedir = '//' + server.name + server.filesPrefix
            basedir = server.filesPrefix
            #print 'BASEDIR:', basedir
            if not os.path.isdir(basedir):
                #basedir = '//' + server.ip + server.filesPrefix
                basedir = server.filesPrefix
                if not os.path.isdir(basedir):
                    print "bad directory, skipping:", basedir
                    continue
            for rig in rigs:
                #if str(rig.name).count('Spengler'):
                #    continue            
                print "Checking %s on %s..." % (rig.name, basedir)
                try:
                    rigdir = basedir + rig.name + '/'
                    if (os.path.isdir(rigdir)):
                        #fldrs =  os.listdir(rigdir)
                        fldrs = filter(lambda d: os.path.isdir(rigdir + d) and not d.count('_DELETED'), os.listdir(rigdir))
                        for fldr in fldrs:
                            if fldr[0] == '.': continue
                            #elif rig.name.lower().count('spengler'):
                            #    print "SPENGLER FOLDER:", fldr
                            fpath = rigdir + fldr
                            #print 'FPATH:', fpath
                            numFiles = len(filter(lambda fn: fn.count('.dat'), os.listdir(fpath)))
                            dirStat = os.stat(fpath)
                            t = datetime.datetime.fromtimestamp(dirStat.st_ctime)
                            log = getLog(fpath)
                            if log:
                               # convert the log file into a nice array/dictionary object
                                parsedLog = parseLog(log)

                                # update a few global tables in the database with possible new info from this log file
                                #handleLog(parsedLog)

                                # extract important explog fields, for use later in populating our new Folder table entry
                                project = parsedLog['Project']
                                sample = parsedLog['Sample']
                                library = parsedLog['Library']
                                chipBarcode = parsedLog['ChipBarCode']
                                seqBarCode = parsedLog.get('SeqBarCode',"")
                                cycles = parsedLog['Cycles']
                                chipType = parsedLog['ChipType']
                                try:
                                    notes = parsedLog['User Notes']
                                except:
                                    notes = ''
                                

                                # here, we are just adding the number of files to the log text, may want to use this later
                                log += ('\nNumber of files: %d' % numFiles)
                            else:
                                # just set a bunch of defaults
                                log = ''
                                project = ''
                                sample = ''
                                library = ''
                                notes = ''

                            # this name is really the PGM's folder path (not full path) with a leading slash
                            #it is now....  JB
                            name = makeName(fpath).split("/")[-1]
                            #print name
                            runName = parsedLog['Experiment Name'].split('_')[-1]
                            #print runName
                            storage = 'BAD' if numFiles < 10 and str(name).lower().count('pre') > 0 else 'RAND'
                            f = Experiment(expDir=fpath, pgmName=rig.name, date=t, expName=runName, log=str(log), 
                                           storage_options=storage, project=project, sample=sample, 
                                           library=library, notes=notes, chipBarcode=chipBarcode, 
                                           seqKitBarcode=seqBarCode, chipType=chipType, cycles=cycles, unique=str(fpath))
                            prevSaved = Experiment.objects.filter(unique=f.unique)
                            checkForLog = lambda: len(prevSaved) == 1 \
                                and ((str(prevSaved[0].log).count(':') <= 0 \
                                and str(f.log).count(':') > 0) \
                                or extractNumFiles(prevSaved[0].log) != extractNumFiles(log))
                            if not len(prevSaved):# or checkForLog()):
                                f.save()
                                numSaved += 1
                except:
                    print "Error getting data from rig %s." % rig
                    traceback.print_exc()
        except:
            print 'An error occurred while attempting to poll server [', server.name, ']'
            traceback.print_exc()
    return numSaved

if __name__ == '__main__':
    import subprocess
    if len(sys.argv) != 2:
        print 'Usage: --once | --many'
        exit(0)
    if sys.argv[1] == '--once':
        print pollFolders()
        exit(0)
    if sys.argv[1] == '--many':
        while True:
            ecode = subprocess.call('python pollfolder.py --once', shell=True)
            time.sleep(60)
