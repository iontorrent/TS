#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys
import json
import traceback
import subprocess
import logging
import logging.handlers


class PickleLogger(object):
    """Gentle reader,
    In Celery, we're passing around a TSconfig object on which we call methods
    in order to perform the different stages of an upgrade, and because of this,
    that object is getting pickled and because of that picking the logger's
    file object needs to be delt with, and rather than manually removing it
    every time we pickle a TSconfig object and reinstantiating it every time it
    is unpickled, I wrote this ... Proxy for the logger which implements some
    of pickles serializer methods.
    Sincerely,
    Brian Kennedy
     """

    def __init__(self):
        self._logger = self.setup_logging()

    def __getstate__(self):
        obj_dict = self.__dict__.copy()
        obj_dict['_logger'] = None
        return obj_dict

    def __setstate__(self, obj_dict):
        self.__dict__.update(obj_dict)
        self._logger = self.setup_logging(False)

    def __getattr__(self, item):
        return getattr(self._logger, item)

    def __setattr__(self, key, value):
        if key == "_logger":
            self.__dict__["_logger"] = value
        else:
            setattr(self._logger, key, value)

    def setup_logging(self, rollover=True):
        """Configure the logger and return it.  This is a function because we want
        to use the logger in celery tasks as well
        """
        logger = logging.getLogger("tsconfig")
        logfile = "/var/log/ion/tsconfig_gui.log"
        if not logger.handlers:
            logger.propagate = False
            logger.setLevel(logging.DEBUG)
            hand = logging.handlers.RotatingFileHandler(logfile, backupCount=5)
            logger.addHandler(hand)
            format = logging.Formatter("%(asctime)s\t%(levelname)s\t%(funcName)s\t%(message)s")
            hand.setFormatter(format)
            # we handle the rollover manually in order to ensure that the records of a
            # single upgrade reside in a single log file.
        if rollover and os.stat(logfile).st_size > 1024 * 1024:
            hand.doRollover()
        return logger

logger = PickleLogger()

################################################################################
#
# States
#
################################################################################
#1:"Polling Failed"
#2:"No Updates Available"
#3:"Updates Available"
#4:"Package File Download Incomplete"
#5:"Package File Download Complete"
#6:"Package File Download Failed"
#7:"Package Install Incomplete"
#8:"Package Install Failed"
#9:"Package Install Complete" # Should be same as No Updates Available


################################################################################
#
# Utility functions
#
################################################################################
def host_is_master():
    '''Returns true if current host is configured as master node'''

    CONF_FILE="/etc/torrentserver/tsconf.conf"

    if os.path.isfile(CONF_FILE):
        try:
            for line in open(CONF_FILE):
                if "mode:master" in line:
                    logger.debug("Found mode:master in %s" % CONF_FILE)
                    return True
                elif "mode:compute" in line:
                    logger.debug("Found mode:compute in %s" % CONF_FILE)
                    return False
                else:
                    pass
        except IOError as err:
            logger.error(err.message)
            
    if os.path.isfile("/opt/ion/.masternode") and not os.path.isfile("/opt/ion/.computenode"):
        logger.debug("Using flag files to determine masterhost status")
        return True
        
    raise OSError("Host not configured as either master or compute node")
    


# This tells apt-get not to expect access to standard in.
os.environ['DEBIAN_FRONTEND'] = 'noninteractive'


################################################################################
#
#Ion database access
#Only head nodes will have dbase access
#
################################################################################
sys.path.append('/opt/ion/')
if host_is_master():
    os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
    from iondb.rundb import models


################################################################################
#
# Class Definition: TSconfig
#
################################################################################
class TSconfig (object):
    
    def __init__(self):
                
        # Lists of ubuntu packages required for Torrent Server.
        # See function updatePackageLists()
        self.SYS_PKG_LIST=[]
        self.SYS_PKG_LIST_MASTER_ONLY=[]
        self.ION_PKG_LIST=[]
        self.ION_PKG_LIST_MASTER_ONLY=[]
        
        # Internal states
        self.upst = {
            'U':'Unknown',
            'C':'Checking for update',
            'A':'Available',
            'N':'No updates',
            'UF':'Update failure',
            'DL':'Downloading',
            'DF':'Download failure',
            'RI':'Ready to install',
            'I':'Installing',
            'IF':'Install failure',
            'F':'Finished installing',
        }
    
        # User-facing status messages
        self.user_status_msgs = {
            '':'Updates available',
            '':'System is up to date',
            '':'Error during update check',
            '':'Updates downloading',
            '':'Updates installing',
            '':'Error downloading',
            '':'Error installing',
            '':'Error configuring',
        }
        
        self.state = 'U'                    # Internal state of the object
        self.autodownloadenabled = False    # Auto-download flag
        self.userackdownload = False        # User acknowledged Download
        self.userackinstall = False         # User acknowledged Install
        self.testrun = False                # Flag for debugging without making any changes to system
        self.status_for_user = None         # Status string entry in database
        self.pkgprogress = None             # String of format "3/12" where package number of total package number progress.
        self.dbaccess = False               # Set when we can talk to database
        self.pkglist = []                   # List of packages with an update available
        self.logger = logger
        self.packageListFile = os.path.join('/','usr','share','ion-tsconfig','torrentsuite-packagelist.json')
        self.updatePackageLists()
        
        if not host_is_master():
            self.dbaccess = False
            self.logger.info("Dbase access disabled; not a head node")
        else:
            try:
                gc = models.GlobalConfig.objects.all()[0]
                self.dbaccess = True
                self.logger.info("Dbase access enabled")
            except:
                self.dbaccess = False
                self.logger.info("Dbase access disabled")
        
        self.logger.info("TSconfig.__init__() executing")
            
    #--- End of __init__ ---#

    def reload_logger(self):
        logging.shutdown()
        self.logger = PickleLogger()

    def set_testrun(self,flag):
        self.testrun = flag
        
    def get_state(self):
        return self.state
    
    def get_state_msg(self):
        return self.upst.get(self.state,'Developer Error')
        
    def set_state(self, new_state):
        if new_state == self.state:
            return
        try:
            models.GlobalConfig.objects.update(ts_update_status=self.upst[new_state])
            self.state = new_state
        except Exception as err:
            self.logger.error("Failed setting GlobalConfig ts_update_status to '%s'" % new_state)
            raise err

    def reset_pkgprogress(self, current=0, total=0):
        self.progress_current = current
        self.progress_total = total
        self.pkgprogress = "%s %d/%d" % (self.upst[self.state], self.progress_current, self.progress_total)


    def add_pkgprogress(self, progress=1):
        self.progress_current += progress
        self.pkgprogress = "%s %d/%d" % (self.upst[self.state], self.progress_current, self.progress_total)
        if self.dbaccess:
            try:
                models.GlobalConfig.objects.update(ts_update_status=self.pkgprogress)
                self.logger.debug("Progress %s" % self.pkgprogress)
            except:
                self.logger.warn("Unable to update database with progress")
                
    def get_pkgprogress(self,current,total):
        return self.pkgprogress
            
    def set_autodownloadflag(self,flag):
        self.autodownloadenabled = flag
            
    def get_autodownloadflag(self):
        return self.autodownloadenabled
            
    def set_userackdownload(self,flag):
        self.userackdownload = flag
            
    def get_userackdownload(self):
        return self.userackdownload
            
    def set_userackinstall(self,flag):
        self.userackinstall = flag
            
    def get_userackinstall(self):
        return self.userackinstall
            
    def get_syspkglist(self):
        if host_is_master():
            list = self.SYS_PKG_LIST_MASTER_ONLY + self.SYS_PKG_LIST
        else:
            list = self.SYS_PKG_LIST
        return list

    def get_ionpkglist(self):
        if host_is_master():
            list = self.ION_PKG_LIST + self.ION_PKG_LIST_MASTER_ONLY
        else:
            list = self.ION_PKG_LIST
        return list
    
    ################################################################################
    #
    # Update internal list of packages to install
    #
    ################################################################################
    def updatePackageLists(self):

        try:
            
            self.logger.info("parsing %s" % self.packageListFile)
            fp = open(self.packageListFile,'r')
            pkgObj = json.load(fp)
        
            self.SYS_PKG_LIST               = pkgObj['packages']['system']['allservers']
            
            self.SYS_PKG_LIST_MASTER_ONLY   = pkgObj['packages']['system']['master']
            
            self.ION_PKG_LIST               = pkgObj['packages']['torrentsuite']['allservers']
            
            self.ION_PKG_LIST_MASTER_ONLY   = pkgObj['packages']['torrentsuite']['master']
            
            fp.close()
            
        except:
            self.logger.exception(traceback.format_exc())
            raise
        
    ################################################################################
    #
    # Update apt repository database
    #
    ################################################################################
    def updatePkgDatabase(self):
        '''Calls apt-get update to update list of available packages.
        NOTE!!! This requires root permission to execute successfully'''
        cmd1 = ['/usr/bin/apt-get',
               'update']
        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p1.communicate()
        if p1.returncode != 0:
            self.logger.error(stderr)
            
        return p1.returncode == 0
    
    ################################################################################
    #
    # Finds out if there are updates to packages
    #
    ################################################################################
    def TSpoll_pkgs(self):
        '''Returns True when there are updates to the Ion Torrent Suite software packages
        Returns list of packages to update'''
        self.set_state('C')
        if not self.updatePkgDatabase():
            self.logger.warn ("Could not update apt package database")
            self.set_state('UF')
            return None
        else:
            self.updatePackageLists()
            status, list = self.pollForUpdates()
            if status and len(list) > 0:
                self.logger.info("There are %d updates!" % len(list))
                self.set_state('A')
                self.pkglist = list
            else:
                self.set_state('N')
            
            return list
    
    ################################################################################
    #
    # Purge package files
    #
    ################################################################################
    def TSpurge_pkgs(self):
        try:
            cmd = ['/usr/bin/apt-get',
                   'autoclean']
            p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout,stderr = p1.communicate()
            if p1.returncode == 0:
                self.logger.info ("autocleaned apt cache directory")
            else:
                self.logger.info ("Error during autoclean: %s" % stderr)
        except:
            self.logger.exception(traceback.format_exc())
    
    ################################################################################
    #
    # Download package files
    #
    ################################################################################
    def TSdownload_pkgs(self,pkglist):
        '''Uses apt-get to download available packages'''
        results = {}
        self.set_state('DL')
        numpkgs = len(pkglist)
        for i,pkg in enumerate(pkglist):
            self.add_pkgprogress()
            cmd = ['/usr/bin/apt-get',
                   '--assume-yes',
                   '--force-yes',
                   '--download-only',
                   'install',
                   pkg]
            p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p1.communicate()
            self.logger.debug("Downloading %d of %d %s" % (i+1,numpkgs,pkg))
            if p1.returncode == 0:
                results[pkg] = True
            else:
                self.logger.error(stderr)
                results[pkg] = False

        return results
    
    ################################################################################
    #
    # Install package files
    #
    ################################################################################
    def TSinstall_pkgs(self,pkglist):
        '''Calls apt-get install on each package in list'''
        self.logger.debug("Inside TSinstall_pkgs")
        installed = {}
        numpkgs = len(pkglist)
        for i,pkg in enumerate(pkglist):
            cmd = ['/usr/bin/apt-get',
                   '--assume-yes',
                   '--force-yes',
                   'install',
                   pkg]
            if not self.testrun:
                self.logger.info("Installing %d of %d %s" % (i+1,numpkgs,pkg))
                self.add_pkgprogress()
                p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = p1.communicate()
                self.logger.debug(stdout)
                if p1.returncode == 0:
                    installed[pkg] = True
                else:
                    self.logger.error("Package '%s' failed:\n%s" % (pkg, stderr))
                    installed[pkg] = False
            else:
                self.add_pkgprogress()
                self.logger.info("FAKE! Installing %d of %d %s" % (i+1,numpkgs,pkg))
                
        failed = [k for k, v in installed.items() if not v]
        if failed:
            dl_err = 'Upgrade failed to install completely.'
            self.logger.error(dl_err)
            # Potentially do something with `failed` here, such as return it.
        return all(installed.values())

    ################################################################################
    #
    # Search for specific files that have an updated version available
    #
    ################################################################################
    def pollForUpdates(self):
        '''Returns 0 when there is an update.
        Checks for any updates for packages in ION_PKG_LIST and ION_PKG_LIST_MASTER_ONLY'''
        self.set_state('C')
        status = False
        pkglist = []
        cmd1 = ['/usr/bin/apt-get',
               'upgrade',
               '--simulate']
        pkgnames = self.ION_PKG_LIST
        pkgnames.extend(self.ION_PKG_LIST_MASTER_ONLY)
        searchterms = ["^Inst "+l for l in pkgnames]
        for pkg in searchterms:
            self.logger.debug("Checking: %s" % pkg)
            cmd2 = ['grep',pkg]
            p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p2 = subprocess.Popen(cmd2, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p2.communicate()
            p1.communicate()
    
            # p1 will have no stdout at this point since it was all read by p2
            if p1.wait() != 0:
                self.logger.error(stderr)
            else:
                if stdout.splitlines():
                    name = stdout.splitlines()[0].split(" ")[1]
                    pkglist.append(name)
                    self.logger.debug("Has update: %s" % stdout.splitlines()[0])
                
        #self.logger.debug("%d updates found." % len(pkglist))
        if len(pkglist) > 0:
            status = True

        return status, pkglist
    
    
    ################################################################################
    #
    # Download sytem and Ion debian package files
    #
    ################################################################################
    def TSexec_download(self):
        self.updatePackageLists()
        syspkglist = self.get_syspkglist()
        ionpkglist = self.get_ionpkglist()
        self.reset_pkgprogress(total=len(syspkglist) + len(ionpkglist))

        #================================
        # autoclean the apt cache
        #================================
        self.TSpurge_pkgs()
        
        #================================
        # Download system packages
        #================================
        sys_status = self.TSdownload_pkgs(syspkglist)
        if not any(sys_status.values()):
            self.logger.error("All system packages failed to download!")
            self.set_state('DF')
        elif not all(sys_status.values()):
            failed = " ".join(p for p, s in sys_status.items() if not s)
            self.logger.warn("sys packages %s failed to download!" % failed)
            self.set_state('DF')

        #================================
        # Download Ion packages
        #================================
        ion_status = self.TSdownload_pkgs(ionpkglist)
        if not any(ion_status.values()):
            self.logger.error("All ion packages failed to download!")
            models.Message.error("All ion packages failed to download!", 'updates')
            self.set_state('DF')
        elif not all(ion_status.values()):
            failed = " ".join(p for p, s in ion_status.items() if not s)
            self.logger.warn("ion packages %s failed to download!" % failed)
            models.Message.warn("Ion packages %s failed to download!" % failed, 'updates')
            self.set_state('DF')

        if all(ion_status.values()) and all(sys_status.values()):
            self.set_state('RI')
        return sys_status.items() + ion_status.items()
    
    ################################################################################
    #
    # Wrappers to TSconfig configuration functions
    #
    ################################################################################
    def TSpreinst_syspkg(self):
        cmd = ["/usr/sbin/TSwrapper"
               " preinst_system_packages"]
        
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        stdout, stderr = p1.communicate()
        self.logger.debug("preinst_system_packages")
        self.logger.debug(stdout)
        if p1.returncode == 0:
            pass
        else:
            self.logger.error(stderr)
        
        
        return
    def TSpostinst_syspkg(self):
        cmd = ["/usr/sbin/TSwrapper"
               " config_system_packages"]
        
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        stdout, stderr = p1.communicate()
        self.logger.debug("config_system_packages")
        self.logger.debug(stdout)
        if p1.returncode == 0:
            pass
        else:
            self.logger.error(stderr)
        
        return
    def TSpreinst_ionpkg(self):
        #================================
        #Nothing to do at the moment
        #================================
        self.logger.debug("TSpreinst_ionpkg")
        pass

        return
    def TSpostinst_ionpkg(self):
        cmd = ["/usr/sbin/TSwrapper"
               " config_ion_packages"]
        
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        stdout, stderr = p1.communicate()
        self.logger.debug("config_ion_packages")
        self.logger.debug(stdout)
        if p1.returncode == 0:
            pass
        else:
            self.logger.error(stderr)
        
        return
    
    ################################################################################
    #
    # Install sytem and Ion debian package files and run config commands
    #
    ################################################################################
    def TSexec_update(self):
        
        try:
            self.set_state('I')
            self.logger.debug("Inside TSexec_update")
            self.logger.debug("%d and %d" % (len(self.get_syspkglist()),len(self.get_ionpkglist())))
            self.updatePackageLists()
            syspkglist = self.get_syspkglist()
            ionpkglist = self.get_ionpkglist()
            self.reset_pkgprogress(total=len(syspkglist) + len(ionpkglist))
            # Install TSconfig first so that it's code, executed through TSwrapper
            # is upgraded Before execution below.
            tsconfig_result = self.TSinstall_pkgs(["ion-tsconfig"])
            if not tsconfig_result:
                self.logger.warning("Could not install ion-tsconfig!")
                return False
    
            #================================
            # Get latest package lists
            #================================
            self.updatePackageLists()
    
            #================================
            # Execute pre-System package install
            #================================
            self.TSpreinst_syspkg()
            
            #================================
            # Install System packages
            #================================
            sys_result = self.TSinstall_pkgs(syspkglist)
            
            #================================
            # Execute System configuration
            #================================
            self.TSpostinst_syspkg()
            
            #================================
            # Execute pre-Ion package install
            #================================
            self.TSpreinst_ionpkg()
            
            #================================
            # Install Ion packages
            #================================
            ion_result = self.TSinstall_pkgs(ionpkglist)
            
            #================================
            # Execute Ion configuration
            #================================
            self.TSpostinst_ionpkg()
            
        except:
            self.logger.exception(traceback.format_exc())
            
        success = sys_result and ion_result
        if success:
            self.logger.info("Successfully TSconfigured !")
        else:
            self.logger.error("Failed to TSconfigure.")
        return success

################################################################################
#
# End Class Definition: TSconfig
#
################################################################################
