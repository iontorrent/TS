#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

'''
    TSconfig.py works with the GUI updates page to check if new version is available and update TS
    In TS version < 5.4 update tasks instantiate TSConfig class directly, in order for this to work
        all class functions listed as called by dbReports must remain intact
    Starting with TS 5.4 GUI update tasks call functions via the script's CLI
'''

import os
import sys
import time
import traceback
import subprocess
import logging
import logging.handlers
import apt
import apt_pkg
import argparse
from distutils.version import LooseVersion


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


def manual_is_upgradable(pkg):
    """This method is used to get around a current bug in aptitude where some versions are being marked as upgradable incorrectly"""
    return LooseVersion(pkg.candidate.version) > LooseVersion(pkg.installed.version)

#-------------------------------------------------------------------------------
#
# Utility functions
#
#-------------------------------------------------------------------------------

# TODO: this is available in ts_params
CONF_FILE = "/etc/torrentserver/tsconf.conf"


def host_is_master():
    '''Returns true if current host is configured as master node'''
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


def get_apt_cache_dir():
    try:
        apt_pkg.InitConfig()
        _dir = os.path.join(
            apt_pkg.config.get("dir"),
            apt_pkg.config.get("dir::cache"),
            apt_pkg.config.get("dir::cache::archives")
        )
    except:
        _dir = "/var/cache/apt/archives"    # default setting
    return _dir


def freespace(directory):
    '''Returns free disk space for given directory in megabytes'''
    try:
        s = os.statvfs(directory)
    except:
        logger.error(traceback.format_exc())
        mbytes = -1
    else:
        mbytes = (s.f_bsize * s.f_bavail) / (1024 * 1024)

    return mbytes

# This tells apt-get not to expect access to standard in.
os.environ['DEBIAN_FRONTEND'] = 'noninteractive'


#-------------------------------------------------------------------------------
#
#Ion database access
#Only head nodes will have dbase access
#
#-------------------------------------------------------------------------------
sys.path.append('/opt/ion/')
if host_is_master():
    os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
    from iondb.rundb import models
    logger.disabled = False


#-------------------------------------------------------------------------------
#
# python-apt
#
#-------------------------------------------------------------------------------

class GetAcquireProgress(apt.progress.base.AcquireProgress):

    '''
    Handle the package download process for apt_pkg.Acquire
    '''

    def __init__(self, tsconfig):
        apt.progress.base.AcquireProgress.__init__(self)
        self._width = 80
        self._id = 1
        self.tsconfig = tsconfig
        pass

    def start(self):
        self.tsconfig.logger.debug("[GetAcquireProgress] StartAcquire")

    def stop(self):
        self.tsconfig.logger.debug("[GetAcquireProgress] StopAcquire")

    def pulse(self, acquire):

        tsc = self.tsconfig
        progress = "Downloading %.1fMB/%.1fMB" % (self.current_bytes/(1024 * 1024), self.total_bytes/(1024 * 1024))
        tsc.update_progress(progress)

        item_idx = self.current_items
        if item_idx == self.total_items:
            item_idx -= 1
        destfile = acquire.items[item_idx].destfile
        destfile = destfile.split('/')[-1]
        debug_string = "[GetAcquireProgress] %s; CPS: %s/s; Bytes: %s/%s; Item: %s/%s" % (
            destfile,
            self.current_cps, self.current_bytes, self.total_bytes,
            item_idx+1, self.total_items
        )
        tsc.logger.debug(debug_string)

        return True


class GetInstallProgress(apt.progress.base.InstallProgress):

    '''
    Handle the package install process for apt.Cache
    '''

    def __init__(self, tsconfig):
        apt.progress.base.InstallProgress.__init__(self)
        self.tsconfig = tsconfig
        pass

    def startUpdate(self):
        self.tsconfig.logger.debug("[GetInstallProgress] StartInstall")

    def finishUpdate(self):
        self.tsconfig.logger.debug("[GetInstallProgress] FinishInstall")

    def status_change(self, pkg, percent, status):
        tsc = self.tsconfig

        progress = "%s %d%%" % (status, percent)
        tsc.update_progress(progress)

        self.tsconfig.logger.debug("[GetInstallProgress] %s [%s/100]" % (status, percent))

    def error(self, pkg, errormsg):
        tsc = self.tsconfig
        tsc.update_progress(errormsg)

        self.tsconfig.logger.error(errormsg)

    def conffile(self, old, new):
        self.tsconfig.logger.debug("conffile: %s to %s" % (old, new))

    def processing(self, pkg, stage):
        self.tsconfig.logger.debug("processing: %s -> %s" % (pkg, stage))

    def dpkg_status_change(self, pkg, status):
        self.tsconfig.logger.debug("status_change: %s %s" % (pkg, stage))


#-------------------------------------------------------------------------------
#
# List of packages used by GUI updates webpage to check if a TS update is Available
# This is NOT the complete list of packages that are installed by TSconfig
#
#-------------------------------------------------------------------------------
ION_PACKAGES = [
    'ion-analysis',
    'ion-chefupdates',
    'ion-dbreports',
    'ion-docs',
    'ion-gpu',
    'ion-onetouchupdater',
    'ion-pgmupdates',
    'ion-pipeline',
    'ion-plugins',
    'ion-protonupdates',
    'ion-publishers',
    'ion-referencelibrary',
    'ion-rsmts',
    'ion-s5updates',
    'ion-sampledata',
    'ion-torrentpy',
    'ion-torrentr',
    'ion-tsconfig',
]


#-------------------------------------------------------------------------------
#
# Class Definition: TSconfig
#   handles download of ion packages and installation of ion-tsconfig
#   launches Ansible script to update software
#
#-------------------------------------------------------------------------------
class TSconfig(object):

    def __init__(self):
        self.logger = logger
        self.ION_PKG_LIST = []
        self.apt_cache = None
        self.aptcachedir = get_apt_cache_dir()

    def updatePkgDatabase(self):
        '''Update apt cache '''
        retry = 5   # try five times
        sleepy = 2  # wait a couple seconds

        while retry:
            try:
                self.apt_cache = apt.Cache()
                self.apt_cache.update()
                self.apt_cache.open(None)
                self.logger.debug("Successfully updated apt cache")
                return True
            except Exception as err:
                logger.warn("Waiting to update apt cache: " + str(err))
                time.sleep(sleepy)
                retry -= 1
        self.logger.debug("Unable to retrieve apt cache")
        return False

    def TSpurge_pkgs(self):
        try:
            cmd = ['sudo', 'apt-get', 'autoclean']
            p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p1.communicate()
            if p1.returncode == 0:
                self.logger.info("autocleaned apt cache directory")
            else:
                self.logger.info("Error during autoclean: %s" % stderr)
        except:
            self.logger.error(traceback.format_exc())

    def update_progress(self, status):
        try:
            models.GlobalConfig.objects.update(ts_update_status=status)
        except:
            from django.db import connection
            connection.close()  # Force a new connection on next transaction
            try:
                models.GlobalConfig.objects.update(ts_update_status=status)
            except:
                self.logger.error("Unable to update database with progress")
                connection.close()  # Force a new connection on next transaction

    def updatePackageLists(self):
        self.ION_PKG_LIST = sorted([pkg for pkg in ION_PACKAGES if self.apt_cache.has_key(pkg) and self.apt_cache[pkg].is_installed])

    def buildUpdateList(self, pkgnames):
        pkglist = []
        for pkg_name in pkgnames:
            pkg = self.apt_cache[pkg_name]
            if manual_is_upgradable(pkg):
                pkglist.append(pkg_name)
                self.logger.debug("%s %s upgradable to %s" % (pkg.name, pkg.installed.version, pkg.candidate.version))
            else:
                self.logger.debug("%s %s" % (pkg.name, pkg.installed.version))

        self.logger.debug("Checked %s packages, found %s upgradable" % (len(pkgnames), len(pkglist)))

        return pkglist

    #-------------------------------------------------------------------------------
    #
    #   Functions called from dbReports
    #
    #-------------------------------------------------------------------------------

    def TSpoll_pkgs(self):
        '''Checks the Ion Torrent Suite software and returns list of packages to update '''
        self.update_progress('Checking for updates')

        ionpkglist = []
        if not self.updatePkgDatabase():
            self.logger.error("Could not update apt package database")
            self.update_progress('Failed checking for updates')
        else:
            self.updatePackageLists()
            ionpkglist = self.buildUpdateList(self.ION_PKG_LIST)
            if len(ionpkglist) > 0:
                self.update_progress('Available')
                self.logger.info("There are %d ion package updates!" % len(ionpkglist))

                # check available disk space
                self.TSpurge_pkgs()
                available = freespace(self.aptcachedir)
                self.apt_cache.upgrade()
                required = self.apt_cache.required_download / (1024 * 1024)
                self.apt_cache.clear()
                self.logger.info("%.1fMB required download space, %.1fMB available in %s." % (required, available, self.aptcachedir))

                if available < required:
                    msg = "WARNING: insufficient disk space for update"
                    self.update_progress(msg)
                    self.logger.debug(msg)
            else:
                self.update_progress('No updates')

        return ionpkglist

    def TSexec_download(self):
        downloaded = []
        success = False
        if not self.updatePkgDatabase():
            self.logger.error("Could not update apt package database")
        else:
            self.TSpurge_pkgs()
            self.update_progress('Downloading')
            self.apt_cache.upgrade()
            required = self.apt_cache.required_download / (1024 * 1024)
            downloaded = [pkg.name for pkg in self.apt_cache.get_changes()]
            self.logger.debug('%d upgradable packages, %.1fMB required download space' % (len(downloaded), required))

            # Download packages
            pm = apt_pkg.PackageManager(self.apt_cache._depcache)
            fetcher = apt_pkg.Acquire(GetAcquireProgress(self))
            try:
                self.apt_cache._fetch_archives(fetcher, pm)
                success = True
            except:
                self.logger.error(traceback.format_exc())

        if success:
            self.update_progress('Ready to install')
            self.logger.info("Successfully downloaded %s packages" % len(downloaded))
        else:
            self.update_progress('Download failure')
            self.logger.error("Failed downloading packages!")

        return downloaded

    def TSexec_update_tsconfig(self):
        if not self.apt_cache and not self.updatePkgDatabase():
            self.logger.error("Could not update apt package database")
            return False

        try:
            pkg = self.apt_cache['ion-tsconfig']
            if manual_is_upgradable(pkg):
                self.update_progress('Installing ion-tsconfig')
                self.apt_cache.clear()
                pkg.mark_upgrade()
                self.apt_cache.commit(GetAcquireProgress(self), GetInstallProgress(self))
                self.logger.info("Installed ion-tsconfig")
                return True
        except:
            self.logger.warning("Could not install ion-tsconfig!")
            self.logger.error(traceback.format_exc())
        return False

    def TSexec_update(self):
        self.logger.info("Starting update via ansible")
        self.update_progress('Installing')
        os.environ["TS_EULA_ACCEPTED"] = "1"
        # First step is to update software packages
        try:
            cmd = ["/usr/sbin/TSconfig", "-s", "--force"]
            p1 = subprocess.call(cmd)
            success = p1 == 0
        except:
            success = False
            self.logger.error(traceback.format_exc())

        # Next step is to configure the server, if update was successful
        if success:
            self.logger.info("Software Updated !")

            try:
                self.logger.debug("Starting configuration")
                cmd = ["/usr/sbin/TSconfig", "--configure-server", "--force", "--noninteractive"]
                p1 = subprocess.call(cmd)
                success = p1 == 0
            except:
                success = False
                self.logger.error(traceback.format_exc())

            if success:
                self.logger.info("TS Configured !")
                self.update_progress("Finished installing")
            else:
                self.logger.error("Failed to Configure TS.")
                self.update_progress("Install failure")

        else:
            self.logger.error("Failed to Update Software.")
            self.update_progress("Install failure")

        # This will start celery if it is not running for any reason after upgrade
        try:
            subprocess.call(['service', 'celeryd', 'start'])
            subprocess.call(['service', 'celerybeat', 'start'])
        except:
            pass

        return success

    def set_securityinstall(self, flag):
        # legacy function, leave it to be able to upgrade from TS < 5.4
        pass

#-------------------------------------------------------------------------------
#
# End Class Definition: TSconfig
#
#-------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TSconfig.py check for and install new TS software updates")
    parser.add_argument('-p', '--poll', dest='poll', action='store_true', default=False, help='check for new ion packages')
    parser.add_argument('-d', '--download', dest='download', action='store_true', default=False, help='download all available packages')
    parser.add_argument('-r', '--refresh', dest='refresh', action='store_true', default=False, help='update ion-tsconfig to latest version')
    parser.add_argument('-s', '--upgrade', dest='upgrade', action='store_true', default=False, help='starts software update via ansible')

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error('no option specified')
        sys.exit(2)

    tsconfig = TSconfig()

    if args.poll:
        try:
            packages = tsconfig.TSpoll_pkgs()
            print(len(packages))
        except Exception as err:
            logger.error(traceback.format_exc())
            sys.exit("Update failure")

    if args.download:
        try:
            downloaded = tsconfig.TSexec_download()
            print(len(downloaded))
        except Exception as err:
            logger.error(traceback.format_exc())
            sys.exit("Download failure")
    
    if args.refresh:
        try:
            is_new = tsconfig.TSexec_update_tsconfig()
            print(is_new)
        except Exception as err:
            logger.error(traceback.format_exc())
            sys.exit("Failed to install ion-tsconfig")

    if args.upgrade:
        # This action will run from a daemonized process
        try:
            success = tsconfig.TSexec_update()
        except Exception as err:
            logger.error(traceback.format_exc())
            success = False

        if not success:
            sys.exit("Install failure")
