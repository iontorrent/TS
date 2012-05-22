#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
################################################################################
#
# The software update procedure currently is handled by a bunch of shell script functions:
# config_ionsoftware calls the following:
#    ion_daemon_ctrl stop
#    install_system_packages
#    config_system_packages
#    install_ion_packages
#
################################################################################

import os
import time
import subprocess
import traceback
import sys
import logging

# Python has an amazing logging system, in production you'd beef it up, but here
# is the quick and easy version
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
        format="%(asctime)s\t%(levelname)s\t%(funcName)s\t%(message)s")

DEBUG = True

# Lists of ubuntu packages required for Torrent Server.  Currently needs to be manually
# synchronized with lists stored in ts_functions.
SYS_PKG_LIST=[
    'traceroute',
    'arp-scan',
    'nmap',
    'links',
    'lynx',
    'minicom',
    'mc',
    'screen',
    'make',
    'gdb',
    'gcc',
    'build-essential',
    'imagemagick',
    'emacs23',
    'vim',
    'nano',
    'curl',
    'whois',
    'figlet',
    'binutils',
    'sshpass',
    'tk8.5',
    'libmotif-dev',
    'libxpm-dev',
    'xorg',
    'xfce4',
    'gridengine-common',
    'gridengine-client',
    'gridengine-exec',
    'libdrmaa1.0',
    'iptables',
    'ntp',
    'nfs-kernel-server',
    'openssh-server',
    'samba',
    'zip',
    'libz-dev',
    'postfix',
    'python-pysam',
    'python-simplejson',
    'python-calabash',
    'python-jsonpipe',
    'perl',
    'perl-doc',
]

SYS_PKG_LIST_MASTER_ONLY=[
    'gridengine-master',
    'gridengine-qmon',
    'rabbitmq-server',
    'vsftpd',
    'postgresql',
    'apache2',
    'apache2-mpm-prefork',
    'libapache2-mod-python',
    'libapache2-mod-php5',
    'dnsmasq',
    'dhcp3-server',
    'tomcat6',
    'tomcat6-admin',
]

ION_PKG_LIST=[
    'ion-gpu',
    'ion-analysis',
    'ion-alignment',
    'ion-pipeline',
    'tmap',
    'samtools',
    'ion-rsmts',
    'ion-samita',
    'ion-torrentr',
]
ION_PKG_LIST_MASTER_ONLY=[
    'ion-plugins',
    'ion-pgmupdates',
    'ion-docs',
    'ion-referencelibrary',
    'ion-sampledata',
    'ion-publishers',
    'ion-onetouchupdater',
    'ion-dbreports',
]

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
                    logging.debug("Found mode:master in %s" % CONF_FILE)
                    return True
                elif "mode:compute" in line:
                    logging.debug("Found mode:compute in %s" % CONF_FILE)
                    return False
        except IOError as err:
            logging.error(err.message)

    if os.path.isfile("/opt/ion/.masternode") and not os.path.isfile("/opt/ion/.computenode"):
        logging.debug("Using flag files to determine master host status")
        return True
        
    raise OSError("Host not configured as either master or compute node")


def get_syspkglist():
    if host_is_master():
        list = SYS_PKG_LIST_MASTER_ONLY + SYS_PKG_LIST
    else:
        list = SYS_PKG_LIST
    return list

def get_ionpkglist():
    if host_is_master():
        list = ION_PKG_LIST_MASTER_ONLY + ION_PKG_LIST
    else:
        list = ION_PKG_LIST
    return list

################################################################################
#
# Finds out if there are updates to packages
#
################################################################################
def TSpoll_pkgs():
    '''Returns True when there are updates to the Ion Torrent Suite software packages
    Returns list of packages to update'''
    if not updatePkgDatabase():
        logging.warn("Could not update apt package database")
        return None
    else:
        err, list = pollForUpdates()
        if not err and len(list) > 0:
            logging.info("There are %d updates!" % len(list))
        return list

################################################################################
#
# Download package files
#
################################################################################
def TSdownload_pkgs(pkglist):
    '''Uses apt-get to download available packages'''
    results = {}
    for pkg in pkglist:
        cmd = ['/usr/bin/apt-get',
               '--assume-yes',
               '--force-yes',
               '--download-only',
               'install',
               pkg]
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p1.communicate()
        logging.debug("Checking %s" % pkg)
        if p1.returncode == 0:
            results[pkg] = True
        else:
            logging.error(stderr)
            results[pkg] = False
    return results

################################################################################
#
# Install package files
#
################################################################################
def TSinstall_pkgs(pkglist):
    '''Calls apt-get install on each package in list'''
    errstatus = False
    for pkg in pkglist:
        cmd = ['/usr/bin/apt-get',
               '--assume-yes',
               '--force-yes',
               'install',
               pkg]
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p1.communicate()
        logging.debug(stdout)
        if p1.returncode != 0:
            logging.error(stderr)
            errstatus = True
    return errstatus

################################################################################
#
# Update apt repository database
#
################################################################################
def updatePkgDatabase():
    '''Calls apt-get update to update list of available packages.
    NOTE!!! This requires root permission to execute successfully'''
    cmd1 = ['/usr/bin/apt-get',
           'update']
    p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p1.communicate()
    if p1.returncode != 0:
        logging.error(stderr)

    return p1.returncode == 0

################################################################################
#
# Search for specific files that have an updated version available
#
################################################################################
def pollForUpdates():
    '''Returns 0 when there is an update.
    Checks for any updated ion-* package, for now.'''
    status = False
    pkglist = []
    cmd1 = ['/usr/bin/apt-get',
           'upgrade',
           '--simulate']
    cmd2 = ['grep','^Inst ion-']
    p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p2.communicate()

    # p1 will have no stdout at this point since it was all read by p2
    if p1.wait() != 0:
        logging.error(stderr)
        status = False
    else:
        # just for fun
        pkglist = [l.split(" ")[1] for l in stdout.splitlines() if len(l) > 0]
        logging.debug("%d updates found." % len(pkglist))
        status = True

    return status, pkglist


################################################################################
#
# Download sytem and Ion debian package files
#
################################################################################
def TSexec_download():
    """Returns True is all packages succeeded in downloading, otherwise False.
    """
    #================================
    # Download system packages
    #================================
    list = get_syspkglist()
    sys_status = TSdownload_pkgs(list)
    if not any(sys_status.values()):
        logging.error("All ion packages failed to download!")
    elif not all(sys_status.values()):
        failed = " ".join(p for p, s in sys_status.items() if not s)
        logging.warn("ion packages %s failed to download!" % failed)
    
    #================================
    # Download Ion packages
    #================================
    list = get_ionpkglist()
    ion_status = TSdownload_pkgs(list)    # TODO: Check return status
    if not any(ion_status.values()):
        logging.error("All ion packages failed to download!")
    elif not all(ion_status.values()):
        failed = " ".join(p for p, s in ion_status.items() if not s)
        logging.warn("ion packages %s failed to download!" % failed)
    return all(ion_status.values()) and all(sys_status.values())

################################################################################
#
# Wrappers to TSconfig configuration functions
#
################################################################################
def TSpreinst_syspkg():
    #================================
    #TODO: cleanup from previous software defined in ts_functions
    #TODO: setup preseeds defined in ts_functions
    #================================
    return
def TSpostinst_syspkg():
    #================================
    #TODO: call config_system_packages() defined in ts_functions
    #================================
    return
def TSpreinst_ionpkg():
    #================================
    #Nothing to do at the moment
    #================================
    return
def TSpostinst_ionpkg():
    #================================
    #TODO: Make sure all Ion daemons are running defined in ts_functions
    #================================
    return

################################################################################
#
# Install sytem and Ion debian package files and run config commands
#
################################################################################
def TSexec_update(syspkglist,ionpkglist):
    # TODO: Remove this to go live
    if DEBUG:
        return
    
    #================================
    # TODO:Execute pre-System package install
    #================================
    TSpreinst_syspkg()
    
    #================================
    # TODO:Install System packages
    #================================
    TSinstall_pkgs(syspkglist)
    
    #================================
    # TODO:Execute System configuration
    #================================
    TSpostinst_syspkg()
    
    #================================
    # TODO:Execute pre-Ion package install
    #================================
    TSpreinst_ionpkg()
    
    #================================
    # TODO:Install Ion packages
    #================================
    TSinstall_pkgs(ionpkglist)
    
    #================================
    # TODO:Execute Ion configuration
    #================================
    TSpostinst_ionpkg()
    
    return

################################################################################
#
# Test code showing complete update process
#
################################################################################
def software_update_loop():
    '''Mostly pseudocode to describe the software update process'''
    loop_enabled = True
    #Status variable
    upst = {
        'U':'Unknown',
        'C':'Checking for update',
        'A':'Available',
        'N':'No updates',
        'DL':'Downloading debian files',
        'RI':'Ready to install',
        'I':'Installing',
        'F':'Finished installing',
    }
    update_status = upst['U']
    
    while loop_enabled:
        #================================
        # Check for new Ion package files
        #================================
        update_status = upst['C']
        
        new_files = TSpoll_pkgs()
        
        if new_files:
            update_status = upst['A']
            
            #================================
            # TODO: Check for auto-download enabled
            #================================
            auto_download = True
            
            if not auto_download:
                #================================
                # TODO:Get User Acknowledge To Initiate Download
                #================================
                user_ack_download = True
            
            if auto_download or user_ack_download:
                
                update_status = upst['DL']

                TSexec_download()
                
                update_status = upst['RI']
            
                #================================
                # TODO:Get User Acknowledge To Execute Software Install
                #================================
                user_acknowledge = False
                
                if user_acknowledge:
                    
                    update_status = upst['I']
                    
                    TSexec_update()
                    
                    update_status = upst['F']

        else:
            update_status = upst['N']
            # No new packages, or error determining.
            # Also, Use Case: No internet access, needs USB?  TSpoll_pkgs() needs to be USB aware?
        
        loop_enabled = False
        # Take a breather
        loop_interval = (3600 if not DEBUG else 5)
        time.sleep(loop_interval)
        
    return None

if __name__ == '__main__':

    # TODO:Check for root permission

    software_update_loop()
    logging.shutdown()
