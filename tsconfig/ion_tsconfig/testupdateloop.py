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

import time
import iondb.bin.djangoinit
#from ion_tsconfig.TSconfig import *
from TSconfig import *

################################################################################
#
# Test code showing complete update process
#
################################################################################
def software_update_loop(TSconfig):
    '''Mostly pseudocode to describe the software update process'''
    loop_enabled = True
    
    while loop_enabled:
        #================================
        # Check for new Ion package files
        #================================
        
        new_files = TSconfig.TSpoll_pkgs()
        space = TSconfig.freespace('/var/')
        print "/var partition: %d mbytes" % space
        
        #if new_files:
        #if True:
        if False:
            #================================
            # Check for auto-download enabled
            #================================
            auto_download = TSconfig.get_autodownloadflag()
            
            if not auto_download:
                #================================
                # Get User Acknowledge To Initiate Download
                #================================
                user_ack_download = TSconfig.get_userackdownload()
            
            if auto_download or user_ack_download:

                TSconfig.TSexec_download()
            
                #================================
                # Get User Acknowledge To Execute Software Install
                #================================
                user_acknowledge = TSconfig.get_userackinstall()
                
                if user_acknowledge:
                    
                    TSconfig.TSexec_update()
        else:
            pass
            # No new packages, or error determining.
            # Also, Use Case: No internet access, needs USB?  TSpoll_pkgs() needs to be USB aware?
        
        loop_enabled = False
        ## Take a breather
        #loop_interval = (3600 if not DEBUG else 5)
        #time.sleep(loop_interval)
        
    return None

if __name__ == '__main__':
    
    # TODO:Check for root permission

    newclass = TSconfig()
    newclass.set_testrun(True)
    newclass.set_autodownloadflag(False)
    newclass.set_userackinstall(False)
    software_update_loop(newclass)
    print "Status at exit %s" % newclass.get_state_msg() 
    
