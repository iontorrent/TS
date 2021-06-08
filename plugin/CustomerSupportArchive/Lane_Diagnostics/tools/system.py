'''
Advanced system functions
'''
import os, sys
import subprocess
from network import get_domain

def makedir( dirname ):
    '''
    Makes the selected directory and parents if not already present
    '''
    if os.path.exists( os.path.normpath( dirname ) ):
        return
    try:
        os.makedirs( os.path.normpath( dirname ) )
    except OSError:
        r = subprocess.Popen('sudo -S mkdir -p %s' % dirname, stdout=subprocess.PIPE , 
                             stdin=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True )
        vacuum = r.communicate('ionadmin\n')[0]
        r = subprocess.Popen('sudo -S chown ionadmin:ionadmin %s' % dirname, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True )
        vacuum = r.communicate('ionadmin\n')[0]
        if not os.path.exists( os.path.normpath( dirname ) ):
            raise

def get_instrument_name( ):
    """Test results folder to grab 2-3 character instrument name"""
    # First try to ask for hostname
    proton = subprocess.Popen(["hostname"], stdout=subprocess.PIPE, shell=True).communicate()[0].splitlines()[0]
    # OK, try to get it from the data directories
    if proton == '':
        folders = subprocess.Popen(["ls /sw_results | grep '-'"], stdout=subprocess.PIPE, shell=True).communicate()[0]
        folders = folders.splitlines()
        for folder in folders:
            if '-' in folder:
                proton = folder.split('-')[0]
                break

    if proton:
        return proton
    raise ValueError( 'Could not determine proton name' )
