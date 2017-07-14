#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

import os
import socket
from iondb.bin import djangoinit
from iondb.rundb import models
from iondb.utils.utils import is_TsVm

def instrument_host_ip():
    '''Returns ip address of the host system'''
    # This only works because S5 has embedded TsVm, thus there should only be
    # one instrument configured in the Rig table.  We grab the one with the most
    # recent init date which is likely to be the correct entry.  instruments
    # record their IP address in the host_address field.
    # the host_address field is always an IP address
    try:
        for rig in models.Rig.objects.all().order_by('-last_init_date'):
            if rig.host_address:
                ipaddr = rig.host_address
                break
    except:
        ipaddr = '127.0.0.1'
    return ipaddr


def gethostip():
    '''Returns IP address of server, or if its a TsVm (S5 virtual machine) it returns
    the IP of the instrument host server'''
    if is_TsVm():
        ipaddr = instrument_host_ip()
    else:
        # Filched from
        # http://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 0))
            ipaddr = s.getsockname()[0]
        except:
            ipaddr = '127.0.0.1'
        finally:
            s.close()
    return ipaddr


def gethostname():
    '''Returns hostname of server, or if its a TsVm (S5 virtual machine) it returns
    the hostname of the instrument host server'''
    if is_TsVm():
        ipaddr = instrument_host_ip()
        try:
            hostname, _, _ = socket.gethostbyaddr(ipaddr)
        except:
            hostname = 'tsvm'
    else:
        hostname = socket.gethostname()
    return hostname


if __name__ == '__main__':
    print(gethostname())
    print(gethostip())
