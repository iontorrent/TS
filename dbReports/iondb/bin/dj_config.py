# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
"""
Reusable functions to determine common configuration information and perform
common tasks related to the environment in which Django functions.

In general, this should not be used by components internal to Django to read
system state.  Instead, settings.py should get this info and settings.py
should be used to access such information.  This will ensure that we maintain
a single authoritative source of such settings with a simple interface.

"Configuration information" which can be expected to change many times during
the life of the Django process should not be defined here.  Put that elsewhere.
"""

import os
import subprocess
import requests


def call(*cmd):
    # Root user at boot requires this addition to its path variable to locate tmap
    env = dict(os.environ)
    env['PATH'] = '/usr/local/bin:' + env['PATH']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    stdout, stderr = proc.communicate()
    if proc.returncode == 0:
        # the call returned successfully
        return stdout.strip()
    else:
        # the call returned with an error
        return None


def get_tmap_version():
    return call("tmap", "index", "--version")


def is_s5_tsvm():
    """This method will return true if we are part of the S5 tsvm, false otherwise"""
    # I am not super excited to use the existance of such a file as there could be a point
    # in time in the future where the 1:1 correlation may change between the existance of
    # this file and this TS being a S5 tsvm instance
    return os.path.exists('/etc/init.d/mountExternal')


def get_s5_ip_addr():
    """get the S5 IP address"""
    try:
        resp = requests.get('http://192.168.122.1/instrument/software/www/config/DataCollect.config', timeout=2.0)
        for line in resp.text.split('\n'):
            if 'IP Address Str' in line:
                return line[15:]
    except requests.ConnectionError:
        # we don't really need to do anything here asides from assume this is not an S5 and move on
        return None
