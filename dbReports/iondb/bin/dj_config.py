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


import subprocess


def call(*cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode == 0:
        # the call returned successfully
        return stdout.strip()
    else:
        # the call returned with an error
        return None


def get_tmap_version():
    return call("tmap", "index", "--version")

