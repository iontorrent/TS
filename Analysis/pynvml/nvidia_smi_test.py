#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

import sys
from os.path import dirname, abspath
from distutils.sysconfig import get_python_lib

fpath = abspath(__file__)
prefix = dirname(dirname(fpath))
pypath = get_python_lib(prefix=prefix)
if pypath not in sys.path:
    sys.path.append(pypath)

import nvidia_smi

print(nvidia_smi.XmlDeviceQuery())
