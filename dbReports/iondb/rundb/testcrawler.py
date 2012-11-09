#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import sys
import os
#sys.path.append('/opt/ion/')
#os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from crawler import *

if __name__ == '__main__':
    f = "/home/bpuc"
    text = load_log(f,"explog.txt")
    d = parse_log(text)
    kwargs,t = exp_kwargs(d,f)
    print kwargs['autoAnalyze']
