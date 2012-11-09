#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import xmlrpclib

if __name__ == '__main__':
    
    # Connect to ionCrawler daemon
    proxy = xmlrpclib.ServerProxy('http://127.0.0.1:%d' % 10001)
    experiment_name = "R_2012_02_08_13_37_53_70dc054aa3"
    pe_forward = ""
    pe_reverse = ""
    proxy.startPE(experiment_name,pe_forward,pe_reverse)
