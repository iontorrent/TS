# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# Test code to hit the xmlrpc server run by backup.py
#
from twisted.web.xmlrpc import Proxy
import xmlrpclib
import time

proxy = xmlrpclib.ServerProxy('http://10.25.3.30:9191')
queryCnt = 0
while True:
    startT = time.time()
    whatsit = proxy.uptime()
    stopT = time.time()
    print "query: %d uptime: %d seconds" % (queryCnt, whatsit)
    time.sleep(5)
    queryCnt += 1
