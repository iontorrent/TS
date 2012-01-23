# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# Test code to hit the xmlrpc server run by backup.py
#
from twisted.web.xmlrpc import Proxy
import xmlrpclib
import time

proxy = xmlrpclib.ServerProxy('http://127.0.0.1:%d' % 10002)
queryCnt = 0
#while True:
#    startT = time.time()
#    whatsit = proxy.next_to_archive()
#    stopT = time.time()
#    print "queries %d %f" % (queryCnt, (stopT-startT))
#    time.sleep(0.1)
#    queryCnt += 1
experiments = proxy.next_to_archive()
print type(experiments)
for key in experiments.keys():
    print "key is %s" % key
    for exp in experiments[key]:
        print "%s %s" % (exp['name'],exp['storage_opt'])