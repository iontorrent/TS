# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# xmlrpc status tests
#
from twisted.web.xmlrpc import Proxy
import xmlrpclib
import time
from os import path
import sys

sys.path.append("/opt/ion/iondb")
import settings

# status of crawler
try:
    url = "http://127.0.0.1:" + str(settings.CRAWLER_PORT)
    proxy = xmlrpclib.ServerProxy(url)
    statusResult = proxy.state()
    # result is a two param list, first item is the status (sellping, processing, etc), second is the time.  If time greater than 60 its bad
    crawlerTime = int(statusResult[1])
    # print "Crawler time: %s" % crawlerTime
    if crawlerTime > 80:
        print "Crawler:error"
    else:
        print "Crawler:ok"
except:
    print "Crawler:offline"

# status of job server
try:
    url = "http://127.0.0.1:" + str(settings.JOBSERVER_PORT)
    proxy = xmlrpclib.ServerProxy(url)
    statusResult = proxy.uptime()
    upTime = int(statusResult)
    # print "Job Server up time: %s" % upTime
    if upTime < 0:
        print "Job:error"
    else:
        print "Job:ok"
except:
    print "Job:offline"

# status of archive server
try:
    url = "http://127.0.0.1:" + str(settings.IARCHIVE_PORT)
    proxy = xmlrpclib.ServerProxy(url)
    # archive server will return NULL when nothing is happening, so that can kick off an xml parse error, so we need to handle this valid case
    try:
        statusResult = proxy.next_to_archive()
        print "Archive:ok"
    except:
        print "Archive:ok"
except:
    print "Archive:offline"

# status of plugin server
try:
    url = "http://127.0.0.1:" + str(settings.IPLUGIN_PORT)
    proxy = xmlrpclib.ServerProxy(url)
    statusResult = proxy.uptime()
    upTime = int(statusResult)
    # print "Plugin Server up time: %s" % upTime
    if upTime < 0:
        print "Plugin:error"
    else:
        print "Plugin:ok"
except:
    print "Plugin:offline"

