#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
import os
import re
import socket
from glob import glob
import zipfile
patterns = [
    "/etc/torrentserver/tsconf.conf",
    "/opt/sge/iontorrent/spool/master/messages",
    "/usr/share/ion-tsconfig/mint-config/*",
    "/var/log/apache2/access.log",
    "/var/log/apache2/access.log.1.gz",
    "/var/log/apache2/error.log",
    "/var/log/apache2/error.log.1.gz",
    "/var/log/ion/crawl.log",
    "/var/log/ion/crawl.log.1",
    "/var/log/ion/django.log",
    "/var/log/ion/django.log.1.gz",
    "/var/log/ion/celery_*.log",
    "/var/log/ion/celery_*.log.1.gz",
    "/var/log/ion/celerybeat.log",
    "/var/log/ion/celerybeat.log.1.gz",
    "/var/log/ion/ionPlugin.log",
    "/var/log/ion/ionPlugin.log.1",
    "/var/log/ion/jobserver.log",
    "/var/log/ion/jobserver.log.1",
    "/var/log/ion/tsconfig_*.log",
    "/var/log/ion/tsconfig_*.log.1",
    "/var/log/ion/tsconf.log",
    "/var/log/ion/RSMAgent.log",
    "/var/log/ion/data_management.log",
    "/var/log/ion/data_management.log.1.gz",
    "/var/log/kern.log",
    "/var/log/postgresql/postgresql-8.4-main.log",
    "/var/log/syslog",
    "/tmp/stats_sys.txt",
]

NATURAL_SORT_PATTERN = re.compile(r'(\d+|\D+)')

def natsort_key(s):
    return [int(s) if s.isdigit() else s for s in NATURAL_SORT_PATTERN.findall(s)]


def get_servicetag():
    '''Return serialnumber from tsconf.conf.  Else return hostname.
    '''
    try:
        with open("/etc/torrentserver/tsconf.conf") as conf:
            for l in conf:
                if l.startswith("serialnumber:"):
                    servicetag = l[len("serialnumber:"):].strip()
                    break
    except IOError:
        servicetag = socket.gethostname()
    except:
        servicetag = "0"
    return servicetag


def makeSSA():
    servicetag = get_servicetag()
    archive_name = "%s_systemStats.zip" % servicetag
    archive_path = os.path.join("/tmp", archive_name)
    ssazip = zipfile.ZipFile(archive_path, mode='w',
        compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    for pattern in patterns:
        files = glob(pattern)
        files.sort(key=natsort_key)
        for filename in files:
            try:
                ssazip.write(filename, os.path.basename(filename))
            except:
                pass
    ssazip.close()
    return archive_path, archive_name
