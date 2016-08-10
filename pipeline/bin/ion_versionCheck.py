#!/usr/bin/python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import socket
from ion import version
import sys
import os

# provide a way to remove ion from the front of a package string name
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-i", "--ion", dest="ion",
                  action="store_true", default=False,
                  help="Remove ion- from the start of a string", metavar="ion")

(options, args) = parser.parse_args()

sys.path.append('/opt/ion/')

os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

from ion.utils.TSversion import findVersions

ret, meta = findVersions()

print "Torrent_Suite=" + version
print "host=" + socket.gethostname()

for package, version in ret.iteritems():
    if not options.ion:
        package = package.replace("ion-", "")
    print "%s=%s" % (package, version)
