#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

################################################################################
##
##	Reports internal number crunchers' software versions
##
################################################################################

echo -e "Torrent_Suite=`python -c 'from ion import version;print version'`"
echo -e "host=`hostname`"
#call the Python script, and pass the first arg along, --ion should be provided to add ion to the front of package names
python /opt/ion/iondb/bin/versionCheck.py $1

exit 0
