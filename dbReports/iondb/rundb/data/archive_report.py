#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
#
# List all Reports and status of file categories, showing archive location
#
import sys
from iondb.bin import djangoinit
from iondb.rundb.models import Results
from iondb.rundb.data import dmactions_types

# Write the column headers
sys.stdout.write("Report Name," + ",".join(dmactions_types.FILESET_TYPES) + "\n")

# Get list of Result objects from database
results = Results.objects.all().order_by('timeStamp')

for result in results:
    sys.stdout.write(result.resultsName)
    # Get DMFileStat objects for this Report
    for dm_type in dmactions_types.FILESET_TYPES:
        dmfilestat = result.get_filestat(dm_type)
        sys.stdout.write(",")
        sys.stdout.write(str(dmfilestat.archivepath))
    print
