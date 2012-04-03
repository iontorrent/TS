#!/usr/bin/env python
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Author: Daniel Cuevas

import call_api as api
import sys
import json
import os.path


# This is the API UID for this publisher
pub_uid = "/rundb/api/v1/publisher/BED/"

# Usage function
def usage():
    print """
submitBeds.py <ID> <DIRECTORY> <UPLOAD_BED> <META_FILE>
Required args:
    <ID>              : Process ID number
    <DIRECTORY>       : Given directory for Publisher
    <UPLOAD_BED>      : BED file to validate
    <META_FILE>       : Metadata JSON file
    """
    sys.exit(1)

# Check input is correct
if len(sys.argv) != 5:
    usage()

id = str(sys.argv[1])
directory = str(sys.argv[2])
bedFile = str(sys.argv[3])
metaFile = str(sys.argv[4])

upload_uid = "/rundb/api/v1/contentupload/%s/" % id

# Open up metaFile to get ref name for pathname purposes
try:
    fh = open(metaFile, "r")
except IOError:
    print >> sys.stderr, "Couldn't open " + metaFile
    sys.exit(1)
line = fh.read().rstrip()
meta = json.loads(line)
ref = meta["reference"]
# Get bed file name without directory path
bedFName = bedFile.split("/").pop()

# File paths to call into publisher
pbed = ref+"/unmerged/plain/"+bedFName
dbed = ref+"/unmerged/detail/"+bedFName
mpbed = ref+"/merged/plain/"+bedFName
mdbed = ref+"/merged/detail/"+bedFName

def register(file):
    full_path = os.path.join(directory, file)
    reg = "/%s" % file
    api.post("content", publisher=pub_uid, meta=line, file=full_path, path=reg, contentupload=upload_uid)

# Register files to Publisher
register(pbed)
register(dbed)
register(mpbed)
register(mdbed)
