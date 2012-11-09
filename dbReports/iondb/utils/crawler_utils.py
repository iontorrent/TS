#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import os
import glob
import string
import datetime

from iondb.rundb import models
TIMESTAMP_RE = models.Experiment.PRETTY_PRINT_RE

def tdelt2secs(td):
    """Convert a ``datetime.timedelta`` object into a floating point
    number of seconds."""
    day_seconds = float(td.days*24*3600)
    ms_seconds = float(td.microseconds)/1000000.0
    return  day_seconds + float(td.seconds) + ms_seconds
    
def getFlowOrder(rawString):

    '''Parses out a nuke flow order string from entry in explog.txt of the form:
    rawString format:  "4 0 r4 1 r3 2 r2 3 r1" or "4 0 T 1 A 2 C 3 G".'''

    #Initialize return value
    flowOrder = ''
    
    #Capitalize all lowercase; strip leading and trailing whitespace
    rawString = string.upper(rawString).strip()
    
    # If there are no space characters, it is 'new' format
    if rawString.find(' ') == -1:
        flowOrder = rawString
    else:
        #Define translation table
        table = {
                "R1":'G',
                "R2":'C',
                "R3":'A',
                "R4":'T',
                "T":'T',
                "A":'A',
                "C":'C',
                "G":'G'}
        #Loop thru the tokenized rawString extracting the nukes in order and append to return string
        for c in rawString.split(" "):
            try:
                flowOrder += table[c]
            except KeyError:
                pass

    # Add a safeguard.
    if len(flowOrder) < 4:
        flowOrder = 'TACG'
        
    return flowOrder

def folder_mtime(folder):
    """Determine the time at which the experiment was performed. In order
    to do this reliably, ``folder_mtime`` tries the following approaches,
    and returns the result of the first successful approach:
    
    #. Parse the name of the folder, looking for a YYYY_MM_DD type
       timestamp
    #. ``stat()`` the folder and return the ``mtime``.
    """
    match = TIMESTAMP_RE.match(os.path.basename(folder))
    if match is not None:
        dt = datetime.datetime(*map(int,match.groups(1)))
        return dt
    else:
        acqfnames = glob.glob(os.path.join(folder, "*.acq"))
        if len(acqfnames) > 0:
            fname = acqfnames[0]
        else:
            fname = folder
        stat = os.stat(fname)
        return datetime.datetime.fromtimestamp(stat.st_mtime)