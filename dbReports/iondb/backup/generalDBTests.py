#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
import os
from os import path
import sys
import commands
import statvfs
import datetime

import iondb.bin.djangoinit
from iondb.rundb import models
from django import shortcuts

import json
from iondb.rundb import json_field

arg0 = False
arg1 = False

metaDataLog = []
ret = shortcuts.get_object_or_404(models.Results, pk=1)


def test():
    update_metaData("Nothing", "Created")
    update_metaData("Exporting", "Exporting to <path>")
    update_metaData("Nothing", "Finished exporting")
    update_metaData("Archiving", "Archiving to <path>")
    update_metaData("Archived", "Finished archiving")

    print "\n", getMetaDataLog()

'''
These methods will normally be in the Results class in models as update_metaData(self, status, info). (And they are, in this repository)
But the results that I'm grabbing from my db is using 2.2, so changes I make to the models file don't apply to it.
In any case, the moral of the story is that keeping a list of old dictionaries should work fine.
Adding more keys won't take any time at all, too.
'''


def update_metaData(status, info):
        ret.metaData["Status"] = status
        ret.metaData["Date"] = "%s" % datetime.datetime.now()
        ret.metaData["Info"] = info
        metaDataLog.append({"Status": ret.metaData.get("Status"), "Date": ret.metaData.get("Date"), "Info": ret.metaData.get("Info")})


def getMetaDataLog():
        logVal = ""
        for datum in metaDataLog:
            logVal += json.dumps(datum) + "\n"
        return logVal

if __name__ == '__main__':
    test()
