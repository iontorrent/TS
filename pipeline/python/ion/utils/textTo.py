# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import os

def textToDict(file):

    if not os.path.exists(file):
        return None

    f = open(file, 'r')
    pk = f.readlines()
    pkDict = {}
    for line in pk:
        try:
            parsline = line.strip().split("=")
            key = parsline[0].strip()
            value = parsline[1].strip()
            pkDict[key]=value
        except IndexError:
            pass
    f.close()
    return pkDict
