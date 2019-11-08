# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
import sys


def textToDict(text):
    pkDict = {}
    # Get Headings for Process Params
    for line in text:
        if "=" in line:
            try:
                parsline = line.strip().split("=")
                key = parsline[0].strip()
                value = parsline[1].strip()
                pkDict[key] = value
            except IndexError:
                pass

    return pkDict


def fileToDict(fileIn):

    if not os.path.exists(fileIn):
        return None

    f = open(fileIn, "r")
    text = f.readlines()
    f.close()
    pkDict = textToDict(text)
    return pkDict


if __name__ == "__main__":

    pkDict = fileToDict(sys.argv[1])
    print(pkDict)
