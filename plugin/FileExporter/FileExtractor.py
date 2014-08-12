#!/usr/bin/python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import sys
import zipfile

if __name__=="__main__":
    argc = len(sys.argv)
    if argc != 2:
        print 'Usage: FileExtractor InputFile.zip'
        quit()

    inputFile = sys.argv[1]
    print 'Extracting all files from zip file: %s' % inputFile

    z = zipfile.ZipFile(inputFile, 'r')
    z.extractall()

    print 'Completed.'

