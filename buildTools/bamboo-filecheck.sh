#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

find Analysis alignTools buildTools dbReports plugin torrentR onetouchupdate tsconfig -type f | \
    grep -v '\.svn' | grep -v '__init__.py' | grep -v jsonencode.py | \
    xargs buildTools/fileChecker.pl

if [ $? != 0 ]; then
    echo "filecheck failed on source tree" >&2
    echo "run ./buildTools/bamboo-filecheck.sh to do this check locally" >&2
    exit 1;
fi
exit 0

