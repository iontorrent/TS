#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

CPPCHECK=`which cppcheck`
if [ $? != 0 ]; then
    sudo apt-get update
    sudo apt-get install -y cppcheck
fi

if [ "$@" ]; then
    DIRS=$@
else
    DIRS=`ls -d * | grep -v external`
fi

LOC=`dirname $0`

cppcheck --enable=all --error-exitcode=1 --quiet --suppressions $LOC/cppcheck-suppressions.txt \
    --template '{id}:{file}:{line} ({severity}) {message}' \
    $DIRS

if [ $? != 0 ]; then
    echo "               " >&2 
    echo "====================cppcheck failed on source tree======================" >&2
    echo "               " >&2 
    echo " Please correct this problem ASAP, or revert changes.                " >&2 
    echo "               " >&2 
    echo " run ./buildTools/bamboo-cppcheck.sh to do this check locally" >&2
    echo " expected warnings can be added to buildTools/cppcheck-suppressions.txt" >&2
    echo "               " >&2 
    echo "               " >&2 
    echo " *TIP* you can save time by running cppcheck on a single module:  " >&2
    echo "                                                  " >&2 
    echo "          ./buildTools/bamboo-cppcheck.sh Analysis" >&2
    echo "                                                  " >&2 
    echo "========================================================================" >&2
    exit 1;
fi
exit 0



