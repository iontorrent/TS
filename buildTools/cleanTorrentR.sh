#!/bin/sh
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Quick script to clean the torrentR binaries as R is not aware of all dependencies
find ../../torrentR/src/ -name '*o' | xargs rm -f
