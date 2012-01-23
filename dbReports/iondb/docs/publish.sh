#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

BUILDDIR=./.build/html/*
OUTDIR=/var/www/dev/docs

scp -r $BUILDDIR ion@ecto3:$OUTDIR
