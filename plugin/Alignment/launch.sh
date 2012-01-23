#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
#AUTORUNDISABLE
#as of 6/21 the stack size being set on SGE nodes is too large, setting manually to the default
ulimit -s 8192
#normal plugin script
VERSION="0.2"
$DIRNAME/alignment.py startplugin.json
cp $DIRNAME/Alignment_block.php $RESULTS_DIR/
