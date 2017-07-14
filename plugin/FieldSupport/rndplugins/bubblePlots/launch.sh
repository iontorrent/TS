#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# $Revision$
VERSION="0.2"

echo "SIGPROC_DIR=$SIGPROC_DIR"

# Some blackbird-thumbnail specific stuff
if [ $TSP_CHIPTYPE = "900" ] || [[ "$TSP_CHIPTYPE" == P* ]]; then
  if [ ${RAW_DATA_DIR##*/} != "thumbnail" ]; then
    RAW_DATA_DIR="$RAW_DATA_DIR/thumbnail"
  fi
fi
echo "RAW_DATA_DIR=$RAW_DATA_DIR"

$DIRNAME/bubblePlots.pl \
  --sigproc-dir $SIGPROC_DIR \
  --raw-dir $RAW_DATA_DIR \
  --analysis-name $TSP_ANALYSIS_NAME \
  --out-dir $TSP_FILEPATH_PLUGIN_DIR

