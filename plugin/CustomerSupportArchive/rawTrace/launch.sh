#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# $Revision$
VERSION="0.3"
isThumbnail="false"
isMultilane="false"


# Some proton-thumbnail specific stuff
if [ $TSP_CHIPTYPE = "900" ] || [[ "$TSP_CHIPTYPE" == P* ]]
then
  # This plugin only takes raw data for thumbnails for proton
  if [ ${RAW_DATA_DIR##*/} != "thumbnail" ]
  then
    RAW_DATA_DIR="$RAW_DATA_DIR/thumbnail"
  fi
fi



# Find multilane flags in explog.txt



isMultilane=true

MSTRING="$(echo `basename $ANALYSIS_DIR` | sed 's/.*ChipLane//' | tr -d _)"

echo $MSTRING

# debug
echo "is this run thumbnail?  $isThumbnail"
echo "is this run multilane?  $isMultilane"


echo "SIGPROC_DIR=$SIGPROC_DIR"
echo "TSP_FILEPATH_PLUGIN_DIR=$TSP_FILEPATH_PLUGIN_DIR"
echo "TSP_LIBRARY_KEY=$TSP_LIBRARY_KEY"
echo "TSP_FLOWORDER=$TSP_FLOWORDER"


if [ "$isMultilane" = true ]
then
        $DIRNAME/rawTraceML.pl \
        --analysis-dir $ANALYSIS_DIR \
        --lib-key $TSP_LIBRARY_KEY \
        --floworder $TSP_FLOWORDER \
        --chip-type $TSP_CHIPTYPE \
        --thumbnail $isThumbnail \
        --out-dir $TSP_FILEPATH_PLUGIN_DIR \
        --active-lanes $MSTRING
      # R --no-save --slave < $DIRNAME/flow-by-flow.R
      # cp $DIRNAME/flow-by-flow.html $TSP_FILEPATH_PLUGIN_DIR

else
    if [ -d "$SIGPROC_DIR/NucStep" ]
    then
      # Latest version is fast as it uses
      # prepared data dumped into NucStep dir
      $DIRNAME/rawTrace.pl \
        --analysis-dir $SIGPROC_DIR \
        --lib-key $TSP_LIBRARY_KEY \
        --floworder $TSP_FLOWORDER \
        --chip-type $TSP_CHIPTYPE \
        --out-dir $TSP_FILEPATH_PLUGIN_DIR
      #R --no-save --slave < $DIRNAME/flow-by-flow.R $SIGPROC_DIR &
      #cp $DIRNAME/flow-by-flow.html $TSP_FILEPATH_PLUGIN_DIR
    else
      # Legacy mode, for TS2.0 and before.
      # Much slower as all DATs are read.
      echo "RAW_DATA_DIR=$RAW_DATA_DIR"
      $DIRNAME/rawTrace_2.0.pl \
        --analysis-dir $SIGPROC_DIR \
        --raw-dir $RAW_DATA_DIR \
        --lib-key $TSP_LIBRARY_KEY \
        --floworder $TSP_FLOWORDER \
        --chip-type $TSP_CHIPTYPE \
        --bam $TSP_FILEPATH_BAM \
        --out-dir $TSP_FILEPATH_PLUGIN_DIR
    fi
fi
