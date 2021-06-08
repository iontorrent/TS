#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# $Revision$
VERSION="0.3"

VERSION="0.2"
isThumbnail="false"
isMultilane="false"


# Find multilane flags in explog.txt
line=$(sed -n -e '/LanesActive\(.*\)yes/p' "$ANALYSIS_DIR/explog.txt")
array=($line)
MSTRING=" "
for key in "${!array[@]}";
    do idxlane="${array[$key]:11:1}"
    echo "$key $idxlane";
    MSTRING=$MSTRING$idxlane
done

if [[ ${#array[@]} -lt 1 ]] 
then
    isMultilane=false
else
    isMultilane=true
fi

# Check if the run is thumbnail or fullchip
if [ -d "$ANALYSIS_DIR/block_X0_Y0" ]
then
    isThumbnail=false
else
    isThumbnail=true
fi

# debug
echo "is this run thumbnail?  $isThumbnail"
echo "is this run multilane?  $isMultilane"
echo ""${array[@]}""

echo "ANALYSIS_DIR=$ANALYSIS_DIR"
echo "SIGPROC_DIR=$SIGPROC_DIR"
echo "BASECALLER_DIR=$BASECALLER_DIR"
echo "ALIGNMENT_DIR=$ALIGNMENT_DIR"

echo "TSP_FILEPATH_PLUGIN_DIR=$TSP_FILEPATH_PLUGIN_DIR"
echo "TSP_LIBRARY_KEY=$TSP_LIBRARY_KEY"
echo "TSP_FLOWORDER=$TSP_FLOWORDER"



if [ "$isMultilane" = true ]
then
        echo "$DIRNAME/rawPlotsML.pl --analysis-dir $ANALYSIS_DIR --alignment-dir $ALIGNMENT_DIR --chip-type $TSP_CHIPTYPE"
        $DIRNAME/rawPlotsML.pl \
        --analysis-dir $ANALYSIS_DIR \
        --sigproc-dir $SIGPROC_DIR \
        --basecaller-dir $BASECALLER_DIR \
        --alignment-dir $ALIGNMENT_DIR \
        --raw-dir $RAW_DATA_DIR \
        --chip-type $TSP_CHIPTYPE \
        --thumbnail $isThumbnail \
        --out-dir $TSP_FILEPATH_PLUGIN_DIR \
        --active-lanes $MSTRING

else
    echo "$DIRNAME/rawPlots.pl --analysis-dir $ANALYSIS_DIR  --sigproc-dir $SIGPROC_DIR --basecaller-dir $BASECALLER_DIR  --alignment-dir $ALIGNMENT_DIR  --raw-dir $RAW_DATA_DIR  --chip-type $TSP_CHIPTYPE  --out-dir $TSP_FILEPATH_PLUGIN_DIR"

    $DIRNAME/rawPlots.pl \
      --analysis-dir $ANALYSIS_DIR \
      --sigproc-dir $SIGPROC_DIR \
      --basecaller-dir $BASECALLER_DIR \
      --alignment-dir $ALIGNMENT_DIR \
      --raw-dir $RAW_DATA_DIR \
      --chip-type $TSP_CHIPTYPE \
      --out-dir $TSP_FILEPATH_PLUGIN_DIR
fi