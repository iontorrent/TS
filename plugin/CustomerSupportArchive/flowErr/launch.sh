#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# $Revision$
VERSION="1.1"

echo "ANALYSIS_DIR=$ANALYSIS_DIR"
echo "TSP_FILEPATH_PLUGIN_DIR=$TSP_FILEPATH_PLUGIN_DIR"
echo "TSP_FLOWORDER=$TSP_FLOWORDER"
echo "TSP_CHIPTYPE=$TSP_CHIPTYPE"
echo "HOSTNAME="`hostname`

export BC_JSON="$TSP_FILEPATH_PLUGIN_DIR/barcodes.json"

#export BC_JSON='barcodes.json'

IONSTATS_BAM_INPUT=`python <<END
import json
import os
with open(str(os.environ['BC_JSON'])) as f:
    data = json.load(f)
bam_files=set([v["bam_filepath"] for k,v in data.iteritems()])
IONSTATS_BAM_INPUT = ",".join(list(bam_files))
print  IONSTATS_BAM_INPUT
END`

echo "==================="
echo $IONSTATS_BAM_INPUT
echo "==================="


# Determin ionstats exe to use - use a local distro-specific one if availalbe, otherwise use system version
IONSTATS_EXE="ionstats"
if [[ -x "$DIRNAME/bin/ionstats" ]]
then
  IONSTATS_EXE="$DIRNAME/bin/ionstats"
fi
echo "IONSTATS_EXE=$IONSTATS_EXE"

# IONSTATS_BAM_INPUT=$(ls $ANALYSIS_DIR/*rawlib.bam | paste -s -d, )
IONSTATS_ERROR_SUMMARY="$TSP_FILEPATH_PLUGIN_DIR/ionstats_error_summary.h5"
IONSTATS_OPTIONS="--evaluate-hp --skip-rg-suffix .nomatch --n-flow $TSP_NUM_FLOWS --max-subregion-hp 6"

# Find multilane flags in explog.txt
line=$(sed -n -e '/LanesActive\(.*\)yes/p' "$ANALYSIS_DIR/explog.txt")
array=($line)
MSTRING=" "
for key in "${!array[@]}";
    do idxlane="${array[$key]:11:1}"
    # echo "$key $idxlane";
    MSTRING=$MSTRING$idxlane
done
MSTRING="$(echo `basename $ANALYSIS_DIR` | sed 's/.*ChipLane//' | tr -d _)"
echo $MSTRING
isThumbnail=true
isMultilane="false"
if [[ ${#array[@]} -lt 1 ]] 
then
    isMultilane=false
else
    isMultilane=true
fi

# debug
echo "is this run thumbnail?  $isThumbnail"
echo "is this run multilane?  $isMultilane"
echo ""${array[@]}""


# Determine the chip-dimension options based on chip type
IONSTATS_CHIP_DIM_OPTIONS=""
IONSTATS_CHIP_SIZE_X=0
IONSTATS_CHIP_SIZE_Y=0

if [[ $TSP_CHIPTYPE == R* || $TSP_CHIPTYPE == 52* ]]
then
  if [ -d "$ANALYSIS_DIR/block_X0_Y0" ]
  then
    IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 7680,1728 --subregion-dim 64,64"
  else
    IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 1200,800 --subregion-dim 50,50"
  fi
elif [[ $TSP_CHIPTYPE == *P0* || $TSP_CHIPTYPE == 53* ]]
then
  if ls $ANALYSIS_DIR/block_* 1> /dev/null 2>&1;
  then
    IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 7680,5312 --subregion-dim 64,64"
  else
    IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 1200,800 --subregion-dim 50,50"
  fi
elif [[ $TSP_CHIPTYPE == *P1* || $TSP_CHIPTYPE == 54* || $TSP_CHIPTYPE == GX5* ]]
then
  if ls $ANALYSIS_DIR/block_* 1> /dev/null 2>&1;
  then
    IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 15456,10656 --subregion-dim 184,148"
    if [ "$isMultilane" = true ]
    then
        IONSTATS_CHIP_DIM_OPTIONS="--chip-dim 3864,10656 --subregion-dim 184,148"
        IONSTATS_CHIP_SIZE_X=3864
        IONSTATS_CHIP_SIZE_Y=10656
    else
        IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 15456,10656 --subregion-dim 184,148"
        IONSTATS_CHIP_SIZE_X=1200
        IONSTATS_CHIP_SIZE_Y=800
    fi
  else
    if [ "$isMultilane" = true ]
        then
        IONSTATS_CHIP_DIM_OPTIONS="--chip-dim 300,800 --subregion-dim 50,50"
        IONSTATS_CHIP_SIZE_X=300
        IONSTATS_CHIP_SIZE_Y=800
    else
        IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 1200,800 --subregion-dim 50,50"
        IONSTATS_CHIP_SIZE_X=1200
        IONSTATS_CHIP_SIZE_Y=800
    fi
  fi
elif [[ $TSP_CHIPTYPE == *P2* || $TSP_CHIPTYPE == 55* ]]
then
  if ls $ANALYSIS_DIR/block_* 1> /dev/null 2>&1;
  then
    IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 30912,21312 --subregion-dim 368,296"
  else
    IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 1200,800 --subregion-dim 50,50"
  fi
elif [[ $TSP_CHIPTYPE == *318* ]]
then
  IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 3392,3792 --subregion-dim 53,48"
elif [[ $TSP_CHIPTYPE == *316v2* ]]
then
  IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 3392,2120 --subregion-dim 53,53"
elif [[ $TSP_CHIPTYPE == *314* ]]
then
  IONSTATS_CHIP_DIM_OPTIONS="--chip-origin 0,0 --chip-dim 1280,1152 --subregion-dim 40,48"
fi


space=" "
comma=","
if [ "$isMultilane" = true ]
    then
        for key in "${!array[@]}";
            do idxlane="${array[$key]:11:1}"
            echo "Lane --- "$idxlane
            # IONSTATS_BAM_INPUT=$(ls $ANALYSIS_DIR/*rawlib.bam | paste -s -d, )
            IONSTATS_CHIP_ORIGIN_X=$((IONSTATS_CHIP_SIZE_X * $(($idxlane - 1)) ))
            IONSTATS_CHIP_ORIGIN_Y=0
            IONSTATS_CHIP_ORIGIN_OPTIONS=" --chip-origin $IONSTATS_CHIP_ORIGIN_X$comma$IONSTATS_CHIP_ORIGIN_Y"
            echo $IONSTATS_CHIP_ORIGIN_OPTIONS
            IONSTATS_CHIP_DIM_OPTION="$IONSTATS_CHIP_DIM_OPTIONS$IONSTATS_CHIP_ORIGIN_OPTIONS"
            echo $IONSTATS_CHIP_DIM_OPTION
            COMMAND="$IONSTATS_EXE alignment $IONSTATS_CHIP_DIM_OPTION $IONSTATS_OPTIONS --output-h5 $IONSTATS_ERROR_SUMMARY -i $IONSTATS_BAM_INPUT"
            echo $COMMAND
            $COMMAND

            ANALYSIS_NAME=`basename $ANALYSIS_DIR`
            COMMAND="$DIRNAME/flowErr2.pl --analysis-name $ANALYSIS_NAME --error-summary-h5 $IONSTATS_ERROR_SUMMARY --floworder $TSP_FLOWORDER --out-dir $TSP_FILEPATH_PLUGIN_DIR --laneid $idxlane --multilane true"
            echo $COMMAND
            $COMMAND

            rm -rf $IONSTATS_ERROR_SUMMARY

            if [ ! -d "$SIGPROC_DIR/block_X0_Y0" ]
            then
                  IONSTATS_BAM_INPUT=$(ls $BASECALLER_DIR/unfiltered.trimmed/*rawlib.bam | paste -s -d, )
                  echo "bam file = "$IONSTATS_BAM_INPUT
                  if [[ ! "$IONSTATS_BAM_INPUT" == "" ]]; then
                    COMMAND="$IONSTATS_EXE alignment $IONSTATS_CHIP_DIM_OPTION $IONSTATS_OPTIONS --output-h5 $IONSTATS_ERROR_SUMMARY -i $IONSTATS_BAM_INPUT"
                    echo $COMMAND
                    $COMMAND

                    COMMAND="$DIRNAME/flowErr2.pl --analysis-name $ANALYSIS_NAME --error-summary-h5 $IONSTATS_ERROR_SUMMARY --floworder $TSP_FLOWORDER --out-dir $TSP_FILEPATH_PLUGIN_DIR --unfiltered 1  --laneid $idxlane --multilane true"
                    echo $COMMAND
                    $COMMAND

                    rm -rf $IONSTATS_ERROR_SUMMARY
                  fi
                  

                  IONSTATS_BAM_INPUT=$(ls $BASECALLER_DIR/unfiltered.untrimmed/*rawlib.bam | paste -s -d, )
                  if [[ ! "$IONSTATS_BAM_INPUT" == "" ]]; then
                    COMMAND="$IONSTATS_EXE alignment $IONSTATS_CHIP_DIM_OPTION $IONSTATS_OPTIONS --output-h5 $IONSTATS_ERROR_SUMMARY -i $IONSTATS_BAM_INPUT"
                    echo $COMMAND
                    $COMMAND

                    COMMAND="$DIRNAME/flowErr2.pl --analysis-name $ANALYSIS_NAME --error-summary-h5 $IONSTATS_ERROR_SUMMARY --floworder $TSP_FLOWORDER --out-dir $TSP_FILEPATH_PLUGIN_DIR --unfiltered 1 --untrimmed 1  --laneid $idxlane --multilane true"
                    echo $COMMAND
                    $COMMAND

                    rm -rf $IONSTATS_ERROR_SUMMARY
                fi
            fi
        done
else   # single lane 
    COMMAND="$IONSTATS_EXE alignment $IONSTATS_CHIP_DIM_OPTIONS $IONSTATS_OPTIONS --output-h5 $IONSTATS_ERROR_SUMMARY -i $IONSTATS_BAM_INPUT"
    echo $COMMAND
    $COMMAND

    ANALYSIS_NAME=`basename $ANALYSIS_DIR`
    COMMAND="$DIRNAME/flowErr2.pl --analysis-name $ANALYSIS_NAME --error-summary-h5 $IONSTATS_ERROR_SUMMARY --floworder $TSP_FLOWORDER --out-dir $TSP_FILEPATH_PLUGIN_DIR"
    echo $COMMAND
    $COMMAND

    rm -rf $IONSTATS_ERROR_SUMMARY

    if [ ! -d "$SIGPROC_DIR/block_X0_Y0" ]
    then
      IONSTATS_BAM_INPUT=$(ls $BASECALLER_DIR/unfiltered.trimmed/*rawlib.bam | paste -s -d, )
      COMMAND="$IONSTATS_EXE alignment $IONSTATS_CHIP_DIM_OPTIONS $IONSTATS_OPTIONS --output-h5 $IONSTATS_ERROR_SUMMARY -i $IONSTATS_BAM_INPUT"
      echo $COMMAND
      $COMMAND

      COMMAND="$DIRNAME/flowErr2.pl --analysis-name $ANALYSIS_NAME --error-summary-h5 $IONSTATS_ERROR_SUMMARY --floworder $TSP_FLOWORDER --out-dir $TSP_FILEPATH_PLUGIN_DIR --unfiltered 1"
      echo $COMMAND
      $COMMAND

      rm -rf $IONSTATS_ERROR_SUMMARY

      IONSTATS_BAM_INPUT=$(ls $BASECALLER_DIR/unfiltered.untrimmed/*rawlib.bam | paste -s -d, )
      COMMAND="$IONSTATS_EXE alignment $IONSTATS_CHIP_DIM_OPTIONS $IONSTATS_OPTIONS --output-h5 $IONSTATS_ERROR_SUMMARY -i $IONSTATS_BAM_INPUT"
      echo $COMMAND
      $COMMAND

      COMMAND="$DIRNAME/flowErr2.pl --analysis-name $ANALYSIS_NAME --error-summary-h5 $IONSTATS_ERROR_SUMMARY --floworder $TSP_FLOWORDER --out-dir $TSP_FILEPATH_PLUGIN_DIR --unfiltered 1 --untrimmed 1"
      echo $COMMAND
      $COMMAND

      rm -rf $IONSTATS_ERROR_SUMMARY
    fi

fi

