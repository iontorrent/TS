#!/bin/bash
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
# $Revision$
VERSION="0.4"

echo "ANALYSIS_DIR=$ANALYSIS_DIR"
echo "TSP_FILEPATH_PLUGIN_DIR=$TSP_FILEPATH_PLUGIN_DIR"
echo "TSP_FLOWORDER=$TSP_FLOWORDER"
echo "TSP_CHIPTYPE=$TSP_CHIPTYPE"
# echo "TSP_FILEPATH_BAM=$TSP_FILEPATH_BAM"
echo "HOSTNAME="`hostname`
echo "TSP_RUNID=$TSP_RUNID"
# export PYTHONPATH=$PYTHONPATH:$DIRNAME


if [ -e $SIGPROC_DIR/separator.spatial.h5 ];then
	echo "Running: $DIRNAME/chip_spatial_plot.py -o $TSP_FILEPATH_PLUGIN_DIR -p $TSP_RUNID $SIGPROC_DIR/separator.spatial.h5"
	python $DIRNAME/chip_spatial_plot.py -o $TSP_FILEPATH_PLUGIN_DIR -p $TSP_RUNID $SIGPROC_DIR/separator.spatial.h5
else
	echo "Running: $DIRNAME/chip_spatial_plot.py -o $TSP_FILEPATH_PLUGIN_DIR -p $TSP_RUNID $SIGPROC_DIR/block_*/separator.spatial.h5"
	python $DIRNAME/chip_spatial_plot.py -o $TSP_FILEPATH_PLUGIN_DIR -p $TSP_RUNID $SIGPROC_DIR/block_*/separator.spatial.h5
fi


cp -r $DIRNAME/assets $TSP_FILEPATH_PLUGIN_DIR
perl $DIRNAME/gather.pl $TSP_RUNID *.png > $TSP_FILEPATH_PLUGIN_DIR/separator_spatial.html
