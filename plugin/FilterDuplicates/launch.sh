#!/bin/bash
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved 

#MAJOR_BLOCK 

VERSION="3.6.58660"

# Change the following line to all CAPS to disable auto-run of this plugin, but do not uncomment
#autorundisable

#as of 6/21 the stack size being set on SGE nodes is too large, setting manually to the default
ulimit -s 8192

# create environment variables from the JSON
makeVarsFromJson ${RUNINFO__ANALYSIS_DIR}/ion_params_00.json;

if [ -z "${TSP_FILEPATH_BARCODE_TXT}" -o ! -e "${TSP_FILEPATH_BARCODE_TXT}" ]; then
  echo "INFO: This is not a barcode run.";
  echo ${TSP_FILEPATH_BARCODE_TXT}; 
else
  echo ${TSP_FILEPATH_BARCODE_TXT};
fi

# Make sure it is empty
if [ -f ${TSP_FILEPATH_PLUGIN_DIR} ]; then
    run "rm -rf ${TSP_FILEPATH_PLUGIN_DIR}";
fi

# create a dummy block
echo "INFO: Mark duplicates was on.";
#cat ${TSP_FILEPATH_BAM}.markduplicates.metrics.txt > ${TSP_FILEPATH_PLUGIN_DIR}/MarkDuplicates_block.html;
python ${DIRNAME}/filterDuplicates.py -a $ANALYSIS_DIR -o $RESULTS_DIR -m Results.json 

