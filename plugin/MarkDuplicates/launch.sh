#!/bin/bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved 

#MAJOR_BLOCK 

# SVN Version
VERSION="3.0.40641"

# Change the following line to all CAPS to disable auto-run of this plugin, but do not uncomment
#autorundisable

#as of 6/21 the stack size being set on SGE nodes is too large, setting manually to the default
ulimit -s 8192

# create environment variables from the JSON
makeVarsFromJson ${RUNINFO__ANALYSIS_DIR}/ion_params_00.json;

# Make sure it is empty
if [ -f ${TSP_FILEPATH_PLUGIN_DIR} ]; then
    run "rm -rf ${TSP_FILEPATH_PLUGIN_DIR}";
fi

# create a dummy block
if [ "true" == ${MARK_DUPLICATES} ]; then
  echo "INFO: Mark duplicates was on.";
  #cat ${TSP_FILEPATH_BAM}.markduplictes.metrics.txt > ${TSP_FILEPATH_PLUGIN_DIR}/MarkDuplicates_block.html;
  python ${DIRNAME}/mark_duplicates.py -m ${TSP_FILEPATH_BAM}.markduplictes.metrics.txt > ${TSP_FILEPATH_PLUGIN_DIR}/MarkDuplicates_block.html;
elif [ "false" == ${MARK_DUPLICATES} ]; then
  echo "INFO: Mark duplicates was off.";
else 
  echo "INFO: Mark duplicates was in an unknown state.";
fi

# dummy JSON
echo '{' > ${TSP_FILEPATH_PLUGIN_DIR}/results.json;
if [ -z "${TSP_FILEPATH_BARCODE_TXT}" -o ! -e "${TSP_FILEPATH_BARCODE_TXT}" ]; then
  echo "INFO: This is not a barcode run.";
else
  echo "INFO: This is a barcode run.";
fi
echo '}' >> ${TSP_FILEPATH_PLUGIN_DIR}/results.json;
