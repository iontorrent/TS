#!/bin/bash
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

#MAJOR_BLOCK
#AUTORUNDISABLED

#as of 6/21 the stack size being set on SGE nodes is too large, setting manually to the default
#ulimit -s 8192
#$ -l mem_free=22G,h_vmem=22G,s_vmem=22G
#normal plugin script
VERSION="5.4.0.46"

echo ${DIRNAME}/variant_caller_plugin.py \
  --install-dir   ${DIRNAME} \
  --output-dir    ${TSP_FILEPATH_PLUGIN_DIR} \
  --output-url    ${TSP_URLPATH_PLUGIN_DIR} \
  --report-dir    ${ANALYSIS_DIR}

echo 'running launch.sh'
${DIRNAME}/variant_caller_plugin.py \
  --install-dir   ${DIRNAME} \
  --output-dir    ${TSP_FILEPATH_PLUGIN_DIR} \
  --output-url    ${TSP_URLPATH_PLUGIN_DIR} \
  --report-dir    ${ANALYSIS_DIR}
