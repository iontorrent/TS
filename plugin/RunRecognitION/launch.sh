#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

VERSION="1.1RC1"
#AUTORUNDISABLE

if ! which python; then
    echo "Error: Could not find python executable"
    exit 1
fi

if [ -e ${RESULTS_DIR}/${PLUGINNAME}_block.html ]; then
    rm -f ${RESULTS_DIR}/${PLUGINNAME}_block.html
fi

if [ -e ${RESULTS_DIR}/leaderboard.html ];  then
    rm -f ${RESULTS_DIR}/leaderboard.html
fi

python $DIRNAME/run_recognition_plugin.py $DIRNAME ${RESULTS_DIR}

if [ -e ${RESULTS_DIR}/startplugin.json ]; then
    rm ${RESULTS_DIR}/startplugin.json
fi