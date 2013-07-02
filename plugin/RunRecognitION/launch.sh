#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

VERSION="3.6.63335"
#AUTORUNDISABLE

if ! which python; then
    echo "Error: Could not find python executable"
    exit 1
fi

rm -f ${RESULTS_DIR}/${PLUGINNAME}_block.html
rm -f ${RESULTS_DIR}/leaderboard.html
python $DIRNAME/run_recognition_plugin.py $DIRNAME ${RESULTS_DIR}
rm ${RESULTS_DIR}/startplugin.json
