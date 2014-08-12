#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved



# please convert this plugin to use the newer method without launch.sh
# so that this plugin version label will update itself automatically
VERSION="4.2-r87667"




#AUTORUNDISABLE

if ! which python; then
    echo "Error: Could not find python executable"
    exit 1
fi

rm -f ${RESULTS_DIR}/${PLUGINNAME}_block.html
rm -f ${RESULTS_DIR}/leaderboard.html
python $DIRNAME/run_recognition_plugin.py $DIRNAME ${RESULTS_DIR}
rm ${RESULTS_DIR}/startplugin.json
