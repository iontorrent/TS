#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

ERR=0
# Rsync over data and then run comparison program.
if [ -e /home/csugnet/public_html/analysis-integration/cropped/B4_231_cropped/acq_0001.dat ]; then
    if [ ! -e cropped/B4_231_cropped/acq_0001.dat ]; then
    ln -s /home/csugnet/public_html/analysis-integration/cropped/ .
    ln -s /home/csugnet/public_html/analysis-integration/gold .
    fi
else
    echo "Syncing cropped data."
    rsync --rsh=ssh -az rnd2:/home/csugnet/public_html/analysis-integration/cropped .
    echo "Syncing gold data."
    rsync --rsh=ssh -az rnd2:/home/csugnet/public_html/analysis-integration/gold .
fi 
if [ "$?" != 0 ]; then
    echo "Couldn't rsync cropped data."
    exit 1
fi

if [ "$?" != 0 ]; then
    echo "Couldn't rsync gold results."
    exit 1
fi
mkdir testing
echo "Running and comparing data"
build/Analysis/AnalysisIntegrationTest  --gtest_output=xml:./ --analysis-test-config buildTools/analysis-integration.tab --prefix `pwd`

if [ "$?" != 0 ]; then
    ERR=$(($ERR + 1))
fi
if [ $ERR != 0 ]; then
    echo -e $ERRMSG
    echo "FAILURES: $ERR integration failues."
    exit $ERR
else
    echo "SUCCESS: All Analysis integration tests passed."
fi

