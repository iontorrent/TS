#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Change the following line to all CAPS to disable auto-run of this plugin, but do not uncomment
#autorundisable

VERSION="0"

COMMANDLINE="$0 $@"
DIRNAME=`dirname $0`
PROGRAM=`basename $0`

failure () {
	echo $1
	echo "Exiting $PROGRAM early"
    exit 1
}

# ======================================
# Parse Command Line Arguments
# ======================================
export RAW_DATA_DIR=""
export ANALYSIS_DIR=""
export LIBRARY_KEY=""
export TESTFRAG_KEY=""
export RESULTS_DIR=""
export URL_ROOT=""
while getopts ":vr:a:l:t:o:p:u:" opt; do
    case $opt in
        v  )
            echo $VERSION
            exit 0
            ;;
        r  )
            RAW_DATA_DIR=$OPTARG
            ;;
        a  )
            ANALYSIS_DIR=$OPTARG
            ;;
        l  )
            LIBRARY_KEY=$OPTARG
            ;;
        t  )
            TESTFRAG_KEY=$OPTARG
            ;;
        o  )
            RESULTS_DIR=$OPTARG
            ;;
		p  )
	    	DIRNAME=$OPTARG
	    	;;
        u  )
        	URL_ROOT=$OPTARG
            ;;
        \? )
            echo "usage: $PROGRAM ..."
            echo "arguments:"
            echo "  -v                  print the version and exit"
            echo "  -r RAW_DATA_DIR     specify the location of the raw data"
            echo "  -a ANALYSIS_DIR     specify the location of the analysis results"
            echo "  -l LIBRARY_KEY      specify the library key"
            echo "  -t TESTFRAG_KEY     specify the testfrag key"
            echo "  -o RESULTS_DIR      specify the plugin result folder"
	    	echo "  -p DIRNAME          specify the plugin directory"
            echo "  -u URL_ROOT         specify the URL root of the results directory"
            exit 0
            ;;
    esac
done
shift $(($OPTIND - 1))

# ========================
# Do some input validation
# ========================
if [ ! -d ${RAW_DATA_DIR} ]
then
	failure "Raw data directory: ${RAW_DATA_DIR}. Invalid"
elif [ ! -r ${RAW_DATA_DIR} ]
then
	failure "Raw data directory: ${RAW_DATA_DIR}. Not readable"
fi

if [ ! -d ${ANALYSIS_DIR} ]
then
	failure "Analysis directory: ${ANALYSIS_DIR}. Invalid"
elif [ ! -r ${ANALYSIS_DIR} ]
then
	failure "Analysis directory: ${ANALYSIS_DIR}. Not readable"
fi

if [ ! -d ${RESULTS_DIR} ]
then
	failure "Results directory: ${RESULTS_DIR}. Invalid"
elif [ ! -w ${RESULTS_DIR} ]
then
	failure "Results directory: ${RESULTS_DIR}. Not writeable"
fi

echo "version=$VERSION"
echo "start time=`date`"
echo "command line=$COMMANDLINE"
echo "$PROGRAM: starting execution of plugin code"
# ===================================================
# Add plugin specific code to execute below this line
# ===================================================

python $DIRNAME/printhtml.py ${RESULTS_DIR}
if [ "$?" -ne 0 ]; then
    failure "command printhtml.py failed."
fi

# ===================================================
# Normal exit value is 0
# ===================================================
echo "$PROGRAM: completing execution of plugin code"
echo "end time=`date`"

exit 0
