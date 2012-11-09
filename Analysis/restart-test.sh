#!/bin/bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

# Test code for restarting background model.
# Usage:
#
#     restart-test raw-data-directory

data=$1
flws=40
blck=20
comm="--local-wells-file off --flowlimit $flws $data"
dir0=$flws
dir1=20+

mkdir $dir0
mkdir $dir1

#rm -rf $dir0/*
#rm -rf $dir1/*

(
    cd $dir0
    echo $dir0
    Analysis $comm > out
)

(
    cd $dir1
    nextDir=`printf "archive.%04d" 0`
    Analysis $comm --start-flow-plus-interval  0,$blck --restart-next $nextDir > out.0000

    for (( i=$blck; i<$flws; i=$i+$blck ))
    do
        fromDir=$nextDir
        nextDir=`printf "archive.%04d" $i`
        outFile=`printf "out.%04d" $i`
        echo $nextDir
        Analysis $comm --start-flow-plus-interval $i,$blck --restart-from $fromDir --restart-next $nextDir > $outFile
    done
)


