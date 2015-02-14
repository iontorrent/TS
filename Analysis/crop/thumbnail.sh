#!/bin/sh
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
## thumbnail.sh inDir nX kernX chip outDir
## thumbnail.sh chip 4 104 314 tn_4x104

## remove tmp files
find /tmp -maxdepth 1 -type d -user $USER -exec rm -fvr {} \;

if [ "$#" -gt 0 ] ; then
	inDir=$1
else
	inDir=$PWD/input
fi

if [ "$#" -gt 1 ] ; then
	nX=$2
else
	nX=2
fi

if [ "$#" -gt 2 ] ; then
	kernX=$3
else
	kernX=64
fi

if [ "$#" -gt 3 ] ; then
	chip=$4
fi

if [ "$#" -gt 4 ] ; then
	outDir=$5
else
	outDir=$PWD
fi

if [ "$#" -gt 5 ] ; then
	flowlimit=$6
fi

## the following line needs to be modified by the user 
codeDir=~/IonSoftware_latest/TS/Analysis-build

if [ -d $inDir ]; then	
	##Thumbnail.cpp
	echo ================================================================================
	echo Thumbnail -i $inDir -x $nX -y $nX -X $kernX -Y $kernX -t $chip -o $outDir 
	$codeDir/Thumbnail -i $inDir -x $nX -y $nX -X $kernX -Y $kernX -t $chip -o $outDir 
else
	echo input directory does not exist: $inDir
	exit
fi
