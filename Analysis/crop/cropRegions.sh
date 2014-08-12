#!/bin/sh
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
if [ "$#" -gt 0 ] ; then
	inDir=$1
else
	inDir=$PWD/input
fi

if [ "$#" -gt 1 ] ; then
	outDir=$2
else
	outDir=$PWD
fi

if [ "$#" -gt 2 ] ; then
	nX=$3
else
	nX=3
fi

if [ "$#" -gt 3 ] ; then
	kernX=$4
else
	kernX=104
fi

if [ "$#" -gt 4 ] ; then
	chipOpt='-t '$5
fi

if [ "$#" -gt 5 ] ; then
	flowOpt='-F '$6
fi

## the following line needs to be modified by the user 
buildDir=/rhome/ewang/IonSoftware_latest/TS/build/Analysis
if [ -d $inDir ]; then
	echo CropRegions -x $nX -y $nX -X $kernX -Y $kernX -i $inDir -o $outDir $chipOpt $flowOpt
	$buildDir/CropRegions -x $nX -y $nX -X $kernX -Y $kernX -i $inDir -o $outDir $chipOpt $flowOpt
else
	echo input directory does not exist: $inDir
	exit
fi
