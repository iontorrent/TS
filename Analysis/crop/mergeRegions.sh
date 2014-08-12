#!/bin/sh
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

if [ "$#" -gt 0 ] ; then
	regionList=$1
else
	regionList=regions.list
fi

if [ "$#" -gt 1 ] ; then
	outOpt='-o '$2
fi

if [ "$#" -gt 2 ] ; then
	inOpt='-i '$3
else
	inOpt='-i CropRegions'
fi

if [ "$#" -gt 3 ] ; then
	flowOpt='-F '$4
fi

## the following line needs to be modified by the user 
buildDir=/rhome/ewang/IonSoftware_latest/TS/build/Analysis

if [ -f $inDir ]; then
	echo MergeRegions -l $regionList
	$buildDir/MergeRegions -l $regionList $outOpt $inOpt $flowOpt
else
	echo regionList does not exist: $regionList
	exit
fi
