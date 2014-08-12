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
	nX=3
fi

if [ "$#" -gt 2 ] ; then
	kernX=$3
else
	kernX=104
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
codeDir=/rhome/ewang/IonSoftware_latest/TS/Analysis/crop

if [ -d $inDir ]; then
	## cropRegions
	echo
	echo ====================================================================================
	echo cropRegions $inDir $outDir $nX $kernX $chip $flowlimit
	echo
	sh $codeDir/cropRegions.sh $inDir $outDir $nX $kernX $chip $flowlimit
	
	## make regionList
	regionList=$outDir/regions.list
	find $outDir -maxdepth 3 -type d -regex ".*x[0-9]*_y[0-9]*" | sort > $regionList
	
	## mergeRegions
	echo
	echo ====================================================================================
	## mergeRegions tn_4x104/regions.list tn_4x104 chip $flowlimit
	echo mergeRegions $regionList $outDir $inDir $flowlimit
	echo
	sh $codeDir/mergeRegions.sh $regionList $outDir $inDir $flowlimit
	
	## push the directory up one level
	if [ ! -d $outDir/thumbnail ]; then
		mkdir $outDir/thumbnail
	fi
	rm -f $outDir/thumbnail/*
	cp -f $outDir/MergeRegions/final_thumbnail/* $outDir/thumbnail/
	echo copied thumbnail from $outDir/MergeRegions/final_thumbnail to $outDir/thumbnail
else
	echo input directory does not exist: $inDir
	exit
fi
