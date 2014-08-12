#!/bin/bash
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
set -e
set -x

list=(
amp_exp100
amp_exp110
)
for item in ${list[@]}; do
    ./harvest_bam.py --destination /media/Data ${item}
done
