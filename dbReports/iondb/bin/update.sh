#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

#This script is called from Torrent Browser to update its own packages.

OUT=/tmp/django-update
UPDATE_STATUS=/tmp/django-update-status

#create a lock file
echo "locked" > $UPDATE_STATUS

echo "=============================" > $OUT
echo "Starting Torrent Suite Update" >> $OUT
echo "=============================" >> $OUT
sudo apt-get update >> $OUT
sudo apt-get -y --force-yes install ion-tsconfig >> $OUT
sudo TSconfig -s >> $OUT

if [ "$?" = "0" ]; then
    echo "=============================" >> $OUT
    echo "Torrent Suite Update is done"  >> $OUT
    echo "=============================" >> $OUT
else
    echo "=============================" >> $OUT
    echo "Torrent Suite Update FAILED!!" >> $OUT
    echo "=============================" >> $OUT
fi


#delete the lock
rm $UPDATE_STATUS
