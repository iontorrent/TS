#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

USER=ion
SERVER=rnd1.ite
PUBDIR=lucid-alpha
PUBPATH=public_html/updates_server_root/updates/software/$PUBDIR
PKGFILE=$1

if [ $# -lt 1 ]
then
   echo "Need to specify a package file to publish."
   exit 1
fi

while [ $# -ne 0 ]; do

	PKGFILE=$1
    
    if [ ! -r "$PKGFILE" ]
    then
		echo "Invalid file specified: $PKGFILE"
    else
		echo "Copying $PKGFILE to server"
    	scp $PKGFILE $USER@$SERVER:$PUBPATH
    	if [ $? -ne 0 ]
    	then
    	   echo "There was an error copying $PKGFILE"
    	fi
    fi

    # Increment the arguments
    shift
    
done

echo "Writing new Packages.gz file"
ssh $USER@$SERVER "cd $PUBPATH/.. && rm -f $PUBDIR/Packages.gz && apt-ftparchive packages $PUBDIR | gzip > $PUBDIR/Packages.gz"
if [ $? -ne 0 ]
then
	echo "There was an error creating the Packages.gz file"
	exit 1
fi

exit 0
