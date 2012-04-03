#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
source version

which equivs-build 1>/dev/null
if [ $? -eq 1 ]; then
	echo
	echo "ERROR. equivs-build is not installed"
    echo "Try this to install:"
    echo "     sudo apt-get install equivs"
    echo
    exit 1
fi

which dch 1>/dev/null
if [ $? -eq 1 ]; then
	echo
	echo "ERROR. debchange is not installed"
    echo "Try this to install:"
    echo "     sudo apt-get install devscripts"
    echo
    exit 1
fi

# auto-increment build number.
#RELEASE=$(($RELEASE+1))
#sed -i "s/^RELEASE.*/RELEASE=$RELEASE/" version
#svn commit version -m"Candidate release build $MAJOR.$MINOR-$RELEASE"
#svn update

dch -b -v $MAJOR.$MINOR-$RELEASE
equivs-build debian/control

exit 0
