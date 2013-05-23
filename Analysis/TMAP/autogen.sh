#!/bin/sh
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

run ()
{
	echo "running: $*"
	eval $*

	if test $? != 0 ; then
		echo "error: while running '$*'"
		exit 1
	fi
}

run aclocal
run autoheader
run automake -a
run autoconf
