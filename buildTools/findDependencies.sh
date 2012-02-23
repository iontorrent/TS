#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Use this script (from the top level folder) to find
# library dependencies in the various packages

FOUNDMODULES=`find build -maxdepth 1 -mindepth 1 -type d`
MODULES=${MODULES-$FOUNDMODULES}

for M in $MODULES; do
	echo $M:
	FILES=`find $M -maxdepth 1 | xargs file | grep ELF | sed 's/:.*//'`
	if [ "$FILES" ]; then
		( for f in $FILES; do ldd $f; done;) | \
			grep '=>' | sed 's/ =>.*//' | sed 's/\t//' | \
			grep -v vdso | sort | uniq | xargs -n1 dpkg -S | \
			sed 's/:.*//' | sort | uniq | perl -pe s'/\n/, /' | \
			perl -pe s'/$/\n/'
	fi
done


