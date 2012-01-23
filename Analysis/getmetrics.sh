#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Generate summary of run data for analysis folders in current directory.
# Format results as csv.

# put results here:
outfile="run_stats.csv"

# need some temp files:
tmp_old=/tmp/tmp_old.$RANDOM
tmp_new=/tmp/tmp_new.$RANDOM

# does $outfile already exist?
if [ -f $outfile ]; then
	# if yes, then will append results from latest analyses to existing file.
	# make a list of analyses already included in existing file:
	perl -a -F\, -ne 'next if $.==1; $F[2]=~s/\"//g; print "$F[2]\n"' $outfile | sort > $tmp_old
else
	# if not, then will generate new file, with column headers:
	(
	echo -n "expdate,exptime,andir,anname,cycles,project,sample,library,machine,";
	echo -n "analysis_vers,alignment_vers,dbreports_vers,sigproc3_vers,";
	echo -n "nwashout,nbead,ndud,nambig,nlive,ntf,nlib,";
	echo -n "coverage,meanlen,longest,";
	echo -n "nread,50Q10,50Q17,50Q20,100Q10,100Q17,100Q20,200Q10,200Q17,200Q20,";
	echo -n "cfscore,iescore,drscore"
	echo
	) >> $outfile

	echo >> $tmp_old
fi

# make a list of all analyses currently on disk:
ls | sort > $tmp_new

# find path to perl script that extract metrics:
plScript=`echo $0 | sed -e 's/sh$/pl/'`

# loop over all analyses not already included in existing $outfile:
for d in `comm -1 -3 $tmp_old $tmp_new`; do
	if [ -d $d ]; then
		 (cd $d; $plScript);
	fi;
done >> $outfile

# delete tmp files:
unlink $tmp_old
unlink $tmp_new


