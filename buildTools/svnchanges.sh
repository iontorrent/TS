#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Generate a report of all change lgo entries, for each directory since the last
# release build.

reportfile="${PWD}/latest_changes"

if [ -f $reportfile ]; then rm -f $reportfile; fi

MODULES=(
	Analysis	
	dbReports	
	gpu	
	ionifier	
	onetouchupdate
	pgmupdates
	plugin
	publishers
	referencelibrary
	rndplugins
	RSM	
	torrentR
	tsconfig
    )

for module in ${MODULES[@]}; do
	echo "======$module======" >> $reportfile
    cd $module
    svn2cl --reparagraph -i -r HEAD:"{`date -d '8 days ago' '+%F %T'`}"
    nline=$(grep -n "Candidate release build" ChangeLog|awk -F: '{print $1}')
    nline=$(echo $nline|awk '{print $1}')
    head -n $nline ChangeLog >> $reportfile
    echo >> $reportfile
    cd -
done


exit 0
