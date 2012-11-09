#!/bin/bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# Script to record file size of the files removed from filesystem
#
# Intended to run as a cron job to purge Report Directory files
# The files removed are customer deliverable files!  To restore the files, the data must be reanalyzed from wells file.
#
output=0
do_remove=0
DAYS=60

while [ $# != 0 ]; do
	case $1 in
    	'-o'|'--output')
        	output=1
        ;;
        '-d'|'--delete')
        	do_remove=1
        ;;
        '--days')
        	shift
            DAYS=$1
        ;;
    esac
    shift
done

if [ "$output" == "1" ]; then
    # Generate a unique log filename
    DATESTAMP=$(date +%Y%m%d_)$RANDOM
    LOGFILE="cron_purge_$DATESTAMP.log"
    LOGDIR=$HOME

    # Store stdout file descriptor
    exec 3>&1
    # Redirect output to a file
    exec 1>$LOGDIR/$LOGFILE
fi


echo "Reports File Purge"
echo $(date)
echo
echo "Files older than $DAYS days selected"
echo
if [ $do_remove -eq 1 ]; then
	echo "Deleting enabled"
else
	echo "Deleting disabled"
fi

cmd="find /results/analysis/output/Home \
	-maxdepth 4 \
    -mtime +$DAYS -name *.bam \
    -o -mtime +$DAYS -name *.bai \
    -o -mtime +$DAYS -name *.sff  \
    -o -mtime +$DAYS -name *.fastq  \
    -o -mtime +$DAYS -name *.zip -size +10M \
    -o -mtime +$DAYS -name MaskBead.mask  \
    -o -mtime +$DAYS -name 1.tau  \
    -o -mtime +$DAYS -name 1.lmres  \
    -o -mtime +$DAYS -name 1.cafie-residuals  \
    -o -mtime +$DAYS -name bg_param.h5  \
    -o -mtime +$DAYS -name separator.h5  \
    -o -mtime +$DAYS -name BkgModel*.txt  \
    -o -mtime +$DAYS -name separator*.txt  \
    "

TOTAL=0
for file in $($cmd); do
	stat $file -c"%12s %n"
    SPACE=$(stat $file -c%s)
    TOTAL=$(($TOTAL+$SPACE))
    if [ $do_remove -eq 1 ]; then
    	rm -f $file
    fi
done

echo
if [ $do_remove -eq 1 ]; then
	echo "Total space freed:"
else
	echo "Total space identified:"
fi
echo "$(($TOTAL/1024/1024)) Mbytes"
echo

# Tests if fd 3 is connected and undoes redirection to log file
if test -t 3; then exec 1>&3 3>&-; fi

exit 0
