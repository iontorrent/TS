#!/bin/bash
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

LSDIR=$1
HTMLOUT=$2

HNAME=$(basename "$HTMLOUT")

echo "<table class='heading'>" > $HTMLOUT
echo "<thead> <tr><th>File Size</th><th>Date</th><th>File</th></tr></thead>" >> $HTMLOUT
for fname in `ls $LSDIR`
do b=`ls -gGhd $LSDIR/$fname | cut -f3,4,6 -d " "`
  # do not include the current file being created nor link to lifechart (for js includes)
  if [ $fname = "$HNAME" -o $fname = "lifechart" ];then
    continue
  fi 
  dataLine=`echo $b | awk '{print "<tr><td>"$1"</td><td>"$2"</td><td>"}'`
  echo "$dataLine<a href=${fname}>${fname}</a></td></tr>" >> $HTMLOUT
done
echo "</table>" >> $HTMLOUT

