#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Produce a padded version of a BED file using BEDtools.
A genome file is required that has two tab-separated fields given chromosome IDs and sizes."
USAGE="USAGE:
 $CMD [options] <BED file> <genome> <pad size> <output file name>"
OPTIONS="OPTIONS:
  -h --help Report usage and help
  -l Log progress to STDERR"

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

SHOWLOG=0

while getopts "hl" opt
do
  case $opt in
    l) SHOWLOG=1;;
    h) echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
       exit 0;;
    \?) echo $USAGE >&2
        exit 1;;
  esac
done
shift `expr $OPTIND - 1`

if [ $# -ne 4 ]; then
  echo "$CMD: Invalid number of arguments." >&2
  echo -e "$USAGE\n$OPTIONS" >&2
  exit 1;
fi

BEDFILE=$1
GENOME=$2
PADSIZE=$3
BEDOUT=$4

#--------- End command arg parsing ---------

if [ $SHOWLOG -eq 1 ]; then
  echo "$CMD BEGIN:" `date` >&2
fi

RUNPTH=`readlink -n -f $0`
RUNDIR=`dirname $RUNPTH`

if [ $PADSIZE -le 0 ];then
  echo "ERROR: Invalid pad size ($PADSIZE); must be >= 0" >&2
  exit 1;
fi
if ! [ -f "$BEDFILE" ]; then
  echo "ERROR: Targets (BED) file does not exist at $BEDFILE" >&2
  exit 1;
fi
if ! [ -f "$GENOME" ]; then
  echo "ERROR: Genome file does not exist at $GENOME" >&2
  exit 1;
fi
if ! [ -f "$RUNDIR/slopBed" ]; then
  echo "ERROR: Could not locate local slopBed in script directory." >&2
  exit 1;
fi

BEDOUTTMP="$BEDOUT.tmp"

BCMD="${RUNDIR}/slopBed -i \"$BEDFILE\" -g \"$GENOME\" -b $PADSIZE > \"$BEDOUTTMP\""
eval "$BCMD" >&2
if [ $? -ne 0 ]; then
  echo "ERROR: BEDTools command failed."
  echo "\$ $BCMD" >&2
  exit 1;
fi
BCMD="${RUNDIR}/mergeBed -i \"$BEDOUTTMP\" > \"$BEDOUT\""
eval "$BCMD" >&2
if [ $? -ne 0 ]; then
  echo "ERROR: BEDTools command failed."
  echo "\$ $BCMD" >&2
  exit 1;
fi

# mergeBed re-orders the regions to alpha-num - need to re-order back to genome order
mv "$BEDOUT" "$BEDOUTTMP"
for CHRID in `awk '{print $1}' "$GENOME"`
do
  AWKCMD="awk '\$1==\"$CHRID\" {print}' \"$BEDOUTTMP\" >> \"$BEDOUT\""
  eval "$AWKCMD"
done
rm -f "$BEDOUTTMP"

if [ $SHOWLOG -eq 1 ]; then
  echo "$PADSIZE base padding of targets complete." >&2
  echo "$CMD END:" `date` >&2
fi

