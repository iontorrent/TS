#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Create reads filtered to unique starts. Output file name will <input file>.ustarts.bam";
USAGE="USAGE:
 $CMD [options] <BAM file>"
OPTIONS="OPTIONS:
  -h --help Report usage and help
  -i Create the BAM INDEX file (<input file>.ustarts.bam.bai).
  -l Log progress to STDERR. A few primary progress messages will still be output.
  -D <dirpath> Path to root Directory where results are written. Default: ./";

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

SHOWLOG=0
WORKDIR="."
MAKEBAI=0

while getopts "hlD:" opt
do
  case $opt in
    D) WORKDIR=$OPTARG;;
    i) MAKEBAI=1;;
    l) SHOWLOG=1;;
    h) echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
       exit 0;;
    \?) echo $USAGE >&2
        exit 1;;
  esac
done
shift `expr $OPTIND - 1`

if [ $# -ne 1 ]; then
  echo "$CMD: Invalid number of arguments." >&2
  echo -e "$USAGE\n$OPTIONS" >&2
  exit 1;
fi

BAMFILE=$1

#--------- End command arg parsing ---------

LOGOPT=''
if [ $SHOWLOG -eq 1 ]; then
  echo -e "\n$CMD BEGIN:" `date` >&2
  LOGOPT='-l'
fi
RUNPTH=`readlink -n -f $0`
RUNDIR=`dirname $RUNPTH`
if [ $SHOWLOG -eq 1 ]; then
  echo -e "RUNDIR=$RUNDIR\n" >&2
fi

# Check environment
if ! [ -d "$RUNDIR" ]; then
  echo "ERROR: Executables directory does not exist at $RUNDIR" >&2
  exit 1;
elif ! [ -d "$WORKDIR" ]; then
  echo "ERROR: Output work directory does not exist at $WORKDIR" >&2
  exit 1;
elif ! [ -f "$BAMFILE" ]; then
  echo "ERROR: Mapped reads (bam) file does not exist at $BAMFILE" >&2
  exit 1;
fi

echo "Filtering reads to unique starts..." >&2
if [ $SHOWLOG -eq 1 ]; then
  echo "" >&2
fi

BAMROOT=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMEXTN=`echo $BAMROOT | awk -F. '{print $NF}'`
BAMNAME=`echo $BAMROOT | sed -e 's/\.[^.]*$//'`
TSAMFILE="$WORKDIR/$BAMNAME.ustarts.sam"
BAMFILE2="$WORKDIR/$BAMNAME.ustarts.$BAMEXTN"

REMDUP="perl $RUNDIR/remove_pgm_duplicates.pl $LOGOPT -u \"$BAMFILE\" > \"$TSAMFILE\""
eval "$REMDUP" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: remove_pgm_duplicates.pl failed." >&2
  echo "\$ $REMDUP" >&2
  #exit 1;
fi

if [ $SHOWLOG -eq 1 ]; then
  echo -e "\nCreating sorted BAM and BAI files..." >&2
fi

SAMCMD="samtools view -S -b -t \"$GENOME\" -o \"$BAMFILE2\" \"$TSAMFILE\" &> /dev/null"
eval "$SAMCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: SAMtools command failed." >&2
  echo "\$ $SAMCMD" >&2
  #exit 1;
else
  if [ $SHOWLOG -eq 1 ]; then
    echo "> $BAMFILE2" >&2
  fi
fi
rm -f "$TSAMFILE"

if [ $MAKEBAI -eq 1 ]; then
  SAMCMD="samtools index \"$BAMFILE2\""
  eval "$SAMCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: SAMtools command failed." >&2
    echo "\$ $SAMCMD" >&2
    #exit 1;
  else
    if [ $SHOWLOG -eq 1 ]; then
      echo "> ${BAMFILE2}.bai" >&2
    fi
  fi
fi

if [ $SHOWLOG -eq 1 ]; then
  echo "Filtering to unique starts complete:" `date` >&2
  echo "" >&2
fi
 
