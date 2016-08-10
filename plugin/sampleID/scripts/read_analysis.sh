#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Create basic reads analysis summary text for sample targets, including male/female specific target reads."
USAGE="USAGE:
  $CMD [options] <reference.fasta> <BAM file>"
OPTIONS="OPTIONS:
  -h --help Report usage and help.
  -l Log progress to STDERR.
  -g Report reads to X/Y targets in -B targets file.
  -B <file> Limit coverage to targets specified in this BED file.
  -D <dirpath> Path to Directory where results are written. Default: ./
  -O <file> Output file name for text data (per analysis). Default: '' (STDOUT).
  -T <text> Output text to descript targets. Default: 'target region'."

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

SHOWLOG=0
BEDFILE=""
WORKDIR="."
OUTFILE=""
GENDERTRGS=0
TARGID="target region"

while getopts "hglB:D:O:T:" opt
do
  case $opt in
    g) GENDERTRGS=1;;
    l) SHOWLOG=1;;
    B) BEDFILE=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    O) OUTFILE=$OPTARG;;
    T) TARGID=$OPTARG;;
    h) echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
       exit 0;;
    \?) echo $USAGE >&2
        exit 1;;
  esac
done
shift `expr $OPTIND - 1`

if [ $# -ne 2 ]; then
  echo "$CMD: Invalid number of arguments." >&2
  echo -e "$USAGE\n$OPTIONS" >&2
  exit 1
fi

REFERENCE=$1
BAMFILE=$2

if [ "$OUTFILE" == "-" ]; then
  OUTFILE=""
fi

#--------- End command arg parsing ---------

RUNPTH=`readlink -n -f $0`
RUNDIR=`dirname $RUNPTH`

# Check environment

BAMROOT=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMNAME=`echo $BAMROOT | sed -e 's/\.[^.]*$//'`

if [ $SHOWLOG -eq 1 ]; then
  echo "$CMD BEGIN:" `date` >&2
  echo "REFERENCE: $REFERENCE" >&2
  echo "MAPPINGS:  $BAMROOT" >&2
  if [ -n "$BEDFILE" ]; then
    echo "TARGETS:   $BEDFILE" >&2
  fi
  echo "WORKDIR:   $WORKDIR" >&2
  if [ -n "$OUTFILE" ];then
    echo "TEXT OUT:  $OUTFILE" >&2
  else
    echo "TEXT OUT:  <STDOUT>" >&2
  fi
fi

if ! [ -d "$RUNDIR" ]; then
  echo "ERROR: Executables directory does not exist at $RUNDIR" >&2
  exit 1
elif ! [ -d "$WORKDIR" ]; then
  echo "ERROR: Output work directory does not exist at $WORKDIR" >&2
  exit 1
elif ! [ -f "$REFERENCE" ]; then
  echo "ERROR: Reference sequence (fasta) file does not exist at $REFERENCE" >&2
  exit 1
elif ! [ -f "$BAMFILE" ]; then
  echo "ERROR: Mapped reads (bam) file does not exist at $BAMFILE" >&2
  exit 1
elif [ -n "$BEDFILE" -a ! -f "$BEDFILE" ]; then
  echo "ERROR: Reference targets (bed) file does not exist at $BEDFILE" >&2
  exit 1
fi

# Get absolute file paths to avoid link issues in HTML
WORKDIR=`readlink -n -f "$WORKDIR"`
REFERENCE=`readlink -n -f "$REFERENCE"`
BAMFILE=`readlink -n -f "$BAMFILE"`

ROOTNAME="$WORKDIR/$BAMNAME"
if [ -n "$OUTFILE" ];then
  OUTFILE="${WORKDIR}/${OUTFILE}"
else
  OUTFILE="/dev/stdout"
fi

############

# Basic on-target stats
if [ -n "$OUTFILE" ]; then
  MREADS=`samtools view -c -F 4 "$BAMFILE"`
  TREADS=$MREADS
  PTREADS="100.0%"
  echo "Number of mapped reads:    $MREADS" > "$OUTFILE"
  if [ -n "$BEDFILE" ]; then
    TREADS=`samtools view -c -F 4 -L "$BEDFILE" "$BAMFILE"`
    if [ "$TREADS" -gt 0 ]; then
      PTREADS=`echo "$TREADS $MREADS" | awk '{printf("%.2f%%"),100*$1/$2}'`
    else
      PTREADS="0%"
    fi
  fi
  echo "Number of reads in ${TARGID}s: $TREADS" >> "$OUTFILE"
  echo "Percent reads in ${TARGID}s:   $PTREADS" >> "$OUTFILE"
  MREADS=`samtools depth -G 4 "$BAMFILE" | awk '{c+=$3} END {printf "%.0f",c+0}'`
  if [ -n "$BEDFILE" ]; then
    TREADS=`samtools depth -G 4 -b "$BEDFILE" "$BAMFILE" | awk '{c+=$3} END {printf "%.0f",c+0}'`
    if [ "$TREADS" -gt 0 ]; then
      PTREADS=`echo "$TREADS $MREADS" | awk '{printf("%.2f%%"),100*$1/$2}'`
    else
      PTREADS="0%"
    fi
  else
    TREADS=$MREADS
    PTREADS="100%"
  fi
  echo "Total base reads in ${TARGID}s:    $TREADS" >> "$OUTFILE"
  echo "Percent base reads in ${TARGID}s: $PTREADS" >> "$OUTFILE"
  if [ -n "$BEDFILE" -a "$GENDERTRGS" -gt 0 ]; then
    GENBED="${WORKDIR}/gender.bed"
    awk '$1~/^chrX/ {print}' "$BEDFILE" > "$GENBED"
    XREADS=`samtools view -c -F 4 -q 11 -L "$GENBED" "$BAMFILE"`
    awk '$1~/^chrY/ {print}' "$BEDFILE" > "$GENBED"
    YREADS=`samtools view -c -F 4 -q 11 -L "$GENBED" "$BAMFILE"`
    rm -f "$GENBED"
    echo "Male ${TARGID} reads:   $YREADS" >> "$OUTFILE"
    echo "Female ${TARGID} reads: $XREADS" >> "$OUTFILE"
  fi
fi

