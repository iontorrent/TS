#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Create tsv and image files of mapped read coverage to a reference."
USAGE="USAGE:
  $CMD [options] <reference.fasta> <BAM file>"
OPTIONS="OPTIONS:
  -h --help Report usage and help
  -l Log progress to STDERR
  -G <file> Genome file. Assumed to be <reference.fasta>.fai if not specified.
  -D <dirpath> Path to Directory where results are written. Default: ./
  -O <file> Output file name for text data (per analysis). Use '-' for STDOUT. Default: 'summary.txt'
  -R <file> Output file name for reads and base coverage data. Default: None created
  -B <file> Limit coverage to targets specified in this BED file
  -V <file> Filepath table list of called Variants
  -P <file> Padded targets BED file for padded target coverage analysis"

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

SHOWLOG=0
BEDFILE=""
GENOME=""
WORKDIR="."
OUTFILE="summary.txt"
PADBED=""
VARSFILE=""
READSTATS=""

while getopts "hlB:G:D:O:P:V:R:" opt
do
  case $opt in
    l) SHOWLOG=1;;
    B) BEDFILE=$OPTARG;;
    G) GENOME=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    O) OUTFILE=$OPTARG;;
    P) PADBED=$OPTARG;;
    V) VARSFILE=$OPTARG;;
    R) READSTATS=$OPTARG;;
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

if [ -z "$GENOME" ]; then
  GENOME="$REFERENCE.fai"
fi
if [ "$OUTFILE" == "-" ]; then
  OUTFILE=""
fi

#--------- End command arg parsing ---------

RUNPTH=`readlink -n -f $0`
RUNDIR=`dirname $RUNPTH`
#echo -e "RUNDIR=$RUNDIR\n" >&2

# Check environment

BAMROOT=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMNAME=`echo $BAMROOT | sed -e 's/\.[^.]*$//'`

if [ $SHOWLOG -eq 1 ]; then
  echo "$CMD BEGIN:" `date` >&2
  echo "REFERENCE: $REFERENCE" >&2
  echo "MAPPINGS:  $BAMROOT" >&2
  echo "GENOME:    $GENOME" >&2
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
elif ! [ -f "$GENOME" ]; then
  echo "ERROR: Genome (.fai) file does not exist at $GENOME" >&2
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
elif [ -n "$SNPSFILE" -a ! -f "$SNPSFILE" ]; then
  echo "ERROR: SNPs file does not exist at $SNPSFILE" >&2
  exit 1;
elif [ -n "$INDELSFILE" -a ! -f "$INDELSFILE" ]; then
  echo "ERROR: INDELs file does not exist at $INDELSFILE" >&2
  exit 1;
elif [ -n "$PADBED" -a ! -f "$PADBED" ]; then
  echo "ERROR: Padded reference targets (bed) file does not exist at $PADBED" >&2
  exit 1;
fi

# Get absolute file paths to avoid link issues in HTML
WORKDIR=`readlink -n -f "$WORKDIR"`
REFERENCE=`readlink -n -f "$REFERENCE"`
BAMFILE=`readlink -n -f "$BAMFILE"`
GENOME=`readlink -n -f "$GENOME"`

ROOTNAME="$WORKDIR/$BAMNAME"
if [ -n "$OUTFILE" ];then
  rm -f "${WORKDIR}/${OUTFILE}"
  OUTCMD=">> \"${WORKDIR}/${OUTFILE}\""
fi

############

# Basic on-target stats
if [ -n "$READSTATS" ]; then
  MREADS=`samtools view -c -F 4 "$BAMFILE"`
  PTREADS="100.0%"
  echo "Number of mapped reads:  $MREADS" > "${WORKDIR}/$READSTATS"
  if [ -n "$BEDFILE" ]; then
    TREADS=`samtools view -c -F 4 -L "$BEDFILE" "$BAMFILE"`
    if [ "$TREADS" -gt 0 ]; then
      PTREADS=`echo "$TREADS $MREADS" | awk '{printf("%.2f%%"),100*$1/$2}'`
    else
      PTREADS="0%"
    fi
  fi
  echo "Percent reads on target: $PTREADS" >> "${WORKDIR}/$READSTATS"
  MREADS=`samtools depth "$BAMFILE" | awk '{c+=$3} END {print c+0}'`
  echo "Number of mapped bases:  $MREADS" >> "${WORKDIR}/$READSTATS"
  PTREADS="100.0%"
  if [ -n "$BEDFILE" ]; then
    TREADS=`samtools depth -b "$BEDFILE" "$BAMFILE" | awk '{c+=$3} END {print c+0}'`
    if [ "$TREADS" -gt 0 ]; then
      PTREADS=`echo "$TREADS $MREADS" | awk '{printf("%.2f%%"),100*$1/$2}'`
    else
      PTREADS="0%"
    fi
  fi
  echo "Percent bases on target: $PTREADS" >> "${WORKDIR}/$READSTATS"
fi
 
# Basic coverage stats
if [ -n "$BEDFILE" ]; then
  gnm_size=`awk 'BEGIN {gs = 0} {gs += $3-$2} END {print gs}' "$BEDFILE"`
  COVERAGE_ANALYSIS="samtools depth -b \"$BEDFILE\" \"$BAMFILE\" 2> /dev/null | awk -f $RUNDIR/coverage_analysis.awk -v genome=$gnm_size"
else
  gnm_size=`awk 'BEGIN {gs = 0} {gs += $2} END {print gs}' "$GENOME"`
  COVERAGE_ANALYSIS="samtools depth \"$BAMFILE\" 2> /dev/null | awk -f $RUNDIR/coverage_analysis.awk -v genome=$gnm_size"
fi
eval "$COVERAGE_ANALYSIS $OUTCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: Command failed:" >&2
  echo "\$ $COVERAGE_ANALYSIS $OUTCMD" >&2
  exit 1;
fi

# Extract numbers of SNP and INDEL calls
AWK_COUNTREC='$0!~/^\s*$/ {if(++c>1){hes+=$3;hms+=$4;hei+=$5;hmi+=$6}} END {printf "Heterozygous SNPs:   %d\nHomozygous SNPs:     %d\nHeterozygous INDELs: %d\nHomozygous INDELs:   %d\n",hes,hms,hei,hmi}'
if [ -n "$VARSFILE" ]; then
  COVERAGE_ANALYSIS="awk '$AWK_COUNTREC' \"$VARSFILE\""
  eval "$COVERAGE_ANALYSIS $OUTCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: Command failed:" >&2
    echo "\$ $COVERAGE_ANALYSIS $OUTCMD" >&2
    exit 1;
  fi
fi
