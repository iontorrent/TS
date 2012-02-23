#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Run coverage_analysis for all reads and reads filtered to unique starts.
Results will go to two directories, .../all_reads and .../filtered_reads, or just the
.../all_reads directory if the -s option is specified. Here '...' is the current
directory or that specified by the -D option. Individual plots and data files are
produced to the output directories and a .html file, for visualizing all results in a
browser, is also produced unless the -x option is provided."
USAGE="USAGE:
 $CMD [options] <reference.fasta> <BAM file>"
OPTIONS="OPTIONS:
  -h --help Report usage and help
  -D <dirpath> Path to root Directory where results are written. Default: ./
  -G <file> Genome file. Assumed to be <reference.fasta>.fai if not specified.
  -O <file> Output file name for text data (per analysis). Use '-' for STDOUT. Default: 'summary.txt'
  -B <file> Limit coverage to targets specified in this BED file
  -P <file> Padded targets BED file for padded target coverage analysis
  -R <file> Name for HTML Results file (in output directory). Default: 'results.html'
  -T <file> Name for HTML Table row summary file (in output directory). Default: '' (=> none created)
  -H <dirpath> Path to directory containing files 'header' and 'footer', used to wrap HTML results file.
  -s Do not produce and analyze the filtered reads file
  -x Do not create the HTML file linking to all results created"

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

MAXCOV=0
BEDFILE=""
GENOME=""
WORKDIR="."
OUTFILE=""
USTARTS=1
MAKEHML=1
RESHTML=""
ROWHTML=""
HAFHTML=""
PADBED=""

while getopts "hsxB:M:G:D:X:O:R:T:H:P:" opt
do
  case $opt in
    B) BEDFILE=$OPTARG;;
    M) MAXCOV=$OPTARG;;
    G) GENOME=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    O) OUTFILE=$OPTARG;;
    R) RESHTML=$OPTARG;;
    T) ROWHTML=$OPTARG;;
    H) HAFHTML=$OPTARG;;
    P) PADBED=$OPTARG;;
    s) USTARTS=0;;
    x) MAKEHML=0;;
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
  exit 1;
fi

REFERENCE=$1
BAMFILE=$2

if [ -z "$GENOME" ]; then
  GENOME="$REFERENCE.fai"
fi
if [ -z "$OUTFILE" ]; then
  OUTFILE="summary.txt"
fi
if [ -z "$RESHTML" ]; then
  RESHTML="results.html"
fi

#--------- End command arg parsing ---------

echo "$CMD BEGIN:" `date` >&2
RUNPTH=`readlink -n -f $0`
RUNDIR=`dirname $RUNPTH`
echo -e "RUNDIR=$RUNDIR\n" >&2

# Check environment

if ! [ -d "$RUNDIR" ]; then
  echo "ERROR: Executables directory does not exist at $RUNDIR" >&2
  exit 1;
elif ! [ -d "$WORKDIR" ]; then
  echo "ERROR: Output work directory does not exist at $WORKDIR" >&2
  exit 1;
elif ! [ -f "$GENOME" ]; then
  echo "ERROR: Genome (.fai) file does not exist at $GENOME" >&2
  exit 1;
elif ! [ -f "$REFERENCE" ]; then
  echo "ERROR: Reference sequence (fasta) file does not exist at $REFERENCE" >&2
  exit 1;
elif ! [ -f "$BAMFILE" ]; then
  echo "ERROR: Mapped reads (bam) file does not exist at $BAMFILE" >&2
  exit 1;
elif [ -n "$BEDFILE" -a ! -f "$BEDFILE" ]; then
  echo "ERROR: Reference targets (bed) file does not exist at $BEDFILE" >&2
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

# Create subfolders if not already present

RESDIR1="all_reads"
RESDIR2="filtered_reads"

WORKDIR1="$WORKDIR/$RESDIR1"
WORKDIR2="$WORKDIR/$RESDIR2"

if ! [ -d "$WORKDIR1" ]; then
  mkdir "$WORKDIR1"
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create work directory $WORKDIR1" >&2
    exit 1;
  fi
fi
if [ $USTARTS -eq 1 ]; then
  if ! [ -d "$WORKDIR2" ]; then
    mkdir "$WORKDIR2"
    if [ $? -ne 0 ]; then
      echo "ERROR: Failed to create work directory $WORKDIR2" >&2
      exit 1;
    fi
  fi
else
  rm -rf "$WORKDIR2"
fi

BAMROOT=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMNAME=`echo $BAMROOT | sed -e 's/\.[^.]*$//'`

############

echo -e "Processing unfiltered reads...\n" >&2

COVER="$RUNDIR/coverage_analysis.sh -H $MAXCOV -O \"$OUTFILE\" -B \"$BEDFILE\" -P \"$PADBED\" -D \"$WORKDIR1\" -G \"$GENOME\" \"$REFERENCE\" \"$BAMFILE\""
eval "$COVER" >&2
if [ $? -ne 0 ]; then
  echo -e "\nFailed to run coverage analysis for unfiltered reads." >&2
  echo "\$ $COVER" >&2
  exit 1;
fi

############

if [ $USTARTS -eq 1 ]; then

  echo -e "\nFiltering reads to unique starts...\n" >&2

  TSAMFILE="$WORKDIR/$BAMNAME.ustarts.sam"
  TBAMSORT="$WORKDIR/$BAMNAME.ustarts.sort"
  BAMFILE2="$WORKDIR/$BAMNAME.ustarts.bam"

  REMDUP="perl $RUNDIR/remove_pgm_duplicates.pl -l -u \"$BAMFILE\" > \"$TSAMFILE\""
  eval "$REMDUP" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: remove_pgm_duplicates.pl failed." >&2
    echo "\$ $REMDUP" >&2
    #exit 1;
  fi

  echo -e "\nCreating sorted BAM and BAI files..." >&2

  SAMCMD="samtools view -S -b -t \"$GENOME\" -o \"$BAMFILE2\" \"$TSAMFILE\" &> /dev/null"
  eval "$SAMCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: SAMtools command failed." >&2
    echo "\$ $SAMCMD" >&2
    #exit 1;
  else
    echo "> $BAMFILE2" >&2
  fi

  SAMCMD="samtools sort \"$BAMFILE2\" \"$TBAMSORT\""
  eval "$SAMCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: SAMtools command failed." >&2
    echo "\$ $SAMCMD" >&2
    #exit 1;
  else
    mv "$TBAMSORT.bam" "$BAMFILE2"
  fi
  rm -f "$TSAMFILE"

  SAMCMD="samtools index \"$BAMFILE2\""
  eval "$SAMCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: SAMtools command failed." >&2
    echo "\$ $SAMCMD" >&2
    #exit 1;
  else
    echo "> ${BAMFILE2}.bai" >&2
  fi

  echo "Filtering to unique starts complete:" `date` >&2

  ############

  echo -e "\nProcessing filtered reads...\n" >&2

  COVER="$RUNDIR/coverage_analysis.sh -H $MAXCOV -O \"$OUTFILE\" -B \"$BEDFILE\" -P \"$PADBED\" -D \"$WORKDIR2\" -G \"$GENOME\" \"$REFERENCE\" \"$BAMFILE2\""
  eval "$COVER" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nFailed to run coverage analysis for filtered reads."
    echo "\$ $COVER" >&2
    #exit 1;
  fi

fi;   # Filtered run condition

############

if [ $MAKEHML -eq 1 ]; then
  echo -e "\nCreating HTML report..." >&2
  if [ $USTARTS -eq 1 ]; then
    TWORES="-R \"$RESDIR2\" -B \"$BAMFILE2\""
  fi
  if [ -n "$ROWHTML" ]; then
    ROWHTML="-T \"$ROWHTML\""
  fi
  if [ -n "$HAFHTML" ]; then
    HAFHTML="-H \"$HAFHTML\""
  fi
  HMLCMD="perl $RUNDIR/coverage_analysis_report.pl -t \"$BAMNAME\" ${ROWHTML} -O \"$RESHTML\" ${HAFHTML} -D \"$WORKDIR\" -S \"$OUTFILE\" ${TWORES} \"$RESDIR1\" \"$BAMFILE\""
  eval "$HMLCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: coverage_analysis_report.pl failed." >&2
    echo "\$ $HMLCMD" >&2
    #exit 1;
  fi
  echo "HTML report complete: " `date` >&2
fi

echo -e "\n$CMD END:" `date` >&2
