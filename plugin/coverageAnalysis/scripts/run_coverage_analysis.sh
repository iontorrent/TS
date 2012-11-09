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
  -d Filter to reomove Duplicate reads removed.
  -u Filter to Uniquely mapped reads (SAM MAPQ>0).
  -B <file> Limit coverage to targets specified in this BED file
  -D <dirpath> Path to root Directory where results are written. Default: ./
  -G <file> Genome file. Assumed to be <reference.fasta>.fai if not specified.
  -H <dirpath> Path to directory containing files 'header' and 'footer', used to wrap HTML results file.
  -O <file> Output file name for text data (per analysis). Use '-' for STDOUT. Default: 'summary.txt'
  -P <file> Padded targets BED file for padded target coverage analysis
  -Q <file> Name for BLOCK HTML results file (in output directory). Default: '' (=> none created)
  -R <file> Name for HTML Results file (in output directory). Default: 'results.html'
  -T <file> Name for HTML Table row summary file (in output directory). Default: '' (=> none created)
  -l Log progress to STDERR. (A few primary progress messages will always be output.)
  -s Single run only. Otherwise unfiltered and filtered (for -d or -u) run passes are made, producing paired sets of reports.
  -x Do not create the HTML file linking to all results created."

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

SHOWLOG=0
MAXCOV=0
BEDFILE=""
GENOME=""
WORKDIR="."
OUTFILE=""
MAKEHML=1
RESHTML=""
ROWHTML=""
HAFHTML=""
PADBED=""
BLOCKFILE=""
DEDUP=0
UNIQUE=0
TWORUNS=1

# enables old strategy of pre-filtering BAM
PREFILTER_BAM=0

while getopts "hlsduxB:M:G:D:X:O:R:T:H:P:Q:" opt
do
  case $opt in
    B) BEDFILE=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    G) GENOME=$OPTARG;;
    H) HAFHTML=$OPTARG;;
    M) MAXCOV=$OPTARG;;
    O) OUTFILE=$OPTARG;;
    P) PADBED=$OPTARG;;
    Q) BLOCKFILE=$OPTARG;;
    R) RESHTML=$OPTARG;;
    T) ROWHTML=$OPTARG;;
    d) DEDUP=1;;
    l) SHOWLOG=1;;
    s) TWORUNS=0;;
    u) UNIQUE=1;;
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

FILTERRUN=0
UNFILTRUN=$TWORUNS
if [ $DEDUP -eq 1 -o $UNIQUE -eq 1 ]; then
  FILTERRUN=1
else
  UNFILTRUN=1
fi

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
#BAMFILE=`readlink -n -f "$BAMFILE"`
GENOME=`readlink -n -f "$GENOME"`

# Create subfolders if not already present

RESDIR1="all_reads"
RESDIR2="filtered_reads"

WORKDIR1="$WORKDIR/$RESDIR1"
WORKDIR2="$WORKDIR/$RESDIR2"

if [ $UNFILTRUN -eq 1 ]; then
  if ! [ -d "$WORKDIR1" ]; then
    mkdir "$WORKDIR1"
    if [ $? -ne 0 ]; then
      echo "ERROR: Failed to create work directory $WORKDIR1" >&2
      exit 1;
    fi
  fi
else
  rm -rf "$WORKDIR1"
fi
if [ $FILTERRUN -eq 1 ]; then
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

if [ $UNFILTRUN -eq 1 ]; then

  echo "Processing unfiltered reads..." >&2
  if [ $SHOWLOG -eq 1 ]; then
    echo "" >&2
  fi
  COVER="$RUNDIR/coverage_analysis.sh $LOGOPT -H $MAXCOV -O \"$OUTFILE\" -B \"$BEDFILE\" -P \"$PADBED\" -D \"$WORKDIR1\" -G \"$GENOME\" \"$REFERENCE\" \"$BAMFILE\""
  eval "$COVER" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nFailed to run coverage analysis for unfiltered reads." >&2
    echo "\$ $COVER" >&2
    exit 1;
  fi

fi;   # Filtered run condition

############

if [ $FILTERRUN -eq 1 ]; then

  if [ $SHOWLOG -eq 1 ]; then
    echo "" >&2
  fi
  BAMFILE2="$BAMFILE"
  FILTEROPTS=""
  if [ $DEDUP -eq 1 ]; then
    if [ $UNIQUE -eq 1 ]; then
      echo "Filtering BAM to uniquely mapped non-duplicate reads..." >&2
      BAMFILTER="-F 0x400 -q 1"
      FILTEROPTS="-d -u"
    else
      echo "Filtering BAM to non-duplicate reads..." >&2
      BAMFILTER="-F 0x400"
      FILTEROPTS="-d"
    fi
  else
    echo "Filtering BAM to uniquely mapped reads..." >&2
    BAMFILTER="-q 1"
    FILTEROPTS="-u"
  fi
  if [ $SHOWLOG -eq 1 ]; then
    echo "" >&2
  fi

  if [ $PREFILTER_BAM -eq 1 ]; then
    FILTEROPTS=""
    BAMFILE2="$WORKDIR/$BAMNAME.filtered.bam"
    REMDUP="samtools view -b -h $BAMFILTER \"$BAMFILE\" > \"$BAMFILE2\" 2> /dev/null"
    eval "$REMDUP" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: BAM filter command failed:" >&2
      echo "\$ $REMDUP" >&2
      echo "Proceeding with unfiltered BAM file." >&2
      BAMFILE2="$BAMFILE"
    else
      if [ $SHOWLOG -eq 1 ]; then
        echo "> $BAMFILE2" >&2
      fi
      SAMCMD="samtools index \"$BAMFILE2\""
      eval "$SAMCMD" >&2
      if [ $? -ne 0 ]; then
        echo -e "\nERROR: BAM indexing command failed:" >&2
        echo "\$ $SAMCMD" >&2
        echo "Proceeding with unfiltered BAM file." >&2
        BAMFILE2="$BAMFILE"
      else
        if [ $SHOWLOG -eq 1 ]; then
          echo "> ${BAMFILE2}.bai" >&2
        fi
      fi
    fi
    if [ $SHOWLOG -eq 1 ]; then
      echo "Filtering complete:" `date` >&2
      echo "" >&2
    fi
  fi
  ############

  echo "Processing filtered reads..." >&2
  if [ $SHOWLOG -eq 1 ]; then
    echo "" >&2
  fi
  COVER="$RUNDIR/coverage_analysis.sh $LOGOPT $FILTEROPTS -H $MAXCOV -O \"$OUTFILE\" -B \"$BEDFILE\" -P \"$PADBED\" -D \"$WORKDIR2\" -G \"$GENOME\" \"$REFERENCE\" \"$BAMFILE2\""
  eval "$COVER" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nFailed to run coverage analysis for filtered reads."
    echo "\$ $COVER" >&2
    #exit 1;
  fi

fi;   # Filtered run condition

############

if [ $MAKEHML -eq 1 ]; then
  if [ $SHOWLOG -eq 1 ]; then
    echo "" >&2
  fi
  echo -e "Creating HTML report..." >&2
  if [ $TWORUNS -eq 1 ]; then
    TWORES="-R \"$RESDIR2\" -B \"$BAMFILE2\""
  elif [ $FILTERRUN -eq 1 ]; then
    RESDIR1="$RESDIR2"
    BAMFILE="$BAMFILE2"
  fi
  if [ $DEDUP -eq 1 ]; then
    if [ $UNIQUE -eq 1 ]; then
      TWORES="$TWORES -p \"Uniquely Mapped Non-duplicate Reads\""
    else
      TWORES="$TWORES -p \"Non-duplicate Reads\""
    fi
  elif [ $UNIQUE -eq 1 ]; then
    TWORES="-p \"Uniquely Mapped Reads\""
  fi
  if [ -n "$ROWHTML" ]; then
    ROWHTML="-T \"$ROWHTML\""
  fi
  if [ -n "$HAFHTML" ]; then
    HAFHTML="-H \"$HAFHTML\""
  fi
  HMLCMD="perl $RUNDIR/coverage_analysis_report.pl -t \"$BAMNAME\" ${ROWHTML} -O \"$RESHTML\" ${HAFHTML} -D \"$WORKDIR\" -S \"$OUTFILE\" ${TWORES} \"$RESDIR1\" \"$BAMFILE\""
  if [ $SHOWLOG -eq 1 ]; then
    echo "\$ $HMLCMD" >&2
  fi
  eval "$HMLCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: coverage_analysis_report.pl failed." >&2
    echo "\$ $HMLCMD" >&2
  else
    if [ $SHOWLOG -eq 1 ]; then
      echo "> ${RESDIR1}/$RESHTML" >&2
    fi
  fi
  if [ $SHOWLOG -eq 1 ]; then
    echo "HTML report complete: " `date` >&2
  fi

  # Block Summary
  if [ -n "$BLOCKFILE" ]; then
    HMLCMD="perl $RUNDIR/coverage_analysis_block.pl -O \"$BLOCKFILE\" -D \"$WORKDIR\" -S \"$OUTFILE\" \"$RESDIR1\" \"$BAMFILE\""
    if [ $SHOWLOG -eq 1 ]; then
      echo "\$ $HMLCMD" >&2
    fi
    eval "$HMLCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: coverage_analysis_block.pl failed." >&2
      echo "\$ $HMLCMD" >&2
    else
      if [ $SHOWLOG -eq 1 ]; then
        echo "> ${WORKDIR}/${BLOCKFILE}" >&2
      fi
    fi
    if [ $SHOWLOG -eq 1 ]; then
      echo "HTML report complete: " `date` >&2
    fi
  fi

fi

if [ $SHOWLOG -eq 1 ]; then
  echo -e "\n$CMD END:" `date` >&2
fi

