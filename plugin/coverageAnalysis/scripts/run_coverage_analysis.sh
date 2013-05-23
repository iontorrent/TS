#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Wrapper for coverage_analysis to also handle read trimming and report workups.
Individual plots and data files are produced to the output directory ('.' unless specified by -D).
An HTML file for visualizing all results in a browser is also produced, unless the -x option is used."
USAGE="USAGE:
 $CMD [options] <reference.fasta> <BAM file>"
OPTIONS="OPTIONS:
  -h --help Report usage and help
  -a Customize output for Amplicon reads.
  -d Filter to remove Duplicate reads removed.
  -u Filter to Uniquely mapped reads (SAM MAPQ>0).
  -r Customize output for AmpliSeq-RNA reads. (Overrides -a.)
  -w Customize output for TargetSeq reads. Forces Warning if targets file (-B) is not provided.
  -t Filter BAM file to trimmed reads using TRIMP.
  -p <int>  Padding value for BED file padding. For reporting only. Default: 0.
  -A <file> Annotate coverage for (annotated) targets specified in this BED file
  -B <file> Limit coverage to targets specified in this BED file
  -C <name> Original name for BED targets selected for reporting (pre-padding, etc.)
  -D <dirpath> Path to root Directory where results are written. Default: ./
  -G <file> Genome file. Assumed to be <reference.fasta>.fai if not specified.
  -O <file> Output file name for text data (per analysis). Default: '' => <BAMROOT>.stats.cov.txt.
  -P <file> Padded targets BED file for padded target coverage analysis
  -Q <file> Name for BLOCK HTML results file (in output directory). Default: '' (=> none created)
  -R <file> Name for HTML Results file (in output directory). Default: 'results.html'
  -S <file> SampleID tracking regions file. Default: '' (=> no tageted reads statistic created)
  -T <file> Name for HTML Table row summary file (in output directory). Default: '' (=> none created)
  -l Log progress to STDERR. (A few primary progress messages will always be output.)
  -x Do not create the HTML file linking to all results created."

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

SHOWLOG=0
BEDFILE=""
GENOME=""
WORKDIR="."
OUTFILE=""
MAKEHML=1
RESHTML=""
ROWHTML=""
PADBED=""
BLOCKFILE=""
DEDUP=0
UNIQUE=0
ANNOBED=""
AMPOPT=""
TRIMP=0
PADVAL=0
TRGSID=""
RNABED=0
CKTARGETSEQ=0
TRACKINGBED=""

while getopts "hladrtuwxp:A:B:C:M:G:D:X:O:R:S:T:P:Q:" opt
do
  case $opt in
    A) ANNOBED=$OPTARG;;
    B) BEDFILE=$OPTARG;;
    C) TRGSID=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    G) GENOME=$OPTARG;;
    O) OUTFILE=$OPTARG;;
    P) PADBED=$OPTARG;;
    Q) BLOCKFILE=$OPTARG;;
    R) RESHTML=$OPTARG;;
    S) TRACKINGBED=$OPTARG;;
    T) ROWHTML=$OPTARG;;
    p) PADVAL=$OPTARG;;
    a) AMPOPT="-a";;
    d) DEDUP=1;;
    r) RNABED=1;;
    t) TRIMP=1;;
    u) UNIQUE=1;;
    w) CKTARGETSEQ=1;;
    x) MAKEHML=0;;
    l) SHOWLOG=1;;
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
if [ -z "$RESHTML" ]; then
  RESHTML="results.html"
fi

BASECOVERAGE=1
if [ $RNABED -eq 1 ]; then
  AMPOPT="-r"
  BASECOVERAGE=0
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

BAMROOT=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMNAME=`echo $BAMROOT | sed -e 's/\.[^.]*$//'`

# Short descript of read filters
RTITLE=""
if [ $TRIMP -eq 1 ]; then
  RTITLE="Trimmed"
fi
if [ $UNIQUE -eq 1 ]; then
  RTITLE="${RTITLE} Uniquely Mapped"
fi
if [ $DEDUP -eq 1 ]; then
  RTITLE="${RTITLE} Non-duplicate"
fi
if [ -z "$RTITLE" ]; then
  RTITLE="All Mapped"
fi
RTITLE="-s \"$RTITLE Reads\""

############

if [ $TRIMP -eq 1 ]; then
  echo "(`date`) Trimming reads to targets..." >&2
  BAMTRIM="${WORKDIR}/${BAMNAME}.trim.bam"
  RT=0
  PTRIMCMD="java -Xmx8G -cp ${DIRNAME}/TRIMP_lib -jar ${DIRNAME}/TRIMP.jar \"$BAMFILE\" \"$BAMTRIM\" \"$REFERENCE\" \"$ANNOBED\""
  if [ $SHOWLOG -gt 0 ]; then
    echo "\$ $PTRIMCMD" >&2
  fi
  eval "$PTRIMCMD" || RT=$?
  if [ $RT -ne 0 ]; then
    echo "WARNING: TRIMP failed..." >&2
    echo "\$ $PTRIMCMD" >&2
  elif [ $SHOWLOG -gt 0 ]; then
    echo "> ${BAMTRIM}" >&2
  fi
  if [ -e "$BAMTRIM" ]; then
    SINDX="samtools index \"$BAMTRIM\"" >&2
    eval "$SINDX" || RT=$?
    if [ $RT -ne 0 ]; then
      echo "WARNING: samtools index failed... Proceeding with pre-trimmed BAM file." >&2
      echo "\$ $SINDX" >&2
    else
      BAMFILE="$BAMTRIM"
      BAMNAME="${BAMNAME}.trim"
      BAMROOT="${BAMNAME}.bam"
      if [ $SHOWLOG -gt 0 ]; then
        echo "> ${BAMTRIM}.bai" >&2
      fi
    fi
  else
    echo "WARNING: No trimmed BAM file found. Proceeding with pre-trimmed BAM file." >&2
  fi
  echo "" >&2
fi

############

if [ "$OUTFILE" == "-" ]; then
  OUTFILE=""
fi
if [ -z "$OUTFILE" ]; then
  OUTFILE="${BAMNAME}.stats.cov.txt"
fi

FILTOPTS=""
if [ $DEDUP -eq 1 ]; then
  FILTOPTS="-d"
fi
if [ $UNIQUE -eq 1 ]; then
  FILTOPTS="$FILTOPTS -u"
fi

if [ $SHOWLOG -eq 1 ]; then
  echo "" >&2
fi
COVER="$RUNDIR/coverage_analysis.sh $LOGOPT $RTITLE $FILTOPTS $AMPOPT -O \"$OUTFILE\" -A \"$ANNOBED\" -B \"$BEDFILE\" -C \"$TRGSID\" -p $PADVAL -P \"$PADBED\" -S \"$TRACKINGBED\" -D \"$WORKDIR\" -G \"$GENOME\" \"$REFERENCE\" \"$BAMFILE\""
eval "$COVER" >&2
if [ $? -ne 0 ]; then
  echo -e "\nFailed to run coverage analysis."
  echo "\$ $COVER" >&2
fi

############

if [ $MAKEHML -eq 1 ]; then
  if [ $SHOWLOG -eq 1 ]; then
    echo "" >&2
  fi
  echo -e "(`date`) Creating HTML report..." >&2
  if [ -n "$ROWHTML" ]; then
    ROWHTML="-T \"$ROWHTML\""
  fi
  GENOPT="-g"
  if [ -n "$BEDFILE" ]; then
    GENOPT=""
    if [ -z "$AMPOPT" -a $TARGETCOVBYBASES -eq 1 ];then
      AMPOPT="-b"
    fi
  elif [ -n "$AMPOPT" -o "$CKTARGETSEQ" -eq 1 ];then
    AMPOPT="-w"
  fi
  SIDOPT=""
  if [ -n "$TRACKINGBED" ]; then
    SIDOPT="-i"
  fi
  if [ $NOTARGETANALYSIS -gt 0 ];then
    AMPOPT=""
  fi
  COVERAGE_HTML="COVERAGE_html"
  PTITLE=`echo $BAMNAME | sed -e 's/\.trim$//'`
  HMLCMD="$RUNDIR/coverage_analysis_report.pl $RTITLE $AMPOPT $ROWHTML $GENOPT $SIDOPT -N \"$BAMNAME\" -t \"$PTITLE\" -D \"$WORKDIR\" \"$COVERAGE_HTML\" \"$OUTFILE\""
  eval "$HMLCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: coverage_analysis_report.pl failed." >&2
    echo "\$ $HMLCMD" >&2
  elif [ $SHOWLOG -eq 1 ]; then
    echo "> ${WORKDIR}/$COVERAGE_HTML" >&2
  fi
  EXTRAHTML="$WORKDIR/tca_auxiliary.htm"
  cat "$EXTRAHTML" >> "${WORKDIR}/$COVERAGE_HTML" 
  rm -f "$EXTRAHTML"

  # Block Summary
  if [ -n "$BLOCKFILE" ]; then
    HMLCMD="perl $RUNDIR/coverage_analysis_block.pl $RTITLE $AMPOPT $GENOPT $SIDOPT -O \"$BLOCKFILE\" -D \"$WORKDIR\" -S \"$OUTFILE\" \"$BAMFILE\""
    eval "$HMLCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: coverage_analysis_block.pl failed." >&2
      echo "\$ $HMLCMD" >&2
    elif [ $SHOWLOG -eq 1 ]; then
      echo "> ${WORKDIR}/${BLOCKFILE}" >&2
    fi
  fi
  if [ $SHOWLOG -eq 1 ]; then
    echo "HTML report complete: " `date` >&2
  fi

fi

############

if [ $SHOWLOG -eq 1 ]; then
  echo -e "\n$CMD END:" `date` >&2
fi

