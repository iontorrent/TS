#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Create tsv and image files of mapped read coverage to a reference."
USAGE="USAGE:
  $CMD [options] <reference.fasta> <BAM file>"
OPTIONS="OPTIONS:
  -h --help Report usage and help
  -G <file> Genome file. Assumed to be <reference.fasta>.fai if not specified.
  -D <dirpath> Path to Directory where results are written. Default: ./
  -O <file> Output file name for text data (per analysis). Use '-' for STDOUT. Default: 'summary.txt'
  -B <file> Limit coverage to targets specified in this BED file
  -P <file> Padded targets BED file for padded target coverage analysis
  -0 Include 0x coverage in (binned) plots
  -H <N> Set maximum coverage for Histogram plot to N. Default: 0
      0 => Full bar plot (not linear)"

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

PLOT0=0
MAXCOV=0
BEDFILE=""
GENOME=""
WORKDIR="."
BINSIZE=0
OUTFILE="summary.txt"
PADBED=""

while getopts "h0B:H:G:D:O:P:" opt
do
  case $opt in
    0) PLOT0=1;;
    B) BEDFILE=$OPTARG;;
    H) MAXCOV=$OPTARG;;
    G) GENOME=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    O) OUTFILE=$OPTARG;;
    P) PADBED=$OPTARG;;
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

echo "$CMD BEGIN:" `date` >&2
RUNPTH=`readlink -n -f $0`
RUNDIR=`dirname $RUNPTH`
#echo -e "RUNDIR=$RUNDIR\n" >&2

# Check environment

BAMROOT=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMNAME=`echo $BAMROOT | sed -e 's/\.[^.]*$//'`

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
echo >&2

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
  touch "$WORKDIR/$OUTFILE"
  OUTFILE=">> \"$WORKDIR/$OUTFILE\""
fi

# delete old run data
rm -f "$WORKDIR"/*

############

CHRCNT_XLS=$ROOTNAME.coverage_by_chrom.xls
CHRMAP_XLS=$ROOTNAME.coverage_map_by_chrom.xls

# Note: supplying genome is useful for getting overriding the chromosome order in the bed file
if [ -n "$BEDFILE" ]; then
  COVERAGE_ANALYSIS="perl $RUNDIR/chart_by_chrom.pl -o -B \"$BEDFILE\" -F \"$CHRMAP_XLS\" -G \"$GENOME\" \"$BAMFILE\" \"$CHRCNT_XLS\""
else
  COVERAGE_ANALYSIS="perl $RUNDIR/chart_by_chrom.pl -o -F \"$CHRMAP_XLS\" -G \"$GENOME\" \"$BAMFILE\" \"$CHRCNT_XLS\""
fi

echo "Calculating read distribution by chromosome..." >&2
eval "$COVERAGE_ANALYSIS $OUTFILE" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: chart_by_chrom.pl failed." >&2
  echo "\$ $COVERAGE_ANALYSIS $OUTFILE" >&2
  exit 1;
fi
echo "> $CHRCNT_XLS" >&2
echo "> $CHRMAP_XLS" >&2

CHRPAD_XLS=$ROOTNAME.coverage_by_chrom_padded_target.xls
if [ -n "$PADBED" ]; then
  COVERAGE_ANALYSIS="perl $RUNDIR/chart_by_chrom.pl -p -B \"$PADBED\" -G \"$GENOME\" \"$BAMFILE\" \"$CHRPAD_XLS\""
  eval "$COVERAGE_ANALYSIS $OUTFILE" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: chart_by_chrom.pl failed for padded targets." >&2
    echo "\$ $COVERAGE_ANALYSIS $OUTFILE" >&2
    exit 1;
  fi
  echo "> $CHRPAD_XLS" >&2
fi
echo "Chromosome coverage analysis done:" `date` >&2

############

FPILEUP="$ROOTNAME.starts.pileup"

if [ -n "$BEDFILE" ]; then
  #PILEUP_STARTS="samtools mpileup -B -l \"$BEDFILE\" -f \"$REFERENCE\" \"$BAMFILE\" 2> /dev/null > \"$FPILEUP\""
  PILEUP_STARTS="samtools depth -b \"$BEDFILE\" \"$BAMFILE\" 2> /dev/null > \"$FPILEUP\""
else
  #PILEUP_STARTS="samtools mpileup -B -f \"$REFERENCE\" \"$BAMFILE\" 2> /dev/null > \"$FPILEUP\""
  PILEUP_STARTS="samtools depth \"$BAMFILE\" 2> /dev/null > \"$FPILEUP\""
fi

echo -e "\nCounting base read depths..." >&2
eval "$PILEUP_STARTS" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: samtools depth failed." >&2
  echo "\$ $PILEUP_STARTS" >&2
  exit 1;
fi
echo "Base read depth counting done:" `date` >&2

############

TARGCOV_XLS=$ROOTNAME.fine_coverage.xls

if [ -n "$BEDFILE" ]; then
  FINE_COVERAGE_ANALYSIS="perl $RUNDIR/bed_covers.pl -G \"$GENOME\" \"$FPILEUP\" \"$BEDFILE\" > \"$TARGCOV_XLS\""
  echo -e "\nCreating individual targets coverage..." >&2
  eval "$FINE_COVERAGE_ANALYSIS" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: bed_covers.pl failed." >&2
    echo "\$ $FINE_COVERAGE_ANALYSIS" >&2
    exit 1;
  fi
  echo ">" $TARGCOV_XLS >&2
  echo "Individual targets coverage done:" `date` >&2
fi

############

OUTCOV_XLS=$ROOTNAME.coverage.xls
OUTCOV_BIN_XLS=$ROOTNAME.coverage_binned.xls

echo -e "\nCalculating coverage..." >&2

if [ -n "$BEDFILE" ]; then
  gnm_size=`awk 'BEGIN {gs = 0} {gs += $3-$2} END {print gs}' "$BEDFILE"`
else
  gnm_size=`awk 'BEGIN {gs = 0} {gs += $2} END {print gs}' "$GENOME"`
fi

base_reads=`samtools depth "$BAMFILE" | awk '{c+=$3} END {print c}'`

COVERAGE_ANALYSIS="awk -f $RUNDIR/coverage_analysis.awk -v basereads=$base_reads -v genome=$gnm_size -v outfile=\"$OUTCOV_BIN_XLS\" -v x1cover=\"$OUTCOV_XLS\" -v plot0x=$PLOT0 -v showlevels=$MAXCOV -v binsize=$BINSIZE \"$FPILEUP\""

eval "$COVERAGE_ANALYSIS $OUTFILE" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: awk command failed." >&2
  echo "\$ $COVERAGE_ANALYSIS $OUTFILE" >&2
  exit 1;
fi
echo ">" $OUTCOV_XLS >&2
echo ">" $OUTCOV_BIN_XLS >&2
echo "Coverage analysis done:" `date` >&2

############

echo -e "\nCreating coverage plots..." >&2

plotError=0

OUTCOV_PNG=$ROOTNAME.coverage.png
PLOTCMD="R --no-save --slave --vanilla --args \"$OUTCOV_XLS\" \"$OUTCOV_PNG\" < $RUNDIR/plot_coverage.R"
eval "$PLOTCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: plot_coverage.R failed." >&2
  plotError=1
else
  echo ">" $OUTCOV_PNG >&2
fi

OUTCOV_PNG=$ROOTNAME.coverage_normalized.png
PLOTCMD="R --no-save --slave --vanilla --args \"$OUTCOV_XLS\" \"$OUTCOV_PNG\" < $RUNDIR/plot_normalized_coverage.R"
eval "$PLOTCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: plot_normalized_coverage.R failed." >&2
  plotError=1
else
  echo ">" $OUTCOV_PNG >&2
fi

OUTCOV_PNG=$ROOTNAME.coverage_binned.png
PLOTCMD="R --no-save --slave --vanilla --args \"$OUTCOV_BIN_XLS\" \"$OUTCOV_PNG\" 0 < $RUNDIR/plot_binned_coverage.R"
eval "$PLOTCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: plot_binned_coverage.R failed for binned read coverage." >&2
  plotError=1
else
  echo ">" $OUTCOV_PNG >&2
fi

# disabled as plot is not considered of use at this time
#OUTCOV_PNG=$ROOTNAME.coverage_distribution.png
#PLOTCMD="R --no-save --slave --vanilla --args \"$OUTCOV_BIN_XLS\" \"$OUTCOV_PNG\" 1 < $RUNDIR/plot_binned_coverage.R"
#eval "$PLOTCMD" >&2
#if [ $? -ne 0 ]; then
#  echo -e "\nERROR: plot_binned_coverage.R failed for binned read distribution." >&2
#  plotError=1
#else
#  echo ">" $OUTCOV_PNG >&2
#fi

OUTCOV_PNG=$ROOTNAME.coverage_onoff_target.png
PLOTCMD="R --no-save --slave --vanilla --args \"$CHRCNT_XLS\" \"$OUTCOV_PNG\" < $RUNDIR/plot_onoff_target.R"
eval "$PLOTCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: plot_onoff_target.R failed." >&2
  plotError=1
else
  echo ">" $OUTCOV_PNG >&2
fi

if [ -n "$PADBED" ]; then
  OUTCOV_PNG=$ROOTNAME.coverage_onoff_padded_target.png
  PLOTCMD="R --no-save --slave --vanilla --args \"$CHRPAD_XLS\" \"$OUTCOV_PNG\" < $RUNDIR/plot_onoff_padded_target.R"
  eval "$PLOTCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: plot_onoff_padded_target.R failed." >&2
    plotError=1
  else
    echo ">" $OUTCOV_PNG >&2
  fi
fi
 
if [ -n "$BEDFILE" ]; then
  CHRMAP_PNG=$ROOTNAME.coverage_on_target.png
  PLOTCMD="R --no-save --slave --vanilla --args \"$TARGCOV_XLS\" \"$CHRMAP_PNG\" < $RUNDIR/plot_on_target.R"
  eval "$PLOTCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: plot_on_target.R failed." >&2
    plotError=1
  else
    echo ">" $CHRMAP_PNG >&2
  fi
fi

CHRMAP_PNG=$ROOTNAME.coverage_map_onoff_target.png
PLOTCMD="R --no-save --slave --vanilla --args \"$CHRMAP_XLS\" \"$CHRMAP_PNG\" < $RUNDIR/plot_map_onoff_target.R"
eval "$PLOTCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: plot_map_onoff_target.R failed." >&2
  plotError=1
else
  echo ">" $CHRMAP_PNG >&2
fi


rm -rf "$FPILEUP"
#if [ $plotError -eq 1 ]; then
#  exit 1;
#fi
echo -e "\n$CMD END:" `date` >&2
