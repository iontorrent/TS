#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Create tsv and image files of mapped read coverage to a reference."
USAGE="USAGE:
  $CMD [options] <reference.fasta> <BAM file>"
OPTIONS="OPTIONS:
  -h --help Report usage and help.
  -l Log progress to STDERR.
  -a Customize output for Amplicon targets - use assigned reads rather than bases
  -d Ignore Duplicate reads.
  -u Include only Uniquely mapped reads (MAPQ > 1).
  -p <number> Padding value used (for report). Default: 0.
  -s <text> Single line of text reflecting user options selected. Default: 'All Reads'.
  -A <file> Annotated (non-merged) targets BED file for per-target coverage analysis.
  -B <file> General BED file specifying (merged) target regions for on-target base coverage analysis.
  -C <name> Original name for BED targets selected for reporting (pre-padding, etc.)
  -D <dirpath> Path to Directory where results are written.
  -G <file> Genome file. Assumed to be <reference.fasta>.fai if not specified.
  -O <file> Output file name for text data (per analysis). Default: '' => <BAMROOT>.stats.cov.txt.
  -P <file> Padded targets BED file for padded target coverage analysis."

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

SHOWLOG=0
TRACK=1
ANNOBED=""
BEDFILE=""
GENOME=""
WORKDIR="."
PADVAL=0
OUTFILE=""
PADBED=""
RUNOPTS="All Reads"
NONDUPREADS=0
UNIQUEREADS=0
AMPLICONS=0
PROPPLOTS=1
TRGSIG=""
PLOTREPLEN=0

while getopts "hladup:s:A:B:C:G:D:O:P:" opt
do
  case $opt in
    a) AMPLICONS=1;;
    d) NONDUPREADS=1;;
    u) UNIQUEREADS=1;;
    l) SHOWLOG=1;;
    p) PADVAL=$OPTARG;;
    s) USROPTS=$OPTARG;;
    A) ANNOBED=$OPTARG;;
    B) BEDFILE=$OPTARG;;
    C) TRGSID=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    G) GENOME=$OPTARG;;
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

RUNPTH=`readlink -n -f $0`
RUNDIR=`dirname $RUNPTH`

# Check environment

BAMROOT=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMNAME=`echo $BAMROOT | sed -e 's/\.[^.]*$//'`

if [ $SHOWLOG -eq 1 ]; then
  echo "(`date`) $CMD started." >&2
  echo "REFERENCE: $REFERENCE" >&2
  echo "MAPPINGS:  $BAMROOT" >&2
  echo "GENOME:    $GENOME" >&2
  if [ -n "$BEDFILE" ]; then
    echo "TARGETS:   $BEDFILE" >&2
  fi
  echo "WORKDIR:   $WORKDIR" >&2
  if [ -n "$OUTFILE" ];then
    echo "STATS OUT:  $OUTFILE" >&2
  fi
  echo >&2
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
elif [ -n "$ANNOBED" -a ! -f "$ANNOBED" ]; then
  echo "ERROR: Annotated targets (bed) file does not exist at $ANNOBED" >&2
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
#BAMFILE=`readlink -n -f "$BAMFILE"`
GENOME=`readlink -n -f "$GENOME"`

ROOTNAME="$WORKDIR/$BAMNAME"
AUXFILEROOT="$WORKDIR/tca_auxiliary"

if [ -z "$OUTFILE" ];then
  OUTFILE="${ROOTNAME}.stats.cov.txt"
fi
OUTFILE="$WORKDIR/$OUTFILE"

FILTOPTS=""
SAMVIEWOPT="-F 4"
if [ $NONDUPREADS -eq 1 ];then
  FILTOPTS="-d"
  SAMVIEWOPT="-F 0x404"
fi
if [ $UNIQUEREADS -eq 1 ];then
  FILTOPTS="$FILTOPTS -u"
  SAMVIEWOPT="$SAMVIEWOPT -q 1"
fi

# BEDFILE and ANNOFILE become the same if only one is defined
BEDOPT=''
if [ -n "$BEDFILE" ]; then
  BEDOPT="-B \"$BEDFILE\""
fi
if [ -n "$ANNOBED" ]; then
  ANNOBEDOPT="-B \"$ANNOBED\""
  if [ -z "$BEDFILE" ]; then
    BEDFILE=$ANNOBED
    BEDOPT=$ANNOBEDOPT
  fi
else
  ANNOBED=$BEDFILE
  ANNOBEDOPT=$BEDOPT
fi

REPLENPLOT=$AMPLICONS

PLOTERROR=0

########### Capture Title & User Options to Summary Text #########

echo -e "Coverage Analysis Report\n" > "$OUTFILE"
REFNAME=`echo "$REFERENCE" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
echo "Reference (File): $REFNAME" >> "$OUTFILE"
if [ -n "$BEDFILE" ]; then
  if [ -n "$TRGSID" ]; then
    TRGNAME=$TRGSID
  else
    TRGNAME=`echo "$BEDFILE" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
  fi
  echo "Targeted Regions: $TRGNAME" >> "$OUTFILE"
  if [ $PADVAL -gt 0 ]; then
    echo "Target Padding: $PADVAL" >> "$OUTFILE"
  fi
fi
ALMNAME=`echo "$BAMNAME" | sed -e 's/\.trim$//'`
echo "Alignments: $ALMNAME" >> "$OUTFILE"
echo -e "Using: $USROPTS\n" >> "$OUTFILE"

########### Create BBC files #########

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Creating base coverage files..." >&2
fi
BASEREADS="${AUXFILEROOT}.rds"
BBCFILE="${AUXFILEROOT}.bbc"
BBCMD="$RUNDIR/bbcCreate.pl $FILTOPTS $BEDOPT -p -O \"$BBCFILE\" \"$GENOME\" \"$BAMFILE\" > \"$BASEREADS\""
eval "$BBCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: bbcCreate.pl failed." >&2
  echo "\$ $BBCMD" >&2
  exit 1;
elif [ "$SHOWLOG" -eq 1 ]; then
  echo "> $BBCFILE" >&2
  echo ">" `cat $BASEREADS` "base reads" >&2
fi

BBCMD="$RUNDIR/bbcIndex.pl \"$BBCFILE\""
eval "$BBCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: bbcIndex.pl failed." >&2
  echo "\$ $BBCMD" >&2
  exit 1;
elif [ "$SHOWLOG" -eq 1 ]; then
  echo "> ${AUXFILEROOT}.bci" >&2
fi

########### Generate the Coverage Overview Plot #########

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Calculating effective reference coverage overview plot..." >&2
fi
COVOVR_XLS="$ROOTNAME.covoverview.xls"
COVCMD="$RUNDIR/bbcOverview.pl $BEDOPT \"$BBCFILE\" > \"$COVOVR_XLS\""
eval "$COVCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: bbcOverview.pl failed." >&2
  echo "\$ $COVCMD" >&2
  exit 1;
elif [ $SHOWLOG -eq 1 ]; then
  echo "> $COVOVR_XLS" >&2
fi

COVOVR_PNG="$ROOTNAME.covoverview.png"
PLOTCMD="R --no-save --slave --vanilla --args \"$COVOVR_XLS\" \"$COVOVR_PNG\" < $RUNDIR/plot_overview.R"
eval "$PLOTCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: plot_overview.R failed." >&2
  PLOTERROR=1
elif [ $SHOWLOG -eq 1 ]; then
  echo "> $COVOVR_PNG" >&2
fi

########### Read Coverage Analysis #########

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Generating basic reads stats..." >&2
fi
MAPPED_READS=`samtools view -c $SAMVIEWOPT "$BAMFILE"`
echo -e "Number of mapped reads:         $MAPPED_READS" >> "$OUTFILE"

if [ -n "$BEDOPT" ]; then
  # switch to flagstat if total #reads wanted (has exact same performance as view -c)
  TREADS=`samtools view -c $SAMVIEWOPT -L "$BEDFILE" "$BAMFILE"`
  FREADS=`echo "$TREADS $MAPPED_READS" | awk '{if($2<=0){$1=0;$2=1}printf "%.2f", 100*$1/$2}'`
  #echo "Number of reads on target:      $TREADS" >> "$OUTFILE"
  echo "Percent reads on target:        $FREADS%" >> "$OUTFILE"
  if [ -n "$PADBED" ]; then
    TREADS=`samtools view -c $SAMVIEWOPT -L "$PADBED" "$BAMFILE"`
    FREADS=`echo "$TREADS $MAPPED_READS" | awk '{if($2<=0){$1=0;$2=1}printf "%.2f", 100*$1/$2}'`
    echo "Percent reads on padded target: $FREADS%" >> "$OUTFILE"
  fi
fi
echo "" >> "$OUTFILE"

if [ -n "$ANNOBEDOPT" ]; then

  if [ $AMPLICONS -eq 0 ]; then
    TARGETCOVFILE="$ROOTNAME.target.cov.xls"
    COVCMD="$RUNDIR/bbcTargetAnno.pl \"$BBCFILE\" \"$ANNOBED\" > \"$TARGETCOVFILE\""
    TARGETMSG='target base'
  else
    TARGETCOVFILE="$ROOTNAME.amplicon.cov.xls"
    COVCMD="$RUNDIR/targetReadCoverage.pl $FILTOPTS \"$BAMFILE\" \"$ANNOBED\" > \"$TARGETCOVFILE\""
    TARGETMSG='amplicon read'
  fi
  if [ $TRACK -eq 1 ]; then
    echo "(`date`) Analyzing $TARGETMSG coverage..." >&2
  fi
  eval "$COVCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: $TARGETMSG analysis failed." >&2
    echo "\$ $COVCMD" >&2
    exit 1;
  elif [ $SHOWLOG -eq 1 ]; then
    echo "> $TARGETCOVFILE" >&2
  fi

  if [ $SHOWLOG -eq 1 ]; then
    echo "(`date`) Sorting $TARGETMSG coverage results to increasing read depth order..." >&2
  fi
  TMPFILE="$AUXFILEROOT.sort.tmp"
  COVCMD="head -1 \"$TARGETCOVFILE\" > \"$TMPFILE\"; tail --lines +2 \"$TARGETCOVFILE\" | sort -t \$'\t' -k 10n,10 -k 1d,1 -n -k 2n,2 -k 3n,3 >> \"$TMPFILE\""
  eval "$COVCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: bash sort failed." >&2
    echo "\$ $COVCMD" >&2
    exit 1;
  fi
  mv "$TMPFILE" "$TARGETCOVFILE"
  if [ $SHOWLOG -eq 1 ]; then
    echo "> $TARGETCOVFILE (sorted)" >&2
  fi

  if [ $TRACK -eq 1 ]; then
    echo "(`date`) Generating start-up $TARGETMSG Coverage Chart data..." >&2
  fi
  TCCINITFILE="$AUXFILEROOT.ttc.xls"
  COVCMD="$RUNDIR/target_coverage.pl -G \"$GENOME\" \"$TARGETCOVFILE\" - - 0 100000000 100 0 100 -1 > \"$TCCINITFILE\""
  eval "$COVCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: target_coverage.pl failed." >&2
    echo "\$ $COVCMD" >&2
    exit 1;
  elif [ $SHOWLOG -eq 1 ]; then
    echo "> $TCCINITFILE" >&2
  fi

  if [ $PROPPLOTS -eq 1 ]; then
    if [ $TRACK -eq 1 ]; then
      echo "(`date`) Generating static representation plots..." >&2
    fi
    TARGETCOV_GC_PNG=`echo $TARGETCOVFILE | sed 's/\.xls$/.gc.png/'`
    TARGETCOV_LEN_PNG=`echo $TARGETCOVFILE | sed 's/\.xls$/.len.png/'`
    TARGETCOV_REP_PNG=`echo $TARGETCOVFILE | sed 's/\.xls$/.fedora.png/'`
    TARGETCOV_RPL_PNG=`echo $TARGETCOVFILE | sed 's/\.xls$/.fedlen.png/'`
    PLOTOPT="FG"
    if [ $REPLENPLOT -eq 1 ]; then
      PLOTOPT="FGKL"
    fi
    if [ $AMPLICONS -ne 0 ]; then
      PLOTOPT="${PLOTOPT}a"
    fi
    PLOTCMD="R --no-save --slave --vanilla --args \"$TARGETCOVFILE\" $PLOTOPT < $RUNDIR/plot_gc.R"
    eval "$PLOTCMD" >&2
    if [ $? -ne 0 ]; then
      echo "ERROR: plot_gc.R failed." >&2
      PLOTERROR=1
    elif [ $SHOWLOG -eq 1 ]; then
      echo "> $TARGETCOV_REP_PNG" >&2
      echo "> $TARGETCOV_GC_PNG" >&2
      if [ $REPLENPLOT -eq 1 ]; then
        echo "> $TARGETCOV_RPL_PNG" >&2
        echo "> $TARGETCOV_LEN_PNG" >&2
      fi
    fi
  fi

fi

########### Depth of Read Coverage Analysis #########

if [ $AMPLICONS -ne 0 ]; then
  if [ $TRACK -eq 1 ]; then
    echo "(`date`) Analyzing depth of $TARGETMSG coverage..." >&2
  fi
  # For now there is no point in creating the DOC distribution file since this is useless for
  # Depth of Coverage plots with few amplicons (targets). Most read depths are only covered once
  # and this information is given in the fine coverage file.
  COVERAGE_ANALYSIS="$RUNDIR/targetReadStats.pl -a -M $MAPPED_READS \"$TARGETCOVFILE\" >> \"$OUTFILE\""
  eval "$COVERAGE_ANALYSIS >> \"$OUTFILE\"" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: targetReadStats.pl failed." >&2
    echo "\$ $COVERAGE_ANALYSIS >> \"$OUTFILE\"" >&2
    exit 1;
  fi
fi

########### Depth of Base Coverage Analysis #########

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Analyzing depth of base coverage..." >&2
fi
DOCFILE="$ROOTNAME.base.cov.xls"
TRGOPTS="-g"
if [ -n "$BEDFILE" ]; then
  trgsize=`awk 'BEGIN {gs = 0} {gs += $3-$2} END {printf "%.0f",gs+0}' "$BEDFILE"`
  basereads=`cat $BASEREADS`
  TRGOPTS="-C $basereads -T $trgsize"
fi
COVERAGE_ANALYSIS="$RUNDIR/bbcStats.pl $TRGOPTS -D \"$DOCFILE\" \"$BBCFILE\""
if [ $AMPLICONS -ne 0 ]; then
  echo "" >> "$OUTFILE"
fi
eval "$COVERAGE_ANALYSIS >> \"$OUTFILE\"" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: bbcStats.pl failed." >&2
  echo "\$ $COVERAGE_ANALYSIS >> \"$OUTFILE\"" >&2
  exit 1;
elif [ $SHOWLOG -eq 1 ]; then
  echo ">" $DOCFILE >&2
fi

########### Chromosome Coverage Analysis #########

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Generating reference coverage files..." >&2
fi
CBCFILE="$AUXFILEROOT.cbc"
CHRCOVFILE="$ROOTNAME.chr.cov.xls"
WGNCOVFILE="$ROOTNAME.wgn.cov.xls"
TRGOPTS=""
if [ -n "$BEDFILE" ]; then
  TRGOPTS="-t"
fi
COVCMD="$RUNDIR/bbcCoarseCov.pl $TRGOPTS -O \"$CBCFILE\" -C \"$CHRCOVFILE\" -W \"$WGNCOVFILE\" \"$BBCFILE\""
eval "$COVCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: bbcCourseCov.pl failed." >&2
  echo "\$ $COVCMD" >&2
  exit 1;
elif [ $SHOWLOG -eq 1 ]; then
  echo "> $CBCFILE" >&2
  echo "> $CHRCOVFILE" >&2
  echo "> $WGNCOVFILE" >&2
fi

########### Write HTML fragment for table #########

EXTRAHTML="$AUXFILEROOT.htm"
# files openned directly by the JS have to be with a local path (as opposed to those passed through to perl)
DOCFILE=`echo $DOCFILE | sed -e 's/^.*\///'`
echo "<br/> <div id='DepthOfCoverageChart' datafile='$DOCFILE' class='center' style='width:800px;height:300px'></div>" > "$EXTRAHTML"
REFGEN="genome"
COLLAPSERCC=""; # RCC will be expanded unless no targets are defined (=> Whole Genome but not necessarily!)
if [ -n "$BEDFILE" ]; then
  if [ $PROPPLOTS -eq 1 ]; then
    TARGETCOV_GC_PNG=`echo $TARGETCOV_GC_PNG | sed -e 's/^.*\///'`
    TARGETCOV_LEN_PNG=`echo $TARGETCOV_LEN_PNG | sed -e 's/^.*\///'`
    TARGETCOV_REP_PNG=`echo $TARGETCOV_REP_PNG | sed -e 's/^.*\///'`
    TARGETCOV_RPL_PNG=`echo $TARGETCOV_RPL_PNG | sed -e 's/^.*\///'`
    REPOPT="gccovfile='$TARGETCOV_GC_PNG' fedorafile='$TARGETCOV_REP_PNG'"
    if [ $REPLENPLOT -eq 1 ]; then
      REPOPT="$REPOPT fedlenfile='$TARGETCOV_RPL_PNG' lencovfile='$TARGETCOV_LEN_PNG'"
    fi
    echo "<br/> <div id='PictureFrame' $REPOPT class='center' style='width:800px;height:300px'></div>" >> "$EXTRAHTML"
  fi
  REFGEN=""
  COLLAPSERCC="collapse"
  TCCINITFILE=`echo $TCCINITFILE | sed -e 's/^.*\///'`
  echo "<br/> <div id='TargetCoverageChart' amplicons=$AMPLICONS datafile='$TARGETCOVFILE' initfile='$TCCINITFILE' class='center' style='width:800px;height:300px'></div>" >> "$EXTRAHTML"
fi
CHRCOVFILE=`echo $CHRCOVFILE | sed -e 's/^.*\///'`
WGNCOVFILE=`echo $WGNCOVFILE | sed -e 's/^.*\///'`
echo "<br/> <div id='ReferenceCoverageChart' $REFGEN $COLLAPSERCC bbcfile='$BBCFILE' cbcfile='$CBCFILE' chrcovfile='$CHRCOVFILE' wgncovfile='$WGNCOVFILE' class='center' style='width:800px;height:300px'></div>" >> "$EXTRAHTML"
echo "<br/> <div id='FileLinksTable' fileurl='filelinks.xls' class='center' style='width:440px'></div>" >> "$EXTRAHTML"
echo "<br/>" >> "$EXTRAHTML"

# create local igv session file
TRACKOPT=''
if [ -n "$ANNOBEDOPT" ]; then
  ANNOBED=`echo $ANNOBED | sed -e 's/^.*\///'`
  TRACKOPT="-a \"$ANNOBED\""
fi
BAMFILE=`echo $BAMFILE | sed -e 's/^.*\///'`
COVCMD="$RUNDIR/create_igv_link.py -r ${WORKDIR} -b ${BAMFILE} $TRACKOPT -g ${TSP_LIBRARY} -s igv_session.xml"
eval "$COVCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nWARNING: create_igv_link.py failed." >&2
  echo "\$ $COVCMD" >&2
elif [ $SHOWLOG -eq 1 ]; then
  echo "> igv_session.xml" >&2
fi

########### Finished #########

#if [ $PLOTERROR -eq 1 ]; then
#  exit 1;
#fi
if [ $SHOWLOG -eq 1 ]; then
  echo -e "\n(`date`) $CMD completed." >&2
fi

