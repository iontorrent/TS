#!/bin/bash
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Description:
Run coverage analysis for read alignments to (regions of the) reference genome.
Individual plots and data files are produced to the output directory ('.' unless specified by -D).
A full HTML report is created if the -R option is employed, with interactive charts and links to output files.

Usage notes:
* For whole genome coverage the target files (-A and -B options) should not be supplied.
* For target base coverage just the -B option target file should be suppied.
* For target coverage summary file production the -g and/or -A options are required.
  (Note: -g should not be used with -A if the targets file is already in GC annotated BED format.)
* To generate the Amplicons Read Coverage report the -a (AmpliSeq) option should be supplied.
  Example: $CMD -ag -D results -R report.html -B ampliseq.bed hg19.fasta IonXpress_038_rawlib.bam
* To generate the Targets Coverage report the -c (TargetSeq) option should be supplied.
  For TargetSeq the -A file should not contain overlapping targets to ensure correct base coverage per target.
  The Ion published merged detail file should be used to ensure the Target Coverage report is accurate.
  A (merged) padded targets file is passed with the -B option, typically with -A for the unpadded targets.

Technical notes:
* This script depends on other code accessible from the original parent (coverageAnalysis) folder.
"

USAGE="USAGE:
 $CMD [options] <reference.fasta> <BAM file>"

OPTIONS="OPTIONS:
  -a Customize output for Amplicon reads. (AmpliSeq option.) (Overrides -c.)
  -b Base coverage only. No individual target coverage analysis. (Lite version.)
  -c Customize output for contig reads.
     Overrides -a and -t options for amplicon/target coverage statistics.
     If no targets BED file is provided an additional contig reads coverage file will be produced.
     If the targets BED file is provided this is assumed to specify (a subset) or whole reference contigs or chromosomes.
     (The chromosome base coverage will not be created in this case.)
  -d Ignore Duplicate reads. (By SAM FLAG.)
  -g Add GC annotation to targets file. By default the annotated BED file provided by the -A option is assumed to be
     correctly formatted (4 standard fields plus Ion auxiliary plus target GC count). With the -g option specified the
     -A BED file (or -B BED file if -A is not used) is re-formated and GC annotated.
  -r Customize output for targeted reads. (AmpliSeq-RNA option.) (Overrides -a and -t.)
  -t Add Target Coverage statistics by mean base read depth. (TargetSeq option.) See notes on -c option.
  -u Filter to Uniquely mapped reads. (By SAM MAPQ>0.)
  -A <file> Annotate coverage for (GC annotated) targets specified in this BED file. See -g option.
  -B <file> Limit coverage to targets specified in this BED file. Defaults to -A file if not provided.
  -C <file> Original name for BED targets selected for reporting and linking. Defaults to -B file if not provided.
  -D <dirpath> Path to root Directory where results are written. Default: ./
  -E <name> File name stem for auxilary output files. Default: 'tca_auxiliary'.
  -F <name> File name stem for analysis output files. Default: Use BAM file name provided (w/o extension).
  -G <file> Genome file. Assumed to be <reference.fasta>.fai if not specified.
  -L <name> Reference Library name, e.g. hg19. Defaults to <reference> if not supplied.
  -N <name> Sample name for use in summary output. Default: 'None'
  -M <int>  Minimium Mapped read length filter. Default: 0.
  -O <file> Output file name for text data (per analysis). Default: '' => <BAMNAME>.stats.cov.txt.
  -P <int>  Padding value used for BED file padding. Assumes -B targets file is padded if the pad value is > 0 and
     base coverage stats will be generated using this file BUT read stats will use the annotated -A targets file,
     which is required for this option. The padding value itself is only used for reporting. Default: 0.
  -Q <int>  Minimum read mapping Quality (MAPQ). Default: 0.
  -R <file> Name for local HTML Results file (in output directory). Default: '' (=> no HTML report created)
  -S <file> SampleID tracking regions file. Default: '' (=> no tracking reads statistics reported)
  -l Log progress to STDERR. (A few primary progress messages will always be output.)
  -h --help Report full description, usage and options."

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

SHOWLOG=0
BEDFILE=""
GENOME=""
WORKDIR="."
STATSTEM=""
RESHTML=""
DEDUP=0
UNIQUE=0
ANNOBED=""
AMPOPT=""
PADVAL=0
ORGBED=""
RNABED=0
TRACKINGBED=""
SAMPLENAME="None"
NOTARGETANAL=0
TRGCOVBYBASES=0
ANNOBEDFORMAT=0
LIBRARY=""
FILESTEM=""
AUXSTEM=""
CONTIGS=0
MINMAPLEN=0
MINMAPQUAL=0

TRACK=1
PLOTERROR=0
LINKANNOBED=1
TRGCOVDEPTH=1
AMPE2EREADS=1

while getopts "hlabcdgurtA:B:C:D:E:F:G:L:M:N:O:P:Q:R:S:" opt
do
  case $opt in
    A) ANNOBED=$OPTARG;;
    B) BEDFILE=$OPTARG;;
    C) ORGBED=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    E) AUXSTEM=$OPTARG;;
    F) FILESTEM=$OPTARG;;
    G) GENOME=$OPTARG;;
    L) LIBRARY=$OPTARG;;
    M) MINMAPLEN=$OPTARG;;
    N) SAMPLENAME=$OPTARG;;
    O) STATSTEM=$OPTARG;;
    P) PADVAL=$OPTARG;;
    Q) MINMAPQUAL=$OPTARG;;
    R) RESHTML=$OPTARG;;
    S) TRACKINGBED=$OPTARG;;
    a) AMPOPT="-a";;
    b) NOTARGETANAL=1;;
    c) CONTIGS=1;;
    d) DEDUP=1;;
    g) ANNOBEDFORMAT=1;;
    r) RNABED=1;;
    t) TRGCOVBYBASES=1;;
    u) UNIQUE=1;;
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

if [ -z "$BEDFILE" -a -n "$ANNOBED" ];then
  echo "WARNING: -A option suppied without -B option. Assuming -B option intended." >&2
  BEDFILE="$ANNOBED"
  ANNOBED=""
fi
if [ -z "$GENOME" ]; then
  GENOME="${REFERENCE}.fai"
fi
if [ -z "$LIBRARY" ]; then
  LIBRARY=`echo $REFERENCE | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
  echo "WARNING: -L option not supplied. Reference library name assumed to be '$LIBRARY'." >&2
fi
if [ -n "$RESHTML" ]; then
  RESHTML="${WORKDIR}/$RESHTML"
  rm -f "$RESHTML"
fi

if [ "$STATSTEM" = "-" ]; then
  STATSTEM=""
fi

RUNPTH=`readlink -n -f $0`
WORKDIR=`readlink -n -f "$WORKDIR"`
REFERENCE=`readlink -n -f "$REFERENCE"`
GENOME=`readlink -n -f "$GENOME"`

RUNDIR=`dirname $RUNPTH`
BINDIR="${RUNDIR}/bin"
RUNDIR="${RUNDIR}/scripts"
BAMBAI="${BAMFILE}.bai"
BAMNAME=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMSTEM=`echo $BAMNAME | sed -e 's/\.[^.]*$//'`

# if local (override) version of bbctools not present default to system version
BBCTOOLS="$BINDIR/bbctools"
if [ ! -f "$BBCTOOLS" ]; then
  BBCTOOLS="bbctools"
  command -v bbctools || { echo "ERROR: 'bbctools' is not available as a command on this system!" >&2; exit 1; }
fi

if [ "$FILESTEM" = "" -o "$FILESTEM" = "-" ];then
  FILESTEM="$BAMSTEM"
else
  FILESTEM=`echo $FILESTEM | sed -e 's/^.*\///'`
fi
if [ "$AUXSTEM" = "" -o "$AUXSTEM" = "-" ];then
  AUXSTEM="tca_auxiliary"
else
  AUXSTEM=`echo $AUXSTEM | sed -e 's/^.*\///'`
fi
OUTFILEROOT="$WORKDIR/$FILESTEM"
AUXFILEROOT="$WORKDIR/$AUXSTEM"

if [ -z "$STATSTEM" ]; then
  STATSTEM="${FILESTEM}.stats.cov.txt"
fi
STATSFILE="$WORKDIR/$STATSTEM"

#--------- End command arg parsing ---------

# Echo primary input args

if [ $SHOWLOG -eq 1 ]; then
  echo -e "\n$CMD BEGIN:" `date` >&2
  echo "(`date`) $CMD started." >&2
  echo "REFERENCE: $REFERENCE" >&2
  echo "MAPPINGS:  $BAMNAME" >&2
  echo "GENOME:    $GENOME" >&2
  if [ -n "$BEDFILE" ]; then
    echo "TARGETS:   $BEDFILE" >&2
  fi
  if [ -n "$STATSFILE" ];then
    echo "STATSFILE: $STATSFILE" >&2
  fi
  echo "RUNDIR:    $RUNDIR" >&2
  echo "WORKDIR:   $WORKDIR" >&2
  echo "FILESTEM:  $FILESTEM" >&2
  echo "AUXSTEM:   $AUXSTEM" >&2
  echo >&2
fi

# Check compatible options

if [ $NOTARGETANAL -eq 1 ];then
  if [ -n "$AMPOPT" ];then
    echo "WARNING: $AMPOPT option suppressed by -b option." >&2
    AMPOPT=""
  fi
  if [ $TRGCOVBYBASES -ne 0 ];then
    echo "WARNING: -t option suppressed by -b option." >&2
    TRGCOVBYBASES=0
  fi
fi
if [ -n "$AMPOPT" ];then
  if [ -z "$BEDFILE" ];then
    echo "ERROR: Targets file (-B option) required with $AMPOPT option." >&2
    exit 1
  fi
  if [ -z "$ANNOBED" -a "$ANNOBEDFORMAT" -eq 0 ];then
    echo "ERROR: Annotated targets file (-g or -A option) required with $AMPOPT option." >&2
    exit 1
  fi
  if [ -z "$ANNOBED" -a "$PADVAL" -ne 0 ];then
    echo "WARNING: Padding indicated but unpadded targets not supplied using -A option." >&2
  fi
fi

PROPPLOTS=1
BASECOVERAGE=1
WGN_CONTIGS=0
RNA_CONTIGS=0
if [ $CONTIGS -ne 0 ];then
  PROPPLOTS=0
  if [ -z "$BEDFILE" ];then
    # whole genome base coverage and rad coverage now available w/o target file for bbctools
    BASECOVERAGE=1
  elif [ $TRGCOVBYBASES -ne 0 ];then
    WGN_CONTIGS=1
  elif [ $RNABED -ne 0 ];then
    RNA_CONTIGS=1
    BASECOVERAGE=0
  fi
fi

# Set derived options and overrides

LOGOPT=''
if [ $SHOWLOG -eq 1 ]; then
  LOGOPT='-l'
fi
AMPCOVOPTS='-a'
AMPLICONS=0
TARGETTYPE='target'
if [ "$AMPOPT" = "-a" ]; then
  TARGETTYPE='amplicon'
  AMPLICONS=1
elif [ $RNABED -eq 1 ]; then
  TARGETTYPE='amplicon'
  AMPLICONS=2
  AMPOPT="-r"
  BASECOVERAGE=0
  TRGCOVDEPTH=0
  REP_OVERVIEW=1
else
  AMPCOVOPTS=''
  AMPE2EREADS=0
fi
REPLENPLOT=$AMPLICONS

ANNOBEDOPT=''
if [ -n "$ANNOBED" ]; then
  ANNOBEDOPT="-R '$ANNOBED'"
else
  TRGCOVBYBASES=0
fi
BEDOPT=''
if [ -n "$BEDFILE" ]; then
  BEDOPT="-R '$BEDFILE'"
else
  TRGCOVDEPTH=0
  AMPE2EREADS=0
fi

# Check environment files and directories

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
elif [ -n "$ANNOBED" -a ! -f "$ANNOBED" ]; then
  echo "ERROR: Annotated targets (bed) file does not exist at $ANNOBED" >&2
  exit 1;
elif [ -n "$TRACKINGBED" -a ! -f "$TRACKINGBED" ]; then
  echo "ERROR: Sample tracking targets (bed) file does not exist at $TRACKINGBED" >&2
  exit 1;
fi

# Short descriptons of read filters and translated filter options

RFTITLE=""
FILTOPTS=""
if [ -n "$TRACKINGBED" ]; then
  RFTITLE="Sample tracking"
fi
if [ $DEDUP -eq 1 ]; then
  FILTOPTS="-d"
  RFTITLE="${RFTITLE}, Non-duplicate"
fi
if [ $UNIQUE -eq 1 ]; then
  RFTITLE="${RFTITLE}, Uniquely mapped"
  FILTOPTS="$FILTOPTS -u"
fi
if [ $MINMAPLEN -gt 0 ];then
  RFTITLE="${RFTITLE}, Minimum aligned length = $MINMAPLEN"
  FILTOPTS="$FILTOPTS -L $MINMAPLEN"
fi
if [ $MINMAPQUAL -gt 0 ]; then
  RFTITLE="${RFTITLE}, Minimum mapping quality = $MINMAPQUAL"
  FILTOPTS="$FILTOPTS -Q $MINMAPQUAL"
fi
RFTITLE=`echo $RFTITLE | sed -e "s/^, //"`

TRGSID=`echo $ORGBED | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
if [ -z "$TRGSID" ];then
  TRGSID=`echo $BEDFILE | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
fi

########### Create local annotated bedfile given format (typically for cmd-line usage) ########### 

if [ "$ANNOBEDFORMAT" -ne 0 -a $NOTARGETANAL -eq 0 ];then
  INBEDFILE="$ANNOBED"
  if [ -z "$INBEDFILE" ];then
    INBEDFILE="$BEDFILE"
  fi
  if [ -n "$INBEDFILE" ]; then
    echo "(`date`) Creating GC annotated targets file..." >&2
    ANNOBED="${AUXFILEROOT}.gc.bed"
    ${RUNDIR}/../bed/gcAnnoBed.pl -a -s -w -t "$WORKDIR" "$INBEDFILE" "$REFERENCE" > "$ANNOBED"
    LINKANNOBED=0
    ANNOBEDOPT="-R '$ANNOBED'"
    if [ $SHOWLOG -eq 1 ]; then
      echo "> $ANNOBED" >&2
    fi
  fi
fi

########### Capture Title & User Options to Summary #########

if [ $SHOWLOG -eq 1 ]; then
  echo "" >&2
fi
echo -e "Coverage Analysis Report\n" > "$STATSFILE"
echo "Sample Name: $SAMPLENAME" >> "$STATSFILE"
REFNAME=`echo "$REFERENCE" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
echo "Reference Genome: $REFNAME" >> "$STATSFILE"
if [ -n "$BEDFILE" -a $CONTIGS -eq 0 ]; then
  echo "Target Regions: $TRGSID" >> "$STATSFILE"
  if [ $PADVAL -gt 0 ]; then
    echo "Target Padding: $PADVAL" >> "$STATSFILE"
  fi
fi
echo "Alignments: $BAMSTEM" >> "$STATSFILE"
if [ -n "$RFTITLE" ];then
  echo "Read Filters: $RFTITLE" >> "$STATSFILE"
fi
eval "$BBCTOOLS version" >> "$STATSFILE"
echo "" >> "$STATSFILE"

########### Create BBC and target (read) coverage result files ###########

# Set up for output of base coverage

BBCFILE="${AUXFILEROOT}.bbc"
BCIFILE="${AUXFILEROOT}.bci"
CBCFILE="${AUXFILEROOT}.cbc"

BASECOVOPTS=""
if [ $BASECOVERAGE -eq 1 ]; then
  BASECOVOPTS="-ciB '$AUXFILEROOT'"
  # Using padded targes is a special case where bbctools has to be used twice...
  if [ $PADVAL -gt 0 ];then
    if [ $TRACK -eq 1 ]; then
      echo "(`date`) Creating base coverage files for padded targets..." >&2
    fi
    # Here the sumStats will have on-target counts using the padded targets.
    # On-target reads could be gathered here vs. padded targets but was not done in original version
    BBCMD="$BBCTOOLS create $FILTOPTS $BEDOPT $BASECOVOPTS '$BAMFILE'"
    eval "$BBCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: bbctools create failed." >&2
      echo "\$ $BBCMD" >&2
      exit 1;
    elif [ "$SHOWLOG" -eq 1 ]; then
      if [ -n "$BASECOVOPTS" ]; then
        echo "> $BBCFILE" >&2
        echo "> $BCIFILE" >&2
        echo "> $CBCFILE" >&2
      fi
    fi
    BASECOVOPTS=""
    SSTOPT=""
  fi
fi

# Set up for output of on-target reads tracking

DEPTHS=""
if [ $RNABED -ne 0 ];then
  # disable depth of base coverage report for RNA applications
  DEPTH="-D ''"
fi
TARGETCOVFILE=""
TARGETCOVOPTS=""
DOSORT=0
if [ -n "$ANNOBEDOPT" -a $NOTARGETANAL -eq 0 ]; then
  XTRAFIELDS="3:region_id,-2:attributes,-1:gc_count"
  if [ $AMPLICONS -eq 0 ]; then
    TARGETCOVFILE="${OUTFILEROOT}.target.cov.xls"
    TARGETCOVOPTS="-C '$TARGETCOVFILE' -T trgbases -A '$XTRAFIELDS'"
  else
    TRGCOVBYBASES=0
    if [ $RNA_CONTIGS -ne 0 ];then
      TARGETCOVFILE="${OUTFILEROOT}.contig.cov.xls"
      TARGETCOVOPTS="-C '$TARGETCOVFILE' $DEPTH -T trgreads -A '$XTRAFIELDS'"
    else
      TARGETCOVFILE="${OUTFILEROOT}.amplicon.cov.xls"
      TARGETCOVOPTS="-C '$TARGETCOVFILE' $DEPTH -T AmpliSeq -A '$XTRAFIELDS'"
    fi
  fi
  DOSORT=1
elif [ $CONTIGS -ne 0 -a -z "$BEDFILE" ];then
  # -c option with no targets provided
  TARGETCOVFILE="${OUTFILEROOT}.ctg.cov.xls"
  TARGETCOVOPTS="-C '$TARGETCOVFILE' -T trgreads -D ''"
fi

if [ $TRACK -eq 1 ]; then
  if [ -n "$BASECOVOPTS" ]; then
    if [ -n "$TARGETCOVOPTS" ]; then
      echo "(`date`) Creating base coverage files and analyzing $TARGETTYPE coverage..." >&2
    else
      echo "(`date`) Creating base coverage files..." >&2
    fi
  elif [ -n "$TARGETCOVOPTS" ]; then
    echo "(`date`) Analyzing $TARGETTYPE coverage..." >&2
  fi
 
fi

SSTFILE="${OUTFILEROOT}.chr.reads.xls"
SSTOPT="-S '$SSTFILE'"

BBCMD="$BBCTOOLS create $FILTOPTS $ANNOBEDOPT $BASECOVOPTS $SSTOPT $TARGETCOVOPTS '$BAMFILE'"
eval "$BBCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: bbctools create failed." >&2
  echo "\$ $BBCMD" >&2
  exit 1;
elif [ "$SHOWLOG" -eq 1 ]; then
  if [ -n "$BASECOVOPTS" ]; then
    echo "> $BBCFILE" >&2
    echo "> $BCIFILE" >&2
    echo "> $CBCFILE" >&2
    echo "> $SSTFILE" >&2
  fi
  if [ -n "$TARGETCOVOPTS" ]; then
    echo "> $TARGETCOVFILE" >&2
  fi
fi

if [ $DOSORT -ne 0 ]; then
  if [ $TRACK -eq 1 ]; then
    echo "(`date`) Sorting $TARGETTYPE coverage results to increasing read depth order..." >&2
  fi
  TMPFILE="${AUXFILEROOT}.sort.tmp"
  COVCMD="head -1 '$TARGETCOVFILE' > '$TMPFILE'; tail --lines +2 '$TARGETCOVFILE' | sort -t \$'\t' -k 10n,10 -k 1d,1 -n -k 2n,2 -k 3n,3 >> '$TMPFILE'"
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
fi

########### Chromosome Coverage Analysis #########

if [ $BASECOVERAGE -eq 1 ]; then
  if [ $NOTARGETANAL -ne 0 ];then
    echo "(`date`) Skipping generating contig coverage files..." >&2
  elif [ $RNABED -eq 0 ];then
    if [ $TRACK -eq 1 ]; then
      echo "(`date`) Generating contig coverage files..." >&2
    fi
    TRGOPTS=""
    FTITLES="chrom,start,end,fwd_basereads,rev_basereads"
    if [ -n "$BEDFILE" -a $WGN_CONTIGS -eq 0 ]; then
      FTITLES="$FTITLES,fwd_trg_basereads,rev_trg_basereads"
    else
      TRGOPTS="-t"
    fi
    CHRCOVFILE="${OUTFILEROOT}.chr.cov.xls"
    COVCMD="$BBCTOOLS view -i $TRGOPTS -H $FTITLES '$BBCFILE' > '$CHRCOVFILE'"
    eval "$COVCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: bbctools view failed." >&2
      echo "\$ $COVCMD" >&2
      exit 1;
    elif [ $SHOWLOG -eq 1 ]; then
      echo "> $CHRCOVFILE" >&2
    fi
    # add target annotation fields to whole reference histogram
    if [ -n "$ANNOBEDOPT" -a $NOTARGETANAL -eq 0 ]; then
      ANNOTRG="$ANNOBEDOPT -A 3"
      FTITLES="$FTITLES,targets"
    fi
    WGNCOVFILE="${OUTFILEROOT}.wgn.cov.xls"
    COVCMD="$BBCTOOLS view -i -N 200 $TRGOPTS $ANNOTRG -H $FTITLES '$BBCFILE' > '$WGNCOVFILE'"
    eval "$COVCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: bbctools view failed." >&2
      echo "\$ $COVCMD" >&2
      exit 1;
    elif [ $SHOWLOG -eq 1 ]; then
      echo "> $WGNCOVFILE" >&2
    fi
    # take total reads per contig and add to the base cunts per target
    TMPFILE="${AUXFILEROOT}.chrcov.tmp"
    COVCMD="$RUNDIR/addContigReads.pl '$CHRCOVFILE' '$SSTFILE' > '$TMPFILE'"
    eval "$COVCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nWARNING: addContigReads.pl failed - read counts not added to $CHRCOVFILE.\n" >&2
    else
      mv "$TMPFILE" "$CHRCOVFILE"
    fi
  fi
fi

########### Generate the Coverage Overview Plot #########

if [ $BASECOVERAGE -eq 1 ]; then
  if [ $NOTARGETANAL -ne 0 ];then
    echo "(`date`) Skipping coverage overview analysis..." >&2
  elif [ $RNABED -eq 0 ];then
    if [ $TRACK -eq 1 ]; then
      echo "(`date`) Calculating effective reference coverage overview plot..." >&2
    fi
    COVOVR_XLS="${OUTFILEROOT}.covoverview.xls"
    COVCMD="$BBCTOOLS view $BEDOPT -N 600 -pst -H contigs,reads '$BBCFILE' > '$COVOVR_XLS'"
    eval "$COVCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: bbctools view failed." >&2
      echo "\$ $COVCMD" >&2
      exit 1;
    elif [ $SHOWLOG -eq 1 ]; then
      echo "> $COVOVR_XLS" >&2
    fi
    COVOVR_PNG="${OUTFILEROOT}.covoverview.png"
    PLOTCMD="R --no-save --slave --vanilla --args '$COVOVR_XLS' '$COVOVR_PNG' < $RUNDIR/plot_overview.R"
    eval "$PLOTCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: plot_overview.R failed." >&2
      PLOTERROR=1
    elif [ $SHOWLOG -eq 1 ]; then
      echo "> $COVOVR_PNG" >&2
    fi
  fi
fi

########### Sample Tracking Coverage Analysis #########

if [ -n "$TRACKINGBED" ]; then
  if [ $TRACK -eq 1 ]; then
    echo "(`date`) Analyzing sample tracking reads..." >&2
  fi
  # Assume sampleID tracking must be for AmpliSeq type analysis
  TRACKING_READS=`$BBCTOOLS create -R "$TRACKINGBED" -C - -T amplicon -P 30 -E 2 -D "" "$BAMFILE" | awk 'BEGIN {FS="\t"} NR>1 {c+=$7} END {print c}'`
fi

########### Basic Coverage Statistics and Depth of Coverage ###########

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Analyzing depth of coverage..." >&2
fi

read MAPPED_READS TARGET_READS <<< `awk 'BEGIN {FS="\t"} NR>1 {r+=$2+$3;t+=$4+$5} END {print r+0,t+0}' "$SSTFILE"`
echo -e "Number of mapped reads:         $MAPPED_READS" >> "$STATSFILE"

if [ -n "$BEDOPT" ]; then
  PC_TARGET_READS=`echo "$TARGET_READS $MAPPED_READS" | awk '{if($2<=0){$1=0;$2=1}printf "%.2f", 100*$1/$2}'`
# echo "Number of reads on target:      $TARGET_READS" >> "$STATSFILE"
  echo "Percent reads on target:        $PC_TARGET_READS%" >> "$STATSFILE"
fi
if [ -n "$TRACKINGBED" ]; then
# echo "Number of tracking reads:       $TRACKING_READS" >> "$STATSFILE"
  PC_TRACK_READS=`echo "$TRACKING_READS $MAPPED_READS" | awk '{if($2<=0){$1=0;$2=1}printf "%.2f", 100*$1/$2}'`
  echo "Percent sample tracking reads:  $PC_TRACK_READS%" >> "$STATSFILE"
fi
echo "" >> "$STATSFILE"

if [ $CONTIGS -ne 0 ];then
  # AMPOPT to distinuish RNA-Seq
  STATCMD="$RUNDIR/targetReadStats.pl -c $AMPOPT -M $MAPPED_READS '$SSTFILE'"
elif [ $AMPLICONS -ne 0 ]; then
  STATCMD="$RUNDIR/targetReadStats.pl $AMPOPT -M $MAPPED_READS '$TARGETCOVFILE'"
else
  STATCMD="$RUNDIR/targetReadStats.pl -b -P $PC_TARGET_READS '$TARGETCOVFILE'"
fi
eval "$STATCMD >> '$STATSFILE'" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: targetReadStats.pl failed." >&2
  echo "\$ $STATCMD >> '$STATSFILE'" >&2
  exit 1;
fi
if [ -n "$ANNOBEDOPT" ]; then
  CMPBIAS=`R --no-save --slave --vanilla --args "$TARGETCOVFILE" < $RUNDIR/gcbias.R`
  if [ $AMPLICONS -ne 0 ]; then
    echo "Amplicon base composition bias:     $CMPBIAS" >> "$STATSFILE"
  else
    echo "Target base composition bias:       $CMPBIAS" >> "$STATSFILE"
  fi
fi

########### Base depth of coverage and summary report ###########

if [ $BASECOVERAGE -eq 1 ]; then
  echo "" >> "$STATSFILE"
  # new command needs lack of BED file to indicate genome coverage
  if [ $CONTIGS -eq 0 ];then
    TBEDOPT=$BEDOPT
  fi
  DOCFILE="${OUTFILEROOT}.base.cov.xls"
  DOCCMD="$BBCTOOLS report -g $TBEDOPT -C '$DOCFILE' '$BBCFILE' >> '$STATSFILE'"
  eval "$DOCCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: bbctools view failed." >&2
    echo "\$ $DOCCMD" >&2
    exit 1;
  elif [ $SHOWLOG -eq 1 ]; then
    echo "> $DOCFILE" >&2
  fi
fi

########### AmpliSeq option for end-to-end reads ###########

if [ $AMPE2EREADS -eq 1 -a $CONTIGS -eq 0 ]; then
  # for RNA AmpliSeq this is a bonus stat added (only) to summary file in 5.2
  if [ $RNABED -ne 0 ];then
    echo "" >> "$STATSFILE"
  fi
  pce2erds=`awk 'BEGIN {FS="\t"} NR>1 {e2e+=$8+$9;tr+=$10} END {if(tr>0){printf "%.2f",100*e2e/tr}else{print "0.00"}}' $TARGETCOVFILE`
  echo "Percent end-to-end reads:          ${pce2erds}%" >> "$STATSFILE"
fi

######### Create static target representation plots and initial chart view #########

if [ $NOTARGETANAL -ne 0 ]; then
  echo "(`date`) Skipping fine coverage analysis..." >&2
elif [ -n "$ANNOBEDOPT" ]; then
  if [ $TRACK -eq 1 ]; then
    echo "(`date`) Generating start-up $TARGETTYPE coverage chart data..." >&2
  fi
  TCCINITFILE="${AUXFILEROOT}.ttc.xls"
  GENOPT="-c"
  if [ $RNABED -eq 0 ]; then
    GENOPT="-G '$GENOME'"
  fi
  COVCMD="$RUNDIR/target_coverage.pl $GENOPT '$TARGETCOVFILE' - - 0 100000000 100 0 100 -1 > '$TCCINITFILE'"
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
    TARGETCOV_GC_PNG="${OUTFILEROOT}.gc.png"
    TARGETCOV_LEN_PNG="${OUTFILEROOT}.len.png"
    TARGETCOV_REP_PNG="${OUTFILEROOT}.gc_rep.png"
    TARGETCOV_RPL_PNG="${OUTFILEROOT}.ln_rep.png"
    TARGETCOV_RPP_PNG="${OUTFILEROOT}.pool.png"
    PLOTOPT="FGp"
    if [ $REPLENPLOT -eq 1 ]; then
      PLOTOPT="FGKLp"
    elif [ $REPLENPLOT -eq 2 ]; then
      PLOTOPT="p"
    fi
    if [ $AMPLICONS -ne 0 ]; then
      PLOTOPT="${PLOTOPT}a"
    fi
    PLOTCMD="R --no-save --slave --vanilla --args '$TARGETCOVFILE' $PLOTOPT '${OUTFILEROOT}' < $RUNDIR/plot_gc.R"
    eval "$PLOTCMD" >&2
    if [ $? -ne 0 ]; then
      echo "ERROR: plot_gc.R failed." >&2
      PLOTERROR=1
    elif [ $SHOWLOG -eq 1 ]; then
      echo "> $TARGETCOV_REP_PNG" >&2
      echo "> $TARGETCOV_GC_PNG" >&2
      if [ -e "$TARGETCOV_RPP_PNG" ];then
        echo "> $TARGETCOV_RPP_PNG" >&2
      fi
      if [ $REPLENPLOT -eq 1 ]; then
        echo "> $TARGETCOV_RPL_PNG" >&2
        echo "> $TARGETCOV_LEN_PNG" >&2
      fi
    fi
  fi
  if [ $RNABED -eq 1 ]; then
    if [ $TRACK -eq 1 ]; then
      echo "(`date`) Generating amplicon representation overview plot..." >&2
    fi
    if [ $CONTIGS -ne 0 ];then
      REPOVRTRG='Contigs'
    else
      REPOVRTRG='Amplicons'
    fi
    REPOVR_PNG="${OUTFILEROOT}.repoverview.png"
    PLOTCMD="R --no-save --slave --vanilla --args '$TARGETCOVFILE' '$REPOVR_PNG' $REPOVRTRG < $RUNDIR/plot_rna_rep.R"
    eval "$PLOTCMD" >&2
    if [ $? -ne 0 ]; then
      echo "ERROR: plot_rna_rep.R failed." >&2
      PLOTERROR=1
    elif [ $SHOWLOG -eq 1 ]; then
      echo "> $REPOVR_PNG" >&2
    fi
  fi
fi

# ------------- Analysis Complete -------------

# Create table of primary output files and their descriptions.
# - Temporary code until a json solution is implemented for both templates and cmd-line

if [ -n "$ORGBED" ];then
 PLUGIN_OUT_BEDPAGE=`echo $ORGBED | sed -e 's/^.*\/uploads\/BED\/\([0-9][0-9]*\)\/.*/\/rundb\/uploadstatus\/\1/'`
fi
# Create direct links to BED files IF original BED file was not converted to a link to the TSS BED upload page
if [ "$PLUGIN_OUT_BEDPAGE" = "$ORGBED" ];then
  # only one file link for contigs mode
  if [ -f "$BEDFILE" -a $CONTIGS -eq 0 ];then
    PLUGIN_OUT_BEDFILE_MERGED=`echo "$BEDFILE" | sed -e 's/^.*\///'`
    ln -sf "$BEDFILE" "${WORKDIR}/$PLUGIN_OUT_BEDFILE_MERGED"
  fi
  if [ -f "$ANNOBED" -a "$LINKANNOBED" -ne 0 ];then
    PLUGIN_OUT_BEDFILE_UNMERGED=`echo "$ANNOBED" | sed -e 's/^.*\///'`
    ln -sf "$ANNOBED" "${WORKDIR}/$PLUGIN_OUT_BEDFILE_UNMERGED"
  fi
fi
PLUGIN_OUT_BAMFILE="$BAMNAME"
PLUGIN_OUT_BAIFILE="${BAMNAME}.bai"
PLUGIN_OUT_STATSFILE="${FILESTEM}.stats.cov.txt"
PLUGIN_OUT_DOCFILE="${FILESTEM}.base.cov.xls"
if [ $RNABED -ne 0 ];then
  PLUGIN_OUT_RNACOVFILE="${FILESTEM}.amplicon.cov.xls"
else
  PLUGIN_OUT_AMPCOVFILE="${FILESTEM}.amplicon.cov.xls"
fi
PLUGIN_OUT_TRGCOVFILE="${FILESTEM}.target.cov.xls"
if [ $BASECOVERAGE -ne 0 ];then
  PLUGIN_OUT_CHRCOVFILE="${FILESTEM}.chr.cov.xls"
elif [ $RNA_CONTIGS -ne 0 ];then
  PLUGIN_OUT_CTGCOVFILE="${FILESTEM}.contig.cov.xls"
fi
PLUGIN_OUT_WGNCOVFILE="${FILESTEM}.wgn.cov.xls"
source "${RUNDIR}/fileLinks.sh"
write_file_links "$WORKDIR" "filelinks.xls";

# Create local igv session file - again perhaps sould be moved to plugin code
if [ -n "$LIBRARY" ]; then
  TRACKOPT=''
  if [ -n "$ANNOBED" ]; then
    ABED=`echo $ANNOBED | sed -e 's/^.*\///'`
    TRACKOPT="-a '$ABED'"
  fi
  COVCMD="$RUNDIR/create_igv_link.py -r ${WORKDIR} -b ${BAMNAME} $TRACKOPT -g ${LIBRARY} -s igv_session.xml"
  eval "$COVCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nWARNING: create_igv_link.py failed." >&2
    echo "\$ $COVCMD" >&2
  elif [ $SHOWLOG -eq 1 ]; then
    echo "> igv_session.xml" >&2
  fi
fi

############# Stand-alone HTML report writer #############

if [ -n "$RESHTML" ]; then
  if [ $SHOWLOG -eq 1 ]; then
    echo "" >&2
  fi
  echo -e "(`date`) Creating HTML report..." >&2
  GENOPT="-g"
  if [ -n "$BEDFILE" ]; then
    GENOPT=""
    if [ -z "$AMPOPT" -a $TRGCOVBYBASES -eq 1 ];then
      AMPOPT="-b"
    fi
  elif [ -n "$AMPOPT" ];then
    AMPOPT="-w"
  fi
  SIDOPT=""
  if [ -n "$TRACKINGBED" ]; then
    SIDOPT="-i"
  fi
  if [ $NOTARGETANAL -gt 0 ];then
    AMPOPT=""
  fi
  WARNMSG=''
  if ! [ -f "$BAMBAI" ];then
    WARNMSG="-W \"<h4 style='text-align:center;color:red'>WARNING: BAM index file not found. Assignments of reads to amplicons not performed.</h4>\""
  fi
  COVERAGE_HTML="COVERAGE_html"
  STATSDICT="$RUNDIR/../templates/help_dict.json"
  HMLCMD="$RUNDIR/coverage_analysis_report.pl $RFTITLE $AMPOPT $GENOPT $SIDOPT $WARNMSG -A '$STATSDICT' -N '$FILESTEM' -t '$FILESTEM' -D '$WORKDIR' '$COVERAGE_HTML' '$STATSTEM'"
  eval "$HMLCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: coverage_analysis_report.pl failed." >&2
    echo "\$ $HMLCMD" >&2
  elif [ $SHOWLOG -eq 1 ]; then
    echo "> ${WORKDIR}/$COVERAGE_HTML" >&2
  fi

  # attempt to link style and javascript files locally
  ln -sf "${RUNDIR}/../flot" "${WORKDIR}/"
  ln -sf "${RUNDIR}/../lifechart" "${WORKDIR}/"
  ln -sf "${RUNDIR}/igv.php3" "${WORKDIR}/"
  if [ -f "${WORKDIR}/lifechart/TCA.head.html" ];then
    # charts should be active if this file exists
    cat "${WORKDIR}/lifechart/TCA.head.html" > "$RESHTML"
    if [ "$AMPOPT" != "-r" ]; then
      echo '<script language="javascript" type="text/javascript" src="lifechart/DepthOfCoverageChart.js"></script>' >> "$RESHTML"
      echo '<script language="javascript" type="text/javascript" src="lifechart/ReferenceCoverageChart.js"></script>' >> "$RESHTML"
    fi
    if [ -n "$BEDFILE" ]; then
      echo '<script language="javascript" type="text/javascript" src="lifechart/TargetCoverageChart.js"></script>' >> "$RESHTML"
    fi
  else
    # else give basic header - interactive chart links should not appear
  echo '<?xml version="1.0" encoding="iso-8859-1"?>' > "$RESHTML"
    echo -e '<!DOCTYPE HTML>\n<html>\n<head><base target="_parent"/>\n' >> "$RESHTML"
    echo '<meta http-equiv="Content-Type" content="text/html; charset=utf-8">' >> "$RESHTML"
  fi
  echo '</head>' >> "$RESHTML"
  echo '<body>' >> "$RESHTML"
  echo "<div class='center' style='width:100%;height:100%'>" >> "$RESHTML"
  echo "<h1><center>Coverage Analysis Report</center></h1>" >> "$RESHTML"
  if [ -n "$SAMPLENAME" -a "$SAMPLENAME" != "None" ];then
    echo "<h3><center>Sample Name: $SAMPLENAME</center></h3>" >> "$RESHTML"
  fi

  # add in the coverage stats tables
  cat "${WORKDIR}/$COVERAGE_HTML" >> "$RESHTML"
  rm -f "${WORKDIR}/$COVERAGE_HTML"

  # add in the charts HTML (that used to be handled by coverage_analysis.sh)
  if [ $BASECOVERAGE -eq 1 ]; then
    # files openned directly by the JS have to be with a local path (as opposed to those passed through to perl)
    DOCFILE=`echo $DOCFILE | sed -e 's/^.*\///'`
    DOCFILE="$FILESTEM.base.cov.xls"
    echo "<br/> <div id='DepthOfCoverageChart' datafile='$DOCFILE' class='center' style='width:800px;height:300px'></div>" >> "$RESHTML"
  fi
  REFGEN="genome"
  COLLAPSERCC=""; # RCC will be expanded unless no targets are defined (=> Whole Genome but not necessarily!)
  TARGTYPE='target'
  if [ $AMPLICONS -gt 0 ];then
    TARGTYPE='amplicon'
  fi
  if [ $NOTARGETANAL -ne 0 ]; then
    TARGETREP='target base'
    if [ $AMPLICONS -gt 0 ];then
      TARGETREP='amplicon read'
    fi
    echo "(`date`) Skipping output for $TARGETREP representation and coverage chart..." >&2
  elif [ -n "$ANNOBED" ]; then
    COLLAPSEPFP="collapse"
    if [ $AMPLICONS -ne 2 -a $PROPPLOTS -ne 0 ]; then
      TARGETCOV_GC_PNG="$FILESTEM.gc.png"
      TARGETCOV_LEN_PNG="$FILESTEM.ln.png"
      TARGETCOV_REP_PNG="$FILESTEM.gc_rep.png"
      TARGETCOV_RPL_PNG="$FILESTEM.ln_rep.png"
      TARGETCOV_RPP_PNG="$FILESTEM.pool.png"
      REPOPT="gccovfile='$TARGETCOV_GC_PNG' fedorafile='$TARGETCOV_REP_PNG'"
      if [ -e "$WORKDIR/$TARGETCOV_RPP_PNG" ];then
        REPOPT="$REPOPT poolcovfile='$TARGETCOV_RPP_PNG'"
      fi
      if [ $AMPLICONS -eq 1 ]; then
        REPOPT="$REPOPT fedlenfile='$TARGETCOV_RPL_PNG' lencovfile='$TARGETCOV_LEN_PNG'"
      fi
      echo "<br/> <div id='PictureFrame' $REPOPT $COLLAPSEPFP class='center' style='width:800px;height:300px'></div>" >> "$RESHTML"
    fi
    COLLAPSERCC="collapse"
    TARGETCOVFILE="${OUTFILEROOT}.$TARGTYPE.cov.xls"
    TCCINITFILE="$AUXSTEM.ttc.xls"
    echo "<br/> <div id='TargetCoverageChart' amplicons=$AMPLICONS datafile='$TARGETCOVFILE' initfile='$TCCINITFILE' class='center' style='width:800px;height:300px'></div>" >> "$RESHTML"
  fi

  if [ $BASECOVERAGE -eq 1 -a $NOTARGETANAL -eq 0 ]; then
    if [ -n "$BEDFILE" ];then
      REFGEN=""
    fi
    BBCFILE="$AUXFILEROOT.bbc"
    CBCFILE="$AUXFILEROOT.cbc"
    CHRCOVFILE="$FILESTEM.chr.cov.xls"
    WGNCOVFILE="$FILESTEM.wgn.cov.xls"
    echo "<br/> <div id='ReferenceCoverageChart' $REFGEN $COLLAPSERCC bbcfile='$BBCFILE' annofile='$ANNOBED' chrcovfile='$CHRCOVFILE' wgncovfile='$WGNCOVFILE' class='center' style='width:800px;height:300px'></div>" >> "$RESHTML"
  fi

  echo "<br/> <div id='FileLinksTable' fileurl='filelinks.xls' class='center' style='width:420px'></div>" >> "$RESHTML"
  echo "<br/>" >> "$RESHTML"
  echo '<div></body></html>' >> "$RESHTML"

  if [ $SHOWLOG -eq 1 ]; then
    echo "HTML report complete: " `date` >&2
  fi
fi

# Report error for simple failures, such as in plot generation
#if [ $PLOTERROR -eq 1 ]; then
#  exit 1;
#fi

if [ $SHOWLOG -eq 1 ]; then
  echo -e "\n$CMD END:" `date` >&2
fi

