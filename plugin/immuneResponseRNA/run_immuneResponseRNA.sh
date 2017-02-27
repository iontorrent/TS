#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Description:
Run Immune Resposne RNA analysis for read alignments to (regions of the) reference transcriptome.
Individual plots and data files are produced to the output directory ('.' unless specified by -D).
A targets file is necessary to assign reads to individual targets. If this is a Ion Torrent BED file
the -g option should be used to ensure target GC counts and auxiliary data is retained to the output.
Note: This script depends on other code being accessible from the original parent folder.
"

USAGE="USAGE:
 $CMD [options] <reference.fasta> <BAM file> <BED file>"

OPTIONS="OPTIONS:
  -a Assign reads to targets assuming amplicon reads: Ends and orientation of reads are taken into account.
     Default: Assign reads purely by maximum overlap with targets, resolving ties by most 5' starting target.
  -d Ignore reads marked as Duplicate reads in the BAM file.
  -g Add GC and gene/auxiliary annotation to targets file. Without this option it assumed these fields already
     exist in the input BED file (if they do not then the gene_id and gc_count output fields will be invalid).
  -u Filter to Uniquely mapped reads (SAM MAPQ>0).
  -D <dirpath> Path to root Directory where results are written. Default: ./
  -F <name> File name stem for analysis output files. Default: Use BAM file name provided (w/o extension).
  -G <file> Genome file. Assumed to be <reference.fasta>.fai if not specified.
  -L <name> Reference Library name, e.g. hg19. Defaults to parsing from reference FATSA file name.
  -N <name> Sample name for use in summary output. Default: 'None'
  -O <file> Output file name for text data (per analysis). Default: '' => <BAMNAME>.stats.cov.txt.
  -Q <int>  Read mapping Quality filter: MAPQ has to be >= <N>. Default: 0. (Overrides -u filter.)
  -R <int>  Read length filter: Aligned read length has to be >= <N>. Default: 0.
  -S <file> ERCC tracking regions file. Default: '' (=> no tracking reads statistics reported)
  -T <name> Original Targets BED file used for naming reports and TS link. Default: Use supplied BED file.
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
STATSFILE=""
DEDUP=0
UNIQUE=0
AMPOPT=""
TRACKINGBED=""
SAMPLENAME="None"
ANNOBEDFORMAT=0
LIBRARY=""
FILESTEM=""
TRGSID=""
MAPQMIN=0
ALENMIN=0

TRACK=1
PLOTERROR=0
LINKANNOBED=1

while getopts "hladguD:F:G:L:N:O:Q:R:S:T:" opt
do
  case $opt in
    D) WORKDIR=$OPTARG;;
    F) FILESTEM=$OPTARG;;
    G) GENOME=$OPTARG;;
    L) LIBRARY=$OPTARG;;
    N) SAMPLENAME=$OPTARG;;
    O) STATSFILE=$OPTARG;;
    Q) MAPQMIN=$OPTARG;;
    R) ALENMIN=$OPTARG;;
    S) TRACKINGBED=$OPTARG;;
    T) TRGSID=$OPTARG;;
    a) AMPOPT="-a";;
    d) DEDUP=1;;
    g) ANNOBEDFORMAT=1;;
    u) UNIQUE=1;;
    l) SHOWLOG=1;;
    h) echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
       exit 0;;
    \?) echo $USAGE >&2
        exit 1;;
  esac
done
shift `expr $OPTIND - 1`

if [ $# -ne 3 ]; then
  echo "$CMD: Invalid number of arguments." >&2
  echo -e "$USAGE\n$OPTIONS" >&2
  exit 1;
fi

REFERENCE=$1
BAMFILE=$2
BEDFILE=$3

if [ -z "$GENOME" ]; then
  GENOME="${REFERENCE}.fai"
fi
if [ -z "$LIBRARY" ]; then
  LIBRARY=`echo $REFERENCE | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
  echo "WARNING: -L option not supplied. Reference library name assumed to be '$LIBRARY'." >&2
fi
if [ "$STATSFILE" = "-" ]; then
  STATSFILE=""
fi

RUNPTH=`readlink -n -f $0`
WORKDIR=`readlink -n -f "$WORKDIR"`
REFERENCE=`readlink -n -f "$REFERENCE"`
GENOME=`readlink -n -f "$GENOME"`

RUNDIR=`dirname $RUNPTH`
RUNDIR="${RUNDIR}/scripts"
BAMBAI="${BAMFILE}.bai"
BAMNAME=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMSTEM=`echo $BAMNAME | sed -e 's/\.[^.]*$//'`

if [ "$FILESTEM" = "" -o "$FILESTEM" = "-" ];then
  FILESTEM="$BAMSTEM"
else
  FILESTEM=`echo $FILESTEM | sed -e 's/^.*\///'`
fi
OUTFILEROOT="$WORKDIR/$FILESTEM"

if [ -z "$STATSFILE" ]; then
  STATSFILE="${OUTFILEROOT}.stats.cov.txt"
fi

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
    echo "STATSFILE:  $STATSFILE" >&2
  fi
  echo "RUNDIR:    $RUNDIR" >&2
  echo "WORKDIR:   $WORKDIR" >&2
  echo "FILESTEM:  $FILESTEM" >&2
  echo >&2
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
elif [ -n "$TRACKINGBED" -a ! -f "$TRACKINGBED" ]; then
  echo "ERROR: ERCC tracking targets (bed) file does not exist at $TRACKINGBED" >&2
  exit 1;
fi

# Short descriptons of read filters and translated options

LOGOPT=''
if [ $SHOWLOG -eq 1 ]; then
  LOGOPT='-l'
fi
if [ -n "$ORIGBED" ];then
  TRGSID=`echo $ORIGBED | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
else
  TRGSID=`echo $BEDFILE | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
fi
TARGETTYPE='amplicon'
BEDOPT="-B \"$BEDFILE\""
RFTITLE=""
FILTOPTS="$AMPOPT"
if [ -n "$TRACKINGBED" ]; then
  RFTITLE="ERCC tracking"
fi
if [ "$MAPQMIN" -gt 0 ];then
  RFTITLE="${RFTITLE}, Mapping quality (${MAPQMIN}+)"
  FILTOPTS="$FILTOPTS -Q $MAPQMIN"
elif [ $UNIQUE -eq 1 ]; then
  RFTITLE="${RFTITLE}, Uniquely mapped"
  FILTOPTS="$FILTOPTS -u"
fi
if [ $DEDUP -eq 1 ]; then
  FILTOPTS="$FILTOPTS -d"
  RFTITLE="${RFTITLE}, Non-duplicate"
fi
if [ "$ALENMIN" -gt 0 ];then
  RFTITLE="${RFTITLE}, Alignment length (${ALENMIN}+)"
  FILTOPTS="$FILTOPTS -L $ALENMIN"
fi
RFTITLE=`echo $RFTITLE | sed -e "s/^, //"`

########### Create local annotated bedfile given format (typically for cmd-line usage) ########### 

if [ "$ANNOBEDFORMAT" -ne 0 ];then
  INBEDFILE="$BEDFILE"
  if [ -n "$INBEDFILE" ]; then
    echo "(`date`) Creating GC annotated targets file..." >&2
    ANNOBED="targets.gc.bed"
    ${RUNDIR}/gcAnnoBed.pl -a -s -w -t "$WORKDIR" "$BEDFILE" "$REFERENCE" > "$ANNOBED"
    LINKANNOBED=0
    if [ $SHOWLOG -eq 1 ]; then
      echo "> $ANNOBED" >&2
    fi
  fi
else
  ANNOBED="$BEDFILE"
fi

########### Capture Title & User Options to Summary #########

if [ $SHOWLOG -eq 1 ]; then
  echo "" >&2
fi
echo -e "Coverage Analysis Report\n" > "$STATSFILE"
echo "Sample Name: $SAMPLENAME" >> "$STATSFILE"
REFNAME=`echo "$REFERENCE" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
echo "Reference Genome: $REFNAME" >> "$STATSFILE"
if [ -n "$BEDFILE" ]; then
  echo "Target Regions: $TRGSID" >> "$STATSFILE"
fi
echo "Alignments: $BAMSTEM" >> "$STATSFILE"
if [ -n "$RFTITLE" ];then
  echo "Read Filters: $RFTITLE" >> "$STATSFILE"
fi
echo "" >> "$STATSFILE"

########### Read Coverage Analysis #########

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Generating basic reads stats..." >&2
fi

# basic read mappings from samtools
read TOTAL_READS MAPPED_READS <<<$(samtools flagstat "$BAMFILE" | awk '$0~/in total/||$0~/mapped \(/ {print $1}')
ONTRG_READS=`samtools view -c -F 4 -L "$BEDFILE" "$BAMFILE"`
echo "Number of total reads:         $TOTAL_READS" >> "$STATSFILE"
echo "Number of mapped reads:        $MAPPED_READS" >> "$STATSFILE"
echo "Number of on-target reads:     $ONTRG_READS" >> "$STATSFILE"

# primary read assignment analysis
TARGETCOVFILE="${OUTFILEROOT}.amplicon.cov.xls"
COVCMD="$RUNDIR/targetReadCoverage.pl $FILTOPTS -C 50 \"$BAMFILE\" \"$ANNOBED\" > \"$TARGETCOVFILE\""
if [ $TRACK -eq 1 ]; then
  echo "(`date`) Analyzing $TARGETTYPE coverage..." >&2
fi
eval "$COVCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: $TARGETTYPE analysis failed." >&2
  echo "\$ $COVCMD" >&2
  exit 1;
elif [ $SHOWLOG -eq 1 ]; then
  echo "> $TARGETCOVFILE" >&2
fi
# grab assigned reads and complete basic stats output
ASN_READS=`awk '++c>1 {t+=$10} END {printf "%.0f",t}' "$TARGETCOVFILE"`
echo "Number of assigned reads:      $ASN_READS" >> "$STATSFILE"
PC_ONTRG_READS=`echo "$ONTRG_READS $MAPPED_READS" | awk '{if($2<=0){$1=0;$2=1}printf "%.2f", 100*$1/$2}'`
PC_ASN_READS=`echo "$ASN_READS $MAPPED_READS" | awk '{if($2<=0){$1=0;$2=1}printf "%.2f", 100*$1/$2}'`
echo "Percent reads on target:       $PC_ONTRG_READS%" >> "$STATSFILE"
echo "Percent assigned reads:        $PC_ASN_READS%" >> "$STATSFILE"

# add RPM to target coverage file
RPM_FACTOR=0
if [ "$ASN_READS" -gt 0 ];then
  RPM_FACTOR=`awk "BEGIN{printf \"%.9f\",1000000/$ASN_READS}"`
fi
TMPFILE="fincov.rpm.tmp"
awk "BEGIN {OFS=\"\t\"} {if(++c>1){lc=sprintf(\"%.3f\",$RPM_FACTOR*\$10)}else{lc=\"RPM\"}print \$0,lc}" "$TARGETCOVFILE" > "$TMPFILE"
mv "$TMPFILE" "$TARGETCOVFILE"

# add ERCC mapping stats if expected
if [ -n "$TRACKINGBED" ]; then
  TRACKING_READS=`samtools view -c -F 4 -L "$TRACKINGBED" "$BAMFILE"`
  PC_TRACKING_READS=`echo "$TRACKING_READS $MAPPED_READS" | awk '{if($2<=0){$1=0;$2=1}printf "%.2f", 100*$1/$2}'`
  echo "Number of ERCC tracking reads: $TRACKING_READS" >> "$STATSFILE"
  echo "Percent ERCC tracking reads:   $PC_TRACKING_READS%" >> "$STATSFILE"
fi
echo "" >> "$STATSFILE"

if [ $SHOWLOG -eq 1 ]; then
  echo "(`date`) Sorting $TARGETTYPE coverage results to increasing read depth order..." >&2
fi
TMPFILE="fincov.sort.tmp"
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
  echo "(`date`) Generating amplicon representation overview plot..." >&2
fi
AMPREP_OVE_PNG=`echo $TARGETCOVFILE | sed 's/\.amplicon\.cov\.xls$/.repoverview.png/'`
PLOTCMD="R --no-save --slave --vanilla --args \"$TARGETCOVFILE\" \"$AMPREP_OVE_PNG\" < $RUNDIR/plot_rna_rep.R"
eval "$PLOTCMD" >&2
if [ $? -ne 0 ]; then
  echo "ERROR: plot_rna_rep.R failed." >&2
  PLOTERROR=1
elif [ $SHOWLOG -eq 1 ]; then
  echo "> $AMPREP_OVE_PNG" >&2
fi

########### Depth of Read Coverage Analysis #########

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Analyzing depth of $TARGETTYPE coverage..." >&2
fi
COVERAGE_ANALYSIS="$RUNDIR/targetReadStats.pl -r -M $MAPPED_READS \"$TARGETCOVFILE\""
eval "$COVERAGE_ANALYSIS >> \"$STATSFILE\"" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: targetReadStats.pl failed." >&2
  echo "\$ $COVERAGE_ANALYSIS >> \"$STATSFILE\"" >&2
  exit 1;
fi

# ------------- Analysis Complete -------------

# Create table of primary output files and their descriptions.
# - Temporary code until a json solution is implemented for both templates and cmd-line

PLUGIN_OUT_BEDPAGE=''
PLUGIN_OUT_BEDFILE_MERGED=''
if [ -n "$ORIGBED" ];then
  PLUGIN_OUT_BEDPAGE=`echo $ORIGBED | sed -e 's/^.*\/uploads\/BED\/\([0-9][0-9]*\)\/.*/\/rundb\/uploadstatus\/\1/'`
else
  PLUGIN_OUT_BEDFILE_MERGED=`echo "$BEDFILE" | sed -e 's/^.*\///'`
  ln -sf "$BEDFILE" "${WORKDIR}/$PLUGIN_OUT_BEDFILE_MERGED"
fi
PLUGIN_OUT_BAMFILE="$BAMNAME"
PLUGIN_OUT_BAIFILE="${BAMNAME}.bai"
PLUGIN_OUT_STATSFILE="${FILESTEM}.stats.cov.txt"
PLUGIN_OUT_DOCFILE="${FILESTEM}.base.cov.xls"
PLUGIN_OUT_AMPCOVFILE="${FILESTEM}.amplicon.cov.xls"
source "${RUNDIR}/fileLinks.sh"
write_file_links "$WORKDIR" "filelinks.xls";

# Report error for simple failures, such as in plot generation
#if [ $PLOTERROR -eq 1 ]; then
#  exit 1;
#fi

if [ $SHOWLOG -eq 1 ]; then
  echo -e "\n$CMD END:" `date` >&2
fi

