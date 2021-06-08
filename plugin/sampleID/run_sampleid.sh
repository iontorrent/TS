#!/bin/bash -E
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR=" Run sample ID analysis for read alignments to sample tracking targets and produce
coverage and calling summary files to the output direcory. Two target files are required
to specify both expected targets and the subset of SNPs that produce the genotype sample ID.
This script depends on other code accessible from the original parent (sampleID) folder."

USAGE="USAGE:
 $CMD [options] <reference.fasta> <BAM file> <targets BED file> <SNPs BED file>"

OPTIONS="OPTIONS:
  -A <file> Output file for SNP Allele coverage and calls. Default: 'allele_counts.xls'.
  -D <dirpath> Path to root Directory where results are written. Default: ./
  -F <name> File name stem for analysis output files. Default: Use BAM file name provided (w/o extension).
  -N <name> Sample Name for adding to reports. Default: '' ('' => 'None').
  -R <file> Output file name for general Read coverage. Default: 'read_stats.txt'
  -S <file> Output file name for SNPS BED coverage. Default: 'on_loci_stats.txt'
  -T <file> Output file name for Targets BED coverage. Default: 'on_target_stats.txt'
  -l Log progress to STDERR. (A few primary progress messages will always be output.)
  -h --help Report full description, usage and options."

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

WORKDIR="."
FILESTEM=""
SAMPLE_NAME=""
ALLELE_COV_OUT="allele_counts.xls"
READ_STATS_OUT="read_stats.txt"
SNP_STATS_OUT="on_loci_stats.txt"
TARGET_STATS_OUT="on_target_stats.txt"

while getopts "hlA:D:F:G:L:N:R:S:T:" opt
do
  case $opt in
    A) ALLELE_COV_OUT=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    F) FILESTEM=$OPTARG;;
    N) SAMPLE_NAME=$OPTARG;;
    R) READ_STATS_OUT=$OPTARG;;
    S) SNP_STATS_OUT=$OPTARG;;
    T) TARGET_STATS_OUT=$OPTARG;;
    l) SHOWLOG=1;;
    h) echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
       exit 0;;
    \?) echo $USAGE >&2
        exit 1;;
  esac
done
shift `expr $OPTIND - 1`

if [ $# -ne 4 ]; then
  echo "$CMD: Invalid number of arguments." >&2
  echo -e "$USAGE\n$OPTIONS" >&2
  exit 1;
fi

REFERENCE=$1
BAMFILE=$2
TARGETS_BEDFILE=$3
SNPS_BEDFILE=$4

RUNPTH=`readlink -n -f $0`
RUNDIR=`dirname $RUNPTH`
RUNDIR="${RUNDIR}/scripts"

WORKDIR=`readlink -n -f "$WORKDIR"`
BAMBAI="${BAMFILE}.bai"
BAMNAME=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMSTEM=`echo $BAMNAME | sed -e 's/\.[^.]*$//'`

if [ "$FILESTEM" = "" -o "$FILESTEM" = "-" ];then
  FILESTEM="$BAMSTEM"
else
  FILESTEM=`echo $FILESTEM | sed -e 's/^.*\///'`
fi
OUTFILEROOT="$WORKDIR/$FILESTEM"

if [ $CHIP_LEVEL_ANALYSIS_PATH ]; then
  samtoolsPath="${DIRNAME}/bin/samtools"
else
  samtoolsPath="samtools"
fi



#--------- End command arg parsing ---------
    
# Generate allele counts if hotspots loci BED provided
echo "Generating base pileup for SNP loci..." >&2
RAW_COV_OUT="${WORKDIR}/snps_raw_cov.txt"
MPILEUP_EXE="${WORKDIR}/mpileup.sh"
MPILEUP_OUT="${WORKDIR}/mpileup.txt"
awk -v ref="$REFERENCE" -v bam="$BAMFILE" '$0!~/^track/ {print "${samtoolsPath} mpileup -BQ0 -d1000000 -f "ref" -r "$1":"$2"-"$3" "bam}' "$SNPS_BEDFILE" > "$MPILEUP_EXE"
source "$MPILEUP_EXE" > "$MPILEUP_OUT" 2> /dev/null
# mpileup is no longer piped because separating bed targets into separate calls per region is >17x faster
${RUNDIR}/allele_from_mpileup.py "$MPILEUP_OUT" > "$RAW_COV_OUT"

# Note: if not already done or unecessary, SNPs BED file should be left aligned using the following command:
# java -jar -Xmx1500m $RUNDIR/LeftAlignBed.jar "$SNPS_BEDFILE" "${WORKDIR}/leftalign.bed" $RUNDIR/GenomeAnalysisTK.jar $REFERENCE
${RUNDIR}/writeAlleles.py "$RAW_COV_OUT" "${WORKDIR}/$ALLELE_COV_OUT" "$SNPS_BEDFILE"

# Generate simple coverage statistics, including number of male/female reads
echo "Generating coverage statistics and sample identification calls..." >&2

${RUNDIR}/read_analysis.sh $LOGOPT -g -T "sample ID region" -O "$READ_STATS_OUT" -B "$TARGETS_BEDFILE" -D "$WORKDIR" "$REFERENCE" "$BAMFILE"

echo "Completed read_analysis.sh" >&2
# Make the sample ID call string - including gender
HAPLOCODE=`${RUNDIR}/extractBarcode.pl -R "${WORKDIR}/$READ_STATS_OUT" "${WORKDIR}/$ALLELE_COV_OUT"`

if [ -z "$SAMPLE_NAME" ];then
  SAMPLE_NAME="None"
fi
TMPFILE="${WORKDIR}/sampleid.tmp"
echo "Sample Name: $SAMPLE_NAME" > "$TMPFILE"
echo "Sample ID:   $HAPLOCODE" >> "$TMPFILE"
cat "${WORKDIR}/$READ_STATS_OUT" >> "$TMPFILE"
mv "$TMPFILE" "${WORKDIR}/$READ_STATS_OUT"

# Target coverage stats, discounting chrY targets for female sample for correct uniformity
TARGETS_BED="$TARGETS_BEDFILE"
if [[ "$HAPLOCODE" =~ ^F ]]; then
  TARGETS_BED="$OUTFILEROOT.noY.bed"
  awk '$1!~/^chrY/ {print}' "$TARGETS_BEDFILE" > "$TARGETS_BED"
fi
OUTCMD=">> \"${WORKDIR}/$TARGET_STATS_OUT\""
gnm_size=`awk 'BEGIN {gs = 0} NR>1 {gs += $3-$2} END {printf "%.0f",gs+0}' "$TARGETS_BED"`

#COVERAGE_ANALYSIS="\"${samtoolsPath}\" depth -G 4 -b \"$TARGETS_BED\" \"$BAMFILE\" 2> /dev/null | awk -f ${RUNDIR}/coverage_analysis.awk -v genome=$gnm_size"
# as with mpileup, samtools dept is much quicker using multiple region analysis than bed files
STDEPTH_EXE="${WORKDIR}/stdepth.sh"
awk -v sam=$samtoolsPath -v bam="$BAMFILE" '$0!~/^track/ {print sam" depth -G 4 -r "$1":"$2+1"-"$3" "bam}' "$TARGETS_BEDFILE" > "$STDEPTH_EXE"
COVERAGE_ANALYSIS="source '$STDEPTH_EXE' 2> /dev/null | awk -f ${RUNDIR}/coverage_analysis.awk -v genome=$gnm_size"

eval "$COVERAGE_ANALYSIS $OUTCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: Command failed:" >&2
  echo "\$ $COVERAGE_ANALYSIS $OUTCMD" >&2
  exit 1;
fi

# SNPs coverage stats
OUTCMD=">> \"${WORKDIR}/$SNP_STATS_OUT\""
gnm_size=`awk 'BEGIN {gs = 0} NR>1 {gs += $3-$2} END {printf "%.0f",gs+0}' "$SNPS_BEDFILE"`

#COVERAGE_ANALYSIS="\"${samtoolsPath}\" depth -G 4 -b \"$SNPS_BEDFILE\" \"$BAMFILE\" 2> /dev/null | awk -f ${RUNDIR}/coverage_analysis.awk -v genome=$gnm_size"
# SNV coverage is measured now from Cov+ + Cov-, since inserts are ignored and deletions do not count by definition (samtools depth)
STDEPTH_OUT="${WORKDIR}/stdepth.out"
awk 'NR>1 {print $1,$2,$15+$16}' "${WORKDIR}/$ALLELE_COV_OUT" > "$STDEPTH_OUT"
COVERAGE_ANALYSIS="awk -f ${RUNDIR}/coverage_analysis.awk -v genome=$gnm_size '$STDEPTH_OUT'"

eval "$COVERAGE_ANALYSIS $OUTCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: Command failed:" >&2
  echo "\$ $COVERAGE_ANALYSIS $OUTCMD" >&2
  exit 1;
fi

# temporary file clean
if [[ "$HAPLOCODE" =~ ^F ]]; then
  rm -f "$TARGETS_BED"
fi
rm -f "$RAW_COV_OUT" "$MPILEUP_EXE" "$MPILEUP_OUT" "$STDEPTH_EXE" "$STDEPTH_OUT"

