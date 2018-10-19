#!/bin/bash -E
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
# Wrapper to set up files and run run_ERCC_analysis.py (formerly named ERCC_analysis_plugin.py).

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Run ERCC analysis for read alignments to transcript targets and produce an ERCC
analysis report for a single BAM file. If the BAM file provided does not already have
ERCC transcript reads it is remapped to the ERCC reference (if provided).
If remapped, the original pipeline first filters reads based on length base QV.";

USAGE="USAGE:
 $CMD [options] <BAM file> <ERCC Pool ID>"

OPTIONS="OPTIONS:
  -B <name> Current barcode ID for html report. Default: '' (not a barcoded run)
  -D <dirpath> Path to root Directory where results are written. Default: ./
  -F <name> File name stem for analysis output files. Default: Use BAM file name provided (w/o extension).
  -H <name> Output file name for HTML report. Default: 'report.html'
  -N <name> Sample Name for adding to reports. Default: '' ('' => 'None')
  -M <int> Minimum number of reads for ERCC target to be considered in dose/response analysis. Default: 10
  -O <name> Output file name for text data (per analysis). Default: '' => <file stem>.stats.txt (see -F)
  -R <file> Remapping reference file. Default: '' => error produced if BAM not already mapped to ERCC transcripts
  -T <float> Threshold for alerts based on dose/response R-squaed correlation value. Default: 0.9
  -a Use only the AmpliSeq ERCC subset (10) for reference and correlation plot. Default: use all detected 92 ERCCs.
  -f Use Forward strand reads only for analysis. Default: Use both forward and reverse reads.
  -l Log progress to STDERR. (A few primary progress messages will always be output.)
  -h --help Report full description, usage and options."

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

SHOWLOG=0
BARCODE=""
REFERENCE=""
WORKDIR="."
FILESTEM=""
SAMPLENAME=""
MINREADS=10
R2THRESH=0.9
STATSTEM=""
RESHTML="report.html"
FWD_READS="N"
AMPLICONS=0

while getopts "hlafB:D:F:H:M:N:O:R:T:" opt
do
  case $opt in
    B) BARCODE=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    F) FILESTEM=$OPTARG;;
    H) RESHTML=$OPTARG;;
    M) MINREADS=$OPTARG;;
    N) SAMPLENAME=$OPTARG;;
    O) STATSTEM=$OPTARG;;
    R) REFERENCE=$OPTARG;;
    T) R2THRESH=$OPTARG;;
    a) AMPLICONS=1;;
    l) SHOWLOG=1;;
    f) FWD_READS="Y";;
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

BAMFILE=$1
ERCC_POOL=$2

RUNPTH=`readlink -n -f $0`
RUNDIR=`dirname $RUNPTH`
PLGDIR="$RUNDIR"
RUNDIR="${RUNDIR}/code"
PLGNAME=`echo $PLGDIR | sed -e 's/^.*\///'`

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

if [ -z "$STATSTEM" ]; then
  STATSTEM="${FILESTEM}.stats.txt"
fi
STATSFILE="$WORKDIR/$STATSTEM"

if ! [ -d "$RUNDIR" ]; then
  echo "ERROR: Executables directory does not exist at $RUNDIR" >&2
  exit 1;
elif ! [ -d "$WORKDIR" ]; then
  echo "ERROR: Output work directory does not exist at $WORKDIR" >&2
  exit 1;
elif ! [ -f "$REFERENCE" ]; then
  echo "ERROR: Reference sequence (fasta) file does not exist at $REFERENCE" >&2
  exit 1;
elif ! [ -f "$BAMFILE" ]; then
  echo "ERROR: Mapped reads (bam) file does not exist at $BAMFILE" >&2
  exit 1;
fi

#--------- End command arg parsing ---------

# original code requires specific SAM file to work on
SAMFILE='tmap.sam'
FASTQ="filtered.fastq"

# Record basic inputs to stats file header
echo -e "ERCC Analysis Report\n" > "$STATSFILE"
echo "Sample Name: $SAMPLENAME" >> "$STATSFILE"
echo "Alignments: $BAMNAME" >> "$STATSFILE"

# Check if BAM file appears to be mapped to any ERCC transcripts in reference
NUM_ERCC_REF=`samtools view -H "$BAMFILE" | awk '$1=="@SQ"&&$2~/^SN:ERCC-/ {++c} END {print c+0}'`
echo "ERCC transcripts in BAM reference: $NUM_ERCC_REF" >> "$STATSFILE"
if [ $NUM_ERCC_REF -gt 0 ];then
  echo "Remap to ERCC reference: No" >> "$STATSFILE"
  echo "(`date`) Converting ERCC mapped BAM to SAM..." >&2
  SAMCMD="samtools view -h '$BAMFILE' > '$WORKDIR/$SAMFILE'"
  eval "$SAMCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: samtools view failed. Likely due to issue with BAM file." >&2
    echo "\$ $SAMCMD" >&2
    exit 1;
  elif [ "$SHOWLOG" -eq 1 ]; then
    echo "> $SAMFILE" >&2
  fi
else
  echo "Remap to ERCC reference: Yes" >> "$STATSFILE"
  REFNAME=`echo "$REFERENCE" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
  #echo "ERCC Reference: $REFNAME" >> "$STATSFILE"
  echo "(`date`) Converting BAM to filtered FASTQ..." >&2
  CQFCMD="python $RUNDIR/preproc_fastq.py '$BAMFILE' 'N' '$WORKDIR' &> '$WORKDIR/preproc_fastq.log'"
  eval "$CQFCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: preproc_fastq.py failed. See preproc_fastq.log for details." >&2
    echo "\$ $CQFCMD" >&2
    exit 1;
  fi
  if [ -f "$WORKDIR/$FASTQ" ];then
    if [ "$SHOWLOG" -eq 1 ]; then
      echo "> $FASTQ" >&2
    fi
  fi
  echo "(`date`) Mapping filtered FASTQ to ERCC reference..." >&2
  MAPCMD="tmap mapall -f '$REFERENCE' -r '$WORKDIR/$FASTQ' -s '$WORKDIR/$SAMFILE' -a 1 -g 0 -n 8 stage1 map1 --seed-length 18 stage2 map2 map3 --seed-length 18 2> '$WORKDIR/tmap.log'"
  echo 
  eval "$MAPCMD" >&2
  if [ $? -ne 0 -o ! -f "$WORKDIR/$SAMFILE" ]; then
    echo -e "\nERROR: tmap re-alignment produced no output. See tmap.log for details." >&2
    echo "\$ $MAPCMD" >&2
    exit 1;
  elif [ "$SHOWLOG" -eq 1 ]; then
    echo "> $SAMFILE" >&2
  fi
fi

# Pre-check ERCC SAM file for sufficient number of targets - to catch errors early with better report
NUM_ERCC_REF=`samtools view -S -F 4 "$WORKDIR/$SAMFILE" | awk "{++c[\\$3]} END {for(x in c){if(c[x]>=$MINREADS){++n}}print 0+n}"`
if [ $NUM_ERCC_REF -lt 3 ];then
  echo -e "\nFailed Analysis: Insufficient ERCC targets ($NUM_ERCC_REF) had sufficient mapped reads ($MINREADS)." >> "$STATSFILE"
  exit 0;
fi

# Use ERCC SAM file to create HTML report...
echo "(`date`) Analyzing ERCC coverage and generating report..." >&2

# Args 2 and 3 (ANALYSIS_DIR and URL_ROOT) appear to be unused, arg 9 is always 'Y' since this indicates no error thus far (not if fastq created!)
if [ $AMPLICONS -eq 0 ]; then
  ERCCREF="ercc.genome"
else
  ERCCREF="ampliseq.ercc.genome"
fi
REPORT="$WORKDIR/$RESHTML"
REPCMD="python $PLGDIR/run_ERCC_analysis.py '$WORKDIR' . . $PLGNAME '$R2THRESH' $PLGDIR '$MINREADS' '$ERCC_POOL' 'Y' '$BARCODE' '$FWD_READS' $ERCCREF > '$REPORT' 2> '$WORKDIR/ERCC_analysis.log'"
eval "$REPCMD" >&2
if [ $? -ne 0 ]; then
  echo -e "\nERROR: run_ERCC_analysis.py failed. See ERCC_analysis.log for details." >&2
  echo "\$ $REPCMD" >&2
  exit 1;
elif [ "$SHOWLOG" -eq 1 ]; then
  echo "> $RESHTML" >&2
fi

# Check for and extract error report
ERRMSG=`awk 'NR>10 {exit} /<title>/ {sub(/.*<title>/,"");sub(/<.*/,"");if(/ERCC Plot/){err=1}else{exit}} err==1&&/<p>/ {sub(/.*<p>/,"");sub(/<.*/,"");print;exit}' "$REPORT"`

NUMTARGS="NA"
RSQUARED="NA"
PCONTARG="NA"
PASSR2="NA"

if [ -z "$ERRMSG" ];then
  # harvest statistics
  CORSTATS="$WORKDIR/dose_response.dat"
  if [ -f "$CORSTATS" ];then
    NUMTARGS=`awk '$1=="N" {print 0+$3;exit}' "$CORSTATS"`
    RSQUARED=`awk '$1=="r^2" {print 0+$3;exit}' "$CORSTATS"`
  fi
  PCONTARG=`awk '/ERCC \(pct\)/ {sub(/.*=<\/td><td>/,"");sub(/<.*/,"");print;exit}' "$REPORT"`
  PCONTARG="$PCONTARG%"
  PASSR2=`awk "BEGIN {if($RSQUARED>=$R2THRESH){print \"Yes\"}else{print \"No\"}exit}"`
else
  echo -e "\nFailed Analysis: $ERRMSG\n" >> "$STATSFILE"
fi
echo "Passes Correlation Threshold: $PASSR2" >> "$STATSFILE"
echo "ERCC Targets Detected: $NUMTARGS" >> "$STATSFILE"
echo "Percent ERCC tracking reads: $PCONTARG" >> "$STATSFILE"
echo "Dose-response R-squared: $RSQUARED" >> "$STATSFILE"

# remove temporary files - handled as option in plugin template script
#rm -f "$WORKDIR/$FASTQ" "$WORKDIR/$SAMFILE"
#rm -f "$WORKDIR/preproc_fastq.log" "$WORKDIR/tmap.log" "$WORKDIR/ERCC_analysis.log" "

