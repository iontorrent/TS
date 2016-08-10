#!/bin/bash
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Description:
Run RNA sequencing alignment based on STAR+bowtie2 alignment of the reads to the reference (e.g. hg19),
assuming the reads cDNA and composed of exon-exon splices or fusions. Perform various analysis on the
aligne reads..."

USAGE="USAGE:
 $CMD [options] <reference.fasta> <reads.bam>"

OPTION="OPTIONS:
  -A <str>  Adapter DNA sequence to check for and cut from 5' end of reads. Default: ''.
  -D <dirpath> Path to root Directory where results are written. Default: '' (current directory).
  -F <name> File name stem for analysis output files. Default: Use BAM file name provided (w/o extension).
  -G <file> Genome file. Assumed to be <reference.fasta>.fai if not specified.
  -L <name> Reference Library name, e.g. hg19. Defaults to <reference> if not supplied.
  -O <file> Output file name for text data (per analysis). Default: '' => <BAMNAME>.stats.cov.txt.
  -P <real> Proportion of reads to use in analysis as a fraction. Default: 1.
  -S <dirpath> Path to location where genome indexing/annotation file exist or will be created. Default: -D target.
  -l Log progress to STDERR. (A few primary progress messages will always be output.)
  -h --help Report full description, usage and options."

SHOWLOG=0
ADAPTER=""
WORKDIR="."
FILESTEM=""
GENOME=""
REF_NAME=""
STATSTEM=""
FRACREADS=1
SCRATCH=""

while getopts "hlA:D:F:G:L:O:P:S:" opt
do
  case $opt in
    A) ADAPTER=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    F) FILESTEM=$OPTARG;;
    G) GENOME=$OPTARG;;
    L) REF_NAME=$OPTARG;;
    O) STATSTEM=$OPTARG;;
    P) FRACREADS=$OPTARG;;
    S) SCRATCH=$OPTARG;;
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
  GENOME="${REFERENCE}.fai"
fi
if [ -z "$REF_NAME" ]; then
  REF_NAME=`echo $REFERENCE | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
  echo "WARNING: -L option not supplied. Reference name assumed to be '$REF_NAME'." >&2
fi
if [ "$STATSTEM" = "-" ]; then
  STATSTEM=""
fi

RUNPTH=`readlink -n -f $0`
WORKDIR=`readlink -n -f "$WORKDIR"`
REFERENCE=`readlink -n -f "$REFERENCE"`
GENOME=`readlink -n -f "$GENOME"`

RUNDIR=`dirname $RUNPTH`
BAMNAME=`echo $BAMFILE | sed -e 's/^.*\///'`
BAMSTEM=`echo $BAMNAME | sed -e 's/\.[^.]*$//'`

if [ "$FILESTEM" = "" -o "$FILESTEM" = "-" ];then
  FILESTEM="$BAMSTEM"
else
  FILESTEM=`echo $FILESTEM | sed -e 's/^.*\///'`
fi
OUTFILEROOT="$WORKDIR/$FILESTEM"

if [ -z "$STATSTEM" ]; then
  STATSTEM="${FILESTEM}.stats.cov.txt"
fi
STATSFILE="$WORKDIR/$STATSTEM"

if [ -z "$SCRATCH" ];then
  SCRATCH=$WORKDIR
fi

#--------- End command arg parsing ---------

run ()
{
  echo "running: $*";
  eval "$*";
  EXIT_CODE="$?";
}

# Echo primary input args

if [ $SHOWLOG -eq 1 ]; then
  echo -e "\n$CMD BEGIN:" `date` >&2
  echo "(`date`) $CMD started." >&2
  echo "REF_NAME:  $REF_NAME" >&2
  echo "REFERENCE: $REFERENCE" >&2
  echo "READS:     $BAMNAME" >&2
  if [ -n "$STATSFILE" ];then
    echo "STATSFILE: $STATSFILE" >&2
  fi
  echo "RUNDIR:    $RUNDIR" >&2
  echo "WORKDIR:   $WORKDIR" >&2
  echo "SCRATCH:   $SCRATCH" >&2
  echo "FILESTEM:  $FILESTEM" >&2
  echo >&2
fi

# Specify data source and program

CUTADAPT="PYTHONPATH=$RUNDIR/bin/cutadapt-0.9.5/build/lib.linux-x86_64-2.6 $RUNDIR/bin/cutadapt-0.9.5/cutadapt"
BOWTIE2=$RUNDIR/bin/bowtie2
HTCOUNT="PYTHONPATH=$RUNDIR/bin/HTSeq-0.5.3p9/build/lib.linux-x86_64-2.6 $RUNDIR/bin/HTSeq-0.5.3p9/build/scripts-2.6/htseq-count"
STAR=$RUNDIR/bin/STAR
PICARD=/opt/picard/picard-tools-current

ANNOTATION=$RUNDIR/annotations/$REF_NAME
#SCRATCH=/results/plugins/scratch/RNASeqAnalysis/$REF_NAME
SCRATCH="$SCRATCH/$REF_NAME"

REF_FLAT=$ANNOTATION/refFlat
RNA_INT=$ANNOTATION/rRNA.interval
GENE_GTF=$ANNOTATION/gene.gtf
STAR_INDEX=$SCRATCH/STAR
BOWTIE2_INDEX=$SCRATCH/bowtie2

# extra rRNA references for alignment
XREF_NAME="xrRNA"
XREFERENCE="$ANNOTATION/$XREF_NAME.fasta"
XREF_INDEX="$SCRATCH/$XREF_NAME"

if [ ! -d $ANNOTATION ]; then
  echo "Cannot find $ANNOTATION. Please make sure gene.gtf, refFlat and rRNA.interval files are present in that directory!" >&2
  exit 1
fi

if [ ! -d $STAR_INDEX ]; then
  mkdir -p $SCRATCH
  # set full access to circumvent update resetting owner - too late if this has already happened!
  chmod 777 $SCRATCH
  TMPDIR=`mktemp -d -p $SCRATCH`
  echo "Created temp STAR index dir: $TMPDIR" >&2
  echo "(`date`) Running STAR --runMode genomeGenerate for reference '$REF_NAME' ..." >&2
  $RUNDIR/bin/STAR --runThreadN 12 --runMode genomeGenerate --genomeDir $TMPDIR --genomeFastaFiles $REFERENCE --sjdbGTFfile $GENE_GTF --sjdbOverhang 75
  mv $TMPDIR $STAR_INDEX
  chmod 777 $STAR_INDEX
  echo "" >&2
fi

if [ ! -d $BOWTIE2_INDEX ]; then
  mkdir -p $SCRATCH
  TMPDIR=`mktemp -d -p $SCRATCH`
  echo "Created temp bowtie2 index dir: $TMPDIR" >&2
  echo "(`date`) Running bowtie2-build for reference '$REF_NAME' ..." >&2
  $RUNDIR/bin/bowtie2-build $REFERENCE $TMPDIR/bowtie2 >& $TMPDIR/bowtie2-build.log
  mv $TMPDIR $BOWTIE2_INDEX
  chmod 777 $BOWTIE2_INDEX
  echo "" >&2
fi

if [ -f $XREFERENCE ]; then
 if [ ! -d $XREF_INDEX ]; then
  mkdir -p $SCRATCH
  TMPDIR=`mktemp -d -p $SCRATCH`
  echo "Created temp bowtie2 index dir: $TMPDIR" >&2
  echo "(`date`) Running bowtie2-build for reference '$XREF_NAME' ..." >&2
  $RUNDIR/bin/bowtie2-build $XREFERENCE $TMPDIR/bowtie2 >& $TMPDIR/bowtie2-build.xrRNA.log
  mv $TMPDIR $XREF_INDEX
  chmod 777 $XREF_INDEX
  echo "" >&2
 fi
fi

# ===================================================
# Run the Plugin - here for just one BAM (reads) file
# ===================================================

cd $WORKDIR

###-------- Get fastq files if not already there

input=$BAMSTEM

if [ "$FRACREADS" != "" ] && [ "$FRACREADS" != "1" ]; then
  if [ ! -s $input.$FRACREADS.bam ]; then
    echo "(`date`) Converting reads in BAM file to FASTQ..." >&2
    samtools view -bhs $FRACREADS $input.bam > $input.$FRACREADS.bam
  fi
  input=$BAMSTEM.$FRACREADS
fi

if [ ! -s $input.fastq ];then
  run $RUNDIR/bin/bam2fastq -o $input.fastq $input.bam
fi

###----------- Cut adaptor
if [ "$ADAPTER" != "" ] && [ "$ADAPTER" != "None" ]; then
  if [ ! -s $input.cutadapt.fastq ]; then
    echo "(`date`) Trimming adpater sequence from ends of reads..." >&2
    run $CUTADAPT -m 16 -b $ADAPTER -o $input.cutadapt.fastq $input.fastq
  fi
  input=$input.cutadapt
fi
 
fastqName=$input

################# Align FASTQ Reads #################

echo "(`date`) Running STAR aligner..." >&2
run $STAR --genomeDir $STAR_INDEX --runThreadN 12 --readFilesIn $fastqName.fastq \
  --outSAMunmapped unmappedSTAR.sam --outReadsUnmapped Fastx \
  --chimSegmentMin 18 --chimScoreMin 12 

echo "(`date`) Sorting SAM for STAR aligned reads..." >&2
run "samtools view -bS Aligned.out.sam | samtools sort - alignedSTAR"

######### Set up for Picard tools calls
if [ -e "${PICARD}/picard.jar" ];then
  PICHEAD="picard.jar "
  PICTAIL=""
else
  PICHEAD=""
  PICTAIL=".jar"
fi
 
run_picard ()
{
  cmd=$1
  shift
  run java -jar ${PICARD}/${PICHEAD}${cmd}${PICTAIL} $*
}

### skip generating alignment metrics for STAR only alignment 

#run_picard CollectAlignmentSummaryMetrics \
#I=alignedSTAR.bam O=${fastqName}_STARonly.alignmentSummary.txt \
#R=${REFERENCE} LEVEL=ALL_READS
 
echo "(`date`) Running bowtie2 aligner..." >&2
run "$BOWTIE2 --local --very-sensitive-local -p 8 -q --mm -x $BOWTIE2_INDEX/bowtie2 -U Unmapped.out.mate1 --un sbt2_unmap.fq \
 | samtools view -bS - | samtools sort - unmapped_remapBowtie2"

### skip generating alignment metrics for BOWTIE2 only alignment 

#run_picard CollectAlignmentSummaryMetrics \
# I=unmapped_remapBowtie2.bam O=${fastqName}_Bowtie2only.alignmentSummary.txt \
# R=${REFERENCE} LEVEL=ALL_READS
 
echo "(`date`) Merging and indexing STAR and bowtie2 aligned reads..." >&2
run_picard MergeSamFiles USE_THREADING=true MSD=true AS=true \
 I=alignedSTAR.bam I=unmapped_remapBowtie2.bam \
 O=${fastqName}.STARBowtie2.bam
 
run samtools index ${fastqName}.STARBowtie2.bam
 
## Perform extra alignment for xrRNA sequences
if [ -d "$XREF_INDEX" -a -f "sbt2_unmap.fq" ]; then
  echo "(`date`) Aligning unmapped reads to supplementary reference..." >&2
  # need sorted BAM for samtools depth
  run "$BOWTIE2 --local --very-sensitive-local -p 8 -q --mm -x $XREF_INDEX/bowtie2 -U sbt2_unmap.fq | samtools view -bS - | samtools sort - '$XREF_NAME'"
  if [ -f "$XREF_NAME.bam" ]; then
    run samtools depth -G 4 xrRNA.bam | awk '{c+=$3} END {print c}' > "$XREF_NAME.basereads"
  else
    echo "0" > "$XREF_NAME.basereads"
  fi
  rm -f "sbt2_unmap.fq"
  #rm -f "$XREF_NAME.bam"
fi

##- Generate alignment Stats and RNA metrics
echo "(`date`) Analyzing aligned reads..." >&2
run_picard CollectAlignmentSummaryMetrics \
 I=${fastqName}.STARBowtie2.bam O=${fastqName}.STARBowtie2.alignmentSummary.txt \
 R=${REFERENCE} LEVEL=ALL_READS
 
run_picard CollectRnaSeqMetrics REF_FLAT=$REF_FLAT \
 RIBOSOMAL_INTERVALS=$RNA_INT STRAND=FIRST_READ_TRANSCRIPTION_STRAND \
 MINIMUM_LENGTH=100 LEVEL=ALL_READS \
 I=${fastqName}.STARBowtie2.bam R=${REFERENCE} O=${fastqName}.STARBowtie2.RNAmetrics.txt
 
######################### htcount

run "samtools view -F4 ${fastqName}.STARBowtie2.bam | $HTCOUNT -q -t exon -i gene_name - $GENE_GTF \
 > ${fastqName}.STARBowtie2.gene.count"

########### cufflinks ########
CUFFLINKS=$RUNDIR/bin/cufflinks-2.2.1/cufflinks
RRNA_MASK=${ANNOTATION}/rRNA_mask.gtf
OUT_CUFFLINKS=output_cufflinks
if [ ! -d $OUT_CUFFLINKS ]; then
  mkdir -p $OUT_CUFFLINKS
fi

run "$CUFFLINKS -q -p 12 -m 100 -s 60 -G $GENE_GTF -M $RRNA_MASK \
  --library-type fr-secondstrand --max-bundle-length 3500000 \
  -o $OUT_CUFFLINKS --no-update-check \
  ${fastqName}.STARBowtie2.bam"

###### rename cufflinks output files and link to parent folder (for HUB scraping)
for file in $OUT_CUFFLINKS/*
do
  fname=`echo $file | sed -e 's/^.*\///'`
  nname="$FILESTEM.$fname"
  mv $file "$OUT_CUFFLINKS/$nname"
  ln -s "$OUT_CUFFLINKS/$nname" "./$nname"
done

###### remove unneeded files

rm -f Aligned.out.sam
rm -f Unmapped.out.mate1
rm -rf _tmp

cd -

