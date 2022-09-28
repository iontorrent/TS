#!/bin/bash
# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Description:
Use data in existing coverageAnalysis with local BED files to calculate gene base uniformity (GBU) and panel base uniformity.
Individual plots and data files are produced to the output directory ('.' unless specified by -D).
"

USAGE="USAGE:
 $CMD [options] <coverageAnalysis/barcode path> <panel targets BED>"

OPTIONS="OPTIONS:
  -a Auto-correct GENE_ID values to assumed HGNC for master gene CDS list matching. (Remove _*)
  -d Output gene coverage WIG files based on assay design rather than GBU region. Overrides -w option.
  -c Generate CDS output (0 padding) in additon to that with -P padding.
  -g Create GBU based on the full amplicon designs. Output to <prefix>.amp.gbu.csv - see -F option.
  -i Reduce CDS targets to the interesection with the panel targets. (Affects GBU output file format.)
  -w Output gene coverage WIG files as <gene>.wig to the output folder. See -d option.
  -D <dirpath> Path to root Directory where results are written. Default: ./
  -F <name> Output File name prefix. Default: 'GPU'
  -M <num>  Mean base read depth for GBU calculations. Default: '' => Use mean over each gene region.
  -N <name> Sample name for use in summary output. Default: 'None'
  -O <file> Output file name for text data (per analysis). Default: '' => '<file name prefix>.stats.txt'.
  -P <int>  Padding value to extend <new targets BED file> regions by. Default: 0 => CDS+<pad> not generated.
  -T <file> Tube dispensing dropouts file. If provided, filter out amplicons in tube dropouts (for plate/rack).
  -l Log progress to STDERR. (A few primary progress messages will always be output.)
  -h --help Report full description, usage and options."

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

TRACK=1

SHOWLOG=0
STATSTEM=""
SAMPLENAME="None"
FILESTEM="GPU"
WORKDIR="."
PAD0=0
PADDING=0
MBRD=""
CGENEID=""
CDSISCT=0
TUBEDROP=""
WIGOUT=0
DESWIG=0
DESGBU=0

while getopts "acdghilwD:F:M:N:O:P:T:" opt
do
  case $opt in
    a) CGENEID="-a";;
    c) PAD0=1;;
    d) DESWIG=1;;
    g) DESGBU=1;;
    i) CDSISCT=1;;
    w) WIGOUT=1;;
    D) WORKDIR=$OPTARG;;
    F) FILESTEM=$OPTARG;;
    M) MBRD=$OPTARG;;
    N) SAMPLENAME=$OPTARG;;
    O) STATSTEM=$OPTARG;;
    P) PADDING=$OPTARG;;
    T) TUBEDROP=$OPTARG;;
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

TCADIR=$1
TCABED=$2

WORKDIR=`readlink -n -f "$WORKDIR"`

RUNPTH=`readlink -n -f $0`
RUNDIR=`dirname $RUNPTH`
BINDIR="$RUNDIR/bin"

BARCODE=`basename "$TCADIR"`
TCAPATH=`dirname "$TCADIR"`
TCASRCP=`basename "$TCAPATH"`

if [ -z "$STATSTEM" ]; then
  STATSTEM="$FILESTEM.stats.txt"
fi
STATSFILE="$WORKDIR/$STATSTEM"

FILEOUT="$WORKDIR/$FILESTEM"
TMPBED="$WORKDIR/tmp.bed"
TMPTXT="$WORKDIR/tmp.txt"
TMPTSV="$WORKDIR/tmp.tsv"

TRGSID=`echo "$TCABED" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`

# if interesection flagged assume GBU over full gene is also wanted
GLENOPT=""
if [ $CDSISCT -gt 0 ]; then
  GLENOPT="-G $TMPTSV"
fi
# Add "PP" to indicate a partial plate number if bed file name is an integer
if [[ "$TRGSID" =~ ^[0-9] ]]; then
  TRGSID="PP$TRGSID"
  # for partial plate extra full CDS fields are not useful(?)
  GLENOPT=""
fi
if [ $DESWIG -gt 0 ]; then
  WIGOUT=1
fi
WIGDIR=""
if [ $WIGOUT -gt 0 ]; then
  WIGDIR="-W $WORKDIR"
fi

MASTERCDSBED="$RUNDIR/data/refGene.20170417.symbol.cleaned.uniqcds_target.tab"
MASTERHSBED="$RUNDIR/data/univ_hotspots.20190118.annotations.CONFIDENTIAL.bed"

########## Autocorrect MASTERHSCBED if the length of a hotspot is zero ##########
MASTERHSCBED="$WORKDIR/univ_hotspots.20190118.noamplicons.CONFIDENTIAL.corrected.bed"
awk '{if ($2!=$3) {print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6} else {print $1 "\t" $2 "\t" $3+1 "\t" $4 "\t" $5 "\t" $6}}' "$MASTERHSBED" > "$MASTERHSCBED"

# if local (override) version of bbctools not present default to system version
BBCTOOLS="$BINDIR/bbctools"
if [ ! -f "$BBCTOOLS" ]; then
  BBCTOOLS="bbctools"
  command -v bbctools || { echo "ERROR: 'bbctools' is not available as a command on this system!" >&2; exit 1; }
fi
BBCFILE="${TCADIR}/tca_auxiliary.bbc"

########### Capture Title & User Options to Summary #########

if [ $SHOWLOG -eq 1 ]; then
  echo "" >&2
fi
echo -e "PhoenixGBU Report\n" > "$STATSFILE"
echo "Barcode: $BARCODE" >> "$STATSFILE"
echo "Sample Name: $SAMPLENAME" >> "$STATSFILE"
echo "Panel Target Regions: $TRGSID" >> "$STATSFILE"
echo "Coverage Analysis Source: $TCASRCP" >> "$STATSFILE"
if [ $PADDING -gt 0 ]; then
  echo "CDS Target Padding: $PADDING" >> "$STATSFILE"
fi
#eval "$BBCTOOLS version" >> "$STATSFILE"
echo "" >> "$STATSFILE"
if [ -n "$MBRD" ];then
  echo "Mean base read depth for GBU: $MBRD" >> "$STATSFILE"
fi

########### Generate GBU regions for CDS regions of genes matched to panel ###########

# CDS regions done first just for convenience of adding gene matching data to STATSFILE earlier

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Extracting BED for CDS GBU and gene ID validation" >&2
fi
CDSBED="$FILEOUT.cds.gbu.bed"
UNMATCH="$FILEOUT.unmatched_geneids.xls"
$RUNDIR/createGeneBed.pl $CGENEID -S "$TMPTXT" -U "$UNMATCH" "$TCABED" "$MASTERCDSBED" > "$TMPBED"
sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$CDSBED"
cat "$TMPTXT" >> "$STATSFILE"
echo "" >> "$STATSFILE"

# generate a bed file filted for amplicons in tube dropouts
# - from here on this becomes the definition of the panel, i.e. for PGI MBRD coverage
FTDBED="$TCABED"
if [ -n "$TUBEDROP" ]; then
  FTDBED="$FILEOUT.ftd_amps.bed"
  $RUNDIR/filterTubeDrop.pl -S "$STATSFILE" "$TCABED" "$TUBEDROP" > "$FTDBED"
fi
# set up for whole amplicons WIG
WIGBEDOPT=""
if [ $DESWIG -gt 0 ]; then
  WIGBEDOPT="-X $FTDBED"
fi

if [ $PAD0 -eq 1 ]; then
  if [ $TRACK -eq 1 ]; then
    echo "(`date`) Extracting effective Panel-Gene-Intersection for CDS GBU" >&2
  fi
  # grab all CDS regions overlapped by all amplicons for full CDS lengths
  CDSBED="$FILEOUT.cds.gbu.bed"
  $RUNDIR/createGeneBed.pl $CGENEID "$TCABED" "$MASTERCDSBED" > "$TMPBED"
  sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$CDSBED"
  awk '$4!=c {if(c){print c"\t"d}c=$4;d=0} {d+=$3-$2} END {if(c){print c"\t"d}}' "$CDSBED" > "$TMPTSV"
  # grab all CDS regions overlapped by amplicon not ignored due to tube dropouts
  if [ -n "$TUBEDROP" ]; then
    $RUNDIR/createGeneBed.pl $CGENEID "$FTDBED" "$MASTERCDSBED" > "$TMPBED"
    sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$CDSBED"
  fi
  # get total CDS bases covered for weighted coverage
  GENECOV=`bedtools sort -i "$CDSBED" | bedtools merge | awk '{c+=$3-$2} END {print c}'`
  # get the panel-gene-intersection to avoid coverage by (non-overlaping) partial primer digestion
  if [ $CDSISCT -eq 1 ]; then
    bedtools intersect -a "$CDSBED" -b "$FTDBED" > "$TMPBED"
    sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$CDSBED"
  fi
  CDSCSV="$FILEOUT.cds.gbu.csv"
  CDSSTS="$FILEOUT.cds.pbu.txt"
  if [ $TRACK -eq 1 ]; then
    echo "(`date`) Calculating Gene Base Uniformity for $(basename $CDSBED)" >&2
  fi
  $RUNDIR/calculateGBU.pl -E "$BBCTOOLS" $GLENOPT -L $GENECOV -M "$MBRD" -T 0.2 -S "$CDSSTS" $WIGDIR $WIGBEDOPT "$BBCFILE" "$CDSBED" > "$CDSCSV"
  PGIBAS=`awk '{if(sub("Bases in target regions: *","")){print}}' "$CDSSTS"`
  PGIPBU=`awk '{if(sub("Uniformity of base coverage: *","")){print}}' "$CDSSTS"`
  PRJPBU=`echo "$PGIPBU $PGIBAS $GENECOV" | awk '{printf "%.2",$1*$2/$3}'`
  echo "PGI Panel Base Uniformity (CDS+5): $PGIPBU" >> "$STATSFILE"
  echo "Panel Base Uniformity (CDS+5):     $PRJPBU" >> "$STATSFILE"
fi

########### Generate GBU regions for designs == amplicon inserts ###########

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Extracting BED for design GBU" >&2
fi
AMPBED="$FILEOUT.amp.gbu.bed"
$RUNDIR/createGeneBed.pl $CGENEID "$TCABED" > "$TMPBED"
sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$AMPBED"

if [ $TRACK -eq 1 ]; then
  echo "(`date`) Calculating Gene Base Uniformity for $(basename $AMPBED)" >&2
fi
AMPCSV="$FILEOUT.amp.gbu.csv"
AMPSTS="$FILEOUT.amp.pbu.txt"
if [ $DESGBU -gt 0 ]; then
  $RUNDIR/calculateGBU.pl -E "$BBCTOOLS" -M "$MBRD" -T 0.2 -S "$AMPSTS" "$BBCFILE" "$AMPBED" > "$AMPCSV"
else
  $RUNDIR/calculateGBU.pl -p -E "$BBCTOOLS" "$BBCFILE" "$AMPBED" > "$AMPSTS"
fi

awk '{if(sub("Average base coverage depth: *","Panel Mean Base Read Depth:        ")){print}}' "$AMPSTS" >> "$STATSFILE"
awk '{if(sub("Uniformity of base coverage: *","Panel Base Uniformity (Design):    ")){print}}' "$AMPSTS" >> "$STATSFILE"
if [ $PAD0 -eq 1 ]; then
  awk '{if(sub("Uniformity of base coverage: *","Panel Base Uniformity (CDS):       ")){print}}' "$CDSSTS" >> "$STATSFILE"
fi

########### Generate GBU regions for CDS+5 regions of genes matched to panel ###########

if [ "$PADDING" -gt 0 ]; then
  if [ $TRACK -eq 1 ]; then
    echo "(`date`) Extracting effective Panel-Gene-Intersection for CDS+$PADDING GBU" >&2
  fi
  # grab all CDS+5 regions overlapped by all amplicons for full CDS+pad lengths
  CDSPADBED="$FILEOUT.cds_pad$PADDING.gbu.bed"
  $RUNDIR/createGeneBed.pl $CGENEID -P $PADDING "$TCABED" "$MASTERCDSBED" > "$TMPBED"
  sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$CDSPADBED"
  awk '$4!=c {if(c){print c"\t"d}c=$4;d=0} {d+=$3-$2} END {if(c){print c"\t"d}}' "$CDSPADBED" > "$TMPTSV"
  # grab all CDS+5 regions overlapped by amplicon not ignored due to tube dropouts
  if [ -n "$TUBEDROP" ]; then
    $RUNDIR/createGeneBed.pl $CGENEID -P $PADDING "$FTDBED" "$MASTERCDSBED" > "$TMPBED"
    sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$CDSPADBED"
  fi
  # get total CDS+5 bases covered for weighted coverage
  GENECOV=`bedtools sort -i "$CDSPADBED" | bedtools merge | awk '{c+=$3-$2} END {print c}'`
  # get the panel-gene-intersection to avoid coverage by (non-overlaping) partial primer digestion
  if [ $CDSISCT -eq 1 ]; then
    bedtools intersect -a "$CDSPADBED" -b "$FTDBED" > "$TMPBED"
    sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$CDSPADBED"
  fi
  if [ $TRACK -eq 1 ]; then
    echo "(`date`) Calculating Gene Base Uniformity for $(basename $CDSPADBED)" >&2
  fi
  CDSPADCSV="$FILEOUT.cds_pad$PADDING.gbu.csv"
  CDSPADCSV_DOWN="$FILEOUT.cds_pad$PADDING.gbu.down.csv"
  CDSPADSTS="$FILEOUT.cds_pad$PADDING.pbu.txt"
  $RUNDIR/calculateGBU.pl -E "$BBCTOOLS" $GLENOPT -L $GENECOV -M "$MBRD" -T 0.2 -S "$CDSPADSTS" $WIGDIR $WIGBEDOPT "$BBCFILE" "$CDSPADBED" > "$CDSPADCSV"
  awk 'BEGIN {FS = ","}; {print $1 "," $2 "," $3 "," $14 "," $7 "," $8 "," $9 "," $10}' "$CDSPADCSV"  > "$CDSPADCSV_DOWN"
  PGIBAS=`awk '{if(sub("Bases in target regions: *","")){print}}' "$CDSPADSTS"`
  PGIPBU=`awk '{if(sub("Uniformity of base coverage: *","")){print}}' "$CDSPADSTS"`
  PGICOV=`echo "$PGIBAS $GENECOV" | awk '{printf "%.2f",100*$1/$2}'`
  PRJPBU=`echo "$PGIPBU $PGIBAS $GENECOV" | awk '{printf "%.2f",$1*$2/$3}'`
  echo "Panel-gene-intersection (CDS+5):   $PGICOV%" >> "$STATSFILE"
  echo "PGI Panel Base Uniformity (CDS+5): $PGIPBU" >> "$STATSFILE"
  echo "Panel Base Uniformity (CDS+5):     $PRJPBU%" >> "$STATSFILE"
fi

########### Generate Hotspot Base Uniformity ###########

MASTERUNIQHSBED="$FILEOUT.uniq.hotspot.bed"
MASTERHSCDSBED="$FILEOUT.hotspot.intersect.cds.bed"
awk '{print $1 "\t" $2 "\t" $3}' $MASTERHSCBED | sort -k1,1 -k2,2g -k3,3g | bedtools merge > $MASTERUNIQHSBED
awk '{print $4 "\t" $7 "\t" $8 "\t" $3}' $MASTERCDSBED | bedtools intersect -a - -b $MASTERUNIQHSBED -wo |  awk '{print 0 "\t" 0 "\t" $4 "\t" $5 "\t-\tht\t" $6 "\t" $7}' | sort | uniq > $MASTERHSCDSBED
# grab all Hospot regions overlapped by all amplicons for full CDS+pad lengths
HSBED="$FILEOUT.gene.hbu.bed"
$RUNDIR/createGeneBed.pl $CGENEID "$TCABED" "$MASTERHSCDSBED" > "$TMPBED"
sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$HSBED"
awk '$4!=c {if(c){print c"\t"d}c=$4;d=0} {d+=$3-$2} END {if(c){print c"\t"d}}' "$HSBED" > "$TMPTSV"
# grab all HS regions overlapped by amplicon not ignored due to tube dropouts
if [ -n "$TUBEDROP" ]; then
  $RUNDIR/createGeneBed.pl $CGENEID "$FTDBED" "$MASTERHSCDSBED" > "$TMPBED"
  sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$HSBED"
fi
# get total hotspot bases covered for weighted coverage
GENECOV=`bedtools sort -i "$HSBED" | bedtools merge | awk '{c+=$3-$2} END {print c}'`
# get the panel-gene-intersection to avoid coverage by (non-overlaping) partial primer digestion
if [ $CDSISCT -eq 1 ]; then
  bedtools intersect -a "$HSBED" -b "$FTDBED" > "$TMPBED"
  sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" > "$HSBED"
fi
if [ $TRACK -eq 1 ]; then
  echo "(`date`) Calculating Hotspot Base Uniformity for $(basename $HSBED)" >&2
fi
HSCSV1="$FILEOUT.gene.hbu.0.1x.csv"
HSCSV2="$FILEOUT.gene.hbu.0.2x.csv"
HSCSV1_DOWN="$FILEOUT.gene.hbu.0.1x.down.csv"
HSCSV2_DOWN="$FILEOUT.gene.hbu.0.2x.down.csv"
HSSTS="$FILEOUT.panel.hbu.txt"

$RUNDIR/calculateGBU.pl -E "$BBCTOOLS" $GLENOPT -L $GENECOV -M "$MBRD" -T 0.1 -S "$HSSTS" "$BBCFILE" "$HSBED" > "$HSCSV1"    # run calculateGBU to generate file for collectGBUstats.pl, which is run in GBU_HBU_Analysis_plugin.py
$RUNDIR/calculateGBU.pl -E "$BBCTOOLS" $GLENOPT -L $GENECOV -M "$MBRD" -T 0.2 -S "$HSSTS" "$BBCFILE" "$HSBED" > "$HSCSV2"    # run calculateGBU to generate file for collectGBUstats.pl, which is run in GBU_HBU_Analysis_plugin.py

$RUNDIR/calculateHBU.pl -E "$BBCTOOLS" -L "$GENECOV" -T 0.1 -F "$TCABED" -S "$HSSTS" "$BBCFILE" "$HSBED" | awk 'BEGIN {FS = ","}; {print $1 "," $2 "," $3 "," $7 "," $9 "," $10 "," $11 "," $12 "," $13}' | sed 's/PGIBU/HBU/g' | sed 's/PGILen/HLen/g' > "$HSCSV1_DOWN"
$RUNDIR/calculateHBU.pl -E "$BBCTOOLS" -L "$GENECOV" -T 0.2 -F "$TCABED" -S "$HSSTS" "$BBCFILE" "$HSBED" | awk 'BEGIN {FS = ","}; {print $1 "," $2 "," $3 "," $7 "," $9 "," $10 "," $11 "," $12 "," $13}' | sed 's/PGIBU/HBU/g' | sed 's/PGILen/HLen/g' > "$HSCSV2_DOWN"
PGIBAS=`awk '{if(sub("Bases in target regions: *","")){print}}' "$HSSTS"`
PGIPBU=`awk '{if(sub("Uniformity of base coverage: *","")){print}}' "$HSSTS"`
PGICOV=`echo "$PGIBAS $GENECOV" | awk '{printf "%.2f",100*$1/$2}'`
PRJPBU=`echo "$PGIPBU $PGIBAS $GENECOV" | awk '{printf "%.2f",$1*$2/$3}'`
echo "PGI Panel Base Uniformity (Hotspot): $PGIPBU" >> "$STATSFILE"

########### Generate Hotspot Minimal Coverage ###########

MASTERHSCDSBED="$FILEOUT.hotspot.intersect.cds.bed"
HSBED="$FILEOUT.gene.hbu.bed"
awk '{print $4 "\t" $7 "\t" $8 "\t" $3}' "$MASTERCDSBED" | bedtools intersect -b - -a "$MASTERHSCBED" -wa |  awk '{print $1 "\t" $2 "\t" $3 "\t" $4}' | sort | uniq > "$MASTERHSCDSBED"
bedtools intersect -a "$MASTERHSCDSBED" -b "$TCABED" -wa | sort | uniq > "$HSBED"
# grab all HS regions overlapped by amplicon not ignored due to tube dropouts
if [ -n "$TUBEDROP" ]; then
  bedtools intersect -a "$MASTERHSCDSBED" -b "$FTDBED" -wa | sort | uniq > "$HSBED"
fi
# get total hotspot bases covered for weighted coverage
GENECOV=`bedtools sort -i "$HSBED" | bedtools merge | awk '{c+=$3-$2} END {print c}'`
# get the panel-gene-intersection to avoid coverage by (non-overlaping) partial primer digestion
if [ $CDSISCT -eq 1 ]; then
  bedtools intersect -a "$HSBED" -b "$FTDBED" -wa > "$TMPBED"
  sort -d -k4,4 -k1,1 -k2,2n -k3,3n "$TMPBED" | sort | uniq > "$HSBED"
fi
if [ $TRACK -eq 1 ]; then
  echo "(`date`) Calculating Hotspot Coverage for $(basename $HSBED)" >&2
fi

HSCOVCSV="$FILEOUT.gene.hbu.cov.csv"
echo "Chr,Start,End,COSMIC,Protein,Gene,Coverage,Normalized Coverage" > "$HSCOVCSV"
awk '{print $4 "\t" $7 "\t" $8 "\t" $3}' "$MASTERCDSBED" > "$TMPBED"
$RUNDIR/calculateHBU.pl -E "$BBCTOOLS" -L "$GENECOV" -F "$TCABED" -T 0.1 -S "HSSTS" "$BBCFILE" "$HSBED" | awk 'BEGIN {FS = ","}; {print $1 "\t" $2 "\t" $4}' | sort -k1,1 > "$TMPTSV"
sort -k4,4 "$HSBED" | join -1 1 -2 4 "$TMPTSV" - | awk '{print $4 "\t" $5 "\t" $6 "\t" $1 "\t" $2 "\t" $3}' | bedtools intersect -a - -b "$TMPBED" -wo | awk '{print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $10 "\t" $5 "\t" $6}' | awk '{print $1 "_" $2 "_" $3 "_" $4 "\t" $5 "\t" $6 "\t" $7}' | sort -k1,1 > "$TMPTXT"

# add protein annotation
awk '{print $1 "_" $2 "_" $3 "_" $4 "\t" $6}' "$MASTERHSCBED" | sort -k1,1 | join -1 1 -2 1 "$TMPTXT" - -t $'\t' | awk -F '[_|\t]' '{print $1 "," $2 "," $3 "," $4 "," $8 "," $5 "," $6 "," $7}' | sort -k1,1 -k2,2g -k3,3g -t ',' >> "$HSCOVCSV"

########### Finish up ###########

rm -f "$TMPTXT" "$TMPBED" "$TMPTSV"

if [ $SHOWLOG -eq 1 ]; then
  echo -e "\n$CMD END:" `date` >&2
fi

