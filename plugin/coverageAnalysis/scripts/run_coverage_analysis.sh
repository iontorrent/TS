#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

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
  -c Add target Coverage statistics by mean base read depth. (TargetSeq option.)
  -g Add GC annotation to targets file. By default the annotated BED file provided by the -A option is assumed to be
     correctly formatted (4 standard fields plus Ion auxiliary plus target GC count). With the -g option specified the
     -A BED file (or -B BED file if -A is not used) is re-formated and GC annotated. (Output as tca_auxiliary.gc.bed)
  -d Ignore Duplicate reads.
  -u Filter to Uniquely mapped reads (SAM MAPQ>0).
  -r Customize output for AmpliSeq-RNA reads. (Overrides -a and -c.)
  -p <int>  Padding value for BED file padding. For reporting only. Default: 0.
  -A <file> Annotate coverage for (GC annotated) targets specified in this BED file. See -F option.
  -B <file> Limit coverage to targets specified in this BED file
  -C <name> Original name for BED targets selected for reporting (pre-padding, etc.)
  -D <dirpath> Path to root Directory where results are written. Default: ./
  -G <file> Genome file. Assumed to be <reference.fasta>.fai if not specified.
  -L <name> Reference Library name, e.g. hg19. Defaults to <reference> if not supplied.
  -N <name> Sample name for use in summary output. Default: 'None'
  -O <file> Output file name for text data (per analysis). Default: '' => <BAMROOT>.stats.cov.txt.
  -Q <file> Name for BLOCK HTML results file (in output directory). Default: '' (=> none created)
  -R <file> Name for local HTML Results file (in output directory). Default: '' (HTML fragment written to COVERAGE_html)
  -S <file> SampleID tracking regions file. Default: '' (=> no tracking reads statistics reported)
  -T <file> Name for HTML Table row summary file (in output directory). Default: '' (=> none created)
  -w Display a Warning in HTML report if targets file was expected but not provided, e.g. for targeted analysis defaulting to Whole Genome.
  -x Disable HTML file creation (-R ignored and no COVERAGE_html written).
  -l Log progress to STDERR. (A few primary progress messages will always be output.)
  -h --help Report full description, usage and options."

# TRIMP code still supported but not advertized since TRIMP library code is no longer provided in plugin folder.
# -t Filter BAM file to trimmed reads using TRIMP.
# Padding file is supported but not advertized as it is ineffective and always passed as "" by plugin. (Retained for future usage.)
#  -P <file> Padded (and merged) targets BED file for padded target coverage analysis. 

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n\n$USAGE\n$OPTIONS" >&2
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
SAMPLENAME="None"
NOTARGETANAL=0
TRGCOVBYBASES=0
ANNOBEDFORMAT=0
LIBRARY=""

while getopts "hlabcdgrtuwxp:A:B:C:D:G:L:N:O:P:Q:R:S:T:" opt
do
  case $opt in
    A) ANNOBED=$OPTARG;;
    B) BEDFILE=$OPTARG;;
    C) TRGSID=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    G) GENOME=$OPTARG;;
    L) LIBRARY=$OPTARG;;
    N) SAMPLENAME=$OPTARG;;
    O) OUTFILE=$OPTARG;;
    P) PADBED=$OPTARG;;
    Q) BLOCKFILE=$OPTARG;;
    R) RESHTML=$OPTARG;;
    S) TRACKINGBED=$OPTARG;;
    T) ROWHTML=$OPTARG;;
    p) PADVAL=$OPTARG;;
    a) AMPOPT="-a";;
    b) NOTARGETANAL=1;;
    c) TRGCOVBYBASES=1;;
    d) DEDUP=1;;
    g) ANNOBEDFORMAT=1;;
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

BASECOVERAGE=1
if [ $RNABED -eq 1 ]; then
  AMPOPT="-r"
  BASECOVERAGE=0
fi

BAMBAI="${BAMFILE}.bai"

# Check compatible options - assume human error
if [ $NOTARGETANAL -eq 1 ];then
  if [ -n "$AMPOPT" ];then
    echo "WARNING: $AMPOPT option suppressed by -b option."
    AMPOPT=""
  fi
  if [ $TRGCOVBYBASES -ne 0 ];then
    echo "WARNING: -c option suppressed by -b option."
    TRGCOVBYBASES=0
  fi
fi
if [ -n "$AMPOPT" ];then
  if [ -z "$BEDFILE" ];then
    echo "ERROR: Targets file (-B option) required with $AMPOPT option."
    exit 1
  fi
  if [ -z "$ANNOBED" -a "$ANNOBEDFORMAT" -eq 0 ];then
    echo "ERROR: Annotated targets file (-g or -A option) required with $AMPOPT option."
    exit 1
  fi
  if [ -z "$ANNOBED" -a "$PADVAL" -ne 0 ];then
    echo "WARNING: Padding indicated but unpadded targets not supplied using -A option." >&2
  fi
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
elif [ -n "$ANNOBED" -a ! -f "$ANNOBED" ]; then
  echo "ERROR: Annotated targets (bed) file does not exist at $ANNOBED" >&2
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

NOTINPWD=1
if [ "$WORKDIR" = "$PWD" ];then
  NOTINPWD=0
fi

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

# Create local annotated bedfile given format

LINKANNOBED=1
if [ "$ANNOBEDFORMAT" -ne 0 -a $NOTARGETANAL -eq 0 ];then
  INBEDFILE="$ANNOBED"
  if [ -z "$INBEDFILE" ];then
    INBEDFILE="$BEDFILE"
  fi
  if [ -n "$INBEDFILE" ]; then
    echo "(`date`) Creating GC annotated targets file..." >&2
    ANNOBED="${WORKDIR}/tca_auxiliary.gc.bed"
    ${RUNDIR}/gcAnnoBed.pl -a -s -w -t "$WORKDIR" "$BEDFILE" "$REFERENCE" > "$ANNOBED"
    LINKANNOBED=0
    if [ $SHOWLOG -eq 1 ]; then
      echo "> $ANNOBED" >&2
    fi
  fi
fi

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
# tag on extra control options here - change if need FILTOPTS elsewhere in script
if [ $NOTARGETANAL -eq 1 ];then
  FILTOPTS="$FILTOPTS -b"
fi
if [ $TRGCOVBYBASES -eq 1 ];then
  FILTOPTS="$FILTOPTS -c"
fi

if [ $SHOWLOG -eq 1 ]; then
  echo "" >&2
fi
COVER="$RUNDIR/coverage_analysis.sh $LOGOPT $RTITLE $FILTOPTS $AMPOPT -N \"$SAMPLENAME\" -O \"$OUTFILE\" -A \"$ANNOBED\" -B \"$BEDFILE\" -C \"$TRGSID\" -p $PADVAL -P \"$PADBED\" -S \"$TRACKINGBED\" -D \"$WORKDIR\" -G \"$GENOME\" -L \"$LIBRARY\" \"$REFERENCE\" \"$BAMFILE\""
eval "$COVER" >&2
if [ $? -ne 0 ]; then
  echo -e "\nFailed to run coverage analysis."
  echo "\$ $COVER" >&2
  exit 1
fi
EXTRAHTML="$WORKDIR/tca_auxiliary.htm"

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
    if [ -z "$AMPOPT" -a $TRGCOVBYBASES -eq 1 ];then
      AMPOPT="-b"
    fi
  elif [ -n "$AMPOPT" -o "$CKTARGETSEQ" -eq 1 ];then
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
  PTITLE=`echo $BAMNAME | sed -e 's/\.trim$//'`
  HMLCMD="$RUNDIR/coverage_analysis_report.pl $RTITLE $AMPOPT $ROWHTML $GENOPT $SIDOPT $WARNMSG -N \"$BAMNAME\" -t \"$PTITLE\" -D \"$WORKDIR\" \"$COVERAGE_HTML\" \"$OUTFILE\""
  eval "$HMLCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: coverage_analysis_report.pl failed." >&2
    echo "\$ $HMLCMD" >&2
  elif [ $SHOWLOG -eq 1 ]; then
    echo "> ${WORKDIR}/$COVERAGE_HTML" >&2
  fi
  cat "$EXTRAHTML" >> "${WORKDIR}/$COVERAGE_HTML" 

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
  
  # Create a more useful HTML report for stand-alone mode
  if [ -n "$RESHTML" ]; then
    # Definition of coverage output file names expected in the file links table
    if [ -f "$BAMFILE" ];then
      PLUGIN_OUT_BAMFILE="$BAMROOT"
      if [ $NOTINPWD -eq 1 ];then
        ln -sf "$PWD/$BAMFILE" "${WORKDIR}/$PLUGIN_OUT_BAMFILE"
      fi
    fi
    if [ -f "$BAMBAI" ];then
      PLUGIN_OUT_BAIFILE="${BAMROOT}.bai"
      if [ $NOTINPWD -eq 1 ];then
        ln -sf "$PWD/$BAMBAI" "${WORKDIR}/$PLUGIN_OUT_BAIFILE"
      fi
    fi
    if [ -f "$BEDFILE" ];then
      PLUGIN_OUT_BEDFILE_MERGED=`echo "$BEDFILE" | sed -e 's/^.*\///'`
      if [ $NOTINPWD -eq 1 ];then
        ln -sf "$PWD/$BEDFILE" "${WORKDIR}/$PLUGIN_OUT_BEDFILE_MERGED"
      fi
    fi
    if [ -f "$ANNOBED" -a "$LINKANNOBED" -ne 0 ];then
      PLUGIN_OUT_BEDFILE_UNMERGED=`echo "$ANNOBED" | sed -e 's/^.*\///'`
      if [ $NOTINPWD -eq 1 ];then
        ln -sf "$PWD/$ANNOBED" "${WORKDIR}/$PLUGIN_OUT_BEDFILE_UNMERGED"
      fi
    fi
    PLUGIN_OUT_STATSFILE="${BAMNAME}.stats.cov.txt"
    PLUGIN_OUT_DOCFILE="${BAMNAME}.base.cov.xls"
    PLUGIN_OUT_AMPCOVFILE="${BAMNAME}.amplicon.cov.xls"
    PLUGIN_OUT_TRGCOVFILE="${BAMNAME}.target.cov.xls"
    PLUGIN_OUT_CHRCOVFILE="${BAMNAME}.chr.cov.xls"
    PLUGIN_OUT_WGNCOVFILE="${BAMNAME}.wgn.cov.xls"
    source "${RUNDIR}/../functions/fileLinks.sh"
    write_file_links "$WORKDIR" "filelinks.xls";

    # attempt to link style and javascript files locally
    ln -sf "${RUNDIR}/../css" "${WORKDIR}/"
    ln -sf "${RUNDIR}/../js" "${WORKDIR}/"
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
    echo "<div class=\"center\" style=\"width:100%;height:100%\">" >> "$RESHTML"
    echo "<h1><center>Coverage Analysis Report</center></h1>" >> "$RESHTML"
    if [ -n "$SAMPLENAME" -a "$SAMPLENAME" != "None" ];then
      echo "<h3><center>Sample Name: $SAMPLENAME</center></h3>" >> "$RESHTML"
    fi
    cat "${WORKDIR}/$COVERAGE_HTML" >> "$RESHTML"
    echo '<div></body></html>' >> "$RESHTML"
    rm -f "${WORKDIR}/$COVERAGE_HTML"
  fi
  if [ $SHOWLOG -eq 1 ]; then
    echo "HTML report complete: " `date` >&2
  fi
fi
rm -f "$EXTRAHTML"

############

if [ $SHOWLOG -eq 1 ]; then
  echo -e "\n$CMD END:" `date` >&2
fi

