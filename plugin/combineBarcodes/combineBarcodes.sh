#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

CMD=`echo $0 | sed -e 's/^.*\///'`
DESCR="Create a set of barcode bam files and create a new grouping of barcode bam files."
USAGE="USAGE:
 $CMD [options] <root barcode BAM filepath>"
OPTIONS="OPTIONS:
  -h --help Report usage and help
  -i Create the BAM INDEX file(s) (e.g. <input file>.bam.bai).
  -l Write additonal Log information to STDERR.
  -u Create unique starts and from new barcode groups.
  -g Group unique starts. Create unique starts for each barcode group then combine them, otherwise create unique starts for the combined groups.
  -c Collapse groupings: Take groups from just effective barcodes. Otherwise group by absolute barcode position, even if no reads.
  -a Combine All groups. Create an additional output of all combined barcode groups, and grouped unique starts if -u or -g specified.
  -B <file> Input Barcode list file. Ordered list of barcode ID's to be group merged. If not speficied then options -G and -E are ignored and
     no merging is done, but unique starts may still be generated.
  -G <N> Grouping number. E.g. N=4 bams merged per new barcode. Default: 0 (all bams merged to one).
  -E <N> For Each offset. E.g. E=2 means every other BAM is grouped to N and count cycles. Default: 1.
  -T <N> Barcode mapping Threshold. Only include barcodes read with at least N mapped reads. Default: 0.
  -L <N> Difference between 'duplicate' reads must be less than N (by binning). Default: 0.
  -S <file> Output barcode grouping Statistics in HTML table (rows) format to this file, if specified.
  -O Merged BAM file name. Default = 'combineBarcodes'.
  -U <url> Url path to directory where results are written. Default: .
  -D <dirpath> Path to Directory where results are written. Default: .";

# should scan all args first for --X options
if [ "$1" = "--help" ]; then
    echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
    exit 0
fi

OUTPUTNAME="combineBarcodes"
SHOWLOG=0
WORKDIR="."
MAKEBAI=0
BCLIST=""
GROUPING=0
EVERYOBC=1
MINMAPS=0
GRPUSTS=0
COLGROUP=0
COMBINEGRPS=0
DUPLENVAR=0
STATSFILE=""
URLPATH="."

while getopts "hlucigaD:B:G:E:T:L:O:S:U:" opt
do
  case $opt in
    B) BCLIST=$OPTARG;;
    G) GROUPING=$OPTARG;;
    E) EVERYOBC=$OPTARG;;
    T) MINMAPS=$OPTARG;;
    L) DUPLENVAR=$OPTARG;;
    O) OUTPUTNAME=$OPTARG;;
    S) STATSFILE=$OPTARG;;
    D) WORKDIR=$OPTARG;;
    U) URLPATH=$OPTARG;;
    i) MAKEBAI=1;;
    l) SHOWLOG=1;;
    u) USTARTS=1;;
    c) COLGROUP=1;;
    g) GRPUSTS=1;;
    a) COMBINEGRPS=1;;
    h) echo -e "$DESCR\n$USAGE\n$OPTIONS" >&2
       exit 0;;
    \?) echo $USAGE >&2
        exit 1;;
  esac
done
shift `expr $OPTIND - 1`

if [ $# -ne 1 ]; then
  echo "$CMD: Invalid number of arguments." >&2
  echo -e "$USAGE\n$OPTIONS" >&2
  exit 1;
fi

BAMFILE=$1

# get path, remove extension, etc.
#BAMFILE=`readlink -n -f "$BAMFILE"`
BAMPATH=$(dirname ${BAMFILE})
BAMFILE=`echo "$BAMFILE" | sed -e 's/\.bam$//'`
BAMNAME=`echo "$BAMFILE" | sed -e 's/^.*\///'`

if [ $EVERYOBC -lt 1 ]; then
  EVERYOBC=1
fi
if [ $MINMAPS -lt 0 ]; then
  MINMAPS=0
fi

#--------- End command arg parsing ---------

run ()
{
    local EXIT_CODE=0
    eval $* >&2 || EXIT_CODE="$?"
    if [ ${EXIT_CODE} != 0 ]; then
        echo -e "ERROR: Status code '${EXIT_CODE}' while running\n\$$*" >&2
        exit 1
    fi
}

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
fi

# Ensure set up for no grouping #1
NOGROUP=0
if [ $GROUPING -lt 1 ]; then
  COLGROUP=1
  NOGROUP=1
fi

# Get effective input barcode list...
echo "Checking barcode BAMs for numbers of mapped reads..."
BCN=0
UBCN=0
for BCLINE in `cat ${BCLIST} | grep "^barcode"`
do
  # stupid bash!
  if [ "$BCLINE" == "barcode" ]; then
    continue;
  fi
  BCN=`expr ${BCN} + 1`
  BARCODE=`echo ${BCLINE} | awk 'BEGIN{FS=","} {print $2}'`
  BARCODEBAM="${BAMPATH}/${BARCODE}_${BAMNAME}.bam"

  if [ -f "$BARCODEBAM" ]; then
    NMAP=`samtools view -c -F 4 "$BARCODEBAM"`
    if [ "$NMAP" -ge "$MINMAPS" ]; then
      echo "Barcode ${BARCODE} has "$NMAP" mapped reads." >&2
      BAMLIST[$UBCN]="${BARCODE}_${BAMNAME}"
      BARCNUM[$UBCN]=$BCN
      UBCN=`expr ${UBCN} + 1`
    elif [ $COLGROUP -eq 0 ]; then
      echo "Barcode ${BARCODE} has "$NMAP" mapped reads. Omitting but retaining grouping position." >&2
      BAMLIST[$UBCN]=""
      BARCNUM[$UBCN]=$BCN
      UBCN=`expr ${UBCN} + 1`
    else
      echo "Barcode ${BARCODE} has "$NMAP" mapped reads. Omitting." >&2
    fi
  elif [ $COLGROUP -eq 0 ]; then
    echo "Barcode ${BARCODE} has no reads. Omitting but retaining grouping position." >&2
    BAMLIST[$UBCN]=""
    BARCNUM[$UBCN]=$BCN
    UBCN=`expr ${UBCN} + 1`
  else
    echo "Barcode ${BARCODE} has no reads. Omitting." >&2
  fi
done

# Ensure set up for no grouping #2
if [ $NOGROUP -eq 1 -o $GROUPING -ge $UBCN ]; then
  GROUPING=$UBCN
fi

OUTFILE="${WORKDIR}/$OUTPUTNAME"
OUTFILEUS="${WORKDIR}/${OUTPUTNAME}.ustarts"
BCLISTOUT="${OUTFILE}.barcodeList.txt"
BCLISTOUTUS="${OUTFILEUS}.barcodeList.txt"
if [ $NOGROUP -eq 0 ]; then
  # the scoring info. is fake but there in case file parser expects something
  echo -e "file_id ${OUTPUTNAME}\nscore_mode 1\nscore_cutoff 2.0" > "$BCLISTOUT"
  if [ $USTARTS -eq 1 ]; then
    echo -e "file_id ${OUTPUTNAME}.ustarts\nscore_mode 1\nscore_cutoff 2.0" > "$BCLISTOUTUS"
  fi
fi

if [ -f "$STATSFILE" ]; then
  rm -f "$STATSFILE"
fi

# -g option forces -u option
PREGRP_USTS=0
PSTGRP_USTS=0
if [ $GRPUSTS -eq 1 ]; then
  USTARTS=1
  PREGRP_USTS=1
elif [ $USTARTS -eq 1 ]; then
  PSTGRP_USTS=1
fi

# Perform the grouped merges...
# Note that code is complicated because no bam may be writen until group merge is completed (to prevent it being ls'd)
if [ $UBCN -gt 0 ]; then
  if [ $NOGROUP -eq 1 ]; then
    echo "Merging $UBCN files of $BCN..." >&2
  else
    echo "Group merging $UBCN files of $BCN..." >&2
  fi
  GCNT=0
  ECNT=0
  PBAR=0
  GRPNUM=0
  NMAP=0
  COMBOLIST=""
  for((BCC=0;BCC<$UBCN;BCN++))
  do
    BCC=`expr ${BCC} + 1`
    BARCODEBAM="${BAMPATH}/${BAMLIST[$PBAR]}"
    LOCALBAM="cbc.$BCC.local.bamx"
    if [ -n "${BAMLIST[$PBAR]}" ]; then
      NMAP=`expr ${NMAP} + 1`
      if [ "$NMAP" -eq 1 ]; then
        COMBOLIST="${BARCNUM[$PBAR]}"
      else
        COMBOLIST="${COMBOLIST},${BARCNUM[$PBAR]}"
      fi
      run "ln -sf \"${BARCODEBAM}.bam\" \"${WORKDIR}/$LOCALBAM\""
      if [ $PREGRP_USTS -eq 1 ]; then
        echo "Creating unique starts for ${LOCALBAM}..." >&2
        echo "  (${BAMLIST[$PBAR]}.bam)" >&2
        run "${RUNDIR}/create_unique_starts.sh $LOGOPT -L $DUPLENVAR -D \"${WORKDIR}\" \"${WORKDIR}/$LOCALBAM\""
      fi
    fi
    GCNT=`expr ${GCNT} + 1`
    if [ $GCNT -eq $GROUPING -o $BCC -eq $UBCN ]; then
      GRPNUM=`expr ${GRPNUM} + 1`
      if [ $NOGROUP -eq 0 ]; then
        OUTFILE="${WORKDIR}/${OUTPUTNAME}_${GRPNUM}-${NMAP}"
        OUTFILEUS="${WORKDIR}/${OUTPUTNAME}.ustarts_${GRPNUM}-${NMAP}"
      fi
      if [ $NMAP -eq 0 ]; then
        echo "Ignoring empty set of barcodes $OUTFILE.."
      elif [ $NMAP -eq 1 ]; then
        run "mv \"${WORKDIR}/$LOCALBAM\" \"${OUTFILE}.bam\""
        echo "> ${OUTFILE}.bam" >&2
        if [ $MAKEBAI -eq 1 ]; then
          run "ln -sf \"${BARCODEBAM}.bam.bai\" \"${OUTFILE}.bam.bai\""
          echo "> ${OUTFILE}.bam.bai" >&2
        fi
        if [ $USTARTS -eq 1 ]; then
          if [ "$PSTGRP_USTS" -eq 1 ]; then
            echo "Creating unique starts for ${LOCALBAM}..." >&2
            echo "  (${BAMLIST[$PBAR]}.bam)" >&2
            run "${RUNDIR}/create_unique_starts.sh $LOGOPT -L $DUPLENVAR -D \"${WORKDIR}\" \"${WORKDIR}/$LOCALBAM\""
          fi
          run "mv ${WORKDIR}/*.ustarts.bamx ${OUTFILEUS}.bam"
          echo "> ${OUTFILEUS}.bam" >&2
          if [ $MAKEBAI -eq 1 ]; then
            run "samtools index \"${OUTFILEUS}.bam\""
            echo "> ${OUTFILEUS}.bam.bai" >&2
          fi
        fi
      else
        echo "Combining $GROUPING barcoded alignments"
        run "samtools merge -f \"${OUTFILE}.tmp\" ${WORKDIR}/*.local.bamx >&2"
        run "mv \"${OUTFILE}.tmp\" \"${OUTFILE}.bam\""
        run "rm -f ${WORKDIR}/*.local.bamx"
        echo "> ${OUTFILE}.bam" >&2
        if [ $MAKEBAI -eq 1 ]; then
          run "samtools index \"${OUTFILE}.bam\""
          echo "> ${OUTFILE}.bam.bai" >&2
        fi
        if [ $USTARTS -eq 1 ]; then
          if [ $PREGRP_USTS -eq 1 ]; then
            echo "Combining $GROUPING unique starts alignments"
            run "samtools merge -f \"${OUTFILEUS}.tmp\" ${WORKDIR}/*.ustarts.bamx >&2"
            run "mv \"${OUTFILEUS}.tmp\" \"${OUTFILEUS}.bam\""
            run "rm -f ${WORKDIR}/*.ustarts.bamx"
          else
            echo "Creating unique starts alignments for combined barcodes..." >&2
            run "${RUNDIR}/create_unique_starts.sh $LOGOPT -L $DUPLENVAR -D \"${WORKDIR}\" \"${OUTFILE}.bam\""
          fi
          echo "> ${OUTFILEUS}.bam" >&2
          if [ $MAKEBAI -eq 1 ]; then
            run "samtools index \"${OUTFILEUS}.bam\""
            echo "> ${OUTFILEUS}.bam.bai" >&2
          fi
        fi
      fi
      # collect output statistics and write to barcodeList.txt
      if [ $NMAP -gt 0 ]; then
        if [ $NOGROUP -eq 0 ]; then
          # again, some of the fields here are faked to keep format parsable
          echo "barcode $GRPNUM,${OUTPUTNAME}_${GRPNUM}-${NMAP},MIXED,GAT,,none,10," >> "$BCLISTOUT"
          if [ $USTARTS -eq 1 ]; then
            echo "barcode $GRPNUM,${OUTPUTNAME}.ustarts_${GRPNUM}-${NMAP},MIXED,GAT,,none,10," >> "$BCLISTOUTUS"
          fi
        fi
        if [ -n "$STATSFILE" ]; then
          RMAP=`samtools view -c -F 4 "${OUTFILE}.bam"`
          echo -n "<tr><td>${COMBOLIST}</td><td>$RMAP</td>" >> "$STATSFILE"
          if [ $USTARTS -eq 1 ]; then
            NUS=`samtools view -c -F 4 "${OUTFILEUS}.bam"`
            echo -n "<td>$NUS</td>" >> "$STATSFILE"
          fi
          if [ $NOGROUP -eq 0 ]; then
            URL="${URLPATH}/${OUTPUTNAME}_${GRPNUM}-${NMAP}.bam"
          else
            URL="${URLPATH}/${OUTPUTNAME}.bam"
          fi
          echo -n "<td><a href=\"${URL}\">BAM</a> &nbsp; <a href=\"${URL}.bai\">BAI</a></td>" >> "$STATSFILE"
          if [ $USTARTS -eq 1 ]; then
            if [ $NOGROUP -eq 0 ]; then
              URL="${URLPATH}/${OUTPUTNAME}.ustarts_${GRPNUM}-${NMAP}.bam"
            else
              URL="${URLPATH}/${OUTPUTNAME}.ustarts.bam"
            fi
            echo -n "<td><a href=\"${URL}\">BAM</a> &nbsp; <a href=\"${URL}.bai\">BAI</a></td>" >> "$STATSFILE"
          fi
          echo "</tr>" >> "$STATSFILE"
        fi
      fi
      NMAP=0
      GCNT=0
      COMBOLIST=""
    fi
    # handle the group spacing option
    PBAR=`expr ${PBAR} + ${EVERYOBC}`
    if [ $PBAR -ge $UBCN ]; then
      ECNT=`expr ${ECNT} + 1`
      PBAR=$ECNT
    fi
  done
  if [ $COMBINEGRPS -eq 1 ]; then
    echo "Combining barcode groups..." >&2
    OUTFILE="${WORKDIR}/${OUTPUTNAME}"
    OUTFILEUS="${OUTFILE}.ustarts"
    run "samtools merge -f \"${OUTFILE}.tmp\" ${OUTFILE}_*.bam >&2"
    run "mv \"${OUTFILE}.tmp\" \"${OUTFILE}.bam\""
    echo "> ${OUTFILE}.bam" >&2
    if [ $MAKEBAI -eq 1 ]; then
      run "samtools index \"${OUTFILE}.bam\""
      echo "> ${OUTFILE}.bam.bai" >&2
    fi
    if [ $USTARTS -eq 1 ]; then
      echo "Combining unique starts groups..." >&2
      run "samtools merge -f \"${OUTFILEUS}.tmp\" ${OUTFILEUS}_*.bam >&2"
      run "mv \"${OUTFILEUS}.tmp\" \"${OUTFILEUS}.bam\""
      echo "> ${OUTFILEUS}.bam" >&2
      if [ $MAKEBAI -eq 1 ]; then
        run "samtools index \"${OUTFILEUS}.bam\""
        echo "> ${OUTFILEUS}.bam.bai" >&2
      fi
    fi
    if [ -n "$STATSFILE" ]; then
      echo "Collecting combined groups statistics..." >&2
      RMAP=`samtools view -c -F 4 "${OUTFILE}.bam"`
      echo -n "<tr><td>All groups</td><td>$RMAP</td>" >> "$STATSFILE"
      if [ $USTARTS -eq 1 ]; then
        NUS=`samtools view -c -F 4 "${OUTFILEUS}.bam"`
        echo -n "<td>$NUS</td>" >> "$STATSFILE"
      fi
      URL="${URLPATH}/${OUTPUTNAME}.bam"
      echo -n "<td><a href=\"${URL}\">BAM</a> &nbsp; <a href=\"${URL}.bai\">BAI</a></td>" >> "$STATSFILE"
      if [ $USTARTS -eq 1 ]; then
        URL="${URLPATH}/${OUTPUTNAME}.ustarts.bam"
        echo -n "<td><a href=\"${URL}\">BAM</a> &nbsp; <a href=\"${URL}.bai\">BAI</a></td>" >> "$STATSFILE"
      fi
      echo "</tr>" >> "$STATSFILE"
    fi
  fi
else
  if [ $USTARTS -eq 1 ]; then
    echo "Filtering reads to unique starts..." >&2
    LOCALBAM="${WORKDIR}/combineBarcodes.bamx"
    run "ln -sf \"${BAMFILE}.bam\" \"${LOCALBAM}\""
    run "${DIRNAME}/create_unique_starts.sh $LOGOPT -L $DUPLENVAR -D \"${WORKDIR}\" \"${LOCALBAM}\""
    run "mv \"${LOCALBAM}\" \"${OUTFILE}.bam\""
    echo "> ${OUTPUTNAME}.bam" >&2
    if [ $MAKEBAI -eq 1 ]; then
      run "ln -sf \"${BAMFILE}.bam.bai\" \"${OUTFILE}.bam.bai\""
      echo "> ${OUTPUTNAME}.bam.bai" >&2
    fi
    run "mv \"${WORKDIR}/combineBarcodes.ustarts.bamx\" \"${OUTFILEUS}.bam\""
    echo "> ${OUTPUTNAME}.ustarts.bam" >&2
    if [ $MAKEBAI -eq 1 ]; then
      run "samtools index \"${OUTFILEUS}.bam\""
      echo "> ${OUTPUTNAME}.ustarts.bam.bai" >&2
    fi
    if [ -n "$STATSFILE" ]; then
      RMAP=`samtools -c -F 4 view "${OUTFILE}.bam"`
      NUS=`samtools -c -F 4 view "${OUTFILEUS}.bam"`
      echo -n "<tr><td>No barcodes</td><td>${RMAP}</td><td>$NUS</td>" > "$STATSFILE"
      URL="${URLPATH}/${OUTPUTNAME}.bam"
      echo -n "<td><a href=\"${URL}\">BAM</a> &nbsp; <a href=\"${URL}.bai\">BAI</a></td>" >> "$STATSFILE"
      URL="${URLPATH}/${OUTPUTNAME}.ustarts.bam"
      echo "<td><a href=\"${URL}\">BAM</a> &nbsp; <a href=\"${URL}.bai\">BAI</a></td></tr>" >> "$STATSFILE"
    fi
  else
    run "ln -sf \"$BAMFILE\" \"${OUTFILE}.bam\""
    echo "> ${OUTPUTNAME}.bam" >&2
    if [ $MAKEBAI -eq 1 ]; then
      run "ln -sf \"${BAMFILE}.bam.bai\" \"${OUTFILE}.bam.bai\""
      echo "> ${OUTPUTNAME}.bam.bai" >&2
    fi
    if [ -n "$STATSFILE" ]; then
      NMAP=`samtools view -c -F 4 "${OUTFILE}.bam"`
      echo -n "<tr><td>-</td><td>${NMAP}</td>" > "$STATSFILE"
      URL="${URLPATH}/${OUTPUTNAME}.bam"
      echo "<td><a href=\"${URL}\">BAM</a> &nbsp; <a href=\"${URL}.bai\">BAI</a></td></tr>" >> "$STATSFILE"
    fi
  fi
fi

if [ $SHOWLOG -eq 1 ]; then
  echo "Merging barcodes complete:" `date` >&2
  echo "" >&2
fi
 
