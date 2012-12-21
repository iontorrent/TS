#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#MAJOR_BLOCK

VERSION="3.4.49291"

# Disable excess debug output for test machine
#set +o xtrace

# DEVELOPMENT/DEBUG options:
# NOTE: the following should be set to 0 in production mode
PLUGIN_DEV_FULL_LOG=0;          # 1 for coverage analysis log, 2 for additional xtrace (not recommended)
CONTINUE_AFTER_BARCODE_ERROR=1;	# 0 to have plugin fail after first barcode failure

# Get the (non-bc) run name from the BAM file - should be the same as ${TSP_RUN_NAME}_${TSP_ANALYSIS_NAME}
PLUGIN_BAM_FILE=`echo "$TSP_FILEPATH_BAM" | sed -e 's/^.*\///'`
PLUGIN_BAM_NAME=`echo $PLUGIN_BAM_FILE | sed -e 's/\.[^.]*$//'`
PLUGIN_RUN_NAME="$TSP_FILEPATH_OUTPUT_STEM"
REFERENCE="$TSP_FILEPATH_GENOME_FASTA"

# Check for by-pass PUI
if [ -z "$PLUGINCONFIG__LIBRARYTYPE_ID" ]; then
  OLD_IFS="$IFS"
  IFS=";"
  PLAN_INFO=(`${DIRNAME}/parse_plan.py ${TSP_FILEPATH_PLUGIN_DIR}/startplugin.json`)
  IFS=$OLD_IFS
  PLUGINCONFIG__LIBRARYTYPE=${PLAN_INFO[0]}
  PLUGINCONFIG__TARGETREGIONS=${PLAN_INFO[1]}
  PLUGINCONFIG__TRIMREADS="No"
  PLUGINCONFIG__PADTARGETS=0
  PLUGINCONFIG__UNIQUEMAPS="Yes"
  PLUGINCONFIG__NONDUPS="Yes"
  if [ -z "$PLUGINCONFIG__LIBRARYTYPE" ]; then
    rm -f "${TSP_FILEPATH_PLUGIN_DIR}/results.json"
    HTML="${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
    echo '<html><body>' > "$HTML"
    if [ -f "${DIRNAME}/html/logo.sh" ]; then
      source "${DIRNAME}/html/logo.sh"
      print_html_logo >> "$HTML";
    fi
    echo "<h3><center>${PLUGIN_RUN_NAME}</center></h3>" >> "$HTML"
    echo "<br/><h2 style=\"text-align:center;color:red\">*** Automatic analysis was not performed for PGM run. ***</h2>" >> "$HTML"
    echo "<br/><h3 style=\"text-align:center\">(Requires an associated Plan that is not a GENS Runtype.)</h3></br>" >> "$HTML"
    echo '</body></html>' >> "$HTML"
    exit
  elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "ampliseq" ]; then
    PLUGINCONFIG__LIBRARYTYPE_ID="Ion AmpliSeq"
    PLUGINCONFIG__TRIMREADS="Yes"
    PLUGINCONFIG__NONDUPS="No"
  elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "targetseq" ]; then
    PLUGINCONFIG__LIBRARYTYPE_ID="Ion TargetSeq"
  elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "wholegenome" ]; then
    PLUGINCONFIG__LIBRARYTYPE_ID="Whole Genome"
  else
    echo "ERROR: Unexpected Library Type: $PLUGINCONFIG__LIBRARYTYPE" >&2
    exit 1
  fi
  if [ -z "$PLUGINCONFIG__TARGETREGIONS" ]; then
    PLUGINCONFIG__TARGETREGIONS_ID=""
  else
    PLUGINCONFIG__TARGETREGIONS_ID=`echo "$PLUGINCONFIG__TARGETREGIONS" | sed -e 's/^.*\///' | sed -e 's/\.bed$//'`
  fi
else
  # Grab PUI parameters
  PLUGINCONFIG__LIBRARYTYPE_ID=`echo "$PLUGINCONFIG__LIBRARYTYPE_ID" | sed -e 's/_/ /g'`
  PLUGINCONFIG__TARGETREGIONS_ID=`echo "$PLUGINCONFIG__TARGETREGIONS_ID" | sed -e 's/_/ /g'`
  if [ -n "$PLUGINCONFIG__TRIMREADS" ]; then
    PLUGINCONFIG__TRIMREADS="Yes"
  else
    PLUGINCONFIG__TRIMREADS="No"
  fi
  if [ -n "$PLUGINCONFIG__UNIQUEMAPS" ]; then
    PLUGINCONFIG__UNIQUEMAPS="Yes"
  else
    PLUGINCONFIG__UNIQUEMAPS="No"
  fi
  if [ -n "$PLUGINCONFIG__NONDUPLICATES" ]; then
    PLUGINCONFIG__NONDUPLICATES="Yes"
  else
    PLUGINCONFIG__NONDUPLICATES="No"
  fi
fi

# Customize analysis options based on library type
PLUGIN_DETAIL_TARGETS=$PLUGINCONFIG__TARGETREGIONS
if [ "$PLUGIN_DETAIL_TARGETS" = "none" ]; then
  PLUGIN_DETAIL_TARGETS=""
fi
PLUGIN_RUNTYPE=$PLUGINCONFIG__LIBRARYTYPE
PLUGIN_TARGETS=`echo "$PLUGIN_DETAIL_TARGETS" | sed -e 's/\/unmerged\/detail\//\/merged\/plain\//'`
PLUGIN_ANNOFIELDS=""
AMPOPT=""
if [ "$PLUGIN_RUNTYPE" = "ampliseq" ]; then
  PLUGIN_ANNOFIELDS="-f 4,8"
  AMPOPT="-a"
elif [ "$PLUGIN_RUNTYPE" = "targetseq" ]; then
  # used merged detail target for base coverage to assigned targets
  PLUGIN_DETAIL_TARGETS=`echo "$PLUGIN_DETAIL_TARGETS" | sed -e 's/\/unmerged\//\/merged\//'`
  PLUGIN_ANNOFIELDS="-f 4,8"
fi
PLUGIN_TRIMREADS=$PLUGINCONFIG__TRIMREADS
PLUGIN_PADSIZE=$PLUGINCONFIG__PADTARGETS
PLUGIN_UMAPS=$PLUGINCONFIG__UNIQUEMAPS
PLUGIN_NONDUPS=$PLUGINCONFIG__NONDUPLICATES
PLUGIN_TRGSID=`echo "$PLUGIN_TARGETS" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`

# Override possible non-sense parameter combinations (?)
#if [ "$PLUGINCONFIG__LIBRARYTYPE" == "ampliseq" ]; then
#  PLUGIN_UMAPS="Yes"
#  PLUGIN_NONDUPS="No"
#elif [ "$PLUGINCONFIG__LIBRARYTYPE" == "wholegenome" ]; then
#  PLUGIN_PADSIZE=0
#fi

# Used to check for for merged BAM file override here
PLUGIN_CHECKBC=1

echo "Selected run options:" >&2
echo "  Library Type:    $PLUGINCONFIG__LIBRARYTYPE_ID" >&2
echo "  Target Regions:  $PLUGINCONFIG__TARGETREGIONS_ID" >&2
echo "  Target Padding:  $PLUGINCONFIG__PADTARGETS" >&2
echo "  Trim Reads:      $PLUGINCONFIG__TRIMREADS" >&2
echo "  Uniquely Mapped: $PLUGINCONFIG__UNIQUEMAPS" >&2
echo "  Non-duplicate:   $PLUGINCONFIG__NONDUPLICATES" >&2

echo "Employed run options:" >&2
echo "  Reference Genome: $REFERENCE" >&2
echo "  Aligned Reads:    $TSP_FILEPATH_BAM" >&2
echo "  Library Type:     $PLUGIN_RUNTYPE" >&2
echo "  Target Regions:   $PLUGIN_DETAIL_TARGETS" >&2
echo "  Merged Regions:   $PLUGIN_TARGETS" >&2
echo "  Target Padding:   $PLUGIN_PADSIZE" >&2
echo "  Trim Reads:       $PLUGIN_TRIMREADS" >&2
echo "  Uniquely Mapped:  $PLUGIN_UMAPS" >&2
echo "  Non-duplicate:    $PLUGIN_NONDUPS" >&2

# Check for missing files
if [ -n "$PLUGIN_DETAIL_TARGETS" ]; then
  if [ ! -e "$PLUGIN_DETAIL_TARGETS" ]; then
    echo "ERROR: Cannot locate target regions file: ${PLUGIN_DETAIL_TARGETS}" >&2
    exit 1
  fi
fi
if ! [ -d "$TSP_FILEPATH_PLUGIN_DIR" ]; then
  echo "ERROR: Failed to locate output directory $TSP_FILEPATH_PLUGIN_DIR" >&2
  exit 1
fi

# Definition of file names, etc.
LIFECHART="${DIRNAME}/lifechart"
PLUGIN_OUT_COVERAGE_HTML="COVERAGE_html"
BARCODES_LIST="${TSP_FILEPATH_PLUGIN_DIR}/barcodeList.txt"
SCRIPTSDIR="${DIRNAME}/scripts"
JSON_RESULTS="${TSP_FILEPATH_PLUGIN_DIR}/results.json"
HTML_RESULTS="${PLUGINNAME}.html"
HTML_BLOCK="${PLUGINNAME}_block.html"
HTML_ROWSUMS="${PLUGINNAME}_rowsum"
PLUGIN_OUT_FILELINKS="filelinks.xls"

# Definition of fields displayed in barcode link/summary table
HTML_TORRENT_WRAPPER=1
PLUGIN_FILTER_READS=0
PLUGIN_INFO_FILTERED="Coverage statistics for uniquely mapped non-duplicate reads."
if [ $PLUGIN_UMAPS = "Yes" ];then
  PLUGIN_FILTER_READS=1
  if [ $PLUGIN_NONDUPS = "No" ];then 
    PLUGIN_INFO_FILTERED="Coverage statistics for uniquely mapped reads."
  fi
elif [ $PLUGIN_NONDUPS = "Yes" ];then
  PLUGIN_FILTER_READS=1
  PLUGIN_INFO_FILTERED="Coverage statistics for non-duplicate reads."
fi
PLUGIN_INFO_ALLREADS="Coverage statistics for all (unfiltered) aligned reads."

BC_COL_TITLE[0]="Mapped Reads"
BC_COL_HELP[0]="Number of reads that were mapped to the full reference genome."
BC_COL_TITLE[1]="On Target"
BC_COL_HELP[1]="Percentage of mapped reads that were aligned over a target region."
BC_COL_TITLE[2]="Mean Depth"
BC_COL_HELP[2]="Mean average target base read depth, including non-covered target bases."
BC_COL_HELP[3]="Percentage of target bases covered by at least 0.2x the average base read depth."
BC_COL_TITLE[3]="Uniformity"

# Set up report page layout and help text
COV_PAGE_WIDTH="700px"
BC_SUM_ROWS=4
FILTOPTS=""
if [ $PLUGIN_FILTER_READS -eq 1 ];then
  # Option to run twice with and w/o filters, producing old-style report
  BC_TITLE_INFO="Coverage summary statistics for filtered aligned barcoded reads."
  if [ $PLUGIN_NONDUPS = "Yes" ];then
    FILTOPTS="$FILTOPTS -d"
  fi
  if [ $PLUGIN_UMAPS = "Yes" ];then
    FILTOPTS="$FILTOPTS -u"
  fi
else
  BC_TITLE_INFO="Coverage summary statistics for all (un-filtered) aligned barcoded reads."
fi

# Set up log options
LOGOPT=""
if [ "$PLUGIN_DEV_FULL_LOG" -gt 0 ]; then
  LOGOPT='-l'
  if [ "$PLUGIN_DEV_FULL_LOG" -gt 1 ]; then
    set -o xtrace
  fi
fi

# Direct PLUGIN_TRIMREADS to direct cmd option
TRIMOPT=""
if [ "$PLUGIN_TRIMREADS" = "Yes" ]; then
  TRIMOPT="-t"
fi

# Source the HTML files for shell functions and define others below
for BASH_FILE in `find ${DIRNAME}/functions/ | grep .sh$`
do
  source ${BASH_FILE};
done

# --------- Start processing the data ----------

# Local copy of sorted barcode list file
if [ ! -f $TSP_FILEPATH_BARCODE_TXT ]; then
   PLUGIN_CHECKBC=0
fi
if [ $PLUGIN_CHECKBC -eq 1 ]; then
  # use the old barcode list - rely on number of BAMs in folder for actual list
  run "sort -t ' ' -k 2n,2 \"$TSP_FILEPATH_BARCODE_TXT\" > \"$BARCODES_LIST\"";
fi

# Create links to files required for (barcode) report summary table
run "ln -sf ${DIRNAME}/js ${TSP_FILEPATH_PLUGIN_DIR}/.";
run "ln -sf ${DIRNAME}/css ${TSP_FILEPATH_PLUGIN_DIR}/.";

echo -e "\nResults folder initialized." >&2

if [ "$PLUGIN_DEV_FULL_LOG" -ne 0 ]; then
  echo "" >&2
fi

# Create padded targets file
PLUGIN_EFF_TARGETS="$PLUGIN_TARGETS"
PADDED_TARGETS=""
if [ $PLUGIN_PADSIZE -gt 0 ];then
  GENOME="${REFERENCE}.fai"
  if ! [ -f "$GENOME" ]; then
    echo "WARNING: Could not create padded targets file; genome (.fai) file does not exist at $GENOME" >&2
    echo "- Continuing without padded targets analysis." >&2
  else
    PADDED_TARGETS="${TSP_FILEPATH_PLUGIN_DIR}/padded_targets_$PLUGIN_PADSIZE.bed"
    PADCMD="${DIRNAME}/padbed/padbed.sh $LOGOPT \"$PLUGIN_TARGETS\" \"$GENOME\" $PLUGIN_PADSIZE \"$PADDED_TARGETS\""
    eval "$PADCMD" >&2
    if [ $? -ne 0 ]; then
      echo "WARNING: Could not create padded targets file; padbed.sh failed." >&2
      echo "\$ $REMDUP" >&2
      echo "- Continuing without padded targets analysis." >&2
      PADDED_TARGETS=""
    else
      # as of 3.0 the bed file becomes the padded bedfile so all analysis is done on the padded targets
      PLUGIN_EFF_TARGETS="$PADDED_TARGETS"
      PADDED_TARGETS=""
    fi
  fi
  echo >&2
fi

# Create GC annotated BED file for read-to-target assignment
GCANNOBED=""
if [ -n "$PLUGIN_DETAIL_TARGETS" ]; then
  if [ $PLUGIN_DEV_FULL_LOG -gt 0 ]; then
    echo "Adding GC count information to annotated targets file..." >&2
  fi
  GCANNOBED="${TSP_FILEPATH_PLUGIN_DIR}/tca_auxiliary.gc.bed"
  GCANNOCMD="${SCRIPTSDIR}/gcAnnoBed.pl -s -w -t \"$TSP_FILEPATH_PLUGIN_DIR\" $PLUGIN_ANNOFIELDS \"$PLUGIN_DETAIL_TARGETS\" \"$REFERENCE\" > \"$GCANNOBED\""
  eval "$GCANNOCMD" >&2
  if [ $? -ne 0 ]; then
    echo -e "\nERROR: gcAnnoBed.pl failed." >&2
    echo "\$ $GCANNOCMD" >&2
    exit 1;
  elif [ $PLUGIN_DEV_FULL_LOG -gt 0 ]; then
    echo "> $GCANNOBED" >&2
  fi
fi

# Remove previous results to avoid displaying old before ready
PLUGIN_OUT_STATSFILE="${PLUGIN_BAM_NAME}.stats.cov.txt"

rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}" "${TSP_FILEPATH_PLUGIN_DIR}/$HTML_BLOCK" "$JSON_RESULTS"
rm -f "$PLUGIN_OUT_COVERAGE_HTML"
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*.stats.cov.txt ${TSP_FILEPATH_PLUGIN_DIR}/*.xls ${TSP_FILEPATH_PLUGIN_DIR}/*.png
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*.bam*

PLUGIN_OUT_STATSFILE=""

# Creates the body of the detailed report post-analysis
write_html_results ()
{
  local RUNID=${1}
  local OUTDIR=${2}
  local OUTURL=${3}
  local BAMFILE=${4}

  # test for trimmed bam file based results
  local BAMROOT="$RUNID"
  if [ "$PLUGIN_TRIMREADS" = "Yes" ]; then
    PLUGIN_OUT_TRIMPBAM=`echo $BAMFILE | sed -e 's/.bam$/\.trim\.bam/'`
    if [ -e "${OUTDIR}/$PLUGIN_OUT_TRIMPBAM" ]; then
      PLUGIN_OUT_TRIMPBAI="${PLUGIN_OUT_TRIMPBAM}.bai"
      BAMROOT="${RUNID}.trim";
    else
      PLUGIN_OUT_TRIMPBAM=""
    fi
  fi
  # Definition of coverage output file names expected in the file links table
  PLUGIN_OUT_BAMFILE="${RUNID}.bam"
  PLUGIN_OUT_BAIFILE="${PLUGIN_OUT_BAMFILE}.bai"
  PLUGIN_OUT_STATSFILE="${BAMROOT}.stats.cov.txt" ; # also needed for json output
  PLUGIN_OUT_DOCFILE="${BAMROOT}.base.cov.xls"
  PLUGIN_OUT_AMPCOVFILE="${BAMROOT}.amplicon.cov.xls"
  PLUGIN_OUT_TRGCOVFILE="${BAMROOT}.target.cov.xls"
  PLUGIN_OUT_CHRCOVFILE="${BAMROOT}.chr.cov.xls"
  PLUGIN_OUT_WGNCOVFILE="${BAMROOT}.wgn.cov.xls"

  # Links to folders/files required for html report pages (inside firewall)
  run "ln -sf \"${DIRNAME}/flot\" \"${OUTDIR}/\"";
  run "ln -sf \"${LIFECHART}\" \"${OUTDIR}/\"";
  run "ln -sf \"${SCRIPTSDIR}/igv.php3\" \"${OUTDIR}/\"";

  # Create the html report page
  echo "(`date`) Publishing HTML report page..." >&2
  write_file_links "$OUTDIR" "$PLUGIN_OUT_FILELINKS";
  local HTMLOUT="${OUTDIR}/${HTML_RESULTS}";
  write_page_header "$LIFECHART/TCA.head.html" "$HTMLOUT";
  cat "${OUTDIR}/$PLUGIN_OUT_COVERAGE_HTML" >> "$HTMLOUT"
  write_page_footer "$HTMLOUT";

  # Remove temporary files (in each barcode folder)
  run "rm -f ${OUTDIR}/${PLUGIN_OUT_COVERAGE_HTML}"
}

# Check for barcodes
if [ $PLUGIN_CHECKBC -eq 1 ]; then
  barcode;
else
  # Write a front page for non-barcode run
  HTML="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
  write_html_header "$HTML" 15;
  echo "<h3><center>${PLUGIN_RUN_NAME}</center></h3>" >> "$HTML"
  display_static_progress "$HTML";
  write_html_footer "$HTML";
  # need to create link early so the correct name gets used if a PTRIM file is created
  if [ -e "$TSP_FILEPATH_BAM" ]; then
    TESTBAM=$TSP_FILEPATH_BAM  
  else
    TESTBAM="${ANALYSIS_DIR}/${PLUGIN_RUN_NAME}.bam"
  fi
  run "ln -sf \"${TESTBAM}\" \"${PLUGIN_RUN_NAME}.bam\""
  run "ln -sf \"${TESTBAM}.bai\" \"${PLUGIN_RUN_NAME}.bam.bai\""
  # Run on single bam
  RT=0
  eval "${SCRIPTSDIR}/run_coverage_analysis.sh $LOGOPT $FILTOPTS $AMPOPT $TRIMOPT -R \"$HTML_RESULTS\" -D \"$TSP_FILEPATH_PLUGIN_DIR\" -A \"$GCANNOBED\" -B \"$PLUGIN_EFF_TARGETS\" -C \"$PLUGIN_TRGSID\" -P \"$PADDED_TARGETS\" -p $PLUGIN_PADSIZE -Q \"$HTML_BLOCK\" \"$REFERENCE\" \"${PLUGIN_RUN_NAME}.bam\"" || RT=$?
  if [ $RT -ne 0 ]; then
    write_html_header "$HTML";
    echo "<h3><center>${PLUGIN_RUN_NAME}</center></h3>" >> "$HTML"
    echo "<br/><h3 style=\"text-align:center;color:red\">*** An error occurred - check Log File for details ***</h3><br/>" >> "$HTML"
    write_html_footer "$HTML";
    exit 1
  fi
  # Collect results for detail html report and clean up - also sets $PLUGIN_OUT_STATSFILE
  write_html_results "$PLUGIN_RUN_NAME" "$TSP_FILEPATH_PLUGIN_DIR" "." "${PLUGIN_RUN_NAME}.bam"
  # Write json output
  write_json_header;
  write_json_inner "${TSP_FILEPATH_PLUGIN_DIR}" "$PLUGIN_OUT_STATSFILE" "" 2;
  write_json_footer;
fi
# Remove after successful completion
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/header" "${TSP_FILEPATH_PLUGIN_DIR}/footer" "$PADDED_TARGETS" "$BARCODES_LIST"
echo "(`date`) Completed with statitics output to results.json."


