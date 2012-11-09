#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#MAJOR_BLOCK

VERSION="3.2.1-44445"

# Disable excess debug output for test machine
#set +o xtrace

# DEVELOPMENT/DEBUG options:
# NOTE: the following should be set to 0 in production mode
PLUGIN_DEV_FULL_LOG=0;          # 1 for coverage analysis log, 2 for additional xtrace (not recommended)
CONTINUE_AFTER_BARCODE_ERROR=1;	# 0 to have plugin fail after first barcode failure
TWO_SETS_OUTPUT=0;		# 1 to revert to 2.2 style of barcode summary report

# Check for by-pass PUI
if [ -z "$PLUGINCONFIG__LIBRARYTYPE_ID" ]; then
  OLD_IFS="$IFS"
  IFS=";"
  PLAN_INFO=(`${DIRNAME}/parse_plan.py ${TSP_FILEPATH_PLUGIN_DIR}/startplugin.json`)
  IFS=$OLD_IFS
  PLUGINCONFIG__LIBRARYTYPE=${PLAN_INFO[0]}
  PLUGINCONFIG__TARGETREGIONS=${PLAN_INFO[1]}
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
  elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "targetseq" ]; then
    PLUGINCONFIG__LIBRARYTYPE_ID="Ion TargetSeq"
  elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "fullgenome" ]; then
    PLUGINCONFIG__LIBRARYTYPE_ID="Full Genome"
  else
    echo "ERROR: Unexpected Library Type: $PLUGINCONFIG__LIBRARYTYPE" >&2
    exit 1
  fi
  if [ -z "$PLUGINCONFIG__TARGETREGIONS" ]; then
    PLUGINCONFIG__TARGETREGIONS_ID=""
  else
    PLUGINCONFIG__TARGETREGIONS_ID=`echo "$PLUGINCONFIG__TARGETREGIONS" | sed -e 's/^.*\///' | sed -e 's/\.bed$//'`
    PLUGINCONFIG__TARGETREGIONS=`echo "$PLUGINCONFIG__TARGETREGIONS" | sed -e 's/\/unmerged\/detail\//\/merged\/plain\//'`
  fi
  PLUGINCONFIG__PADTARGETS=0
  PLUGINCONFIG__UNIQUEMAPS="No"
  PLUGINCONFIG__NONDUPS="No"
else
  # Grab PUI parameters
  PLUGINCONFIG__LIBRARYTYPE_ID=`echo "$PLUGINCONFIG__LIBRARYTYPE_ID" | sed -e 's/_/ /g'`
  PLUGINCONFIG__TARGETREGIONS_ID=`echo "$PLUGINCONFIG__TARGETREGIONS_ID" | sed -e 's/_/ /g'`
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

PLUGIN_TARGETS=$PLUGINCONFIG__UNPTARGETS
PLUGIN_PADSIZE=$PLUGINCONFIG__PADTARGETS
PLUGIN_UMAPS=$PLUGINCONFIG__UNIQUEMAPS
PLUGIN_NONDUPS=$PLUGINCONFIG__NONDUPLICATES

# Override possible non-sense parameter combinations
if [ "$PLUGINCONFIG__LIBRARYTYPE" == "ampliseq" ]; then
  PLUGIN_UMAPS="Yes"
  PLUGIN_NONDUPS="No"
elif [ "$PLUGINCONFIG__LIBRARYTYPE" == "fullgenome" ]; then
  PLUGIN_PADSIZE=0
fi

# Check for merged BAM file override
PLUGIN_CHECKBC=1
if [ -n "$PLUGINCONFIG__MERGEDBAM" ]; then
    TSP_FILEPATH_BAM=$PLUGINCONFIG__MERGEDBAM
    PLUGIN_CHECKBC=0
else
    PLUGINCONFIG__MERGEDBAM_ID='Current Report'
fi

echo "Selected run options:" >&2
echo "  Aligned Reads:   $PLUGINCONFIG__MERGEDBAM_ID" >&2
echo "  Library Type:    $PLUGINCONFIG__LIBRARYTYPE_ID" >&2
echo "  Target regions:  $PLUGINCONFIG__TARGETREGIONS_ID" >&2
echo "  Target padding:  $PLUGINCONFIG__PADTARGETS" >&2
echo "  Uniquely mapped: $PLUGINCONFIG__UNIQUEMAPS" >&2
echo "  Non-duplicate:   $PLUGINCONFIG__NONDUPLICATES" >&2

echo "Employed run options:" >&2
echo "  Reference Genome: $TSP_FILEPATH_GENOME_FASTA" >&2
echo "  Aligned Reads:    $TSP_FILEPATH_BAM" >&2
echo "  Library Type:     $PLUGINCONFIG__LIBRARYTYPE" >&2
echo "  Target Regions:   $PLUGIN_TARGETS" >&2
echo "  Target padding:   $PLUGIN_PADSIZE" >&2
echo "  Uniquely mapped:  $PLUGIN_UMAPS" >&2
echo "  Non-duplicate:    $PLUGIN_NONDUPS" >&2

# Check for missing files
if [ -n "$PLUGIN_TARGETS" ]; then
  if [ ! -f "$PLUGIN_TARGETS" ]; then
    echo "ERROR: Cannot locate target regions file: ${PLUGIN_TARGETS}" >&2
    exit 1
  fi
fi
if ! [ -d "$TSP_FILEPATH_PLUGIN_DIR" ]; then
  echo "ERROR: Failed to locate output directory $TSP_FILEPATH_PLUGIN_DIR" >&2
  exit 1
fi

# Definition of file names, etc.
PLUGIN_BAM_FILE=`echo "$TSP_FILEPATH_BAM" | sed -e 's/^.*\///'`
PLUGIN_BAM_NAME=`echo $PLUGIN_BAM_FILE | sed -e 's/\.[^.]*$//'`
PLUGIN_RUN_NAME="$TSP_RUN_NAME"
BARCODES_LIST="${TSP_FILEPATH_PLUGIN_DIR}/barcodeList.txt"
SCRIPTSDIR="${DIRNAME}/scripts"
JSON_RESULTS="${TSP_FILEPATH_PLUGIN_DIR}/results.json"
HTML_RESULTS="${PLUGINNAME}.html"
HTML_BLOCK="${PLUGINNAME}_block.html"
HTML_ROWSUMS="${PLUGINNAME}_rowsum"
HTML_TORRENT_WRAPPER=1

# Definition of fields displayed in barcode link/summary table
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
BC_COL_TITLE[1]="On Target"
BC_COL_TITLE[2]="Mean Depth"
BC_COL_HELP[0]="Number of reads that were mapped to the full."
BC_COL_HELP[1]="Percentage of mapped reads that were aligned over a target region."
BC_COL_HELP[2]="Mean average target base read depth, including non-covered target bases."
if [ $TWO_SETS_OUTPUT -ne 0 ]; then
  BC_COL_TITLE[3]="Coverage"
  BC_COL_TITLE[4]="Mapped Reads"
  BC_COL_TITLE[5]="On Target"
  BC_COL_TITLE[6]="Mean Depth"
  BC_COL_TITLE[7]="Coverage"
  BC_COL_HELP[3]="Percentage of target bases that were covered by at least one read."
  BC_COL_HELP[4]="Number of unique starts reads that were mapped to the full reference."
  BC_COL_HELP[5]="Percentage of unique starts that were aligned over a target region."
  BC_COL_HELP[6]="Mean average target base read depth using unique starts, including non-covered target bases."
  BC_COL_HELP[7]="Percentage of target bases that were covered by at least one read using unique starts."
else
  BC_COL_TITLE[3]="Uniformity"
  BC_COL_TITLE[4]="1x Coverage"
  BC_COL_TITLE[5]="20x Coverage"
  BC_COL_TITLE[6]="100x Coverage"
  BC_COL_TITLE[7]="500x Coverage"
  BC_COL_HELP[3]="Percentage of target bases covered by at least 0.2x the average base coverage depth."
  BC_COL_HELP[4]="Percentage of target bases that were covered by at least one read."
  BC_COL_HELP[5]="Percentage of target bases covered by at least twenty reads."
  BC_COL_HELP[6]="Percentage of target bases covered by at least one hundred reads."
  BC_COL_HELP[7]="Percentage of target bases covered by at least five hundred reads."
fi

# Set up report page layout and help text
BC_COV_PAGE_WIDTH="960px"
COV_PAGE_WIDTH="480px"
BC_SUM_ROWS=7
RUNCOV_OPTS="-s"
if [ $PLUGIN_FILTER_READS -eq 1 ];then
  # Option to run twice with and w/o filters, producing old-style report
  if [ $TWO_SETS_OUTPUT -ne 0 ]; then
    RUNCOV_OPTS=""
    BC_TITLE_INFO="Coverage summary statistics for all and filtered sampled barcoded reads."
    BC_SUM_ROWS=8
    COV_PAGE_WIDTH="960px"
  else
    BC_TITLE_INFO="Coverage summary statistics for filtered aligned barcoded reads."
  fi
  if [ $PLUGIN_NONDUPS = "Yes" ];then
    RUNCOV_OPTS="$RUNCOV_OPTS -d"
  fi
  if [ $PLUGIN_UMAPS = "Yes" ];then
    RUNCOV_OPTS="$RUNCOV_OPTS -u"
  fi
else
  BC_TITLE_INFO="Coverage summary statistics for all (un-filtered) aligned barcoded reads."
  #BC_SUM_ROWS=4
  #BC_COV_PAGE_WIDTH="620px"
fi

# Set up log options
LOGOPT=""
if [ "$PLUGIN_DEV_FULL_LOG" -gt 0 ]; then
  LOGOPT='-l'
  if [ "$PLUGIN_DEV_FULL_LOG" -gt 1 ]; then
    set -o xtrace
  fi
fi

# Source the HTML files for shell functions and define others below
for HTML_FILE in `find ${DIRNAME}/html/ | grep .sh$`
do
  source ${HTML_FILE};
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

# Link local copy of js and css
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
  GENOME="${TSP_FILEPATH_GENOME_FASTA}.fai"
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

# Generate header.html and footer.html for use in secondary results pages
#  -uses COV_PAGE_WIDTH to specify the inner page width
write_html_header
write_html_footer

# Reset COV_PAGE_WIDTH to specify the inner page width for barcode table
COV_PAGE_WIDTH=$BC_COV_PAGE_WIDTH

# Remove previous results to avoid displaying old before ready
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}" "${TSP_FILEPATH_PLUGIN_DIR}/$HTML_BLOCK" "$JSON_RESULTS"

# Check for barcodes
if [ $PLUGIN_CHECKBC -eq 1 ]; then
  barcode;
else
  # Link BAM to here for download links - hard link if using a report merged bam
  if [ -n "$PLUGINCONFIG__MERGEDBAM" ]; then
    ln -f "$TSP_FILEPATH_BAM" .
    ln -f "${TSP_FILEPATH_BAM}.bai" .
  else
    ln -sf "$TSP_FILEPATH_BAM" .
    ln -sf "${TSP_FILEPATH_BAM}.bai" .
  fi
  # Write a front page for non-barcode run
  HTML="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
  write_html_header "$HTML" 15;
  echo "<h3><center>${PLUGIN_RUN_NAME}</center></h3>" >> "$HTML"
  display_static_progress "$HTML";
  write_html_footer "$HTML";
  # Run on single bam
  RT=0
  eval "${SCRIPTSDIR}/run_coverage_analysis.sh $LOGOPT $RUNCOV_OPTS -R \"$HTML_RESULTS\" -T \"$HTML_ROWSUMS\" -H \"${TSP_FILEPATH_PLUGIN_DIR}\" -D \"$TSP_FILEPATH_PLUGIN_DIR\" -B \"$PLUGIN_EFF_TARGETS\" -P \"$PADDED_TARGETS\" -Q \"$HTML_BLOCK\" \"$TSP_FILEPATH_GENOME_FASTA\" \"${TSP_FILEPATH_BAM}\"" || RT=$?
  if [ $RT -ne 0 ]; then
    write_html_header "$HTML";
    echo "<h3><center>${PLUGIN_RUN_NAME}</center></h3>" >> "$HTML"
    echo "<br/><h3 style=\"text-align:center;color:red\">*** An error occurred - check Log File for details ***</h3><br/>" >> "$HTML"
    write_html_footer "$HTML";
    exit 1
  fi
  # Write json output
  write_json_header;
  if [ $TWO_SETS_OUTPUT -ne 0 ]; then
    write_json_inner "${TSP_FILEPATH_PLUGIN_DIR}/all_reads" "summary.txt" "all_reads" 2;
    if [ "$PLUGIN_FILTER_READS" -eq 1 ];then
      echo "," >> "$JSON_RESULTS"
      write_json_inner "${TSP_FILEPATH_PLUGIN_DIR}/filtered_reads" "summary.txt" "filtered_reads" 2;
    fi
  else
    if [ "$PLUGIN_FILTER_READS" -eq 1 ];then
      write_json_inner "${TSP_FILEPATH_PLUGIN_DIR}/filtered_reads" "summary.txt" "" 2;
    else
      write_json_inner "${TSP_FILEPATH_PLUGIN_DIR}/all_reads" "summary.txt" "" 2;
    fi
  fi
  write_json_footer;
  rm -f "${TSP_FILEPATH_PLUGIN_DIR}/$HTML_ROWSUMS"
fi
# Remove after successful completion
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/header" "${TSP_FILEPATH_PLUGIN_DIR}/footer" "$PADDED_TARGETS" "$BARCODES_LIST"

