#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#MAJOR_BLOCK

VERSION="3.4.51836"

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
  PLUGINCONFIG__SAMPLEID="No"
  PLUGINCONFIG__TRIMREADS="No"
  PLUGINCONFIG__PADTARGETS=0
  PLUGINCONFIG__UNIQUEMAPS="Yes"
  PLUGINCONFIG__NONDUPS="Yes"
  PLUGINCONFIG__BARCODETARGETREGIONS=""
  if [ -z "$PLUGINCONFIG__LIBRARYTYPE" ]; then
    rm -f "${TSP_FILEPATH_PLUGIN_DIR}/results.json"
    HTML="${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
    echo '<html><body>' > "$HTML"
    if [ -f "${DIRNAME}/html/logo.sh" ]; then
      source "${DIRNAME}/html/logo.sh"
      print_html_logo >> "$HTML";
    fi
    echo "<h3><center>${PLUGIN_RUN_NAME}</center></h3>" >> "$HTML"
    echo "<br/><h2 style=\"text-align:center;color:red\">*** Automatic analysis was not performed. ***</h2>" >> "$HTML"
    echo "<br/><h3 style=\"text-align:center\">(Requires an associated Plan to specify the Run Type.)</h3></br>" >> "$HTML"
    echo '</body></html>' >> "$HTML"
    exit
  elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "ampliseq" ]; then
    PLUGINCONFIG__LIBRARYTYPE_ID="Ion AmpliSeq"
    # Primer trimming disabled until future resolution of undesired side-efects
    #PLUGINCONFIG__TRIMREADS="Yes"
    PLUGINCONFIG__NONDUPS="No"
  elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "ampliseq-rna" ]; then
    PLUGINCONFIG__LIBRARYTYPE_ID="Ion AmpliSeq RNA"
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
  if [ -n "$PLUGINCONFIG__SAMPLEID" ]; then
    PLUGINCONFIG__SAMPLEID="Yes"
  else
    PLUGINCONFIG__SAMPLEID="No"
  fi
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
PLUGIN_ANNOFIELDS="-f 4,8"
PLUGIN_READCOV="e2e"
AMPOPT=""
if [ "$PLUGIN_RUNTYPE" = "ampliseq" ]; then
  AMPOPT="-a"
elif [ "$PLUGIN_RUNTYPE" = "ampliseq-rna" ]; then
  AMPOPT="-r"
  #PLUGIN_READCOV="c70"
else
  # used merged detail target for base coverage to assigned targets
  # otherwise base coverage may be over-counted where targets overlap
  PLUGIN_DETAIL_TARGETS=`echo "$PLUGIN_DETAIL_TARGETS" | sed -e 's/\/unmerged\//\/merged\//'`
fi
PLUGIN_SAMPLEID=$PLUGINCONFIG__SAMPLEID
PLUGIN_TRIMREADS=$PLUGINCONFIG__TRIMREADS
PLUGIN_PADSIZE=$PLUGINCONFIG__PADTARGETS
PLUGIN_UMAPS=$PLUGINCONFIG__UNIQUEMAPS
PLUGIN_NONDUPS=$PLUGINCONFIG__NONDUPLICATES
PLUGIN_TRGSID=`echo "$PLUGIN_TARGETS" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`

PLUGIN_USE_TARGETS=0
if [ -n "$PLUGIN_TARGETS" ];then
  PLUGIN_USE_TARGETS=1
fi
PLUGIN_BC_TARGETS=$PLUGINCONFIG__BARCODETARGETREGIONS
PLUGIN_MULTIBED="No"
if [ -n "$PLUGIN_BC_TARGETS" ];then
  PLUGIN_MULTIBED="Yes"
  PLUGIN_USE_TARGETS=1
fi
BC_MAPPED_BED=""
PLUGIN_CHECKBC=1

# Absolute plugin path to fixed sampleID panel (may become dynamic later)

PLUGIN_ROOT=`dirname $DIRNAME`
PLUGIN_SAMPLEID_REGIONS=""
if [ "$PLUGIN_SAMPLEID" = "Yes" ];then
  PLUGIN_SAMPLEID_REGIONS="${PLUGIN_ROOT}/sampleID/targets/KIDDAME_sampleID_regions.bed"
  if [ ! -e "$PLUGIN_SAMPLEID_REGIONS" ]; then
    echo "WARNING: Cannot locate sampleID regions file: ${PLUGIN_SAMPLEID_REGIONS}" >&2
    echo " -- Continuing analysis without SampleID Tracking option." >&2
    PLUGIN_SAMPLEID_REGIONS=""
    PLUGIN_SAMPLEID="No"
  fi
fi

# Report on user/plan option selection and processed options

echo "Selected run options:" >&2
echo "  Library Type:      $PLUGINCONFIG__LIBRARYTYPE_ID" >&2
echo "  Target Regions:    $PLUGINCONFIG__TARGETREGIONS_ID" >&2
echo "  Target Padding:    $PLUGINCONFIG__PADTARGETS" >&2
echo "  SampleID Tracking: $PLUGINCONFIG__SAMPLEID" >&2
echo "  Barcoded Targets:  $PLUGIN_MULTIBED" >&2
echo "  Trim Reads:        $PLUGINCONFIG__TRIMREADS" >&2
echo "  Uniquely Mapped:   $PLUGINCONFIG__UNIQUEMAPS" >&2
echo "  Non-duplicate:     $PLUGINCONFIG__NONDUPLICATES" >&2

echo "Employed run options:" >&2
echo "  Reference Genome:  $REFERENCE" >&2
echo "  Aligned Reads:     $TSP_FILEPATH_BAM" >&2
echo "  Library Type:      $PLUGIN_RUNTYPE" >&2
echo "  Target Regions:    $PLUGIN_DETAIL_TARGETS" >&2
echo "  Merged Regions:    $PLUGIN_TARGETS" >&2
echo "  SampleID Tracking: $PLUGIN_SAMPLEID" >&2
echo "  Target Padding:    $PLUGIN_PADSIZE" >&2
echo "  Trim Reads:        $PLUGIN_TRIMREADS" >&2
echo "  Uniquely Mapped:   $PLUGIN_UMAPS" >&2
echo "  Non-duplicate:     $PLUGIN_NONDUPS" >&2

declare -A BARCODE_TARGET_MAP
if [ -n "$PLUGIN_BC_TARGETS" ];then
  IFS=';' read -a BC_TARGET_MAP <<< "$PLUGIN_BC_TARGETS"
  echo "  Barcoded Targets:" >&2
  for BCTRGMAP in "${BC_TARGET_MAP[@]}"
  do
    BCKEY=`echo "$BCTRGMAP" | sed -e 's/=.*$//'`
    BCVAL=`echo "$BCTRGMAP" | sed -e 's/^.*=//'`
    echo "    $BCKEY -> $BCVAL" >&2
    BARCODE_TARGET_MAP["$BCKEY"]="$BCVAL"
  done
fi

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

# Help text (etc.) for filterin messages (legacy)
HTML_TORRENT_WRAPPER=1
PLUGIN_FILTER_READS=0
PLUGIN_INFO_FILTERED="Coverage statistics for uniquely mapped non-duplicate reads."
if [ "$PLUGIN_UMAPS" = "Yes" ];then
  PLUGIN_FILTER_READS=1
  if [ "$PLUGIN_NONDUPS" = "No" ];then 
    PLUGIN_INFO_FILTERED="Coverage statistics for uniquely mapped reads."
  fi
elif [ "$PLUGIN_NONDUPS" = "Yes" ];then
  PLUGIN_FILTER_READS=1
  PLUGIN_INFO_FILTERED="Coverage statistics for non-duplicate reads."
fi
PLUGIN_INFO_ALLREADS="Coverage statistics for all (unfiltered) aligned reads."

# Definition of fields and customization in barcode summary table
BC_COL_TITLE[0]="Mapped Reads"
BC_COL_HELP[0]="Number of reads that were mapped to the full reference genome."
BC_COL_TITLE[1]="On Target"
BC_COL_HELP[1]="Percentage of mapped reads that were aligned over a target region."
BC_COL_TITLE[2]="SampleID"
BC_COL_HELP[2]="The percentage of filtered reads mapped to any targeted region used for sample identification."
BC_COL_TITLE[3]="Mean Depth"
BC_COL_HELP[3]="Mean average target base read depth, including non-covered target bases."
BC_COL_TITLE[4]="Uniformity"
BC_COL_HELP[4]="Percentage of target bases covered by at least 0.2x the average base read depth."

COV_PAGE_WIDTH="720px"
BC_SUM_ROWS=5
if [ "$AMPOPT" = "-r" ]; then
  # no mean depth/uniformity columns
  if [ "$PLUGIN_SAMPLEID" = "Yes" ];then
    BC_SUM_ROWS=3
  else
    BC_SUM_ROWS=2
  fi
elif [ -z "$PLUGIN_DETAIL_TARGETS" -a -z "$PLUGIN_BC_TARGETS" ]; then
  if [ "$PLUGIN_SAMPLEID" = "Yes" ];then
    # remove On Target
    BC_SUM_ROWS=4
    BC_COL_TITLE[1]=${BC_COL_TITLE[2]}
    BC_COL_HELP[1]=${BC_COL_HELP[2]}
    BC_COL_TITLE[2]=${BC_COL_TITLE[3]}
    BC_COL_HELP[2]=${BC_COL_HELP[3]}
    BC_COL_TITLE[3]=${BC_COL_TITLE[4]}
    BC_COL_HELP[3]=${BC_COL_HELP[4]}
  else
    # remove On Target and SampleID
    BC_SUM_ROWS=3
    BC_COL_TITLE[1]=${BC_COL_TITLE[3]}
    BC_COL_HELP[1]=${BC_COL_HELP[3]}
    BC_COL_TITLE[2]=${BC_COL_TITLE[4]}
    BC_COL_HELP[2]=${BC_COL_HELP[4]}
  fi
elif [ "$PLUGIN_SAMPLEID" = "No" ];then
  # remove SampleID
  BC_SUM_ROWS=4
  BC_COL_TITLE[2]=${BC_COL_TITLE[3]}
  BC_COL_HELP[2]=${BC_COL_HELP[3]}
  BC_COL_TITLE[3]=${BC_COL_TITLE[4]}
  BC_COL_HELP[3]=${BC_COL_HELP[4]}
fi

FILTOPTS=""
if [ $PLUGIN_FILTER_READS -eq 1 ];then
  # Option to run twice with and w/o filters, producing old-style report
  BC_TITLE_INFO="Coverage summary statistics for filtered aligned barcoded reads."
  if [ "$PLUGIN_NONDUPS" = "Yes" ];then
    FILTOPTS="$FILTOPTS -d"
  fi
  if [ "$PLUGIN_UMAPS" = "Yes" ];then
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

# Direct PLUGIN_TRIMREADS to detect trimp option
TRIMOPT=""
if [ "$PLUGIN_TRIMREADS" = "Yes" ]; then
  TRIMOPT="-t"
fi

# --------- Source the functions/*.sh files for (shell) functions ----------

for BASH_FILE in `find ${DIRNAME}/functions/ | grep .sh$`
do
  source ${BASH_FILE};
done

# --------- Start processing the data ----------

# Remove previous results to avoid displaying old before ready
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}" "${TSP_FILEPATH_PLUGIN_DIR}/$HTML_BLOCK" "$JSON_RESULTS"
rm -f "$PLUGIN_OUT_COVERAGE_HTML"
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*.stats.cov.txt ${TSP_FILEPATH_PLUGIN_DIR}/*.xls ${TSP_FILEPATH_PLUGIN_DIR}/*.png
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*.bam* ${TSP_FILEPATH_PLUGIN_DIR}/*.bed*
rm -rf ${TSP_FILEPATH_PLUGIN_DIR}/static_links

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
create_padded_targets "$PLUGIN_TARGETS" $PLUGIN_PADSIZE "$TSP_FILEPATH_PLUGIN_DIR"
PLUGIN_EFF_TARGETS=$CREATE_PADDED_TARGETS

# Reserved for re-version of script to producing padded targets as parallel report
PADDED_TARGETS=""

# Create GC annotated BED file for read-to-target assignment
gc_annotate_bed "$PLUGIN_DETAIL_TARGETS" "$TSP_FILEPATH_PLUGIN_DIR"
GCANNOBED=$GC_ANNOTATE_BED

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
  eval "${SCRIPTSDIR}/run_coverage_analysis.sh $LOGOPT $FILTOPTS $AMPOPT $TRIMOPT -R \"$HTML_RESULTS\" -D \"$TSP_FILEPATH_PLUGIN_DIR\" -A \"$GCANNOBED\" -B \"$PLUGIN_EFF_TARGETS\" -C \"$PLUGIN_TRGSID\" -P \"$PADDED_TARGETS\" -p $PLUGIN_PADSIZE -Q \"$HTML_BLOCK\" -S \"$PLUGIN_SAMPLEID_REGIONS\" \"$REFERENCE\" \"${PLUGIN_RUN_NAME}.bam\"" || RT=$?
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

