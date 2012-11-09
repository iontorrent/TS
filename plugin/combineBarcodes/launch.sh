#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
VERSION="2.2.0.0"

PLUGIN_DEV_FULL_LOG=0;

# User inputs (default for autorun)
INPUT_GROUPING=0
INPUT_SPACING=1
INPUT_MIN_MAP=5000
INPUT_COL_GRP='-c'
INPUT_USTARTS='-u'
INPUT_DUP_LENVAR=0
INPUT_GRP_USTARTS='-g'
INPUT_COMBINE_ALL=''
if [ -n "$PLUGINCONFIG__GROUPING" ]; then
  INPUT_GROUPING=$PLUGINCONFIG__GROUPING
  INPUT_SPACING=$PLUGINCONFIG__SPACING
  INPUT_MIN_MAP=$PLUGINCONFIG__READSTHRESHOLD
  if [ "$PLUGINCONFIG__COLLAPSE" != "Yes" ]; then
    INPUT_COL_GRP=''
    PLUGINCONFIG__COLLAPSE="No"
  fi
  if [ "$PLUGINCONFIG__UNIQUESTARTS" != "Yes" ]; then
    INPUT_USTARTS=''
    PLUGINCONFIG__UNIQUESTARTS="No"
    PLUGINCONFIG__GROUPUNIQUESTARTS="No"
    PLUGINCONFIG__LENGTHVAR=0
  fi
  INPUT_DUP_LENVAR=$PLUGINCONFIG__LENGTHVAR
  if [ "$PLUGINCONFIG__GROUPUNIQUESTARTS" != "Yes" ]; then
    INPUT_GRP_USTARTS=''
    PLUGINCONFIG__GROUPUNIQUESTARTS="No"
  fi
  if [ "$PLUGINCONFIG__COMBINEGROUPS" == "Yes" ]; then
    INPUT_COMBINE_ALL='-a'
  else
    PLUGINCONFIG__COMBINEGROUPS="No"
  fi
else
  PLUGINCONFIG__GROUPING=$INPUT_GROUPING
  PLUGINCONFIG__SPACING=$INPUT_SPACING
  PLUGINCONFIG__READSTHRESHOLD=$INPUT_MIN_MAP
  PLUGINCONFIG__COLLAPSE="Yes"
  PLUGINCONFIG__LENGTHVAR=0
  PLUGINCONFIG__UNIQUESTARTS="Yes"
  PLUGINCONFIG__GROUPUNIQUESTARTS="Yes"
  PLUGINCONFIG__COMBINEGROUPS="No"
fi

echo "Selected run options:" >&2
echo "  Barcode Grouping:       $PLUGINCONFIG__GROUPING" >&2
echo "  Barcode Spacing:        $PLUGINCONFIG__SPACING" >&2
echo "  Mapped Reads Threshold: $PLUGINCONFIG__READSTHRESHOLD" >&2
echo "  Collapse Grouping:      $PLUGINCONFIG__COLLAPSE" >&2
echo "  Create Unique Starts:   $PLUGINCONFIG__UNIQUESTARTS" >&2
if [ "$PLUGINCONFIG__UNIQUESTARTS" = "Yes" ]; then
  echo "  Read End Resolution:    $PLUGINCONFIG__LENGTHVAR" >&2
  echo "  Group Unique Starts:    $PLUGINCONFIG__GROUPUNIQUESTARTS" >&2
fi
echo "  Combine Groups:         $PLUGINCONFIG__COMBINEGROUPS" >&2

PLUGIN_BAM_FILE=`echo "$TSP_FILEPATH_BAM" | sed -e 's/^.*\///'`
PLUGIN_BAM_NAME=`echo $PLUGIN_BAM_FILE | sed -e 's/\.[^.]*$//'`

PLUGIN_OUT_BAMDIR="${TSP_FILEPATH_PLUGIN_DIR}/barcodeGroups"

PLUGIN_OUT_BAMFILE="combineBarcodes.bam"
PLUGIN_OUT_BAIFILE="${PLUGIN_OUT_BAMFILE}.bai"
PLUGIN_OUT_USTARTS_BAMFILE="combineBarcodes.ustarts.bam"
PLUGIN_OUT_USTARTS_BAIFILE="${PLUGIN_OUT_USTARTS_BAMFILE}.bai"
PLUGIN_OUT_BARCODES="combineBarcodes.bclist.bam"

PLUGIN_OUT_STATS_HTML="${TSP_FILEPATH_PLUGIN_DIR}/bcgroup_stats_html"

# Definition of file names, locations, etc., used by plugin
BARCODES_LIST="${TSP_FILEPATH_PLUGIN_DIR}/barcodeList.txt"
PLUGIN_RUN_NAME="$TSP_RUN_NAME"
JSON_RESULTS="${TSP_FILEPATH_PLUGIN_DIR}/results.json"
HTML_RESULTS="${PLUGINNAME}.html"

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

# Page width to center on and set table sizes
COV_PAGE_WIDTH=800

# For collecting simple results stats
MERGE_REPORT=""

# Creates the body of the detailed report post-analysis
# This would be modified to handle individual barcode reports
# Pass "" as 4th arg for non-complete page draw.
write_html_results ()
{
  local RUNID=${1}
  local OUTDIR=${2}
  local OUTURL=${3}
  local BAMFILE=${4}

  local HTMLOUT="${OUTDIR}/${HTML_RESULTS}";

  # Create the html report page
  echo "Generating html report..." >&2
  if [ -n "$BAMFILE" ]; then
    write_html_header "$HTMLOUT";
  else
    write_html_header "$HTMLOUT" 15;
  fi
  echo "<h3 style=\"text-align:center\">${RUNID}</h3><br/>" >> "$HTMLOUT"
  if [ -n "$BAMFILE" ]; then
    if [ -n "$INPUT_USTARTS" ]; then
      echo '<table style="width:100%;text-align:center;margin-left:auto;margin-right:auto;background:white">' >> "$HTMLOUT"
      echo "<tr><th style='text-align:center;width:240px !important'>Combined Barcodes</th>" >> "$HTMLOUT"
      echo "<th style='text-align:center;width:100px !important'>Mapped Reads</th>" >> "$HTMLOUT"
      echo "<th style='text-align:center;width:100px !important'>Unique Starts</th>" >> "$HTMLOUT"
      echo "<th style='text-align:center;width:100px !important'>All Reads</th>" >> "$HTMLOUT"
      echo "<th style='text-align:center;width:100px !important'>Filtered Reads</th></tr>" >> "$HTMLOUT"
    else
      echo '<table style="width:100%;text-align:center;margin-left:auto;margin-right:auto;background:white">' >> "$HTMLOUT"
      echo "<tr><th style='text-align:center;width:340px !important'>Combine Barcodes</th>" >> "$HTMLOUT"
      echo "<th style='text-align:center;width:100px !important'>Mapped Reads</th>" >> "$HTMLOUT"
      echo "<th style='text-align:center;width:100px !important'>All Reads</th></tr>" >> "$HTMLOUT"
    fi
    cat "$PLUGIN_OUT_STATS_HTML" >> "$HTMLOUT"
    echo '</table><br/>' >> "$HTMLOUT"
    #if [ -n "$INPUT_COMBINE_ALL" ]; then
    #  write_html_file_links "$OUTURL" "$OUTDIR" >> "$HTMLOUT";
    #fi
  else
    display_static_progress "$HTMLOUT";
  fi
  write_html_footer "$HTMLOUT"
  return 0
}

# Remove previous results to avoid displaying old before ready (or to support hard links from other plugins)
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}" "$JSON_RESULTS"
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*.bam ${TSP_FILEPATH_PLUGIN_DIR}/*.bam.bai
rm -rf "$PLUGIN_OUT_BAMDIR"

# create a new folder to hold the new groups of barcodes
if [ "$INPUT_GROUPING" -gt 0 ]; then
  run "mkdir \"$PLUGIN_OUT_BAMDIR\""
else
  PLUGIN_OUT_BAMDIR="$TSP_FILEPATH_PLUGIN_DIR"
fi

# Make link to css folder
run "ln -sf ${DIRNAME}/css ${TSP_FILEPATH_PLUGIN_DIR}/.";

# Write not-completed HTML output
write_html_results "$PLUGIN_RUN_NAME" "$TSP_FILEPATH_PLUGIN_DIR" "." "";

# Check if running on a barcoded run
if [ -f $TSP_FILEPATH_BARCODE_TXT ]; then
  # Local copy of sorted barcode list file
  run "sort -t ' ' -k 2n,2 \"$TSP_FILEPATH_BARCODE_TXT\" > \"$BARCODES_LIST\"";
  # Run grouped barcode generater
  run "${DIRNAME}/combineBarcodes.sh -i -B \"$BARCODES_LIST\" $LOGOPT -L $INPUT_DUP_LENVAR $INPUT_COL_GRP $INPUT_USTARTS $INPUT_GRP_USTARTS $INPUT_COMBINE_ALL -G $INPUT_GROUPING -E $INPUT_SPACING -T $INPUT_MIN_MAP -D \"${PLUGIN_OUT_BAMDIR}\" -S \"$PLUGIN_OUT_STATS_HTML\" \"$TSP_FILEPATH_BAM\""
  # Optional complete merge of resulting groups
  if [ -n "$INPUT_COMBINE_ALL" -a "$INPUT_GROUPING" -gt 0 ]; then
    echo "Relocating combined barcode group files..." >&2
    run "mv \"${PLUGIN_OUT_BAMDIR}/combineBarcodes.bam\" \"${TSP_FILEPATH_PLUGIN_DIR}/combineBarcodes.bam\""
    run "mv \"${PLUGIN_OUT_BAMDIR}/combineBarcodes.bam.bai\" \"${TSP_FILEPATH_PLUGIN_DIR}/combineBarcodes.bam.bai\""
    if [ -n "$INPUT_USTARTS" ]; then
      run "mv \"${PLUGIN_OUT_BAMDIR}/combineBarcodes.ustarts.bam\" \"${TSP_FILEPATH_PLUGIN_DIR}/combineBarcodes.ustarts.bam\""
      run "mv \"${PLUGIN_OUT_BAMDIR}/combineBarcodes.ustarts.bam.bai\" \"${TSP_FILEPATH_PLUGIN_DIR}/combineBarcodes.ustarts.bam.bai\""
    fi
  else
    # Create empty files for user plugins to pick up
    touch "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_BAMFILE}"
    if [ -n "$INPUT_USTARTS" ]; then
      touch "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_USTARTS_BAMFILE}"
    fi
  fi
else
  # Run barcode generater just to get unique starts
  run "${DIRNAME}/combineBarcodes.sh -i $LOGOPT -L $INPUT_DUP_LENVAR $INPUT_USTARTS -D \"${TSP_FILEPATH_PLUGIN_DIR}\" -S \"$PLUGIN_OUT_STATS_HTML\" \"$TSP_FILEPATH_BAM\""
fi

# Write JSON output

# Write completed HTML output (bamfile not used but arg needs to be non-empty)
write_html_results "$PLUGIN_RUN_NAME" "$TSP_FILEPATH_PLUGIN_DIR" "." "$PLUGIN_OUT_BAMFILE";

# Remove temporary files after successful completion
rm -f "$BARCODES_LIST" "${TSP_FILEPATH_PLUGIN_DIR}/startplugin.json"
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*.bamx
rm -f "$PLUGIN_OUT_STATS_HTML"

