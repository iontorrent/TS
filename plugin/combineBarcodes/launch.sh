#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
VERSION="2.2.0.0"

PLUGIN_DEV_FULL_LOG=0;
PLUGIN_CREATE_USTARTS=1;

REFERENCE="$TSP_FILEPATH_GENOME_FASTA"
REFERENCE_FAI="${REFERENCE}.fai"

PLUGIN_BAM_FILE=`echo "$TSP_FILEPATH_BAM" | sed -e 's/^.*\///'`
PLUGIN_BAM_NAME=`echo $PLUGIN_BAM_FILE | sed -e 's/\.[^.]*$//'`

PLUGIN_OUT_BAMFILE="combineBarcodes.bam"
PLUGIN_OUT_BAIFILE="${PLUGIN_OUT_BAMFILE}.bai"
PLUGIN_OUT_USTARTS_BAMFILE="combineBarcodes.ustarts.bam"
PLUGIN_OUT_USTARTS_BAIFILE="${PLUGIN_OUT_USTARTS_BAMFILE}.bai"

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
    echo '<style> td.inleft {width:75% !important;text-align:left;border:0;padding:1px} </style>' >> "$HTMLOUT"
    echo '<style> td.inright {width:25% !important;text-align:right;border:0;padding:1px} </style>' >> "$HTMLOUT"
    echo '<table style="width:330px;margin-left:auto;margin-right:auto;background:transparent">' >> "$HTMLOUT"
    echo "$MERGE_REPORT" >> "$HTMLOUT"
    echo "<tr><td class='inleft'>Total reads</td><td class='inright'>${STATS1[0]}</td></tr>" >> "$HTMLOUT"
    echo "<tr><td class='inleft'>Total mapped reads</td><td class='inright'>${STATS1[1]}</td></tr>" >> "$HTMLOUT"
    if [ $PLUGIN_CREATE_USTARTS -eq 1 ]; then
      echo "<tr><td class='inleft'>Total unique starts</td><td class='inright'>$STATS2</td></tr>" >> "$HTMLOUT"
    fi
    echo "<tr><td class='inleft'></td><td class='inright'>${STATS2[1]}</td></tr>" >> "$HTMLOUT"
    echo '</table><br/>' >> "$HTMLOUT"
    write_html_file_links "$OUTURL" "$OUTDIR" >> "$HTMLOUT";
  else
    display_static_progress "$HTMLOUT";
  fi
  write_html_footer "$HTMLOUT"
  return 0
}

# Remove previous results to avoid displaying old before ready (or to support hard links from other plugins)
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}" "$JSON_RESULTS"
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*.bam ${TSP_FILEPATH_PLUGIN_DIR}/*.bam.bai

# Make link to css folder
run "ln -sf ${DIRNAME}/css ${TSP_FILEPATH_PLUGIN_DIR}/.";

# Write not-completed HTML output
write_html_results "$PLUGIN_RUN_NAME" "$TSP_FILEPATH_PLUGIN_DIR" "." "";

# Check if running on a barcoded run
# NOTE: Following code is complicated because no .BAM files must be seen until completely created
if [ -f $TSP_FILEPATH_BARCODE_TXT ]; then
  # Local copy of sorted barcode list file
  run "sort -t ' ' -k 2n,2 \"$TSP_FILEPATH_BARCODE_TXT\" > \"$BARCODES_LIST\"";
  # Loop over (effective) barcodes
  BCN=0
  UBCN=0
  for BARCODE_LINE in `cat ${BARCODES_LIST} | grep "^barcode"`
  do
    BARCODE=`echo ${BARCODE_LINE} | awk 'BEGIN{FS=","} {print $2}'`
    BARCODE_BAM="${ANALYSIS_DIR}/${BARCODE}_${PLUGIN_BAM_FILE}"
    LOCALBC_BAM="${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE}.all.bamx"
    if [ -f "$BARCODE_BAM" ]; then
      echo "Processing barcode ${BARCODE}..." >&2
      UBCN=`expr ${UBCN} + 1`
      run "ln -sf \"$BARCODE_BAM\" \"$LOCALBC_BAM\""
      if [ $PLUGIN_CREATE_USTARTS -eq 1 ]; then
        run "${DIRNAME}/create_unique_starts.sh $LOGOPT -D \"${TSP_FILEPATH_PLUGIN_DIR}\" \"$LOCALBC_BAM\""
      fi
      echo "" >&2
    else
      echo -e "Skipping barcode ${BARCODE}...\n" >&2
    fi
    BCN=`expr ${BCN} + 1`
  done
  echo "Merging $UBCN barcoded alignments..." >&2
  LOCALBC_BAM="${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_BAMFILE}"
  run "samtools merge \"${LOCALBC_BAM}.tmp\" ${TSP_FILEPATH_PLUGIN_DIR}/*.all.bamx >&2" 
  run "mv \"${LOCALBC_BAM}.tmp\" \"${LOCALBC_BAM}\" "
  run "samtools index \"${PLUGIN_OUT_BAMFILE}\"" 
  MERGE_REPORT="<tr><td class='inleft'>Barcode alignments combined</td><td class='inright'>$UBCN of $BCN</td></tr>"
  if [ $PLUGIN_CREATE_USTARTS -eq 1 ]; then
    echo "Merging $UBCN barcoded unique starts alignments..." >&2
    LOCALBC_BAM="${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_USTARTS_BAMFILE}"
    run "samtools merge \"${LOCALBC_BAM}.tmp\" ${TSP_FILEPATH_PLUGIN_DIR}/*.ustarts.bamx >&2" 
    run "mv \"${LOCALBC_BAM}.tmp\" \"${LOCALBC_BAM}\" "
    run "samtools index \"${LOCALBC_BAM}\"" 
  fi
else
  MERGE_REPORT="<tr class='inleft'><td>Original reads were not barcoded.</td><td></td></tr>"
  if [ $PLUGIN_CREATE_USTARTS -eq 1 ]; then
    echo "Filtering reads to unique starts..." >&2
    LOCALBC_BAM="${TSP_FILEPATH_PLUGIN_DIR}/combineBarcodes.bamx"
    run "ln -sf \"$TSP_FILEPATH_BAM\" \"${LOCALBC_BAM}\""
    run "${DIRNAME}/create_unique_starts.sh $LOGOPT -D \"${TSP_FILEPATH_PLUGIN_DIR}\" \"${LOCALBC_BAM}\""
    run "mv \"${LOCALBC_BAM}\" \"${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_BAMFILE}\" "
    run "mv \"${TSP_FILEPATH_PLUGIN_DIR}/combineBarcodes.ustarts.bamx\" \"${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_USTARTS_BAMFILE}\" "
    run "samtools index \"${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_USTARTS_BAMFILE}\"" 
  else
    run "ln -sf \"$TSP_FILEPATH_BAM\" \"${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_BAMFILE}\""
  fi
fi

echo "Created merged barcode alignment files:" >&2
echo "> ${PLUGIN_OUT_BAMFILE}" >&2
echo "> ${PLUGIN_OUT_BAIFILE}" >&2
if [ $PLUGIN_CREATE_USTARTS -eq 1 ]; then
  echo "> ${PLUGIN_OUT_USTARTS_BAMFILE}" >&2
  echo "> ${PLUGIN_OUT_USTARTS_BAIFILE}" >&2
fi

echo "Collecting mapping statistics..." >&2

OLD_IFS="$IFS"
IFS=";"
STATS1=(`samtools flagstat "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_BAMFILE}" | awk '{++c;a[c]=$1} END {print a[1]";"a[3]}'`)
IFS="$OLD_IFS"
if [ $PLUGIN_CREATE_USTARTS -eq 1 ]; then
  STATS2=`samtools flagstat "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_USTARTS_BAMFILE}" | awk '{++c;if(c==3){print $1}}'`
fi

# Write JSON output

# Write completed HTML output (bamfile not used but arg needs to be non-empty)
write_html_results "$PLUGIN_RUN_NAME" "$TSP_FILEPATH_PLUGIN_DIR" "." "$PLUGIN_OUT_BAMFILE";

# Remove temporary files after successful completion
rm -f "$BARCODES_LIST" "${TSP_FILEPATH_PLUGIN_DIR}/startplugin.json"
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*.bamx

