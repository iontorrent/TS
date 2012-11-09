#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
#AUTORUNDISABLED
VERSION="2.2.3-1"

# Option for how to use plugin; set to 1 to make each report be a different html
ALLOW_MULTIPLE_OUTPUTS_PER_RUN=0

PLUGIN_RESULTS_NAME="$PLUGINCONFIG__BAM_NAME"
PLUGIN_RUN_LIST="$PLUGINCONFIG__RUN_LIST"
PLUGIN_NUM_RUNS="$PLUGINCONFIG__NUM_RUNS"
PLUGIN_TOTAL_AQ17="$PLUGINCONFIG__TOTAL_AQ17"
PLUGIN_HOST_URL="$PLUGINCONFIG__HOST_URL"

PLUGIN_OUT_BAMFILE="${PLUGIN_RESULTS_NAME}.bam"
PLUGIN_OUT_BAIFILE="${PLUGIN_OUT_BAMFILE}.bai"

REFERENCE="$TSP_FILEPATH_GENOME_FASTA"
REFERENCE_FAI="${REFERENCE}.fai"

echo "Employed run settings:" >&2
echo "  Reference Genome:  $REFERENCE" >&2
echo "  PGM Flow Order:    $TSP_FLOWORDER" >&2
echo "  Output File Name:  $PLUGIN_RESULTS_NAME" >&2
echo "  Runs To Merge:     $PLUGIN_NUM_RUNS" >&2
echo "  Total AQ17 Reads:  $PLUGIN_TOTAL_AQ17" >&2
#echo "  Host URL:          $PLUGIN_HOST_URL" >&2
echo "  API URL:           $RUNINFO__API_URL" >&2
echo "Selected runs to combine:" >&2
echo "  $PLUGIN_RUN_LIST" >&2
echo "" >&2

# Definition of file names, locations, etc., used by plugin
PLUGIN_RUN_NAME="$TSP_RUN_NAME"
JSON_RESULTS="${TSP_FILEPATH_PLUGIN_DIR}/results.json"

if [ $ALLOW_MULTIPLE_OUTPUTS_PER_RUN -eq 0 ]; then
    HTML_RESULTS="${PLUGINNAME}.html"
else
    HTML_RESULTS="${PLUGIN_RESULTS_NAME}.html"
fi

PLUGIN_OUT_LIST_HTML="REPORTLIST_html"
PLUGIN_OUT_RESULTS_HTML="MERGEREPORT_html"
PLUGIN_OUT_LIST_BAMS="${TSP_FILEPATH_PLUGIN_DIR}/bamfiles.txt"

# Defines the overall width of the page drawn
COV_PAGE_WIDTH=1000

# Source the HTML files for shell functions and define others below
for HTML_FILE in `find ${DIRNAME}/html/ | grep .sh$`
do
    source ${HTML_FILE};
done

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
    echo " <h3 style=\"text-align:center\">${RUNID}</h3><br/>" >> "$HTMLOUT"
    echo " <h4>Reports selected: ${PLUGIN_NUM_RUNS} &nbsp;&nbsp;&nbsp;Total AQ17 Reads: ${PLUGIN_TOTAL_AQ17}</h4><br/>" >> "$HTMLOUT"
    write_html_report_list "${OUTDIR}/$PLUGIN_OUT_LIST_HTML" >> "$HTMLOUT"
    if [ -n "$BAMFILE" ]; then
        cat "${OUTDIR}/$PLUGIN_OUT_RESULTS_HTML" >> "$HTMLOUT"
        write_html_file_links "$OUTURL" "$OUTDIR" >> "$HTMLOUT";
    else
        display_static_progress "$HTMLOUT";
    fi
    write_html_footer "$HTMLOUT";

    # Remove temporary files
    if [ -n "$BAMFILE" ]; then
        rm -f "${OUTDIR}/$PLUGIN_OUT_LIST_HTML" "${OUTDIR}/$PLUGIN_OUT_RESULTS_HTML" "$PLUGIN_OUT_LIST_BAMS"
    fi
    return 0
}

# Remove previous results to avoid displaying old before ready (or to support hard links from other plugins)
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}" "$JSON_RESULTS"
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LIST_HTML" "${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_RESULTS_HTML" "$PLUGIN_OUT_LIST_BAMS"
if [ $ALLOW_MULTIPLE_OUTPUTS_PER_RUN -eq 0 ]; then
  rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*.bam ${TSP_FILEPATH_PLUGIN_DIR}/*.bam.bai
else
  rm -f "${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_BAMFILE" "${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_BAIFILE"
fi

# Make local link to css for html
run "ln -sf ${DIRNAME}/css ${TSP_FILEPATH_PLUGIN_DIR}/.";

# Generate list of reports to combine and BAM file list
run "${DIRNAME}/fetchReportData.pl -x -H \"$RUNINFO__API_URL\" -B \"${PLUGIN_OUT_LIST_BAMS}\" \"$PLUGIN_RUN_LIST\" >> \"${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LIST_HTML\""

# Write a front page for non-barcode run
write_html_results "$PLUGIN_RESULTS_NAME" "$TSP_FILEPATH_PLUGIN_DIR" "." "";

# Combine BAM files with pre-check
run "${DIRNAME}/mergeBams.pl -f -i -x \"${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_BAMFILE\" \"${PLUGIN_OUT_LIST_BAMS}\" >> \"${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_RESULTS_HTML\""
echo "> ${PLUGIN_OUT_BAIFILE}" >&2

# Re-draw results pages with statistics, warnings, or error message
write_html_results "$PLUGIN_RESULTS_NAME" "$TSP_FILEPATH_PLUGIN_DIR" "." "$PLUGIN_OUT_BAMFILE";

# Write json output
write_json;

# remove temporary files after successful completion
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/startplugin.json" ${TSP_FILEPATH_PLUGIN_DIR}/*.log

