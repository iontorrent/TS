#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

VERSION="2.0.0.0"

#AUTORUNDISABLE

# disable excess debug output for this test machine
set +o xtrace

# grab PUI parameters
PLUGIN_TARGETS=$PLUGINCONFIG__UNPTARGETS
PLUGIN_PADSIZE=$PLUGINCONFIG__PADTARGETS
PLUGIN_USTARTS=$PLUGINCONFIG__UNIQUESTARTS
PLUGIN_TARGETID=$PLUGINCONFIG__TARGETSEQID

# temporary measure to avoid bug wrt to spaces in passing data from instance.html
PLUGIN_TARGETID=`echo "$PLUGIN_TARGETS" | sed -e 's/^.*targetseq\.//' | sed -e 's/\.bed$//' | sed -e 's/^\s+//' | sed -e 's/\./: /' | sed -e 's/_/ /g'`
PLUGIN_TARGETID=`echo "$PLUGIN_TARGETID" | sed -e 's/^\(.*: \)\(.\)\(.*\)$/\1\u\2\3/'`

if [ -n "$PLUGIN_USTARTS" ]; then
  PLUGIN_USTARTS="Yes"
else
  PLUGIN_USTARTS="No"
fi

echo "Selected run options:" >&2
echo "  Target regions: $PLUGIN_TARGETID" >&2
echo "  Target padding: $PLUGIN_PADSIZE" >&2
echo "  Examine unique starts: $PLUGIN_USTARTS" >&2

if ! [ -d "$TSP_FILEPATH_PLUGIN_DIR" ]; then
    echo "ERROR: Failed to locate output directory $TSP_FILEPATH_PLUGIN_DIR" >&2
    exit 1
fi

# --- the following code allows the re-use of the barcodes() table presentation ---

# Source the HTML files
for HTML_FILE in `find ${DIRNAME}/html/ | grep .sh$`
do
  source ${HTML_FILE};
done

#*! @function
#  @param  $*  the command to be executed
run ()
{
  eval $* >&2
  EXIT_CODE="$?"
  if [ ${EXIT_CODE} != 0 ]; then
    rm -v "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
    echo "status code '${EXIT_CODE}' while running '$*'" >&2
    exit 1
  fi
}

#*! @function
#  @param $1 Directory path to create output
#  @param $2 Filepath to BAM file
run_targetseq_analysis ()
{
  local RESDIR="$1"
  local BAMFILE="$2"
  local RUNCOV="${RUNDIR}/run_coverage_analysis.sh $RUNCOV_OPTS -R \"$HTML_RESULTS\" -T \"$HTML_ROWSUMS\" -H \"${TSP_FILEPATH_PLUGIN_DIR}\" -D \"$RESDIR\" -B \"$PLUGIN_TARGETS\" -P \"$PADDED_TARGETS\" \"$TSP_FILEPATH_GENOME_FASTA\" \"$BAMFILE\""
  eval "$RUNCOV || cleanup_on_error" >&2
}

#*! @function
#  @param $1 Name of JSON file to append to
#  @param $2 Path to file composed of <name>:<value> lines
#  @param $3 dataset (e.g. "filtered_reads")
#  @param $4 printing indent level. Default: 2
append_to_json_results ()
{
  local JSONFILE="$1"
  local DATAFILE="$2"
  local DATASET="$3"
  local INDENTLEV="$4"
  if [ -z $INDENTLEV ]; then
    INDENTLEV=2
  fi
  local JSONCMD="perl ${RUNDIR}/coverage_analysis_json.pl -a -I $INDENTLEV -B \"$DATASET\" \"$DATAFILE\" \"$JSONFILE\""
  eval "$JSONCMD || echo \"WARNING: Failed to write to JSON from $DATAFILE\"" >&2
}

#*! @function
cleanup_on_error ()
{
  echo -e "\nRemoving temporary files..." >&2
  # avoiding using rm !(drmaa_stdout.txt) because of danger
  TMPFILE="/results/plugins/scratch/${PLUGINNAME}_$RANDOM$RANDOM_drmaa_stdout.txt"
  mv "${TSP_FILEPATH_PLUGIN_DIR}/drmaa_stdout.txt" "$TMPFILE"
  # no log output so last error is easy to visualize
  rm -rf "$TSP_FILEPATH_PLUGIN_DIR"/*
  mv "$TMPFILE" "${TSP_FILEPATH_PLUGIN_DIR}/drmaa_stdout.txt"
  exit 1;
}

run "mkdir -p ${TSP_FILEPATH_PLUGIN_DIR}/js";
run "cp ${DIRNAME}/js/*.js ${TSP_FILEPATH_PLUGIN_DIR}/js/.";
run "mkdir -p ${TSP_FILEPATH_PLUGIN_DIR}/css";
run "cp ${DIRNAME}/css/*.css ${TSP_FILEPATH_PLUGIN_DIR}/css/.";

echo -e "\nResults folder initialized.\n" >&2

RUNDIR="${DIRNAME}/scripts"

# Create padded targets file
PADDED_TARGETS=""
if [ $PLUGIN_PADSIZE -gt 0 ];then
  GENOME="${TSP_FILEPATH_GENOME_FASTA}.fai"
  if ! [ -f "$GENOME" ]; then
    echo "WARNING: Could not create padded targets file; genome (.fai) file does not exist at $GENOME" >&2
    echo "- Continuing without padded targets analysis." >&2
  else
    PADDED_TARGETS="${TSP_FILEPATH_PLUGIN_DIR}/padded_targets_$PLUGIN_PADSIZE.bed"
    PADCMD="$RUNDIR/../padbed/padbed.sh \"$PLUGIN_TARGETS\" \"$GENOME\" $PLUGIN_PADSIZE \"$PADDED_TARGETS\""
    eval "$PADCMD" >&2
    if [ $? -ne 0 ]; then
      echo "WARNING: Could not create padded targets file; padbed.sh failed." >&2
      echo "\$ $REMDUP" >&2
      echo "- Continuing without padded targets analysis." >&2
      PADDED_TARGETS=""
    fi
  fi
  echo >&2
fi

PLUGIN_OUT_BAM_NAME=`echo ${TSP_FILEPATH_BAM} | sed -e 's_.*/__g'`

JSON_RESULTS="${TSP_FILEPATH_PLUGIN_DIR}/results.json"
HTML_RESULTS="${PLUGINNAME}.html"
HTML_ROWSUMS="${PLUGINNAME}_rowsum"

if [ "$PLUGIN_USTARTS" == "Yes" ];then
  RUNCOV_OPTS=""
  BC_SUM_ROWS=8
  BC_COV_PAGE_WIDTH=960
  COV_PAGE_WIDTH=960
else
  RUNCOV_OPTS="-s"
  BC_SUM_ROWS=4
  BC_COV_PAGE_WIDTH=620
  COV_PAGE_WIDTH=480
fi

PLUGIN_INFO_ALLREADS="All mapped reads assigned to this barcode set."
PLUGIN_INFO_USTARTS="Uniquely mapped reads sampled for one starting alignment to each reference base in both read orientations."

# definition of fields displayed in barcode link/summary table
BC_COL_TITLE[0]="Mapped Reads"
BC_COL_TITLE[1]="On Target"
BC_COL_TITLE[2]="Mean Depth"
BC_COL_TITLE[3]="Coverage"
BC_COL_TITLE[4]="Mapped Reads"
BC_COL_TITLE[5]="On Target"
BC_COL_TITLE[6]="Mean Depth"
BC_COL_TITLE[7]="Coverage"
BC_COL_HELP[0]="Number of reads that were mapped to the full reference for this barcode set."
BC_COL_HELP[1]="Percentage of mapped reads that were aligned over a target region."
BC_COL_HELP[2]="Mean average target base read depth, including non-covered target bases."
BC_COL_HELP[3]="Percentage of all target bases that were covered to at least 1x read depth."
BC_COL_HELP[4]="Number of unique starts that were mapped to the full reference for this barcode set."
BC_COL_HELP[5]="Percentage of unique starts that were aligned over a target region."
BC_COL_HELP[6]="Mean average target base read depth using unique starts, including non-covered target bases."
BC_COL_HELP[7]="Percentage of all target bases that were covered to at least 1x read depth using unique starts."

# Generate header.html and footer.html for use in secondary results pages
# Users COV_PAGE_WIDTH to specify the inner page width
write_html_header
write_html_footer

# Reset COV_PAGE_WIDTH to specify the inner page width for barcode table
COV_PAGE_WIDTH=$BC_COV_PAGE_WIDTH

# Remove previous results to avoid displaying old before ready
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"

# Check for barcodes
if [ -f ${TSP_FILEPATH_BARCODE_TXT} ]; then
  barcode;
else
  # link BAM to here for download links
  ln -sf "$TSP_FILEPATH_BAM" .
  ln -sf "${TSP_FILEPATH_BAM}.bai" .
  # write a front page for non-barcode run
  HTML="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
  write_html_header "$HTML" 15;
  echo "<h3><center>${PLUGIN_OUT_BAM_NAME}</center></h3>" >> "$HTML"
  display_static_progress "$HTML";
  write_html_footer "$HTML";
  # run on single bam
  run_targetseq_analysis "$TSP_FILEPATH_PLUGIN_DIR" "$TSP_FILEPATH_BAM"
  # write json output
  write_json_header;
  write_json_inner "$TSP_FILEPATH_PLUGIN_DIR" "all_reads";
  write_json_inner "$TSP_FILEPATH_PLUGIN_DIR" "filtered_reads";
  write_json_footer;
fi
# Remove after successful completion
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/header" "${TSP_FILEPATH_PLUGIN_DIR}/footer" "${TSP_FILEPATH_PLUGIN_DIR}/startplugin.json" "$PADDED_TARGETS"

