#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#*! @function
#  @param  $*  the command to be executed
run ()
{
  if [ "$PLUGIN_DEV_FULL_LOG" -gt 0 ]; then
    echo "\$ $*" >&2
  fi
  local EXIT_CODE=0
  eval $* >&2 || EXIT_CODE="$?"
  if [ ${EXIT_CODE} != 0 ]; then
    echo -e "ERROR: Status code '${EXIT_CODE}' while running\n\$$*" >&2
    if [ "$CONTINUE_AFTER_BARCODE_ERROR" -eq 0 ]; then
      # partially produced barcode html might be useful but is left in an auto-update mode
      rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}" "$JSON_RESULTS"
    fi
    # still have to produce error here to prevent calling code from continuing
    exit 1
  fi
}

# produces a title with fly-over help showing users parameters
write_page_title ()
{
    local ALIGNEDREADS=""
    #if [ "$PLUGINCONFIG__MERGEDBAM_ID" != "Current Report" ]; then
    #    ALIGNEDREADS="Aligned Reads='${PLUGINCONFIG__MERGEDBAM_ID}'   "
    #fi
    local TARGETS=""
    if [ -n "$BC_MAPPED_BED" ]; then
      TARGETS=`echo "$BC_MAPPED_BED" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
      TARGETS="    Target Regions='${TARGETS}'"
    elif [ -n "$PLUGIN_BC_TARGETS" ];then
      TARGETS="    Target Regions=[Barcoded Targets]"
    elif [ -n "$PLUGIN_EFF_TARGETS" ]; then
      TARGETS="    Target Regions='${PLUGINCONFIG__TARGETREGIONS_ID}'"
    fi
    local OPTIONS=""
    if [ "$PLUGIN_SAMPLEID" = "Yes" ];then
        OPTIONS="$OPTIONS   SampleID Tracking"
    fi
    if [ "$INPUT_TRIM_READS" = "Yes" ]; then
        OPTIONS="$OPTIONS   Trim Reads"
    fi
    if [ "$PLUGIN_PADSIZE" -gt 0 ]; then
        OPTIONS="$OPTIONS   Target Padding: $PLUGIN_PADSIZE"
    fi
    if [ "$PLUGIN_TRIMREADS" = "Yes" ]; then
        OPTIONS="$OPTIONS   Trim Reads"
    fi
    if [ "$PLUGIN_UMAPS" = "Yes" ]; then
        OPTIONS="$OPTIONS   Uniquely Mapped"
    fi
    if [ "$PLUGIN_NONDUPS" = "Yes" ]; then
        OPTIONS="$OPTIONS   Non-duplicates"
    fi
    echo "<h1><center><span style=\"cursor:help\" title=\"${ALIGNEDREADS}Library Type='${PLUGINCONFIG__LIBRARYTYPE_ID}'${TARGETS}${OPTIONS}\">Coverage Analysis Report</span></center></h1>" >> "$1"
}

write_page_header ()
{
    local HEADFILE="$1"
    local HTML="${TSP_FILEPATH_PLUGIN_DIR}/header"
    if [ -n "$2" ]; then
        HTML="$2"
    fi
    cat "$HEADFILE" > "$HTML"
    if [ "$AMPOPT" != "-r" ]; then
      echo '<script language="javascript" type="text/javascript" src="lifechart/DepthOfCoverageChart.js"></script>' >> "$HTML"
      echo '<script language="javascript" type="text/javascript" src="lifechart/ReferenceCoverageChart.js"></script>' >> "$HTML"
    fi
    if [ -n "$PLUGIN_TARGETS" -o "$PLUGIN_BC_TARGETS" ]; then
      echo '<script language="javascript" type="text/javascript" src="lifechart/TargetCoverageChart.js"></script>' >> "$HTML"
    fi
    echo '</head>' >> "$HTML"
    echo '<body>' >> "$HTML"
    print_html_logo >> "$HTML";
    echo "<div class=\"center\" style=\"width:100%;height:100%\">" >> "$HTML"
    write_page_title "$HTML";
}

write_page_footer ()
{
    local HTML="${TSP_FILEPATH_PLUGIN_DIR}/footer"
    if [ -n "$1" ]; then
        HTML="$1"
    fi
    print_html_footer >> "$HTML"
    echo '<br/><br/></div>' >> "$HTML"
    echo '</body></html>' >> "$HTML"
}

# old header/footer code - still used for barcodes table
write_html_header ()
{
    local HTML="${TSP_FILEPATH_PLUGIN_DIR}/header"
    if [ -n "$1" ]; then
        HTML="$1"
    fi
    local REFRESHRATE=0
    if [ -n "$2" ]; then
        if [ $2 -gt 0 ]; then
	    REFRESHRATE=$2
        fi
    fi
    echo '<?xml version="1.0" encoding="iso-8859-1"?>' > "$HTML"
    echo '<!DOCTYPE html>' >> "$HTML"
    echo '<html>' >> "$HTML"
    print_html_head $REFRESHRATE >> "$HTML"
    if [ "$HTML_TORRENT_WRAPPER" -eq 1 ]; then
      echo '<title>Torrent Coverage Analysis Report</title>' >> "$HTML"
      echo '<body>' >> "$HTML"
      print_html_logo >> "$HTML";
    else
      echo '<body>' >> "$HTML"
    fi

    if [ -z "$COV_PAGE_WIDTH" ];then
	echo '<div id="inner">' >> "$HTML"
    else
	echo "<div style=\"width:${COV_PAGE_WIDTH};margin-left:auto;margin-right:auto;height:100%\">" >> "$HTML"
    fi
    if [ "$HTML_TORRENT_WRAPPER" -eq 1 ]; then
      write_page_title "$HTML";
    fi
}

write_html_footer ()
{
    local HTML="${TSP_FILEPATH_PLUGIN_DIR}/footer"
    if [ -n "$1" ]; then
        HTML="$1"
    fi
    print_html_end_javascript >> "$HTML"
    if [ "$HTML_TORRENT_WRAPPER" -eq 1 ]; then
      print_html_footer >> "$HTML"
      echo '<br/><br/>' >> "$HTML"
    fi
    echo '</div></body></html>' >> "$HTML"
}

display_static_progress ()
{
    local HTML="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
    if [ -n "$1" ]; then
        HTML="$1"
    fi
    echo "<br/><h3 style=\"text-align:center;color:red;margin:0\">*** Analysis is not complete ***</h3><br/>" >> "$HTML"
    echo "<a href=\"javascript:document.location.reload();\" ONMOUSEOVER=\"window.status='Refresh'; return true\">" >> "$HTML"
    echo "<div style=\"text-align:center\" title=\"Reload page\">Click here to refresh</div></a><br/>" >> "$HTML"
}

write_json_header ()
{
    local haveBC="false"
    if [ -n "$1" ]; then
	if [ $1 -eq 0 ]; then
	    haveBC="false"
	else
	    haveBC="true"
	fi
    fi
    echo "{" > "$JSON_RESULTS"
    echo "  \"Targetted regions\" : \"$PLUGIN_TARGETS\"," >> "$JSON_RESULTS"
    echo "  \"Target padding\" : \"$PLUGIN_PADSIZE\"," >> "$JSON_RESULTS"
    echo "  \"Uniquely mapped\" : \"$PLUGIN_UMAPS\"," >> "$JSON_RESULTS"
    echo "  \"Non-duplicate\" : \"$PLUGIN_NONDUPS\"," >> "$JSON_RESULTS"
    echo "  \"barcoded\" : \"$haveBC\"," >> "$JSON_RESULTS"
    if [ "$haveBC" = "true" ]; then
        echo "  \"barcodes\" : {" >> "$JSON_RESULTS"
    fi
}

write_json_footer ()
{
    if [ -n "$1" -a "$1" -ne 0 ]; then
        echo -e "\n  }" >> "$JSON_RESULTS"
    else
        echo "" >> "$JSON_RESULTS"
    fi
    echo "}" >> "$JSON_RESULTS"
}

write_json_inner ()
{
    local DATADIR=$1
    local DATASET=$2
    local BLOCKID=$3
    local INDENTLEV=$4
    if [ -f "${DATADIR}/${DATASET}" ];then
        append_to_json_results "$JSON_RESULTS" "${DATADIR}/${DATASET}" "$BLOCKID" $INDENTLEV;
    fi
}

append_to_json_results ()
{
  local JSONFILE="$1"
  local DATAFILE="$2"
  local DATASET="$3"
  local INDENTLEV="$4"
  if [ -z $INDENTLEV ]; then
    INDENTLEV=2
  fi
  if [ -n $DATASET ]; then
    DATASET="-B \"$DATASET\""
  fi
  local JSONCMD="perl ${SCRIPTSDIR}/coverage_analysis_json.pl -a -I $INDENTLEV $DATASET \"$DATAFILE\" \"$JSONFILE\""
  eval "$JSONCMD || echo \"WARNING: Failed to write to JSON from $DATAFILE\"" >&2
}

# Create padded targets file and return filename as $CREATE_PADDED_TARGETS
create_padded_targets ()
{
  local BEDFILE=$1
  local PADDING=$2
  local OUTDIR=$3
  local BEDROOT
  local GENOME
  local PADCMD
  CREATE_PADDED_TARGETS=$BEDFILE
  if [ -n "$BEDFILE" -a -n "$PADDING" -a $PADDING -gt 0 ];then
    echo "(`date`) Creating padded targets file..." >&2
    GENOME="${REFERENCE}.fai"
    if ! [ -f "$GENOME" ]; then
      echo "WARNING: Could not create padded targets file; genome (.fai) file does not exist at $GENOME" >&2
      echo "- Continuing without padded targets analysis." >&2
    else
      BEDROOT=`echo "$BEDFILE" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
      CREATE_PADDED_TARGETS="${OUTDIR}/${BEDROOT}_$PLUGIN_PADSIZE.bed"
      PADCMD="${DIRNAME}/padbed/padbed.sh $LOGOPT \"$BEDFILE\" \"$GENOME\" $PADDING \"$CREATE_PADDED_TARGETS\""
      eval "$PADCMD" >&2
      if [ $? -ne 0 ]; then
        echo "WARNING: Could not create padded targets file; padbed.sh failed." >&2
        echo "\$ $REMDUP" >&2
        echo "- Continuing without padded targets analysis." >&2
        CREATE_PADDED_TARGETS=$BEDFILE
      elif [ $PLUGIN_DEV_FULL_LOG -gt 0 ]; then
        echo "> $CREATE_PADDED_TARGETS" >&2
      fi
    fi
  fi
}

# Create GC annotated BED file and return to filename $GC_ANNOTATE_BED
gc_annotate_bed ()
{
  local BEDFILE=$1
  local OUTDIR=$2
  local GCANNOCMD
  GC_ANNOTATE_BED="$BEDFILE"
  if [ -n "$BEDFILE" ]; then
    echo "(`date`) Adding GC count information to annotated targets file..." >&2
    GC_ANNOTATE_BED="${OUTDIR}/tca_auxiliary.gc.bed"
    GCANNOCMD="${SCRIPTSDIR}/gcAnnoBed.pl -s -w -t \"$OUTDIR\" $PLUGIN_ANNOFIELDS \"$BEDFILE\" \"$REFERENCE\" > \"$GC_ANNOTATE_BED\""
    eval "$GCANNOCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: gcAnnoBed.pl failed. GC Annotation will be missing from subsequent analyses." >&2
      echo "\$ $GCANNOCMD" >&2
      GC_ANNOTATE_BED="$BEDFILE"
      # TO DO: add a default field for GC to prevent downstream failure...
    elif [ $PLUGIN_DEV_FULL_LOG -gt 0 ]; then
      echo "> $GC_ANNOTATE_BED" >&2
    fi
  fi
}

# Link specific file names with more non-run-speicific names (in a diferent folder).
create_scraper_links ()
{
  local NAMEROOT=${1}
  local LINKNAME=${2}
  local OUTDIR=${3}
  local SAFEROOT=`echo "$NAMEROOT" | sed -e 's/^.*\///' | sed -e 's/[][\.*^$(){}?+|/]/\\\&/g'`
  for FNAME in `eval "ls -1 ${NAMEROOT}.* 2> /dev/null"`
  do
    LNAME=`echo "$FNAME" | sed -e 's/^.*\///' | sed -e "s/$SAFEROOT\./$LINKNAME\./"`
    FNAME=`readlink -f $FNAME`
    ln -sf "$FNAME" "${OUTDIR}/${LNAME}"
  done
}

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

  # Create scraper directory containings links to all 'visible' output files
  echo "(`date`) Creating scraper folder..." >&2
  run "mkdir ${OUTDIR}/scraper"
  create_scraper_links "${OUTDIR}/$BAMROOT" "link" "${OUTDIR}/scraper"
}

