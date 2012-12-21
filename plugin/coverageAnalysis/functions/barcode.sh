#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

barcode_load_list ()
{
  local ROWSUM_NODATA=""
  local NTAB
  for((NTAB=0;NTAB<${BC_SUM_ROWS};NTAB++)); do
    ROWSUM_NODATA="${ROWSUM_NODATA}<td>N/A</td> "
  done
  
  local BCN=0
  local BARCODE_BAM
  local BARCODE_LINE

  local FILTERCMD="grep ^barcode \"${BARCODES_LIST}\" | cut -d, -f2";
  # check for backwards compatibility - currently relying on 2.2 barcodesList.txt annd presence of BAMs for filtering
  #local FLINE=`head -1 "${BARCODES_LIST}" | cut -d, -f1`
  #if [ "$FLINE" = "BarcodeId" ]; then
  #  FILTERCMD="tail -n +2 \"${BARCODES_LIST}\" | cut -d, -f1";
  #fi
  for BARCODE_LINE in `eval "$FILTERCMD"`
  do
    BARCODES[$BCN]=${BARCODE_LINE}
    BARCODE_ROWSUM[$BCN]=$ROWSUM_NODATA
    BARCODE_BAM="${ANALYSIS_DIR}/${BARCODE_LINE}_${PLUGIN_BAM_FILE}"
    if [ -e ${BARCODE_BAM} ]; then
      BARCODES_OK[${BCN}]=1
      BARCODE_BAMNAME[${BCN}]=$PLUGIN_BAM_FILE
    else
      BARCODE_BAM="${ANALYSIS_DIR}/${BARCODE_LINE}_${PLUGIN_RUN_NAME}.bam"
      if [ -e ${BARCODE_BAM} ]; then
        BARCODES_OK[${BCN}]=1
        BARCODE_BAMNAME[${BCN}]="${PLUGIN_RUN_NAME}.bam"
      else
        BARCODES_OK[${BCN}]=0
        BARCODE_BAMNAME[${BCN}]=""
      fi
    fi
    BCN=`expr ${BCN} + 1`
  done
}

barcode_partial_table ()
{
  local HTML="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
  if [ -n "$1" ]; then
    HTML="$1"
  fi
  local NLINES=0
  if [ -n "$2" ]; then
    NLINES="$2"
  fi
  local REFRESHRATE=15
  if [ "$NLINES" = "$NBARCODES" ]; then
    REFRESHRATE=0
  fi
  write_html_header "$HTML" $REFRESHRATE
  if [ "$HTML_TORRENT_WRAPPER" -eq 1 ]; then
    echo "<h3><center>$PLUGIN_RUN_NAME</center></h3>" >> "$HTML"
  fi
  barcode_links "$HTML" $NLINES
  # insert any extra text as raw html below table
  if [ -n "$3" ]; then
    echo -e "$3" >> "$HTML"
  fi
  write_html_footer "$HTML"
}

barcode_links ()
{
  local HTML="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
  if [ -n "1" ]; then
    HTML="$1"
  fi
  local NLINES=-1;  # -1 => all, 0 => none
  if [ -n "$2" ]; then
    NLINES="$2"
  fi
  # html has compromises so as to appear almost identical on Firefox vs. IE8
  echo " <div id=\"BarcodeList\" class=\"report_block\"/>" >> "$HTML"
  echo "  <style type=\"text/css\">" >> "$HTML"
  echo "   th {text-align:center;width=100%}" >> "$HTML"
  echo "   td {text-align:center;width=100%}" >> "$HTML"
  echo "   .help {cursor:help; border-bottom: 1px dotted #A9A9A9}" >> "$HTML"
  echo "   .report_block > h2 {margin:0;padding:5px}" >> "$HTML"
  echo "   .report_block {margin:0px 0px 1px 0px;padding:0px}" >> "$HTML"
  echo "  </style>" >> "$HTML"
  echo "  <h2><span class=\"help\" title=\"${BC_TITLE_INFO}\">Barcode Summary Report</span></h2>" >> "$HTML"
  echo "  <div>" >> "$HTML"
  echo "   <table class=\"noheading\" style=\"table-layout:fixed\">" >> "$HTML"
  echo "   <tr>" >> "$HTML"
  echo "  <th><span class=\"help\" style=\"width:110px !important\" title=\"Name of the barcode sequence and link to detailed report for reads associated with that barcode.\">Barcode ID</span></th>" >> "$HTML"
  local BCN
  local CWIDTH=100
  for((BCN=0;BCN<${BC_SUM_ROWS};BCN++))
  do
    echo "  <th style=\"width:${CWIDTH}px !important\"><span class=\"help\" title=\"${BC_COL_HELP[$BCN]}\">${BC_COL_TITLE[$BCN]}</span></th>" >> "$HTML"
  done
  echo "   </tr>" >> "$HTML"

  local BARCODE
  local UNFIN=0
  for((BCN=0;BCN<${#BARCODES[@]};BCN++))
  do
    if [ $NLINES -ge 0 -a $BCN -ge $NLINES ]; then
      UNFIN=1
      break
    fi
    BARCODE=${BARCODES[$BCN]}
    if [ ${BARCODES_OK[$BCN]} -eq 1 ]; then
      echo "<tr><td style=\"text-align:left\"><a style=\"cursor:help\" href=\"${BARCODE}/${HTML_RESULTS}\"><span title=\"Click to view the detailed coverage report for barcode ${BARCODE}\">${BARCODE}</span></a></td>" >> "$HTML"
      echo "${BARCODE_ROWSUM[$BCN]}</tr>" >> "$HTML"
    elif [ ${BARCODES_OK[$BCN]} -eq 2 ]; then
      echo "<tr><td style=\"text-align:left\"><span class=\"help\" title=\"An error occurred while processing data for barcode ${BARCODE}\" style=\"color:red\">${BARCODE}</span></td>" >> "$HTML"
      echo "${BARCODE_ROWSUM[$BCN]}</tr>" >> "$HTML"
    fi
  done
  echo "  </table></div>" >> "$HTML"
  echo " </div>" >> "$HTML"
  if [ $UNFIN -eq 1 ]; then
    display_static_progress "$HTML"
  fi
}

barcode_create_summary_matrix ()
{
  local PROPINDEX=$1
  if [ -z "$PROPINDEX" ]; then
    PROPINDEX=9
  fi
  local OUTFILE=$2
  if [ -z "$OUTFILE" ]; then
    OUTFILE="${PLUGIN_RUN_NAME}.bcmatrix.xls"
  fi
  local BARCODE
  local BARCODE_DIR
  local FILEEXT
  # use globbing to find files needed for each barcode
  OLDOPT=`shopt -p nullglob | awk '{print $2}'`
  shopt -s nullglob
  if [ "$AMPOPT" = "-a" ]; then
    FILEEXT="*.amplicon.cov.xls"
  else
    FILEEXT="*.target.cov.xls"
  fi
  local BCN
  local FILELIST=''
  for((BCN=0;BCN<${#BARCODES[@]};BCN++))
  do
    if [ ${BARCODES_OK[$BCN]} -eq 1 ]; then
      BARCODE=${BARCODES[$BCN]}
      # should match only one file
      FILES="${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE}/${BARCODE}_$FILEEXT"
      for covfile in $FILES
      do
        FILELIST="$FILELIST $covfile"
      done
    fi
  done
  shopt $OLDOPT nullglob
  # build the barcode matrix for reads/base coverage
  if [ -n "$FILELIST" ]; then
    echo "" >&2
    run "${SCRIPTSDIR}/barcodeMatrix.pl \"${REFERENCE}.fai\" $PROPINDEX $FILELIST > \"$OUTFILE\""
  fi
}

barcode_append_to_json_results ()
{
  local DATADIR=$1
  local BARCODE=$2
  if [ -n "$3" ]; then
    if [ "$3" -gt 1 ]; then
      echo "," >> "$JSON_RESULTS"
    fi
  fi
  echo "  \"$BARCODE\" : {" >> "$JSON_RESULTS"
  write_json_inner "${DATADIR}" "summary.txt" "" 4;
  echo "" >> "$JSON_RESULTS"
  echo -n "  }" >> "$JSON_RESULTS"
}

barcode ()
{
  local HTMLOUT="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"

  # Yes, there are barcodes
  echo "There are barcodes!" >&2
  
  local LOGOPT="> /dev/null"
  if [ "$PLUGIN_DEV_FULL_LOG" -eq 1 ]; then
    echo "" >&2
    LOGOPT=""
  fi
  # Load bar code data
  local BC_ERROR=0
  local BARCODES
  local BARCODES_OK
  local BARCODE_ROWSUM
  barcode_load_list;

  # Start json file
  write_json_header 1;

  # Empty Table - BARCODE set because header file expects this load lavascript
  local BARCODE="TOCOME"

  barcode_partial_table "$HTMLOUT";
  NBARCODES=${#BARCODES[@]}
  
  # Go through the barcodes 
  local BARCODE_DIR
  local BARCODE_BAM
  local NLINE
  local BCN
  local NUSED=0
  for((BCN=0;BCN<${NBARCODES};BCN++))
  do
    BARCODE=${BARCODES[$BCN]}
    BARCODE_DIR="${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE}"
    BARCODE_URL="."
    BARCODE_BAM="${BARCODE}_${BARCODE_BAMNAME[${BCN}]}"
    NLINE=`expr ${BCN} + 1`

    # ensure old data is not retained
    run "rm -rf ${BARCODE_DIR}"
    if [ ${BARCODES_OK[$BCN]} -eq 0 ]; then
      echo -e "\nSkipping ${BARCODE}" >&2
    else
      echo -e "\nProcessing barcode ${BARCODE}" >&2
      run "mkdir -p ${BARCODE_DIR}"
      run "ln -s ${GCANNOBED} ${BARCODE_DIR}/"
      # need to create link early so the correct name gets used if a PTRIM file is created
      BARCODE_LINK_BAM="${BARCODE_DIR}/${BARCODE}_${PLUGIN_RUN_NAME}.bam"
      run "ln -sf \"${ANALYSIS_DIR}/${BARCODE_BAM}\" \"${BARCODE_LINK_BAM}\""
      run "ln -sf \"${ANALYSIS_DIR}/${BARCODE_BAM}.bai\" \"${BARCODE_LINK_BAM}.bai\""
      local RT=0
      eval "${SCRIPTSDIR}/run_coverage_analysis.sh $LOGOPT $FILTOPTS $AMPOPT $TRIMOPT -R \"$HTML_RESULTS\" -T \"$HTML_ROWSUMS\" -D \"$BARCODE_DIR\" -A \"$GCANNOBED\" -B \"$PLUGIN_EFF_TARGETS\" -C \"$PLUGIN_TRGSID\" -p $PLUGIN_PADSIZE -P \"$PADDED_TARGETS\" \"$REFERENCE\" \"$BARCODE_LINK_BAM\"" || RT=$?
      if [ $RT -ne 0 ]; then
        BC_ERROR=1
        if [ "$CONTINUE_AFTER_BARCODE_ERROR" -eq 0 ]; then
          exit 1
        else
          BARCODES_OK[${BCN}]=2
        fi
      else
        # process all result files to detailed html page and clean up
        write_html_results "${BARCODE}_${PLUGIN_RUN_NAME}" "$BARCODE_DIR" "$BARCODE_URL" "${BARCODE}_${PLUGIN_RUN_NAME}.bam"
        # collect table summary results
        if [ -f "${BARCODE_DIR}/${HTML_ROWSUMS}" ]; then
          BARCODE_ROWSUM[$BCN]=`cat "${BARCODE_DIR}/$HTML_ROWSUMS"`
          rm -f "${BARCODE_DIR}/${HTML_ROWSUMS}"
        fi
        NUSED=`expr ${NUSED} + 1`
        barcode_append_to_json_results "$BARCODE_DIR" $BARCODE $NUSED;
      fi
    fi
    barcode_partial_table "$HTMLOUT" $NLINE
  done
  # create barcode * amplicon matrix or targeted coverage
  if [ -n "$PLUGIN_EFF_TARGETS" ]; then
    BCMATRIX="${PLUGIN_RUN_NAME}.bcmatrix.xls"
    barcode_create_summary_matrix 9 "$BCMATRIX"
    TITLESTR="target"
    if [ "$AMPOPT" = "-a" ]; then
      TITLESTR="amplicon"
    fi
    INSERT_HTML="\n<br/><a href='$BCMATRIX' title='Click to download a table file of coverage for individual ${TITLESTR}s for each barcode.'>Download barcode/$TITLESTR coverage matrix</a>\n"
    # have to redraw page to get link in right place
    barcode_partial_table "$HTMLOUT" $NLINE "$INSERT_HTML";
  fi
  # write raw table as block_html (for 3.0 summary)
  COV_PAGE_WIDTH="auto"
  HTML_TORRENT_WRAPPER=0
  barcode_partial_table "$HTML_BLOCK" $NLINE;
  # finish up with json
  write_json_footer 1;
  if [ "$BC_ERROR" -ne 0 ]; then
    exit 1
  fi
}
