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
    SAMPNAME=`echo "$PLUGIN_SAMPLE_NAMES" | sed -e "s/.*;$BARCODE_LINE=\([^;]*\).*/\1/"`
    if [ -z "$SAMPNAME" -o "$SAMPNAME" = "$PLUGIN_SAMPLE_NAMES" ];then
      SAMPNAME='None'
    fi
    if [ -e ${BARCODE_BAM} ]; then
      BARCODES_OK[${BCN}]=1
      BARCODE_BAMNAME[${BCN}]=$PLUGIN_BAM_FILE
      BARCODE_SAMPNAME[${BCN}]="$SAMPNAME"
    else
      BARCODE_BAM="${ANALYSIS_DIR}/${BARCODE_LINE}_${PLUGIN_RUN_NAME}.bam"
      if [ -e ${BARCODE_BAM} ]; then
        BARCODES_OK[${BCN}]=1
        BARCODE_BAMNAME[${BCN}]="${PLUGIN_RUN_NAME}.bam"
        BARCODE_SAMPNAME[${BCN}]="$SAMPNAME"
      else
        BARCODES_OK[${BCN}]=0
        BARCODE_BAMNAME[${BCN}]=""
        BARCODE_SAMPNAME[${BCN}]=""
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
  #if [ "$HTML_TORRENT_WRAPPER" -eq 1 ]; then
    #echo "<h3><center>$PLUGIN_RUN_NAME</center></h3>" >> "$HTML"
  #fi
  barcode_links "$HTML" $NLINES
  # insert any extra text as raw html below table
  if [ -n "$3" ]; then
    echo -e "$3" >> "$HTML"
  fi
  write_html_footer "$HTML"
}

barcode_links ()
{
  # Define BC_COL array data.
  declare -A BC_COL_TITLE
  declare -A BC_COL_HELP
  BC_COL_TITLE[0]=$BCT0
  BC_COL_HELP[0]=$BCH0
  BC_COL_TITLE[1]=$BCT1
  BC_COL_HELP[1]=$BCH1
  if [ $BC_SUM_ROWS -gt 2 ]; then
    BC_COL_TITLE[2]=$BCT2
    BC_COL_HELP[2]=$BCH2
  fi
  if [ $BC_SUM_ROWS -gt 3 ]; then
    BC_COL_TITLE[3]=$BCT3
    BC_COL_HELP[3]=$BCH3
  fi
  if [ $BC_SUM_ROWS -gt 4 ]; then
    BC_COL_TITLE[4]=$BCT4
    BC_COL_HELP[4]=$BCH4
  fi
  
  local HTML="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
  if [ -n "$1" ]; then
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
  echo "  <th><span class=\"help\" style=\"width:100px !important\" title=\"Name of the barcode sequence and link to detailed report for reads associated with that barcode.\">Barcode ID</span></th>" >> "$HTML"
  echo "  <th><span class=\"help\" style=\"width:100px !important\" title=\"Sample Name associated with this barcode in the experiment plan.\">Sample Name</span></th>" >> "$HTML"
  local BCN
  local CWIDTH
  local CTITLE
  BARCODE_TITLES=''
  for((BCN=0;BCN<${BC_SUM_ROWS};BCN++))
  do
    CTITLE=${BC_COL_TITLE[$BCN]}
    if [ "$CTITLE" = "Mapped Reads" ];then
      CWIDTH=100
    elif [ "$CTITLE" = "Mean Depth" ];then
      CWIDTH=80
    else
      CWIDTH=70
    fi
    echo "  <th style=\"width:${CWIDTH}px !important\"><span class=\"help\" title=\"${BC_COL_HELP[$BCN]}\">${CTITLE}</span></th>" >> "$HTML"
    if [ $BCN -eq 0 ];then
      BARCODE_TITLES=$CTITLE
      continue
    fi
    BARCODE_TITLES="$BARCODE_TITLES\t$CTITLE"
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
      echo "<td style=\"text-align:left\">${BARCODE_SAMPNAME[$BCN]}</td>" >> "$HTML"
      echo "${BARCODE_ROWSUM[$BCN]}</tr>" >> "$HTML"
    elif [ ${BARCODES_OK[$BCN]} -eq 2 ]; then
      echo "<tr><td style=\"text-align:left\"><span class=\"help\" title=\"An error occurred while processing data for barcode ${BARCODE}\" style=\"color:red\">${BARCODE}</span></td>" >> "$HTML"
      echo "<td style=\"text-align:left\">${BARCODE_SAMPNAME[$BCN]}</td>" >> "$HTML"
      echo "${BARCODE_ROWSUM[$BCN]}</tr>" >> "$HTML"
    fi
  done
  echo "  </table></div>" >> "$HTML"
  echo " </div>" >> "$HTML"
  if [ $UNFIN -eq 1 ]; then
    display_static_progress "$HTML"
  fi
}

barcode_summary_to_file ()
{
  local FILEOUT=$1
  if [ -z "$FILEOUT" ]; then
    FILEOUT="barcode_summary.xls"
  fi
  FILEOUT="${TSP_FILEPATH_PLUGIN_DIR}/$FILEOUT"
  echo -e "Barcode ID\tSample Name\t$BARCODE_TITLES" > "$FILEOUT"
  local BCN
  for((BCN=0;BCN<${#BARCODES[@]};BCN++))
  do
    BARCODE=${BARCODES[$BCN]}
    if [ ${BARCODES_OK[$BCN]} -eq 1 ]; then
      SAMPNAME=${BARCODE_SAMPNAME[$BCN]}
      ROWSUM=`echo "${BARCODE_ROWSUM[$BCN]}" | sed -e 's/[ \t]*<td>//g' | sed -e 's/<\/td>/\t/g' | sed -e 's/[ \t]*$//'`
      echo -e "$BARCODE\t$SAMPNAME\t$ROWSUM" >> "$FILEOUT"
    fi
  done
}

barcode_create_summary_matrix ()
{
  local FILEEXT=$1
  if [ -z "$FILEEXT" ]; then
    FILEEXT="amplicon.cov.xls"
  fi
  FILEEXT="*.$FILEEXT"
  local PROPID=$2
  if [ -z "$PROPID" ]; then
    PROPID=9
  fi
  local OUTFILE=$3
  if [ -z "$OUTFILE" ]; then
    OUTFILE="${PLUGIN_RUN_NAME}.bcmatrix.xls"
  fi
  # use globbing to find files needed for each barcode
  OLDOPT=`shopt -p nullglob | awk '{print $2}'`
  shopt -s nullglob
  local BCN
  local BARCODE
  local FILES
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
    run "${SCRIPTSDIR}/barcodeMatrix.pl \"${REFERENCE}.fai\" $PROPID $FILELIST > \"$OUTFILE\""
  fi
}

barcode_append_to_json_results ()
{
  local DATADIR=$1
  local BARCODE=$2
  local SUMFILE=$3
  if [ -n "$4" ]; then
    if [ "$4" -gt 1 ]; then
      echo "," >> "$JSON_RESULTS"
    fi
  fi
  echo "    \"$BARCODE\" : {" >> "$JSON_RESULTS"
  write_json_inner "${DATADIR}" "$SUMFILE" "" 6;
  echo "" >> "$JSON_RESULTS"
  echo -n "    }" >> "$JSON_RESULTS"
}

barcode ()
{
  local HTMLOUT="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
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
  local BARCODE_TITLES
  local BARCODE_BAMNAME
  local BARCODE_SAMPNAME
  barcode_load_list;

  # Start json file
  write_json_header 1;

  # Empty Table - BARCODE set because header file expects this load javascript
  local BARCODE="TOCOME"

  barcode_partial_table "$HTMLOUT";
  NBARCODES=${#BARCODES[@]}
  
  # Convert python-generated list to bash list.
  # Start by making a list of the keys...
  barCount=1
  declare -A BARCODE_TARGET_MAP_contig
  for barVal in $T_BARCODE_TARGET_MAP_VALS
  do
    BARCODE_TARGET_MAP_contig[$barCount]=$barVal
    barCount=$[barCount+1]
  done
  # ...Then put the values in accordingly.
  barCount=1
  declare -A BARCODE_TARGET_MAP
  for barVal in $T_BARCODE_TARGET_MAP_KEYS
  do
    #BARCODE_TARGET_MAP[${BARCODE_TARGET_MAP_contig[$barCount]}]=$barVal
    BARCODE_TARGET_MAP[$barVal]=${BARCODE_TARGET_MAP_contig[$barCount]}
    barCount=$[barCount+1]
  done
  
  # Go through the barcodes 
  local BARCODE_DIR
  local BARCODE_BAM
  local NLINE
  local BCN
  local BC_GCANNOBED
  local BC_MERGEBED
  local BC_PADBED
  local BC_BED
  local BC_TRGSID
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
      echo -e "\nSkipping ${BARCODE}:- No BAM file found in analysis directory." >&2
    else
      echo -e "\nProcessing barcode ${BARCODE}..." >&2
      run "mkdir -p ${BARCODE_DIR}"
      # code block to handle barcode to bed mapping
      BC_BED="$PLUGIN_DETAIL_TARGETS"
      BC_GCANNOBED="$GCANNOBED"
      BC_MERGEBED="$PLUGIN_EFF_TARGETS"
      BC_PADBED="$PADDED_TARGETS"
      BC_TRGSID="$PLUGIN_TRGSID"
      if [ -n "$PLUGIN_BC_TARGETS" ];then
        BC_MAPPED_BED=${BARCODE_TARGET_MAP[$BARCODE]}
        if [ -z "$BC_MAPPED_BED" ];then
          if [ -z "$PLUGIN_EFF_TARGETS" ];then
            echo  "- Skipping:- No assigned or default barcode." >&2
            BARCODES_OK[${BCN}]=0
            run "rm -rf ${BARCODE_DIR}"
            continue
          fi
          echo "Employing default targets: $BC_TRGSID" >&2
          BC_MAPPED_BED="$PLUGINCONFIG__TARGETREGIONS_ID"
          if [ -n "$BC_GCANNOBED" ]; then
            run "ln -s ${BC_GCANNOBED} ${BARCODE_DIR}/"
          fi
        else
          BC_BED="$BC_MAPPED_BED"
          BC_TRGSID=`echo "$BC_MAPPED_BED" | sed -e 's/^.*\///' | sed -e 's/\.[^.]*$//'`
          echo "Employing barcoded targets: $BC_TRGSID" >&2
          # get the correct detail/plain merged/unmerged versions
          BC_GCANNOBED=$BC_MAPPED_BED
          if [ -z "$AMPOPT" ]; then
            BC_GCANNOBED=`echo "$BC_GCANNOBED" | sed -e 's/\/unmerged\//\/merged\//'`
          fi
          BC_MAPPED_BED=`echo "$BC_MAPPED_BED" | sed -e 's/\/unmerged\/detail\//\/merged\/plain\//'`
          # perform padding and/or GC annotation
          create_padded_targets "$BC_MAPPED_BED" $PLUGIN_PADSIZE "$BARCODE_DIR"
          BC_MERGEBED=$CREATE_PADDED_TARGETS
          gc_annotate_bed "$BC_GCANNOBED" "$BARCODE_DIR"
          BC_GCANNOBED=$GC_ANNOTATE_BED
          # ensure specific padded bed disabled here (retained for future dev.)
          BC_PADBED=""
        fi
      elif [ -n "$BC_GCANNOBED" ]; then
        run "ln -s ${BC_GCANNOBED} ${BARCODE_DIR}/"
      fi
      # need to create link early so the correct name gets used if a PTRIM file is created
      SAMPLENAME="${BARCODE_SAMPNAME[$BCN]}"
      BARCODE_LINK_BAM="${BARCODE_DIR}/${BARCODE}_${PLUGIN_RUN_NAME}.bam"
      run "ln -sf \"${ANALYSIS_DIR}/${BARCODE_BAM}\" \"${BARCODE_LINK_BAM}\""
      run "ln -sf \"${ANALYSIS_DIR}/${BARCODE_BAM}.bai\" \"${BARCODE_LINK_BAM}.bai\""
      local RT=0
      eval "${SCRIPTSDIR}/run_coverage_analysis.sh $LOGOPT $FILTOPTS $AMPOPT $TRIMOPT -N \"$SAMPLENAME\" -R \"$HTML_RESULTS\" -T \"$HTML_ROWSUMS\" -D \"$BARCODE_DIR\" -A \"$BC_GCANNOBED\" -B \"$BC_MERGEBED\" -C \"$BC_TRGSID\" -p $PLUGIN_PADSIZE -P \"$BC_PADBED\" -S \"$PLUGIN_SAMPLEID_REGIONS\" \"$REFERENCE\" \"$BARCODE_LINK_BAM\"" || RT=$?
      if [ $RT -ne 0 ]; then
        BC_ERROR=1
        if [ "$CONTINUE_AFTER_BARCODE_ERROR" -eq 0 ]; then
          exit 1
        else
          BARCODES_OK[${BCN}]=2
        fi
      else
        # process all result files to detailed html page and clean up
        write_html_results "${BARCODE}_${PLUGIN_RUN_NAME}" "$BARCODE_DIR" "$BARCODE_URL" "${BARCODE}_${PLUGIN_RUN_NAME}.bam" "$SAMPLENAME" "$BC_BED"
        # collect table summary results
        if [ -f "${BARCODE_DIR}/${HTML_ROWSUMS}" ]; then
          BARCODE_ROWSUM[$BCN]=`cat "${BARCODE_DIR}/$HTML_ROWSUMS"`
          rm -f "${BARCODE_DIR}/${HTML_ROWSUMS}"
        fi
        NUSED=`expr ${NUSED} + 1`
        if [ "$TRIMOPT" = "-t" ]; then
          STATSFILE="${BARCODE}_${PLUGIN_RUN_NAME}.trim.stats.cov.txt"
        else
          STATSFILE="${BARCODE}_${PLUGIN_RUN_NAME}.stats.cov.txt"
        fi
        barcode_append_to_json_results "$BARCODE_DIR" $BARCODE "$STATSFILE" $NUSED;
      fi
    fi
    BC_MAPPED_BED="";  # ensure title in barcode summary page is not for most recent run
    barcode_partial_table "$HTMLOUT" $NLINE
  done

  # create barcode * (amplicon | contig)matrix or targeted coverage
  BCMATRIX="${PLUGIN_RUN_NAME}.bcmatrix.xls"
  FIELDID="chrom"
  FILEEXT="chr.cov.xls";
  TITLESTR="contig"
  if [ $PLUGIN_USE_TARGETS -gt 0 ]; then
    FIELDID=9
    if [ "$AMPOPT" = "-a" -o "$AMPOPT" = "-r" ]; then
      FILEEXT="amplicon.cov.xls"
      TITLESTR="amplicon"
    else
      FILEEXT="target.cov.xls"
      TITLESTR="target"
    fi
  fi
  echo "" >&2
  echo "(`date`) Creating barcode/${TITLESTR}s coverage matrix..." >&2
  BCSUMMARY="${PLUGIN_RUN_NAME}.barcode_summary.xls"
  barcode_summary_to_file "$BCSUMMARY"
  INSERT_HTML="\n<br/><a href='$BCSUMMARY' title='Click to download a table file of the Barcode Summary Report presented above.'>Download Barcode Summary Report</a>\n"
  barcode_create_summary_matrix $FILEEXT $FIELDID "$BCMATRIX"
  INSERT_HTML="$INSERT_HTML\n<br/><a href='$BCMATRIX' title='Click to download a table file of coverage for individual ${TITLESTR}s for each barcode.'>Download barcode/$TITLESTR coverage matrix</a>\n"
  # redraw page to get new link in right place
  barcode_partial_table "$HTMLOUT" $NLINE "$INSERT_HTML";

  # write raw table as block_html (for 3.0 summary)
  COV_PAGE_WIDTH="auto"
  HTML_TORRENT_WRAPPER=0
  barcode_partial_table "$HTML_BLOCK" $NLINE;

  # finish up with json
  write_json_footer 1;

  # create the scraper folder for across barcodes analysis results - assumes PLUGIN_RUN_NAME is the output root
  local SCRAPERDIR="${TSP_FILEPATH_PLUGIN_DIR}/scraper"
  run "rm -rf ${SCRAPERDIR}"
  run "mkdir ${SCRAPERDIR}"
  create_scraper_links "${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_RUN_NAME" "link" "${SCRAPERDIR}"
  if [ "$BC_ERROR" -ne 0 ]; then
    exit 1
  fi
}

