#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

# Legacy html results writing support for barcode runs.
write_html_results ()
{
    local RUNID=${1}
    local OUTDIR=${2}
    local OUTURL=${3}
    local BAMFILE=${4}
    local SAMPLENAME=${5}

    # Create softlink to js/css folders and php scripts
    run "ln -sf \"${DIRNAME}/slickgrid\" \"${OUTDIR}/\"";
    run "ln -sf \"${DIRNAME}/lifegrid\" \"${OUTDIR}/\"";
    run "ln -sf ${DIRNAME}/scripts/*.php3 \"${OUTDIR}/\"";

    # Link bam/bed files from plugin dir and create local URL names for fileLinks table
    PLUGIN_OUT_BAMFILE=`echo "$BAMFILE" | sed -e 's/^.*\///'`
    PLUGIN_OUT_BAIFILE="${PLUGIN_OUT_BAMFILE}.bai"
    if [ -n "$BAMFILE" ]; then
	# create hard links if this was a combineAlignment file - shouldn't be barcodes!
	if [ -n "$PLUGINCONFIG__MERGEDBAM" ]; then
	    run "ln -f ${BAMFILE} ${OUTDIR}/$PLUGIN_OUT_BAMFILE"
	    run "ln -f ${BAMFILE}.bai ${OUTDIR}/$PLUGIN_OUT_BAIFILE"
	else
	    run "ln -sf ${BAMFILE} ${OUTDIR}/$PLUGIN_OUT_BAMFILE"
	    run "ln -sf ${BAMFILE}.bai ${OUTDIR}/$PLUGIN_OUT_BAIFILE"
	fi
    fi
    if [ "$OUTDIR" != "$TSP_FILEPATH_PLUGIN_DIR" ]; then
	if [ -n "$INPUT_BED_FILE" ]; then
	    run "ln -sf ${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_BEDFILE ${OUTDIR}/$PLUGIN_OUT_BEDFILE"
	fi
	if [ -n "$INPUT_SNP_BED_FILE" ]; then
	    run "ln -sf ${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE ${OUTDIR}/$PLUGIN_OUT_LOCI_BEDFILE"
	fi
    fi

    # Create the html report page
    echo "Generating html report..." >&2
    local HTMLOUT="${OUTDIR}/${HTML_RESULTS}";
    write_page_header "$LIFEGRIDDIR/SNPID.head.html" "$HTMLOUT";
    if [ -n "$SAMPLENAME" -a "$SAMPLENAME" != "None" ];then
        echo "<h3><center>Sample Name: $SAMPLENAME</center></h3>" >> "$HTMLOUT"
    fi
    cat "${OUTDIR}/$PLUGIN_OUT_COVERAGE_HTML" >> "$HTMLOUT"
    echo "<div id=\"sampleIDalleleCoverageTable\" fileurl=\"${PLUGIN_OUT_COV}\" class=\"center\"></div>" >> "$HTMLOUT"

    # Create partial html report (w/o file liinks section) and convert to PDF
    local HTMLTMP="${HTMLOUT}_.html"
    awk '$0!~/fileLinksTable/ {print}' "$HTMLOUT" > "$HTMLTMP"
    echo '</div></body></html>' >> "$HTMLTMP"
    PLUGIN_OUT_PDFFILE=`echo "$PLUGIN_OUT_BAMFILE" | sed -e 's/\.[^.]*$//'`
    PLUGIN_OUT_PDFFILE="${PLUGIN_OUT_PDFFILE}.${PLUGINNAME}.pdf"
    local PDFCMD="${BINDIR}/wkhtmltopdf-amd64 --load-error-handling ignore --no-background \"$HTMLTMP\" \"${OUTDIR}/$PLUGIN_OUT_PDFFILE\""
    eval "$PDFCMD >& /dev/null" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nWarning: No PDF view created. Command failed:" >&2
      echo "\$ $PDFCMD" >&2
    fi
    rm -f "$HTMLTMP"

    # Add in the full files links to the report
    write_file_links "$OUTDIR" "$PLUGIN_OUT_FILELINKS" >> "$HTMLOUT";
    echo "<div id=\"fileLinksTable\" fileurl=\"${PLUGIN_OUT_FILELINKS}\" class=\"center\"></div>" >> "$HTMLOUT"
    write_page_footer "$HTMLOUT";

    # Remove temporary files
    rm -f "${OUTDIR}/$PLUGIN_OUT_COVERAGE_HTML"
    return 0
}

barcode_load_list ()
{
    local ROWSUM_NODATA=""
    local NTAB=0
    for((NTAB=0;NTAB<${BC_SUM_ROWS};NTAB++)); do
        ROWSUM_NODATA="${ROWSUM_NODATA}<td>N/A</td> "
    done
    
    local BCN=0
    local BARCODE_BAM
    local BARCODE_LINE

    local FILTERCMD="grep ^barcode \"${BARCODES_LIST}\" | cut -d, -f2";
    for BARCODE_LINE in `eval "$FILTERCMD"`
    do
        BARCODES[$BCN]=${BARCODE_LINE}
        BARCODE_ROWSUM[$BCN]=$ROWSUM_NODATA
        BARCODE_BAM="${ANALYSIS_DIR}/${BARCODE_LINE}_${PLUGIN_BAM_FILE}"
        SAMPNAME=`echo "$PLUGIN_SAMPLE_NAMES" | sed -e "s/.*;$BARCODE_LINE=\([^;]*\).*/\1/"`
        if [ -z "$SAMPNAME" -o "$SAMPNAME" = "$PLUGIN_SAMPLE_NAMES" ];then
            SAMPNAME='None'
        fi
        if [ -f "$BARCODE_BAM" ]; then
            BARCODES_OK[${BCN}]=1
            BARCODE_SAMPNAME[${BCN}]="$SAMPNAME"
        else
            BARCODES_OK[${BCN}]=0
            BARCODE_SAMPNAME[${BCN}]=""
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
    if [ $NLINES -eq $NBARCODES ]; then
	REFRESHRATE=0
    fi
    write_html_header "$HTML" $REFRESHRATE
    barcode_links "$HTML" $NLINES
    write_html_footer "$HTML"
}

barcode_links ()
{
  # Define BC_COL arrays.
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
    echo "    <tr>" >> "$HTML"
    echo "     <th><span class=\"help\" style=\"width:100px !important\" title=\"Name of the barcode sequence and link to detailed report for reads associated with that barcode.\">Barcode ID</span></th>" >> "$HTML"
    echo "     <th><span class=\"help\" style=\"width:100px !important\" title=\"Sample Name associated with this barcode in the experiment plan.\">Sample Name</span></th>" >> "$HTML"
    local BCN
    local CWIDTH=120
    for((BCN=0;BCN<${BC_SUM_ROWS};BCN++))
    do
        if [ $BCN -eq 1 ]; then
            CWIDTH=66
        fi
        echo "     <th style=\"width:${CWIDTH}px !important\"><span class=\"help\" title=\"${BC_COL_HELP[$BCN]}\">${BC_COL_TITLE[$BCN]}</span></th>" >> "$HTML"
    done
    echo "    </tr>" >> "$HTML"

    local BARCODE
    local UNFIN=0
    for((BCN=0;BCN<${#BARCODES[@]};BCN++))
    do
        if [ $NLINES -ge 0 -a $BCN -ge $NLINES ]; then
            UNFIN=1
            break
        fi
        BARCODE=${BARCODES[$BCN]}
        if [[ ${BARCODES_OK[$BCN]} -eq 1 ]]; then
            echo "     <tr><td style=\"text-align:left\"><a style=\"cursor:help\" href=\"${BARCODE}/${HTML_RESULTS}\"><span title=\"Click to view the detailed report for barcode ${BARCODE}\">${BARCODE}</span></a></td>" >> "$HTML"
            echo "<td style=\"text-align:left\">${BARCODE_SAMPNAME[$BCN]}</td>" >> "$HTML"
            echo "      ${BARCODE_ROWSUM[$BCN]}</tr>" >> "$HTML"
        elif [[ ${BARCODES_OK[$BCN]} -eq 2 ]]; then
            echo "     <tr><td style=\"text-align:left\"><span class=\"help\" title=\"An error occurred while processing data for barcode ${BARCODE}\" style=\"color:red\">${BARCODE}</span></td>" >> "$HTML"
            echo "<td style=\"text-align:left\">${BARCODE_SAMPNAME[$BCN]}</td>" >> "$HTML"
            echo "      ${BARCODE_ROWSUM[$BCN]}</tr>" >> "$HTML"
        fi
    done

    echo "  </table></div>" >> "$HTML"
    echo " </div>" >> "$HTML"
    if [ $UNFIN -eq 1 ]; then
	display_static_progress "$HTML"
    fi
}

barcode_append_to_json_results ()
{
    local BARCODE=$1
    if [ -n "$2" ]; then
        if [ "$2" -gt 1 ]; then
            echo "," >> "$JSON_RESULTS"
        fi
    fi
    echo "    \"$BARCODE\" : {" >> "$JSON_RESULTS"
    echo -n "    " >> "$JSON_RESULTS"
    cat "$PLUGIN_OUT_HAPLOCODE" >> "$JSON_RESULTS"
    write_json_inner "$BARCODE_DIR" "$PLUGIN_OUT_READ_STATS" "mapped_reads" 6;
    echo "," >> "$JSON_RESULTS"
    write_json_inner "$BARCODE_DIR" "$PLUGIN_OUT_TARGET_STATS" "target_coverage" 6;
    if [ $BC_HAVE_LOCI -ne 0 ];then
        echo "," >> "$JSON_RESULTS"
	write_json_inner "$BARCODE_DIR" "$PLUGIN_OUT_LOCI_STATS" "hotspot_coverage" 6;
    fi
    echo "" >> "$JSON_RESULTS"
    echo -n "    }" >> "$JSON_RESULTS"
}

barcode ()
{
    # Load bar code ID and check for corresponding BAM files
    local BARCODES
    local BARCODE_IDS
    local BARCODES_OK
    local BARCODE_ROWSUM
    local BARCODE_SAMPNAME
    barcode_load_list;
    NBARCODES=${#BARCODES[@]}
    echo "Processing $NBARCODES barcodes..." >&2

    # Start json file
    write_json_header 1;

    # Empty Table - BARCODE set because header file expects this load javascript
    local BARCODE="TOCOME"
    local HTMLOUT="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
    barcode_partial_table "$HTMLOUT";
    
    # Go through the barcodes 
    local BARCODE_DIR
    local BARCODE_BAM
    local NLINE
    local BCN
    local BC_DONE
    local NJSON=0
    for((BCN=0;BCN<${NBARCODES};BCN++))
    do
        BARCODE=${BARCODES[$BCN]}
        BARCODE_DIR="${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE}"
        BARCODE_URL="."
        BARCODE_BAM="${ANALYSIS_DIR}/${BARCODE}_${PLUGIN_BAM_FILE}"
        NLINE=`expr ${BCN} + 1`

        if [[ ${BARCODES_OK[$BCN]} -eq 0 ]]; then
            echo -e "\nSkipping ${BARCODE}" >&2
        else
            # perform coverage anaysis and write content
            echo -e "\nProcessing barcode ${BARCODE}" >&2
            run "mkdir -p ${BARCODE_DIR}"
            SAMPLENAME="${BARCODE_SAMPNAME[$BCN]}"
            local RT=0
            eval "${SCRIPTSDIR}/call_variants.sh \"${BARCODE}_${PLUGIN_RUN_NAME}\" \"$BARCODE_DIR\" \"$BARCODE_URL\" \"$BARCODE_BAM\"" >&2 || RT=$?
            if [ $RT -ne 0 ]; then
                BC_ERROR=1
                if [ "$CONTINUE_AFTER_BARCODE_ERROR" -eq 0 ]; then
                    exit 1
                else
                    BARCODES_OK[${BCN}]=2
                fi
            else
                # process all result files to detailed html page and clean up
                write_html_results "${BARCODE}_${PLUGIN_RUN_NAME}" "$BARCODE_DIR" "$BARCODE_URL" "$BARCODE_BAM" "$SAMPLENAME"
                # collect table summary results
                if [ -f "${BARCODE_DIR}/${HTML_ROWSUMS}" ]; then
                    BARCODE_ROWSUM[$BCN]=`cat "${BARCODE_DIR}/$HTML_ROWSUMS"`
                fi
                NJSON=`expr ${NJSON} + 1`
	        barcode_append_to_json_results $BARCODE $NJSON;
            fi
            if [ "$PLUGIN_DEV_KEEP_INTERMEDIATE_FILES" -eq 0 ]; then
                rm -f ${BARCODE_DIR}/*.txt "${BARCODE_DIR}/$HTML_ROWSUMS"
            fi
        fi
	barcode_partial_table "$HTMLOUT" $NLINE
    done
    # write raw table as block_html (for 3.0 summary)
    COV_PAGE_WIDTH="auto"
    HTML_TORRENT_WRAPPER=0
    barcode_partial_table $HTML_BLOCK $NLINE;
    # finish up with json
    write_json_footer 1;
    if [ "$BC_ERROR" -ne 0 ]; then
        exit 1
    fi
}
