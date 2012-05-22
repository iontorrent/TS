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
    local BARCODE
    local BARCODE_ID
    local BARCODE_BAM
    local BARCODE_LINE

    for BARCODE_LINE in `cat ${BARCODES_LIST} | grep "^barcode"`
    do
        BARCODE_ID=`echo ${BARCODE_LINE} | awk 'BEGIN{FS=","} {print $1}'`
        BARCODE=`echo ${BARCODE_LINE} | awk 'BEGIN{FS=","} {print $2}'`

        BARCODES[$BCN]=${BARCODE}
        BARCODE_IDS[$BCN]=${BARCODE_ID}
        BARCODE_ROWSUM[$BCN]=$ROWSUM_NODATA

        BARCODE_BAM="${ANALYSIS_DIR}/${BARCODE}_${PLUGIN_BAM_FILE}"
        if [ -f ${BARCODE_BAM} ]; then
            BARCODES_OK[${BCN}]=1
        else
            BARCODES_OK[${BCN}]=0
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
    barcode_links "$HTML" $NLINES
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
    echo "   <div id=\"BarcodeList\" class=\"report_block\"/>" >> "$HTML"
    echo "    <h2>Barcode Coverage Reports</h2>" >> "$HTML"
    echo "    <div>" >> "$HTML"
    echo "     <br/>" >> "$HTML"
    echo "     <style type=\"text/css\">" >> "$HTML"
    echo "      th {text-align:center;width=100%}" >> "$HTML"
    echo "      td {text-align:right;width=100%}" >> "$HTML"
    echo "      .help {cursor:help; border-bottom: 1px dotted #A9A9A9}" >> "$HTML"
    echo "     </style>" >> "$HTML"
    echo "     <table class=\"noheading\" style=\"table-layout:fixed\">" >> "$HTML"
    echo "     <tr>" >> "$HTML"
    echo "      <th style=\"width:200px !important\" rowspan=\"2\"><span class=\"help\" title=\"The barcode ID for each set of reads.\">Coverage Analysis Reports</span></th>" >> "$HTML"
    echo "      <th style=\"width:300px !important\" colspan=\"4\"><span class=\"help\" title=\"${PLUGIN_INFO_ALLREADS}\">All Reads</span></th>" >> "$HTML"
    if [ $BC_SUM_ROWS -ge 8 ];then
	echo "      <th style=\"width:300px !important\" colspan=\"4\"><span class=\"help\" title=\"${PLUGIN_INFO_USTARTS}\">Unique Starts</span></th>" >> "$HTML"
    fi
    echo "     </tr>" >> "$HTML"
    echo "      <tr>" >> "$HTML"

    local BCN
    for((BCN=0;BCN<${BC_SUM_ROWS};BCN++))
    do
        echo "       <th><span class=\"help\" title=\"${BC_COL_HELP[$BCN]}\">${BC_COL_TITLE[$BCN]}</span></th>" >> "$HTML"
    done
    echo "      </tr>" >> "$HTML"

    local BARCODE
    local UNFIN=0
    for((BCN=0;BCN<${#BARCODES[@]};BCN++))
    do
        if [ $NLINES -ge 0 -a $BCN -ge $NLINES ]; then
            UNFIN=1
            break
        fi
        BARCODE=${BARCODES[$BCN]}
        echo "      <tr>" >> "$HTML"
        if [ ${BARCODES_OK[$BCN]} -eq 1 ]; then
            echo "       <td style=\"text-align:left\"><a style=\"cursor:help\" href=\"${BARCODE}/${HTML_RESULTS}\"><span title=\"Click to view the detailed coverage report for barcode ${BARCODE}\">${BARCODE}</span></a></td>" >> "$HTML"
        elif [ ${BARCODES_OK[$BCN]} -eq 2 ]; then
            echo "       <td style=\"text-align:left\"><span class=\"help\" title=\"An error occurred while processing data for barcode ${BARCODE}\" style=\"color:red\">${BARCODE}</span></td>" >> "$HTML"
        else
            echo "       <td style=\"text-align:left\"><span class=\"help\" title=\"No Data for barcode ${BARCODE}\" style=\"color:grey\">${BARCODE}</span></td>" >> "$HTML"
        fi
        echo "           ${BARCODE_ROWSUM[$BCN]}" >> "$HTML"
        echo "      </tr>" >> "$HTML"
    done

    echo "     </table></div>" >> "$HTML"
    if [ $UNFIN -eq 1 ]; then
	display_static_progress "$HTML"
    fi
    echo "   </div>" >> "$HTML"
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
    write_json_inner "${BARCODE_DIR}/all_reads" "summary.txt" "all_reads" 6;
    if [ $BC_SUM_ROWS -ge 8 ];then
        echo "," >> "$JSON_RESULTS"
	write_json_inner "${BARCODE_DIR}/filtered_reads" "summary.txt" "filtered_reads" 6;
    fi
    echo "" >> "$JSON_RESULTS"
    echo -n "    }" >> "$JSON_RESULTS"
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
    local BARCODES
    local BARCODE_IDS
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
        BARCODE_BAM="${BARCODE}_${PLUGIN_BAM_FILE}"
        NLINE=`expr ${BCN} + 1`

        # ensure old data is not retained
        run "rm -rf ${BARCODE_DIR}"
        if [ ${BARCODES_OK[$BCN]} -eq 0 ]; then
            echo -e "\nSkipping ${BARCODE}" >&2
        else
            echo -e "\nProcessing barcode ${BARCODE}" >&2
            run "mkdir -p ${BARCODE_DIR}"
            run "ln -s ${ANALYSIS_DIR}/${BARCODE_BAM} ${BARCODE_DIR}/${BARCODE_BAM}";
            run "ln -s ${ANALYSIS_DIR}/${BARCODE_BAM}.bai ${BARCODE_DIR}/${BARCODE_BAM}.bai";
            # unfortunately the run_coverage_analysis() function cannot be called here
            # as the error from subshell needs to be fully processed in calling method
            # - returning from any function after subshell appears to exit entire process
            local RT=0
            eval "${SCRIPTSDIR}/run_coverage_analysis.sh $LOGOPT $RUNCOV_OPTS -R \"$HTML_RESULTS\" -T \"$HTML_ROWSUMS\" -H \"${TSP_FILEPATH_PLUGIN_DIR}\" -D \"$BARCODE_DIR\" -B \"$PLUGIN_TARGETS\" -P \"$PADDED_TARGETS\" \"$TSP_FILEPATH_GENOME_FASTA\" \"${BARCODE_DIR}/${BARCODE_BAM}\"" || RT=$?
            if [ $RT -ne 0 ]; then
                BC_ERROR=1
                if [ "$CONTINUE_AFTER_BARCODE_ERROR" -eq 0 ]; then
                    exit 1
                else
                    BARCODES_OK[${BCN}]=2
                fi
            else
                if [ -f "${BARCODE_DIR}/${HTML_ROWSUMS}" ]; then
                    BARCODE_ROWSUM[$BCN]=`cat "${BARCODE_DIR}/$HTML_ROWSUMS"`
                    rm -f "${BARCODE_DIR}/${HTML_ROWSUMS}"
                fi
                NUSED=`expr ${NUSED} + 1`
	        barcode_append_to_json_results $BARCODE $NUSED;
            fi
        fi
	barcode_partial_table "$HTMLOUT" $NLINE
    done
    # finish up with json
    write_json_footer 1;
    if [ "$BC_ERROR" -ne 0 ]; then
        exit 1
    fi
}
