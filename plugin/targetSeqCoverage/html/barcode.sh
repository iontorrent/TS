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

    for BARCODE_LINE in `cat ${TSP_FILEPATH_BARCODE_TXT} | grep "^barcode"`
    do
        BARCODE_ID=`echo ${BARCODE_LINE} | awk 'BEGIN{FS=","} {print $1}'`
        BARCODE=`echo ${BARCODE_LINE} | awk 'BEGIN{FS=","} {print $2}'`

        BARCODES[$BCN]=${BARCODE}
        BARCODE_IDS[$BCN]=${BARCODE_ID}
        BARCODE_ROWSUM[$BCN]=$ROWSUM_NODATA

        BARCODE_BAM="${BARCODE}_${PLUGIN_OUT_BAM_NAME}"
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
    if [ "$NLINES" -eq "$NBARCODES" ]; then
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
    echo "    <h2>Barcode Target Coverage Reports</h2>" >> "$HTML"
    echo "    <div>" >> "$HTML"
    echo "     <br/>" >> "$HTML"
    echo "     <style type=\"text/css\">" >> "$HTML"
    echo "      th {text-align:center;width=100%}" >> "$HTML"
    echo "      td {text-align:right;width=100%}" >> "$HTML"
    echo "      .help {cursor:help; border-bottom: 1px dotted #A9A9A9}" >> "$HTML"
    echo "     </style>" >> "$HTML"
    echo "     <table class=\"noheading\" style=\"table-layout:fixed\">" >> "$HTML"
    echo "     <tr>" >> "$HTML"
    echo "      <th style=\"width:200px !important\" rowspan=\"2\"><span class=\"help\" title=\"The barcode ID for each set of reads.\">Target Coverage Reports</span></th>" >> "$HTML"
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
        else
            echo "       <td style=\"text-align:left\"><span class=\"help\" title=\"No Data for barcode ${BARCODE}\" style=\"color:darkred\">${BARCODE}</span></td>" >> "$HTML"
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
    echo "," >> "$JSON_RESULTS"
    echo "  \"barcode\" : {" >> "$JSON_RESULTS"
    echo -n "    \"barcode_id\" : \"$BARCODE\"" >> "$JSON_RESULTS"
    write_json_inner "$BARCODE_DIR" "all_reads" 4;
    if [ $BC_SUM_ROWS -ge 8 ];then
	write_json_inner "$BARCODE_DIR" "filtered_reads" 4;
    fi
    echo "" >> "$JSON_RESULTS"
    echo -n "  }" >> "$JSON_RESULTS"
}

barcode ()
{
    local HTMLOUT="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"

    # Yes, there are barcodes
    echo "There are barcodes!" >&2
    
    # Unzip them here
    if [ -f ${ANALYSIS_DIR}/*.barcode.bam.zip ]; then
        unzip -n ${ANALYSIS_DIR}/*.barcode.bam.zip
    fi
    if [ -f ${ANALYSIS_DIR}/*.barcode.bam.bai.zip ]; then
        unzip -n ${ANALYSIS_DIR}/*.barcode.bam.bai.zip
    fi
    if [ -f ${ANALYSIS_DIR}/*.barcode.bai.zip ]; then
        unzip -n ${ANALYSIS_DIR}/*.barcode.bai.zip
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
    for((BCN=0;BCN<${NBARCODES};BCN++))
    do
        BARCODE=${BARCODES[$BCN]}
        BARCODE_DIR="${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE}"
        BARCODE_BAM="${BARCODE}_${PLUGIN_OUT_BAM_NAME}"

	# make barcode subdirectory and move unzipped bam files there
        run "mkdir -p ${BARCODE}"
        if [ ${BARCODES_OK[$BCN]} -eq 0 ]; then
            echo -e "\nSkipping ${BARCODE}" >&2
            echo "<h2>There was no BAM file for the barcode ${BARCODE}</h2>" > "${BARCODE_DIR}/${HTML_RESULTS}"
        else
            echo -e "\nProcessing barcode ${BARCODE}" >&2
            if [ ! -f ${BARCODE}/${BARCODE_BAM} ]; then
                run "mv ${BARCODE_BAM} ${BARCODE}/."
            fi
            if [ ! -f ${BARCODE}/${BARCODE_BAM}.bai ]; then
                if [ ! -f ${BARCODE_BAM}.bai ]; then
                    run "samtools index ${BARCODE_BAM}";
                fi
                run "mv ${BARCODE_BAM}.bai ${BARCODE}/."
            fi
            run_targetseq_analysis "$BARCODE_DIR" "${BARCODE}/$BARCODE_BAM"
            if [ -f "${BARCODE_DIR}/${HTML_ROWSUMS}" ]; then
                BARCODE_ROWSUM[$BCN]=`cat "${BARCODE_DIR}/$HTML_ROWSUMS"`
            fi
	    barcode_append_to_json_results $BARCODE;
        fi
        NLINE=`expr ${BCN} + 1`
	barcode_partial_table "$HTMLOUT" $NLINE
    done

    # finish up with json
    write_json_footer;

    # remove excess bam/bai files not transfer to subdirectories
    run "rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*bam ${TSP_FILEPATH_PLUGIN_DIR}/*bam.bai";

    ### block html ### 
    #-- hidden because of presentation issues, e.g. large vertical whitespace on close & redraw issues when scrolling
    #barcode_partial_table "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
}
