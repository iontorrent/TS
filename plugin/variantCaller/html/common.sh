#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

write_html_header ()
{
    local HTML="${TSP_FILEPATH_PLUGIN_DIR}/header"
    if [ -n "$1" ]; then
        HTML="$1"
    fi
    local REFRESHRATE=0
    if [ -n "$2" ]; then
	REFRESHRATE=$2
    fi
    echo '<?xml version="1.0" encoding="iso-8859-1"?>' > "$HTML"
    echo '<!DOCTYPE html>' >> "$HTML"
    echo '<html>' >> "$HTML"
    print_html_head $REFRESHRATE >> "$HTML"
    echo ' <body>' >> "$HTML"
    print_html_logo >> "$HTML";

    if [ -z "$COV_PAGE_WIDTH" ];then
	echo '  <div id="inner">' >> "$HTML"
    else
	echo "  <div style=\"width:${COV_PAGE_WIDTH}px;margin-left:auto;margin-right:auto;height:100%\">" >> "$HTML"
    fi
    echo "   <h1><center><span style=\"cursor:help\" title=\"Library Type='${PLUGINCONFIG__LIBRARYTYPE_ID}'   Variant Frequency='${PLUGINCONFIG__VARIATIONTYPE_ID}' Target Regions='${PLUGINCONFIG__TARGETREGIONS_ID}' Target loci='${PLUGINCONFIG__TARGETLOCI_ID}'\">Variant Caller Report</span></center></h1>" >> "$HTML"
}

write_html_footer ()
{
    local HTML="${TSP_FILEPATH_PLUGIN_DIR}/footer"
    if [ -n "$1" ]; then
        HTML="$1"
    fi
    print_html_end_javascript >> "$HTML"
    print_html_footer >> "$HTML"
    echo '  <br/><br/></div>' >> "$HTML"
    echo ' </body></html>' >> "$HTML"
}

display_static_progress ()
{
    local HTML="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
    if [ -n "$1" ]; then
        HTML="$1"
    fi
    echo "    <br/><h3 style=\"text-align:center;color:red\">*** Analysis is not complete ***</h3>" >> "$HTML"
    echo "    <a href=\"javascript:document.location.reload();\" ONMOUSEOVER=\"window.status='Refresh'; return true\">" >> "$HTML"
    echo "    <div style=\"text-align:center\">Click here to refresh</div></a>" >> "$HTML"
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
    echo "  \"Library Type\" : \"$PLUGINCONFIG__LIBRARYTYPE_ID\"," >> "$JSON_RESULTS"
    echo "  \"Variant Detection\"  : \"$PLUGINCONFIG__VARIATIONTYPE_ID\"," >> "$JSON_RESULTS"
    echo "  \"Target Regions\" : \"$PLUGINCONFIG__TARGETREGIONS_ID\"," >> "$JSON_RESULTS"
    echo "  \"Target Loci\"  : \"$PLUGINCONFIG__TARGETLOCI_ID\"," >> "$JSON_RESULTS"
    echo "  \"barcoded\" : \"$haveBC\"," >> "$JSON_RESULTS"
    if [ "$haveBC" = "true" ]; then
        echo "  \"barcodes\" : {" >> "$JSON_RESULTS"
    fi
}

write_json_footer ()
{
    if [ -n "$1" ]; then
	if [ $1 -ne 0 ]; then
	    echo -e "\n  }" >> "$JSON_RESULTS"
        fi
    fi
    echo -e "\n}" >> "$JSON_RESULTS"
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

