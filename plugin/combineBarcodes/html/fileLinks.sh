#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

write_html_file_links()
{
local OUTURL="$OUTURL"
if [ -n ${1} ]; then
    OUTURL=${1}
fi
local OUTDIR="$TSP_FILEPATH_PLUGIN_DIR"
if [ -n ${2} ]; then
    OUTDIR=${2}
fi
echo "<div id=\"FileLinks\" class=\"report_block\">"
echo " <h2>File Links</h2>"
echo " <div class=\"demo_jui\">"
echo "  </br>"
echo "  <table class=\"noheading\" id=\"filelinks\">"
if [ -f "${OUTDIR}/$PLUGIN_OUT_BAMFILE" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_BAMFILE\">Download the combined aligned reads file.</a></td></tr>"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BAIFILE" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_BAIFILE\">Download the combined aligned reads index file.</a></td></tr>"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_USTARTS_BAMFILE" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_USTARTS_BAMFILE\">Download the combined unique reads file.</a></td></tr>"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_USTARTS_BAIFILE" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_USTARTS_BAIFILE\">Download the combined unique reads index file.</a></td></tr>"
fi
echo -e "  </table>\n </div>\n</div>"
}
