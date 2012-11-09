#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

write_file_links()
{
local OUTDIR="$TSP_FILEPATH_PLUGIN_DIR"
if [ -n ${1} ]; then
    OUTDIR=${1}
fi
local FILENAME="filelinks.xls"
if [ -n ${2} ]; then
    FILENAME=${2}
fi
local OUTFILE="${OUTDIR}/${FILENAME}"

echo -e "Text\tLink" > "$OUTFILE"
if [ -f "${OUTDIR}/$PLUGIN_OUT_PDFFILE" ]; then
  echo -e "Download a hard-copy A4 image of this report page. (binaryfile.pdf)\t${PLUGIN_OUT_PDFFILE}" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_COV" ]; then
  echo -e "Download all variant calls as a table file. (textfile.xls)\t${PLUGIN_OUT_COV}" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BEDFILE" ]; then
  echo -e "Download the target regions file. (textfile.bed)\t${PLUGIN_OUT_BEDFILE}" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_LOCI_BEDFILE" ]; then
  echo -e "Download the target hotspots file. (textfile.bed)\t${PLUGIN_OUT_LOCI_BEDFILE}" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BAMFILE" ]; then
  echo -e "Download the mapped reads file. (binaryfile.bam)\t${PLUGIN_OUT_BAMFILE}" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BAIFILE" ]; then
  echo -e "Download the mapped reads index file. (binaryfile.bai)\t${PLUGIN_OUT_BAIFILE}" >> "$OUTFILE"
fi
}
