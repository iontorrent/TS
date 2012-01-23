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
echo "<style type=\"text/css\">td {width:100% !important;padding:4px;text-align:left}</style>"
echo "<div id=\"FileLinks\" class=\"report_block\">"
echo " <h2>File Links</h2>"
echo " <div class=\"demo_jui\">"
echo "  </br>"
echo "  <table class=\"noheading\" id=\"filelinks\">"
echo "   <tr><td><a href="http://www.broadinstitute.org/igv/projects/current/igv.php">Open IGV to import genome.</a></td></tr>"
if [ -f "${OUTDIR}/$PLUGIN_OUT_ALLVARS" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_ALLVARS\">Download all variant calls as a table file. (textfile.xls)</a></td></tr>"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_COV" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_COV\">Download all allele counts as a table file. (textfile.xls)</a></td></tr>"
fi
if [ -f "${OUTDIR}/${PLUGIN_OUT_SNPS_VCF}.gz" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/${PLUGIN_OUT_SNPS_VCF}.gz\">Download the SNP calls as a VCF file. (binaryfile.vcf.gz)</a></td></tr>"
fi
if [ -f "${OUTDIR}/${PLUGIN_OUT_SNPS_VCF}.gz.tbi" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/${PLUGIN_OUT_SNPS_VCF}.gz.tbi\">Download the SNP calls VCF index file. (binaryfile.vcf.gz.tbi)</a></td></tr>"
fi
if [ -f "${OUTDIR}/${PLUGIN_OUT_INDELS_VCF}.gz" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/${PLUGIN_OUT_INDELS_VCF}.gz\">Download the INDEL calls as a VCF file. (binaryfile.vcf.gz)</a></td></tr>"
fi
if [ -f "${OUTDIR}/${PLUGIN_OUT_INDELS_VCF}.gz.tbi" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/${PLUGIN_OUT_INDELS_VCF}.gz.tbi\">Download the INDEL calls VCF index file. (binaryfile.vcf.gz.tbi)</a></td></tr>"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BEDFILE" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_BEDFILE\">Download the target regions file. (textfile.bed)</a></td></tr>"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_LOCI_BEDFILE" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_LOCI_BEDFILE\">Download the target hotspots file. (textfile.bed)</a></td></tr>"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BAMFILE" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_BAMFILE\">Download the mapped reads file. (binaryfile.bam)</a></td></tr>"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BAIFILE" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_BAIFILE\">Download the mapped reads index file. (binaryfile.bai)</a></td></tr>"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_TRIMPBAM" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_TRIMPBAM\">Download the primer-trimmed mapped reads file. (binaryfile.bam)</a></td></tr>"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_TRIMPBAI" ]; then
  echo "   <tr><td><a href=\"${OUTURL}/$PLUGIN_OUT_TRIMPBAI\">Download the primer-trimmed mapped reads index file. (binaryfile.bam)</a></td></tr>"
fi
echo -e "  </table>\n </div>\n</div>"
}
