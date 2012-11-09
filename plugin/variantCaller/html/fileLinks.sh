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
# Link to Broad IGV
#echo -e "Open IGV to import genome.\thttp://www.broadinstitute.org/igv/projects/current/igv.php" >> "$OUTFILE"
# Link to internal IGV
echo -e "Open internal IGV to import genome.\t"+ ${RUNINFO__NET_LOCATION}+":8080/IgvServlet/igv" >> "$OUTFILE"
if [ -f "${OUTDIR}/$PLUGIN_OUT_ALLVARS" ]; then
  echo -e "Download all variant calls as a table file. (textfile.xls)\t${PLUGIN_OUT_ALLVARS}" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_COV" ]; then
  echo -e "Download all allele counts as a table file. (textfile.xls)\t${PLUGIN_OUT_COV}" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/${PLUGIN_OUT_SNPS_VCF}.gz" ]; then
  echo -e "Download the SNP calls as a VCF file. (binaryfile.vcf.gz)\t${PLUGIN_OUT_SNPS_VCF}.gz" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/${PLUGIN_OUT_SNPS_VCF}.gz.tbi" ]; then
  echo -e "Download the SNP calls VCF index file. (binaryfile.vcf.gz.tbi)\t${PLUGIN_OUT_SNPS_VCF}.gz.tbi" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/${PLUGIN_OUT_INDELS_VCF}.gz" ]; then
  echo -e "Download the INDEL calls as a VCF file. (binaryfile.vcf.gz)\t${PLUGIN_OUT_INDELS_VCF}.gz" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/${PLUGIN_OUT_INDELS_VCF}.gz.tbi" ]; then
  echo -e "Download the INDEL calls VCF index file. (binaryfile.vcf.gz.tbi)\t${PLUGIN_OUT_INDELS_VCF}.gz.tbi" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/${PLUGIN_OUT_MERGED_VCF}.gz" ]; then
  echo -e "Download combined SNP and INDEL calls as a VCF file. (binaryfile.vcf.gz)\t${PLUGIN_OUT_MERGED_VCF}.gz" >> "$OUTFILE"
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
# may be possible for both unique starts and trimming to be applied
local PROCESS="mapped"
if [ -f "${OUTDIR}/$PLUGIN_OUT_USTARTSBAM" ]; then
  echo -e "Download the unique starts filtered reads file. (binaryfile.bam)\t${PLUGIN_OUT_USTARTSBAM}" >> "$OUTFILE"
  PROCESS="filtered"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_USTARTSBAI" ]; then
  echo -e "Download the unique starts filtered reads index file. (binaryfile.bam)\t${PLUGIN_OUT_USTARTSBAI}" >> "$OUTFILE"
  PROCESS="filtered"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_TRIMPBAM" ]; then
  echo -e "Download the primer-trimmed $PROCESS reads file. (binaryfile.bam)\t${PLUGIN_OUT_TRIMPBAM}" >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_TRIMPBAI" ]; then
  echo -e "Download the primer-trimmed $PROCESS reads index file. (binaryfile.bai)\t${PLUGIN_OUT_TRIMPBAI}" >> "$OUTFILE"
fi
}
