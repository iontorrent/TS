#!/bin/bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

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

local HTSV="This is a tab-separated-values text file with a .xls filename extension."

echo -e "Text\tLink\tDescription\tFormat" > "$OUTFILE"
# This has alias for IGV that FileLinksTable.js know how to substitute for local TS path
#echo -e "Open local IGV to import genome.\tIGV" >> "$OUTFILE"
if [ -f "${OUTDIR}/$PLUGIN_OUT_STATSFILE" ]; then
  echo -e "Coverage statistics summary file.\t${PLUGIN_OUT_STATSFILE}\tThis file is a text summary of the statistics presented in the tables at the beginning of this report.\tThis text file does not have a standard format.#The first line is the title of this plugin. Each subsequent line is either blank or a particular statistic title followed by a colon (:) and its value." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_AMPCOVFILE" ]; then
  COVFLD="#fwd_cov: The number of assigned forward strand reads that align to least 70% of the amplicon region.#rev_cov: The number of assigned reverse strand reads that align to least 70% of the amplicon region."
  echo -e "Amplicon coverage summary file.\t${PLUGIN_OUT_AMPCOVFILE}\tFine coverage summary data used to create the Amplicon Coverage Chart.\t$HTSV #It has 12 named fields:#contig_id: The name of the chromosome or contig of the reference for this amplicon.#contig_srt: The start location of the amplicon target region. Note that this coordinate is 1-based, unlike the corresponding 0-based coordinate in the original targets BED file.#contig_end: The last base coordinate of this amplicon target region. Note that the length of the amplicon target is given as tlen = (contig_end - contig_srt + 1).#region_id: The ID for this amplicon as given as the 4th column of the targets BED file.#gene_id or attributes: The gene symbol or attributes field as provided in the targets BED file.#gc_count: The number of G and C bases in the target region. Hence, %GC = 100% * gc_count / tlen.#overlaps: The number of times this target was overlapped by any read by at least one base. Note that individual reads might overlap multiple amplicons where the amplicon regions themselves overlap.${COVFLD}#total_reads: The total number of reads assigned to this amplicon. This value equals (fwd_reads + rev_reads) and is the field that rows of this file are ordered by (then by contig id, srt and end).#fwd_reads: The number of forward strand reads assigned to this amplicon.#rev_reads: The number of reverse strand reads assigned to this amplicon." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BAMFILE" ]; then
  echo -e "Aligned reads BAM file.\t${PLUGIN_OUT_BAMFILE}\tAll reads aligned used to generate this report page.\tBinary form of the SAM format file that records individual reads and their alignment to the reference genome.#Refer to the current SAM tools documentation for more file format information." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BAIFILE" ]; then
  echo -e "Aligned reads BAI file.\t${PLUGIN_OUT_BAIFILE}\tBinary BAM index file as required by some analysis tools and alignment viewers such as IGV.\tThis file may be generated from an ordered BAM file using SAM tools.#Refer to the current SAM tools documentation for more information." >> "$OUTFILE"
fi
if [ -n "$PLUGIN_OUT_BEDPAGE" ]; then
  echo -e "Link to targets (BED) file upload page.\t${PLUGIN_OUT_BEDPAGE}\tLink to the target regions upload page where you can review and download the BED file that defined the regions of the reference analyzed in this report.\tFollow link for target regions file description and processing details." >> "$OUTFILE"
fi
}
