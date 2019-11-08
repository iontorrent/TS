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
if [ "$PLUGIN_READCOV" = "c70" ]; then
  COVFLD="#fwd_cov: The number of assigned forward strand reads that align to least 70% of the amplicon region.#rev_cov: The number of assigned reverse strand reads that align to least 70% of the amplicon region."
else
  COVFLD="#fwd_e2e: The number of assigned forward strand reads that read from one end of the amplicon region to the other.#rev_e2e: The number of assigned reverse strand reads that read from one end of the amplicon region to the other."
fi

echo -e "Text\tLink\tDescription\tFormat" > "$OUTFILE"
# This has alias for IGV that FileLinksTable.js know how to substitute for local TS path
#echo -e "Open local IGV to import genome.\tIGV" >> "$OUTFILE"
if [ -f "${OUTDIR}/$PLUGIN_OUT_STATSFILE" ]; then
  echo -e "Coverage statistics summary file.\t${PLUGIN_OUT_STATSFILE}\tThis file is a text summary of the statistics presented in the tables at the beginning of this report.\tThis text file does not have a standard format.#The first line is the title of this plugin. Each subsequent line is either blank or a particular statistic title followed by a colon (:) and its value." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_AMPCOVFILE" ]; then
  echo -e "Amplicon molecular coverage summary file.\t${PLUGIN_OUT_AMPCOVFILE}\tAmplicon molecular coverage summary table, used to create the Amplicon molecular Coverage Chart.\t$HTSV #It has 16 named fields:#contig_id: The name of the chromosome or contig of the reference for this amplicon.#contig_srt: The start location of the amplicon target region. Note that this coordinate is 1-based, unlike the corresponding 0-based coordinate in the original targets BED file.#contig_end: The last base coordinate of this amplicon target region. Note that the length of the amplicon target is given as (contig_end - contig_srt + 1).#region_id: The ID for this amplicon as given as the 4th column of the targets BED file.#gene_id or attributes: The gene symbol or attributes field as provided in the targets BED file.#func_mol_cov: The number of molecules (functional molecules) which are available for variantCaller.#lod: LOD (limitation of detection) calulated from the number of functional molecules. #strict_func_umt_rate: the percentage of functional molecules used with strict molecular tags. #func_mol_cov_loss_due_to_strand: the percentage of functional molecules loss due to strand bias. #fwd_only_mol_cov: the number of molecules containing forward strand only.#rev_only_mol_cov: the number of molecules containing reverse strand only.#both_strands_mol_cov: the number of molecules containing both forward strand and reverse strand. #perc_functional_reads: the percentage of reads contributed to functional molecules. #reads_per_func_mol: Average reads per functional molecules. #perc_to_mol_(<3_reads): the percentage of reads contributed to small size molecules(size<3).#perc_to_mol_(>=3&<30_reads): the percentage of reads contributed to median size molecules(size>=3 && size<30).#perc_to_mol_(>=30_reads): the percentage of reads contributed to large size molecules(size>30)." >> "$OUTFILE"
fi
if [ -n "$PLUGIN_OUT_BEDPAGE" ]; then
  echo -e "Link to targets (BED) file upload page.\t${PLUGIN_OUT_BEDPAGE}\tLink to the target regions upload page where you can review and download the BED file that defined the regions of the reference analyzed in this report.\tFollow link for target regions file description and processing details." >> "$OUTFILE"
fi
  echo -e "Zip report.\tzipReport.php3\tDownload a zip file containing a PDF of the current report page and primary coverage analysis files.\tOnce downloaded this file (molecularCoverageAnalysisReport.zip) may be unzipped to extract an extended PDF of the current report page and primary coverage analysis files." >> "$OUTFILE"
}
