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
if [ -f "${OUTDIR}/$PLUGIN_OUT_DOCFILE" ]; then
  echo -e "Base depth of coverage file.\t${PLUGIN_OUT_DOCFILE}\tBase read depths, used to create the Depth of Coverage Chart.\t$HTSV #It has 5 named fields:#read_depth: The depth at which a (targeted) reference base has been read.#base_cov: The number of times any base was read (covered) at this depth.#base_cum_cov: The cumulative number of reads (coverage) at this read depth or greater.#norm_read_depth: The normalized read depth (depth divided by average base read depth).#pc_base_cum_cov: As base_cum_cov but represented as a percentage of the total base reads.#NOTE: Lines (read depths) for which base_cov is 0 are omitted to avoid excessively large files being produced in certain situations." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_AMPCOVFILE" ]; then
  echo -e "Amplicon coverage summary file.\t${PLUGIN_OUT_AMPCOVFILE}\tAmplicon coverage summary table, used to create the Amplicon Coverage Chart.\t$HTSV #It has 12 named fields:#contig_id: The name of the chromosome or contig of the reference for this amplicon.#contig_srt: The start location of the amplicon target region. Note that this coordinate is 1-based, unlike the corresponding 0-based coordinate in the original targets BED file.#contig_end: The last base coordinate of this amplicon target region. Note that the length of the amplicon target is given as (contig_end - contig_srt + 1).#region_id: The ID for this amplicon as given as the 4th column of the targets BED file.#gene_id or attributes: The gene symbol or attributes field as provided in the targets BED file.#gc_count: The number of G and C bases in the target region. The %GC reported uses this count divided by the amplicon (insert) length.#overlaps: The number of times this target was overlapped by any read by at least one base. Note that individual reads might overlap multiple amplicons where the amplicon regions themselves overlap.${COVFLD}#total_reads: The total number of reads assigned to this amplicon. This value equals (fwd_reads + rev_reads) and is the field that rows of this file are ordered by (then by contig id, srt and end).#fwd_reads: The number of forward strand reads assigned to this amplicon.#rev_reads: The number of reverse strand reads assigned to this amplicon.#covNx: The number of bases of the amplicon target that had at least N reads. There are 3 such columns for the specified coverage tiers, which by default are cov20x, cov100x and cov500x." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_RNACOVFILE" ];then
  echo -e "Amplicon coverage summary file.\t${PLUGIN_OUT_RNACOVFILE}\tAmplicon reads summary table, used to create the Amplicon Coverage Chart.\t$HTSV #It has 12 named fields:#contig_id: The name of the chromosome or contig of the reference for this amplicon.#contig_srt: The start location of the amplicon target region. Note that this coordinate is 1-based, unlike the corresponding 0-based coordinate in the original targets BED file.#contig_end: The last base coordinate of this amplicon target region. Note that the length of the amplicon target is given as (contig_end - contig_srt + 1).#region_id: The ID for this amplicon as given as the 4th column of the targets BED file.#gene_id or attributes: The gene symbol or attributes field as provided in the targets BED file.#gc_count: The number of G and C bases in the target region. The %GC reported uses this count divided by the amplicon (insert) length.#overlaps: The number of times this target was overlapped by any read by at least one base. Note that individual reads might overlap multiple amplicons where the amplicon regions themselves overlap.${COVFLD}#total_reads: The total number of reads assigned to this amplicon. This value equals (fwd_reads + rev_reads) and is the field that rows of this file are ordered by (then by contig id, srt and end).#fwd_reads: The number of forward strand reads assigned to this amplicon.#rev_reads: The number of reverse strand reads assigned to this amplicon." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_CTGCOVFILE" ]; then
  echo -e "Contig coverage summary file.\t${PLUGIN_OUT_CTGCOVFILE}\tContig reads summary table, used to create the Contig Coverage Chart.\t$HTSV #It has 12 named fields:#contig_id: The name of the contig sequence (or chromosome or transcript) of the reference.#contig_srt: The start location of the contig target region. This field has the value 1, corresponding to the first base of the contig.#contig_end: The last base coordinate of this contig, also it's length in bases.#region_id: Same as contig_id.#attributes: The attributes field (unused).#gc_count: The number of G and C bases in the target region. The %GC reported uses this count divided by the contig length, which may may be inconsisent (lower) than given by other defintions for references that have large numbers of degenerate or unknown bases (e.g. N's) included.#overlaps: The number of times this contig target was overlapped by any read by at least one base. This field should be the same as total_reads.${COVFLD}#total_reads: The total number of reads assigned to this amplicon. This value equals (fwd_reads + rev_reads) and is the field that rows of this file are ordered by (then by contig id, srt and end).#fwd_reads: The number of forward strand reads assigned to this amplicon.#rev_reads: The number of reverse strand reads assigned to this amplicon." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_TRGCOVFILE" ]; then
  echo -e "Target coverage summary file.\t${PLUGIN_OUT_TRGCOVFILE}\tTarget coverage summary table, used to create the Target Coverage Chart.\t$HTSV #It has 12 named fields:#contig_id: The name of the chromosome or contig of the reference for this target.#contig_srt: The start location of the target region. Note that this coordinate is 1-based, unlike the corresponding 0-based coordinate in the original targets BED file.#contig_end: The last base coordinate of this target region. Note that the length of the target is given as (contig_end - contig_srt + 1).#region_id: The ID for this target as given as the 4th column of the targets BED file.#gene_id: The gene symbol as given as the last field of the targets BED file.#gc_count: The number of G and C bases in the target region. The number of G and C bases in the target region. The %GC reported uses this count divided by the target length. Note that this may appear to be low for chromosome targets that have large numbers of degenerate or unknown bases (e.g. N's) included.#covered: The number of bases of this target that were covered by at least one read. Note that this may include deletions or inner regions that are uncovered by reads.#uncov_5p: The number of bases that were not covered a the 5' (upstream) end of the forward DNA strand.#uncov_3p: The number of bases that were not covered a the 3' (downstream) end of the forward DNA strand. For TargetSeq this may indicate poor probe coverage at this end of the target.#ave_basereads: The average target base read depth. This value equals the total base reads for the target divided by the target length. The rows in this file are ordered by this value (then by contig id, srt and end).#fwd_basereads: The number of forward strand base reads assigned to this target.#rev_basereads: The number of reverse strand base reads assigned to this target.#covNx: The number of bases of the amplicon target that had at least N reads. There are 3 such columns for the specified coverage tiers, which by default are cov20x, cov100x and cov500x." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_CHRCOVFILE" ]; then
  echo -e "Chromosome base coverage summary file.\t${PLUGIN_OUT_CHRCOVFILE}\tBase reads per chromosome, used to create the default view of the Reference Coverage Chart.\t$HTSV #It has 6 or 8 fields:#chrom: The name of the chromosome or contig of the reference.#start: Coordinate of the first base in this chromosome. This is always 1.#end: Coordinate of the last base of this chromsome. Also its length in bases.#fwd_basereads: Total number of forward strand base reads for the chromosome.#rev_basereads: Total number reverse strand base reads for the chromosome.#fwd_trg_basereads (if present): Total number of forward strand base reads that mapped over at least one target region.#rev_trg_basereads (if present): Total number of reverse strand base reads that mapped over at least one target region.#total_reads: Total (sequencing) reads mapped to individual contigs." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BAMFILE" ]; then
  echo -e "Aligned reads BAM file.\t${PLUGIN_OUT_BAMFILE}\tAll reads aligned used to generate this report page.\tBinary form of the SAM format file that records individual reads and their alignment to the reference genome.#Refer to the current SAM tools documentation for more file format information." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BAIFILE" ]; then
  echo -e "Aligned reads BAI file.\t${PLUGIN_OUT_BAIFILE}\tBinary BAM index file as required by some analysis tools and alignment viewers such as IGV.\tThis file may be generated from an ordered BAM file using SAM tools.#Refer to the current SAM tools documentation for more information." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_TRIMPBAM" ]; then
  echo -e "Primer-trimmed reads BAM file.\t${PLUGIN_OUT_TRIMPBAM}\tBinary primer-trimmed aligned reads. Created from the original alignment file by trimming reads to specific amplicons regions they are assigned to, where necessary to resolve overlaps with multiple amplicon target regions.\tThis file may be generated from an ordered (primer trimmed) BAM file using SAM tools.#Refer to the current SAM tools documentation for more file format information." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_TRIMPBAI" ]; then
  echo -e "Primer-trimmed reads BAI file.\t${PLUGIN_OUT_TRIMPBAI}\tBinary BAM index file as required by some analysis tools and alignment viewers such as IGV.\tThis file may be generated from an ordered BAM file using SAM tools.#Refer to the current SAM tools documentation for more information." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BEDFILE_MERGED" ]; then
  echo -e "Targets BED file.\t${PLUGIN_OUT_BEDFILE_MERGED}\tTargets BED file used to generate base coverage analysis.\tMay be standard BED format if only base coverage is required." >> "$OUTFILE"
fi
if [ -f "${OUTDIR}/$PLUGIN_OUT_BEDFILE_UNMERGED" ]; then
  echo -e "Targets BED file (annotated).\t${PLUGIN_OUT_BEDFILE_UNMERGED}\tTargets BED file used to generate target coverage analsyis (assumed unmerged for AmpliSeq).\tMust be generated by the gcAnnoBed.pl script from a published ionTorrent BED format or with the bedDetail format as used by the UCSC Genome Browser." >> "$OUTFILE"
fi
if [ -n "$PLUGIN_OUT_BEDPAGE" ]; then
  echo -e "Link to targets (BED) file upload page.\t${PLUGIN_OUT_BEDPAGE}\tLink to the target regions upload page where you can review and download the BED file that defined the regions of the reference analyzed in this report.\tFollow link for target regions file description and processing details." >> "$OUTFILE"
fi
  echo -e "Download ZIP report.\tzipReport.php3\tDownload a zip file containing a PDF of the current report page and primary coverage analysis files.\tOnce downloaded this file (coverageAnalysisReport.zip) may be unzipped to extract an extended PDF of the current report page and primary coverage analysis files." >> "$OUTFILE"
}
