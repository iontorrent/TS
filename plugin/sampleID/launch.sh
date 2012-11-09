#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#MAJOR_BLOCK

VERSION="3.2.43647"

# DEVELOPMENT/DEBUG options:
# NOTE: the following should all be set to 0 in production mode
export PLUGIN_DEV_KEEP_INTERMEDIATE_FILES=0;
export PLUGIN_DEV_FULL_LOG=0;          # 1 for variant calling log, 2 for additional xtrace (not recommended)

export CONTINUE_AFTER_BARCODE_ERROR=1;
export ENABLE_HOTSPOT_LEFT_ALIGNMENT=0;

# Setup environment

export INPUT_BED_FILE="targets/KIDDAME_sampleID_regions.bed"
export INPUT_SNP_BED_FILE="targets/KIDDAME_sampleID_loci.bed"

export REFERENCE="$TSP_FILEPATH_GENOME_FASTA"
export REFERENCE_FAI="${REFERENCE}.fai"

echo "Employed run options:" >&2
echo "  Reference Genome:  $REFERENCE" >&2
echo "  Aligned Reads:     $TSP_FILEPATH_BAM" >&2
echo "  Target Regions:    $INPUT_BED_FILE" >&2
echo "  Target Loci:       $INPUT_SNP_BED_FILE" >&2

# Set up logging options for commands
export ERROUT="2> /dev/null"
export LOGOPT=""
if [ "$PLUGIN_DEV_FULL_LOG" -gt 0 ]; then
    ERROUT=""
    LOGOPT="-l"
    if [ "$PLUGIN_DEV_FULL_LOG" -gt 1 ]; then
        set -o xtrace
    fi
fi

export PLUGIN_BAM_FILE=`echo "$TSP_FILEPATH_BAM" | sed -e 's/^.*\///'`
export PLUGIN_BAM_NAME=`echo $PLUGIN_BAM_FILE | sed -e 's/\.[^.]*$//'`
export PLUGIN_RUN_NAME="$TSP_RUN_NAME"

export PLUGIN_HS_ALIGN_DIR="${TSP_FILEPATH_PLUGIN_DIR}/hs_align"
export PLUGIN_HS_ALIGN_BED="${PLUGIN_HS_ALIGN_DIR}/hs_align.bed"

export PLUGIN_OUT_READ_STATS="read_stats.txt"
export PLUGIN_OUT_TARGET_STATS="on_target_stats.txt"
export PLUGIN_OUT_LOCI_STATS="on_loci_stats.txt"

export PLUGIN_OUT_COV_RAW="allele_counts.txt"
export PLUGIN_OUT_COV="allele_counts.xls"

export PLUGIN_OUT_COVERAGE_HTML="COVERAGE_html"
export PLUGIN_OUT_FILELINKS="filelinks.xls"

export PLUGIN_CHROM_X_TARGETS="${TSP_FILEPATH_PLUGIN_DIR}/CHROMX.bed"
export PLUGIN_CHROM_Y_TARGETS="${TSP_FILEPATH_PLUGIN_DIR}/CHROMY.bed"
export PLUGIN_CHROM_NO_Y_TARGETS="${TSP_FILEPATH_PLUGIN_DIR}/TARGETNOY.bed"

export BARCODES_LIST="${TSP_FILEPATH_PLUGIN_DIR}/barcodeList.txt"
export SCRIPTSDIR="${DIRNAME}/scripts"
export BINDIR="${DIRNAME}/bin"
export JSON_RESULTS="${TSP_FILEPATH_PLUGIN_DIR}/results.json"
export HTML_RESULTS="${PLUGINNAME}.html"
export HTML_BLOCK="${PLUGINNAME}_block.html"
export HTML_ROWSUMS="${PLUGINNAME}_rowsum"
export HTML_TORRENT_WRAPPER=1

export PLUGIN_OUT_HAPLOCODE="${TSP_FILEPATH_PLUGIN_DIR}/haplocode"

INPUT_BED_FILE="${DIRNAME}/${INPUT_BED_FILE}"
INPUT_SNP_BED_FILE="${DIRNAME}/${INPUT_SNP_BED_FILE}"

# These are only used to define HTML layouts (handled below)
LIFEGRIDDIR="${DIRNAME}/lifegrid"
BC_SUM_ROWS=5
COV_PAGE_WIDTH="700px"
BC_TITLE_INFO="Sample ID sequence and marker coverage summary statistics for barcoded aligned reads."
BC_COL_TITLE[0]="Sample ID"
BC_COL_TITLE[1]="Reads On-Target"
BC_COL_TITLE[2]="Read Depth"
BC_COL_TITLE[3]="20x Coverage"
BC_COL_TITLE[4]="100x Coverage"
BC_COL_HELP[0]="Sample identification code, based on gender and homozygous or heterozygous calls made for reads at target loci."
BC_COL_HELP[1]="Number of mapped reads that were aligned over gender and sample identification regions."
BC_COL_HELP[2]="Average target base read depth over all sample identification loci."
BC_COL_HELP[3]="Percentage of all sample identification loci that were covered to at least 20x read depth."
BC_COL_HELP[4]="Percentage of all sample identification loci that were covered to at least 100x read depth."

# Flag for extra tables columns needed for hotspots, etc.
BC_HAVE_LOCI=0
if [ -n "$INPUT_SNP_BED_FILE" ]; then
  BC_HAVE_LOCI=1
fi

# Set by barcode iterator if there is a failure of any single barcode
BC_ERROR=0

# Source the HTML files for shell functions and define others below
for HTML_FILE in `find ${DIRNAME}/html/ | grep .sh$`
do
    source ${HTML_FILE};
done

# Creates the body of the detailed report post-analysis
write_html_results ()
{
    local RUNID=${1}
    local OUTDIR=${2}
    local OUTURL=${3}
    local BAMFILE=${4}

    # Create softlink to js/css folders and php scripts
    run "ln -sf \"${DIRNAME}/slickgrid\" \"${OUTDIR}/\"";
    run "ln -sf \"${DIRNAME}/lifegrid\" \"${OUTDIR}/\"";
    run "ln -sf ${DIRNAME}/scripts/*.php3 \"${OUTDIR}/\"";

    # Link bam/bed files from plugin dir and create local URL names for fileLinks table
    PLUGIN_OUT_BAMFILE=`echo "$BAMFILE" | sed -e 's/^.*\///'`
    PLUGIN_OUT_BAIFILE="${PLUGIN_OUT_BAMFILE}.bai"
    if [ -n "$BAMFILE" ]; then
        # create hard links if this was a combineAlignment file - shouldn't be barcodes!
        if [ -n "$PLUGINCONFIG__MERGEDBAM" ]; then
            run "ln -f ${BAMFILE} ${OUTDIR}/$PLUGIN_OUT_BAMFILE"
            run "ln -f ${BAMFILE}.bai ${OUTDIR}/$PLUGIN_OUT_BAIFILE"
        else
            run "ln -sf ${BAMFILE} ${OUTDIR}/$PLUGIN_OUT_BAMFILE"
            run "ln -sf ${BAMFILE}.bai ${OUTDIR}/$PLUGIN_OUT_BAIFILE"
        fi
    fi
    if [ "$OUTDIR" != "$TSP_FILEPATH_PLUGIN_DIR" ]; then
        if [ -n "$INPUT_BED_FILE" ]; then
            run "ln -sf ${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_BEDFILE ${OUTDIR}/$PLUGIN_OUT_BEDFILE"
        fi
        if [ -n "$INPUT_SNP_BED_FILE" ]; then
            run "ln -sf ${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE ${OUTDIR}/$PLUGIN_OUT_LOCI_BEDFILE"
        fi
    fi

    # Create the html report page
    echo "Generating html report..." >&2
    local HTMLOUT="${OUTDIR}/${HTML_RESULTS}";
    write_page_header "$LIFEGRIDDIR/SNPID.head.html" "$HTMLOUT";
    cat "${OUTDIR}/$PLUGIN_OUT_COVERAGE_HTML" >> "$HTMLOUT"
    echo "<div id=\"sampleIDalleleCoverageTable\" fileurl=\"${PLUGIN_OUT_COV}\" class=\"center\"></div>" >> "$HTMLOUT"

    # Create partial html report (w/o file liinks section) and convert to PDF
    local HTMLTMP="${HTMLOUT}_.html"
    awk '$0!~/fileLinksTable/ {print}' "$HTMLOUT" > "$HTMLTMP"
    echo '</div></body></html>' >> "$HTMLTMP"
    PLUGIN_OUT_PDFFILE=`echo "$PLUGIN_OUT_BAMFILE" | sed -e 's/\.[^.]*$//'`
    PLUGIN_OUT_PDFFILE="${PLUGIN_OUT_PDFFILE}.${PLUGINNAME}.pdf"
    local PDFCMD="${BINDIR}/wkhtmltopdf-amd64 --load-error-handling ignore --no-background \"$HTMLTMP\" \"${OUTDIR}/$PLUGIN_OUT_PDFFILE\""
    eval "$PDFCMD >& /dev/null" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nWarning: No PDF view created. Command failed:" >&2
      echo "\$ $PDFCMD" >&2
    fi
    rm -f "$HTMLTMP"

    # Add in the full files links to the report
    write_file_links "$OUTDIR" "$PLUGIN_OUT_FILELINKS" >> "$HTMLOUT";
    echo "<div id=\"fileLinksTable\" fileurl=\"${PLUGIN_OUT_FILELINKS}\" class=\"center\"></div>" >> "$HTMLOUT"
    write_page_footer "$HTMLOUT";

    # Remove temporary files
    rm -f "${OUTDIR}/$PLUGIN_OUT_COVERAGE_HTML"
    return 0
}

# Local copy of sorted barcode list file
PLUGIN_CHECKBC=0
if [ -f $TSP_FILEPATH_BARCODE_TXT ]; then
   PLUGIN_CHECKBC=1
   run "sort -t ' ' -k 2n,2 \"$TSP_FILEPATH_BARCODE_TXT\" > \"$BARCODES_LIST\"";
fi
export PLUGIN_CHECKBC

# Remove previous results to avoid displaying old before ready (some of these prempt TVC calling in place)
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}" "${TSP_FILEPATH_PLUGIN_DIR}/$HTML_BLOCK" "$JSON_RESULTS" ${TSP_FILEPATH_PLUGIN_DIR}/*.bed
rm -rf ${TSP_FILEPATH_PLUGIN_DIR}/*.bam* ${TSP_FILEPATH_PLUGIN_DIR}/dibayes* ${PLUGIN_HS_ALIGN_DIR}
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/hotspot* ${TSP_FILEPATH_PLUGIN_DIR}/variant* ${TSP_FILEPATH_PLUGIN_DIR}/allele*
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*_stats.txt ${TSP_FILEPATH_PLUGIN_DIR}/*.xls ${TSP_FILEPATH_PLUGIN_DIR}/*.log

# Link bed files locally - assumed to static to plugin
export PLUGIN_OUT_BEDFILE=`echo "$INPUT_BED_FILE" | sed -e 's/^.*\///'`
export PLUGIN_OUT_LOCI_BEDFILE=`echo "$INPUT_SNP_BED_FILE" | sed -e 's/^.*\///'`
if [ -n "$INPUT_BED_FILE" ]; then
    run "ln -sf ${INPUT_BED_FILE} ${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_BEDFILE"
fi
if [ -n "$INPUT_SNP_BED_FILE" ]; then
    run "ln -sf ${INPUT_SNP_BED_FILE} ${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE"
fi

echo -e "\nResults folder initialized.\n" >&2

# Process HotSpot BED file for left-alignment
if [ -n "$INPUT_SNP_BED_FILE" -a "$ENABLE_HOTSPOT_LEFT_ALIGNMENT" -eq 1 ]; then
    echo "Ensuring left-alignment of HotSpot regions..." >&2
    run "mkdir -p ${PLUGIN_HS_ALIGN_DIR}"
    run "ln -sf ${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE ${PLUGIN_HS_ALIGN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE"
    ALBCMD="java -jar -Xmx1500m ${DIRNAME}/LeftAlignBed.jar ${PLUGIN_HS_ALIGN_DIR}/${PLUGIN_OUT_LOCI_BEDFILE} ${PLUGIN_HS_ALIGN_BED} ${DIRNAME}/GenomeAnalysisTK.jar ${REFERENCE}"
    if [ "$PLUGIN_DEV_FULL_LOG" -gt 0 ]; then
        echo "\$ $ALBCMD > ${PLUGIN_HS_ALIGN_DIR}/LeftAlignBed.log 2>&1" >&2
    fi
    # NOTE: if java fails due to lack of virtual memory, this error is not trapped by eval
    eval "$ALBCMD > ${PLUGIN_HS_ALIGN_DIR}/LeftAlignBed.log 2>&1" >&2
    grep "Skipped (invalid) records:" ${PLUGIN_HS_ALIGN_DIR}/LeftAlignBed.log >&2
    if [ $? -ne 0 -o ! -f "$PLUGIN_HS_ALIGN_BED" ]; then
        echo "WARNING: Left alignment of HotSpot BED file failed:" >&2
        echo "\$ $ALBCMD > ${PLUGIN_HS_ALIGN_DIR}/LeftAlignBed.log 2>&1" >&2
        echo " - Continuing with original HotSpot BED file..." >&2
        PLUGIN_HS_ALIGN_BED="${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE"
    fi
    echo "" >&2
else
    PLUGIN_HS_ALIGN_BED="${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE"
fi

# Create temporary BED file containing X/Y identification targets
if [ -n "$INPUT_BED_FILE" ]; then
    awk '$1=="chrX" {print}' "$INPUT_BED_FILE" > "$PLUGIN_CHROM_X_TARGETS"
    awk '$1=="chrY" {print}' "$INPUT_BED_FILE" > "$PLUGIN_CHROM_Y_TARGETS"
    awk '$1!="chrY" {print}' "$INPUT_BED_FILE" > "$PLUGIN_CHROM_NO_Y_TARGETS"
fi

# Make links to js/css used for barcodes table and empty results page
run "ln -sf \"${DIRNAME}/js\" \"${TSP_FILEPATH_PLUGIN_DIR}/\"";
run "ln -sf \"${DIRNAME}/css\" \"${TSP_FILEPATH_PLUGIN_DIR}/\"";

# Run for barcodes or single page
if [ $PLUGIN_CHECKBC -eq 1 ]; then
    barcode;
else
    # Write a front page for non-barcode run
    HTML="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
    write_html_header "$HTML" 15;
    echo "<h3><center>${PLUGIN_RUN_NAME}</center></h3>" >> "$HTML"
    display_static_progress "$HTML";
    write_html_footer "$HTML";
    # Perform the analysis
    RT=0
    eval "${SCRIPTSDIR}/call_variants.sh \"${PLUGIN_RUN_NAME}\" \"$TSP_FILEPATH_PLUGIN_DIR\" . \"$TSP_FILEPATH_BAM\"" >&2 || RT=$?
    if [ $RT -ne 0 ]; then
        write_html_header "$HTML";
        echo "<h3><center>${PLUGIN_RUN_NAME}</center></h3>" >> "$HTML"
        echo "<br/><h3 style=\"text-align:center;color:red\">*** An error occurred - check Log File for details ***</h3><br/>" >> "$HTML"
        write_html_footer "$HTML";
        exit 1
    fi
    # Collect results for detail html report and clean up
    write_html_results "$PLUGIN_RUN_NAME" "$TSP_FILEPATH_PLUGIN_DIR" "." "$TSP_FILEPATH_BAM"
    # Write json output
    write_json_header 0;
    write_json_inner "$TSP_FILEPATH_PLUGIN_DIR" "$PLUGIN_OUT_READ_STATS" "mapped_reads" 2;
    echo "," >> "$JSON_RESULTS"
    write_json_inner "$TSP_FILEPATH_PLUGIN_DIR" "$PLUGIN_OUT_TARGET_STATS" "target_coverage" 2;
    if [ $BC_HAVE_LOCI -ne 0 ];then
        echo "," >> "$JSON_RESULTS"
        write_json_inner "$TSP_FILEPATH_PLUGIN_DIR" "$PLUGIN_OUT_LOCI_STATS" "hotspot_coverage" 2;
    fi
    write_json_footer;
    if [ "$PLUGIN_DEV_KEEP_INTERMEDIATE_FILES" -eq 0 ]; then
        rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*_stats.txt "${TSP_FILEPATH_PLUGIN_DIR}/$HTML_ROWSUMS"
    fi
fi

# remove temporary files after successful completion
rm -rf ${PLUGIN_HS_ALIGN_DIR}
rm -f "$BARCODES_LIST" "$PLUGIN_CHROM_X_TARGETS" "$PLUGIN_CHROM_Y_TARGETS" "$PLUGIN_CHROM_NO_Y_TARGETS" "$PLUGIN_OUT_HAPLOCODE"

