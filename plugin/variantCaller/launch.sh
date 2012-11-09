#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved 

#MAJOR_BLOCK 

#as of 6/21 the stack size being set on SGE nodes is too large, setting manually to the default
#ulimit -s 8192
#$ -l mem_free=22G,h_vmem=22G,s_vmem=22G
#normal plugin script
VERSION="3.2.45211"

# DEVELOPMENT/DEBUG options:
# NOTE: the following should all be set to 0 in production mode
export PLUGIN_DEV_KEEP_INTERMEDIATE_FILES=0;   # use prior to PLUGIN_DEV_RECALL_VARIANTS=1 to re-process from temporary results
export PLUGIN_DEV_SKIP_VARIANT_CALLING=0;      # 1 to skip variant calling - use previous calls
export PLUGIN_DEV_FULL_LOG=0;          # 1 for variant calling log, 2 for additional xtrace (not recommended)

export CONTINUE_AFTER_BARCODE_ERROR=1;
export ENABLE_HOTSPOT_LEFT_ALIGNMENT=1;

# Minimum barcode BAM size required for variant calling. 50,000 bytes ~ 100-400 reads.
export BCFILE_MIN_SIZE="50000"

# Check for by-pass PUI
if [ -z "$PLUGINCONFIG__LIBRARYTYPE_ID" ]; then
    OLD_IFS="$IFS"
    IFS=";"
    PLAN_INFO=(`${DIRNAME}/parse_plan.py ${TSP_FILEPATH_PLUGIN_DIR}/startplugin.json`)
    IFS="$OLD_IFS"
    PLUGINCONFIG__LIBRARYTYPE=${PLAN_INFO[0]}
    PLUGINCONFIG__VARIATIONTYPE=${PLAN_INFO[1]}
    PLUGINCONFIG__TARGETREGIONS=${PLAN_INFO[2]}
    PLUGINCONFIG__TARGETLOCI=${PLAN_INFO[3]}
    if [ "$PLUGINCONFIG__LIBRARYTYPE" = "ampliseq" ]; then
        PLUGINCONFIG__LIBRARYTYPE_ID="Ion AmpliSeq"
        PLUGINCONFIG__TRIMREADS="Yes"
        PLUGINCONFIG__UNIQUESTARTS="No"
        PLUGINCONFIG__PADTARGETS=0
    elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "targetseq" ]; then
        PLUGINCONFIG__LIBRARYTYPE_ID="Ion TargetSeq"
        PLUGINCONFIG__TRIMREADS="No"
        PLUGINCONFIG__UNIQUESTARTS="No"
        PLUGINCONFIG__PADTARGETS=0
    elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "fullgenome" ]; then
        PLUGINCONFIG__LIBRARYTYPE_ID="Full Genome"
        PLUGINCONFIG__TRIMREADS="No"
        PLUGINCONFIG__UNIQUESTARTS="No"
        PLUGINCONFIG__PADTARGETS=0
    else
        rm -f "${TSP_FILEPATH_PLUGIN_DIR}/results.json"
        HTML="${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
        echo '<html><body>' > "$HTML"
        if [ -f "${DIRNAME}/html/logo.sh" ]; then
            source "${DIRNAME}/html/logo.sh"
            print_html_logo >> "$HTML";
        fi
        echo "<h3><center>${PLUGIN_RUN_NAME}</center></h3>" >> "$HTML"
        echo "<br/><h2 style=\"text-align:center;color:red\">*** Automatic analysis was not performed for PGM run. ***</h2>" >> "$HTML"
        echo "<br/><h3 style=\"text-align:center\">(Runtype '$PLUGINCONFIG__LIBRARYTYPE' is not supported.)</h3></br>" >> "$HTML"
        echo '</body></html>' >> "$HTML"
        exit
    fi
    if [ "$PLUGINCONFIG__VARIATIONTYPE" = "Germ_Line" ]; then
        PLUGINCONFIG__VARIATIONTYPE_ID="Germ Line"
    elif [ "$PLUGINCONFIG__VARIATIONTYPE" = "Somatic" ]; then
        PLUGINCONFIG__VARIATIONTYPE_ID="Somatic"
    else
        echo "ERROR: Unexpected Variation Frequency: $PLUGINCONFIG__VARIATIONTYPE" >&2
        exit 1
    fi
    # Correct for changes to default regions values
    if [ "$PLUGINCONFIG__TARGETREGIONS" = "none" ]; then
        PLUGINCONFIG__TARGETREGIONS=""
    fi
    if [ "$PLUGINCONFIG__TARGETLOCI" = "none" ]; then
        PLUGINCONFIG__TARGETLOCI=""
    fi
    if [ -z "$PLUGINCONFIG__TARGETREGIONS" ]; then
        PLUGINCONFIG__TARGETREGIONS_ID=""
        PLUGINCONFIG__TARGETREGIONS_MERGE=""
    else
        PLUGINCONFIG__TARGETREGIONS_ID=`echo "$PLUGINCONFIG__TARGETREGIONS" | sed -e 's/^.*\///' | sed -e 's/\.bed$//'`
        PLUGINCONFIG__TARGETREGIONS_MERGE=`echo "$PLUGINCONFIG__TARGETREGIONS" | sed -e 's/\/unmerged\/detail\//\/merged\/plain\//'`
    fi
    if [ -z "$PLUGINCONFIG__TARGETLOCI" ]; then
        PLUGINCONFIG__TARGETLOCI_ID=""
        PLUGINCONFIG__TARGETLOCI_MERGE=""
    else
        PLUGINCONFIG__TARGETLOCI_ID=`echo "$PLUGINCONFIG__TARGETLOCI" | sed -e 's/^.*\///' | sed -e 's/\.bed$//'`
        PLUGINCONFIG__TARGETLOCI_MERGE=`echo "$PLUGINCONFIG__TARGETLOCI" | sed -e 's/\/unmerged\/detail\//\/merged\/plain\//'`
    fi
else
    # temporary fix for user selections: bug means result only passed up to first space to UNIX VARS
    # - hence convert back underscores to spaces
    PLUGINCONFIG__LIBRARYTYPE_ID=`echo "$PLUGINCONFIG__LIBRARYTYPE_ID" | sed -e 's/_/ /g'`
    PLUGINCONFIG__VARIATIONTYPE_ID=`echo "$PLUGINCONFIG__VARIATIONTYPE_ID" | sed -e 's/_/ /g'`
    PLUGINCONFIG__TARGETREGIONS_ID=`echo "$PLUGINCONFIG__TARGETREGIONS_ID" | sed -e 's/_/ /g'`
    PLUGINCONFIG__TARGETLOCI_ID=`echo "$PLUGINCONFIG__TARGETLOCI_ID" | sed -e 's/_/ /g'`
fi

# Convert library type and variant type to parameter file names
export INPUT_VC_PARAMFILE="${DIRNAME}/paramFiles/${PLUGINCONFIG__LIBRARYTYPE}.${PLUGINCONFIG__VARIATIONTYPE}.json"

# Check for merged BAM file override
SNPSOPT=""
PLUGIN_CHECKBC=1
if [ -n "$PLUGINCONFIG__MERGEDBAM" ]; then
    #SNPSOPT="-s"
    PLUGIN_CHECKBC=0
    TSP_FILEPATH_BAM=$PLUGINCONFIG__MERGEDBAM
else
    PLUGINCONFIG__MERGEDBAM_ID='Current Report'
fi

# Parameters from plugin customization
echo "Selected run options:" >&2
echo "  Aligned Reads:     $PLUGINCONFIG__MERGEDBAM_ID" >&2
echo "  Library Type:      $PLUGINCONFIG__LIBRARYTYPE_ID" >&2
echo "  Variant Detection: $PLUGINCONFIG__VARIATIONTYPE_ID" >&2
echo "  Target Regions:    $PLUGINCONFIG__TARGETREGIONS_ID" >&2
echo "  Target Loci:       $PLUGINCONFIG__TARGETLOCI_ID" >&2

# Check for missing files
if [ ! -f "$INPUT_VC_PARAMFILE" ]; then
    echo "ERROR: Cannot locate variant calling parameters file: ${INPUT_VC_PARAMFILE}" >&2
    exit 1
fi
if [ -n "$PLUGINCONFIG__TARGETREGIONS" ]; then
    if [ ! -f "$PLUGINCONFIG__TARGETREGIONS" ]; then
        echo "ERROR: Cannot locate target regions file: ${PLUGINCONFIG__TARGETREGIONS}" >&2
        exit 1
    fi
    if [ ! -f "$PLUGINCONFIG__TARGETREGIONS_MERGE" ]; then
        echo "ERROR: Cannot locate merged target regions file - assuming target regions file is merged." >&2
        exit 1
    fi
fi
if [ -n "$PLUGINCONFIG__TARGETLOCI" ]; then
    if [ ! -f "$PLUGINCONFIG__TARGETLOCI" ]; then
        echo "ERROR: Cannot locate hotspot regions file: ${PLUGINCONFIG__TARGETLOCI}" >&2
        exit 1
    fi
    if [ ! -f "$PLUGINCONFIG__TARGETLOCI_MERGE" ]; then
        echo "ERROR: Cannot locate merged hotspot regions file - assuming hotspot regions file is merged." >&2
        exit 1
    fi
fi

# Setup environment

export INPUT_BED_FILE="$PLUGINCONFIG__TARGETREGIONS"
export INPUT_SNP_BED_FILE="$PLUGINCONFIG__TARGETLOCI"
export INPUT_BED_MERGE="$PLUGINCONFIG__TARGETREGIONS_MERGE"
export INPUT_SNP_BED_MERGE="$PLUGINCONFIG__TARGETLOCI_MERGE"
export INPUT_TRIM_READS="$PLUGINCONFIG__TRIMREADS"
export INPUT_TARGET_PADDING="$PLUGINCONFIG__PADTARGETS"
export INPUT_USE_USTARTS="$PLUGINCONFIG__UNIQUESTARTS"

if [ -z "$INPUT_TRIM_READS" ]; then
  INPUT_TRIM_READS="No"
fi
if [ -z "$INPUT_USE_USTARTS" ]; then
  INPUT_USE_USTARTS="No"
fi

#export REFERENCE="/results/referenceLibrary/tmap-f2/hg19/hg19.fasta"
export REFERENCE="$TSP_FILEPATH_GENOME_FASTA"
export REFERENCE_FAI="${REFERENCE}.fai"

echo "Employed run options:" >&2
echo "  Reference Genome:  $REFERENCE" >&2
echo "  Aligned Reads:     $TSP_FILEPATH_BAM" >&2
echo "  Library Type:      $PLUGINCONFIG__LIBRARYTYPE" >&2
echo "  Variant Detection: $INPUT_VC_PARAMFILE" >&2
echo "  Target Regions:    $INPUT_BED_FILE" >&2
echo "   - Merged:         $INPUT_BED_MERGE" >&2
echo "  Target Loci:       $INPUT_SNP_BED_FILE" >&2
echo "   - Merged:         $INPUT_SNP_BED_MERGE" >&2
echo "  Trim Reads:        $INPUT_TRIM_READS" >&2
echo "  Target Padding:    $INPUT_TARGET_PADDING" >&2
echo "  Use Unique Starts: $INPUT_USE_USTARTS" >&2

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

# Definition of file names used by plugin
export VCFTOOLS="${DIRNAME}/vcftools";
export LOG4CXX_CONFIGURATION="${DIRNAME}/log4j.properties";

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
export PLUGIN_OUT_SNPS="SNP_variants.xls"
export PLUGIN_OUT_INDELS="indel_variants.xls"
export PLUGIN_OUT_ALLVARS="variants.xls"
export PLUGIN_OUT_CHRVARS="variants_per_chromosome.xls"
export PLUGIN_OUT_FILELINKS="filelinks.xls"

export PLUGIN_OUT_LOCI_SNPS="hotspot_SNP_variants.xls"
export PLUGIN_OUT_LOCI_INDELS="hotspot_indel_variants.xls"
export PLUGIN_OUT_LOCI_SNPS_VCF="hotspot_SNP_variants.vcf"
export PLUGIN_OUT_LOCI_INDELS_VCF="hotspot_indel_variants.vcf"
export PLUGIN_OUT_LOCI_CHRVARS="hotspot_variants_per_chromosome.xls"

export PLUGIN_OUT_COVERAGE_HTML="COVERAGE_html"
export HTML_BLOCK="variantCaller_block.html";

export PLUGIN_OUT_VCTRACE="variantCaller.log"
export PLUGIN_OUT_VCWARN="variantCaller.warn"

export BARCODES_LIST="${TSP_FILEPATH_PLUGIN_DIR}/barcodeList.txt";
export SCRIPTSDIR="${DIRNAME}/scripts"
export JSON_RESULTS="${TSP_FILEPATH_PLUGIN_DIR}/results.json"
export HTML_RESULTS="${PLUGINNAME}.html"
export HTML_ROWSUMS="${PLUGINNAME}_rowsum"

# These are hard-coded in to varinatCaller.py, as are its other output file names
export PLUGIN_OUT_SNPS_VCF="SNP_variants.vcf"
export PLUGIN_OUT_INDELS_VCF="indel_variants.vcf"
export PLUGIN_OUT_MERGED_VCF="TSVC_variants.vcf"

# These are only used to define HTML layouts (handled below)
LIFEGRIDDIR="${DIRNAME}/lifegrid"
BC_SUM_ROWS=8
BC_COV_PAGE_WIDTH=1040
COV_PAGE_WIDTH=1040
BC_COL_TITLE[0]="Mapped Reads"
BC_COL_TITLE[1]="Reads On-Target"
BC_COL_TITLE[2]="Bases On-Target"
BC_COL_TITLE[3]="Read Depth"
BC_COL_TITLE[4]="1x Coverage"
BC_COL_TITLE[5]="20x Coverage"
BC_COL_TITLE[6]="100x Coverage"
BC_COL_TITLE[7]="Variants Detected"
BC_COL_HELP[0]="Number of reads that were mapped to the full reference for this barcode set."
BC_COL_HELP[1]="Percentage of mapped reads that were aligned over a target region."
BC_COL_HELP[2]="Percentage of mapped bases that were aligned over a target region."
BC_COL_HELP[3]="Mean average target base read depth, including non-covered target bases."
BC_COL_HELP[4]="Percentage of all target bases that were covered to at least 1x read depth."
BC_COL_HELP[5]="Percentage of all target bases that were covered to at least 20x read depth."
BC_COL_HELP[6]="Percentage of all target bases that were covered to at least 100x read depth."
BC_COL_HELP[7]="Number of variant calls made."

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
    PLUGIN_OUT_TRIMPBAM=`echo "$PLUGIN_OUT_BAMFILE" | sed -e 's/\.[^.]*$//'`
    PLUGIN_OUT_USTARTSBAM="$PLUGIN_OUT_TRIMPBAM"
    PLUGIN_OUT_TRIMPBAM="${PLUGIN_OUT_TRIMPBAM}_PTRIM.bam"
    PLUGIN_OUT_TRIMPBAI="${PLUGIN_OUT_TRIMPBAM}.bai"
    PLUGIN_OUT_USTARTSBAM="${PLUGIN_OUT_USTARTSBAM}_USTARTS.bam"
    PLUGIN_OUT_USTARTSBAI="${PLUGIN_OUT_USTARTSBAM}.bai"
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
    write_page_header "$LIFEGRIDDIR/TVC.head.html" "$HTMLOUT";
    cat "${OUTDIR}/$PLUGIN_OUT_COVERAGE_HTML" >> "$HTMLOUT"
    echo "<div id=\"variantCallsSummaryTable\" fileurl=\"${PLUGIN_OUT_CHRVARS}\" class=\"center\"></div><br/>" >> "$HTMLOUT"
    echo "<div id=\"variantCallsTable\" fileurl=\"${PLUGIN_OUT_ALLVARS}\" class=\"center\"></div><br/>" >> "$HTMLOUT"
    # output varinat caller warning messages, if any
    local KEEP_TMP_FILES="$PLUGIN_DEV_KEEP_INTERMEDIATE_FILES"
    if [ -f "${OUTDIR}/$PLUGIN_OUT_VCWARN" ];then
        echo "<h3 style=\"text-align:center;color:red\">Warning: " >> "$HTMLOUT"
        cat "${OUTDIR}/$PLUGIN_OUT_VCWARN" >> "$HTMLOUT"
        echo "</h3><br/>" >> "$HTMLOUT"
        KEEP_TMP_FILES=1
    fi
    # no alleles table if hotspots not used
    if [ -n "$INPUT_SNP_BED_FILE" ]; then
        echo "<div id=\"alleleCoverageTable\" fileurl=\"${PLUGIN_OUT_COV}\" class=\"center\"></div>" >> "$HTMLOUT"
    fi
    write_file_links "$OUTDIR" "$PLUGIN_OUT_FILELINKS" >> "$HTMLOUT";
    echo "<div id=\"fileLinksTable\" fileurl=\"${PLUGIN_OUT_FILELINKS}\" class=\"center\"></div>" >> "$HTMLOUT"
    write_page_footer "$HTMLOUT";

    # Remove temporary files
    # rm -f "${OUTDIR}/$PLUGIN_OUT_COVERAGE_HTML"
    if [ "$KEEP_TMP_FILES" -eq 0 ]; then
        rm -f "${OUTDIR}/${PLUGIN_OUT_LOCI_CHRVARS}" "${OUTDIR}/bayesian_scorer.vcf"
        rm -f "${OUTDIR}/downsampled.vcf" "${OUTDIR}/gatk_prefiltered.vcf" "${OUTDIR}/filtered.non-downsampled.vcf"
        rm -f ${OUTDIR}/*.log ${OUTDIR}/*.bak ${OUTDIR}/variantCalls.*
    fi
    return 0
}

# Local copy of sorted barcode list file
if [ ! -f $TSP_FILEPATH_BARCODE_TXT ]; then
   PLUGIN_CHECKBC=0
fi
if [ $PLUGIN_CHECKBC -eq 1 ]; then
   run "sort -t ' ' -k 2n,2 \"$TSP_FILEPATH_BARCODE_TXT\" > \"$BARCODES_LIST\"";
fi

# Remove previous results to avoid displaying old before ready
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}" "$JSON_RESULTS" ${TSP_FILEPATH_PLUGIN_DIR}/*.bed
rm -rf ${TSP_FILEPATH_PLUGIN_DIR}/*.bam* ${TSP_FILEPATH_PLUGIN_DIR}/dibayes* ${PLUGIN_HS_ALIGN_DIR}
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/hotspot* ${TSP_FILEPATH_PLUGIN_DIR}/variant* ${TSP_FILEPATH_PLUGIN_DIR}/allele*
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*_stats.txt ${TSP_FILEPATH_PLUGIN_DIR}/*.xls
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*.warn ${TSP_FILEPATH_PLUGIN_DIR}/*.log
if [ $PLUGIN_DEV_SKIP_VARIANT_CALLING -eq 0 ]; then
   rm -f ${TSP_FILEPATH_PLUGIN_DIR}/SNP* ${TSP_FILEPATH_PLUGIN_DIR}/indel*
   rm -f  ${TSP_FILEPATH_PLUGIN_DIR}/*.vcf
fi

# Get local copy of BED files (may be deleted from system later)
export PLUGIN_OUT_BEDFILE=`echo "$INPUT_BED_FILE" | sed -e 's/^.*\///'`
export PLUGIN_OUT_LOCI_BEDFILE=`echo "$INPUT_SNP_BED_FILE" | sed -e 's/^.*\///'`
if [ -n "$INPUT_BED_FILE" ]; then
    run "cp -f ${INPUT_BED_FILE} ${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_BEDFILE"
fi
if [ -n "$INPUT_SNP_BED_FILE" ]; then
    run "cp -f ${INPUT_SNP_BED_FILE} ${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE"
fi

echo -e "\nResults folder initialized.\n" >&2

# Process HotSpot BED file for left-alignment
if [ -n "$INPUT_SNP_BED_FILE" -a "$ENABLE_HOTSPOT_LEFT_ALIGNMENT" -eq 1 ]; then
    echo "Ensuring left-alignment of HotSpot regions..." >&2
    run "mkdir -p ${PLUGIN_HS_ALIGN_DIR}"
    run "ln -sf ${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE ${PLUGIN_HS_ALIGN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE"
    ALBCMD="java -Xmx1500m -jar ${DIRNAME}/LeftAlignBed.jar ${PLUGIN_HS_ALIGN_DIR}/${PLUGIN_OUT_LOCI_BEDFILE} ${PLUGIN_HS_ALIGN_BED} ${DIRNAME}/TVC/jar/GenomeAnalysisTK.jar ${REFERENCE}"
    if [ "$PLUGIN_DEV_FULL_LOG" -gt 0 ]; then
        echo "\$ $ALBCMD > ${PLUGIN_HS_ALIGN_DIR}/LeftAlignBed.log 2>&2" >&2
    fi
    # NOTE: if java fails due to lack of virtual memory, this error is not trapped by eval
    eval "$ALBCMD > ${PLUGIN_HS_ALIGN_DIR}/LeftAlignBed.log 2>&2" >&2
    grep "Skipped (invalid) records:" ${PLUGIN_HS_ALIGN_DIR}/LeftAlignBed.log >&2
    if [ $? -ne 0 -o ! -f "$PLUGIN_HS_ALIGN_BED" ]; then
        echo "WARNING: Left alignment of HotSpot BED file failed:" >&2
        echo "\$ $ALBCMD > ${PLUGIN_HS_ALIGN_DIR}/LeftAlignBed.log 2>&2" >&2
        echo " - Continuing with original HotSpot BED file..." >&2
        PLUGIN_HS_ALIGN_BED="${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE"
    fi
    echo "" >&2
else
    PLUGIN_HS_ALIGN_BED="${TSP_FILEPATH_PLUGIN_DIR}/$PLUGIN_OUT_LOCI_BEDFILE"
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
rm -f "$BARCODES_LIST" "${TSP_FILEPATH_PLUGIN_DIR}/startplugin.json"
rm -rf ${PLUGIN_HS_ALIGN_DIR}
