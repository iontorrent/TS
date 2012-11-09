#!/bin/bash -E
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

vc_main ()
{
    # check arguments
    if [ -z ${1} ]; then
        failure "Error: Arg 1: Input run name missing";
    elif [ -z ${2} ]; then
        failure "Error: Arg 2: Results dir missing";
    elif [ -z ${3} ]; then
        failure "Error: Arg 3: Results URL missing";
    elif [ -z ${4} ]; then
        failure "Error: Arg 4: Input flowspace BAM missing";
    fi
    local RUNID=${1}
    local OUTDIR=${2}
    local OUTURL=${3}
    local BAMFILE=${4}
    
    local PLUGIN_OUT_BAMFILE=`echo "$BAMFILE" | sed -e 's/^.*\///'`

    # Create xml template required for adding IGV links
    run "\"${DIRNAME}/scripts/create_igv_link.py\" -r ${OUTDIR} -b ${PLUGIN_OUT_BAMFILE} -g ${TSP_LIBRARY} -s igv_session.xml"

    # Generate allele counts if hotspots loci BED provided
    echo "Generating base pileup for SNP loci..." >&2
    run "samtools mpileup -BQ0 -d1000000 -f \"$REFERENCE\" -l ${INPUT_SNP_BED_FILE} ${BAMFILE} $ERROUT | ${SCRIPTSDIR}/allele_from_mpileup.py > ${OUTDIR}/$PLUGIN_OUT_COV_RAW";
    run "\"${SCRIPTSDIR}/writeAlleles.py\" \"${OUTDIR}/$PLUGIN_OUT_COV_RAW\" \"${OUTDIR}/$PLUGIN_OUT_COV\" \"$PLUGIN_HS_ALIGN_BED\"";
    rm -rf "${OUTDIR}/$PLUGIN_OUT_COV_RAW"


    # Generate simple coverage statistics, including number of male/female reads
    echo "Generating coverage statistics and sample identification calls..." >&2
    run "${SCRIPTSDIR}/coverage_analysis.sh $LOGOPT -O \"$PLUGIN_OUT_TARGET_STATS\" -R \"$PLUGIN_OUT_READ_STATS\" -B \"$INPUT_BED_FILE\" -V \"$CHRVARS_TAB\" -D \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\""
    run "${SCRIPTSDIR}/coverage_analysis.sh $LOGOPT -O \"$PLUGIN_OUT_LOCI_STATS\" -B \"$INPUT_SNP_BED_FILE\" -V \"$LOCI_CHRVARS_TAB\" -D \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\""

    # Make the sample ID call string - including gender
    local HAPLOCODE=`${SCRIPTSDIR}/extractBarcode.pl -R "${OUTDIR}/$PLUGIN_OUT_READ_STATS" "${OUTDIR}/$PLUGIN_OUT_COV"`
    local TARGETS_BED="$INPUT_BED_FILE"
    if [[ "$HAPLOCODE" =~ ^F ]]; then
      TARGETS_BED="$PLUGIN_CHROM_NO_Y_TARGETS"
    fi

    # Basic coverage stats - moved from coverage_analysis.sh since need to know gender call first
    local OUTCMD=">> \"${OUTDIR}/$PLUGIN_OUT_TARGET_STATS\""
    local gnm_size=`awk 'BEGIN {gs = 0} {gs += $3-$2} END {printf "%.0f",gs+0}' "$TARGETS_BED"`
    local COVERAGE_ANALYSIS="samtools depth -b \"$TARGETS_BED\" \"$BAMFILE\" 2> /dev/null | awk -f ${SCRIPTSDIR}/coverage_analysis.awk -v genome=$gnm_size"
    eval "$COVERAGE_ANALYSIS $OUTCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: Command failed:" >&2
      echo "\$ $COVERAGE_ANALYSIS $OUTCMD" >&2
      exit 1;
    fi
    OUTCMD=">> \"${OUTDIR}/$PLUGIN_OUT_LOCI_STATS\""
    gnm_size=`awk 'BEGIN {gs = 0} {gs += $3-$2} END {printf "%.0f",gs+0}' "$INPUT_SNP_BED_FILE"`
    COVERAGE_ANALYSIS="samtools depth -b \"$INPUT_SNP_BED_FILE\" \"$BAMFILE\" 2> /dev/null | awk -f ${SCRIPTSDIR}/coverage_analysis.awk -v genome=$gnm_size"
    eval "$COVERAGE_ANALYSIS $OUTCMD" >&2
    if [ $? -ne 0 ]; then
      echo -e "\nERROR: Command failed:" >&2
      echo "\$ $COVERAGE_ANALYSIS $OUTCMD" >&2
      exit 1;
    fi

    # Create detailed html report and pass sample ID for collection to barcode summary
    local COVERAGE_HTML="${OUTDIR}/$PLUGIN_OUT_COVERAGE_HTML"
    run "${SCRIPTSDIR}/coverage_analysis_report.pl -t \"$RUNID\" -B "$HAPLOCODE" -R \"${OUTDIR}/$PLUGIN_OUT_READ_STATS\" -S \"${OUTDIR}/$PLUGIN_OUT_LOCI_STATS\" -T \"${OUTDIR}/$HTML_ROWSUMS\" \"$COVERAGE_HTML\" \"${OUTDIR}/$PLUGIN_OUT_TARGET_STATS\""
    echo "  \"SampleID\" : \"$HAPLOCODE\"," > "$PLUGIN_OUT_HAPLOCODE"

    # Create block.html for summary (non-barcoded run)
    if [ $PLUGIN_CHECKBC -eq 0 ]; then
      HELPTXT="Sample ID: Sex (M/F) and list of alleles called from base read coverage at the sample identification loci."
      BLOCKHTML="${OUTDIR}/$HTML_BLOCK"
      COVSTAT=`awk '/100x/ {print}' "${OUTDIR}/$PLUGIN_OUT_LOCI_STATS"`
      echo '<?xml version="1.0" encoding="iso-8859-1"?>' > "$BLOCKHTML"
      echo '<!DOCTYPE html>' >> "$BLOCKHTML"
      echo '<html><body>' >> "$BLOCKHTML"
      echo "<h1 title='$HELPTXT' style='cursor:help;text-align:center;color:darkred'>$HAPLOCODE</h1>" >> "$BLOCKHTML"
      HELPTXT="Percentage of marker bases coveraged by at least 100 reads."
      echo "<div title='$HELPTXT' style='cursor:help;text-align:center'>$COVSTAT</div>" >> "${OUTDIR}/$HTML_BLOCK"
      echo '</body></html>' >> "$BLOCKHTML"
    fi
}

vc_main $1 $2 $3 $4;

