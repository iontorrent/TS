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

    # Generate coverage statistics
    echo "Generating coverage statistics and sample identification calls..." >&2
    run "${SCRIPTSDIR}/coverage_analysis.sh $LOGOPT -O \"$PLUGIN_OUT_TARGET_STATS\" -R \"$PLUGIN_OUT_READ_STATS\" -B \"$INPUT_BED_FILE\" -V \"$CHRVARS_TAB\" -D \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\""

    run "${SCRIPTSDIR}/coverage_analysis.sh $LOGOPT -O \"$PLUGIN_OUT_LOCI_STATS\" -B \"$INPUT_SNP_BED_FILE\" -V \"$LOCI_CHRVARS_TAB\" -D \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\""

    local HAPLOCODE=`${SCRIPTSDIR}/extractBarcode.pl -R "${OUTDIR}/$PLUGIN_OUT_READ_STATS" "${OUTDIR}/$PLUGIN_OUT_COV"`
    local COVERAGE_HTML="${OUTDIR}/$PLUGIN_OUT_COVERAGE_HTML"
    run "${SCRIPTSDIR}/coverage_analysis_report.pl -t \"$RUNID\" -B "$HAPLOCODE" -R \"${OUTDIR}/$PLUGIN_OUT_READ_STATS\" -S \"${OUTDIR}/$PLUGIN_OUT_LOCI_STATS\" -T \"${OUTDIR}/$HTML_ROWSUMS\" \"$COVERAGE_HTML\" \"${OUTDIR}/$PLUGIN_OUT_TARGET_STATS\""
    echo "  \"SampleID\" : \"$HAPLOCODE\"," > "$PLUGIN_OUT_HAPLOCODE"
}

vc_main $1 $2 $3 $4;

