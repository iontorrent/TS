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
    local PLUGIN_OUT_TRIMPBAM=`echo "$PLUGIN_OUT_BAMFILE" | sed -e 's/\.[^.]*$//'`
    local PLUGIN_OUT_TRIMPBAM="${PLUGIN_OUT_TRIMPBAM}_PTRIM.bam"
    local PLUGIN_OUT_USTARTSSAM="${PLUGIN_OUT_BAMFILE}_USTARTS.sam"
    local PLUGIN_OUT_USTARTSBAM="${PLUGIN_OUT_BAMFILE}_USTARTS.bam"

    local VCLOGOUT="> \"${OUTDIR}/$PLUGIN_OUT_VCTRACE\""
    local VCWRNOUT="-W \"${OUTDIR}/$PLUGIN_OUT_VCWARN\""
    local USOPT=""
    local BAM_UNTRIM=""
    local PADBED="${OUTDIR}/paddedTargets_${INPUT_TARGET_PADDING}.bed"
    if [ "$PLUGIN_DEV_SKIP_VARIANT_CALLING" -gt 0 ]; then
        echo "Skipping calling variants on mapped reads..." >&2
    else
        if [ -n "$INPUT_BED_FILE" ]; then
            if [ "$INPUT_TARGET_PADDING" -gt 0 ]; then
                echo "Padding merged target regions..." >&2
                run "${SCRIPTSDIR}/padbed.sh ${INPUT_BED_FILE} ${REFERENCE_FAI} $INPUT_TARGET_PADDING ${PADBED}"
                INPUT_BED_MERGE="$PADBED"
            fi
            if [ "$INPUT_USE_USTARTS" = "Yes" ]; then
                echo "Filtering reads to unique starts..." >&2
                run "${SCRIPTSDIR}/remove_pgm_duplicates.pl $LOGOPT -u \"$BAMFILE\" > \"$PLUGIN_OUT_USTARTSSAM\""
                run "samtools view -S -b -t \"$REFERENCE_FAI\" -o \"$PLUGIN_OUT_USTARTSBAM\" \"$PLUGIN_OUT_USTARTSSAM\" &> /dev/null"
                run "samtools index \"$PLUGIN_OUT_USTARTSBAM\""
                rm -f "$PLUGIN_OUT_USTARTSSAM"
                USOPT="-U \"$BAMFILE\""
                BAMFILE="${OUTDIR}/${PLUGIN_OUT_USTARTSBAM}"
            fi
            if [ "$INPUT_TRIM_READS" = "Yes" ]; then
                echo "Trimming reads to targets..." >&2
                BAM_UNTRIM="$BAMFILE"
                BAMFILE="${OUTDIR}/$PLUGIN_OUT_TRIMPBAM"
                run "java -cp ${DIRNAME}/TRIMP_lib -jar ${DIRNAME}/TRIMP.jar $BAM_UNTRIM $BAMFILE $REFERENCE $INPUT_BED_FILE"
		run "samtools index $BAMFILE"
                if [ -f "$BAMFILE" ]; then
                    if [ "$PLUGIN_DEV_FULL_LOG" -gt 0 ]; then
                        echo "> $BAMFILE" >&2
                    fi
                else
                    echo "WARNING: Read trimming failed. Proceeding with pre-trimmed BAM file." >&2
                    BAMFILE="$BAM_UNTRIM"
                fi
            fi
        fi
        echo "Calling variants on mapped reads..." >&2
        if [ -n "$INPUT_BED_FILE" ]; then
            run "${DIRNAME}/variantCaller.py $LOGOPT $VCWRNOUT -o $TSP_FLOWORDER -p \"$INPUT_VC_PARAMFILE\" -r \"$DIRNAME\" -b \"$INPUT_BED_MERGE\" \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\" $VCLOGOUT" 
        else
            run "${DIRNAME}/variantCaller.py $LOGOPT $VCWRNOUT -o $TSP_FLOWORDER -p \"$INPUT_VC_PARAMFILE\" -r \"$DIRNAME\" \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\" $VCLOGOUT" 
        fi
        if [ ! -f "${OUTDIR}/$PLUGIN_OUT_INDELS_VCF" ]; then
            touch "${OUTDIR}/$PLUGIN_OUT_INDELS_VCF"
        fi
    fi

    # Create xml template required for adding IGV links
    run "\"${DIRNAME}/scripts/create_igv_link.py\" -r ${OUTDIR} -b ${PLUGIN_OUT_BAMFILE} -v ${PLUGIN_OUT_SNPS_VCF}.gz -V ${PLUGIN_OUT_INDELS_VCF}.gz -g ${TSP_LIBRARY} -s igv_session.xml"

    # Create table files from vcf - both or either BED file may be ""
    local ALLVAR_TAB="${OUTDIR}/$PLUGIN_OUT_ALLVARS"
    run "\"${DIRNAME}/parse_variants_dibayes.py\" \"${OUTDIR}/$PLUGIN_OUT_SNPS_VCF\" \"${OUTDIR}/$PLUGIN_OUT_SNPS\" \"$INPUT_BED_FILE\"";
    run "\"${DIRNAME}/parse_variants_indels.py\"  \"${OUTDIR}/$PLUGIN_OUT_INDELS_VCF\" \"${OUTDIR}/$PLUGIN_OUT_INDELS\" \"$INPUT_BED_FILE\"";
    if [ -n "$INPUT_SNP_BED_FILE" ]; then
        run "\"$SCRIPTSDIR/collate_variants.pl\" $LOGOPT -G \"$REFERENCE_FAI\" -B \"$PLUGIN_HS_ALIGN_BED\" \"$ALLVAR_TAB\" \"${OUTDIR}/$PLUGIN_OUT_SNPS\" \"${OUTDIR}/$PLUGIN_OUT_INDELS\"";
    else
        run "\"$SCRIPTSDIR/collate_variants.pl\" $LOGOPT -G \"$REFERENCE_FAI\" \"$ALLVAR_TAB\" \"${OUTDIR}/$PLUGIN_OUT_SNPS\" \"${OUTDIR}/$PLUGIN_OUT_INDELS\"";
    fi

    # Filter SNPs/INDELs for hotspots (statistics)
    local HSOPT=""
    local LOCI_CHRVARS_TAB="${OUTDIR}/$PLUGIN_OUT_LOCI_CHRVARS"
    if [ -n "$INPUT_SNP_BED_FILE" ]; then
        HSOPT="-v"
        echo "Collecting HotSpot variants..." >&2
        if [ -f "${OUTDIR}/${PLUGIN_OUT_SNPS_VCF}" ]; then
            run "${VCFTOOLS} --vcf ${OUTDIR}/${PLUGIN_OUT_SNPS_VCF} --bed ${PLUGIN_HS_ALIGN_BED} --out ${OUTDIR}/hotspot.snps --recode --keep-INFO-all > /dev/null";
        fi
        if [ -f "${OUTDIR}/hotspot.snps.recode.vcf" ]; then
            run "${DIRNAME}/vcf-sort ${OUTDIR}/hotspot.snps.recode.vcf > ${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS_VCF";
            if [ "$PLUGIN_DEV_KEEP_INTERMEDIATE_FILES" -eq 0 ]; then
                rm -f "${OUTDIR}/hotspot.snps.recode.vcf"
            fi
        else
            touch "${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS_VCF"
        fi
        if [ -f "${OUTDIR}/${PLUGIN_OUT_INDELS_VCF}" ]; then
            run "${VCFTOOLS} --vcf ${OUTDIR}/${PLUGIN_OUT_INDELS_VCF} --bed ${PLUGIN_HS_ALIGN_BED} --out ${OUTDIR}/hotspot.indels --recode --keep-INFO-all > /dev/null";
        fi
        if [ -f "${OUTDIR}/hotspot.indels.recode.vcf" ]; then
            run "${DIRNAME}/vcf-sort ${OUTDIR}/hotspot.indels.recode.vcf > ${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS_VCF";
            if [ "$PLUGIN_DEV_KEEP_INTERMEDIATE_FILES" -eq 0 ]; then
              rm -f "${OUTDIR}/hotspot.indels.recode.vcf"
            fi
        else
            touch "${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS_VCF"
        fi
        # temporary tables of hotspot SNPs and INDELs created just for summary report
        run "\"${DIRNAME}/parse_variants_dibayes.py\" \"${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS_VCF\" \"${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS\" \"$PLUGIN_HS_ALIGN_BED\"";
        run "\"${DIRNAME}/parse_variants_indels.py\"  \"${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS_VCF\" \"${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS\" \"$PLUGIN_HS_ALIGN_BED\"";
        run "\"$SCRIPTSDIR/snps_summary_report.pl\" $LOGOPT \"$LOCI_CHRVARS_TAB\" \"${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS\" \"${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS\"";
        if [ "$PLUGIN_DEV_KEEP_INTERMEDIATE_FILES" -eq 0 ]; then
            rm -f "${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS" "${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS"
            rm -f "${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS_VCF" "${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS_VCF"
        fi
    fi

    # Collect per-chromosome variant stats
    echo "Collecting per-chromosome variants summary..." >&2
    local CHRVARS_TAB="${OUTDIR}/$PLUGIN_OUT_CHRVARS"
    run "\"$SCRIPTSDIR/snps_summary_report.pl\" $LOGOPT $HSOPT -G \"$REFERENCE_FAI\" \"$CHRVARS_TAB\" \"$ALLVAR_TAB\"";

    # Generate allele counts if hotspots loci BED provided
    if [ -n "$INPUT_SNP_BED_FILE" ]; then
        echo "Generating base pileup for hotspot alleles..." >&2
        run "samtools mpileup -BQ0 -d1000000 -f \"$REFERENCE\" -l ${INPUT_SNP_BED_MERGE} ${BAMFILE} $ERROUT | ${DIRNAME}/allele_count_mpileup_stdin.py > ${OUTDIR}/$PLUGIN_OUT_COV_RAW";
        run "\"${DIRNAME}/print_allele_counts.py\" \"${OUTDIR}/$PLUGIN_OUT_COV_RAW\" \"${OUTDIR}/$PLUGIN_OUT_COV\" \"$PLUGIN_HS_ALIGN_BED\"";
    fi

    # Generate coverage statistics
    echo "Generating coverage statistics..." >&2
    run "${SCRIPTSDIR}/coverage_analysis.sh $LOGOPT $USOPT -O \"$PLUGIN_OUT_TARGET_STATS\" -R \"$PLUGIN_OUT_READ_STATS\" -B \"$INPUT_BED_MERGE\" -V \"$CHRVARS_TAB\" -D \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\""
    if [ -n "$INPUT_SNP_BED_MERGE" ]; then
        run "${SCRIPTSDIR}/coverage_analysis.sh $LOGOPT -O \"$PLUGIN_OUT_LOCI_STATS\" -B \"$INPUT_SNP_BED_MERGE\" -V \"$LOCI_CHRVARS_TAB\" -D \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\""
    fi
    local COVERAGE_HTML="${OUTDIR}/$PLUGIN_OUT_COVERAGE_HTML"
    if [ -n "$INPUT_SNP_BED_MERGE" ]; then
        run "${SCRIPTSDIR}/coverage_analysis_report.pl -t \"$RUNID\" -R \"${OUTDIR}/$PLUGIN_OUT_READ_STATS\" -S \"${OUTDIR}/$PLUGIN_OUT_LOCI_STATS\" -T \"${OUTDIR}/$HTML_ROWSUMS\" \"$COVERAGE_HTML\" \"${OUTDIR}/$PLUGIN_OUT_TARGET_STATS\""
    else
        run "${SCRIPTSDIR}/coverage_analysis_report.pl -t \"$RUNID\" -R \"${OUTDIR}/$PLUGIN_OUT_READ_STATS\" -T \"${OUTDIR}/$HTML_ROWSUMS\" \"$COVERAGE_HTML\" \"${OUTDIR}/$PLUGIN_OUT_TARGET_STATS\""
    fi

    if [ "$PLUGIN_DEV_KEEP_INTERMEDIATE_FILES" -eq 0 ]; then
        rm -f "${OUTDIR}/$PLUGIN_OUT_SNPS" "${OUTDIR}/$PLUGIN_OUT_INDELS" "$PADBED"
    fi
}

vc_main $1 $2 $3 $4;

