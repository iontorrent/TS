#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#as of 6/21 the stack size being set on SGE nodes is too large, setting manually to the default
ulimit -s 8192
#$ -l mem_free=22G,h_vmem=22G,s_vmem=22G
#normal plugin script
VERSION="2.0.1.1"

# DEVELOPMENT/DEBUG options:
# NOTE: the following should all be set to 0 in production mode
PLUGIN_DEV_KEEP_INTERMEDIATE_FILES=0;   # use prior to PLUGIN_DEV_RECALL_VARIANTS=1 to re-process from temporary results
PLUGIN_DEV_SKIP_VARIANT_CALLING=0;      # 1 to skip variant calling - use previous calls
PLUGIN_DEV_FULL_LOG=0;          # 1 for variant calling log, 2 for additional xtrace (not recommended)

ENABLE_HOTSPOT_LEFT_ALIGNMENT=1;
ENABLE_PRIMER_TRIMMING_IN_AMPLISEQ=1;

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
    if [ -z "$PLUGINCONFIG__LIBRARYTYPE" ]; then
        rm -f "${TSP_FILEPATH_PLUGIN_DIR}/results.json"
        HTML="${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
        echo '<html><body>' > "$HTML"
        if [ -f "${DIRNAME}/html/logo.sh" ]; then
            source "${DIRNAME}/html/logo.sh"
            print_html_logo >> "$HTML";
        fi
        echo "<h3><center>${PLUGIN_RUNNAME}</center></h3>" >> "$HTML"
        echo "<br/><h2 style=\"text-align:center;color:red\">*** Automatic analysis was not performed for PGM run. ***</h2>" >> "$HTML"
        echo "<br/><h3 style=\"text-align:center\">(Requires an associated Plan that is not a GENS Runtype.)</h3></br>" >> "$HTML"
        echo '</body></html>' >> "$HTML"
        exit
    elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "ampliseq" ]; then
        PLUGINCONFIG__LIBRARYTYPE_ID="Ion AmpliSeq"
    elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "targetseq" ]; then
        PLUGINCONFIG__LIBRARYTYPE_ID="Ion TargetSeq"
    elif [ "$PLUGINCONFIG__LIBRARYTYPE" = "fullgenome" ]; then
        PLUGINCONFIG__LIBRARYTYPE_ID="Full Genome"
    else
        echo "ERROR: Unexpected Library Type: $PLUGINCONFIG__LIBRARYTYPE" >&2
        exit 1
    fi
    if [ "$PLUGINCONFIG__VARIATIONTYPE" = "Germ_Line" ]; then
        PLUGINCONFIG__VARIATIONTYPE_ID="Germ Line"
    elif [ "$PLUGINCONFIG__VARIATIONTYPE" = "Somatic" ]; then
        PLUGINCONFIG__VARIATIONTYPE_ID="Somatic"
    else
        echo "ERROR: Unexpected Variation Frequency: $PLUGINCONFIG__VARIATIONTYPE" >&2
        exit 1
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
INPUT_VC_PARAMFILE="${DIRNAME}/paramFiles/${PLUGINCONFIG__LIBRARYTYPE}.${PLUGINCONFIG__VARIATIONTYPE}.json"

# Parameters from plugin customization
echo "Selected run options:" >&2
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

INPUT_BED_FILE="$PLUGINCONFIG__TARGETREGIONS"
INPUT_SNP_BED_FILE="$PLUGINCONFIG__TARGETLOCI"
INPUT_BED_MERGE="$PLUGINCONFIG__TARGETREGIONS_MERGE"
INPUT_SNP_BED_MERGE="$PLUGINCONFIG__TARGETLOCI_MERGE"

#REFERENCE="/results/referenceLibrary/tmap-f2/hg19/hg19.fasta"
REFERENCE="$TSP_FILEPATH_GENOME_FASTA"
REFERENCE_FAI="${REFERENCE}.fai"

echo "Employed run options:" >&2
echo "  Reference Genome:  $REFERENCE" >&2
echo "  PGM Flow Order:    $TSP_FLOWORDER" >&2
echo "  Library Type:      $PLUGINCONFIG__LIBRARYTYPE" >&2
echo "  Variant Detection: $INPUT_VC_PARAMFILE" >&2
echo "  Target Regions:    $INPUT_BED_FILE" >&2
echo "   - Merged:         $INPUT_BED_MERGE" >&2
echo "  Target Loci:       $INPUT_SNP_BED_FILE" >&2
echo "   - Merged:         $INPUT_SNP_BED_MERGE" >&2

# Definition of file names, etc., used by plugin
VCFTOOLS="${DIRNAME}/vcftools";
LOG4CXX_CONFIGURATION="${DIRNAME}/log4j.properties";
export LOG4CXX_CONFIGURATION;

PLUGIN_BAM_FILE=`echo "$TSP_FILEPATH_BAM" | sed -e 's/^.*\///'`
PLUGIN_BAM_NAME=`echo $PLUGIN_BAM_FILE | sed -e 's/\.[^.]*$//'`
PLUGIN_RUN_NAME="$TSP_RUN_NAME"

PLUGIN_HS_ALIGN_DIR="${TSP_FILEPATH_PLUGIN_DIR}/hs_align"
PLUGIN_HS_ALIGN_BED="${PLUGIN_HS_ALIGN_DIR}/hs_align.bed"

PLUGIN_OUT_READ_STATS="read_stats.txt"
PLUGIN_OUT_TARGET_STATS="on_target_stats.txt"
PLUGIN_OUT_COV_RAW="allele_counts.txt"
PLUGIN_OUT_COV="allele_counts.xls"
PLUGIN_OUT_SNPS="SNP_variants.xls"
PLUGIN_OUT_INDELS="indel_variants.xls"
PLUGIN_OUT_ALLVARS="variants.xls"
PLUGIN_OUT_CHRVARS="variants_per_chromosome.xls"
PLUGIN_OUT_SNPS_VCF="SNP_variants.vcf"
PLUGIN_OUT_INDELS_VCF="indel_variants.vcf"

PLUGIN_OUT_LOCI_STATS="on_loci_stats.txt"
PLUGIN_OUT_LOCI_SNPS="hotspot_SNP_variants.xls"
PLUGIN_OUT_LOCI_INDELS="hotspot_indel_variants.xls"
PLUGIN_OUT_LOCI_SNPS_VCF="hotspot_SNP_variants.vcf"
PLUGIN_OUT_LOCI_INDELS_VCF="hotspot_indel_variants.vcf"
PLUGIN_OUT_LOCI_CHRVARS="hotspot_variants_per_chromosome.xls"

PLUGIN_OUT_VCTRACE="variantCaller.log"

BARCODES_LIST="${TSP_FILEPATH_PLUGIN_DIR}/barcodeList.txt";
SCRIPTSDIR="${DIRNAME}/scripts"
JSON_RESULTS="${TSP_FILEPATH_PLUGIN_DIR}/results.json"
HTML_RESULTS="${PLUGINNAME}.html"
HTML_ROWSUMS="${PLUGINNAME}_rowsum"

# Definition of fields displayed in barcode link/summary table
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

BC_HAVE_LOCI=0
if [ -n "$INPUT_SNP_BED_FILE" ]; then
  BC_HAVE_LOCI=1
fi

# Set up log options
ERROUT="2> /dev/null"
LOGOPT=""
if [ "$PLUGIN_DEV_FULL_LOG" -gt 0 ]; then
    ERROUT=""
    LOGOPT="-l"
    if [ "$PLUGIN_DEV_FULL_LOG" -gt 1 ]; then
        set -o xtrace
    fi
fi

# Source the HTML files for shell functions and define others below
for HTML_FILE in `find ${DIRNAME}/html/ | grep .sh$`
do
    source ${HTML_FILE};
done

#*! @function
#  @param  $*  the command to be executed
run ()
{
    if [ "$PLUGIN_DEV_FULL_LOG" -gt 0 ]; then
        echo "\$ $*" >&2
    fi
    eval $* >&2
    EXIT_CODE="$?"
    if test ${EXIT_CODE} != 0; then
        echo "status code '${EXIT_CODE}' while running '$*'" >&2
        # only minimal cleanup so most log files (for barcode that failed) are retained
        rm -f "$VARS_HTML" "$ALLELES_HTML" "$CHRVARS_HTML" "$JSON_RESULTS"
        # partially produced barcode html might be useful but is left in an auto-update mode
        rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
        exit 1
    fi
}

#*! @function
#  @param $1 Name of JSON file to append to
#  @param $2 Path to file composed of <name>:<value> lines
#  @param $3 block name (e.g. "filtered_reads")
#  @param $4 printing indent level. Default: 2
append_to_json_results ()
{
    local JSONFILE="$1"
    local DATAFILE="$2"
    local DATASET="$3"
    local INDENTLEV="$4"
    if [ -z $INDENTLEV ]; then
        INDENTLEV=2
    fi
    local JSONCMD="${SCRIPTSDIR}/coverage_analysis_json.pl -a -I $INDENTLEV -B \"$DATASET\" \"$DATAFILE\" \"$JSONFILE\""
    eval "$JSONCMD || echo \"WARNING: Failed to write to JSON from $DATAFILE\"" >&2
}

call_variants()
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
    
    PLUGIN_OUT_BAMFILE=`echo "$BAMFILE" | sed -e 's/^.*\///'`
    PLUGIN_OUT_BAIFILE="${PLUGIN_OUT_BAMFILE}.bai"
    PLUGIN_OUT_TRIMPBAM=`echo "$PLUGIN_OUT_BAMFILE" | sed -e 's/\.[^.]*$//'`
    PLUGIN_OUT_TRIMPBAM="${PLUGIN_OUT_TRIMPBAM}_PTRIM.bam"
    PLUGIN_OUT_TRIMPBAI="${PLUGIN_OUT_TRIMPBAM}.bai"

    local VCLOGOUT="> \"${OUTDIR}/$PLUGIN_OUT_VCTRACE\""
    local BAM_UNTRIM=""
    if [ "$PLUGIN_DEV_SKIP_VARIANT_CALLING" -gt 0 ]; then
        echo "Skipping calling variants on mapped reads..." >&2
    else
        echo "Calling variants on mapped reads..." >&2
        if [ -n "$INPUT_BED_FILE" ]; then
            if [ "$PLUGINCONFIG__LIBRARYTYPE" = "ampliseq" -a "$ENABLE_PRIMER_TRIMMING_IN_AMPLISEQ" -eq 1 ]; then
                echo "Trimming primer sequence..." >&2
                BAM_UNTRIM="$BAMFILE"
                BAMFILE="${OUTDIR}/$PLUGIN_OUT_TRIMPBAM"
                run "java -cp ${DIRNAME}/TRIMP_lib -jar ${DIRNAME}/TRIMP.jar $BAM_UNTRIM $BAMFILE $REFERENCE $INPUT_BED_FILE"
		run "samtools index $BAMFILE"
                if [ -f "$BAMFILE" ]; then
                    if [ "$PLUGIN_DEV_FULL_LOG" -gt 0 ]; then
                        echo "> $BAMFILE" >&2
                    fi
                else
                    echo "WARNING: Primer trimming failed. Using original BAM file." >&2
                    BAMFILE="$BAM_UNTRIM"
                fi
            fi
            run "${DIRNAME}/variantCaller.py $LOGOPT -o $TSP_FLOWORDER -p \"$INPUT_VC_PARAMFILE\" -r \"$DIRNAME\" -b \"$INPUT_BED_MERGE\" \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\" $VCLOGOUT" 
        else
            run "${DIRNAME}/variantCaller.py $LOGOPT -o $TSP_FLOWORDER -p \"$INPUT_VC_PARAMFILE\" -r \"$DIRNAME\" \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\" $VCLOGOUT" 
        fi
    fi

    # Create softlink to javascript and css folders - required for html headers
     if [ ! -d "${OUTDIR}/js" ]; then
        run "ln -sf \"${OUTDIR}/../js\" \"${OUTDIR}/\"";
        run "ln -sf \"${OUTDIR}/../css\" \"${OUTDIR}/\"";
    fi
     
    # combine all contig SNP variants into one file and blow away all original output
    local chrid
    if [ -d "${OUTDIR}/dibayes_out" ]; then
        echo "Merging SNP calls..." >&2
        rm -f "${OUTDIR}/$PLUGIN_OUT_SNPS_VCF"
        filtOn=0
        for chrid in `cat ${REFERENCE_FAI} | awk '{print $1}'`; do
            fname="${OUTDIR}/dibayes_out/diBayes_run_${chrid}_SNP.vcf"
            if [ -f "$fname" ]; then
                if [ "$filtOn" -eq 0 ]; then
                    filtOn=1
                else
                    sed -i -e "/^#/d" "$fname"
                fi
            cat "$fname" >> "${OUTDIR}/$PLUGIN_OUT_SNPS_VCF"
            fi
        done
        rm -rf "${OUTDIR}/dibayes_out"
    fi
    if [ ! -f "${OUTDIR}/$PLUGIN_OUT_SNPS_VCF" ]; then
        touch "${OUTDIR}/$PLUGIN_OUT_SNPS_VCF"
    fi

    # Filter INDELs from variant-hunter to regions
    if [ -f "${OUTDIR}/indels.recode.vcf" ]; then
        run "${DIRNAME}/vcf-sort ${OUTDIR}/indels.recode.vcf > ${OUTDIR}/$PLUGIN_OUT_INDELS_VCF";
        rm -f "${OUTDIR}/indels.recode.vcf"
    else
        touch "${OUTDIR}/$PLUGIN_OUT_INDELS_VCF";
    fi

    # Create local link to igv.php3 (for barcodes) and xml template required for adding IGV links
    # - will fail if the short name is not know to IGV
    if [ "$OUTDIR" != "$TSP_FILEPATH_PLUGIN_DIR" ]; then
        run "ln -sf ${TSP_FILEPATH_PLUGIN_DIR}/igv.php3 ${OUTDIR}/igv.php3"
    fi
    run "\"${DIRNAME}/scripts/create_igv_link.py\" -r ${OUTDIR} -b ${PLUGIN_OUT_BAMFILE} -v ${PLUGIN_OUT_SNPS_VCF}.gz -V ${PLUGIN_OUT_INDELS_VCF}.gz -g ${TSP_LIBRARY} -s igv_session.xml"

    # Create table files from vcf - both or either BED file may be ""
    local ALLVAR_TAB="${OUTDIR}/$PLUGIN_OUT_ALLVARS"
    run "\"${DIRNAME}/parse_variants_dibayes.py\" \"${OUTDIR}/$PLUGIN_OUT_SNPS_VCF\" \"${OUTDIR}/$PLUGIN_OUT_SNPS\" \"$INPUT_BED_FILE\"";
    run "\"${DIRNAME}/parse_variants_indels.py\"  \"${OUTDIR}/$PLUGIN_OUT_INDELS_VCF\" \"${OUTDIR}/$PLUGIN_OUT_INDELS\" \"$INPUT_BED_FILE\"";
    if [ -n "$INPUT_SNP_BED_FILE" ]; then
        run "\"$SCRIPTSDIR/collate_variants.pl\" $LOGOPT -v -G \"$REFERENCE_FAI\" -B \"$PLUGIN_HS_ALIGN_BED\" \"$ALLVAR_TAB\" \"${OUTDIR}/$PLUGIN_OUT_SNPS\" \"${OUTDIR}/$PLUGIN_OUT_INDELS\"";
    else
        run "\"$SCRIPTSDIR/collate_variants.pl\" $LOGOPT -v -G \"$REFERENCE_FAI\" \"$ALLVAR_TAB\" \"${OUTDIR}/$PLUGIN_OUT_SNPS\" \"${OUTDIR}/$PLUGIN_OUT_INDELS\"";
    fi
    rm -f "${OUTDIR}/$PLUGIN_OUT_SNPS" "${OUTDIR}/$PLUGIN_OUT_INDELS"

    # Convert VCF files to bgz and create index files for uploading to IGV
    if [ ! -s "${OUTDIR}/$PLUGIN_OUT_SNPS_VCF" ]; then
        echo "##fileformat=VCFv4.1" > "${OUTDIR}/$PLUGIN_OUT_SNPS_VCF"
        echo -e "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample" >> "${OUTDIR}/$PLUGIN_OUT_SNPS_VCF"
    fi
    if [ ! -s "${OUTDIR}/$PLUGIN_OUT_INDELS_VCF" ] ; then
        echo "##fileformat=VCFv4.1" > "${OUTDIR}/$PLUGIN_OUT_INDELS_VCF"
        echo -e "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample" >> "${OUTDIR}/$PLUGIN_OUT_INDELS_VCF"
    fi

    run "${DIRNAME}/bgzip -c \"${OUTDIR}/${PLUGIN_OUT_SNPS_VCF}\" > \"${OUTDIR}/${PLUGIN_OUT_SNPS_VCF}.gz\""
    run "${DIRNAME}/tabix -p vcf \"${OUTDIR}/${PLUGIN_OUT_SNPS_VCF}.gz\""
    run "${DIRNAME}/bgzip -c \"${OUTDIR}/${PLUGIN_OUT_INDELS_VCF}\" > \"${OUTDIR}/${PLUGIN_OUT_INDELS_VCF}.gz\""
    run "${DIRNAME}/tabix -p vcf \"${OUTDIR}/${PLUGIN_OUT_INDELS_VCF}.gz\""

    # Filter SNPs/INDELs for hotspots (statistics)
    local HAVE_LOCI=0
    local HSOPT=""
    local LOCI_CHRVARS_TAB="${OUTDIR}/$PLUGIN_OUT_LOCI_CHRVARS"
    if [ -n "$INPUT_SNP_BED_FILE" ]; then
        HAVE_LOCI=1
        HSOPT="-v"
        echo "Collecting HotSpot variants..." >&2
        run "${VCFTOOLS} --vcf ${OUTDIR}/${PLUGIN_OUT_SNPS_VCF} --bed ${PLUGIN_HS_ALIGN_BED} --out ${OUTDIR}/hotspot.snps --recode --keep-INFO-all > /dev/null";
        if [ -f "${OUTDIR}/hotspot.snps.recode.vcf" ]; then
            run "${DIRNAME}/vcf-sort ${OUTDIR}/hotspot.snps.recode.vcf > ${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS_VCF";
            rm -f "${OUTDIR}/hotspot.snps.recode.vcf"
        else
            touch "${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS_VCF"
        fi
        run "${VCFTOOLS} --vcf ${OUTDIR}/${PLUGIN_OUT_INDELS_VCF} --bed ${PLUGIN_HS_ALIGN_BED} --out ${OUTDIR}/hotspot.indels --recode --keep-INFO-all > /dev/null";
        if [ -f "${OUTDIR}/hotspot.indels.recode.vcf" ]; then
            run "${DIRNAME}/vcf-sort ${OUTDIR}/hotspot.indels.recode.vcf > ${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS_VCF";
            rm -f "${OUTDIR}/hotspot.indels.recode.vcf"
        else
            touch "${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS_VCF"
        fi
        # temporary tables of hotspot SNPs and INDELs created just for summary report
        run "\"${DIRNAME}/parse_variants_dibayes.py\" \"${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS_VCF\" \"${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS\" \"$PLUGIN_HS_ALIGN_BED\"";
        run "\"${DIRNAME}/parse_variants_indels.py\"  \"${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS_VCF\" \"${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS\" \"$PLUGIN_HS_ALIGN_BED\"";
        run "\"$SCRIPTSDIR/snps_summary_report.pl\" $LOGOPT \"$LOCI_CHRVARS_TAB\" \"${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS\" \"${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS\"";
        rm -f "${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS" "${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS"
        rm -f "${OUTDIR}/$PLUGIN_OUT_LOCI_SNPS_VCF" "${OUTDIR}/$PLUGIN_OUT_LOCI_INDELS_VCF"
    fi

    # Collect per-chromosome variant stats
    echo "Collecting per-chromosome variants summary..." >&2
    local CHRVARS_TAB="${OUTDIR}/$PLUGIN_OUT_CHRVARS"
    local CHRVARS_HTML="${OUTDIR}/CHRVARS_html"
    run "\"$SCRIPTSDIR/snps_summary_report.pl\" $LOGOPT $HSOPT -s -G \"$REFERENCE_FAI\" -T \"$CHRVARS_HTML\" \"$CHRVARS_TAB\" \"$ALLVAR_TAB\"";

    # Generate allele counts if hotspots loci BED provided
    local ALLELES_HTML="${OUTDIR}/ALLELES_html"
    if [ -n "$INPUT_SNP_BED_FILE" ]; then
        echo "Generating base pileup for hotspot alleles..." >&2
        run "samtools mpileup -BQ0 -d1000000 -f \"$REFERENCE\" -l ${INPUT_SNP_BED_MERGE} ${BAMFILE} $ERROUT | ${DIRNAME}/allele_count_mpileup_stdin.py > ${OUTDIR}/$PLUGIN_OUT_COV_RAW";
        run "\"${DIRNAME}/print_allele_counts.py\" \"${OUTDIR}/$PLUGIN_OUT_COV_RAW\" \"${OUTDIR}/$PLUGIN_OUT_COV\" \"$PLUGIN_HS_ALIGN_BED\" > \"$ALLELES_HTML\"";
    fi

    # Generate coverage statistics
    echo "Generating coverage statistics..." >&2
    run "${SCRIPTSDIR}/coverage_analysis.sh $LOGOPT -O \"$PLUGIN_OUT_TARGET_STATS\" -R \"$PLUGIN_OUT_READ_STATS\" -B \"$INPUT_BED_MERGE\" -V \"$CHRVARS_TAB\" -D \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\""
    if [ -n "$INPUT_SNP_BED_MERGE" ]; then
        run "${SCRIPTSDIR}/coverage_analysis.sh $LOGOPT -O \"$PLUGIN_OUT_LOCI_STATS\" -B \"$INPUT_SNP_BED_MERGE\" -V \"$LOCI_CHRVARS_TAB\" -D \"$OUTDIR\" \"$REFERENCE\" \"$BAMFILE\""
    fi

    # Link bam/bed files from plugin dir - avoid redundant copies and make generic to linker
    if [ -n "$BAM_UNTRIM" ]; then
        BAMFILE="$BAM_UNTRIM"
    fi
    if [ -n "$BAMFILE" ]; then
        run "ln -sf ${BAMFILE} ${OUTDIR}/$PLUGIN_OUT_BAMFILE"
        run "ln -sf ${BAMFILE}.bai ${OUTDIR}/$PLUGIN_OUT_BAIFILE"
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
    write_html_header "$HTMLOUT";
    if [ -n "$INPUT_SNP_BED_MERGE" ]; then
        run "${SCRIPTSDIR}/coverage_analysis_report.pl -t \"$RUNID\" -R \"${OUTDIR}/$PLUGIN_OUT_READ_STATS\" -S \"${OUTDIR}/$PLUGIN_OUT_LOCI_STATS\" -T \"${OUTDIR}/$HTML_ROWSUMS\" \"$HTMLOUT\" \"${OUTDIR}/$PLUGIN_OUT_TARGET_STATS\""
    else
        run "${SCRIPTSDIR}/coverage_analysis_report.pl -t \"$RUNID\" -R \"${OUTDIR}/$PLUGIN_OUT_READ_STATS\" -T \"${OUTDIR}/$HTML_ROWSUMS\" \"$HTMLOUT\" \"${OUTDIR}/$PLUGIN_OUT_TARGET_STATS\""
    fi
    write_html_chromo_variants "$CHRVARS_HTML" $HAVE_LOCI >> "$HTMLOUT";
    local VARS_HTML="${OUTDIR}/VARIANTS_html"
    run "${SCRIPTSDIR}/table2html.pl \"$ALLVAR_TAB\" > \"$VARS_HTML\""
    write_html_variants "$VARS_HTML" $HAVE_LOCI >> "$HTMLOUT";
    # no alleles table if hotspots not used
    if [ -n "$INPUT_SNP_BED_FILE" ]; then
        write_html_allele_coverage "$ALLELES_HTML" >> "$HTMLOUT";
    fi
    write_html_file_links "$OUTURL" "$OUTDIR" >> "$HTMLOUT";
    write_html_footer "$HTMLOUT";

    # Remove temporary files
    rm -f "$VARS_HTML" "$ALLELES_HTML" "$CHRVARS_HTML"
    if [ "$PLUGIN_DEV_KEEP_INTERMEDIATE_FILES" -eq 0 ]; then
        rm -f ${CHRVARS_TAB} ${LOCI_CHRVARS_TAB} ${OUTDIR}/variantCalls.*
        rm -f ${OUTDIR}/*.log ${OUTDIR}/*.bak ${OUTDIR}/bayesian_scorer.vcf
    fi
}

# Local copy of sorted barcode list file
if [ -f $TSP_FILEPATH_BARCODE_TXT ]; then
   run "sort -t ' ' -k 2n,2 \"$TSP_FILEPATH_BARCODE_TXT\" > \"$BARCODES_LIST\"";
fi

# Remove previous results to avoid displaying old before ready
rm -f "${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}" "$JSON_RESULTS" ${TSP_FILEPATH_PLUGIN_DIR}/*.bed
rm -rf ${TSP_FILEPATH_PLUGIN_DIR}/*.bam* ${TSP_FILEPATH_PLUGIN_DIR}/dibayes* ${PLUGIN_HS_ALIGN_DIR}
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/hotspot* ${TSP_FILEPATH_PLUGIN_DIR}/variant* ${TSP_FILEPATH_PLUGIN_DIR}/allele*
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/*_stats.txt ${TSP_FILEPATH_PLUGIN_DIR}/*.xls ${TSP_FILEPATH_PLUGIN_DIR}/*.log
if [ $PLUGIN_DEV_SKIP_VARIANT_CALLING -eq 0 ]; then
   rm -f ${TSP_FILEPATH_PLUGIN_DIR}/SNP* ${TSP_FILEPATH_PLUGIN_DIR}/indel*
   rm -f  ${TSP_FILEPATH_PLUGIN_DIR}/*.vcf
fi

# Get local copy of js and css
run "mkdir -p ${TSP_FILEPATH_PLUGIN_DIR}/js";
run "cp ${DIRNAME}/js/*.js ${TSP_FILEPATH_PLUGIN_DIR}/js/.";
run "mkdir -p ${TSP_FILEPATH_PLUGIN_DIR}/css";
run "cp ${DIRNAME}/css/*.css ${TSP_FILEPATH_PLUGIN_DIR}/css/.";

# Get local copy of php3 linkin script to ensure report tables alway work (even if plugin changes)
run "cp ${DIRNAME}/scripts/igv.php3 ${TSP_FILEPATH_PLUGIN_DIR}/igv.php3"

# Get local copy of BED files (may be deleted from system later)
PLUGIN_OUT_BEDFILE=`echo "$INPUT_BED_FILE" | sed -e 's/^.*\///'`
PLUGIN_OUT_LOCI_BEDFILE=`echo "$INPUT_SNP_BED_FILE" | sed -e 's/^.*\///'`
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

# Run for barcodes or single page
if [ -f $TSP_FILEPATH_BARCODE_TXT ]; then
    barcode;
else
    # Write a front page for non-barcode run
    HTML="${TSP_FILEPATH_PLUGIN_DIR}/${HTML_RESULTS}"
    write_html_header "$HTML" 15;
    echo "<h3><center>${PLUGIN_RUNNAME}</center></h3>" >> "$HTML"
    display_static_progress "$HTML";
    write_html_footer "$HTML";
    # Perform the analysis
    call_variants "$PLUGIN_RUN_NAME" "$TSP_FILEPATH_PLUGIN_DIR" "." "$TSP_FILEPATH_BAM"
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
