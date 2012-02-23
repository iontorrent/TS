#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
#AUTORUNDISABLE
#as of 6/21 the stack size being set on SGE nodes is too large, setting manually to the default
ulimit -s 8192
#$ -l mem_free=22G,h_vmem=22G,s_vmem=22G
#normal plugin script
VERSION="0.1"
#setup environment

if [ ! -d /results/referenceLibrary/tmap-f2/hg19 ]; then
    echo "<html>" > "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
    echo "<head>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
    echo "<body>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
echo "<h4>This Plugin is is designed to work only with the human hg19 reference.<br> The following directions will help you install this reference on your Torrent Server <br>
    1. Download the FASTA file from <a href=\"http://updates.iontorrent.com/reference/hg19.zip\">http://updates.iontorrent.com/reference/hg19.zip</a> <br>
    2. Upload the zip file using the reference upload UI<br>
    3. During the upload process the short name of the genome should be set to hg19 <br>
    4. The index creation will take a few hours, once complete rerun the plugin <br>
    </h4>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
    echo "</body>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
    echo "</html>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html"
    exit 1
fi

# Source the HTML files
for HTML_FILE in `find ${DIRNAME}/html/ | grep .sh$`
do
    source ${HTML_FILE};
done

#*! @function
#  @param  $*  the command to be executed
run ()
{
    echo "running: $*" >> ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_DETAILED_LOG};
    eval $*;
    EXIT_CODE="$?";
        if test ${EXIT_CODE} != 0; then
        # Remove the HTML file
            rm -v "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
            failure "status code '${EXIT_CODE}' while running '$*'";
        fi
}



set_output_paths ()
{
    PLUGIN_OUT_COV=allele_counts.txt;
    PLUGIN_OUT_STATS=target_stats.txt;
    PLUGIN_OUT_DETAILED_LOG=${PLUGINNAME}_log.txt;
    PLUGIN_OUT_GATK_VARIANTS=variants.vcf;
}
set_output_paths;
INPUT_BED_FILE="/results/plugins/AmpliSeqCancerVariantCaller/bedfiles/400_hsm_v12_1_seq.bed";
INPUT_SNP_BED_FILE="/results/plugins/AmpliSeqCancerVariantCaller/bedfiles/HSM_ver12_1_loci.bed";
LOG4CXX_CONFIGURATION="/results/plugins/AmpliSeqCancerVariantCaller/log4j.properties";
export LOG4CXX_CONFIGURATION;
GATK="/results/plugins/AmpliSeqCancerVariantCaller/GATK/dist/GenomeAnalysisTK.jar";
BGZIP="/results/plugins/Germ-lineVariantCaller/bgzip";
TABIX="/results/plugins/Germ-lineVariantCaller/tabix";
VCFTOOLS="/results/plugins/AmpliSeqCancerVariantCaller/vcftools";


# Run SNP calling

python ${DIRNAME}/ampliSeqCancerVariantCaller.py ${TSP_FILEPATH_PLUGIN_DIR}/startplugin.json >> ${TSP_FILEPATH_PLUGIN_DIR}/launch_sh_output.txt;
wait;

parse_variants() {

    if [ -z ${1} ]; then
        failure "Error: Arg 1: Results dir missing";
    elif [ -z {2} ]; then
        failure "Error: Arg 2 : Plugin name missing";
    elif [ -z {3} ]; then
        failure "Error: Arg 3: URL path missing";
    elif [ -z {4} ]; then
        failure "Error: Arg 4: Input BAM missing";
    elif [ -z {5} ]; then
        failure "Error: Arg 5: Input flowspace BAM missing";
    fi

    local TSP_FILEPATH_PLUGIN_DIR=${1};
    local PLUGINNAME=${2};
    local TSP_URLPATH_PLUGIN_DIR=${3};
    local INPUT_BAM=${4};
    local INPUT_FLOW_BAM=${5};

# delete older results files if present

if [ -f ${TSP_FILEPATH_PLUGIN_DIR}/indel_variants.vcf.gz ]; then
    rm -f "${TSP_FILEPATH_PLUGIN_DIR}/indel_variants.vcf.gz";
    rm -f "${TSP_FILEPATH_PLUGIN_DIR}/variantCalls.filtered.vcf";
fi

# filter indel calls
run "python ${DIRNAME}/filter_indels.py ${TSP_FILEPATH_PLUGIN_DIR}/variantCalls.vcf ${TSP_FILEPATH_PLUGIN_DIR}/variantCalls.filtered.vcf";
run "${VCFTOOLS} --vcf ${TSP_FILEPATH_PLUGIN_DIR}/variantCalls.filtered.vcf --bed ${INPUT_SNP_BED_FILE} --out indels --recode --keep-INFO-all";

if [ -f ${TSP_FILEPATH_PLUGIN_DIR}/indels.recode.vcf ]; then
    run "${DIRNAME}/vcf-sort ${TSP_FILEPATH_PLUGIN_DIR}/indels.recode.vcf > ${TSP_FILEPATH_PLUGIN_DIR}/indel_variants.vcf";
else
    run "touch ${TSP_FILEPATH_PLUGIN_DIR}/indel_variants.vcf";
fi

# clean up consensus files from dibayes directory
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/*.fasta;
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/*.txt;
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/*.gff3;

# remove header lines from all per chromsome files apart from chr1

for i in `seq 2 22` X Y M;
do
sed -i -e "/^#/d" ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/*chr$i\_SNP.vcf;
done

#combine per contig vcf files from diBayes

for i in `seq 1 22` X Y M;
do 
cat ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/*chr$i\_SNP.vcf >> ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/SNP_variants.vcf;
done


# delete per chromsome files
rm -f ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/*chr*_SNP.vcf;


# Run Allele Counts

run "samtools mpileup -BQ0 -d1000000 -f /results/referenceLibrary/tmap-f2/hg19/hg19.fasta -l ${INPUT_SNP_BED_FILE} ${TSP_FILEPATH_PLUGIN_DIR}/${INPUT_BAM} | ${DIRNAME}/allele_count_mpileup_stdin.py  > ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_COV}";

# Count on/off target reads
run "samtools view ${TSP_FILEPATH_PLUGIN_DIR}/${INPUT_BAM} | wc -l > ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_STATS}";
run "samtools view -F 0x4 ${TSP_FILEPATH_PLUGIN_DIR}/${INPUT_BAM} | wc -l >> ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_STATS}";
run "samtools view -F 0x4 -L ${INPUT_BED_FILE} ${TSP_FILEPATH_PLUGIN_DIR}/${INPUT_BAM} | wc -l >> ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_STATS}";
run "samtools depth -b ${INPUT_BED_FILE} ${TSP_FILEPATH_PLUGIN_DIR}/${INPUT_BAM} | ${DIRNAME}/depth.py >> ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_STATS}";


# bgzip and index the variant files
${BGZIP} -f ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/SNP_variants.vcf;
${TABIX} -fp vcf  ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/SNP_variants.vcf.gz;
${BGZIP} -f ${TSP_FILEPATH_PLUGIN_DIR}/indel_variants.vcf;
${TABIX} -fp vcf ${TSP_FILEPATH_PLUGIN_DIR}/indel_variants.vcf.gz;

# copy bed files from plugin dir
cp -v ${DIRNAME}/bedfiles/400_hsm_v12_1_seq.bed ${TSP_FILEPATH_PLUGIN_DIR}/amplicon_seq.bed;
cp -v ${DIRNAME}/bedfiles/HSM_ver12_1_loci.bed ${TSP_FILEPATH_PLUGIN_DIR}/Cancer_panel_loci.bed;

# ===================================================
# Create the html report page
# ===================================================

mkdir -vp "${TSP_FILEPATH_PLUGIN_DIR}/js";
cp -v ${DIRNAME}/js/*.js "${TSP_FILEPATH_PLUGIN_DIR}/js/.";
mkdir -vp "${TSP_FILEPATH_PLUGIN_DIR}/css";
cp -v ${DIRNAME}/css/*.css "${TSP_FILEPATH_PLUGIN_DIR}/css/.";

# HTML tag
echo -n '' > "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo '<html>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Head
print_html_head >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Body start
echo -e '\t<body>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Ion Torrent logo
print_html_logo >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Inner division (start)
echo -n \
'        <div id="inner">
            <h1><center>AmpliSeq Cancer Variant Caller</center></h1>
            <h5><center>This variant caller is specific to the Ampliseq Cancer Panel Kit. 
            For general variant calling please use the Germ-line Variant Caller. </center></h5>
' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Variant table - diBayes
print_html_variants_dibayes >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Variant table - indels
print_html_variants_indels >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Allele counts
allele_coverage >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Target statistics
target_stats >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
#File Links
print_html_file_links >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Ending Javascript
print_html_end_javascript >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Footer
print_html_footer >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Inner division (end)
echo -n \
'           </div>
    </body>
</html>
' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
}

if [ ! -f $TSP_FILEPATH_BARCODE_TXT ]
then
    parse_variants ${TSP_FILEPATH_PLUGIN_DIR} ${PLUGINNAME} ${TSP_URLPATH_PLUGIN_DIR} "sorted.bam" "sorted_flowspace.bam";
else
    barcode;
fi

