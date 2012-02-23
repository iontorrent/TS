#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Change the following line to all CAPS to disable auto-run of this plugin, but do not uncomment
#autorundisable

VERSION="0.1.0" # major.minor.bug
#HTML_ONLY=0; # 1 outputs the html portion only, 0 performs variant calling as well
#ALL_VARIANTS=0; # 1 this to give all positions, 0 for only variants

# ===================================================
# Add plugin specific code to execute below this line
# ===================================================

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

#*! @function
#  @param  $*  the command to be executed
print_version ()
{
	echo -n "$* " >> ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_DETAILED_LOG};
	# Note: samtools and tabix have no "--version" flag, and return non-zero
	( set +o pipefail
	eval $* 2>&1 | grep Version >> ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_DETAILED_LOG};
	)
}

#*! @function
set_output_paths ()
{
	PLUGIN_OUT_BAM_NAME=`echo ${TSP_FILEPATH_BAM} | sed -e 's_.*/__g'`; # Just the BAM file name, with no path/url
	PLUGIN_OUT_BCF=tmp.bcf;
	PLUGIN_OUT_VCF_GZ_VARIANTS=variants.vcf.gz;
	PLUGIN_OUT_VCF_GZ_VARIANTS_NAME=`echo ${PLUGIN_OUT_VCF_GZ_VARIANTS} | sed -e 's_.*/__g'`; 
	PLUGIN_OUT_VCF_GZ_ALL=all.positions.vcf.gz;
	PLUGIN_OUT_DETAILED_LOG=${PLUGINNAME}_log.txt;
	PLUGIN_OUT_SESSION_XML_NAME="igv_session.xml";
}

#*! @function
set_variant_calling_defaults ()
{
	SAMTOOLS_MAX_DEPTH=10000; # -d
	SAMTOOLS_MAX_DEPTH_INDEL=1000; # -L
	SAMTOOLS_MIN_BASEQ=7; # -Q
	SAMTOOLS_HP_COEFF=50; # -h
	SAMTOOLS_PHRED_GAPO=10; # -o
	SAMTOOLS_PHRED_GAPE=17; # -e 
	SAMTOOLS_MIN_GAPPED_READS=4; # -m
	VCF_FILTER_MIN_READ_DEPTH=2; # -d
	VCF_FILTER_MAX_READ_DEPTH=10000000; # -D
	VCF_FILTER_WIN_SNP_ADJ_GAPS=0; # -w
	VCF_FILTER_WIN_INDEL_ADJ_GAPS=0; # -W
	VCF_FILTER_SNPS_STRAND_DEPTH=1; # -s
	VCF_FILTER_INDELS_STRAND_DEPTH=1; # -S
	VCF_FILTER_HPS_STRAND_DEPTH=2; # -H
	BCFTOOLS_INDEL_TO_SUB_RATIO=-1; # -i
	PLUGIN_OUT_TOP_NUM=250; # the number of variants to output in the HTML table
	PLOTS_MAX_INDEL_LENGTH=10;
	PLOTS_MAX_COVERAGE=100;
}

#*! @function
set_variant_calling_params_from_json ()
{
	if [[ ${PLUGINCONFIG__SAMTOOLS_MAX_DEPTH} ]] && [ "${PLUGINCONFIG__SAMTOOLS_MAX_DEPTH}" != "default" ]; then
		SAMTOOLS_MAX_DEPTH=${PLUGINCONFIG__SAMTOOLS_MAX_DEPTH};
	fi
	if [[ ${PLUGINCONFIG__SAMTOOLS_MAX_DEPTH_INDEL} ]] && [ "${PLUGINCONFIG__SAMTOOLS_MAX_DEPTH_INDEL}" != "default" ]; then
		SAMTOOLS_MAX_DEPTH_INDEL=${PLUGINCONFIG__SAMTOOLS_MAX_DEPTH_INDEL};
	fi
	if [[ ${PLUGINCONFIG__SAMTOOLS_MIN_BASEQ} ]] && [ "${PLUGINCONFIG__SAMTOOLS_MIN_BASEQ}" != "default" ]; then
		SAMTOOLS_MIN_BASEQ=${PLUGINCONFIG__SAMTOOLS_MIN_BASEQ};
	fi
	if [[ ${PLUGINCONFIG__SAMTOOLS_HP_COEFF} ]] && [ "${PLUGINCONFIG__SAMTOOLS_HP_COEFF}" != "default" ]; then
		SAMTOOLS_HP_COEFF=${PLUGINCONFIG__SAMTOOLS_HP_COEFF};
	fi
	if [[ ${PLUGINCONFIG__SAMTOOLS_PHRED_GAPO} ]] && [ "${PLUGINCONFIG__SAMTOOLS_PHRED_GAPO}" != "default" ]; then
		SAMTOOLS_PHRED_GAPO=${PLUGINCONFIG__SAMTOOLS_PHRED_GAPO};
	fi
	if [[ ${PLUGINCONFIG__SAMTOOLS_PHRED_GAPE} ]] && [ "${PLUGINCONFIG__SAMTOOLS_PHRED_GAPE}" != "default" ]; then
		SAMTOOLS_PHRED_GAPE=${PLUGINCONFIG__SAMTOOLS_PHRED_GAPE};
	fi
	if [[ ${PLUGINCONFIG__SAMTOOLS_MIN_GAPPED_READS} ]] && [ "${PLUGINCONFIG__SAMTOOLS_MIN_GAPPED_READS}" != "default" ]; then
		SAMTOOLS_MIN_GAPPED_READS=${PLUGINCONFIG__SAMTOOLS_MIN_GAPPED_READS};
	fi
	if [[ ${PLUGINCONFIG__VCF_FILTER_MIN_READ_DEPTH} ]] && [ "${PLUGINCONFIG__VCF_FILTER_MIN_READ_DEPTH}" != "default" ]; then
		VCF_FILTER_MIN_READ_DEPTH=${PLUGINCONFIG__VCF_FILTER_MIN_READ_DEPTH};
	fi
	if [[ ${PLUGINCONFIG__VCF_FILTER_MAX_READ_DEPTH} ]] && [ "${PLUGINCONFIG__VCF_FILTER_MAX_READ_DEPTH}" != "default" ]; then
		VCF_FILTER_MAX_READ_DEPTH=${PLUGINCONFIG__VCF_FILTER_MAX_READ_DEPTH};
	fi
	if [[ ${PLUGINCONFIG__VCF_FILTER_WIN_SNP_ADJ_GAPS} ]] && [ "${PLUGINCONFIG__VCF_FILTER_WIN_SNP_ADJ_GAPS}" != "default" ]; then
		VCF_FILTER_WIN_SNP_ADJ_GAPS=${PLUGINCONFIG__VCF_FILTER_WIN_SNP_ADJ_GAPS};
	fi
	if [[ ${PLUGINCONFIG__VCF_FILTER_WIN_INDEL_ADJ_GAPS} ]] && [ "${PLUGINCONFIG__VCF_FILTER_WIN_INDEL_ADJ_GAPS}" != "default" ]; then
		VCF_FILTER_WIN_INDEL_ADJ_GAPS=${PLUGINCONFIG__VCF_FILTER_WIN_INDEL_ADJ_GAPS};
	fi
	if [[ ${PLUGINCONFIG__VCF_FILTER_SNPS_STRAND_DEPTH} ]] && [ "${PLUGINCONFIG__VCF_FILTER_SNPS_STRAND_DEPTH}" != "default" ]; then
		VCF_FILTER_SNPS_STRAND_DEPTH=${PLUGINCONFIG__VCF_FILTER_SNPS_STRAND_DEPTH};
	fi
	if [[ ${PLUGINCONFIG__VCF_FILTER_INDELS_STRAND_DEPTH} ]] && [ "${PLUGINCONFIG__VCF_FILTER_INDELS_STRAND_DEPTH}" != "default" ]; then
		VCF_FILTER_INDELS_STRAND_DEPTH=${PLUGINCONFIG__VCF_FILTER_INDELS_STRAND_DEPTH};
	fi
	if [[ ${PLUGINCONFIG__VCF_FILTER_HPS_STRAND_DEPTH} ]] && [ "${PLUGINCONFIG__VCF_FILTER_HPS_STRAND_DEPTH}" != "default" ]; then
		VCF_FILTER_HPS_STRAND_DEPTH=${PLUGINCONFIG__VCF_FILTER_HPS_STRAND_DEPTH};
	fi
	if [[ ${PLUGINCONFIG__BCFTOOLS_INDEL_TO_SUB_RATIO} ]] && [ "${PLUGINCONFIG__BCFTOOLS_INDEL_TO_SUB_RATIO}" != "default" ]; then
		BCFTOOLS_INDEL_TO_SUB_RATIO=${PLUGINCONFIG__BCFTOOLS_INDEL_TO_SUB_RATIO};
	fi
	if [[ ${PLUGINCONFIG__PLUGIN_OUT_TOP_NUM} ]] && [ "${PLUGINCONFIG__PLUGIN_OUT_TOP_NUM}" != "default" ]; then
		PLUGIN_OUT_TOP_NUM=${PLUGINCONFIG__PLUGIN_OUT_TOP_NUM};
	fi
	if [[ ${PLUGINCONFIG__PLOTS_MAX_INDEL_LENGTH} ]] && [ "${PLUGINCONFIG__PLOTS_MAX_INDEL_LENGTH}" != "default" ]; then
		PLOTS_MAX_INDEL_LENGTH=${PLUGINCONFIG__PLOTS_MAX_INDEL_LENGTH};
	fi
	if [[ ${PLUGINCONFIG__PLOTS_MAX_COVERAGE} ]] && [ "${PLUGINCONFIG__PLOTS_MAX_COVERAGE}" != "default" ]; then
		PLOTS_MAX_COVERAGE=${PLUGINCONFIG__PLOTS_MAX_COVERAGE};
	fi
	if [[ ${PLUGINCONFIG__HTML_ONLY} ]] && [ "${PLUGINCONFIG__HTML_ONLY}" != "default" ]; then
		HTML_ONLY=1;
	fi
}

# ===================================================
# Plugin initialization
# ===================================================
	
# HTML_ONLY: 1 this prints the html only, 0 re-calls variants and prints the html
if [ -z ${HTML_ONLY} ]; then
	HTML_ONLY=0; 
fi

# ALL_VARIANTS: 1 this to give all positions, 0 for only variants
if [ -z ${ALL_VARIANTS} ]; then
	ALL_VARIANTS=0; 
fi

# Set defaults
set_output_paths;
set_variant_calling_defaults;
set_variant_calling_params_from_json; # overwrite defaults if necessary
TABIX_PATH=/results/plugins/Germ-lineVariantCaller; # TODO: hard-coded

# Test for the existence of the reference genome files.
test_for_file "${TSP_FILEPATH_GENOME_FASTA}";
if [ ! -f ${TSP_FILEPATH_GENOME_FASTA}.fai ]; then
	echo -n '' > "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	echo "<html>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	echo -e "\t<body>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	echo "\t\t\t<h2>The genome index was missing: ${TSP_FILEPATH_GENOME_FASTA}.fai!<br>Please try rebuilding the genome FASTA from the TorrentSuite</h2>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	echo -e "\t</body>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	echo -e "</html>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	failure "The genome index was missing: ${TSP_FILEPATH_GENOME_FASTA}.fai!\nPlease try rebuilding the genome FASTA from the TorrentSuite";
fi
test_for_file "${TSP_FILEPATH_GENOME_FASTA}.fai"; # warn the user later

# Test for the existence of the relevant executables.
test_for_executable "samtools";
test_for_executable "bcftools";
test_for_executable "${TABIX_PATH}/tabix";
test_for_executable "${TABIX_PATH}/bgzip";
# Note: assuming python, awk, sed, cut, grep, head, wc, and echo are all available

run "mkdir -vp ${TSP_FILEPATH_PLUGIN_DIR}/js"; 
run "cp -v ${DIRNAME}/js/*.js ${TSP_FILEPATH_PLUGIN_DIR}/js/."; 
run "mkdir -vp ${TSP_FILEPATH_PLUGIN_DIR}/css"; 
run "cp -v ${DIRNAME}/css/*.css ${TSP_FILEPATH_PLUGIN_DIR}/css/.";
#copy the php file
run "cp -v ${DIRNAME}/scripts/igv.php3 ${TSP_FILEPATH_PLUGIN_DIR}/igv.php3";

# Notes: set the following:
# - TSP_FILEPATH_BAM: filepath of the BM
# - PLUGIN_OUT_BAM_NAME: name of the BAM
# - TSP_FILEPATH_PLUGIN_DIR: where the plugin output exists, and where the BAM file exists
# - PLUGIN_OUT_BAM_URL: the relative path from the TSP_FILEPATH_PLUGIN_DIR to the BAM file
run_variant_calling ()
{
	if [ -z ${1} ]; then
		failure "Error: argument one required";
	elif [ -z ${2} ]; then
		failure "Error: argument two required";
	elif [ -z ${3} ]; then
		failure "Error: argument three required";
	elif [ -z ${4} ]; then
		failure "Error: argument four required";
	fi

	local TSP_FILEPATH_BAM=${1}; # This masks the outside TSP_FILEPATH_BAM
	local PLUGIN_OUT_BAM_NAME=${2}; # This masks the outside PLUGIN_OUT_BAM_NAME
	local TSP_FILEPATH_PLUGIN_DIR=${3}; # This masks the outside TSP_FILEPATH_PLUGIN_DIR
	local PLUGIN_OUT_BAM_URL=${4}; # Relative to the variant report page

	# Make a directory to store the intermediates and results.
	run "mkdir -p ${TSP_FILEPATH_PLUGIN_DIR}";
	if [ ! -f ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_DETAILED_LOG} ]; then
		run "touch ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_DETAILED_LOG}";
	fi

	# Test for input files
	test_for_file "${TSP_FILEPATH_BAM}";
	test_for_file "${TSP_FILEPATH_BAM}.bai";

	# Print out the hostname for good measure
	hostname >> ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_DETAILED_LOG};

	# ===================================================
	# Call the variants
	# ===================================================

	# Print out the version #s
	print_version "samtools";
	print_version "${TABIX_PATH}/tabix";

	# Run the variant calling
	if test 0 = ${HTML_ONLY}; then
		if test 0 = ${ALL_VARIANTS}; then
			# Take the sorted BAM, and call it creating a compressed vcf
			run "samtools mpileup \
			-d ${SAMTOOLS_MAX_DEPTH} \
			-L ${SAMTOOLS_MAX_DEPTH_INDEL} \
			-Q ${SAMTOOLS_MIN_BASEQ} \
			-h ${SAMTOOLS_HP_COEFF} \
			-o ${SAMTOOLS_PHRED_GAPO} \
			-e ${SAMTOOLS_PHRED_GAPE} \
			-m ${SAMTOOLS_MIN_GAPPED_READS} \
			-f ${TSP_FILEPATH_GENOME_FASTA} \
			-g \
			${TSP_FILEPATH_BAM} \
			| bcftools view \
			-e -g -c -v -N \
			-i ${BCFTOOLS_INDEL_TO_SUB_RATIO} \
			- \
			| perl ${DIRNAME}/scripts/vcf_filter.pl \
			-d ${VCF_FILTER_MIN_READ_DEPTH} \
			-D ${VCF_FILTER_MAX_READ_DEPTH} \
			-w ${VCF_FILTER_WIN_SNP_ADJ_GAPS} \
			-W ${VCF_FILTER_WIN_INDEL_ADJ_GAPS} \
			-s ${VCF_FILTER_SNPS_STRAND_DEPTH} \
			-S ${VCF_FILTER_INDELS_STRAND_DEPTH} \
			-H ${VCF_FILTER_HPS_STRAND_DEPTH} \
			| ${TABIX_PATH}/bgzip > ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_VCF_GZ_VARIANTS}";
			# Index the vcf with tabix
			run "${TABIX_PATH}/tabix -p vcf ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_VCF_GZ_VARIANTS}";
		else
			# Take the sorted BAM, and create an uncalled bcf
			run "samtools mpileup \
			-d ${SAMTOOLS_MAX_DEPTH} \
			-L ${SAMTOOLS_MAX_DEPTH_INDEL} \
			-Q ${SAMTOOLS_MIN_BASEQ} \
			-h ${SAMTOOLS_HP_COEFF} \
			-o ${SAMTOOLS_PHRED_GAPO} \
			-e ${SAMTOOLS_PHRED_GAPE} \
			-f ${TSP_FILEPATH_GENOME_FASTA} \
			-g \
			${TSP_FILEPATH_BAM} \
			> ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_BCF}";

			# Take the uncalled bcf and call it creating a compressed vcf
			run "bcftools view \
			-e -g -c -N \
			-i ${BCFTOOLS_INDEL_TO_SUB_RATIO} \
			${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_BCF} \
			| ${TABIX_PATH}/bgzip > ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_VCF_GZ_ALL}";
			# Index the vcf with tabix
			run "${TABIX_PATH}/tabix -p vcf ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_VCF_GZ_ALL}";

			# Take the uncalled bcf and call it creating a compressed vcf
			run "bcftools view \
			-e -g -c -v -N \
			-i ${BCFTOOLS_INDEL_TO_SUB_RATIO} \
			${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_BCF} \
			| perl ${DIRNAME}/scripts/vcf_filter.pl \
			-d ${VCF_FILTER_MIN_READ_DEPTH} \
			-D ${VCF_FILTER_MAX_READ_DEPTH} \
			-w ${VCF_FILTER_WIN_SNP_ADJ_GAPS} \
			-W ${VCF_FILTER_WIN_INDEL_ADJ_GAPS} \
			-s ${VCF_FILTER_SNPS_STRAND_DEPTH} \
			-S ${VCF_FILTER_INDELS_STRAND_DEPTH} \
			-H ${VCF_FILTER_HPS_STRAND_DEPTH} \
			| ${TABIX_PATH}/bgzip > ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_VCF_GZ_VARIANTS}";
			# Index the vcf with tabix
			run "${TABIX_PATH}/tabix -p vcf ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_VCF_GZ_VARIANTS}";

			# Clean up
			run "rm -v ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_BCF}";
		fi
	fi

	# ===================================================
	# Create the IGV link
	# ===================================================

	# Set Locus
	# Get the first region to display
	LOCUS_CHR="";
	FIRST_LINE=`set +o pipefail; samtools view ${TSP_FILEPATH_BAM} | head -n 1`; # get the first line in the BAM
	if [ ! "" == "${FIRST_LINE}" ]; then
		LOCUS_CHR=`echo "${FIRST_LINE}" | awk '{print $3}'`; # get the chromosome name
		LOCUS_START=`echo "${FIRST_LINE}" | awk '{print $4}'`; # get the start position
		LOCUS_END=`set +o pipefail; samtools view ${TSP_FILEPATH_BAM} | grep -v "^@" | head -n 1000 | grep "${LOCUS_CHR}" | tail -n 1 | awk '{print $4}'`; # grab a later alignment
		if [ "" == "${LOCUS_END}" ]; then
# no later alignment, use the first region
			LOCUS_END=`echo "${FIRST_LINE}" | awk '{print $4}'`;
		else
# bound the region to 1000 bases 
			if [ `expr ${LOCUS_START} + 999` -lt ${LOCUS_END} ]; then
				LOCUS_END=`expr ${LOCUS_START} + 999`;
			fi
		fi
	fi
	LOCUS="${LOCUS_CHR}:${LOCUS_START}-${LOCUS_END}"
	run "python ${DIRNAME}/scripts/create_igv_link.py \
	--results-dir=${TSP_FILEPATH_PLUGIN_DIR} \
	--bam-file=${PLUGIN_OUT_BAM_NAME} \
	--vcf-file=${PLUGIN_OUT_VCF_GZ_VARIANTS_NAME} \
	--locus=\"${LOCUS}\" \
	--genome-name=${TSP_LIBRARY} \
	--session-xml-name=${PLUGIN_OUT_SESSION_XML_NAME}";

	# ===================================================
	# Create some debugging plots 
	# ===================================================
	run "python ${DIRNAME}/scripts/plots_variant_calling.py \
	--vcf-file=${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_VCF_GZ_VARIANTS} \
	--output-dir=${TSP_FILEPATH_PLUGIN_DIR} \
	--max-indel-length=${PLOTS_MAX_INDEL_LENGTH} \
	--max-coverage=${PLOTS_MAX_COVERAGE}";

	# ===================================================
	# Create the html report page
	# ===================================================

	# Get the # of variants that will be displayed
	PLUGIN_OUT_TOTAL_NUM=`set +o pipefail; gunzip -c ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_VCF_GZ_VARIANTS} | grep -v ^# | wc -l`;
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
	<h5><center>
	This Germ-line Variant Caller uses accepted and generic methodology based on SAM Tools and VCF Tools.  It is configured with a standard set of <a href="#about">parameters</a>.  If you wish to fine tune variant calling towards your own level tolerance for either false positives or false negatives, we would recommend the more adaptable SW available through our third party software partners.
	</center></h5>
	<h1><center>Germ-line Variant Caller Plugin</center></h1>
	' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	# Variant summary table
	if [ -f ${TSP_FILEPATH_PLUGIN_DIR}/results.json ]; then
		rm -v ${TSP_FILEPATH_PLUGIN_DIR}/results.json;
	fi
	print_html_variant_summary_table "0" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	# Top X variants
	print_html_top_X_variants >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	# File Links
	print_html_file_links >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	# Appendix
	print_html_appendix >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	# About
	print_html_about >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	# Ending Javascript
	print_html_end_javascript >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	# Footer
	print_html_footer >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	# Innert division (end)
	echo -n \
	'		</div>
	</body>
	</html>
	' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
}

# Check for barcodes
if [ -f ${TSP_FILEPATH_BARCODE_TXT} ]; then
	barcode;
else
	run_variant_calling ${TSP_FILEPATH_BAM} ${PLUGIN_OUT_BAM_NAME} ${TSP_FILEPATH_PLUGIN_DIR} "../../${PLUGIN_OUT_BAM_NAME}";

	###  block run ### 
	echo -n '' > "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	echo '<html>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	print_html_head >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	echo -e '\t<body>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	print_html_variant_summary_table "0" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	print_html_end_javascript >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	echo -n '</body></html>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
fi
