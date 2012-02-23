#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

barcode_links ()
{
	local HTML="${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
	if [ ! -z ${1} ]; then
		HTML=${1};
	fi
	echo -e "\t\t\t<div id=\"BarcodeList\" class=\"report_block\"/>" >> ${HTML};
	echo -e "\t\t\t\t<h2>Barcode Variant Reports</h2>" >> ${HTML};
	echo -e "\t\t\t\t<div>" >> ${HTML};
	echo -e "\t\t\t\t\t<br/>" >> ${HTML};
	echo -e "\t\t\t\t\t<table class=\"noheading\">" >> ${HTML};
	echo -e "\t\t\t\t\t\t<tr>" >> ${HTML};
	echo -e "\t\t\t\t\t\t\t<th><span class=\"tip\" title=\"The integer value unique to the sequence within this set\"><span class=\"tippy\">Barcode Index</span></span></th>" >> ${HTML};
	echo -e "\t\t\t\t\t\t\t<th><span class=\"tip\" title=\"The name for the individual sequence\"><span class=\"tippy\">Barcode Identifier</span></span></th>" >> ${HTML};
	echo -e "\t\t\t\t\t\t\t<th><span class=\"tip\" title=\"The sequence of bases defining the barcode\"><span class=\"tippy\">Barcode Sequence</span></span></th>" >> ${HTML};
	echo -e "\t\t\t\t\t\t\t<th><span class=\"tip\" title=\"The individual variant report for this barcode\"><span class=\"tippy\">Barcode Variant Report</span></span></th>" >> ${HTML};
	echo -e "\t\t\t\t\t\t</tr>" >> ${HTML};
	CTR=0;
	for BARCODE_LINE in `cat ${TSP_FILEPATH_BARCODE_TXT} | grep "^barcode"`
	do
		BARCODE_IDX=`echo ${BARCODE_LINE} | awk 'BEGIN{FS=","} {print $1}'`;
		BARCODE_ID=`echo ${BARCODE_LINE} | awk 'BEGIN{FS=","} {print $2}'`;
		BARCODE_SEQ=`echo ${BARCODE_LINE} | awk 'BEGIN{FS=","} {print $3}'`;
		BARCODE_BAM_NAME="${BARCODE_ID}_${PLUGIN_OUT_BAM_NAME}";
		BARCODE_BAM="${ANALYSIS_DIR}/${BARCODE_ID}_${PLUGIN_OUT_BAM_NAME}";
		BARCODE_IDXS[${CTR}]=${BARCODE_IDX};
		BARCODE_IDS[${CTR}]=${BARCODE_ID};
		BARCODE_SEQS[${CTR}]=${BARCODE_SEQ};
		BARCODE_BAM_NAMES[${CTR}]=${BARCODE_BAM_NAME};
		BARCODE_BAMS[${CTR}]=${BARCODE_BAM};
		echo -e "\t\t\t\t\t\t<tr>" >> ${HTML};
		echo -e "\t\t\t\t\t\t\t<td>${BARCODE_IDX}</td>" >> ${HTML};
		echo -e "\t\t\t\t\t\t\t<td>${BARCODE_ID}</td>" >> ${HTML};
		echo -e "\t\t\t\t\t\t\t<td>${BARCODE_SEQ}</td>" >> ${HTML};
		
		# Check to see if the BAM exists
		if [ -f ${BARCODE_BAM} ]; then # exists
			if [ ! -f ${BARCODE_BAM}.bai ]; then
				run "samtools index ${BARCODE_BAM}";
			fi
			echo -e "\t\t\t\t\t\t\t<td><a href=\"${BARCODE_SEQ}/${PLUGINNAME}.html\">Variant Report</a></td>" >> ${HTML};
			BARCODES_OK[${CTR}]="1";
		else
			echo -e "\t\t\t\t\t\t\t<td>No data</td>" >> ${HTML};
			# Nullify
			BARCODES_OK[${CTR}]="0";
		fi
		echo -e "\t\t\t\t\t\t</tr>" >> ${HTML};
		CTR=`expr ${CTR} + 1`;
	done
	echo -e "\t\t\t\t\t</table>" >> ${HTML};
	echo -e "\t\t\t\t\t<br/>" >> ${HTML};
	echo -e "\t\t\t\t</div>" >> ${HTML};
	echo -e "\t\t\t</div>" >> ${HTML};
}

barcode ()
{
	local CTR;
	local BARCODE_IDX;
	local BARCODE_ID;
	local BARCODE_SEQ;
	local BARCODE_IDXS;
	local BARCODE_IDS;
	local BARCODE_SEQS;
	local BARCODES_OK;
	local BARCODE_BAM;
	local BARCODE_BAM_NAME;
	local BARCODE_LINE;

	# Yes, there are barcodes
	echo "There are barcodes!" >&2;
	
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

	# Link Table
	barcode_links;
	
	# Go through the barcodes 
	for((CTR=0;CTR<${#BARCODE_SEQS[@]};CTR++))
	do
		BARCODE_IDX=${BARCODE_IDXS[${CTR}]};
		BARCODE_ID=${BARCODE_IDS[${CTR}]};
		BARCODE_SEQ=${BARCODE_SEQS[${CTR}]};
		BARCODE_BAM_NAME="${BARCODE_BAM_NAMES[${CTR}]}";
		BARCODE_BAM="${BARCODE_BAMS[${CTR}]}";
		run "mkdir -p ${BARCODE_SEQ}"; # make a new dir
		if [ "0" == ${BARCODES_OK[${CTR}]} ]; then
			# No barcode!
			echo "Skipping ${BARCODE_SEQ}" >&2;
                        echo -n '' > "${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE_SEQ}/${PLUGINNAME}.html";
                        echo -e "\t\t\t<h2>There was no BAM file for the barcode ${BARCODE_SEQ}</h2>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE_SEQ}/${PLUGINNAME}.html";
		else
			echo "Variant calling ${BARCODE_SEQ}" >&2;
			# copy the php file for the IGV links, this could be referenced differently in the future (TODO)
			run "cp -v ${DIRNAME}/scripts/igv.php3 ${BARCODE_SEQ}/igv.php3";
			# run the variant calling
			run_variant_calling "${BARCODE_BAM}" "${BARCODE_BAM_NAME}" "${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE_SEQ}" "../../../${BARCODE_BAM_NAME}";
			# Variant summary table NB: we must fake the results directory
			TMP_TSP_FILEPATH_PLUGIN_DIR=${TSP_FILEPATH_PLUGIN_DIR};
			TSP_FILEPATH_PLUGIN_DIR=${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE_SEQ};
			print_html_variant_summary_table "1" >> "${TMP_TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
			TSP_FILEPATH_PLUGIN_DIR=${TMP_TSP_FILEPATH_PLUGIN_DIR};
		fi
	done
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

	# unset variables
	unset BARCODE_IDX;
	unset BARCODE_ID;
	unset BARCODE_SEQ;

	###  block run ### 
	echo -n '' > "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	echo '<html>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	print_html_head >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	echo -e '\t<body>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	barcode_links "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	print_html_end_javascript >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	echo -n '</body></html>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
}
