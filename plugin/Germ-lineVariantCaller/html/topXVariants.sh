#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


print_html_top_X_variants()
{
# PLUGIN_OUT_TOTAL_DISPLAYED should be a global variable;
if [ ${PLUGIN_OUT_TOTAL_NUM} -ge ${PLUGIN_OUT_TOP_NUM} ]; then
	PLUGIN_OUT_TOTAL_DISPLAYED=${PLUGIN_OUT_TOP_NUM};
else
	PLUGIN_OUT_TOTAL_DISPLAYED=${PLUGIN_OUT_TOTAL_NUM};
fi
echo \
"            <div id=\"VariantSample\" class=\"report_block\">
                <h2>Variant Sample</h2>
                <div calss=\"demo_jui\">
					<br/>
                    Displaying the top ${PLUGIN_OUT_TOTAL_DISPLAYED} variants out of the ${PLUGIN_OUT_TOTAL_NUM} total called variants, sorted by quality.
					<br/>
					Click on a column heading to sort the table, enter a search term to filter the variants, or scroll right to see more information.
					<br/>
					<br/>
					<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" class=\"noheading\" id=\"datatable\" style=\"word-wrap:break-word\">
						<thead>
							<tr>
								<th class=\"col_small\">
								</th>
								<th class=\"col_large\">
								</th>
								<th class=\"col_med\">
								</th>
								<th class=\"col_med\">
								</th>
								<th class=\"col_med\">
								</th>
								<th class=\"col_small\">
								</th>
								<th class=\"col_small\">
								</th>
								<th class=\"col_small\">
								</th>
								<th colspan=\"5\" class=\"col_span\">
								<span class=\"tip\" title=\"The coverage information at this position\">
										<span class=\"tippy\">Coverage Information</span>
									</span>
								</th>
							</tr>
							<tr>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"Click the link to view the variant in IGV\">
										<span class=\"tippy\">View in IGV</span>
									</span>
								</th>
								<th class=\"col_large\">
									<span class=\"tip\" title=\"The sequence (contig) name in the reference genome FASTA\">
										<span class=\"tippy\">Sequence Name</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"The one-based position in the reference genome\">
										<span class=\"tippy\">Position</span>
									</span>
								</th>
								<th class=\"col_med\">
								<span class=\"tip\" title=\"The reference base(s)\">
										<span class=\"tippy\">Reference</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"Comma separated list of alternate non-reference alleles\">
										<span class=\"tippy\">Variant</span>
									</span>
								</th>
								<th class=\"col_small\">
									<span class=\"tip\" title=\"The type of the variant\">
										<span class=\"tippy\">Type</span>
									</span>
								</th>
								<th class=\"col_small\">
									<span class=\"tip\" title=\"The ploidy of the variant\">
										<span class=\"tippy\">Ploidy</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"The phred-scaled quality score that the variant is incorrect (high quality scores indicate high confidence variant calls)\">
										<span class=\"tippy\">Quality</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"The total reads covering the position.  This includes low quality bases and so may be greater than the sum of the forward/reverse reference/variant counts.\">
										<span class=\"tippy\">Total Coverage</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"The number of high quality reads covering the reference on the forward strand\">
										<span class=\"tippy\">Reference Forward</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"The number of high quality reads covering the reference on the reverse strand\">
										<span class=\"tippy\">Reference Reverse</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"The number of high quality reads covering the variant on the forward strand\">
										<span class=\"tippy\">Variant Forward</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"The number of high quality reads covering the variant on the reverse strand\">
										<span class=\"tippy\">Variant Reverse</span>
									</span>
								</th>
							</tr>
						</thead>
						<tbody>
";
#echo "<tr><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td></tr>\n";
run python ${DIRNAME}/scripts/table_top_x_variants.py \
-v ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_VCF_GZ_VARIANTS} \
-n ${PLUGIN_OUT_TOP_NUM} \
-s ${PLUGIN_OUT_SESSION_XML_NAME};
echo -n \
'						</tbody>
                    </table>
					<div class=\"spacer\"></div>
                </div>
            </div>
';
}
