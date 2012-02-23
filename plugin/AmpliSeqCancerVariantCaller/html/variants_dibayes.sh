#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


print_html_variants_dibayes()
{
echo \
"            <div id=\"VariantCalls\" class=\"report_block\">
                <h2>Variant Calls (SNP) with frequency > 5%</h2>
                <div calss=\"demo_jui\">
					<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" class=\"noheading\" id=\"dibayestable\" style=\"word-wrap:break-word\">
						<thead>
							<tr>
								<th class=\"col_med\">
								</th>
								<th class=\"col_med\">
								</th>
								<th class=\"col_small\">
								</th>
								<th class=\"col_med\">
								</th>
								<th class=\"col_med\">
								</th>
								<th class=\"col_small\">
								</th>
								<th class=\"col_med\">
								</th>
								<th colspan=\"3\" class=\"col_span\">
								<span class=\"tip\" title=\"The coverage information at this position\">
										<span class=\"tippy\">Coverage Information</span>
									</span>
								</th>
							</tr>
							<tr>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"The sequence (contig) name in the reference genome FASTA\">
										<span class=\"tippy\">Sequence Name</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"The one-based position in the reference genome\">
										<span class=\"tippy\">Position</span>
									</span>
								</th>
								<th class=\"col_small\">
									<span class=\"tip\" title=\"Name of the Gene\">
										<span class=\"tippy\">Gene Name</span>
									</span>
								</th>
								<th class=\"col_med\">
								<span class=\"tip\" title=\"The reference base(s)\">
										<span class=\"tippy\">Reference</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"Variant allele\">
										<span class=\"tippy\">Variant</span>
									</span>
								</th>
								<th class=\"col_small\">
									<span class=\"tip\" title=\"Frequency of the variant allele\">
										<span class=\"tippy\">VarFreq</span>
									</span>
								</th>
								<th class=\"col_med\">
								<span class=\"tip\" title=\"P-value\">
										<span class=\"tippy\">P-value</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"The total reads covering the position.\">
										<span class=\"tippy\">Total Coverage</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"The number of reads covering the reference allele\">
										<span class=\"tippy\">Reference Coverage</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"The number of reads covering the variant allele\">
										<span class=\"tippy\">Variant Coverage</span>
									</span>
								</th>
							</tr>
						</thead>
						<tbody>
";
#echo "<tr><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td></tr>\n";
run python ${DIRNAME}/parse_variants_dibayes.py  ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/SNP_variants.vcf.gz ${TSP_FILEPATH_PLUGIN_DIR}/dibayes_out/SNP_variants.xls;
echo -n \
"						</tbody>
                    </table>
					<div class=\"spacer\"></div>
                    <br>
                    <input value=\"Export displayed table as CSV\" type=\"button\" onclick=\"\$('#dibayestable').table2CSV({
                    header:['Sequence Name', 'Position', 'Gene Name', 'Reference', 'Variant', 'VarFreq','P-value','Total Coverage','Reference Coverage', 'Variant Coverage']})\">
                </div>
            </div>
";
}
