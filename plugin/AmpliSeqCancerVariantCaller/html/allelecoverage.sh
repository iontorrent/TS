#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


allele_coverage()
{

echo \
"            <div id=\"AlleleCoverage\" class=\"report_block\">
                <h2>Allele Counts for all loci within the Cancer Panel</h2>
                <div calss=\"demo_jui\">
					<br/>
					<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" class=\"noheading\" id=\"alleletable\" style=\"word-wrap:break-word\">
						<thead>
							<tr>
								<th class=\"col_large\">
								</th>
								<th class=\"col_med\">
								</th>
								<th class=\"col_small\">
								</th>
								<th class=\"col_med\">
								</th>
								<th class=\"col_med\">
								</th>
								<th colspan=\"5\" class=\"col_span\">
								<span class=\"tip\" title=\"The coverage information at this position\">
										<span class=\"tippy\">Coverage Information</span>
									</span>
								</th>
							</tr>
							<tr>
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
								<th class=\"col_small\">
									<span class=\"tip\" title=\"Name of the gene\">
										<span class=\"tippy\">Gene Name</span>
									</span>
								</th>
								<th class=\"col_med\">
								<span class=\"tip\" title=\"The reference base(s)\">
										<span class=\"tippy\">Reference</span>
									</span>
								</th>
                                                                <th class=\"col_small\">
								<span class=\"tip\" title=\"The total reads covering the position.\">
										<span class=\"tippy\">Total Coverage</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"Number of reads calling A\">
										<span class=\"tippy\">A</span>
									</span>
								</th>
								<th class=\"col_small\">
									<span class=\"tip\" title=\"Number of reads calling C\">
										<span class=\"tippy\">C</span>
									</span>
								</th>
								<th class=\"col_small\">
									<span class=\"tip\" title=\"Number of reads calling G\">
										<span class=\"tippy\">G</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"Number of reads calling T\">
										<span class=\"tippy\">T</span>
									</span>
								</th>
								<th class=\"col_small\">
								<span class=\"tip\" title=\"Number of reads calling N\">
										<span class=\"tippy\">N</span>
									</span>
								</th>
							</tr>
						</thead>
						<tbody>
";
#echo "<tr><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td><td>A</td></tr>\n";
run python ${DIRNAME}/print_allele_counts.py ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_COV} ${TSP_FILEPATH_PLUGIN_DIR}/allele_counts.xls;
echo -n \
"						</tbody>
                    </table>
					<div class=\"spacer\"></div>
                <br>
                    <input value=\"Export displayed table as CSV\" type=\"button\" onclick=\"\$('#alleletable').table2CSV({header:['Sequence Name', 'Position', 'Gene Name', 'Reference', 'Total Coverage', 'A','C','G','T', 'N']})\">
                </div>
            </div>
";
}
