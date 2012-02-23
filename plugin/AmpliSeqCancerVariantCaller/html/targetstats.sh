#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


target_stats()
{

echo \
"            <div id=\"TargetStats\" class=\"report_block\">
                <h2>On Target Stats</h2>
                <div calss=\"demo_jui\">
					<br/>
					<table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" class=\"noheading\" id=\"targettable\" style=\"word-wrap:break-word\">
						<thead>
							<tr>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"Total number of reads\">
										<span class=\"tippy\">Total Reads</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"The number of mapped reads\">
										<span class=\"tippy\">Mapped Reads</span>
									</span>
								</th>
								<th class=\"col_med\">
								<span class=\"tip\" title=\"The number of reads in target regions\">
										<span class=\"tippy\">On Target Reads</span>
									</span>
								</th>
                                                                <th class=\"col_med\">
								<span class=\"tip\" title=\"% of reads on target\">
										<span class=\"tippy\">% on target</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"% of reads off target\">
										<span class=\"tippy\">% off target</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"% of unmapped reads\">
										<span class=\"tippy\">% unmapped</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"Total number of covered bases\">
										<span class=\"tippy\">Total coverage</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"Average depth of coverage\">
										<span class=\"tippy\">Average coverage</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"% of Target bases with 0 coverage\">
										<span class=\"tippy\">% of uncovered bases</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"% of Target bases with at least 1x coverage\">
										<span class=\"tippy\">% of target bases at 1x</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"% of Target bases with at least 10x coverage\">
										<span class=\"tippy\">% of target bases at 10x</span>
									</span>
								</th>
								<th class=\"col_med\">
									<span class=\"tip\" title=\"% of Target bases with at least 100x coverage\">
										<span class=\"tippy\">% of target bases at 100x</span>
									</span>
								</th>
							</tr>
						</thead>
						<tbody>
";
run python ${DIRNAME}/parse_stats.py  ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_STATS} ${TSP_FILEPATH_PLUGIN_DIR}/results.json;
echo -n \
'                                               </tbody>
                    </table>
                                        <div class=\"spacer\"></div>

                </div>
            </div>
';

}
