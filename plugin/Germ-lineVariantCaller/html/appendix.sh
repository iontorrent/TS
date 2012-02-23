#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

print_html_appendix()
{
echo -n \
"            <div id=\"Appendix\" class=\"report_block\">
                <h2>Appendix</h2>
                <div>
					<br/>
					Here we provide a collection of graphs to help explore the quality of the observed variant calls. 
					Note that these plots are most useful when a large number (>1000) of variants are present. 
					<br/>
					<br/>
                    <table class=\"noheading\">
                        <tr><td>
							<span class=\"tip\" title=\"The fraction of reads observing the variants across variant positions\"><span class=\"tippy\">
								<a href=\"variantFractions.png\">Fraction of reads with a variant at variant positions</a>
							</span></span>
						</td></tr> 
                        <tr><td>
							<span class=\"tip\" title=\"The deletion length distrubition for heterozygous deletions\"><span class=\"tippy\">
								<a href=\"indelLength.Deletions.Het.png\">Heterozygous deletion length distribution</a>
							</span></span>
						</td></tr>
                        <tr><td>
							<span class=\"tip\" title=\"The deletion length distrubition for homozygous deletions\"><span class=\"tippy\">
								<a href=\"indelLength.Deletions.Hom.png\">Homozygous deletion length distribution</a>
							</span></span>
						</td></tr>
                        <tr><td>
							<span class=\"tip\" title=\"The insertion length distrubition for heterozygous insertions\"><span class=\"tippy\">
								<a href=\"indelLength.Insertions.Het.png\">Heterozygous insertion length distribution</a>
							</span></span>
						</td></tr>
                        <tr><td>
							<span class=\"tip\" title=\"The insertion length distrubition for homozygous insertions\"><span class=\"tippy\">
								<a href=\"indelLength.Insertions.Hom.png\">Homozygous insertion length distribution</a>
							</span></span>
						</td></tr>
                        <tr><td>
							<span class=\"tip\" title=\"The base substitution frequency for all possible alleles\"><span class=\"tippy\">
								<a href=\"snpInfo.png\">SNP mutation info</a>
							</span></span>
						</td></tr>
                        <tr><td>
							<span class=\"tip\" title=\"The sequencing strand distributions across variants\"><span class=\"tippy\">
								<a href=\"strandFrequency.png\">Strand frequencies at variant positions</a>
							</span></span>
						</td></tr>
						<tr><td>
							<span class=\"tip\" title=\"The coverage distributions at variant positons\"><span class=\"tippy\">
								<a href=\"coverage.png\">Total coverage at variant positions</a>
							</span></span>
						</td></tr>
                        <tr><td>
							<span class=\"tip\" title=\"The variant allele coverage distributions at variant positions\"><span class=\"tippy\">
								<a href=\"variantCoverage.png\">Variant coverage at variant positions</a>
							</span></span>
						</td></tr>
                    </table>
                </div>
            </div> <!--  end of plots-->";
}
