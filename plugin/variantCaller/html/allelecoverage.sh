#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

write_html_allele_coverage()
{
echo \
"<div id=\"AlleleCoverage\" class=\"report_block\">
 <h2>Allele Coverage for all bases in HotSpot Regions<sup>&dagger;</sup></h2>
 <div class=\"demo_jui\">
  <table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" class=\"noheading\" id=\"alleletable\" style=\"word-wrap:break-word\">
   <thead>
    <tr>
      <th style=\"width:90px !important\">
       <span class=\"tip\" title=\"The chromosome (or contig) name in the reference genome.\"><span class=\"tippy\">Chrom</span></span>
      </th>
      <th style=\"width:100px !important\">
       <span class=\"tip\" title=\"The one-based position in the reference genome.\"><span class=\"tippy\">Position</span></span>
      </th>
      <th style=\"width:100px !important\">
       <span class=\"tip\" title=\"Name of the target region containing the HotSpot variation site.\"><span class=\"tippy\">Target ID</span></span>
      </th>
      <th style=\"width:110px !important\">
       <span class=\"tip\" title=\"Name of the HotSpot variant (site).\"><span class=\"tippy\">HotSpot ID</span></span>
      </th>
      <th style=\"width:50px !important\">
       <span class=\"tip\" title=\"The reference base(s).\"><span class=\"tippy\">Ref</span></span>
      </th>
      <th style=\"width:60px !important\">
       <span class=\"tip\" title=\"The total reads covering the position, including deletions.\"><span class=\"tippy\">Cov</span></span>
      </th>
      <th style=\"width:55px !important\">
       <span class=\"tip\" title=\"Number of reads calling A.\"><span class=\"tippy\">A</span></span>
      </th>
      <th style=\"width:55px !important\">
       <span class=\"tip\" title=\"Number of reads calling C.\"><span class=\"tippy\">C</span></span>
      </th>
      <th style=\"width:55px !important\">
       <span class=\"tip\" title=\"Number of reads calling G.\"><span class=\"tippy\">G</span></span>
      </th>
      <th style=\"width:55px !important\">
       <span class=\"tip\" title=\"Number of reads calling T.\"><span class=\"tippy\">T</span></span>
      </th>
      <th style=\"width:55px !important\">
       <span class=\"tip\" title=\"Number of forward reads aligned over the reference base that did not produce a base deletion call.\"><span class=\"tippy\">Cov (+)</span></span>
      </th>
      <th style=\"width:55px !important\">
       <span class=\"tip\" title=\"Number of reverse reads aligned over the reference base that did not produce a base deletion call.\"><span class=\"tippy\">Cov (-)</span></span>
      </th>
      <th style=\"width:100% !important\">
       <span class=\"tip\" title=\"Number of reads calling deletion at this base location.\"><span class=\"tippy\">Cov DEL</span></span>
      </th>
     </tr>
    </thead>
   <tbody>
";
if [ -f "$1" ]; then
    cat "$1" 2> /dev/null;
fi
echo -n \
"   </tbody>
  </table>
  <br>
  <input value=\"Export table as tab-delimited file\" type=\"button\" onclick=\"\$('#alleletable').table2CSV({
header:['Chromosome', 'Position', 'Target ID', 'HotSpot ID', 'Ref', 'Coverage', 'A','C','G','T', 'Cov (+)', 'Cov (-)', 'Cov DEL']})\">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <em><sup><b>&dagger;</b></sup>Table does not include coverage for HotSpot insertion variants.</em>
 </div>
</div>
";
}
