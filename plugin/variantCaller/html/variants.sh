#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

write_html_variants()
{
local EXCOL=0
if [ -n "$2" ]; then
  EXCOL=$2
fi
local EXTITLE=""
echo \
"<div id=\"VariantCalls\" class=\"report_block\">
 <h2>Variant Calls</h2>
 <div class=\"demo_jui\">
  <table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" class=\"noheading\" id=\"dibayestable\" style=\"word-wrap:break-word\">
   <thead>
    <tr>
      <th style=\"width:35px !important\">
       <span class=\"tip\" title=\"Click the IGV link to open the variant in Intergrated Genome Viewer to see all reads covering the variant.\"><span class=\"tippy\">View</span></span>
      <th style=\"width:70px !important\">
       <span class=\"tip\" title=\"The chromosome (or contig) name in the reference genome.\"><span class=\"tippy\">Chrom</span></span>
      </th>
      <th style=\"width:84px !important\">
       <span class=\"tip\" title=\"The one-based position in the reference genome.\"><span class=\"tippy\">Position</span></span>
      </th>
      <th style=\"width:75px !important\">
       <span class=\"tip\" title=\"Gene symbol for the gene where the variant is located. This value is not available (N/A) if no target regions were defined (full genome was used).\"><span class=\"tippy\">Gene Sym</span></span>
      </th>
      <th style=\"width:75px !important\">
       <span class=\"tip\" title=\"Name of the target region where the variant is located. This value is not available (N/A) if no target regions were defined (full genome was used).\"><span class=\"tippy\">Target ID</span></span>
      </th>
      <th style=\"width:40px !important\">
       <span class=\"tip\" title=\"Type of variantion detected (SNP/INDEL).\"><span class=\"tippy\">Type</span></span>
      </th>
      <th style=\"width:40px !important\">
       <span class=\"tip\" title=\"Assigned ploidy of the variation: Homozygous (Hom), Heterozygous (Het) or No Call (NC).\"><span class=\"tippy\">Ploidy</span></span>
      </th>
      <th style=\"width:35px !important\">
       <span class=\"tip\" title=\"The reference base(s).\"><span class=\"tippy\">Ref</span></span>
      </th>
      <th style=\"width:50px !important\">
       <span class=\"tip\" title=\"Variant allele base(s).\"><span class=\"tippy\">Variant</span></span>
      </th>";
if [ $EXCOL -eq 0 ]; then
echo \
"      <th style=\"width:72px !important\">
       <span class=\"tip\" title=\"Frequency of the variant allele.\"><span class=\"tippy\">Var Freq</span></span>
      </th>
      <th style=\"width:72px !important\">
       <span class=\"tip\" title=\"Estimated probability that the variant could be produced by chance.\"><span class=\"tippy\">P-value</span></span>
      </th>
      <th style=\"width:72px !important\">
       <span class=\"tip\" title=\"The total reads covering the position.\"><span class=\"tippy\">Cov</span></span>
      </th>
      <th style=\"width:72px !important\">
       <span class=\"tip\" title=\"The number of reads covering the reference allele.\"><span class=\"tippy\">Ref Cov</span></span>
      </th>
      <th style=\"width:100% !important\">
       <span class=\"tip\" title=\"The number of reads covering the variant allele.\"><span class=\"tippy\">Var Cov</span></span>
      </th>";
else
EXTITLE="HotSpot ID"
echo \
"      <th style=\"width:60px !important\">
       <span class=\"tip\" title=\"Frequency of the variant allele.\"><span class=\"tippy\">Var Freq</span></span>
      </th>
      <th style=\"width:71px !important\">
       <span class=\"tip\" title=\"Estimated probability that the variant could be produced by chance.\"><span class=\"tippy\">P-value</span></span>
      </th>
      <th style=\"width:55px !important\">
       <span class=\"tip\" title=\"The total reads covering the position.\"><span class=\"tippy\">Cov</span></span>
      </th>
      <th style=\"width:55px !important\">
       <span class=\"tip\" title=\"The number of reads covering the reference allele.\"><span class=\"tippy\">Ref Cov</span></span>
      </th>
      <th style=\"width:55px !important\">
       <span class=\"tip\" title=\"The number of reads covering the variant allele.\"><span class=\"tippy\">Var Cov</span></span>
      </th>
      <th style=\"width:100% !important\">
       <span class=\"tip\" title=\"$EXTITLE for one or more starting locations matching the identified variant.\"><span class=\"tippy\">$EXTITLE</span></span>
      </th>";
EXTITLE=",'$EXTITLE'"
fi
echo \
"     </tr>
    </thead>
   <tbody>";
if [ -f "$1" ]; then
    cat "$1" 2> /dev/null;
fi
echo \
"   </tbody>
  </table>
  <br>
  <input value=\"Export table as tab-delimited file\" type=\"button\" onclick=\"\$('#dibayestable').table2CSV({
header:['Chromosome','Position','Gene Sym','Target ID','Var Type','Ploidy','Ref','Variant','Var Freq','P-value','Coverage','Ref Cov','Var Cov'${EXTITLE}]})\">
 </div>
</div>";
}
