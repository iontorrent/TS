#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


write_html_chromo_variants()
{
local EXCOL=0
if [ -n "$2" ]; then
  EXCOL=$2
fi
local EXTITLE=""
local TWIDTH=700
if [ $EXCOL -ne 0 ]; then
  TWIDTH=700
fi
echo \
"<div id=\"ChromosomeVariants\" class=\"report_block\">
 <h2>Variant Calls Summary</h2>
 <div class=\"demo_jui\" style=\"width:${TWIDTH}px !important;margin-left:auto;margin-right:auto\">
  <table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" border=\"1\" class=\"noheading\" id=\"chromstatstable\" style=\"word-wrap:break-word\">
   <thead>
    <tr>
      <th style=\"width:90px !important\">
       <span class=\"tip\" title=\"The chromosome (or contig) name in the reference genome.\"><span class=\"tippy\">Chrom</span></span>
      </th>
      <th style=\"width:90px !important; text-align:center\">
       <span class=\"tip\" title=\"The total number of variants called in (the target regions of) the reference.\"><span class=\"tippy\">Variants</span></span>
      </th>
      <th style=\"width:90px !important; text-align:center\">
       <span class=\"tip\" title=\"The total number of heterozygous SNPs called in (the target regions of) the reference.\"><span class=\"tippy\">Het SNPs</span></span>
      </th>
      <th style=\"width:90px !important; text-align:center\">
       <span class=\"tip\" title=\"The total number of homozygous SNPs called in (the target regions of) the reference.\"><span class=\"tippy\">Hom SNPs</span></span>
      </th>
      <th style=\"width:90px !important; text-align:center\">
       <span class=\"tip\" title=\"The total number of heterozygous INDELs called in (the target regions of) the reference.\"><span class=\"tippy\">Het INDELs</span></span>
      </th>";
if [ $EXCOL -eq 0 ]; then
echo \
"      <th style=\"width:100% !important; text-align:center\">
       <span class=\"tip\" title=\"The total number of homozygous INDELs called in (the target regions of) the reference.\"><span class=\"tippy\">Hom INDELs</span></span>
      </th>";
else
EXTITLE="HotSpots"
echo \
"      <th style=\"width:90px !important\">
       <span class=\"tip\" title=\"The total number of homozygous INDELs called in (the target regions of) the reference.\"><span class=\"tippy\">Hom INDELs</span></span>
      </th>
      <th style=\"width:100% !important\">
       <span class=\"tip\" title=\"The total number of variants identified with one or more $EXTITLE.\"><span class=\"tippy\">$EXTITLE</span></span>
      </th>";
EXTITLE=",'$EXTITLE'"
fi
echo "     </tr>"
echo "    </thead>"
echo "   <tbody>"
if [ -f "$1" ]; then
    cat "$1" 2> /dev/null;
fi
echo \
"   </tbody>
  </table>
  <br>
  <input value=\"Export table as tab-delimited file\" type=\"button\" onclick=\"\$('#chromstatstable').table2CSV({
header:['Chromosome','Variants','Het SNPs','Hom SNPs','Het INDELs','Hom INDELs'${EXTITLE}]})\">
 </div>
</div>";
}
