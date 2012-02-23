#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

print_html_variant_summary_table()
{
# NB: no shell variable checking
local BARCODE_VARIANT_REPORT=${1};
local DIV1_ID="VariantSummary";
local H2="<h2>Variant Summary</h2>";
if [ ! -z ${BARCODE_IDX} ]; then # BARCODE
	DIV1_ID="${DIV1_ID}${BARCODE_SEQ}";
	H2="<h2>Variant Summary (${BARCODE_ID})</h2>";
fi

echo -n \
"            <div id=\"${DIV1_ID}\" name=\"${DIV1_ID}\" class=\"report_block\">
		${H2}
                <div>
                    <br/>
";
run "python ${DIRNAME}/scripts/table_variant_summary.py \
    --vcf-file=${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_VCF_GZ_VARIANTS} \
    --results-json-file=${TSP_FILEPATH_PLUGIN_DIR}/results.json";
# Should e write a link to the barcode report?
if [[ ! -z ${BARCODE_IDX} ]] && [[ "1" == ${BARCODE_VARIANT_REPORT} ]]; then 
echo -n \
"                </div>
                <br/>
                <h3 class=\"report_block\"><a href=\"${BARCODE_SEQ}/${PLUGINNAME}.html\">Variant Report (${BARCODE_ID})</a></h3>
            </div>
";
else
echo \
'                </div>
            </div>
';
fi
}
