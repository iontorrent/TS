#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

barcode () {
for barcodes in `cat ${TSP_FILEPATH_BARCODE_TXT} | grep "^barcode"`
do
    BARCODE=`echo $barcodes | awk 'BEGIN{FS=","} {print $2}'`;
    if [ -d ${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE} ]; then
        parse_variants "${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE}" ${PLUGINNAME} "${TSP_URLPATH_PLUGIN_DIR}/${BARCODE}" "${BARCODE}_sorted.bam" "${BARCODE}_sorted_flowspace.bam";
    fi
done


mkdir -vp "${TSP_FILEPATH_PLUGIN_DIR}/js";
cp -v ${DIRNAME}/js/*.js "${TSP_FILEPATH_PLUGIN_DIR}/js/.";
mkdir -vp "${TSP_FILEPATH_PLUGIN_DIR}/css";
cp -v ${DIRNAME}/css/*.css "${TSP_FILEPATH_PLUGIN_DIR}/css/.";


echo -n '' > "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo '<html>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
print_html_head >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e '\t<body>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";

echo -e "\t\t\t<div id=\"BarcodeList\" class=\"report_block\"/>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t\t<h2>Barcode Variant Reports</h2>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t\t<div>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t\t\t<br/>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t\t\t<table class=\"noheading\">" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t\t\t\t<tr>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";

echo -e "\t\t\t\t\t\t\t<th><span class=\"tip\" title=\"The name for the individual sequence\"><span class=\"tippy\">Barcode Identifier</span></span></th>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t\t\t\t\t<th><span class=\"tip\" title=\"The sequence of bases defining the barcode\"><span class=\"tippy\">Barcode Sequence</span></span></th>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t\t\t\t\t<th><span class=\"tip\" title=\"The individual variant report for this barcode\"><span class=\"tippy\">Barcode Variant Report</span></span></th>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t\t\t\t</tr>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";

for barcodes in `cat ${TSP_FILEPATH_BARCODE_TXT} | grep "^barcode"`
do
    BARCODE=`echo $barcodes | awk 'BEGIN{FS=","} {print $2}'`;
    BARCODE_SEQ=`echo $barcodes | awk 'BEGIN{FS=","} {print $3}'`;
    if [ -d ${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE} ]; then
	echo -e "\t\t\t\t\t\t<tr>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	echo -e "\t\t\t\t\t\t\t<td>${BARCODE}</td>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
        echo -e "\t\t\t\t\t\t\t<td>${BARCODE_SEQ}</td>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	echo -e "\t\t\t\t\t\t\t<td><a href=\"${TSP_URLPATH_PLUGIN_DIR}/${BARCODE}/${PLUGINNAME}.html\">Variant Report</a></td>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
	echo -e "\t\t\t\t\t\t</tr>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
    fi
done
echo -e "\t\t\t\t\t</table>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t\t\t<br/>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t\t</div>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -e "\t\t\t</div>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";


print_html_end_javascript >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";
echo -n '</body></html>' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}_block.html";


# Plugin Main page

# HTML tag
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
            <h1><center>AmpliSeq Cancer Variant Caller</center></h1>
            <h5><center>This variant caller is specific to the Ampliseq Cancer Panel Kit. 
            For general variant calling please use the Germ-line Variant Caller. </center></h5>
' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";

echo -e "\t\t\t<div id=\"BarcodeList\" class=\"report_block\"/>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t\t<h2>Barcode Variant Reports</h2>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t\t<div>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t\t\t<br/>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t\t\t<table class=\"noheading\">" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t\t\t\t<tr>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";

echo -e "\t\t\t\t\t\t\t<th><span class=\"tip\" title=\"The name for the individual sequence\"><span class=\"tippy\">Barcode Identifier</span></span></th>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t\t\t\t\t<th><span class=\"tip\" title=\"The sequence of bases defining the barcode\"><span class=\"tippy\">Barcode Sequence</span></span></th>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t\t\t\t\t<th><span class=\"tip\" title=\"The individual variant report for this barcode\"><span class=\"tippy\">Barcode Variant Report</span></span></th>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t\t\t\t</tr>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
        
for barcodes in `cat ${TSP_FILEPATH_BARCODE_TXT} | grep "^barcode"`
do      
    BARCODE=`echo $barcodes | awk 'BEGIN{FS=","} {print $2}'`;
    BARCODE_SEQ=`echo $barcodes | awk 'BEGIN{FS=","} {print $3}'`;
    if [ -d ${TSP_FILEPATH_PLUGIN_DIR}/${BARCODE} ]; then
        echo -e "\t\t\t\t\t\t<tr>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
        echo -e "\t\t\t\t\t\t\t<td>${BARCODE}</td>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
        echo -e "\t\t\t\t\t\t\t<td>${BARCODE_SEQ}</td>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
        echo -e "\t\t\t\t\t\t\t<td><a href=\"${TSP_URLPATH_PLUGIN_DIR}/${BARCODE}/${PLUGINNAME}.html\">Variant Report</a></td>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
        echo -e "\t\t\t\t\t\t</tr>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
    fi
done
echo -e "\t\t\t\t\t</table>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t\t\t<br/>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t\t</div>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
echo -e "\t\t\t</div>" >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";


# Ending Javascript
print_html_end_javascript >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Footer
print_html_footer >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";
# Inner division (end)
echo -n \
'           </div>
    </body>
</html>
' >> "${TSP_FILEPATH_PLUGIN_DIR}/${PLUGINNAME}.html";

}
