<?php
header('Content-Type: application/zip');
header('Content-Disposition: attachment; filename="molecularCoverageAnalysisReport.zip"');
header('Content-Transfer-Encoding: binary');
ob_clean();
flush();

$fp = popen('zip -9qr - * -i barcodes.json startplugin.json drmaa_stdout.txt  local_parameters.json \*.bcmatrix.xls \*.bc_summary.xls \*.pdf \*.amplicon.cov.xls ', 'r');

while (!feof($fp)) {
    echo fread($fp, 8192);
    ob_flush();
    flush();
}

pclose($fp);
?>
