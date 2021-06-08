<?php
header('Content-Type: application/zip');
header('Content-Disposition: attachment; filename="coverageAnalysisReport.zip"');
header('Content-Transfer-Encoding: binary');
ob_clean();
flush();

$fp = popen('zip -9qr - * -i barcodes.json startplugin.json drmaa_stdout.txt \*.bcmatrix.xls \*.bc_summary.xls \*.pdf \*.chr.cov.xls \*.contig.cov.xls \*.amplicon.cov.xls \*.target.cov.xls -x \*scraper\*', 'r');

while (!feof($fp)) {
    echo fread($fp, 8192);
    ob_flush();
    flush();
}

pclose($fp);
?>
