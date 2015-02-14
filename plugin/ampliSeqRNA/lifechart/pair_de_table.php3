<?php
$data = $_GET['data'];
$threshold = $_GET['threshold'];
$thresAction = $_GET['thresAction'];
$bc1 = $_GET['bc1'];
$bc2 = $_GET['bc2'];
$bccount = $_GET['bccount'];
$fname = 'ampliSeqRNA.DE.'.$bc1.'-'.$bc2.'.xls';

header("Content-Type: text/csv");
header("Content-Disposition: attachment; filename=\"$fname\"");
ob_clean();
flush();

$fp = popen("../scripts/tableDE.pl -B '$bc1,$bc2' -r -N 1000000 -S RPM -T $threshold -A $thresAction -a -M $bccount '$data'", 'r');
while (!feof($fp)) {
  echo fread($fp, 8192);
  ob_flush();
  flush();
}
pclose($fp);
?>
