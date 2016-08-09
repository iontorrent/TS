<?php
$dataFile = $_GET['dataFile'];
$rows = $_GET['rows'];

$dtfh = fopen($dataFile,"r");
if( $dtfh == 0 || feof($dtfh) ) {
  print 'Could open data file '.$dataFile;
  exit(0);
}
# This an html script but cannot use .html/.php without it being seen by plugin status table.
# Using other mime types result in page being simply viewed as text in the browser.
$taqmanFile = "taqManRequest.htm";
$tqfh = fopen($taqmanFile,"w");
if( $tqfh == 0 ) {
  fclose($dtfh);
  print 'ERROR: Unable to write to file '.$taqmanFile;
  exit(0);
}

fwrite($tqfh,"<html><h2>Re-directing to Life Technologies TaqMan Genotyping Assay Search...</h2>\n<div style='display:none'>\n");
fwrite($tqfh,"<form id='search_form' action='https://www.thermofisher.com/order/genome-database/MultipleTargets' method='POST' enctype='multipart/form-data'>");
fwrite($tqfh,"<input name='productTypeSelect' value='genotyping'/>\n");
fwrite($tqfh,"<input name='targetTypeSelect' value='snp_all'/>\n");
fwrite($tqfh,"<input name='species' value='' disabled='disabled'>\n");
fwrite($tqfh,"<textarea name='batchText'>\n");
fwrite($tqfh,"##fileformat=TVCF-TaqMan\n");
fwrite($tqfh,"#CHROM\tPOS\tID\tREF\tALT\n");

$recNum = 0;
$line = "";
$i = 0;
while( ($j = strpos( $rows, ",", $i )) !== false ) {
  $r = substr( $rows, $i, $j-$i ) + 2;
  $i = $j+1;
  while( !feof($dtfh) )
  {
    $line = fgets($dtfh);
    ++$recNum;
    if( $recNum == $r ) break;
  }
  if( $recNum != $r ) break;
  $fields = explode("\t",$line);
  //use existing ref and alt and coord
  fwrite($tqfh,"$fields[0]\t$fields[14]\t.\t$fields[2]\t$fields[7]\n");
}
fclose($dtfh);

fwrite($tqfh,"</textarea>\n");
fwrite($tqfh,"<input name='CID' value='ION2TAQMAN'/>\n");
fwrite($tqfh,"<input name='btnSearch' value='y' hidden='y'/>\n");
fwrite($tqfh,"<input type='Submit'/>\n");
fwrite($tqfh,"</form></div>\n");
fwrite($tqfh,"<script>document.getElementById('search_form').submit();</script>\n");
fwrite($tqfh,"<html>\n");
fclose($tqfh);
header("Location: ".$taqmanFile);
?>
