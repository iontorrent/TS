<?php
$dataFile = $_GET['dataFile'];
$rows = $_GET['rows'];

$dtfh = fopen($dataFile,"r");
if( $dtfh == 0 || feof($dtfh) ) {
  print 'Could open data file '.$dataFile;
  exit(0);
}
$tableFile = "subtable.xls";
$tbfh = fopen($tableFile,"w");
if( $tbfh == 0 ) {
  fclose($dtfh);
  print 'Could write to file '.$tableFile;
  exit(0);
}
// keep header line
$line = fgets($dtfh);
fwrite($tbfh,$line);

$recNum = 0;
$i = 0;
while( ($j = strpos( $rows, ",", $i )) !== false ) {
  $r = substr( $rows, $i, $j-$i ) + 1;
  $i = $j+1;
  while( !feof($dtfh) )
  {
    $line = fgets($dtfh);
    ++$recNum;
    if( $recNum == $r ) break;
  }
  if( $recNum != $r ) break;
  fwrite($tbfh,$line);
}
fclose($dtfh);
fclose($tbfh);
header("Location: ".$tableFile);
?>
