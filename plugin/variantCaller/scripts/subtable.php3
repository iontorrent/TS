<?php
$dataFile = $_GET['dataFile'];
if (isset($_GET['keyrows'])) {
  $keyrows = $_GET['keyrows'];
} else {
  $keyrows = $_GET['rows'];
}


$i = 0;
$key = array();
while( ($j = strpos( $keyrows, ",", $i )) !== false ) {
  $r = substr( $keyrows, $i, $j-$i ) + 1;
  $i = $j+1;
  array_push($key, $r);
}
$skey = $key;
sort($skey);
var_dump($key);
var_dump($skey);

$dtfh = fopen($dataFile,"r");
if( $dtfh == 0 || feof($dtfh) ) {
  print 'Could open data file '.$dataFile;
  exit(0);
}
$tableFile = "subtable.xls";
$tbfh = fopen($tableFile,"w");
if( $tbfh == 0 ) {
  fclose($dtfh);
  print 'ERROR: Unable to write to file '.$tableFile;
  exit(0);
}
// keep header line
$line = fgets($dtfh);
fwrite($tbfh,$line);

$recNum = 0;
foreach($skey as $val) {
  while( !feof($dtfh) ) {
    $line = fgets($dtfh);
    ++$recNum;
    if( $recNum == $val ) break;
  }
  if( $recNum != $val) break;
  $arr[$val] = $line;
}
fclose($dtfh);

foreach($key as $val) {
  fwrite($tbfh, $arr[$val]);
}
fclose($tbfh);
header("Location: ".$tableFile);
?>
