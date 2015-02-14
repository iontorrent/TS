<?php
$filename = $_GET['filename'];
$outfile = $_GET['outfile'];
$numlines = intval($_GET['numlines']);
$startline = intval($_GET['startline']);
$numfields = intval($_GET['numfields']);
$startfield = intval($_GET['startfield']);
$headlines = intval($_GET['headlines']);
$bedcoords = intval($_GET['bedcoords']);
$binsize = floatval($_GET['binsize']);

# line and fields positions are 1-based
if( $startline < 1 ) $startline = 1;
if( $startfield < 1 ) $startfield = 1;
if( $binsize < 1 ) $binsize = 1;

$skiphead = 0;
if( $headlines < 0 ) {
  $headlines = -$headlines;
  $skiphead = 1;
}

if( !file_exists($filename) ) {
  echo "Cannot open file '$filename'.";
  exit(1);
}

$nl = "<br/>\n";
if( $outfile != "" && $outfile != "-" ) {
  header("Cache-Control: ");
  header("Content-type: text/csv");
  header("Content-disposition: attachment;filename=$outfile");
  $nl = "\n";
}

$binCnt = 0;
$binMod = 0;
$recsout = 0;
$f = fopen($filename,"r");
while( !feof($f) ) { 
  $line = preg_replace('/\r?\n$/', '', fgets($f));
  if( $ishead = ($headlines > 0) ) {
    --$headlines;
    if( $skiphead ) continue;
  } elseif( $startline > 1 ) {
    --$startline;
    continue;
  } elseif( $numlines > 0 ) {
    if( ++$recsout > $numlines ) break;
  }
  # either a header or content line is reduced to specified fields
  $fields = explode("\t",$line);
  $nflds = sizeof($fields);
  if( $nflds < $startfield ) continue;
  # skip or accumulate bin output
  if( $ishead || $binsize == 1 ) {
    $recflds = $fields;
  } else {
    $binMod += 1.0;
    if( ++$binCnt == 1 ) {
      $recflds = $fields;
    } else {
      for( $j = 2; $j < $nflds; ++$j ) {
        $recflds[$j] += $fields[$j];
      }
    }
    if( $binMod < $binsize ) continue;
    if( $binCnt > 1 ) $recflds[0] .= " - ".$fields[0];
    $binMod -= $binsize;
    $binCnt = 0;
  }
  $endfield = $startfield + $numfields - 1;
  if( $numfields < 1 || $endfield > $nflds ) $endfield = $nflds;
  print $recflds[$startfield-1];
  for( $i = $startfield, $j = 0; $i < $endfield; ++$i,++$j ) {
    if( $bedcoords && $j == 0 ) --$recflds[$i];
    print "\t".$recflds[$i];
  }
  print $nl;
}
fclose($f);
# check for last record in binning mode in case of sumup ound-off error
if( $binCnt > 0 ) {
  if( $binCnt > 1 ) $recflds[0] .= " - ".$fields[0];
  $endfield = $startfield + $numfields - 1;
  if( $numfields < 1 || $endfield > $nflds ) $endfield = $nflds;
  print $recflds[$startfield-1];
  for( $i = $startfield, $j = 0; $i < $endfield; ++$i,++$j ) {
    if( $bedcoords && $j == 0 ) --$recflds[$i];
    print "\t".$recflds[$i];
  }
  print $nl;
}
?>

