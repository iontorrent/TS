<?php
$oktoWriteFile = true;
$options = $_GET['options'];
$dataFile = $_GET['dataFile'];
$chrom = $_GET['chrom'];
$gene = $_GET['gene'];
$covmin = $_GET['covmin'];
$covmax = $_GET['covmax'];
$maxrows = intval($_GET['maxrows']);
$clipleft = $_GET['clipleft'];
$clipright = $_GET['clipright'];
$numrec = intval($_GET['numrec']);
if( $dataFile == "" ) $dataFile = '-';
if( $chrom == "" ) $chrom = '-';
if( $gene == "" ) $gene = '-';
$tobed = strpos( $options, '-b' ) !== false ;
$allout = $tobed || (strpos( $options, '-a' ) !== false) ;
if( $allout )
{
  $subtable = preg_replace( '/\.xls$/', '', $dataFile ).".range.".($tobed ? "bed" : "xls");
  $filename = preg_replace( '/^.*\//', '', $subtable );
  $redirect = '';
  if( $tobed || !$oktoWriteFile ) {
    header("Content-type: text/csv");
    header("Content-disposition: attachment;filename=$filename");
  } else {
    # allows it to be openned in excel automatically
    header("Location: ../".$filename);
    $redirect = "> \"$subtable\"";
  }
  system("../scripts/target_coverage.pl $options \"$dataFile\" \"$chrom\" \"$gene\" $covmin $covmax $maxrows $clipleft $clipright $numrec $redirect");
}
else
{
  system("../scripts/target_coverage.pl $options \"$dataFile\" \"$chrom\" \"$gene\" $covmin $covmax $maxrows $clipleft $clipright $numrec");
}
?>
