<?php
$bbcfile = $_GET['bbcfile'];
$cbcfile = $_GET['cbcfile'];
$chrom = $_GET['chrom'];
$pos_srt = intval($_GET['pos_srt']);
$pos_end = intval($_GET['pos_end']);
$numbins = intval($_GET['maxrows']);
$srt_bin = intval($_GET['srt_bin']);
$end_bin = intval($_GET['end_bin']);
$outfile = $_GET['outfile'];
$options = $_GET['options'];

if( $chrom == "" ) $chrom = '-';
$cbcopt = "";
if( $cbcfile != "" ) $cbcopt = "-C \"$cbcfile\"";

if( $outfile != "" && $outfile != "-" ) {
  header("Cache-Control: ");
  header("Content-type: text/csv");
  header("Content-disposition: attachment;filename=$outfile");
}
system("../scripts/bbcRegionView.pl $options -b $numbins -s $srt_bin -e $end_bin $cbcopt \"$bbcfile\" \"$chrom\" $pos_srt $pos_end");
?>
