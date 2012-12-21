<?php
$bbcfile = $_GET['bbcfile'];
$cbcfile = $_GET['cbcfile'];
$chrom = $_GET['chrom'];
$pos_srt = intval($_GET['pos_srt']);
$pos_end = intval($_GET['pos_end']);
$numbins = intval($_GET['maxrows']);
if( $chrom == "" ) $chrom = '-';
$cbcopt = "";
if( $cbcfile != "" ) $cbcopt = "-C \"$cbcfile\"";
system("../scripts/bbcRegionView.pl -b $numbins $cbcopt \"$bbcfile\" \"$chrom\" $pos_srt $pos_end");
?>
