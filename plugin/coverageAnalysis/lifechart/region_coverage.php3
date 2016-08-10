<?php
$bbcfile = $_GET['bbcfile'];
$annofile = $_GET['annofile'];
$annofields = $_GET['annofields'];
$annotitles = $_GET['annotitles'];
$chrom = $_GET['chrom'];
$pos_srt = intval($_GET['pos_srt']);
$pos_end = intval($_GET['pos_end']);
$numbins = intval($_GET['maxrows']);
$srt_bin = intval($_GET['srt_bin']);
$end_bin = intval($_GET['end_bin']);
$outfile = $_GET['outfile'];
$options = $_GET['options'];

$header = "-H chrom,start,end,fwd_basereads,rev_basereads,fwd_trg_basereads,rev_trg_basereads";
if( $options == "-bl" ) {
  $header = "";
}

if( $chrom != "" ) {
  if( $pos_srt > 0 ) {
    $chrom = "$chrom:$pos_srt";
    if( $pos_end > 0 ) $chrom = "$chrom-$pos_end";
  } else if( $pos_end > 0 ) {
    $chrom = "$chrom:1-$pos_end";
  }
  $chrom = "\"$chrom\"";
}
$aflds = "";
if( $annofile != "" && $annofields != "" ) {
  $aflds = "-R \"$annofile\" -A \"$annofields\"";
  $header = "$header,$annotitles";
}

if( $outfile != "" && $outfile != "-" ) {
  header("Cache-Control: ");
  header("Content-type: text/csv");
  header("Content-disposition: attachment;filename=$outfile");
}
# check for local-version override: if fail use system version
$bbctools = '../bin/bbctools';
if( !file_exists($bbctools) ) {
  $bbctools = "bbctools";
}
system("$bbctools view $options $aflds -N $numbins -S $srt_bin -E $end_bin $header \"$bbcfile\" $chrom");
?>
