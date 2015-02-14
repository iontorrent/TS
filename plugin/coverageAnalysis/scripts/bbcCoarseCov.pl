#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# NOTE: This code relies on the BAM and BED files supplied being correctly ordered.

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create a Coarse Base Coverage (binary CBC) file from a BBC file.
This will consist of binned summed base coverage over every 1000 (N) bases of the genome.
Options allow additonal (text) files to be written for whole-genome coverage.";
my $USAGE = "Usage:\n\t$CMD [options] <BBC file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -l Log progess to STDERR.
  -s <int> Binning size used in binned Summed read counts file. Default: 1000.
  -t Output additional coverage for on-target base reads. (Will produce larger output files.)
  -w <int> Binning size used for Whole genome coverage (-W option). Default: 200.
  -C <file> Optional Chromosome coverage tsv file: On/off target coverage for each chromosome/contig.
  -W <file> Optional Whole genome coverage tsv file: On/off target bins bounded by contigs.
  -O <file> Output file name. Default <BBC file>.cbc (replacing .bbc extension if present).";

my $logopt = 0;
my $cbcsize = 1000;
my $wgnumBins = 200;
my $targetcov = 0;
my $chrcovfile = "";
my $wgncovfile = "";
my $cbcfile = "";

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-O') {$cbcfile = shift;}
  elsif($opt eq '-t') {$targetcov = 1;}
  elsif($opt eq '-C') {$chrcovfile = shift;}
  elsif($opt eq '-W') {$wgncovfile = shift;}
  elsif($opt eq '-w') {$wgnumBins = int(shift);}
  elsif($opt eq '-s') {$cbcsize = int(shift);}
  elsif($opt eq '-l') {$logopt = 1;}
  elsif($opt eq '-h' || $opt eq "?" || $opt eq '--help') {$help = 1;}
  else
  {
    print STDERR "$CMD: Invalid option argument: $opt\n";
    print STDERR "$OPTIONS\n";
    exit 1;
  }
}
my $nargs = scalar @ARGV;
if( $help )
{
  print STDERR "$DESCR\n";
  print STDERR "$USAGE\n";
  print STDERR "$OPTIONS\n";
  exit 1;
}
elsif( $nargs != 1 )
{
  print STDERR "$CMD: Invalid number of arguments.\n";
  print STDERR "$USAGE\n";
  exit 1;
}

my $bbcfile = shift(@ARGV);

$wgnumBins = 200 if( $wgnumBins < 1 );
$cbcsize = 1000 if( $cbcsize < 1 );

if( $targEnd < $targStart )
{
  print STDERR "$CMD: Error: Invalid arguments: <end> value must be >= <start>.\n";
  exit 1;
}

my $haveChrCov = ($chrcovfile ne "" && $chrcovfile ne "-" );
my $haveWgnCov = ($wgncovfile ne "" && $wgncovfile ne "-" );

if( $cbcfile eq "" || $cbcfile eq "-" )
{
  $cbcfile = $bbcfile;
  $cbcfile =~ s/\.bbc$//;
  $cbcfile .= ".cbc";
}

#--------- End command arg parsing ---------

my $header = "chrom\tstart\tend\tfwd_basereads\trev_basereads";
$header .= "\tfwd_trg_basereads\trev_trg_basereads" if( $targetcov );

# open BBCFILE and read contig header string
open( BBCFILE, "<:raw", $bbcfile ) || die "Failed to open BBC file $bbcfile\n";
chomp( my $contigList = <BBCFILE> );
my @chromName = split('\t',$contigList);
my $numChroms = scalar(@chromName);
my @chromSize;
my $genomeSize = 0;
for( my $i = 0; $i < $numChroms; ++$i )
{
  my @fields = split('\$',$chromName[$i]);
  $chromName[$i] = $fields[0];
  $chromSize[$i] = int($fields[1]);
  $genomeSize += $chromSize[$i];
}
print STDERR "Read $numChroms contig names and lengths. Total contig size: $genomeSize\n" if( $logopt );

# reset $cbcsize if genome is small (<100Kb)
if( $genomeSize < 100000 && $cbcsize > 100 ) {
  if( $genomeSize < 1000 ) {
    $cbcsize = 1;
  } elsif( $genomeSize < 10000 ) {
    $cbcsize = 10;
  } else {
    $cbcsize = 100;
  }
  print STDERR "Reset CBC binning size to $cbcsize for small reference ($genomeSize bases)\n" if( $logopt );
}

# for whole genome coverage try to divide bins evenly of genome but at least one per contig
my @chromBins = ((1) x ($numChroms+1));
if( $haveWgnCov )
{
  if( $wgnumBins < $numChroms )
  {
    print STDERR "$CMD: Warning: Too few whole genome bins specified ($wgnumBins). Employing one per contig ($numChroms).\n";
    $wgnumBins = $numChroms;
  }
  # this method asigns 1 bin for every contig then iteratively adds one to the largest remaining contig,
  # dividing the contig by the number of bins to leave virtual contigs for comparison
  my $xbins = $wgnumBins - $numChroms;
  my @echrsize = @chromSize;
  while( $xbins > 0 )
  {
    my $mi = 0;
    my $max = $echrsize[0];
    for( my $j = 1; $j < $numChroms; ++$j )
    {
      if( $echrsize[$j] > $max )
      {
        $max = $echrsize[$j];
        $mi = $j;
      }
    }
    ++$chromBins[$mi];
    $echrsize[$mi] = $chromSize[$mi] / $chromBins[$mi];
    --$xbins;
  }
}

# Chrom Coverage arrays
my @chrCovFwd = ((0) x ($numChroms+1));
my @chrCovRev = ((0) x ($numChroms+1));
my @chrCovFOT = ((0) x ($numChroms+1));
my @chrCovROT = ((0) x ($numChroms+1));

# Whole genome coverage arrays
my @wgnCovFwd = ((0) x $wgnumBins);
my @wgnCovRev = ((0) x $wgnumBins);
my @wgnCovFOT = ((0) x $wgnumBins);
my @wgnCovROT = ((0) x $wgnumBins);
my @wgnBinEnd = ((0) x $wgnumBins);

# Read coverage from BBC file but outputing to CBC for each contig at a time
open( CBCFILE, ">:raw", $cbcfile ) || die "Failed to open binary file for output '$cbcfile'.\n";

my $numValues = $targetcov ? 4 : 2;
my $intsize = 4;
my $headbytes = 2 * $intsize;
my ($pos,$cd,$wrdsz,$rdlen,$fcov,$rcov,$upcstr,$nrd,$ontarg);
my @outAry;
my ($chromBinOffset,$totalBins) = (0,0);
my ($arySize,$numBins) = (0,0);
my ($chromCount,$chromIndex) = (0,0);
while( $chromCount <= $numChroms )
{
  last if( read( BBCFILE , my $buffer, $headbytes) != $headbytes );
  ($pos,$cd) = unpack "L2", $buffer;
  if( $pos == 0 )
  {
    # process coverage for previous contig
    processContigReads() if( $chromCount );
    # check for expected chromosome start - some might be missing but contig number must be increasing
    unless( $cd )
    {
      print STDERR "Error reading BBC file: Apparently no reads to targetted reference.\n";
      exit 1;
    }
    if( $cd-1 < $chromIndex )
    {
      print STDERR "Error reading BBC file: Contig $cd out of order vs count $chromCount.\n";
      exit 1;
    }
    $chromIndex = $cd-1;
    $chrid = $chromName[$chromIndex];
    ++$chromCount;
    print STDERR "Processing reads for contig#$chromCount: $chrid (#$cd)...\n" if( $logopt );
    # set up to collect data for this contig
    $numBins = $chromSize[$chromIndex] / $cbcsize;
    $numBins = ($numBins == int($numBins)) ? $numBins : int($numBins + 1);
    $arySize = $numValues * $numBins;
    @outAry = ((0) x $arySize);
    next;
  }
  $wdsiz = ($cd & 6) >> 1;
  next if( $wdsiz == 0 );  # ignore 0 read regions (for now)
  $rdlen = $cd >> 3;
  $ontarg = $cd & 1;
  if( $wdsiz == 3 ) {$upcstr = "L2";}
  elsif( $wdsiz == 2 ) {$upcstr = "S2";}
  else {$upcstr = "C2";}
  $wdsiz = 1 << $wdsiz;
  --$pos;
  while( $rdlen )
  {
    read( BBCFILE, my $buffer, $wdsiz );
    ($fcov,$rcov) = unpack $upcstr, $buffer;
    $binNum = $numValues * int($pos/$cbcsize);
    $outAry[$binNum] += $fcov;
    $outAry[++$binNum] += $rcov;
    if( $targetcov && $ontarg )
    {
      $outAry[++$binNum] += $fcov;
      $outAry[++$binNum] += $rcov;
    }
    --$rdlen;
    ++$pos;
  }
}
# process last contig read
processContigReads() if( $chromCount );
close( BBCFILE );

# --------------- Output optional whole genome coverage files --------------

if( $logopt )
{
  my $size = (stat $cbcfile)[7];
  my $exps = $totalBins * 4 * $numValues;
  print STDERR "Created file $cbcfile: size = $totalBins * $numValues ints = $size bytes (expected $exps)\n";
}
if( $haveChrCov ) 
{ 
  open( CHRCOV, ">$chrcovfile" ) or die "Failed to open file for output $chrcovfile.";
  print CHRCOV "$header\n";
  for( my $cn = 0; $cn < $numChroms; ++$cn ) 
  { 
    print CHRCOV "$chromName[$cn]\t1\t$chromSize[$cn]\t$chrCovFwd[$cn]\t$chrCovRev[$cn]";
    print CHRCOV "\t$chrCovFOT[$cn]\t$chrCovROT[$cn]" if( $targetcov );
    print CHRCOV "\n";
  }
  close( CHRCOV );
  print STDERR "Created chromosome coverage summary file $chrcovfile.\n" if( $logopt );
} 
if( $haveWgnCov )
{
  open( WGNCOV, ">$wgncovfile" ) or die "Failed to open file for output $wgncovfile.";
  print WGNCOV "$header\n";
  my $a = 0;
  for( my $cn = 0; $cn < $numChroms; ++$cn )
  {
    my $bend = 0;
    for( my $bn = 0; $bn < $chromBins[$cn]; ++$bn, ++$a )
    {
      my $bsrt = $bend + 1;
      $bend += $wgnBinEnd[$a];
      print WGNCOV "$chromName[$cn]\t$bsrt\t$bend\t$wgnCovFwd[$a]\t$wgnCovRev[$a]";
      print WGNCOV "\t$wgnCovFOT[$a]\t$wgnCovROT[$a]" if( $targetcov );
      print WGNCOV "\n";
    }
  }
  close( WGNCOV );
  print STDERR "Created genome coverage summary file $wgncovfile.\n" if( $logopt );
}

# ----------------END-------------------

sub processContigReads
{
  # write CBC for contig
  my $fp = tell(CBCFILE);
  print STDERR "Filing $chromName[$chromIndex]: $chromSize[$chromIndex] -> $numBins bins (offset $totalBins)\n" if( $logopt );
  $totalBins += $numBins;
  print CBCFILE pack "L[$arySize]", @outAry;
  # collect whole chromosome coverage stats
  if( $haveChrCov )
  {
    my $a = 0;
    for( my $j = 0; $j < $numBins; ++$j )
    {
      $chrCovFwd[$chromIndex] += $outAry[$a++];
      $chrCovRev[$chromIndex] += $outAry[$a++];
      if( $targetcov )
      {
        $chrCovFOT[$chromIndex] += $outAry[$a++];
        $chrCovROT[$chromIndex] += $outAry[$a++];
      }
    }
  }
  # collect binned whole genome coverage stats
  if( $haveWgnCov )
  {
    my ($a,$bn) = (0,0);
    my $ncb = $chromBins[$chromIndex] / $numBins;
    for( my $j = 0; $j < $numBins; ++$j )
    {
      $bn = $chromBinOffset + int($j * $ncb);
      $wgnBinEnd[$bn] += $cbcsize;
      $wgnCovFwd[$bn] += $outAry[$a++];
      $wgnCovRev[$bn] += $outAry[$a++];
      if( $targetcov )
      {
        $wgnCovFOT[$bn] += $outAry[$a++];
        $wgnCovROT[$bn] += $outAry[$a++];
      }
    }
    $chromBinOffset += $chromBins[$chromIndex];
    # correct last bin size to actual size used
    my $lastBinSize = $chromSize[$chromIndex] % $cbcsize;
    $wgnBinEnd[$bn] += $lastBinSize - $cbcsize if( $lastBinSize );
  }
}

