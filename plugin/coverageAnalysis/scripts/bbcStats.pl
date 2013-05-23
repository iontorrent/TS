#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create statistics for a base coverage (bbc) file (to STDOUT).
Options include output of a depth of coverage distribution table.";
my $USAGE = "Usage:\n\t$CMD [options] <covfile.bbc>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -f Output Full statistics. This adds a few more statistics to the output, e.g. those output by previous version of TCA.
  -g Produce base coverage stats all for reads of the Genome (reference).
  -l Show extra log information to STDERR.
  -w Print Warning messages for potential BED file issues to STDOUT.
  -B <file> Input Bed file to use to restrict coverage region analysis to.
     (Faster if an index covfile.bci or covfile.bbc.bci exists.)
  -C <int> Count of total base reads used for % read on target stat. Default: 0 (None output).
     A -ve value indicates to use total read count. This is useful for on/off target stats limited
     to regions using the -B option but may take if only target coverage stats are required.
  -D <file> Output file name for depth of coverage Distribution table (tsv) file. Default: '' (None output).
     Output fields created depend on provided options -B, -g and -T.
  -R <int> Threshold for Read coverage for strand bias to be counted. Default: 10 reads.
  -S <int> Threshold for Strand bias counting based on percent forward reads being between <int> and 100-<int>. Default: 70.
  -T <int> Target size in bases. Required (>0) for coverage stats vs. total target region. Default: 0";

my $docfile = '';
my $bedfile = '';
my $logopt = 0;
my $genomeStat = 0;
my $targetSize = 0;
my $basereads = 0;
my $bedwarn = 0;
my $fullstats = 0;
my $thresBias = 70;
my $thresReads = 10;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-B') {$bedfile = shift;}
  elsif($opt eq '-C') {$basereads = int(shift);}
  elsif($opt eq '-D') {$docfile = shift;}
  elsif($opt eq '-R') {$thresReads = int(shift);}
  elsif($opt eq '-S') {$thresBias = int(shift);}
  elsif($opt eq '-T') {$targetSize = int(shift);}
  elsif($opt eq '-f') {$fullstats = 1;}
  elsif($opt eq '-g') {$genomeStat = 1;}
  elsif($opt eq '-w') {$bedwarn = 1;}
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

my $havedoc = ($docfile ne "" && $docfile ne "-" );
my $havebed = ($bedfile ne "" && $bedfile ne "-" );

$targetSize = 0 if( $targetSize <= 0 );
$thresBias = 0 if( $thresBias <= 0 );
$thresBias *= 0.01; # % -> fraction

unless( $genomeStat || $targetSize || $havebed )
{
  print STDERR "$CMD: Requires at least one of options -B, -G or -T to be specified.\n";
  exit 1;
}

#--------- End command arg parsing ---------

die "Cannot find depth file $bbcfile" unless( -e $bbcfile );

# open BBCFILE and read contig header string
open( BBCFILE, "<:raw", $bbcfile ) || die "Failed to open BBC file $bbcfile\n";
chomp( my $contigList = <BBCFILE> );
my @chromName = split('\t',$contigList );
my %chromNum;
my $numChroms = scalar(@chromName);
my $genomeSize = 0;
for( my $i = 0; $i < $numChroms; ++$i )
{
  my @fields = split('\$',$chromName[$i]);
  $chromNum{$chrid} = $i+1;
  $chromName[$i] = $fields[0];
  $genomeSize += int($fields[1]);
}
print STDERR "Read $numChroms contig names and lengths. Total contig size: $genomeSize\n" if( $logopt );

# set the upper and lower boundaries for counting base coverage bias
my $lthresBias = 1 - $thresBias;
if( $lthresBias > $thresBias )
{
  $thresBias = $lthresBias;
  $lthresBias = 1 - $thresBias;
}

my (%targSrts,%targEnds);
my $bedtarSize = 0;
loadBedRegions() if( $haveTargets );

# collect depth distribution : pre-allocate for performance
my @genomeDist = $genomeStat > 0 ? ((0)x20000) : ();
my @targetDist = $targetSize > 0 ? ((0)x20000) : ();
my ($genomeMaxDepth,$targetMaxDepth) = (0,0);
my ($genomeNoBias,$targetNoBias) = (0,0);

my $baseReads = 0;
my $intsize = 4;
my $headbytes = 2 * $intsize;
my ($pos,$cd,$wrdsz,$rdlen,$fcov,$rcov,$upcstr,$nrd,$bias,$ontarg);

my $skipOffTarget = (!$genomeStat && $basereads >= 0);

while(1)
{
  last if( read( BBCFILE , my $buffer, $headbytes) != $headbytes );
  ($pos,$cd) = unpack "L2", $buffer;
  if( $pos == 0 )
  {
    my $cnum = $cd;
    $chrid = $chromName[$cnum-1];
    print STDERR "Found start of contig $chrid ($cnum)\n" if( $logopt );
    next;
  }
  $wdsiz = ($cd & 6) >> 1;
  next if( $wdsiz == 0 );  # ignore 0 read regions (for now)
  $rdlen = $cd >> 3;
  $ontarg = $cd & 1;
  if( $skipOffTarget && !$ontarg )
  {
    seek( BBCFILE, $rdlen << $wdsiz, 1 );
    next;
  }
  if( $wdsiz == 3 ) {$upcstr = "L2";}
  elsif( $wdsiz == 2 ) {$upcstr = "S2";}
  else {$upcstr = "C2";}
  $wdsiz = 1 << $wdsiz;
  while( $rdlen )
  {
    read( BBCFILE, my $buffer, $wdsiz );
    ($fcov,$rcov) = unpack $upcstr, $buffer;
    $nrd = $fcov+$rcov;
    $baseReads += $nrd;
    $bias = $nrd ? $fcov / $nrd : 0.5;
    if( $genomeStat )
    {
      ++$genomeDist[$nrd];
      $genomeMaxDepth = $nrd if( $nrd > $genomeMaxDepth );
      ++$genomeNoBias unless( $nrd >= $thresReads && ($bias < $lthresBias || $bias > $thresBias) );
    }
    if( $targetSize && $ontarg )
    {
      ++$targetDist[$nrd];
      $targetMaxDepth = $nrd if( $nrd > $targetMaxDepth );
      ++$targetNoBias unless( $nrd >= $thresReads && ($bias < $lthresBias || $bias > $thresBias) );
    }
    --$rdlen;
  }
}

# create output stats
my ($genomeCumd,$genomeABC,$targetCumd,$targetABC);
$baseReads = $basereads if( $basereads >= 0 );
($targetCumd,$targetABC) = outputStats( 'Target', \@targetDist, $targetMaxDepth, $targetSize, $baseReads, $targetNoBias ) if( $targetSize );
($genomeCumd,$genomeABC) = outputStats( 'Genome', \@genomeDist, $genomeMaxDepth, $genomeSize, 0, $genomeNoBias ) if( $genomeStat );

if( $havedoc )
{
  my $header = "read_depth";
  if( $targetSize && $genomeStat )
  {
    $header .= "\ttarget_base_cov\ttarget_base_cum_cov\tnorm_read_depth\tpc_target_base_cum_cov";
    $header .= "\tgenome_base_cov\tgenome_base_cum_cov\tnorm_read_depth\tpc_genome_base_cum_cov";
  }
  else
  {
    $header .= "\tbase_cov\tbase_cum_cov\tnorm_read_depth\tpc_base_cum_cov";
  }
  open( DOCOUT, ">$docfile" ) || die "Cannot open file for writing $docfile.\n";  
  print DOCOUT "$header\n";
  my $maxDepth = $genomeMaxDepth > $targetMaxDepth ? $genomeMaxDepth : $targetMaxDepth;
  my ($tot,$cov,$cumcov,$normdepth,$pccumcov);
  for( my $d = 0; $d <= $maxDepth; ++$d )
  {
    print DOCOUT $d;
    if( $targetSize )
    {
      if( $d <= $targetMaxDepth )
      {
        $tot = @{$targetCumd}[0];
        $cov = $targetDist[$d];
        $cumcov = @{$targetCumd}[$d];
        $normdepth = $targetABC > 0 ? $d / $targetABC : 0;
        $pccumcov = $tot > 0 ? 100 * $cumcov / $tot : 0;
      }
      else {$cov = $cumcov = $normdepth = $pccumcov = 0;}
      printf DOCOUT "\t%.0f\t%.0f\t%.4f\t%.2f", $cov, $cumcov, $normdepth, $pccumcov;
    }
    if( $genomeStat )
    {
      if( $d <= $genomeMaxDepth )
      {
        $tot = @{$genomeCumd}[0];
        $cov = $genomeDist[$d];
        $cumcov = @{$genomeCumd}[$d];
        $normdepth = $genomeABC > 0 ? $d / $genomeABC : 0;
        $pccumcov = $tot > 0 ? 100 * $cumcov / $tot : 0;
      }
      else {$cov = $cumcov = $normdepth = $pccumcov = 0;}
      printf DOCOUT "\t%.0f\t%.0f\t%.4f\t%.2f", $cov, $cumcov, $normdepth, $pccumcov;
    }
    print DOCOUT "\n";
  }
  close( DOCOUT );
}

#-------------------------- End ------------------------

# Load all BED file regions in to memory and validate BED file.
# This functions is for code organization only and not intended to be general function.
sub loadBedRegions
{
  my ($lastChr,$lastSrt,$lastEnd,$numTracks,$numTargets,$numTargReads,$numWarns) = (0,0,0,0,0,0,0);
  open( BEDFILE, "$bedfile" ) || die "Cannot open targets file $bedfile.\n";
  while( <BEDFILE> )
  {
    my ($chrid,$srt,$end) = split('\t',$_);
    next if( $chrid !~ /\S/ );
    if( $chrid =~ /^track / )
    {
      ++$numTracks;
      if( $numTracks > 1 )
      {
        print STDERR "\nWARNING: Bed file has multiple tracks. Ignoring tracks after the first.\n";
        ++$numWarns;
        last;
      }
      if( $numTargets > 0 )
      {
        print STDERR "\nERROR: Bed file incorrectly formatted: Contains targets before first track statement.\n";
        exit 1;
      }
      next;
    }
    if( !defined($chromNum{$chrid}) )
    {
      print STDERR "\nERROR: Target region ($chrid:$srt-$end) not present in specified genome.\n";
      exit 1;
    }
    if( $chromNum{$chrid} != $lastChr )
    {
      if( $chromNum{$chrid} < $lastChr )
      {
        print STDERR "\nERROR: BED file is not ordered ($chrid out of order vs. genome file).\n";
        exit 1;
      }
      $lastChr = $chromNum{$chrid};
      $lastSrt = 0;
      $lastEnd = 0;
    }
    ++$numTargReads;
    ++$srt;
    $end += 0;
    if( $srt < $lastSrt )
    {
      print STDERR "ERROR: Region $chrid:$srt-$end is out-of-order vs. previous region $chrid:$lastSrt-$lastEnd.\n";
      exit 1;
    }
    $lastSrt = $srt;
    if( $srt <= $lastEnd )
    {
      ++$numWarn;
      if( $end <= $lastEnd )
      {
        print STDERR "Warning: Region $chrid:$srt-$end is entirely overlapped previous region $chrid:$lastSrt-$lastEnd.\n" if( $bedwarn );
        $lastEnd = $end;
        next;
      }
      print STDERR "Warning: Region $chrid:$srt-$end overlaps previous region $chrid:$lastSrt-$lastEnd.\n" if( $bedwarn );
      $srt = $lastEnd + 1;
    }
    $lastEnd = $end;
    ++$numTargets;
    push( @{$targSrts{$chrid}}, $srt );
    push( @{$targEnds{$chrid}}, $end );
    $bedtarSize += $end - $srt + 1;
  }
  close( BEDFILE );
  print STDERR "$CMD: $numWarns BED file warnings were detected!\n" if( $numWarns );
  print STDERR " - Re-run with the -w option to see individual warning messages.\n" if( !$bedwarn );
  print STDERR "Read $numTargets of $numTargReads target regions from $bedfile\n" if( $logopt || $numWarns );
}

# generates output stats or given depth array and returns reference to cumulative depth array
sub outputStats
{
  my ($tag,$hist,$maxDepth,$targSize,$baseReads,$noBiasReads) = @_;
  my @dist = @{$hist};
  my @cumd;
  my $tagL = lc($tag);
  my $tagU = ucfirst($tagL);
  my ($reads,$sum_reads,$sum_dreads,$cumcov) = (0,0,0,0);
  for( my $depth = $maxDepth; $depth > 0; --$depth )
  {
    $dist[$depth] += 0; # force value
    $cumcov += $dist[$depth];
    $cumd[$depth] = $cumcov; # for medians
    $reads = $depth * $dist[$depth];
    # sums for variance calculation
    $sum_reads += $reads;
    $sum_dreads += $depth * $reads;
  }
  # have to address the element directly, since dist is a copy (?)
  #$dist[0] = $targSize - $cumcov;
  ${$_[1]}[0] = $targSize - $cumcov;
  $cumd[0] = $targSize;
  # mean read depth
  my $abc = $sum_reads / $targSize;
  # mean and stddev for reads with at least 1x coverage ($cumcov == $cumd[1])
  my $ave = $cumcov > 0 ? $sum_reads/$cumcov : 0;
  my $std = $cumcov > 1 ? sqrt(($sum_dreads - $ave*$ave*$cumcov)/($cumcov-1)) : 0;
  my $scl = 100 / $targSize;
  my $p2m = int(0.2*$abc+0.5);
  if( $baseReads > 0 )
  {
    printf "Total aligned base reads:          %.0f\n",$baseReads;
    printf "Total base reads on target:        %.0f\n",$sum_reads;
  }
  else
  {
    printf "Total base reads on target:        %.0f\n",$sum_reads;
  }
  if( $tagL eq "genome" )
  {
    printf "Bases in reference %s:         %.0f\n",$tagL,$targSize;
  }
  else
  {
    printf "Bases in %s regions:           %.0f\n",$tagL,$targSize;
  }
  if( $baseReads > 0 )
  {
    printf "Percent base reads on target:      %.2f%%\n",100*($sum_reads/$baseReads);
  }
  my $sig = sigfig($abc);
  printf "Average base coverage depth:       %.${sig}f\n",$abc;
  printf "Uniformity of base coverage:       %.2f%%\n",$cumd[$p2m]*$scl;
  if( $fullstats )
  {
    printf "Bases covered (at least 1x):       %.0f\n",$cumcov;
    printf "Maximum base read depth:           %.0f\n",$maxDepth;
    $sig = sigfig($ave);
    printf "Average base read depth:           %.${sig}f\n",$ave;
    $sig = sigfig($std);
    printf "Std.Dev base read depth:           %.${sig}f\n",$std;
  }
  printf "$tagU base coverage at 1x:        %.2f%%\n",$cumd[1]*$scl;
  printf "$tagU base coverage at 10x:       %.2f%%\n",$cumd[10]*$scl if( $fullstats );
  printf "$tagU base coverage at 20x:       %.2f%%\n",$cumd[20]*$scl;
  printf "$tagU base coverage at 50x:       %.2f%%\n",$cumd[50]*$scl if( $fullstats );
  printf "$tagU base coverage at 100x:      %.2f%%\n",$cumd[100]*$scl;
  printf "$tagU base coverage at 500x:      %.2f%%\n",$cumd[500]*$scl;
  printf "$tagU bases with no strand bias:  %.2f%%\n",100*$noBiasReads/$targSize;
  return (\@cumd,$abc);
}

sub sigfig
{
  my $val = $_[0];
  return 0 if( $val ) >= 1000;
  return 1 if( $val ) >= 100;
  return 2 if( $val ) >= 10;
  return 3;
}

