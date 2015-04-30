#!/usr/bin/perl
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create statistics for a contigs read coverage file (to STDOUT).
It is assumed the input file is a tsv file with header and by default, that the first column is contig names and the last sequence reads.";
my $USAGE = "Usage:\n\t$CMD [options] <contig coverage file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -d Data field (0-based). Default: -1 (the last)";

my $logopt = 0;
my $contig_fld = 0;
my $data_fld = -1;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-c') {$contig_fld = (shift)+0;}
  elsif($opt eq '-d') {$data_fld = (shift)+0;}
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

my $trdfile = shift(@ARGV);

#--------- End command arg parsing ---------

die "Cannot find depth file $trdfile" unless( -e $trdfile );

# open TRDFILE and read header string to detect BED format (ionVersion)
open( TRDFILE, $trdfile ) || die "Failed to open target reads file $trdfile\n";
chomp( my $fieldIDs = <TRDFILE> );

# collect depth distribution : pre-allocate for performance
my @targetDist = ((0)x20000);
my ($targetMaxDepth,$numTargets) = (0,0);

# collect depth-of-reads distribution and read threshold stats
while(<TRDFILE>)
{
  chomp;
  my @fields = split('\t',$_);
  next if( $fields[0] !~ /\S/ );
  my $depth = $fields[$data_fld];
  $targetMaxDepth = $depth if( $depth > $targetMaxDepth );
  ++$numTargets;
  ++$targetDist[int($depth)];
}
close(TRDFILE);

# create output stats
my $targType = 'Contig';
my $targetCumd = outputStats( $targType, \@targetDist, $targetMaxDepth, $numTargets, $sumBaseDepth );

#-------------------------- End ------------------------

# generates output stats or given depth array and returns reference to cumulative depth array
sub outputStats
{
  my ($tag,$hist,$maxDepth,$numTargs,$sumDepth) = @_;
  my @dist = @{$hist};
  my @cumd;
  my $tagL = lc($tag);
  my $tagU = ucfirst($tagL);
  my ($reads,$sum_reads,$sum_dreads,$cumcov) = (0,0,0,0);
  for( my $depth = int($maxDepth); $depth > 0; --$depth ) {
    $dist[$depth] += 0; # force value
    $cumcov += $dist[$depth];
    $cumd[$depth] = $cumcov; # for medians
    $reads = $depth * $dist[$depth];
    # sums for variance calculation
    $sum_reads += $reads;
    $sum_dreads += $depth * $reads;
  }
  # have to address the element directly, since dist is a copy (?)
  ${$_[1]}[0] = $numTargs - $cumcov;
  $cumd[0] = $numTargs;
  # mean read depth
  my $abc = $sum_reads / $numTargs;
  # mean and stddev for reads with at least 1x coverage ($cumcov == $cumd[1])
  my $ave = $cumcov > 0 ? $sum_reads/$cumcov : 0;
  my $std = $cumcov > 1 ? sqrt(($sum_dreads - $ave*$ave*$cumcov)/($cumcov-1)) : 0;
  my $scl = 100 / $numTargs;
  my $sig = sigfig($abc);
  printf "Number of %ss:                %.0f\n",$tagL,$numTargs;
  printf "Average reads per $tagL:         %.${sig}f\n",$abc;
  printf "%ss with at least 1 read:     %d\n",$tagU,$cumd[1];
  printf "%ss with at least 10 reads:   %d\n",$tagU,$cumd[10];
  printf "%ss with at least 100 reads:  %d\n",$tagU,$cumd[100];
  printf "%ss with at least 1000 reads: %d\n",$tagU,$cumd[1000];
  printf "%ss with at least 10K reads:  %d\n",$tagU,$cumd[10000];
  printf "%ss with at least 100K reads: %d\n",$tagU,$cumd[100000];
  return \@cumd;
}

sub sigfig
{
  my $val = $_[0];
  return 0 if( $val ) >= 1000;
  return 1 if( $val ) >= 100;
  return 2 if( $val ) >= 10;
  return 3;
}

