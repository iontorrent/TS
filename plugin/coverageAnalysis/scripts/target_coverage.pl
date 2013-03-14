#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
# Targets table file sampler in perl for PHP script. Not intended for cmd-line use outside of testing.
# The -G option is only used when a scan of the whole file is wanted with the chromosomes in genome order output.
# The -a option specifies to output all data passing filters in range
#  - Max rows output and binning is ignorred. No exra fields for # number filtered and contig list are output.
# The -b option is the same as -a except the output format is plain (no header) 4-column BED.
# The -c option indicates that no contig (field #1) list is returned (with retval < 0 or -G option)

(my $CMD = $0) =~ s{^(.*/)+}{};

my $genome = '';
my $allout = 0;
my $bedout = 0;
my $chrret = 1;

while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-G') {$genome = shift;}
  elsif($opt eq '-a') {$allout = 1;}
  elsif($opt eq '-b') {$allout = 1; $bedout = 1;}
  elsif($opt eq '-c') {$chrret = 0;}
  else
  {
    print STDERR "$CMD: Invalid option argument: $opt\n";
    exit 1;
  }
}
if( scalar(@ARGV) < 8 )
{
  print STDERR "$CMD: Invalid number of arguments.\n";
  exit 1;
}
my $dataFile = shift;
my $chrom = shift;
my $gene = uc(shift);
my $covmin = shift;
my $covmax = shift;
my $maxrows = shift;
my $clipleft = shift;
my $clipright = shift;

# optional arg gives total number of records in file so this does not have to be pre-determined
my $numrec = scalar(@ARGV) ? shift : 0;
if( scalar(@ARGV) > 0 )
{
  print STDERR "$CMD: Invalid number of arguments.\n";
  exit 1;
}

$chrom = "" if( $chrom eq "-" );
$gene = "" if( $gene eq "-" );
$genome = "" if( !$chrret );

unless( $genome eq '' || $genome eq '-' )
{
  unless( -e $genome )
  {
    print STDERR "$CMD: Warning: Failed to read contig order from genome file '$genome'\n";
    $genome = '';
  }
}

open( TSVFILE, $dataFile );
if( !TSVFILE || eof(TSVFILE) ) {
  print "Error: Could not open data file $dataFile\n";
  exit(0);
}
# first pass to determine number of records matching query
my %chrid;
my $chrList = '';
my $numHits = $numrec;
if( $numrec <= 0 )
{
  $numHits = 0;
  <TSVFILE>;
  while( <TSVFILE> )
  {
    my @fields = split;
    if( $chrret && !defined($chrid{$fields[0]}) )
    {
      $chrList .= $fields[0] . ':';
      $chrid{$fields[0]} = 1;
    }
    next if( $chrom ne "" && $chrom ne $fields[0] );
    next if( $gene ne "" && $gene ne uc($fields[4]) );
    next if( $covmin > $fields[9] );
    next if( $covmax < $fields[9] );
    ++$numHits;
  }
  # rewind for re-read
  seek(TSVFILE,0,0);
}
# read chromosome list from genome file if provided
if( $genome ne '' )
{
  $chrList = '';
  open( GENOME, $genome ) || die "Cannot read genome info. from $genome.\n";
  while( <GENOME> )
  {
    chomp;
    my ($chr) = split;
    # only include chromosome with targets
    $chrList .= $chr . ':' if( $numrec > 0 || defined($chrid{$chr}) );
  }
  close( GENOME );
}
# output header line with extra first field for numHits to query
# - also output all chromosomes as extra field if asked for by supplying $numrec < 0
my $line = <TSVFILE>;
if( !$allout ) {
  print "$numHits\t";
  print "$chrList\t" if( $chrret && $numrec < 0 );
}
print $line if( !$bedout );
exit(0) if( $numHits == 0 );
exit(0) if( $clipleft > 100 || $clipright < $clipleft );

$maxrows = $numHits if( $maxrows > $numHits || $maxrows == 0 );

$clipleft = 0 if( $clipleft <= 0 );
$clipright = 100 if( $clipright > 100 );

# done this way so representation (total number of binned tatgets) can be reproduced using elsewhere
my $cliprows = int(0.5 + $numHits * 0.01 * ($clipright - $clipleft));
my $skipStart = 1+int(0.01 * $clipleft * $numHits);
my $binsize = $cliprows / $maxrows;

my $bin = 0;
my $bincnt = 0;
my $sumLen = 0;
my $nrec = 0;
my $nout = 0;
my $ccnt = 0;
my $gcbias = 0;
my $slen = 0;
my @fields;
my @record;

$" = "\t";

while( <TSVFILE> )
{
  @fields = split;
  next if( $chrom ne "" && $chrom ne $fields[0] );
  next if( $gene ne "" && $gene ne uc($fields[4]) );
  next if( $covmin > $fields[9] );
  next if( $covmax < $fields[9] );
  next if( ++$nrec < $skipStart );
  if( $allout ) {
    if( $bedout ) {
      --$fields[1];
      print "$fields[0]\t$fields[1]\t$fields[2]\t$fields[3]\n";
    } else {
      print "@fields\n";
    }
    $bin += 1.0;
    if( $bin >= $binsize ) {
      $bin -= $binsize;
      last if( ++$nout >= $maxrows );
    }
    next;
  }
  if( $binsize <= 1 ) {
    print "@fields\n";
    last if( ++$nout >= $maxrows );
    next;
  }
  $bin += 1.0;
  ++$bincnt;
  $slen = $fields[2] - $fields[1] + 1;
  if( $bincnt == 1 ) {
    @record = @fields;
    $sumLen = $slen;
    $gcbias = abs(($fields[5] / $slen)-0.5);
    $ccnt = 0;
  } else {
    $sumLen += $slen;
    $record[5] += $fields[5]; # gc count
    $record[6] += $fields[6]; # bases covered
    $record[7] += $fields[7]; # bases uncov 3'
    $record[8] += $fields[8]; # bases uncov 3'
    $record[10] += $fields[10]; # fwd base reads
    $record[11] += $fields[11]; # rev base reads
    $gcbias += abs(($fields[5] / $slen)-0.5);
    ++$ccnt if( $record[0] ne $fields[0] );
  }
  if( $bin >= $binsize ) {
    $record[0] = "Multiple" if( $ccnt );
    if( $bincnt > 1 )
    {
      $record[1] = $bincnt;
      $record[2] = $sumLen;
      $record[3] = 0; # => no amplicon ID => must be binned
      $record[4] = sprintf("%.3f", $gcbias);
    }
    # this could be normalized read counts or base counts
    $record[9] = sprintf("%.3f", ($record[10]+$record[11])/$sumLen);
    print "@record\n";
    $bin -= $binsize;
    $bincnt = 0;
    last if( ++$nout >= $maxrows );
  }
}
# dump last bin results if round-up error on last bin
if( $nout < $maxrows && !$allout ) {
  $record[0] = "Multiple" if( $ccnt );
  if( $bincnt > 1 )
  {
    $record[1] = $bincnt;
    $record[2] = $sumLen;
    $record[3] = 0; # => no amplicon ID => must be binned
    $record[4] = sprintf("%.3f", $gcbias);
  }
  $record[9] = sprintf("%.3f", ($record[10]+$record[11])/$sumLen);
  print "@record\n";
}
close(TSVFILE);

