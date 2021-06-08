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

my $accuracy = 1e-10;
my $contigSep = '&';

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
my $gene = uc(shift);  # assume case insensitive for user convenience - later this could be an issue
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

# set target ID check as default match if gene symbol not matched
# - disabled using tab if no gene query or explicit KVP query
my $checkID = ($gene eq "" || index($gene,'=') > 0) ? '\t' : uc($gene);

open( TSVFILE, $dataFile );
if( !TSVFILE || eof(TSVFILE) ) {
  print "Error: Could not open data file $dataFile\n";
  exit(0);
}
# first pass to determine number of records matching query
my %chrid;
my $chrList = '';
my $numHits = $numrec;
my $checkGene = 0;
my $checkNVP = 0;
my $keyswp = '';
if( $numrec <= 0 )
{
  $numHits = 0;
  # check if this had gene id column of old format - if not assume it is the ionVersion=4.0 KVP fields
  my $line = <TSVFILE>;
  my $genq = $gene;
  if( $genq ne "" ) {
    $checkGene = ($line =~ m/\sgene_id\s/);
    $checkNVP = !$checkGene;
    if( $checkNVP ) {
      # make default key GENE_ID
      my $i = index($genq,"=");
      if( $i < 0 ) {
        $genq = "GENE_ID=".$genq;
        $keyswp = ";GENE_ID=";
      } else {
        $keyswp = ";".substr($genq,0,$i+1);
      }
      # for performance - allows for direct string search rather than complex regex
      $genq = ';'.$genq if( $genq !~ /^;/ );
      $genq .= ';' if( $genq !~ /;$/ );
    }
  }
  while( <TSVFILE> )
  {
    my @fields = split('\t',$_);
    if( $chrret && !defined($chrid{$fields[0]}) )
    {
      $chrList .= $fields[0] . $contigSep;
      $chrid{$fields[0]} = 1;
    }
    next if( $chrom ne "" && $chrom ne $fields[0] );
    next if( $covmin > $fields[9] );
    next if( $covmax < $fields[9] );
    if( $checkNVP ) {
      # allow to work on merged fields by replacing merge separator character (@) with a key match
      my $fld = ';'.uc($fields[4]).';';
      $fld =~ s/&/$keyswp/g;
      next if( index( $fld, $genq ) < 0 && $checkID ne uc($fields[3]) );
    } elsif( $checkGene ) {
      # support for pre-4.0 format
      next if( $genq ne uc($fields[4]) && $checkID ne uc($fields[3]) );
    }
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
    $chrList .= $chr . $contigSep if( $numrec > 0 || defined($chrid{$chr}) );
  }
  close( GENOME );
}
# output header line with extra first field for numHits to query
# - also output all chromosomes as extra field if asked for by supplying $numrec < 0
my $line = <TSVFILE>;
my @fields= split('\t',$line);
my $numFields = scalar(@fields);
# code is repeated here since earlier pre-scan might not have been perfomed
if( $gene ne "" ) {
  $checkGene = ($line =~ m/\sgene_id\s/);
  $checkNVP = !$checkGene;
  if( $checkNVP ) {
    my $i = index($gene,"=");
    if( $i < 0 ) {
      $gene = "GENE_ID=".$gene;
      $keyswp = ";GENE_ID=";
    } else {
      $keyswp = ";".substr($gene,0,$i+1);
    }
    $gene = ';'.$gene if( $gene !~ /^;/ );
    $gene .= ';' if( $gene !~ /;$/ );
  }
}
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
my $firstRow = 1+int(0.01 * $clipright * $numHits)-$cliprows;  # handles effective round vs clipleft 
my $binsize = $cliprows / $maxrows;

my $bin = 0;
my $bincnt = 0;
my $sumLen = 0;
my $nrec = 0;
my $nout = 0;
my $ccnt = 0;
my $gcbias = 0;
my $slen = 0;
my @record;

$" = "\t";

while( <TSVFILE> )
{
  chomp;
  @fields = split('\t',$_);
  next if( $chrom ne "" && $chrom ne $fields[0] );
  next if( $covmin > $fields[9] );
  next if( $covmax < $fields[9] );
  if( $checkNVP ) {
    # allow to work on merged fields by replacing merge separator character (@)
    my $fld = ';'.uc($fields[4]).';';  # do not want to affect output string!
    $fld =~ s/&/$keyswp/g;
    next if( index( $fld, $gene ) < 0 && $checkID ne uc($fields[3]) );
  } elsif( $checkGene ) {
    # support for pre-4.0 format
    next if( $gene ne uc($fields[4]) && $checkID ne uc($fields[3]) );
  }
  # to get to correct window this must be done after all filters
  next if( ++$nrec < $firstRow );
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
    $sumLen = $slen;
    @record = @fields;
    $gcbias = abs(($fields[5] / $slen)-0.5);
    $ccnt = 0;
  } else {
    # sum count fields for averaging
    $sumLen += $slen;
    for( my $f = 5; $f < $numFields; ++$f ) {
      $record[$f] += $fields[$f];
    }
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
    $bin -= $binsize;
    $bincnt = 0;
    print "@record\n";
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
  print "@record\n";
}
close(TSVFILE);

