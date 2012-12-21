#!/usr/bin/perl
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

use File::Basename;

(my $CMD = $0) =~ s{^(.*/)+}{};
if( scalar(@ARGV) < 3 )
{
  print STDERR "Error: Insuficient args: Usage:\n  $CMD <genome file (.fai)> <Property Index> <file1> [<file2] ...]\n";
  exit 0;
}
my $genome = shift;
unless( -e $genome )
{
  print STDERR "Error at $CMD arg#1: Genome file not found: '$genome'.\n";
  exit 0;
}
my $propNum = int(shift);
if( $propNum <= 0 )
{
  print STDERR "Error at $CMD arg#2: <Property Index> must be an integer > 0.\n";
  exit 0;
}

# flag for extra target location output (useful for debugging sort)
my $addCoords = 0;

# check/read genome file (for chromosome order relative to genome)
my @chromName;
my $numChroms = 0;
open( GENOME, $genome ) || die "Cannot read genome info. from $genome.\n";
while( <GENOME> )
{
  my ($chrid) = split;
  next if( $chrid !~ /\S/ );
  $chromName[$numChroms++] = $chrid;
}
close( GENOME );

my %targets;
my %sortlist;
my $barcode;
my $barcode_fields;
while(<>)
{
  my $fn = basename(dirname($ARGV));
  if( $fn ne $barcode )
  {
    $barcode = $fn;
    $barcode_fields .= $fn."\t";
    next;  # skip header line
  }
  my @fields = split;
  # assume amplicon ID is unique (filled in for only 3 columns)
  my $trgid = $fields[3];
  if( defined($targets{$trgid}) )
  {
    $targets{$trgid} .= "\t";
  }
  else
  {
    # build lists for sorting (by chromsome)
    push( @{$sortlist{$fields[0]}}, [$fields[1],$fields[2],$fields[4],$fields[3]] );
  }
  $targets{$trgid} .= $fields[$propNum];
}
# sort arrays for each chromosome on target start then stop
while( my ($chr,$ary) = each(%sortlist) )
{
  @$ary = sort { $a->[0] <=> $b->[0] || $a->[1] <=> $b->[1] } @$ary;
}
# output matrix using amplicons sorted by chromosome, start, stop (even thouh these fields are not output)
print "Chrom\tStart\tEnd\t" if( $addCoords );
print "Gene\tTarget\t$barcode_fields\n";
for( my $chrn = 0; $chrn < $numChroms; ++$chrn )
{
  my $chr = $chromName[$chrn];
  my $ary = $sortlist{$chr};
  for( my $i = 0; $i < scalar(@$ary); ++$i )
  {
    my $subary = $ary->[$i];
    my $trgid = $subary->[3];
    print "$chr\t$subary->[0]\t$subary->[1]\t" if( $addCoords );
    print "$subary->[2]\t$subary->[3]\t$targets{$trgid}\n";
  }
}

