#!/usr/bin/perl
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

use File::Basename;

(my $CMD = $0) =~ s{^(.*/)+}{};
if( scalar(@ARGV) < 3 ) {
  print STDERR "Error: Insuficient args: Usage:\n  $CMD <reference.fai file [,file2, ...]> <Property Index> <file1> [<file2> ...]\n";
  exit 0;
}
my $genome = shift;

my $propType = shift;
my $idField = 3;
my $propNum = int($propType+0);
my $contigRep = 0;
# allow customization for reading the chromosome coverage summary file
if( $propType eq "chrom" || $propType eq "contig" ) {
  $idField = 0;
  $propNum = -1;
  $contigRep = 1;
} elsif( $propType eq "chrombase" ) {
  $idField = 0;
  $propNum = 3;
  $contigRep = 2;
} elsif( $propNum <= 0 ) {
  print STDERR "Error at $CMD arg#2: <Property Index> must be an integer > 0.\n";
  exit 1;
}

# flag for extra target location output (useful for debugging sort)
my $addCoords = 0;

# check/read genome file (for chromosome order relative to genome)
my @chromName;
my %chromIDs;
my $numChroms = 0;
if( $genome ne '-' ) {
  # split multiple files by ','
  my @files = split(',',$genome);
  foreach my $file (@files) {
    open( GENOME, $file ) || die "Cannot read genome info. from $file.\n";
    while( <GENOME> ) {
      chomp;
      my ($chrid) = split;
      next if( $chrid !~ /\S/ );
      unless( defined($chromIDs{$chrid}) ) {
        $chromIDs{$chrid} = $numChroms;
        $chromName[$numChroms++] = $chrid;
      }
    }
    close( GENOME );
  }
}

my %targets;
my %sortlist;
my $barcode;
my $barcode_fields;
my $fnum = 0;
while(<>) {
  my $fn = basename(dirname($ARGV));
  if( $fn ne $barcode ) {
    # add blank columns for existing targets not covered to number of bacodes considered so far
    if( ++$fnum > 1 ) {
      $barcode_fields .= "\t";
      while( my ($id,$str) = each(%targets) )
      {
        my $cnt = ($str =~ tr/\t//)+2;
        $targets{$id} .= "\t" if( $cnt < $fnum );
      }
    }
    $barcode = $fn;
    $barcode_fields .= $fn;
    next;  # skip header line
  }
  chomp;
  my @fields = split('\t',$_);
  # collect contig IDs if not provided by genome file for ordering
  my $chr = $fields[0];
  unless( defined($chromIDs{$chr}) ) {
    $chromIDs{$chr} = $numChroms;
    $chromName[$numChroms++] = $chr;
  }
  # assume target ID plus start+end location is unique
  my $trgid = $fields[$idField].':'.$fields[1].'-'.$fields[2];
  if( defined($targets{$trgid}) ) {
    $targets{$trgid} .= "\t";
  } else {
    # build lists for sorting (by chromsome)
    my $gene = ';'.$fields[4].';';
    unless( index($gene,"=") < 0 ) {
      # try to extract gene ID from KVP field
      if( $gene =~ m/;GENE_ID=(.*?);/ ) {
        $gene = $1;
      } else {
        my $fusion = '';
        while( $gene =~ s/GENE_ID=(.*?);// ) {
          $fusion .= '-/-' if( $fusion ne '' );
          $fusion .= $1;
          $gene .= ';';
        }
        $gene = $fusion;
      }
    } else {
      # old TS BED format
      $gene = $fields[4];
    }
    #$gene = ($gene =~ m/;GENE_ID=(.*?);/) ? $1 : (index($gene,"=") < 0 ? $fields[4] : '');
    $gene = "N/A" if( $gene eq '.' || $gene !~ '\S' );
    push( @{$sortlist{$chr}}, [$fields[1],$fields[2],$gene,$fields[$idField]] );
    # add empty fields for previously unmatched barcodes
    for( my $tnum = $fnum; $tnum > 1; --$tnum ) {
      $targets{$trgid} .= "\t";
    }
  }
  my $prop = $fields[$propNum];
  # chrombase mode outputs mean base read depth per contig
  if( $contigRep == 2 ) {
    # maybe DIV#0 error if an invalid file was provided
    $prop = ($prop+$fields[4])/$fields[2];
    $prop = sprintf( "%.3f", $prop ) if( $prop >= 0.5 );
  }
  $targets{$trgid} .= $prop;
}
# sort arrays for each chromosome on target start then stop
while( my ($chr,$ary) = each(%sortlist) ) {
  @$ary = sort { $a->[0] <=> $b->[0] || $a->[1] <=> $b->[1] } @$ary;
}
# output matrix using amplicons sorted by chromosome, start, stop (even though these fields are not output)
if( $contigRep ) {
  print $propType eq "contig" ? "Contig\t" : "Chrom\t";
  print "Start\tEnd\t" if( $addCoords );
  print "$barcode_fields\n";
} else {
  print "Chrom\tStart\tEnd\t" if( $addCoords );
  print "Gene\tTarget\t$barcode_fields\n";
}
for( my $chrn = 0; $chrn < $numChroms; ++$chrn ) {
  my $chr = $chromName[$chrn];
  my $ary = $sortlist{$chr};
  for( my $i = 0; $i < scalar(@$ary); ++$i ) {
    my $subary = $ary->[$i];
    my $trgid = $subary->[3].':'.$subary->[0].'-'.$subary->[1];
    if( $contigRep ) {
      print "$chr\t";
      print "$subary->[0]\t$subary->[1]\t" if( $addCoords );
      print "$targets{$trgid}\n";
    } else {
      print "$chr\t$subary->[0]\t$subary->[1]\t" if( $addCoords );
      print "$subary->[2]\t$subary->[3]\t$targets{$trgid}\n";
    }
  }
}

