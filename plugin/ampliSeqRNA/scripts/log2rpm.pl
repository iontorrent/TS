#!/usr/bin/perl
# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
# Utility script for converting RPM reads to log2(rpm+1), as inermediate for conversion to CHP format.
# This script is for general usage and performs no parameter or file format checking, etc.
# First arg is number of barcodes followed by (one or more) RPM input file

my $ln2 = 1/log(2);

my $nbc = 1+shift(@ARGV);
my $nline = 0;
my $gene = "";
while(<>) {
  chomp;
  my @fields = split('\t',$_);
  for( my $i = 0; $i <= $nbc; ++$i ) {
    my $v = $fields[$i];
    if( $i == 0 ) {
      # record gene name for preferential output
      $gene = $v;
      next;
    }
    if( $i == 1 ) {
      # use gene name if known - else ampID
      $v = $gene if( $nline && $gene && $gene ne "N/A" );
      # there is a hard limit on amplion/probe id string length in TAC
      print substr($v,0,16);
      next;
    } else {
      print "\t";
    }
    if( $nline ) {
      # conversion of RPM values
      print $ln2*log($v+1);
    } else {
      # field name for barcode id fields
      print $v;
    }
  }
  print "\n";
  ++$nline;
}

