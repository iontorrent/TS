#!/usr/bin/perl
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

# Simple script to re-format gene counts file for direct download

(my $CMD = $0) =~ s{^(.*/)+}{};
if( scalar(@ARGV) != 1 ) {
  print STDERR "Error: Invalid number of args: Usage:\n  $CMD <gene counts file>\n";
  exit 1;
}

my $genecountsfile = shift(@ARGV);

# always have at least header in output
print "Gene\tReads\n";

my %mapstats = ('no_feature'=>1,'ambiguous'=>1,'too_low_aQual'=>1,'not_aligned'=>1,'alignment_not_unique'=>1);
unless( -e $genecountsfile ) {
  print STDERR "WARNING: $CMD did not find gene counts file '$genecountsfile'\n";
  exit 0; # not a fatal for this script
}
open(GENECOUNTS,$genecountsfile) || die "ERROR: $cmd Could not open '$genecountsfile'";
while(<GENECOUNTS>) {
  chomp;
  my ($id,$cnt) = split;
  # skip unwanted stats
  next if( defined($mapstats{$id}) );
  print "$id\t$cnt\n";
}
close(GENECOUNTS);

