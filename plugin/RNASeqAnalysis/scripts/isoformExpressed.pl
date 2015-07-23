#!/usr/bin/perl
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
# Extracts the transcript isoform data to represented isoforms per gene

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Extracts the transcript isoform data for cufflinks output file to represented isoforms
per gene in the format: gene_id<tab>num_isoforms<tab>isoforms_detected<nl>. (Output to STDOUT.)
Whether an isoform is detected is judged by a threshol n its FPKM value.";
my $USAGE = "Usage:\n\t$CMD [options] <cufflinks isoforms.fpkm_tracking file>\n";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -F <value> FPKP threshold used to count detected isoforms. Default: 0.3";

my $fpkm_thres = 0.3;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-F') {$fpkm_thres = 0+shift;}
  elsif($opt eq '-h' || $opt eq "?" || $opt eq '--help') {$help = 1;}
  else {
    print STDERR "$CMD: Invalid option argument: $opt\n";
    print STDERR "$OPTIONS\n";
    exit 1;
  }
}
my $nargs = scalar(@ARGV);
if( $help ) {
  print STDERR "$DESCR\n";
  print STDERR "$USAGE\n";
  print STDERR "$OPTIONS\n";
  exit 1;
} elsif( $nargs != 1 ) {
  print STDERR "$CMD: Invalid number of arguments.\n";
  print STDERR "$USAGE\n";
  exit 1;
}

my $isofile = shift;

# ---------------- End of cmd arg processing -------------------

my $nisoforms = -1;
my %nisos,%risos;
open(ISOFORMS,$isofile) || die "$CMD: Error: could not open isoforms representation file '$isofile'\n";
while(<ISOFORMS>) {
  # skip field titles
  next unless( ++$nisoforms );
  my @fields = split('\t',$_);
  my ($gene,$fpkm) = ($fields[3],$fields[9]);
  ++$nisos{$gene};
  ++$risos{$gene} if( $fpkm >= $fpkm_thres);
}
close(ISOFORMS);
# output data to table
print "gene_id\tnum_isoforms\tisoforms_detected\n";
while( my ($gene_id,$numisos) = each %nisos ) {
  printf "$gene_id\t$numisos\t%d\n",$risos{$gene_id};
}

