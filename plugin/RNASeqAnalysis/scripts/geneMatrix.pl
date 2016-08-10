#!/usr/bin/perl
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
# Only partly re-factored for more general usage.

use File::Basename;

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create a barcodes x targets property the same field extracted from a number of (barcode) table files.
This assumes the first row contains (unique) gene names and the second their read counts.
It also expects that all files have the same number of rows in the same order (checked).
Column names reflect the folder names each file resides in.";
my $USAGE = "Usage:\n\t$CMD [options] <file1> [<file2] ...]\n";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -r Convert read counts to RPM values by multiplying by 100000/total reads per barcode (file).";

my $toRPM = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-r') {$toRPM = 1;}
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
} elsif( $nargs < 1 ) {
  print STDERR "$CMD: Invalid number of arguments.\n";
  print STDERR "$USAGE\n";
  exit 1;
}

my %grows;
my (@genes,@reads);
my ($barcode,$barcode_fields,$firstArg);
my ($fnum,$lnum,$nrds);
while(<>) {
  my $fn = basename(dirname($ARGV));
  if( $fn ne $barcode ) {
    # process pending list of reads
    if( ++$fnum == 1 ) {
      $firstArg = dirname($ARGV);
    } else {
      my $rpm = ($toRPM && $nrds) ? 1000000 / $nrds : 1;
      for( my $i = 0; $i < scalar(@reads); ++$i ) {
        my $rds = $reads[$i] * $rpm;
        $grows{$genes[$i]} .= sprintf($toRPM ? "\t%.3f" : "\t%.0f",$rds);
      }
    }
    $barcode = $fn;
    $barcode_fields .= "\t$fn";
    $lnum = 0;
    $nrds = 0;
    next;  # skip header line
  }
  my @fields = split;
  # check id are consisent in all files
  if( $fnum == 1 ) {
    push(@genes,$fields[0]);
    push(@reads,0+$fields[1]);
  } elsif( $genes[$lnum] ne $fields[0] ) {
    printf STDERR "ERROR: $CMD: Gene name at line %d of file %d (%s) '%s' does not match that in file 1 (%s) = %s.\n",
      $lnum+1, $fnum, dirname($ARGV), $fields[0], $firstArg, $genes[$lnum];
    exit 1;
  } else {
    $reads[$lnum] = 0+$fields[1];
  }
  $nrds += $fields[1];
  ++$lnum;
}
# add in data for last file
if( $fnum ) {
  my $rpm = ($toRPM && $nrds) ? 1000000 / $nrds : 1;
  for( my $i = 0; $i < scalar(@reads); ++$i ) {
    my $rds = $reads[$i] * $rpm;
    $grows{$genes[$i]} .= sprintf($toRPM ? "\t%.3f" : "\t%.0f",$rds);
  }
}
# output all rows to table
print "Gene$barcode_fields\n";
for( my $i = 0; $i < scalar(@genes); ++$i ) {
  printf "%s%s\n", $genes[$i], $grows{$genes[$i]};
}

