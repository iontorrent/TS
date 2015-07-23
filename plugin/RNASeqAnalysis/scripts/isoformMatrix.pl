#!/usr/bin/perl
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
# Only partly re-factored for more general usage.

use File::Basename;

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create a barcodes x isoform FPKM matrix by extracting fields of list of individual files.";
my $USAGE = "Usage:\n\t$CMD [options] <file1> [<file2] ...]\n";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -s Strict matching of isoform ids by corresponding rows";

my $sep = "_";
my $matchId = 1;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-s') {$matchId = 0;}
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

my (%bcdata,%isoidx);
my (@isoform,@fpkms);
my ($barcode,$barcode_fields,$firstArg);
my ($fnum,$lnum);
while(<>) {
  my $fn = basename(dirname($ARGV));
  if( $fn ne $barcode ) {
    # record first file name for error msgs or completed row data
    if( ++$fnum == 1 ) {
      $firstArg = dirname($ARGV);
    } else {
      for( my $i = 0; $i < scalar(@fpkms); ++$i ) {
        $bcdata{$isoform[$i]} .= "\t".$fpkms[$i];
      }
    }
    $barcode = $fn;
    $barcode_fields .= "\t$fn";
    $lnum = 0;
    next;  # skip header line
  }
  my @fields = split;
  # check id are consistent in all files
  my $isoId = $fields[4].$sep.$fields[0];
  if( $fnum == 1 ) {
    $isoidx{$isoId} = scalar(@isoform);
    push(@isoform,$isoId);
    push(@fpkms,0+$fields[9]);
  } elsif( $matchId ) {
    if( defined($isoidx{$isoId}) ) {
      $fpkms[$isoidx{$isoId}] = 0+$fields[9];
    } else {
      printf STDERR "ERROR: $CMD: Iosoform name at line %d of file %d (%s) '%s' does not match any in file 1 (%s) = %s.\n",
        $lnum+1, $fnum, dirname($ARGV), $isoId, $firstArg, $isoform[$lnum];
      exit 1;
    }
  } elsif( $isoform[$lnum] ne $isoId ) {
    printf STDERR "ERROR: $CMD: Iosoform name at line %d of file %d (%s) '%s' does not match at same line in in file 1 (%s) = %s.\n",
      $lnum+1, $fnum, dirname($ARGV), $isoId, $firstArg, $isoform[$lnum];
    exit 1;
  } else {
    $fpkms[$lnum] = 0+$fields[9];
  }
  ++$lnum;
}
# add in data for last file
if( $fnum > 0 ) {
  for( my $i = 0; $i < scalar(@fpkms); ++$i ) {
    $bcdata{$isoform[$i]} .= "\t".$fpkms[$i];
  }
}
# output all rows to table
print "Gene_Isoform$barcode_fields\n";
for( my $i = 0; $i < scalar(@isoform); ++$i ) {
  printf "%s%s\n", $isoform[$i], $bcdata{$isoform[$i]};
}

