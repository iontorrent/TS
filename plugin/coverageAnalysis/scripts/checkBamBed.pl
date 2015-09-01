#!/usr/bin/perl
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Checks a given BED file only contains contigs that are present in given BAM file.
If they appear to be consistent then the script also checks to see if the BAM has any reads.
If no BED file is supplied, only the BAM file is checked.
Produces an error message to STDOUT if an issue if found or '' otherwise.
An error status is only issued for serious errors, e.g. missing files.";
my $USAGE = "Usage:\n\t$CMD [options] <bam file> <bed file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information";

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-h' || $opt eq "?" || $opt eq '--help') {$help = 1;}
  else {
    print STDERR "$CMD: Invalid option argument: $opt\n";
    print STDERR "$OPTIONS\n";
    exit 1;
  }
}
my $nargs = scalar @ARGV;
if( $help ) {
  print STDERR "$DESCR\n";
  print STDERR "$USAGE\n";
  print STDERR "$OPTIONS\n";
  exit 1;
} elsif( $nargs < 1 || $nargs > 2 ) {
  print STDERR "$CMD: Invalid number of arguments.\n";
  print STDERR "$USAGE\n";
  exit 1;
}

my $bamfile = shift(@ARGV);
my $bedfile = $nargs > 1 ? shift(@ARGV) : "";

#--------- End command arg parsing ---------

unless( -e $bamfile ) {
  print "Cannot find BAM file '$bamfile'\n";
  exit 1;
}

my %bamchrs;
my $nreads = 0;

unless( open( BAMREAD, "samtools idxstats '$bamfile' |" ) ) {
  print "Failed to open $bamfile - may not be BAM formatted.\n";
  exit 1;
}
while(<BAMREAD>) {
  next if(/^[*]/);
  my ($chr,$siz,$rds) = split('\t',$_);
  $bamchrs{$chr} = $siz+0;
  $nreads += $rds;
}
close(BAMREAD);

#print STDERR "Number of mapped reads = $nreads\n";

if( $bedfile ) {
  my $ntrack = 0;
  unless( -e $bedfile ) {
    print "Cannot find BED file '$bedfile'\n";
    exit 1;
  }
  unless( open( BEDREAD, $bedfile ) ) {
    print "Failed to open $bedfile.\n";
  }
  while(<BEDREAD>) {
    if( /^track/ ) {
      if( ++$ntrack > 1 ) {
        print "BED file has multiple tracks. Validation undefined.\n";
        exit 0;
      }
      next;
    }
    my ($chr,$srt,$end) = split('\t',$_);
    unless( defined( $bamchrs{$chr} ) ) {
      print "BED file is not valid for BAM file: Target '$chr' not found in BAM header.\n";
      exit 0;
    }
    if( $end > $bamchrs{$chr} ) {
      print "BED file is not valid for BAM file: Target '$chr:$srt-$end' extends beyond the size of the contig ($bamchrs{$chr}).\n";
      exit 0;
    }
  }
  close(BEDREAD);
}  
unless( $nreads ) {
  print "BAM file has no mapped reads.\n";
  exit 0;
}
exit 0;

