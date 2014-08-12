#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Add an extra column to the input file or total chromosome/contig reads. Output to STDOUT.
Assumes the input file is TSV with the first field representing the chromosome/contig names.";
my $USAGE = "Usage:\n\t$CMD [options] <contig cov file> <BAM file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -d Ignore (PCR) Duplicate reads.
  -u Include only Uniquely mapped reads (MAPQ > 1).
  -m <int> Many contigs threshold for switching counting algorithms. Default: 50.";

my $nondupreads = 0;
my $uniquereads = 0;
my $manyContigs = 50;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-d') {$nondupreads = 1;}
  elsif($opt eq '-u') {$uniquereads = 1;}
  elsif($opt eq '-m') {$manyContigs = int(shift);}
  elsif($opt eq '-h' || $opt eq "?" || $opt eq '--help') {$help = 1;}
  else {
    print STDERR "$CMD: Invalid option argument: $opt\n";
    print STDERR "$OPTIONS\n";
    exit 1;
  }
}
if( $help ) {
  print STDERR "$DESCR\n";
  print STDERR "$USAGE\n";
  print STDERR "$OPTIONS\n";
  exit 1;
}
elsif( scalar @ARGV != 2 ) {
  print STDERR "$CMD: Invalid number of arguments.";
  print STDERR "$USAGE\n";
  exit 1;
}

my $tsvfile = shift(@ARGV);
my $bamfile = shift(@ARGV);

#--------- End command arg parsing ---------

my $samopt= ($nondupreads ? "-F 0x704" : "-F 0x304").($uniquereads ? " -q 1" : "");

# make quick test to see if this is a valid bam file
my $numContigs = 0;
open( BAMTEST, "samtools view -H \"$bamfile\" |" ) || die "Cannot open BAM file '$bamfile'.";
while(<BAMTEST>) {
  ++$numContigs if(/^\@SQ/);
}
close(BAMTEST);
#print STDERR "Found $numContigs contigs\n";
unless( $numContigs ) {
  print STDERR "Error: BAM file unaligned or badly formatted.\n";
  exit 1;
}

# if 'many' contigs it is more efficient to parse once for contig counts than using many samtools view calls
my %chrids;
my $chrid;
unless( $numContigs < $manyContigs ) {
  open( BAM, "samtools view $samopt \"$bamfile\" |" ) || die "Cannot open BAM file '$bamfile'.";
  while(<BAM>) {
    $chrid = (split('\t',$_))[2];
    ++$chrids{$chrid};
  }
}

my $nlines = 0;
my $line;
open( TSVFILE, "$tsvfile" ) || die "Cannot open input TSV file $tsvfile.\n";

# some code duplication to avoid using inner condition
if( $numContigs < $manyContigs ) {
  while(<TSVFILE>) {
    chomp;
    next unless(/\S/);
    if( ++$nlines == 1 ) {
      print "$_\tseq_reads\n";
      next;
    }
    $line = $_;
    $chrid = (split('\t',$_))[0];
    next if( $chrid !~ /\S/ );
    my $nreads = `samtools view -c $samopt $bamfile "$chrid"`;
    printf "$line\t%d\n",$nreads;
  }
} else {
  while(<TSVFILE>) {
    chomp;
    next unless(/\S/);
    if( ++$nlines == 1 ) {
      print "$_\tseq_reads\n";
      next;
    }
    $line = $_;
    s/\t.*$//;
    printf "$line\t%d\n",$chrids{$_};
}

}
close(TSVFILE);

