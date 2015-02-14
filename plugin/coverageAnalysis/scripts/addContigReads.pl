#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create a file of forward, reverse and total contig read counts for a given file.
Or add an extra column to the input file or total chromosome/contig reads. Output to STDOUT.";
my $USAGE = "Usage:\n\t$CMD [options] <BAM file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -a Add forward and reverse read count fields (in addition to total).
  -d Ignore (PCR) Duplicate reads.
  -u Include only Uniquely mapped reads (MAPQ > 1).
  -M <int>  Many contigs threshold for switching counting algorithms. Default: 50.
  -C <file> Contig coverage TSV file to add total contig reads to as last field.";

my $add_fwd_rev = 0;
my $nondupreads = 0;
my $uniquereads = 0;
my $manyContigs = 50;
my $tsvfile = "";

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-a') {$add_fwd_rev = 1;}
  elsif($opt eq '-d') {$nondupreads = 1;}
  elsif($opt eq '-u') {$uniquereads = 1;}
  elsif($opt eq '-M') {$manyContigs = int(shift);}
  elsif($opt eq '-C') {$tsvfile = shift;}
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
elsif( scalar @ARGV != 1 ) {
  print STDERR "$CMD: Invalid number of arguments.";
  print STDERR "$USAGE\n";
  exit 1;
}

$tsvfile = '' if( $tsvfile eq '-' );
my $bamfile = shift(@ARGV);

#--------- End command arg parsing ---------

my $samopt= ($nondupreads ? "-F 0x704" : "-F 0x304").($uniquereads ? " -q 1" : "");

# read total reads and contig info using idxstats
my (@chrids,@chrszs,@chrrds,@chrFwd,@chrRev);
my %chridx;
my $numContigs = 0;
open( BAMTEST, "samtools idxstats '$bamfile' |" ) || die "Cannot open BAM file '$bamfile'.";
while(<BAMTEST>) {
  my @fields = split('\t',$_);
  next if( $fields[0] eq '*' );
  $chrids[$numContigs] = $fields[0];
  $chrszs[$numContigs] = $fields[1];
  $chrrds[$numContigs] = $fields[2];
  $chridx{$fields[0]} = $numContigs;
  ++$numContigs;
}
close(BAMTEST);

unless( $numContigs ) {
  print STDERR "Error: BAM file unaligned or badly formatted.\n";
  exit 1;
}

# unless idxstats is sufficient, get the contigs read stats from the bamfile
if( $nondupreads || $uniquereads || $add_fwd_rev ) {
  # for 'many' contigs it is more efficient to parse once for contig counts than using many samtools view calls
  if( $numContigs < $manyContigs ) {
    my $samopt_f = ($nondupreads ? "-F 0x714" : "-F 0x314").($uniquereads ? " -q 1" : "");
    my $samopt_r = $samopt." -f 16";
    for( my $chrIdx = 0; $chrIdx < $numContigs; ++$chrIdx ) {
      $chrFwd[$chrIdx] = `samtools view -c $samopt_f "$bamfile" "$chrids[$chrIdx]"`;
      $chrRev[$chrIdx] = `samtools view -c $samopt_r "$bamfile" "$chrids[$chrIdx]"`;
    }
  } else {
    my $lastChr = ''; # to avoid excessive hash lookups
    my $chrIdx = 0;
    open( BAM, "samtools view $samopt '$bamfile' |" ) || die "Cannot open BAM file '$bamfile'.";
    while(<BAM>) {
      my ($rdid,$flg,$chr) = split('\t',$_);
      if( $chr ne $lastChr ) {
        $chrIdx = $chridx{$chr};
        $lastChr = $chr;
      }
      if( $flg & 16 ) { ++$chrRev[$chrIdx] } else { ++$chrFwd[$chrIdx] }
    }
  }
  # overwrite idxstats read counts with filtered read counts
  for( my $chrIdx = 0; $chrIdx < $numContigs; ++$chrIdx ) {
    $chrrds[$chrIdx] = $chrFwd[$chrIdx] + $chrRev[$chrIdx];
  }
}

if( $tsvfile ) {
  my $nlines = 0;
  my ($line,$chr);
  open( TSVFILE, "$tsvfile" ) || die "Cannot open input TSV file $tsvfile.\n";
  while(<TSVFILE>) {
    chomp;
    next unless(/\S/);
    if( ++$nlines == 1 ) {
      printf "$_\t%stotal_reads\n", $add_fwd_rev ? "fwd_reads\trev_reads\t" : "";
      next;
    }
    $line = $_;
    s/\t.*$//;
    $chrIdx = $chridx{$_};
    if( $add_fwd_rev ) {
      printf "$line\t%d\t%d\t%d\n",$chrFwd[$chrIdx],$chrRev[$chrIdx],$chrrds[$chrIdx];
    } else {
      printf "$line\t%d\n",$chrrds[$chrIdx];
    }
  }
  close(TSVFILE);
} else {
  printf "contig\tstart\tend\t%stotal_reads\n", $add_fwd_rev ? "fwd_reads\trev_reads\t" : "";
  for( my $chrIdx = 0; $chrIdx < $numContigs; ++$chrIdx ) {
    if( $add_fwd_rev ) {
      printf "%s\t1\t%d\t%d\t%d\t%d\n",$chrids[$chrIdx],$chrszs[$chrIdx],$chrFwd[$chrIdx],$chrRev[$chrIdx],$chrrds[$chrIdx];
    } else {
      printf "%s\t1\t%d\t%d\n",$chrids[$chrIdx],$chrszs[$chrIdx],$chrrds[$chrIdx];
    }
  }
}

