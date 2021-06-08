#!/usr/bin/perl
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Take the contig reads from one file and append as addition fields to the (base
coverage) records in another. This is a simple suport script that assumes a 1:1 correspondence
o the rows. Output to STDOUT.";
my $USAGE = "Usage:\n\t$CMD [options] <contig base reads file> <contig reads file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -a Add forward and reverse read count fields, rather than total.
  -t Also add total on-target reads, or forward and reverse ion-target reads with -a option.";

my $add_fwd_rev = 0;
my $inc_ontarget = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-a') {$add_fwd_rev = 1;}
  elsif($opt eq '-t') {$inc_ontarget = 1;}
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

my $basesfile = shift(@ARGV);
my $readsfile = shift(@ARGV);

#--------- End command arg parsing ---------

my $nlines = 0;
open( BASESFILE, "$basesfile" ) || die "Cannot open input base coverage file $basesfile.\n";
open( READSFILE, "$readsfile" ) || die "Cannot open input read coverage file $readsfile.\n";
while(<BASESFILE>) {
  chomp;
  print;
  if( ++$nlines == 1 ) {
    if( $add_fwd_rev ) {
      print "\tfwd_reads\trev_reads"; 
      print "\tfwd_trg_reads\trev_trg_reads" if( $inc_ontarget ); 
    } else {
      print "\ttotal_reads"; 
      print "\ttotal_trg_reads" if( $inc_ontarget ); 
    }
    printf "\n";
    <READSFILE>;
    next;
  }
  chomp( $_ = <READSFILE> );
  my @fields = split('\t',$_);
  if( $add_fwd_rev ) {
    print "\t$fields[1]\t$fields[2]"; 
    print "\t$fields[3]\t$fields[4]" if( $inc_ontarget ); 
  } else {
    printf "\t%.0f", $fields[1]+$fields[2]; 
    printf "\t%.0f", $fields[3]+$fields[4] if( $inc_ontarget ); 
  }
  printf "\n";
}
close(READSFILE);
close(BASESFILE);

