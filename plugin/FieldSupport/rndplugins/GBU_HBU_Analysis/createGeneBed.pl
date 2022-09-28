#!/usr/bin/perl
# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create a BED file of regions for gene CDS regions given intersection with TSS targets BED file.
An optional padding may be applied for the regions. Output to STDOUT.";
my $USAGE = "Usage:\n\t$CMD [options] <amplicon bed file> <gene CDS tab file>";
my $OPTIONS = "Options:
  -a        Auto-correct gene names by removing trailing characters after and including '_', e.g. for HGNC gene name
            matching. A warning will be output to STDERR for the number GENE_ID values corrected this way, if any.
  -t        Output input BED track line. Default: Ingore rack line(s) (output is easier to sort).
  -P <int>  Padding for output regions. (No merging is peformed.) Default: 0.
  -S <file> File for output of BED loading statistics. Default: '' => no output
  -U <file> File for output of genes that were Unmatched to genes in the master list. Ignored if gene CDS flle not provided.
  -h ? --help Display Help information.";

my $padding = 0;
my $fixgeneid = 0;
my $keepTrack = 0;
my $statout = '';
my $unmatchout = '';

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-a') {$fixgeneid = 1}
  elsif($opt eq '-t') {$keepTrack = 1}
  elsif($opt eq '-P') {$padding = int(shift)}
  elsif($opt eq '-S') {$statout = shift}
  elsif($opt eq '-U') {$unmatchout = shift}
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
my $nargs = scalar(@ARGV);
if( $nargs < 1 || $nargs > 2 ) {
  print STDERR "$CMD: Invalid number of arguments.";
  print STDERR "$USAGE\n";
  exit 1;
}

my $bedfile = shift(@ARGV);
my $cdsfile = shift(@ARGV);

#--------- End command arg parsing ---------

my %genes, %fixedids;
my ($namps,$ngenes,$ndups) = (0,0,0);
my $fixedGenes = 0;
my $track = "";
open( BEDFILE, "$bedfile" ) || die "Cannot open targets BED file $bedfile.\n";
while(<BEDFILE>) {
  if( /^track/ ) {
    $track = $_ if( $keepTrack );
    next;
  }
  next unless( /\S/ );
  chomp;
  if ( /GENE_ID=sid/ ) {
    next;
  }
  my @fields = split('\t',$_);
  my $aux = $fields[-1];
  ++$namps;
  # check for and get gene id(s)
  (my $gens = $aux) =~ s/(^|.*;)(GENE_ID=)([^;]+)(.*$)/\3/;
  next unless($2);
  my $nugens = 0;
  my %ugenes = ();
  foreach my $gene (split(',',$gens)) {
    $gene =~ s/^\s+|\s+$//g;
    if( $fixgeneid && $gene =~ m/_.*/ ) {
      ++$fixedGenes if( ++$fixedids{$gene} == 1 );
    }
    $gene =~ s/_.*// if( $fixgeneid );
    ++$ngenes if( ++$genes{$gene} == 1 );
    ++$nugens if( ++$ugenes{$gene} == 1 );
    # output the (padded filtered) amplicon inserts if not to be replaced by CDS regions
    # note: output is no longer grouped by gene ID if multiple genes per amplicon
    unless( $cdsfile ) {
      if( $header ) {
        print $header;
        $header = "";
      }
      my $srt = $fields[1]-$padding;
      my $end = $fields[2]+$padding;
      print "$fields[0]\t$srt\t$end\t$gene\n";
    }
  }
  ++$ndups if( $nugens > 1 );
}
close(BEDFILE);

my ($ncds,$ncdsgenes) = (0,0);
if( $cdsfile ) {
  my %cdsgenes;
  open( CDSFILE, "$cdsfile" ) || die "Cannot open input read coverage file $cdsfile.\n";
  while(<CDSFILE>) {
    my @fields = split('\t',$_);
    my $gene = $fields[2];
    next unless( defined($genes{$gene}) );
    ++$ncdsgenes if( ++$cdsgenes{$gene} == 1 );
    ++$ncds;
    my $srt = $fields[6]-$padding;
    my $end = $fields[7]+$padding;
    if( $header ) {
      print $header;
      $header = "";
    }
    print "$fields[3]\t$srt\t$end\t$gene\n";
  }
  close(CDSFILE);
  if( $unmatchout && $ncdsgenes != $ngenes ) {
    open( UNMOUT, ">", "$unmatchout" ) || die "Cannot open unmatched genes output file $unmatchout.\n";
    print UNMOUT "Gene ID\tAmplicon Refs\n";
    foreach my $gene ( sort(keys %genes) ) {
      print UNMOUT "$gene\t$genes{$gene}\n" unless( defined($cdsgenes{$gene}) );
    }
    close(UNMOUT);
  }
}

print STDERR "Warning: $fixedGenes gene IDs were auto-corrected to assumed HGNC name.\n" if( $fixedGenes );

if( $statout ) {
  if( open(STATOUT,">$statout") ) {
    print STATOUT "Number of genes:     $ngenes\n";
    print STATOUT "Number of amplicons: $namps\n";
    print STATOUT "Number of CDS genes matched:   $ncdsgenes\n";
    print STATOUT "Number of CDS regions covered: $ncds\n\n";
    print STATOUT "Number of gene names auto-corrected:   $fixedGenes\n" if( $fixgeneid );
    print STATOUT "Number of amplicons in multiple genes: $ndups\n";
    close(STATOUT);
  } else {
    print STDERR "WARNING: Could not open '$statout' for writing target stats.\n";
  }
}

