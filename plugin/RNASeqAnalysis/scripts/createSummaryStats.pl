#!/usr/bin/perl
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved

# Simple script to do a quick coverage analysis for the targets.
# Basically create summary stats for the reports. Output to STDERR as stat:value (like basic json).

use File::Basename;

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Collect statistcs from input files and write summary to STDOUT (as a json-like format).
Also collect alignment feature reads counts to the specified file for report summary plots.";
my $USAGE = "Usage:\n$CMD [options] <alignment summary file> <gene counts file> <RNA summary file> <feature counts output file>\n";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -d Add some extra statists to the output for debuging pursposes. (Independent from those added by the -f option.)
  -f Add Full statistics, including ambigous mappings and mismatch/indel rates.
  -r <file> Take ribo reads counts from the given file and adjust other statistics accordingly. Adjusted percentages are
     calculated from modified total aligned reads so to be consistent with (picard) overcounting in separate catagories.";

my $debug = 0;
my $fullstats = 0;
my $xrRNAfile = "";

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-d') {$debug = 1;}
  elsif($opt eq '-f') {$fullstats = 1;}
  elsif($opt eq '-r') {$xrRNAfile = shift;}
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
} elsif( $nargs != 4 ) {
  print STDERR "$CMD: Invalid number of arguments.\n";
  print STDERR "$USAGE\n";
  exit 1;
}

my $alignstatsfile = shift(@ARGV);
my $genecountsfile = shift(@ARGV);
my $RNAhitsfile = shift(@ARGV);
my $featureoutfile = shift(@ARGV);

#----------- End arg processing ----------

# grab all the fields from the alignment summary file
my @fieldTitles = ();
my %mapMetrics;
my $grabLine = 0;
open(ALIGNSTATS,$alignstatsfile) || die "Could not open '$alignstatsfile'";
while(<ALIGNSTATS>) {
  chomp;
  my @fields = split('\t',$_);
  if( $fields[0] eq "## METRICS CLASS" ) {
    $grabLine = 1;
    next;
  }
  next unless( $grabLine );
  if( ++$grabLine == 2 ) {
   @fieldTitles = @fields;
   next;
  }
  last if( $grabLine > 3 );
  for( my $i = 0; $i < scalar(@fieldTitles); ++$i ) {
    $mapMetrics{$fieldTitles[$i]} = $fields[$i];
  }
}
close(ALIGNSTATS);

my @covlevs = (1,10,100,1000,10000);
my $nlevs = scalar(@covlevs);
my @covcnts = ((0) x $nlevs);
my @mapstats = ('no_feature','ambiguous','too_low_aQual','not_aligned','alignment_not_unique');
my %genecovstats;
@genecovstats{@mapstats} = ((0) x scalar(@mapstats));
my ($numGenes,$numGeneMaps) = (0,0);
if( -e $genecountsfile ) {
  open(GENECOUNTS,$genecountsfile) || die "Could not open '$genecountsfile'";
  while(<GENECOUNTS>) {
    my ($id,$cnt) = split;
    if( defined($genecovstats{$id}) ) {
      $genecovstats{$id} = $cnt+0;
      next;
    }
    ++$numGenes;
    $numGeneMaps += $cnt;
    for( my $i = 0; $i < $nlevs; ++$i ) {
      ++$covcnts[$i] if( $cnt >= $covlevs[$i] );
    }
  }
  close(GENECOUNTS);
}

# grab all the fields from the RNA summary file
@fieldTitles = ();
my %rnaMetrics;
my $grabLine = 0;
open(RNASTATS,$RNAhitsfile) || die "Could not open '$RNAhitsfile'";
while(<RNASTATS>) {
  chomp;
  my @fields = split('\t',$_);
  if( $fields[0] eq "## METRICS CLASS" ) {
    $grabLine = 1;
    next;
  }
  next unless( $grabLine );
  if( ++$grabLine == 2 ) {
   @fieldTitles = @fields;
   next;
  }
  last if( $grabLine > 3 );
  for( my $i = 0; $i < scalar(@fieldTitles); ++$i ) {
    $rnaMetrics{$fieldTitles[$i]} = $fields[$i]+0;
  }
}
close(RNASTATS);

# read option statistics file(s) - currently just for xrRNA
my $xrRNA = 0;
if( $xrRNAfile ne "" && $mapMetrics{'TOTAL_READS'} > 0 ) {
  if( open(XRFILE,$xrRNAfile) ) {
    # only a single number read for now but latter could have more formatted data
    while(<XRFILE>) {
      $xrRNA = int($_);
      last;
    }
    close(XRFILE);
  } else {
    print STDERR "WARNING: $CMD could not open file '$xrRNAfile'\n";
  }
}
# adjust stats using extra ribo reads (allows for negative values, although not expected)
if( $xrRNA != 0 ) {
  # some stats cannot be adjusted, e.g. PF_MISMATCH_RATE, but is probably unnecesary
  my $tot = $rnaMetrics{'PF_ALIGNED_BASES'} + $xrRNA;
  $rnaMetrics{'PF_ALIGNED_BASES'} = $tot;
  $rnaMetrics{'RIBOSOMAL_BASES'} += $xrRNA;
  $rnaMetrics{'PCT_CODING_BASES'} = $rnaMetrics{'CODING_BASES'} / $tot;
  $rnaMetrics{'PCT_UTR_BASES'} = $rnaMetrics{'UTR_BASES'} / $tot;
  $rnaMetrics{'PCT_MRNA_BASES'} = $rnaMetrics{'PCT_CODING_BASES'} + $rnaMetrics{'PCT_UTR_BASES'};
  $rnaMetrics{'PCT_RIBOSOMAL_BASES'} = $rnaMetrics{'RIBOSOMAL_BASES'} / $tot;
  $rnaMetrics{'PCT_INTRONIC_BASES'} = $rnaMetrics{'INTRONIC_BASES'} / $tot;
  $rnaMetrics{'PCT_INTERGENIC_BASES'} = $rnaMetrics{'INTERGENIC_BASES'} / $tot;
  $rnaMetrics{'PCT_PF_READS_ALIGNED'} = $tot / $rnaMetrics{'PF_BASES'};
  $rnaMetrics{'PCT_USABLE_BASES'} = $rnaMetrics{'PCT_PF_READS_ALIGNED'} * $rnaMetrics{'PCT_MRNA_BASES'};
}

printf "Total Reads:      %.0f\n",$mapMetrics{'TOTAL_READS'};
printf "Aligned Reads:    %.0f\n",$mapMetrics{'PF_READS_ALIGNED'};
printf "Pct Aligned:      %.2f%%\n",100*$mapMetrics{'PCT_PF_READS_ALIGNED'};
printf "Mean Read Length: %.1f\n",$mapMetrics{'MEAN_READ_LENGTH'};
printf "Strand Balance:   %.4f\n",$mapMetrics{'STRAND_BALANCE'};
print "\n";
printf "Reference Genes:  %.0f\n",$numGenes;
printf "Reads Mapped to Genes: %.0f\n",$numGeneMaps;
for( my $i = 0; $i < $nlevs; ++$i ) {
  printf "Genes with %d+ reads: %.0f\n",$covlevs[$i], $covcnts[$i];
}
if( $fullstats ) {
  print "\n";
  printf "Unmapped Reads:  %.0f\n",$genecovstats{'not_aligned'};
  printf "Low Qual Reads:  %.0f\n",$genecovstats{'too_low_aQual'};
  printf "Non-unique Maps: %.0f\n",$genecovstats{'alignment_not_unique'};
  printf "No Feature Maps: %.0f\n",$genecovstats{'no_feature'};
  printf "Ambiguous Maps:  %.0f\n",$genecovstats{'ambiguous'};
  printf "Pct Fusions:     %.2f%%\n",100*$mapMetrics{'PCT_CHIMERAS'};
  printf "Mismatch Rate:   %.4f\n",$mapMetrics{'PF_MISMATCH_RATE'};
  printf "InDel Rate:      %.4f\n",$mapMetrics{'PF_INDEL_RATE'};
}
print "\n";
my $totalBases = $rnaMetrics{'PF_BASES'};
my $alignBases = $rnaMetrics{'PF_ALIGNED_BASES'};
printf "Total Base Reads:     %.0f\n",$totalBases;
printf "  Pct Aligned Bases:  %.2f%%\n",($totalBases > 0 ? 100*$alignBases/$totalBases : 0);
printf "  Pct Usable Bases:   %.2f%%\n",100*$rnaMetrics{'PCT_USABLE_BASES'};
printf "Total Aligned Bases:  %.0f\n",$alignBases;
printf "Pct mRNA Bases:       %.2f%%\n",100*$rnaMetrics{'PCT_MRNA_BASES'};
printf "  Pct Coding Bases:   %.2f%%\n",100*$rnaMetrics{'PCT_CODING_BASES'};
printf "  Pct UTR Bases:      %.2f%%\n",100*$rnaMetrics{'PCT_UTR_BASES'};
printf "Pct Ribosomal Bases:  %.2f%%\n",100*$rnaMetrics{'PCT_RIBOSOMAL_BASES'};
printf "Pct Intronic Bases:   %.2f%%\n",100*$rnaMetrics{'PCT_INTRONIC_BASES'};
printf "Pct Intergenic Bases: %.2f%%\n",100*$rnaMetrics{'PCT_INTERGENIC_BASES'};

unless( open( CNTOUT, ">$featureoutfile" ) ) {
  print STDERR "WARNING: $CMD: Could open '$featureoutfile' for output.\n";
  exit 0; # do not consider a serious error here
}

# Note: order changed here for sake of pie chart (Ribo low frac between big slices for readability)
print CNTOUT "Feature_Name\tReads\n";
printf CNTOUT "Coding\t%.0f\n",$rnaMetrics{'CODING_BASES'};
printf CNTOUT "Ribosomal\t%.0f\n",$rnaMetrics{'RIBOSOMAL_BASES'};
printf CNTOUT "UTR\t%.0f\n",$rnaMetrics{'UTR_BASES'};
printf CNTOUT "Intronic\t%.0f\n",$rnaMetrics{'INTRONIC_BASES'};
printf CNTOUT "Intergenic\t%.0f\n",$rnaMetrics{'INTERGENIC_BASES'};

if( $debug ) {
  printf CNTOUT "Sum Feature\t%.0f\n",$rnaMetrics{'CODING_BASES'}+$rnaMetrics{'UTR_BASES'}+$rnaMetrics{'RIBOSOMAL_BASES'}+$rnaMetrics{'INTRONIC_BASES'}+$rnaMetrics{'INTERGENIC_BASES'};
  printf CNTOUT "Aligned Bases\t%.0f\n",$rnaMetrics{'PF_ALIGNED_BASES'};
  printf CNTOUT "\nTotal Bases\t%.0f\n",$rnaMetrics{'PF_BASES'};
  printf CNTOUT "Fraction Aligned Bases\t%f\n",$alignBases/$totalBases;
  printf CNTOUT "Fraction Usable Bases\t%f\n",$rnaMetrics{'PCT_USABLE_BASES'};
  printf CNTOUT "Usable Aligned (of total)\t%.0f\n",$rnaMetrics{'PF_BASES'}*$rnaMetrics{'PCT_USABLE_BASES'};
  printf CNTOUT "Ignored Reads\t%.0f\n",$rnaMetrics{'IGNORED_READS'};
}
close(CNTOUT);

