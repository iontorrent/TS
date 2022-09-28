#!/usr/bin/perl
# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create Gene Base Uniformiy given a BBC file and target regions BED file.
Optionally report Panel Base Uniformity and create gene base coverage files in WIG (compatible) format.
NOTE: The input BED file must have 4 columns and (group) sorted by gene name as the 4th field and then by start/end
coordinates for the GBU grouping. Output for GBU are a set of base coverage stats in CSV (table) format to STDOUT.";
my $USAGE = "Usage:\n\t$CMD [options] <BBC file> <gene-grouped sorted BED file>";
my $OPTIONS = "Options:
  -p Output PBU stats to STDOUT and do not create GBU and WIG output to STDOUT. All other options are ignored.
  -E <exec> Path to BBCtools executable. Default: 'bbctools' (executable in path).
  -F <file> Full panel BED file for panel stats and MBRD threshold (w/o -M). Default: '' => is input BED.
  -G <file> Input file of total target region length of each gene. If this is provided the output will have
            additional fields and different field names to distinguish Panel-Gene-Intersect vs. Gene (full CDS).
  -L <int>  Number of bases expected in projected regions of interest. Default: Use sum of gene lengths proved by -G.
  -M <num>  Mean read depth used for GBU/PBU calculations. Default: Use mean base read depth for region provided.
  -T <num>  Percentage threshold of mean read depth to compare with. Default: 0.2
  -S <file> File for output for PBU statistics. Default: '' => no output.
  -W <path> Create gene coverage WIG files using output path provided. Use '.' for current dir. Default: '' => no output.
  -X <file> Create WIG files using this gene-grouped sorted BED file rather than the input (where GBU is calculated).
  -h ? --help Display Help information.";

my $bbctools = "bbctools";
my $statout = '';
my $mbrd = 0;
my $mthr = 0.2;
my $cdslenFile = '';
my $roiTotalLen = 0;
my $wigDir = "";
my $fullpanel = '';
my $wigPanel = "";
my $noGBUout = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-p') {$noGBUout = 1}
  elsif($opt eq '-E') {$bbctools = shift}
  elsif($opt eq '-F') {$fullpanel = shift}
  elsif($opt eq '-G') {$cdslenFile = shift}
  elsif($opt eq '-L') {$roiTotalLen = int(shift)}
  elsif($opt eq '-M') {$mbrd = shift}
  elsif($opt eq '-T') {$mthr = shift}
  elsif($opt eq '-S') {$statout = shift}
  elsif($opt eq '-W') {$wigDir = shift}
  elsif($opt eq '-X') {$wigPanel = shift}
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
if( $nargs != 2 ) {
  print STDERR "$CMD: Invalid number of arguments.";
  print STDERR "$USAGE\n";
  exit 1;
}

my $bbcfile = shift(@ARGV);
my $bedfile = shift(@ARGV);

$mbrd = -1 if( $mbrd == "" );  # empty string -> -1 => use base MRD from whole panel
$mbrd += 0;                    # non-empty, non-numeric string becomes 0

$wigDir = "" if( $wigDir eq "-" );
$wigPanel = "" if( $wigPanel eq "-" );
$wigDir = "." if( $wigPanel && $wigDir eq "" );

# disable any conflicting options with noGBUout option
if( $noGBUout ) {
  $statsout = "";
  $fullpanel = "";
}
$fullpanel = $bedfile if( $fullpanel eq "" || $fullpanel eq "-" );

#--------- End command arg parsing ---------

# generate the PBU stats for the whole panel first - this gives the mean base read depth required for GBU

if( $statout ) {
  open(STATOUT,">",$statout) || die "Could not open output file $statout.\n";
}
my ($abrd,$totb) = (-1,-1,-1);
open(PBUOUT,"$bbctools report -gR '$fullpanel' '$bbcfile' |") || die "Could not run BBCtools report command: $!\n";
while(<PBUOUT>) {
  chomp;
  next unless( m/(.*)(:[ ]*)(.*$)/ );
  my $stat = $1;
  my $val = 0+$3;
  $abrd = $val if( $stat eq "Average base coverage depth" );
  $totb = $val if( $stat eq "Total base reads on target" );
  print STATOUT "$_\n" if( $statout );
  print "$_\n" if( $noGBUout );
}
close(PBUOUT);
if( $statout ) {
  # the following tracks the actual base MRD used but removed here to keep output all for whole panel
  #print STATOUT "\nMean base read depth used for GBU: $mbrd\n";
  close(STATOUT);
}
# do nothing else with noGBUout option
exit 0 if( $noGBUout );

# load gene 'CDS' lengths if provided
my %CDSLen = ();
my %outputGBU = ();
my $haveGeneLen = 0;
my $totalCDSLen = 0;
if( $cdslenFile && $cdslenFile ne "-" ) {
  $haveGeneLen = open(CDSLEN,$cdslenFile);
  if( $haveGeneLen ) {
    while(<CDSLEN>) {
      chomp;
      my ($gene,$length) = split('\t',$_);
      $CDSLen{$gene} = 0+$length;
      $totalCDSLen += $length;
    }
    close(CDSLEN);
  } else {
    print STDERR "WARNING; Could not open file '$cdslenFile'. Total gene TargetLen not added to output.\n";
  }
}

# load wigPanel to memory
my %wigAmplicons = ();
if( $wigDir && $wigPanel ) {
  my $genekey = "GENE_ID=$lastGene;";
  open(WIGBED,$wigPanel) || die "Could not read from (-X) '$wigPanel'\n";
  while(<WIGBED>) {
    next unless( /\S/ );
    next if( /^track\s/ );
    chomp;
    if( /(.*GENE_ID=)(.+?)(;.*)/ ) {
      my $geneList = $2;
      for my $gene (split(',',$geneList)) {
        $gene =~ s/^\s+|\s+$//g;
        push( @{$wigAmplicons{$gene}}, $_ );
      }
    }
  }
  close(WIGBED);
}

# get mean base read depth for panel BED as passed
$mbrd = $abrd if( $mbrd < 0 );
if( $mbrd < 0 ) {
  print STDERR "ERROR: $CMD could not read base MRD from '$bbcfile'\n";
  exit(1);
}
# Get adjusted mean base read depth for full gene CDS regions. This is to test how including 0x
# coverage beyond panel/gene instersection (PGI) affects the numerator of GBU calculation.
# (Neglible where little in-silico 0x)
# But this is not necessary since the MBRD is now used as the threshold by definition.
$totalCDSLen = $roiTotalLen if( $roiTotalLen > 0 );
my $gmbrd = $totalCDSLen ? $totb / $totalCDSLen : 0;

# output the GBU for each set of target regions assmed to be grouped by gene (4th field of BED)
# Note: stats recorded no longer match original APD_plugin output
printf "Gene,MinCov,MaxCov,PGIMBRD,PGIBU,PGISUB,Cov1x,Cov100x,Cov350x,Cov500x,PGILen%s\n",($haveGeneLen ? ",GLen,GMBRD,GBU,GSUB" : "");

my $tmpgbubed = "gene_cds.tmp.bed";
my $lastGene, $lastChr;
my ($reglen,$lastSrt,$lastEnd) = (0,0,-1);

open(BEDIN,$bedfile) || die "Could not read from '$bedfile'\n";
while(<BEDIN>) {
  next unless( /\S/ );
  next if( /^track\s/ );
  chomp;
  my ($chr,$srt,$end,$gene) = split('\t',$_);
  if( $gene ne $lastGene ) {
    # output pending merged region
    if( $lastEnd > 0 ) {
      $reglen += $lastEnd - $lastSrt;
      print TMPBED "$lastChr\t$lastSrt\t$lastEnd\t$lastGene\n";
      close(TMPBED);
      calcGBU();
    }
    $lastGene = $gene;
    $lastChr = $chr;
    $lastEnd = -1;  # in case next region actually starts at 0 (typically impossible for genes)
    $reglen = 0;
    open(TMPBED,">",$tmpgbubed) || die "Could not write local file '$tmpgbubed'\n";
  }
  # delay output of region until established it does nt overlap another for the same gene
  # note: assumes the regions are correctly ordered for the gene group
  if( $srt > $lastEnd ) {
    if( $lastEnd > 0 ) {
      $reglen += $lastEnd - $lastSrt;
      print TMPBED "$lastChr\t$lastSrt\t$lastEnd\t$lastGene\n";
    }
    $lastSrt = $srt;
    $lastEnd = $end;
  } elsif( $end > $lastEnd ) {
    # merge to (expand) pending region
    $lastEnd = $end;
  }
}
close(BEDIN);
if( $lastEnd > 0 ) {
  $reglen += $lastEnd - $lastSrt;
  print TMPBED "$lastChr\t$lastSrt\t$lastEnd\t$lastGene\n";
  close(TMPBED);
  calcGBU();
}
unlink($tmpgbubed);

# create empty records to capture CDS length for genes with no GBU output (if available via -G)
if( $haveGeneLen ) {
  for my $gene (sort keys %CDSLen) {
    print "$gene,0,0,0,0,0,0,0,0$CDSLen{$gene},0,0,0\n" unless($outputGBU{$gene});
  }
}
exit(0);

# ---------------------------------------

sub calcGBU {
  # This method uses bbctools view on the merged BED file to reproduce original APD_plugin report
  # IF min/max and WIG base coverage is not required then bbctools report could be used on unmerged BED
  # - Would be quicker but requires more parsing
  # Also for WIG the output could be generated for reads with one command except for normalization of reads
  my ($cov,$ncov,$maxCov,$totCov,$cov20,$gcov20,$cov100,$cov350,$cov500) = (0,0,0,0,0,0,0,0,0);
  my $minCov = 999999999;
  my $subC20 = $mthr * $mbrd;
  my $gsubC20 = $mthr * $gmbrd;
  my $wigNorm = $mbrd > 0 ? 100/$mbrd : 0;
  my $writeWIG = 0;
  if( $wigDir ) {
    my $wigfile = "$wigDir/$lastGene.wig";
    open(WIG,">",$wigfile) || die "Cannot open WIG file for output at '$wigfile'.";
    print WIG "variableStep chrom=$lastChr\n";
    $writeWIG = 1 unless( $wigPanel );
  }
  my $pos;
  open(BBCVIEW,"$bbctools view -nstR '$tmpgbubed' '$bbcfile' |") || die "Could not run BBCtools view command: $!\n";
  while(<BBCVIEW>) {
    ($pos,$cov) = split('\t',$_);
    $cov += 0; # avoid multiple str->num conversions
    ++$ncov;
    $totCov += $cov;
    ++$cov20 if( $cov < $subC20 );
    ++$gcov20 if( $cov < $gsubC20 );
    ++$cov100 if( $cov >=100 );
    ++$cov350 if( $cov >=350 );
    ++$cov500 if( $cov >=500 );
    $maxCov = $cov if( $cov > $maxCov );
    $minCov = $cov if( $cov < $minCov );
    printf WIG "$pos\t%.3f\n",$cov*$wigNorm if( $writeWIG );
  }
  close(BBCVIEW);
  my $cov0 = $reglen - $ncov;
  $minCov = 0 if( $cov0 );  # at least one base had 0 coverage
  $cov20 += $cov0;
  $gcov20 += $cov0;
  printf "$lastGene,$minCov,$maxCov,%.2f,%.4f,$cov20,$ncov,$cov100,$cov350,$cov500,$reglen", $totCov/$reglen, ($reglen-$cov20)/$reglen;
  # extra fields output if full gene lengths provided (vs. Panel-Gene-Intersection)
  if( $haveGeneLen ) {
    my $cdslen = 0+$CDSLen{$lastGene};
    if( $cdslen > 0 ) {
      # $cdslen-$reglen are uncovered bases added to $gcov20: ncov20 = cdslen-gcov20-(cdslen-reglen)
      printf ",$cdslen,%.2f,%.4f,$gcov20", $totCov/$cdslen, ($reglen-$gcov20)/$cdslen;
    } else {
      print ",NA,NA,NA,NA";
    }
  }
  print "\n";
  ++$outputGBU{$lastGene};
  # check if a second pass is required for a more specific gene WIG coverage
  if( $wigDir && $wigPanel ) {
    # create new temporary bed for amplicons with matching GENE_ID
    open(TMPBED,">",$tmpgbubed) || die "Could not write local file '$tmpgbubed'\n";
    for my $line (@{$wigAmplicons{$lastGene}}) {
      print TMPBED "$line\n";
    }
    close(TMPBED);
    # repeat coverage scan over new temporary bed
    open(BBCVIEW,"$bbctools view -nstR '$tmpgbubed' '$bbcfile' |") || die "Could not run BBCtools view command: $!\n";
    while(<BBCVIEW>) {
      ($pos,$cov) = split('\t',$_);
      printf WIG "$pos\t%.3f\n",$cov*$wigNorm;
    }
    close(BBCVIEW);
  }
  close(WIG) if( $wigDir );
}

