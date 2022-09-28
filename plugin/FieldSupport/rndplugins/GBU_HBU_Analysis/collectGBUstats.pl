#!/usr/bin/perl
# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved

use File::Basename;

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Collect individual panel GBU data and produce weighted mean GBU, etc., stats for each gene.
Full racks gene representation is assued to be the panel(s) with the highest coverage. Output TSV to STDOUT.";
my $USAGE = "Usage:\n\t$CMD [options] <gbu1.csv> [<gbu2.csv> ...]";
my $OPTIONS = "Options:
  -d          Delete input WIG file(s). Only effective if -C option is provided.
  -w          Weight GBU values by gene mean base read depth for individual inputs (see -W)
  -C <dir>    Generate combined weighted Coverage WIG files (per gene) to this folder. Default: '' => none output.
  -G <gene>   Track results for individual Gene to STDERR. (Debugging option.)
  -R <file>   Gene to rack TSV file in case full rack panel is not run. Default: '' => estimate from file names.
  -W <method> Replicate inputs weighting method applied. Allowed values are (case-insensitive):
     Mean  => Use mean of relicates, equivalent to standard weighting of results across panels.
     Best  => Take highest mean base read depth for each gene from each replicate.
     Worst => Take lowest mean base read depth for each gene from each replicate.
     MBRD  => Weight replicates by mean base read depth for each gene. (-w option is assumed)
  -h ? --help Display Help information.";

my $GOI = "";
my $repWeight = "MEAN";  # same as treating replicates as just more data points
my $weightMBRD = 0;
my $geneRackFile = "";
my $wigDir = "";
my $deleteInputWig = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-d') {$deleteInputWig = 1}
  elsif($opt eq '-w') {$weightMBRD = 1}
  elsif($opt eq '-C') {$wigDir = shift}
  elsif($opt eq '-G') {$GOI = shift}
  elsif($opt eq '-R') {$geneRackFile = shift}
  elsif($opt eq '-W') {$repWeight = shift}
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
if( $nargs < 1 ) {
  print STDERR "$CMD: Invalid number of arguments.";
  print STDERR "$USAGE\n";
  exit 1;
}

$repWeight = uc($repWeight);
my $validRepWeight = " MEAN, BEST, MBRD, WORST,";
if( index( $validRepWeight, ' '.$repWeight.',' ) < 0 ) {
  print STDERR "ERROR: $CMD -W option must one of $validRepWeight to be valid.\n";
  exit 1;
}
if( $repWeight eq "MBRD" ) {
  $repWeight = "Mean";
  $weightMBRD = 1;
}

$geneRackFile = "" if( $geneRackFile eq "-" );

# ------------ End Cmd Args processing -----------

# load explicit gene to rack maps from if file provided
my $geneRack;
if( $geneRackFile ) {
  open( GENERACK, $geneRackFile ) || die "ERROR: $CMD: Failed to open file '$geneRackFile'";
  while(<GENERACK>) {
    chomp;
    my ($gene,$rack) = split('\t',$_);
    $geneRack{$gene} = $rack;
  }
  close(GENERACK);
}

# load all the data from all files to associative array
my %geneData, %geneCDSLen, %genePGILeni, %wigpath;
my $fname, $panel, $bcrep;
my $nwarn = 0;
while(<>) {
  if( $ARGV ne $fname ) {
    $fname = $ARGV;
    $panel = basename($fname);
    $panel =~ s/\.[^.]*$//;
    $bcrep = "1";
    # specific parsing out of panel ID if present - else no replaicate detections
    if( $panel =~ s/\.GBU\..*// ) {
      if( $panel =~ m/(^.*)_(.+)/ ) {
        $bcrep = $1; # barcode
        $panel = $2; # plateID
        $wigpath{$bcrep} = dirname($fname);
      }
    }
    ++$nwarn if( $bcrep eq "1" );
    next;  # skip header
  }
  chomp;
  my @fields = split(',',$_);
  my $gene = $fields[0];
  my $rlen = 0+$fields[7];
  my $clen = 0+$fields[8];
  # extract just necessary parts: MeanCov (for weighting?), SubGBUBases (for GBU recalc), RegionLen (for weighting)
  $geneData{$gene}{$panel}{$bcrep} = [ 0+$fields[3], 0+$fields[5], $rlen ];
  # track max. coverage vs. CDS and CDS length (constants for gene)
  # - separated in case full rack result has full gene tube dropouts (only CDS length recorded)
  $genePGILen{$gene} = $rlen if( $rlen > $genePGILen{$gene} );
  $geneCDSLen{$gene} = $clen if( $clen );
}
if( $nwarn ) {
  print STDERR "Warning: $nwarn input files could not be parsed to separate replicate/panel IDs.\n";
}

# File header line: CDS here is gene region covered as passed in - typically Gene CDS+5b padding
# Key: w = Weighted, m = Mean, sd = Std Dev, G = Gene (CDS), B = Base, U = Uniformity,
#      P = Panel, I = Intersection, NFC = Number full coverage panels, NPC = Number full coverage panels
print "Gene\tRackID\tAmpCov\twmGBU\twsdGBU\tCDS.Len\tPGI.Len\twmPGIBU\twsdPGIBU\tNFC\tmFPGIBU\tsdFPGIBU\tNPC\twmPPGIBU\twsdPPGIBU\n";

# process data by gene in alphanum order
my $init_nrep = ($repWeight eq "BEST" || $repWeight eq "WORST") ? 1 : 0;
my $init_bcov = ($repWeight eq "WORST") ? 99999999 : 0;
my $wigHeader, $wigFile;
my %wigData;
for my $gene (sort keys %geneData) {
  my $FPRack = $geneRack{$gene} || "NA";
  my $PGILen = $genePGILen{$gene};
  my $CDSLen = $geneCDSLen{$gene};
  unless( $CDSLen ) {
    print STDERR "ERROR: CDS length for gene $gene is unavailable - final GBU value and WIG file omitted from output.\n";
    next;
  }
  my ($NFC,$mFPGIBU,$sdFPGIBU,$mFPGIBU2,$sdFPGIBU2) = (0,0,0,0,0);
  my ($NPC,$wmPPGIBU,$wsdPPGIBU,$wmPPGIBU2,$wsdPPGIBU2) = (0,0,0,0,0);
  if( $wigDir ) {
    my $wigfile = "$wigDir/$gene.wig";
    open(WIG,">",$wigfile) || die "Could not open WIG file for output at '$wigfile'";
    %wigData = ();
  }
  for my $panel (keys %{$geneData{$gene}}) {
    my $isPartial = substr($panel,0,2) eq "PP";
    $FPRack = $panel unless( $isPartial || $geneRackFile );
    # iterate over replicates
    my ($rwGBU,$rwSWt,$rwGBU2,$rwSWt2) = (0,0,0,0);
    my $bcov = $init_bcov;
    my $nrep = $init_nrep;
    for my $bcrep (keys %{$geneData{$gene}{$panel}}) {
      my ($cov,$sub,$len) = @{$geneData{$gene}{$panel}{$bcrep}};
      next unless( $len );
      $isPartial = ($len < $PGILen);   # redefined as full coverage - should be same over all replicates!
      my $gbu = 1.0 - ($sub / $len);   # GBU recalculated for accuracy (to avoid round off)
      my $ppwt = $len / $PGILen;       # PP weighting by panel/CDS base coverage intersection (1 for full rack)
      # barcode/replicate weighting options
      if( $weightMBRD ) {
        $ppwt *= $cov;
      }
      my $wgbu = $gbu * $ppwt;
      if( $repWeight eq "BEST" ) {
        if( $cov > $bcov ) {
          $cov = $bcov;
          $rwGBU = $wgbu;
          $rwSWt = $ppwt;
          $rwGBU2 = $wgbu * $gbu;
          $rwSWt2 = $ppwt * $ppwt;
          $wigFile = "$wigpath{$bcrep}/$gene.wig";
        }
      } elsif( $repWeight eq "WORST" ) {
        if( $cov < $bcov ) {
          $cov = $bcov;
          $rwGBU = $wgbu;
          $rwSWt = $ppwt;
          $rwGBU2 = $wgbu * $gbu;
          $rwSWt2 = $ppwt * $ppwt;
          $wigFile = "$wigpath{$bcrep}/$gene.wig";
        }
      } else {
        ++$nrep;
        $rwGBU += $wgbu;
        $rwSWt += $ppwt;
        $rwGBU2 += $wgbu * $gbu;  # w.x^2
        $rwSWt2 += $ppwt * $ppwt;
        sumWigfile("$wigpath{$bcrep}/$gene.wig",$ppwt) if( $wigDir );
      }
      # DEBUG (for specific gene) option
      if( $gene eq $GOI ) {
        print STDERR "$gene ($panel:$bcrep): mrd=$cov, sub=$sub, pgi=$len/$PGILen/$CDSLen, GBU = $gbu -> $wgbu\n";
      }
    }
    if( $wigDir && $repWeight ne "MEAN" ) {
      sumWigfile($wigFile,$rwSWt);
    }
    # collect stats according to the 'panel' fully or parially covered the gene
    # note that variable names below do not match their final contents but the summed components needed for calculation
    if( $isPartial ) {
      $NPC += $nrep;
      $wmPPGIBU += $rwGBU;
      $wsdPPGIBU += $rwSWt;
      $wmPPGIBU2 += $rwGBU2;
      $wsdPPGIBU2 += $rwSWt2;
    } else {
      $NFC += $nrep;
      $mFPGIBU += $rwGBU;
      $sdFPGIBU += $rwSWt;   # should = $NFC
      $mFPGIBU2 += $rwGBU2;
      $sdFPGIBU2 += $rwSWt2; # should = $NFC
    }
  }
  # DEBUG (for specific gene) option
  if( $gene eq $GOI ) {
    print STDERR "$gene: $NFC full cov: S(PGIBU) = $mFPGIBU, S(w) = $sdFPGIBU2, S(w.PGIBU^2) = $mFPGIBU2, S(w^2) = $sdFPGIBU2\n";
    print STDERR "$gene: $NPC part cov: S(w.PGIBU) = $wmPPGIBU, S(w) = $wsdPPGIBU, S(w.PGIBU^2) = $wmPPGIBU2, S(w^2) = $wsdPPGIBU2\n";
  }
  if( $wigDir ) {
    writeWigfile("$wigDir/$gene.wig",$wsdPPGIBU+$sdFPGIBU);
  }
  my $PanelCov = $PGILen / $CDSLen;
  my ($wmPGIBU,$wsdPGIBU) = weightedMean( $mFPGIBU+$wmPPGIBU, $sdFPGIBU+$wsdPPGIBU, $mFPGIBU2+$wmPPGIBU2, $sdFPGIBU2+$wsdPPGIBU2 );
  my ($wmGBU,$wsdGBU)     = ( $PanelCov*$wmPGIBU, $PanelCov*$wsdPGIBU );
  ($mFPGIBU,$sdFPGIBU)    = weightedMean( $mFPGIBU, $sdFPGIBU, $mFPGIBU2, $sdFPGIBU2 );
  ($wmPPGIBU,$wsdPPGIBU)  = weightedMean( $wmPPGIBU, $wsdPPGIBU, $wmPPGIBU2, $wsdPPGIBU2 );
  printf "$gene\t$FPRack\t%.2f%%\t%.4f\t%.4f\t$CDSLen\t$PGILen\t%.4f\t%.4f\t$NFC\t%.4f\t%.4f\t$NPC\t%.4f\t%.4f\n",
    100*$PanelCov, $wmGBU, $wsdGBU, $wmPGIBU, $wsdPGIBU, $mFPGIBU, $sdFPGIBU, $wmPPGIBU, $wsdPPGIBU;
}

exit 0;

# ---------------------------------------------------

sub weightedMean {
  my ($s1,$w1,$s2,$w2) = @_;
  my $wm = $w1 > 0 ? $s1/$w1 : 0;
  # some potential for roundoff error leading to -ve sums
  # in this case the result should be tiny and taking a value of 0 is appropriate
  my $wf = $w1 > 0 ? $w1-($w2/$w1) : 0;
  my $wv = $wf > 0 ? ($s2-$w1*$wm*$wm)/$wf : 0;
  # with only 1 data point ($wf==0) StdDev is strictly undefined but is here defined as 0
  return ( $wm, ($wv > 0 ? sqrt($wv) : 0) );
}

sub sumWigfile {
  my ($wigfile,$wgt) = @_;
  unless( open(WIG,$wigfile) ) {
    print STDERR "WARNING: $CMD: Could not open WIG file at '$wigfile'.\n";
    return;
  }
  $wigHeader = <WIG>;
  while(<WIG>) {
    my ($pos,$cov) = split('\t',$_);
    $wigData{0+$pos} += $cov * $wgt;
  }
  close(WIG);
  unlink($wigfile) if( $deleteInputWig );
}

sub writeWigfile {
  my ($wigfile,$sumwgt) = @_;
  unless( $wigHeader ) {
    print STDERR "WARNING: $CMD: No WIG files read for output to '$wigfile'.\n";
    return;
  }
  unless( open(WIG,">",$wigfile) ) {
    print STDERR "WARNING: $CMD: Could not open WIG file for output at '$wigfile'.\n";
    return;
  }
  print WIG $wigHeader;
  # no warning for 0 lines output because techically that is possibly correct
  my $wgt = $sumwgt > 0 ? 1/$sumwgt : 0;
  print STDERR "WARNING: wgt = 0 for $wigfile\n" unless($wgt);
  for my $pos (sort {$a <=> $b} keys %wigData) {
    printf WIG "$pos\t%.2f\n", $wigData{$pos}*$wgt;
  }
  close(WIG);
}

