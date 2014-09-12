#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

# Note: script is overly complicated due to supporting older versions.

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create a table of Differential Expression analysis. Output to STDOUT.
If 3 or fewer barcodes are avaliable/reported then effective (RPM) values for these barcodes are output.
Otherwise default output fields are the target ID and gene, min/max/ave reads (RPM) and largest expression ratio, to minimum threshold.
Options exist so this can be run on raw data, normaizied data or using thresholds on raw but reporting on normalized data.";
my $USAGE = "Usage:\n\t$CMD [options] <target x BC coverage file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -a Retain Annotation fields, being all fields after barcode fields. (Requires -M specified.)
  -r Output up/down Regulation instead of RPM ratio, assuming first barcode is control.
  -t Output Threshold modified values for min/max or barcode RPMs. Default: output direct RPM values.
  -B <list> Comma-separated subset of Barcode IDs to use to produce DE table. Thresholded barcode RPM's output if <= 3 specified barcodes.
  -L <int> Number of leading (row ID) fields before barcodes, which are copied to output file. Default: 2.
  -M <int> Maximum number of barcode fields to read. Default: 0 (assume all fields are barcode data).
  -N <value> Normalize read counts by sum of reads per barcode (column) to this value (e.g. 1000000 for RPM). Default: 0 (do not normalize).
  -S <label> Suffix label for fields. Default: 'Reads' (e.g minReads, maxReads, aveReads)
     Ignored if less than 4 barcodes used - field titles will be barcode names instead.
  -T <int> Threshold for minimum number of reads for ratio calculation. Default: 10.";

my $regulation = 0;
my $leadFields = 2;
my $keepAnno = 0;
my $rd_thres = 10;
my $fsuf = "Reads";
my $normNom = 0;
my $barcodeList = "";
my $maxBarcodes = 0;
my $thresOut = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-a') {$keepAnno = 1;}
  elsif($opt eq '-r') {$regulation = 1;}
  elsif($opt eq '-t') {$thresOut = 1;}
  elsif($opt eq '-B') {$barcodeList = shift;}
  elsif($opt eq '-L') {$leadFields = int(shift);}
  elsif($opt eq '-M') {$maxBarcodes = int(shift);}
  elsif($opt eq '-N') {$normNom = int(shift);}
  elsif($opt eq '-S') {$fsuf = shift;}
  elsif($opt eq '-T') {$rd_thres = int(shift);}
  elsif($opt eq '-h' || $opt eq "?" || $opt eq '--help') {$help = 1;}
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
} elsif( $nargs != 1 ) {
  print STDERR "$CMD: Invalid number of arguments.\n";
  print STDERR "$USAGE\n";
  exit 1;
}

my $matfile = shift(@ARGV);

if( $normNom < 0 ) {
  print STDERR "Warning: Negative normalization numerator (-N) value ($normNom) ignored.";
  $normNom = 0;
}
if( $rd_thres < 1 ) {
  print STDERR "Warning: Reads threshold (-T) value ($rd_thres) reset to minimum value of 1.";
  $rd_thres = 1;
}

# for RPM normalization factors are calcualted even if not directly applied (to avoid extra logic in code)
my $normalize = ($normNom > 0) ? 1 : 0;
$normNom = 1000000 if( $normNom <= 0 );

$maxBarcodes = 0 if( $maxBarcodes < 0 );
$leadFields = 0 if( $leadFields < 0 );

my $rpmTitle = $regulation ? "log2(DER)" : "DERatio";

#--------- End command arg parsing ---------

die "Cannot find depth file $matfile" unless( -e $matfile );

# open MATFILE and read header string to detect BED format (ionVersion)
open( MATFILE, $matfile ) || die "Failed to open target reads file $matfile\n";
chomp( my $titleLine = <MATFILE> );
my @fieldIDs = split('\t',$titleLine);
my $nfields = scalar(@fieldIDs);
my $nbc = $nfields-$leadFields;
if( $nbc < 1 ) {
  print STDERR "Input matrix $matfile does not appear to have the correct format\n";
  exit 1;
}
$nbc = $maxBarcodes if( $maxBarcodes > 0 && $nbc > $maxBarcodes );
my $annoSrt = $nbc+$leadFields;

# set up for only using a subset of all barcodes (e.g. just two)
my $controlbc = -1;
my $nubc = $nbc;
if( $barcodeList ne "" ) {
  my @bcarray = split(',',$barcodeList);
  my %barcodes = map { $_ => 1 } @bcarray;
  for( my $i = 0; $i < $nbc; ++$i ) {
    $controlbc = $i if( $fieldIDs[$i+$leadFields] eq $bcarray[0] );
    unless( defined($barcodes{$fieldIDs[$i+$leadFields]}) ) {
      --$nubc;
      $fieldIDs[$i+$leadFields] = "";
    }
  }
  if( $nubc < 1 ) {
    print STDERR "Input matrix $matfile does have any barcodes specified by -B '$barcodeList'\n";
    exit 1;
  }
}
# change titles to reflect barcode names if not enough to average
my @genTitles = ( "\tmin".$fsuf, "\tmax".$fsuf, "\tave".$fsuf );
if( $nubc <= 3 ) {
  my $j = 0;
  for( my $i = 0; $i < $nbc; ++$i ) {
    $genTitles[$j++] = $fieldIDs[$i+$leadFields].".$fsuf\t" if( $fieldIDs[$i+$leadFields] );
  }
  while( $j < 3 ) {
    $genTitles[$j++] = "";
  }
}

# create normalization factors - used tracking total read counts even if normalization not appplied
my @norms = ((0) x $nbc);
while(<MATFILE>) {
  chomp;
  my @fields = split('\t',$_);
  next if( $fields[0] !~ /\S/ );
  for( my $i = 0; $i < $nbc; ++$i ) {
    $norms[$i] += $fields[$i+$leadFields];
  }
}
for( my $i = 0; $i < $nbc; ++$i ) {
  $norms[$i] = ($normalize * $norms[$i] > 0) ? $normNom/$norms[$i] : 1;
}
seek(MATFILE,0,0);
<MATFILE>;

my $clog2 = log(2);

# print output field titles
for( my $i = 0; $i < $leadFields; ++$i ) {
  printf "%s\t", $fieldIDs[$i];
}
print "$genTitles[0]$genTitles[1]$genTitles[2]$rpmTitle";
if( $keepAnno ) {
  for( my $i = $annoSrt; $i < $nfields; ++$i ) {
    printf "\t%s", $fieldIDs[$i];
  }
}
print "\n";

# analyze the representation of taget reads per bacode
while(<MATFILE>) {
  chomp;
  my @fields = split('\t',$_);
  next if( $fields[0] !~ /\S/ );
  my ($min,$max,$ave,$minCor,$maxCor) = (-1,0,0,0,0);
  my ($maxRdBelowT,$norFcBelowT) = (-1,0);
  my ($controlNrds,$controlTrds) = (0,0);
  my @origRds = ();
  for( my $i = 0; $i < $nbc; ++$i ) {
    next unless( $fieldIDs[$i+$leadFields] );
    my $nrds = $fields[$i+$leadFields];
    my $rds = $nrds * $norms[$i];
    my $trds = thresReads($nrds) * $norms[$i];
    # track barcode (thresholded) reads for -B output option
    push( @origRds, $thresOut ? $trds : $rds );
    # if regulation mode the control overrides minRPM and is not included in maxRPM (unless all are below threshold)
    if( $i == $controlbc ) {
      $controlNrds = $nrds;
      $controlTrds = $trds;
      next;
    };
    # min/max/ave are raw RPM values over all barcodes
    $ave += $rds;
    $min = $rds if( $rds < $min || $min < 0 );
    $max = $rds if( $rds > $max );
    # collect min/max RPM for reads >= threshold
    if( $nrds >= $rd_thres ) {
      $minCor = $rds if( $rds < $minCor || $minCor == 0 );
      $maxCor = $rds if( $rds > $maxCor );
      next;
    }
    # select representative barcode (values) of those with reads < threshold
    # - use highest number of raw reads and split ties by most total barcode reads (=> lower normalization factor)
    if( $nrds > $maxRdBelowT || ($nrds == $maxRdBelowT && $norms[$i] < $norFcBelowT) ) {
      $maxRdBelowT = $nrds;
      $norFcBelowT = $norms[$i];
    }
  }
  # set min/max for DE based of values for selected barcodes above and below threshold
  if( $maxRdBelowT >= 0 ) {
    # reads for best barcode below threshold set to threshold
    my $rds = $rd_thres * $norFcBelowT;
    # ...but if control reads were less than threshold override with this for set, so all low => DER = 1.0
    $rds = $controlTrds if( $regulation && $controlNrds < $rd_thres );
    $minCor = $rds if( $rds < $minCor || $minCor == 0 );
    $maxCor = $rds if( $rds > $maxCor );
  }
  $ave /= ($regulation && $nbc > 1) ? $nbc-1 : $nbc;
  # print out row definition fields
  for( my $i = 0; $i < $leadFields; ++$i ) {
    printf "%s\t", $fields[$i];
  }
  if( $nubc <= 3 ) {
    for( my $i = 0; $i < $nubc; ++$i ) {
      if( $normalize ) {
        printf "%.3f\t",$origRds[$i];
      } else {
        print "$origRds[$i]\t";
      }
    }
  } else {
    unless( $thresOut ) {
      $min = $minCor;
      $max = $maxCor;
    }
    if( $normalize ) {
      printf "%.3f\t%.3f\t%.3f\t",$min,$max,$ave;
    } else {
      printf "$min\t$max\t$ave\t";
    }
  }
  # override minimum with control in regulation mode (-r)
  my $der = $regulation ? log($maxCor/$controlTrds)/$clog2 : $maxCor/$minCor;
  printf "%.3f", $der;
  # append assumed annotation fields
  if( $keepAnno ) {
    for( my $i = $annoSrt; $i < $nfields; ++$i ) {
      printf "\t%s", $fields[$i];
    }
  }
  print "\n";
}
close(MATFILE);

sub thresReads {
  return $_[0] < $rd_thres ? $rd_thres : $_[0];
}

