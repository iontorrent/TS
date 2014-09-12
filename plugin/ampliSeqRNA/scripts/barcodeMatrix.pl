#!/usr/bin/perl
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
# Only partly re-factored for more general usage.

use File::Basename;

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create a barcodes x targets property the same field(s) extracted from a number of (barcode) table files.";
my $USAGE = "Usage:\n\t$CMD [options] <genome file (.fai)> <Property Index> <file1> [<file2] ...]\n";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -A <prefix> Annotation attribute prefix for exracting attributes to add as extra fields to output. Default: '' (do not add any).
     For example, -A 'A_' would extract value 'xyz' from 'A_OMIN=xyz' and add under an extra field column 'OMIN'. 
  -a <integer> Attribute field to extract GENE_ID and other annotation from. Default: 5.";

my $attField = 5;
my $attPrefix = '';
my $attNullVal = '.';

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 ) {
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-a') {$attField = int(shift);}
  elsif($opt eq '-A') {$attPrefix = shift;}
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
} elsif( $nargs < 3 ) {
  print STDERR "$CMD: Invalid number of arguments.\n";
  print STDERR "$USAGE\n";
  exit 1;
}

my $genome = shift(@ARGV);
unless( -e $genome ) {
  print STDERR "Error at $CMD arg#1: Genome file not found: '$genome'.\n";
  exit 0;
}
# allow customization for reading the chromosome coverage summary file: arg2 = "chrom"
my $propType = shift(@ARGV);
my $idField = 3;
my $propNum = int($propType+0);
if( $propType eq "chrom" ) {
  $idField = 0;
  $propNum = 3;
} elsif( $propNum <= 0 ) {
  print STDERR "Error at $CMD arg#2: <Property Index> must be an integer > 0.\n";
  exit 0;
}
my $propNum2 = $propNum+1;

my $attPrefixLen = length($attPrefix);
--$attField;

# ------------------ End of arg parsing -------------------

# flag for extra target location output (useful for debugging sort)
my $addCoords = 0;

# check/read genome file (for chromosome order relative to genome)
my @chromName;
my $numChroms = 0;
open( GENOME, $genome ) || die "Cannot read genome info. from $genome.\n";
while( <GENOME> ) {
  my ($chrid) = split;
  next if( $chrid !~ /\S/ );
  $chromName[$numChroms++] = $chrid;
}
close( GENOME );

my @annoOrder = ();
my %annoFields;
my %annoData;

my %targets;
my %sortlist;
my $barcode;
my $barcode_fields;
my $fnum = 0;
while(<>) {
  my $fn = basename(dirname($ARGV));
  if( $fn ne $barcode ) {
    # add blank columns for existing targets not covered to number of bacodes considered so far
    if( ++$fnum > 1 ) {
      $barcode_fields .= "\t";
      while( my ($id,$str) = each(%targets) ) {
        my $cnt = ($str =~ tr/\t//)+2;
        $targets{$id} .= "\t" if( $cnt < $fnum );
      }
    }
    $barcode = $fn;
    $barcode_fields .= $fn;
    next;  # skip header line
  }
  my @fields = split;
  # assume target ID plus start+end location is unique
  my $trgid = $fields[$idField].':'.$fields[1].'-'.$fields[2];
  if( defined($targets{$trgid}) ) {
    $targets{$trgid} .= "\t";
  } else {
    # extract and track additonal annotations
    # build lists for sorting (by chromsome)
    my $gene = ';'.$fields[$attField].';';
    $gene = ($gene =~ m/;GENE_ID=(.*?);/) ? $1 : (index($gene,"=") < 0 ? $fields[$attField] : '');
    $gene = "N/A" if( $gene eq '.' || $gene !~ '\S' );
    push( @{$sortlist{$fields[0]}}, [$fields[1],$fields[2],$gene,$fields[$idField]] );
    # add empty fields for previously unmatched barcodes
    for( my $tnum = $fnum; $tnum > 1; --$tnum ) {
      $targets{$trgid} .= "\t";
    }
    # add additional attributes field (as extracted kvp)
    if( $attPrefixLen ) {
      my $exAnno = '';
      my @anno = split(';',$fields[$attField]);
      foreach (@anno) {
        next unless( /^$attPrefix/ );
        my $kvp = substr($_,$attPrefixLen);
        if( $kvp =~ /\s*(\S+?)=(\S+)/ ) {
          unless( defined($annoFields{$1}) ) {
            $annoFields{$1} = scalar(@annoOrder);
            push(@annoOrder,$1);
          }
        }
        $exAnno .= $kvp.';';
      }
      $annoData{$trgid} = $exAnno;
    }
  }
  my $prop = $fields[$propNum];
  $prop += $fields[$propNum2] if( $propType eq "chrom" );
  $targets{$trgid} .= $prop;
}
# sort arrays for each chromosome on target start then stop
while( my ($chr,$ary) = each(%sortlist) ) {
  @$ary = sort { $a->[0] <=> $b->[0] || $a->[1] <=> $b->[1] } @$ary;
}
# output matrix using amplicons sorted by chromosome, start, stop (even though these fields are not output)
if( $propType eq "chrom" ) {
  print "Chrom\t";
  print "Start\tEnd\t" if( $addCoords );
  print "$barcode_fields";
} else {
  print "Chrom\tStart\tEnd\t" if( $addCoords );
  print "Gene\tTarget\t$barcode_fields";
}
# add extracted annotation fields
my $numAttr = scalar(@annoOrder);
if( $numAttr ) {
  foreach (@annoOrder) { print "\t$_"; }
}
print "\n";

for( my $chrn = 0; $chrn < $numChroms; ++$chrn ) {
  my $chr = $chromName[$chrn];
  my $ary = $sortlist{$chr};
  for( my $i = 0; $i < scalar(@$ary); ++$i ) {
    my $subary = $ary->[$i];
    my $trgid = $subary->[3].':'.$subary->[0].'-'.$subary->[1];
    if( $propType eq "chrom" ) {
      print "$chr\t";
      print "$subary->[0]\t$subary->[1]\t" if( $addCoords );
      print "$targets{$trgid}";
    } else {
      print "$chr\t$subary->[0]\t$subary->[1]\t" if( $addCoords );
      print "$subary->[2]\t$subary->[3]\t$targets{$trgid}";
    }
    # output selected attribute to fields, regardless of KVP order and with blanks for missing attrbute keys
    if( $numAttr ) {
      my @attOut = (($attNullVal) x $numAttr);
      foreach (split(';',$annoData{$trgid})) {
        if( /\s*(\S+?)=(\S+)/ ) {
          $attOut[$annoFields{$1}] = $2;
        }
      }
      print "\t".join("\t",@attOut);
    }
    print "\n";
  }
}

