#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

use File::Basename;
use FindBin qw($Bin);

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create GC augmented BED file (write to STDOUT).";
my $USAGE = "Usage:\n\t$CMD <BED file> <FASTA file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -s Strict validation. Error out for potential issues, e.g. records out-of-order.
  -w Print Warning messages for potential BED file issues to STDOUT (-l also has to be specified).
  -f <list> Add extra Fields defined by comma-separated list of fields positions (1 based).
  -t <dirpath> Path to use for temporary file output. Default: '.'";

my $extraFields = "";
my $strictbed = 0;
my $bedwarn = 0;
my $tmpdir = '.';

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-f') {$extraFields = shift;}
    elsif($opt eq '-s') {$strictbed = 1;}
    elsif($opt eq '-w') {$bedwarn = 1;}
    elsif($opt eq '-t') {$tmpdir = shift;}
    elsif($opt eq '-h' || $opt eq "?" || $opt eq '--help') {$help = 1;}
    else
    {
        print STDERR "$CMD: Invalid option argument: $opt\n";
        print STDERR "$OPTIONS\n";
        exit 1;
    }
}
if( $help )
{
    print STDERR "$DESCR\n";
    print STDERR "$USAGE\n";
    print STDERR "$OPTIONS\n";
    exit 1;
}
elsif( scalar @ARGV != 2 )
{
    print STDERR "$CMD: Invalid number of arguments.";
    print STDERR "$USAGE\n";
    exit 1;
}

my $bedfile = shift(@ARGV);
my $fastafile = shift(@ARGV);

#--------- End command arg parsing ---------

my @exFields = split(',',$extraFields);

# Create temporary fasta-tab file
my($fastatmp,$folder,$ext) = fileparse($bedfile, qr/\.[^.]*$/);
my $fastatmp = "$tmpdir/$fastatmp.tmp";
system "$Bin/fastaFromBed -fi \"$fastafile\" -bed \"$bedfile\" -tab -fo \"$fastatmp\"";

$" = "\t";

my @fields;
my %chromNum;
my ($line,$gc,$lastChr,$lastSrt,$lastEnd);
my ($numTargets,$numTargReads,$numTracks,$chromCnt) = (0,0,0,0);
my ($numWarns,$numErrors,$trackWarns) = (0,0,0);

# Assume equivalant ordering and re-write input BED to STDOUT
open( FASTABED, "$fastatmp" ) || die "Cannot open fastaFromBed result file $fastatmp.\n";
open( BEDFILE, "$bedfile" ) || die "Cannot open targets file $bedfile.\n"; 
while( <BEDFILE> )
{
  chomp;
  @fields = split('\t',$_);
  next if( $fields[0] !~ /\S/ );
  if( $fields[0] =~ /^track / )
  {
    ++$numTracks;
    if( $numTracks > 1 )
    {
      print STDERR "WARNING: Bed file has multiple tracks.\n" if( !$trackWarns && $bedwarn );
      print STDERR " - Ignoring tracks after the first.\n" if( $strictbed );
      ++$numWarns;
      ++$trackWarns;
      last if( $strictbed );
    }
    if( $numTargets > 0 )
    {
      ++$numErrors;
      print STDERR "ERROR: Bed file incorrectly formatted: Contains targets before first track statement.\n";
      exit 1 if( $strictbed );
    }
    print "$_\n";
    next;
  }
  # read corresponding GC line
  chomp($line = <FASTABED>);
  # validate last region - may wish to discard
  my $chrid = $fields[0];
  unless( defined($chromNum{$chrid}) )
  {
    $chromNum{$chrid} = ++$chromCnt;
    $lastChr = $chromNum{$chrid};
    $lastSrt = 0;
    $lastEnd = 0;
  }
  if( $chromNum{$chrid} < $lastChr )
  {
    ++$numErrors;
    print STDERR "ERROR: BED file is not ordered: $chrid does not appear in a single section.\n";
    exit 1 if( $strictbed );
  }
  ++$numTargReads;
  my $srt = $fields[1]+1;
  my $end = $fields[2]+0;
  if( $srt < $lastSrt )
  {
    ++$numErrors;
    print STDERR "ERROR: Region $chrid:$srt-$end is out-of-order vs. previous region $chrid:$lastSrt-$lastEnd.\n";
    exit 1 if( $strictbed );
  }
  if( $srt <= $lastEnd )
  {
    if( $end <= $lastEnd )
    {
      ++$numWarn;
      print STDERR "Warning: Region $chrid:$srt-$end is entirely overlapped previous region $chrid:$lastSrt-$lastEnd.\n" if( $bedwarn );
      #next;  # if want to discard this region
    }
    #print STDERR "Warning: Region $chrid:$srt-$end overlaps previous region $chrid:$lastSrt-$lastEnd.\n" if( $bedwarn );
  }
  $lastSrt = $srt;
  $lastEnd = $end;
  my @seq = split('\t',$line);
  my $gc = ($seq[1] =~ tr/cgGC/*/);
  print "$fields[0]\t$fields[1]\t$fields[2]";
  for( my $i = 0; $i < scalar(@exFields); ++$i )
  {
    my $f = $exFields[$i]-1;
    next if( $f < 0 );
    print "\t$fields[$f]";
  }
  print "\t$gc\n";
}
close( BEDFILE );
close( FASTABED );
unlink($fastatmp);

print STDERR "$CMD: Completed with $numWarn warnings.\n" if( $numWarn );
print STDERR "$CMD: Completed by ignorring $numErrors potential BED format errors.\n" if( $numErrors );

