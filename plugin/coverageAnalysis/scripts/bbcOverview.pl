#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

# get current running script dir
use FindBin qw($Bin);

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create binned coverage across the whole of the effective reference from a Base Coverage file. (Output to STDOUT.)";
my $USAGE = "Usage:\n\t$CMD [options] <BBC file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -l Print extra log information to STDERR.
  -w Print Warning messages for potential BED file issues to STDOUT.
  -b <int> Number of bins to output. Default: 600.
  -B <file> Optional BED file to define effective reference.";

my $bedfile="";
my $numbins=600;
my $bedwarn=0;
my $logopt=0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-B') {$bedfile = shift;}
    elsif($opt eq '-b') {$numbins = shift;}
    elsif($opt eq '-w') {$bedwarn = 1;}
    elsif($opt eq '-l') {$logopt = 1;}
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
elsif( scalar @ARGV != 1 )
{
    print STDERR "$CMD: Invalid number of arguments.\n";
    print STDERR "$USAGE\n";
    exit 1;
}

my $bbcfile = shift(@ARGV);

$binsize += 0;
$binsize = 600 if( $binsize < 1 );

my $haveBed = ($bedfile ne "");

#--------- End command arg parsing ---------

# Open BBCFILE and read contig header string - this is to get distribution over the genome per contig
open( BBCFILE, "<:raw", $bbcfile ) || die "Failed to open BBC file $bbcfile\n";
chomp( my $contigList = <BBCFILE> );
my @chromName = split('\t',$contigList );
my $numChroms = scalar(@chromName);
my @chromSize;
my $genomeSize = 0;
my (%chromMaps,%chromNum);
for( my $i = 0; $i < $numChroms; ++$i )
{
  my @fields = split('\$',$chromName[$i]);
  $chromNum{$fields[0]} = $i+1;
  $chromName[$i] = $fields[0];
  $chromSize[$i] = int($fields[1]);
  $chromMaps{$fields[0]} = $genomeSize;
  $genomeSize += $chromSize[$i];
}
print STDERR "Read $numChroms contig names and lengths. Total contig size: $genomeSize\n" if( $logopt );

# create hash arrays of target starts and ends and cumulative target position for binning
my (%targSrts,%targEnds,%targMaps);
my $targetSize = 0;
loadBedRegions() if( $haveBed );

# Check/open bamfile and set up bin IDs for whole genome/ targeted modes
my @binid;
if( $haveBed )
{
  $numbins = $targetSize if( $targetSize < $numbins );
  $binsize = $targetSize / $numbins;
  print STDERR "targetSize = $targetSize, numbins = $numbins -> binsize = $binsize\n" if( $logopt );
  my $lastChrom = "";
  my $binnum = 0;
  for( my $cn = 0; $cn < $numChroms; ++$cn )
  {
    my $chrid = $chromName[$cn];
    next if( !defined($targMaps{$chrid}) );
    my @targMap = @{$targMaps{$chrid}};
    my $lbin = $targMap[scalar(@targMap)-1] / $binsize;
    my $ibin = int($lbin);
    if( $lbin != $ibin && $lastChrom ne "" )
    {
      $binid[$binnum++] = $lastChrom . '--' . $chrid;
    }
    while( $binnum < $ibin )
    {
      $binid[$binnum++] = $chrid;
    }
    $lastChrom = $chrid;
  }
  # finish up last few bins (if any)
  while( $binnum < $numbins )
  {
    $binid[$binnum++] = $lastChrom;
  }
  open( BBCVIEW, "$Bin/bbcView.pl -B \"$bedfile\" \"$bbcfile\" |" ) || die "Cannot read base coverage from $bbcfile.\n";
}
else
{
  $numbins = $genomeSize if( $genomeSize < $numbins );
  $binsize = $genomeSize / $numbins;
  my $chrid = $chromName[0];
  my $chrsz = $numChroms > 1 ? $chromMaps{$chromName[1]} : $genomeSize;
  print STDERR "genomeSize = $genomeSize, numbins = $numbins -> binsize = $binsize\n" if( $logopt );
  my $chrn = 1;
  my $pos = 0;
  my $i = 0;
  for( ; $i < $numbins; ++$i )
  {
    $binid[$i] = $chrid;
    $pos = int(($i+1) * $binsize);
    while( $pos > $chrsz )
    {
      $chrid = $chromName[$chrn];
      #$binid[$i] .= "-" . $chrid if( int($pos) > $chrsz );
      ++$chrn;
      $chrsz = ($chrn >= $numChroms) ? $genomeSize : $chromMaps{$chromName[$chrn]};
    }
    $binid[$i] .= '--' . $chrid if( $binid[$i] ne $chrid );
  }
  open( BBCVIEW, "$Bin/bbcView.pl \"$bbcfile\" |" ) || die "Cannot read base coverage from $bbcfile.\n";
}

my @covbin;
my $bin = 0;
my $lastBin = 1;
my $lastChr = "";
my ($targNum,$targSrt,$targEnd,$targAryLen,$chromMap);
my (@targSrtAry,@targEndAry,@targMapAry);

while( <BBCVIEW> )
{
  my ($chrid,$pos,$ontarg,$fcov,$rcov) = split('\t',$_);
  if( $haveBed )
  {
    next if( !defined($targMaps{$chrid}) );
    if( $chrid ne $lastChr )
    {
      $lastChr = $chrid;
      @targSrtAry = @{$targSrts{$chrid}};
      @targEndAry = @{$targEnds{$chrid}};
      @targMapAry = @{$targMaps{$chrid}};
      $targAryLen = scalar(@targSrtAry);
      $targNum = 0;
      $targSrt = $targSrtAry[0];
      $targEnd = $targEndAry[0];
      $targMap = $targMapAry[0];
      print STDERR "Binning reads over $chrid for $targAryLen targets...\n" if( $logopt );
    }
    # find the target this read belongs to
    next if( $targNum >= $targAryLen );
    if( $pos < $targSrt || $pos > $targEnd )
    {
      while( ++$targNum < $targAryLen )
      {
        last if( $pos <= $targEndAry[$targNum] );
      }
      next if( $targNum >= $targAryLen );
      $targSrt = $targSrtAry[$targNum];
      $targEnd = $targEndAry[$targNum];
      $targMap = $targMapAry[$targNum];
      next if( $pos < $targSrt );
    }
    $bin = int( ($targMap + $pos - $targSrt)/ $binsize );
  }
  else
  {
    if( $chrid ne $lastChr )
    {
      $lastChr = $chrid;
      $targMap = $chromMaps{$chrid};
      print STDERR "Binning reads over $chrid...\n" if( $logopt );
    }
    $bin = int(($pos+$targMap-1) / $binsize);
  }
  $covbin[$bin] += $fcov+$rcov;
}
close( BBCVIEW );

# Dump out the bins to STDOUT
print "contigs\treads\n";
for( my $i = 0; $i < $numbins; $i++ )
{
    printf "%s\t%.0f\n", $binid[$i], $covbin[$i];
}

# ----------------END-------------------

# Load all BED file regions in to memory and validate BED file.
# This functions is for code organization only and not intended to be general function.
sub loadBedRegions
{
  my ($lastChr,$lastSrt,$lastEnd,$numTracks,$numTargets,$numTargReads,$numWarns) = (0,0,0,0,0,0,0);
  open( BEDFILE, "$bedfile" ) || die "Cannot open targets file $bedfile.\n";
  while( <BEDFILE> )
  {
    my ($chrid,$srt,$end) = split('\t',$_);
    next if( $chrid !~ /\S/ );
    if( $chrid =~ /^track / )
    {
      ++$numTracks;
      if( $numTracks > 1 )
      {
        print STDERR "\nWARNING: Bed file has multiple tracks. Ignoring tracks after the first.\n";
        ++$numWarns;
        last;
      }
      if( $numTargets > 0 )
      {
        print STDERR "\nERROR: Bed file incorrectly formatted: Contains targets before first track statement.\n";
        exit 1;
      }
      next;
    }
    unless( defined($chromNum{$chrid}) )
    {
      print STDERR "\nERROR: Target region ($chrid:$srt-$end) not present in specified genome.\n";
      exit 1;
    }
    if( $chromNum{$chrid} != $lastChr )
    {
      if( $chromNum{$chrid} < $lastChr )
      {
        print STDERR "\nERROR: BED file is not ordered ($chrid out of order vs. genome file).\n";
        exit 1;
      }
      # add an extra value for the total target size to avoid having to look to next target start
      push( @{$targMaps{$chromName[$lastChr-1]}}, $targetSize ) if( $lastChr );
      $lastChr = $chromNum{$chrid};
      $lastSrt = 0;
      $lastEnd = 0;
    }
    ++$numTargReads;
    ++$srt;
    $end += 0;
    if( $srt < $lastSrt )
    {
      print STDERR "ERROR: Region $chrid:$srt-$end is out-of-order vs. previous region $chrid:$lastSrt-$lastEnd.\n";
      exit 1;
    }
    if( $srt <= $lastEnd )
    {
      ++$numWarn;
      if( $end <= $lastEnd )
      {
        print STDERR "Warning: Region $chrid:$srt-$end is entirely overlapped previous region $chrid:$lastSrt-$lastEnd.\n" if( $bedwarn );
        #print STDERR " - This region will be excluded from the output file.\n" if( $bedwarn );
        next;
      }
      print STDERR "Warning: Region $chrid:$srt-$end overlaps previous region $chrid:$lastSrt-$lastEnd.\n" if( $bedwarn );
      $srt = $lastEnd + 1;
      #print STDERR " - Report will contain partial coverage for the overlap region $chrid:$srt-$end.\n" if( $bedwarn );
    }
    $lastSrt = $srt;
    $lastEnd = $end;
    ++$numTargets;
    push( @{$targSrts{$chrid}}, $srt );
    push( @{$targEnds{$chrid}}, $end );
    push( @{$targMaps{$chrid}}, $targetSize );
    $targetSize += $end - $srt + 1;
  }
  # add final target size
  push( @{$targMaps{$chromName[$lastChr-1]}}, $targetSize ) if( $lastChr );
  close( BEDFILE );
  if( $numWarns )
  {
    print STDERR "$CMD: $numWarns BED file warnings were detected!\n";
    print STDERR " - Re-run with the -w option to see individual warning messages.\n" if( !$bedwarn );
  }
  print STDERR "Read $numTargets of $numTargReads target regions from $bedfile\n" if( $logopt || $numWarns );
  print STDERR "Total non-overlapping target region size: $targetSize bases.\n" if( $logopt );
}

