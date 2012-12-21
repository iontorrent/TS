#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create table file (tsv) of bed target regions read coverage for BAM file. (Output to STDOUT.)
The coverage information is added to fields in the BED but assumes certain fields are present (e.g. from gcAnnoBed.pl).
Individual reads will first be assigned to individual targets, using limits specified by optional arguments.";
my $USAGE = "Usage:\n\t$CMD [options] <BAM file> <Annotated BED file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -l Print extra Log information to STDERR.
  -a Assign reads assuming Amplicon/AmpliSeq targets. (See -D/-U options below.)
     Default: Assign reads by maximum overlap with target. (Max. overlap also used to assign -a ties.)
  -b Output file is extended BED format (with BED coordinates and bedDetail track line).
     Default: Text table file (tab-separated text fields) with 1-based coordinates assumed field names.
  -d Ignore (PCR) Duplicate reads.
  -u Include only Uniquely mapped reads (MAPQ > 1).
  -n Normalize read counts by the size of the targets. Affects what is output to the 'scoring' field (#9),
     which will be given the field ID 'norm_reads' or 'total_reads'.
  -E <int> End-to-end read proximity limit. If a read covers up to this disance from both ends of a region
     it will be counted as an end-to-end read. (Set to 0 if given as negative.) Default: 2.
  -D <int> Downstream limit for matching read start to target end (appropriate for +/- strand mapping).
     This assignment parameter is only employed if the -a option is provided. Default: 5.
  -U <int> Upstream limit for matching read start to target end (appropriate for +/- strand mapping).
     This assignment parameter is only employed if the -a option is provided. Default: 30.";

my $logopt = 0;
my $bedout = 0;
my $ampreads = 0;
my $dsLimit = 5;
my $usLimit = 30;
my $e2eLimit = 2;
my $nondupreads = 0;
my $uniquereads = 0;
my $normreads = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-l') {$logopt = 1;}
  elsif($opt eq '-a') {$ampreads = 1;}
  elsif($opt eq '-b') {$bedout = 1;}
  elsif($opt eq '-d') {$nondupreads = 1;}
  elsif($opt eq '-u') {$uniquereads = 1;}
  elsif($opt eq '-n') {$normreads = 1;}
  elsif($opt eq '-D') {$dsLimit = int(shift);}
  elsif($opt eq '-E') {$e2eLimit = int(shift);}
  elsif($opt eq '-U') {$usLimit = int(shift);}
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

my $bamfile = shift(@ARGV);
my $bedfile = shift(@ARGV);

$e2eLimit = 0 if( $e2eLimit < 0 );
$usLimit *= -1;  # more convenient for testing

#--------- End command arg parsing ---------

my $samopt= ($nondupreads ? "-F 0x404" : "-F 0x704").($uniquereads ? " -q 1" : "");

# create hash arrays of target starts and ends from reading the BED file
print STDERR "Reading and validating BED file...\n" if( $logopt );
my (%targSrts,%targEnds,%chromNum);
my (%targFwdReads,%targRevReads,%targFwdE2E,%targRevE2E,%targOvpReads);
loadBedRegions();

my $lastChrid = "";
my $numTracks = 0;
my ($nTrgs,$tSrts,$tEnds);
my ($tFwdRds,$tRevRds,$tFwdE2E,$tRevE2E,$tOvpRds);

print STDERR "Reading BAM file using BED filter...\n" if( $logopt );
$| = 1;  # autoflush

open( MAPPINGS, "samtools view $samopt -L \"$bedfile\" \"$bamfile\" |" )
  || die "Failed to pipe reads from $bamfile for regions in $bedfile\n";
while( <MAPPINGS> )
{
  next if(/^@/);
  my ($rid,$flag,$chrid,$srt,$scr,$cig) = split;
  my $end = $srt-1;
  my $rev = $flag & 16;
  # set up arrays for new contig (to avoid repeated hash-lookups)
  if( $chrid ne $lastChrid )
  {
    $lastChrid = $chrid;
    $tSrts = \@{$targSrts{$chrid}};
    $tEnds = \@{$targEnds{$chrid}};
    $tFwdRds = \@{$targFwdReads{$chrid}};
    $tRevRds = \@{$targRevReads{$chrid}};
    $tFwdE2E = \@{$targFwdE2E{$chrid}};
    $tRevE2E = \@{$targRevE2E{$chrid}};
    $tOvpRds = \@{$targOvpReads{$chrid}};
    $nTrgs = scalar(@{$tSrts});
  }
  while( $cig =~ s/^(\d+)(.)// )
  {
    $end += $1 if( $2 eq "M" || $2 eq "D" || $2 eq "X" || $2 eq "=" );
  }
  # Here reads are expected to be bigger than targets so cannot rely on start position
  # being >= a particular region start (hence no binary search on ordered start locations).
  # But can find the last region start that is <= than read end and scan backwards.
  my $tn = floor_bsearch($end,$tSrts);
  if( $tn < 0 )
  {
    # filter reads should overlap at least one region !
    print STDERR "$CMD: Warning: Filtered read $chrid:$srt-$end did not overlap any target region!\n";
    next;
  }
  my $maxOvlp = -1;
  my $bestTn = $tn;
  my $maxEndDist;
  for( ; $tn >= 0; --$tn )
  {
    # test for no overlap with this region (but possibly previous regions since ends not ordered)
    # Note: Unfortunately no early ending if any large, overlapping amplicons allowed (e.g. if region[0] is whole chromosome!)
    next if( $tEnds->[$tn] < $srt );
    my $rSrt = $tSrts->[$tn];
    my $rEnd = $tEnds->[$tn];
    # record a hit for any read overlap
    ++$tOvpRds->[$tn];
    $dSrt = $srt - $rSrt;
    $dEnd = $rEnd - $end;
    # test if this can be assigned to an amplicon
    if( $ampreads )
    {
      my $aSrt = $rev ? $dEnd : $dSrt;
      next if( $aSrt < $usLimit || $aSrt > $dsLimit );
    }
    # save region number for max overlap
    $dSrt = 0 if( $dSrt < 0 );
    $dEnd = 0 if( $dEnd < 0 );
    $rSrt = $rEnd - $rSrt - $dSrt - $dEnd; # actually 1 less than overlap
    if( $rSrt > $maxOvlp )
    {
      $maxOvlp = $rSrt;
      $bestTn = $tn;
      $maxEndDist = $dSrt > $dEnd ? $dSrt : $dEnd;
    }
  }
  if( $maxOvlp >= 0 )
  {
    if( $rev )
    {
      ++$tRevRds->[$bestTn];
      ++$tRevE2E->[$bestTn] if( $maxEndDist <= $e2eLimit );
    }
    else
    {
      ++$tFwdRds->[$bestTn];
      ++$tFwdE2E->[$bestTn] if( $maxEndDist <= $e2eLimit );
    }
  }
}
close( MAPPINGS );

print STDERR "Creating output...\n" if( $logopt );
my $headerLine = "track type=bedDetail";
if( !$bedout )
{
  # replace default bed track line assumed field titles (by position)
  $headerLine = "";
  my @titles = ("contig_id","contig_srt","contig_end","region_id","gene_id","gc");
  for( my $i = 0; $i < scalar(@titles); ++$i )
  {
    if( defined($titles[$i]) ) { $headerLine .= "$titles[$i]\t"; }
    else { $headerLine .= sprintf("field%d\t", $i+1); }
  }
  # these are the added 6 base coverage fields
  my $nrdsid = $normreads ? 'norm_reads' : 'total_reads';
  $headerLine .= "overlaps\tfwd_e2e\trev_e2e\t$nrdsid\tfwd_reads\trev_reads";
  print "$headerLine\n";
}

$lastChrid = "";
$numTracks = 0;
my $targNum;
open( BEDFILE, "$bedfile" ) || die "Cannot open targets file $bedfile.\n";
while( <BEDFILE> )
{
  chomp;
  my @fields = split("\t",$_);
  my $chrid = $fields[0];
  next if( $chrid !~ /\S/ );
  # silently ignore extra tracks and duplicate amplicons this time
  if( $chrid =~ /^track/ )
  {
    last if( ++$numTracks > 1 );
    print "$_\n" if( $bedout );
    next;
  }
  if( $chrid ne $lastChrid )
  {
    $lastChrid = $chrid;
    $lastSrt = 0;
    $lastEnd = 0;
    $targNum = 0;
    $tSrts = \@{$targSrts{$chrid}};
    $tEnds = \@{$targEnds{$chrid}};
    $tFwdRds = \@{$targFwdReads{$chrid}};
    $tRevRds = \@{$targRevReads{$chrid}};
    $tFwdE2E = \@{$targFwdE2E{$chrid}};
    $tRevE2E = \@{$targRevE2E{$chrid}};
    $tOvpRds = \@{$targOvpReads{$chrid}};
  }
  my $srt = $fields[1]+1;
  my $end = $fields[2]+0;
  next if( $srt == $lastSrt && $end == $lastEnd );
  $lastSrt = $srt;
  $lastEnd = $end;
  if( $tSrts->[$targNum] != $srt || $tEnds->[$targNum] != $end )
  {
     print STDERR "$CMD: ERROR: BED file region $chrid:$srt-$end does not match expected region $chrid:$tSrts->[$targNum]-$tEnds->[$targNum] at index $targNum\n";
     exit 1;
  }
  --$srt if( $bedout );
  print "$chrid\t$srt\t";
  for( my $i = 2; $i < scalar(@fields); ++$i ) { print "$fields[$i]\t"; }
  print "$tOvpRds->[$targNum]\t$tFwdE2E->[$targNum]\t$tRevE2E->[$targNum]\t";
  my $tLen = $fields[2] - $fields[1];
  my $nreads = $tFwdRds->[$targNum] + $tRevRds->[$targNum];
  if( $normreads )
  {
    $nreads = $tLen > 0 ? $nreads / $tLen : 0;
    printf "%.3f\t", $nreads;
  }
  else
  {
    printf "%.0f\t", $nreads;
  }
  print "$tFwdRds->[$targNum]\t$tRevRds->[$targNum]\n";
  ++$targNum;
}
close( BEDFILE );

# ----------------END-------------------

# Load all BED file regions in to memory and validate BED file.
# This functions is for code organization only and not intended to be general function.
# This method does not expect a load of the genome so cannot tell if chromsomes are out of order>
sub loadBedRegions
{
  my ($lastChr,$lastSrt,$lastEnd,$numTracks,$numTargets,$numTargReads,$numWarns) = (0,0,0,0,0,0,0);
  open( BEDFILE, "$bedfile" ) || die "Cannot open targets file $bedfile.\n";
  while( <BEDFILE> )
  {
    my ($chrid,$srt,$end) = split('\t',$_);
    next if( $chrid !~ /\S/ );
    if( $chrid =~ /^track/ )
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
      $chromNum{$chrid} = ++$lastChr;
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
    # Anticipate overlapping regions here and only warn for obscured regions
    if( $srt <= $lastEnd )
    {
      if( $srt == $lastSrt && $end == $lastEnd )
      {
        ++$numWarn;
        print STDERR "Warning: Region $chrid:$srt-$end is a repeat - skipping.\n" if( $bedwarn );
        next;
      }
      if( $end <= $lastEnd )
      {
        ++$numWarn;
        print STDERR "Warning: Region $chrid:$srt-$end is entirely overlapped previous region $chrid:$lastSrt-$lastEnd.\n" if( $bedwarn );
        #next;
      }
    }
    $lastSrt = $srt;
    $lastEnd = $end;
    ++$numTargets;
    push( @{$targSrts{$chrid}}, $srt );
    push( @{$targEnds{$chrid}}, $end );
    push( @{$targFwdReads{$chrid}}, 0 );
    push( @{$targRevReads{$chrid}}, 0 );
    push( @{$targFwdE2E{$chrid}}, 0 );
    push( @{$targRevE2E{$chrid}}, 0 );
    push( @{$targOvpReads{$chrid}}, 0 );
  }
  close( BEDFILE );
  if( $numWarns )
  {
    print STDERR "$CMD: $numWarns BED file warnings were detected!\n";
    print STDERR " - Re-run with the -w option to see individual warning messages.\n" if( !$bedwarn );
  }
  print STDERR "Read $numTargets of $numTargReads target regions from $bedfile\n" if( $logopt || $numWarns );
}

sub floor_bsearch
{
  # Return highest index for which lims[index] <= val (< lims[index+1])
  # given value $_[0] (val) and assending-order values array $_[1] (lims).
  # If val >= all values in lims[] then the last index (size-1) is returned.
  # If val  < all values in lims[] then -1 is returned.
  my ($val,$lims) = @_;
  # NOTE: return -1 if value is less than the first value in the array
  if( $lims->[0] > $val ) { return -1; }
  my ($l,$u) = (0, scalar(@{$lims})-1);
  # return last index if value is >= the last value in the array
  if( $lims->[$u] <= $val ) { return $u; }
  # value must be within ranges
  while(1)
  {
    my $m = int( ($l + $u)/2 );
    if( $val < $lims->[$m] ) { $u = $m; }
    elsif( $val < $lims->[$m+1] ) { return $m; }
    else { $l = $m+1; }
  }
}

