#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Append target base coverage depth values to a given table file of target regions at specified depths.
The first 3 fields in this file must be the chromosome/contig name, start and end coordinates of the targets,
and rows ordered by these fields as in the corresponding BBC file, e.g. as produced by bbcTargetAnno.pl. Output to STDOUT.";
my $USAGE = "Usage:\n\t$CMD [options] <BBC file> <target table file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -l Print extra Log information to STDERR.
  -w Print Warning messages for potential file issues to STDOUT.";

my $logopt = 0;
my $bedwarn = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-l') {$logopt = 1;}
  elsif($opt eq '-w') {$bedwarn = 1;}
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

my $bbcfile = shift(@ARGV);
my $trgfile = shift(@ARGV);

#--------- End command arg parsing ---------

use constant BIGWORD => 2**32;

# Open BBCFILE and read contig header string
open( BBCFILE, "<:raw", $bbcfile ) || die "Failed to open BBC file $bbcfile\n";
chomp( my $contigList = <BBCFILE> );
my @chromName = split('\t',$contigList );
my $numChroms = scalar(@chromName);
my @chromSize;
my $genomeSize = 0;
my %chromIndex;
for( my $i = 0; $i < $numChroms; ++$i )
{
  my @fields = split('\$',$chromName[$i]);
  $chromName[$i] = $fields[0];
  $chromSize[$i] = int($fields[1]);
  $chromIndex{$fields[0]} = $i;
  $genomeSize += $chromSize[$i];
}
print STDERR "Read $numChroms contig names and lengths. Total contig size: $genomeSize\n" if( $logopt );

my $bciLastHead = tell(BBCFILE); # start of chrom#0

my $lastChrIdx = -1;
my ($numTargets,$numTracks,$targetsOut) = (0,0,0);
my ($targSrt,$targEnd);
my ($cd,$rdlen,$wdsiz,$fcov,$rcov);

my $intsize = 4;
my $headbytes = 2 * $intsize;

my $detailLog = 0; # set for debugging
my $numBedWarnings = 0;

# load BCI file indexes
my ($bciNumChroms,$bciBlockSize,$bciLastChr);
my (@bciIndex,@bciBlockOffsets);
my $bciIndexSize = 0;  # indicates no BCI load
exit 0 unless( loadBCI($bbcfile) );

my $linenum = 0;
open( TRGFILE, "$trgfile" ) || die "Cannot open targets file $trgfile.\n"; 
while( <TRGFILE> )
{
  chomp;
  @fields = split('\t',$_);
  my $chrid = $fields[0];
  next if( $chrid !~ /\S/ );
  if( ++$linenum == 1 ) {
    print "$_\tcov20x\tcov100x\tcov500x\n";
    next;
  }
  if( !defined($chromIndex{$chrid}) )
  {
    print STDERR "ERROR: Target contig ($chrid) not present in genome as defined in the BBC file.\n";
    exit 1;
  }
  ++$numTargets;
  my $chrIdx = $chromIndex{$chrid};
  my $srt = $fields[1]+0;
  my $end = $fields[2]+0;
  print STDERR "Read target region for chrom#$chrIdx $chrid:$srt:$end\n" if( $detailLog );
  # check how new target overlaps the old (for correct annoatation of things like proper GC counts this better no happen!)
  if( $chrIdx != $lastChrIdx )
  {
    if( $chrIdx < $lastChrIdx )
    {
      # only a performance issue, and then only if contig order is swapping every few targets
      print STDERR "Warning: Target contig ($chrid) is out-of-order vs. that defined in the BBC file.\n";
    }
    $lastChrIdx = $chrIdx;
    $targSrt = 0;
    $targEnd = 0;
  }
  if( $srt < $targSrt )
  {
    print STDERR "ERROR: Region $chrid:$srt-$end is out-of-order vs. previous region $chrid:$targSrt-$targEnd.\n";
    exit 1;
  }
  # optional warnings for overlaping regions
  my $sameSrt = ($srt == $targSrt);
  $targSrt = $srt;
  if( $srt <= $targEnd )
  {
    ++$numBedWarnings;
    if( $bedwarn )
    {
      if( $end <= $targEnd ) {
        print STDERR "Warning: Region $chrid:$srt-$end is entirely overlapped by previous region $chrid:$targSrt-$targEnd.\n";
      } elsif( $sameSrt ) {
        print STDERR "Warning: Region $chrid:$srt-$end entirely overlaps previous region $chrid:$targSrt-$targEnd.\n";
      } else {
        print STDERR "Warning: Region $chrid:$srt-$end overlaps previous region $chrid:$targSrt-$targEnd.\n";
      }
      print STDERR " - Base coverage over this region will be counted in multiple targets.\n";
    }
  }
  $targEnd = $end;
  # reset cursor to start of last section looked at (for adjacent/overlapping targets)
  seek(BBCFILE,$bciLastHead,0);
  bciSeekForward($chrIdx,$targSrt);
  # read from current BBC cursor to end of current target region
  my ($pos,$cov20x,$cov100x,$cov500x) = (0,0,0,0);
  while( $pos <= $targEnd )
  {
    # record the start of the next BBC region in case there adjacent/overlapping targets
    $bciLastHead = tell(BBCFILE);
    last if( read( BBCFILE, my $buffer, $headbytes) != $headbytes );
    ($pos,$cd) = unpack "L2", $buffer;
    last if( $pos == 0 ); # start of next contig or error
    last if( $pos > $targEnd );
    $wdsiz = ($cd & 6) >> 1;
    next if( $wdsiz == 0 );  # ignore 0 read regions (for now)
    $rdlen = $cd >> 3;
    printf STDERR "Read BBC region $pos:$rdlen.$wdsiz.%d\n",($cd&1) if( $detailLog );
    if( $pos+$rdlen < $targSrt )
    {
      # ignore regions ending before target start
      last unless( seek( BBCFILE, $rdlen << $wdsiz, 1 ) );
      next;
    }
    # skips read to align for regions overlapping target start
    if( $pos < $targSrt )
    {
      my $skip = $targSrt - $pos;
      last unless( seek( BBCFILE, $skip << $wdsiz, 1 ) );
      $rdlen -= $skip;
      $pos = $targSrt;
    }
    # debatable if reading whole block into memory would be useful here - reading is buffered anyway
    if( $wdsiz == 3 ) {$upcstr = "L2";}
    elsif( $wdsiz == 2 ) {$upcstr = "S2";}
    else {$upcstr = "C2";}
    $wdsiz = 1 << $wdsiz;
    # note: whether the read is actually denoted as on-target is ignorred here - BED definition overrides
    while( $rdlen )
    {
      read( BBCFILE, my $buffer, $wdsiz );
      if( $pos >= $targSrt )
      {
        # counts at base coverage levels
        ($fcov,$rcov) = unpack $upcstr, $buffer;
        my $tcov = $fcov+$rcov;
        ++$cov20x  if( $tcov >= 20 );
        ++$cov100x if( $tcov >= 100 );
        ++$cov500x if( $tcov >= 500 );
      }
      --$rdlen;
      last if( ++$pos > $targEnd );
    }
  }
  # output original BED file fields with appended coverage data
  ++$targetsOut;
  printf "%s\t%d\t%d\t%d\n", join("\t",@fields), $cov20x, $cov100x, $cov500x;
}
close( TRGFILE );

if( $numBedWarnings )
{
  print STDERR "$CMD: $numBedWarnings BED file warnings were detected!\n";
  print STDERR " - Re-run with the -w option to see individual warning messages.\n" if( !$bedwarn );
  $logopt = 1;
}
print STDERR "Output coverage for $targetsOut regions of $numTargets.\n" if( $logopt );

# ----------------END-------------------

# Load the BCI file into arrays for the bci file corresponding to the specified bcc file.
sub loadBCI
{
  my $bbcfile = $_[0];
  my $bcifile = $bbcfile . '.bci';
  unless( -e $bcifile )
  {
    $bcifile = $bbcfile;
    $bcifile =~ s/\.bbc$//;
    $bcifile .= ".bci";
  }
  $bciIndexSize = 0;  # in case error occurs
  unless( -e $bcifile )
  {
    print STDERR "Warning: no BCI file coresponding to BBC file $bbcfile found.\n";
    return 1;
  }
  open BCIFILE, "<:raw", $bcifile or die "Could not open $bcifile or $bbcfile.bci: $!\n";
  my $fsize = (stat $bcifile)[7];
  my $intsize = 4;
  my $numbytes = $intsize * 2;
  read(BCIFILE, my $buffer, $numbytes) == $numbytes or die "Did not read $numbytes bytes from $bcifile\n";
  ($bciBlockSize,$bciNumChroms) = unpack "L2", $buffer;
  print STDERR "Read blocksize ($bciBlockSize) and number of contigs ($bciNumChroms) from $bcifile\n" if( $logopt );
  if( $bciBlockSize <= 0 )
  {
    close(BCIFILE);
    return 0;
  }
  $numbytes = $bciNumChroms * $intsize;
  read(BCIFILE, $buffer, $numbytes) == $numbytes or die "Did not read $numbytes bytes from $bcifile\n";
  @bciBlockOffsets = unpack "L[$bciNumChroms]", $buffer;
  # read remainder of file to unsigned int array
  $numbytes = $fsize - (2 + $bciNumChroms) * $intsize;
  read(BCIFILE, $buffer, $numbytes) == $numbytes or die "Did not read $numbytes bytes from $bcifile\n";
  $bciIndexSize = $numbytes >> 2;
  @bciIndex = unpack "L[$bciIndexSize]", $buffer;
  close(BCIFILE);
  $bciLastChr = -1;
  # convert bciIndex to double integers
  my $highWord = 0;
  my @bigints = (($highBit) x $bciIndexSize);
  $bigints[0] = $bciIndex[0];
  for( my $i = 1; $i < $bciIndexSize; ++$i ) {
    $highWord += BIGWORD if( $bciIndex[$i] > 0 && $bciIndex[$i] < $bciIndex[$i-1] );
    $bigints[$i] = $bciIndex[$i] + $highWord;
  }
  @bciIndex = @bigints;
  return 1;
}

# Moves current file cursor to that for the given contig index and position
# If ahead of current position and contig index has changed.
# returns 0 if location is out of ranges else 1
sub bciSeekForward
{
  my ($chromIdx,$srt) = @_;
  return 0 if( $chromIdx < 0 || $chromIdx >= $bciNumChroms );
  my $chrBlockSrt = $bciBlockOffsets[$chromIdx];
  my $blockIdx = $chrBlockSrt+int(($srt-1)/$bciBlockSize);
  return 0 if( $srt < 1 || $blockIdx >= $bciIndexSize );
  my $blockSrt = $bciIndex[$blockIdx];
  seek(BBCFILE,$blockSrt,0) if( $chromIdx != $bciLastChr || $blockSrt > $bciLastHead );
  $bciLastChr = $chromIdx;
  return 1;
}
