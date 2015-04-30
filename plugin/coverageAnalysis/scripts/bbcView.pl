#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Output data one line a time from a BBC file to STDOUT. Output has the format the format
contig<tab>position<tab>on-target<tab>forward reads<tab>reverse reads. The on-target field is either
0 or 1 and reflects the value in the BBC file, regardless of whether this overlaps with regions
specified by the targets file (-B option).";
my $USAGE = "Usage:\n\t$CMD [options] <BBC file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -l Print extra Log information to STDERR.
  -t Output reads for on-Target denoted bases only.
  -w Print Warning messages for potential BED file issues to STDOUT.
  -B <file> Bed file specifying target regions to report base coverage over.";

my $logopt = 0;
my $bedfile = '';
my $skipOffTarget = 0;
my $bedwarn = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-l') {$logopt = 1;}
  elsif($opt eq '-t') {$skipOffTarget = 1;}
  elsif($opt eq '-w') {$bedwarn = 1;}
  elsif($opt eq '-B') {$bedfile = shift;}
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

my $haveBed = ($bedfile ne '' && $bedfile ne '-');

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

my $detailLog = 0; # set for debugging

my $intsize = 4;
my $headbytes = 2 * $intsize;
my ($pos, $cd,$rdlen,$wdsiz,$fcov,$rcov,$ontarg);

# read BBC directly if there is no BED regions defined
if( !$haveBed )
{
  my $chrIdx = -1;
  while(1)
  {
    last if( read( BBCFILE , my $buffer, $headbytes) != $headbytes );
    ($pos,$cd) = unpack "L2", $buffer;
    if( $pos == 0 )
    {
      $chrIdx = $cd-1;
      $chrid = $chromName[$chrIdx];
      print STDERR "Found start of contig#$chrIdx $chrid ($cd)\n" if( $logopt );
      next;
    }
    $wdsiz = ($cd & 6) >> 1;
    next if( $wdsiz == 0 );  # ignore 0 read regions (for now)
    $rdlen = $cd >> 3;
    $ontarg = $cd & 1;
    if( $skipOffTarget && !$ontarg )
    {
      seek( BBCFILE, $rdlen << $wdsiz, 1 );
      next;
    }
    if( $wdsiz == 3 ) {$upcstr = "L2";}
    elsif( $wdsiz == 2 ) {$upcstr = "S2";}
    else {$upcstr = "C2";}
    $wdsiz = 1 << $wdsiz;
    while( $rdlen )
    {
      read( BBCFILE, my $buffer, $wdsiz );
      ($fcov,$rcov) = unpack $upcstr, $buffer;
      print "$chrid\t$pos\t$ontarg\t$fcov\t$rcov\n" if( $fcov+$rcov );
      --$rdlen;
      ++$pos;
    }
  }
  exit 0;
}

# considerably more processing using target file
my $lastHead = tell(BBCFILE); # start of chrom#0
my $lastChrIdx = -1;
my ($numTargets,$numTracks,$targetsOut) = (0,0,0);
my ($targSrt,$targEnd);

# load BCI file indexes
my ($bciNumChroms,$bciBlockSize,$bciLastChr);
my (@bciIndex,@bciBlockOffsets);
my $bciIndexSize = 0;  # indicates no BCI load
exit 0 unless( loadBCI($bbcfile) );

# read base coverage just in BED regions
my $numBedWarnings = 0;
open( BEDFILE, "$bedfile" ) || die "Cannot open targets file $bedfile.\n"; 
while( <BEDFILE> )
{
  chomp;
  @fields = split('\t',$_);
  my $chrid = $fields[0];
  next if( $chrid !~ /\S/ );
  if( $chrid =~ /^track / )
  {
    ++$numTracks;
    if( $numTracks > 1 )
    {
      ++$numBedWarnings;
      print STDERR "WARNING: Bed file has multiple tracks. Ignoring tracks after the first.\n";
      last;
    }
    if( $numTargets > 0 )
    {
      print STDERR "ERROR: Bed file incorrectly formatted: Contains targets before first track statement.\n";
      exit 1;
    }
    next;
  }
  if( !defined($chromIndex{$chrid}) )
  {
    print STDERR "ERROR: Target contig ($chrid) not present in genome as defined in the BBC file.\n";
    exit 1;
  }
  ++$numTargets;
  my $chrIdx = $chromIndex{$chrid};
  my $srt = $fields[1]+1;
  my $end = $fields[2]+0;
  print STDERR "Read target region for chrom#$chrIdx $chrid:$srt:$end\n" if( $detailLog );
  # check how new target overlaps the old (for correct annoatation of things like proper GC counts this better no happen!)
  if( $chrIdx != $lastChrIdx )
  {
    if( $chrIdx < $lastChrIdx )
    {
      print STDERR "ERROR: Target contig ($chrid) is out-of-order vs. that defined in the BBC file.\n";
      exit 1;
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
  my $sameSrt = ($srt == $targSrt);
  $targSrt = $srt;
  if( $srt <= $targEnd || $sameSrt )
  {
    ++$numBedWarnings;
    if( $end <= $targEnd || $sameSrt )
    {
      if( $bedwarn ) {
        printf STDERR "Warning: Region $chrid:$srt-$end %s previous region $chrid:$targSrt-$targEnd.\n",
          $end <= $targEnd ? "is entirely overlapped by" : "entirely overlaps";
      }
      $targEnd = $end;
      next;
    }
    print STDERR "Warning: Region $chrid:$srt-$end is overlaps previous region $chrid:$targSrt-$targEnd.\n" if( $bedwarn );
    $srt = $targEnd + 1;
  }
  $targEnd = $end;
  # reset cursor to start of last section looked at (for adjacent/overlapping targets)
  seek(BBCFILE,$lastHead,0);
  next unless( bciSeekForward($chrIdx,$targSrt) );
  # read from current BBC cursor to end of current target region
  $pos = 0;
  while( $pos <= $targEnd )
  {
    # record the start of the next BBC region in case there adjacent/overlapping targets
    $lastHead = tell(BBCFILE);
    last if( read( BBCFILE, my $buffer, $headbytes) != $headbytes );
    ($pos,$cd) = unpack "L2", $buffer;
    last if( $pos == 0 ); # start of next contig
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
    $ontarg = $cd & 1;
    if( $skipOffTarget && !$ontarg )
    {
      seek( BBCFILE, $rdlen << $wdsiz, 1 );
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
    while( $rdlen )
    {
      read( BBCFILE, my $buffer, $wdsiz );
      if( $pos >= $targSrt )
      {
        # record coverage where base coverage is non-zero over target
        ($fcov,$rcov) = unpack $upcstr, $buffer;
        print "$chrid\t$pos\t$ontarg\t$fcov\t$rcov\n" if( $fcov+$rcov );
      }
      --$rdlen;
      last if( ++$pos > $targEnd );
    }
  }
}
close( BEDFILE );

if( $bedwarn && $numBedWarnings )
{
  print STDERR "$CMD: $numBedWarnings BED file warnings were detected!\n";
}

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
  if( $detailLog ) {
    print STDERR "Num indexes @ $bciBlockSize = $bciIndexSize\n";
    for( my $i = 0; $i < $bciNumChroms; ++$i ) {
      my $j = $bciBlockOffsets[$i];
      print STDERR "Contig #$i @ $j -> file pos $bciIndex[$j]\n";
    }
    for( my $i = 0; $i < $bciIndexSize; ++$i ) {
      print STDERR "Offset $i -> file pos $bciIndex[$i]\n";
    }
  }
  return 1;
}

# Move the open BCC file cursor to the position for reading the specified chromosome index (0-based)
# and read position (1-based) <= specified coordinate. Here it is assumed read locations will be
# given in order so no reset of the file cursor is only allowed to be forwards.
# Returns 0 if the contig read for unanticipated input or there are no reads at all for the current contig/read block.
sub bciSeekForward
{
  if( !$bciIndexSize )
  {
    print STDERR "Warning: bciSeekForward called with no BCI index loaded.\n" if( $logopt );
    return 0;
  }
  my ($chromIdx,$srt) = @_;
  return 0 if( $chromIdx < $bciLastChr || $chromIdx >= $bciNumChroms );
  my $chrBlockSrt = $bciBlockOffsets[$chromIdx];
  my $blockIdx = $chrBlockSrt+int(($srt-1)/$bciBlockSize);
  if( $srt < 1 || $blockIdx >= $bciIndexSize )
  {
    print STDERR "Warning: bciSeekForward called with coordinate $srt out of range for contig index $chromIdx.\n";
    return 0;
  }
  $bciLastChr = $chromIdx;
  my $blockSrt = $bciIndex[$blockIdx];
  printf STDERR "Block start = $blockSrt at index $blockIdx for $chromIdx,$srt; file at %d\n", tell(BBCFILE) if( $detailLog );
  # skip non-represented contigs - ok since BBCFILE starts with contig names - 0 seek not allowed
  return 0 unless( $blockSrt );
  seek(BBCFILE,$blockSrt,0) if( $blockSrt > tell(BBCFILE) );
  return 1;
}

