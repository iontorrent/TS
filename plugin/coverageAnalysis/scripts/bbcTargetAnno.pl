#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create table file (tsv.xls) of bed target regions coverage for given base coverage file. (Output to STDOUT.)
The BED file is assumed to be that produced for PGM files processed by gcAnnoBed.pl for standard (table file) output.";
my $USAGE = "Usage:\n\t$CMD [options] <BBC file> <Annotated BED file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -l Print extra Log information to STDERR.
  -w Print Warning messages for potential BED file issues to STDOUT.
  -b Output file is extended BED format (with BED coordinates and bedDetail track line).
     Default: Text table file (tab-separated text fields) with 1-based coordinates assumed field names.";

my $logopt = 0;
my $bedout = 0;
my $bedwarn = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-l') {$logopt = 1;}
  elsif($opt eq '-b') {$bedout = 1;}
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
my $bedfile = shift(@ARGV);

my $onebase = !$bedout;

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

my $lastHead = tell(BBCFILE); # start of chrom#0

my $lastChrIdx = -1;
my ($numTargets,$numTracks,$targetsOut) = (0,0,0);
my ($targSrt,$targEnd);
my ($cd,$rdlen,$wdsiz,$fcov,$rcov);

my $intsize = 4;
my $headbytes = 2 * $intsize;

my $detailLog = 0; # set for debugging

# generate output header line
my $headerLine = "track type=bedDetail";
if( !$bedout )
{
  # replace default bed track line assumed field titles (by position)
  $headerLine = "";
  my @titles = ("contig_id","contig_srt","contig_end","region_id","gene_id","gc_count");
  for( my $i = 0; $i < scalar(@titles); ++$i )
  {
    if( defined($titles[$i]) ) { $headerLine .= "$titles[$i]\t"; }
    else { $headerLine .= sprintf("field%d\t", $i+1); }
  }
  # these are the added 6 base coverage fields
  $headerLine .= "covered\tuncov_5p\tuncov_3p\tave_basereads\tfwd_basereads\trev_basereads";
}

my $numBedWarnings = 0;

# load BCI file indexes
my ($bciNumChroms,$bciBlockSize,$bciLastChr);
my (@bciIndex,@bciBlockOffsets);
my $bciIndexSize = 0;  # indicates no BCI load
exit 0 unless( loadBCI($bbcfile) );

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
      print STDERR "WARNING: Bed file has multiple tracks. Ignoring tracks after the first.\n" if( $bedwarn );
      ++$numBedWarnings;
      last;
    }
    if( $numTargets > 0 )
    {
      print STDERR "ERROR: Bed file incorrectly formatted: Contains targets before first track statement.\n";
      exit 1;
    }
    if( $bedout ) {
      $headerLine = $_;
    } elsif( m/ionVersion=([\S\.]*)/ ) {
      my $tsv = $1;
      if( $tsv =~ m/^(\d+\.\d+)(\..*)?$/ ) {
        $tsv = $1;
        $headerLine =~ s/gene_id/attributes/ if( $tsv >= 4 );
      }
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
  my $sameSrt = ($srt == $targSrt);
  $targSrt = $srt;
  if( $srt < $targSrt )
  {
    print STDERR "ERROR: Region $chrid:$srt-$end is out-of-order vs. previous region $chrid:$targSrt-$targEnd.\n";
    exit 1;
  }
  if( $srt <= $targEnd )
  {
    ++$numBedWarnings;
    if( $end <= $targEnd || $sameSrt )
    {
      if( $bedwarn ) {
        printf STDERR "Warning: Region $chrid:$srt-$end %s previous region $chrid:$targSrt-$targEnd.\n",
          $end <= $targEnd ? "is entirely overlapped by" : "entirely overlaps";
        print STDERR " - This region will be excluded from the output file.\n";
      }
      $targEnd = $end;
      next;
    }
    print STDERR "Warning: Region $chrid:$srt-$end overlaps previous region $chrid:$targSrt-$targEnd.\n" if( $bedwarn );
    $srt = $targEnd + 1;
    print STDERR " - Report will contain partial coverage for the overlap region $chrid:$srt-$end.\n" if( $bedwarn );
  }
  $targEnd = $end;
  # reset cursor to start of last section looked at (for adjacent/overlapping targets)
  seek(BBCFILE,$lastHead,0);
  my ($pos,$nfwd,$nrev,$fcov5p,$lcov3p,$tcov) = (0,0,0,0,0,0);
  if( bciSeekForward($chrIdx,$targSrt) )
  {
    # read from current BBC cursor to end of current target region
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
          # record coverage where base coverage is non-zero over target
          ($fcov,$rcov) = unpack $upcstr, $buffer;
          if( $fcov+$rcov )
          {
            $nfwd += $fcov; # forward base reads
            $nrev += $rcov; # reverse base reads
            $fcov5p = $pos if( !$fcov5p ); # position in target of first base covered
            $lcov3p = $pos; # position in target of last base covered
            ++$tcov; # total base reads on target (not including deletions)
          }
        }
        --$rdlen;
        last if( ++$pos > $targEnd );
      }
    }
  }
  # output original BED file fields with appended coverage data
  print "$headerLine\n" if( ++$targetsOut == 1 );
  my $len = $targEnd - $targSrt + 1;
  my $depth = $len > 0 ? ($nfwd + $nrev) / $len : 0;
  ++$fields[1] if( $onebase );
  for( my $i = 0; $i < scalar(@fields); ++$i )
  {
    print "$fields[$i]\t";
  }
  $fcov5p = $fcov5p ? $fcov5p-$targSrt : $len;
  $lcov3p = $lcov3p ? $targEnd-$lcov3p : $len;
  printf "%d\t%d\t%d\t%.3f\t%d\t%d\n", $tcov, $fcov5p, $lcov3p, $depth, $nfwd, $nrev;
}
close( BEDFILE );

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

# Move the open BCC file cursor to the position for reading the specified chromosome index (0-based)
# and read position (1-based) <= specified coordinate. Here it is assumed read locations will be
# given in order so reset of the file cursor is only allowed to be forwards.
# Code exists with error message if reading not initiated or contig/position is out of range.
# Returns 0 if the contig read for unanticipated input or there are no reads at all for the current contig/read block.
sub bciSeekForward
{
  if( !$bciIndexSize )
  {
    print STDERR "ERROR: bciSeekForward called with no BCI index loaded.\n" if( $logopt && !$numBciWarn );
    exit 1;
  }
  my ($chromIdx,$srt) = @_;
  return 0 if( $chromIdx < $bciLastChr || $chromIdx >= $bciNumChroms );
  my $chrBlockSrt = $bciBlockOffsets[$chromIdx];
  my $blockIdx = $chrBlockSrt+int(($srt-1)/$bciBlockSize);
  if( $srt < 1 || $blockIdx >= $bciIndexSize )
  {
    print STDERR "ERROR: bciSeekForward called with coordinate $srt out of range for contig index $chromIdx.\n";
    exit 1;
  }
  $bciLastChr = $chromIdx;
  my $blockSrt = $bciIndex[$blockIdx];
  printf STDERR "Block start = $blockSrt at index $blockIdx, file at %d\n", tell(BBCFILE) if( $detailLog );
  # skip non-represented contigs - ok since BBCFILE starts with contig names - 0 seek not allowed
  return 0 unless( $blockSrt );
  my $fpos = tell(BBCFILE);
  seek(BBCFILE,$blockSrt,0) if( $blockSrt > $fpos );
  return 1;
}

