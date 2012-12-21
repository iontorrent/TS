#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create a Binary Base Coverage file from a BAM file.
This file will contain base depth-of-coverage information for forward and reverse base reads per reference base.
A genome file, specifying contigs and lengths such as a fasta.fai, is required for vaidation and indexing.
If a targets (BED) file is given it will also distinguish coverage for bases within and without those regions.
The default output file will be <infile>.bbc for input file <infile>.bam.";
my $USAGE = "Usage:\n\t$CMD [options] <genome file> <BAM file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -l Show extra log information to STDERR.
  -d Ignore Duplicate reads.
  -u Include only Uniquely mapped reads (MAPQ > 1).
  -p Print the total number of base reads output to STDOUT at the end of the run.
  -t Output BBC file in text format (for testing only).
  -w Print Warning messages for potential BED file issues to STDOUT (-l also has to be specified).
  -B <file> BED file to define regions of interest.
  -O <file> Output file name. Default <BAM file>.bbc (replacing .bam extension if present).
  -S <int> Block Size for indexing points. (Max. region block size is 1/10 this.) Default: 100000.";

my $bedfile = "";
my $bbcfile = "";
my $dfiBlocksize = 100000;
my $logopt = 0;
my $txtout = 0;
my $nondupreads = 0;
my $uniquereads = 0;
my $bedwarn = 0;
my $printreads = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-B') {$bedfile = shift;}
  elsif($opt eq '-O') {$bbcfile = shift;}
  elsif($opt eq '-S') {$dfiBlocksize = shift;}
  elsif($opt eq '-d') {$nondupreads = 1;}
  elsif($opt eq '-u') {$uniquereads = 1;}
  elsif($opt eq '-p') {$printreads = 1;}
  elsif($opt eq '-t') {$txtout = 1;}
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
elsif( scalar @ARGV != 2 )
{
  print STDERR "$CMD: Invalid number of arguments.";
  print STDERR "$USAGE\n";
  exit 1;
}

my $genome = shift(@ARGV);
my $bamfile = shift(@ARGV);

if( $bbcfile eq "" || $bbcfile eq "-" )
{
  $bbcfile = $bamfile;
  $bbcfile =~ s/\.bam$//;
  $bbcfile .= ".bbc";
}
$dfiBlocksize = 100000 if( $dfiBlocksize < 1 );

my $haveTargets = ($bedfile ne "" && $bedfile ne "-");

#--------- End command arg parsing ---------

print STDERR "$CMD started at ".localtime()."\n" if( $logopt );

# read expected contigs
my @chromSize;
my @chromName;
my %chromNum;
my $numChroms = 0;
my $genomeSize = 0;
my $genomeString = '';

# check/read genome file
open( GENOME, $genome ) || die "Cannot read genome info. from $genome.\n";
while( <GENOME> )
{
  chomp;
  my ($chrid,$chrlen) = split;
  $chromNum{$chrid} = ++$numChroms;
  $chromName[$numChroms] = $chrid;
  $chromSize[$numChroms] = $chrlen;
  $genomeSize += $chrlen;
  $genomeString .= "\t" if( $numChroms > 1 );
  $genomeString .= $chrid . "\$" . $chrlen;
}
close( GENOME );
print STDERR "Read $numChroms contigs from genome file. Total size = $genomeSize.\n" if( $logopt );

my (%targSrts,%targEnds);
loadBedRegions() if( $haveTargets );

print STDERR "Analyzing forward and reverse read depths...\n" if( $logopt );

my $samopt= ($nondupreads ? "-F 0x714" : "-F 0x314").($uniquereads ? " -Q 1" : "");
my $fwdcov = "samtools depth $samopt \"$bamfile\" 2> /dev/null |";
open(FWDCOV,$fwdcov) || die "Could not open forward coverage pipe '$fwdcov'.";

$samopt= ($nondupreads ? "-F 0x704" : "-F 0x304 ").($uniquereads ? " -Q 1" : "");
my $bamcov = "samtools depth $samopt \"$bamfile\" 2> /dev/null |";
open(BAMCOV,$bamcov) || die "Could not open total coverage pipe '$bamcov'.";

open(BBCFILE,">$bbcfile") || die "Could not write to '$bbcfile'.";
print BBCFILE "$genomeString\n";

# Global variables for collecting buffered data for output
my $recHeadSize = 8; # 2x integers
my $wordToggleSize = 2 * $recHeadSize;
my $bufSize = int($dfiBlocksize/10); # must be < 2^29
my @ioBuffer = ((0) x ($bufSize*2));
my $nReads = 0;
my ($srtPos,$lstPos,$lstOntarg,$curChrom);
my ($wordSize,$backWrdsz,$backStep);

my ($checkTargets,$ontarg,$totalReads,$chrReads) = (0,0,0,0);
my (@targSrtAry,@targEndAry,$targAryLen,$targNum,$targSrt,$targEnd);
my ($chrid,$pos,$cov,$fchr,$fpos,$fcov,$lchr,$line);
unless( eof(FWDCOV) )
{
  chomp($line = <FWDCOV>);
  ($fchr,$fpos,$fcov) = split('\t',$line);
}
while(<BAMCOV>)
{
  chomp;
  ($chrid,$pos,$cov) = split('\t',$_);
  if( $chrid ne $lchr )
  {
    set_chrom($chrid);
    print STDERR "Processing reads for $chrid\n" if( $logopt );
    $lchr = $chrid;
    if( $haveTargets )
    {
      $targAryLen = 0;
      $checkTargets = defined($chromNum{$chrid});
      if( $checkTargets )
      {
        @targSrtAry = @{$targSrts{$chrid}};
        @targEndAry = @{$targEnds{$chrid}};
        $targAryLen = scalar(@targSrtAry);
        $targNum = 0;
        $targSrt = $targSrtAry[0];
        $targEnd = $targEndAry[0];
      }
      print STDERR "- checking for $targAryLen targets...\n" if( $logopt );
    }
  }
  if( $checkTargets )
  {
    if( $pos < $targSrt )
    {
      $ontarg = 0;
    }
    elsif( $pos <= $targEnd )
    {
      $ontarg = 1;
    }
    else
    {
      while( ++$targNum < $targAryLen )
      {
        last if( $pos <= $targEndAry[$targNum] );
      }
      if( $targNum < $targAryLen )
      {
        $targSrt = $targSrtAry[$targNum];
        $targEnd = $targEndAry[$targNum];
        $ontarg = ($pos >= $targSrt) ? 1 : 0;
      }
      else
      {
        $checkTargets = $ontarg = 0;
      }
    }
  }
  $chrReads += $cov;
  if( $chrid ne $fchr || $pos < $fpos )
  {
    pack_reads($pos,0,$cov,$ontarg);
    next;
  }
  pack_reads($pos,$fcov,($cov-$fcov),$ontarg);
  if( eof(FWDCOV) )
  {
    $fchr = "";
  }
  else
  {
    chomp($line = <FWDCOV>);
    ($fchr,$fpos,$fcov) = split(' ',$line);
  }
}
close(FWDCOV);
close(BAMCOV);
set_chrom("");
close(BBCFILE);
if( $logopt )
{
  print STDERR "Output $totalReads base reads to $bbcfile\n";
  print STDERR "$CMD completed at ".localtime()."\n";
}
print "$totalReads\n" if( $printreads );

#------------------------------ END -----------------------------

# Load all BED file regions in to memory and validate BED file.
# This functions is for code organization only and not intended to be general function.
sub loadBedRegions
{
  my ($lastChr,$lastSrt,$lastEnd,$numTracks,$numTargets,$numTargReads,$numWarns) = (0,0,0,0,0,0,0);
  open( BEDFILE, "$bedfile" ) || die "Cannot open targets file $bedfile.\n";
  while( <BEDFILE> )
  {
    my ($chrid,$srt,$end) = split;
    next if( $chrid !~ /\S/ );
    if( $chrid eq "track" )
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
    if( !defined($chromNum{$chrid}) )
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
        next;
      }
      print STDERR "Warning: Region $chrid:$srt-$end overlaps previous region $chrid:$lastSrt-$lastEnd.\n" if( $bedwarn );
      $srt = $lastEnd + 1;
    }
    $lastSrt = $srt;
    $lastEnd = $end;
    ++$numTargets;
    push( @{$targSrts{$chrid}}, $srt );
    push( @{$targEnds{$chrid}}, $end );
  }
  close( BEDFILE );
  if( $logopt )
  {
    print STDERR "$CMD: $numWarns BED file warnings were detected!\n" if( $numWarns );
    print STDERR " - Re-run with the -w option to see individual warning messages.\n" if( $numWarns && !$bedwarn );
    print STDERR "Read $numTargets of $numTargReads target regions from $bedfile\n";
  }
}

# create hash arrays of target starts and ends - use "" for $_[0] for last call (at EOF)
sub set_chrom
{
  if( $nReads )
  {
    dump_reads();
    print STDERR "$curChrom had $chrReads base reads\n" if( $logopt );
    $totalReads += $chrReads;
    $nReads = $chrReads = 0;
  }
  $curChrom = $_[0];
  if( $curChrom ne "" )
  {
    if( $txtout ) { print BBCFILE ":\t$curChrom\n"; }
    else { print BBCFILE pack "L2", 0, $chromNum{$curChrom}; }    # denote start of next chromosome
  }
}

sub pack_reads
{
  local ($rPos,$rFwd,$rRev,$rOntarg) = @_;
  local $wrdsz = $rFwd >= $rRev ? $rFwd : $rRev;
  if( $wrdsz >= 65536 ) {$wrdsz = 8;}
  elsif( $wrdsz >= 256 ) {$wrdsz = 4;}
  elsif( $wrdsz > 0 ) {$wrdsz = 2;}
  if( $nReads == 0 )
  {
    # reset for new contig start (not new region since this leaves $nReads == 1)
    $backWrdsz = $backStep = 0;
    $srtPos = $rPos;
    $lstPos = $rPos-1;
    $lstOntarg = $rOntarg;
    $wordSize = $wrdsz;
  }
  elsif( $nReads >= $bufSize || $rPos - $lstPos > 8 || $lstOntarg != $rOntarg || $wrdsz > $wordSize )
  {
    # force dump if buffer size met, at least 8 position gap, or region switches type
    # also if word size increases then output (assumes space saving is preemptive - otherwise would need minor recursion?)
    dump_reads();
  }
  elsif( $wrdsz < $wordSize )
  {
    # check to see if most recent coverage could be saved more efficiently with a scale change
    # i.e. difference in bytes is greater than 2 inserted headers (in case immediately change up again)
    $backWrdsz = $wrdsz if( $wrdsz > $backWrdsz );
    ++$backStep;
    back_dump_reads() if( $backStep*($wordSize-$backWrdsz) > $wordToggleSize );
  }
  else
  {
    # ignore non-contiguous drops in counts scale
    $backWrdsz = $backStep = 0;
  }
  # account for possible small gaps in depth output (that do not merit a jump)
  my $idx = $nReads << 1;
  while( ++$lstPos < $rPos )
  {
    $ioBuffer[$idx++] = 0;
    $ioBuffer[$idx++] = 0;
    ++$nReads;
  }
  $ioBuffer[$idx] = $rFwd;
  $ioBuffer[++$idx] = $rRev;
  ++$nReads;
}

# dump_reads() is only called from pack_reads() as it inherits its local variables
sub dump_reads
{
  # dump current reads in buffer
  if( $txtout )
  {
    my $ws = $wordSize == 8 ? 3 : ($wordSize >> 1);
    print BBCFILE "$srtPos | $nReads.$ws.$lstOntarg\n";
    my $idx = 0;
    for( my $i = $0; $i < $nReads; ++$i,++$idx )
    {
      print BBCFILE "$ioBuffer[$idx]\t";
      ++$idx;
      print BBCFILE "$ioBuffer[$idx]\n";
    }
  }
  else
  {
    # pack position + length.wordSizeCode.onTargetBit
    my $cd = ($nReads << 3) | ($wordSize == 8 ? 6 : $wordSize) | $lstOntarg;
    print BBCFILE pack "L2", $srtPos, $cd;
    # pack length of array according to word size (none if wordSize == 0)
    my $arySiz = $nReads << 1;
    if   ( $wordSize == 8 ) { print BBCFILE pack "L[$arySiz]", @ioBuffer; }
    elsif( $wordSize == 4 ) { print BBCFILE pack "S[$arySiz]", @ioBuffer; }
    elsif( $wordSize == 2 ) { print BBCFILE pack "C[$arySiz]", @ioBuffer; }
  }
  # reset for processing next region
  $backWrdsz = $backStep = 0;
  $lstOntarg = $rOntarg;
  $wordSize = $wrdsz;
  $nReads = 0; # -> 1 on return when current read saved to array
  $srtPos = $rPos;
  $lstPos = $srtPos-1;
}

sub back_dump_reads
{
#print "back_dump_reads at $rPos: $nReads backed by $backStep, $wordSize -> $backWrdsz\n";
  # dump reads up to -$backStep positions to allow word size change
  my $eReads = $backStep-1; # grab $backStep; -1 because last read processed on return
  $nReads -= $eReads;     # only this number of reads are output
  my $j = $nReads << 1;   # record start of reads to be retained
  $wrdsz = $backWrdsz;    # becomes the new $wordSize
  dump_reads();
  # dump_reads() resets for continued reading - copy back original $backStep reads
  $nReads = $eReads;
  my $k = $nReads << 1;
  for( my $i = 0; $i < $k; ++$i, ++$j )
  {
    $ioBuffer[$i] = $ioBuffer[$j];
  }
  $srtPos -= $eReads; # $lstPos still good since this is the previous read position
}

