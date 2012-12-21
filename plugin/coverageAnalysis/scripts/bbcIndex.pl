#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create a Base Coverage Index file (.bci) for a Binary Base Coverage (.bbc) file (as created by bam2bbc.pl).
Output file is the <bbc file>.bci.";
my $USAGE = "Usage:\n\t$CMD [options] <BBC file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -l Print extra Log information to STDERR.
  -O <file> Output file name. Default <BBC file>.bci (replacing .bbc extension if present).
  -S <int> Block Size for indexing points. Default: 100000.";

my $bciBlocksize = 100000;
my $logopt = 0;
my $bcifile = "";

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-S') {$bciBlocksize = shift;}
  elsif($opt eq '-O') {$bcifile = shift;}
  elsif($opt eq '-l') {$logopt = 1;}
  elsif($opt eq '-h' || $opt eq "?" || $opt eq '--help') {$help = 1;}
  else
  {
    print "$CMD: Invalid option argument: $opt\n";
    print "$OPTIONS\n";
    exit 1;
  }
}
my $nargs = scalar @ARGV;
if( $help )
{
  print "$DESCR\n";
  print "$USAGE\n";
  print "$OPTIONS\n";
  exit 1;
}
elsif( $nargs != 1 )
{
  print "$CMD: Invalid number of arguments.\n";
  print "$USAGE\n";
  exit 1;
}

my $bbcfile = shift(@ARGV);

$bciBlocksize = 100000 if( $bciBlocksize < 1 );

if( $bcifile eq "" || $bcifile eq "-" )
{
  $bcifile = $bbcfile;
  $bcifile =~ s/\.bbc$//;
  $bcifile .= ".bci";
}

#--------- End command arg parsing ---------

# open BBCFILE and read contig header string
open( BBCFILE, "<:raw", $bbcfile ) || die "Failed to open BBC file $bbcfile\n";
chomp( my $contigList = <BBCFILE> );
my @chromName = split('\t',$contigList );
my $numChroms = scalar(@chromName);
my @chromSize;
my $genomeSize = 0;
for( my $i = 0; $i < $numChroms; ++$i )
{
  my @fields = split('\$',$chromName[$i]);
  $chromName[$i] = $fields[0];
  $chromSize[$i] = int($fields[1]);
  $genomeSize += $chromSize[$i];
}
print STDERR "Read $numChroms contig names and lengths. Total contig size: $genomeSize\n" if( $logopt );
printf STDERR "Header string length: %d\n", length($contigList)+1 if( $logopt );

# create depth index file header
my @bciBlocks = ((0) x ($numChroms+1));
my $bciCumBlocks = 0;
my $headSize = $numChroms + 1;
my @bciBlockOffsets = ((0) x $headSize);
$bciBlockOffsets[0] = $numChroms;
for( my $cn = 1; $cn <= $numChroms; ++$cn )
{
  my $nblocks = $chromSize[$cn-1] / $bciBlocksize;
  $bciBlocks[$cn] = ($nblocks == int($nblocks)) ? $nblocks : int($nblocks + 1);
  $bciBlockOffsets[$cn] = $bciCumBlocks;
  $bciCumBlocks += $bciBlocks[$cn];
}
open( BCIFILE, ">:raw", $bcifile ) || die "Failed to open binary BCI file for writing $bcifile\n";
print BCIFILE pack "L L[$headSize]", $bciBlocksize, @bciBlockOffsets;

# The depth file MUST have same contigs in the same order as in the genome file for indexing - other wise an error is issued
my @bciChromOffsets;
my ($numIndexed,$chrid,$bciChromIndex,$bciCoordCheck);
my ($chrnum,$bciCoordCheck) = (0,0);
my $intsize = 4;
my $headbytes = 2 * $intsize;
my ($pos,$cd,$wrdsz);
while(1)
{
  last if( read( BBCFILE , my $buffer, $headbytes) != $headbytes );
  my ($pos,$cd) = unpack "L2", $buffer;
  if( $pos == 0 )
  {
    # test next contig number is as expected - no error unless the bbc file was created correctly
    my $cnum = $cd;
    $chrid = $chromName[$cnum-1];
    if( !defined($chromName[$cnum-1]) )
    {
      print STDERR "$CMD: Error: Contig number $cd is not specified in bbc contig list.\n";
      close(BBCFILE);
      close(BCIFILE);
      unlink($bcifile);
      exit 1;
    }
    # output index array for last contig
    if( $bciCoordCheck )
    {
      print BCIFILE pack "L[$numIndexes]", @bciChromOffsets;
      print "Created $numIndexes indexes for contig #$chrnum = $chromName[$chrnum-1] (seek offset: $bciChromOffsets[0])\n" if( $logopt );
    }
    # output empty index arrays for contigs with no coverage
    for( my $cn = $chrnum+1; $cn < $cnum; ++$cn )
    {
      $numIndexes = $bciBlocks[$cn];
      @bciChromOffsets = ((0) x $numIndexes);
      print BCIFILE pack "L[$numIndexes]", @bciChromOffsets;
      print "Created $numIndexes x 0 indexes for contig #$cn = $chromName[$cn-1]  (seek offset: $bciChromOffsets[0])\n" if( $logopt );
    }
    # prepare new array to get block starts
    print "Found start of contig $chrid ($cnum)\n" if( $logopt );
    $chrnum = $cnum;
    $seekDepthChrom = tell(BBCFILE);
    $numIndexes = $bciBlocks[$chrnum];
    @bciChromOffsets = ((0) x $numIndexes);
    $bciChromIndex = 0;
    $bciCoordCheck = 1;
    next;
  }
  while( $pos >= $bciCoordCheck )
  {
    #print "Recorded offset $bciChromIndex @ $seekDepthChrom for $pos >= $bciBlocksize\n" if( $logopt );
    $bciChromOffsets[$bciChromIndex++] = $seekDepthChrom;
    $bciCoordCheck += $bciBlocksize;
  }
  $wrdsz = ($cd >> 1) & 3;
  seek( BBCFILE, ($cd >> 3) << $wrdsz, 1 ) if( $wrdsz );
  $seekDepthChrom = tell(BBCFILE);
}
# Output index array for last contig if generated
if( $bciCoordCheck )
{
  print BCIFILE pack "L[$numIndexes]", @bciChromOffsets;
  print "End: created $numIndexes indexes for contig #$chrnum = $chromName[$chrnum-1]  (seek offset: $bciChromOffsets[0])\n" if( $logopt );
}
# Create 0 indexes for non-covered contigs remaining
for( my $cn = $chrnum+1; $cn <= $numChroms; ++$cn )
{
  $numIndexes = $bciBlocks[$cn];
  @bciChromOffsets = ((0) x $numIndexes);
  print BCIFILE pack "L[$numIndexes]", @bciChromOffsets;
  print "End: created $numIndexes x 0 indexes for contig $chromName[$cn-1] ($cn) (seek offset: $bciChromOffsets[0])\n" if( $logopt );
}
close( BBCFILE );
close( BCIFILE );

if( $logopt )
{
  my $size = (stat $bcifile)[7];
  my $exps = (2 + $numChroms + $bciCumBlocks) * 4;
  print "Created file $bcifile: size = $size bytes (expected $exps)\n";
}

