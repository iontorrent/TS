#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create binned coverage across the whole of the effective reference. (Output to STDOUT.)";
my $USAGE = "Usage:\n\t$CMD [options] <BAM file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -b <int> Number of bins to output. Default: 600.
  -d Ignore Duplicate reads.
  -u Include only Uniquely mapped reads (MAPQ > 1).
  -B <file> Optional BED file to define effective genome. Required genome/fai file not specified (-G option).
  -G <file> Genome/FASTA index (fai) data file (chromosomes+lengths) used to specify expected chromosome order.";

my $genome="";
my $bedfile="";
my $numbins=600;
my $nondupreads=0;
my $uniquereads=0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-G') {$genome = shift;}
    elsif($opt eq '-B') {$bedfile = shift;}
    elsif($opt eq '-b') {$numbins = shift;}
    elsif($opt eq '-d') {$nondupreads = 1;}
    elsif($opt eq '-u') {$uniquereads = 1;}
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
    print STDERR "$CMD: Invalid number of arguments.";
    print STDERR "$USAGE\n";
    exit 1;
}

my $bamfile = shift(@ARGV);

$binsize += 0;
$binsize = 600 if( $binsize < 1 );

my $haveGenome = ($genome ne "");
my $haveBed = ($bedfile ne "");

#--------- End command arg parsing ---------

# read expected chromosomes
my @chromNames;
my %chromMaps;
my $numChroms = 0;
my $genomeSize = 0;

# if supplied, use genome/fai file to specify contigs and contig order
# - otherwise order is defined by bed file
if( $haveGenome )
{
    open( GENOME, $genome ) || die "Cannot read genome info. from $genome.\n";
    while( <GENOME> )
    {
        chomp;
        my ($chrid,$chrlen) = split;
        $chromNames[++$numChroms] = $chrid;
        $chromMaps{$chrid} = $genomeSize;
        $genomeSize += $chrlen;
    }
    close( GENOME );
}
elsif( !$haveBed )
{
   die "Genome file must be specified if not using target regions (BED file).\n";
}

# create hash arrays of target starts and ends and cumulative target position for binning
my %targSrts;
my %targEnds;
my %targMaps;
my $targetSize = 0;
my $numTracks = 0;
if( $haveBed )
{
  open( BEDFILE, "$bedfile" ) || die "Cannot open targets file $bedfile.\n"; 
  while( <BEDFILE> )
  {
    chomp;
    @fields = split;
    my $chrid = $fields[0];
    next if( $chrid !~ /\S/ );
    if( $chrid eq "track" )
    {
	++$numTracks;
	if( $numTracks > 1 )
	{
            print STDERR "\nWARNING: Bed file has multiple tracks. Ingoring tracks after the first.\n";
            last;
	}
	if( $targetSize > 0 )
	{
            print STDERR "\nERROR: Bed file incorrectly formatted: Contains targets before first track statement.\n";
            exit 1;
	}
	next;
    }
    if( !defined($chromMaps{$chrid}) )
    {
        if($haveGenome)
        {
            print STDERR "\nERROR: Target fragment ($chrid) not present in specified genome.\n";
            exit 1;
        }
        # if genome undefined use bedfile to define chromosomes and order
        $chromNames[++$numChroms] = $chrid;
        $chromMaps{$chrid} = 1;
    }
    push( @{$targSrts{$chrid}}, $fields[1]+1 );
    push( @{$targEnds{$chrid}}, $fields[2] );
    push( @{$targMaps{$chrid}}, $targetSize );
    $targetSize += $fields[2] - $fields[1];
  }
  close( BEDFILE );
}

# explicit option to ignore filtering of non-duplicates
my $samopt= ($nondupreads ? "" : "-F 0x304 ").($uniquereads ? "-Q 1" : "");

# Check/open bamfile and set up bin IDs for whole genome/ targeted modes
my @binid;
if( $haveBed )
{
  $binsize = $targetSize / $numbins;
  my $lastChrom;
  my $binnum = 0;
  for( my $cn = 1; $cn <= $numChroms; ++$cn )
  {
    my $chrid = $chromNames[$cn];
    next if( !defined($targMaps{$chrid}) );
    my @targMap = @{$targMaps{$chrid}};
    my $lbin = $targMap[scalar(@targMap)-1] / $binsize;
    my $ibin = int($lbin);
    if( $cn > 1 && $lbin != $ibin )
    {
      $binid[$binnum++] = $lastChrom . '-' . $chrid;
    }
    while( $binnum <= $ibin )
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
  open( PILEUP, "samtools depth $samopt -b $bedfile $bamfile |" ) || die "Cannot read base coverage from $bamfile.\n";
}
else
{
  $binsize = $genomeSize / $numbins;
  my $chrid = $chromNames[1];
  my $chrsz = $numChroms > 1 ? $chromMaps{$chromNames[2]} : $genomeSize;
  my $chrn = 2;
  my $pos = 0;
  for( my $i = 0; $i < $numbins; ++$i )
  {
    $binid[$i] = $chrid;
    $pos = $i * $binsize;
    while( $pos > $chrsz )
    {
      $chrid = $chromNames[$chrn];
      $binid[$i] .= "-" . $chrid;
      $chrsz = $chrn < $numChroms ? $chromMaps{$chromNames[++$chrn]} : $genomeSize;
    }
  }
  open( PILEUP, "samtools depth $samopt $bamfile |" ) || die "Cannot read base coverage from $bamfile.\n";
}

my @covbin;
my $bin = 0;
my $lastBin = 1;
my $lastChr = "";
while( <PILEUP> )
{
    my $line = $_;
    my ($chrid,$pos,$cnt) = split;
    if( !defined($chromMaps{$chrid}) )
    {
        if( $haveGenome )
        {
            print STDERR "\nERROR: Target fragment ($chrid) not present in specified genome.\n";
            exit 1;
        }
        # if genome undefined use bedfile to define chromosomes and order
        $chromNames[++$numChroms] = $chrid;
        $chromMaps{$chrid} = 1;
    }
    # look for hit on target
    if( $haveBed )
    {
      next if( !defined($targMaps{$chrid}) );
      my $flid = floor_bsearch($pos,\@{$targSrts{$chrid}});
      if( $flid < 0 || $pos > $targEnds{$chrid}[$flid] )
      {
        # base coverage not in targets shouldn't happen here
        print STDERR "$CMD: Warning: $chrid:$pos was not on-target (ti = $flid)\n";
        next;
      }
      $bin = int( ($targMaps{$chrid}[$flid] + $pos - $targSrts{$chrid}[$flid])/ $binsize );
    }
    else
    {
      $bin = int(($pos+$chromMaps{$chrid}-1) / $binsize);
    }
    $covbin[$bin] += $cnt;
}
close( PILEUP );

# Dump out the bins to STDOUT
print "contigs\treads\n";
for( my $i = 0; $i < $numbins; $i++ )
{
    printf "%s\t%.0f\n", $binid[$i], $covbin[$i];
}

# ----------------END-------------------

sub floor_bsearch
{
    # return lowest index for which lows[index] <= val (<= lows[index+1])
    # assumes 2nd arg is a pointer to a non-empty array of assending-sorted values
    my ($val,$lows) = @_;
    # return -1 if value is less than the first value in the array
    if( $lows->[0] > $val ) { return -1; }
    my ($l,$u) = (0, scalar(@{$lows})-1);
    # return last index if value is >= the last value in the array
    if( $lows->[$u] <= $val ) { return $u; }
    # value must be within ranges
    while(1)
    {
	my $i = int( ($l + $u)/2 );
	if( $val < $lows->[$i] ) { $u = $i; }
        elsif( $val < $lows->[$i+1] ) { return $i; }
	else { $l = $i+1; } 
    }
}
