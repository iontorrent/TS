#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create tsv of bed target regions coverage for given pileup file. (Output to STDOUT.)";
my $USAGE = "Usage:\n\t$CMD [options] <SAM pileup/depth file> <BED file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -g Create GC coverage data for each covered target. Requires SAM pileup file as input, rather than SAM depth file.
  -G <file> Genome data file (chromosomes+lengths) used to specify expected chromosome names";

my $genome="";
my $addGCdata = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-G') {$genome = shift;}
    elsif($opt eq '-g') {$addGCdata = 1;}
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

my $pileupfile = shift(@ARGV);
my $bedfile = shift(@ARGV);

my $haveGenome = ($genome ne "");

#--------- End command arg parsing ---------

# read expected chromosomes
my %corder;
my %chrom_num_target;
my $num_chroms = 0;

if( $haveGenome )
{
    open( GENOME, $genome ) || die "Cannot read genome info. from $genome.\n";
    while( <GENOME> )
    {
        chomp;
        my $chrid = (split)[0];
        $chrom_num_target{$chrid} = 0;
        $num_chroms++;
        $corder{$num_chroms} = $chrid;
    }
    close( GENOME );
}

# create hash arrays of target starts and ends
my %chrom_starts;
my %chrom_ends;

my $numTracks = 0;
my $numTargets = 0;

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
	if( $numTargets > 0 )
	{
            print STDERR "\nERROR: Bed file incorrectly formatted: Contains targets before first track statement.\n";
            exit 1;
	}
	next;
    }
    if( !defined($chrom_num_target{$chrid}) )
    {
        if($haveGenome)
        {
            print STDERR "\nERROR: Target fragment ($chrid) not present in specified genome.\n";
            exit 1;
        }
        # if genome undefined use bedfile to define chromosomes and order
        $chrom_num_target{$chrid} = 0;
        $num_chroms++;
        $corder{$num_chroms} = $chrid;
    }
    ++$numTargets;
    push( @{$chrom_starts{$chrid}}, $fields[1]+1 );
    push( @{$chrom_ends{$chrid}}, $fields[2] );
}
close( BEDFILE );
for( my $i = 1; $i <= $num_chroms; $i++ )
{
    my $k = $corder{$i};
    $chrom_num_target{$k} = scalar(@{$chrom_starts{$k}});
}

my @chrom_target_cov;
my @chrom_target_reads;
my @chrom_target_gc;

open( PILEUP, "$pileupfile" ) || die "Cannot read base coverage from $pileupfile.\n";

while( <PILEUP> )
{
    my ($chrid,$pos,$refb,$cnt) = split;
    $cnt = $refb if( $cnt eq "" ); # only 3 fields for samtools depth output
    if( !defined($chrom_num_target{$chrid}) )
    {
        if( $haveGenome )
        {
            print STDERR "\nERROR: Target fragment ($chrid) not present in specified genome.\n";
            exit 1;
        }
        # if genome undefined use bedfile to define chromosomes and order
        $chrom_num_target{$chrid} = 0;
        $num_chroms++;
        $corder{$num_chroms} = $chrid;
    }
    # look for hit on target
    if( $chrom_num_target{$chrid} )
    {
        my $flid = floor_bsearch($pos,\@{$chrom_starts{$chrid}});
        if( $flid >= 0 && $pos <= $chrom_ends{$chrid}[$flid] )
        {
            ++$chrom_target_cov{$chrid}[$flid];
	    $chrom_target_reads{$chrid}[$flid] += $cnt;
	    ++$chrom_target_gc{$chrid}[$flid] if( $refb eq "G" || $refb eq "C" );
        }
    }
}
close( PILEUP );

print "chrom_id\tstart_pos\tend_pos\tcoverage\tpc_coverage\tbase_reads\tnorm_reads";
$addGCdata ? print("\tpc_gc_reads\n") : print("\n");
for( my $i = 1; $i <= $num_chroms; $i++ )
{
    my $chrid = $corder{$i};
    my $ntarg = $chrom_num_target{$chrid};
    my @norm;
    for( my $j = 0; $j < $ntarg; ++$j )
    {
	$norm[$j] = $chrom_target_reads{$chrid}[$j] / ($chrom_ends{$chrid}[$j] - $chrom_starts{$chrid}[$j] + 1);
    }
    my @idx = sort{ $norm[$a] <=> $norm[$b] } (0..($ntarg-1));
    for( my $k = 0; $k < $ntarg; ++$k )
    {
	$j = $idx[$k];
        my $tlen = $chrom_ends{$chrid}[$j] - $chrom_starts{$chrid}[$j] + 1;
        my $ncov = $chrom_target_cov{$chrid}[$j]+0;
        my $nhit = $chrom_target_reads{$chrid}[$j]+0;
	my $pcgc = 100 * $chrom_target_gc{$chrid}[$j] / $tlen;
        my $covp = 100 * $ncov / $tlen;
        printf "%s\t%d\t%d\t%d\t%.2f\t%d\t%.3f",
            $chrid, $chrom_starts{$chrid}[$j], $chrom_ends{$chrid}[$j], $ncov, $covp, $nhit, $norm[$j];
	$addGCdata ? printf("\t%.1f\n",$pcgc) : print("\n");
    }
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
