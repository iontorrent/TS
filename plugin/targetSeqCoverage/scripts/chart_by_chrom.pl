#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create tsv list of number of reads to choromosomes.";
my $USAGE = "Usage:\n\t$CMD [options] <BAM | SAM file> <output file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -s Reads file is in SAM format (default is BAM)
  -o Output summary statistics to STDOUT
  -p Output padded target summary statistics to STDOUT (overrides -o)
  -G <file> Genome data file (chromosomes+lengths) used to specify expected chromosome names
  -B <file> Limit coverage to targets specified in this BED file
  -F <file> Output position frequency analysis to this file
  -S <N> Bin Size for location frequency output (if -F used). Default: 100
  -C <N> Minimum Count for location frequency output (if -F used). Default: 5
  -t Two-tone with 0 spacers (if -F used). Default: Set 4-Tone output for on-target values";

my $bedfile = "";
my $genome="";
my $readtype = 0;
my $freqfile = "";
my $binsize = 100;
my $minpcnt = 5;
my $spacepeaks = 0;
my $colorpeaks = 1;
my $statsout = 0;
my $padstats = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-B') {$bedfile = shift;}
    elsif($opt eq '-G') {$genome = shift;}
    elsif($opt eq '-F') {$freqfile = shift;}
    elsif($opt eq '-S') {$binsize = shift;}
    elsif($opt eq '-C') {$minpcnt = shift;}
    elsif($opt eq '-s') {$readtype = 1;}
    elsif($opt eq '-o') {$statsout = 1;}
    elsif($opt eq '-p') {$padstats = 1;}
    elsif($opt eq '-t') {$spacepeaks = 1;$colorpeaks=0}
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

if( $binsize < 1 )
{
    print STDERR "ERROR: Distance frequency bin size ($binsize) must be > 0.\n";
    exit 1;
}

my $mappings = shift;
my $outfile = shift;

my $haveGenome = ($genome ne "");
my $haveTargets = ($bedfile ne "");
my $doAnalysis = ($freqfile ne "");

#--------- End command arg parsing ---------

# read expected chromosomes
my %corder;
my %chrom_counts;
my %chrom_on_target;
my %chrom_num_target;
my $num_chroms = 0;

if( $haveGenome )
{
    open( GENOME, $genome ) || die "Cannot read genome info. from $genome.\n";
    while( <GENOME> )
    {
        chomp;
        my $chrid = (split)[0];
        $chrom_counts{$chrid} = 0;
        $chrom_num_target{$chrid} = 0;
        $num_chroms++;
        $corder{$num_chroms} = $chrid;
    }
    close( GENOME );
}

my $numTracks = 0;
my $numTargets = 0;

# scan bedfile for targets that might not be in the genome (an error) or reads
if( $haveTargets )
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
            if( $numTargets > 0 )
	    {
        	print STDERR "\nERROR: Bed file incorrectly formatted: Contains targets before first track statement.\n";
        	exit 1;
	    }
	    next;
	}
        if( !defined($chrom_counts{$chrid}) )
        {
            if($haveGenome)
            {
                print STDERR "\nERROR: Target fragment ($chrid) not present in specified genome.\n";
                exit 1;
            }
            # if genome undefined use bedfile to define chromosomes and order
            $chrom_counts{$chrid} = 0;
            $chrom_num_target{$chrid} = 0;
            $num_chroms++;
            $corder{$num_chroms} = $chrid;
        }
        ++$chrom_num_target{$chrid};
	++$numTargets;
    }
    close( BEDFILE );
}

# set up stuff for read pileup analysis
my %pos_counts;
my %pos_ontarg;
my %pos_maxhit;
my $posbin;
my $num_reads = 0;
my $num_ontarg = 0;

# make initial read of mappings to get all hits, i.e. including off-target hits
my $xSam = $readtype == 1 ? "-S" : "";
open( MAPPINGS, "samtools view -F 4 $xSam \"$mappings\" |" ) || die "Cannot read mapped reads from $mappings.\n";
while( <MAPPINGS> )
{
    next if(/^@/);
    my ($rid,$flag,$chrid,$pos,$scr,$cig) = split;
    if( !defined($chrom_counts{$chrid}) )
    {
	if( $haveGenome )
	{
	    print STDERR "\nERROR: Target fragment ($chrid) not present in specified genome.\n";
	    exit 1;
	}
	# if genome undefined use abm to identify ALL chromosomes
	$chrom_counts{$chrid} = 0;
	$chrom_num_target{$chrid} = 0;
	$num_chroms++;
	$corder{$num_chroms} = $chrid;
    }
    ++$num_reads;
    ++$chrom_counts{$chrid};
    # record binned start position hit frequencies
    if( $doAnalysis )
    {
        # for reverse reads calculate the actual start alignment position
        # taking into account the pileup overlap (matching + deletion regions)
        if( $flag == 16 )
        {
            while( $cig =~ s/^(\d+)(.)// )
            {
		$pos += $1 if( $2 eq "M" || $2 eq "D" || $2 eq "X" || $2 eq "=" );
            }
            --$pos;	# adjust for 1-base
        }
        $posbin = int(($pos-1)/$binsize);
        ++$pos_counts{$chrid}[$posbin];
        $pos_maxhit{$chrid} = $posbin if( $posbin > $pos_maxhit{$chrid} );
    }
}
close( MAPPINGS );

open( OUTPUT, ">$outfile" ) || die "Cannot open tsv output file $outfile.\n";

# re-open for counting chromosome on-target hits
if( $haveTargets )
{
    open( MAPPINGS, "samtools view -F 4 -L \"$bedfile\" $xSam \"$mappings\" |" ) || die "Cannot read mapped reads from $mappings.\n";
    while( <MAPPINGS> )
    {
	next if(/^@/);
	my ($rid,$flag,$chrid,$pos,$scr,$cig) = split;
        # check if this was a bed target (safety check since already samtools filtered by bed file)
	if( $chrom_num_target{$chrid} )
	{
	    ++$num_ontarg;
	    ++$chrom_on_target{$chrid};
            if( $doAnalysis )
            {
                if( $flag == 16 )
                {
                    while( $cig =~ s/^(\d+)(.)// )
                    {
			$pos += $1 if( $2 eq "M" || $2 eq "D" || $2 eq "X" || $2 eq "=" );
                    }
                    --$pos;	# adjust for 1-base
                }
                $posbin = int(($pos-1)/$binsize);
                $pos_ontarg{$chrid}[$posbin] = 1;
            }
	}
    }
    close( MAPPINGS );

    print OUTPUT "Chromosome\tMapped_Reads\tOn_Target\n";
    for( my $i = 1; $i <= $num_chroms; $i++ )
    {
        my $k = $corder{$i};
        printf OUTPUT "%s\t%d\t%d\n", $k, $chrom_counts{$k}, $chrom_on_target{$k};
    }
}
else
{
    print OUTPUT "Chromosome\tMapped_Reads\n";
    for( my $i = 1; $i <= $num_chroms; $i++ )
    {
        my $k = $corder{$i};
        printf OUTPUT "%s\t%d\n", $k, $chrom_counts{$k};
    }
}
close( OUTPUT );
# remove files in preference to just having header line
unlink( $outfile ) if( $num_chroms == 0 );

if( $doAnalysis )
{
    open( FREQFILE, ">$freqfile" ) || die "Cannot write to file $freqfile.\n";
    print FREQFILE "Chromosome\tStart\tEnd\tCount\tOn_Target\n";
    my $rows_out = 0;
    for( my $i = 1; $i <= $num_chroms; $i++ )
    {
        my $k = $corder{$i};
	my ($lastpos, $color, $colorphase) = (0,0,0);
	for( my $pos = 0; $pos < $pos_maxhit{$k}; $lastpos = $pos, ++$pos )
	{
	    next if( $pos_counts{$k}[$pos] < $minpcnt );
	    if( $colorpeaks )
	    {
		$colorphase = 2-$colorphase if( $pos_counts{$k}[$lastpos] < $minpcnt );
		$color = $colorphase + ($haveTargets ? $pos_ontarg{$k}[$pos] : 0);
	    }
	    else
	    {
		$color = $haveTargets ? $pos_ontarg{$k}[$pos]+0 : 0;
		if( $spacepeaks && $pos_counts{$k}[$lastpos] < $minpcnt )
		{
		    print FREQFILE "$k\t$lastpos\t0\t0\n";
		}
	    }
	    my $start = 1 + $pos*$binsize;
	    my $end = $start + $binsize - 1;
	    printf FREQFILE "%s\t%d\t%d\t%d\t%d\n", $k, $start, $end, $pos_counts{$k}[$pos], $color;
	    ++$rows_out;
	}
    }
    close( FREQFILE );
    unlink( $freqfile ) if( $rows_out == 0 );
}

$num_ontarg = $num_reads if( !$haveTargets );
my $pcOntarg = $num_reads == 0 ? 0 : 100 * $num_ontarg / $num_reads;
if( $padstats )
{
    printf "Percent reads on padded target: %.2f%%\n", $pcOntarg;
}
elsif( $statsout )
{
    printf "Number of mapped reads:         %d\n", $num_reads;
    printf "Number of reads on target:      %d\n", $num_ontarg;
    printf "Percent reads on target:        %.2f%%\n", $pcOntarg;
}

