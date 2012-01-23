#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Remove duplicate reads from a Ion PGM read mapping file.
The input file can be in BAM or SAM format and does not have to be sorted.
Output (to STDOUT) is SAM format with reads in order encountered in input.";
my $USAGE = "Usage:\n\t$CMD [options] <BAM | SAM file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -s Reads file is in SAM format (default is BAM)
  -l Print log of duplicate removal statistics to STDERR
  -k Keep unmapped reads and/or those with unexpected flag values in the output.
  -u Filter non-unique mappings (MAPQ = 0) prior to duplicate checking.
  -D <file> Output duplicate Distribution to <file>. (Default: no output)
  -L <N> Difference between 'duplicate' reads must be less than N (by binning). Default: 0
      0 => no length considered; equal positions on the same chromosome => duplicates
      1 => reads of the length and equal positions on the same chromosome => duplicates";

my $readtype = 0;
my $maxlendiff = 0;
my $report = 0;
my $filter_nonunque = 0;
my $keep_unmap = 0;
my $dup_dist_file = "";

my $maxdistplot = 50;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-L') {$maxlendiff = shift;}
    elsif($opt eq '-D') {$dup_dist_file = shift}
    elsif($opt eq '-s') {$readtype = 1}
    elsif($opt eq '-l') {$report = 1}
    elsif($opt eq '-k') {$keep_unmap = 1}
    elsif($opt eq '-u') {$filter_nonunque = 1}
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

my $mappings = shift(@ARGV);

#--------- End command arg parsing ---------

# $readtype: 0=BAM (default), 1=SAM - may have others later
if( $readtype == 1 )
{
    open( MAPPINGS, "$mappings" ) || die "Cannot read mapped reads from $mappings.\n";
}
else
{
    open( MAPPINGS, "samtools view -h \"$mappings\" |" ) || die "Cannot read mapped reads from $mappings.\n";
}

my %freads;
my %rreads;
my $num_reads = 0;
my $num_maps  = 0;
my $num_unmap = 0;
my $num_unpro = 0;
my $num_nonun = 0;
my $num_fread = 0;
my $num_fdups = 0;
my $num_rread = 0;
my $num_rdups = 0;

my $rlen = 0;

while( <MAPPINGS> )
{
    if(/^@/)
    {
	# keep header info for conversion to good bam
	print;
	next;
    }
    my ($rid,$flag,$chr,$pos,$scr,$cig) = split;
    ++$num_reads;
    if( $chr eq "*" || $flag == 4 )
    {
	++$num_unmap;
        print if( $keep_unmap );
        next;
    }
    if( $flag != 0 && $flag != 16 )
    {
	++$num_unpro;
        print if( $keep_unmap );
        next;
    }
    if( $scr == 0 )
    {
	++$num_nonun;
	next if( $filter_nonunque );
    }
    ++$num_maps;
    # calculate true sequence overlap length
    my $slen = 0;
    my $cigar = $cig;
    while( $cig =~ s/^(\d+)(.)// )
    {
	# this is probably quicker than using regex, after first copying $1 and $2 so they are not corrupted
	$slen += $1 if( $2 eq "M" || $2 eq "D" || $2 eq "X" || $2 eq "=" );
    }
    # correct starting location for reverse reads
    $pos += $slen-1 if( $flag == 16 );
    # for binning relative to seq size
    $rlen = int(($slen-1)/$maxlendiff) if( $maxlendiff );
    if( $flag == 0 )
    {
	if( ++$freads{$chr}{$pos}{$rlen} == 1 )
	{
	    ++$num_fread;
	    print;
	}
	else
	{
	    ++$num_fdups;
	}
    }
    else
    {
	if( ++$rreads{$chr}{$pos}{$rlen} == 1 )
	{
	    ++$num_rread;
	    print;
	}
	else
	{
	    ++$num_rdups;
	}
    }
}
close( MAPPINGS );

if( $report )
{
    printf STDERR "Total reads:      %9d\n",$num_reads;
    printf STDERR "  Unmapped reads: %9d%s\n",$num_unmap, ($keep_unmap ? "  (retained)" : "");
    printf STDERR "  Unknown flag:   %9d%s\n",$num_unpro, ($keep_unmap ? "  (retained)" : "");
    printf STDERR "  Non-unique:     %9d%s\n",$num_nonun, ($filter_nonunque ? "  (pre-filtered)" : "");
    printf STDERR "Mapped reads:     %9d\n",$num_maps;
    printf STDERR "  Forward reads:  %9d\n",$num_fread+$num_fdups;
    printf STDERR "     Duplicates:  %9d\n",$num_fdups;
    printf STDERR "  Reverse reads:  %9d\n",$num_rread+$num_rdups;
    printf STDERR "     Duplicates:  %9d\n",$num_rdups;
    printf STDERR "Retained starts:  %9d\n",$num_fread+$num_rread;
    printf STDERR "  Forward reads:  %9d\n",$num_fread;
    printf STDERR "  Reverse reads:  %9d\n",$num_rread;
}

if( $dup_dist_file ne "" )
{
    open( DUPDIST, ">$dup_dist_file" ) || die "Cannot open output file $dup_dist_file.\n";
    my @fdup_dist;
    my @rdup_dist;
    while( my ($chr,$h1) = each %freads )
    {
	while( my ($pos,$h2) = each %$h1 )
	{
	    while( my ($len,$cnt) = each %$h2 )
	    {
		++$fdup_dist[$cnt];
	    }
	}
    }
    while( my ($chr,$h1) = each %rreads )
    {
	while( my ($pos,$h2) = each %$h1 )
	{
	    while( my ($len,$cnt) = each %$h2 )
	    {
		++$rdup_dist[$cnt];
	    }
	}
    }
    print DUPDIST "Duplication\tForward\tReverse\n";
    my $ds = scalar @fdup_dist;
    my $fs = scalar @rdup_dist;
    $ds = $fs if( $fs > $ds );
    if( $ds > $maxdistplot )
    {
	$fs = 1+$maxdistplot;
	for( my $d = 1+$fs; $d <= $ds; ++$d )
	{
	    $fdup_dist[$fs] += $fdup_dist[$d];
	    $rdup_dist[$fs] += $rdup_dist[$d];
	}
	$ds = $maxdistplot;
	$maxdistplot = -1;
    }
    for( my $d = 1; $d <= $ds; ++$d )
    {
	printf DUPDIST "%d\t%d\t%d\n",$d,$fdup_dist[$d],$rdup_dist[$d];
    }
    if( $maxdistplot < 0 )
    {
	printf DUPDIST ">%d\t%d\t%d\n",$ds,$fdup_dist[$fs],$rdup_dist[$fs];
    }
    close( DUPDIST );
    print STDERR "Duplicate distribution output to\n  $dup_dist_file\n";
}
