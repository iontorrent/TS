#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

use File::Basename;

# get current running script dir
use FindBin qw($Bin);

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Merge and sort a list of variant tables and optionally add experted HotSpot IDs.
Input and output files are tab-delimited text (tsv) files, with headings (taken from first input).";
my $USAGE = "Usage:\n\t$CMD [options] <output tsv file> <input tsv file> [<tsv2> ...]";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -l Log progress to STDERR
  -v Add IGV viewer link on rows
  -G <file> Genome file (.fasta.fai) for ordering output by chromosome
  -B <file> Name for BED file specifying HotSpot locations.";

my $hotspotfile = "";
my $genomefile = "";
my $logprogress = 0;
my $addviewer = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-B') {$hotspotfile = shift;}
    elsif($opt eq '-l') {$logprogress = 1;}
    elsif($opt eq '-v') {$addviewer = 1;}
    elsif($opt eq '-G') {$genomefile = shift;}
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
elsif( scalar @ARGV < 2 )
{
    print STDERR "$CMD: Invalid number of arguments.";
    print STDERR "$USAGE\n";
    exit 1;
}

my $outfile = shift;

my $havehotspots = ($hotspotfile ne "");

#--------- End command arg parsing ---------

my %chr_counts;
my %chr_fields;
my @chr_order;
my $nchr = 0;
my $readLines = 0;
my $headerline="";
while( scalar @ARGV )
{
    my $filein = shift;
    open( INFILE, $filein ) || die "$CMD: Cannot open variants file $filein.\n";
    my $firstline = 1;
    while( <INFILE> )
    {
	chomp;
	next if( /^\s*$/ );
        if( $firstline )
        {
            $headerline = $_ if( $headerline eq "" );
            $firstline = 0;
            next;
        }
	@_ = split(/\t/);
	my $c = shift @_;
        my @d = @_;
	$chr_order[$nchr++] = $c if( ++$chr_counts{$c} == 1 );
        push( @{$chr_fields{$c}}, \@d );
	++$readLines;
    }
    close( INFILE );
}

if( $headerline eq "" )
{
    print STDERR "\nERROR: $CMD: No fields header line in input TSV file(s).\n";
    exit 1;
}
# re-order by genome file
if( $genomefile ne "" )
{
    $nchr = 0;
    if( open( GENOME, "$genomefile" ) )
    {
        while( <GENOME> )
        {
            chomp;
            next if( /^\s*$/ );
            @_ = split;
	    $chr_order[$nchr++] = $_[0];
        }
        close( GENOME );
    }
    else
    {
        print STDERR "WARNING: $CMD: Could not open genome file $genomefile\n";
    }
}

open( OUTFILE, ">$outfile" ) || die "$CMD: Cannot open output file $outfile.\n";

$headerline = "View\t" . $headerline if( $addviewer );

my %hotspot_starts;
my %hotspot_ends;
my %hotspot_ids;
my %hotspot_infos;
my $vart = 0; # position must be first field (after chromosome)
if( $havehotspots )
{
    # check for VarType field
    my @fields = split(/\t/,$headerline);
    my @j = 1;
    for( my $i = 1; $i < scalar(@fields); ++$i )
    {
	if( $fields[$i] =~ m/^Type$/i )
        {
            $vart = $i-1;
            last;
        }
    }
    print STDERR "WARNING: $CMD: No VarType field found for identifying DEL for position correction.\n" if( $vart == 0 );
    printf OUTFILE "%s\tHotSpot ID\n", $headerline;
    # read in ORDERED BED file into start/stop locations and hotspot ID per chromosome
    my $numTracks = 0;
    my $numTargets = 0;
    open( BEDFILE, "$hotspotfile" ) || die "$CMD: Cannot open output file $hotspotfile.\n";
    while( <BEDFILE> )
    {
        chomp;
        @fields = split(/\t/);
        my $chrid = $fields[0];
        next if( $chrid !~ /\S/ );
        if( $chrid =~ /^track / )
        {
            ++$numTracks;
            if( $numTracks > 1 )
            {
                print STDERR "\nWARNING: $CMD: Bed file has multiple tracks. Ingoring tracks after the first.\n";
                last;
            }
            if( $numTargets > 0 )
            {
                print STDERR "\nERROR: $CMD: Bed file incorrectly formatted: Contains targets before first track statement.\n";
                exit 1;
            }
            next;
        }
        ++$numTargets;
	push( @{$hotspot_starts{$chrid}}, $fields[1]+1 );
        #push( @{$hotspot_ends{$chrid}}, $fields[2] );
        push( @{$hotspot_ids{$chrid}}, $fields[3] );
	if(scalar(@fields) > 6) {
		push( @{$hotspot_infos{$chrid}}, $fields[6] );
	}
    }
    close( BEDFILE );
    print STDERR " - $numTargets HotSpot sites read from bed file.\n" if( $logprogress );
}
else
{
    printf OUTFILE "%s\n", $headerline;
}

my $writeLines = 0;
for( my $i = 0; $i < $nchr; ++$i )
{
    my $c = $chr_order[$i];
    next if( !defined($chr_counts{$c}) );
    $writeLines += $chr_counts{$c};
    @{$chr_fields{$c}} = sort { $a->[0] <=> $b->[0] || $a->[1] <=> $b->[1] } @{$chr_fields{$c}};
    my $num_hotspots = $havehotspots ? scalar(@{$hotspot_starts{$c}})+0 : 0;
    foreach (@{$chr_fields{$c}})
    {
        my @fields = @{$_};
        my $spos = $fields[0];
        if( !$havehotspots )
        {
            printf OUTFILE "<a class='igvTable' data-locus='%s:%s'>IGV</a>\t",$c,$spos if( $addviewer );
            printf OUTFILE "%s\t%s\n",$c,join("\t",@fields);
            next;
        }
        # list ALL hotspot IDs that START at the found variant location, adjusting for deletions
        
	++$spos if( $fields[$vart] eq "DEL" );
        my $hotspot = "---";
        if( $num_hotspots > 0 )
        {
            my $hs = floor_bstart( $spos, \@{$hotspot_starts{$c}} );
            if( $hs >= 0 )
            {
                $hotspot = "";
                while( $hs < $num_hotspots )
                {
                    last if( $hotspot_starts{$c}[$hs] != $spos );
		    if(($hotspot_infos ne " ") & ($hotspot_infos{$c} ne " ") & ($hotspot_infos{$c}[$hs] ne " ")){
			my $hotSpotInfo = $hotspot_infos{$c}[$hs];
			if(exact_match($hotSpotInfo, @fields)) {
				$hotspot .= $hotspot_ids{$c}[$hs] .";";
		    	}
		    }
		    else {
                    		print STDERR "WARNING: No HotSpotAlleles information found for the hotspot id:" .$hotspot_ids{$c}[$hs]. ". So matching will be done using Start position only.";
				$hotspot .= $hotspot_ids{$c}[$hs].";";
		    }
		    $hs++;
                }
            }
        }
        $spos = $fields[0];
	printf OUTFILE "<a class='igvTable' data-locus='%s:%s'>IGV</a>\t",$c,$spos if( $addviewer );
        #$fields[0] = $spos;
        printf OUTFILE "%s\t%s\t%s\n", $c, join("\t",@fields), $hotspot;
    }
}
close( OUTFILE );
print STDERR " - $writeLines variants reported of $readLines variant lines read.\n" if( $logprogress );

# ----------------END-------------------

sub exact_match
{
    #return true if the types and alleles match exactly
    #1st argument is the info field for the hotspot
    #2nd argument is variant record tokens
    my ($hotspotInfo, @variantFields) = @_;
    my $refAllele = "NONE";
    my $obsAllele = "NONE";
    my @hotspotAlleles = split(/;/, $hotspotInfo);
    my @refAlleleInfo = (split(/=/, $hotspotAlleles[0]));
    if($refAlleleInfo[0] eq "REF") {
    	$refAllele = $refAlleleInfo[1];
    }
    my @obsAlleleInfo = (split(/=/, $hotspotAlleles[1]));
    if($obsAlleleInfo[0] eq "OBS") {
 	$obsAllele = $obsAlleleInfo[1];
    }
    if($refAllele eq "NONE" || $obsAllele eq "NONE") {
	return 1; #return true 
    }
    
    
    my $hotspotType = "";
    if((length($refAllele) == 1) & (length($obsAllele) == 1)) {
	$hotspotType = "SNP";
    }
    elsif(length($refAllele) > length($obsAllele)) {
    	$hotspotType = "DEL";
    }
    elsif(length($refAllele) < length($obsAllele)) {
	$hotspotType = "INS";
    }
    my $variantType = $variantFields[$vart];
    
    $obsAllele =~ s/U|R|Y|M|K|W|S|B|D|H|V|N/\./g;
    $refAllele =~ s/U|R|Y|M|K|W|S|B|D|H|V|N/\./g;
    
    if (($variantType eq "SNP") && ($hotspotType eq "SNP") ){
	my $variantAltAllele =  $variantFields[6];
	if($variantAltAllele =~ m/$obsAllele/) {
		return 1;
	}
	else {
		return 0;
	}
    }
    if($variantType eq "DEL") {
	my $variantAllele = substr($variantFields[5], 1);
	if($variantAllele =~ m/$refAllele/) {
		return 1;
	}
	else {
		return 0;
	}
    }
    if($variantType eq "INS") {
	my $variantAllele = substr($variantFields[6], 1);
	if($variantAllele =~ m/$obsAllele/) {
		return 1;
	}
	else {
		return 0;
	}
    }
    return 0;
}

sub floor_bstart
{
    # return the first index for which pos[index] == val, or -1
    # assumes 2nd arg is a pointer to a non-empty array of assending-sorted values
    my ($val,$lows) = @_;
    # return -1 if value is less than the first value or greater than last
    my ($l,$u) = (0, scalar(@{$lows})-1);
    if( $lows->[$l] > $val || $lows->[$u] < $val ) { return -1; }
    my $i = 0;
    while(1)
    {
        $i = int( ($l + $u)/2 );
        last if( $val == $lows->[$i] );
        return -1 if( $l == $u );
        if( $val < $lows->[$i] ) { $u = $i; }
        else { $l = $i+1; }
    }
    # walk back for lowest index match
    while( $i )
    {
        last if( $lows->[$i-1] < $val );
        --$i;
    }
    return $i;
}

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

