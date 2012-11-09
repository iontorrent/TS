#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

use File::Basename;

# get current running script dir
use FindBin qw($Bin);

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create table and optional html files for SNP detection per chromosome from one or more list of variants.
Input and output files are tab-delimited text (tsv) files, with headings.";
my $USAGE = "Usage:\n\t$CMD [options] <output tsv file> <input tsv file> [<tsv2> ...]";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -l Log progress to STDERR
  -s Skip first column as non-data (e.g. IGV link)
  -v Add an extra column for identified variants, based on contents of the 'TaqMan Assay ID' fields
  -G <file> Genome file (.fasta.fai) for ordering output by chromosome
  -T <file> Name for HTML Table row summary file";

my $rowsumfile = "";
my $genomefile = "";
my $logprogress = 0;
my $cntvariants = 0;
my $skipfirstcol = 0;

my $hotspothead = "TaqMan Assay ID";

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-T') {$rowsumfile = shift;}
    elsif($opt eq '-l') {$logprogress = 1;}
    elsif($opt eq '-s') {$skipfirstcol = 1;}
    elsif($opt eq '-v') {$cntvariants = 1;}
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

my $haverowsum = ($rowsumfile ne "");

#--------- End command arg parsing ---------

my %chr_counts;
my %chr_vartps;
my @chr_order;
my $nchr = 0;
my $readLines = 0;
my $type;
my $vart = 0;
while( scalar @ARGV )
{
    my $filein = shift;
    open( INFILE, $filein ) || die "Cannot open variants file $filein.\n";
    while( <INFILE> )
    {
	chomp;
	next if( /^\s*$/ );
	my @fields = split(/\t/);
        shift(@fields) if( $skipfirstcol );
	my $c = $fields[0];
        # assume this indicates this file has header row
        if( $c eq "Chromosome" )
        {
            # identify TaqMan Assay ID row
            if( $cntvariants && $vart == 0 )
            {
                for( my $i = 1; $i < scalar(@fields); ++$i )
                {
                    if( $fields[$i] eq $hotspothead )
                    {
                        $vart = $i;
                        last;
                    }
                }
            }
            next;
        }
	$chr_order[$nchr++] = $c if( ++$chr_counts{$c} == 1 );
        $type = $fields[4];
        $type = "INDEL" if( $type eq "DEL" || $type eq "INS" );
        $type = "SNP" if( $type eq "MNP" );
	++$chr_vartps{$c}{$fields[5]." ".$type};
        if( $vart > 0 )
        {
            my $id = $fields[$vart];
            ++$chr_vartps{$c}{'hotspot'} unless( $id eq "---" || $id eq "" || $id eq "N/A" );
        }
	++$readLines;
    }
    close( INFILE );
}
if( $cntvariants && $vart == 0 )
{
    print STDERR "WARNING: $CMD: '$hotspothead' field not located: counts will default to 0.\n";
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
        print STDERR "Warning: Could not open genome file $genomefile\n";
    }
}
open( OUTFILE, ">$outfile" ) || die "Cannot open output file $outfile.\n";
if( $haverowsum )
{
    open( ROWSUM, ">$rowsumfile" ) || die "Cannot open output file $rowsumfile.\n";
}
print OUTFILE "Chromosome\tVariants\tHet SNPs\tHom SNPs\tHet INDELs\tHom INDELs";
print OUTFILE ($vart > 0) ? "Hotspots\n" : "\n";
my $writeLines = 0;
for( my $i = 0; $i < $nchr; ++$i )
{
    my $c = $chr_order[$i];
    next if( !defined($chr_counts{$c}) );
    $writeLines += $chr_counts{$c};
    printf OUTFILE "%s\t%d\t%d\t%d\t%d\t%d", $c, $chr_counts{$c}+0,
	$chr_vartps{$c}{'Het SNP'}, $chr_vartps{$c}{'Hom SNP'}, $chr_vartps{$c}{'Het INDEL'}, $chr_vartps{$c}{'Hom INDEL'};
    if( $vart > 0)
    {
        printf OUTFILE "\t%d\n", $chr_vartps{$c}{'hotspot'};
    }
    else
    {
        printf OUTFILE "\n";
    }
    next if( !$haverowsum );
    printf ROWSUM "<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td>", $c, $chr_counts{$c}+0,
        $chr_vartps{$c}{'Het SNP'}, $chr_vartps{$c}{'Hom SNP'}, $chr_vartps{$c}{'Het INDEL'}, $chr_vartps{$c}{'Hom INDEL'};
    if( $vart > 0 )
    {
        printf ROWSUM "<td>%d</td></tr>\n", $chr_vartps{$c}{'hotspot'};
    }
    else
    {
        printf ROWSUM "</tr>\n";
    }
}
close( ROWSUM ) if( $haverowsum );
close( OUTFILE );
if( $logprogress )
{
    print STDERR " - $writeLines variants reported of $readLines variant lines read.\n";
}
