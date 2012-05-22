#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Calls from allele coverage table file and write a barcode string to STDOUT";
my $USAGE = "Usage:\n\t$CMD [options] <input table file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -d Assume first line starts with data (no column titles).
  -r <N> Minimum total gender sampling Reads required to make gender call (with -R). Default: 30.
  -y <N> Minimum percentage of gender sampling reads that must be the male target (Y) to make a male gender call. Default: 20(%).
  -R <file> Read stats file. If provided, male/female targets read counts are extracted and used to assign gender call.
  -C <str> Name or number of field that contains the individual allele calls. Default 'Call' or the 5th field.";

my $haveheader = 1;
my $readstats = "";
my $callfield = "Call";

# cutoffs for calling gender
my $minReads = 30;
my $minYPC = 20;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-d') {$haveheader = 0;}
    elsif($opt eq '-r') {$minReads = shift;}
    elsif($opt eq '-y') {$minYPC = shift;}
    elsif($opt eq '-R') {$readstats = shift;}
    elsif($opt eq '-C') {$callfield = shift;}
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

my $infile = shift;

#--------- End command arg parsing ---------

my $haplo = "";

my $callFieldNum = 5;
my $firstline = $haveheader;
open( INFILE, $infile ) || die "Cannot open table file $infile.\n";
while( <INFILE> )
{
    chomp;
    next if( /^\s*$/ );
    # assume this indicates this file has header row
    my @fields= split(/\t/);
    if( $firstline )
    {
        $firstline = 0;
        $fieldNum = 0;
        foreach (@fields)
        {
            ++$fieldNum;
            if( $_ eq $callfield || $fieldNum == $callfield )
            {
                 $callFieldNum = $fieldNum;
                 last;
            }
        }
        next;
    }
    $haplo .= $fields[$fieldNum-1];
}
close( INFILE );

my $gender="";
if( $readstats ne "" )
{
    $gender="?-";
    if( open( READSTATS, $readstats ) )
    {
        my $nm = 0;
        my $nf = 0;
        while( <READSTATS> )
        {
            chomp;
            if( m/Male target reads:/ )
            {
                s/Male target reads: //;
                $nm = $_+0;
            }
            elsif( m/Female target reads:/ )
            {
                s/Female target reads: //;
                $nf = $_+0;
            }
        }
        close( READSTATS );
        if( $nm+$nf >= $minReads )
        {
            $pcMale = 100 * $nm / ($nm + $nf);
            $gender = ($pcMale >= $minYPC) ? "M-" : "F-";
        }
    }
    else
    {
        print STDERR "Could not open read stats file: $readstats\n";
    }
}

# if half of the length of the id string is unknown (?) then replace with "N/A" or "<M/F>-?" => no useful ID
my $clen = length($haplo);
my $cqry = $haplo =~ tr/?//;
$haplo = $gender.$haplo;
$haplo = ($gender eq "?-" || $gender eq "") ? "N/A" : $gender."?" if( $cqry >= $clen/2 );
print $haplo;

