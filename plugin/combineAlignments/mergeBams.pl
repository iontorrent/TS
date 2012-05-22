#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

use JSON;
use LWP;
use File::Copy 'move';

# get current running script dir
use FindBin qw($Bin);

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Create a single BAM file from a list of BAM files.
Files are pre-checked for compatibility then combined using 'samtools merge'.
Summary statistics and any warning messages are output to STDOUT.";
my $USAGE = "Usage:\n\t$CMD [options] <output file name> <file1> [<file2> ...]";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -f Input files are lists of bam files rather than the files themselves.
  -i Create bam index files.
  -t Do not perform merge; Test only mode.
  -x Add HTML tags around warning messages.";

my $bamlists = 0;
my $bamindex = 0;
my $htmlify = 0;
my $nomerge = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-f') {$bamlists = 1;}
    elsif($opt eq '-i') {$bamindex = 1;}
    elsif($opt eq '-t') {$nomerge = 1;}
    elsif($opt eq '-x') {$htmlify = 1;}
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
    print STDERR "$CMD: Invalid number of arguments.\n";
    print STDERR "$USAGE\n";
    exit 1;
}

my $outputbam = shift;

#--------- End command arg parsing ---------

my @bamlist;
if( $bamlists )
{
    while(<>)
    {
        chomp;
        push @bamlist, $_;
    }
}
else
{
    while( scalar(@ARGV) > 0 )
    {
        push @bamlist, shift;
    }
}

my $numBams = scalar(@bamlist);
die "ERROR: No BAM files given for merging." if( $numBams == 0 );

my @warnings;
my $tmapCheck = '';
my $tmapWarn = 0;
my $flowCheck = '';
my $flowWarn = 0;

for( my $i = 0; $i < $numBams; ++$i )
{
    my $bamfile = $bamlist[$i];
    print STDERR "Analyzing $bamfile\n";
    my $isTmap = 0;
    my $isFlow = 0;
    open( BAMFILE, "samtools view -H \"$bamfile\" |" ) || die "ERROR: Could not open bam file $bamfile";
    while( <BAMFILE> )
    {
        chomp;
        @_ = split('\t');
        if( $_[0] eq '@RG' )
        {
            for( my $i = 1; $i < scalar(@_); ++$i )
            {
                 if( $_[$i] =~ m/^FO:/ )
                 {
                     $isFlow = 1;
                     $flowCheck = $_[$i] if( $flowCheck eq '' );
                     $flowWarn = 1 if( $flowCheck ne $_[$i] );
                     last;
                 }
            }
        }
        if( $_[0] eq '@PG' && $_[1] eq 'ID:tmap' )
        {
            $isTmap = 1;
            $tmapCheck = $_[2] if( $tmapCheck eq '' );
            $tmapWarn = 1 if( $tmapCheck ne $_[2] );
        }
        last if( $_[0] !~ m/^@/ );
    }
    close( BAMFILE );
    $bamfile =~ s/.*\///;
    push @warnings, "$bamfile was not created using TMAP." if( $isTmap == 0 );
    push @warnings, "$bamfile does not contain flowspace information." if( $isFlow == 0 );
}

push @warnings, "Not all alignments were generated using the same version of TMAP." if( $tmapWarn );
push @warnings, "Not all reads were generated using the same nucleotide flow sequence.$brk\n" if( $flowWarn );

my $brk = $htmlify ? '<br/>' : '';
my $bul = $htmlify ? '&#149;' : '*';
my $numWarn = scalar(@warnings);
if( $numWarn > 0 )
{
    print "<h4>" if( $htmlify );
    print "*** WARNINGS ***\n";
    print "</h4><h4 style=\"color:red\">\n" if( $htmlify );
    if( $numWarn > 0 )
    {
    }
    for( my $w = 0; $w < $numWarn; ++$w )
    {
        print "$bul $warnings[$w]$brk\n";
    }
    print "</h4>\n" if( $htmlify );
}

exit 0 if( $nomerge );

print STDERR "Merging bam files...\n";
#my $bamfilelist = join(' ',@bamlist);
#my $mergeCmd = "samtools merge -f $outputbam.tmp";
my $bamfilelist = 'I='.join(' I=',@bamlist);
my $mergeCmd = "java -jar $Bin/MergeSamFiles.jar O=$outputbam.tmp USE_THREADING=true".($bamindex ? " CREATE_INDEX=true" : "");
if( system("$mergeCmd $bamfilelist 2> mergeBams.log") )
{
    print STDERR "$CMD: File merge command failed:\n\$ $mergeCmd $bamfilelist\n";
    print STDERR "Check mergeBam.log for more information.\n";
    exit 1;
}
move "$outputbam.tmp" , $outputbam;

if( $bamindex )
{
#    print STDERR "Indexing $outputbam.tmp...\n";
#    system("samtools index $outputbam.tmp");
    move "$outputbam.tmp.bai" , "$outputbam.bai";
}

print STDERR "Analyzing $outputbam...\n";
my $spc = $htmlify ? '&nbsp;&nbsp;&nbsp;' : '';
open( FLAGSTAT, "samtools flagstat \"$outputbam\" |" ) || die "ERROR: Could not open bam file $outputbam";
print "<h4>\n" if( $htmlify );
my $fsline = 0;
while( <FLAGSTAT> )
{
    @_ = split;
    ++$fsline;
    print "Total reads:  $_[0]$spc\n" if( $fsline == 1 );
    print "Mapped reads: $_[0]$brk\n" if( $fsline == 3 );
}
close( FLAGSTAT );
print "</h4>\n" if( $htmlify );

