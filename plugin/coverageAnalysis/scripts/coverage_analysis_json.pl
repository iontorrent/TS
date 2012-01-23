#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

use File::Basename;

# get current running script dir
use FindBin qw($Bin);

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Write the contents of a given data file to json format, with options.
Input file assumed to have data by line and separates by first ':' to form name/value pair.
White space in name string is coverted to underscores.";
my $USAGE = "Usage:\n\t$CMD [options] <data file> <json file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -a Append to existing jason file. Default: Create stand-alone, with surrounding {}
  -B <name> Block name. Default: none - no name or {} node wrapper.
  -I <N> Set the initial indentation level. Default 2.";

my $appendJson = 0;
my $indentLvl = 2;
my $blockname = "";

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-a') {$appendJson = 1;}
    elsif($opt eq '-I') {$indentLvl = shift;}
    elsif($opt eq '-B') {$blockname = shift;}
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

$indentLvl = 0 if( 0+$indentLvl < 0 );

my $datafile = shift;
my $jsonfile = shift;

#--------- End command arg parsing ---------

unless( open( DATAFILE, "$datafile" ) )
{
    print STDERR "Could not locate data file $datafile\n";
    exit 1;
}
if( $appendJson )
{
    unless( open( JSONFILE, ">>$jsonfile" ) ) 
    {
	close( DATAFILE );
	print STDERR "Could not append to json file $jsonfile\n";
	exit 1;
    }
}
else
{
    unless( open( JSONFILE, ">$jsonfile" ) ) 
    {
	close( DATAFILE );
	print STDERR "Could not write to json file $jsonfile\n";
	exit 1;
    }
    print JSONFILE "{\n";
}

if( $blockname ne "" )
{
    print JSONFILE (" " x $indentLvl)."\"$blockname\" : {\n";
    $indentLvl += 2;
}

my $linenum = 0;
while( <DATAFILE> )
{
    print JSONFILE ",\n" if( ++$linenum != 1 );
    print JSONFILE jsonNameValue($_);
}
close( DATAFILE );

if( $blockname ne "" )
{
    $indentLvl -= 2;
    print JSONFILE "\n".(" " x $indentLvl)."}";
}
close( JSONFILE );

# ---------- END ----------

sub jsonNameValue
{
    my ($n,$v) = split(/:/,$_[0]);
    $n =~ s/^\s+|\s+$//g;
    return "" if( $n eq "" );
    $v =~ s/^\s+|\s+$//g;
    $v = "N/A" if( $v eq "" );
    #$n =~ s/\s/_/g;
    return (" " x $indentLvl)."\"$n\" : \"$v\"";
}
