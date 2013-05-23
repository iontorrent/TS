#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Read tab-separate value file (tsv) and output as html table rows (to STDOUT).";
my $USAGE = "Usage:\n\t$CMD [options] <input table file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -d Assume first line starts with data (no column titles)
  -t Add the first line as table column Titles in the output html, unless -d specified
  -v Add IGV viewer link on rows";

my $haveheader = 1;
my $writeheader = 0;
my $addviewer = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-d') {$haveheader = 0;}
    elsif($opt eq '-t') {$writeheader = 1;}
    elsif($opt eq '-v') {$addviewer = 1;}
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
        if( $writeheader )
        {
            print "<tr> ";
            print " <th>View</th>" if( $addviewer );
            foreach (@fields)
            {
                printf " <th>%s</th>",$_;
            }
            print " </tr>\n";
        }
        next;
    }
    print "<tr> ";
    printf " <td><a class='igvTable' data-locus='%s:%s'>IGV</a></td>",$fields[0],$fields[1] if( $addviewer );
    foreach (@fields)
    {
        printf " <td>%s</td>",$_;
    }
    print " </tr>\n";
}
close( INFILE );

