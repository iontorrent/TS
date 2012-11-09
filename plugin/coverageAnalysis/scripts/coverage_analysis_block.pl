#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

use File::Basename;

# get current running script dir
use FindBin qw($Bin);

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Format the output from a double_coverage_analysis run to an html page.
The two required arguments are the path to the result files and original mapped reads file.
The results directory is relative to the top level directory (specified by -D option) for HMTL.";
my $USAGE = "Usage:\n\t$CMD [options] <results directory> <data link file name>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -O <file> Output file name (relative to output directory). Should have .html extension. Default: block.html
  -D <dirpath> Path to Directory where html page is written. Default: '' (=> use path given by Output file name)
  -S <file> Input Statistics file name. Default: '-' (no summary file)
  -A <file> Auxillary help text file defining fly-over help for HTML titles. Default: <script dir>/help_tags.txt
  -p <title> Primary table header text. Default 'All Reads'.
  -t <title> Secondary title for report. Default: ''";

my $outfile = "block.html";
my $workdir=".";
my $statsfile = "";
my $plotsize = 400;
my $title = "";
my $helpfile ="";
my $thead1 = "All Reads";

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-O') {$outfile = shift;}
    elsif($opt eq '-D') {$workdir = shift;}
    elsif($opt eq '-S') {$statsfile = shift;}
    elsif($opt eq '-A') {$helpfile = shift;}
    elsif($opt eq '-p') {$thead1 = shift;}
    elsif($opt eq '-t') {$title = shift;}
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

my $results1 = shift;
my $bamfile1 = shift;

$statsfile = "" if( $statsfile eq "-" );

# extract root name for output files from bam file names
my($runid1,$folder,$ext) = fileparse($bamfile1, qr/\.[^.]*$/);

#--------- End command arg parsing ---------

my $workdir1 = "$workdir/$results1";

# check data folders
die "No output directory found at $workdir" unless( -d "$workdir" );
die "No results directory found at $workdir1" unless( -d "$workdir1" );

my %helptext;
loadHelpText( "$helpfile" );

$outfile = "$workdir/$outfile" if( $workdir ne "" );
open( OUTFILE, ">$outfile" ) || die "Cannot open output file $outfile.\n";

print OUTFILE "<html><body>\n";
print OUTFILE "<div style=\"width:760px;margin-left:auto;margin-right:auto;\">\n";
#print OUTFILE "<h1><center>Coverage Analysis Report</center></h1>\n";
if( $title ne "" )
{
    # add simple formatting if no html tag indicated
    $title = "<h3><center>$title</center></h3>" if( $title !~ /^\s*</ );
    print OUTFILE "$title\n";
}
# html has compromises so as to appear almost identical on Firefox vs. IE8
print OUTFILE "<style type=\"text/css\">\n";
print OUTFILE "  table {width:100% !important;border-collapse:collapse;margin:0;table-layout:fixed}\n";
print OUTFILE "  th,td {font-family:\"Lucida Sans Unicode\",\"Lucida Grande\",Sans-Serif;font-size:14px;line-height:1.2em;font-weight:normal}\n";
print OUTFILE "  th,td {border:0px;padding:0px;text-align:center}\n";
print OUTFILE "  td {padding-top:5px;padding-bottom:5px}\n";
print OUTFILE "  td.inleft  {width:75% !important;border-width:0;text-align:left;padding:2px;padding-left:50px}\n";
print OUTFILE "  td.inright {width:25% !important;border-width:0;text-align:right;padding:2px;padding-right:10px}\n";
print OUTFILE "  img.frm {display:block;margin-left:auto;margin-right:auto;margin-top:10px;margin-bottom:10px;width:400;height:200;border-width:0;cursor:help}\n";
print OUTFILE "  .thelp {cursor:help}\n";
print OUTFILE "</style>\n";

print OUTFILE "<center><table><tr>\n";
displayResults( "covoverview.png", "covoverview.xls", "Coverage Overview", 1, "height:100px" );
my @keylist = ( "Number of mapped reads", "Percent reads on target", "Percent base reads on target" );
print OUTFILE "<td>";
print OUTFILE subTable( "$workdir1/$statsfile", \@keylist );
print OUTFILE "</td></tr><tr>\n";
displayResults( "coverage.png", "coverage.xls", "Target Coverage" );
@keylist = ( "Average base coverage depth", "Uniformity of coverage",
    "Target coverage at 1x", "Target coverage at 20x", "Target coverage at 100x", "Target coverage at 500x" );
print OUTFILE "<td>";
print OUTFILE subTable( "$workdir1/$statsfile", \@keylist );
print OUTFILE "</td></tr></table></center>\n";

print OUTFILE "</div></body></html>\n";
close( OUTFILE );

#-------------------------END-------------------------

sub commify
{
    (my $num = $_[0]) =~ s/\G(\d{1,3})(?=(?:\d\d\d)+(?:\.|$))/$1,/g;
    return $num;
}

sub subTable
{
    my @statlist;
    unless( open( STATFILE, "$_[0]" ) )
    {
        print STDERR "Could not locate text file $_[0]\n";
        return "Data unavailable";
    }
    while( <STATFILE> )
    {
        push( @statlist, $_ );
    }
    close( STATFILE );
    my @keylist = @{$_[1]};
    my $htmlText = "<table>\n";
    foreach $keystr (@keylist)
    {
        my $foundKey = 0;
        foreach( @statlist )
        {
            my ($n,$v) = split(/:/);
            if( $n eq $keystr )
            {
                $v =~ s/^\s*//;
                my $nf = ($v =~ /^(\d*)(\.?.*)/) ? commify($1).$2 : $v;
                $n = getHelp($n,1);
                $htmlText .= "<tr><td class=\"inleft\">$n</td>";
                $htmlText .= ($v ne "") ? " <td class=\"inright\">$nf</td></tr>\n" : "</td>\n";
                ++$foundKey;
                last;
            }
        }
        if( $foundKey == 0 )
        {
            $htmlText .= "<td>N/A</td>";
            print STDERR "No value found for statistic '$keystr'\n";
        }
    }
    return $htmlText."</table>";
}

sub displayResults
{
    my ($pic,$tsv,$alt,$skip,$style) = ($_[0],$_[1],$_[2],$_[3],$_[4]);
    # if skip is defined (e.g. 1) do not output anything if the first data file is missing
    # i.e. it is expected for this set of data to be missing and therefore skipped
    if( $skip != 0 )
    {
	return unless( -e "$workdir/$results1/$runid1.$tsv" );
    }
    my $desc = getHelp($alt);
    writeLinksToFiles( "$results1/$runid1.$pic", "$results1/$runid1.$tsv", $alt, $desc, $style );
}

sub loadHelpText
{
    my $hfile = $_[0];
    $hfile = "$Bin/help_tags.txt" if( $hfile eq "" );
    unless( open( HELPFILE, $hfile ) )
    {
	print STDERR "Warning: no help text file found at $hfile\n";
	return;
    }
    my $title = "";
    my $text = "";
    while( <HELPFILE> )
    {
	chomp;
	next if( ! /\S/ );
	if( s/^@// )
	{
	    $helptext{$title} = $text if( $title ne "" );
	    $title = $_;
	    $text = "";
	}
	else
	{
	    $text .= "$_ ";
	}
    }
    $helptext{$title} = $text if( $title ne "" );
    close( HELPFILE );
}

sub getHelp
{
    my $help = $helptext{$_[0]};
    my $htmlWrap = $_[1];
    $help = $_[0] if( $help eq "" );
    $help = "<span class=\"thelp\" title=\"$help\">$_[0]</span>" if( $htmlWrap == 1 );
    return $help;
}

sub writeLinksToFiles
{
    my ($pic,$tsv,$alt,$desc,$style) = ($_[0],$_[1],$_[2],$_[3],$_[4]);
    $style = " style=\"$style\"" if( $style ne "" );
    if( -f "$workdir/$pic" )
    {
        #print OUTFILE "<td><a style=\"cursor:help\" href=\"$pic\" title=\"$desc\">$alt<br/><img class=\"frm\"$style src=\"$pic\" alt=\"$alt\"/></a> ";
        print OUTFILE "<td><a style=\"cursor:help\" href=\"$pic\" title=\"$desc\"><img class=\"frm\"$style src=\"$pic\" alt=\"$alt\"/></a> ";
    }
    else
    {
        print STDERR "WARNING: Could not locate plot file $workdir/$pic\n";
        print OUTFILE "<td $style>$alt plot unavailable.<br/>";
    }
#    if( -f "$workdir/$tsv" )
#    {
#        print OUTFILE "<a href=\"$tsv\">Download data file</a></td>";
#    }
#    else
#    {
#        print STDERR "WARNING: Could not locate data file $workdir/$tsv\n";
#        print OUTFILE "Data file unavailable.</td>";
#    }
}

