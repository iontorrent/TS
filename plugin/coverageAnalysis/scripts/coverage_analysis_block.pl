#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

use File::Basename;

# get current running script dir
use FindBin qw($Bin);

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Collect output from coverageAnalysis run for block/summary HTML page.";
my $USAGE = "Usage:\n\t$CMD [options] <bam file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -a Customize to Amplicon reads coverage information (if a distinction is made)
  -g Expect 'Genome' rather than 'Target' as the tag used for base statistics summary (and use for output).
  -i Output sample Identification tracking reads in summary statistics.
  -r AmpliSeq RNA report. No output associated with base coverage and uniformity of coverage. (Overrides -a)
  -O <file> Output file name (relative to output directory). Should have .html extension. Default: ./block.html
  -D <dirpath> Path to Directory where html page is written. Default: '' (=> use path given by Output file name)
  -S <file> Input Statistics file name. Default: '-' (no summary file)
  -A <file> Auxillary help text file defining fly-over help for HTML titles. Default: ./help_tags.txt
  -s <title> Stats table header text (Plain text. Fly-over help added if <title> is matched to help text.) Default 'All Reads'.
  -t <title> Title for report. (Plain text or HTML.) Default: ''";

my $outfile = "block.html";
my $workdir=".";
my $statsfile = "";
my $plotsize = 400;
my $title = "";
my $helpfile ="";
my $tabhead = "All Reads";
my $genome = 0;
my $amplicons = 0;
my $rnacoverage = 0;
my $sampleid = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-O') {$outfile = shift;}
    elsif($opt eq '-D') {$workdir = shift;}
    elsif($opt eq '-S') {$statsfile = shift;}
    elsif($opt eq '-A') {$helpfile = shift;}
    elsif($opt eq '-a') {$amplicons = 1;}
    elsif($opt eq '-g') {$genome = 1;}
    elsif($opt eq '-i') {$sampleid = 1;}
    elsif($opt eq '-r') {$rnacoverage = 1;}
    elsif($opt eq '-s') {$tabhead = shift;}
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
elsif( scalar @ARGV != 1 )
{
    print STDERR "$CMD: Invalid number of arguments.";
    print STDERR "$USAGE\n";
    exit 1;
}

my $bamfile = shift;

$statsfile = "" if( $statsfile eq "-" );

# tied/derive options - in case futher customization is required
$amplicons = 0 if( $rnacoverage );
$ampcoverage  = ($amplicons || $rnacoverage );
$basecoverage = ($genome || !$rnacoverage );

# extract root name for output files from bam file names
my($runid,$folder,$ext) = fileparse($bamfile, qr/\.[^.]*$/);

#--------- End command arg parsing ---------

# check data folders
die "No output directory found at $workdir" unless( -d "$workdir" );

my %helptext;
loadHelpText( "$helpfile" );

$outfile = "$workdir/$outfile" if( $workdir ne "" );
open( OUTFILE, ">$outfile" ) || die "Cannot open output file $outfile.\n";

# Common header + link to stylesheet
print OUTFILE "<?xml version=\"1.0\" encoding=\"iso-8859-1\"?>\n";
print OUTFILE "<!DOCTYPE HTML>\n";
print OUTFILE "<html>\n<head>\n<base target=\"_parent\"/>\n";
print OUTFILE "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\">\n";
print OUTFILE "<link rel=\"stylesheet\" type=\"text/css\" href=\"lifechart/lifegrid.css\"/>\n";
print OUTFILE "</head>\n";
print OUTFILE "<body>\n";
print OUTFILE "<div style=\"margin-left:auto;margin-right:auto;\">\n";
if( $title ne "" )
{
    # add simple formatting if no html tag indicated
    $title = "<h3><center>$title</center></h3>" if( $title !~ /^\s*</ );
    print OUTFILE "$title\n";
}
print OUTFILE "<table class=\"center\"><tr>\n";
my $t2width = 350;
if( $rnacoverage )
{
  displayResults( "repoverview.png", "amplicon.cov.xls", "Representation Overview", 1, "height:100px" );
  print OUTFILE "</td>\n";
  $t2width = 320;
}
else
{
  displayResults( "covoverview.png", "covoverview.xls", "Coverage Overview", 1, "height:100px" );
  print OUTFILE "</td>\n";
  $t2width = 340;
}

my @keylist = ( "Number of mapped reads" );
push( @keylist, "Percent reads on target" ) if( !$genome );
push( @keylist, "Percent sample tracking reads" ) if( $sampleid );
push( @keylist, ( "Average base coverage depth", "Uniformity of base coverage" ) ) if( !$rnacoverage );
printf OUTFILE "<td><div class=\"statsdata\" style=\"width:%dpx\">\n", $t2width;
print OUTFILE subTable( "$workdir/$statsfile", \@keylist );
print OUTFILE "</div></td></tr>\n";
print OUTFILE "</table>\n";
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
        if( $keystr eq "" )
        {
            # treat empty string as spacer
            $htmlText .= "  <tr><td class=\"inleft\">&nbsp;</td><td class=\"inright\">&nbsp;</td></tr>\n";
            next;
        }
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
            $htmlText .= "  <tr><td class=\"inleft\">$keystr</td><td class=\"inright\">?</td></tr>\n";
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
	return unless( -e "$workdir/$runid.$tsv" );
    }
    my $desc = getHelp($alt);
    writeLinksToFiles( "$runid.$pic", "$runid.$tsv", $alt, $desc, $style );
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
    $style = "style=\"$style\"" if( $style ne "" );
    if( -f "$workdir/$pic" )
    {
        print OUTFILE "<td class=\"imageplot\" style=\"border:0\"><a href=\"$pic\" title=\"$desc\"><img $style src=\"$pic\" alt=\"$alt\"/></a> ";
    }
    else
    {
        print STDERR "WARNING: Could not locate plot file $workdir/$pic\n";
        print OUTFILE "<td $style>$alt plot unavailable.<br/>";
    }
}

