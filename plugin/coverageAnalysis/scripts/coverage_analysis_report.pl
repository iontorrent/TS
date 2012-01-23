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
my $USAGE = "Usage:\n\t$CMD [options] <results directory> <bam file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information
  -O <file> Output file name (relative to output directory). Should have .html extension. Default: results.html
  -D <dirpath> Path to Directory where html page is written. Default: '' (=> use path given by Output file name)
  -R <dirpath> Path to secondary Results directory for the run. Default: '' (=> only one set of results)
  -B <file> Filepath to secondary Run ID. Default: '' (=> assume the same run id as given by <run ID>)
  -S <file> Statistics file name, relative to results directory. Default: '-' (no summary file)
  -A <file> Auxillary help text file defining fly-over help for HTML titles. Default: <script dir>/help_tags.txt
  -T <file> Name for HTML Table row summary file (in output directory). Default: '' (=> none created)
  -H <dirpath> Path to directory containing files 'header' and 'footer'. If present, used to wrap the output file name.
  -t <title> Secondary title for report. Default: ''";

my $outfile = "results.html";
my $workdir=".";
my $results2 = "";
my $bamfile2 = "";
my $statsfile = "";
my $plotsize = 400;
my $rowsumfile = "";
my $headfoot = "";
my $title = "";
my $helpfile ="";

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-O') {$outfile = shift;}
    elsif($opt eq '-D') {$workdir = shift;}
    elsif($opt eq '-R') {$results2 = shift;}
    elsif($opt eq '-B') {$bamfile2 = shift;}
    elsif($opt eq '-S') {$statsfile = shift;}
    elsif($opt eq '-A') {$helpfile = shift;}
    elsif($opt eq '-T') {$rowsumfile = shift;}
    elsif($opt eq '-H') {$headfoot = shift;}
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

my $have2runs = ($results2 ne "");
$statsfile = "" if( $statsfile eq "-" );

# extract root name for output files from bam file names
my($runid1,$folder,$ext) = fileparse($bamfile1, qr/\.[^.]*$/);
my $runid2 = $runid1;
if( $have2runs && $bamfile2 ne "" )
{
    ($runid2,$folder,$ext) = fileparse($bamfile2, qr/\.[^.]*$/);
}

#--------- End command arg parsing ---------

my $workdir1 = "$workdir/$results1";
my $workdir2 = "$workdir/$results2";

# check data folders
die "No output directory found at $workdir" unless( -d "$workdir" );
die "No results directory found at $workdir1" unless( -d "$workdir1" );

if( $have2runs )
{
    unless( -d "$workdir2" )
    {
	print STDERR "WARNING: Could not locate secondary results directory at $workdir2\nContinuing without...\n";
	$have2runs = 0;
    }
}

my %helptext;
loadHelpText( "$helpfile" );

$outfile = "$workdir/$outfile" if( $workdir ne "" );
open( OUTFILE, ">$outfile" ) || die "Cannot open output file $outfile.\n";

# write html header
if( $headfoot ne "" )
{
    if( open( HEADER, "$headfoot/header" ) )
    {
        unless( open( FOOTER, "$headfoot/footer" ) )
        {
	    close( HEADER );
	    print STDERR "Could not open $headfoot/footer\n";
	    $headfoot = "";
        }
    }
    else
    {
	print STDERR "Could not open $headfoot/header\n";
	$headfoot = "";
    }
}
if( $headfoot ne "" )
{
    while( <HEADER> )
    {
	print OUTFILE;
    }
    close( HEADER );
}
else
{
    print OUTFILE "<html><body>\n";
    if( $have2runs )
    {
	print OUTFILE "<div style=\"width:960px;margin-left:auto;margin-right:auto;height:100%\">\n";
    }
    else
    {
	print OUTFILE "<div style=\"width:480px;margin-left:auto;margin-right:auto;height:100%\">\n";
    }
    print OUTFILE "<h1><center>Coverage Analysis Report</center></h1>\n";
}
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
print OUTFILE "  th,td {width:50% !important;border:1px solid #bbbbbb;padding:5px;text-align:center}\n";
print OUTFILE "  td {padding-top:20px;padding-bottom:20px}\n";
print OUTFILE "  td.inleft  {width:75% !important;border-width:0;text-align:left;padding:2px;padding-left:40px}\n";
print OUTFILE "  td.inright {width:25% !important;border-width:0;text-align:right;padding:2px;padding-right:40px}\n";
print OUTFILE "  img.frm {display:block;margin-left:auto;margin-right:auto;margin-top:10px;margin-bottom:10px;width:400;height:400;border-width:0;cursor:help}\n";
print OUTFILE "  .thelp {cursor:help}\n";
print OUTFILE "</style>\n";

# table header
my $hotLable = getHelp("All Reads",1);
print OUTFILE "<center><table>\n<tr><th>$hotLable</th>";
$hotLable = getHelp("Unique Starts",1);
print OUTFILE "<th>$hotLable</th>" if( $have2runs );
print OUTFILE "</tr>\n";

if( $statsfile ne "" )
{
    my $txt = readTextAsTableFormat("$workdir1/$statsfile");
    print OUTFILE "<tr><td>$txt</td>";
    if( $have2runs )
    {
	$txt = readTextAsTableFormat("$workdir2/$statsfile");
	print OUTFILE "\n<td>$txt</td>";
    }
    print OUTFILE "</tr>\n";
    if( $rowsumfile ne "" )
    {
	$rowsumfile = "$workdir/$rowsumfile" if( $workdir ne "" );
	writeRowSum( "$rowsumfile", "$workdir1/$statsfile", "$workdir2/$statsfile" );
    }
}

displayResults( "coverage.png", "coverage.xls", "Target Coverage" );
displayResults( "coverage_binned.png", "coverage_binned.xls", "Binned Target Coverage" );
displayResults( "coverage_onoff_target.png", "coverage_by_chrom.xls", "Target Coverage by Chromosome" );
displayResults( "coverage_onoff_padded_target.png", "coverage_by_chrom_padded_target.xls", "Padded Target Coverage by Chromosome", 1 );
displayResults( "coverage_on_target.png", "fine_coverage.xls", "Individual Target Coverage" );
displayResults( "coverage_map_onoff_target.png", "coverage_map_by_chrom.xls", "On/Off Target Read Alignment" );
displayResults( "coverage_normalized.png", "coverage.xls", "Normalized Target Coverage" );
#displayResults( "coverage_distribution.png", "coverage_binned.xls", "Target Coverage Distribution" );

print OUTFILE "<tr>\n";
writeBamLinks( "$runid1.bam" );
writeBamLinks( "$runid2.bam" ) if( $have2runs );
print OUTFILE "<tr>\n";

# write html footer
print OUTFILE "</table></center>\n";
if( $headfoot ne "" )
{
    while( <FOOTER> )
    {
	print OUTFILE;
    }
    close( FOOTER );
}
else
{
    print OUTFILE "</div></body></html>\n";
    close( OUTFILE );
}

#print STDERR "> $outfile\n";

#-------------------------END-------------------------

sub readTextAsTableFormat
{
    unless( open( TEXTFILE, "$_[0]" ) )
    {
	print STDERR "Could not locate text file $_[0]\n";
	return "Data unavailable";
    }
    my $htmlText = "<table>\n";
    while( <TEXTFILE> )
    {
	my ($n,$v) = split(/:/);
	$v =~ s/^\s*//;
	# format leading numeric string using commas
	my $nf = ($v =~ /^(\d*)(\.?.*)/) ? commify($1).$2 : $v;
	$n = getHelp($n,1);
	$htmlText .= "<tr><td class=\"inleft\">$n</td>";
	$htmlText .= ($v ne "") ? " <td class=\"inright\">$nf</td></tr>\n" : "</td>\n";
    }
    close( TEXTFILE );
    return $htmlText."</table>";
}

sub commify
{
    (my $num = $_[0]) =~ s/\G(\d{1,3})(?=(?:\d\d\d)+(?:\.|$))/$1,/g;
    return $num;
}

sub writeRowSum
{
    unless( open( ROWSUM, ">$_[0]" ) )
    {
	print STDERR "Could not write to $_[0]\n";
	return;
    }
    print ROWSUM readRowSum($_[1]);
    if( $have2runs )
    {
	print ROWSUM readRowSum($_[2]);
    }
    #print STDERR "> $_[0]\n";
}

sub readRowSum
{
    my $htmlText = "";
    my $nread = 0;
    if( open( STATFILE, "$_[0]" ) )
    {
	while( <STATFILE> )
	{
	    my ($n,$v) = split(/:/);
	    if( $n eq "Number of mapped reads" ||
		$n eq "Percent reads on target" ||
		$n eq "Average base coverage depth" ||
		$n eq "Target coverage at 1x" )
	    {
		$v =~ s/^\s*//;
		my $nf = ($v =~ /^(\d*)(\.?.*)/) ? commify($1).$2 : $v;
		$htmlText .= "<td>$nf</td> ";
		++$nread;
	    }
	}
	close( STATFILE );
    }
    for( my $i = $nread; $i < 4; ++$i )
    {
        $htmlText .= "<td>N/A</td> ";
    }
    return $htmlText;
}

sub displayResults
{
    my ($pic,$tsv,$alt,$skip) = ($_[0],$_[1],$_[2],$_[3]);
    # if skip is defined (e.g. 1) do not output anything if the first data file is missing
    # i.e. it is expected for this set of data to be missing and therefore skipped
    if( $skip != 0 )
    {
	return unless( -e "$workdir/$results1/$runid1.$tsv" );
    }
    my $desc = getHelp($alt);
    print OUTFILE "<tr>\n";
    writeLinksToFiles( "$results1/$runid1.$pic", "$results1/$runid1.$tsv", $alt, $desc );
    writeLinksToFiles( "$results2/$runid2.$pic", "$results2/$runid2.$tsv", $alt, $desc ) if( $have2runs );
    print OUTFILE "\n</tr>\n\n";
}

sub writeLinksToFiles
{
    my ($pic,$tsv,$alt,$desc) = ($_[0],$_[1],$_[2],$_[3]);
    if( -f "$workdir/$pic" )
    {
	print OUTFILE "<td><a style=\"cursor:help\" href=\"$pic\" title=\"$desc\">$alt<br/><img class=\"frm\" src=\"$pic\" alt=\"$alt\"/></a> ";
    }
    else
    {
	print STDERR "WARNING: Could not locate plot file $workdir/$pic\n";
	print OUTFILE "<td>Plot unavailable<br/>";
    }
    if( -f "$workdir/$tsv" )
    {
	print OUTFILE "<a href=\"$tsv\">Download data file</a></td>";
    }
    else
    {
	print STDERR "WARNING: Could not locate data file $workdir/$tsv\n";
	print OUTFILE "No data</td>";
    }
}

sub writeBamLinks
{
    my $bam = $_[0];
    if( -f "$workdir/$bam" )
    {
	print OUTFILE "<td><a href=\"$bam\">Download BAM file</a><br/>\n";
	print OUTFILE "<a href=\"$bam.bai\">Download BAM index file</a></td>\n";
    }
    else
    {
	print STDERR "WARNING: Could not locate bam file $workdir/$bam\n";
	print OUTFILE "<td>BAM file unavailable</td>\n";
    }
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
