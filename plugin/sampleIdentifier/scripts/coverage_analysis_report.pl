#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

use File::Basename;

# get current running script dir
use FindBin qw($Bin);

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Format the output for coverage vs. one or two targets to an html page.
The two required arguments are the path to the result files and original mapped reads file.
The results directory is relative to the top level directory (specified by -D option) for HMTL.";
my $USAGE = "Usage:\n\t$CMD [options] <output html file> <stats file>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -R <file> Reads statsitics file (input top summary table).
  -S <file> Secondary statistics file (input for side summary table).
  -A <file> Auxillary help text file defining fly-over help for HTML titles. Default: 'help_tags.txt'.
  -T <file> Name for HTML Table row summary file.
  -B <ID> Sample Barcode identified, printed at top of page with help link.
  -t <title> Secondary title for report. Default: ''.";

my $readsfile = "";
my $statsfile2 = "";
my $rowsumfile = "";
my $helpfile ="";
my $title="";
my $sampleid="";

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-R') {$readsfile = shift;}
    elsif($opt eq '-S') {$statsfile2 = shift;}
    elsif($opt eq '-A') {$helpfile = shift;}
    elsif($opt eq '-B') {$sampleid = shift;}
    elsif($opt eq '-T') {$rowsumfile = shift;}
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

my $outfile = shift;
my $statsfile1 = shift;

my $haverowsum = ($rowsumfile ne "");
my $have2stats = ($statsfile2 ne "");
$statsfile1 = "" if( $statsfile1 eq "-" );

#--------- End command arg parsing ---------

# check data folders
die "No statistics summary file found at $statsfile1" unless( -f "$statsfile1" );
if( $have2stats )
{
    die "No statistics summary file found at $statsfile2" unless( -f "$statsfile2" );
}

my %helptext;
loadHelpText( "$helpfile" );

if( $haverowsum )
{
    # remove any old file since calls will append to this
    unlink( $rowsumfile );
}
open( OUTFILE, ">>$outfile" ) || die "Cannot open output file $outfile.\n";

if( $title ne "" )
{
    # add simple formatting if no html tag indicated
    $title = "<h3><center>$title</center></h3>" if( $title !~ /^\s*</ );
    print OUTFILE "$title\n";
}
if( $sampleid ne "" )
{
    my $help = "Sample ID: Sex (M/F) and list of alleles called from base read coverage at the sample identification loci.";
    print OUTFILE "<br/><h1 title=\"$help\" style=\"cursor:help;text-align:center;color:darkred\">$sampleid</h1>\n";
}

# Output read summary
if( $readsfile ne "" )
{
    my $readstable = readTextAsTableFormat( $readsfile );
    if( $readstable ne "" )
    {
        print OUTFILE "<br/>\n";
        print OUTFILE "<div class=\"statsdata center\" style=\"width:340px\">\n";
        print OUTFILE "$readstable\n";
        print OUTFILE "</div><br/>\n";
        if( $haverowsum )
        {
            if( $sampleid ne "" )
            {
                writeRowSum( $rowsumfile, $sampleid );
            }
            my @keylist = ( "Number of mapped reads", "Percent reads on target" );
            writeRowSum( $rowsumfile, $readsfile, \@keylist );
        }
    }
}

# table headers
printf OUTFILE "<div class=\"statshead center\" style=\"width:%dpx\">\n", ($have2stats ? 730 : 370);
my $hotLable = getHelp("Sample ID Regions",1);
print OUTFILE "<table>\n<tr><th>$hotLable</th>";
$hotLable = getHelp("Sample ID SNPs",1);
print OUTFILE "<th>$hotLable</th>" if( $have2stats );
print OUTFILE "</tr>\n";

my $txt = readTextAsTableFormat("$statsfile1");
print OUTFILE "<tr><td><div class=\"statsdata\">$txt</div></td>";
if( $have2stats )
{
    $txt = readTextAsTableFormat("$statsfile2");
    print OUTFILE "\n<td><div class=\"statsdata\">$txt</div></td>";
}
print OUTFILE "</tr>\n";
if( $haverowsum )
{
    # secondary (loci coverage) statistics are given in preference to primary (region coverage) statistics
    my @keylist = ( "Average base coverage depth", "Coverage at 20x", "Coverage at 100x" );
    writeRowSum( $rowsumfile, ($have2stats ? $statsfile2 : $statsfile1), \@keylist );
}

# write table foot
print OUTFILE "</table></div>\n<br/><br/>\n";

#print STDERR "> $outfile\n";
#print STDERR "> $rowsumfile\n" if( $haverowsum );

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

# args: 0 => output file (append), 1 => input file, 2 => array of keys to read
# if $_[2] not passed then just append $_[1] as the value
sub writeRowSum
{
    unless( open( ROWSUM, ">>$_[0]" ) )
    {
	print STDERR "Could not file for append at $_[0]\n";
	return;
    }
    if( scalar(@_) >= 3 )
    {
        print ROWSUM readRowSum($_[1],$_[2]);
    }
    else
    {
        print ROWSUM "<td>$_[1]</td> ";
    }
    close( ROWSUM );
}

# args: 0 => input file, 1 => array of keys to read
sub readRowSum
{
    my @statlist;
    return "" unless( open( STATFILE, "$_[0]" ) );
    while( <STATFILE> )
    {
        push( @statlist, $_ );
    }
    close( STATFILE );
    my @keylist = @{$_[1]};
    my $htmlText = "";
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
		$htmlText .= "<td>$nf</td> ";
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
    return $htmlText;
}

# args: 0 => input file, 1 => array of keys to read and sum (if present)
sub sumRowSum
{
    return 0 unless( open( STATFILE, "$_[0]" ) );
    my @statlist;
    while( <STATFILE> )
    {
        push( @statlist, $_ );
    }
    close( STATFILE );
    my @keylist = @{$_[1]};
    my $sumval = 0;
    foreach $keystr (@keylist)
    {
        my $foundKey = 0;
        foreach( @statlist )
        {
            my ($n,$v) = split(/:/);
            if( $n eq $keystr )
            {
                $v =~ s/^\s*//;
                $v =~ /^\d*\.?\d*/;
                $sumval += $v+0;
                ++$foundKey;
                last;
            }
        }
        print STDERR "No value found for statistic $keystr\n" if( $foundKey == 0 );
    }
    return $sumval;
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
    $help = "<span title=\"$help\">$_[0]</span>" if( $htmlWrap == 1 );
    return $help;
}
