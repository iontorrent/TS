#!/usr/bin/perl
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

# This file is now used for generating partial a HTML report for cmd-line run_coverage_analysis.sh

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
  -a Extract Amplicon reads coverage information from stats file as a parallel report table. (Overrides -b)
  -b Extract targets coverage by Base coverage information from stats file as a parallel report table. 
  -g Expect 'Genome' rather than 'Target' as the tag used for base statistics summary (and use for output).
  -i Output sample Identification tracking reads in summary statistics.
  -r AmpliSeq RNA report. No output associated with base coverage and uniformity of coverage. (Overrides -a)
  -w Indicates to put a warning banner for missing targets file.
  -W <msg> General Warning message to be output to the report (after -w). Default: ''
  -D <dir> Directory path for working directory where input files are found and output files saved.
  -N <title> Name prefix for any output files for display and links. Default: 'tca_auxillary'.
  -A <file> Auxillary help file for fly-over help for HTML titles. Assumes JSON format if ext is '.json'. Default: <script dir>/help_tags.txt.
  -s <title> Stats table header text (Plain text. Fly-over help added if <title> is matched to help text.) Default 'All Reads'.
  -T <file> Name for HTML Table row summary file.
  -t <title> Secondary title for report. Default: ''.";

my $readsfile = "";
my $rowsumfile = "";
my $helpfile ="";
my $title = "";
my $workdir = "";
my $runid = "tca_auxillary";
my $tabhead = "All Reads";
my $amplicons = 0;
my $genome = 0;
my $rnacoverage = 0;
my $trgcoverage = 0;
my $sampleid = 0;
my $warnBanner = 0;
my $warnMessage = '';

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
  last if($ARGV[0] !~ /^-/);
  my $opt = shift;
  if($opt eq '-R') {$readsfile = shift;}
  elsif($opt eq '-D') {$workdir = shift;}
  elsif($opt eq '-N') {$runid = shift;}
  elsif($opt eq '-A') {$helpfile = shift;}
  elsif($opt eq '-T') {$rowsumfile = shift;}
  elsif($opt eq '-a') {$amplicons = 1;}
  elsif($opt eq '-b') {$trgcoverage = 1;}
  elsif($opt eq '-g') {$genome = 1;}
  elsif($opt eq '-i') {$sampleid = 1;}
  elsif($opt eq '-r') {$rnacoverage = 1;}
  elsif($opt eq '-s') {$tabhead = shift;}
  elsif($opt eq '-t') {$title = shift;}
  elsif($opt eq '-w') {$warnBanner = 1;}
  elsif($opt eq '-W') {$warnMessage = shift;}
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
my $statsfile = shift;

# option dependencies and overrides

if( $warnBanner ) {
  $amplicons = 0;
  $rnacoverage = 0;
}
$amplicons = 0 if( $rnacoverage );
$ampcoverage  = ($amplicons || $rnacoverage );
$basecoverage = ($genome || !$rnacoverage );
$trgcoverage = 0 if( $ampcoverage );

$statsfile = "" if( $statsfile eq "-" );
$rowsumfile = "" if( $rowsumfile eq "-" );

my $haverowsum = ($rowsumfile ne "");
my $have2stats = ($ampcoverage || $trgcoverage) && $basecoverage;
my $passingcov = 0; #$rnacoverage

$workdir = "." if( $workdir eq "" || $workdir eq "-" );
$statsfile = "$workdir/$statsfile";
$rowsumfile = "$workdir/$rowsumfile";

#--------- End command arg parsing ---------

# customize for genome base coverage
# TO DO modify for genome coverage
my $tagU = $genome ? "Genome" : "Target";
my $tagL = lc($tagU);

# check data folders
die "No statistics summary file found at $statsfile" unless( -f $statsfile );

my %helptext;
loadHelpText( "$helpfile" );

if( $haverowsum ) {
  # remove any old file since calls will append to this
  unlink( $rowsumfile );
}

open( OUTFILE, ">$workdir/$outfile" ) || die "Cannot open output file $workdir/$outfile.\n";

print OUTFILE "<div style=\"width:860px;margin-left:auto;margin-right:auto;\">\n";

if( $title ne "" ) {
  # add simple formatting if no html tag indicated
  $title = "<h3><center>$title</center></h3>" if( $title !~ /^\s*</ );
  print OUTFILE "$title\n";
}
if( $warnBanner ) {
  print OUTFILE "<h4 style='color:red'><center>Warning: No targets region specified as expected for Library Type.</center></h4>\n";
}
if( $warnMessage ne '' ) {
  print OUTFILE "$warnMessage\n";
}

# overview plot
print OUTFILE "<table class=\"center\"><tr>\n";
my $t2width = 350;
if( $rnacoverage ) {
  displayResults( "repoverview.png", "amplicon.cov.xls", "Representation Overview", 1, "height:100px" );
  print OUTFILE "</td>\n";
  $t2width = 320;
} else {
  displayResults( "covoverview.png", "covoverview.xls", "Coverage Overview", 1, "height:100px" );
  print OUTFILE "</td>\n";
  $t2width = 340;
}
my @keylist = ( "Number of mapped reads" );
push( @keylist, "Percent reads on target" ) if( !$genome );
push( @keylist, "Percent sample tracking reads" ) if( $sampleid );
push( @keylist, ( "Average base coverage depth", "Uniformity of base coverage" ) ) if( !$rnacoverage );
printf OUTFILE "<td><div class=\"statsdata\" style=\"width:%dpx\">\n",$t2width;
print OUTFILE subTable( $statsfile, \@keylist );
print OUTFILE "</div></td></tr>\n";
print OUTFILE "</table>\n";
print OUTFILE "<br/>\n";

# table headers
printf OUTFILE "<div class=\"statshead center\" style=\"width:%dpx\">\n", ($have2stats ? ($trgcoverage ? 780 : 730) : 380);
print OUTFILE "<table>\n <tr>\n";
if( $ampcoverage ) {
  $hotLable = getHelp("Amplicon Read Coverage",1);
  print OUTFILE "  <th>$hotLable</th>\n";
} elsif ( $trgcoverage ) {
  $hotLable = getHelp("Target Coverage",1);
  print OUTFILE "  <th>$hotLable</th>\n";
}
if( $basecoverage ) {
  my $hotLable = getHelp("$tagU Base Coverage",1);
  print OUTFILE "  <th>$hotLable</th>\n";
}
print OUTFILE " </tr>\n <tr>\n";
my @keylist;
if( $ampcoverage )
{
  @keylist = ( "Number of amplicons" );
  if( $amplicons )
  {
    push( @keylist, (
      "Percent assigned amplicon reads", "Average reads per amplicon", "Uniformity of amplicon coverage",
      "Amplicons with at least 1 read", "Amplicons with at least 20 reads",
      "Amplicons with at least 100 reads", "Amplicons with at least 500 reads" ) );
  }
  else
  {
    push( @keylist, (
      "Amplicons with at least 1 read", "Amplicons with at least 10 reads",
      "Amplicons with at least 100 reads", "Amplicons with at least 1000 reads",
      "Amplicons with at least 10K reads", "Amplicons with at least 100K reads" ) );
  }
  push( @keylist, "Amplicons with no strand bias" );
  push( @keylist, $passingcov ? "Amplicons with passing coverage" : "Amplicons reading end-to-end" );
  $txt = subTable( $statsfile, \@keylist );
  print OUTFILE "  <td><div class=\"statsdata\">$txt</div></td>\n";
} elsif( $trgcoverage ) {
  @keylist = ( "Number of unmerged targets", "Percent assigned target reads", "Average base coverage depth per target",
    "Uniformity of base coverage per target", "Targets with base coverage at 1x", "Targets with base coverage at 20x",
    "Targets with base coverage at 100x", "Targets with base coverage at 500x", "Targets with no strand bias", "Targets with full coverage" );
  $txt = subTable( $statsfile, \@keylist );
  print OUTFILE "  <td><div class=\"statsdata\">$txt</div></td>\n";
}
if( $basecoverage )
{
  if( $genome )
  {
    @keylist = ( "Bases in reference $tagL" );
  }
  else
  {
    @keylist = ( "Bases in $tagL regions", "Percent base reads on $tagL" );
  }
  push( @keylist, ( "Average base coverage depth", "Uniformity of base coverage", "$tagU base coverage at 1x",
    "$tagU base coverage at 20x", "$tagU base coverage at 100x", "$tagU base coverage at 500x", "$tagU bases with no strand bias" ) );
  push( @keylist, "Percent end-to-end reads" ) if( $amplicons );
  push( @keylist, '' ) if( $trgcoverage );
  my $txt = subTable( $statsfile, \@keylist );
  print OUTFILE "  <td><div class=\"statsdata\">$txt</div></td>\n";
}
print OUTFILE " </tr>\n</table>\n</div>\n</div>\n";

# create table row for separate summary page
if( $haverowsum )
{
  # Should this also be the same for ampliSeq?
  @keylist = ( "Number of mapped reads", "Percent reads on target" );
  push( @keylist, "Percent sample tracking reads" ) if( $sampleid );
  push( @keylist, ( "Average base coverage depth", "Uniformity of base coverage" ) ) if( !$rnacoverage );
  writeRowSum( $rowsumfile, $statsfile, \@keylist );
}

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
sub writeRowSum
{
  unless( open( ROWSUM, ">>$_[0]" ) )
  {
    print STDERR "Could not file for append at $_[0]\n";
    return;
  }
  print ROWSUM readRowSum($_[1],$_[2]);
  close( ROWSUM );
}

# args: 0 => input file, 1 => array of keys to read
sub readRowSum
{
  return "" unless( open( STATFILE, "$_[0]" ) );
  my @statlist;
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
    if( $foundKey == 0 && $keystr != $optionalKeyField )
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
    #print STDERR "No value found for statistic $keystr\n" if( $foundKey == 0 );
  }
  return $sumval;
}

sub loadHelpText
{
  my $hfile = $_[0];
  $hfile = "$Bin/help_tags.txt" if( $hfile eq "" );
  loadHelpJson($hfile) if( $hfile =~ /\.json$/ );
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
    next unless( /\S/ );
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

sub loadHelpJson
{
  # This only read pseudo-json: 1 level deep and no arrays
  my $hfile = $_[0];
  $hfile = "$Bin/../templates/stats_dict.json" if( $hfile eq "" );
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
    next unless( /\S/ );
    next if( /^\s*[{}]/ );
    ($title,$text) = split(':',$_,2);
    $title =~ s/^\s+|\s+$//g;
    $text =~ s/^\s+|\s+$//g;
    $title =~ s/^"|"$//g;
    $text =~ s/^"|"$//g;
    $helptext{$title} = $text if( $title ne "" );
  }
  close( HELPFILE );
}

sub getHelp
{
  my $help = $_[0];
  $help =~ s/[^0-9A-Za-z]/_/g unless( defined($helptext{$help}) );
  $help =~ s/^Amplicons with at/nAmplicons with at/ if( $rnacoverage );
  $help = $helptext{$help};
  my $htmlWrap = $_[1];
  $help = $_[0] if( $help eq "" );
  $help = "<span title=\"$help\">$_[0]</span>" if( $htmlWrap == 1 );
  return $help;
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

sub writeLinksToFiles
{
  my ($pic,$tsv,$alt,$desc,$style) = ($_[0],$_[1],$_[2],$_[3],$_[4]);
  $style = " style=\"$style\"" if( $style ne "" );
  if( -f "$workdir/$pic" )
  {
    print OUTFILE "<td class=\"imageplot\" style=\"background-color:#F5F5F5\"><a href=\"$pic\" title=\"$desc\"><img $style src=\"$pic\" alt=\"$alt\"/></a>";
  }
  else
  {
    print STDERR "WARNING: Could not locate plot file $workdir/$pic\n";
    print OUTFILE "<td $style>$alt plot unavailable.<br/>";
  }
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
  my $htmlText = " <table>\n";
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
        # corection for percentages vs. numbers for AmpliSeq-RNA
        $n = getHelp($n,1);
        $htmlText .= "  <tr><td class=\"inleft\">$n</td>";
        $htmlText .= ($v ne "") ? "<td class=\"inright\">$nf</td></tr>\n" : "</td>\n";
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
  return $htmlText." </table>";
}

