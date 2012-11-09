#!/usr/bin/perl
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

use JSON;
use LWP;

#--------- Begin command arg parsing ---------

(my $CMD = $0) =~ s{^(.*/)+}{};
my $DESCR = "Extract data from a comma-separate list of report IDs (no spaces).
Output is to STDOUT as text of HTML table rows. Optionally create a file of alignment files.";
my $USAGE = "Usage:\n\t$CMD [options] -H <host_domain_name> <reportID[,reportID...]>";
my $OPTIONS = "Options:
  -h ? --help Display Help information.
  -x Output as HTML table rows.
  -f Return error if nucleotide Flow sequences are not the same for all reports.
  -B <file> Output BAM file list to this file. Default: no output.
  -H <host> Remote host name or IP.";

my $bamlist = '';
my $htmlout = 0;
my $errFlows = 0;

my $help = (scalar(@ARGV) == 0);
while( scalar(@ARGV) > 0 )
{
    last if($ARGV[0] !~ /^-/);
    my $opt = shift;
    if($opt eq '-x') {$htmlout = 1;}
    elsif($opt eq '-f') {$errFlows = 1;}
    elsif($opt eq '-B') {$bamlist = shift;}
    elsif($opt eq '-H') {$host = shift;}
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
    print STDERR "$CMD: Invalid number of arguments.\n";
    print STDERR "$USAGE\n";
    exit 1;
}

my $reportList = shift;
my $makebamlist = ($bamlist ne '');

#--------- End command arg parsing ---------

my @months = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec');
my $useragent = LWP::UserAgent->new;

my @url_parts = split(/\//,$host);
$host = join("/",@url_parts[0..2]);


my $reportName;
my $projectName;
my $tmapVer;
my $aq17;
my $reportDate;
my $reportLink;
my $flowSeq;
my $bamfile;
my $filepath;

if( $makebamlist )
{
    open( BAMLIST, ">$bamlist" ) || die "$CMD: Cannot open alignment file list file for output: $bamlist.\n";
}

my $sumAQ17 = 0;
my $flowCheck = '.';
my $flowWarn = 0;
my $tmapCheck = '.';
my $tmapWarn = 0;

my @reports = split(',',$reportList);
my $numReports = scalar(@reports);

for( my $r = 0; $r < $numReports; ++$r )
{
    my $rID = $reports[$r];
    getReportData($rID);
    if( $htmlout == 0 )
    {
        print "Collected data for report id '$rID'\n";
        print "  REPORT:  $reportName\n";
        print "  PROJECT: $projectName\n";
        print "  AQ17:    $aq17\n";
        print "  TMAPVER: $tmapVer\n";
        print "  DATE:    $reportDate\n";
        print "  REPLINK: $reportLink\n";
        print "  FLOWSEQ: $flowSeq\n";
        print "  BAMFILE: $bamfile\n";
        #print "  FILESAT: $filepath\n";
    }
    else
    {
        print STDERR "Collected data for report id '$rID'\n";
        print "<tr><td><a href=\"$reportLink\">$reportName</a></td> <td>$projectName</td>";
        print " <td style=\"text-align:center\">$aq17</td> <td style=\"text-align:center\">$tmapVer</td>";
        print " <td style=\"text-align:center\">$reportDate</td> </tr>\n";
    }
    print BAMLIST $bamfile,"\n" if( $makebamlist );
    $sumAQ17 += $aq17;
    $flowCheck = $flowSeq if( $flowCheck eq '.' );
    $flowWarn = 1 if( $flowSeq ne $flowCheck );
    $tmapCheck = $tmapVer if( $tmapCheck eq '.' );
    $tmapWarn = 1 if( $tmapVer ne $tmapCheck );
}
close( BAMLIST ) if( $makebamlist );

printf STDERR "Collected data for %d reports.\n",$numReports;
printf STDERR "Total AQ17 Reads: %d\n",$sumAQ17;
print STDERR "WARNING: Not all runs were performed using the same version of TMAP.\n" if( $tmapWarn );
if( $flowWarn )
{
    if( $errFlows )
    {
        print STDERR "ERROR: Not all runs were performed using the same nucleotide flow sequence.\n";
        exit 1;
    }
    print STDERR "WARNING: Not all runs were performed using the same nucleotide flow sequence.\n";
}

# ----------------END-------------------

sub getReportData
{
    my $json_hash = getRequest("/rundb/api/v1/results/".$_[0]);
    $reportName = $json_hash->{'resultsName'};
    $reportLink = $json_hash->{'reportLink'};
    $filepath = $json_hash->{'filesystempath'};
    $bamfile = $json_hash->{'bamLink'};
    $bamfile =~ s/.*\///;
    $bamfile = $filepath.'/'.$bamfile;
    $reportDate = $json_hash->{'timeStamp'};
    $reportDate =~ s/T.*$//;
    my @ymd = split('-',$reportDate);
    my $mon = $months[$ymd[1]-1];
    $reportDate = "$mon $ymd[2] $ymd[0]";
    $tmapVer = $json_hash->{'analysisVersion'};
    $tmapVer =~ s/^.*?tm://;
    $tmapVer =~ s/,.*$//;

    my $exp = $json_hash->{'experiment'};
    my $lib = $json_hash->{'libmetrics'}[0];

    $json_hash = getRequest($exp);
    $projectName = $json_hash->{'project'};
    $flowSeq = $json_hash->{'flowsInOrder'};

    $json_hash = getRequest($lib);
    $aq17 = $json_hash->{'q17_alignments'};
}

sub getRequest
{
    my $reqstr = $host.$_[0]."/?noplugin=1";
    my $errMsg = "DB request failed for $reqstr";
    my $request = HTTP::Request->new(GET => $reqstr);
    $request->content_type('application/json');
    my $response = $useragent->request($request);
    die $errMsg if( !$response->is_success );
    return decode_json($response->content);
}

sub testConnection
{
	my $request = HTTP::Request->new(GET => $_[0]);
	my $response = $useragent->request($request);
	return $response->is_success;
}
