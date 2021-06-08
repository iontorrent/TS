#!/usr/bin/env perl
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# Script to wrap around an R analysis to display various spatial signal distributions

use strict;
use warnings;
use FileHandle;
use Cwd;
use File::Temp;
use File::Path qw(mkpath);
use File::Basename;
use FileHandle;
use Getopt::Long;

my $opt = {
  "analysis-dir"  => undef,
  "analysis-name" => undef,
  "floworder"     => "TACGTACGTCTGAGCATCGATCGATGTACAGC",
  "bam"           => undef,
  "out-dir"       => ".",
  "help"          => 0,
  "debug"         => 0,
  "html-only"     => 0,
  "plugin-name"   => "flowErr",
  "alignStats"    => "alignStats",
};

GetOptions(
    "a|analysis-dir=s"     => \$opt->{"analysis-dir"},
    "n|analysis-name=s"    => \$opt->{"analysis-name"},
    "f|floworder=s"        => \$opt->{"floworder"},
    "b|bam=s"              => \$opt->{"bam"},
    "o|out-dir=s"          => \$opt->{"out-dir"},
    "s|alignStats=s"       => \$opt->{"alignStats"},
    "h|help"               => \$opt->{"help"},
    "debug"                => \$opt->{"debug"},
    "html-only"            => \$opt->{"html-only"},
);

&usage() if(
    $opt->{"help"}
    || !defined($opt->{"bam"})
    || !defined($opt->{"out-dir"})
);

sub usage () {
    print STDERR << "EOF";

    usage: $0 [-o outDir -n analysisName -a myAnalysisDir] -b bamFile
     -a,--analysis-dir   : directory with analysis results - used for plot names
     -n,--analysis-name  : Name for plots
     -f,--floworder      : Nuc flow order [TACGTACGTCTGAGCATCGATCGATGTACAGC]
     -b,--bam            : BAM file of alignments
     -o,--out-dir        : directory in which to write results [.]
     --debug             : Just write the R script and exit
     --html-only         : skip analysis and plotting, just make html
     -h,--help           : this (help) message
EOF
  exit(1);
}

my $hostname = `hostname`;
print "HOSTNAME=$hostname\n";

# Locate the R script and make sure we can read it
my $rScriptFile = dirname($0) . "/" .$opt->{"plugin-name"} . ".R";
die "$0: unable to find R script $rScriptFile\n" if (! -e $rScriptFile);
die "$0: unable to read R script $rScriptFile\n" if (! -r $rScriptFile);

mkpath $opt->{"out-dir"} || die "$0: unable to make directory ".$opt->{"out-dir"}.": $!\n";
my $htmlFile = $opt->{"out-dir"}."/".$opt->{"plugin-name"}.".html";
my $htmlFh = FileHandle->new("> $htmlFile") || die "$0: problem writing $htmlFile: $!\n";
if(!defined($opt->{"analysis-name"})) {
  $opt->{"analysis-name"} = defined($opt->{"analysis-dir"}) ? basename($opt->{"analysis-dir"}) : "NoName";
}
my $htmlTitle = $opt->{"plugin-name"} . " for " . $opt->{"analysis-name"};
my $htmlHeaderLine = $htmlTitle;
&writeHtmlHeader($htmlTitle,$htmlHeaderLine,$htmlFh);

my $plotDir = "plots";

# Write out the header of the R script
my $tempDirRscript = &File::Temp::tempdir(CLEANUP => ($opt->{"debug"}==0));
my $rTempFile = "$tempDirRscript/rawTrace.R";
my $rTempFh = FileHandle->new("> $rTempFile") || die "$0: unable to open R script file $rTempFile for write: $!\n";

# Analyze BAM file to determine per-flow error rate
my $tempDir = &File::Temp::tempdir(CLEANUP => ($opt->{"debug"}==0));
my $flowErrFile = &bamToFlowErr($opt->{"bam"},$tempDir,$opt);

print $rTempFh "analysisName <- \"".$opt->{"analysis-name"}."\"\n";
my $jsonResultsFile       = "results.json";
print $rTempFh "jsonResultsFile <- \"".$opt->{"out-dir"}."/$jsonResultsFile\"\n";
print $rTempFh "flowErrFile <- \"$flowErrFile\"\n";
print $rTempFh "plotDir <- \"".$opt->{"out-dir"}."/$plotDir\"\n";
if(defined($opt->{"floworder"})) {
  print $rTempFh "floworder <- \"".$opt->{"floworder"}."\"\n";
} else {
  print $rTempFh "floworder <- NA\n";
}
open(RSCRIPT,$rScriptFile) || die "$0: failed to open R script file $rScriptFile: $!\n";
while(<RSCRIPT>) {
  print $rTempFh $_;
}
close RSCRIPT;
close $rTempFh;
my $rLogFile = $opt->{"out-dir"} . "/" . $opt->{"plugin-name"} . ".Rout";
my $retVal = 0;
if(!$opt->{"html-only"}) {
  if($opt->{"debug"}) {
    print "\n";
    print "Wrote R script to $rTempFile\n";
  } else {
    my $command = "R CMD BATCH --slave --no-save --no-timing $rTempFile $rLogFile";
    $retVal = system($command);
  }
  &rLogPrintHtml($htmlFh,$rLogFile,"<br>Problem running R\n") if($retVal != 0);
}



my $blockHtmlFile = $opt->{"out-dir"}."/".$opt->{"plugin-name"}."_block.html";
my $blockHtmlFh = FileHandle->new("> $blockHtmlFile") || die "$0: problem writing $blockHtmlFile: $!\n";

chdir $opt->{"out-dir"};

&writeBlockHtml($blockHtmlFh,$plotDir,$opt);
&finishHtml($blockHtmlFh);
close $blockHtmlFh;

&writeHtmlPlots($htmlFh,$plotDir);

if(!$opt->{"debug"}) {
  &finishHtml($htmlFh);
}


sub writeHtmlPlots {
  my($htmlFh,$plotDir) = @_;

  print $htmlFh "<table border=2 cellpadding=5>\n";
  my @plotTypes  = ("allFlows.png","100Flows.png");
  print $htmlFh "  <tr>";
  foreach my $plotType (@plotTypes) {
    my @files = glob(sprintf("%s/*.%s",$plotDir,$plotType));
    if(@files==1) {
      print $htmlFh "<td><img src=\"".$files[0]."\"/></td>";
    } else {
      print $htmlFh "<td>NA</td>";
    }
  }
  print $htmlFh "  <tr>\n";
  print $htmlFh "</table>\n";
}

sub writeHtmlHeader {
  my($title,$header,$fh) = @_;
  print $fh "<head>\n";
  print $fh "  <title>$title</title>\n";
  print $fh "</head>\n";
  print $fh "<body>\n";
  print $fh "<h1>$header</h1>\n";
}

sub finishHtml {
    my $fh = shift(@_);
    print $fh "</body>\n";
    close $fh;
}

sub rLogPrintHtml {
    my($outFh,$rLogFile,$errString) = @_;
    `cat $rLogFile`;
    my $rLogFh = FileHandle->new("< $rLogFile");
    if($rLogFh) {
	print $outFh $errString."  <br>R log file listed below:\n\n";
	print $outFh "  <pre>\n\n";
	while(<$rLogFh>) {
	    print $outFh $_;
	}
	print $outFh "  </pre>\n";
	close $rLogFh;
    } else {
	print $outFh $errString."  <br>No R log file found.\n\n";
    }
}

sub executeSystemCall {
  my ($command,$returnVal) = @_;

  # Initialize status tracking
  my $exeFail  = 0;
  my $died     = 0;
  my $core     = 0;
  my $exitCode = 0;

  # Run command
  if(!defined($returnVal)) {
    system($command);
  } else {
    $$returnVal = `$command`;
  }

  # Check status
  if ($? == -1) {
    $exeFail = 1;
  } elsif ($? & 127) {
   $died = ($? & 127);
   $core = 1 if($? & 128);
  } else {
    $exitCode = $? >> 8;
  }

  my $problem = 0;
  if($exeFail || $died || $exitCode) {
    print STDERR "$0: problem encountered running command \"$command\"\n";
    if($exeFail) {
      print STDERR "Failed to execute command: $!\n";
    } elsif ($died) {
      print STDERR sprintf("Child died with signal %d, %s coredump\n", $died,  $core ? 'with' : 'without');
    } else {
      print STDERR "Child exited with value $exitCode\n";
    }
    $problem = 1;
  }

  return($problem);
}

sub writeBlockHtml {
  my($fh,$plotDir,$opt) = @_;
  my $htmlTitle = $opt->{"plugin-name"} . " block_html for " . $opt->{"analysis-name"};
  &writeHtmlHeader($htmlTitle,"",$fh);

  my $imgHeight=250;
  my @files = glob(sprintf("%s/*.allFlows.png",$plotDir));
  print $fh "  <a href=\"".$files[0]."\"><img src=\"".$files[0]."\" height=$imgHeight />\n" if(@files==1);
  print $fh "  <br>\n";
}

sub bamToFlowErr {
  my($bamFile,$outDir,$opt) = @_;

  my $flowErrFile = "flowErr.txt";
  my $cwd = &getcwd();
  chdir $outDir;
  if(!$opt->{"html-only"}) {
    my $command = $opt->{"alignStats"};
    $command .= " -n 12";
    $command .= " -i $bamFile";
    $command .= " --flowErrFile $flowErrFile";
    $command .= " --flowOrder ".$opt->{"floworder"};
    $command .= " --3primeClip 10 ";
    warn "$0: Failed to generate flowErr file $flowErrFile from bam file while in dir $outDir with the following command:\n$command\n" if(&executeSystemCall($command));
  }
  chdir $cwd;
  $flowErrFile = "$outDir/$flowErrFile";
  
  return($flowErrFile);
}
