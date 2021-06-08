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
  "analysis-name"    => "NoName",
  "error-summary-h5" => undef,
  "floworder"        => "TACGTACGTCTGAGCATCGATCGATGTACAGC",
  "out-dir"          => ".",
  "help"             => 0,
  "debug"            => 0,
  "html-only"        => 0,
  "plugin-name"      => "flowErr2",
  "unfiltered"       => 0,
  "untrimmed"        => 0,
  "multilane"        => "false",
  "laneid"           => 0
};

GetOptions(
    "n|analysis-name=s"    => \$opt->{"analysis-name"},
    "e|error-summary-h5=s" => \$opt->{"error-summary-h5"},
    "f|floworder=s"        => \$opt->{"floworder"},
    "o|out-dir=s"          => \$opt->{"out-dir"},
    "h|help"               => \$opt->{"help"},
    "debug"                => \$opt->{"debug"},
    "html-only"            => \$opt->{"html-only"},
    "unfiltered"           => \$opt->{"unfiltered"},
    "untrimmed"            => \$opt->{"untrimmed"},
    "multilane"            => \$opt->{"multilane"},
    "laneid=i"               => \$opt->{"laneid"},
);

&usage() if(
    $opt->{"help"}
    || !defined($opt->{"error-summary-h5"})
    || !defined($opt->{"floworder"})
    || !defined($opt->{"out-dir"})
);

sub usage () {
    print STDERR << "EOF";

    usage: $0 [-o outDir -n analysisName -f floworder] -e ionstats_error_summary.h5
     -n,--analysis-name    : Name for plots [NoName]
     -e,--error-summary-h5 : HDF5 file produced by ionstats alignment
     -f,--floworder        : Nuc flow order [TACGTACGTCTGAGCATCGATCGATGTACAGC]
     -o,--out-dir          : directory in which to write results [.]
     --debug               : Just write the R script and exit
     --html-only           : skip analysis and plotting, just make html
     --unfiltered          : use unfiltered bam
     --untrimmed           : use untrimmed bam
     -h,--help             : this (help) message
EOF
  exit(1);
}

# Locate the R script and make sure we can read it
my $rScriptFile = dirname($0) . "/" .$opt->{"plugin-name"} . ".R";
die "$0: unable to find R script $rScriptFile\n" if (! -e $rScriptFile);
die "$0: unable to read R script $rScriptFile\n" if (! -r $rScriptFile);
my $rFnsFile    = dirname($0) . "/" .$opt->{"plugin-name"} . ".fns.R";
die "$0: unable to find R functions file $rFnsFile\n" if (! -e $rFnsFile);
die "$0: unable to read R functions file $rFnsFile\n" if (! -r $rFnsFile);

# Find the R lib dir - to hande the fact that we have different R versions on different OSes
my $rLibDir = &findRLibDir($0);

my $outDir = $opt->{"out-dir"};

mkpath $outDir || die "$0: unable to make directory ".$outDir.": $!\n";
my $htmlFile = "";
if($opt->{"multilane"}) {
  $htmlFile = $outDir."/".$opt->{"plugin-name"}."_lane_".$opt->{"laneid"}.".html";
}else{
  $htmlFile = $outDir."/".$opt->{"plugin-name"}.".html";
}

if($opt->{"unfiltered"}) {
  if($opt->{"untrimmed"}) {
    $htmlFile = $outDir."/".$opt->{"plugin-name"}."_unfiltered_untrimmed.html";
  } else {
    $htmlFile = $outDir."/".$opt->{"plugin-name"}."_unfiltered_trimmed.html";
  }
}
my $htmlFh = FileHandle->new("> $htmlFile") || die "$0: problem writing $htmlFile: $!\n";
my $htmlTitle = $opt->{"plugin-name"} . " for " . $opt->{"analysis-name"};
my $htmlHeaderLine = $htmlTitle;
&writeHtmlHeader($htmlTitle,$htmlHeaderLine,$htmlFh);

my $plotDir = "plots";


if($opt->{"multilane"}) {
    $plotDir = $plotDir."_lane_".$opt->{"laneid"};
}

if($opt->{"unfiltered"}) {
  if($opt->{"untrimmed"}) {
    $plotDir = $plotDir."_unfiltered_untrimmed";
  } else {
    $plotDir = $plotDir."_unfiltered_trimmed";
  }
}
# Write out the header of the R script
my $tempDirRscript = &File::Temp::tempdir(CLEANUP => ($opt->{"debug"}==0));
my $rTempFile = "$tempDirRscript/flowErr.R";
my $rTempFh = FileHandle->new("> $rTempFile") || die "$0: unable to open R script file $rTempFile for write: $!\n";

if($opt->{"multilane"}) {
  print $rTempFh "analysisName <- \"".$opt->{"analysis-name"}."_Lane_".$opt->{"laneid"}."\"\n";
}else{
  print $rTempFh "analysisName <- \"".$opt->{"analysis-name"}."\"\n";
}
print $rTempFh "errorSummaryH5 <- \"".$opt->{"error-summary-h5"}."\"\n";
my $jsonResultsFile = "results.json";
print $rTempFh "jsonResultsFile <- \"".$outDir."/$jsonResultsFile\"\n";
print $rTempFh "plotDir <- \"".$outDir."/$plotDir\"\n";
print $rTempFh "unfiltered <- \"".$opt->{"unfiltered"}."\"\n";
print $rTempFh "untrimmed <- \"".$opt->{"untrimmed"}."\"\n";
if(defined($opt->{"floworder"})) {
  print $rTempFh "floworder <- \"".$opt->{"floworder"}."\"\n";
} else {
  print $rTempFh "floworder <- NA\n";
}
print $rTempFh "source(\"$rFnsFile\")\n";
print $rTempFh "libDir <- \"$rLibDir\"\n";
open(RSCRIPT,$rScriptFile) || die "$0: failed to open R script file $rScriptFile: $!\n";
while(<RSCRIPT>) {
  print $rTempFh $_;
}
# {
#   my $command = "cat $rTempFile";
#   system($command)
# }
close RSCRIPT;
close $rTempFh;
my $rLogFile = $outDir."/".$opt->{"plugin-name"}.".Rout";
if($opt->{"unfiltered"}) {
  if($opt->{"untrimmed"}) {
    $rLogFile = $outDir."/".$opt->{"plugin-name"}."_unfiltered_untrimmed.Rout";
  } else {
    $rLogFile = $outDir."/".$opt->{"plugin-name"}."_unfiltered_trimmed.Rout";
  }
}
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

my $blockHtmlFile = $outDir."/".$opt->{"plugin-name"}."_block.html";
if($opt->{"unfiltered"}) {
  if($opt->{"untrimmed"}) {
    $blockHtmlFile = $outDir."/".$opt->{"plugin-name"}."_block_unfiltered_untrimmed.html";
  } else {
    $blockHtmlFile = $outDir."/".$opt->{"plugin-name"}."_block_unfiltered_trimmed.html";
  }
}
my $blockHtmlFh = FileHandle->new("> $blockHtmlFile") || die "$0: problem writing $blockHtmlFile: $!\n";

chdir $outDir;

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

  my @plotTypes = ();
  my $plotType = "";

  @plotTypes = (
    "per_base.regional.error.dynamic.png",
    "per_base.regional.aligned.dynamic.png",
    "spatial_filter.png"
  );
  print $htmlFh "  <tr>\n";
  for(my $i=0; $i <= $#plotTypes; $i++) {
    my $plotType = $plotTypes[$i];
    my @files = glob(sprintf("%s/*.%s",$plotDir,$plotType));
    if(@files==1) {
      print $htmlFh "    <td><img src=\"".$files[0]."\"/></td>\n";
    } else {
      print $htmlFh "    <td>NA</td>\n";
    }
  }
  print $htmlFh "  </tr>\n";
  print $htmlFh "</table>\n";

  # Per-HP results
  print $htmlFh "<h1>Per-HP Error Rates</h1>\n";
  print $htmlFh "<table border=2 cellpadding=5>\n";

  print $htmlFh "  <tr>\n";
  my @files = glob(sprintf("%s/*.%s",$plotDir,"per_hp.err_distribution.png"));
  if(@files==1) {
    print $htmlFh "    <td><img src=\"".$files[0]."\"/></td>\n";
  } else {
    print $htmlFh "    <td>NA</td>\n";
  }
  print $htmlFh "  </tr>\n";

  @plotTypes = (
    "per_hp.total.error.png",
    "per_hp.over-under.error.png",
    "per_hp.total.phred.png",
    "per_hp.over-under.phred.png",
  );
  print $htmlFh "  <tr>\n";
  for(my $i=0; $i <= $#plotTypes; $i++) {
    my $plotType = $plotTypes[$i];
    my @files = glob(sprintf("%s/*.%s",$plotDir,$plotType));
    if(@files==1) {
      print $htmlFh "    <td><img src=\"".$files[0]."\"/></td>\n";
    } else {
      print $htmlFh "    <td>NA</td>\n";
    }
  }
  print $htmlFh "  </tr>\n";

  $plotType = "*mer_error.regional.png";
  @files = glob(sprintf("%s/*.%s",$plotDir,$plotType));
  if(@files > 0) {
    print $htmlFh "  <tr>\n";
    foreach my $file (@files) {
      my $movie = ($file =~ /^(.+)\.(\d+)mer_error.regional.png$/) ? "$1.per_flow.movie.regional.".$2."mer.gif" : "NA";
      print $htmlFh "    <td><center>(Click image for per-flow animated gif)</center><a href=\"".$movie."\"/> <img src=\"$file\"/> </a></td>\n";
    }
    print $htmlFh "  </tr>\n";
  }

  $plotType = "per_flow.*mer_err.png";
  @files = glob(sprintf("%s/*.%s",$plotDir,$plotType));
  if(@files > 0) {
    print $htmlFh "  <tr>\n";
    foreach my $file (@files) {
      print $htmlFh "    <td><img src=\"".$file."\"/></td>\n";
    }
    print $htmlFh "  </tr>\n";
  }

  print $htmlFh "</table>\n";

  print $htmlFh "<h1>Per-Flow Error Rates</h1>\n";
  print $htmlFh "<table border=2 cellpadding=5>\n";
  @plotTypes = (
    "per_flow.err.png",
    "per_flow.sub.png",
    "per_flow.ins.png",
    "per_flow.del.png"
  );
  print $htmlFh "  <tr>\n";
  for(my $i=0; $i <= $#plotTypes; $i++) {
    my $plotType = $plotTypes[$i];
    my @files = glob(sprintf("%s/*.%s",$plotDir,$plotType));
    if(@files==1) {
      print $htmlFh "    <td><img src=\"".$files[0]."\"/></td>\n";
    } else {
      print $htmlFh "    <td>NA</td>\n";
    }
  }
  print $htmlFh "  </tr>\n";

  @plotTypes = (
    "per_incorporating_flow.err.png",
    "per_incorporating_flow.sub.png",
    "per_incorporating_flow.ins.png",
    "per_incorporating_flow.del.png"
  );
  print $htmlFh "  <tr>\n";
  for(my $i=0; $i <= $#plotTypes; $i++) {
    my $plotType = $plotTypes[$i];
    my @files = glob(sprintf("%s/*.%s",$plotDir,$plotType));
    if(@files==1) {
      print $htmlFh "    <td><img src=\"".$files[0]."\"/></td>\n";
    } else {
      print $htmlFh "    <td>NA</td>\n";
    }
  }
  print $htmlFh "  </tr>\n";
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
  my @files = glob(sprintf("%s/*.per_incorporating_flow.err.png",$plotDir));
  print $fh "  <a href=\"".$files[0]."\"><img src=\"".$files[0]."\" height=$imgHeight />\n" if(@files==1);
  @files = glob(sprintf("%s/*.per_base.png",$plotDir));
  print $fh "  <a href=\"".$files[0]."\"><img src=\"".$files[0]."\" height=$imgHeight />\n" if(@files==1);
  @files = glob(sprintf("%s/*.per_hp.total.error.png",$plotDir));
  print $fh "  <a href=\"".$files[0]."\"><img src=\"".$files[0]."\" height=$imgHeight />\n" if(@files==1);
  @files = glob(sprintf("%s/*.per_base.regional.error.dynamic.png",$plotDir));
  print $fh "  <a href=\"".$files[0]."\"><img src=\"".$files[0]."\" height=$imgHeight />\n" if(@files==1);
  @files = glob(sprintf("%s/*.per_base.regional.aligned.dynamic.png",$plotDir));
  print $fh "  <a href=\"".$files[0]."\"><img src=\"".$files[0]."\" height=$imgHeight />\n" if(@files==1);
  print $fh "  <br>\n";
}

sub findRLibDir {
  my($exe) = @_;

  # Determine the R version
  my $rVersion = 0;
  my $command = "R --version | head -n 1 | cut -f1 -d. | cut -f3 -d\\ ";
  $rVersion = `$command`;
  chomp $rVersion;

  my $baseDir = dirname($exe);
  die "$0: unexpected R version: $rVersion\n" if(! $rVersion =~ /^(2|3)$/);
  my $rLibDir = "$baseDir/lib/Rv$rVersion";
  die "$0: unable to find R lib dir $rLibDir\n" if (! -e $rLibDir);
  return($rLibDir);
}
