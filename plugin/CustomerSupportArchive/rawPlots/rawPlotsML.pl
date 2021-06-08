#!/usr/bin/env perl
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Script to wrap around an R analysis to display raw data off the chip

use strict;
use warnings;
use FileHandle;
use Cwd;
use File::Temp;
use File::Basename;
use Getopt::Long;
use File::Copy;

my $opt = {
  "raw-dir"       => undef,
  "analysis-dir"  => undef,
  "alignment-dir" => undef,
  "sigproc-dir" => undef,
  "basecaller-dir" => undef,
  "analysis-name" => undef,
  "chip-type"     => undef,
  "out-dir"       => ".",
  "help"          => 0,
  "debug"         => 0,
  "just-html"     => 0,
  "base-name"     => "rawPlots",
  "thumbnail"     => "true",
  "active-lanes"  => ""
};

GetOptions(
    "r|raw-dir=s"          => \$opt->{"raw-dir"},
    "a|analysis-dir=s"     => \$opt->{"analysis-dir"},
    "s|sigproc-dir=s"     => \$opt->{"sigproc-dir"},
    "b|basecaller-dir=s"     => \$opt->{"basecaller-dir"},
    "l|alignment-dir=s"     => \$opt->{"alignment-dir"},
    "n|analysis-name=s"    => \$opt->{"analysis-name"},
    "c|chip-type=s"        => \$opt->{"chip-type"},
    "o|out-dir=s"          => \$opt->{"out-dir"},
    "j|just-html"          => \$opt->{"just-html"},
    "h|help"               => \$opt->{"help"},
    "d|debug"              => \$opt->{"debug"},
    "t|thumbnail=s"        => \$opt->{"thumbnail"},
    "active-lanes=s"       => \$opt->{"active-lanes"},
    );

&usage() if(
    $opt->{"help"}
    || !defined($opt->{"analysis-dir"})
    || !defined($opt->{"raw-dir"})
    || !defined($opt->{"out-dir"})
    );

sub usage () {
    print STDERR << "EOF";

    usage: $0 [-o outDir -d -h] -r myAcqDir -a myAnalysisDir -o out
     -r,--raw-dir acqdir        : directory with the acq files
     -a,--analysis-dir myInput  : directory with analysis results
     -o,--out-dir myOutput      : directory in which to write results
     -c,--chip-type 314R        : specify chip type
     -j,--just-html             : rerun html generation not plots
     -d,--debug                 : Just write the R script and exit
     -h,--help                  : this (help) message
EOF
  exit(1);
}

# Check that we have some R scripts
my $rScriptFile = dirname($0) . "/" .$opt->{"base-name"} . ".R";
die "$0: unable to find R script $rScriptFile\n" if (! -e $rScriptFile);
die "$0: unable to read R script $rScriptFile\n" if (! -r $rScriptFile);
my $rFunctionFile = dirname($0) . "/" .$opt->{"base-name"} . ".fns.R";
die "$0: unable to find R script $rFunctionFile\n" if (! -e $rFunctionFile);
die "$0: unable to read R script $rFunctionFile\n" if (! -r $rFunctionFile);

mkdir $opt->{"out-dir"} || die "$0: unable to make directory ".$opt->{"out-dir"}.": $!\n";


#my $rLogBase = $opt->{"base-name"}.".R.log";
#my $rLogFile = $opt->{"out-dir"}."/".$rLogBase;
my $analysisDir =  $opt->{"analysis-dir"};
my $sigprocDir =  $opt->{"sigproc-dir"};
my $basecallerDir =  $opt->{"basecaller-dir"};
my $alignmentDir =  $opt->{"alignment-dir"};
my $acqDir = $opt->{"raw-dir"};
my $metaFile = $opt->{"analysis-dir"} . "/expMeta.dat";
my $analysisName = $opt->{"analysis-name"};
if (!defined($analysisName)) {
  my $metaFh;
  if($metaFh = FileHandle->new($metaFile)) {
    while (my $l = <$metaFh>) {
      chomp($l);
      if ($l =~ /Analysis Name = (\S+)/g) {
        $analysisName = $1;
      }
    }
  } else {
    warn "$0: unable to read $metaFile: $!\n";
  }
}
$analysisName = "NA" if (!defined($analysisName));


# Write and run the R script
my @flowsToPlot  = (0, 1, 2, 3, 4, 5, 6, 7);
#@flowsToPlot = (@flowsToPlot, 60,61,62,63, 120,121,122,123, 180,181,182,183);
my $numFlowsToPlot = @flowsToPlot;


if(!defined($opt->{"chip-type"})) {
  $opt->{"chip-type"} = &getChipType($analysisDir);
}

# First try separator mask
my $maskFile = "$sigprocDir/separator.mask.bin";

# Then try analysis mask
if (! (-e $maskFile)) {
    print "Maskfile 1: $maskFile does not exist.\n";
  $maskFile = "$sigprocDir/bfmask.bin";
}
# Then if nothing else exclusion mask
if (! (-e $maskFile) ) {
    print "Maskfile 2: $maskFile does not exist.\n";
  $maskFile = "$sigprocDir/analysis.bfmask.bin";
  if (! (-e $maskFile) ) {
     print "Maskfile 3: $maskFile does not exist.\n";
      if($opt->{"chip-type"} =~ /(3\d+)/) {
	  $opt->{"chip-type"} = $1;
      }
      $maskFile = "/opt/ion/config/exclusionMask_".$opt->{"chip-type"}.".bin";
  }
}
    print "Maskfile final: $maskFile .\n";
my $retVal = 0;
my $rTempFile = "";
if (! $opt->{"just-html"}) {
  if(length( $opt->{"active-lanes"} ) != 0){
    my $lane_str = $opt->{"active-lanes"};
    my @lanes = split('', $lane_str);
    foreach my $lane (@lanes) {
      print "**********Lane $lane\n";
      mkdir $opt->{"out-dir"}."/plots_lane_".$lane || die "$0: unable to make directory ".$opt->{"out-dir"}.": $!\n";
      my $rLogFile = $opt->{"out-dir"} . "/" . $opt->{"base-name"} . "_lane_" . $lane . ".Rout";
      my $tempDirRscript = &File::Temp::tempdir(CLEANUP => ($opt->{"debug"}==0));
      $rTempFile = "$tempDirRscript/rawPlots.R";
      my $rTempFh = FileHandle->new("> $rTempFile") || die "$0: unable to open R script file $rTempFile for write: $!\n";
      print $rTempFh "library(torrentR)\n";
      print $rTempFh "library(Hmisc)\n";
      print $rTempFh "library(reshape2)\n";
      print $rTempFh "library(ggplot2)\n";
      print $rTempFh "library(rjson)\n";
      print $rTempFh "library(RColorBrewer)\n";

      print $rTempFh "expName = '$analysisName'\n";
      print $rTempFh "analysisDir = '$analysisDir'\n";

      print $rTempFh "sigprocDir = '$sigprocDir'\n";
      print $rTempFh "basecallerDir = '$basecallerDir'\n";
      print $rTempFh "alignmentDir = '$alignmentDir'\n";

      print $rTempFh "datDir = '$acqDir'\n";
      print $rTempFh "plotDir = '" . $opt->{"out-dir"}."/plots_lane_".$lane . "'\n";
      print $rTempFh "maskFile = '$maskFile'\n";
      print $rTempFh "chipType = '" . $opt->{"chip-type"} . "'\n";
      print $rTempFh "multiLane = 'True" . "'\n";
      print $rTempFh "activeLanes = " . $lane . "\n";
      print $rTempFh "flows = c(" . join(",",@flowsToPlot) . ")\n";
      print $rTempFh "source(\"" . $rFunctionFile . "\")\n";
      open(RSCRIPT,$rScriptFile) || die "$0: failed to open R script file $rScriptFile: $!\n";
      while(<RSCRIPT>) {
        print $rTempFh $_;
      }
      {
        my $command = "cat $rTempFile";
        system($command)
      }

      close RSCRIPT;
      close $rTempFh;

      if($opt->{"debug"}) {
        print "Wrote R script to $rTempFile\n";
      } else {
        my $command = "R CMD BATCH --slave --no-save --no-timing $rTempFile $rLogFile";
        $retVal = system($command);
      }

      # Write the html output
      &writeHtml($rLogFile,$opt,$analysisName, $retVal, $rTempFile, $analysisName, $lane, @flowsToPlot);
    }
  }
}



exit(0);



sub writeHtml {
    my($rLogFile,$opt,$n,$retVal,$rFile, $analysisName, $laneID, @flowsToPlot) = @_;
    my $numPlotted = @flowsToPlot;
    my $cwd = &Cwd::getcwd();
    chdir $opt->{"out-dir"};

    my $htmlFile = $opt->{"base-name"}."_lane_". $laneID .".html";
    my $htmlFh = FileHandle->new("> $htmlFile") || die "$0: problem writing $htmlFile: $!\n";
    my $plotDir = "./plots_lane_".$laneID; #$opt->{"out-dir"}; 
	# $opt->{"base-name"}.".plot";
    my $title = $opt->{"base-name"};
    &writeHtmlHeader($title,$n." Raw Plots",$htmlFh);

	my @bfPlots = (
          #sprintf("%s/bfDetail-%s.png",$plotDir, $analysisName),
          sprintf("%s/regionMap-%s.png",$plotDir, $analysisName)
        );
	foreach my $plotFile (@bfPlots) {
		print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);
	}

	my @plotOrder = sort {($a%4)<=>($b%4)} @flowsToPlot;  # T's first, then A's, etc.
    my @nucs = ("T","A","C","G");
    for (my $i = 0; $i < $numPlotted; $i++) {
	my $flow = $plotOrder[$i];
	my $base = $nucs[$flow % 4];
	my $plotFile = sprintf("%s/raw-flow-%s-%d-%s.png",$plotDir, $base, $flow, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);
    }

    for (my $i = 0; $i < $numPlotted; $i++) {
	my $flow = $plotOrder[$i];
	my $base = $nucs[$flow % 4];
	my $plotFile = sprintf("%s/sub-%s-%d-bg-%s.png",$plotDir, $base, $flow, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);
    }

    for (my $i = 0; $i < $numPlotted; $i++) {
	my $flow = $plotOrder[$i];
	my $base = $nucs[$flow % 4];
	my $plotFile = sprintf("%s/raw-flow-zoom-%s-%d-%s.png",$plotDir, $base, $flow, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);
    }

    for (my $i = 0; $i < $numPlotted; $i++) {
	my $flow = $plotOrder[$i];
	my $base = $nucs[$flow % 4];
	my $plotFile = sprintf("%s/sub-%s-%d-bg-zoom-%s.png",$plotDir, $base, $flow, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);
    }
    
    my $plotFile = sprintf("%s/raw-prebeadfind-1-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/sub-prebeadfind-1-bg-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/raw-zoom-prebeadfind-1-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/sub-zoom-prebeadfind-1-bg-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/empty-sub-prebeadfind-1-bg-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);


     $plotFile = sprintf("%s/raw-prebeadfind-3-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/sub-prebeadfind-3-bg-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/raw-zoom-prebeadfind-3-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/sub-zoom-prebeadfind-3-bg-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/empty-sub-prebeadfind-3-bg-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

     $plotFile = sprintf("%s/raw-prebeadfind-4-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/sub-prebeadfind-4-bg-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/raw-zoom-prebeadfind-4-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/sub-zoom-prebeadfind-4-bg-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/raw-postbeadfind-3-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/sub-postbeadfind-3-bg-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/raw-zoom-postbeadfind-3-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    $plotFile = sprintf("%s/sub-zoom-postbeadfind-3-bg-%s.png",$plotDir, $analysisName);
	print $htmlFh "      <img src=\"$plotFile\"/>\n" if(-e $plotFile);

    # If we didn't get a good return value then print the log and quit
    if ($retVal != 0) {
	&rLogPrintHtml($htmlFh,$rLogFile,"  <h3>Error</h3>\n    Problem running R.\n\n");
	finishHtml($htmlFh);
	close($htmlFh);
	die "Problem running R. (R temp scrpt file: $rFile)";
    }

    finishHtml($htmlFh);
    system "cp", "$htmlFile", $opt->{"out-dir"};
  return;
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

sub writeHtmlHeader() {
    my $title = shift(@_);
    my $header = shift(@_);
    my $fh = shift(@_);
    print $fh "<head>\n";
    print $fh "  <title>$title</title>\n";
    print $fh "</head>\n";
    print $fh "<body>\n";
    print $fh "<h3>$header</h3>\n";
}

sub getChipType {
  my $logFile = shift(@_) . "/expMeta.dat";
  my $chipType = "NA";
  my $inFh = FileHandle->new($logFile) or return($chipType);
  while (my $line = <$inFh>) {
    chomp($line);
    $line =~ s/\"//g; 
    if ($line =~ /^Chip Type\s*=\s*(\w+)/) {
      $chipType = $1;
    }
  }
  return ($chipType);
}
