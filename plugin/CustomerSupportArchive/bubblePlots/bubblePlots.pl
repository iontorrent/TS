#!/usr/bin/env perl
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Script to wrap around an R analysis to display various spatial signal distributions


use strict;
use warnings;
use FileHandle;
use Cwd;
use File::Temp;
use File::Basename;
use Getopt::Long;

my $opt = {
  "raw-dir"       => undef,
  "sigproc-dir"  => undef,
  "analysis-name" => undef,
  "out-dir"       => ".",
  "help"          => 0,
  "debug"         => 0,
  "just-html"     => 0,
  "base-name"     => "bubblePlots",
};

GetOptions(
    "r|raw-dir=s"          => \$opt->{"raw-dir"},
    "s|sigproc-dir=s"     => \$opt->{"sigproc-dir"},
    "n|analysis-name=s"    => \$opt->{"analysis-name"},
    "o|out-dir=s"          => \$opt->{"out-dir"},
    "j|just-html"          => \$opt->{"just-html"},
    "d|debug"              => \$opt->{"debug"},
    "h|help"               => \$opt->{"help"},
    );

&usage() if(
    $opt->{"help"}
    || !defined($opt->{"raw-dir"})
    || !defined($opt->{"out-dir"})
    );

sub usage () {
    print STDERR << "EOF";

    usage: $0 [-o outDir -a sigprocDir -d -h] -r myAcqDir -o out
     -r,--raw-dir acqdir        : directory with the acq files
     -s,--sigproc-dir myInput   : directory with signal processing results
     -o,--out-dir myOutput      : directory in which to write results
     -j,--just-html             : rerun html generation not plots
     -d,--debug                 : write R script and exit
     -h,--help                  : this (help) message
EOF
  exit(1);
}

mkdir $opt->{"out-dir"} || die "$0: unable to make directory ".$opt->{"out-dir"}.": $!\n";
my $workDir = sprintf("%s/to-delete",$opt->{"out-dir"});
mkdir $workDir  || die "$0: unable to make directory " . $workDir . ": $!\n";

my $plotSubDir = sprintf("%s",$opt->{"out-dir"});
#mkdir $plotSubDir || die "$0: unable to make directory " . $plotSubDir . ": $!\n";

my $rLogBase = $opt->{"base-name"}.".R.log";
my $rLogFile = $opt->{"out-dir"}."/".$rLogBase;
my $sigprocDir = defined($opt->{"sigproc-dir"}) ?  $opt->{"sigproc-dir"} : "";

my $acqDir = $opt->{"raw-dir"};

#generate analysis name in a clean fashion
#look it up in the expMeta.dat file
my $analysisName = $opt->{"analysis-name"};
if(defined($opt->{"sigproc-dir"})) {
  my $metaFile = $opt->{"sigproc-dir"} . "/expMeta.dat";
  my $metaFh = FileHandle->new($metaFile);
  if (!defined($analysisName) && (-r $metaFile)) {
      while (my $l = <$metaFh>) {
  	chomp($l);
  	if ($l =~ /Analysis Name = (\S+)/g) {
  	    $analysisName = $1;
  	}
      }
  }
}
$analysisName = "NA" if(!defined($analysisName));

# Write and run the R script


my($rTempFh,$rTempFile) = &File::Temp::tempfile();

my $retVal = 0;
if (! $opt->{"just-html"}) {
    #find the executable somewhere
    #important - this must have a path, or it crashes - remember when testing in the same directory!
    my ($tmpfile,$plugindir) = fileparse($0);

    my $executable=""; # not used in this script but a useful slot to pass to the Rscript
    &rScript($rTempFh, $sigprocDir, $acqDir, $analysisName,$plotSubDir, $executable,$plugindir,$opt->{"out-dir"});
    my $command = "R CMD BATCH --slave --no-save --no-timing $rTempFile $rLogFile";
    if($opt->{"debug"}) {
      print "$command\n";
      exit(0);
    }
    $retVal = system($command);
}

# Cleanup
my $outDir = $opt->{"out-dir"};
system("rm -rf $outDir/to-delete");


# Write the html output
&writeHtml($rLogBase,$opt,$analysisName, $retVal, $rTempFile, $analysisName);


######SubRoutines below here
sub writeHtml {
    my($rLogFile,$opt,$n,$retVal,$rFile,$analysisName) = @_;
  
    my $cwd = &Cwd::getcwd();
    chdir $opt->{"out-dir"};

    my $htmlFile = $opt->{"base-name"}.".html";
    my $htmlFh = FileHandle->new("> $htmlFile") || die "$0: problem writing $htmlFile: $!\n";
    my $plotDir = "."; #$opt->{"out-dir"}; 
# $opt->{"base-name"}.".plot";
    my $title = $opt->{"base-name"};
    &writeHtmlHeader($title,$n." Bubble Animated Plots",$htmlFh);


    # If we didn't get a good return value then print the log and quit
    if ($retVal != 0) {
	&rLogPrintHtml($htmlFh,$rLogFile,"  <h3>Error</h3>\n    Problem running R.\n\n");
	finishHtml($htmlFh);
	close($htmlFh);
	die "Problem running R. (R temp scrpt file: $rFile)";
    }
	my $summaryFile = sprintf("%s/%s.flow.all.csv",$plotDir,$analysisName);
	if(-e $summaryFile) {
	    print $htmlFh "      <a href=\"$summaryFile\">flow.all.csv</a>\n" if(-e $summaryFile);
	}

print $htmlFh "<h3> 5 bubble/outlier plots below: ever damaged,  log(sd), lagged(sd), sd(residuals), time-delay </h3>\n";
print $htmlFh "<h3> Revised: subsamples chip for more speed, samples flows out to end </h3>\n";

print $htmlFh "<h3> All wells obscured by bubbles</h3>\n";

my $sumplotFile = sprintf("%s/%s-outlier-by-flow.png",$plotDir,$analysisName);
	if(-e $sumplotFile) {
	    print $htmlFh "      <img src=\"$sumplotFile\" width=800 height=800 />\n" if(-e $sumplotFile);
	}


my $wplotFile = sprintf("%s/%s.outlier-all.png",$plotDir,$analysisName);
	if(-e $wplotFile) {
	    print $htmlFh "      <img src=\"$wplotFile\" width=800 height=800 />\n" if(-e $wplotFile);
	}


print $htmlFh "<h3> Plotting variation to look for unusual wells</h3>\n";

	my $bplotFile = sprintf("%s/%s.flow.gif",$plotDir, $analysisName);
	if(-e $bplotFile) {
	    print $htmlFh "      <img src=\"$bplotFile\" width=800 height=800 />\n" if(-e $bplotFile);
	}

print $htmlFh "<h3> Plotting lagged variation to look for unusual wells</h3>\n";

	my $lplotFile = sprintf("%s/%s.lag.flow.gif",$plotDir, $analysisName);
	if(-e $lplotFile) {
	    print $htmlFh "      <img src=\"$lplotFile\" width=800 height=800 />\n" if(-e $lplotFile);
	}

print $htmlFh "<h3> Plotting sd(residuals) to look for unusual wells</h3>\n";

	my $rplotFile = sprintf("%s/%s.flow.res.gif",$plotDir, $analysisName);
	if(-e $rplotFile) {
	    print $htmlFh "      <img src=\"$rplotFile\" width=800 height=800 />\n" if(-e $rplotFile);
	}

print $htmlFh "<h3> Plotting time-warp to look for unusual wells</h3>\n";

	my $tplotFile = sprintf("%s/%s.flow.time.gif",$plotDir, $analysisName);
	if(-e $tplotFile) {
	    print $htmlFh "      <img src=\"$tplotFile\" width=800 height=800 />\n" if(-e $tplotFile);
	}

  return;
}


sub finishHtml {
    my $fh = shift(@_);
    print $fh "</body>\n";
    print $fh "</html>\n";
    close $fh;
}

sub rScript {
    my($rFh,$sigprocDir,$acqDir,$analysisName,$outDir,$executable,$pluginloc,$masteroutDir) = @_;
    print $rFh "
expName = '$analysisName'
sigprocDir = '$sigprocDir'
executable = '$executable'
datDir = '$acqDir'
outDir = '$outDir'
masteroutDir = '$masteroutDir'
plotDir = '$outDir'
";

#read in R script from file in original plugin directory!
my $rscript_file = $pluginloc . "/bubble3.plots.R";
open(RSCRIPT, $rscript_file) || die("Could not open $rscript_file");
my @rscript=<RSCRIPT>;
close(RSCRIPT);

print $rFh @rscript;
    return;
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
