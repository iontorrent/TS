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
  "lib-key"       => "TCAG",
  "floworder"     => "TACGTACGTCTGAGCATCGATCGATGTACAGC",
  "chip-type"     => "unkown",
  "out-dir"       => ".",
  "acq-limit"     => undef,
  "help"          => 0,
  "debug"         => 0,
  "html-only"     => 0,
  "plugin-name"   => "rawTrace",
  "alignStats"    => "alignStats",
  "thumbnail"     => "true",
  "active-lanes"  => ""
};

GetOptions(
    "a|analysis-dir=s"     => \$opt->{"analysis-dir"},
    "n|analysis-name=s"    => \$opt->{"analysis-name"},
    "c|chip-type=s"        => \$opt->{"chip-type"},
    "o|out-dir=s"          => \$opt->{"out-dir"},
    "l|lib-key=s"          => \$opt->{"lib-key"},
    "f|floworder=s"        => \$opt->{"floworder"},
    "h|help"               => \$opt->{"help"},
    "acq-limit=i"          => \$opt->{"acq-limit"},
    "debug"                => \$opt->{"debug"},
    "html-only"            => \$opt->{"html-only"},
    "t|thumbnail=s"        => \$opt->{"thumbnail"},
    "active-lanes=s"       => \$opt->{"active-lanes"},
);

&usage() if(
    $opt->{"help"}
    || !defined($opt->{"analysis-dir"})
    || !defined($opt->{"out-dir"})
);

sub usage () {
    print STDERR << "EOF";

    usage: $0 [-o outDir -n analysisName] -a myAnalysisDir -r myRawDir
     -a,--analysis-dir   : directory with analysis results
     -n,--analysis-name  : Name for plots
     -l,--lib-key        : Library key (TCAG)
     -f,--floworder      : Nuc flow order (TACGTACGTCTGAGCATCGATCGATGTACAGC)
     -c,--chip-type      : Type of chip ("unknown")
     -o,--out-dir        : directory in which to write results
     --acq-limit         : limit on number of acq flows to use (def: no limit)
     --debug             : Just write the R script and exit
     --html-only         : skip analysis and plotting, just make html
     -h,--help           : this (help) message
     -t,--thumbnail      : is thumnail or not
     --active-lanes      : active-lanes
EOF
  exit(1);
}

my $hostname = `hostname`;
print "HOSTNAME=$hostname\n";
# Locate the R script and make sure we can read it
my $rScriptFile = dirname($0) . "/" .$opt->{"plugin-name"} . ".R";
die "$0: unable to find R script $rScriptFile\n" if (! -e $rScriptFile);
die "$0: unable to read R script $rScriptFile\n" if (! -r $rScriptFile);

# Make dir
mkpath $opt->{"out-dir"} || die "$0: unable to make directory ".$opt->{"out-dir"}.": $!\n";
# my $htmlFile = $opt->{"out-dir"}."/".$opt->{"plugin-name"}.".html";
# my $htmlFh = FileHandle->new("> $htmlFile") || die "$0: problem writing $htmlFile: $!\n";
# $opt->{"analysis-name"} = basename($opt->{"analysis-dir"}) if(!defined($opt->{"analysis-name"}));
# my $htmlTitle = $opt->{"plugin-name"} . " for " . $opt->{"analysis-name"};
# my $htmlHeaderLine = $htmlTitle;
# &writeHtmlHeader($htmlTitle,$htmlHeaderLine,$htmlFh);

my $analysisDir = $opt->{"analysis-dir"};
my $bAnalysisDir=$ENV{"SIGPROC_DIR"};
# diretory for flow-by-flow

# block ids
my @x = (1, 4, 7, 10);
my @y = (6, 3, 1);
my $x_delta = 1288;
my $y_delta = 1332;
my $raw_dir = undef;
my $path = undef;

# if($opt->{"thumbnail"}){
#   print "thumbnail!";
#   $path = $ENV{'raw'};
#   $bAnalysisDir="$analysisDir/rawdata/onboard_results/sigproc_results/";
#   if ($ENV{"CHIP_LEVEL_ANALYSIS_PATH"}) {
#       $raw_dir = $ENV{"RAW_DATA_DIR"};
#       print "orig: $raw_dir";
#       $raw_dir =~ s/thumbnail$//g;
#       print "changed: $raw_dir";
#       $bAnalysisDir="$raw_dir/onboard_results/sigproc_results/";
#   }
# }


if(length( $opt->{"active-lanes"} ) != 0){
    my $lane_str = $opt->{"active-lanes"};
    my @lanes = split('', $lane_str);
    foreach my $lane (@lanes) {
      print "**********Lane $lane\n";
      my $htmlFile = $opt->{"out-dir"}."/".$opt->{"plugin-name"}."_lane_"."$lane".".html";
      my $htmlFh = FileHandle->new("> $htmlFile") || die "$0: problem writing $htmlFile: $!\n";
      $opt->{"analysis-name"} = basename($opt->{"analysis-dir"}) if(!defined($opt->{"analysis-name"}));
      my $htmlTitle = $opt->{"plugin-name"} . " for " . $opt->{"analysis-name"} . "(Lane #" . $lane . ")";
      my $htmlHeaderLine = $htmlTitle;
      &writeHtmlHeader($htmlTitle,$htmlHeaderLine,$htmlFh);

      my $x_block = $x[$lane - 1] * $x_delta;   
      my @y_block = map { $_ * $y_delta } @y;
      my $block_id = 0;
      my @blockDirs = ();
      my @nucStepDirs = ();
      my $nucStepDir = "";
      my @nucStepTimes = ();
      my $nucStepTime = "";
      my $nucStepRegions = "";
      my %fileLocation = ();
      my $plotDir = "plots_lane_"."$lane";
      my $sigDirff = $analysisDir;

      foreach (@y_block) {
        my $blockAnalysisDir = $bAnalysisDir."/block_X".$x_block."_Y".$_;
        $blockDirs[$block_id] = $blockAnalysisDir;

        if($block_id == 1){
          $sigDirff = $blockDirs[$block_id];
        }
        # print "/block_X".$x_block."_Y".$_."/sigproc_results"."\n";
        # Locate requried files, complain if not found
        my %requiredFiles = (
          "bfMaskFile"     => "$blockAnalysisDir/bfmask.bin",
        );
        my $problem = 0;
        
        foreach my $fileType (sort keys %requiredFiles) {
          my @fileList = glob($requiredFiles{$fileType});
          if(@fileList == 0) {
            my $errString = "Unable to find $fileType matching pattern $requiredFiles{$fileType}";
            print $htmlFh "\n<h3>WARNING</h3>\n$errString\n";
            print STDERR "$errString\n";
            $fileLocation{$fileType} = undef;
            $problem = 1;
          } elsif (@fileList > 1) {
            print $htmlFh "\n<h3>WARNING</h3>\nWhen looking for $fileType, found multiple matches to pattern $requiredFiles{$fileType}\n";
            print $htmlFh "\nFiles found are: " . join(", ",@fileList) . "\n";
            print $htmlFh "\nProceeding with first match, which might not be the right thing to do.\n";
            $fileLocation{$fileType} = $fileList[0];
          } elsif (! -e $fileList[0]) {
            my $errString = "File $requiredFiles{$fileType} does not exist\n";
            print $htmlFh "\n<h3>WARNING</h3>\n$errString\n";
            print STDERR "$errString\n";
            $problem = 1;
          } else {
            $fileLocation{$fileType} = $fileList[0];
          }
        }

        # Find nuc step dir and associated files
        my $errMsg="";
        ($nucStepDir,$nucStepTime,$nucStepRegions) = &findNucStepDir($blockAnalysisDir,\$problem,\$errMsg);
        $nucStepDirs[$block_id] = $nucStepDir;
        $nucStepTimes[$block_id] = $nucStepTime;
        print $htmlFh "\n<h3>WARNING</h3>\n$errMsg\n" if($errMsg ne "");

        if($problem) {
          &finishHtml($htmlFh);
          die "$0: problem finding input files.  Note this analysis will not run on a --from-wells report\n";
        }

        $block_id = $block_id + 1;
      }
        # Write out the header of the R script
        print "writing out the header of the R script\n";
        my $tempDirRscript = &File::Temp::tempdir(CLEANUP => ($opt->{"debug"}==0));
        my $rTempFile = "$tempDirRscript/rawTrace.R";
        my $rTempFh = FileHandle->new("> $rTempFile") || die "$0: unable to open R script file $rTempFile for write: $!\n";
        foreach my $fileType (sort keys %fileLocation) {
          my $val = defined($fileLocation{$fileType}) ? "\"".$fileLocation{$fileType}."\"" : "NA";
          print $rTempFh "$fileType <- $val\n";
        }
        print $rTempFh "nucStepDir     <- c(\"".join("\",\"",@nucStepDirs)."\")\n";
        print $rTempFh "nucStepTime    <- c(\"".join("\",\"",@nucStepTimes)."\")\n";
        print $rTempFh "nucStepRegions <- c(\"".join("\",\"",@$nucStepRegions)."\")\n";
        print $rTempFh "chipType <- \"".$opt->{"chip-type"}."\"\n";
        print $rTempFh "analysisName <- \"".$opt->{"analysis-name"}."\"\n";
        my $jsonResultsFile       = "results.json";
        print $rTempFh "jsonResultsFile <- \"".$opt->{"out-dir"}."/$jsonResultsFile\"\n";
        print $rTempFh "plotDir <- \"".$opt->{"out-dir"}."/$plotDir\"\n";
        if(defined($opt->{"lib-key"})) {
          print $rTempFh "libKey <- \"".$opt->{"lib-key"}."\"\n";
        } else {
          print $rTempFh "libKey <- NA\n";
        }
        if(defined($opt->{"acq-limit"})) {
          print $rTempFh "acqLimit <- ".$opt->{"acq-limit"}."\n";
        } else {
          print $rTempFh "acqLimit <- NA\n";
        }
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
        
        #check file
        # open TEST,"$rTempFile";
        # while(<TEST>){
        #   chomp;
        #   print "$_\n";
        # }
        # close TEST;
        
        my $rLogFile = $opt->{"out-dir"} . "/" . $opt->{"plugin-name"} . "_lane_"."$lane".".Rout";
        my $retVal = 0;
        if(!$opt->{"html-only"}) {
          if($opt->{"debug"}) {
            print "\n";
            print "Wrote R script to $rTempFile\n";
          } else {
            my $command = "R CMD BATCH --slave --no-save --no-timing $rTempFile $rLogFile" ;
            print $command."\n";
            $retVal = system($command);
          }
          &rLogPrintHtml($htmlFh,$rLogFile,"<br>Problem running R\n") if($retVal != 0);
        }



        my $blockHtmlFile = $opt->{"out-dir"}."/".$opt->{"plugin-name"}."_lane_"."$lane"."_block.html";
        print $blockHtmlFile."\n";
        
        # my $blockHtmlFh = FileHandle->new("> $blockHtmlFile") || die "$0: problem writing $blockHtmlFile: $!\n";

        chdir $opt->{"out-dir"};

        # &writeBlockHtml($blockHtmlFh,$plotDir,$opt);
        # &finishHtml($blockHtmlFh);
        # close $blockHtmlFh;

        &writeHtmlPlots($htmlFh,$plotDir);
        ## flow-by-flow
        my $rScriptff = dirname($0) . "/flow-by-flow.R";
        my $rffLogFile = $opt->{"out-dir"} . "/" . $opt->{"plugin-name"} . "_ff_lane_"."$lane".".Rout";
        my $command = "Rscript --vanilla $rScriptff $sigDirff $rffLogFile";
        print $command;
        $retVal = system($command);
    }
    # if(!$opt->{"debug"}) {
    #   &finishHtml($htmlFh);
    # }
}
  

sub writeHtmlPlots {
  my($htmlFh,$plotDir) = @_;

  # Determine region names, put bestBead at front of list
  my @files = glob(sprintf("%s/nucSteps.*.empty.png",$plotDir));
  my @regions = sort map {s/\.empty\.png$//; s/.+nucSteps\.//; $_} @files;

  # Sub-dirs for the regions
  my $htmlSubDir = "html_"."$plotDir";
  mkdir $htmlSubDir || die "$0: probem making html sub dir $htmlSubDir: $!\n";
  print $htmlFh "\n";
  foreach my $region (@regions) {
    my $htmlSubFile = "$htmlSubDir/region.$region.html";
    my $htmlSubFh = FileHandle->new("> $htmlSubFile") || die "$0: problem writing $htmlSubFile: $!\n";
    &writeHtmlHeader($region,$region,$htmlSubFh);
    &writeSubHtml($region,$htmlSubFh,$plotDir);
    &finishHtml($htmlSubFh);
    close $htmlSubFh;
    print $htmlFh "<br><a href=\"$htmlSubFile\">$region details</a>\n";
  }
  print $htmlFh "\n";

  print $htmlFh "<hr>\n";
  print $htmlFh "<table border=2 cellpadding=5>\n";
  my @plotTypes  = ("stepSize","keyBkgSub","keyFlows","flowEmptySd","flowResidual");
  my @plotSuffix = (".empty"  ,""         ,""        ,""           ,".nbrsub"     );
  for(my $iPlot=0; $iPlot < @plotTypes; $iPlot++) {
    my $plotType = $plotTypes[$iPlot];
    my $plotSuffix = $plotSuffix[$iPlot];
    print $htmlFh "  <tr>".join("",map {"<th>$_</th>"} @regions)."</tr>\n";
    print $htmlFh "  <tr>";
    foreach my $region (@regions) {
      my $baseName = sprintf("%s/%s.%s%s",$plotDir,$plotType,$region,$plotSuffix);
      if(! -e "$baseName.png") {
        print $htmlFh "<td>NA</td>";
      } else {
        if(! -e "$baseName.json") {
          print $htmlFh "<td><img src=\"$baseName.png\"/></td>";
        } else {
          print $htmlFh "<td><a href=\"$baseName.json\"><img src=\"$baseName.png\"/></a></td>";
        }
      }
    }
    print $htmlFh "  <tr>\n";
  }
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

sub lockHtml {
  my($fh,$plotDir,$opt) = @_;
  my $htmlTitle = $opt->{"plugin-name"} . " block_html for " . $opt->{"analysis-name"};
  &writeHtmlHeader($htmlTitle,"",$fh);

  my $plotFile;

  my $imgHeight=250;
  $plotFile = sprintf("%s/nucSteps.middle.empty.png",$plotDir);
  print $fh "  <img src=\"$plotFile\" height=$imgHeight />\n" if(-e $plotFile);
  $plotFile = sprintf("%s/stepSize.middle.empty.png",$plotDir);
  print $fh "  <img src=\"$plotFile\" height=$imgHeight />\n" if(-e $plotFile);
  print $fh "  <br>\n";

  $plotFile = sprintf("%s/keyBkgSub.middle.png",$plotDir);
  print $fh "  <img src=\"$plotFile\" height=$imgHeight />\n" if(-e $plotFile);
  $plotFile = sprintf("%s/flowResidual.middle.nbrsub.png",$plotDir);
  print $fh "  <img src=\"$plotFile\" height=$imgHeight />\n" if(-e $plotFile);
  $plotFile = sprintf("%s/flowEmptySd.middle.png",$plotDir);
  print $fh "  <img src=\"$plotFile\" height=$imgHeight />\n" if(-e $plotFile);
}

sub findNucStepDir {
  my ($analysisDir,$problem,$errMsg) = @_;

  my $nucStepDir = "$analysisDir/NucStep";
  if(! -e $nucStepDir) {
    $$errMsg = "$0: did not find nuc step dir $nucStepDir: $!\n";
    $$problem = 1;
    return;
  }
  if(! -r $nucStepDir) {
    $$errMsg = "$0: unable to read nuc step dir $nucStepDir: $!\n";
    $$problem = 1;
    return;
  }

  my $nucStepTime = "$nucStepDir/NucStep_frametime.txt";
  if(! -e $nucStepTime) {
    $$errMsg = "$0: did not find nuc step timing file $nucStepTime: $0\n";
    $$problem = 1;
    return;
  }
  if(! -r $nucStepTime) {
    $$errMsg = "$0: unable to read nuc step timing file $nucStepTime: $0\n";
    $$problem = 1;
    return;
  }
  
  my $nucStepFilePattern = "$nucStepDir/NucStep_*_step.txt";
  my @nucStepFiles = glob($nucStepFilePattern);
  if(@nucStepFiles==0) {
    $$errMsg = "$0: found no nuc step files matching pattern $nucStepFilePattern\n";
    $$problem = 1;
    return;
  }
  my @nucStepRegions = ();
  foreach my $file (@nucStepFiles) {
    if($file =~ /\/NucStep_(.+)_step.txt$/) {
      push(@nucStepRegions,$1);
    }
  }
  @nucStepRegions = sort(@nucStepRegions);

  return($nucStepDir,$nucStepTime,\@nucStepRegions);
}

sub writeSubHtml {
  my($region,$htmlSubFh,$plotDir) = @_;

  print $htmlSubFh "<hr>\n";
  print $htmlSubFh "<table border=2 cellpadding=5>\n";
  my $imgHeight=300;
  my @plotTypes = ("stepSize", "flowResidual", "nucSteps", "nucSmooth");
  my @wellTypes = ("bead","empty","nbrsub");
  foreach my $wellType (@wellTypes) {
    print $htmlSubFh "<th>$wellType</th>";
    foreach my $plotType (@plotTypes) {
      my $baseName = sprintf("%s/%s.%s.%s",$plotDir,$plotType,$region,$wellType);
      if(! -e "$baseName.png") {
        print $htmlSubFh "<td>NA</td>";
      } else {
        if(! -e "$baseName.json") {
          print $htmlSubFh "<td><img src=\"../$baseName.png\" height=$imgHeight /></td>";
        } else {
          print $htmlSubFh "<td><a href=\"../$baseName.json\"><img src=\"$baseName.png\" height=$imgHeight /></a></td>";
        }
      }
    }
    print $htmlSubFh "  <tr>\n";
  }
  print $htmlSubFh "</table>\n";
}

sub writeBlockHtml {
  my($fh,$plotDir,$opt) = @_;
  my $htmlTitle = $opt->{"plugin-name"} . " block_html for " . $opt->{"analysis-name"};
  &writeHtmlHeader($htmlTitle,"",$fh);

  my $plotFile;

  my $imgHeight=250;
  $plotFile = sprintf("%s/nucSteps.Middle.png",$plotDir);
  print $fh "  <img src=\"$plotFile\" height=$imgHeight /><br>\n" if(-e $plotFile);

  $imgHeight=250;
  $plotFile = sprintf("%s/nucStepSize.Middle.png",$plotDir);
  print $fh "  <img src=\"$plotFile\" height=$imgHeight />\n" if(-e $plotFile);
  $plotFile = sprintf("%s/keyZeroSubBead.Middle.png",$plotDir);
  print $fh "  <img src=\"$plotFile\" height=$imgHeight />\n" if(-e $plotFile);
  $plotFile = sprintf("%s/keyZeroSubBest.png",$plotDir);
  print $fh "  <img src=\"$plotFile\" height=$imgHeight />\n" if(-e $plotFile);
}

