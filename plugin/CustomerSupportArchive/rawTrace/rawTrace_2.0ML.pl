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
  "raw-dir"       => undef,
  "analysis-name" => undef,
  "lib-key"       => "TCAG",
  "floworder"     => "TACGTACGTCTGAGCATCGATCGATGTACAGC",
  "chip-type"     => "unkown",
  "bamFile"       => undef,
  "out-dir"       => ".",
  "acq-limit"     => undef,
  "help"          => 0,
  "debug"         => 0,
  "html-only"     => 0,
  "plugin-name"   => "rawTrace_2.0ML",
  "alignStats"    => "alignStats",
  "thumbnail"     => "true",
  "active-lanes"  => ""
};

GetOptions(
    "a|analysis-dir=s"     => \$opt->{"analysis-dir"},
    "r|raw-dir=s"          => \$opt->{"raw-dir"},
    "n|analysis-name=s"    => \$opt->{"analysis-name"},
    "c|chip-type=s"        => \$opt->{"chip-type"},
    "o|out-dir=s"          => \$opt->{"out-dir"},
    "b|bam=s"              => \$opt->{"bamFile"},
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
    || !defined($opt->{"raw-dir"})
    || !defined($opt->{"analysis-dir"})
    || !defined($opt->{"out-dir"})
);

sub usage () {
    print STDERR << "EOF";

    usage: $0 [-o outDir -n analysisName] -a myAnalysisDir -r myRawDir
     -a,--analysis-dir   : directory with analysis results
     -r,--raw-dir        : directory with analysis results
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

mkpath $opt->{"out-dir"} || die "$0: unable to make directory ".$opt->{"out-dir"}.": $!\n";



if(length( $opt->{"active-lanes"} ) != 0){
  my $lane_str = $opt->{"active-lanes"};
  my @lanes = split('', $lane_str);

  my $sigProDir = $ENV{"SIGPROC_DIR"};
  foreach my $lane (@lanes) {
    print "**********Lane $lane\n";
    my $plotDir = "plots_lane_"."$lane";
    my $htmlFile = $opt->{"out-dir"}."/".$opt->{"plugin-name"}."_lane_"."$lane".".html";
    my $htmlFh = FileHandle->new("> $htmlFile") || die "$0: problem writing $htmlFile: $!\n";
    $opt->{"analysis-name"} = basename($opt->{"analysis-dir"}) if(!defined($opt->{"analysis-name"}));
    my $htmlTitle = $opt->{"plugin-name"} . " for " . $opt->{"analysis-name"} . "(Lane #" . $lane . ")";
    my $htmlHeaderLine = $htmlTitle;
    &writeHtmlHeader($htmlTitle,$htmlHeaderLine,$htmlFh);


    # Locate requried files, complain if not found
    my %requiredFiles = (
      "bfMaskFile"     => "$sigProDir/bfmask.bin",
    );
    my $problem = 0;
    my %fileLocation = ();
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
    # Convert bam file to Default.sam.parsed
    my $tempDir = &File::Temp::tempdir(CLEANUP => ($opt->{"debug"}==0));
    # if(defined($opt->{"bamFile"})) {
    #   $fileLocation{"samParsedFile"} = &bamToSamParsed($opt->{"bamFile"},$tempDir, $opt);
    # } else {
    #   $fileLocation{"samParsedFile"} = undef;
    # }

    # Write out the header of the R script
    my $tempDirRscript = &File::Temp::tempdir(CLEANUP => ($opt->{"debug"}==0));
    my $rTempFile = "$tempDirRscript/rawTrace_2.0ML.R";
    my $rTempFh = FileHandle->new("> $rTempFile") || die "$0: unable to open R script file $rTempFile for write: $!\n";
    foreach my $fileType (sort keys %fileLocation) {
      my $val = defined($fileLocation{$fileType}) ? "\"".$fileLocation{$fileType}."\"" : "NA";
      print $rTempFh "$fileType <- $val\n";
    }
    print $rTempFh "chipType <- \"".$opt->{"chip-type"}."\"\n";
    print $rTempFh "rawDir <- \"".$opt->{"raw-dir"}."\"\n";
    print $rTempFh "analysisName <- \"".$opt->{"analysis-name"}."\"\n";
    my $jsonResultsFile       = "results.json";
    print $rTempFh "jsonResultsFile <- \"".$opt->{"out-dir"}."/$jsonResultsFile\"\n";
    print $rTempFh "plotDir <- \"".$opt->{"out-dir"}."/$plotDir\"\n";
    if(defined($opt->{"active-lanes"})){
      print $rTempFh "lane <- \"".$lane."\"\n";
      }else{
        print $rTempFh "lane <- 0\n";
      }

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
    open TEST,"$rTempFile";
    # my $count = 0;
    while(<TEST>){
      chomp;
      print "$_\n";
      # $count = $count + 1;
      # if($count > 130){
        # last;
      # }
    }
    close TEST;

    my $rLogFile = $opt->{"out-dir"} . "/" . $opt->{"plugin-name"} . "_lane_"."$lane".".Rout";
    my $retVal = 0;
    if(!$opt->{"html-only"}) {
      if($opt->{"debug"}) {
        print "\n";
        print "Wrote R script to $rTempFile\n";
        print "tempDir is $tempDir\n";
      } else {
        my $command = "R CMD BATCH --slave --no-save --no-timing $rTempFile $rLogFile";
        $retVal = system($command);
      }
      &rLogPrintHtml($htmlFh,$rLogFile,"<br>Problem running R\n") if($retVal != 0);
    }



    my $blockHtmlFile = $opt->{"out-dir"}."/".$opt->{"plugin-name"}."_lane_"."$lane"."_block.html";
    my $blockHtmlFh = FileHandle->new("> $blockHtmlFile") || die "$0: problem writing $blockHtmlFile: $!\n";

    chdir $opt->{"out-dir"};

    &writeBlockHtml($blockHtmlFh,$plotDir,$opt);
    &finishHtml($blockHtmlFh);
    close $blockHtmlFh;

    &writeHtmlPlots($htmlFh,$plotDir);
    &writeHtmlSubPlots($htmlFh,"beadFind",$htmlTitle,"beadfind_pre",$plotDir);
    &writeHtmlSubPlots($htmlFh,"nucFlow", $htmlTitle,"acq",         $plotDir);

    if(!$opt->{"debug"}) {
      &finishHtml($htmlFh);
    }
  }# end of each lane

} # end of the main

sub writeHtmlSubPlots {
  my($htmlFh,$linkName,$htmlTitle,$plotFilePrefix,$plotDir) = @_;

  my $htmlSubDir = "html";
  mkdir $htmlSubDir || die "$0: probem making html sub dir $htmlSubDir: $!\n";
  my $htmlSubFile = "$htmlSubDir/$linkName.html";
  my $htmlSubFh = FileHandle->new("> $htmlSubFile") || die "$0: problem writing $htmlSubFile: $!\n";
  &writeHtmlHeader($htmlTitle,$htmlTitle,$htmlSubFh);
  my @plots = glob(sprintf("%s/%s*",$plotDir,$plotFilePrefix));
  foreach my $plotFile (@plots) {
    print $htmlSubFh "      <img src=\"../$plotFile\"/>\n" if(-e $plotFile);
  }
  &finishHtml($htmlSubFh);
  close $htmlSubFh;

  print $htmlFh "\n<br><a href=\"$htmlSubFile\">$linkName</a>\n";
}

sub writeHtmlPlots {
  my($htmlFh,$plotDir) = @_;

  # Determine region names, put bestBead at front of list
  my @files = glob(sprintf("%s/nucSteps.*.png",$plotDir));
  my @regions = sort map {s/\.png$//; s/.+nucSteps\.//; $_} @files;
  my $bestIndex = undef;
  for(my $i=0; $i<@regions; $i++) {
    $bestIndex = $i if($regions[$i] eq 'bestBead');
  }
  if(defined($bestIndex)) {
    my $temp = splice(@regions,$bestIndex,1);
    @regions = ($temp,@regions);
  }
  
  # Best key and zero-sub key
  print $htmlFh "<br>\n";
  my @plots = (
    sprintf("%s/bestKey.png",$plotDir),
    sprintf("%s/keyZeroSubBest.png",$plotDir)
  );
  foreach my $plotFile (@plots) {
    print $htmlFh "<img src=\"$plotFile\"/>\n" if(-e $plotFile);
  }

  # Nuc steps
  print $htmlFh "<hr>\n";
  print $htmlFh "<table border=2 cellpadding=5>\n";
  print $htmlFh "  <tr><th>Region</th><th>Nuc Step</th></tr>\n";
  foreach my $region (@regions) {
    my $plotFile = sprintf("%s/nucSteps.%s.png",$plotDir,$region);
    print $htmlFh "  <tr><td>$region</td>";
    print $htmlFh (-e $plotFile) ? "<td><img src=\"$plotFile\"/></td>" : "<td>NA</td>";
    print $htmlFh "</tr>\n";
  }

  # All the other plots
  print $htmlFh "<hr>\n";
  print $htmlFh "<table border=2 cellpadding=5>\n";
  my @plotTypes = ("nucStepSize","keyFlows","keyZeroSubBead");
  foreach my $plotType (@plotTypes) {
    print $htmlFh "  <tr>".join("",map {"<th>$_</th>"} @regions)."</tr>\n";
    print $htmlFh "  <tr>";
    foreach my $region (@regions) {
      my $baseName = sprintf("%s/%s.%s",$plotDir,$plotType,$region);
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

sub bamToSamParsed {
  my($bamFile,$outDir,$opt) = @_;

  my $samParsedFile = undef;

  my $cwd = &getcwd();
  chdir $outDir;
  if(!$opt->{"html-only"}) {
    my $command = $opt->{"alignStats"}." -n 6 -i $bamFile -p 1";
    if(&executeSystemCall($command)) {
      warn "$0: Failed to generate sam.parsed file from bam file while in dir $outDir with the following command:\n$command\n"
    } else {
      $samParsedFile = "$outDir/Default.sam.parsed";
    }
  }
  chdir $cwd;
  
  return($samParsedFile);
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
