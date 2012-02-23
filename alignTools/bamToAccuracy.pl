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
  "bam"         => undef,
  "max-length"  => 400,
  "out-dir"     => ".",
  "alignStats"  => "alignStats",
  "debug"       => 0,
  "help"        => 0,
};

GetOptions(
    "b|bam=s"        => \$opt->{"bam"},
    "m|max-length=i" => \$opt->{"max-length"},
    "o|out-dir=s"    => \$opt->{"out-dir"},
    "h|help"         => \$opt->{"help"},
    "d|debug"        => \$opt->{"debug"},
);

&usage() if(
    $opt->{"help"}
    || !defined($opt->{"bam"})
);

sub usage () {
    print STDERR << "EOF";

  usage: $0 -b file.bam
    -b,--bam file.bam      : The bam file to analyze
    -m,--max-length 400    : max read position for which to report error rate
    -o,--out-dir out       : directory in which to write results
    -h,--help              : this (help) message
    -d,--debug             : run in debug mode

  dependencies (satisifed on most torrent servers):
    alignStats must be in the \$PATH
    R and the R libraries torrentR and rjson must be installed

EOF
  exit(1);
}

mkpath $opt->{"out-dir"} || die "$0: unable to make directory ".$opt->{"out-dir"}.": $!\n";

die "$0: bam file ".$opt->{"bam"}." not found: $!\n" if(! -e $opt->{"bam"});
die "$0: bam file ".$opt->{"bam"}." not readable: $!\n" if(! -r $opt->{"bam"});

# Process the bam files
my $tempDir = &File::Temp::tempdir(CLEANUP => ($opt->{"debug"}==0));
my $alignStatsFile = &processBam($opt->{"bam"}, $tempDir, $opt->{"max-length"});

# Write the R script
my $rTempFile = "$tempDir/accuracyPlot.R";
my $rTempFh = FileHandle->new("> $rTempFile") || die "$0: unable to open R script file $rTempFile for write: $!\n";
my $outFileBase = basename($opt->{"bam"});
my $jsonResultsFile = $opt->{"out-dir"} . "/$outFileBase.accuracy.json";
my $txtResultsFile = $opt->{"out-dir"} . "/$outFileBase.accuracy.txt";
print $rTempFh "jsonResultsFile <- \"$jsonResultsFile\"\n";
print $rTempFh "txtResultsFile <- \"$txtResultsFile\"\n";
print $rTempFh "plotDir <- \"".$opt->{"out-dir"}."\"\n";
print $rTempFh "alignStatsFile <- \"$alignStatsFile\"\n";
print $rTempFh q~
pngWidth <- 400
pngHeight <- 400

library(torrentR)
library(rjson)

toPhred <- function(p) {
  p[p<=0] <- NA
  -10*log10(p)
}

x <- readTSV(alignStatsFile)
xNames <- names(x)
x <- matrix(unlist(x),ncol=length(x))
colnames(x) <- xNames
x <- cbind(x,"aligned"=x[,"nread"]-apply(x[,c("unalign","excluded","clipped")],1,sum))
x <- cbind(x,"errAtPosition"=x[,"totErr"]/x[,"aligned"])
x <- cbind(x,"errCumulative"=cumsum(x[,"totErr"])/cumsum(x[,"aligned"]))

alignLenLimit <- x[sum(x[,"aligned"]>0),"readLen"]
xLim <- c(1,max(alignLenLimit,150,na.rm=TRUE))

yLim <- c(10,25)
png(plotFile <- sprintf("%s/perBaseAccuracy.phred.png",plotDir),width=pngWidth,height=pngHeight)
plot(xLim,yLim,xlab="Position",ylab="Qscore",type="n")
abline(h=seq(min(yLim),max(yLim),by=1),lty=2,col="grey")
abline(v=seq(min(xLim),max(xLim),by=50),lty=2,col="grey")
lines(x[,"readLen"],  toPhred(x[,"errAtPosition"]  ),col="blue")
lines(x[,"readLen"],  toPhred(x[,"errCumulative"]  ),col="red",lty=2)
legend("topright",inset=0.01,c("per-base","cumulative"),lwd=2,lty=c(1,2),col=c("blue","red"))
title(sprintf("Accuracy (Phred scale)\nbased on %d aligned of %d total reads",x[1,"aligned"],x[1,"nread"]))
dev.off()
#system(sprintf("eog %s",plotFile))

yLim <- c(0.9,1)
png(plotFile <- sprintf("%s/perBaseAccuracy.linear.png",plotDir),width=pngWidth,height=pngHeight)
plot(xLim,yLim,xlab="Position",ylab="Accuracy",type="n")
abline(h=seq(min(yLim),max(yLim),by=1),lty=2,col="grey")
abline(v=seq(min(xLim),max(xLim),by=50),lty=2,col="grey")
lines(x[,"readLen"],  1-x[,"errAtPosition"], col="blue")
lines(x[,"readLen"],  1-x[,"errCumulative"], col="red",lty=2)
legend("topright",inset=0.01,c("per-base","cumulative"),lwd=2,lty=c(1,2),col=c("blue","red"))
title(sprintf("Accuracy (linear scale)\nbased on %d aligned of %d total reads",x[1,"aligned"],x[1,"nread"]))
dev.off()
#system(sprintf("eog %s",plotFile))

resultList <- list(
  "position"      = as.numeric(x[,"readLen"]),
  "errAtPosition" = as.numeric(x[,"errAtPosition"]),
  "errCumulative" = as.numeric(x[,"errCumulative"]),
  "aligned"       = as.numeric(x[,"aligned"])
)
write(toJSON(resultList),file=jsonResultsFile)
resultTable <- as.data.frame(resultList)
write.table(resultTable,file=txtResultsFile,quote=FALSE,sep="\t",row.names=FALSE)
~;

close $rTempFh;
my $rLogFile = "$tempDir/results.Rout";
my $retVal = 0;
if($opt->{"debug"}) {
  print "\n";
  print "Wrote R script to $rTempFile\n";
  print "tempDir is $tempDir\n";
} else {
  my $command = "R CMD BATCH --slave --no-save --no-timing $rTempFile $rLogFile";
  warn "$0: bad exit from R command: $command" if(system($command));
}
  
exit(0);

sub processBam {
  my($bamFile,$outDir,$maxLength) = @_;

  my $cwd = &getcwd();
  chdir $outDir;
  my $alignTableFile = "alignStats.txt";
  my $command = $opt->{"alignStats"};
  $command .= " -n 6";
  $command .= " -i $bamFile";
  $command .= " --3primeClip 10";
  $command .= " --alignSummaryFile $alignTableFile";
  $command .= " --alignSummaryMinLen 1";
  $command .= " --alignSummaryMaxLen $maxLength";
  $command .= " --alignSummaryLenStep 1";
  $command .= " --alignSummaryMaxErr 10";
  warn "$0: Failed to process bam file while in dir $outDir with the following command:\n$command\n" if(&executeSystemCall($command));
  my $resultFile = "$outDir/$alignTableFile";
  chdir $cwd;
  
  return($resultFile);
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
