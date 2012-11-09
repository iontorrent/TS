#!/usr/bin/env perl
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

use strict;
use warnings;
use POSIX;
use File::Basename;
use Getopt::Long;

my $opt = {
  "readFile"               => [],
  "readFileBase"           => [],
  "readFileType"           => undef,
  "genome"                 => undef,
  "filter-length"          => undef,
  "out-base-name"          => undef,
  "start-slop"             => 0,
  "sample-size"            => 0,
  "max-plot-read-len"      => 400,
  "qscores"                => "7,10,17,20,30,47",
  "threads"                => &numCores(),
  "aligner"                => "tmap",
  "aligner-opts-rg"        => undef,                 # primary options (for -R to TMAP)
  "aligner-opts-extra"     => "stage1 map4", # this could include stage and algorithm options
  "aligner-opts-pairing"   => "-Q 2 -S 1 -b 200 -c 30 -d 3", # read pairing-specific options
  "mark-duplicates"        => 0,
  "bidirectional"          => 0, # if true then will pass tmap --bidirectional option
  "skip-alignStats"        => 0, # if true then will disable alignStats
  "aligner-format-version" => undef,
  "align-all-reads"        => 0,
  "genome-path"            => ["/referenceLibrary","/results/referenceLibrary","/opt/ion/referenceLibrary"],
  "sam-parsed"             => 0,
  "realign"                => 0,
  "help"                   => 0,
  "default-sample-size"    => 10000,
  "default-exclude-length" => 20,
  "logfile"                => "alignmentQC_out.txt",
  "output-dir"             => "./",
};

GetOptions(
  "i|input=s@"                => \$opt->{"readFile"},
  "g|genome=s"               => \$opt->{"genome"},
  "o|out-base-name=s"        => \$opt->{"out-base-name"},
  "s|start-slop=i"           => \$opt->{"start-slop"},
  "n|sample-size=i"          => \$opt->{"sample-size"},
  "l|filter-length=i"        => \$opt->{"filter-length"},
  "m|max-plot-read-len=s"    => \$opt->{"max-plot-read-len"},
  "q|qscores=s"              => \$opt->{"qscores"},
  "b|threads=i"              => \$opt->{"threads"},
  "d|aligner=s"              => \$opt->{"aligner"},
  "aligner-opts-rg=s"        => \$opt->{"aligner-opts-rg"},
  "aligner-opts-extra=s"     => \$opt->{"aligner-opts-extra"},
  "aligner-opts-pairing=s"   => \$opt->{"aligner-opts-pairing"},
  "mark-duplicates"          => \$opt->{"mark-duplicates"},
  "bidirectional"            => \$opt->{"bidirectional"},
  "skip-alignStats"          => \$opt->{"skip-alignStats"},
  "c|align-all-reads"        => \$opt->{"align-all-reads"},
  "a|genome-path=s@"         => \$opt->{"genome-path"},
  "p|sam-parsed"             => \$opt->{"sam-parsed"},
  "r|realign"                => \$opt->{"realign"},
  "aligner-format-version=s" => \$opt->{"aligner-format-version"},
  "h|help"                   => \$opt->{"help"},
  "output-dir=s"             => \$opt->{"output-dir"},
  "logfile=s"                => \$opt->{"logfile"},
);

&checkArgs($opt);

unlink($opt->{"logfile"}) if(-e $opt->{"logfile"});

# Determine how many reads are being aligned, make sure there is at least one
my @nReads = ();
foreach my $readFile (@{$opt->{"readFile"}}) {
  my $n = &getReadNumber($readFile,$opt->{"readFileType"});
  print STDOUT "WARNING: $0: no reads to align in $readFile\n" if($n==0);
  push(@nReads,$n);
  if($n != $nReads[0]) {
    print STDOUT "WARNING: $0: expected ".$nReads[0]." reads but found $n while parsing $readFile\n";
    exit(1);
  }
}
my $nReads = $nReads[0];
print "Aligning $nReads reads from ".join(" and ",@{$opt->{"readFile"}})."\n";

# Find the location of the genome index
my $indexVersion = defined($opt->{"aligner-format-version"}) ? $opt->{"aligner-format-version"} : &getIndexVersion();
my($refDir,$refInfo,$refFasta,$infoFile) = &findReference($opt->{"genome-path"},$opt->{"genome"},$opt->{"aligner"},$indexVersion);

if( -f 'explog.txt') {
  my $project = `grep ^Project: explog.txt|cut -f2- -d" "`; chomp $project;
  print "Checking if $refDir/project/$project exists\n";
  if ($project && -r "$refDir/project/$project") {
    print "Found hard-coded genomes, checking if readFile pattern matches\n";
    open(PROJ, "$refDir/project/$project");
    my $readFileName = $opt->{"readFile"}[0];
    $readFileName =~ s/.*\///;
    while (<PROJ>) {
      chomp;
      my ($pattern, $genome) = split /\t/;
      if ($readFileName =~ /^$pattern/) {
        print "Attemping to change genome from ".$opt->{"genome"}." to $genome for ".$opt->{"readFile"}[0]."\n";
        $opt->{"genome"} = $genome;
        my ($newrefDir,$newrefInfo,$newrefFasta,$newinfoFile) = &findReference($opt->{"genome-path"},$opt->{"genome"},$opt->{"aligner"},$indexVersion);
        if ($newrefDir) {
          ($refDir,$refInfo,$refFasta,$infoFile) = ($newrefDir,$newrefInfo,$newrefFasta,$newinfoFile);
          last;
          }
        else {
          print "Cannot find hard-coded genome $genome\n";
        }
      }
    }
    close(PROJ);
  }
}

print "Aligning to reference genome in $refDir\n";


# If out base name was not specified then derive from the base name of the input file
$opt->{"out-base-name"} = $opt->{"readFileBase"}[0] if(!defined($opt->{"out-base-name"}));

# If not specifying that all reads be aligned, if sample size is not set via command-line and if the genome
# info file has a specification, then set sampling according to it.
if((!$opt->{"align-all-reads"}) && ($opt->{"sample-size"} == 0) && exists($refInfo->{"read_sample_size"})) {
  $opt->{"sample-size"} = $refInfo->{"read_sample_size"};
}


# Implement random sampling, so long as align-all-reads has not been specified
if($opt->{"sample-size"} > 0) {
  if($opt->{"align-all-reads"}) {
    print "Request for sampling overridden by request to align all reads\n";
  } elsif($opt->{"sample-size"} >= $nReads) {
    print "Sample size is greater than number of reads, aligning everything\n";
    $opt->{"sample-size"} = 0;
  } else {
    die "$0: sampling is only possible when input file is in sff format\n" if($opt->{"readFileType"} ne "sff");
    print "Aligning random sample of ".$opt->{"sample-size"}." from total of $nReads reads\n";
    for(my $iReadFile=0; $iReadFile < @{$opt->{"readFile"}}; $iReadFile++) {
      my $readFile = $opt->{"readFile"}[$iReadFile];
      my $sampledSff = &extendSuffix($readFile,"sff","sampled");

      my $command1 = sprintf("SFFRandom -n %s -o %s %s 2>>%s",$opt->{"sample-size"},$sampledSff,$readFile,$opt->{"logfile"});
      die "$0: Failure during random sampling of reads\n" if(&executeSystemCall($command1));

      $opt->{"readFile"}[$iReadFile] = $sampledSff;
    }
  }
}


# Set the min length of alignments to retain if it wasn't specified on command line
if(!defined($opt->{"filter-length"})) {
  $opt->{"filter-length"} = exists($refInfo->{"read_exclude_length"}) ?  $refInfo->{"read_exclude_length"} : $opt->{"default-exclude-length"};
}


# Do the alignment
my $bamBase = $opt->{"output-dir"} . "/" . basename($opt->{"out-base-name"});
my $bamFile = $bamBase.".bam";
my $alignStartTime = time();
if($opt->{"aligner"} eq "tmap") {
  my $command = "tmap mapall";
  $command .= " -n ".$opt->{"threads"};
  $command .= " -f $refFasta";
  $command .= " -r ".join(" -r ",@{$opt->{"readFile"}});
  $command .= " -v";
  # For the moment it seems -Y cannot be used with tmap pairing options - need to fix
  if(@{$opt->{"readFile"}} > 1) {
    $command .= " ".$opt->{"aligner-opts-pairing"};
  } else {
    $command .= " -Y";
  }
  $command .= " --bidirectional" if($opt->{"bidirectional"});
  $command .= " ".$opt->{"aligner-opts-rg"} if(defined($opt->{"aligner-opts-rg"}));
  $command .= " -u -o 0"; # NB: random seed based on read and outputs SAM
  die if(!defined($opt->{"aligner-opts-extra"}));
  $command .= " ".$opt->{"aligner-opts-extra"};
  $command .= " 2>> ".$opt->{"logfile"};
  if(0 == $opt->{"mark-duplicates"}) {
      $command .= " | java -Xmx12G -jar /opt/picard/picard-tools-current/SortSam.jar I=/dev/stdin O=$bamFile QUIET=TRUE SO=coordinate";
  }
  else {
      $command .= " | java -Xmx12G -jar /opt/picard/picard-tools-current/SortSam.jar I=/dev/stdin O=$bamFile.tmp QUIET=TRUE SO=coordinate";
  }
  print "  $command\n";
  die "$0: Failure during read mapping\n" if(&executeSystemCall($command));
} else {
  die "$0: invalid aligner option: ".$opt->{"aligner"}."\n";
}
if(1 == $opt->{"mark-duplicates"}) {
    # TODO: how do we get the '/usr/local/bin/' path?
    my $command = "java -Xmx8G -jar /usr/local/bin/MarkDuplicates.jar I=$bamFile.tmp O=$bamFile M=$bamFile.markduplictes.metrics.txt AS=true VALIDATION_STRINGENCY=SILENT"; # FIXME
	print "  $command\n";
    die "$0: Failure during mark duplicates\n" if(&executeSystemCall($command));
    $command = "rm -v $bamFile.tmp";
	print "  $command\n";
    die "$0: Failure mark duplicates cleanup\n" if(&executeSystemCall($command));
}
die "$0: Failure during bam indexing\n" if(&executeSystemCall("samtools index $bamFile"));

my $alignStopTime = time();
my $alignTime = &ceil(($alignStopTime-$alignStartTime)/60);
print "Alignment time: $alignTime minutes\n";

if($opt->{"skip-alignStats"} == 0){
    # Post-process alignments to derive summary statistics
    my $postAlignStartTime = time();
    my $commandPostProcess = "alignStats";
    $commandPostProcess .= " --infile "      . $bamFile;
    $commandPostProcess .= " --genomeinfo "  . $infoFile;
    $commandPostProcess .= " --numThreads "  . $opt->{"threads"};
    $commandPostProcess .= " --qScores "     . $opt->{"qscores"};
    $commandPostProcess .= " --startslop "   . $opt->{"start-slop"};
    $commandPostProcess .= " --alignSummaryFilterLen " . $opt->{"filter-length"};
    $commandPostProcess .= " --alignSummaryMaxLen "    . $opt->{"max-plot-read-len"};
    $commandPostProcess .= " --errTableMaxLen "        . $opt->{"max-plot-read-len"};
    $commandPostProcess .= " --errTableTxtFile "       . "alignStats_err.txt";
    $commandPostProcess .= " --errTableJsonFile "      . "alignStats_err.json";
    $commandPostProcess .= " --samParsedFlag 1" if($opt->{"sam-parsed"});
    if($opt->{"sample-size"}) {
      $commandPostProcess .= " --totalReads $nReads";
      $commandPostProcess .= " --sampleSize ".$opt->{"sample-size"};
    }
    if($opt->{"output-dir"}) {
      $commandPostProcess .= " --outputDir ".$opt->{"output-dir"};
    }
    $commandPostProcess .= " 2>> ".$opt->{"logfile"};
    print "  $commandPostProcess\n";
    die "$0: Failure during alignment post-processing (take 1)\n" if(&executeSystemCall($commandPostProcess));
    my $postAlignStopTime = time();
    my $postAlignTime = &ceil(($postAlignStopTime-$postAlignStartTime)/60);
    print "Alignment post-processing time: $postAlignTime minutes\n";
}

# flowspace realignment, if requested
#if($opt->{"realign"}) {
#  my $command = "tmap sam2fs -S $samFile -l 2 -O 1 > ".$opt->{"out-base-name"}.".flowspace.sam";
#  die "$0: failure during flowspace realignment\n" if(&executeSystemCall($command));
#}

# Cleanup files
my @qScores = split(",",$opt->{"qscores"});
foreach my $qScore (@qScores) {
  my $covFile = $opt->{"out-base-name"}.".$qScore.coverage";
  unlink($covFile) if(-e $covFile);
}

exit(0);




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
    print STDOUT "$0: problem encountered running command \"$command\"\n";
    if($exeFail) {
      print STDOUT "Failed to execute command: $!\n";
    } elsif ($died) {
      print STDOUT sprintf("Child died with signal %d, %s coredump\n", $died,  $core ? 'with' : 'without');
    } else {
      print STDOUT "Child exited with value $exitCode\n";
    }
    $problem = 1;
  }

  return($problem);
}

sub checkArgs {
  my($opt) =@_;

  if($opt->{"help"}) {
    &usage();
    exit(0);
  }
  
  print "  printing alignmentQC.pl options:\n";
  for my $key (keys %$opt) {
    if(!defined($opt->{$key})) {
      print "    $key:\"undef\"\n";
    } elsif (! ref($opt->{$key})) {
      print "    $key: \"" . $opt->{$key} . "\"\n";
    } elsif (UNIVERSAL::isa($opt->{$key},'ARRAY')) {
      print "    $key: \"" . join("\",\n      \"",@{$opt->{$key}}) . "\"\n";
    } else {
      print "    $key: UNRECOGNIZED TYPE\n";
    }
  }
  
  # Check args for things that are not allowed
  my $badArgs = 0;
  if($opt->{"threads"} < 1) {
    $badArgs = 1;
    print STDOUT "ERROR: $0: must specify a positive number of threads\n";
  }
  my $nReadFiles = scalar @{$opt->{"readFile"}};
  if($nReadFiles < 1 || $nReadFiles > 2) {
    $badArgs = 1;
    print STDOUT "ERROR: $0: must specify at least one file of input reads with -i or --input option\n" if($nReadFiles < 1);
    print STDOUT "ERROR: $0: must specify at most two files of input reads with -i or --input option\n" if($nReadFiles > 2);
  } else {
    my $first=1;
    foreach my $readFile (@{$opt->{"readFile"}}) {
      if($readFile =~ /^(.+)\.(fasta|fastq|sff|bam)$/i) {
        my $readFileType = $2;
        push(@{$opt->{"readFileBase"}},$1);
        if($first) {
          $first = 0;
          $opt->{"readFileType"} = lc($readFileType);
        } elsif($opt->{"readFileType"} ne lc($readFileType)) {
          $badArgs = 1;
          print STDOUT "ERROR: $0: input read files should have same suffix\n";
        }
      } else {
        $badArgs = 1;
        print STDOUT "ERROR: $0: suffix of input reads filename $readFile should be one of (.fasta, .fastq, .sff)\n";
      }
    }
  }
  if(!defined($opt->{"genome"})) {
    $badArgs = 1;
    print STDOUT "ERROR: $0: must specify reference genome with -g or --genome option\n";
  }
  if($badArgs) {
    &usage();
    exit(1);
  }

  # Check args for things that might be problems
  if($opt->{"threads"} > &numCores()) {
    print STDOUT "WARNING: $0: number of threads is larger than number available, limiting.\n";
    $opt->{"threads"} = &numCores();
  }
}

sub usage () {
  print STDOUT << "EOF";

usage: $0
  Required args:
    -i,--input myReads         : File to align (fasta/fastq/sff/bam).  Use twice
                                 for paired read mapping
    -g,--genome mygenome       : Genome to which to align
  Optional args:
    -o,--out-base-name          : Base name for output files.  Default is to
                                  use same base name as (first) input file
    -l,--filter-length len      : alignments based on reads this short or
                                  shorter will be ignored when compiling
                                  alignment summary statistics.  If not
                                  specified will be taken from
                                  genome.info.txt, otherwise will be set to 20bp
    -s,--start-slop nBases      : Number of bases at 5' end of read that can be
                                  ignored when scoring alignments
    -n,--sample-size nReads     : Number of reads to sample.  If not specified
                                  will be taken from genome.info.txt
    -m,--max-plot-read-len len  : Maximum read length for read length histograms
                                  Default is 200bp
    -q,--qscores 10,20,30       : Comma-separated list of q-scores at which to
                                  evaluate lengths.  Default is 7,10,17,20,47
    -b,--threads nThreads       : Number of threads to use - default is the
                                  number of physical cores
    -d,--aligner tmap           : The aligner to use - currently only tmap
    --aligner-opts-rg opts      : SAM Read Group options aligner-specific options
                                  (ex. "-R" for TMAP)
    --aligner-opts-extra opts   : Additional extra options to pass to aligner 
                                  (if this is not specified, "stage1 map1 map2 map3" 
                                  will be used).
    --aligner-opts-pairing opts : Pairing options to supply to tmap when two input
                                  read files are provided (if this is not specified,
                                  "-Q 2 -S 1 -b 200 -c 100 -d 3 -L -l 5" will be used).
    --bidirectional             : Indicates that reads are bidirectional merged reads
                                  so that the BAM flag will be appropriately set
    --skip-alignStats           : Indicate whether alignStats will be skippeed
    -c,--align-all-reads        : Over-ride possible sampling, align all reads
    -a,--genome-path /dir       : Path in which references can be found.  Can
                                  be specified multiple times.  By default
                                  /opt/ion/referenceLibrary then
                                  /results/referenceLibrary are searched.
    -p,--sam-parsed             : Generate .sam.parsed file (not a standard
                                  format - do not rely on it)
    -r,--realign                : Create a flowalign.sam with flow-space
                                  realignment (experimental)
    --output-dir                : Output directory for stats output
    -h,--help                   : This help message
EOF
}

sub numCores {

  my $commandCore = "cat /proc/cpuinfo | grep \"core id\" | sort -u | wc -l";
  my $nCore = 0;
  die "$0: Failed to determine number of cores\n" if(&executeSystemCall($commandCore,\$nCore));
  chomp $nCore;

  my $commandProc = "cat /proc/cpuinfo | grep \"physical id\" | sort -u | wc -l";
  my $nProc = 0;
  die "$0: Failed to determine number of processors\n" if(&executeSystemCall($commandProc,\$nProc));
  chomp $nProc;

  my $nTotal = $nCore * $nProc;
  $nTotal = 1 if($nTotal < 1);

  return($nTotal);
}


sub findReference {
  my($genomePath,$genome,$aligner,$indexVersion) = @_;

  die "ERROR: $0: no base paths defined to search for reference library\n" if(@$genomePath == 0);

  my $dirName = "$indexVersion/$genome";
  my $found = 0;
  my $refLocation = undef;
  foreach my $baseDir (reverse @$genomePath) {
    $refLocation = "$baseDir/$dirName";
    if(-e $refLocation) {
      $found = 1;
      last;
    }
  }

  if(!$found) {
    print STDOUT "ERROR: $0: unable to find reference genome $dirName\n";
    print STDOUT "Searched in the following locations:\n";
    print STDOUT join("\n",map {"$_/$dirName"} reverse(@$genomePath))."\n";
    die;
  }

  my $fastaFile = "$refLocation/$genome.fasta";
  die "ERROR: $0: unable to find reference fasta file $fastaFile\n" if(! -e $fastaFile);

  my $infoFile = "$refLocation/$genome.info.txt";
  open(INFO,$infoFile) || die "$0: unable to open reference genome info file $infoFile: $!\n";
  my $info = {};
  my $lineCount = 0;
  while(<INFO>) {
    $lineCount++;
    next if(/^\s*#/ || /^\s*$/);
    chomp;
    my @F = split "\t";
    die "$0: bad format in line $lineCount of genome info file $infoFile\n" if(@F != 2);
    $info->{$F[0]} = $F[1];
  }
  close(INFO);


  return($refLocation,$info,$fastaFile,$infoFile);
}

sub getReadNumber {
  my($readFile,$fileType) = @_;

  my $nReads=0;
  if($fileType eq "fastq") {
    my $command = "cat $readFile | wc -l";
    die "$0: Failed to determine number of reads from $readFile\n" if(&executeSystemCall($command,\$nReads));
    chomp $nReads;
    $nReads =~ s/\s+//g;
    $nReads /= 4;
  } elsif($fileType eq "fasta") {
    my $command = "grep -c \"^>\" $readFile";
    die "$0: Failed to determine number of reads from $readFile\n" if(&executeSystemCall($command,\$nReads));
    chomp $nReads;
    $nReads =~ s/\s+//g;
  } elsif($fileType eq "sff") {
    my $command = "SFFSummary -q 0 -m 0 -s $readFile | grep \"^reads\" | cut -f2- -d\\ ";
    die "$0: Failed to determine number of reads from $readFile\n" if(&executeSystemCall($command,\$nReads));
    chomp $nReads;
    $nReads =~ s/\s+//g;
  } elsif($fileType eq "bam") {
    my $command = "samtools flagstat $readFile |head -1 | cut -f1 -d+";
    die "$0: Failed to determine number of reads from $readFile\n" if(&executeSystemCall($command,\$nReads));
    chomp $nReads;
    $nReads =~ s/\s+//g;
  } else {
    die "$0: don't know how to determine number of reads for file type $fileType\n";
  }

  return($nReads);
}

sub extendSuffix {
  my($file,$suffix,$extension) = @_;

  my $newFile = "$file.$extension";
  if($file =~ /^(.+)\.($suffix)$/i) {
    $newFile = "$1.$extension.$2";
  }

  return($newFile);
}

sub getIndexVersion {

  my $command = "tmap index --version";
  my $indexVersion = undef;
  die "$0: Problem encountered determining tmap format version\n" if(&executeSystemCall($command,\$indexVersion));
  chomp $indexVersion if(defined($indexVersion));

  return($indexVersion);
}
