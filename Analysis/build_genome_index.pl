#!/usr/bin/env perl
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

use strict;
use warnings;
use FileHandle;
use File::Copy;
use Cwd;
use Getopt::Long;
#use Path::Class qw / dir file /;

my $opt = {
  "autoFix"           => undef,
  "fastaFile"         => undef,
  "bedfileName"       => undef,
  "genomeNameShort"   => undef,
  "genomeNameLong"    => undef,
  "genomeVersion"     => undef,
  "compressed"        => undef,
  "tmapDir"           => "/usr/local/bin/",
  "picardDir"         => "/opt/picard/picard-tools-current/",
  "readSampleSize"    => 0,
  "readExcludeLength" => 0,
  "help"              => 0,
  "backwardCompat"    => 0,
};

GetOptions(
  "a|auto-fix"              => \$opt->{"autoFix"},
  "f|fasta=s"               => \$opt->{"fastaFile"},
  "m|reference-mask=s"      => \$opt->{"bedfileName"},
  "b|backward-compat"       => \$opt->{"backwardCompat"},
  "s|genome-name-short=s"   => \$opt->{"genomeNameShort"},
  "l|genome-name-long=s"    => \$opt->{"genomeNameLong"},
  "v|genome-version=s"      => \$opt->{"genomeVersion"},
  "c|compressed=s"          => \$opt->{"compressed"},
  "t|tmap-dir=s"            => \$opt->{"tmapDir"},
  "p|picard-dir=s"          => \$opt->{"picardDir"},
  "read-sample-size=i"      => \$opt->{"readSampleSize"},
  "read-exclude-length=i"   => \$opt->{"readExcludeLength"},
  "h|help"                  => \$opt->{"help"},
);
if($opt->{"help"}) {
  &usage();
  exit(0);
}

# args checking
my $badArgs = 0;
if(!defined($opt->{"fastaFile"})) {
  $badArgs = 1;
  print STDERR "ERROR: $0: must specify input fasta file with -f or --fasta option\n";
}
if(!defined($opt->{"genomeNameShort"})) {
  $badArgs = 1;
  print STDERR "ERROR: $0: must specify short genome name with -s or --genome-name-short option\n";
}
if(!defined($opt->{"genomeNameLong"})) {
  $badArgs = 1;
  print STDERR "ERROR: $0: must specify long genome name with -l or --genome-name-long option\n";
}
if(!defined($opt->{"genomeVersion"})) {
  $badArgs = 1;
  print STDERR "ERROR: $0: must specify genome version with -v or --genome-version option\n";
}
if($badArgs) {
  &usage();
  exit(1);
}

sub usage () {
  print STDERR << "EOF";

usage: $0
  Required args:
    -f,--fasta my.fasta                : Single fasta file of genome sequence(s)
    -s,--genome-name-short human       : Short form of genome name
    -l,--genome-name-long "H. Sapiens" : Long form of genome name
    -v,--genome-version hg19           : Genome version
  Optional args:
    -a,--auto-fix                      : Attempt to fix common fasta format issues
    -m,--reference-mask                : Masking bed file for reference sequence
    -c,--compressed hg19.fasta         : Expand a compressed zip. Requires name
    									 fasta as an argument.
    -t,--tmapDir /path/to/tmap         : Location of TMAP executable
    -p,--picard-dir /path/to/picard    : Location of Picard jar files
    --read-sample-size 10000           : Number of reads to randomly sample for
                                         alignment.  Default is to align all
    --read-exclude-length 20           : Alignments of this length or less will
                                         be ignored in summary statistics.
                                         Default value is 20.
    -b,--backward-compat               : Provide better backward compatibility
    -h,--help                          : This help message
EOF
}

my $outDir    = $opt->{"genomeNameShort"};
my $shortName = $opt->{"genomeNameShort"};
my $fastaFile = $opt->{"fastaFile"};
my $bedfile = "";

if (defined($opt->{"bedfileName"})) {
  $bedfile = $opt->{"bedfileName"};
}
#$opt->{'tmapDir'} = dir($opt->{'tmapDir'})->absolute();

#try to uncompress a zip file
if($opt->{"compressed"}) {
	&uncompress($opt->{"fastaFile"}, $opt->{"compressed"});
	#If the uncompression did not die, then use a resulting fasta from this point on
	$fastaFile = $opt->{"compressed"};
}


die "$0: input file $fastaFile does not exist\n" if(! -e $fastaFile);
die "$0: input file $fastaFile exists but is not readable\n" if(! -r $fastaFile);
if ($bedfile ne "") {
  die "$0: input file $bedfile does not exist\n" if(! -e $bedfile);
  die "$0: input file $bedfile exists but is not readable\n" if(! -r $bedfile);
}
my($origFastaFile,$origFastaChecksum,$fastaChecksum) = &checkInput($fastaFile,defined($opt->{"autoFix"})?1:0);
&prepareOutDir($outDir);
my $indexVersion = &getIndexVersion();
my $genomeLength = &makeIndex($outDir,$shortName,$fastaFile,$origFastaFile,$bedfile, $indexVersion);
&makeInfoFile($outDir,$shortName,$genomeLength,$indexVersion,$origFastaChecksum,$fastaChecksum,$opt);
`md5sum $outDir/* > $outDir/$shortName.md5sum.txt`;
exit(0);

sub checkInput {
  my($fastaFile, $autoFix) = @_;
  my $path = $0;
  $path =~ s/\/[^\/]+$//;
  my $command = "$path/validate_reference.pl -f $fastaFile 2>&1";
  $command .= " -u -o -a > $fastaFile.fix" if $autoFix;
  my $returnString = "";
  my $returnErrString = "";
  my $origFastaFile = "";
  my $origFastaChecksum = "";
  if(&executeSystemCall($command,\$returnString,\$returnErrString)) {
    print STDERR "$returnErrString\n";
    print STDERR "$returnString\n";
    print STDERR "$0: Invalid fasta file supplied, fix and retry.\n\n";
    die;
  }
  elsif($autoFix && $returnString) {
    print STDERR "$returnErrString\n";
    print STDERR "$returnString\n";
    $origFastaFile = "$fastaFile.orig";
    print STDERR "$0: $fastaFile is fixed and the original is kept as $origFastaFile \n\n";
    if(&executeSystemCall("mv $fastaFile $origFastaFile; mv $fastaFile.fix $fastaFile",\$returnString,\$returnErrString)) {
	print STDERR "$returnErrString\n";
        print STDERR "$returnString\n";
        print STDERR "$0: problem encountered when moving $fastaFile to $origFastaFile and $fastaFile.fix to $fastaFile.\n\n";
       	die;
    }
    $origFastaChecksum = &getChecksum($origFastaFile);
  }
  elsif($autoFix) {
    if(&executeSystemCall("rm -f $fastaFile.fix",\$returnString,\$returnErrString)) {
	print STDERR "$returnErrString\n";
        print STDERR "$returnString\n";
        print STDERR "$0: problem encountered when removing $fastaFile.fix.\n\n";
    }
  }
  my $fastaChecksum = &getChecksum($fastaFile);

  return($origFastaFile,$origFastaChecksum,$fastaChecksum);
}

sub getIndexVersion {

  my $command = "$opt->{'tmapDir'}/tmap index --version";
  my $indexVersion = undef;
  die "$0: Problem encountered determining tmap format version\n" if(&executeSystemCall($command,\$indexVersion));
  chomp $indexVersion if(defined($indexVersion));

  return($indexVersion);
}

sub makeInfoFile {
  my($outDir,$shortName,$genomeLength,$indexVersion,$origFastaChecksum,$fastaChecksum,$opt) = @_;

  my $infoFileName = "$outDir/$shortName.info.txt";
  my $infoFh = FileHandle->new("> $infoFileName") || die "$0: problem writing $infoFileName: $!\n";
  # Write the requried fields
  print $infoFh join("\t",("genome_name",   $opt->{"genomeNameLong"}))."\n";
  print $infoFh join("\t",("genome_version",$opt->{"genomeVersion"}))."\n";
  print $infoFh join("\t",("genome_length", $genomeLength))."\n";
  print $infoFh join("\t",("index_version", $indexVersion))."\n";
  print $infoFh join("\t",("fasta_md5checksum", $fastaChecksum))."\n";
  # Write the optional fields
  print $infoFh join("\t",("original_fasta_md5checksum", $origFastaChecksum ))."\n"  if($origFastaChecksum ne "");
  print $infoFh join("\t",("read_sample_size",   $opt->{"readSampleSize"}   ))."\n" if($opt->{"readSampleSize"}    > 0);
  print $infoFh join("\t",("read_exclude_length",$opt->{"readExcludeLength"}))."\n" if($opt->{"readExcludeLength"} > 0);
  close($infoFh);
}

sub makeIndex {
  my($outDir,$shortName,$fastaFile,$origFastaFile,$bedfile, $indexVersion) = @_;

  my $genomeLength = 0;

  my $cwd = &Cwd::getcwd();

  # copy fasta file to index dir
  
  my $fastaFileCopy = "$shortName.fasta";
  if ($bedfile ne "") {
    #ZZ, with bed file, generate trimmed version of fasta and a masked file.
    #chdir $outDir || die "$0: unable to chdir to output dir $outDir\n";
    #print STDOUT "Making masked and small reference files\n";
    my $tmapLogFile = "tmap.log";
    #my $command = "$opt->{'tmapDir'}/tmap mask -o $outDir/$fastaFileCopy.noMask $bedfile $fastaFile > $outDir/$fastaFileCopy 2>> $outDir/$tmapLogFile";
    my $command = "$opt->{'tmapDir'}/tmap mask $bedfile $fastaFile > $outDir/$fastaFileCopy 2>> $outDir/$tmapLogFile";
    die "$0: Problem encountered making tmap mask, check tmap log file $outDir/$tmapLogFile for details.\n" if(&executeSystemCall($command));
    print STDOUT "  ...tmap mask complete\n";
    #chdir $cwd || die "$0: unable to chdir to top-level dir $cwd\n";
    &copy($bedfile, "$outDir/maskfile_donot_remove.bed") || die  "$0: Problem copying $bedfile to $outDir/maskfile_donot_remove.bed: $!\n";
  } else {
    my $command = "cp $fastaFile $outDir/$fastaFileCopy";
    print STDOUT "Copying $fastaFile to $outDir/$fastaFileCopy...\n";
    &copy($fastaFile,"$outDir/$fastaFileCopy") || die "$0: Problem copying $fastaFile to $outDir/$fastaFileCopy: $!\n";
    print STDOUT "  ...copy complete\n";
  }

  # copy original fasta file to index dir, if it exists
  if($origFastaFile ne "") {
    my $origFastaFileCopy = "$shortName.fasta.orig";
    my $command = "cp $origFastaFile $outDir/$origFastaFileCopy";
    print STDOUT "Copying $origFastaFile to $outDir/$origFastaFileCopy...\n";
    &copy($origFastaFile,"$outDir/$origFastaFileCopy") || die "$0: Problem copying $origFastaFile to $outDir/$origFastaFileCopy: $!\n";
    print STDOUT "  ...copy complete\n";
  }

  # make tmap index
  chdir $outDir || die "$0: unable to chdir to output dir $outDir\n";
  print STDOUT "Making tmap index...\n";
  my $tmapLogFile = "tmap.log";
  my $backwardCompat = '';
  $backwardCompat = '-p' if ($opt-> {'backwardCompat'} ne 0);

  my $command = "$opt->{'tmapDir'}/tmap index -f $fastaFileCopy -v $backwardCompat 2>> $tmapLogFile";
  die "$0: Problem encountered making tmap index, check tmap log file $outDir/$tmapLogFile for details.\n" if(&executeSystemCall($command));
  print STDOUT "  ...tmap index complete\n";

  #if ($bedfile ne "") {
    #remove the masked file, replace it with the trimmed file
    #my $command = "mv -f $fastaFileCopy.noMask $fastaFileCopy";
    #die "can not rename file.\n" if(&executeSystemCall($command));
  #}  

  # make samtools index.  For now we're putting it in the same place
  # as the tmap index, though that's not a very natural place for it
  # and in future we should rearchitect how and where genome index
  # files are stored.
  print STDOUT "Making samtools index...\n";
  my $samtoolsLogFile = "samtools.log";
  $command = "samtools faidx $fastaFileCopy 2>> $samtoolsLogFile";
  die "$0: Problem encountered making samtools index, check samtools log file $outDir/$samtoolsLogFile for details.\n" if(&executeSystemCall($command));
  print STDOUT "  ...samtools index complete\n";

  # make picard .dict file and put it with tmap index
  my $picardErr = "";
  $picardErr = &makePicardDictFile($opt->{"picardDir"},$fastaFileCopy,$outDir) if(defined($opt->{"picardDir"})); 
  warn "WARNING: $picardErr\n" if($picardErr ne "");

  # determine genome length
  $command = "$opt->{'tmapDir'}/tmap refinfo $fastaFileCopy | grep \"^length\" | tail -n 1 | cut -f2";
  die "$0: Problem encountered determining genome length\n" if(&executeSystemCall($command,\$genomeLength));
  chomp $genomeLength;

  chdir $cwd || die "$0: unable to chdir to top-level dir $cwd\n";

  return $genomeLength;
}

sub prepareOutDir {
  my($outDir) = @_;

  die "$0: output directory $outDir already exits, aborting.\n" if(-e $outDir);
  mkdir $outDir || die "$0: unable to make directory $outDir: $!\n";
}

sub uncompress{
  my ($compressed, $zipped_fasta_name) = @_;
  
  #Test the zip file
  &zip_test(@_);

  #Run system calls to uncompress the data
  #unzip expects only 1 file to be in the .zip file and redirects the out to 
  #a file named <shortname>.fasta
  my $unzip_command = join(" ", "unzip", $compressed, $zipped_fasta_name);
  
  die "$0: Problem encountered with unziping file's system call\n" if(&executeSystemCall($unzip_command,\$compressed));

}

sub zip_test{
  #Make sure the zip file passes CRC checks
  my ($compressed) = @_;
  my $unzip_command_test = "unzip -t " . $compressed ;
  die "$0: Problem encountered with zip file CRC test - zip file may be corrupt\n" if(&executeSystemCall($unzip_command_test,\$compressed));
}

sub executeSystemCall {
  my ($command,$returnVal,$returnString) = @_;

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
  my $errString = "";
  if($exeFail || $died || $exitCode) {
    $errString .= "$0: problem encountered running command \"$command\"\n";
    if($exeFail) {
      $errString .= "Failed to execute command: $!\n";
    } elsif ($died) {
      $errString .= sprintf("Child died with signal %d, %s coredump\n", $died,  $core ? 'with' : 'without');
    } else {
      $errString .= "Child exited with value $exitCode\n";
    }
    $problem = 1;
  }
  $$returnString = $errString if(defined($returnString));

  return($problem);
}

sub makePicardDictFile {
  my($picardDir,$fastaFile,$outDir) = @_;

  my $jarFile = "$picardDir/picard.jar";
  if(! -e $picardDir) {
    return("$0: Picard dir $picardDir does not exist - unable to create Picard .dict file\n");
  } elsif(! -r $picardDir) {
    return("$0: Picard dir $picardDir exists but is not readable - unable to create Picard .dict file\n");
  } elsif(! -e $jarFile) {
    return("$0: jar file $jarFile does not exist - unable to create Picard .dict file\n");
  } elsif (! -r $jarFile) {
    return("$0: jar file $jarFile exists but is not readable - unable to create Picard .dict file\n");
  } else {
    print STDOUT "Making Picard .dict file...\n";
    my $picardDictLogFile = "CreateSequenceDictionary.log";
    my $dictFile = $fastaFile;
    $dictFile = $1 if($dictFile =~ /^(.+)\.fasta$/);
    $dictFile .= ".dict";
    my $command = "java -Xmx2g -jar $jarFile CreateSequenceDictionary R=$fastaFile O=$dictFile 2>> $picardDictLogFile ";
    return("$0: Problem encountered making Picard .dict file, check log file $outDir/$picardDictLogFile for details.\n") if(&executeSystemCall($command));
    print STDOUT "  ...Picard .dict file complete\n";
  }

  return("");
}


sub getChecksum {
  my ($inFile) = @_;

  my $command = "md5sum $inFile";
  my $returnString = "";
  my $returnErrString = "";
  if(&executeSystemCall($command,\$returnString,\$returnErrString)) {
    print STDERR "$returnString\n";
    print STDERR "$0: problem encountered getting checksum of $inFile - command is:\n$command\n";
    die;
  }
  $returnString = $1 if($returnString =~ /^(\S+)\s+/);

  return($returnString);
}
