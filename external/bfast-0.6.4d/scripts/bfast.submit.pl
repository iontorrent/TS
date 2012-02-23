#!/usr/bin/perl

# Please see the LICENSE accompanying this distribution for 
# details.  Report all bugs to nhomer@cs.ucla.edu or 
# bfast-help@lists.sourceforge.net.  For documentation, use the
# -man option.

use strict;
use warnings FATAL => qw( all );
use File::Path;
use XML::Simple; 
use Data::Dumper;
use Getopt::Long;
use Pod::Usage;
use File::Path; # for directory creation
use Cwd;

# TODO:
# - input values could be recognizable strings, such as 10GB for maxMemory...

my %QUEUETYPES = ("SGE" => 0, "PBS" => 1);
my %SPACE = ("NT" => 0, "CS" => 1);
my %TIMING = ("ON" => 1);
my %LOCALALIGNMENTTYPE = ("GAPPED" => 0, "UNGAPPED" => 1);
my %STRAND = ("BOTH" => 0, "FORWARD" => 1, "REVERSE" => 2);
my %STARTSTEP = ("match" => 0, "localalign" => 1, "postprocess" => 2, "sam" => 3);
my %OUTTYPES = (0 => "baf", 1 => "maf", 2 => "gff", 3 => "sam");
my %COMPRESSION = ("none" => ".fastq", "gz" => "-z", "bz2" => "-z");
my $FAKEQSUBID = 0;

use constant {
	OPTIONAL => 0,
	REQUIRED => 1,
	BREAKLINE => "************************************************************\n",
	MERGE_LOG_BASE => 8,
	QSUBNOJOB => "QSUBNOJOB"
};

my $config;
my ($man, $print_schema, $help, $quiet, $start_step, $dryrun) = (0, 0, 0, 0, "match", 0);
my $version = "0.1.1";

GetOptions('help|?' => \$help, 
	man => \$man, 
	'schema' => \$print_schema, 
	'quiet' => \$quiet, 
	'startstep=s' => \$start_step,
	'dryrun' => \$dryrun,
	'config=s' => \$config)
	or pod2usage(1);
Schema() if ($print_schema);
pod2usage(-exitstatus => 0, -verbose => 2) if $man;
pod2usage(1) if ($help or !defined($config));

if(!defined($STARTSTEP{$start_step})) {
	print STDERR "Error. Illegal value to the option -startstep.\n";
	pod2usage(1);
}
$start_step = $STARTSTEP{$start_step};

if(!$quiet) {
	print STDOUT BREAKLINE;
}

# Read in from the config file
my $xml = new XML::Simple;
my $data = $xml->XMLin("$config");

# Validate data
ValidateData($data);

# Submit jobs
CreateJobs($data, $quiet, $start_step, $dryrun);

if(!$quiet) {
	print STDOUT BREAKLINE;
}

sub Schema {
	# print the schema
	my $schema = <<END;
	<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="bfastConfig">
	<xs:complexType>
	  <xs:sequence>
		<xs:element name="globalOptions">
		  <xs:complexType>
			<xs:sequence>
			  <xs:element name="bfastBin" type="directoryPath"/>
			  <xs:element name="samtoolsBin" type="directoryPath"/>
			  <xs:element name="picardBin" type="directoryPath"/>
			  <xs:element name="javaBin" type="directoryPath"/>
			  <xs:element name="qsubBin" type="directoryPath"/>
			  <xs:element name="fastaFileName" type="filePath" use="required"/>
			  <xs:element name="runDirectory" type="directoryPath" use="required">
			  <xs:element name="readsDirectory" type="directoryPath" use="required"/>
			  <xs:element name="outputDirectory" type="directoryPath" use="required"/>
			  <xs:element name="tmpDirectory" type="directoryPath" use="required"/>
			  <xs:element name="outputID" type="xs:string" use="required"/>
			  <xs:element name="cleanUsedIntermediateFiles" type="xs:integer" use="optional"/>
			  <xs:element name="numReadsPerFASTQ" type="positiveInteger">
				<xs:complexType>
				  <xs:attribute name="matchSplit" type="positiveInteger" use="required"/>
				  <xs:attribute name="localalignSplit" type="positiveInteger" use="required"/>
				</xs:complexType>
			  </xs:element>
			  <xs:element name="timing">
				<xs:simpleType>
				  <xs:restriction base="xs:string">
					<xs:enumeration value="ON"/>
				  </xs:restriction>
				</xs:simpleType>
			  </xs:element>
			  <xs:element name="queueType" use="required">
				<xs:simpleType>
				  <xs:restriction base="xs:string">
					<xs:enumeration value="SGE"/>
					<xs:enumeration value="PBS"/>
				  </xs:restriction>
				</xs:simpleType>
			  </xs:element>
			  <xs:element name="space" use="required">
				<xs:simpleType>
				  <xs:restriction base="xs:string">
					<xs:enumeration value="NT"/>
					<xs:enumeration value="CS"/>
				  </xs:restriction>
				</xs:simpleType>
			  </xs:element>
			</xs:sequence>
		  </xs:complexType>
		</xs:element>
		<xs:element name="matchOptions">
		  <xs:complexType>
			<xs:sequence>
			  <xs:element name="mainIndexes" type="xs:string"/>
			  <xs:element name="secondaryIndexes" type="xs:string"/>
			  <xs:element name="offsets" type="xs:string"/>
			  <xs:element name="loadAllIndexes" type="xs:integer"/>
			  <xs:element name="readsCompression">
				<xs:simpleType>
				  <xs:restriction base="xs:string">
					<xs:enumeration value="none"/>
					<xs:enumeration value="gz"/>
					<xs:enumeration value="bz2"/>
				  </xs:restriction>
				</xs:simpleType>
			  </xs:element>
			  <xs:element name="keySize" type="positiveInteger"/>
			  <xs:element name="maxKeyMatches" type="positiveInteger"/>
			  <xs:element name="maxNumMatches" type="positiveInteger"/>
			  <xs:element name="strand">
				<xs:simpleType>
				  <xs:restriction base="xs:string">
					<xs:enumeration value="BOTH"/>
					<xs:enumeration value="FORWARD"/>
					<xs:enumeration value="REVERSE"/>
				  </xs:restriction>
				</xs:simpleType>
			  </xs:element>
			  <xs:element name="threads" type="positiveInteger"/>
			  <xs:element name="queueLength" type="positiveInteger"/>
			  <xs:element name="mergeSeparate" type="xs:integer"/>
			  <xs:element name="qsubQueue" type="xs:string"/>
			  <xs:element name="qsubArgs" type="xs:string"/>
			</xs:sequence>
		  </xs:complexType>
		</xs:element>
		<xs:element name="localalignOptions">
		  <xs:complexType>
			<xs:sequence>
			  <xs:element name="scoringMatrix" type="filePath"/>
			  <xs:element name="ungapped" type="xs:integer"/>
			  <xs:element name="unconstrained" type="xs:integer"/>
			  <xs:element name="offset" type="nonNegativeInteger"/>
			  <xs:element name="maxNumMatches" type="positiveInteger"/>
			  <xs:element name="mismatchQuality" type="positiveInteger"/>
			  <xs:element name="threads" type="positiveInteger"/>
			  <xs:element name="queueLength" type="positiveInteger"/>
			  <xs:element name="pairedEndLength" type="xs:integer"/>
			  <xs:element name="mirrorType" type="xs:integer"/>
			  <xs:element name="forceMirror">
				<xs:simpleType>
				  <xs:restriction base="xs:integer">
					<xs:minInclusive value="0"/>
					<xs:maxInclusive value="3"/>
				  </xs:restriction>
				</xs:simpleType>
			  </xs:element>
			  <xs:element name="qsubQueue" type="xs:string"/>
			  <xs:element name="qsubArgs" type="xs:string"/>
			</xs:sequence>
		  </xs:complexType>
		</xs:element>
		<xs:element name="postprocessOptions">
		  <xs:complexType>
			<xs:sequence>
			  <xs:element name="algorithm">
				<xs:simpleType>
				  <xs:restriction base="xs:integer">
					<xs:minInclusive value="0"/>
					<xs:maxInclusive value="3"/>
				  </xs:restriction>
				</xs:simpleType>
			  </xs:element>
			  <xs:element name="unpaired" type="xs:integer"/>
			  <xs:element name="reversePaired" type="xs:integer"/>
			  <xs:element name="outputFormat" type="xs:integer"/>
			  <xs:element name="unmappedFile" type="xs:string"/>
			  <xs:element name="readGroupFile" type="xs:string"/>
			  <xs:element name="threads" type="positiveInteger"/>
			  <xs:element name="queueLength" type="positiveInteger"/>
			  <xs:element name="qsubQueue" type="xs:string"/>
			  <xs:element name="qsubArgs" type="xs:string"/>
			</xs:sequence>
		  </xs:complexType>
		</xs:element>
		<xs:element name="samOptions">
		  <xs:complexType>
			<xs:sequence>
			  <xs:element name="samtools" type="integer" use="required"/>
			  <xs:element name="maximumMemory" type="positiveInteger"/>
			  <xs:element name="qsubQueue" type="xs:string"/>
			  <xs:element name="qsubArgs" type="xs:string"/>
			</xs:sequence>
		  </xs:complexType>
		</xs:element>
	  </xs:sequence>
	</xs:complexType>
  </xs:element>
  <xs:simpleType name="filePath">
	<xs:restriction base="xs:string">
	  <xs:pattern value="\\S+"/>
	</xs:restriction>
  </xs:simpleType>
  <xs:simpleType name="directoryPath">
	<xs:restriction base="xs:string">
	  <xs:pattern value="\\S+/"/>
	</xs:restriction>
  </xs:simpleType>
  <xs:simpleType name="nonNegativeInteger">
	<xs:restriction base="xs:integer">
	  <xs:minInclusive value="0"/>
	</xs:restriction>
  </xs:simpleType>
  <xs:simpleType name="positiveInteger">
	<xs:restriction base="xs:integer">
	  <xs:minInclusive value="1"/>
	</xs:restriction>
  </xs:simpleType>
</xs:schema>
END

	print STDOUT $schema;
	exit 1;
}

sub ValidateData {
	my $data = shift;

	# global options
	die("The global options were not found.\n") unless (defined($data->{'globalOptions'})); 
	ValidatePath($data->{'globalOptions'},         'bfastBin',                                 OPTIONAL); 
	ValidatePath($data->{'globalOptions'},         'samtoolsBin',                              OPTIONAL); 
	ValidatePath($data->{'globalOptions'},         'picardBin',                                OPTIONAL); 
	ValidatePath($data->{'globalOptions'},         'javaBin',                                OPTIONAL); 
	ValidatePath($data->{'globalOptions'},         'qsubBin',                                  OPTIONAL); 
	ValidateOptions($data->{'globalOptions'},      'queueType',          \%QUEUETYPES,         REQUIRED);
	ValidateOptions($data->{'globalOptions'},      'space',              \%SPACE,              REQUIRED);
	ValidateFile($data->{'globalOptions'},         'fastaFileName',                            REQUIRED);
	ValidateOptions($data->{'globalOptions'},      'timing',             \%TIMING,             OPTIONAL);
	ValidatePath($data->{'globalOptions'},         'runDirectory',                             REQUIRED); 
	ValidatePath($data->{'globalOptions'},         'readsDirectory',                           REQUIRED); 
	ValidatePath($data->{'globalOptions'},         'outputDirectory',                          REQUIRED); 
	ValidatePath($data->{'globalOptions'},         'tmpDirectory',                             REQUIRED); 
	ValidateOption($data->{'globalOptions'},       'outputID',                                 REQUIRED); 
	ValidateOption($data->{'globalOptions'},       'numReadsPerFASTQ',                         REQUIRED);
	ValidateOption($data->{'globalOptions'},       'cleanUsedIntermediateFiles',               REQUIRED);
	die "Attribute matchSplit required with numReadsPerFASTQ\n" if (!defined($data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'matchSplit'}));
	die "Attribute localalignSplit required with numReadsPerFASTQ\n" if (!defined($data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'localalignSplit'}));
	die "Attribute matchSplit must be <= numReadsPerFASTQ\n" if ($data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'content'} < $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'matchSplit'});
	die "Attribute localalignSplit must be <= matchSplit\n" if ($data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'matchSplit'} < $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'localalignSplit'});

	# match
	die("The match options were not found.\n") unless (defined($data->{'matchOptions'})); 
	ValidateOption($data->{'matchOptions'},     'mainIndexes',                              OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'secondaryIndexes',                         OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'offsets',                                  OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'loadAllIndexes',                           OPTIONAL);
	ValidateOptions($data->{'matchOptions'},    'readCompression',       \%COMPRESSION,     OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'keySize',                                  OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'maxKeyMatches',                            OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'maxNumMatches',                            OPTIONAL);
	ValidateOptions($data->{'matchOptions'},    'strand',             \%STRAND,             OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'threads',                                  OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'queueLength',                              OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'mergeSeparate',                            OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'qsubQueue',                                OPTIONAL);
	ValidateOption($data->{'matchOptions'},     'qsubArgs',                                 OPTIONAL);

	# localalign
	die("The localalign options were not found.\n") unless (defined($data->{'localalignOptions'})); 
	ValidateFile($data->{'localalignOptions'},         'scoringMatrix',                            OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'ungapped',                            	   OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'unconstrained',                            OPTIONAL);
	ValidateFile($data->{'localalignOptions'},         'offset',                                   OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'maxNumMatches',                            OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'mismatchQuality',                          OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'queueLength',                              OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'pairedEndLength',                          OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'mirrorType',                               OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'forceMirror',                              OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'threads',                                  OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'qsubQueue',                                OPTIONAL);
	ValidateOption($data->{'localalignOptions'},       'qsubArgs',                                 OPTIONAL);

	# postprocess
	die("The postprocess options were not found.\n") unless (defined($data->{'postprocessOptions'})); 
	ValidateOption($data->{'postprocessOptions'}, 'algorithm',                                OPTIONAL);
	ValidateOption($data->{'postprocessOptions'}, 'unpaired',                                  OPTIONAL);
	ValidateOption($data->{'postprocessOptions'}, 'reversePaired',                            OPTIONAL);
	ValidateOption($data->{'unmappedFile'}, 'unmappedFile',                                   OPTIONAL);
	ValidateFile($data->{'postprocessOptions'}, 'readGroupFile',                              OPTIONAL);
	ValidateOptions($data->{'postprocessOptions'}, 'outputFormat', \%OUTTYPES,                OPTIONAL);
	ValidateOption($data->{'postprocessOptions'}, 'threads',                                  OPTIONAL);
	ValidateOption($data->{'postprocessOptions'}, 'queueLength',                              OPTIONAL);
	ValidateOption($data->{'postprocessOptions'}, 'qsubQueue',                                OPTIONAL);
	ValidateOption($data->{'postprocessOptions'}, 'qsubArgs',                                 OPTIONAL);

	# samtools/picard
	if(defined($data->{'samOptions'})) {
		ValidateOption($data->{'samOptions'},     'samtools',								  REQUIRED);
		ValidateOption($data->{'samOptions'},     'maximumMemory',                            OPTIONAL);
		ValidateOption($data->{'samOptions'},     'qsubQueue',                                OPTIONAL);
		ValidateOption($data->{'samOptions'},     'qsubArgs',                                 OPTIONAL);
	}
	else {
		warn("The SAM options were not found.  Skipping SAM...\n");
	}
}

sub ValidateOption {
	my ($hash, $option, $required) = @_;

	return 1 if (defined($hash->{$option}));
	return 0 if (OPTIONAL == $required);

	die("Option '$option' was not found.\n");
}

sub ValidatePath {
	my ($hash, $option, $required) = @_; 

	if(0 != ValidateOption($hash, $option, $required) and $hash->{$option} !~ m/\S+\//) { # very liberal check
		die("Option '$option' did not give a valid path.\n");
	}
}

sub ValidateFile {
	my ($hash, $option, $required) = @_;

	if(0 != ValidateOption($hash, $option, $required) and $hash->{$option} !~ m/\S+/) { # very liberal check
		die("Option '$option' did not give a valid file name.\n");
	}
}

sub ValidateOptions {
	my ($hash, $option, $values, $required) = @_;

	if(0 != ValidateOption($hash, $option, $required) and !defined($values->{$hash->{$option}})) {
		die("The value '".($hash->{$option})."' for option '$option' was not valid.\n");
	}
}

sub GetDirContents {
	my ($dir, $dirs, $suffix) = @_;

	if(!($dir =~ m/\/$/)) {
		$dir .= "/";
	}

	local *DIR;
	opendir(DIR, "$dir") or die("Error.  Could not open $dir.  Terminating!\n");
	@$dirs = grep !/^\.\.?$/, readdir DIR;
	for(my $i=0;$i<scalar(@$dirs);$i++) {
		@$dirs[$i] = $dir."".@$dirs[$i];
	}
	close(DIR);

	@$dirs= grep /\.$suffix$/, @$dirs;
	if(0 == scalar(@$dirs)) {
		die("Did not find any '$suffix' files\n");;
	}

	@$dirs = sort { $a cmp $b } @$dirs;
}

sub CreateJobs {
	my ($data, $quiet, $start_step, $dryrun) = @_;

	my @match_output_ids = ();
	my @localalign_output_ids = ();
	my @matchJobIDs = ();
	my @localalignJobIDs = ();
	my @postprocessJobIDs = ();

	# Create directories - Error checking...
	mkpath([$data->{'globalOptions'}->{'runDirectory'}],    ($quiet) ? 0 : 1, 0755);
	mkpath([$data->{'globalOptions'}->{'outputDirectory'}], ($quiet) ? 0 : 1, 0755);
	mkpath([$data->{'globalOptions'}->{'tmpDirectory'}],    ($quiet) ? 0 : 1, 0755);

	CreateJobsMatch($data, $quiet, $start_step, $dryrun, \@matchJobIDs,     \@match_output_ids);
	CreateJobsLocalalign($data, $quiet, $start_step, $dryrun, \@matchJobIDs,     \@localalignJobIDs,       \@match_output_ids, \@localalign_output_ids);
	CreateJobsPostprocess($data, $quiet, $start_step, $dryrun, \@localalignJobIDs,       \@postprocessJobIDs, \@localalign_output_ids);
	if(defined($data->{'samOptions'})) {
		if(!defined($data->{'postprocessOptions'}->{'outputFormat'}) ||
			3 == $data->{'postprocessOptions'}->{'outputFormat'}) {
			CreateJobsSAM($data, $quiet, $start_step, $dryrun, \@postprocessJobIDs, \@localalign_output_ids);
		}
	}
}

sub CreateRunFile {
	my ($data, $type, $output_id) = @_;

	return $data->{'globalOptions'}->{'runDirectory'}."$type.$output_id.sh";
}

sub GetMatchesFile {
	my ($data, $output_id) = @_;

	return sprintf("%sbfast.matches.file.%s.bmf",
		$data->{'globalOptions'}->{'outputDirectory'},
		$output_id);
}

sub GetAlignFile {
	my ($data, $output_id) = @_;

	return sprintf("%sbfast.aligned.file.%s.baf",
		$data->{'globalOptions'}->{'outputDirectory'},
		$output_id);
}

sub GetReportedFile {
	my ($data, $output_id, $type) = @_;

	return sprintf("%sbfast.reported.file.%s.%s",
		$data->{'globalOptions'}->{'outputDirectory'},
		$output_id,
		$OUTTYPES{$type});
}

sub GetUnmappedFile {
	my ($data, $output_id, $type) = @_;

	return sprintf("%sbfast.not.reported.file.%s.%s",
		$data->{'globalOptions'}->{'outputDirectory'},
		$output_id,
		$OUTTYPES{$type});
}

sub GetIndexes {
	my ($data, $indexNumbers) = @_;

	if(defined($data->{'matchOptions'}->{'secondaryIndexes'})) {
		die("Option \"secondaryIndexes\" not supported when using \"mergeSeparate\"");
	}
	if(defined($data->{'matchOptions'}->{'mainIndexes'})) {
		# This could be supported, but currently not
		die("Option \"mainIndexes\" currently not supported when using \"mergeSeparate\"");
	}
	else {
		my $space = "nt";
		$space = "cs" if ("CS" eq $data->{'globalOptions'}->{'space'});

		my $dir = $data->{'globalOptions'}->{'fastaFileName'};

		# infer directory
		if($dir =~ m/^(.*\/)/) {
			$dir = $1;
		}
		else {
			$dir = "./";
		}

		local *DIR;
		opendir(DIR, "$dir") or die("Error.  Could not open $dir.  Terminating!\n");
		my @dirs = grep !/^\.\.?$/, readdir DIR;
		close(DIR);

		@dirs = grep m/$space\.\d+\.1\.bif$/, @dirs; # indexes only
		for(my $i=0;$i<scalar(@dirs);$i++) {
			if($dirs[$i] =~ m/$space\.(\d+)\.1\.bif$/) {
				push(@$indexNumbers, $1);
			}
			else {
				die;
			}
		}
	}

	@$indexNumbers = sort { $a <=> $b } @$indexNumbers;

	die if(0 == scalar(@$indexNumbers));
}

sub CreateJobsMatch {
	my ($data, $quiet, $start_step, $dryrun, $qsub_ids, $output_ids) = @_;
	my @read_files = ();

	# Get reads
	my $file_ext = "fastq";
	$file_ext .= ".".$data->{'matchOptions'}->{'readCompression'} if(defined($data->{'matchOptions'}->{'readCompression'}));
	GetDirContents($data->{'globalOptions'}->{'readsDirectory'}, \@read_files, $file_ext);

	my @indexNumbers = ();
	if(defined($data->{'matchOptions'}->{'mergeSeparate'})) {
		GetIndexes($data, \@indexNumbers);
	}

	# The number of match to perform per read file
	my $num_split_files = int(0.5 + ($data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'content'}/$data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'matchSplit'}));

	# Go through each
	foreach my $read_file (@read_files) {
		my ($cur_read_num_start, $cur_read_num_end) = (1, $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'matchSplit'}); 
		for(my $i=0;$i<$num_split_files;$i++) {
			my $output_id = $read_file; 
			$output_id =~ s/.*\///;
			if($output_id =~ m/(\d+).$file_ext/) {
				$output_id = $1;                
				$output_id = $data->{'globalOptions'}->{'outputID'}.".$output_id.reads.$cur_read_num_start-$cur_read_num_end";            
			}   
			else {
				$output_id =~ s/.$file_ext//;
				$output_id = $data->{'globalOptions'}->{'outputID'}.".reads.$i";            
			}

			if(defined($data->{'matchOptions'}->{'mergeSeparate'}) && 1 == $data->{'matchOptions'}->{'mergeSeparate'}) {
				# Run each BMF separately
				my @qsub_ids_sub = ();
				my @output_ids_sub = ();
				my @bmf_files_sub = ();
				for(my $j=0;$j<scalar(@indexNumbers);$j++) {
					my $output_id_sub = $output_id.".".(1+$j)."";
					my $run_file_sub = CreateRunFile($data, 'match', $output_id_sub);
					my $bmf_file_sub = GetMatchesFile($data, $output_id_sub);
					my $cmd_sub = "";
					$cmd_sub .= $data->{'globalOptions'}->{'bfastBin'}.""      if defined($data->{'globalOptions'}->{'bfastBin'});
					$cmd_sub .= "bfast match";
					$cmd_sub .= " -f ".$data->{'globalOptions'}->{'fastaFileName'};
					$cmd_sub .= " -i ".(1+$j);
					$cmd_sub .= " -r ".$read_file;
					$cmd_sub .= " -o ".$data->{'matchOptions'}->{'offsets'}          if defined($data->{'matchOptions'}->{'offsets'});
					$cmd_sub .= " ".$COMPRESSION{$data->{'matchOptions'}->{'readCompression'}} if(defined($data->{'matchOptions'}->{'readCompression'}));
					$cmd_sub .= " -A 1"                                                 if ("CS" eq $data->{'globalOptions'}->{'space'});
					$cmd_sub .= " -s $cur_read_num_start -e $cur_read_num_end";
					$cmd_sub .= " -k ".$data->{'matchOptions'}->{'keySize'}          if defined($data->{'matchOptions'}->{'keySize'});
					$cmd_sub .= " -K ".$data->{'matchOptions'}->{'maxKeyMatches'}    if defined($data->{'matchOptions'}->{'maxKeyMatches'});
					$cmd_sub .= " -M ".$data->{'matchOptions'}->{'maxNumMatches'}    if defined($data->{'matchOptions'}->{'maxNumMatches'});
					$cmd_sub .= " -w ".$STRAND{$data->{'matchOptions'}->{'strand'}}           if defined($data->{'matchOptions'}->{'strand'});
					$cmd_sub .= " -n ".$data->{'matchOptions'}->{'threads'}          if defined($data->{'matchOptions'}->{'threads'});
					$cmd_sub .= " -Q ".$data->{'matchOptions'}->{'queueLength'}      if defined($data->{'matchOptions'}->{'queueLength'});
					$cmd_sub .= " -T ".$data->{'globalOptions'}->{'tmpDirectory'};
					$cmd_sub .= " -t"                                                   if defined($data->{'globalOptions'}->{'timing'});
					$cmd_sub .= " > ".$bmf_file_sub;

					# Submit the job
					my @a_sub = (); # empty array for job dependencies
					my $qsub_id_sub = SubmitJob($run_file_sub, $quiet, ($start_step <= $STARTSTEP{"match"}) ? 1 : 0, 0, $dryrun, $cmd_sub, $data, 'matchOptions', $output_id_sub, \@a_sub);
					push(@qsub_ids_sub, $qsub_id_sub) if (QSUBNOJOB ne $qsub_id_sub);
					push(@output_ids_sub, $output_id_sub);
					push(@bmf_files_sub, $bmf_file_sub);
				}

				# Temporarily nullify threads  
				my $tmp_threads = -1;
				if(defined($data->{'matchOptions'}->{'threads'}) && 1 < $data->{'matchOptions'}->{'threads'}) {
					$tmp_threads = $data->{'matchOptions'}->{'threads'};
					$data->{'matchOptions'}->{'threads'} = 1;
				}

				# Merge results
				my $run_file = CreateRunFile($data, 'match', $output_id);
				my $bmf_file = GetMatchesFile($data, $output_id);
				my $cmd = "";
				$cmd .= $data->{'globalOptions'}->{'bfastBin'}.""      if defined($data->{'globalOptions'}->{'bfastBin'});
				$cmd .= "bmfmerge";
				$cmd .= " -M ".$data->{'matchOptions'}->{'maxNumMatches'}    if defined($data->{'matchOptions'}->{'maxNumMatches'});
				$cmd .= " -Q ".$data->{'matchOptions'}->{'queueLength'}      if defined($data->{'matchOptions'}->{'queueLength'});
				for(my $j=0;$j<scalar(@bmf_files_sub);$j++) {
					$cmd .= " ".$bmf_files_sub[$j];
				}
				$cmd .= " > ".$bmf_file;
				my @a = @qsub_ids_sub; 
				my $qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"match"}) ? 1 : 0, 1, $dryrun, $cmd, $data, 'matchOptions', $output_id, \@a);
				push(@$qsub_ids, $qsub_id) if (QSUBNOJOB ne $qsub_id);
				push(@$output_ids, $output_id);

				# Clean merged results
				if(defined($data->{'globalOptions'}->{'cleanUsedIntermediateFiles'}) &&
					$data->{'globalOptions'}->{'cleanUsedIntermediateFiles'} == 1) {
					$run_file = CreateRunFile($data, 'clean.bmfmerge', $output_id);
					$output_id = "clean.bmfmerge.$output_id";
					$cmd = "rm -v";
					for(my $j=0;$j<scalar(@bmf_files_sub);$j++) {
						$cmd .= " ".$bmf_files_sub[$j];
					}
					@a = (); push(@a, $qsub_id); # depend on the previous
					$qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"match"}) ? 1 : 0, 1, $dryrun, $cmd, $data, 'matchOptions', $output_id, \@a);
				}
				
				# Restore threads
				if(defined($data->{'matchOptions'}->{'threads'}) && 1 < $data->{'matchOptions'}->{'threads'}) {
					$data->{'matchOptions'}->{'threads'} = $tmp_threads;
				}

				$cur_read_num_start += $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'matchSplit'};
				$cur_read_num_end += $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'matchSplit'};
			}
			else {
				my $run_file = CreateRunFile($data, 'match', $output_id);
				my $bmf_file = GetMatchesFile($data, $output_id);
				my $cmd = "";
				$cmd .= $data->{'globalOptions'}->{'bfastBin'}.""      if defined($data->{'globalOptions'}->{'bfastBin'});
				$cmd .= "bfast match";
				$cmd .= " -f ".$data->{'globalOptions'}->{'fastaFileName'};
				$cmd .= " -i ".$data->{'matchOptions'}->{'mainIndexes'} if defined($data->{'matchOptions'}->{'mainIndexes'});
				$cmd .= " -I ".$data->{'matchOptions'}->{'secondaryIndexes'} if defined($data->{'matchOptions'}->{'secondaryIndexes'});
				$cmd .= " -r ".$read_file;
				$cmd .= " -o ".$data->{'matchOptions'}->{'offsets'}          if defined($data->{'matchOptions'}->{'offsets'});
				$cmd .= " -l " if defined($data->{'matchOptions'}->{'loadAllIndexes'});
				$cmd .= " ".$COMPRESSION{$data->{'matchOptions'}->{'readCompression'}} if(defined($data->{'matchOptions'}->{'readCompression'}));
				$cmd .= " -A 1"                                                 if ("CS" eq $data->{'globalOptions'}->{'space'});
				$cmd .= " -s $cur_read_num_start -e $cur_read_num_end";
				$cmd .= " -k ".$data->{'matchOptions'}->{'keySize'}          if defined($data->{'matchOptions'}->{'keySize'});
				$cmd .= " -K ".$data->{'matchOptions'}->{'maxKeyMatches'}    if defined($data->{'matchOptions'}->{'maxKeyMatches'});
				$cmd .= " -M ".$data->{'matchOptions'}->{'maxNumMatches'}    if defined($data->{'matchOptions'}->{'maxNumMatches'});
				$cmd .= " -w ".$STRAND{$data->{'matchOptions'}->{'strand'}}           if defined($data->{'matchOptions'}->{'strand'});
				$cmd .= " -n ".$data->{'matchOptions'}->{'threads'}          if defined($data->{'matchOptions'}->{'threads'});
				$cmd .= " -Q ".$data->{'matchOptions'}->{'queueLength'}      if defined($data->{'matchOptions'}->{'queueLength'});
				$cmd .= " -T ".$data->{'globalOptions'}->{'tmpDirectory'};
				$cmd .= " -t"                                                   if defined($data->{'globalOptions'}->{'timing'});
				$cmd .= " > ".$bmf_file;

				# Submit the job
				my @a = (); # empty array for job dependencies
				my $qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"match"}) ? 1 : 0, 0, $dryrun, $cmd, $data, 'matchOptions', $output_id, \@a);
				push(@$qsub_ids, $qsub_id) if (QSUBNOJOB ne $qsub_id);
				push(@$output_ids, $output_id);
				$cur_read_num_start += $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'matchSplit'};
				$cur_read_num_end += $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'matchSplit'};
			}
		}
	}
}

# One dependent id for each output id
sub CreateJobsLocalalign {
	my ($data, $quiet, $start_step, $dryrun, $dependent_ids, $qsub_ids, $input_ids, $output_ids) = @_;

	# The number of match to perform per read file
	my $num_split_files = int(0.5 + ($data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'matchSplit'}/$data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'localalignSplit'}));

	# Go through each
	for(my $i=0;$i<scalar(@$input_ids);$i++) {
		my ($cur_read_num_start, $cur_read_num_end) = (1, $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'localalignSplit'});
		my ($output_id_read_num_start, $output_id_read_num_end) = (0, 0);
		my $dependent_job = (0 < scalar(@$dependent_ids)) ? $dependent_ids->[$i] : QSUBNOJOB;
		my $input_id = $input_ids->[$i];
		my $input_id_no_read_num ="";
		if($input_id =~ m/(.+)\.(\d+)\-\d+$/) {
			$input_id_no_read_num = $1;
			$output_id_read_num_start += $2; 
			$output_id_read_num_end += $2 + $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'localalignSplit'} - 1;
		}
		else {
			$input_id_no_read_num = $input_id;
			$output_id_read_num_start = 1;
			$output_id_read_num_end = $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'localalignSplit'};
		} 
		my $bmf_file = GetMatchesFile($data, $input_id);
		my @bmf_file_clean_ids = ();
		for(my $j=0;$j<$num_split_files;$j++) {
			my $output_id = "$input_id_no_read_num.$output_id_read_num_start-$output_id_read_num_end";
			my $run_file = CreateRunFile($data, 'localalign', $output_id);
			my $baf_file = GetAlignFile($data, $output_id);

			my $cmd = "";
			$cmd .= $data->{'globalOptions'}->{'bfastBin'}.""        if defined($data->{'globalOptions'}->{'bfastBin'});
			$cmd .= "bfast localalign";
			$cmd .= " -f ".$data->{'globalOptions'}->{'fastaFileName'};
			$cmd .= " -m $bmf_file";
			$cmd .= " -x ".$data->{'localalignOptions'}->{'scoringMatrix'}      if defined($data->{'localalignOptions'}->{'scoringMatrix'});
			$cmd .= " -u" if defined($data->{'localalignOptions'}->{'ungapped'});
			$cmd .= " -c" if defined($data->{'localalignOptions'}->{'unconstrained'});
			$cmd .= " -A 1"                                                 if ("CS" eq $data->{'globalOptions'}->{'space'});
			$cmd .= " -o ".$data->{'localalignOptions'}->{'offset'}             if defined($data->{'localalignOptions'}->{'offset'});
			$cmd .= " -M ".$data->{'localalignOptions'}->{'maxNumMatches'}      if defined($data->{'localalignOptions'}->{'maxNumMatches'});
			$cmd .= " -q ".$data->{'localalignOptions'}->{'mismatchQuality'}    if defined($data->{'localalignOptions'}->{'mismatchQuality'});
			$cmd .= " -n ".$data->{'localalignOptions'}->{'threads'}            if defined($data->{'localalignOptions'}->{'threads'});
			$cmd .= " -Q ".$data->{'localalignOptions'}->{'queueLength'}        if defined($data->{'localalignOptions'}->{'queueLength'});
			$cmd .= " -l ".$data->{'localalignOptions'}->{'pairedEndLength'}    if defined($data->{'localalignOptions'}->{'pairedEndLength'});
			$cmd .= " -L ".$data->{'localalignOptions'}->{'mirroringType'}      if defined($data->{'localalignOptions'}->{'mirroringType'});
			$cmd .= " -F"                                                   if defined($data->{'localalignOptions'}->{'forceMirror'});
			$cmd .= " -s $cur_read_num_start -e $cur_read_num_end";
			$cmd .= " -t"                                                   if defined($data->{'globalOptions'}->{'timing'});
			$cmd .= " > ".$baf_file;

			# Submit the job
			my @a = (); push(@a, $dependent_job) if(QSUBNOJOB ne $dependent_job);
			my $qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"localalign"}) ? 1 : 0, ($start_step <= $STARTSTEP{"match"}) ? 1 : 0, $dryrun, $cmd, $data, 'localalignOptions', $output_id, \@a);
			push(@$qsub_ids, $qsub_id) if (QSUBNOJOB ne $qsub_id);
			push(@$output_ids, $output_id);
			$cur_read_num_start += $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'localalignSplit'};
			$cur_read_num_end += $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'localalignSplit'};
			$output_id_read_num_start += $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'localalignSplit'};
			$output_id_read_num_end += $data->{'globalOptions'}->{'numReadsPerFASTQ'}->{'localalignSplit'};
			push(@bmf_file_clean_ids, $qsub_id) if (QSUBNOJOB ne $qsub_id);
		}
		if(defined($data->{'globalOptions'}->{'cleanUsedIntermediateFiles'}) &&
			$data->{'globalOptions'}->{'cleanUsedIntermediateFiles'} == 1 &&
			0 < scalar(@bmf_file_clean_ids)) {
			my $cmd = "rm -v $bmf_file";
			my $run_file = CreateRunFile($data, 'clean.bmf', $input_id);
			my $output_id = "clean.bmf.$input_id";
			my $qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"localalign"}) ? 1 : 0, ($start_step <= $STARTSTEP{"match"}) ? 1 : 0, $dryrun, $cmd, $data, 'globalOptions', $output_id, \@bmf_file_clean_ids);
		}
	}
}

sub CreateJobsPostprocess {
	my ($data, $quiet, $start_step, $dryrun, $dependent_ids, $qsub_ids, $output_ids) = @_;

	# Go through each
	for(my $i=0;$i<scalar(@$output_ids);$i++) {
		my $output_id = $output_ids->[$i];
		my $input_id = $output_ids->[$i];
		my $dependent_job = (0 < scalar(@$dependent_ids)) ? $dependent_ids->[$i] : QSUBNOJOB;
		my $baf_file = GetAlignFile($data, $output_id);
		my $run_file = CreateRunFile($data, 'postprocess', $output_id);
		my $sam_file = GetReportedFile($data, $output_id, (defined($data->{'postprocessOptions'}->{'outputFormat'})) ? $data->{'postprocessOptions'}->{'outputFormat'} : 3);
		my $unmapped_file = GetUnmappedFile($data, $output_id, 0);

		my $cmd = "";
		$cmd .= $data->{'globalOptions'}->{'bfastBin'}."" if defined($data->{'globalOptions'}->{'bfastBin'});
		$cmd .= "bfast postprocess";
		$cmd .= " -f ".$data->{'globalOptions'}->{'fastaFileName'};
		$cmd .= " -i $baf_file";
		$cmd .= " -a ".$data->{'postprocessOptions'}->{'algorithm'}   if defined($data->{'postprocessOptions'}->{'algorithm'});
		$cmd .= " -A 1"                                                 if ("CS" eq $data->{'globalOptions'}->{'space'});
		$cmd .= " -U" if defined($data->{'postprocessOptions'}->{'unpaired'});
		$cmd .= " -R" if defined($data->{'postprocessOptions'}->{'reversePaired'});
		$cmd .= " -q ".$data->{'localalignOptions'}->{'mismatchQuality'}    if defined($data->{'localalignOptions'}->{'mismatchQuality'});
		$cmd .= " -x ".$data->{'localalignOptions'}->{'scoringMatrix'}      if defined($data->{'localalignOptions'}->{'scoringMatrix'});
		$cmd .= " -u ".$unmapped_file if defined($data->{'postprocessOptions'}->{'unmappedFile'});
		$cmd .= " -O ".$data->{'postprocessOptions'}->{'outputFormat'}   if defined($data->{'postprocessOptions'}->{'outputFormat'});
		$cmd .= " -r ".$data->{'postprocessOptions'}->{'readGroupFile'}.""   if defined($data->{'postprocessOptions'}->{'readGroupFile'});
		$cmd .= " -n ".$data->{'postprocessOptions'}->{'threads'}          if defined($data->{'postprocessOptions'}->{'threads'});
		$cmd .= " -Q ".$data->{'postprocessOptions'}->{'queueLength'} if defined($data->{'postprocessOptions'}->{'queueLength'});
		$cmd .= " -t"                                                  if defined($data->{'globalOptions'}->{'timing'});
		$cmd .= " > ".$sam_file;

		# Submit the job
		my @a = (); push(@a, $dependent_job) if(QSUBNOJOB ne $dependent_job);
		my $qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"postprocess"}) ? 1 : 0, ($start_step <= $STARTSTEP{"localalign"}) ? 1 : 0, $dryrun, $cmd, $data, 'postprocessOptions', $output_id, \@a);
		push(@$qsub_ids, $qsub_id) if (QSUBNOJOB ne $qsub_id);

		if(defined($data->{'globalOptions'}->{'cleanUsedIntermediateFiles'}) &&
			$data->{'globalOptions'}->{'cleanUsedIntermediateFiles'} == 1 &&
			QSUBNOJOB ne $qsub_id) {

			my @a = (); 
			push(@a, $qsub_id) if (QSUBNOJOB ne $qsub_id);
			my $cmd = "rm -v $baf_file";
			my $run_file = CreateRunFile($data, 'clean.baf', $input_id);
			my $output_id = "clean.baf.$input_id";
			$qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"postprocess"}) ? 1 : 0, ($start_step <= $STARTSTEP{"localalign"}) ? 1 : 0, $dryrun, $cmd, $data, 'globalOptions', $output_id, \@a);
		}
	}
}

sub CreateJobsSAM {
	my ($data, $quiet, $start_step, $dryrun, $dependent_ids, $output_ids) = @_;
	my @qsub_ids = ();
	my ($cmd, $run_file, $output_id, $qsub_id);

	my $type = ($data->{'samOptions'}->{'samtools'} == 0) ? 'picard' : 'samtools';
	my @reported_bams = ();
	if(0 == $data->{'samOptions'}->{'samtools'}) {
		if(!defined($data->{'globalOptions'}->{'picardBin'})) { die("Picard bin required") };
		if(!defined($data->{'globalOptions'}->{'javaBin'})) { die("Java bin required") };
	}

	# Go through each
	for(my $i=0;$i<scalar(@$output_ids);$i++) {
		$output_id = $output_ids->[$i];
		my $dependent_job = (0 < scalar(@$dependent_ids)) ? $dependent_ids->[$i] : QSUBNOJOB;
		my $sam_file = GetReportedFile($data, $output_id, 3);
		$run_file = CreateRunFile($data, $type, $output_id);

		if(0 == $data->{'samOptions'}->{'samtools'}) {
			$cmd = $data->{'globalOptions'}->{'javaBin'}."java";
			$cmd .= " -Xmx2g";
			$cmd .= " -jar ".$data->{'globalOptions'}->{'picardBin'}."SortSam.jar";
			$cmd .= " I=$sam_file";
			$cmd .= " O=".$data->{'globalOptions'}->{'outputDirectory'}."bfast.reported.file.$output_id.bam";
			$cmd .= " SO=coordinate";
			$cmd .= " TMP_DIR=".$data->{'globalOptions'}->{'tmpDirectory'};
			$cmd .= " VALIDATION_STRINGENCY=SILENT";
		}
		else {
			$cmd = "";
			$cmd .= "".$data->{'globalOptions'}->{'samtoolsBin'} if defined($data->{'globalOptions'}->{'samtoolsBin'});
			$cmd .= "samtools view -S -b";
			$cmd .= " -T ".$data->{'globalOptions'}->{'fastaFileName'};
			$cmd .= " $sam_file | ";
			$cmd .= $data->{'globalOptions'}->{'samtoolsBin'}            if defined($data->{'globalOptions'}->{'samtoolsBin'});
			$cmd .= "samtools sort";
			$cmd .= " -m ".$data->{'samOptions'}->{'maximumMemory'} if defined($data->{'samOptions'}->{'maximumMemory'});
			$cmd .= " - ".$data->{'globalOptions'}->{'outputDirectory'};
			$cmd .= "bfast.reported.file.$output_id";
		}
		push(@reported_bams, $data->{'globalOptions'}->{'outputDirectory'}."bfast.reported.file.$output_id.bam");

# Submit the job
		my @a = (); push(@a, $dependent_job) if(QSUBNOJOB ne $dependent_job);
		$qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"sam"}) ? 1 : 0, ($start_step <= $STARTSTEP{"postprocess"}) ? 1 : 0, $dryrun, $cmd, $data, 'samOptions', $output_id, \@a);
		if(QSUBNOJOB ne $qsub_id) {
			push(@qsub_ids, $qsub_id);
		}
		else {
			# currently it must be submitted
			die;
		}

		if(defined($data->{'globalOptions'}->{'cleanUsedIntermediateFiles'}) &&
			$data->{'globalOptions'}->{'cleanUsedIntermediateFiles'} == 1 &&
			QSUBNOJOB ne $qsub_id) {
			my @a = (); 
			push(@a, $qsub_id) if (QSUBNOJOB ne $qsub_id);
			my $cmd = "rm -v $sam_file";
			my $input_id = $output_id;
			my $run_file = CreateRunFile($data, 'clean.sam', $input_id);
			my $output_id = "clean.sam.$input_id";
			$qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"sam"}) ? 1 : 0, ($start_step <= $STARTSTEP{"postprocess"}) ? 1 : 0, $dryrun, $cmd, $data, 'globalOptions', $output_id, \@a);
		}
	}

	# Merge script(s)
	# Note: there could be too many dependencies, so lets just create dummy jobs to "merge" the dependencies
	# Note: what todo if there are too many inputs?
	my $merge_lvl = 0;
	while(MERGE_LOG_BASE < scalar(@qsub_ids)) { # while we must merge
		$merge_lvl++;
		my $ctr = 0;
		my @cur_ids = @qsub_ids;
		@qsub_ids = ();
		for(my $i=0;$i<scalar(@cur_ids);$i+=MERGE_LOG_BASE) {
			$ctr++;
			# Get the subset of dependent jobs
			my @dependent_jobs = ();
			for(my $j=$i;$j<scalar(@cur_ids) && $j<$i+MERGE_LOG_BASE;$j++) {
				push(@dependent_jobs, $cur_ids[$j]);
			}
			# Create the command
			$output_id = "merge.".$data->{'globalOptions'}->{'outputID'}.".$merge_lvl.$ctr";
			$run_file = $data->{'globalOptions'}->{'runDirectory'}."$type.".$output_id.".sh";
			$cmd = "echo \"Merging $merge_lvl / $ctr\"\n";
			$qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"sam"}) ? 1 : 0, 1, $dryrun, $cmd, $data, 'samOptions', $output_id, \@dependent_jobs);
			if(QSUBNOJOB ne $qsub_id) {
				push(@qsub_ids, $qsub_id);
			}
			else {
				# currently it must be submitted
				die;
			}
		}
	}

	$output_id = "merge.".$data->{'globalOptions'}->{'outputID'};
	$run_file = $data->{'globalOptions'}->{'runDirectory'}."$type.".$output_id.".sh";
	if(1 < scalar(@qsub_ids)) {
		if(0 == $data->{'samOptions'}->{'samtools'}) {
			if(!defined($data->{'globalOptions'}->{'picardBin'})) { die("Picard bin required") };
			$cmd = $data->{'globalOptions'}->{'javaBin'}."java";
			$cmd .= " -Xmx2g";
			$cmd .= " -jar ".$data->{'globalOptions'}->{'picardBin'}."MergeSamFiles.jar";
			foreach my $bam (@reported_bams) {
				$cmd .= " I=$bam";
			}
			$cmd .= " O=".$data->{'globalOptions'}->{'outputDirectory'}."bfast.".$data->{'globalOptions'}->{'outputID'}.".bam";
			$cmd .= " SO=coordinate";
			$cmd .= " AS=true";
			$cmd .= " TMP_DIR=".$data->{'globalOptions'}->{'tmpDirectory'};
			$cmd .= " VALIDATION_STRINGENCY=SILENT";
		}
		else {
			$cmd = "";
			$cmd .= $data->{'globalOptions'}->{'samtoolsBin'} if defined($data->{'globalOptions'}->{'samtoolsBin'});
			$cmd .= "samtools merge";
			$cmd .= " ".$data->{'globalOptions'}->{'outputDirectory'}."bfast.".$data->{'globalOptions'}->{'outputID'}.".bam";
			$cmd .= " ".$data->{'globalOptions'}->{'outputDirectory'}."bfast.reported.file.".$data->{'globalOptions'}->{'outputID'}."*bam";
		}
	}
	else {
		my $output_id = $output_ids->[0];
		my $sam_file = $data->{'globalOptions'}->{'outputDirectory'}."bfast.reported.file.$output_id.bam";
		$cmd = "cp -v $sam_file ".$data->{'globalOptions'}->{'outputDirectory'}."bfast.".$data->{'globalOptions'}->{'outputID'}.".bam";
	}
	$qsub_id = SubmitJob($run_file , $quiet, ($start_step <= $STARTSTEP{"sam"}) ? 1 : 0, 1, $dryrun, $cmd, $data, 'samOptions', $output_id, \@qsub_ids);
	if(QSUBNOJOB ne $qsub_id) {
		push(@qsub_ids, $qsub_id);
	}
	else {
		# currently it must be submitted
		die;
	}

	# clean up SAM files
	# What if there are too many BAMS?
	if(defined($data->{'globalOptions'}->{'cleanUsedIntermediateFiles'}) &&
		$data->{'globalOptions'}->{'cleanUsedIntermediateFiles'} == 1 &&
		QSUBNOJOB ne $qsub_id) {
		my @a = (); 
		push(@a, $qsub_id) if (QSUBNOJOB ne $qsub_id);
		my $cmd = "rm -v @reported_bams";
		my $output_id = "clean.bams";
		my $run_file = CreateRunFile($data, 'clean', "bams");
		$qsub_id = SubmitJob($run_file, $quiet, ($start_step <= $STARTSTEP{"sam"}) ? 1 : 0, ($start_step <= $STARTSTEP{"postprocess"}) ? 1 : 0, $dryrun, $cmd, $data, 'globalOptions', $output_id, \@a);
	}
}

sub SubmitJob {
	my ($run_file, $quiet, $should_run, $should_depend, $dryrun, $command, $data, $type, $output_id, $dependent_job_ids) = @_;
	$output_id = "$type.$output_id"; $output_id =~ s/Options//g;

	if(!$quiet) {
		print STDERR "[bfast submit] RUNFILE=$run_file\n";
	}
	if(1 == $should_run) {
		my $output = <<END_OUTPUT;
run ()
{
	echo "running: \$*" 2>&1;
	eval \$*;
	if test \$? != 0 ; then
	echo "error: while running '\$*'";
	exit 100;
	fi
}
END_OUTPUT
		$output .= "\nrun \"hostname\";\n";
		# Redirect PBS stderr/stdout, since it buffers them
		if ("PBS" eq $data->{'globalOptions'}->{'queueType'}) {
			my $pbs_stderr_redirect = "$run_file.stderr.redirect";
			my $pbs_stdout_redirect = "$run_file.stdout.redirect";
			$output .= "run \"$command 2> $pbs_stderr_redirect > $pbs_stdout_redirect\";\n";
		}
		else {
			$output .= "run \"$command\";\n";
		}
		$output .= "exit 0;\n";
		open(FH, ">$run_file") or die("Error.  Could not open $run_file for writing!\n");
		print FH "$output";
		close(FH);
	}

	# Create qsub command
	my $qsub = "";
	$qsub .= $data->{'globalOptions'}->{'qsubBin'} if defined($data->{'globalOptions'}->{'qsubBin'});
	$qsub .= "qsub";

	if(0 < scalar(@$dependent_job_ids) && 1 == $should_depend) {
		$qsub .= " -hold_jid ".join(",", @$dependent_job_ids)         if ("SGE" eq $data->{'globalOptions'}->{'queueType'});
		$qsub .= " -W depend=afterok:".join(":", @$dependent_job_ids) if ("PBS" eq $data->{'globalOptions'}->{'queueType'});
	}
	if(defined($data->{$type}->{'threads'}) && 1 < $data->{$type}->{'threads'}) {
		$qsub .= " -pe serial ".$data->{$type}->{'threads'}     if ("SGE" eq $data->{'globalOptions'}->{'queueType'});;
		$qsub .= " -l nodes=1:ppn=".$data->{$type}->{'threads'} if ("PBS" eq $data->{'globalOptions'}->{'queueType'});;
	}
	$qsub .= " -q ".$data->{$type}->{'qsubQueue'} if defined($data->{$type}->{'qsubQueue'});
	$qsub .= " ".$data->{$type}->{'qsubArgs'} if defined($data->{$type}->{'qsubArgs'});
	$qsub .= " -N $output_id -o $run_file.out -e $run_file.err $run_file";

	if(1 == $should_run) {
		if(1 == $dryrun) {
			$FAKEQSUBID++;
			print STDERR "[bfast submit] NAME=$output_id QSUBID=$FAKEQSUBID\n";
			return $FAKEQSUBID;
		}

		# Submit the qsub command
		my $qsub_id=`$qsub`;
		$qsub_id = "$qsub_id";
		chomp($qsub_id);

		# There has to be a better way to get the job ids (?)
		if($qsub_id =~ m/Your job (\d+)/) {
			$qsub_id = $1;
		}
		die("Error submitting QSUB_COMMAND=$qsub\nQSUB_ID=$qsub_id\n") unless (0 < length($qsub_id));
		if($qsub_id !~ m/^\S+$/) {
			die("Error submitting QSUB_COMMAND=$qsub\nQSUB_ID=$qsub_id\n") unless (0 < length($qsub_id));
		}

		if(!$quiet) {
			print STDERR "[bfast submit] NAME=$output_id QSUBID=$qsub_id\n";
		}

		return $qsub_id;
	}
	else {
		if(!$quiet) {
			print STDERR "[bfast submit] NAME=$output_id QSUBID=Not submitted\n";
		}

		return QSUBNOJOB;
	}
}

__END__
=head1 SYNOPSIS

bfast.submit.pl [options] 

=head1 OPTIONS

=over 8

=item B<-help>
Print a brief help message and exits.

=item B<-schema>
Print the configuration XML schema.

=item B<-man>
Prints the manual page and exits.

=item B<-quiet>
Do not print any submit messages.

=item B<-startstep>
Specifies on which step of the alignment process to start (default: match). The
values can be "match", "localalign", "postprocess", or "sam".

=item B<-dryrun>
Do everything but submit the jobs.

=item B<-config>
The XML configuration file.

=back

=head1 DESCRIPTION

B<bfast.submit.pl> will create the necessary shell scripts for B<BFAST> to be
run on a supported cluster (SGE and PBS).  It will also submit each script
supporting job dependencies.  The input to B<bfast.submit.pl> is an XML
configuration file.  To view the schema for this file, please use the 
I<-schema> option.

To use this script to run B<BFAST>, the necessary input files must be created. 
This includes creating a BFAST reference genome file (with an additional color space version
for ABI SOLiD data), the BFAST index file(s), and a 
samtools indexed reference FASTA file (using 'samtools faidx').  Additionally,
the reads, if not already in B<FASTQ> format then the input files must be 
properly reformatted and can be optionally split for parallel processing 
(please observe B<BFAST>'s paired-end or mate-end B<FASTQ> 
format).  Optional input files must also be created before using this script.
For a description of how to create these files, please see the 
B<bfast-book.pdf> in the B<BFAST> distribution.

The behaviour of this script is as follows.  For each B<FASTQ> file found in the
reads directory, one index search process using B<bfast match>, one local alignment
process using B<bfast align> will be performed, one post-processing process using 
B<bfast postprocess>, and one import to the B<SAM> format using B<SAMtools> (see 
http://samtools.sourceforge.net) or B<Picard> (see 
http://picard.sourceforge.net) will be performed.  Finally, we merge all B<SAM> 
files that have been created for each input B<FASTQ> file.  Observe that all the
B<SAM> files will be sorted by location.  We submit each job separately using
the job dependency capabilities of the cluster scheduler.  All output files
will be created in the output directory, all run files as well as the stdout and
stderr streams will be found in the run files directory, and all temporary files
will be created in the temporary directory.

Please report all bugs to nhomer@cs.ucla.edu or bfast-help@lists.sourceforge.net.

=head1 REQUIREMENTS

This script requires the XML::Simple library, which can be found at:
http://search.cpan.org/dist/XML-Simple

=cut
