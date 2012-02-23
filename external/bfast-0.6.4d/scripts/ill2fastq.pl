#!/usr/bin/perl -w

# Author: Nils Homer
# At some point, a C implementation of this script would be nice.

use strict;
use warnings;
use Getopt::Std;

select STDERR; $| = 1;  # make unbuffered

my %opts;
my $version = '0.1.1';
my $usage = qq{
Usage: ill2fastq.pl [[ -b <bar code length> | -B ] -n <number of reads> -o <output prefix> -q -s] <input prefix>

This script will convert Illumina output files (*qseq.txt or 
*sequence files) to the BFAST fastq multi-end format.  For 
single-end reads that do not have more than one end (neither 
mate-end nor paired end), the output format is strictly fastq 
format.  For multi-end data (mate-end, paired end ect.) each 
end is outputted consecutively in 5'->3' ordering in fastq 
blocks.  Therefore, the output reamins strictly conforming 
to the fastq format, but keeps intact the relationships between
sequences that all originate from the same peice of DNA.
We assume for paired end data that all data is able to be
paired.  The -n option is useful to split the output into 
separate files each consisting of the specified number of reads.
In this case, you will need to also specify an output prefix 
using the -o option.


All *qseq.txt or *sequence.txt files will be inferred from the 
specified directory and accompanying prefix.  For paired end data, 
reads in
<prefix>_1_XXXX_qseq.txt and <prefix>_2_XXXX_qseq.txt will be
paired (similarly for *sequence.txt).  For example, if we wish 
to create a paired end fastq file the first lane, the command 
should be:

ill2fastq.pl s_1

Using the -B option will try to determine the barcode length 
automatically using some simple heuristics. This will be much
slower because it makes several passes through the datafiles.
};

my $ROTATE_NUM = 100000;

getopts('b:Cn:o:sqB', \%opts);
die($usage) if (@ARGV < 1);

my $num_reads = -1;
my $output_prefix = "";
my $input_suffix = "";
my $input_suffix_state = -1; # 0 - qseq, 1 -> sequence
if(defined($opts{'n'})) {
	$num_reads = $opts{'n'};
	if(!defined($opts{'o'})) {
		die("Error.  The -o option must be specified when using the -n option.  Terminating!\n");
	}
	die if ($num_reads < 1);
	$output_prefix = $opts{'o'};
	die if (length($output_prefix) <= 0);
}
else {
	if(defined($opts{'o'})) {
		die("Error.  The -o option was specified without the -n option.  Terminating!\n");
	}
}
if(defined($opts{'q'}) && defined($opts{'s'})) {
	die("Error.  Both -q and -s options were specified.  Terminating!\n");
}
elsif(defined($opts{'q'})) {
	$input_suffix = "qseq.txt";
	$input_suffix_state = 0;
}
elsif(defined($opts{'s'})) {
	$input_suffix = "sequence.txt";
	$input_suffix_state = 1;
}
else {
	die("Error.  The -q or -s option must be specified.  Terminating!\n");
}

if(defined($opts{'B'}) && defined($opts{'b'})) {
	die("Error. Both -b and -B options were specified. Terminating!\n");
}

my $infer_barcode_length = 0;
if(defined($opts{'B'})) {
	$infer_barcode_length = 1;
}

my $barcode_length = 0;
if(defined($opts{'b'})) {
	$barcode_length = $opts{'b'};
	die "barcode length is not a number. Terminating!\n" unless($barcode_length =~ /\d+/);
}

my $input_prefix = shift @ARGV;

# Get input files
my @files_one = ();
my @files_two = ();
GetDirContents($input_prefix."_1", \@files_one);
GetDirContents($input_prefix."_2", \@files_two);

if(0 == scalar(@files_one)) {
	die("Error.  Could not find any files.  Terminating!\n");
}
elsif(0 < scalar(@files_two) && scalar(@files_one) != scalar(@files_two)) {
	die("Error.  Did not find an equal number of paired end files.  Terminating!\n");
}

# Sort the file names
@files_one = sort @files_one;
@files_two = sort @files_two;

# If '-B' was specified. Try to figure out what the barcode length is
my $has_illumina_barcode = 0;
if($infer_barcode_length) {
	# Check if there are illuminia barcodes.
	my $qseq_dir = $input_prefix;
	$qseq_dir =~ /([^\/]+)$/;
	$qseq_dir =~ s/$1$//;

	if( -e "$qseq_dir/config.xml" ) {
		open(FH, "$qseq_dir/config.xml") || die;
		while(<FH>) {
			if(/<Barcode>/) {
				$has_illumina_barcode = 1;
				last;
			}
		}
		close(FH) || die;
	}

	if(0 == $has_illumina_barcode) {
		# Check every 1000th read
		$barcode_length = &infer_barcode_len(1000);
	}
}

my $FH_index = 0;
my $output_file_num = 1;
my $output_num_written = 0;

if(0 < $num_reads) {
	open(FHout, ">$output_prefix.$output_file_num.fastq") || die;
}

if(1 == $has_illumina_barcode) { # Illumina barcodes in second qseq file.
	while($FH_index < scalar(@files_one)) {
		my $min_read_name = "";

		open(FH_one, "$files_one[$FH_index]") || die;
		open(FH_two, "$files_two[$FH_index]") || die;
		my %read_one = ();
		my %read_two = ();
		while(1 == get_read(*FH_one, \%read_one, $barcode_length, 1, $input_suffix_state) &&
			1 == get_read(*FH_two, \%read_two, $barcode_length, 2, $input_suffix_state)) {
			if(0 != cmp_read_names($read_one{"NAME"}, $read_two{"NAME"})) {
				print STDERR "".$read_one{"NAME"}."\t".$read_two{"NAME"}."\n";
				die;
			}
			if(0 < $num_reads) {
				if($num_reads <= $output_num_written) {
					close(FHout);
					$output_file_num++;
					$output_num_written=0;
					open(FHout, ">$output_prefix.$output_file_num.fastq") || die;
				}
				print FHout "".$read_one{"NAME"}."_BC:".$read_two{"SEQ"}."\n".$read_one{"SEQ"}."\n+\n".$read_one{"QUAL"}."\n";

				$output_num_written++;
			}
			else {
				print STDOUT "".$read_one{"NAME"}."_BC:".$read_two{"SEQ"}."\n".$read_one{"SEQ"}."\n+\n".$read_one{"QUAL"}."\n";
#				print STDOUT "".$read_two{"NAME"}."\n".$read_two{"SEQ"}."\n+\n".$read_two{"QUAL"}."\n";
			}
		}
		close(FH_one);
		close(FH_two);
		$FH_index++;
	}


}
elsif(0 == scalar(@files_two)) { # Single end
	while($FH_index < scalar(@files_one)) {
		open(FH_one, "$files_one[$FH_index]") || die;
		my %read = ();
		while(1 == get_read(*FH_one, \%read, $barcode_length, 1, $input_suffix_state)) {
			if(0 < $num_reads) {
				if($num_reads <= $output_num_written) {
					close(FHout);
					$output_file_num++;
					$output_num_written=0;
					open(FHout, ">$output_prefix.$output_file_num.fastq") || die;
				}
				print FHout "".$read{"NAME"}."\n".$read{"SEQ"}."\n+\n".$read{"QUAL"}."\n";
				$output_num_written++;
			}
			else {
				print STDOUT "".$read{"NAME"}."\n".$read{"SEQ"}."\n+\n".$read{"QUAL"}."\n";
			}
			%read = ();
		}
		close(FH_one);
		$FH_index++;
	}
}
else { # Paired end
	while($FH_index < scalar(@files_one)) {
		print "ON $FH_index\n";
		my $min_read_name = "";

		open(FH_one, "$files_one[$FH_index]") || die;
		open(FH_two, "$files_two[$FH_index]") || die;
		my %read_one = ();
		my %read_two = ();
		while(1 == get_read(*FH_one, \%read_one, $barcode_length, 1, $input_suffix_state) &&
			1 == get_read(*FH_two, \%read_two, $barcode_length, 2, $input_suffix_state)) {
			if(0 != cmp_read_names($read_one{"NAME"}, $read_two{"NAME"})) {
				print STDERR "".$read_one{"NAME"}."\t".$read_two{"NAME"}."\n";
				die;
			}
			if(0 < $num_reads) {
				if($num_reads <= $output_num_written) {
					close(FHout);
					$output_file_num++;
					$output_num_written=0;
					open(FHout, ">$output_prefix.$output_file_num.fastq") || die;
				}
				print FHout "".$read_one{"NAME"}."\n".$read_one{"SEQ"}."\n+\n".$read_one{"QUAL"}."\n";
				print FHout "".$read_two{"NAME"}."\n".$read_two{"SEQ"}."\n+\n".$read_two{"QUAL"}."\n";
				$output_num_written++;
			}
			else {
				print STDOUT "".$read_one{"NAME"}."\n".$read_one{"SEQ"}."\n+\n".$read_one{"QUAL"}."\n";
				print STDOUT "".$read_two{"NAME"}."\n".$read_two{"SEQ"}."\n+\n".$read_two{"QUAL"}."\n";
			}
		}
		close(FH_one);
		close(FH_two);
		$FH_index++;
	}
}
if(0 < $num_reads) {
	close(FHout);
}

sub cmp_read_names {
	my ($a, $b) = @_;

	# Remove the bar codes if necessarily
	$a =~ s/_BC:.+//;
	$b =~ s/_BC:.+//;

	# characters, the numbers, then characters, then numbers, ...
	# recursion is for sissies
	while(0 != length($a) &&
		0 != length($b)) {
		my $a_char = "";
		my $b_char = "";
		my $a_num = 0;
		my $b_num = 0;
		if($a =~ m/^(\d+)(.*?)$/) {
			$a_num = $1;
			$a = $2;
		}
		elsif($a =~ m/^(\D+)(.*?)$/) {
			$a_char = $1;
			$a = $2;
		}
		if($b =~ m/^(\d+)(.*?)$/) {
			$b_num = $1;
			$b = $2;
		}
		elsif($b =~ m/^(\D+)(.*?)$/) {
			$b_char = $1;
			$b = $2;
		}
		# Compare numbers then letters
		if(!($a_char eq $b_char)) {
			return ($a_char cmp $b_char);
		}
		elsif($a_num != $b_num) {
			return ($a_num <=> $b_num);
		}
	}

	return (length($a) <=> length($b));
}

sub GetDirContents {
	my ($prefix, $dirs) = @_;

	my $dir = "";
	if($prefix =~ m/^(.+\/)([^\/]*)/) {
		$dir = $1;
		$prefix = $2;
	}
	else {
		$dir = "./";
	}

	local *DIR;
	opendir(DIR, "$dir") || die("Error.  Could not open $dir.  Terminating!\n");
	@$dirs = grep !/^\.\.?$/, readdir DIR;
	@$dirs = sort grep /^$prefix.*$input_suffix/, @$dirs;
	for(my $i=0;$i<scalar(@$dirs);$i++) {
		@$dirs[$i] = $dir."".@$dirs[$i];
	}
	close(DIR);
}

sub get_read {
	my ($FH, $read, $barcode_length, $end, $input_suffix_state) = @_;

	if(0 == $input_suffix_state) {
		return parse_qseq_line($FH, $read, $barcode_length, $end);
	}
	elsif(1 == $input_suffix_state) {
		return parse_sequence_line($FH, $read, $barcode_length, $end);
	}
	else {
		die("Error.  Could not understand input suffix state: $input_suffix_state.  Terminating!\n");
	}
	return 0;
}

sub parse_qseq_line {
	my ($FH, $read, $barcode_length, $end) = @_;

	if(defined(my $line = <$FH>)) {
		my @arr = split(/\s+/, $line);

		my $name = "@".$arr[0]."_".$arr[1]."_".$arr[2]."_".$arr[3]."_".$arr[4]."_".$arr[5]."_".$arr[6]."";
		my $qual = $arr[9];
		my $seq = $arr[8];

		($read->{"NAME"}, $read->{"SEQ"}, $read->{"QUAL"}) = convert_ill($name, $seq, $qual, $barcode_length, $end);

		return 1;
	}
	else {
		return 0;
	}
}

sub parse_sequence_line {
	my ($FH, $read, $barcode_lenghth, $end) = @_;

	if(defined(my $name = <$FH>) &&
		defined(my $seq = <$FH>) && 
		defined(my $comment = <$FH>) &&
		defined(my $qual = <$FH>)) {
	
		chomp($name); chomp($seq); chomp($qual);

		($read->{"NAME"}, $read->{"SEQ"}, $read->{"QUAL"}) = convert_ill($name, $seq, $qual, $barcode_length, $end);

		# strip end ID off of read name
		$read->{"NAME"} =~ s/\/$end$//;

		return 1;
	}
	else {
		return 0;
	}
}

sub convert_ill {
	my ($name, $seq, $qual, $barcode_length, $end) = @_;
	
	# Convert Illumina PHRED to Sanger PHRED
	for(my $i=0;$i<length($qual);$i++) {
		my $Q = ord(substr($qual, $i, 1)) - 64;
		substr($qual, $i, 1) = chr(($Q<=93? $Q : 93) + 33);
	}

	$seq =~ tr/\./N/; # 'bfast postprocess' bails on '.'

	if(0 < $barcode_length) {
		$name .= "_BC:".substr($seq, 0, $barcode_length)."";
		$seq = substr($seq, $barcode_length);
		$qual = substr($qual, $barcode_length);
	}

	if(1 != $end) {
		$seq = reverse($seq);
		$seq =~ tr/ACGTacgt/TGCAtgca/;
		$qual = reverse($qual);
	}
	return ($name, $seq, $qual);
}

sub infer_barcode_len() {
	my ($skip_reads) = @_;

	my @reads; # collect sample reads;

	my $seen_reads = 0;
	my $barcode_length = 0; # required by get_read()
	my $FH_index = 0;

	if(0 == scalar(@files_two)) { # Single end
		while($FH_index < scalar(@files_one)) {
			open(FH_one, "$files_one[$FH_index]") || die;
			my %read = ();
			while(1 == get_read(*FH_one, \%read, $barcode_length, 1, $input_suffix_state)) {
				$seen_reads++;
				unless($seen_reads % $skip_reads) { # grab one read for every $skip_reads
					push(@reads, $read{"SEQ"});
				}
				%read = ();
			}
			close(FH_one);
			$FH_index++;
		}
	} else {                      # Paired end
		while($FH_index < scalar(@files_one)) {
			open(FH_one, "$files_one[$FH_index]") || die;
			open(FH_two, "$files_two[$FH_index]") || die;
			my %read_one = ();
			my %read_two = ();

			# Notice we pass '1' as the 4th argument -- we do not want get_read() to do a reverse
			# compliment when trying to deterime the barcode.
			while(1 == get_read(*FH_one, \%read_one, $barcode_length, 1, $input_suffix_state) &&
				1 == get_read(*FH_two, \%read_two, $barcode_length, 1, $input_suffix_state)) {
				if(0 != cmp_read_names($read_one{"NAME"}, $read_two{"NAME"})) {
					print STDERR "".$read_one{"NAME"}."\t".$read_two{"NAME"}."\n";
					die;
				}
				$seen_reads++;
				unless($seen_reads % $skip_reads) {
					push(@reads, $read_one{"SEQ"});
					push(@reads, $read_two{"SEQ"});
				}
			}
			close(FH_one);
			close(FH_two);
			$FH_index++;
		}
	}

	# Once the reads have been sampled check the heuristic.
	my $num_reads = scalar(@reads);

	if($num_reads < 1000) {
		die("Not enough reads to guess barcode length. Terminating!");
	}

	# barcodes are between 11 and 3 bases
	for my $num_bc_bases (qw(10 9 8 7 6 5 4 3 2)) {
		my $t_count = 0; # number of T's at given bp position accross the sample reads
		for my $read (@reads) {
			my @bases = split(//, $read);
			$t_count++ if($bases[$num_bc_bases] =~ /([Tt])/);
		}

		if( ( $t_count / $num_reads) > 0.66) { # this is probably the barcode
			return $num_bc_bases + 1;
		}
	}

	return 0; # Doesn't seem to have a barcode.
}

