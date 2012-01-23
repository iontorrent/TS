#!/usr/bin/perl -w
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# generate a summary of results for experiment data in current directory
# format output as comma separated list.

use strict;
use warnings;
use Cwd;

my $expdate = "";
my $exptime = "";
my $anname  = "";
my $cycles  = 0;
my $project = "";
my $sample  = "";
my $library = "";
my $machine = "";

my $nwashout = 0;
my $nbead    = 0;
my $ndud     = 0;
my $nambig   = 0;
my $nlive    = 0;
my $ntf      = 0;
my $nlib     = 0;

my $nread    = 0;
my $coverage = 0;
my $meanlen  = 0;
my $longest  = 0;
my $Q050_10  = 0;
my $Q050_17  = 0;
my $Q050_20  = 0;
my $Q100_10  = 0;
my $Q100_17  = 0;
my $Q100_20  = 0;
my $Q200_10  = 0;
my $Q200_17  = 0;
my $Q200_20  = 0;

my $cfmetric = -1.0;
my $iemetric = -1.0;
my $drmetric = -1.0;

my $analysis_vers  = "";
my $alignment_vers = "";
my $dbreports_vers = "";
my $sigproc3_vers  = "";

#$filM = "filterMetrics.txt";
my $cafM = "cafieMetrics.txt";
my $algn = "alignment.summary";
my $mask = "bfmask.stats";
my $meta = "expMeta.dat";
my $vers = "version.txt";
#$keyp = "keypass.summary";
#$post = "postbfmask.stats";
my $stat = "status.txt";

sub ParseCafM()
{
	my $n = 2;
	if(open CAFM, $cafM){
		while(<CAFM>){
			#$n = 0 if /^TF\s+=\s+BHB TF2.2/;
			#$n = 0 if /^TF\s+=\s+BHB TF1.2/;
			#$n = 0 if /^TF\s+=\s+cf_08_06_09_08/;
			$n = 0 if /^TF\s+=\s+cf_10_08_10_10/;

			if($n == 1){
		        s/^.+=\s*//;
        		my @sig = split;
				#$cfmetric = $sig[24];
				#$iemetric = $sig[32];
				# TF1.2:
				#$cfmetric = $sig[18];
				#$iemetric = $sig[17];
				# cf_10:
				if($#sig >= 77){
					my $peak37 = $sig[29] + $sig[33] + $sig[37] + $sig[41] + $sig[45];
					my $peak69 = $sig[61] + $sig[65] + $sig[69] + $sig[73] + $sig[77];
					$drmetric  = $peak37  / $peak69;
				}
				
				if($#sig >= 41){
					$iemetric  = $sig[41] / $sig[37];
				}
				
				if($#sig >= 37){
					$cfmetric  = $sig[33] / $sig[37];
				}
			}
			
			++$n;
		}
	}
}

sub ParseMeta()
{
	my $date = "";
	if(open META, $meta){
		while(<META>){
			$date    = $1 if /Experiment Date = (.+)$/;
			$date    = $1 if /Run Date = (.+)$/;
			$anname  = $1 if /Analysis Name = (.+)$/;
			$cycles  = $1 if /Analysis Cycles = (\d+)/;
			$project = $1 if /Project = (.+)$/;
			$sample  = $1 if /Sample = (\S+)/;
			$library = $1 if /Library = (.+)$/;
			$machine = $1 if /PGM = (.+)$/;
		}
	}
	
	($expdate, $exptime) = split /\s+/, $date if $date;
}

sub ParseMask()
{
	if(open MASK, $mask){
		while(<MASK>){
			$nwashout = $1 if /Washout Wells = (\d+)/;
			$nbead    = $1 if /Bead Wells = (\d+)/;
			$ndud     = $1 if /Dud Beads = (\d+)/;
			$nambig   = $1 if /Ambiguous Beads = (\d+)/;
			$nlive    = $1 if /Live Beads = (\d+)/;
			$ntf      = $1 if /Test Fragment Beads = (\d+)/;
			$nlib     = $1 if /Library Beads = (\d+)/;
		}
	}
}

sub ParseAlgn()
{
	if(open ALGN, $algn){
		while(<ALGN>){
			$nread    = $1 if /Total number of Reads = (\d+)/;
			$coverage = $1 if /Filtered .+ Q10 Coverage Percentage = (\d+.\d+)/;
			$coverage = $1 if /Filtered .+ relaxed BLAST coverage percentage = (\d+.\d+)/;
			$meanlen  = $1 if /Filtered .+ Mean Alignment Length = (\d+)/;
			$longest  = $1 if /Filtered .+ Longest Alignment = (\d+)/;
			$Q050_10  = $1 if /Filtered .+ 50Q10 Reads = (\d+)/;
			$Q050_17  = $1 if /Filtered .+ 50Q17 Reads = (\d+)/;
			$Q050_20  = $1 if /Filtered .+ 50Q20 Reads = (\d+)/;
			$Q100_10  = $1 if /Filtered .+ 100Q10 Reads = (\d+)/;
			$Q100_17  = $1 if /Filtered .+ 100Q17 Reads = (\d+)/;
			$Q100_20  = $1 if /Filtered .+ 100Q20 Reads = (\d+)/;
			$Q200_10  = $1 if /Filtered .+ 200Q10 Reads = (\d+)/;
			$Q200_17  = $1 if /Filtered .+ 200Q17 Reads = (\d+)/;
			$Q200_20  = $1 if /Filtered .+ 200Q20 Reads = (\d+)/;
		}
	}
}

sub ParseVers()
{
	if(open VERS, $vers){
		while(<VERS>){
			$analysis_vers  = $1 if /analysis=(\S+)/;
			$alignment_vers = $1 if /alignment=(\S+)/;
			$dbreports_vers = $1 if /dbreports=(\S+)/;
			$sigproc3_vers  = $1 if /sigproc3=(\S+)/;
		}
	}
}


# hack for Broad DOE runs for which run number was not encoded in
# Sample field of expMeta.dat:
#%doe_number_not_in_sample_field = (
#	HEI51	=> 1,
#	FRA140	=> 1,
#	DIR135	=> 1,
#	BOH148	=> 1,
#	HEI50	=> 1,
#	FRA139	=> 1,
#	DIR133	=> 1,
#	BOH146	=> 1,
#);

# Only process runs that have completed Analysis pipeline:
if(-f $stat){
	my @stat = stat "status.txt";
	my $modt = $stat[9];
	my $time = time;
	if($time - $modt > 1000){
		ParseMeta();
		ParseMask();
		ParseAlgn();
		ParseVers();
		ParseCafM();

		my @anpath = split /\//, getcwd;
		my $andir  = $anpath[$#anpath];

		# for Broad DOE July 2010:
		#($snap, $run) = ("", "");
		#($snap, $run) = ($1, $2) if $sample =~ /(.+)_(Run.+)/;
		#if(($snap and $run) or defined $doe_number_not_in_sample_field{$anname}){
		if($cfmetric >= 0 or $iemetric >= 0 or $drmetric >= 0){
			print qq($expdate,$exptime,"$andir","$anname",$cycles,"$project","$sample","$library","$machine",);
			print qq("$analysis_vers","$alignment_vers","$dbreports_vers","$sigproc3_vers",);
			print qq($nwashout,$nbead,$ndud,$nambig,$nlive,$ntf,$nlib,);
			print qq($coverage,$meanlen,$longest,);
			print qq($nread,$Q050_10,$Q050_17,$Q050_20,$Q100_10,$Q100_17,$Q100_20,$Q200_10,$Q200_17,$Q200_20,);
			printf("%.3f,%.3f,%.3f", $cfmetric, $iemetric, $drmetric);
			print qq(\n);

			printf STDERR "done:  %s\n", getcwd;
		}else{
			printf STDERR "skip:  %s\n", getcwd;
		}
	}
}

