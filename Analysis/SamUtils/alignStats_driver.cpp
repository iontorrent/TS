/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 *  driver.cpp
 *  SamUtils
 *
 *  Created by Michael Lyons on 2/2/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */




#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include "Utils.h"
#include "OptArgs.h"
#include "alignStats.h"
#include <sys/types.h>
#include <sys/stat.h>

//#include <tr1/unordered_map>

using namespace std;


void usage();
void get_options(AlignStats::options& opt, OptArgs& opts);
bool check_args(AlignStats::options& opt);


int main(int argc, const char* argv[])
{
	
	OptArgs opts;
	opts.ParseCmdLine(argc, argv);
	AlignStats::options opt;
	get_options(opt, opts);
	opt.score_flows = (opt.flow_order != "");
	
	// Open a BAM file:	
	string extension = get_file_extension(opt.bam_file);
	cerr << "[alignStats] sam/bam file: " << opt.bam_file << endl;
	
	if (opt.list_of_files.length() > 0) {
		
		ifstream in(opt.list_of_files.c_str());
		if (in.fail()) {
			cerr << "[alignStats] list of input sams/bams file: " << opt.list_of_files << " couldn't be opened" << endl;
			exit(1);
		} else {
			string line;
			while (in.good()) {
				getline(in, line);
				//line has stuff from genome info now
				opt.out_file = line;
				opt.bam_file = line;
				AlignStats stats(opt);
				cerr << "[alignStats] threads: " << opt.num_threads << " buffer size: " << opt.buffer_size << std::endl;
				stats.go();
								
			}
			in.close();
		}
		
		
		
		
	} else {
		AlignStats stats(opt);
		
		

		if (opt.debug_flag) {
			cerr << "[alignStats] debugging on" << endl;
		}
		//
		if (opt.skip_cov_flag) {
			cerr << "[alignStats] skipping coverage calculations" << endl;
			
		}
		cerr << "[alignStats] threads: " << opt.num_threads << " buffer size: " << opt.buffer_size << std::endl;
		stats.go();
		
	}

	
		
	
}
//utility functions, boring stuff

//available alphabet letters
/* r */

//should get rid of -r
void get_options(AlignStats::options& opt, OptArgs& opts) {
	opts.GetOption(opt.bam_file,				"",					'i',"infile");
	opts.GetOption(opt.out_file,				opt.out_file,				'o',"outfile");
	opts.GetOption(opt.genome_info,				opt.genome_info,			'g',"genomeinfo");
	opts.GetOption(opt.flow_order,				"",					'F',"flowOrder");
	opts.GetOption(opt.read_to_keep_file,			"",					'K',"readsToKeep");
	opts.GetOption(opt.read_to_reject_file,			"",					'R',"readsToReject");
	opts.GetOption(opt.filter_length,			"20",					'l',"filterlength");
	opts.GetOption(opt.q_scores,				opt.q_scores,				'q',"qScores");
	opts.GetOption(opt.start_slop,				"0",					's',"startslop");
	opts.GetOption(opt.sam_parsed_flag,			"0",					'p',"samParsedFlag");
	opts.GetOption(opt.total_reads,				"0",					't',"totalReads");
	opts.GetOption(opt.read_limit,				"0",					'-',"readLimit");
	opts.GetOption(opt.sample_size,				"0",					'z',"sampleSize");
	opts.GetOption(opt.help_flag,				"false",				'h',"help");
	opts.GetOption(opt.num_threads,				"4",					'n',"numThreads");
	opts.GetOption(opt.buffer_size,				"10000",				'B',"bufferSize");
	opts.GetOption(opt.skip_cov_flag,			"false",				'x',"skipCoverage");
	opts.GetOption(opt.debug_flag,				"false",				'y',"debug");
	opts.GetOption(opt.iupac_flag,				"true",					'u',"iupac");
	opts.GetOption(opt.keep_iupac,				"false",				'k',"keepIUPAC");
	opts.GetOption(opt.max_coverage,			"20000",				'v',"maxCoverage");
	opts.GetOption(opt.flow_err_file,			opt.flow_err_file,			'-' ,"flowErrFile");
	opts.GetOption(opt.align_summary_file,			opt.align_summary_file,			'a',"alignSummaryFile");
	opts.GetOption(opt.align_summary_min_len,		"50",					'c',"alignSummaryMinLen");
	opts.GetOption(opt.align_summary_max_len,		"400",					'm',"alignSummaryMaxLen");
	opts.GetOption(opt.align_summary_len_step,		"50",					'e',"alignSummaryLenStep");
	opts.GetOption(opt.align_summary_max_errors,		 "3",					'j',"alignSummaryMaxErr");
	opts.GetOption(opt.stdin_sam_flag,			 "false",				'S',"stdinSam");
	opts.GetOption(opt.stdin_bam_flag,			 "false",				'b',"stdinBam");
	opts.GetOption(opt.list_of_files,			 opt.list_of_files,			'L',"listOfFiles");
	opts.GetOption(opt.align_summary_filter_len, 		"50",					'f',"alignSummaryFilterLen");
	opts.GetOption(opt.align_summary_filter_accuracy, 	"0.9",					'w',"alignSummaryFilterAcc");
	opts.GetOption(opt.truncate_soft_clipped,		"true",					'T',"cutClippedBases");
	opts.GetOption(opt.three_prime_clip,			"0",					'C',"3primeClip");
	opts.GetOption(opt.round_phred_scores,			"true", 				'X',"roundPhredScores");
    opts.GetOption(opt.five_prime_justify,			"true", 				'P',"5primeJustify");
    opts.GetOption(opt.output_dir,                "",                       'd',"outputDir");
    opts.GetOption(opt.merged_region_file,                "",                       'r',"mergedRegionFile");

	if(opt.read_limit <= 0)
		opt.read_limit = LONG_MAX;









	if (!check_args(opt)) {
		usage();
		exit(1);
	}
}

void usage() {
	AlignStats::options tmp_opt;
	
	cout << endl
	<< "[alignStats] - create alignment statistics information to be displayed on torrent browser." << endl
	<< endl
	<< "usage: " << endl
	<< "  alignStats [-i samfile.or.bam]" << endl
	<< endl
	<< "Help: " << endl
	<< "  -h"<<"\t\t\t\t\t"<<": this (help) message" << endl
	<< "I/O options" << endl
	<< "  -i,--infile"<<"\t\t\t\t"<<": sam file of alignments"<< endl
	<< "  -b,--stdinBam"<<"\t\t\t\t"<<": input is binary and from stdin"<< endl
	<< "  -S,--stdinSam"<<"\t\t\t\t"<<": input is raw text sam file and from stdin"<< endl
	<< "  -o,--out"<<"\t\t\t\t"<<": output name, prepended to default output files [Default value:  Default]" << endl
	<< "  -r, --mergedRegion"<<"\t\t\t\t"<<": file including base postitions for merged region"<< endl
	<<"General options:" << endl
	<< "  -g,--genomeinfo"<<"\t\t\t"<<": genome.info.txt " << endl
	<< "  -K,--readsToKeep"<<"\t\t\t"<<": file specifying list of read names to restrict to" << endl
	<< "  -R,--readsToReject"<<"\t\t\t"<<": file specifying list of read names to exclude" << endl
	<< "  -l,--filterlength"<<"\t\t\t"<<": filtered length [Default value:  \"20\"]" << endl
	<< "  -q,--qScores"<<"\t\t\t\t"<<": comma seperated list of qscores ex.: 7,10,17,20,47.  [Default value:  7,10,17,20,47]"<<endl
	<< "  -s,--startslop"<<"\t\t\t"<<": alignment slop bases (bases to ignore from start of read) [Default value:  0]"<<endl
	<< "  -p,--samParseFlag"<<"\t\t\t"<<": generate sam.parsed(1) on, (0) off [Default value: 0]"<<endl
	<< "  -u,--iupac"<<"\t\t\t\t"<<": user supply either true/false: to correct IUPAC's from reference to read."
									<<"\n\t\t\t\t\t  For example, if the reference contains ATGY and the read is ATGC it will be considered [Default value:  true]"
									<<"\n\t\t\t\t\t  a perfect alignment."<< endl
	<< "  -k,--keepIUPAC"<<	"\t\t\t"<<": user supply either true/false: keep IUPAC codes in qdna and tdna strings.  "
									<<"\n\t\t\t\t\t  use only in conjunction with -u,--iupac.  Parameter has no effect otherwise [Default value: true] "<< endl
	<< "  -t,--totalReads"<<"\t\t\t"<<": total number of reads in original fastq file [Default value: none]" << endl
	<< "  --readLimit"<<"\t\t\t"<<": total number of reads to process [Default: 0 which means all]" << endl
	<< "  -z,--sampleSize"<<"\t\t\t"<<": size of sampled alignment (or from genome.info.txt) [Default value: 0]"<<endl
	<< "  -X,--roundPhredScores"<<"\t\t\t"<<": boolean value to allow rounding in the phred score calculations to handle the colloquial definition\n\t\t\t\t\t  of a \"100 Q17\" which means 2 errors in 100 bases. [Default value:  true]" << endl
        << "  -P,--5primeJustify " <<"\t\t\t"<<": boolean value to follow indel justification \n\t\t\t\t\t  in SAM or force 5' justification.  Use this flag to force  5' justification [Default value:  true]" << endl
	<< "\n\nPerformance Options:" << endl
	<< "  -n,--numThreads"<<"\t\t\t"<<": number of threads.  This is used to create the number of worker threads. "
									 <<"\n\t\t\t\t\t  There is a constant IO thread in conjunction with this # [Default value: 4]" << endl
	<< "  -B,--bufferSize"<<"\t\t\t"<<": buffer size for each thread.  [Default value: 10000]" << endl
	<< "  -v,--maxCoverage"<<"\t\t\t"<<": code will occasionally look ahead when computing coverage to avoid overlapping regions."<<endl
						   <<"\t\t\t\t\t"<<"  This sets a maximum on the number of reads that are allowed in the look ahead queue. [Default value: 20000]" << endl
	<< "  -T,--cutClippedBases"<<"\t\t\t"<<": A flag that will ignore soft clipped bases in all error calculations. [Default value:  true]"<<endl	
	<< "  -3,--3primeClip"<<"\t\t\t"<<": An integer value that will ignore a specific # of soft clipped bases when calculating alignment statistics.  [Default value:  0]" << endl
	<<"Error Table options:"<<endl
	<< "  --flowErrFile"<<"\t\t\t"<<": optional file with reporting on number of HP and base errors per flow"<<endl
	<< "  -a,--alignSummaryFile"<<"\t\t\t"<<": optional file that will produce a tab delimited table containing read lengths and the # of reads aligned"<<endl
										  <<"\t\t\t\t\t  and the total # of reads with 0-alignSummaryMaxErr [Default value: alignTable.txt]" << endl
	<< "  -c,--alignSummaryMinLen"<<"\t\t"<<": minimum length for read to be considered in table [Default value: 50]" << endl
	<< "  -m,--alignSummaryMaxLen"<<"\t\t"<<": maximum length for read to be considered in table [Default value: 400]" << endl
	<< "  -e,--alignSummaryLenStep"<<"\t\t"<<": step size.  table will be in incremeents of minimum length + alignSummaryLenStep, etc. [Default value: 50]" << endl
	<< "  -j,--alignSummaryMaxErr"<<"\t\t"<<": maximum number of errors to track in output [Default value: 3]" << endl
	<<"Error table filtering explained: ( (alignSummaryFilterLen - errors within this length of the read) / alignSummaryFilterLen ) >= alignSummaryFilterAcc" << endl
	<< "  -f,--alignSummaryFilterLen"<<"\t\t"<<": alignment table filtering criteria length [Default value: 50]" << endl
	<< "  -w,--alignSummaryFilterAcc"<<"\t\t"<<": alignment table filtering criteria accuracy [Default value: 0.9]" << endl
	
	
	<< endl;
}

	
bool make_output_dir(string output_dir) { 
  //just to simplify, set output prefix to output_dir + "/" + output_prefix
  if (mkdir( output_dir.c_str(), 0777) == -1) {
    return false;
  } else {
    return true;
  }
  
}

bool check_args(AlignStats::options& opt) {
	if (opt.help_flag) {
		return false;
	}
	if (!opt.stdin_sam_flag && !opt.stdin_bam_flag && opt.list_of_files.length() == 0) {
		string extension = get_file_extension(opt.bam_file);		
		if ((extension != "sam" && extension != "bam") && (extension != "SAM" && extension != "BAM")) {
                    cerr << "[alignStats] Error!\n"
                            "[alignStats] The user needs to provide an input file [-i]\n"
                            "[alignStats] Exiting." << endl;
                        return false;
		}
	}	
  if ( opt.output_dir.length() > 0) {
    struct stat st;
    if (stat(opt.output_dir.c_str(), &st) != 0) {
      if ( !make_output_dir( opt.output_dir ) ) {
        cerr << "[alignStats] Error!" << endl
        << "[alignStats] Unable to create output directory: " << opt.output_dir << endl
        << "[alignStats] Exiting." << endl;
        return false;
      }
    }
    opt.out_file = opt.output_dir + "/" + opt.out_file;
    opt.align_summary_file = opt.output_dir + "/" + opt.align_summary_file;
  }
	return true;
}



