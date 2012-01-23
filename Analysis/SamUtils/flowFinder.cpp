/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 *  flowFinder.cpp
 *  SamUtils_pileup
 *
 *  Created by Michael Lyons on 5/10/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */


#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <string>

#include <fstream>
#include "bam.h"
#include "Utils.h"
#include "OptArgs.h"
#include "BAMReader.h"
#include "types/BAMRead.h"
#include "BAMUtils.h"
#include "file-io/ion_util.h"
#include "flow_utils.h"

using namespace std;


struct options {
	options()
	: bam_file("")
	, bam_region("")
	, region_list("")
	, flow_order("")
	, key_seq("")
	, help_flag(false)
	{}
	string			bam_file; //-i,--i
	string			bam_region; //like this:  chr1:500-700
	string			region_list;
	string			flow_order;
	string			key_seq;
	bool			help_flag;
};
void parse_region_and_output(options& opt, BAMReader& the_reader );


void usage();
void comp(char& nuke);
void get_options(options& opt, OptArgs& opts);
bool check_args(options& opt);


typedef const bam_pileup1_t* pileup_ptr;
typedef std::list<BAMRead> pileup_reads;
typedef std::vector<float>     ionogram;

typedef std::pair<pileup_reads, int> pileup_data;


int main(int argc, const char* argv[]) {
	
		
	
	OptArgs opts;
	opts.ParseCmdLine(argc, argv);
	options opt;
	get_options(opt, opts);
	
	// Open a BAM file:	
	string extension = get_file_extension(opt.bam_file);
	cerr << "[flowFinder] bam file: " << opt.bam_file << endl;
	BAMReader reader;
	reader.open(opt.bam_file);
	if (!reader.is_open()) {
		cerr << "[flowFinder] failed to open: " << opt.bam_file << endl;
		exit(1);
	}
	
	
	std::vector<std::string> regions;
	if (opt.region_list.size() > 0) {
		ifstream in(opt.region_list.c_str());;

		if (!in) {
			cerr << "[flowFinder] unable to open region list, file name: " << opt.region_list << endl;
		} else {
			cout << "region" << '\t' << "read name" << '\t' << "col" << '\t' << "row" << '\t' << "position" << '\t' << "adjusted"<< '\t' << "nuc" << endl;

			std::string line;
			while (!in.eof()) {
				in >> line;
				opt.bam_region = line;
				parse_region_and_output(opt, reader);
			}
		}

		
	} else {
		if (opt.bam_region.size() > 0) {
			cout << "region" << '\t' << "read name" << '\t' << "col" << '\t' << "row" << '\t' << "position" << '\t' << "adjusted"<< '\t' << "nuc";
			if (opt.flow_order.size() > 0) cout << '\t' << "flow";
			cout << endl;

			parse_region_and_output(opt, reader);
		}
	}

	
	
	
	
		
	
}
//utility functions, boring stuff

void parse_region_and_output(options& opt, BAMReader& the_reader) {
	for(BAMReader::pileup_generator generator = the_reader.get_generator(opt.bam_region); generator.good(); generator.next()) {
		pileup_data the_data = generator.get();
		int qpos = the_data.second + 1; //make it 1 based instead of 0.  
		//const bam_pileup1_t* p;
		pileup_reads& reads = the_data.first;
		for (pileup_reads::iterator i = reads.begin(); i != reads.end(); ++i) {
			
			
			BAMUtils util(*i);
			int adj_qpos = qpos;
			while (util.is_deletion(adj_qpos)) {
				adj_qpos--;//should move in the 5' direction assuming reference fasta file is 5' -> 3'
				//base = util.get_query_base(p->qpos - 1);
			}
			
			char base;
			int row = -1;
			int col = -1;
			ion_readname_to_rowcol(util.get_name().c_str(), &row, &col);
			
			int flow_num = -1;
			//flowgram_it ionogram;
			ionogram ion;
			//seq2flow(const std::string& seq, const std::string& flow_order, FlowIt dest)
			std::string seq;
			if (util.get_bamread().mapped_reverse_strand()) {
				std::string tmp(util.get_qdna());
				seq.resize(tmp.size());
				std::remove_copy( tmp.begin(), tmp.end(), seq.begin(), '-');
				seq = opt.key_seq + seq;
				flow_num =	getFlowNum(seq, opt.flow_order, (((util.get_t_start() + util.get_t_length() - 1) - adj_qpos) + opt.key_seq.length()));
				base = seq[ (((util.get_t_start() + util.get_t_length() - 1) - adj_qpos) + opt.key_seq.length()) ];
				comp(base);
			} else {
				seq = std::string(opt.key_seq + i->get_seq().to_string());

				flow_num =	getFlowNum(seq, opt.flow_order, (adj_qpos - util.get_t_start())+ opt.key_seq.length() + 1);
				base = util.get_query_base(adj_qpos);
			}			
			//cerr << seq << " " << util.get_t_start() << " " << util.get_t_length() << endl;
			cout << opt.bam_region << '\t' << util.get_name() << '\t' << col << '\t' << row << '\t' << qpos << '\t' << adj_qpos<< '\t' << base; 
			if (flow_num != -1) cout << '\t' << flow_num;
			cout <<endl;
		}
		
	}
	
	
}


void get_options(options& opt, OptArgs& opts) {
	opts.GetOption(opt.bam_file		,"",'i',"infile");
	opts.GetOption(opt.bam_region	,"",'r',"region");
	opts.GetOption(opt.region_list	,"",'l',"regionList");
	opts.GetOption(opt.flow_order	,"",'f',"flowOrder");
	opts.GetOption(opt.key_seq		,"",'k',"keySeq");


	if (!check_args(opt)) {
		usage();
		exit(1);
	}
}

void usage() {
	options tmp_opt;
	
	cout << endl
	<< "flowFinder - emulates samtools pileup with some ion specific tweaks." << endl
	<< endl
	<< "usage: " << endl
	<< "  flowFinder [-i -r (-l -f)]" << endl
	<< endl
	<< "options: " << endl
	<< "  -h"<<"\t\t\t\t\t"<<": this (help) message" << endl
	<< "  -i,--infile"<<"\t\t\t\t"<<": sorted and indexed bam file"<< endl
	<< "  -r,--region"<<"\t\t\t\t"<<": region to parse.  example from hg19 chromosome 1:  chr1:1-1000 " << endl
	<< "  -l,--regionList"<<"\t\t\t\t"<<": list of regions in a file.  one per line. " << endl
	<< "  -f,--flowOrder"<<"\t\t\t\t"<<": flow order used during experiment " << endl
	<< endl;
	
	
	
}


void comp(char& nuke) {
	
	switch (nuke) {
		case 'A':
			nuke = 'T';
			break;
		case 'a':
			nuke = 'T';
			break;
		case 'G':
			nuke = 'C';
			break;
		case 'g':
			nuke = 'C';
			break;
		case 'C':
			nuke = 'G';
			break;
		case 'c':
			nuke = 'G';
			break;
		case 'T':
			nuke = 'A';
			break;
		case 't':
			nuke = 'A';
			break;

		default:
			break;
	}
	
}

bool check_args(options& opt) {
	if (opt.help_flag) {
		return false;
	}
	string extension = get_file_extension(opt.bam_file);
	
	if (( extension != "bam") && (extension != "BAM")) {
		usage();
		exit(1);
	}
	if (!(opt.bam_file.length() > 0)) {
		cerr << "[flowFinder] no input file " << endl;
		return false;
	}
	
	
	return true;
	
}


