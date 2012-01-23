/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 *  api_example_with_iteration_and_cigar.cpp
 *  SamUtils
 *
 *	has to be compiled with -std=c++0x
 *
 *
 *  Created by Michael Lyons on 6/14/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */

#include <string>
#include "BAMReader.h"
#include "types/BAMRead.h"
#include "types/Cigar.h"
#include "BAMUtils.h"
#include <map>

using namespace std;
int main(int argc, const char* argv[]) {

	
	string bam_file = string(argv[1]);
	
	BAMReader reader(bam_file);
	reader.open(); //don't like doing this, should just be able to get an iterator
		
	std::map<int, int>	read_length_histo;
	std::map<int, int>  padded_histo;
	std::map<int, int>	nqlen_histo;
	std::map<int, int>	nqlen_clipped_histo;
	int histo_lower_bound = 25;
	int histo_upper_bound = 400;
	char delim = '\t';

	for (BAMReader::iterator i = reader.get_iterator(); i.good(); i.next()) {
		
		BAMRead read = i.get();
		
		
		int length = read.get_tlen();
		Cigar cig = read.get_cigar();
		int clipped_len = 0;
		//find padded element.  this is slow because it's either at the first position or the last, but it's the simplest
		//and shows the basic way to handle cigar elements
		for (Cigar::iterator cig_itr = cig.get_iterator(); cig_itr.good(); cig_itr.next()) {
			if (cig_itr.op() == 'S') {
				clipped_len += cig_itr.len();
			}
		}
		
					//BAMRead, qscores, start slop
		BAMUtils util(read, "7,10,17,20,47", 0);

		for (int j = histo_lower_bound; j <= histo_upper_bound; j+=histo_lower_bound) {
			if (length >= j) {
				read_length_histo[ j ]++;
			}
			
			if ((length - clipped_len) >= j) {
				padded_histo[ j ]++;
			}
			
			if (util.get_q_length() >= j) {
				nqlen_histo[ j ]++;
			}
			
			if (util.get_q_length() + util.get_soft_clipped_bases() >= j) {
				nqlen_clipped_histo[ j ]++;
			}
			
		}
		cerr << read.get_qname() << delim << "length:" << length << delim << "padded_len:" << clipped_len << delim << "len w/o padding:" << (length - clipped_len);
		cerr << delim << "nqlen:" << util.get_q_length() << delim << "nqlen + clipped:" << delim << (util.get_q_length() + util.get_soft_clipped_bases()) <<  endl;
		
	}

	cout << "length" << delim << "total read" << delim << "clipped reads" << delim << "qLen" << delim << "qLen + clipped" << endl;
	
	for (auto j = read_length_histo.begin(); j != read_length_histo.end(); ++j) {
		cout << j->first << delim << j->second << delim << padded_histo[ j->first ] << delim << nqlen_histo[ j->first ] << delim << nqlen_clipped_histo[ j->first ] << endl;
	}
	

	
	
}
