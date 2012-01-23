/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 *  api_example_pileup.cpp
 *  SamUtils
 *
 *	has to be compiled with -std=c++0x
 *
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
#include "types/Pileup.h"
#include <map>

using namespace std;
int main(int argc, const char* argv[]) {
	
	
	string bam_file = string(argv[1]);
	
	BAMReader reader(bam_file);
	reader.open(); //don't like doing this, should just be able to get an iterator
	
	
	char delim = '\t';
	
	//setup required data
	int ref_index			= 0;
	coord_t region_begin	= 100;
	coord_t region_end		= 200;
	
	std::vector<int> phreds;
	phreds.push_back(7);
	phreds.push_back(10);
	phreds.push_back(17);
	phreds.push_back(20);
	phreds.push_back(47);
	
	
	
	PileupFactory pileups(ref_index, region_begin, region_end);
	for (BAMReader::iterator i = reader.get_iterator(ref_index, region_begin, region_end); i.good(); i.next()) {		
		BAMRead read = i.get();
		BAMUtils util(read);
		cerr << util.get_name() <<delim<<read.get_pos()<< endl;
		pileups.insert_util(util);
			
	}
	
	
	//iterate over pileups
	cerr << "Iteration type 1, unordered" << endl;
	for (PileupFactory::pileup_iterator itr = pileups.get_pileup_iterator(); itr.good(); itr.next()) {
		Pileup& p = itr.get();
		cerr <<delim<< "Ref id:"<<delim<< p.get_tid() <<delim<< "position in genome:"<<delim<< p.get_pos() <<delim<<"num reads:"<<delim<< p.get_num_reads() << endl;
	}
	cerr << "Iteration type 2, order is important" << endl;
	//iterate a slightly different way
	for (int pos_in_genome = region_begin; pos_in_genome <= region_end; pos_in_genome++) {
		Pileup p = pileups.get_pileup(pos_in_genome);
		cerr <<delim<< "Ref id:"<<delim<< p.get_tid() <<delim<< "position in genome:"<<delim<< p.get_pos() <<delim<<"num reads:"<<delim<< p.get_num_reads() << endl;

	}
	
	
	
	
		
	
	
}
