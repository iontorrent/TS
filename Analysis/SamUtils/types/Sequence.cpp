/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 *  Sequence.cpp
 *  SamUtils
 *
 *  Created by Michael Lyons on 1/4/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */


#include <sstream>
#include <iostream>
#include "Sequence.h"




Sequence::Sequence(const bam1_t* BAM_record) {
	
	_init();
	bam_record = BAM_record;
	seq_length = bam_record->core.l_qseq;
	
}


void Sequence::_init() {
	
	bam_record = NULL;
	seq_length = -1;
	
}


long Sequence::get_length() const {
	return seq_length;
	
}


Sequence::iterator Sequence::get_iterator() {
	return Sequence::iterator(bam_record);
	
}


std::string Sequence::to_string() {
	std::ostringstream strm;
	
	for (Sequence::iterator i = get_iterator(); i.good(); i.next()) {
		strm << i.get();
		
	}
	return strm.str();
	
}


//iterator functionality
Sequence::iterator::iterator(const bam1_t* BAM_record) {
	_init();
	bam_record = BAM_record;
	bam_seq = bam1_seq(bam_record);
	seq_len = bam_record->core.l_qseq;
	
}

void Sequence::iterator::_init() {
	bam_record = NULL;
	position = 0;

}

void Sequence::iterator::next() {
	position++;	//next SHOULD handle this, and in order to do so
				//I am pre-incrementing position before returning the character
}

char Sequence::iterator::get() {
	return bam_nt16_rev_table[bam1_seqi(bam_seq, position)];
	
}

bool Sequence::iterator::good() const {
	
	if (position < seq_len) {
		return  true;
	} else {
		return false;
	}
	
}

void Sequence::iterator::set_position(size_t pos) {
	//just to be safe
	if (pos < seq_len) {
		position = pos;
	} else {
		position = seq_len; //iteration isn't possible at this point which 
							//is what we want
	}


}

