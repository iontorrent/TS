/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 *  MD.cpp
 *  SamUtils
 *
 *  Created by Michael Lyons on 1/4/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */


#include <sstream>
#include <iostream>
#include <locale>
#include "MD.h"




MD::MD(const bam1_t* BAM_record) {
	_init();
	bam_record = BAM_record;
	md_length = bam_record->core.l_qseq;
}


void MD::_init() {
	bam_record = NULL;
	md_length = -1;
	
}


long MD::get_length() const {
	return md_length;
	
}


MD::iterator MD::get_iterator() {
	return MD::iterator(bam_record);
	
}


std::string MD::to_string() {
	std::ostringstream strm;
	for (MD::iterator i = get_iterator(); i.good(); i.next()) {
			strm << i.get();
			
	}
		
	
	return strm.str();
	
}


//iterator functionality
/* 
 uint8_t *md_ptr = bam_aux_get(bam_record, "MD");
 if (md_ptr) {
 return bam_aux2Z(md_ptr);
 } else {
 return NULL;
 }
 
 */
MD::iterator::iterator(const bam1_t* BAM_record) {
	_init();
	bam_record = BAM_record;
	uint8_t* tmp_ptr = bam_aux_get(bam_record, "MD");
	if (tmp_ptr) {
		md_ptr = bam_aux2Z(tmp_ptr);
	} else {
		md_ptr = NULL;
	}

	//md_len = bam_record->core.l_qseq; //this should be a save assumption
	
}

void MD::iterator::_init() {
	bam_record = NULL;
	position = 0;
	
}

void MD::iterator::next() {
	position++;	//next SHOULD handle this, and in order to do so
	//I am pre-incrementing position before returning the character
}

char MD::iterator::get() {
	//std::cerr << "md_ptr: " << int(md_ptr[position]) << std::endl;
	return md_ptr[position];
}

bool MD::iterator::good() const {
	
	if (md_ptr == NULL) {
		return false;
	}
	if (md_ptr[position] != 0) {
		return true;
	}
	else {
		return false;
	}
	
}

