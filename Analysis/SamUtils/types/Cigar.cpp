/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */


/*
 *  Cigar.cpp
 *  SamUtils
 *
 *  Created by Michael Lyons on 12/23/10.
 *  Copyright 2010 Life Technologies. All rights reserved.
 *
 */



#include <iostream>
#include <sstream>
#include "Cigar.h"
#include "sam.h"




Cigar::Cigar() {
	_init();
}


Cigar::Cigar(const bam1_t* BAM_record) {
	_init();
	bam_record = BAM_record;
	cigar_len =  bam_record->core.n_cigar;


}
	
	
void Cigar::_init() {
	cigar_len = -1;
	bam_record = NULL;
}
	
	
		
		//is iterator good?
bool Cigar::iterator::good() const {
			
			if (bam_record == NULL) {
				return false;
			}
			if (pos <= ((bam_record->core.n_cigar)) ) {
				
				return true;	
			} else {
				return false;
			}
			
}
		
//get next cigar element
void Cigar::iterator::next() {
	_set_len_and_op();
	pos++;	
	cigar_ptr++;
	
}
int Cigar::iterator::len() {
	
	return length;
	
}
//operation
char Cigar::iterator::op() const {
	char ret;
	switch (operation) {
		case BAM_CMATCH:
			ret = 'M';
			break;
		case BAM_CINS:
			ret =  'I';	
			break;
		case BAM_CDEL:
			ret =  'D';
			break;
		case BAM_CREF_SKIP:
			ret =  'N';
			break;
		case BAM_CSOFT_CLIP:
			ret =  'S';
			break;
		case BAM_CPAD:
			ret =  'P';
			break;
		case BAM_CHARD_CLIP:
			ret =  'H';
			break;
		default:
			ret =  '*';
			break;
	}
	
	return ret;
	
}


//length of operation



		
Cigar::iterator::iterator() {
			_init();
}
		
Cigar::iterator::iterator(const bam1_t* BAM_record) {
			_init();
			bam_record = BAM_record;
			cigar_ptr = bam1_cigar(BAM_record);
			next();
}
		
void Cigar::iterator::_init() {
			bam_record = NULL;
			cigar_ptr = NULL;
			length = 0;
			operation = 0;
			pos = 0;
}
		
		//private functions
void Cigar::iterator::_set_len_and_op() {
			uint32_t cigar_i = *cigar_ptr;
			length = cigar_i >> BAM_CIGAR_SHIFT;
			operation = cigar_i & BAM_CIGAR_MASK;
}
		
				
	
	
	
	
Cigar::iterator Cigar::get_iterator() {
		if (bam_record != NULL) {
			return iterator(bam_record);
		} else {
			return iterator();
		}
}
	
	
std::string Cigar::to_string() {
		std::ostringstream strm;
		for (Cigar::iterator i = get_iterator(); i.good(); i.next()) {
			strm << i.len();
			strm << i.op();
	//		std::cout << i.len() << i.op();
			
		}
		
	//std::cout << std::endl;
		//std::cout << strm.str();
	if (strm.str().length() < 1) {
		strm << "*";
	}
	
	return strm.str();
	//std::cout << "blah.length(): " << blah.length() << std::endl;
		
}
	




