/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 *  Qual.cpp
 *  SamUtils
 *
 *  Created by Michael Lyons on 1/4/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */

#include <sstream>
#include <iostream>

#include "Qual.h"



Qual::Qual() {
	_init();

}


Qual::Qual(const bam1_t* BAM_record) {

	bam_record = BAM_record;
	qual_length = bam_record->core.l_qseq;

}


void Qual::_init() {
	qual_length = -1;
	bam_record = NULL;

}

long Qual::get_length()	const {
	return qual_length;

}

Qual::iterator Qual::get_iterator() {
	
	return iterator(bam_record);
	
}

std::string	Qual::to_string() {
	
	std::ostringstream strm;
	for (iterator i = get_iterator(); i.good(); i.next()) {
		strm << i.get();
		//std::cerr << i.get();
	}
	//std::cerr << std::endl;
	return strm.str();
}

//iterator functionality
Qual::iterator::iterator(const bam1_t* BAM_record) {
	_init();
	bam_record = BAM_record;
	qual_len = bam_record->core.l_qseq;
	bam_qual = bam1_qual(bam_record);


}

void Qual::iterator::_init() {
	bam_record = NULL;
	qual_len = -1;
	position = 0;
	bam_qual = NULL;
	

}


bool Qual::iterator::good()	const{
	if (position < qual_len) {
		return true;
	} else {
		return false;
	}


}


void Qual::iterator::next()	{
	position++;

}


char Qual::iterator::get() {

	//std::cerr << (char)(bam_qual[position]+33);
	return (char)(bam_qual[position]+33);

}

