/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 *  BAMReader.cpp
 *  SamUtils
 *
 *  Created by Michael Lyons on 12/9/10.
 *  
 *
 */

#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
#include "Utils.h"
#include "BAMReader.h"

using namespace std;


/* constructors */



BAMReader::BAMReader() {
	Initialize();
	
	
}
BAMReader::BAMReader(const std::string& BAM_file) {
	Initialize();
	bam_file = BAM_file;
}
BAMReader::BAMReader(const std::string& BAM_file, const std::string& Index_File) {
	Initialize();
	bam_file = BAM_file;
	index_file = Index_File;
	
	
}


/* initialization code */

void BAMReader::Initialize() {
	file_p  = NULL;
	bam_index = NULL;
	file_open = false;
	bam_header = bam_header_init();
	
}

// Destructor closes all open files:

BAMReader::~BAMReader() {
	
	this->close();
	
}
 



/* functionality */
void BAMReader::open() {
	_open();
}

// Open a bam file or bam/index pair:
void BAMReader::open(const std::string& BAM_file) {
	//might as well grab it here
	bam_file = BAM_file;
	_open();
	
	
}


void BAMReader::open(const std::string& mode_stdin, const std::string& type_of_stdin) {
		
	file_p = samopen(mode_stdin.c_str(), type_of_stdin.c_str(), 0);
	if (file_p == NULL) {
		std::cout << "[BAMReader] file:"<< bam_file << " doesn't appear to exist" << std::endl;
		file_open = false;
	} else {
		if (file_p->header) {
			std::cerr << "[BAMReader] file opened" << std::endl;
			_init();

		} else {
			std::cerr << "[BAMReader] invalid input." << std::endl;
		}

		
	}
	
}


void BAMReader::_open() {
	//std::cerr << "index_file.capacity(): " << index_file.capacity() << std::endl; 
	string extension = get_file_extension(bam_file);
	//make lower case
	std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
	if (extension == "bam") { //binary sam file (bam)
		if (index_file.capacity() > 0) {
			file_p = samopen(bam_file.c_str(), "rb", 0);
			
		} else {
			file_p = samopen(bam_file.c_str(), "rb", 0);
			
		}
		
	} else { //must be a sam file
		if (index_file.capacity() > 0) {
			file_p = samopen(bam_file.c_str(), "r", 0);
		
		} else {
			file_p = samopen(bam_file.c_str(), "r", 0);
		
		}
	}
	
	//error handling
	if (file_p == NULL) {
		std::cout << "[BAMReader] file:"<< bam_file << " doesn't appear to exist" << std::endl;
		file_open = false;
	} else {
                std::cerr << "[BAMReader] file opened" << std::endl;
		_init();
	}

	
}


void BAMReader::_init() {
	
	file_open = true;
	bam_header = file_p->header;

	for (int n_target = 0; n_target < bam_header->n_targets; n_target++) {
		std::string isrt(bam_header->target_name[n_target]);
		ref_list.push_back(isrt);
		ref_lens.push_back(bam_header->target_len[n_target]);
	}

}


void BAMReader::close() {
	
	if (is_open()) {
		samclose(file_p);
		file_open = false;
	} 
	
}


// Was the open successful?
bool BAMReader::is_open()   const {
	return file_open;
}


BAMReader::operator bool() const {
	return is_open();
}


//return number of references
int BAMReader::num_refs() const {
	if (bam_header != NULL) {
		return bam_header->n_targets;
	} else {
		return -1;
	}

}



//return names strvec(string vector) of names
const strvec& BAMReader::refs() const {	
	return ref_list;
	
}

//return lenvec(vector of integers) of lengths of the references
const lenvec& BAMReader::lens() const {
	return ref_lens;
	
}




/**** iterator implementations *****/

	/* iterator gets */
BAMReader::iterator BAMReader::get_iterator() {
	return BAMReader::iterator(file_p, ref_lens);
	
}


BAMReader::iterator BAMReader::get_iterator(int ref_index) {
	
	return BAMReader::iterator(file_p, ref_index, bam_file, ref_lens);
	
}

BAMReader::iterator BAMReader::get_iterator(const std::string& ref_seq_name) {
		//couldn't find ref if ref_index == -1
	int ref_index = _findRef(ref_seq_name);
	if (ref_index > -1) {
		return BAMReader::iterator(file_p, ref_index, bam_file, ref_lens);

	} else {
		//possibly just want to exit here
		return BAMReader::iterator(); //basically a null iterator.  is_good == false, so you can't do any damage here
	}
	
}




BAMReader::iterator BAMReader::get_iterator(int ref_index, coord_t begin, coord_t end) {
	return BAMReader::iterator(file_p, ref_index, bam_file, begin, end, ref_lens);
	
}

BAMReader::iterator BAMReader::get_iterator(const std::string& ref_seq_name, coord_t begin, coord_t end) {
	
	int ref_index = _findRef(ref_seq_name);
	if (ref_index > -1) {
		return BAMReader::iterator(file_p, ref_index, bam_file, begin,end, ref_lens);
	} else {
		return BAMReader::iterator(); //basically a null iterator.  is_good == false, so you can't do any damage here
	}

}


//gets a ref_index 
int BAMReader::_findRef(const std::string& ref_seq_name) const {
		
	int ref_index = -1;
	
	int num_refs = file_p->header->n_targets;
	for (int i = 0; i <= num_refs; i++) {
		std::string ref(file_p->header->target_name[i]);
		
		if (ref == ref_seq_name) {
			ref_index = i;
			return ref_index;
		}
	}
	return ref_index;
	
}

	/*iterator ctors/dtors and initialization*/

BAMReader::iterator::iterator() {
	_init();
}

BAMReader::iterator::iterator(samfile_t* fp, lenvec& ref_lens) {
	_init();
	itr_file = fp;
	tid_lens = ref_lens;
	//std::cerr << "[BAMReader::iterator] itr_file->type: " << itr_file->type << std::endl;
	if (itr_file->type == 2) {
		is_bam = false;
	} else if (itr_file->type&1) {
		is_bam = true;
	} else {
		is_bam = false;
		is_good = false;
	}

	//cerr << "constructed:  BAMReader::iterator::iterator(samfile_t* fp)" << endl;
	//need to prime iteration to get first record ready:
	next();
}

BAMReader::iterator::iterator(samfile_t* fp, int ref_index, const std::string& BAM_file, lenvec& ref_lens ) {
	_init(fp, ref_index, BAM_file);
	tid_lens = ref_lens;
}

BAMReader::iterator::iterator(samfile_t* fp, int ref_index, const std::string& BAM_file, coord_t begin, coord_t end, lenvec& ref_lens) {
	_init(fp, ref_index, BAM_file);
	assert(itr_file);
	if (itr_file->type & 1) {
		is_bam = true;
	}
	
	tid_lens = ref_lens;
	is_good = _set_iter(ref_index, begin, end); 
	if (is_good) {
		cerr << "[BAMReader] iter set: ref[ "<< itr_file->header->target_name[ref_index] << " ] start: "<< begin << " end: "<< end << endl;
	} else {
		cerr << "[BAMReader] iter not set" << endl;
	}
}


void BAMReader::iterator::_init() {
	itr_file = NULL;
	itr_record = bam_init1();
	is_good = false;
	bam_iter = NULL; // 12-14 maybe set this to 0
}


void BAMReader::iterator::_init(samfile_t* fp, int ref_index, const std::string& BAM_file) {
	_init();
	itr_file = fp;
	itr_index = bam_index_load(BAM_file.c_str());
	if (itr_index != NULL) {
		//index loaded
		cerr << "[BAMReader] setting iter to ref_index: " << ref_index <<  ": " 
		<< itr_file->header->target_name[ref_index] << endl;
		int beg = 0;
		
		int end =itr_file->header->target_len[ref_index];
		is_good = _set_iter(ref_index, beg, end); 
		if (is_good) {
			cerr << "[BAMReader] iter set" << endl;
		} else {
			cerr << "[BAMReader] iter not set" << endl;
		}
	} else {
		std::cerr << "[BAMReader] bam not indexed.  indexing..." << std::endl;
		_index(BAM_file);
		itr_index = bam_index_load(BAM_file.c_str());
		if (itr_index == NULL) { 
			std::cerr << "[BAMReader] something serioulsy wrong with indexing: " 
			<< BAM_file << "\nexiting..." << std::endl;
			
		}
	}
	
	
}

bool BAMReader::iterator::_set_iter(int tid, int beg, int end) {
		
	if (itr_index == NULL) {
		cerr << "[BAMReader] itr_index == null" << endl;
	}
	
	cerr << "[BAMReader] bam_iter_query(itr_index, "<< tid << "," << beg << "," << end << ");" << endl;;
	bam_iter = bam_iter_query(itr_index, tid, beg, end);
	if (bam_iter == NULL) {
		cerr << "[BAMReader] bam_iter == NULL" << endl;
	}
	//prime the iterator to first position
	int bytes = bam_iter_read(itr_file->x.bam, bam_iter, itr_record);
	if (bytes < 0) {
		
		return false;
	} else if (bytes >= 0){
		return true;
	} else {
		return false;
	}

}

void BAMReader::iterator::_index(const std::string& BAM_file) {
	
	bam_index_build(BAM_file.c_str());
	//documentation says status is always 0 for now. 
}

/*
BAMReader::iterator::~iterator() {
	
	bam_iter_destroy(bam_iter);
}
 */

	/*	iteration	*/



void BAMReader::iterator::next() {
	//if (bam_iter != NULL) {
	if (is_bam) { //bamfile
		
			int bytes = bam_iter_read(itr_file->x.bam, bam_iter, itr_record);
		//std::cerr << "next() bytes: " << bytes << std::endl;
			
			if (bytes == -1) {
				is_good = false;
			}
			else {
				is_good = true;
			}

		
		
	} //end bam_iter != NULL
	else if (itr_file != NULL) { //must be sam or non-regional iteration

		int bytes = samread(itr_file, itr_record);
		//cerr << "bytes read: " << bytes << endl;
		if (bytes == -1) {
			is_good = false;
		} 
		else {
			is_good = true;
		}

	}
	
	else {
		is_good = false;
		//return false;
	}
		
		
}
	





