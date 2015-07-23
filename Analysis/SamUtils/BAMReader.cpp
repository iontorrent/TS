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
#include <exception>
#include "Utils.h"
#include "BAMReader.h"
#include "sam_header.h"

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
  bam_header = NULL;
  //bam_header = bam_header_init();
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
		std::cerr << "[BAMReader] file:"<< bam_file << " doesn't appear to exist" << std::endl;
		file_open = false;
	} else {
		if (file_p->header) {
			std::cout << "[BAMReader] file opened" << std::endl;
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
		std::cerr << "[BAMReader] file:"<< bam_file << " doesn't appear to exist" << std::endl;
		file_open = false;
	} else {
                std::cout << "[BAMReader] file opened" << std::endl;
		_init();
	}

	
}

void BAMReader::_init() {
	
	file_open = true;
	bam_header = file_p->header;
        bam_header_object.init( bam_header );

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
	if (itr_file->file->is_bin) {
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
	if (itr_file->file->is_bin) {
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

BAMReader::iterator::~iterator() {
    if(NULL != itr_record) {
        bam_destroy1(itr_record);
        itr_record = NULL;
    }
}
	
/* BAMHeader implementations */

//ctor
BAMReader::BAMHeader::BAMHeader() {
  //setup default read group
  //maybe?
}



void BAMReader::BAMHeader::init(bam_header_t* header) {
  
  header_ptr = header;
  
  //not sure what this one does, Heng Li's code:
  //if (header_ptr->l_text >= 3) {
  //  if (header_ptr->sdict == 0)
  //    header_ptr->sdict = sam_header_parse2( header_ptr->text );
  //}
  int num_entries = 0;
  char type[2] = { 'R', 'G' };
  char key_tag[2] = { 'I', 'D' };
  char **rg_list = NULL;
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  if ( rg_list == NULL ) { 
    /*
     I hate doing this.  If there's a better way to check 
     that the ID is or isn't there, I'd like to know
     */
    cerr << "[BAMReader::BAMHeader] RG is missing required ID field.  Ignoring read groups." << endl;
      cerr << "lawlllll" << endl;
    read_groups.push_back(BAMReader::BAMHeader::ReadGroup());
    return;
  } else {
    for( int j = 0; j < num_entries; j++ ) {
        read_groups.push_back(BAMReader::BAMHeader::ReadGroup());
    }
  }
   
  //get flow order
  key_tag[0] = 'F'; key_tag[1] = 'O';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {
    try {
      read_groups[j].flow_order = rg_list[j];
      read_groups[j].num_flows = read_groups[j].flow_order.length();
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }  
  //get key sequence
  key_tag[0] = 'K'; key_tag[1] = 'S';
  num_entries = 0;
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {   
    try {
      read_groups[j].key_sequence = rg_list[j];
      read_groups[j].key_length = read_groups[j].key_sequence.length();
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }
  //get ID
  key_tag[0] = 'I'; key_tag[1] = 'D';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {    
    try {
      read_groups[j].group_id = rg_list[j];
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }
  //get CN
  key_tag[0] = 'C'; key_tag[1] = 'N';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {    
    try {
      read_groups[j].sequencing_center = rg_list[j];
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }
  //get DS
  key_tag[0] = 'D'; key_tag[1] = 'S';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {    
    try {
      read_groups[j].description = rg_list[j];
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }
  //get DT
  key_tag[0] = 'D'; key_tag[1] = 'T';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {    
    try {
      read_groups[j].date_time = rg_list[j];
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }
  //get LB
  key_tag[0] = 'L'; key_tag[1] = 'B';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {   
    try {
      read_groups[j].library = rg_list[j];
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }
  //get PG
  key_tag[0] = 'P'; key_tag[1] = 'G';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) { 
    try {
      read_groups[j].program = rg_list[j];
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }
  //get PI
  key_tag[0] = 'P'; key_tag[1] = 'I';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {    
    try { 
      read_groups[j].predicted_median_insert_size = strtol(rg_list[j], NULL, 10);
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }
  //get PL
  key_tag[0] = 'P'; key_tag[1] = 'L';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {    
    try {
      read_groups[j].sequencing_platform = rg_list[j];
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }
  //get PU
  key_tag[0] = 'P'; key_tag[1] = 'U';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {    
    try {
      read_groups[j].platform_unit = rg_list[j];
    } catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }
  //get SM
  key_tag[0] = 'S'; key_tag[1] = 'M';
  rg_list = parse_read_group( rg_list, type, key_tag, &num_entries );
  for (int j = 0; j < num_entries; j++ ) {    
    try {
      read_groups[j].sample_name = rg_list[j];
    }catch ( std::exception& e) {
      //defaults are in place for the ReadGroup
    }
  }

  free(rg_list); 
}

char** BAMReader::BAMHeader::parse_read_group( char** rg_list, char type[2], char key_tag[2], int* num_entries ) {
  if (rg_list != NULL)
    free(rg_list);
  rg_list = sam_header2list( header_ptr->sdict, type, key_tag, num_entries );
  return rg_list;
}

BAMReader::BAMHeader::ReadGroup& BAMReader::BAMHeader::get_read_group(unsigned int id) {
  if (read_groups.size() >= id) {
     return read_groups[ id ];
  } else {
    //FIXME:  throw an exception instead
    return default_group;
  }
}

BAMReader::BAMHeader::ReadGroup::ReadGroup()
:
group_id("nothing meaningful"),
sequencing_center("Timbuktu" ),
flow_order("N"),
key_sequence("N"),
description("Default ReadGroup"),
date_time("1/1/12"),
library("not a library"),
num_flows(-1),
key_length(-1)
{
  
}

std::string BAMReader::BAMHeader::ReadGroup::to_string() {
  
  std::ostringstream strm;
  strm << get_read_group_id() <<" "<< get_sequencing_center() <<" "<< get_description() <<" "<< get_date_time() 
  <<" "<< get_flow_order() <<" "<< get_key_sequence() <<" "<< get_library() <<" "<< get_program() 
  <<" "<< get_predicted_median_insert_size() <<" "<< get_sequencing_platform() <<" "<< get_platform_unit()
  <<" "<< get_sample_name();
  
  return strm.str();
    
  

}
