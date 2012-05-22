/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 *  BAMRead.cpp
 *  SamUtils
 *
 *  Created by Michael Lyons on 12/21/10.
 *  *
 */

#include <iostream>
#include <sstream>
#include "BAMRead.h"


using namespace samutils_types;

/*TODO:
 1:  be able to get # of flows
 2:  get flow order
 3:  return flow signals
 */

std::vector<int>  BAMRead::get_fz(int num_flows) {
  //return flow signals from this read
  std::vector<int> flow_sigs(num_flows);
  
  uint8_t* flow_ptr = bam_aux_get(bam_record.get(), "FZ" );
  // for number of flows, set the corresponding flow in the vector 
  // set each flow value to the vector
  for( int i = 0; i < num_flows; i++ ) {
    flow_sigs[i] = *(flow_ptr + i);
  }
  
  return flow_sigs;
}



Cigar BAMRead::get_cigar() {
	
	return Cigar(bam_record.get());
	
}


str_ptr  BAMRead::get_rnext()  {
	
	assert(bam_record.get());
	
	int mtid = bam_record->core.mtid;
	str_ptr rnext;
	if (mtid >= 0) {
		
		
		rnext = bam_header->target_name[mtid];
		
		
	} else {
		rnext = NULL;
	}

		

	return rnext;
	
}

Sequence BAMRead::get_seq() {
	
	return Sequence(bam_record.get());
}


Qual BAMRead::get_qual() {
		
	return Qual(bam_record.get());
	
}


/*** optional tags ***
 -have to implement MD at very least
 */








std::string BAMRead::to_string() {
	return to_string_1_2();
}







std::string	BAMRead::to_string_1_2() {
	
	//qname		flag	rname	pos		mapq	cigar	pnext	rnext	tlen	seq		qual
	//*qname() + "\t" + *flag() + "\t" + *rname() + "\t" + *pos() + "\t" + *cigar() + "\t" + *pnext() + "\t" + rnext() + "\t" + tlen() + "\t" + seq() + "\t" + qual();
	
	
	std::ostringstream strm;
	
	
	
	strm << std::string(get_qname()) << "\t";
	
	
	
	strm << get_flag() << "\t";
	 if (get_rname() != NULL) {
	 	 strm << std::string(get_rname()) << "\t";
		 //std::cerr << get_rname() << std::endl
	 } else {
			strm << "*" << "\t";
	}
	 strm << get_pos() << "\t";  //should be 0 if not there
	strm << get_mapq() << "\t";
	strm << get_cigar().to_string() << "\t"; //if null should be "*"
	if (get_rnext() != NULL) {
		strm << get_rnext() << "\t"; //MRNM

	} else {
		strm << "*" << "\t";
	}

	strm << get_pnext() << "\t"; //MPOS
	strm << get_tlen() << "\t"; //ISIZE
	strm << get_seq().to_string() << "\t";
	strm << get_qual().to_string() << "\t"; //if fail should be "*"
	
	//optional fields
	if (get_rg() != NULL) {
		strm << "RG:Z:" << std::string(get_rg()) << "\t";
	}
	
	if (get_pg() != NULL) {
		strm << "PG:Z:" << std::string(get_pg()) << "\t";
	}
	if (get_as() != -1) {
		strm << "AS:i:";
		coord_t score = get_as();
		if (score == -2147483648 ) {
			strm << "-" << "\t";
		} else {
			strm << score << "\t";
		}

		
	}
	
	if (get_nm() != -1) {
		strm << "NM:i:";
		strm << get_nm() << "\t";
	}
	
	if (get_nh() != -1) {
		strm << "NH:i:";
		strm << get_nh() << "\t";
		
	}
	if (get_ih() != -1) {
		strm << "IH:i:";
		strm << get_ih() << "\t";
	}
	
	if (get_hi() != -1) {
		strm << "HI:i:";
		strm << get_hi() << "\t";
	}
	
	if (get_md().to_string().capacity() > 0) {
		strm << "MD:Z:";
		strm << get_md().to_string() << "\t";
	}
	
	
	if (get_xa() != NULL) {
		strm << "XA:Z:";
		strm << std::string(get_xa()) << "\t";
	}
	
	if (get_xs() != -1) { 
		strm << "XS:i:" << get_xs() << "\t";
	}
	if (get_xt() != -1) { 
		strm << "XT:i:" << get_xt();
	}
	
	
	
	/*
	 if (rg().get() != NULL) 
	 strm << *(rg()) << "\t";
	 */
	strm << "\n";
	
	return strm.str();
	
}


