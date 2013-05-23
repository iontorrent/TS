/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/*score_alignmentst_dna
 *  BAMUtils.cpp
 *  SamUtils
 *
 *  Created by Michael Lyons on 1/6/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */
#include "BAMUtils.h"
#include "Utils.h"
#include <math.h>
#include <locale>
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>

//constructors

BAMUtils::BAMUtils() {
	_init();
	total_three_prime_ignore = 0;
}
 
BAMUtils::BAMUtils(BAMRead& BAM_record) { 
	_init();
	total_three_prime_ignore = 0;

	bam_record = BAM_record;
	q_scores = std::string("7,10,17,20,47");

	_crunch_data();

}


BAMUtils::BAMUtils(BAMRead& BAM_record, std::string qscores, coord_t slop)
	: bam_record(BAM_record), q_scores(qscores), num_slop(slop) {
	_init();
	total_three_prime_ignore = 0;
	set_genome_length(bam_record.get_ref_len());
	_crunch_data();
}


BAMUtils::BAMUtils(BAMRead& BAM_record, std::string qscores, coord_t slop, 
                bool IUPAC_Flag, bool Keep_IUPAC, bool Truncate_Soft_Clipped, 
                int error_table_min_len, int error_table_max_len, 
                int error_table_step_size, int three_prime_clip, 
                bool rounding_phred_scores, bool Five_Prime_Justify, std::string &Flow_Order):
          bam_record(BAM_record), 
          q_scores(qscores), 
          num_slop(slop), 
          total_three_prime_ignore(three_prime_clip)
{
	_init();
	
	flow_order = Flow_Order;
	
	iupac_flag = IUPAC_Flag;
	keep_iupac = Keep_IUPAC;
	truncate_soft_clipped = Truncate_Soft_Clipped;
	//set_genome_length(genome_length);	
	set_genome_length(bam_record.get_ref_len());
	set_slop_bases(slop);
	init_error_lens(error_table_min_len, error_table_max_len, error_table_step_size);
	round_phred_scores = rounding_phred_scores;
    five_prime_justify = Five_Prime_Justify;
	_crunch_data();

}


BAMUtils::BAMUtils(BAMRead* BAM_Record_Ptr, std::string qscores, coord_t slop) {
	_init();
	
	
	bam_rec_ptr = BAM_Record_Ptr;
	bam_record = *bam_rec_ptr;
	q_scores = qscores;
	
	set_slop_bases(slop);
	_crunch_data();
	
	
	
}



BAMUtils::~BAMUtils() {
	if (bam_rec_ptr != NULL) {
		//delete(bam_rec_ptr);
	}
}


void BAMUtils::init_error_lens(int min_len, int max_len, int step_size) {
	for (int i = min_len; i <= max_len; i+=step_size) {
		error_lens.push_back(i);
	}
}


//init
void BAMUtils::_init() {
	t_diff			= -1;
	t_len			= -1;
	num_slop		= 0;
	max_slop		=0;
	n_qlen			= -1;
	match_base		= -1;
	q_err			= -1;
	mm_err			= -1;
	homo_err			= -1;
	indelErr		= -1;
	genome_len	= -1;
	soft_clipped_bases = 0;
	adjusted_three_prime_trim_len = 0;
	bam_rec_ptr = NULL;
	iupac_flag = true;
	keep_iupac = true;
	truncate_soft_clipped = true;
	is_three_prime_soft_clipped = false;
    round_phred_scores = true;
    five_prime_justify = false;
    _region_homo_err = 0;
    _region_mm_err = 0;
    _region_indel_err = 0;
    _region_ins_err = 0;
    _region_del_err = 0;
    _region_clipped = false;
	flow_order = "";
	max_aligned_flow = -1;
}


//does everything to fill in all the data
void BAMUtils::_crunch_data() {
	
	dna();
	
	padded_alignment();
	
	if (get_bamread().mapped_reverse_strand()) {
		reverse();
	}
        
    //shift indels to 5'
    if (five_prime_justify)
        left_justify();
	//cip soft clipped portions of read
	if (total_three_prime_ignore > 0)
		remove_bases_before_soft_clipped();
	
	int start_base = 0;
	int stop_base = 0;
	bool calcRegionError = bam_record.get_region_base_positions(&start_base, &stop_base);
	
	_region_clipped = false;
		
    if(calcRegionError){
    	adjust_region_base_positions(&start_base, &stop_base, pad_target, &_region_clipped);
    	save_region_error_positions(pad_match, _region_error_positions, start_base, stop_base);
    }
	
	score_alignments(q_scores, calcRegionError, start_base, stop_base);
	
	calc_error();
	
	
}

void BAMUtils::remove_bases_before_soft_clipped() {


	if (total_three_prime_ignore > 0 ) {

		if (static_cast<unsigned int>(total_three_prime_ignore) < pad_match.length() ) {

			adjusted_three_prime_trim_len = 0;
			int num_matches = 0;
			for (int j = (pad_match.length() - 1); ( (j >= 0) && (num_matches < total_three_prime_ignore) ); j--) {
				//if (pad_match[j] != '|') {
				if (pad_target[j] == '-') {
					adjusted_three_prime_trim_len++;
				} else {
					num_matches++;
				}

			}
			//cerr << "adjusted trim len: " << adjusted_three_prime_trim_len << endl;
			adjusted_three_prime_trim_len += num_matches;
			pad_match.erase(pad_match.length() - adjusted_three_prime_trim_len);
			pad_source.erase(pad_source.length() - adjusted_three_prime_trim_len);
			pad_target.erase(pad_target.length() - adjusted_three_prime_trim_len);
		}
	}

}


std::string BAMUtils::to_string() {
	std::ostringstream strm(std::ostringstream::app);
	strm << get_name(); //0
	strm << "\t";
	strm << get_strand(); //1

	strm << "\t";
	strm << get_t_start(); //2

	strm << "\t";
	strm << get_t_length(); //3

	strm << "\t";
	strm << get_q_length();//4

	strm << "\t";
	strm << get_match();//5
	strm << "\t";
	strm << get_percent_id();//6
	strm << "\t";
	strm << get_q_error();//7
	strm << "\t";

	strm << get_homo_errors();//8
	strm << "\t";
	strm << get_mismatch_errors();//9
	strm << "\t";
	strm << get_indel_errors();//10
	strm << "\t";

	strm << get_qdna();//12
	strm << "\t";
	strm << get_matcha();//13
	strm << "\t";
	strm << get_tdna();//14
	strm << "\t";
	strm << get_rname();//15
	strm << "\t";

	if (num_slop >= 0) {
		strm << num_slop;
		
	}else {
		strm << 0;//16
	}
	strm << "\t";
	
	std::vector<std::string> q_score_vec;
	split(q_scores, ',', q_score_vec);
	for (std::vector<std::string>::size_type i = 0; i < q_score_vec.size(); i++) {
		//strm << phred_lens[i];
		strm << get_phred_len((strtol(q_score_vec[i].c_str(),NULL, 10)));
		strm << "\t";
	}

	strm << get_full_q_length();
	return strm.str();
	
}


void BAMUtils::dna() {
	
	
	
	MD md = bam_record.get_md();
	Cigar cig = bam_record.get_cigar();
	Sequence qseq = bam_record.get_seq();
	


	int position = 0;
	std::string seq;
	Sequence::iterator qseq_itr = qseq.get_iterator();
	for (Cigar::iterator i = cig.get_iterator(); i.good(); i.next()) {
		
		
		if (i.op() == 'M') {
			int count = 0;
			while (qseq_itr.good()) {
				
				if (count >= i.len()) {
					break;
				} else {
					seq += qseq_itr.get();
					qseq_itr.next();
					count++;

				}
			}
			

		} else if ((i.op() == 'I') || (i.op() == 'S')) {
			int count = 0;
			while (qseq_itr.good()) {
				if (count >= i.len()) {
					break;
				}				
				qseq_itr.next();
				count++;
				
			}
			//bool is_error = false;

			if (i.op() == 'S') {
				soft_clipped_bases += i.len();
				//is_error = true;

			}

			
		} 
		position++;
	}
	
	
	t_dna.reserve(seq.length());
	int start = 0;
	MD::iterator md_itr = md.get_iterator();
	std::string num;
	coord_t md_len = 0;
	char cur;

	while (md_itr.good()) {
		cur = md_itr.get();
		
		if (std::isdigit(cur)) {
			num+=cur;
			//md_itr.next();
		}
		else {
			if (num.length() > 0) {
				md_len = convert(num);
				num.clear();
			
				t_dna += seq.substr(start, md_len);
				start += md_len;
				
			}
			
		}
				
		if (cur == '^') {
			//get nuc
			md_itr.next();
			char nuc = md_itr.get();
			while (std::isalpha(nuc)) {
				t_dna += nuc;
				md_itr.next();
				nuc = md_itr.get();
			}
			num += nuc; //it's a number now will
						//lose this value if i don't do it here
			//cur = nuc;				
			
		} else if (std::isalpha(cur)) {
			t_dna += cur;
			start++;

		}
		md_itr.next();
		

	}
	//clean up residual num if there is any
	if (num.length() > 0) {
		md_len = convert(num);
		num.clear();
		t_dna += seq.substr(start, md_len);
		start += md_len;
	}
	

	
}

//helper function to condense some code, converts a string to a coord_t
//assumes input string is only numeric.
coord_t BAMUtils::convert(std::string num) {
	
	return strtol(num.c_str(),NULL, 10);
	
	
}

int BAMUtils::left_justify() {
    char c;
    int prev_del = 0;
    int prev_ins = 0; 
    int start_ins = 0;
    int start_del = 0;
    int end_ins = 0;
    int end_del = 0;
    int justified = 0;
    unsigned int i;

    for (i = 0; i < pad_match.length(); i++) {
        if('-' == pad_target[i]) { // deletion
            if(0 == prev_del) {
                start_del = i;
            }
            prev_del = 1;
            end_del = i;
            prev_ins = 0;
            start_ins = end_ins = -1;
            i++;
        }
        else if('-' == pad_source[i]) { // insert
            if(0 == prev_ins) {
                start_ins = i;
            }
            prev_ins = 1;
            end_ins = i;
            prev_del = 0;
            start_del = -1;
            end_del = -1;
            i++;
        }
        else {
            if(1 == prev_del) { // previous was an deletion
                start_del--;
                while(0 <= start_del && // bases remaining to examine 
                      pad_target[start_del] != '-' && // hit another deletion 
                      pad_source[start_del] != '-' && // hit an insertion 
                      pad_source[start_del] == pad_source[end_del]) { // src pad_target base matches pad_target base 
                    // swap end_del and start_del for the target and pad_match
                    c = pad_target[end_del]; pad_target[end_del] = pad_target[start_del]; pad_target[start_del] = c;
                    c = pad_match[end_del]; pad_match[end_del] = pad_match[start_del]; pad_match[start_del] = c;
                    start_del--;
                    end_del--;
                    justified = 1;
                }
                end_del++; // we decremented when we exited the loop 
                i = end_del;
            }
            else if(1 == prev_ins) { // previous was an insertion
                start_ins--;
                while(0 <= start_ins && // bases remaining to examine
                      pad_target[start_ins] != '-' && // hit another deletion 
                      pad_source[start_ins] != '-' && // hit an insertion 
                      pad_target[start_ins] == pad_target[end_ins]) { // src target base matches dest target base 
                    // swap end_ins and start_ins for the pad_target and pad_match
                    c = pad_source[end_ins]; pad_source[end_ins] = pad_source[start_ins]; pad_source[start_ins] = c;
                    c = pad_match[end_ins]; pad_match[end_ins] = pad_match[start_ins]; pad_match[start_ins] = c;
                    start_ins--;
                    end_ins--;
                    justified = 1;
                }
                end_ins++; // we decremented when we exited the loop 
                i = end_ins;                
            }
            else {
                //i++;
            }
            // reset
            prev_del = prev_ins = 0;
            start_del = start_ins = end_del = end_ins = -1;
        }
    }     
    return justified;
}

void BAMUtils::padded_alignment() {
	Cigar cig = bam_record.get_cigar();
	Sequence tdna = bam_record.get_seq();

	int sdna_pos = 0;
	int tdna_pos = 0;
	pad_source.reserve(t_dna.length());
	pad_target.reserve(t_dna.length());
	pad_match.reserve(t_dna.length());
	Sequence::iterator tdna_itr = tdna.get_iterator();
	int tot = 0;
	//find out if the first cigar op could be soft clipped or not
	is_three_prime_soft_clipped = false;


	for (Cigar::iterator i = cig.get_iterator(); i.good(); i.next()) {
		//i.op();		i.len();
		if (this->bam_record.mapped_reverse_strand()) {
			if (tot > ( cig.get_length( ) - 3) ){
				if (i.op() == 'S')
					is_three_prime_soft_clipped = true;
				else
					is_three_prime_soft_clipped = false;

			}
		} else {
			if (tot < 2) {
				if (i.op() == 'S')
					is_three_prime_soft_clipped = true;
				else
					is_three_prime_soft_clipped = false;

			}
		}

		if (i.op() == 'I' ) {
			pad_source.append(i.len(), '-');
					
			int count = 0;
			tdna_itr.set_position(tdna_pos);
			
			while (tdna_itr.good()) {
				if (count >= i.len()) {
					break;
				} else {
					pad_target += tdna_itr.get();
					tdna_itr.next();
					
					tdna_pos++;
					count++;
				}
				

			}
			pad_match.append(i.len(), '+');
		}
		else if(i.op() == 'D' || i.op() == 'N') {
			pad_source.append( t_dna.substr(sdna_pos, i.len()));
			sdna_pos += i.len();
			pad_target.append(i.len(), '-');
			pad_match.append(i.len(), '-');
			
			
		}
		else if(i.op() == 'P') {
			pad_source.append(i.len(), '*');

			pad_target.append(i.len(), '*');
			pad_match.append(i.len(), ' ');
			
			
			
			
		} else if (i.op() == 'S') {

			if (!truncate_soft_clipped) {

					pad_source.append(i.len(), '-');
					pad_match.append(i.len(), '+');
					pad_target.append(i.len(), '+');

			}	
			int count = 0;
			while (tdna_itr.good()) {
				if (count >= i.len()) {
					break;
				}		
				tdna_pos++;
				tdna_itr.next();

				count++;
			}
			

						
		}
		
		else if (i.op() == 'H') {
			//nothing for clipped bases
		}else {
			std::string ps, pt, pm;
			ps.reserve(i.len());
			pm.reserve(i.len());

			ps = t_dna.substr(sdna_pos,i.len()); //tdna is really qdna

			tdna_itr.set_position(tdna_pos);
			int count = 0;
			
			while (tdna_itr.good()) {
				if (count < i.len()) {
					pt += tdna_itr.get();
				} else {
					break;
				}

				tdna_itr.next();
				count++;

			}
			for (unsigned int z = 0; z < ps.length(); z++) {
				if (ps[z] == pt[z]) {
					pad_match += '|';
				} else if (ps[z] != 'A' || ps[z] != 'C' || ps[z] != 'G' || ps[z] != 'T') {
					if (iupac_flag) {
						
						std::vector<char> nukes(IUPAC::get_base(ps[z]));
						bool replaced = false;
						unsigned int nuke_ptr = 0;
						for (unsigned int n = 0; n < nukes.size(); n++) {
							if (nukes[n] == pt[z]) {
								pad_match += '|';
								replaced  = true;
								nuke_ptr = n;
								break;
							}
							//nuke_ptr++;
						}
						if (!replaced) {
							pad_match += ' ';
						}
						else if (!keep_iupac) {
							//std::cerr << "nukes["<<nuke_ptr<<"]: " << nukes[nuke_ptr] << " nukes.size() " << nukes.size() << std::endl;
							ps[z] = nukes[nuke_ptr];
						}//keep_iupac
					}//iupac_flag
					else {
						pad_match += ' ';
					}
				}//end else if checking ps[z] agianst nukes
				else {
					pad_match += ' ';
				}


			}//end for loop
			pad_source += ps;
			pad_target += pt;
			sdna_pos += i.len();
			tdna_pos += i.len();

			
			
		}
		tot++;

	}
	/*
	std::cerr << "pad_source: " << pad_source << std::endl;
	std::cerr << "pad_target: " << pad_target << std::endl;
	std::cerr << "pad_match : " << pad_match << std::endl;
	*/
}

void BAMUtils::reverse() {
	reverse_comp(pad_source);
	std::reverse(pad_match.begin(), pad_match.end());
	reverse_comp(pad_target);
}




//modifies existing, careful
//this could probably be faster -- maybe with an std::transform
void BAMUtils::reverse_comp(std::string& c_dna) {
	for (unsigned int i = 0; i<c_dna.length(); i++) {
		switch (c_dna[i]) {
			case 'A':
				c_dna[i] = 'T';
				break;
			case 'T':
				c_dna[i] = 'A';
				break;
			case 'C':
				c_dna[i] = 'G';
				break;
			case 'G':
				c_dna[i] = 'C';
				break;
			case '-':
				c_dna[i] = '-';
				break;

			default:
				break;
		}
	}
	std::reverse(c_dna.begin(), c_dna.end());
	
}

//faster if you leave this part out
void BAMUtils::calc_error() {
	q_err = 0;
	homo_err = 0;
	mm_err = 0;
	indelErr = 0;
	
	
	std::vector<std::string> q_score_vec;
	split(q_scores, ',', q_score_vec);
	
    int loop_limit = static_cast<int>( pad_source.length() );//get_phred_len(strtol(q_score_vec[0].c_str(), NULL, 10));
	for (int i = get_slop(); i < loop_limit; i++) {
		if (pad_match[i] != '|') {
			q_err++;
			//check homopol
			if (pad_source[i] != '-' && 
				((i > 0 && (pad_source[i-1] == pad_source[i])) ||
				 (i+1 < static_cast<int>( pad_target.length() ) && (pad_source[i+1] == pad_source[i])))) {
					homo_err++;
			}
			else if(pad_target[i] != '-' &&
					((i > 0 && (pad_target[i-1] == pad_target[i])) ||
					 (i+1 < static_cast<int>( pad_target.length() ) && (pad_target[i+1] == pad_target[i])))) {
						
					homo_err++;
			}
			if ((pad_source[i] != '-') && (pad_target[i] != '-')) {
				mm_err++;
			} else {
				indelErr++;
			}
			
		}
	}

}
	
	
void BAMUtils::score_alignments(const std::string& qscores, bool calc_region_error, int startBase, int stopBase) {

	
	
	t_diff = 0;
	match_base = 0;
	n_qlen = 0;
	t_len = 0;
	phred_lens.clear();

	int first_aligned_flow=0;
	std::string first_aligned_flow_bam_tag = "ZF";
	bool score_flows = (flow_order != "");
	if(score_flows && bam_record.get_optional_field(first_aligned_flow_bam_tag,first_aligned_flow)) {
		score_flows = true;
		//std::cout << get_name() << "\tfirst aligned flow = " << first_aligned_flow << ", flow order = " << flow_order << "\n";
	}

	equiv_len = std::tr1::unordered_map<coord_t, coord_t>(pad_source.length());
	int consecutive_error = 0;

	// flow-based error determination
	int current_flow = first_aligned_flow;
	max_aligned_flow = -1;
	char prev_target_base = ' ';
	char next_target_base = ' ';
	unsigned int next_target_base_index = 0;

	//region-errors
	_region_homo_err = 0;
	_region_mm_err = 0;
	_region_indel_err = 0;
	_region_ins_err = 0;
	_region_del_err = 0;	
	
	//using namespace std;
	for (int i = 0; (unsigned int)i < pad_source.length(); i++) {
		//std::cerr << " i: " << i << " n_qlen: " << n_qlen << " t_len: " << t_len << " t_diff: " << t_diff << std::endl;
		if (pad_source[i] != '-') {
			t_len = t_len + 1;
		}
		
		if (pad_match[i] != '|') {
			t_diff = t_diff + 1;
			//region-errors: start			
			if(calc_region_error && i>=startBase && i <=stopBase){
				if (pad_source[i] == '-' && 
						((i > 0 && (pad_target[i-1] == pad_target[i])) ||
								(i+1 < static_cast<int>( pad_target.length() ) && (pad_target[i+1] == pad_target[i])))) {
						_region_homo_err++; 
				}
				else if(pad_target[i] == '-' &&
						((i > 0 && (pad_source[i-1] == pad_source[i])) ||
								(i+1 < static_cast<int>( pad_source.length() ) && (pad_source[i+1] == pad_source[i])))) {						
						_region_homo_err++;
				}
				if ((pad_source[i] != '-') && (pad_target[i] != '-')) {
					_region_mm_err++;
				} else {
					_region_indel_err++;
					if((pad_source[i] == '-') && (pad_target[i] != '-')){
						_region_ins_err++;
					}
					if((pad_source[i] != '-') && (pad_target[i] == '-')){
						_region_del_err++;
					}								
				}			
			}
			//region-errors end
			
			//need to check if this is worthy of a CF/IE style error
			/*
			 * Rationale for the below
			 * i > 0
			 * 		due to a few i-1's need to make sure we can't have a negative index into string
			 * pad_match[i-1] != '|'
			 * 		this means there is SOME error at this position
			 * ( ( pad_target[i] == pad_target[i - 1] ) || pad_match[i] == '-' )
			 * 		 if the reference sequence's previous base is equal to this reference sequence base i'm
			 * 		 going to favor the reference being correct and imply that it's a CF/IE error in the read
			 * 		 OR
			 * 		 if the error is an indel i'm going to favor CF/IE as being the cause as well
			 */
			if (i > 0 && pad_match[i-1] != '|' && ( ( pad_target[i] == pad_target[i - 1] ) || pad_match[i] == '-' ) ) {
				consecutive_error = consecutive_error + 1;
			} else {
				consecutive_error = 1;
			}
		} else {
			consecutive_error = 0;
			match_base = match_base + 1;
		}
		if (pad_target[i] != '-') {
			n_qlen = n_qlen + 1;
		}
		equiv_len[i] = t_len;
		if (error_lens.size() != 0 && pad_match[i] != '|') {
			if(pad_match[i] == ' ')
				error_table_mis[n_qlen]++;
			else if(pad_match[i] == '+')
				error_table_ins[n_qlen]++;
			else
				error_table_del[n_qlen]++;
		}

		if(score_flows) {
			// Figure out what flow we are in
			char thisBase = pad_target[i];

			if (pad_target[i] != '-') {
				// Not a deletion in the read, so only advance flow if we're in a new homopolymer
				if(prev_target_base != ' ' && thisBase != prev_target_base) {
					while(flow_order[(++current_flow) % flow_order.length()] != thisBase) {
					}
				}
				prev_target_base = thisBase;
			} else {
				// In a deletion.  So what to do depends on the deleted base:
				//   if alignment hasn't started yet or the deleted base matches previous read HP
				//     don't advance the flow
				//   otherwise
				//     if there is not another read base then we're done
				//     if there is another read base
				//       if deletion matches next read base, advance the flow to next read base
				//       otherwise look for an intermediate flow (possibly the current flow) matching the deletion
				//       if none found, leave flow as-is
				char deleted_base = pad_source[i];
				if((prev_target_base != ' ') && (deleted_base != prev_target_base)) {
					// find next target base, if we don't already know it
					if(next_target_base_index <= (unsigned int)i) {
						for(next_target_base_index=i+1; next_target_base_index < pad_target.length(); next_target_base_index++) {
							if(pad_target[next_target_base_index] != '-') {
								next_target_base = pad_target[next_target_base_index];
								break;
							}
						}
					}
					if(next_target_base != ' ') {
						// find flow for next target base
						int next_flow = current_flow;
						if(next_target_base != prev_target_base) {
							while(flow_order[next_flow % flow_order.length()] != next_target_base) {
								next_flow++;
							}
						}
						if(deleted_base == next_target_base) {
							current_flow = next_flow;
							prev_target_base = next_target_base;
						} else {
							int flow = 0;
							for(flow=current_flow; flow < next_flow; flow++) {
								if(flow_order[flow % flow_order.length()] == deleted_base) {
									current_flow = flow;
									prev_target_base = deleted_base;
									break;
								}
							}
						}
					}
				}
			}
  
			if(current_flow > max_aligned_flow)
				max_aligned_flow = current_flow;
			if(consecutive_error > 0) {
				if(flow_err.size()<1 || flow_err[flow_err.size()-1]!=current_flow) 
                                  	{flow_err.push_back(current_flow); 
					flow_err_bases.push_back(consecutive_error);
                                 	}
				else    
					{flow_err_bases[flow_err_bases.size()-1]+=consecutive_error;
                                 	}

				flow_err_bases.push_back(consecutive_error);
			}
		}
	}
	// A sanity check for debugging purposes: ensure all errrors are accounted for
	//int err_total = 0;
	//for(unsigned int i=0; i<error_table_mis.size(); i++) {
	//	err_total += error_table_mis[i];
	//	err_total += error_table_ins[i];
	//	err_total += error_table_del[i];
	//}
	//if(err_total != t_diff) {
	//	std::cout << "ERROR: " << get_name() << " err_total = " << err_total << " and t_diff = " << t_diff << "\n";
	//}

	//if(score_flows) {
	//	std::cout << "aligned to flow " << max_aligned_flow << "\n";
	//	std::cout << flow_err.size() << " flow errors\n";
	//	for(unsigned int i=0; i<flow_err.size(); i++)
	//		std::cout << flow_err[i] << "\t" << flow_err_bases[i] << "\n";
	//}
	
	//get qual vals from  bam_record
	std::vector<std::string> q_score_vec;
	split(qscores, ',', q_score_vec);
	phred_lens.reserve(q_score_vec.size());
	std::vector<coord_t> q_len_vec(q_score_vec.size(), 0);
	std::vector<double> Q;
	
	//setting acceptable error rates for each q score, defaults are
	//7,10,17,20,47
	for (std::vector<std::string>::size_type i = 0; i < q_score_vec.size(); i++) {
		phred_lens.push_back(0);
                int phred_val = strtol(q_score_vec[i].c_str(),NULL, 10);
		if (round_phred_scores) {
                    if (phred_val == 7) {
                       Q.push_back(0.2);
                    }
                    else if(phred_val == 10) {
                       Q.push_back(0.1);
                    }
                    else if(phred_val == 17) {
                       Q.push_back(0.02);
                    }
                    else if(phred_val == 20) {
                       Q.push_back(0.01);
                    }
                    else if(phred_val == 47) {
                       Q.push_back(0.00002);
                    } else {
                      Q.push_back(  phred_to_upper_prob( phred_val ) );
                    }
		} else {
			Q.push_back( calculate_phred( static_cast<double>( phred_val) ) );
		}

	}
	

	coord_t prev_t_diff = 0;
	coord_t prev_loc_len = 0;
	coord_t i = pad_source.length() - 1;

	for (std::vector<std::string>::size_type k =0; k < Q.size(); k++) {
		coord_t loc_len = n_qlen;
		coord_t loc_err = t_diff;
		if (k > 0) {
			loc_len = prev_loc_len;
			loc_err = prev_t_diff;
		}
		
		while ((loc_len > 0) && (static_cast<int>(i) >= num_slop) && i > 0) {

			if (q_len_vec[k] == 0 && (((loc_err / static_cast<double>(loc_len))) <= Q[k]) /*&& (equivalent_length(loc_len) != 0)*/) {
				//coord_t eqv_len = equivalent_length(loc_len);

				coord_t eqv_len = loc_len;
				q_len_vec[k] = eqv_len;
				phred_lens[k] = eqv_len;
				int key = strtol(q_score_vec[k].c_str(),NULL, 10);

				if (phreds.count(key) > 0) {
					phreds.erase(phreds.find(key));
				}
				phreds[key] = eqv_len;
				

				
				prev_t_diff = loc_err;
				prev_loc_len = loc_len;
				break;
			}
			if (pad_match[i] != '|') {
				loc_err--;
			}


			if (pad_target[i] != '-') {

				loc_len--;
			}

			i--;
			

		}
		
	}
	
	
}


void BAMUtils::adjust_region_base_positions(int* start_base, int* stop_base, std::string this_pad_target, bool *region_clipped)
{
	int non_del_base = 0;
	bool start_base_counted = false;
	bool stop_base_counted = false;

	for (int i = 0; (unsigned int)i < this_pad_target.size(); i++) {	
		if(this_pad_target[i] != '-') non_del_base++; 
		if(*start_base == (non_del_base-1) && start_base_counted == false ){
			*start_base = i;
			start_base_counted = true;			
		}
		if(*stop_base == (non_del_base-1)){
			*stop_base = i;
			stop_base_counted = true;	
			break;
		}	
	}
	if(stop_base_counted==false){
		*stop_base = (*stop_base < (static_cast<int>(this_pad_target.size())-1)?(*stop_base):(static_cast<int>(this_pad_target.size())-1));
		*region_clipped = true;
		}
	else{
		if(this_pad_target[*start_base-1] == '-')*start_base = *start_base - 1;
		if(this_pad_target[*stop_base-1] == '-')*stop_base = *stop_base - 1;	
		if( (this_pad_target.size()-*stop_base)<this_pad_target.size()/10) *region_clipped = true; 
	}
	
}


void BAMUtils::save_region_error_positions(std::string my_pad_match, std::vector<std::pair<long,int> >  &region_error_positions, int start_base, int stop_base)
{
	coord_t ref_pos = bam_record.get_pos();
	coord_t error_pos = ref_pos;
	int j = 0;
	int num_errors = 0;
	std::pair <long, int> error_pos_pair;
	
	
	for(int i = 0; i <= stop_base; i++){
		if(i >= start_base && my_pad_match.at(i) == '+') { //insertion
			num_errors++;
			if((i < stop_base && my_pad_match.at(i+1) != '+') || i == stop_base){				
				error_pos = ref_pos + j - 1;
				error_pos_pair.first = error_pos;
				error_pos_pair.second = num_errors;
				region_error_positions.push_back(error_pos_pair);
				num_errors = 0;
			}
		}		
		else{
			j++;
			if(i >= start_base && my_pad_match.at(i) != '|'){
				num_errors = 1;
				error_pos = ref_pos + j - 1;
				error_pos_pair.first = error_pos;
				error_pos_pair.second = num_errors;				
				region_error_positions.push_back(error_pos_pair);
			}
		}		
	}
	
	return;
	
}
