/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BAMUTILS_H
#define BAMUTILS_H


/*pad_so
 *  BAMUtils.h
 *  SamUtils
 *
 *  Created by Michael Lyons on 12/9/10.
 *  Copyright 2010 Life Technologies. All rights reserved.
 *	
 *
 */

#include <vector>
#include <map>
#include <algorithm>
#include <math.h>
#include <tr1/unordered_map>
#include "types/samutils_types.h"
#include "types/BAMRead.h"


/**
 A class that provides common utility functions for Ion Torrent R&D/production
 code.  
 
 Works on alignments from SAM/BAM files.  
 */

class BAMUtils {

public:
	
	
	//constructors
	/**
	 Default constructor.  a requirement to be part of an STL container
	 */
	BAMUtils();
	
	/**
	 Creates a BAMUtils object using all default values for the phred scores, start slop, IUPAC and soft clip truncation flags, and error table values.
	 Defaults are:
	 qscores:				7,10,17,20,47
	 slop:					0
	 iupac_flag:			true
	 keep_iupac:			true
	 trunace_soft_clipped:	true
	 error_table_min_len:	0
	 error_table_max_len:	0
	 error_table_step_size:	0
	 
	 @param		BAMRead&	a valid BAMRead object
	 @return	BAMUtils	constructed BAMUtils object
	 */
	BAMUtils(BAMRead& BAM_record);
	/**
	 Creates a BAMUtils object using default values for the IUPAC and soft clip truncation flags, and error table values.
	 Defaults are:
	 iupac_flag:			true
	 keep_iupac:			true
	 trunace_soft_clipped:	true
	 error_table_min_len:	0
	 error_table_max_len:	0
	 error_table_step_size:	0
	 
	 
	 @param		BAMRead*		a pointer to a BAMRead
	 @param		std::string		a csv string representing the phred scores ie: "7,10,17,20,47"
	 @param		long			a value representing # of bases to ignore from the start of the read for 
								error calculations
	 @return	BAMUtils		constructed BAMUtils object
	 
	 */
	BAMUtils(BAMRead* BAM_Record_Ptr, std::string qscores, coord_t slop);
	
	/**
	 Creates a BAMUtils object using default values for the IUPAC and soft clip truncation flags, and error table values.
	 Defaults are:
	 iupac_flag:			true
	 keep_iupac:			true
	 trunace_soft_clipped:	true
	 error_table_min_len:	0
	 error_table_max_len:	0
	 error_table_step_size:	0
	 
	 
	 @param		BAMRead&		a reference to a BAMRead
	 @param		std::string		a csv string representing the phred scores ie: "7,10,17,20,47"
	 @param		long			a value representing # of bases to ignore from the start of the read for 
	 error calculations
	 @return	BAMUtils		constructed BAMUtils object
	 
	 */
	
	BAMUtils(BAMRead& BAM_record, std::string qscores, coord_t slop);

	/**
	 Creates a BAMUtils object using default values for the IUPAC and soft clip truncation flags, and error table values.
	
	 
	 
	 @param		BAMRead&						a reference to a BAMRead
	 @param		std::string qscores				a csv string representing the phred scores ie: "7,10,17,20,47"
	 @param		long start_slop					a value representing # of bases to ignore from the start of the read for 
												error calculations
	 @param		bool IUPAC_Flag					if true, will try and validate IUPAC alignment mismatches in MD string
	 @param		bool Keep_IUPAC					if true, will keep IUPAC's in tDNA string instead of nucleotide bases
	 @param		bool Truncate_Soft_Clipped		if true, ignores soft clipped regions of reads in error calculations
	 @param		int error_table_min_len			minimum length for error table record keeping
	 @param		int	error_table_max_len			maximum length for error table record keeping
	 @param		int error_table_step_size		interval for error table.  
         @param         int three_prime_clip                    amount of 3 prime bases to ignore
         @param         bool rounding_phred_scores              bool on wehther to follow colloqiual phred scores
         @param         bool Five_Prime_Justify                 bool to follow mapper's indel justification (false) or force 5prime(true)
         @param         std::string Flow_Order                  Sequencing flow order
	 @return	BAMUtils						constructed BAMUtils object
	 
	 */	
	BAMUtils(BAMRead& BAM_record, std::string qscores, coord_t slop, 
                bool IUPAC_Flag, bool Keep_IUPAC, bool Truncate_Soft_Clipped, 
                int error_table_min_len, int error_table_max_len, 
                int error_table_step_size, int three_prime_clip, 
                bool rounding_phred_scores, bool Five_Prime_Justify, std::string &Flow_Order);
	
	~BAMUtils();
	
	
	typedef	std::tr1::unordered_map<int, coord_t>					q_lens;  //example key, value (17, 100)
	typedef std::tr1::unordered_map<int, coord_t>					q_lens_itr;
	typedef std::tr1::unordered_map<int, int>::const_iterator		error_table_iterator;
	
	//Default.sam.parsed field accessors
	/**
	 get_name() returns name of read.  matches name from fastq and sam file
	 */
	std::string					get_name(); 
	/**
	 get_strand() returns an integer representing the strand.  0 is the positive strand
	 16 is the negative strand
	 */
	int							get_strand();
	/**
	 get_t_start() is the start of the alignment in the genome
	 */
	coord_t						get_t_start();
	/**
	 get_t_length() returns the length of the alignment with respect to the reference
	 */
	coord_t						get_t_length();
	/**
	 get_q_length() returns the length of the aligned portion of the read
	 Note on this field:
	 	 In discussions on bugs in alignStats over the last few months this field commonly has been
	 	 referred to as "read space."  This is only the length of the read that does not contain these cigar operations:
	 	 D, S, H (or any or type of deletion type or clipping type).

	 */
	coord_t						get_q_length();
	/**
	 get_full_q_length() returns the full query length, inclusive of any alignment soft clipping
	 */
	coord_t						get_full_q_length();
	/**
	 get_match() returns number of bases that match the reference
	 */
	coord_t						get_match();
	/**
	 get_percent_id() returns the % similarity between the reference and read
	 */
	double						get_percent_id();
	/**
	 get_q_error() returns the # of errors in the lowest Q-score.  By default this is Q7
	 */
	coord_t						get_q_error();
	/**
	 get_homo_errors() returns the # of errors caused by homopolymers in the read
	 */
	int							get_homo_errors();
	/**
	 get_mismatch_errors() returns the # of errors caused by mismatches in the read
	 */
	int							get_mismatch_errors();
	/**
	 get_indel_errors() returns the # of errors caused by indels in the read
	 */
	int							get_indel_errors();
	/**
	 get_qdna() returns the query dna sequence as a string
	 */	
	std::string					get_qdna();
	/**
	 get_matcha() returns the annotated match sequence, for example "||||+---    || | | -++|+"
	 */
	std::string					get_matcha();
	/**
	 get_tdna() returns the template dna sequence
	 */
	std::string					get_tdna();
	/**
	 get_rname() returns the name of the contig this read aligned too
	 */
	std::string					get_rname();
	/**
	 get_slop() returns the # of bases at the start of the read which are ignored in error calculation
	 */
	int							get_slop();
	/**
	 get_phred_len() returns the length of the read given a Q# score.  
	 */
	coord_t						get_phred_len(int phred_score);
	/**
	 get_t_diff() returns the # of differences in the alignment between the template and query sequences 
	 */
	inline coord_t get_t_diff() { return t_diff; }

	/**
	 Returns QUAL field from SAM/BAM.  The string is in the orientation of the read.
	 
	 If the read matches to the reverse strand, the QUAL string is in a 3'->5' orientation.
	 
	 This is ultimately a convenience to the user, becuase you can achieve this by calling
	 BAMRead::get_qual().to_string()  -- however, that produces a string that is always 5'->3' in orientation
	 
	 @return	std::string	representation of QUAL field
	 */
	std::string					get_qual() {
		
		std::string qual = get_bamread().get_qual().to_string();
		if (get_bamread().mapped_reverse_strand()) {
			std::reverse(qual.begin(), qual.end());

		}
		return qual;
			
	}
	
	
	//getters & setters for mutatable fields
	coord_t			get_genome_length();
	void			set_genome_length(coord_t len);
	void			set_slop_bases(coord_t slop);
	void			set_flow_order(const std::string &fo);
	
	/**
	 get_soft_clipped_bases() returns the length of the read that
	 was not included in the alignment, aka soft clipped
	 */
	int				get_soft_clipped_bases() {
		return soft_clipped_bases;
	}
	
	//to help out pileup
	/**
	 A convenience function that checks to see if this read covers a specific portion of the genome
	 
	 @param	int pos_in_genome		1-based position in genome
	 @return	bool				true if position is covered
	 */
	inline	bool			is_pos_covered(int pos_in_genome) {
		coord_t read_end = (get_t_start() + get_t_length());  
		if (pos_in_genome > read_end || pos_in_genome < get_t_start()) {
			return false;
		} else {
			return true;
		}		
	}
	
	/**
	 A convenience function that checks to see if this read covers a specific portion of the genome
	 for a specific quality level on a phred scale.
	 
	 @param	int pos_in_genome		1-based position in genome
	 @param	int phred_num			phred score of interest
	 @return	bool				true if position is covered
	 */
	inline bool	is_pos_covered(int pos_in_genome, int phred_num) {
		coord_t phred_len = get_phred_len(phred_num);
		coord_t read_end = -1;
		coord_t read_start = -1;
		if (phred_len == 0) {
			return false;
		} else {
			
			if (get_bamread().mapped_reverse_strand()) {
				if (get_t_start() + get_t_length() <= get_genome_length()) {
					if (get_t_length() >= phred_len) {
						read_start = get_t_start() + get_t_length() - phred_len;
						read_end = get_t_start() + get_t_length();
					}
					else if( get_t_length() < phred_len) {
						read_start = get_t_start();
						read_end = get_t_start() + get_t_length();
					}
				} else if( get_t_start() + get_t_length() > get_genome_length() ){
					if (get_t_length() >= phred_len) {
						read_start = get_t_start() + get_t_length() - phred_len;
						read_end = get_genome_length();
					} else if (get_t_length() < phred_len) {
						read_start = get_t_start();
						read_end = get_genome_length();
					}
				}

				
				
			} else {
				read_start = get_t_start();
				read_end = get_t_start() + phred_len;
			}
				
			if (pos_in_genome > read_end || pos_in_genome < (read_start)) {
				return false;
			} else {
				return true;
			}

			
		}

		
		
		
		
	}
	
	/**
	 assumes fasta orientation of read
	 get_query_base() returns the base
	*/
	inline	char	get_query_base(int pos_in_genome) {
		
		coord_t read_end = (get_t_start() + get_t_length());
		if (pos_in_genome > read_end && pos_in_genome < get_t_start()) {
			return 'N';
		} else {
			//return true;
			//have to subtract 1 from get_t_start() becuase it's adjusted from 0 based to 1 based indexing
			// int adj_pos = ((get_t_start()) + get_t_length()) - pos_in_genome;
			int adj_pos;
			
			if (get_bamread().mapped_reverse_strand()) {
				
				adj_pos = (get_t_start() + get_t_length()) - pos_in_genome;
				
			}else {
				adj_pos = pos_in_genome - get_t_start();

			}

			std::string::size_type size = adj_pos;
			 if (size < pad_target.size()) {
				 return pad_target[adj_pos];
			 } else {
				 return 'N';
			 }
			
		}
		
	}
	
	
	
	bool	pass_filtering(int length, double accuracy) {
		
		int errors = error_table[length];
		
		if (length > get_q_length()) {

			return false;
		} 
		else if ( length < errors || length <= 0) {
			return false;
        }
		else {
			

			double acc = (double)(length - errors) / (double)length;
			
			if (acc >= accuracy) {

				return true;
			} else {

				return false;
			}
		}

		

		
	}
	/**
	 returns cumulative error at this length
	 */
	int get_total_error_at_length(int length) {
		std::tr1::unordered_map<int, int>::iterator itr;
		int total_errors = 0;
		for(int i = 1; i <= length; i++){ //length is 1 based
			itr = error_table.find(i);
			if (itr != error_table.end()) {
				total_errors += itr->second;
			}
		}
		return total_errors;


	}
	
	/** 
	 takes 1 based position in read.  Assumes input position is in the 5' orientation.  
	 so, position 1 is always the first based sequenced by the instrument.  Position 2 is always the 2nd base sequenced, etc
	 
	 */
	inline
	bool is_position_an_error(int position_in_read) {
		

		if (get_total_position_errors(position_in_read) > 0) {
			return true;
		} else {
			return false;
		}

		
			
	}
	/**
	 *takes 1 based position in read.
	 *@param position_in_read	1-based position in read
	 *@return	total errors at that position
	 */
	inline
	int	get_total_position_errors(int position_in_read) {
		//using namespace std;
		unsigned int adjusted_pos = position_in_read;
		if (adjusted_pos < ( pad_match.length() ) ) {

			return error_table[  adjusted_pos  ];
		} else {
			return 0;
		}

	}
	
	/**
	 Returns the index of the maximum aligned flow
	 
	 @return	int	max_aligned_flow
	 */
	int get_max_aligned_flow() { return max_aligned_flow; }
	
	/**
	 Returns a reference to the internal flow_err vector which contains
	 flow indices for erroneous flows
	 
	 @return	std::vector<coord_t>&	flow_err
	 */
	std::vector<uint16_t>&	 get_flow_err() { return flow_err; }
	
	/**
	 Returns a reference to the internal flow_err_bases vector
	 which is a companion vector for flow_err and specifies the
	 the number of base errors in each erroneous flow
	 
	 @return	std::vector<uint16_t>&	flow_err_bases
	 */
	std::vector<uint16_t>&	 get_flow_err_bases() { return flow_err_bases; }
	
	/**
	 Returns a reference to the internal BAMRead object
	 
	 @return	BAMRead&	BAMRead used for this utility 
	 */
	BAMRead&	 get_bamread() { return bam_record; }
	
	/**
	 tab delimited string of the get funcitons in this class
	 
	 @return	string		string representation of class
	 */
	std::string		to_string();
	
	
	/**
	 *
	 */
	int			get_adjusted_three_prime_trim_length() { return adjusted_three_prime_trim_len; }
	/*
	 Utilities from BAMRead
	 */
	
	
	
	
	error_table_iterator error_table_begin() {
		return error_table.begin();
	}
	
	error_table_iterator error_table_end() {
		return error_table.end();
	}

	void get_region_errors(int *region_homo_err, int *region_mm_err, int *region_indel_err, int *region_ins_err, int *region_del_err, bool *region_clipped){		
		*region_homo_err = _region_homo_err;
		*region_mm_err = _region_mm_err; 		
		*region_indel_err = _region_indel_err;
		*region_ins_err = _region_ins_err;
		*region_del_err = _region_del_err; 
		*region_clipped = _region_clipped; 
		return;	
	}


private:
	
	
	
	
	void		init_error_table(int min_len, int max_len, int step_size);
	void		_crunch_data();
	void		dna(); 
	void		padded_alignment();
	void		reverse_comp(std::string& c_dna);
	void		score_alignments();
	void		score_alignments(const std::string& qscores, bool calc_region_error, int start_base, int stop_base);
	void		reverse();
	coord_t		equivalent_length(coord_t q_len);
    int left_justify();
	
	std::string	reverse_comp(Sequence c_dna);
	
	void		calc_error();
	void		_init();
	
	//parse alignment string and set values.  will try and only do this once
	void		parse_alignment();
	
	//utility functions
	coord_t convert(std::string num);
	
	void   remove_bases_before_soft_clipped();
	
	//data members  
	BAMRead bam_record;
	BAMRead* bam_rec_ptr;
	
	std::string t_dna;
	std::string pad_match;
	std::string	pad_source;
	std::string pad_target;
	std::string q_scores;
	std::tr1::unordered_map<coord_t, coord_t> equiv_len;
	std::vector<coord_t> phred_lens;
	
	
	coord_t t_diff;
	coord_t t_len;
	coord_t		num_slop;
	coord_t max_slop;
	coord_t	n_qlen;
	coord_t match_base;
	int		q_err;
	int		mm_err;
	int		homo_err;
	int		indelErr;
	int		soft_clipped_bases;
	int		total_three_prime_ignore; //total 3 prime soft clipped bases to ignore when calculating error
	int		adjusted_three_prime_trim_len; //actual number of characters cut off of pad_match/target/seq strings
							   //total_three_prime_ignore but in alignment coordinates
	bool	is_three_prime_soft_clipped;
	bool	round_phred_scores;
	coord_t	genome_len;
	q_lens  phreds;

	// Things relating to evaluation of errors in flow space
	std::string flow_order;
	int max_aligned_flow;
	std::vector<uint16_t> flow_err;
	std::vector<uint16_t> flow_err_bases;
	
	bool	iupac_flag;
	bool	keep_iupac;
	bool	truncate_soft_clipped;
        bool    five_prime_justify; //whether or not to justify indels in the 5'
	//error table data members
						 //len,  errors
	std::tr1::unordered_map<int, int> error_table;
	std::vector<int>				  error_lens;

	inline
	double calculate_phred(double phred) {
	    return pow( 10.0, (phred / -10.0) );
	}
	
	inline
	double phred_to_upper_prob(double phred) {
		return( calculate_phred(phred - 0.5) );
	}

	inline
	double phred_to_lower_prob(double phred) {
		return( calculate_phred(phred + 0.5) );

	}
	

	int		_region_mm_err;
	int		_region_homo_err;
	int		_region_indel_err;
	int		_region_ins_err;
	int		_region_del_err;	
	bool		_region_clipped;
	
	void adjust_region_base_positions(int* startBase, int* stopBase, std::string padTarget, bool *region_clipped);
	//void calc_region_errors(int startBase, int stopBase);

};

//Default.sam.parsed field accessors

//$name in perl
inline
std::string BAMUtils::get_name() {
	return std::string(bam_record.get_qname());
	
}


//$strand in perl
inline
int	BAMUtils::get_strand() {
	
	return bam_record.get_flag();
	
}


//$tStart in perl
inline
coord_t	BAMUtils::get_t_start() {
	//might need to be mpos() 
	//instead of pos()
	return bam_record.get_pos();
	
}


//$tLen in perl
inline
coord_t			BAMUtils::get_t_length() {
	return t_len; //make it 1 based
	
}




//$qLen in perl
inline
coord_t			BAMUtils::get_q_length() {
	
	return n_qlen; //make it 1 based
}

// full query length (inclusive of any alignment soft clipping that may be present)
inline
coord_t			BAMUtils::get_full_q_length() {
	
	return get_q_length() + get_soft_clipped_bases();
}

//match position
//$matchBase in perl
inline
coord_t			BAMUtils::get_match() {
	return	match_base;
	
}

//$percent in perl
inline
double			BAMUtils::get_percent_id() {
	
	if (get_q_length() != 0) {
		return (get_match() / ((double)get_q_length() + get_soft_clipped_bases()));
	} else {
		return 0.0;
	}
	
	
	
}

//$qErr in perl
inline
coord_t			BAMUtils::get_q_error() {
	
	return q_err;
}


//$homoErr
inline
int				BAMUtils::get_homo_errors() {
	return homo_err;
	
}

//$mmErr
inline
int				BAMUtils::get_mismatch_errors() {
	return mm_err;
	
}


//$indelErr
inline
int				BAMUtils::get_indel_errors() {
	
	return indelErr;
	
}

//$qDNA_pad
inline
std::string		BAMUtils::get_qdna() {
	
	return	pad_target;
	
}

//$match
inline
std::string		BAMUtils::get_matcha() {
	
	return	pad_match;
	
}

//$tDNA_pad
inline
std::string		BAMUtils::get_tdna() {
	
	return pad_source;
	
}

//$tName
inline
std::string		BAMUtils::get_rname() {
  if (bam_record.get_rname())
    return std::string(bam_record.get_rname());
  else
    return "*";
}

//$numSlop
inline
int				BAMUtils::get_slop() {
	return num_slop;
	
}


inline
coord_t	 BAMUtils::get_phred_len(int phred_score) {
	
	std::tr1::unordered_map<int, coord_t>::iterator itr = phreds.find(phred_score);
	if (itr != phreds.end()) {
		return itr->second;
	} else {
		return 0;
	}
	
}

inline
void			BAMUtils::set_genome_length(coord_t len) {
	
	genome_len = len;
}
inline
coord_t			BAMUtils::get_genome_length() {
	return genome_len;
	
}

inline
void			BAMUtils::set_slop_bases(coord_t slop) {
	num_slop = slop;
}

inline
void			BAMUtils::set_flow_order(const std::string &fo) {
	flow_order = fo;
}


inline
coord_t		BAMUtils::equivalent_length(coord_t q_len) {
	
	std::tr1::unordered_map<coord_t, coord_t>::iterator it;
	it = equiv_len.find(q_len);
	
	if (it != equiv_len.end()) {
		return it->second;
	} else {
		return 0;
	}

}


#endif // BAMUTILS_H

