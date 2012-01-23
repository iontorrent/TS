/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/*
 *  Cigar.h
 *  SamUtils
 *
 *  Created by Michael Lyons on 12/22/10.
 *  Copyright 2010 Life Technologies. All rights reserved.
 *
 */
#ifndef CIGAR_H
#define CIGAR_H

#include <sstream>
#include "sam.h"

/**
 A class representing the CIGAR field from a SAM/BAM
 
 Requires the BAMRead that the Cigar is retrieved from to still remain in scope.  

 
 CIGAR: CIGAR string. The CIGAR operations are given in the following table (set `*' if un-
 available):
 Op		BAM		Description
 M		0		alignment match (can be a sequence match or mismatch)
 I		1		insertion to the reference
 D		2		deletion from the reference
 N		3		skipped region from the reference
 S		4		soft clipping (clipped sequences present in SEQ)
 H		5		hard clipping (clipped sequences NOT present in SEQ)
 P		6		padding (silent deletion from padded reference)
 =		7		sequence match
 X		8		sequence mismatch
 Notes:
 -H can only be present as the first and/or last operation.
 
 -S may only have H operations between them and the ends of the CIGAR string.
 
 -For mRNA-to-genome alignment, an N operation represents an intron. For other types of
  alignments, the interpretation of N is not defined.
 
 
 
 */
class Cigar {

	public:
	/**
	 Default constructor.  You cannot do anything with this class if not constructed with a 
	 bam1_t*
	 @return	Cigar		default constructed Cigar object
	 */
	Cigar();
	/**
	 Standard constructor.  Needs a bam1_t in order to extract the cigar information
	 
	 @params	const bam1_t*	an alignment from a SAM/BAM
	 @return	Cigar			constructed Cigar object
	 */
	Cigar(const bam1_t* BAM_record);		
	
	
	/**
	 Returns cigar string that you'd find in BAM/SAM.  
	 
	 @return	std::string		string version of CIGAR
	 */
	std::string to_string();
	/**
	 Returns length of CIGAR string itself.
	 
	 @return	long			length of cigar string
	 */
	int	get_length()	const {
		return cigar_len;
	}
		
	
	/**
	 Cigar::iterator class to help a user of this class navigate the Cigar string in a 
	 stress free manner.  This iterator is forward only for now.
	 
	 This requires an existing Cigar object to be constructed.  You can retrieve an iterator by
	 calling Cigar::get_iterator()
	 
	 example:
	 //assume a BAMRead already exists named read;
	 
	 Cigar cigar = read.get_cigar();
	 Cigar::iteartor cigar_iterator = cigar.get_iterator();
	 
	 
	 The iterator advances by Cigar operation.  For each element in the Cigar there's an
	 operation and an associated length of the operation.  
	 Cigar::iterator::len() corresponds to the operation length
	 Cigar::iterator::op() gives the actual operation 
	 
	 */
	class iterator {
		
		public:
		
		/**
		 Are there still iterable elements of the string?
		 
		 @return	bool	true if there are still more elements
		 */
		bool good() const;		
		/**
		 advances iterator to the next cigar element
		 
		 */
		void next();
		/**
		 Returns length of cigar operation
		 
		 @return	int		length of cigar op
		 */
		int len();
		/**
		 Returns the actual cigar operation
		 
		 @return	char	cigar op
		 */
		char op() const; 		
		
			
			
			
				
		private:
		iterator();
		iterator(const bam1_t* BAM_record);		
		void _init(); 		
			//private functions
		void _set_len_and_op();		
		
			//member vars
			friend class Cigar;
			const bam1_t*  bam_record;
			uint32_t length;
			uint32_t operation;
			size_t		pos; //position in cigar list
			uint32_t*	cigar_ptr;

	};
	
	
	/**
	 Gives you an iterator to the Cigar
	 
	 @return	Cigar::iterator
	 */
	iterator get_iterator();
		
	
	
		
private:
	
	//funcs
	void _init();
	
	//CigarElementsArray	elements;
	const bam1_t*	 bam_record;
	int				cigar_len;

};


#endif // CIGAR_H

