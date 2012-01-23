/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef MD_H
#define MD_H

/*
 *  MD.h
 *  SamUtils
 *
 *  Created by Michael Lyons on 1/5/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */


#include "sam.h"

/*
 uint8_t *md_ptr = bam_aux_get(bam_record, "MD");
 if (md_ptr) {
 return bam_aux2Z(md_ptr);
 } else {
 return NULL;
 }
 
 
  */

/**
	An iterable container class for the MD field of a SAM/BAM.  Requires the BAMRead that the MD
	is retrieved from to still remain in scope. 
 
	The MD field aims to achieve SNP/indel calling without looking at the reference. For example, a string `10A5^AC6'
	means from the leftmost reference base in the alignment, there are 10 matches followed by an A on the reference which
	is different from the aligned read base; the next 5 reference bases are matches followed by a 2bp deletion from the
	reference; the deleted sequence is AC; the last 6 bases are matches. The MD field ought to match the CIGAR string.

	String for mismatching positions. Regex : [0-9]+(([A-Z]|\^[A-Z]+)[0-9]+)
*/
class MD {
public:
	/**
	 Default constructor.  Class isn't of use if it's default constructed.
	 
	 @return	MD		default constructed MD object
	 */
	MD();
	/**
	 Standard constructor.  Requires a bam1_t* in order to extract the MD field.
	 
	 @param		bam1_t*		alignment from bam file
	 @return	MD			constructed MD object
	 */
	
	MD(const bam1_t* BAM_record);
	
	/**
	 Returns the length of the MD string
	 @return	long		length of MD string
	 */
	
	long get_length()	const;
	/**
	 MD::iterator is how you access the actual elements of hte MD string.
	 Currently only a forward iterator.
	 
	 //example code.  Assumes a valid BAMRead object named "read"
	 MD md = read.get_md();
	 for(MD::iterator md_iterator = md.get_iterator(); md_iterator.good(); md_iterator.next()) {
		cout << md_iterator.get();
	 }
	 
	 */
	class iterator {
	public:
		/**
		 Returns the status of the iterator.  If true, there are more elements accessible
		 
		 @return	bool		true if iterator has more elements
		 */
		bool	good()	const;
		/**
		 Advances iterator to the next element
		 */		
		void	next();
		/**
		 Returns a character from the MD string
		 @return	char		one character from MD string
		 */		
		char	get();
	private:
		
		//friends
		friend class MD;
		//ctors
		iterator();
		iterator(const bam1_t*	BAM_record);
		//funcs
		void _init();
		//member vars
		const bam1_t*	bam_record;
		size_t			position;
		size_t			md_len;
		const char*		md_ptr;
	};
	/**
	 Gives you an iterator for the MD
	 
	 @return	MD::iterator
	 */
	iterator	get_iterator();
	/**
	 Returns a string representation of the MD
	 
	 @return	string		MD string from SAM/BAM
	 */
	std::string	to_string();
	
private:
	
	//funcs
	void _init();
	
	
	//data members
	long			md_length;
	const bam1_t*	bam_record;
	
	
	
	
};

#endif // MD_H

