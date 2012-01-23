/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef QUAL_H
#define QUAL_H

/*
 *  Qual.h
 *  SamUtils
 *
 *  Created by Michael Lyons on 1/4/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */

#include <string>
#include "sam.h"


/**
 An iterable container class for the Qual field of a SAM/BAM.  Requires the BAMRead that the Qual
 is retrieved from to still remain in scope.  
 
 QUAL: ASCII of base QUALity plus 33 (same as the quality string in the Sanger FASTQ format).
 A base quality is the phred-scaled base error probability which equals 10 log10 Prfbase is wrongg.
 This field can be a `*' when quality is not stored. If not a `*', SEQ must not be a `*' and the
 length of the quality string ought to equal the length of SEQ
*/
class Qual {

public:
	
	/**
	 Default constructor.  Class isn't of use if it's default constructed.
	 
	 @return	Qual		default constructed Qual object
	 */
	Qual();
	
	/**
	 Standard constructor.  Requires a bam1_t* in order to extract the Qual field.
	 
	 @param		bam1_t*		alignment from bam file
	 @return	Qual		constructed Qual object
	 */
	Qual(const bam1_t* BAM_record);
	
	/**
	 Returns the length of the Qual string
	 @return	long		length of qual string
	 */
	long	get_length()	const;
	
	/**
	 Qual::iterator is how you access the actual elements of hte Qual string.
	 Currently only a forward iterator.
	 
	 //example code.  Assumes a valid BAMRead object named "read"
	 Qual qual = read.get_qual();
	 for(Qual::iterator qual_iterator = qual.get_iterator(); qual_iterator.good(); qual_iterator.next()) {
		cout << qual_iterator.get();
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
		 Returns a character from the Qual string
		 @return	char		one character from Qual string
		 */
		char	get();
		
	private:
		//friends
		friend class Qual;
		//ctors
		iterator();
		iterator(const bam1_t* BAM_record);
		//funcs
		void _init();
		
		//members
		const bam1_t*	bam_record;
		size_t			position;
		size_t			qual_len;
		uint8_t*		bam_qual;
	};
	
	/**
	 Gives you an iterator for the Qual
	 
	 @return	Qual::iterator
	 */
	iterator	get_iterator();
	/**
	 Returns a string representation of the Qual
	 
	 @return	string		qual string from SAM/BAM
	 */
	std::string	to_string();
	
private:
	
	//funcs
	void _init();
	
	//data members
	long			qual_length;
	const bam1_t*	bam_record;


};

#endif // QUAL_H

