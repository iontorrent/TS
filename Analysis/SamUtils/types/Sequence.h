/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef SEQUENCE_H
#define SEQUENCE_H

/*
 *  Sequence.h
 *  SamUtils
 *
 *  Created by Michael Lyons on 1/4/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */

#include "sam.h"

/**
 A class representing the SEQ field from a SAM/BAM file
 
 Requires the BAMRead that the Sequence is retrieved from to still remain in scope.  

 
 SEQ: fragment SEQuence. This field can be a `*' when the sequence is not stored. If not a `*',
 the length of the sequence must equal the sum of lengths of M/I/S/=/X operations in CIGAR.
 An `=' denotes the base is identical to the reference base. No assumptions can be made on the
 letter cases.
 
 
 */
class Sequence {
public:
	/**
	 Default constructor.  Class is useless if you use this
	 
	 @return	Sequence	a default constructed Sequence object
	 */
	Sequence();
	
	/**
	 Standard constructor.  Requires a bam1_t* in order to extract the sequence information
	 
	 @param		const bam1_t*	pointer to an alignment from a SAM/BAM
	 @return	Sequence		a constructed and usable Sequence object
	 */
	Sequence(const bam1_t* BAM_record);
	
	/**
	 Returns the length of the Sequence string
	 
	 @return	long		length of sequence string;
	 */
	long get_length()	const;
	
	
	/**
	 Sequence::iterator is how you access the actual elements of the Sequence string.
	 Currently only a forward iterator.
	 
	 //example code.  Assumes a valid BAMRead object named "read"
	 Sequence sequence = read.get_seq();
	 for(Sequence::iterator sequence_iterator = sequence.get_iterator(); sequence_iterator.good(); sequence_iterator.next()) {
		cout << sequence_iterator.get();
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
		 Returns a character from the Sequence string
		 
		 @return	char		one character from Sequence string
		 */
		char	get();
		
		/**
		 Convenience method so you can move around the Sequence string.  The
		 parameter, pos, corresponds to the 0 based index into the String.  It must be less
		 than the length of the Sequence.  
		 
		 If this is called and an iterator exists, the value the iterator points to will be changed.
		 
		 @param	size_t pos		0-based index in Sequence string
		 */
		void	set_position(size_t pos);
		
	private:
		
		//friends
		friend class Sequence;
		//ctors
		iterator();
		iterator(const bam1_t*	BAM_record);
		//funcs
		void _init();
		//member vars
		const bam1_t*	bam_record;
		size_t			position;
		size_t			seq_len;
		uint8_t*		bam_seq;
	};
	/**
	 Gives you an iterator for the Sequence
	 
	 @return	Sequence::iterator
	 */
	iterator	get_iterator();
	/**
	 Returns a string representation of the Sequence
	 
	 @return	string		sequence string from SAM/BAM
	 */
	std::string	to_string();
	
private:
	
	//funcs
	void _init();
	
	
	//data members
	long			seq_length;
	const bam1_t*	bam_record;
		
		
	
	
};

#endif // SEQUENCE_H

