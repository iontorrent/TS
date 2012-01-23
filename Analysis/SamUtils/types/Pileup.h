/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */


/*
 *  PileupFactory.h
 *  SamUtils
 *
 *  Created by Michael Lyons on 3/15/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */

#ifndef PILEUP_H
#define PILEUP_H


#include <algorithm>
#include <iostream>
#include <tr1/memory>
#include <vector>
#include <tr1/unordered_map>
#include <list>

#include "BAMRead.h"
#include "../BAMUtils.h"
#include "samutils_types.h"

using namespace std;
	
/**
 This class aims to encapsulate one position in the reference sequence.  It contains
 all of the reads that overlay this particular position.  The read(s) may or may not
 actually have coverage at this position, however.  
 
 This class is retrieved from a PileupFactory.  It's not recommended that you try and construct them
 yourself.
 
*/
class Pileup {
public:
	
	/**
	 Default constructor.  A necessary evil in order to use these objects in STL containers
	 */
	Pileup(): _tid(-1), _pos(-1), _num_reads(0),_debug(false), all_covered(false) { 
		
	}	
	/**
	 A light constructor that initializes the Pileup with it's position, reference, and Phred scores
	 The phred scores are used to see whether or not this read has some level of Phred coverage.  
	 
	 For instance, given phreds of 7,10,17,20,47 you can see how many Q7 bases cover this position, and
	 how many Q47 bases cover this position, etc.
	 
	 @param	int tid			index of reference in header of SAM/BAM file
	 @param int pos			position in reference
	 @param	vector<int>		a vector of phred scores.  
	 @return	Pileup		a constructed pileup object, but it doesn't contain reads yet.
	 */
	Pileup(int tid, int pos, vector<int>& phreds ) : _tid(tid), _pos(pos),_num_reads(0),_debug(false), all_covered(false) {
		q_scores = phreds;
		q_cov = std::tr1::unordered_map<int, coord_t>(phreds.size());
		for (vector<int>::iterator i = q_scores.begin(); i != q_scores.end(); ++i) {
			
			q_cov[ *i ] = 0;
		}
				
	}
	
	/**
	 Convenience constructor if read already to be placed into the Pileup
	 
	 @param	int tid				index of reference in header of SAM/BAM file
	 @param int pos				position in reference
	 @param	vector<int> phreds	a vector of phred scores.  
	 @param BAMUtils* _p		a pointer to a BAMUtils object that covers this read
	 @return	Pileup			a constructed pileup object, but it doesn't contain reads yet.	 
	 */
	Pileup(int tid, int pos, vector<int>& phreds, BAMUtils* _p ) : _tid(tid), _pos(pos),_num_reads(0),_debug(false), all_covered(false) {
		_debug = false;
		q_scores = phreds;
		q_cov = std::tr1::unordered_map<int, coord_t>(phreds.size());
		for (vector<int>::iterator i = q_scores.begin(); i != q_scores.end(); ++i) {
			
			q_cov[ *i ] = 0;
		}
		insert_util(_p);
		
	}
	
	
	
	/**
	 A copy constructor
	 @param	Pileup const& other		an existing Pileup object
	 @return	Pileup				a copy of other
	 */
	Pileup(Pileup const& other) : _tid(other._tid), _pos(other._pos), _num_reads(other._num_reads), 
	q_cov(other.q_cov), q_scores(other.q_scores), _utils(other._utils),_debug(other._debug),all_covered(other.all_covered)
	{
	}
	
	
	/**
	 Overloaded operator= which calls the copy constructor
	 @param	Pileup that				an existing Pileup object
	 @return	Pileup				a copy of that

	 */
	Pileup& operator=(Pileup that) {
		 swap(*this, that);
		return *this;
	}

	
	/**
	 Inserts a BAMUtils object into this position.  This function will check to make sure the read covers this position.
	 If the read doesn't cover the position, the input pointer is disregarded
	 
	 @param	BAMUtils* _p			pointer to a BAMUtils object that you think covers this position
	 */
	void insert_util(BAMUtils* _p) {
		_num_reads++;
		
		if (_p->get_bamread().get_tid() == get_tid()) {
			
		
			for (vector<int>::size_type i = 0; i < q_scores.size(); i++) {
				if (q_cov[ q_scores[i] ] != 0) {
					all_covered = true;
					continue;
				}
				else {

					if(_p->is_pos_covered(_pos, q_scores[i])) {

						q_cov[ q_scores[i] ]++;
						
					} else {
						all_covered = false;
					}

					
				}
			}
		}
		
	}
	
	/**
	 Returns the index of the reference in the SAM/BAM header file
	 
	 @return	int		index of reference
	 */
	int get_tid() const { return _tid; }
	/**
	 Returns position in reference that this Pileup corresopnds to
	 
	 @return	long	position in genome
	 */
	coord_t get_pos() const { return _pos; }
	/**
	 Returns the number of reads that overlap this position (not necessarily cover it).
	 
	 For instance, if this position in the sample is a deletion to the reference, and all the reads 
	 that overlap this position observe that deletion the position will technically not have any coverage.
	 
	 @return	long	number of reads
	 */
	long get_num_reads() const { return _num_reads; }
	/**
	 A convenience method that will simply return whether or not there is SOMETHING covering this position.
	 This can save you computation if there is a _lot_ of coverage at a position.  
	 
	 @return	bool	true if position is covered
	 */
	bool is_covered() const { return all_covered; }
	/**
	 Returns the level of Phred coverage at this position.  So, if we have 10 Q47 reads that map here
	 and this position is within the Q47 length of all 10 reads, the return value will be 10.
	 
	 @param	int phred	Phred score we're interested in
	 @return	int		level of coverage
	 */
	int get_phred_cov(int phred) {
	
		if (q_cov.count(phred) > 0) {
			return q_cov[phred];
		} else {
			return 0;
		}

	}
	
	
			
	~Pileup() {
		/*_utils.clear();
		q_cov.clear();
		q_scores.clear();*/
		
	}
	
private:
	void swap(Pileup& first, Pileup& second) {
		
		using std::swap; //enables ADL
		swap(first._tid, second._tid);
		swap(first._pos, second._pos);
		swap(first._num_reads, second._num_reads);
		swap(first.q_cov, second.q_cov);
		swap(first.q_scores, second.q_scores);
		swap(first._utils, second._utils);
		swap(first._debug, second._debug);
		swap(first.all_covered, second.all_covered);
		
		
	}
	
		
	int			_tid;
	coord_t		_pos;
	long		_num_reads;
	std::tr1::unordered_map<int, coord_t> q_cov;
	std::vector<int> q_scores;
	std::list<BAMUtils* > _utils;

	bool		_debug;
	bool		all_covered; // this is a speed thing.  not required
	
};



/**
 This class attempts to encapsulate a Genomic window.  Given a start position, and a stop position inside 1 reference sequence
 BAMUtils objects can be inserted.  Upon insertion of reads, assuming they're in the window, the Factory becomes query-able.  
 
 Best practices for use of this factory are to insert all BAMUtils from the window BEFORE trying to use other member functions.
 
 This object can be very memory intensive, and is compute intensive.  It is advised that you keep the windows small if you suspect the 
 possibility of extreme levels of coverage (in excess of 100x).  The advised window length is 10 kilobases.  
 
 
 The start and stop positions of the window are inclusive.  
 
 The start and stop positions are 1-based.
 */

class PileupFactory {
	

public:
	typedef	std::tr1::unordered_map<int, Pileup>	pileup_container_t; /**< an STL container that implements some key,value */

	/**
	 Default constructor.  A necessary evil in order to use these objects in STL containers
	 */
	PileupFactory():_debug(false),_tid(-1), _start_pos(-1), _end_pos(-1) {}
	
	/**
	 Standard constructor.  This constructor will setup a default list of phred scores to observe coverage at:  7,10,17,20,47
	 
	 @param	int tid				index of reference from SAM/BAM header
	 @param	int start_pos		start position of window
	 @param	int	end_pos			end position of window
	 @return	PileupFactory	a constructed PileupFactory that's ready for BAMUtils objects
	 */
	PileupFactory(int tid, int start_pos, int end_pos) :_debug(false), _tid(tid),_start_pos(start_pos), _end_pos(end_pos) { 
		
		_phreds.push_back(7);
		_phreds.push_back(10);
		_phreds.push_back(17);
		_phreds.push_back(20);
		_phreds.push_back(47);
		
	}
	/**
	 If using custom phred scores, you require this constructor.  
	 
	 @param	int tid				index of reference from SAM/BAM header
	 @param	int start_pos		start position of window
	 @param	int	end_pos			end position of window
	 @param vector<int> phreds	list of phred scores to use
	 @return	PileupFactory	a constructed PileupFactory that's ready for BAMUtils objects
	 */
	
	PileupFactory(int tid, int start_pos, int end_pos, vector<int>& phreds) :_debug(false), _tid(tid),_start_pos(start_pos), _end_pos(end_pos), _phreds(phreds) { 
		
		
	}
	
	
	/**
	 I initially designed this constructor to be used in the case of a region that overlaps a previous genomic window.
	 The reads that overlapped would be passed in at construction to make I/O code in AlignStats much simpler.  
	 
	 However, if you already have a list of reads that are within the window for this new Factory you can simply pass them in the constructor here,
	 and the returned PileupFactory will be immediately ready to use.
	 
	 If the input BAMRead vector is large, this constructor can take a large amount of time to execute.  BAMUtils objects are created for each BAMRead, and then
	 inserted into an internal data structure containing a Pileup for each position in the window.
	 
	 @param	int tid							index of reference from SAM/BAM header
	 @param	int start_pos					start position of window
	 @param	int	end_pos						end position of window
	 @param vector<int> phreds				list of phred scores to use
	 @param	vector<BAMRead> the_overlaps	A vector of BAMRead(s) ready to be inserted
	 @return	PileupFactory				a constructed PileupFactory that's ready for BAMUtils objects
	 
	 */
	PileupFactory(int tid, int start_pos, int end_pos, vector<int>& phreds, std::vector<BAMRead>& the_overlaps) :_debug(false), _tid(tid),_start_pos(start_pos), _end_pos(end_pos), _phreds(phreds) { 
		_debug = false; 
		

		set_overlap_reads(the_overlaps);
		
	}
	
	
	/**
	 Copy constructor
	 
	 @param	PileupFactory const& other		Factory to copy
	 @return	PileupFactory				a copy of other
	 */
	PileupFactory(PileupFactory const& other):
	_debug(other._debug), _tid(other._tid), _start_pos(other._start_pos), _end_pos(other._end_pos), _phreds(other._phreds),
	positional_cov(other.positional_cov), overlap_reads(other.overlap_reads), overlap_utils(other.overlap_utils)
	
	{
			
		//cerr << "[PileupFactory] copy ctor" << endl;
	}
	
	
	/**
	 Overloaded operator= that calls the copy constructor
	 */
	PileupFactory& operator=(PileupFactory that) {
		swap(*this, that);
		return *this;
		
	}
	
	/**
	 If performance is on your mind, this sets up the internal data structure 
	 */
	void init_factory() {
		if (((_end_pos - _start_pos) > 1) && ((_end_pos - _start_pos) < 1000000)) {//limit to reduce mem on low coverage runs
			positional_cov.rehash((_end_pos - _start_pos));
		} 

		
	}
		
		
	/**
	 Inserts a BAMUtils object into some Pileup(s).  The function determines which Pileup(s) are effected,
	 finds them, and adds this BAMUtils to it.  
	 
	 @param	BAMUtils& util		a reference to an existing BAMUtils object
	 */
	void insert_util(BAMUtils& util) {
		
			
		BAMRead& align = util.get_bamread();
		//simple attempt to protect against 
		//inserting reads that aren't in this window
		if (align.get_tid() == get_tid()) {
			
			BAMUtils* _ptr = &util;
			int read_start = align.get_pos();
			int calend = align.calend();
			
			int loop_start = 0;
			
			if (read_start >= _start_pos) {
				loop_start = read_start;
			} else {
				loop_start = _start_pos;
			}
			
			
			int loop_end = 0;
			if (calend > _end_pos) {
				loop_end = _end_pos;
			} else {
				loop_end = calend;
			}
			
			for (int j = loop_start; j <= loop_end; j++) {
				pileup_container_t::iterator uom_itr = positional_cov.find(j);
				if (uom_itr != positional_cov.end()) {
					if (uom_itr->second.is_covered()) {
						continue;
					} else {
						uom_itr->second.insert_util(_ptr);
					}
					
				} else {
					positional_cov[j] = Pileup(_tid,j, _phreds, _ptr);
				}
				
				
				
				
				
			}
			
		} 		
	}
	
	/**
	 Returns a Pileup given a position in the genome.  If position is outside of the genomic window a default
	 constructed Pileup is returned.  
	 
	 If the position has no coverage, a default constructed Pileup object is returned
	 
	 @param	int pos_in_genome		1-based position in genome
	 @return	Pileup				a Pileup object for pos_in_genome
	 */
	Pileup get_pileup(int pos_in_genome) {
		//
		
		if (positional_cov.count(pos_in_genome) > 0) {
			return positional_cov[pos_in_genome];
		} else {
			return Pileup();
		}

		

		
	}
	
	/**
	 A convenience method so you can avoid handling Pileup objects all together if you want.
	 Returns true if the position is covered by phred_score reads.  
	 
	 @param		int pos_in_genome		1-based position in genome to inspect
	 @param		int phred_score			Phred score you're interested in
	 @return	bool					true if position is covered by reads in the factory
	 */
	bool is_pos_covered(int pos_in_genome, int phred_score) {
		
		if (_debug) cerr << "[PileupFactory - is_pos_covered()] positional_cov.size(): " << positional_cov.size() << endl;
			
			if (pos_in_genome >= _start_pos && pos_in_genome <= _end_pos) {
				//int _index = pos_in_genome - _start_pos;
				if (positional_cov.count(pos_in_genome) > 0) {
					if (positional_cov[pos_in_genome].get_phred_cov( phred_score ) > 0) {
						return true;
					} else {
						return false;
					}
				}
				return false;

			} else {
				return false;
			}

		
	}
	
	/**
	 the PileupFactory::pileup_iterator class abstracts all the positional information from the user of the class
	 It also removes the possibility of dealing with positions that have no reads overlapping them.  This will iterate only
	 over full Pileup objects (it's possible that the Pileup has no coverage however).
	 
	 Use this for maximum efficiency when looking at the entire window.
	 
	 WARNING!!! The iterator is not guaranteed to iterate the contents of the Factory in order.
	 
	 //example code assuming properly constructed object.  
	 //Assume an existing factory named pileups
	 
	 for (PileupFactory::pileup_iterator itr = pileups.get_pileup_iterator(); itr.good(); itr.next()) {
	 Pileup& p = itr.get();
		cerr <<delim<< "Ref id:"<<delim<< p.get_tid() <<delim<< "position in genome:"<<delim<< p.get_pos() <<delim<<"num reads:"<<delim<< p.get_num_reads() << endl;
	 }
	 */
	class pileup_iterator {
		
	public:	
		/**
		 Returns status of iterator
		 
		 @return	bool	true if more elements in factory
		 */
		bool good() { 
			return (itr != itr_end); 
		}
				
		/**
		 Advances iterator to next Pileup
		 */
		void next() {
			if (itr != itr_end) {
				++itr;
			}
			
		}
		/**
		 Returns a reference to the current Pileup pointed to by the iterator
		 
		 @return	Pileup&		reference to existing Pileup object
		 */
		Pileup& get() {
			return itr->second;
			
		}
		/**
		 Returns genomic 1-based position that iterator is at currently.  
		 */
		int		get_position() {
			if (good()) {
				return itr->first;
			}
		}
		
		
	private:
		friend class PileupFactory;
		pileup_iterator(pileup_container_t::iterator container_start, pileup_container_t::iterator container_end)
		: itr(container_start)
		, itr_start(container_start)
		, itr_end(container_end)
		
		{  };
		
		pileup_container_t::iterator itr;
		pileup_container_t::iterator itr_start;
		pileup_container_t::iterator itr_end;
	};
	
	
	/**
	 Returns an iterator for the Pileups inside the factory.
	 
	 @return	PileupFactory::pileup_iterator
	 */
	pileup_iterator get_pileup_iterator() {
		return pileup_iterator(positional_cov.begin(), positional_cov.end());
	}
	
	/**
	 Allows you to input a vector of BAMRead's.  Warning:  do not do this if you supplied this vector during 
	 construction.  It will overwrite an internal class member.
	 
	 If this function is used, you need to call:
	 
		PileupFactory::handle_overlaps(std::string& q_score_str, coord_t start_slop);
	 
	 The reason that handle_overlaps isn't called is because it can be computationally expensive, and this allows the user
	 of the class to copy the data into the Factory, but not necessarily do the heavy computation aspect of it right away.
	 
	 @param	std::vector<BAMRead>& the_overlaps		reads that mapped to this genomic region
	 */
	void set_overlap_reads(std::vector<BAMRead>& the_overlaps) {

		overlap_reads = the_overlaps;//need to copy instead of swap.  
									 //don't want to assume the caller is done with these reads

	}
	
	/**
	 This function is used ONLY AFTER you have just called:
	
		PileupFactory::set_overlap_reads(std::vector<BAMRead>& the_overlaps) 
	 
	 This is computationally expensive depending on the genomic window size.  It automatically inserts all the reads passed in
	 PileupFactory::set_overlap_reads(std::vector<BAMRead>& the_overlaps) into the internal data structure, 
	 and resulting Pileup objects.
	 
	 @param	std::string	q_score_str		a string representation of the phred scores
	 @param	coord_t	start_slop			number of bases at the start of the read to ignore for error calculations)
	 */
	void handle_overlaps(std::string& q_score_str, coord_t start_slop) {

		for (std::vector<BAMRead>::iterator i = overlap_reads.begin(); i != overlap_reads.end(); ++i) {
			if (i->get_tid() == get_tid()) {
				overlap_utils.push_back(BAMUtils(*i, q_score_str, 0));
				insert_util(overlap_utils.back());
			} 

			
		}
		
	}
	

	/**
	 Clears internal resources
	 */
	void release_resources() {
		//utils.clear();
		overlap_reads.clear();
		overlap_utils.clear();
		//pileups.clear();
		positional_cov.clear();
		
		
		
	}
	
	/**
	 Returns start of genomic window defined in construction
	 @return	int		start position of genomic window
	 */
	int get_start() const { return _start_pos; } 
	/**
	 Returns end of genomic window defined in construction
	 
	 @return	int		end position of genomic window
	 */
	int get_end()	const { return _end_pos; }
	
	/**
	 Returns reference id from header of SAM/BAM file
	 
	 @return	type:int
	 */
	int get_tid()	const { return _tid; }
	
	~PileupFactory() {
		release_resources();		
		
		
	}
	
	
	
private:
		
	void swap(PileupFactory& first, PileupFactory& second) {
		//if (_debug)cerr << "[PileupFactory] Swap" << endl;
		using std::swap; //enables ADL
		swap(first._debug, second._debug);
		swap(first._tid, second._tid);
		swap(first._start_pos, second._start_pos);
		swap(first._end_pos, second._end_pos);
		swap(first._phreds, second._phreds);
		swap(first.positional_cov, second.positional_cov);
		swap(first.overlap_reads, second.overlap_reads);
		swap(first.overlap_utils, second.overlap_utils);
		
	}
	
	
	bool						_debug;
	int							_tid;
	int							_start_pos;
	int							_end_pos;
	std::vector<int>							_phreds;

	pileup_container_t			positional_cov;
	
	std::vector<BAMRead>							overlap_reads;
	std::vector<BAMUtils>							overlap_utils;

	
};   
#endif //PILEUP_H

