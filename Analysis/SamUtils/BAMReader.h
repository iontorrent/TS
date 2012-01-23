/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BAMREADER_H
#define BAMREADER_H

// BAMReader can parse a BAM header, and traverse and parse
// alignments in a BAM file. See also sample calling code in
// example.cpp.

#include <stdint.h>
#include <string>
#include <iostream>
#include <algorithm>
#include "types/BAMRead.h"
#include "types/samutils_types.h"
#include "BAMUtils.h"
#include "Lock.h"
#include <tr1/memory>
#include <tr1/unordered_map>
#include <vector>
#include <queue>
#include <deque>
#include <list>
#include "sam.h"
#include "bam.h"


using namespace samutils_types;

/**
 BAMReader is a class used to read SAM/BAM files or SAM/BAM from stdin
 It's a light weight class that wraps the functionality in samtools
 */
class BAMReader {
public:
	// Create a BAMReader, optionally specifying bam file and index file:
	/**
	 Default constructor.  Initializes internal members.  
	 Use this constructor if you're reading from stdin, and then use
	 BAMReader::open(const std::string& mode_stdin, const std::string& type_of_stdin)
	 
	 @return	BAMReader		a BAMReader that has no file/io
	 */
	BAMReader();
	/**
	 A constructor which takes a string representing the path to a bam file, including the file name.
	 The constructor doesn't open the file.
	 
	 @return	BAMReader		a BAMReader that is ready to be opened
	 */
	BAMReader(const std::string& BAM_file);
	/**
	 A constructor which takes a string representing the path to a bam file, including the file name.
	 It also takes a string representing the path to an index file for this bam.  
	 
	 The constructor doesn't open the file.
	 @return	BAMReader		a BAMReader that is ready to be opened
	 */
	 BAMReader(const std::string& BAM_file, const std::string& Index_File);

	// Destructor closes all open files:
	virtual ~BAMReader();
	// Open a bam file or bam/index pair:
	/**
	 Tries to open file specificed in constructor.  If no file is specified this has no effect
	 */
	void open();
	/**
	 Tries to open file passed in the funciton call.  
	 
	 @param	const std::string& BAM_file		represents path of file to open
	 */
	void open(const std::string& BAM_file);
	/**
	 Tries to open an stdin stream.  
	 Valid arguments to	   mode_stdin: "-"
	 Valid arguments to type_of_stdin:  "r" or "rb".  Use "r" for sams, and "rb" for binary
	 Samtools will exist the program if the above isn't used correctly.
	 
	 @param	const std::string& mode_stdin			currently only '-' is valid input
	 @param const std::string& type_of_stdin		expects "r" or "rb"
	 */
	void open(const std::string& mode_stdin, const std::string& type_of_stdin);
	/**
	 Closes file.  If file is already closed, or not open, this has no effect
	 */
	void close();

	// Was the open successful?
	/**
	 Returns true if file is open.  False if file isn't open
	 
	 @return	bool		true if file is open
	 */
	bool is_open()   const;
	operator bool() const;

	/**
	 Returns the number of references specificed in the header of the SAM/BAM file
	 
	 @return	int		number of references in header
	 */
	int           num_refs() const;
	/**
	 Returns a vector of strings with the reference names from the header
	 
	 @return	strvec&		vector of references in header 
	 */
	const strvec& refs()    const;
	/**
	 Returns a vector of the reference lengths from the header
	 
	 @return	lenvec&		vector of reference lenghts from header
	 */
	const lenvec& lens()    const;

	/**
		A class to allow the user to iterate through the SAM/BAM file/stdin.  
		A file must already be opened by the reader for an iterator to be valid.
	 
		//example code
	 
		string bam_file = string(argv[1]);
		BAMReader reader(bam_file);
		for (BAMReader::iterator i = reader.get_iterator(); i.good(); i.next()) {		
			BAMRead read = i.get();
			cerr << read.to_string();	 
		}
	 
	 */
	class iterator {
	public:
		/**
			Returns the status of the iterator.  The iterator is good if there are still alignments 
			and the file handle is good.
		 
		 @return	bool		true if there are alignments
		 */
		inline bool good() const { 
			
			return is_good; 
		}

		/**
		 Move on to next alignment:
		 */
		void next();
		
		//get a BAMRead in it's entirety:
		/**
		 Returns a fully constructed BAMRead
		 
		 @return	BAMRead		a constructed BAMRead from the file
		 */
		inline BAMRead get() const {
			assert(itr_record);
			//bam1_t *cpy = bam_dup1(itr_record);
			return BAMRead(itr_record, itr_file->header);
		};
		
		/**
		 Returns the reference index of the current alignment pointed to by the iterator.  
		 A convenience function that wont make copies of c-structures
		 
		 @return		int		reference index from header
		 */
		inline int		get_tid() const {
			if (itr_record) {
				return itr_record->core.tid;
			} else {
				return -1;
			}


		}
		/**
		 Returns the length of the reference the read aligns to
		 A convenience function that wont make copies of c-structures

		@return		int			length of reference read aligns to
		 */
		inline int	get_tid_len(int tid) {
			
			return tid_lens[tid];
		}
		
		
	private:
		friend class BAMReader;
		// iterator ctors will go here.
		iterator();
		iterator(samfile_t* fp, lenvec& ref_lens);
		iterator(samfile_t* fp, int ref_index, const std::string& BAM_file, lenvec& ref_lens );
		iterator(samfile_t* fp, int ref_index, const std::string& BAM_file, coord_t begin, coord_t end, lenvec& ref_lens);
		
		//general priv funcs
		void						_init();
		void						_init(samfile_t* fp, int ref_index, const std::string& BAM_file);
		bool						_set_iter(int tid, int beg, int end);
		void						_index(const std::string& BAM_file);
	
		//sam/bam types
		bam1_t*		  itr_record;
		samfile_t*	  itr_file;
		bam_index_t*  itr_index;
		bam_iter_t	  bam_iter;
		lenvec		  tid_lens;
		
		//prims
		bool		  is_good;
		bool		  is_bam;
		
	};
	
	/**
	 An inefficient, but simple version of Pileup which wraps up samtools pileup C functions.  
	 This class will do the same thing as a command line samtools mpileup call.
	 
	 It's not efficient because it will return the same read more than once.
	 
	 The pileup_generator advances by genomic positon, and returns a
		std::pair<pileup_reads, int> pileup_data;
	 
	 Where pileup_reads is:
		std::list<BAMRead>   pileup_reads;
	 
	 */
	class pileup_generator {
		
	public:
		typedef std::list<BAMRead>   pileup_reads;
		typedef const bam_pileup1_t* pileup_ptr;
		typedef std::pair<pileup_reads, int> pileup_data;

		/**
		 Advance the generator to the next read
		 */
		void next() {
			//return a BAMRead from the pileup
			pileups.pop();
		}
		/**
		 Are there still positions in the pileup(s)?
		 */
		bool good() {
			
			return (pileups.size() > 0);
				
		}
		
		/**
		 Returns an std::pair< std::list<BAMRead>, int >
		 
		 The int is the position in the reference sequence this Pileup is
		 
		 @return std::pair< std::list<BAMRead>, int >		a tuple representing the pileup
		 */
		pileup_data get() {
			return pileups.front();
		}
		
		
		~pileup_generator() {
			bam_index_destroy(idx);
			bam_plbuf_destroy(buf);
		}
		
		
	private:
		friend class BAMReader;

		pileup_generator(samfile_t* the_file, std::string the_file_name, std::string the_region, const bam_header_t* header) { 
			generator_ptr = NULL;
			my_header = header;
			my_bam = the_file_name;
			my_region = the_region;
			tmp.in = the_file;
			tmp.me = this;
			if (tmp.in == 0) {
				std::cerr << "[pileup generator] file failed to open" << std::endl;
			}
			idx = bam_index_load(my_bam.c_str());
			
			bam_parse_region(tmp.in->header, my_region.c_str(), &_tid, &_begin, &_end);
			if (_tid < 0) {
				std::cerr << "[pileup generator] invalid region.  use format chr1:1-50" << std::endl;
			}
			tmp.beg = _begin;
			tmp.end = _end;
			std::cerr << "[pileup_generator ctor] the_region: " << the_region << " _tid: " << _tid << " beg: " << tmp.beg << " end: " << tmp.end << std::endl;
			buf = bam_plbuf_init(pileup_func, &tmp);
			bam_fetch(tmp.in->x.bam, idx, _tid, tmp.beg, tmp.end, buf, fetch_func);
			bam_plbuf_push(0, buf); //finalizes pileup  
			
		}
		struct tmpstruct_t{
			int beg, end;
			pileup_generator* me;
			samfile_t *in;
		};
		
		int _tid;
		int _begin;
		int _end;
		std::string my_bam;
		std::string my_region;
		tmpstruct_t tmp;
		
		
		//data structures
		std::queue<pileup_data> pileups;
		
		//samtools specific code inspired/ripped off from samtools-*/examples/calDepth.c
		const bam_header_t* my_header;
		bam_plbuf_t *buf;
		bam_index_t *idx;
		pileup_ptr  generator_ptr;
		
		
				
		// callback for bam_fetch()
		static int fetch_func(const bam1_t *b, void *data)
		{
			bam_plbuf_t *buf = (bam_plbuf_t*)data;
			bam_plbuf_push(b, buf);
			return 0;
		}
		// callback for bam_plbuf_init()
		static int pileup_func(uint32_t tid, uint32_t pos, int n, const bam_pileup1_t *pl, void *data)
		{
			tmpstruct_t *tmp = (tmpstruct_t*)data;
			
			if ((int)pos >= tmp->beg && (int)pos < tmp->end) {
				const bam_pileup1_t* p = pl;
				pileup_reads the_reads;
				for (int i = 1; i <= n; i++) {
					the_reads.push_back(BAMRead(bam_dup1(p->b), tmp->me->my_header));
					
					p = pl + i;
					
				}
						
				pileup_data the_data(the_reads, pos);
				tmp->me->pileups.push(the_data);
				
			}
						
			return 0;
		}
		
		
		
		
	};
	
	// Iterate over all alignemnts:
	/**
	 This returns an iterator for all alignments in the SAM/BAM 
	 
	 @return BAMReader::iterator	iterator to all alignments
	 */
	iterator get_iterator();

	// Iterate over all alignments to a single reference sequence.
	// ref_index is the index of the sequence in the vector of 
	// sequences returned by refs():
	/**
	 Returns an iterator to the specific reference of interest.
	 Will only iterate over those reads
	 
	 @param	int ref_index		index of reference from header
	 @return	BAMReader::iterator		iterator to selected reference
	 */
	iterator get_iterator(int ref_index);
	/**
	 Returns an iterator to the specific reference of interest.
	 Will only iterate over those reads
	 
	 @param		const std::string& ref_index		name of reference
	 @return	BAMReader::iterator					iterator to selected reference
	 */
	iterator get_iterator(const std::string& ref_seq_name);

	
	/**
	 Iterate over all alignments in an interval.
	 Intervals are specified C++-style, as 0-based, and half open:
	 
	 @param		int ref_index			index from header of reference
	 @param		long begin				beginning of region of interest
	 @param		long end				end of region of interest
	 @return	BAMReader::iterator		iterator to selected region

	 */
	iterator get_iterator(int ref_index, coord_t begin, coord_t end);
	/**
	 Iterate over all alignments in an interval.
	 Intervals are specified C++-style, as 0-based, and half open:
	 
	 @param		const std::string& ref_seq_name		name of reference from header
	 @param		long begin							beginning of region of interest
	 @param		long end							end of region of interest
	 @return	BAMReader::iterator					iterator to selected region
	 */
	iterator get_iterator(const std::string& ref_seq_name, coord_t begin, coord_t end);
	
	
	/**
	 Get a pileup generator for a specific region.  The region must follow the format of
	 samtools pileup, ie:  "chr3:1,000-2,000"
	 format is "ref:start-stop"
	 
	 @param		string region					"ref:start-stop"
	 @return	BAMReader::pileup_generator		a pileup generator for the region of interest
	 */
	pileup_generator get_generator(std::string region) {

		return pileup_generator(file_p, bam_file, region, bam_header);
	}
	
	/**
	 Returns a reference to the C-structure representing the header file.  
	 
	 @return	bam_header_t*		pointer to the header
	 */
	const bam_header_t* get_header() const {
			
		return bam_header;
		
	}

private:
	
	//sam/bam types
	samfile_t*		file_p;//file pointer
	bam_index_t*	bam_index;
	bam_header_t*	bam_header;
	
	//typedefs
	strvec			ref_list;
	lenvec			ref_lens;
	
	//STL/stdrd types
	std::string		bam_file;
	std::string		index_file;
	bool			file_open;
	
	// No copying:
	BAMReader(const BAMReader&);
	BAMReader& operator=(const BAMReader&);
	
	
	//private functions
	void Initialize();
	void _open();
	void _init();
	//void _init(samfile_t* fp, int ref_index, const std::string& BAM_file);
	int							_findRef(const std::string& ref_seq_name) const;

};

#endif // BAMREADER_H


