/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef ALIGNSTATS_H
#define ALIGNSTATS_H

/*
 *  alignStats.h
 *  SamUtils
 *
 *  Created by Michael Lyons on 1/31/11.
 *
 */


#include <limits.h>
#include <pthread.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <list>
#include <sstream>
#include <stdlib.h>
#include <tr1/memory>
#include <tr1/unordered_map>
#include <memory>
#include "types/samutils_types.h"
#include "types/BAMRead.h"
#include "types/Pileup.h"
#include "Utils.h"
#include "BAMReader.h"
#include "BAMUtils.h"
#include "BlockingQueue.h"
#include "CircularQueue.h"

using namespace samutils_types;

/**
 *class AlignStats
 *This class really has 1 use, and that is just to 
 *encapsulate portions of this (what used to be alignStats.pl) program to make it easier for a multi-threading implementation.
 */
class AlignStats {

public:
	/**
	 *A structure to keep the command line options for an AlignStats instance
	 *This structure is used to keep the constructors very simple.  
	 */
	struct options {
		options()
		: bam_file("") 
		, bam_index("")
		, out_file("Default")
        , output_dir("")
		, genome_info("")
		, flow_order("")
		, flow_err_file("alignFlowErr.txt")
		, score_flows(false)
		, read_to_keep_file("")
		, read_to_reject_file("")
		, q_scores("7,10,17,20,47")
		, start_slop(0)
		, sam_parsed_flag(0)
		, total_reads(0)
		, read_limit(LONG_MAX)
		, sample_size(0)
		, help_flag(false)
		, skip_cov_flag(false)
		, debug_flag(false)
		, num_threads(4)
		, buffer_size(10000)
		, iupac_flag(true)
		, keep_iupac(false)
		, max_coverage(20000)
		, align_summary_filter_len(20)
		, align_summary_min_len(50)
		, align_summary_max_len(400)
		, align_summary_len_step(50)
		, err_table_txt_file("alignStats_err.txt")
		, err_table_json_file("alignStats_err.json")
		, err_table_filter_len(50) 
		, err_table_filter_accuracy(0.9)
		, err_table_min_len(50)
		, err_table_max_len(400)
		, err_table_len_step(50)
		, err_table_max_errors(3)
		, stdin_sam_flag(false)
		, stdin_bam_flag(false)
		, list_of_files("")
		, truncate_soft_clipped(true)
		, three_prime_clip(0)
		, round_phred_scores(false)
        , five_prime_justify(true)
		, merged_region_file("merge_region.txt")
		{}
		string			bam_file; /**< a string representing the absolute path to a bam file, or just a file name*/
		string			bam_index; /**< a string representing the absolute path to a bam file index*/
		string			out_file;  /**< a string representing the absolute path to the prefix of output file names [Default: "Default"] */
        string          output_dir; /**< a string representing the path to an output directory, will try and make this directory if it doesn't exist already */
		string			genome_info; /**< a string representing the absolute path to a genome.info.txt file (internal Ion Torrent file format [Default: ""] */
		string			flow_order; /**< Sequencing flow order [Default: ""] */
		string			flow_err_file; /**< output file for flow error info */
		bool			score_flows; /**< Sequencing flow order [Default: ""] */
		string			read_to_keep_file; /**< File with list of read IDs to use [Default: ""] */
		string			read_to_reject_file; /**< File with list of read IDs to ignore [Default: ""] */
		string			q_scores;/**< a comma seperate string of integers reprsenting the phred values to use for the alignment.summary output */
		long			start_slop;/**< an integer representing the first N bases to be ignored when considering errors in an alignment */
		int				sam_parsed_flag;/**< a flag to signal whether or not to create the sam.parsed file (can be very large file) */
		long			total_reads;/**< a long to represent the total number of reads being input.  this is used for sampling purposes in order to extrapolate stats */
		long			read_limit;/**< a long to represent the total number of alignments to read - useful to limit when debugging */
		long			sample_size;/**< a long representing the number of reads to sample */
		bool			help_flag;/**< a bool to print the help message to stdout.  exits the program if true */
		bool			skip_cov_flag;/**< a bool that allows the user to skip coverage calculations.  this speeds up the program considerably, and reduces memory use */
		bool			debug_flag;/**< a bool that will print debug messages to stderr.  very verbose, be careful */
		int				num_threads;/**< an integer representing the number of threads the program will try to use */
		int				buffer_size; /**< an integer representing the number of reads that is assigned to each thread for a work load.  If coverage is being calculated the buffer size is 
									  *  only approximately this size.  It can be slightly larger, or smaller 
									  */		
		bool			iupac_flag;/**< a bool flag that will check the MD tag for IUPAC codes and see if they're actually errors to the reference.  
									*	use this if your reference contains IUPAC codes instead of bases in some positions
									*/
		bool			keep_iupac;/**< a bool flag that will put iupac codes into the sam.parsed file.  only has an effect if 
									* iupac_flag == true
									*/
		int				max_coverage;/**< an int representing the maximum coverage that is allowed for 1 position in the genome
									  *   if a particular position has > max_coverage reads aligning to it, only the first
									  *	  N reads are used, where N = max_coverage. [Default: 20000]  
									  */
		long				align_summary_filter_len; /**< the minimum length a read must be to be included in AQ# results output [Default: 20] */
		int				align_summary_min_len;/**< an int representing the minimum length for a read to be considered in the alignment summary table */
		int				align_summary_max_len;/**< an int representing the maximum length for a read to be considered in the alignment summary table */
		int				align_summary_len_step;/**< an int representing the interval of read lengths between the min and max lengths */

		string				err_table_txt_file; /**< a string representing the absolute path for the alignment error table in txt format*/
		string				err_table_json_file; /**< a string representing the absolute path for the alignment error table in json format*/
		int				err_table_filter_len;/**< an int represting the minimum lenght a read must be in order to be considered for the alignment error */
		double				err_table_filter_accuracy; /**< a double representing the % of the read that must be error free in order to be considered for the alignment 
														*   error table.  This filter value doesn't effect the alignment.summary or sam.parsed output.
														*   Filtering formula:  if (((err_table_filter_len - cummaltive_errors_in_read) / err_table_filter_len) > err_table_filter_accuracy)
														*/
		int				err_table_min_len;/**< an int representing the minimum length for a read to be considered in the alignment error table */
		int				err_table_max_len;/**< an int representing the maximum length for a read to be considered in the alignment error table */
		int				err_table_len_step;/**< an int representing the interval of read lengths between the min and max lengths
												*	 in the alignment summary table
												*/
		int				err_table_max_errors;/**< an int representing the maximum number of errors to be output to the table.  one column is produced for each +1 value 
												  *	  until the value stored in err_table_max_errors.  For instance, if this is 3 the columns output would be:
												  *		err1	err2	err3+
												  *	  The final column always has a trailing + character.  This indicates that reads in this column have at least that number of errors
												  */
		
		//stdin options
		bool stdin_sam_flag;/**< a bool representing whether or not the input is from stdin, and raw text input */
		bool stdin_bam_flag;/**< a bool representing whether or not the input is from stdin, and binary input */
		
		//batch processing
		std::string list_of_files;/**< a string representing the absolute path to a file  */
		
		bool truncate_soft_clipped;/**< a bool to represent whether or not to ignore soft clipped bases in the alignment.  This can really effect reads which map to the 
									* reverse strand and have a # of leading soft clipped bases in the alignment [Default: true]
									*/
		int three_prime_clip; /**< an integer value representing the # of 3 prime soft clipped bases to ignore when calculating errors */

		bool round_phred_scores; /**< boolean value to allow rounding in the phred score calculations to handle the colloquial definition
									  of a "100 Q17" which means 2 errors in 100 bases.
									*/
        bool five_prime_justify; /**< boolean whether to follow indel justification done by mapper or force 5prime */
        string merged_region_file;
	};
	//for region-specific errors
	struct read_region{
		int region_start_base;
		int region_stop_base;		
	}; 
	/**
	 * Simple constructor which only takes the AlignStats::options struct as a parameter.  
	 * This unifies the initialization process, and constructors into 1 thing.  
	 */
	AlignStats(options& settings);

	/**
	 * Runs the program
	 * This function calculates statistics given the options passed via the constructor.  
	 * If a new AlignStats::options is set, a new call to go() will re-run the program
	 */
	void go();
	
	/**
	 * Functions for specifying up front read IDs that should be retained/ignored.
	 * If read_to_keep is set then only reads with names matching the specified names
	 * are used.  If read_to_reject is set then any reads matching one listed for 
	 * rejection are excluded.  If a read is on both lists it will be excluded.
	 */
	void set_read_to_keep(std::string readKeepFile);
	void set_read_to_reject(std::string readRejectFile);

	/**
	 * Returns a long which is the AlignStats::options.total_reads field
	 * This represents the total number of reads given in the constructor, or if that wasn't set
	 * it represents the total number of observed reads in the SAM/BAM file
	 */
	long get_total_reads();
	
	
		
private:
		
	
	
	
	//data structs	
	typedef	std::vector<BAMRead>					read_list;
	typedef std::vector<BAMUtils>					util_list;
	typedef std::pair<read_list, PileupFactory>		worker_data;
	typedef std::vector<CircularQueue<worker_data, 20 > > worker_queue;
	worker_queue vector_queue; /**< the producer/consumer queue.
																 *   each thread produces it's own index into this vector
																 *	 the size of the vector equal to AlignStats::options.num_threads
																 */
	
	//optional read filters
	std::map<string,bool> read_to_keep;
	std::map<string,bool> read_to_reject;
	

		
			
	//conditionals	
	NativeMutex					sam_parsed_mutex; /**< a mutex to handle synchronized writes to the sam.parsed file*/
	NativeMutex					worker_mutex; /**< a mutex used throughout consume_read_queue() function.  a general mutex for the
											   *   worker threads, basically
											   */
	
	bool						queue_ready; /**< bool status for whether or not reads were placed into the data structure worker threads pull data from */
	bool						you_should_run;/**< bool status for whether or not the worker threads should continue executing */
	
	size_t my_worker_num; /**< this number is the worker thread id.  it is used to index into the concurrent data structure holding 
						   *	pending worker data
						   */
	long total_reads_to_read; /**< a long representing the total number of alignments to inspect, useful to limit when debugging */
	long total_reads_cached;/**< a long representing the total number of alignments read */
	long total_reads_processed;/**< a long representing the total number of alignments analyzed */
	//i/o members
	
	options opt;/**< an instance of AlignStats::options which holds the configuration for this particular call to AlignStats::go() */
	
	std::ofstream	sam_parsed; /**< output file stream for the sam.parsed file */
	std::ofstream	error_table_file; /**< output file stream for the alignment summary error table */
	BAMReader		reader;/**< a BAMReader instance.  Used to read the SAM/BAM file from AlignStats::options.bam_file */
	bool			is_sorted;/**< a bool that is true if the SAM/BAM is sorted by genome position.  the read_bam() function
							   *	determines this empirically
							   */

	int				reads_in_queue; /**< reads currently in a data structure that is to be passed into the concurrent data structure 
									 *	holding pending worker data (vector_queue)
									 */
	
	

	std::vector<coord_t>				q_sum;/**< represents sum of phred bases. 
											   * each index in the vector corresponds to a different phred score
											   */
	std::vector<coord_t>				q_num;/**< represents the number of reads that pass the minimum length filter for each phred score 
											   * each index in the vector corresponds to a different phred score
											   */
	std::vector<coord_t>				q_longest;/**< longest read for a specific phred score
												   * each index in the vector corresponds to a different phred score
												   */
	std::vector<std::string>			q_scores; /**< phred scores in string form.  
												   * each index in the vector corresponds to a different phred score
												   */
	std::vector<std::vector<coord_t> >	q_histo; /**< a two dimensional vector representing a read length histogram for each phred score
												  *	q_histo[k] holds a vector of size max_len for a particular phred score
												  * each index in the vector corresponds to a different phred score
												  */
	std::vector<coord_t>				coverage;/**< a vector that holds the amount of coverage for a specific phred score 
												  *	each index in the vector corresponds to a different phred score
												  */
	std::vector<int>					phreds; /**< a vector of the phred scores in integer form
												 * each index in the vector corresponds to a different phred score
												 */
	
	//error table data members
	std::vector<int>					alignment_summary_read_lens; /**< read lengths for alignment summary */
	std::vector<int>					error_table_read_lens; /**< read lengths for error table */
	std::vector<long>					error_table_read_totals;/**< total # of reads for each read length in the error_table_read_lens vector */
	std::vector<long>					error_table_errors_at_len; /**< #errors from every read summed for a specific position in the read */
	typedef	std::tr1::unordered_map<int, int>		error_to_length_map; /**< key: error value: total */
	typedef std::tr1::unordered_map<int, error_to_length_map > error_table_type; /**< key: length, value: error_to_length_map */
	typedef std::vector<long>			read_totals_t;  /**< a type representing the totals for each read length we care about in ascending order by length 
														 *  so each index in the vector represents an incremental read length, ie read_totals_t[0] might 
														 * correspond to total # of 25mers
														 */
	typedef std::tr1::unordered_map<int, read_totals_t >	phred_to_totals_t; /**< a map with key: phred #, value: read_totals_t */
	typedef	std::tr1::unordered_map<int, long> length_to_total_map_t; /**< key: length, value: total */
	
	error_table_type		error_table; /**< data structure representing the alignment summary error table.  
										  * key: an int representing the length of the read
										  * value:  an unordered_map of error_to_length_map (typedef is right above this)
										  */
	
	long				mapped_bases;
	long				mapped_reads;
	phred_to_totals_t		alignment_summary_map; /**< data structure representing total reads for each phred score with certain lengths
													* key: phred score
													* value: read_totals_t 
													*/
	
	length_to_total_map_t		unaligned_read_totals;  /**< #unaligned reads at each length */
	length_to_total_map_t		clipped_read_totals;    /**< #clipped reads at each length */
	length_to_total_map_t		filtered_read_totals;   /**< #reads filtered out at each length */
	length_to_total_map_t		total_error_to_length;  /**< total #errors up to and including read position */
	length_to_total_map_t		per_position_mismatch;  /**< total #mismatches at read position */
	length_to_total_map_t		per_position_insertion; /**< total #insertions at read position */
	length_to_total_map_t		per_position_deletion;  /**< total #deletions at read position */
	
	std::string genome_name; /**< output name of genome from AlignStats::options.genome_info file */
	std::string genome_version;/**< output name of genome version from AlignStats::options.genome_info file */
	std::string index_version;/**< output name of genome index version from AlignStats::options.genome_info file */
	long genome_length;/**< length of genome from AlignStats::options.genome_info file */

	std::string			flow_order;	/**< Sequencing flow order */
	bool				score_flows;	/**< true if flow-based error accounting should be performed */
	std::vector<uint32_t>		n_flow_aligned;	/**< vector with i'th element showing number of reads that aligned up to flow i */
	std::vector<uint32_t>		flow_err;	/**< vector with i'th element showing number of erroneous HP calls at flow i */
	std::vector<uint32_t>		flow_err_bases;	/**< vector with i'th element showing number of erroneous base calls at flow i */
	std::string			flow_err_file;	/**< path for output flow error summary table */

	PileupFactory pileups;/**< a pileupfactory, see types/Pileup.h for documentation */
	
	//for region-specific errors: start
	std::tr1::unordered_map<string, read_region> _read_regionmap; 
	bool _region_positions_defined; 

	int	_total_region_homo_err;	
	int _total_region_mm_err;
	int	_total_region_indel_err;
	int	_total_region_ins_err;
	int	_total_region_del_err;
	
	std::ofstream	_region_error_outputfile;
	std::ofstream	_region_error_positions_outputfile;
	NativeMutex	 _region_error_outputfile_mutex; /**< a mutex to handle synchronized writes to the region error file*/
	//*read_regionmap carries the result map
	bool build_read_regionmap(string regionPostionFile, std::tr1::unordered_map<string, read_region> *read_regionmap);
	//for region-specific errors: end
	
	/**
	 * a simple function to initialize individual class members, but not data structures
	 * called inside the constructor 
	 */
	void init();
	/**
	 * initializes all member data structures, called inside constructor
	 */
	void init_data_structs();
	/**
	 * worker threads are basically the consume_read_queue() function.  This is the consumer end of the Producer/Consumer parallel paradigm
	 */
	void consume_read_queue();
	/**
	 *function to parse the file specified by AlignStats::options.genome_info
	 *if that file isn't set this function isn't called, and/or has no effect
	 */
	void set_genome_info();
	/**
	 *if the input is sampled, this will extrapolate the stats calculated to the # provided by AlignStats::options.total_reads
	 */
	void extrapolate();
	/**
	 * writes the header of the sam.parsed file (one line, tab delimited naming all the columns in the file)
	 */
	void write_sam_parsed_header();
	/**
	 * writes the alignment.summary file
	 */
	void write_alignment_summary();
	/**
	 * writes the alignment summary error table file in text format
	 */
	void write_error_table_txt(string outFile);
	/**
	 * writes the alignment summary error table file in json format
	 */
	void write_error_table_json(string outFile);

	/**
	 * writes the alignment flow error summary file
	 */
	void write_flow_error_table();

	/** 
	 * read_bam is the Producer portion of the Producer/Consumer parallel paradigm.  This function does quite a bit
	 * It detects if the BAM/SAM is sorted
	 * It also determines the region of overlap between buffer chunks during coverage.  It creates special lists for these regions that only the 
	 * PileupFactory recieves as input.  This actually duplicates BAMUtils creation for a small % of the input.
	 */
	bool read_bam(BAMReader::iterator& bam_itr, std::vector<pthread_t>& my_threads, int& active_threads);
	/**
	 * This iterates through the pileup_factory to determine the coverage %
	 */
	void sum_coverage(PileupFactory& pileup_factory, vector<coord_t>& my_cov, util_list& util_cache);

	/**
	 * a "thunk" type function to launch threads via pthread.h
	 */
	static void *run(void *arg);

	void add_pair_to_queue(worker_data& the_data, size_t& vector_queue_index);

	// Utility functions for parsing read names from file or stream
	void readNamesFromFile(std::map<string,bool> &nameMap, std::string inFile);
	bool readNextNameFromStream(std::string &readName, ifstream &inStream);

	void countAlignedFlows(std::vector<int> &max_flow, std::vector<uint32_t> &flow_by_position);
};

#endif // ALIGNSTATS_H

