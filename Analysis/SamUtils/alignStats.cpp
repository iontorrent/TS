/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */


/*
 *  alignStats.cpp
 *  SamUtils
 *
 *  Created by Michael Lyons on 1/31/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */

#include <pthread.h>
#include <stddef.h>
#include "alignStats.h"
#include "Utils.h"
#include "IonErr.h"
#include "Lock.h"

using namespace std;


AlignStats::AlignStats(options& settings) : opt(settings) {
	
   
	init();
		
	
	init_data_structs();
	set_genome_info();
	
	//for region-specific errors: start
	_total_region_homo_err = 0;	
	_total_region_mm_err = 0;
	_total_region_indel_err = 0;
	_total_region_ins_err = 0;
	_total_region_del_err = 0;
	
	_region_positions_defined = false; 
	if(opt.merged_region_file!=""){
		_region_positions_defined = build_read_regionmap(opt.merged_region_file.c_str(), &_read_regionmap);
		if(_region_positions_defined){
			  //output errors within the region to file hard-coded file "regionErrors.txt" 
			  string region_error_file_name("region_errors.txt");
			  _region_error_outputfile.open(region_error_file_name.c_str());
	    	  _region_error_outputfile << "#read_id" << "\t"<< "HomoErr" << "\t" << "MmErr" << "\t" << "IndelErr" << "\t" << "InsErr" << "\t" << "DelErr" << "\t" << "RegionClipped" << endl;
		
	    	  string _region_error_positions_file_name("region_error_positions.txt");
			  _region_error_positions_outputfile.open(_region_error_positions_file_name.c_str());
			  _region_error_positions_outputfile << "#Read_ID" << "\t"<< "Number of Positions" << "\t" << "Position 1" << "\t" << "# of Errors at Position1" << "\t" << "..." << endl;
			    	 	  
	
		}
		else{
			cerr << "exit due to wrong region file" << endl;
			exit(1);	
		}			
	}
	//for region-specific errors: end
		
    if(opt.sam_parsed_flag) {
	  string sam_parsed_name(opt.out_file + ".sam.parsed");
	  if (isFile((char *)sam_parsed_name.c_str()))
		std::cerr << "[alignStats] WARNING: sam.parsed file already exists, overwriting: " << sam_parsed_name << endl;
	  sam_parsed.open(sam_parsed_name.c_str());
	  write_sam_parsed_header(); //this will write the header
    }
	
	
	
    if(opt.debug_flag) std::cerr << "[alignStats] constructed" << std::endl;
}

void AlignStats::init() {
	
	reads_in_queue = 0;
	genome_length = 0;
	my_worker_num = 0;
	genome_name = "None";
	genome_version = "None";
	index_version = "None";
	is_sorted = true;
	
		
}

void AlignStats::init_data_structs() {
	
	if (opt.stdin_sam_flag) {
		reader.open("-", "r");
	} else if (opt.stdin_bam_flag) {
		reader.open("-", "rb");
	} else {
		reader.open(opt.bam_file);
	}

    if (!reader.is_open()) {
		if (opt.stdin_sam_flag || opt.stdin_bam_flag) {
			std::cerr << "[alignStats] failed to use stdin as input" << std::endl;
		} else {
			std::cerr << "[alignStats] failed to open file: " << opt.bam_file << std::endl;

		}

		exit(1);
	}
	if (opt.read_to_keep_file != "")
		set_read_to_keep(opt.read_to_keep_file);
	if (opt.read_to_reject_file != "")
		set_read_to_reject(opt.read_to_reject_file);

	const int num_qs = count_char(opt.q_scores, ',') + 1;
	q_sum.assign(num_qs,0);
	q_num.assign(num_qs,0);
	q_longest.assign(num_qs,0);
	/*
	q_fif_num.assign(num_qs,0);
	q_hun_num.assign(num_qs,0);
	q_two_hun_num.assign(num_qs,0);*/
	coverage.assign(num_qs,0);
	split(opt.q_scores, ',', q_scores);
	for (int i = 0; i < num_qs; i++) {
		q_histo.assign(num_qs, vector<coord_t>(opt.align_summary_max_len*2, 0));
	}
	phreds.assign(q_scores.size(), 0);
	for(unsigned int j = 0; j < q_scores.size(); j++) {
		phreds[j] = strtol(q_scores[j].c_str(), NULL, 10);
	}
	
	// init flow-error accounting data structures
	n_flow_aligned.clear();
	flow_err.clear();
	flow_err_bases.clear();

	//init error table
	error_to_length_map initialized_map(opt.err_table_max_errors);
	for (int z = 0; z <= opt.err_table_max_errors; z++) {
		initialized_map[z] = 0;
	}
	mapped_bases = 0;
	mapped_reads = 0;
	for (int i = opt.err_table_min_len; i <= opt.err_table_max_len; i += opt.err_table_len_step) {
		unaligned_read_totals[i] = 0;
		clipped_read_totals[i] = 0;
		filtered_read_totals[i] = 0;
		error_table[i] = initialized_map;
		total_error_to_length[i] = 0;
		per_position_mismatch[i] = 0;
		per_position_insertion[i] = 0;
		per_position_deletion[i] = 0;
		error_table_read_lens.push_back(i);
	}

	// Init AQ length table 
	read_totals_t::size_type read_totals_size = 0;
	for (int i = opt.align_summary_min_len; i <= opt.align_summary_max_len; i += opt.align_summary_len_step) {
		alignment_summary_read_lens.push_back(i);
		read_totals_size++;
	}
	for (unsigned int j = 0; j < phreds.size(); j++) {
		alignment_summary_map[ phreds[j] ] = read_totals_t( read_totals_size );
	}
}


void AlignStats::set_genome_info( ) {
	//if fails from genome_info can be gotten from util
	//util gets this info from the bam_header
	
	ifstream genome_info(opt.genome_info.c_str());
	if (genome_info.fail()) {
		cerr << "[alignStats] genome info file: " << opt.genome_info << " couldn't be opened.  "
		<< "making no assumptions about genome and analyzing all data" << endl;
	} else {
		string line;
		while (genome_info.good()) {
			getline(genome_info, line);
			//line has stuff from genome info now
			vector<string> split_txt;
			split(line, '\t', split_txt);
			
			if (split_txt.size() > 0) {
				if (split_txt[0].find("genome_length") != string::npos) {
					genome_length = strtol(split_txt[1].c_str(), NULL, 10);
				} 
				else if (split_txt[0].find("genome_name") != string::npos) {
					genome_name = split_txt[1];
				}
				else if (split_txt[0].find("genome_version") != string::npos) {
					genome_version = split_txt[1];
				}
				else if (split_txt[0].find("index_version") != string::npos) {
					index_version = split_txt[1];
				}
			}
			
		}
		genome_info.close();
	}
	
	
	
}



static void printProgress(long total_reads_cached, long total_reads_processed)
{
        static int max_length = 0;
        int cur_length = 0;
        std::stringstream stream;
        // make the stream
        stream << "[alignStats] " << total_reads_cached << " alignments read, " << total_reads_processed << " analyzed";
        std::string str = stream.str();
        // get the string
        cur_length = str.length();
        if (max_length < cur_length) {
                max_length = cur_length;
        }
        // print the string
        std::cerr << "\r" << str;
        // print padding, if necessary
        while(0 < max_length - cur_length) { // TODO: only need to print up to the last string's length
            std::cerr << " ";
            cur_length++;
        }
}


void AlignStats::go() {	
		
	total_reads_cached = 0; //1 based
	total_reads_processed = 0; //1 based
	BAMReader::iterator bam_itr = reader.get_iterator();

	if (opt.debug_flag) std::cerr << "[go] going.." << std::endl;	
	//bool queue_status = false;	

	
	//bam_itr_ptr = &bam_itr;
	std::vector<pthread_t> my_threads(opt.num_threads);
	int active_threads = 0;

	you_should_run = true;
	for (int i = 0; i < opt.num_threads; i++) {
		vector_queue.push_back(CircularQueue<worker_data, 20>());
		
	}
	
	/*queue_status = */read_bam(bam_itr, my_threads, active_threads);
	
	if (opt.debug_flag) std::cerr << "[go] out of main loop! "  << std::endl;

	you_should_run = false;
	

    //int rc = 0;
	for (int k = 0; k < active_threads; k++) {
		if (opt.debug_flag) std::cerr << "[go] joining thread["<<k<<"]..." << std::endl;
		/*rc = */pthread_join(my_threads[k], NULL);		
		if (opt.debug_flag) std::cerr << "[go] joined thread["<<k<<"]..." << std::endl;
	}
	if (opt.debug_flag) std::cerr << "[go] worker threads joined" << std::endl;
	if (opt.debug_flag ) { //opt.debug_flag output is cleaner without this, and this information is already there anyway
                printProgress(total_reads_cached, total_reads_processed);
	}

	//for region-specific errors: start
	if(_region_positions_defined){		
		_region_error_outputfile << "#total_errors" << "\t"<< _total_region_homo_err << "\t" << _total_region_mm_err << "\t" << _total_region_indel_err << "\t" << _total_region_ins_err << "\t" << _total_region_del_err << endl;
		_region_error_outputfile.close();
		_region_error_positions_outputfile.close();
	}
	//for region-specific errors: end	

	if (opt.debug_flag) cerr << "[go] bam_itr.good(): " << bam_itr.good() << std::endl;
	sam_parsed.close();
	
	
	//while (true);
	write_alignment_summary();
	
	if (opt.err_table_txt_file.size() > 0) {
		write_error_table_txt(opt.err_table_txt_file);
	}
	
	if (opt.err_table_json_file.size() > 0) {
		write_error_table_json(opt.err_table_json_file);
	}
	
	if (opt.score_flows) {
		write_flow_error_table();
	}

        std::cerr << "\n"; // final newline
	
}
//bam_itr.next();
bool AlignStats::read_bam(BAMReader::iterator& bam_itr, std::vector<pthread_t>& my_threads, int& active_threads) {
	if (opt.debug_flag) std::cerr << "[io - read_bam()] bam_itr.file_good(): " << bam_itr.good() << std::endl;
	
	//setup some conditions/pointers
	int cur_tid = 0;
	int read_tid = 0; //loop control var.  required because bam_itr.get_tid() can be negative
	size_t vector_queue_index = 0;
	read_list overlap_reads;
	int	window_start = -1;
	int window_end = -1;
	int window_tid = -1;
	
	//for threads
	//int rc;
        int stacksize = 8192;
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        pthread_attr_setstacksize(&attr, stacksize);
	while (bam_itr.good() && total_reads_cached < opt.read_limit) {
		reads_in_queue = 0;
		int longest_calend = -1;

		read_tid = bam_itr.get_tid();
		if (opt.debug_flag) std::cerr << "pre cur_tid = " << cur_tid << " read_tid = " << read_tid << " bam_itr.good() = " << bam_itr.good() << " reads_in_queue = " << reads_in_queue << " opt.buffer_size = " << opt.buffer_size << std::endl;

		if (read_tid == -1) { //first read is unmapped.
			read_tid = 0;
		}
		read_list lst;
		cur_tid = read_tid;

		if (overlap_reads.size() > 0) {
			if (overlap_reads.back().get_tid() == cur_tid && overlap_reads.front().get_tid() == cur_tid) {
				
				lst.swap(overlap_reads);

			} else {
				worker_data the_data(overlap_reads, PileupFactory());
				add_pair_to_queue(the_data, vector_queue_index);

				overlap_reads.clear();
			}

		}
		
		//window_tid = cur_tid;
		if (opt.debug_flag) std::cerr << "[io] caching: " << opt.buffer_size << " bam_itr.good(): " << bam_itr.good() << std::endl;

		
		bool first_read = true;
		if (opt.debug_flag) std::cerr << "post cur_tid = " << cur_tid << " read_tid = " << read_tid << " bam_itr.good() = " << bam_itr.good() << " reads_in_queue = " << reads_in_queue << " opt.buffer_size = " << opt.buffer_size << std::endl;
		/*bool one = false;
		bool two = false;
		one = cur_tid == read_tid;
		two = (reads_in_queue < opt.buffer_size);
		if (opt.debug_flag) std::cerr << "cur_tid == read_tid: " << one << " reads_in_queue < opt.buffer_size: " << two << std::endl;*/
		std::tr1::unordered_map<std::string,read_region>::const_iterator found_read_region;
		
		while (cur_tid == read_tid && bam_itr.good() && reads_in_queue < opt.buffer_size && total_reads_cached < opt.read_limit) {
			
			BAMRead tmp = bam_itr.get();			
			//for region-specific errors: start
			if(_region_positions_defined){
				found_read_region = _read_regionmap.find (tmp.get_qname());							
	
				if ( found_read_region != _read_regionmap.end() ){
					tmp.set_region_base_positions(found_read_region->second.region_start_base, found_read_region->second.region_stop_base);
				}		
			}//for region-specific errors: end
			
			
			 			
			if(opt.debug_flag) cerr << tmp.get_qname() << " " << tmp.get_tlen() << " " << tmp.get_seq().to_string() << endl;
			
			if (!opt.skip_cov_flag && is_sorted) {
				//define some_window for pileupfactory
				
				int tmp_calend = tmp.calend();
				
				if (tmp_calend > longest_calend) {
					longest_calend = tmp_calend;  //this will be the end of the window
					
				}
				
			}
			if (tmp.get_bam_ptr() == NULL) { 
				if (opt.debug_flag) std::cerr << "[io] breaking because tmp.get_bam_ptr() == NULL"<< "  bam_itr.good(): " << bam_itr.good() << std::endl;				
				break;
			}
			bool analyze_read=true;
			if((read_to_keep.size() > 0) && (read_to_keep.find(tmp.get_qname()) == read_to_keep.end()) ) {
				analyze_read=false;
				//cout << "rejecting " << tmp.get_qname() << " which fails to match list of ids to retain" << endl;
			}
			if(analyze_read && (read_to_reject.size() > 0) && (read_to_reject.find(tmp.get_qname()) != read_to_reject.end()) ) {
				analyze_read=false;
				//cout << "rejecting " << tmp.get_qname() << " which matches list of ids to reject" << endl;
			}
			if (analyze_read) {
				if (tmp.is_mapped()) {
					if (first_read) {
						
						
						if (lst.size() > 0) {  //some reads in teh overlap queue
							//case 1 we're in the same chromosome.  this one is easy.
	
							if (tmp.get_tid() == window_tid) {
								window_start = window_end + 1;
							} else {
								window_start = 1;  //this could be tmp.get_pos() but i think setting it to 1 guarantees a lot of things
								window_tid = tmp.get_tid();
							}
	
						
						} else {
							window_start = tmp.get_pos();
							window_tid = tmp.get_tid();
						}
	
						
						first_read = false;
					} else {
						if (is_sorted) {
							if (lst.back().get_pos() > tmp.get_pos()) {
								std::cerr << "[alignStats] input file is not sorted.  skipping coverage calculations" << std::endl;
								is_sorted = false;
							}
						}
					}
					
					if (opt.debug_flag) std::cerr << "[io - read_bam()] about to add to list.  bam_itr.get().is_mapped():  "<< tmp.is_mapped() << "  bam_itr.file_good(): " << bam_itr.good() << std::endl;
					lst.push_back(tmp);
					if (opt.debug_flag) std::cerr << "[io - read_bam()] added to list"<< std::endl;
					reads_in_queue++;
				
								
					
					
				} else {
					if (opt.debug_flag) std::cerr << "[io - read_bam()] unmapped read in stdrd loop" << endl;
					for (std::vector<int>::const_iterator read_len_itr = error_table_read_lens.begin(); 
							read_len_itr != error_table_read_lens.end(); ++read_len_itr) {
						
						if (tmp.get_tlen() >= *read_len_itr) {
							
							unaligned_read_totals[*read_len_itr] = unaligned_read_totals[*read_len_itr] + 1;
							
							
						}
						
					}
					
	
				}
			}
			total_reads_cached++;
			if (!opt.debug_flag ) { //opt.debug_flag output is cleaner without this, and this information is already there anyway
				if (total_reads_cached % 10000 == 0 || (0 < total_reads_processed && total_reads_processed % 10000 == 0)) {
                                        printProgress(total_reads_cached, total_reads_processed);
				}
			}
						
			bam_itr.next();
			int tmp_tid = bam_itr.get().get_tid();
			if (tmp_tid != -1) {
				read_tid = tmp_tid;
			}
			if (opt.debug_flag) {
				std::cerr << "[io] cur_tid: " << cur_tid << " == read_tid: " << read_tid << "  && bam_itr.good(): " << bam_itr.good() << "  && reads_in_queue: " << reads_in_queue << " < opt.buffer_size: " << opt.buffer_size << std::endl;
			}
		} //end while (cur_tid == bam_itr.get_tid() && bam_itr.good() && reads_in_queue < opt.buffer_size) {				
	
		
		if (lst.size() > 0 && total_reads_cached < opt.read_limit) {
			if (opt.debug_flag) std::cerr << "[io - read_bam()] adding to list of size:  "<< lst.size() << std::endl;
			/*if (vector_queue_index >= (unsigned int)opt.num_threads) {
				vector_queue_index = 0;
			}*/
			//const bam1_core_t* c = &( lst.back().get_bam_ptr()->core );
			//const bam1_t* b = lst.back().get_bam_ptr();
			//get_bam_ptr()
			window_end = longest_calend;  
			
			int cur_pos = lst.back().get_pos();
			if (!opt.skip_cov_flag && is_sorted) {
				int cov_counter = 0;
				
				while ((read_tid == cur_tid) && bam_itr.good() && (cov_counter < opt.max_coverage)) {
					
					
					BAMRead tmp = bam_itr.get();				

					cur_pos = tmp.get_pos();
					if (cur_pos > window_end) {
						break;
					}
					if (tmp.get_bam_ptr() == NULL) { 
						cerr << "[io] null BAMRead !" << endl;
						break;
					}
					if (tmp.is_mapped()) {
						//for region-specific errors: start
						if(_region_positions_defined){
							if ( found_read_region != _read_regionmap.end() ){
								tmp.set_region_base_positions(found_read_region->second.region_start_base, found_read_region->second.region_stop_base);
							}		
						}
						//for region-specific errors: end
						overlap_reads.push_back(tmp);
						cov_counter++;

					} else {
						if (opt.debug_flag) std::cerr << "[io - read_bam()] unmapped read in overlap loop!" << endl;
						for (std::vector<int>::const_iterator read_len_itr = error_table_read_lens.begin(); 
							 read_len_itr != error_table_read_lens.end(); ++read_len_itr) {
							
							if (tmp.get_tlen() >= *read_len_itr) {
								unaligned_read_totals[*read_len_itr] = unaligned_read_totals[*read_len_itr] + 1;
							}
							
						}
						
					}
					total_reads_cached++;
				
					//}
					bam_itr.next();
					int tmp_tid = bam_itr.get().get_tid();
					if (tmp_tid != -1) {
						read_tid = tmp_tid;
					}
					if (opt.debug_flag) std::cerr << "[io] cur_tid: " << cur_tid << " == read_tid: " << read_tid << "  && bam_itr.good(): " << bam_itr.good() << "  && reads_in_queue: " << reads_in_queue << " < opt.buffer_size: " << opt.buffer_size << std::endl;

				}
	
			
			
			}
			if(window_end < window_start) {
				window_end = window_start;
			} else if (!bam_itr.good() || cur_tid != bam_itr.get_tid()) {
				window_end = bam_itr.get_tid_len(cur_tid);
			} 
				
			
			if (!bam_itr.good()) { //save overlap reads 
				lst.insert(lst.end(), overlap_reads.begin(), overlap_reads.end());
			}
			
				
			PileupFactory the_factory(window_tid, window_start, window_end, phreds, overlap_reads);
			if (!opt.skip_cov_flag && is_sorted) {
				the_factory.init_factory();
			}
			worker_data the_data(lst, the_factory);
			add_pair_to_queue(the_data, vector_queue_index);
			//find a thread with open compute capacity
			
			if (opt.debug_flag) std::cerr << "[io - read_bam()] added lst.size(): "<< lst.size() << " for worker: " << vector_queue_index << endl;

			
			///moved from ::go()
			
			if (active_threads < opt.num_threads) {
				if (opt.debug_flag) std::cerr << "[read_bam] spawning thread: " << active_threads << std::endl;
				/*rc =*/ pthread_create(&my_threads[active_threads], &attr, run, this); //supposedly a data race condition  maybe becuase we're using the same &attr
				
				active_threads++;
				
			}
			
			
					
		

			if (opt.debug_flag) std::cerr << "[io - read_bam()] done caching: " << opt.buffer_size << " cached: " << total_reads_cached << std::endl;
			
			
		  }//end if (lst.size() > 0)

		}//end while (bam_itr.good())




		if (opt.debug_flag) std::cerr << "[io - read_bam()] returning.   bam_itr.good(): " << bam_itr.good() << std::endl;
		return false;
	}
	
 
/**
 * finds thread with approximately the least amount of load and adds data to it
 *
 */
void AlignStats::add_pair_to_queue(worker_data& the_data, size_t& vector_queue_index) {

	bool put_success = false;
	unsigned int least_load = 242; //completely arbitrary number.  it's just much larger than the queue is
	//find queue with least # of items in it
	for(worker_queue::size_type j = 0; j < vector_queue.size(); j++) {
		if (least_load == 242 ) { //first case
			least_load = vector_queue[j].get_size();
			vector_queue_index = j;
		} else if (least_load > vector_queue[j].get_size()) {
			least_load = vector_queue[j].get_size();
			vector_queue_index = j;
		}
	}
	//try to add to queue with least # of items in it,
	while (!put_success) {
		put_success = vector_queue[vector_queue_index].put(the_data);

		if (!put_success) {
			vector_queue_index++;
			if (vector_queue_index >= vector_queue.size()) {
				vector_queue_index = 0;
				}
			}
	}

}









void* AlignStats::run(void* arg) {

 AlignStats *args;
 args = static_cast<AlignStats *> (arg);

 args->consume_read_queue();
 return NULL;

}




void  AlignStats::consume_read_queue() {
	size_t me = 45;
	if (opt.debug_flag) std::cerr <<"[worker thread] thread["<< me << "] hi " << std::endl;
	
	{ 
		ScopedLock lck(&worker_mutex);
		me =  my_worker_num;
		my_worker_num++;	
		if (opt.debug_flag) std::cerr <<"[worker thread] thread["<< me << "] have my worker num " << std::endl;

	}

	//bool first_pass = true;
	//pthread_t stats_thread;
	pthread_attr_t attr;
	
	pthread_attr_init(&attr);
	
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	//setup local data structs
	std::vector<coord_t> my_qnum(q_scores.size(), 0);
	std::vector<coord_t> my_qsum(q_scores.size(), 0);
	std::vector<coord_t> my_qlongest(q_scores.size(), 0);
	std::vector<std::vector<coord_t> >	my_qhisto(q_histo);
	std::vector<coord_t> my_coverage(q_scores.size(), 0);
	
							//len, errors
	long						my_mapped_bases(mapped_bases);
	long						my_mapped_reads(mapped_reads);
	error_table_type					my_error_table(error_table);
	phred_to_totals_t					my_alignment_summary_map(alignment_summary_map);
	length_to_total_map_t				my_clipped_reads(clipped_read_totals);
	length_to_total_map_t				my_filtered_reads(filtered_read_totals);
	length_to_total_map_t				my_total_error_to_length(total_error_to_length);
	length_to_total_map_t				my_per_position_mismatch(per_position_mismatch);
	length_to_total_map_t				my_per_position_insertion(per_position_insertion);
	length_to_total_map_t				my_per_position_deletion(per_position_deletion);

	std::vector<int> my_max_aligned_flow;
	std::vector<uint32_t> my_flow_err;
	std::vector<uint32_t> my_flow_err_bases;
	long my_total_reads_processed=0;
		
	worker_data my_data;
	//bool pop_success = true; //pop_succes
	//std::string test_read("FL9BY:1784:522");

	while (you_should_run || !(vector_queue[me].is_empty())) {
		if (opt.debug_flag) std::cerr <<"[worker thread] thread["<< me << "] top 'o the loop" << std::endl;

		/*pop_success =*/ vector_queue[me].pop(my_data);
	
		
		
		read_list& my_reads = my_data.first;
		if (my_reads.size() == 0) {
			if (opt.debug_flag) std::cerr <<"[worker thread] thread["<< me << "] continuing" << std::endl;
			continue;
		}
		util_list util_cache;//(my_reads.size());
		//setup factory
		PileupFactory& p_fact = my_data.second;
		if (!opt.skip_cov_flag && is_sorted) {
			//p_fact.init_factory();
			p_fact.handle_overlaps(opt.q_scores, opt.start_slop);

		}
		if (opt.debug_flag) std::cerr <<"[worker thread] thread["<< me << "] popped "<< my_reads.size() << endl;

		
		//unsigned int util_cache_index = 0;
		my_total_reads_processed = 0;
		for(read_list::iterator itr = my_reads.begin(); itr != my_reads.end(); ++itr){
		
			util_cache.push_back(BAMUtils(*itr, opt.q_scores, 
                                opt.start_slop, opt.iupac_flag, opt.keep_iupac, 
                                opt.truncate_soft_clipped, opt.err_table_min_len, 
                                opt.err_table_max_len, opt.err_table_len_step, 
                                opt.three_prime_clip, opt.round_phred_scores,
                                opt.five_prime_justify,opt.flow_order));			

			
			BAMUtils& util = util_cache.back();
			
			if (!opt.skip_cov_flag && is_sorted) {
				p_fact.insert_util(util);
			}
			std::vector<coord_t> q_len(q_scores.size(), 0);
			for ( std::vector<std::string>::size_type k = 0; k < q_scores.size(); k++) {
				
				q_len.at(k) = util.get_phred_len(phreds[k]);
				
				if (q_len.at(k) > opt.align_summary_filter_len) {
					my_qnum[k]++;
					my_qsum[k] += q_len.at(k);
					
					for (read_totals_t::size_type z = 0; z < my_alignment_summary_map[ phreds[k] ].size(); z++) {
						if (q_len.at(k) >= alignment_summary_read_lens[ z ] ) {
							my_alignment_summary_map[ phreds[k] ][ z ]++;
						} else {
							break; //save some loop iterations
						}

					}
					
					
					if (q_len.at(k) > my_qlongest[k]) {
						my_qlongest[k] = q_len.at(k);
					}
					if((int) my_qhisto[k].size() < q_len.at(k)) {
						my_qhisto[k].resize(q_len.at(k) * 2);
					}  
					
					my_qhisto[k][q_len.at(k)]++;
				}
			}
			my_mapped_bases += util.get_t_length();
			my_mapped_reads ++;
			if ((util.pass_filtering(opt.err_table_filter_len, opt.err_table_filter_accuracy) || opt.err_table_filter_len == 0 || opt.err_table_filter_accuracy == 0.0)  ) {
				// The read meets our minimal criteria to be used for error assessment
				if(opt.score_flows) {
					my_max_aligned_flow.push_back(util.get_max_aligned_flow());
					std::vector<uint16_t>& read_flow_err = util.get_flow_err();
					std::vector<uint16_t>& read_flow_err_bases = util.get_flow_err_bases();
					for(unsigned int iErr=0; iErr < read_flow_err.size(); iErr++) {
						unsigned int f = read_flow_err[iErr];
						if(f >= my_flow_err.size()) {
							// need to expand summary result vectors
							my_flow_err.resize(f+1,0);
							my_flow_err_bases.resize(f+1,0);
						}
						my_flow_err[f] += 1;
						my_flow_err_bases[f] += read_flow_err_bases[iErr];
					}
				}
					
				int total_errors_in_read = 0;
				for (int len = opt.err_table_min_len; (len <= opt.err_table_max_len) ; len+=opt.err_table_len_step) {
					int errors_at_this_position = util.get_total_position_errors(len);
					total_errors_in_read = util.get_total_error_at_length(len);
					if ( ( len <= (util.get_q_length() ) ) ) {
						if (total_errors_in_read >= 0)	{
							my_total_error_to_length[ len ]  +=  errors_at_this_position;
							my_per_position_mismatch[ len ]  += util.get_total_position_errors_mis(len);
							my_per_position_insertion[ len ] += util.get_total_position_errors_ins(len);
							my_per_position_deletion[ len ]  += util.get_total_position_errors_del(len);
							error_table_type::iterator map_itr = my_error_table.find(len); //len used to be tbl_itr->first
							if (map_itr != my_error_table.end()) {
								//exists in map
								//in_map_itr->first = #of error
								//in_map_itr->second = total # of reads with error
								error_to_length_map::iterator in_map_itr = map_itr->second.find(total_errors_in_read); //tbl_itr->second used to be where errors is

								if (in_map_itr != map_itr->second.end()) {
									//exists in map
									in_map_itr->second = in_map_itr->second +  1;
								} else {
									//my_error_table
									map_itr->second.insert(error_to_length_map::value_type(total_errors_in_read, 1));
								}
							} else {
								//my_error_table[tbl_itr->first] = error_to_length_map;
								error_to_length_map new_map;
								new_map.insert(error_to_length_map::value_type(total_errors_in_read, 1));
								my_error_table.insert(error_table_type::value_type(len, new_map));
							}
						} else {
							/*if ((util.get_q_length() + util.get_soft_clipped_bases() + opt.three_prime_clip ) >= len) {
								my_clipped_reads[ len ]++;
							}*/
						}
					} else {
						/*using namespace std;
						cerr << "util.get_q_length(): " << util.get_q_length() << " opt.three_prime_clip: " <<
								opt.three_prime_clip << " len: " << len << endl;*/
						if ((util.get_q_length() + util.get_soft_clipped_bases() + util.get_adjusted_three_prime_trim_length() ) > len) {
							my_clipped_reads[ len ]++;
						}
					}
				}
			}
			else  {  //doesn't pass filtering or clipped metrics
				for (int i = opt.err_table_min_len; i <= opt.err_table_max_len; i+=opt.err_table_len_step) {
					if (util.get_q_length() >= i) {
						my_filtered_reads[i]++;
					} else {
						break;
					}
				}
			}
			my_total_reads_processed++;
		}
		my_reads.clear();
		if (opt.debug_flag) std::cerr <<"[worker thread] thread["<< me << "] all utils made " << std::endl;

		if (genome_length <= 0) {
			ScopedLock lck(&worker_mutex);

			genome_length = util_cache.back().get_genome_length();
			genome_name = util_cache.back().get_rname();					
			//free mutex

		}
		
		if (!opt.skip_cov_flag && is_sorted) {
			sum_coverage(p_fact, my_coverage, util_cache);
		}
		//for region-specific errors: start
		
		int region_mm_err = 0;
		int	region_homo_err = 0;
		int	region_indel_err = 0;
		int	region_ins_err = 0;
		int	region_del_err = 0;
		bool 	region_clipped = false; 
		std::vector<std::pair<long,int> >  region_error_positions;
		
		if(_region_positions_defined){
			ScopedLock lck(&_region_error_outputfile_mutex);
			for (util_list::iterator w_itr = util_cache.begin(); w_itr != util_cache.end(); ++w_itr) {			
				if((*w_itr).get_bamread().region_error_flag()){
					(*w_itr).get_region_errors(&region_homo_err, &region_mm_err, &region_indel_err,&region_ins_err, &region_del_err, &region_clipped, &region_error_positions);
					if(!region_clipped){
						_region_error_outputfile << (*w_itr).get_bamread().get_qname() << "\t"<< region_homo_err << "\t" << region_mm_err << "\t" << region_indel_err << "\t" << region_ins_err << "\t" << region_del_err << endl; //MJ
						_total_region_homo_err = _total_region_homo_err + region_homo_err;	
						_total_region_mm_err = _total_region_mm_err + region_mm_err;
						_total_region_indel_err = _total_region_indel_err + region_indel_err;
						_total_region_ins_err = _total_region_ins_err + region_ins_err;
						_total_region_del_err = _total_region_del_err + region_del_err;			
						
						int len_region_error = region_error_positions.size();
						
						_region_error_positions_outputfile << (*w_itr).get_bamread().get_qname() << "\t" <<  len_region_error;
						for(int ii=0; ii<len_region_error; ii++){
							_region_error_positions_outputfile << "\t" << (region_error_positions.at(ii)).first << "\t" << (region_error_positions.at(ii)).second; 
						}
						_region_error_positions_outputfile << std::endl;
					}
					
				}			
			}		
		}		
		//for region-specific errors: end	
		
		if (opt.sam_parsed_flag){
			ScopedLock lck(&sam_parsed_mutex);
			for (util_list::iterator w_itr = util_cache.begin(); w_itr != util_cache.end(); ++w_itr) {
				sam_parsed << w_itr->to_string() << endl;
			}
		}
		
		util_cache.clear();

		
		if (opt.debug_flag) std::cerr <<"[worker thread] thread["<< me << "] done with chunk" << std::endl; 

		{
			ScopedLock lck(&worker_mutex);
			total_reads_processed += my_total_reads_processed;
			if (!opt.debug_flag ) { //opt.debug_flag output is cleaner without this, and this information is already there anyway
                                if (10000 < my_total_reads_processed || (int)(total_reads_processed / 10000) != (int)((total_reads_processed-my_total_reads_processed) / 10000)) {
                                        printProgress(total_reads_cached, total_reads_processed);
                                }

			}
		}
		//first_pass = false;
	}

	std::vector<uint32_t> my_n_flow_aligned;
	if(opt.score_flows) {
		// Sort the max flow values so we can know how many times we reached each flow position
		sort(my_max_aligned_flow.begin(),my_max_aligned_flow.end());
		countAlignedFlows(my_max_aligned_flow,my_n_flow_aligned);
		unsigned int max_flow = my_n_flow_aligned.size();
		my_flow_err.resize(max_flow,0);
		my_flow_err_bases.resize(max_flow,0);
	}
	{ //scope
	ScopedLock lck(&worker_mutex);
		for ( std::vector<std::string>::size_type k = 0; k < q_scores.size(); k++) {
			q_num[k] += my_qnum[k];
			q_sum[k] += my_qsum[k];
			if (q_longest[k] < my_qlongest[k]) {
				q_longest[k] = my_qlongest[k];
			}
			
			for (read_totals_t::size_type z = 0; z < my_alignment_summary_map[ phreds[k] ].size(); z++) {
							
				alignment_summary_map[ phreds[k] ][ z ] += my_alignment_summary_map[ phreds[k] ][ z ];
				
			}
			
			if( q_histo[k].size() < my_qhisto[k].size()) {
				q_histo[k].resize(my_qhisto[k].size() * 2);
			}  		
			for (std::vector<coord_t>::size_type j = 0; j < my_qhisto[k].size(); j++) {
						
				q_histo[k][j] += my_qhisto[k][j];
			}
			coverage[k] = coverage[k] +  my_coverage[k];
			if (opt.debug_flag) std::cerr <<"worker ["<< me << "]"<< "coverage["<<k<<"] = " << coverage[k] <<  " my_coverage["<<k<<"] = " << my_coverage[k] << endl;

		 }
		for (error_table_type::const_iterator k = my_error_table.begin(); k != my_error_table.end(); ++k) {
			
														//k->sceond is the nested stl map
			for (error_to_length_map::const_iterator z = k->second.begin(); z != k->second.end(); ++z) {
				
				//k->first is length, z->first is the err#, z->second is the total
												
				error_table[ k->first ][ z->first ] += z->second;
				
				
			}
			
		}
		
		mapped_bases = mapped_bases + my_mapped_bases;
		mapped_reads = mapped_reads + my_mapped_reads;
		for (int i = opt.err_table_min_len; i <= opt.err_table_max_len; i+=opt.err_table_len_step) {
			clipped_read_totals[i]    += my_clipped_reads[i];
			filtered_read_totals[i]   += my_filtered_reads[i];
			total_error_to_length[i]  += my_total_error_to_length[i];
			per_position_mismatch[i]  += my_per_position_mismatch[i];
			per_position_insertion[i] += my_per_position_insertion[i];
			per_position_deletion[i]  += my_per_position_deletion[i];
		}
		
		if(opt.score_flows) {
			unsigned int max_flow = std::max(my_n_flow_aligned.size(),n_flow_aligned.size());
			n_flow_aligned.resize(max_flow,0);
			flow_err.resize(max_flow,0);
			flow_err_bases.resize(max_flow,0);
			my_n_flow_aligned.resize(max_flow,0);
			my_flow_err.resize(max_flow,0);
			my_flow_err_bases.resize(max_flow,0);
			for(unsigned int i_flow=0; i_flow < max_flow; i_flow++) {
				n_flow_aligned[i_flow] += my_n_flow_aligned[i_flow];
				flow_err[i_flow] += my_flow_err[i_flow];
				flow_err_bases[i_flow] += my_flow_err_bases[i_flow];
			}
		}
	}//end scope
	if (opt.debug_flag) std::cerr <<"[worker thread] thread["<< me << "] exiting.. " << std::endl;
	pthread_exit(NULL);
}



//push read and utils into pileup
void AlignStats::sum_coverage(PileupFactory& pileup_factory, vector<coord_t>& my_cov, util_list& util_cache) {
	
	for (PileupFactory::pileup_iterator pileup_itr = pileup_factory.get_pileup_iterator(); pileup_itr.good(); pileup_itr.next()) {
		Pileup& pup = pileup_itr.get();
		for (unsigned int j = 0; j < phreds.size(); j++) {
			if (pup.get_phred_cov( phreds[j] ) > 0) {
				my_cov[j] = my_cov[j] + 1;
			}
		}
		
	}
	
		
	
	
}








long AlignStats::get_total_reads() {
	
	if (opt.total_reads == 0) {
		return total_reads_cached;
	} else {
		return opt.total_reads;
	}

	
}


void AlignStats::extrapolate() {
	
	double scale_factor = 0.0;
	
	scale_factor = (double)get_total_reads() / (double) opt.sample_size;
	for (vector<coord_t>::size_type k = 0; k <q_num.size(); k++) {
		q_num[k] = scale_factor * q_num[k];
		q_sum[k] = scale_factor * q_sum[k];
		for (unsigned int z = 0; z < alignment_summary_map[ phreds[k] ].size(); z++) {
			alignment_summary_map[ phreds[k] ][ z ] = scale_factor * alignment_summary_map[ phreds[k] ][ z ];
		}
	}
	
	
}

void AlignStats::write_alignment_summary() {
	if (opt.debug_flag) std::cerr << "[write_alignment_summary] running.  is_sorted: " << is_sorted << std::endl;
	//long total_reads = get_total_reads();
	
	string alignment_stats_filename;
   
    if( opt.out_file.find("Default") == std::string::npos) {
      alignment_stats_filename = opt.out_file + ".alignment.summary";
	} else {
      if ( opt.output_dir.length() > 0 ) {
        alignment_stats_filename = opt.output_dir + "/alignment.summary";
      } else {
        alignment_stats_filename = "alignment.summary";
      }
			
	}
		
	
	
	ofstream alignment_stats(alignment_stats_filename.c_str());
	if (alignment_stats.fail()) {
		cerr << "[alignStats] couldn't open " << alignment_stats_filename << endl;
		exit(1);
		
	}
	
		
	vector<double> cov_dep(q_num.size(),0.0);
	
		
	if(genome_length > 0){
		for (vector<coord_t>::size_type l = 0; l < coverage.size(); l++) {
			cov_dep[l] = (double)q_sum[l]/(double)genome_length;
			
		}
	}
	

  alignment_stats << "[global]" << endl;
	alignment_stats << "Genome = " << genome_name << endl;
	alignment_stats	<< "Genome Version = " << genome_version << endl;
	alignment_stats	<< "Index Version = " << index_version << endl;
	alignment_stats	<< "Genomesize = " << genome_length << endl;
	
	if (opt.sample_size == 0) {
		alignment_stats << "Total number of Reads = " << get_total_reads() << endl;
		for ( vector<coord_t>::size_type k = 0; k < q_scores.size(); k++) {
			if (genome_length == 0 || (opt.skip_cov_flag || !is_sorted)) {
				alignment_stats	<< "Filtered Q" << q_scores[k] << " Coverage Percentage = 0" << endl; 
				alignment_stats << "Filtered Q" << q_scores[k] << " Mean Coverage Depth = 0" << endl;
			}
			if (genome_length > 0 && (!opt.skip_cov_flag && is_sorted)) {
				if (opt.debug_flag) std::cerr << "[write_alignment_summary] q_scores["<<k<<"] coverage["<<k<<"] = " << coverage[k] << " genome_length = " << genome_length << std::endl;
				alignment_stats << "Filtered Q" << q_scores[k] << " Coverage Percentage = " 
				<< fixed<< setprecision(2) << (100.0 * (coverage[k] / (double)genome_length))
				<<	endl;
				alignment_stats << "Filtered Q" << q_scores[k] 
				<< " Mean Coverage Depth = " << fixed << setprecision(1) << cov_dep[k] << endl;
			}
			alignment_stats << "Filtered Q" << q_scores[k] << " Alignments = " << q_num[k] << endl;
			if (q_num[k] > 0) {
				double align_len = (double)q_sum[k]/(double)q_num[k];
				alignment_stats << "Filtered Q" << q_scores[k] << " Mean Alignment Length = " 
				<<  fixed << setprecision(0) << align_len << endl;
			}
			if (q_num[k] == 0) {
				alignment_stats << "Filtered Q" << q_scores[k] << " Mean Alignment Length = 0" << endl;
			}
			
			alignment_stats << "Filtered Mapped Bases in Q" 
			<< q_scores[k] << " Alignments = " << q_sum[k] << endl
			<< "Filtered Q" << q_scores[k] << " Longest Alignment = " << q_longest[k] << endl;
			
			
			for (unsigned int z = 0; z < alignment_summary_map[ phreds[k] ].size(); z++) {
				alignment_stats << "Filtered " << alignment_summary_read_lens[z] <<"Q"<<  q_scores[k] << " Reads = " << alignment_summary_map[ phreds[k] ][ z ] << endl;
			}
		}
	} else { //must be sampled
		alignment_stats << "Total number of Sampled Reads = " << opt.sample_size << endl;
		for ( vector<coord_t>::size_type k = 0; k < q_scores.size(); k++) {
			if (genome_length == 0) {
				alignment_stats	<< "Sampled Filtered Q" << q_scores[k] << " Coverage Percentage = 0" << endl; 
				alignment_stats << "Sampled Filtered Q" << q_scores[k] << " Mean Coverage Depth = 0" << endl;
				
			}
			if (genome_length > 0) {
				alignment_stats << "Sampled Filtered Q" << q_scores[k] << " Coverage Percentage = " 
				<<  fixed<< setprecision(2) << (100.0 * (coverage[k] / (double)genome_length))
				<<	endl;
				alignment_stats << "Sampled Filtered Q" << q_scores[k] 
				<< " Mean Coverage Depth = "<< fixed << setprecision(1) << cov_dep[k] << endl;
				
			}
			alignment_stats << "Sampled Filtered Q" << q_scores[k] << " Alignments = " << q_num[k] << endl;
			if (q_num[k] > 0) {
				double align_len = (double)q_sum[k]/(double)q_num[k];
				alignment_stats << "Sampled Filtered Q" << q_scores[k] << " Mean Alignment Length = " 
				<<  fixed << setprecision(0) << align_len << endl;
			}
			if (q_num[k] == 0) {
				alignment_stats << "Sampled Filtered Q" << q_scores[k] << " Mean Alignment Length = 0" << endl;
				
			}
			
			alignment_stats << "Sampled Filtered Mapped Bases in Q" 
			<< q_scores[k] << " Alignments = " << q_sum[k] << endl
			<< "Sampled Filtered Q" << q_scores[k] << " Longest Alignment = " << q_longest[k] << endl;
			for (unsigned int z = 0; z < alignment_summary_map[ phreds[k] ].size(); z++) {
				alignment_stats << "Sampled Filtered " << alignment_summary_read_lens[z] <<"Q"<<  q_scores[k] << " Reads = " << alignment_summary_map[ phreds[k] ][ z ] << endl;
			}
			/*
			<< "Sampled Filtered 50Q" << q_scores[k] << " Reads = " << q_fif_num[k] << endl
			<< "Sampled Filtered 100Q" << q_scores[k] << " Reads = " << q_hun_num[k] << endl
			<< "Sampled Filtered 200Q" << q_scores[k] << " Reads = " << q_two_hun_num[k] << endl;*/
		}
		
		
		extrapolate();
		if (genome_length > 0) {
			for (vector<double>::size_type k = 0; k < cov_dep.size(); k++) {
				  cov_dep[k] = (double)  q_sum[k] / (double)genome_length;
			}
		}
		alignment_stats << "Total number of Reads = " << get_total_reads() << endl;
		alignment_stats << "Extrapolated from number of Sampled Reads = " << opt.sample_size << endl;
		for ( vector<coord_t>::size_type k = 0; k < q_scores.size(); k++) {
			if (genome_length == 0) {
				alignment_stats	<< "Extrapolated  Filtered Q" << q_scores[k] << " Coverage Percentage = 0" << endl; 
				alignment_stats << "Extrapolated  Filtered Q" << q_scores[k] << " Mean Coverage Depth = 0" << endl;
				
			}
			if (genome_length > 0) {
				alignment_stats << "Extrapolated Filtered Q" << q_scores[k] << " Coverage Percentage = NA" 
				//%.2f 100*coverage[k]/genome_length
				<<	endl;
				alignment_stats << "Extrapolated Filtered Q" << q_scores[k] 
				<< " Mean Coverage Depth = "<< fixed << setprecision(1) <<   cov_dep[k] << endl;
				
			}
			alignment_stats << "Extrapolated Filtered Q" << q_scores[k] << " Alignments = " <<   q_num[k] << endl;
			if (q_num[k] > 0) {
				double align_len = (double)q_sum[k]/(double)q_num[k];
				alignment_stats << "Extrapolated Filtered Q" << q_scores[k] << " Mean Alignment Length = " 
				<<  fixed << setprecision(0) << align_len << endl;
			}
			if (q_num[k] == 0) {
				alignment_stats << "Extrapolated Filtered Q" << q_scores[k] << " Mean Alignment Length = 0" << endl;
				
			}
			
			alignment_stats << "Extrapolated Filtered Mapped Bases in Q" << q_scores[k] << " Alignments = " <<   q_sum[k] << endl
			<< "Extrapolated Filtered Q" << q_scores[k] << " Longest Alignment = NA" << endl;
			for (unsigned int z = 0; z < alignment_summary_map[ phreds[k] ].size(); z++) {
				alignment_stats << "Extrapolated Filtered " << alignment_summary_read_lens[z] <<"Q"<<  q_scores[k] << " Reads = " << alignment_summary_map[ phreds[k] ][ z ] << endl;
			}
			/*<< "Extrapolated Filtered 50Q" << q_scores[k] << " Reads = " <<   q_fif_num[k] << endl
			<< "Extrapolated Filtered 100Q" << q_scores[k] << " Reads = " <<   q_hun_num[k] << endl
			<< "Extrapolated Filtered 200Q" << q_scores[k] << " Reads = " <<   q_two_hun_num[k] << endl;*/
		}
	}
	alignment_stats.close();
	
	if (!opt.skip_cov_flag && is_sorted) { 
		for ( vector<string>::size_type k = 0; k < q_scores.size(); k++) {
          		string histo_prefix =  "Q" + q_scores[k];
          		ofstream histo_file;
          		if ( opt.output_dir.length() > 0 ) {
            			histo_file.open( string( opt.output_dir + "/" + histo_prefix + ".histo.dat").c_str() );
          		} else {
            			histo_file.open( string(histo_prefix + ".histo.dat").c_str() );
          		}
          		if (histo_file.fail()) {
            			cerr << "[alignStats] couldn't open " << histo_prefix + "histo.dat" << endl;
				exit(1);
		  	}

          		for (int i = 0; i <= opt.align_summary_max_len; i++) {
              			if (i <= opt.align_summary_filter_len) {
                  			histo_file << i << " 0" << endl;
				} else {
                  			histo_file << i << " " << q_histo[k][i] << endl;
              			}
          		}
			histo_file.close();
        	}
	}
	
	if (opt.debug_flag) std::cerr << "[write_alignment_summary] done" << std::endl;
}
	

void AlignStats::write_flow_error_table() {
	ofstream flow_err_out(opt.flow_err_file.c_str());
	char delim = '\t';
	if (flow_err_out.fail()) {
		cerr << "[alignStats] problem writing to flow error file " << opt.flow_err_file << endl;
		return;
	}
	flow_err_out << "flow" << delim << "base" << delim << "n_aligned" << delim << "n_hp_err" << delim << "n_base_err" << endl;
	for(unsigned int i_flow=0; i_flow < n_flow_aligned.size(); i_flow++) {
		flow_err_out << i_flow;
		flow_err_out << delim << opt.flow_order[i_flow % opt.flow_order.size()];
		flow_err_out << delim << n_flow_aligned[i_flow];
		flow_err_out << delim << flow_err[i_flow];
		flow_err_out << delim << flow_err_bases[i_flow];
		flow_err_out << endl;
	}
	flow_err_out.close();
}
	

/* 
 readlen	num_reads	unaligned	err0	err1	err2	err3+
 x		
 */

void AlignStats::write_error_table_txt(string outFile) {
	ofstream alignment_stats(outFile.c_str());
	char delimiter = '\t';
	if (alignment_stats.fail()) {
		cerr << "[alignStats] couldn't open " << outFile << endl;
		return;
	}
	//write header of file
	alignment_stats << "readLen" << delimiter << "nread" << delimiter << "unalign" << delimiter <<"excluded"<< delimiter << "clipped" << delimiter <<"totErr";
	for (int i = 0; i <= opt.err_table_max_errors; i++) {
		alignment_stats << delimiter<<"err"<<i;
		if (i == opt.err_table_max_errors) {
			alignment_stats << "+";
		}
	}
	alignment_stats << endl;
	for (std::vector<int>::iterator itr = error_table_read_lens.begin(); itr != error_table_read_lens.end(); ++itr) {
		
		//*itr length of read
		
		
		long cum_mapped_reads = 0;
		
		for (error_to_length_map::const_iterator map_itr = error_table[*itr].begin(); map_itr != error_table[*itr].end(); ++map_itr) {
			//map_itr->first = length
			//map_itr->second = # of reads
			cum_mapped_reads = cum_mapped_reads + map_itr->second; 
		
			
		}//+ clipped_read_totals[ map_itr->first ] + filtered_read_totals[ map_itr->first ];
		long nread = cum_mapped_reads + unaligned_read_totals[*itr] + clipped_read_totals[ *itr ] + filtered_read_totals[ *itr ];
		alignment_stats <<  *itr							<< delimiter; //readLen
		alignment_stats <<	nread							<< delimiter; //nread
		alignment_stats <<	unaligned_read_totals[ *itr ]	<< delimiter; //unalign
		alignment_stats <<	filtered_read_totals[ *itr ]	<< delimiter; //exclude
		alignment_stats <<	clipped_read_totals[ *itr ]		<< delimiter; //soft clipped
		alignment_stats <<  total_error_to_length[ *itr ]	<< delimiter; //total errors AT this position (not cumulative up until this position)
		
		
				
		int final_place = 0;
		for (error_to_length_map::const_iterator map_itr = error_table[*itr].begin(); map_itr != error_table[*itr].end(); ++map_itr) {
			
				if (map_itr->first >= opt.err_table_max_errors) {
					final_place += map_itr->second;
				}
			
		}
		//sum all errors at the max and after

		for (int i = 0; i < opt.err_table_max_errors; i++) {
			alignment_stats << error_table[*itr][i] << delimiter; //err#
		}
		alignment_stats << final_place;
		
		alignment_stats << endl;
		
	}
	alignment_stats.close();
	
}

void AlignStats::write_error_table_json(string outFile) {
	ofstream alignment_stats(outFile.c_str());
	if (alignment_stats.fail()) {
		cerr << "[alignStats] couldn't open " << outFile << endl;
		return;
	}

	// Collect & compute quantities to write
	std::vector<long> v_read_length;
	std::vector<long> v_nread;
	std::vector<long> v_unaligned;
	std::vector<long> v_filtered;
	std::vector<long> v_clipped;
	std::vector<long> v_aligned;
	std::vector<long> v_cum_aligned;
	std::vector<long> v_n_err_at_position;
	std::vector<long> v_n_mis_at_position;
	long total_mismatch = 0;
	std::vector<long> v_n_ins_at_position;
	long total_insertion = 0;
	std::vector<long> v_n_del_at_position;
	long total_deletion = 0;
	std::vector<long> v_cum_err_at_position;
	std::vector< std::vector<long> > v_err_count(1+opt.err_table_max_errors);
	for (std::vector<int>::iterator itr = error_table_read_lens.begin(); itr != error_table_read_lens.end(); ++itr) {
		long cum_mapped_reads = 0;
		for (error_to_length_map::const_iterator map_itr = error_table[*itr].begin(); map_itr != error_table[*itr].end(); ++map_itr) {
			//map_itr->first = length
			//map_itr->second = # of reads
			cum_mapped_reads = cum_mapped_reads + map_itr->second; 
		}
		long nread = cum_mapped_reads + unaligned_read_totals[*itr] + clipped_read_totals[ *itr ] + filtered_read_totals[ *itr ];
		v_read_length.push_back(*itr);
		v_nread.push_back(nread);
		v_unaligned.push_back(unaligned_read_totals[*itr]);
		v_filtered.push_back(filtered_read_totals[*itr]);
		v_clipped.push_back(clipped_read_totals[*itr]);
		long aligned = nread - unaligned_read_totals[*itr] - filtered_read_totals[*itr] - clipped_read_totals[*itr];
		v_aligned.push_back(aligned);
		long cum_aligned = aligned + ((v_cum_aligned.size() > 0) ? v_cum_aligned.back() : 0);
		v_cum_aligned.push_back(cum_aligned);
		v_n_err_at_position.push_back(total_error_to_length[*itr]);
		v_n_mis_at_position.push_back(per_position_mismatch[*itr]);
		total_mismatch += per_position_mismatch[*itr];
		v_n_ins_at_position.push_back(per_position_insertion[*itr]);
		total_insertion += per_position_insertion[*itr];
		v_n_del_at_position.push_back(per_position_deletion[*itr]);
		total_deletion += per_position_deletion[*itr];
		long cum_err_at_position = total_error_to_length[*itr] + ((v_cum_err_at_position.size() > 0) ? v_cum_err_at_position.back() : 0);
		v_cum_err_at_position.push_back(cum_err_at_position);
		int final_place = 0;
		for (error_to_length_map::const_iterator map_itr = error_table[*itr].begin(); map_itr != error_table[*itr].end(); ++map_itr) {
			if (map_itr->first >= opt.err_table_max_errors) {
				final_place += map_itr->second;
			}
		}
		for (int i = 0; i < opt.err_table_max_errors; i++) {
			v_err_count[i].push_back(error_table[*itr][i]);
		}
		v_err_count[opt.err_table_max_errors].push_back(final_place);
	}

	// write json
	alignment_stats << "{" << endl;

	alignment_stats << "  \"read_length\" : [" << v_read_length[0];
	for(unsigned int i=1; i<v_read_length.size(); i++)
		alignment_stats << ", " << v_read_length[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"nread\" : [" << v_nread[0];
	for(unsigned int i=1; i<v_nread.size(); i++)
		alignment_stats << ", " << v_nread[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"unaligned\" : [" << v_unaligned[0];
	for(unsigned int i=1; i<v_unaligned.size(); i++)
		alignment_stats << ", " << v_unaligned[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"filtered\" : [" << v_filtered[0];
	for(unsigned int i=1; i<v_filtered.size(); i++)
		alignment_stats << ", " << v_filtered[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"clipped\" : [" << v_clipped[0];
	for(unsigned int i=1; i<v_clipped.size(); i++)
		alignment_stats << ", " << v_clipped[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"aligned\" : [" << v_aligned[0];
	for(unsigned int i=1; i<v_aligned.size(); i++)
		alignment_stats << ", " << v_aligned[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"n_err_at_position\" : [" << v_n_err_at_position[0];
	for(unsigned int i=1; i<v_n_err_at_position.size(); i++)
		alignment_stats << ", " << v_n_err_at_position[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"n_mis_at_position\" : [" << v_n_mis_at_position[0];
	for(unsigned int i=1; i<v_n_mis_at_position.size(); i++)
		alignment_stats << ", " << v_n_mis_at_position[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"n_ins_at_position\" : [" << v_n_ins_at_position[0];
	for(unsigned int i=1; i<v_n_ins_at_position.size(); i++)
		alignment_stats << ", " << v_n_ins_at_position[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"n_del_at_position\" : [" << v_n_del_at_position[0];
	for(unsigned int i=1; i<v_n_del_at_position.size(); i++)
		alignment_stats << ", " << v_n_del_at_position[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"cum_aligned\" : [" << v_cum_aligned[0];
	for(unsigned int i=1; i<v_cum_aligned.size(); i++)
		alignment_stats << ", " << v_cum_aligned[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"cum_err_at_position\" : [" << v_cum_err_at_position[0];
	for(unsigned int i=1; i<v_cum_err_at_position.size(); i++)
		alignment_stats << ", " << v_cum_err_at_position[i];
	alignment_stats << "]," << endl;

	alignment_stats << "  \"err_count\" : {" << endl;
	for(unsigned int i=1; i<v_err_count.size(); i++) {
		alignment_stats << "    \"err_count" << i << "\" : [" << v_err_count[i][0];
		for(unsigned int j=1; j<v_err_count[i].size(); j++)
			alignment_stats << ", " << v_err_count[i][j];
                if (i == v_err_count.size()  - 1 ){
		        alignment_stats << "]" << endl;
                }else{
		        alignment_stats << "]," << endl;
                }
	}
	alignment_stats << "  }," << endl;

	alignment_stats << "  \"accuracy_total_bases\" : "            << v_cum_aligned.back()         << "," << endl;
	alignment_stats << "  \"accuracy_total_errors\" : "           << v_cum_err_at_position.back() << "," << endl;
	alignment_stats << "  \"accuracy_total_errors_mismatch\" : "  << total_mismatch               << "," << endl;
	alignment_stats << "  \"accuracy_total_errors_insertion\" : " << total_insertion              << "," << endl;
	alignment_stats << "  \"accuracy_total_errors_deletion\" : "  << total_deletion               << "," << endl;
	alignment_stats << "  \"total_mapped_target_bases\" : "       << mapped_bases                 << "," << endl;
	alignment_stats << "  \"total_mapped_reads\" : "              << mapped_reads                 << endl;

	alignment_stats << "}" << endl;
	alignment_stats.close();
}

//file i/o
void AlignStats::write_sam_parsed_header() {
	
	if (sam_parsed.good()) {
		
			//name	strand	tStart	tLen	qLen	match	percent.id	q7Errs	homErrs	mmErrs	indelErrs	qDNA.a	match.a	tDNA.a	
			//tName	start.a	q7Len	q10Len	q17Len	q20Len	q47Len
			sam_parsed << "name" <<"\t"<<"strand"<<"\t"<<
			"tStart"<<"\t"<<"tLen"<<"\t"<<"qLen"
			<<"\t"<<"match"<<"\t"<<"percent.id"
			<<"\t"<<"q7Errs"<<"\t"<<"homErrs"
			<<"\t"<<"mmErrs"<<"\t"<<"indelErrs"
			<<"\t"<<"qDNA.a"<<"\t"<<"match.a"
			<<"\t"<<"tDNA.a"<<"\t"<<"tName"
			<<"\t"<<"start.a";
			
			for (string::size_type i = 0; i < q_scores.size(); i++) {
				sam_parsed <<"\t"<< "q" << q_scores[i] << "Len";
			}
			sam_parsed <<"\t"<<"qLenFull";
			sam_parsed <<  endl;
	}
		
}


void AlignStats::set_read_to_keep(std::string readKeepFile) {
  readNamesFromFile(read_to_keep, readKeepFile);
}

void AlignStats::set_read_to_reject(std::string readRejectFile) {
  readNamesFromFile(read_to_reject, readRejectFile);
}

void AlignStats::readNamesFromFile(std::map<string,bool> &nameMap, std::string inFile) {
	ifstream inStream;
	if(inFile != "") {
		inStream.open(inFile.c_str());
		if(inStream.fail()) {
			ION_ABORT("Unable to open file " + inFile + " for read");
		} else {
			std::string readName;
			while(readNextNameFromStream(readName, inStream)) {
				nameMap[readName] = true;
			}
		}
	}
}

bool AlignStats::readNextNameFromStream(std::string &readName, ifstream &inStream) {
	if(inStream.good()) {
		char delim = '\t';
		string line;
		getline(inStream,line);
		size_t nameEndPos = line.find(delim, 0);
		if (nameEndPos == string::npos) {
			nameEndPos = line.length();
		}
		if(nameEndPos > 0) {
			readName = line.substr(0, nameEndPos);
			return(true);
		} else {
			return(false);
		}
	} else {
		return(false);
	}
}

bool AlignStats::build_read_regionmap(string region_postion_file, std::tr1::unordered_map<string, read_region> *read_regionmap){

	string read_name;
	int start_pos, stop_pos;
	read_region this_read_region;	

	ifstream file_merged_region;
	file_merged_region.open(region_postion_file.c_str());
	
	if (file_merged_region.fail()) {
		cerr << "merged region file couldn't be opened" << endl;
		//exit(1);
		file_merged_region.close();
		return false;
	} else {				
		while (file_merged_region.good()) {
			file_merged_region >> read_name >> start_pos >> stop_pos;
			this_read_region.region_start_base = start_pos;
			this_read_region.region_stop_base = stop_pos;
			std::pair<std::string, read_region> read_region_pair(read_name, this_read_region);
			(*read_regionmap).insert(read_region_pair);							
		}
		file_merged_region.close();
	} 
	
	return true;		
	
}

void AlignStats::countAlignedFlows(std::vector<int> &max_flow, std::vector<uint32_t> &flow_by_position) {
	// max_flow must be in increasing sorted order, or this will not work
	uint16_t prev_max_flow=USHRT_MAX;
	if(max_flow.size() > 0) {
		flow_by_position.resize(max_flow.back()+1,0);
		prev_max_flow=max_flow.back();
	}
	for(int i_read=max_flow.size()-1; i_read >=0; i_read--) {
		uint16_t this_flow = max_flow[i_read];
		assert(this_flow <= prev_max_flow);
		if(this_flow < prev_max_flow) {
			// just transitioned to a new max flow length, so copy previous values
			for(int i_flow=prev_max_flow-1; i_flow >= this_flow; i_flow--)
				flow_by_position[i_flow] = flow_by_position[prev_max_flow];
			prev_max_flow = this_flow;
		}
		flow_by_position[this_flow] += 1;
	}
	if(prev_max_flow > 0 && prev_max_flow < USHRT_MAX) {
		// copy previous values down to first flow
		for(int i_flow=prev_max_flow-1; i_flow >= 0; i_flow--)
			flow_by_position[i_flow] = flow_by_position[prev_max_flow];
	}
}
