/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Created on: 10/13/2010
//     Author: Keith Moulton
//
// Latest revision:   $Revision: 11029 $
// Last changed by:   $Author: michael.lyons@lifetech.com $
// Last changed date: $Date: 2011-04-08 06:54:09 -0700 (Fri, 08 Apr 2011) $
//

#include <sys/types.h>
#include <dirent.h>
#include <gtest/gtest.h>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include "../SFFTrim/adapter_searcher.h"
#include "../Utils.h"
using namespace std;
class TrimmerTest : public ::testing::Test {
	
protected:
	
	struct config {
		string key;
		string primer;
		string flow_order;
		double cutoff;
		vector<long> primer_starts;
		vector< vector< double > > reads;
		vector<string> reads_base_space;
	};
	
	virtual void SetUp() {
		vector<string> files;
		string dir("./trimdata/");
		get_files(dir, "sim.flow", files);
		
		
		string _s;
		char const row_delim = '\n';
		char const field_delim = '\t';
		
		for (unsigned int j = 0; j < files.size(); j++) {
			
		
			ifstream _sim(files[j].c_str());
			
			config _config;
			fill_config(files[j], dir, _config);
			
			//burn first line
			string brn;
			getline(_sim, brn, row_delim);
			//read rows
			for (string row; getline(_sim, row, row_delim); ) {

				istringstream ss(row);
				unsigned cnt = 0;
				for (string field; getline(ss, field, field_delim); ) {
					if (cnt == 0) {
						_config.reads_base_space.push_back(field);
					}
					else if(cnt == 2) {
						_config.primer_starts.push_back(strtol(field.c_str(), NULL, 10));
						_config.reads.push_back(vector<double>());
					}else if(cnt > 2){
						_config.reads.back().push_back(strtod(field.c_str(), NULL));
					}
					cnt++;
				}
				
				
			}
			
			_configs.push_back(_config);
			_sim.close();
		}
	}
	
	/*equivalent of "ls something*" */
	void get_files(string dir, string file_prefix, vector<string>& files) {
		
		struct dirent *de = NULL;
		DIR *d = NULL;
		d = opendir(dir.c_str());
		ASSERT_TRUE( d != NULL);
		while ((de = readdir(d))) {
			string _f(dir + string(de->d_name));
			if (_f.find(file_prefix) != string::npos) {
				files.push_back(_f);
			}
			
		}
		
	}

	void fill_config(string settings, string dir, config& cfg) {
		/*
		 sim.flow-TACG.adapter-ATCACCGACTGCCCATAGAGAGGCTGAGACTGCCAAGGCACACAGGGGATAGG.txt
		 sim.flow-TACG.adapter-CTGAGACTGCCAAGGCACACAGGGGATAGG.txt
		 sim.flow-TACGTACGTCTGAGCATCGATCGATGTACAGC.adapter-ATCACCGACTGCCCATAGAGAGGCTGAGACTGCCAAGGCACACAGGGGATAGG.txt
		 sim.flow-TACGTACGTCTGAGCATCGATCGATGTACAGC.adapter-CTGAGACTGCCAAGGCACACAGGGGATAGG.txt
		 */
		
		unsigned int pos = settings.find_first_not_of(dir + "sim.flow-");
		unsigned int end = pos;
		while (settings[end++] != '.');
		end--; //just a tad too far
		cfg.flow_order = settings.substr(pos, (end - pos));
		pos = end + 1; //puts as at adapter- in the file name
		pos += 8; //skips adapter-
		end = pos;
		while (settings[end++] != '.');
		end--;
		cfg.primer = settings.substr(pos, (end - pos));
		cfg.key = "TCAG";
		cfg.cutoff = 10.0;
		
	}
		
	vector<config>	_configs;
	
}; //end class def


TEST_F(TrimmerTest, RunSimulatedData) {
	
	for (unsigned int j = 0; j < _configs.size(); j++) {
		
		//cerr << "[TrimmerTest]: simulated reads: " <<  _configs[j].primer_starts.size();
		//cerr << " flow order: " << _configs[j].flow_order << " primer: " << _configs[j].primer << " key: " << _configs[j].key << endl;
		for (unsigned int i = 0; i < _configs[j].primer_starts.size(); i++) {
			
			adapter_searcher as(_configs[j].flow_order, _configs[j].key, _configs[j].primer);
			long as_best_flow = as.search_sff(_configs[j].reads[i].begin(), _configs[j].reads[i].size());	
			//long _best_pos = as.best_pos(<#Ran1 flow_index#>, <#Ran2 base#>, <#int nbases#>)
			long sim_best_flow = _configs[j].primer_starts[i];
			EXPECT_NEAR(as_best_flow, sim_best_flow, 7);// << "_best_flow: " << _best_flow << " expected: " << _configs[j].primer_starts[i];
			
			//double score = (double)as.best_score();
			//EXPECT_LE(score, _configs[j].cutoff);//  << " best score: " << score << " cutoff: " << _configs[j].cutoff;
			
		}
	}	
	
}

/*
int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}*/


