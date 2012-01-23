/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "adapter_searcher.h"

using namespace std;

// Test adapter_searcher against test cases genereatd by phaseSim.
// To build this test program:
// 
//     g++ -g -I../ SFFTrimTest.cpp

int main(int argc, char* argv[])
{
	assert(argc == 3);
	float cutoff = strtod(argv[1], 0);
	ifstream in(argv[2]);
	assert(in);

	string order   = "TACGTACGTCTGAGCATCGATCGATGTACAGC";
	string key     = "TCAG";
	string adapter = "ATCACCGACTGCCCATAGAGAGGCTGAGAC";

	string ignore;
	getline(in, ignore);

	for(string line; getline(in,line);){
		istringstream iss(line);
		string        seq;
		int           pos    = 0;
		float         x      = 0.0;
		int           seqnum = 0;
		vector<float> sig;

		iss >> seqnum >> seq >> pos;
		while(iss >> x)
			sig.push_back(x);

		adapter_searcher as(order, key, adapter);
		int   num_match  = as.find_matches(sig.begin(), sig.end(), cutoff);
		int   best_flow  = sig.size();
		float best_score = cutoff;

		if(num_match){
			adapter_searcher::match match = as.pick_longest(); 
			best_flow  = match._flow;
			best_score = match._score;
		}

		// temp for debug:
		//for(adapter_searcher::match_deq::const_iterator m=as._matches.begin(); m!=as._matches.end(); ++m){
		//	cout << setw(6) << m->_flow
		//         << setw(8) << fixed << setprecision(2) << m->_score
		//	     << endl;
		//}

		cout << setw(6) << pos
		     << setw(6) << best_flow
		     << setw(8) << fixed << setprecision(2) << best_score
			 << endl;
	}
}

