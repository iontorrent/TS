/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/*
	Search an sff file for a fixed nucleotide sequence.
*/

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <limits>
#include <string.h>
#include "sff.h"
#include "sff_file.h"
#include "sff_header.h"
#include "QScoreTrim.h"
#include "OptArgs.h"
#include "adapter_searcher.h" 
#include "IonErr.h"
#include "ion_util.h"

using namespace std;

struct options {
	options()
	: cutoff(0.0)
	, flow_order("TACGTACGTCTGAGCATCGATCGATGTACAGC") // XDB
	, key("TCAG")
	, query("ATCACCGACTGCCCATAGAGAGGCTGAGAC")        // P1 rev comp
	, help(false)
	, print(false)
	, closest(false)
	{}

    double cutoff;
    string flow_order;
    string key;
    string query;
    bool   help;
    string in_sff;
	bool   print;
	bool   closest;
};

void usage();
void get_options(options* opt, OptArgs& opts);
bool check_args(options *opt);

int main(int argc, const char* argv[])
{
	// Parse command line:
	OptArgs opts;
	opts.ParseCmdLine(argc, argv);
	options opt;
	get_options(&opt, opts);

	// Open input sff file:
	sff_file_t *sff_file_in = sff_fopen(opt.in_sff.c_str(), "rb", NULL, NULL);

	int    cnt = 0;
	sff_t* sff = 0;
	for(sff=sff_read(sff_file_in); sff; sff_destroy(sff), sff=sff_read(sff_file_in), ++cnt){
		int32_t base  = -1;
		int32_t flow  = -1;
		double  score = -1.0;
		adapter_searcher as(opt.flow_order, opt.key, opt.query);
		int num_matches = as.find_matches_sff(sff->read->flowgram, sff->gheader->flow_length, opt.cutoff);
		if(num_matches){
			adapter_searcher::match match;
			if(opt.closest)
				match = as.pick_closest();
			else
				match = as.pick_longest();
			score = match._score;
			flow  = match._flow;
			base  = as.flow2pos(sff->read->flow_index, sff->read->bases->s, sff->rheader->n_bases, flow);
		}

		int row = 0;
		int col = 0;
		ion_readname_to_rowcol(sff->rheader->name->s, &row, &col);
		cout << setw(8) << row
		     << setw(8) << col
		     << setw(6) << base
		     << setw(6) << sff->rheader->n_bases
			 << setw(8) << fixed << setprecision(2) << score
		     << setw(6) << flow
			 << endl;

		if(cnt % 10000 == 0) cerr << setw(10) << cnt << "\r";
	}

	// Cleanup:
	sff_fclose(sff_file_in);
	cerr << endl << "done" << endl;
}

void usage()
{
	options tmp_opt;

	cout << endl
	     << "SFFSearch - Search an sff file for a fixed nucleotide sequence."
	     << endl
	     << "options: " << endl
	     << "  -q,--query            Query sequence" << endl
	     << "  -c,--cutoff           Cutoff for score for declaring a match" << endl
	     << "  -i,--in-sff           Input SFF file" << endl
	     << "  -f,--flow-order       Flow order" << endl
	     << "  -k,--key              Key sequence" << endl
	     << "  -e,--pick-closest     Use closest candidate match, rather than longest" << endl
	     << "  -h,--help             This message" << endl
	     << endl;
}

void get_options(options* opt, OptArgs& opts)
{
	opts.GetOption(opt->query,          opt->query,      'q', "query");
	opts.GetOption(opt->cutoff,         "0.0",           'c', "cutoff");
	opts.GetOption(opt->in_sff,         "",              'i', "in-sff");
	opts.GetOption(opt->flow_order,     opt->flow_order, 'f', "flow-order");
	opts.GetOption(opt->key,            opt->key,        'k', "key");
	opts.GetOption(opt->closest,        "false",         'e', "pick-closest");
    opts.GetOption(opt->help,           "false",         'h', "help");
	opts.CheckNoLeftovers();
	
	if (!check_args(opt)) {
		usage();
		exit(1);
	}
}

bool check_args(options *opt)
{
	if (opt->help) {
		return false;
	}

	if (opt->in_sff == "") {
		cerr << "Error: no input sff" << endl;
		return false;
	}

	return true;
}

