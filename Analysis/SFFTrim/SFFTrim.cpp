/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

/*
	Perform adapter trimming and/or quality trimming on all reads in an
	sff file by filling in the clip_adapter_right and clip_qual_right
	fields of the sff read header. Write results to a new sff file.
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

using namespace std;

struct options {
	options()
	: adapter_cutoff(0.0)
	, flow_order("TACGTACGTCTGAGCATCGATCGATGTACAGC") // XDB
	, key("TCAG")
	, adapter("ATCACCGACTGCCCATAGAGAGGCTGAGAC")      // P1 rev comp
	, qual_cutoff(100.0)
	, qual_wsize(30)
	, trim_extra_left(0)
	, min_read_len(8)
	, help(false)
	, in_bead_summary("beadSummary.unfiltered.txt")
	, out_bead_summary("beadSummary.filtered.txt")
	, bead_summary_valid_field("valid")
	, bead_summary_class_field("class")
	, bead_summary_library_row("lib")
	, print(false)
	, closest(false)
	{}

    double adapter_cutoff;
    string flow_order;
    string key;
    string adapter;
	double qual_cutoff;
	int    qual_wsize;
	int    trim_extra_left;
	int    min_read_len;
    bool   help;
    string in_sff;
    string out_sff;
    string in_bead_summary;
    string out_bead_summary;
    string bead_summary_valid_field;
    string bead_summary_class_field;
    string bead_summary_library_row;
	bool   print;
	bool   closest;
};

void     usage();
void     get_options(options* opt, OptArgs& opts);
bool     check_args(options *opt);
uint16_t trim_adapter(const options& opt, sff_t *sff, double& ascore, int32_t& aflow);
uint16_t trim_qual(const options& opt, const sff_t *sff);
int32_t  trim_right(const options& opt, sff_t* sff, int32_t& clip_adapter_left, int32_t& clip_qual_right, int32_t& clip_adapter_right, double& ascore, int32_t& aflow, int32_t &adapter_3prime_trim_len, int32_t &qual_3prime_trim_len);
int32_t  trim_len(int32_t clip_qual_left, int32_t clip_qual_right, int32_t clip_adapter_left, int32_t clip_adapter_right, int32_t &adapter_3prime_trim_len, int32_t &qual_3prime_trim_len);
void updateBeadSummary(const options& opt, int32_t cnt, int32_t nReads, int32_t dropped_by_adapter, int32_t dropped_by_qual);
bool readBeadSummary(ifstream &inFile, map<string, unsigned int> &beadSummaryHeader, vector< vector<string> > &beadSummaryData);
bool writeBeadSummary(ofstream &outFile, map<string, unsigned int> &beadSummaryHeader, vector< vector<string> > &beadSummaryData, unsigned int validFieldNum, unsigned int libRow, int32_t cnt, int32_t nReads, int32_t dropped_by_adapter, int32_t dropped_by_qual);
void init_gc();
inline bool  isgc(int c);
inline float gc(char* bases, uint16_t len);

int main(int argc, const char* argv[])
{
	// Parse command line:
	OptArgs opts;
	opts.ParseCmdLine(argc, argv);
	options opt;
	get_options(&opt, opts);

	// Open input sff file:
	sff_file_t *sff_file_in = sff_fopen(opt.in_sff.c_str(), "rb", NULL, NULL);

	// Count number of reads that retain minimal length after trimming:
	int      cnt                = 0;
	sff_t*   sff                = 0;
	uint32_t nreads             = 0;
	uint32_t dropped_by_adapter = 0;
	uint32_t dropped_by_qual    = 0;
	int32_t  clip_adapter_left  = 0;
	int32_t  clip_qual_right    = 0;
	int32_t  clip_adapter_right = 0;
	int32_t  aflow              = 0;
	double   ascore             = 0.0;

	for(sff=sff_read(sff_file_in); sff; sff_destroy(sff), sff=sff_read(sff_file_in), ++cnt){
		int32_t adapter_3prime_trim_len;
		int32_t qual_3prime_trim_len;
		int32_t tlen = trim_right(opt, sff, clip_adapter_left, clip_qual_right, clip_adapter_right, ascore, aflow, adapter_3prime_trim_len, qual_3prime_trim_len);
		if(tlen >= opt.min_read_len)
			++nreads;
		else if(adapter_3prime_trim_len < opt.min_read_len)
			++dropped_by_adapter;
		else if(qual_3prime_trim_len < opt.min_read_len)
			++dropped_by_qual;

		if(cnt % 10000 == 0) cout << setw(10) << cnt << "\r";
	}

	// Rewind input:
	sff_fclose(sff_file_in);
	sff_file_in = sff_fopen(opt.in_sff.c_str(), "rb", NULL, NULL);

	// Create output file with correct number of reads:
	sff_header_t *new_header = sff_header_clone(sff_file_in->header);
	new_header->n_reads = nreads;
	sff_file_t *sff_file_out = sff_fopen(opt.out_sff.c_str(), "wb", new_header, NULL );

	// Write out the trimmed reads:
	init_gc();
	int printCounter = 0;
	for(sff=sff_read(sff_file_in); sff; sff_destroy(sff), sff=sff_read(sff_file_in), ++printCounter){
		int32_t adapter_3prime_trim_len;
		int32_t qual_3prime_trim_len;
		int32_t tlen = trim_right(opt, sff, clip_adapter_left, clip_qual_right, clip_adapter_right, ascore, aflow, adapter_3prime_trim_len, qual_3prime_trim_len);
		if(tlen >= opt.min_read_len){
			sff_t *out_sff = sff_clone(sff);
			out_sff->rheader->clip_adapter_left  = clip_adapter_left;
			out_sff->rheader->clip_adapter_right = clip_adapter_right;
			out_sff->rheader->clip_qual_right    = clip_qual_right;
			sff_write(sff_file_out, out_sff);
			sff_destroy(out_sff);
		}

		if(opt.print){
			cout << setw(18) << left  << sff->rheader->name->s
			     << setw( 6) << right << clip_qual_right-1
			     << setw( 6) << right << clip_adapter_right-1
			     << setw( 6) << right << sff->rheader->n_bases
				 << setw( 8) << fixed << setprecision(2) << ascore
			     << setw( 6) << right << aflow
				 << setw( 8) << fixed << setprecision(2) << gc(sff->read->bases->s, sff->rheader->n_bases)
				 << endl;
		}

		if(printCounter % 10000 == 0) cout << setw(10) << cnt << "\r";
	}

	// Update bead summary table to reflect any beads that were filtered out after being trimmed short
	updateBeadSummary(opt,cnt,nreads,dropped_by_adapter,dropped_by_qual);


	// Cleanup:
	cout << "writing..." << endl;
	sff_fclose(sff_file_out);
	sff_fclose(sff_file_in);
	cout << endl << "done" << endl;
}

int32_t trim_right(const options& opt, sff_t* sff, int32_t& clip_adapter_left, int32_t& clip_qual_right, int32_t& clip_adapter_right, double& ascore, int32_t& aflow, int32_t &adapter_3prime_trim_len, int32_t &qual_3prime_trim_len)
{
	int32_t clip_qual_left = sff->rheader->clip_qual_left;
	clip_adapter_left = sff->rheader->clip_adapter_left + (opt.trim_extra_left ? opt.trim_extra_left+1:0);
	
	clip_qual_right    = trim_qual(opt, sff);
	clip_adapter_right = trim_adapter(opt, sff, ascore, aflow);

	return trim_len(clip_qual_left, clip_qual_right, clip_adapter_left, clip_adapter_right, adapter_3prime_trim_len, qual_3prime_trim_len);
}

uint16_t trim_qual(const options& opt, const sff_t *sff)
{
	// Optionally perform quality trimming:
	uint16_t clip_qual_right = 0;
	if(opt.qual_cutoff < 100.0){
		ion_string_t *qual = sff->read->quality;
		char         *qbeg = qual->s;
		char         *qend = qbeg + sff->rheader->n_bases;
		char         *clip = QualTrim(qbeg, qend, opt.qual_cutoff, opt.qual_wsize);

		clip_qual_right = clip - qbeg;
	}

	return clip_qual_right;
}

uint16_t trim_adapter(const options& opt, sff_t *sff, double& ascore, int32_t& aflow)
{
	// Optionally perform adapter trimming:
	ascore = opt.adapter_cutoff;
	aflow  = sff->gheader->flow_length;
	uint16_t clip_adapter_right = 0;
	if(opt.adapter_cutoff > 0.0){
		adapter_searcher as(opt.flow_order, opt.key, opt.adapter);
		int num_matches = as.find_matches_sff(sff->read->flowgram, sff->gheader->flow_length, opt.adapter_cutoff);
		if(num_matches){
			adapter_searcher::match match;
			if(opt.closest)
				match = as.pick_closest();
			else
				match = as.pick_longest();
			ascore = match._score;
			aflow  = match._flow;
			clip_adapter_right = as.flow2pos(sff->read->flow_index, sff->read->bases->s, sff->rheader->n_bases, aflow);
		}
	}

	return clip_adapter_right;
}

int32_t trim_len(int32_t clip_qual_left, int32_t clip_qual_right, int32_t clip_adapter_left, int32_t clip_adapter_right, int32_t &adapter_3prime_trim_len, int32_t &qual_3prime_trim_len)
{
	// For purposes of finding the trimmed length, 0 means no trimming.
	// So 0 on the right means infinity on the right:
	clip_qual_right    = clip_qual_right    ? clip_qual_right    : numeric_limits<uint16_t>::max();
	clip_adapter_right = clip_adapter_right ? clip_adapter_right : numeric_limits<uint16_t>::max();

	// Likewise 0 on the left means 1 on the left:
	clip_qual_left    = max(1, clip_qual_left);
	clip_adapter_left = max(1, clip_adapter_left);

	int32_t clip_left = max(clip_adapter_left, clip_qual_left);
	adapter_3prime_trim_len = clip_adapter_right - clip_left + 1;
	qual_3prime_trim_len    = clip_qual_right    - clip_left + 1;

	return min(adapter_3prime_trim_len, qual_3prime_trim_len);
}

void usage()
{
	options tmp_opt;

	cout << endl
	     << "SFFTrim - Perform quality and/or adapter trimming on an sff file." << endl
	     << endl
	     << "options: " << endl
	     << "  -a,--adapter          Reverse complement of adapter sequence" << endl
	     << "  -c,--adapter-cutoff   Cutoff for adapter trimming" << endl
	     << "  -f,--flow-order       Flow order" << endl
	     << "  -h,--help             This message" << endl
	     << "  -i,--in-sff           Input SFF file" << endl
	     << "  -k,--key              Key sequence" << endl
	     << "  -o,--out-sff          Output SFF file" << endl
	     << "  -q,--qual-cutoff      Cutoff for quality trimming" << endl
	     << "  -w,--qual-window-size Window size for quality trimming" << endl
	     << "  -x,--trim-extra-left  Trim a fixed number of additional bases from the 5' end of each read." << endl
	     << "  -m,--min-read-len     Reads trimmed shorter than this are omitted from output" << endl
	     << "  -p,--print-results    Print table summarizing results to stdout" << endl
	     << "  -e,--pick-closest     Use closest candidate match, rather than longest" << endl
	     << endl;
}

void get_options(options* opt, OptArgs& opts)
{
	opts.GetOption(opt->adapter_cutoff, "0.0",           'c', "adapter-cutoff");
	opts.GetOption(opt->flow_order,     opt->flow_order, 'f', "flow-order");
	opts.GetOption(opt->key,            opt->key,        'k', "key");
	opts.GetOption(opt->qual_cutoff,    "100.0",         'q', "qual-cutoff");
	opts.GetOption(opt->qual_wsize,     "30",            'w', "qual-window-size");
	opts.GetOption(opt->trim_extra_left,"0",             'x', "trim-extra-left");
	opts.GetOption(opt->min_read_len,   "8",             'm', "min-read-len");
    opts.GetOption(opt->help,           "false",         'h', "help");
	opts.GetOption(opt->out_sff,        "",              'o', "out-sff");
	opts.GetOption(opt->in_sff,         "",              'i', "in-sff");
	opts.GetOption(opt->adapter,        opt->adapter,    'a', "adapter");
	opts.GetOption(opt->print,          "false",         'p', "print-results");
	opts.GetOption(opt->closest,        "false",         'e', "pick-closest");
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

	if (opt->out_sff == "") {
		cerr << "Error: no output sff" << endl;
		return false;
	}

	return true;
}

void updateBeadSummary(const options& opt, int32_t cnt, int32_t nReads, int32_t dropped_by_adapter, int32_t dropped_by_qual)
{
	ifstream inFile;
	if(opt.in_bead_summary == "") {
		ION_WARN(string("Input bead summary file not specified, unable to update bead summary with reads trimmed to zero length"));
		return;
	} else {
		inFile.open(opt.in_bead_summary.c_str());
		if(inFile.fail()) {
			ION_WARN(string("Unable to open input bead summary file ") + opt.in_bead_summary + string(" for read"));
			return;
		}
        }

	ofstream outFile;
	if(opt.out_bead_summary == "") {
		ION_WARN(string("Output bead summary file not specified, unable to update bead summary with reads trimmed to zero length"));
		return;
	} else {
		outFile.open(opt.out_bead_summary.c_str());
		if(outFile.fail()) {
			ION_WARN(string("Unable to open output bead summary file ") + opt.out_bead_summary + string(" for write"));
			return;
		}
        }

	// Read the original bead summary
	map<string, unsigned int> beadSummaryHeader;
	vector< vector<string> > beadSummaryData;
	bool successfulRead = readBeadSummary(inFile,beadSummaryHeader,beadSummaryData);

	if(!successfulRead) {
		ION_WARN(string("Problem parsing input bead summary file ") + opt.in_bead_summary);
	} else {
		// Check to make sure the field that we want to edit is present
		if(beadSummaryHeader.find(opt.bead_summary_valid_field)==beadSummaryHeader.end()) {
			ION_WARN(string("Did not find header field named ") + opt.bead_summary_valid_field + string(" in file ") + opt.in_bead_summary);
		} else if(beadSummaryHeader.find(opt.bead_summary_class_field)==beadSummaryHeader.end()) {
			ION_WARN(string("Did not find header field named ") + opt.bead_summary_class_field + string(" in file ") + opt.in_bead_summary);
		} else {
			map<string, unsigned int>::iterator it;
			it = beadSummaryHeader.find(opt.bead_summary_valid_field);
			unsigned int validFieldNum = it->second;
			it = beadSummaryHeader.find(opt.bead_summary_class_field);
			unsigned int classFieldNum = it->second;
			bool foundLibRow = false;
			unsigned int libRow = 0;
			for(unsigned int iRow=0; iRow < beadSummaryData.size(); iRow++) {
				if(beadSummaryData[iRow][classFieldNum] == opt.bead_summary_library_row) {
					libRow = iRow;
					foundLibRow = true;
					break;
				}
			}
			if(!foundLibRow) {
				ION_WARN(string("Did not find row for library reads in file ") + opt.in_bead_summary);
			} else {
				// Write the new bead summary
				if(!writeBeadSummary(outFile,beadSummaryHeader,beadSummaryData,validFieldNum,libRow,cnt,nReads,dropped_by_adapter,dropped_by_qual))
					ION_WARN(string("Problem writing bead summary file ") + opt.out_bead_summary);
			}
		}
	}

	inFile.close();
	outFile.close();
}

bool readBeadSummary(ifstream &inFile, map<string, unsigned int> &beadSummaryHeader, vector< vector<string> > &beadSummaryData) {
	string delim = "\t";
	string line;
	unsigned int iLine=0;
	bool firstline = true;
	beadSummaryHeader.clear();
	beadSummaryData.clear();
	while(inFile.good()) {
		// Parse the line
		getline(inFile,line);
		if(line.length() > 0) {
			beadSummaryData.resize(iLine);
			char *token;
			token = strtok((char *) line.c_str(), delim.c_str());
			unsigned int iField=0;
			while(token != NULL) {
				if(firstline) {
					beadSummaryHeader[string(token)] = iField;
				} else {
					beadSummaryData[iLine-1].push_back(string(token));
				}
				iField++;
				token = strtok(NULL, delim.c_str());
			}
			if(firstline)
				firstline=false;
			iLine++;
		}
	}

	bool success = true;
	if(beadSummaryHeader.size() == 0) {
		ION_WARN(string("Problem reading file header"));
		success = false;
	} else {
		for(unsigned int iRow=0; iRow < beadSummaryData.size(); iRow++) {
			if(beadSummaryData[iRow].size() != beadSummaryHeader.size()) {
				ION_WARN(string("All rows should have same number of entries as header"));
				success = false;
				break;
			}
		}
	}

	return(success);
}

bool writeBeadSummary(ofstream &outFile, map<string, unsigned int> &beadSummaryHeader, vector< vector<string> > &beadSummaryData, unsigned int validFieldNum, unsigned int libRow, int32_t cnt, int32_t nReads, int32_t dropped_by_adapter, int32_t dropped_by_qual) {
	string delim = "\t";
	string clipAdapterHeader = "clipAdapter";
	string clipQualHeader = "clipQual";

	vector<unsigned int> beadSummaryHeaderOrder(beadSummaryHeader.size());
	vector<string> beadSummaryHeaderSorted(beadSummaryHeader.size());
	map<string, unsigned int>::iterator it = beadSummaryHeader.begin();
	unsigned int iField=0;
	for( ; iField < beadSummaryHeader.size(); iField++, it++) {
	  beadSummaryHeaderOrder[it->second] = iField;
	  beadSummaryHeaderSorted[it->second] = it->first;
	}

	// Write header
	for(unsigned int iField=0; iField < beadSummaryHeader.size(); iField++) {
		if(iField==validFieldNum) {
			outFile << clipAdapterHeader << delim;
			outFile << clipQualHeader    << delim;
        	}
		outFile << beadSummaryHeaderSorted[iField];
		if(iField < (beadSummaryHeader.size()-1))
			outFile << delim;
	}
	outFile << endl;
	
	// Write Data
	unsigned int success = true;
	for(unsigned int iRow=0; iRow < beadSummaryData.size(); iRow++) {
		for(unsigned int iField=0; iField < beadSummaryHeader.size(); iField++) {
			if(iField==validFieldNum) {
				if(iRow != libRow) {
					outFile << "0" << delim;
					outFile << "0" << delim;
					outFile << beadSummaryData[iRow][iField];
				} else {
					stringstream ss;
					ss << beadSummaryData[iRow][iField];
					int new_valid;
					ss >> new_valid;
					if(ss.fail()) {
						outFile << "0" << delim;
						outFile << "0" << delim;
						outFile << beadSummaryData[iRow][iField];
						success = false;
					} else {
						if(new_valid != (int) cnt) {
							ION_WARN(string("Entry for number of pre-trimmed lib reads does not match what was expected"));
							success = false;
						}
						new_valid -= (int)(dropped_by_adapter + dropped_by_qual);
						if(new_valid != (int) nReads) {
							ION_WARN(string("Entry for number of post-trimmed lib reads does not match what was expected"));
							success = false;
						}
						outFile << dropped_by_adapter << delim;
						outFile << dropped_by_qual    << delim;
						outFile << new_valid;
					}
				}
        		} else {
				outFile << beadSummaryData[iRow][iField];
        		}
			if(iField < (beadSummaryHeader.size()-1))
				outFile << delim;
		}
		outFile << endl;
	}
	
	return(success);
}

static char gc_array[256];

void init_gc()
{
	fill_n(gc_array, 256, 0);
	gc_array['g'] = 1;
	gc_array['G'] = 1;
	gc_array['c'] = 1;
	gc_array['C'] = 1;
}

inline bool isgc(int c)
{
	return gc_array[c];
}

inline float gc(char* bases, uint16_t len)
{
	return 1.0 * count_if(bases, bases+len, isgc) / len;
}



