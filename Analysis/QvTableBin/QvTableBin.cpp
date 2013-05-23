/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#include <math.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

#define QVTABLEBIN_VERSION "1.0.0"

using namespace std;

const size_t kNumPredictors = 6;
const unsigned char kMinQuality = 5;  

unsigned char CalculatePerBaseScore(const vector<vector<float> >& phred_thresholds, const vector<unsigned char>& phred_quality, const vector<float>& pred)
{
	size_t num_phred_cuts = phred_quality.size(); // number of rows/lines in the table

	for(size_t i = 0; i < num_phred_cuts; ++i)
	{
		bool valid_cut = true;

		for(size_t k = 0; k < kNumPredictors && valid_cut; ++k) 
		{
			if(pred[k] > phred_thresholds[k][i]) 
			{
				valid_cut = false;
			}
		}

		if(valid_cut)
		{
			return phred_quality[i];
		}
	}

	return kMinQuality;
}

void usage() 
{
	cerr << "QvTableBin - Convert phred_table_file to a binary table file" << endl;
	cerr << "Usage: " << endl
	   << "  QvTableBin input_file [output_file]" << endl;
	exit(1);
}

int main(int argc, const char *argv[]) 
{
	if(argc < 2)
	{
		usage();
	}
	
	if(argc == 2)
	{
		string option = argv[1];
		if("-h" == option)
		{
			usage();
		}
		else if("-v" == option)
		{
			cerr << "QvTableBin version: " << QVTABLEBIN_VERSION << endl;
			usage();
		}
	}

	string phred_table_file = argv[1];	
	string bin_file(phred_table_file);  
	if(argc > 2)
	{
		bin_file = argv[2];
	}
	else
	{
		bin_file += ".binary";
	}

	ifstream ifs;
	ifs.open(phred_table_file.c_str());
	if(!ifs.is_open())
	{
		cerr << "QvTableBin ERROR: can not open file " << phred_table_file << endl;
	}     

    vector<vector<float> >  phred_thresholds(kNumPredictors);			//!< Predictor threshold table, kNumPredictors x num_phred_cuts.
	vector<unsigned char>   phred_quality;				//!< Quality value associated with each predictor cut.
	vector<vector<float> >  phred_cuts(kNumPredictors);	//!< Predictor threshold cut values.

	float temp;
	while(!ifs.eof()) 
	{
		string line;
		getline(ifs, line);

		if (line.empty())
		{
			break;
		}

		if (line[0] == '#')
		{
			continue;
		}

		stringstream strs(line);
		
		for(size_t k = 0; k < kNumPredictors; ++k)
		{
			strs >> temp;
			phred_thresholds[k].push_back(temp);
		}
		strs >> temp; //skip n-th entry
		strs >> temp;
		phred_quality.push_back(temp);
	}

	ifs.close();

	char buf[100];
	ofstream ofs(bin_file.c_str(), ios::out|ios::binary);
	memcpy(buf, &kNumPredictors, 4);
	ofs.write(buf, 4);
	
	size_t tbSz = 1;

	for(size_t i = 0; i < kNumPredictors; ++i)
	{
        vector<float> vtmp(phred_thresholds[i].size());
		vector<float>::iterator it;
		it = unique_copy(phred_thresholds[i].begin(), phred_thresholds[i].end(), vtmp.begin()); 
		sort(vtmp.begin(), it);
		it = unique_copy (vtmp.begin(), it, vtmp.begin());
		vtmp.resize(distance(vtmp.begin(), it));  
		size_t sz = vtmp.size();
		tbSz *= sz;
		memcpy(buf, &sz, 4);
        ofs.write(buf, 4);

		swap(phred_cuts[i], vtmp);
	}

	for(size_t i = 0; i < kNumPredictors; ++i)
	{
		size_t jj = phred_cuts[i].size();
		for(size_t j = 0; j < jj; ++j)
		{
			temp = phred_cuts[i][j];
			memcpy(buf, &temp, 4);
			ofs.write(buf, 4);
		}
	}

	unsigned char* binTable = new unsigned char[tbSz];
	vector<size_t> vind(kNumPredictors, 0);
	size_t n = 0;
	vector<float> pred(kNumPredictors);
	for(size_t i = 0; i < kNumPredictors; ++i)
	{
		pred[i] = phred_cuts[i][vind[i] ];
	}

	binTable[n] = CalculatePerBaseScore(phred_thresholds, phred_quality, pred);
	++n;
	for(; n < tbSz; ++n)
	{
		size_t ii = kNumPredictors - 1;
		++vind[ii];
		while(vind[ii] == phred_cuts[ii].size())
		{
			vind[ii] = 0;
			--ii;
			++vind[ii];
		}
		
		for(size_t i = 0; i < kNumPredictors; ++i)
		{
			pred[i] = phred_cuts[i][vind[i] ];
		}

		binTable[n] = CalculatePerBaseScore(phred_thresholds, phred_quality, pred);
	}

	ofs.write((char*)binTable, tbSz);

    ofs.close();
	delete [] binTable;

	exit(0);
}
