/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */


#ifndef BAMHELPER_H
#define BAMHELPER_H

#include "api/BamReader.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include "../Analysis/file-io/ion_util.h"

using namespace std;

class MyBamGroup{
public:
	std::vector< std::string > ID;
	std::vector< std::string > FlowOrder;
	std::vector< std::string > KeySequence;
	std::vector< std::string > Description;
	std::vector< std::string > Library;
	std::vector< std::string > PlatformUnit;
	std::vector< std::string > PredictedInsertSize;
	std::vector< std::string > ProductionDate;
	std::vector< std::string > Program;
	std::vector< std::string > Sample;
	std::vector< std::string > SequencingCenter;
	std::vector< std::string > SequencingTechnology;

  std::string errMsg;
  void ReadGroup(char *bamFile);
};

class BamHeaderHelper{
public:
  vector<string> bam_sequence_names;
  vector<string> flow_order_set;

  void GetRefID(BamTools::BamReader &bamReader);
  int IdentifyRefID(string &sequenceName);
  void GetFlowOrder(BamTools::BamReader &bamReader);
};



bool getTagParanoid(BamTools::BamAlignment &alignment, const std::string &tag, int64_t &value);

std::string getQuickStats(const std::string &bamFile, std::map< std::string, int > &keyLen, unsigned int &nFlowFZ, unsigned int &nFlowZM);
bool getNextAlignment(BamTools::BamAlignment &alignment, BamTools::BamReader &bamReader, const std::map<std::string, int> &groupID, std::vector< BamTools::BamAlignment > &alignmentSample, std::map<std::string, int> &wellIndex, unsigned int nSample);


void dna(string &qDNA, const vector<BamTools::CigarOp>& cig, const string& md, string& tDNA);
void padded_alignment(const vector<BamTools::CigarOp>& cig, string& qDNA, string& tDNA, string& pad_query, string& pad_target, string& pad_match, bool isReversed);
void reverse_comp(std::string& c_dna);
std::vector<int> score_alignments(string& pad_source, string& pad_target, string& pad_match );
void OpenMyBam(BamTools::BamReader &bamReader, char *bamFile);

#endif // BAMHELPER_H
