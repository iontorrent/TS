/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#include <sstream>
#include <stdlib.h>
#include <assert.h>
#include <iomanip>
#include <limits>
#include "GenomeDiffStats.h"
#include "file-io/ion_util.h"

using namespace std;

GenomeDiffStats::~GenomeDiffStats() {
  if (mAlignOut.is_open()) {
      mAlignOut.close();
  }
}

void GenomeDiffStats::SetAlignmentsOut(const std::string &alignOut) {
  mAlignOut.open(alignOut.c_str());
  assert(mAlignOut.is_open());
}

void GenomeDiffStats::SetFlowsOut(const std::string &flowsOut) {
  mFlowsOut.open(flowsOut.c_str());
  assert(mFlowsOut.is_open());
}

void GenomeDiffStats::PrintRecord(std::ostream &out,
				const std::string &name,
				const std::string &genomeOrig,
				int gLength,
				const std::vector<int> &gFlow,

				const std::string &readOrig,
				int rLength,
				const std::vector<int> &rFlow) {
  out << "align: " << name << endl;
  out << "-----------" << endl;
  out << "full alignments:" << endl;
  out << genomeOrig << endl;
  out << readOrig << endl;
  out << endl;
  string g = genomeOrig.substr(0, gLength);
  string r = readOrig.substr(0, rLength);
  out << "portion aligned in flow space: " << endl;
  out << g << endl;
  out << r << endl;
  out << endl;
  for (unsigned int i = 0; i < gFlow.size(); i++) {
    out << gFlow[i];
  }
  out << endl;
  for (unsigned int i = 0; i < gFlow.size(); i++) {
    if (gFlow[i] == rFlow[i]) {
      out << "|";
    }
    else {
      out << " ";
    }
  }
  out << endl;
  for (unsigned int i = 0; i < rFlow.size(); i++) {
    out << rFlow[i];
  }
  out << endl;
  for (unsigned int i = 0; i < gFlow.size(); i++) {
    int x = (mKeyFlows.size() + i) % mFlowOrder.size();
    out << mFlowOrder[x];
  }
  out << endl;
  for (unsigned int i = 0; i < gFlow.size(); i++) {
    out << i % 10;
  }
  out << endl;
}



bool GenomeDiffStats::Advance(char extendG, char extendR, char &currentG, char &currentR) {
  char localR = currentR;
  char localG = currentG;
  if(extendG == localG && extendR == localR) {
    return true;
  }
  if(extendG != localG && localG != '-' && extendG != '-') {
    return false;
  }
  if(extendR != localR && localR != '-' && extendR != '-') {
    return false;
  }
  return true;
}

void GenomeDiffStats::FillInPairedFlows(int &seqLength,
				      std::vector<int> &gFlow,
				      std::vector<int> &rFlow,
				      const std::string &genome,
				      const std::string &read,
				      int numFlows) {
  gFlow.resize(numFlows);
  rFlow.resize(numFlows);
  int seqIx = 0;
  int flowIx = 0;
  while (seqIx < (int)genome.length() && seqIx < (int)read.length() && flowIx < numFlows) {
    char gC = genome[seqIx];
    char rC = read[seqIx];
    int gBC = 0;
    int rBC = 0;
    int winStart = seqIx;
    int winEnd = winStart;
    // Count how many of the same characters we have in either alignment
    while ((winEnd < (int)genome.length() && winEnd < (int)read.length()) && 
	   Advance(genome[winEnd],read[winEnd], gC, rC)) {
      if (gC == '-') {
	gC = genome[winEnd];
      }
      if (rC == '-') {
	rC = read[winEnd];
      }
      winEnd++;
    }

    // Count bases in the window.
    for (int i = winStart; i < winEnd; i++) {
      if (genome[i] != '-') {
	gBC++;
      }
      if (read[i] != '-') {
	rBC++;
      }
    }
      
    // If different bases fill in the flows and then skip to the next base;
    if (rC != gC) {
      bool rSeen = false;
      bool gSeen = false;
      while (flowIx < numFlows && (!rSeen || !gSeen) ) {
	int nucIx = (mKeyFlows.size() + flowIx) % mFlowOrder.size();
	if (gC == mFlowOrder[nucIx] || (gC == '-' && !gSeen)) {
	  gFlow[flowIx] = gBC;
	  gSeen = true;
	}
	else {
	  gFlow[flowIx] = 0;
	}
	if (rC == mFlowOrder[nucIx] || (rC == '-' && !rSeen)) {
	  rFlow[flowIx] = rBC;
	  rSeen = true;
	}
	else {
	  rFlow[flowIx] = 0;
	}
	flowIx++;
      }
      seqIx = winEnd;
    }
    // Otherwise when we get to the right flow then fill in results
    else {
      while (flowIx < numFlows) {
	int nucIx = (mKeyFlows.size() + flowIx) % mFlowOrder.size();
	if (gC == mFlowOrder[nucIx]) {
	  gFlow[flowIx] = gBC;
	  rFlow[flowIx] = rBC;
	  seqIx = winEnd;
	  flowIx++;
	  break;
	}
	else {
	  gFlow[flowIx] = 0;
	  rFlow[flowIx] = 0;
	  flowIx++;
	}
      }
    }
  }
  seqLength = seqIx;
}

void GenomeDiffStats::CompareRead(const std::string &name,
				  const std::string &genomeOrig,
				  const std::string &readOrig,
				  int numFlows) {
  vector<int> gFlow, rFlow;
  int length = -1;
  FillInPairedFlows(length, gFlow, rFlow, genomeOrig, readOrig, numFlows);

  int row = -1, col=-1;
  GetRowColFromName(name, row, col);

  CompareFlows(row, col, gFlow, rFlow);
  if ( mFlowsOut.good() ) {
    mFlowsOut << name;
    for (unsigned int i = 0; i < gFlow.size(); i++) {
      mFlowsOut << "\t" << gFlow[i];
    }
    mFlowsOut << endl;
  }
  if (mAlignOut.good()) {
    PrintRecord(mAlignOut, name, genomeOrig, length, gFlow, readOrig, length, rFlow);
  }
}



