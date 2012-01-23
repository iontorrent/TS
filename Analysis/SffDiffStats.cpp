/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#include <sstream>
#include <stdlib.h>
#include <assert.h>
#include <iomanip>
#include <limits>

#include "SffDiffStats.h"
#include "file-io/ion_util.h"
#include "file-io/sff_sort.h"

using namespace std;

SffDiffStats::SffDiffStats(int maxH, unsigned int nCol, unsigned int nRow, 
			   const std::string &qLengthCol, const std::string &run1, 
			   const std::string &run2) : FlowDiffStats(maxH, nCol, nRow, qLengthCol)
{
  Init(run1, run2);
}

SffDiffStats::SffDiffStats(int maxH, unsigned int nCol, unsigned int nRow, const std::string &qLengthCol, const std::string &run1, const std::string &run2, const unsigned int nColBins, const unsigned int nRowBins) : FlowDiffStats(maxH, nCol, nRow, qLengthCol, nColBins, nRowBins) {
  Init(run1, run2);
}

SffDiffStats::~SffDiffStats() {
}

// For efficiency sake I just make a note of the name and I'll
// do the actual comparisons when I iterate in parallel through
// the two sorted SFFs.
void SffDiffStats::CompareRead(const std::string &name,
			       const std::string &genomeOrig,
			       const std::string &readOrig,
			       int numFlows) {
  readsToCompare[name] = true;
}

bool SffDiffStats::ReadNextEntry(SFFWrapper& sffW, const sff_t* &aread, int& row, int& col) {
  bool success;
  aread = sffW.LoadNextEntry(&success);  
  if (!aread) success = false;
  if (success)
    GetRowColFromName(std::string(sff_name(aread)), row, col);
  else
    row = col = -1;
  return success;
}

void SffDiffStats::IndexToFlowspace(std::vector<uint8_t>& flow_index, std::vector<int>& flowspace) {

  // skip passed the key sequence
  unsigned int i = mKeySeq.size();
  while (flow_index[i]==0) { i++; }  // additional bases that match the last nuc of the key

  // encode as number of nuc incorporations at each flow
  for (; i < flow_index.size(); i++) {

    int this_flow_index = flow_index[i];

    while (this_flow_index--)
      flowspace.push_back(0);

    flowspace.back()++;

  }
} 


// Sort the two SFF files.  Iterate through each read.
// If the read passed the filters, then compare flows.
void SffDiffStats::FinishComparison(int numFlows) {

  cout << "Filtering complete." << endl;
  cout << "Sorting SFF file " << sffFileName1 << endl;

  // Sort the two sff files so we can iterate together
  sff_file_t* sff1 = sff_fopen(sffFileName1.c_str(), "rbi", NULL, NULL);
  char fTemplate1[256] = { 0 };
  sprintf(fTemplate1, "/tmp/sff_%d_XXXXXX", getpid()); 
  int tmp1fd = mkstemp(fTemplate1);
  sff_file_t* sff1Sorted = sff_fdopen(tmp1fd, "wbi", sff1->header, sff1->index);
  sff_sort(sff1, sff1Sorted);
  sff_fclose(sff1Sorted);

  cout << "Sorting SFF file " << sffFileName2 << endl;

  sff_file_t* sff2 = sff_fopen(sffFileName2.c_str(), "rbi", NULL, NULL);
  char fTemplate2[256] = { 0 };
  sprintf(fTemplate2, "/tmp/sff_%d_XXXXXX", getpid()); 
  int tmp2fd = mkstemp(fTemplate2);
  sff_file_t* sff2Sorted = sff_fdopen(tmp2fd, "wbi", sff2->header, sff2->index);
  sff_sort(sff2, sff2Sorted);
  sff_fclose(sff2Sorted);
  
  SFFWrapper sff1W, sff2W;
  sff1W.OpenForRead(fTemplate1);
  sff2W.OpenForRead(fTemplate2);

  SetFlowOrder(sff1W.GetHeader()->flow->s);
  SetKeySeq(sff1W.GetHeader()->key->s);

  bool sff1ReadOK, sff2ReadOK;
  int row1, col1;
  int row2, col2;
  const sff_t* read1;
  const sff_t* read2;
  
  cout << "Collecting alignment statistics" << endl;

  sff1ReadOK = ReadNextEntry(sff1W, read1, row1, col1);
  sff2ReadOK = ReadNextEntry(sff2W, read2, row2, col2);

  while (sff1ReadOK && sff2ReadOK) {

    // advance just one or the other if the entry row/col don't match 
    if (row1 < row2 || (row1 == row2 && col1 < col2)) {
      sff1ReadOK = ReadNextEntry(sff1W, read1, row1, col1);
      continue;
    }

    if (row2 < row1 || (row1 == row2 && col2 < col1)) {
      sff2ReadOK = ReadNextEntry(sff2W, read2, row2, col2);
      continue;
    }

    std::string name1(sff_name(read1));
    std::string name2(sff_name(read2));

    assert(row1 == row2 && col1 == col2);

    if (readsToCompare[name1] || readsToCompare[name2]) {

      // build flowspace representation of the sequence from flow_index
      
      uint8_t* fi1 = sff_flow_index(read1);
      uint8_t* fi2 = sff_flow_index(read2);

      std::vector<uint8_t> fi1vec(fi1, fi1 + sff_n_bases(read1));
      std::vector<uint8_t> fi2vec(fi2, fi2 + sff_n_bases(read2));
    
      std::vector<int> fs1vec;
      std::vector<int> fs2vec;

      IndexToFlowspace(fi1vec, fs1vec);
      IndexToFlowspace(fi2vec, fs2vec);

      if (mPairedOut.is_open()) {
	mPairedOut << name1 << "\t";
	for (unsigned int i=0; i < fs1vec.size(); i++) { if (fs1vec[i] > 9) mPairedOut << 'X'; else mPairedOut << fs1vec[i]; } mPairedOut << endl;
	for (unsigned int t=0; t <= name1.size()/8; t++) { mPairedOut << "\t"; }
	for (unsigned int i=0; i < max(fs1vec.size(),fs2vec.size()); i++) {
	  if (i >= fs1vec.size())
	    mPairedOut << 'v';
	  else if (i >= fs2vec.size())
	    mPairedOut << '^';
	  else {
	    if (fs1vec[i]>fs2vec[i]) mPairedOut << (char)toupper(mFlowOrder[i]); else if (fs1vec[i]<fs2vec[i]) mPairedOut << (char)tolower(mFlowOrder[i]); else mPairedOut << ' ';
	  }
	}
	mPairedOut << endl;
	mPairedOut << name2 << "\t";
	for (unsigned int i=0; i < fs2vec.size(); i++) { if (fs2vec[i] > 9) mPairedOut << 'X'; else mPairedOut << fs2vec[i]; } mPairedOut << endl << endl;
      }

      // only analyze first numFlows
      fs1vec.resize(numFlows);
      fs2vec.resize(numFlows);

      // record the stats
      CompareFlows(row1, col1, fs1vec, fs2vec);
    }

    // advance both
    sff1ReadOK = ReadNextEntry(sff1W, read1, row1, col1);
    sff2ReadOK = ReadNextEntry(sff2W, read2, row2, col2);
  }

  sff1W.Close(); unlink(fTemplate1);
  sff2W.Close(); unlink(fTemplate2);

}

void SffDiffStats::Init(const std::string &run1, 
			const std::string &run2) {
  sffFileName1 = run1;
  sffFileName2 = run2;
}


void SffDiffStats::SetPairedOut(const std::string &pairedOut) {
  mPairedOut.open(pairedOut.c_str());
  assert(mPairedOut.is_open());
}

