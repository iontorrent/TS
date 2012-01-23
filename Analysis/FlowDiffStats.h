/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

// Base class to generate statistics comparing two sets of flowgrams.
//   Implement a subclass to read from two sets of analyses on the same wells file (BasecallFlowDiff) or
//   an analysis and a genome alignment (GenomeFlowDiff)

#ifndef FLOWDIFFSTATS_H
#define FLOWDIFFSTATS_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "SampleStats.h"
#include "RawWells.h"
#include "Mask.h"
#include "Utils.h"

const int NUM_NUCLEOTIDES = 4;
const unsigned int COL_BIN_SIZE = 50;
const unsigned int ROW_BIN_SIZE = 50;

int GetRowColFromBfmask(const std::string &bfmask, int *nRow, int *nCol);
void determineFlows(std::vector<unsigned int> &seqFlow, const std::string &seq, const std::string &flowOrder); 


/**
 * Compare different flowgrams implied by the sequence alignments
 * of our reads. Currently takes a conservative approach to assigning
 * blame to differences giving the reads the benefit of the doubt. */
class FlowDiffStats {
  
public: 
  
  /** Constructor. What size homopolymers to max out at. */
  FlowDiffStats(int maxHomopolymers);

  /** Constructor. What size homopolymers to max out at. */
  FlowDiffStats(int maxHomopolymers, unsigned int nCol, unsigned int nRow);

  /** Constructor as above plus name of column to use for filtering (e.g. q7Len) */
  FlowDiffStats(int maxH, unsigned int nCol, unsigned int nRow, const std::string &qLengthCol);
  
  /** Constructor as above plus bin sizes */
  FlowDiffStats(int maxH, unsigned int nCol, unsigned int nRow, const std::string &qLengthCol, const unsigned int nColBins, const unsigned int nRowBins);
  
  /** Destructor - Close files. */
  virtual ~FlowDiffStats();

  /** Set the flow order that generated the sequence. */
  void SetFlowOrder(const std::string &flowOrder) { mFlowOrder = flowOrder; determineFlows(mKeyFlows,mKeySeq,mFlowOrder); }

  /** Set the key seqeunce generated the sequence. */
  void SetKeySeq(const std::string &keySeq) { mKeySeq = keySeq; determineFlows(mKeyFlows,mKeySeq,mFlowOrder); }

  /** Set the indexes of the wells we want to analyze. */
  void SetWellToAnalyze(const std::vector<bool> &wells) { mWellsToUse = wells; }
  
  /** Set the file to be writing out the stats. */
  void SetStatsOut(const std::string &statsFile);

  /** Set the file to be writing out the stats. */
  void SetSNROut(const std::string &snrFile);

  /** Set the file to be writing out the stats. */
  void SetBinnedHpSigOut(const std::string &binnedHpSigFile);

  /** Set the file to be writing out the stats. */
  void SetFlowGCOut(const std::string &gcFile);

  /** Set the file to be writing out the stats. */
  void SetFlowErrOut(const std::string &flowErrFile);

  /** Fill in the flows based on the sequence observed */
  int FillInFlows(int &seqLength, std::vector<int> &flow, const std::string &seq, int nFlows=30);

  /** Translate a nuc A,T,G,C to our indexes */
  int NucToInt(char nuc);

  /** Translate a nuc index to a nucleotide character. */
  std::string IntToNuc(int nuc);

  /** Given a flow intensity for a reference and a read, write out to our summary. */
  void RecordDifference(char nuc, int reference, int read);

  /** Print the summary of our stats to the stream provided. */
  void PrintSumStat(int nucIx, std::ostream &out);

  /** Check for illegal flow sequence. */
  bool CheckIllegalFlowSeq(std::vector<int> &flowSeq);


  /** Compare the flowgrams of reference sequence and read sequence .*/
  void CompareFlows(int row, int col, std::vector<int> &reference, std::vector<int> &read);

  /** OBSOLETE. */
  void CompareFlows(const std::string& name, std::vector<int> &reference, std::vector<int> &read);

  /** Get the raw counts for a stat. */
  int GetCount(int nuc, int referenceHp, int readHp) { return mSumStats[nuc][referenceHp][readHp]; }

  void SetGCWindowSize(unsigned int sz);

  void SetRowCol(int row, int col) {
    mRow = row;
    mCol = col;
  }

  void FillInSubset(const std::string &samFile, int minVal,
                    int minRow, int maxRow, 
                    int minCol, int maxCol,
                    std::vector<int32_t> &xSubset, 
                    std::vector<int32_t> &ySubset);

  void SetWellsFile(const std::string &wells, int nRow, int nCol, int maxFlow,
                    const std::vector<int32_t> &xSubset, const std::vector<int32_t> &ySubset);

  void RecordGCStats(int row, int col, std::vector<int> &reference, std::vector<int> &read);

  void RecordFlowHPStats(int row, int col, std::vector<int> &reference, std::vector<int> &read);

  /** Tokenize a line based on "\t" */
  static void ChopLine(std::vector<std::string> &words, const std::string &line, char delim='\t');

  /** Get the next alignment sequence from the Default.sam.parsed file. */
  bool GetNextAlignment(std::string &name, std::string &genomic, std::string &read, int &row, int &col, int minVal);

  // de-munge the read name from a sam parsed file
  bool GetRowColFromName(const std::string &name, int &row, int &col);

  /** Open up a tab delimited file in the Default.sam.parsed file format. */
  void SetAlignmentInFile(const std::string &samfile);

  /** Open up a file and compare all the alignments. */
  void FilterAndCompare(    int numFlows,
			    std::ostream &out,
			    int minVals=50,
			    int minRow=-1,
			    int maxRow=-1,
			    int minCol=-1,
			    int maxCol=-1);


  // Compare a read.  Called from FilterAndCompare
  virtual void CompareRead(const std::string &name,
			   const std::string &genomeOrig,
			   const std::string &readOrig,
			   int numFlows) = 0;

  // Called after FilterAndCompare completes
  virtual void FinishComparison(int numFlows) {}

 protected:

  /** Initialize with max hp length and name of column to use for filtering (e.g. q7Len) */
  void Init(int maxH, unsigned int nCol, unsigned int nRow, unsigned int colBinSize, unsigned int rowBinSize, const std::string &qLengthCol);

  int mCol, mRow;
  
  // Output filestreams for writing results.
  std::ofstream mFlowStatsOut;
  std::ofstream mFlowSNROut;
  std::ofstream mFlowBinnedHpSigOut;
  std::ofstream mFlowErrOut;
  std::ofstream mFlowGCOut;

  // G+C window size and tallies
  unsigned int mGCWindowSize;
  std::vector < std::vector < std::vector <int> > > mGCCounts; // gc% x ref count x read count

  // Number of col & row bins for regionally binned stats
  unsigned int mColBinSize;
  unsigned int mRowBinSize;
  unsigned int mColBins;
  unsigned int mRowBins;
  
  // Maximum number of homopolymers tracked
  int mMaxHomo;

  std::vector< std::vector<int> > mFlowErrNumerator;
  std::vector< std::vector<int> > mFlowErrDenominator;

  // Summary statistics indexed by nucleotide
  std::vector<std::vector<std::vector<int> > > mSumStats; ///< Nuc x TruHP x ObservedHp
  std::string mFlowOrder;
  std::string mKeySeq;
  std::vector<unsigned int> mKeyFlows;

  std::vector< std::vector<SampleStats<float> > > mFlowValues;  // Flow by HP values based on reference

  std::vector< std::vector< std::vector< std::vector< SampleStats<float> > > > > mFlowBinnedHpSig;  // [colBin][rowBin][flow][hpLen]

  std::string mRawWells;
  RawWells *mWells;
  //  std::vector<std::vector<std::vector<float> > > mWellVals; // row x col x flow
  std::vector<bool> mWellsToUse;

  int mBadFlow;
  int mGoodFlow;

  // Column indexes of interest in the alignment file.
  int mNameIdx;
  int mThresholdIdx;
  int mGenomicIdx;
  int mReadIdx;
  int mSlopIdx;
  int mSlopFiltered;
  int mGoodReads;
  
  // Input alignment file.
  std::ifstream mAlignments;

  std::string mQLengthCol; // "q7Len" or "q10Len" or "q17Len" column names from the sam.parsed file

};

#endif // FLOWDIFFSTATS_H
