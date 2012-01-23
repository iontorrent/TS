/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

// subclass to compare two runs on the same chip

#ifndef SFFDIFFSTATS_H
#define SFFDIFFSTATS_H

#include "FlowDiffStats.h"
#include "SFFWrapper.h"
#include <map>


/**
 * Compare different flowgrams implied by the sequence alignments
 * of our reads. Currently takes a conservative approach to assigning
 * blame to differences giving the reads the benefit of the doubt. */
class SffDiffStats : public FlowDiffStats {
  
public: 
  
  /** Constructor with filenames of two runs */
  SffDiffStats(int maxH, unsigned int nCol, unsigned int nRow, const std::string &qLengthCol, const std::string &run1, const std::string &run2);

  /** Constructor as above plus bin sizes */
  SffDiffStats(int maxH, unsigned int nCol, unsigned int nRow, const std::string &qLengthCol, const std::string &run1, const std::string &run2, const unsigned int nColBins, const unsigned int nRowBins);

  /** Set the genomic flow out. */
  void SetPairedOut(const std::string &pairedOut);

  /** Destructor - Close files. */
  virtual ~SffDiffStats();
  void CompareRead(const std::string &name,
		   const std::string &genomeOrig,
		   const std::string &readOrig,
		   int numFlows);
  void FinishComparison(int numFlows);

 protected:

  void Init(const std::string &run1, const std::string &run2);

  std::string sffFileName1, sffFileName2;

  // option stream to write detailed alignment between runs
  std::ofstream mPairedOut;

  // a entry for each read from the sam alignment that passes threshold
  std::map<std::string, bool> readsToCompare;

  bool ReadNextEntry(SFFWrapper& sffW, const sff_t* &read, int& row, int& col);

  // converts the flow_index to a flowspace representation, removing key
  void IndexToFlowspace(std::vector<uint8_t>& flow_index, std::vector<int>& flowspace);

};

#endif // SFFDIFFSTATS_H
