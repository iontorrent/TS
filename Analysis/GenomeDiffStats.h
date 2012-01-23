/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

// subclass to create a second flow from a genome alignment in a parsed sam file

#ifndef GENOMEDIFFSTATS_H
#define GENOMEDIFFSTATS_H

#include "FlowDiffStats.h"



/**
 * Compare different flowgrams implied by the sequence alignments
 * of our reads. Currently takes a conservative approach to assigning
 * blame to differences giving the reads the benefit of the doubt. */
class GenomeDiffStats : public FlowDiffStats {
  
public: 
  /** Constructors are all the same as base.  Duplicate. */
  /** Constructor. What size homopolymers to max out at. */
 GenomeDiffStats(int maxHomopolymers):FlowDiffStats(maxHomopolymers) {}

  /** Constructor. What size homopolymers to max out at. */
 GenomeDiffStats(int maxHomopolymers, unsigned int nCol, unsigned int nRow):FlowDiffStats(maxHomopolymers,nCol,nRow) {}

  /** Constructor as above plus name of column to use for filtering (e.g. q7Len) */
 GenomeDiffStats(int maxH, unsigned int nCol, unsigned int nRow, std::string qLengthCol) : FlowDiffStats(maxH, nCol, nRow, qLengthCol) {}  
  /** Constructor as above plus bin sizes */
 GenomeDiffStats(int maxH, unsigned int nCol, unsigned int nRow, const std::string &qLengthCol, const unsigned int nColBins, const unsigned int nRowBins) : FlowDiffStats(maxH, nCol, nRow, qLengthCol, nColBins, nRowBins) {}
  
  
  /** Destructor - Close files. */
  virtual ~GenomeDiffStats();

  /** Set the file to write the alignments. */
  void SetAlignmentsOut(const std::string &alignOut);

  /** Set the genomic flow out. */
  void SetFlowsOut(const std::string &flowsOut);

  /** Pull out any alignment type '-' characters */
  void FilterSeq(std::string &seqOut, const std::string &seq);

  /** Print out the results from an alignment. */
  void PrintRecord(std::ostream &out,
		   const std::string &name,
		   const std::string &genomeOrig,
		   int gLength,
		   const std::vector<int> &gFlow,

		   const std::string &readOrig,
		   int rLength,
		   const std::vector<int> &rFlow);
		   
  /** If characters are same then extend. */
  bool Advance(char extendG, char extendR, char &currentG, char &currentR);

  /** Fill in the flows based on the paired alignment. Some things will be ambiguous
      but we try to give benefit of the doubt to read. */
  void FillInPairedFlows(int &seqLength,
			 std::vector<int> &gFlow,
			 std::vector<int> &rFlow,
			 const std::string &genome,
			 const std::string &read,
			 int numFlows);
			 
  /** Compare an alignment for certain number of flows. */
  void CompareRead(const std::string &name,
		   const std::string &genomeOrig,
		   const std::string &readOrig,
		   int numFlows);

 protected:

  /** Initialize with max hp length and name of column to use for filtering (e.g. q7Len) */
  void Init(int maxH, const std::string &qLengthCol, unsigned int nCol, unsigned int nRow, const unsigned int nColBins, const unsigned int nRowBins);

  // Output filestreams for writing results.
  std::ofstream mAlignOut;
  std::ofstream mFlowsOut;

};

#endif // GENOMEDIFFSTATS_H
