/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BFREFERENCE_H
#define BFREFERENCE_H

#include <vector>
#include <string>
#include "GridMesh.h"
#include "Mask.h"
#include "Image.h"

/** 
 * Look for the wells that are buffering the least to use as reference
 * wells for algorithm fitting.
 */
class BFReference {

 public:
  /** Basic well types of interest for reference. */
  enum WellType {
    Unknown,
    Exclude,
    Reference
  };

  /** Basic constructor with default values. See Init() for configuration */
  BFReference();

  /* Accessors */
  int GetDcStart() { return mDcStart; }
  void SetDcStart(int start) { mDcStart = start; }

  int GetDcEnd() { return mDcEnd; }
  void SetDcEnd(int end) { mDcEnd = end; }

  int GetNumRow() { return mGrid.GetRow(); }
  int GetNumCol() const { return mGrid.GetCol(); }
  int GetNumWells() { return mGrid.GetRow() * mGrid.GetCol(); }

  /** Set a debug output file to dump some traces as we rip through file. */
  void SetDebugFile(const std::string &file) { mDebugFile = file; }

  /** Should this well be used as a reference? e.g. are we reasonably
      sure of no buffering due to polymerase, bead, etc. present */
  bool IsReference(int wellIdx) const { return mWells[wellIdx] == Reference; }

  /** Should this well be used as a reference? e.g. are we reasonably
      sure of no buffering due to polymerase, bead, etc. present */
  bool IsReference(int rowIx, int colIx) const { return IsReference(RowColIdx(rowIx, colIx)); }

  /** 
   * Initalize the object by setting the chip size, region to search size and the 
   * quantiles of interest.
   * @param nRow - Number of rows on the chip (Y dimension)
   * @param nCol - Number of columns on the chip (X dimension)
   * @param nRowStep - Size of the regions in row dimesion.
   * @param nColStep - Size of the regions in col dimesion.
   * @param minQ - Minimum buffering quantile to mark as reference. 
   * @param maxQ - Maximum buffering quantile to mark as reference. 
   */
  void Init(int nRow, int nCol, 
	    int nRowStep, int nColStep, 
	    double minQ, double maxQ);

  /**
   * Calculate which wells to use for reference based on buffering
   * relative to neighbors. 
   * @param dataFile - beadfind dat file
   * @param mask - Exclusion mask to use for this chip. 
   */
  void CalcReference(const std::string &datFile, Mask &mask);
  void CalcReference(const std::string &datFile, Mask &mask, std::vector<float> &metric);
  void CalcDualReference(const std::string &datFile1, const std::string &datFile2, Mask &mask);

  void CalcSignalReference(const std::string &datFile, const std::string &bgFile,
			   Mask &mask, int traceFrame=18);
  void CalcSignalReference2(const std::string &datFile, const std::string &bgFile,
			   Mask &mask, int traceFrame=18);

  /**
   * Convert row and column to single index. 
   */
  int RowColIdx(int row, int col) const {
    return row * GetNumCol() + col;
  }

  size_t RowColToIndex(size_t rowIx, size_t colIx) const { 
    return rowIx * GetNumCol() + colIx; 
  }

  /**
   * Rounding for local work @todo - should be some global functions
   */
  static int Round(double x) {
    return (int)(x + .5);
  }

  /**
   * Calculate the wells with the least buffering for a region.
   * @param rStart,rEnd,cStart,cEnd - Rectangle determined by row & column starts ends.
   * @param metric - Buffering metric used to determine well buffering.
   * @param min/max Quantile - Well rankings of bufferings to use for reference.
   * @param wells - classification of wells as reference or not.
   */
  void FillInRegionRef(int rStart, int rEnd, int cStart, int cEnd,
		       std::vector<float> &metric, 
		       double minQuantile, double maxQuantile,
		       std::vector<char> &wells);

  /**
   * Calculate the wells with the least buffering for entire chip.
   * @param wells - classification of wells as reference or not.
   * @param metric - Buffering metric used to determine well buffering.
   * @param grid - Object dividing the chip into regions
   * @param min/max Quantile - Well rankings of bufferings to use for reference.
   */
  void FillInReference(std::vector<char> &wells, 
		       std::vector<float> &metric,
		       GridMesh<int> &grid,
		       double minQuantile,
		       double maxQuantile);
  
  float GetBfMetricVal(size_t wellIdx) const { return mBfMetric[wellIdx]; }
    
  void SetRegionSize(int regionXSize, int regionYSize) { mRegionXSize = regionXSize, mRegionYSize = regionYSize; }
  void GetRegionSize(int &regionXSize, int &regionYSize) {regionXSize = mRegionXSize; regionYSize = mRegionYSize; }
  void SetDoRegional(bool doRegional) { mDoRegionalBgSub = doRegional; }
  bool GetDoRegional() { return mDoRegionalBgSub; }

 private:

  void DebugTraces(const std::string &fileName,  Mask &mask, Image &bfImg);
  bool InSpan(size_t rowIx, size_t colIx,
	      const std::vector<int> &rowStarts,
	      const std::vector<int> &colStarts,
	      int span);
  
  bool mDoRegionalBgSub;
  int mRegionXSize, mRegionYSize;
  int mDcStart;   ///< Start frame for DC offset
  int mDcEnd;     ///< End frame for DC offset
  double mMinQuantile;  ///< Starting quantile to use for reference wells
  double mMaxQuantile;  ///< Ending quantile to use for reference wells
  std::string mDebugFile; ///< File to dump some debug traces to.
  GridMesh<int> mGrid;  ///< GridMesh defining the different regions of the chip
  std::vector<char> mWells;  ///< Classification of wells as reference, exclude, etc.
  std::vector<float> mBfMetric; ///< Metric calculated by beadfind flow and image code
};

#endif // BFREFERENCE_H
