/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef BFREFERENCE_H
#define BFREFERENCE_H

#include <vector>
#include <string>
#include <armadillo>
#include <map>
#include <algorithm>
#include <utility>

#include "GridMesh.h"
#include "Mask.h"
#include "Image.h"
#include "SampleQuantiles.h"
#include "IonErr.h"
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
    Filtered,
    Reference
  };

  /** Beadfind metric type. */
  enum BufferMeasurement {
    BFLegacy = 0,    // Traditional measurement from back in the days
    BFMaxSd = 1,     // What is the dc offset value at the frame with the most variation
    BFIntMaxSd = 2   // Itegrate over a few frames around the most variation
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
  char GetType(int wellIdx) const { return mWells[wellIdx]; }

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
  void CalcReference(const std::string &datFile, Mask &mask, BufferMeasurement bf_type=BFLegacy);
  void CalcReference(const std::string &datFile, Mask &mask, std::vector<float> &metric);
  void CalcReference(Image &bfImg, Mask &mask, BufferMeasurement bf_type);
  void AdjustForT0(int rowStart, int rowEnd, int colStart, int colEnd,
		   int nCol, std::vector<float> &t0, std::vector<char> &filters,
		   std::vector<float> &metric);
  void CalcShiftedReference(const std::string &datFile, Mask &mask, std::vector<float> &metric, BufferMeasurement bf_type=BFLegacy);
  void CalcShiftedReference(Image &bfImg, Mask &mask, std::vector<float> &metric, BufferMeasurement bf_type);
  void CalcSignalShiftedReference(const std::string &datFile, const std::string &bgFile, Mask &mask, std::vector<float> &metric, float minTraceSd, int bfIntegrationWindow, int bfIntegrationWidth, BufferMeasurement bf_type);
  void CalcDualReference(const std::string &datFile1, const std::string &datFile2, Mask &mask);

  void CalcSignalReference(const std::string &datFile, const std::string &bgFile,
			   Mask &mask, int traceFrame=18);
  void CalcSignalReference2(const std::string &datFile, const std::string &bgFile,
			   Mask &mask, int traceFrame=18);
  void FilterRegionOutliers(Image &bfImg, Mask &mask, float iqrThreshold, 
                            int rowStart, int rowEnd, int colStart, int colEnd);

  void FilterForOutliers(Image &bfImg, Mask &mask, float iqrThreshold, int rowStep, int colStep);
  void FilterOutlierSignalWells(int rowStart, int rowEnd, int colStart,  int colEnd, int chipWidth,
                                arma::Mat<float> &data, std::vector<char> &wells);
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
  void  SetBfMetricVal(size_t wellIdx, float val) { mBfMetric[wellIdx] = val; }
  void FillBfMetric(float v) { std::fill(mBfMetric.begin(), mBfMetric.end(), v); }
  void SetRegionSize(int regionXSize, int regionYSize) { mRegionXSize = regionXSize, mRegionYSize = regionYSize; }
  void GetRegionSize(int &regionXSize, int &regionYSize) {regionXSize = mRegionXSize; regionYSize = mRegionYSize; }
  void SetDoRegional(bool doRegional) { mDoRegionalBgSub = doRegional; }
  bool GetDoRegional() { return mDoRegionalBgSub; }
  void SetIqrOutlierMult(float mult) { mIqrOutlierMult = mult; }
  void SetNumEmptiesPerRegion( int num ) { ION_ASSERT(num > 0, "Must specify positive number of empties.");  mNumEmptiesPerRegion = num; }
  void SetDoSdat(bool _doSdat) { doSdat = _doSdat; }
  void SetT0(std::vector<float> &t0) { mT0 = t0; }
  void SetDebugH5(const std::string &file) { mDebugH5File = file; }
  float GetTraceSd(size_t wIx) { return mTraceSd[wIx]; }
  bool IsThumbnail() { return mIsThumbnail; }
  void SetThumbnail(bool isThumbnail) { mIsThumbnail = isThumbnail; }
  void SetComparatorCorrect(bool comparatorCorrect) { mDoComparatorCorrect = comparatorCorrect; }
 private:
  void CalcAvgNeighborBuffering(int rowStart, int rowEnd, int colStart, int colEnd, Mask &mask, int avg_window,
                                std::vector<float> &metric, std::vector<float> &neighbor_avg);
  void CalcNeighborDistribution(int rowStart, int rowEnd, int colStart, int colEnd, Mask &mask, int avg_window,
                                std::vector<float> &metric, std::vector<float> &ksD, std::vector<float> &ksProb,
                                std::map<std::pair<int,double>,double> &mZDCache);
  inline bool IsOk(size_t wellIx, Mask &m, std::vector<char> &filt) { return ((m[wellIx] & MaskPinned) == 0 && (m[wellIx] & MaskExclude) == 0 && filt[wellIx] != Filtered); }
  bool LoadImage(Image &img, const std::string &fileName);
  void DebugTraces(const std::string &fileName,  Mask &mask, Image &bfImg);
  bool InSpan(size_t rowIx, size_t colIx,
	      const std::vector<int> &rowStarts,
	      const std::vector<int> &colStarts,
	      int span);
  void GetNEigenScatter(arma::Mat<float> &YY, arma::Mat<float> &E, int nEigen);  
  void GetEigenProjection(arma::Mat<float> &data, arma::Col<unsigned int> &goodRows, size_t nEigen, arma::Mat<float> &proj);
  bool doSdat;
  bool mDoRegionalBgSub;
  float mIqrOutlierMult;
  arma::fmat mTraces;
  int mRegionXSize, mRegionYSize;
  int mDcStart;   ///< Start frame for DC offset
  int mDcEnd;     ///< End frame for DC offset
  double mMinQuantile;  ///< Starting quantile to use for reference wells
  double mMaxQuantile;  ///< Ending quantile to use for reference wells
  int mNumEmptiesPerRegion;
  std::string mDebugH5File; ///< If set some debug output data will be written
  std::string mDebugFile; ///< File to dump some debug traces to.
  GridMesh<int> mGrid;  ///< GridMesh defining the different regions of the chip
  std::vector<char> mWells;  ///< Classification of wells as reference, exclude, etc.
  std::vector<float> mBfMetric; ///< Metric calculated by beadfind flow and image code
  std::vector<float> mT0;
  SampleQuantiles<float> mMadSample;
  std::vector<float> mTraceSd;
  bool mIsThumbnail;
  bool mDoComparatorCorrect;
  const static int MIN_OK_WELLS = 10;
  arma::Col<float> EVal;
  arma::Mat<float> Y, X, B, Proj, Cov, EVec, Diff, Copy,  Corr;
};

#endif // BFREFERENCE_H
