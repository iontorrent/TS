/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef T0CALCMT_H
#define T0CALCMT_H

#include <iostream>
#include "GridMesh.h"
#include "FindSlopeChange.h"
#include "T0Model.h"
#include "PJobQueue.h"
#include "Mask.h"
#include "Traces.h"

class T0CalcMt;

/** Job class to calculate the t0 on accumulated traces. */
class T0RegionJob : public PJob {
public:
  void Init(T0CalcMt *calc, size_t startBin, size_t endBin, int windowSize, 
            float minFirstSlope, float maxFirstSlope,
            float minSecondSlope, float maxSecondSlope, 
            std::vector<float> *timestamps) {
    mCalc = calc;
    mStartBin = startBin;
    mEndBin = endBin;
    mMaxFirstSlope = maxFirstSlope;
    mMinFirstSlope = minFirstSlope;
    mMaxSecondSlope = maxSecondSlope;
    mMinSecondSlope = minSecondSlope;
    mWindowSize = windowSize;
    mTimeStamps = timestamps;
    mFinder.SetWindowSize(mWindowSize);
    /* mFinder.SetMaxFirstHingeSlope(mMaxFirstSlope); */
    /* mFinder.SetMinFirstHingeSlope(mMinFirstSlope); */
    mFinder.SetFirstSlopeRange(mMinFirstSlope, mMaxFirstSlope);
    mFinder.SetSecondSlopeRange(mMinSecondSlope, mMaxSecondSlope);
  }

  void Run();
  T0CalcMt *mCalc;
  size_t mStartBin, mEndBin;
  T0Finder mFinder;
  float mMaxSecondSlope;
  float mMinSecondSlope;
  float mMaxFirstSlope;
  float mMinFirstSlope;
  int mWindowSize;
  std::vector<float> *mTimeStamps;
};

/** Job class to sum up the dc offset region frame averages. */
class T0AccumulateJob : public PJob {
public:
  void Init(T0CalcMt *calc, short *data, size_t startBin, size_t endBin) {
    mCalc = calc;
    mData = data;
    mStartBin = startBin;
    mEndBin = endBin;
  }

  void Run();
  T0CalcMt *mCalc;
  short *mData;
  size_t mStartBin, mEndBin;
};

/** Some ideas about where we think t0 should be for a region. */
class T0Prior {

public:
  T0Prior() {
    mFrameStart = 0;
    mFrameEnd = std::numeric_limits<size_t>::max();
    mT0Prior = 0;
    mT0Weight = 0;
  }

  size_t mFrameStart, mFrameEnd; ///< Range start and end to investigate.
  double mT0Prior;
  double mT0Weight;
};

/** 
    Multithreaded class to hold information and algorithms for
    calculating time zero (when nuc hits well) for a region. 
*/
class T0CalcMt {

 public:

  T0CalcMt() {
    mMask = NULL;
    mRow = mCol = mFrame = mRowStep = mColStep = 0;
    mWindowSize = 6; 
    mMaxFirstHingeSlope = 0;
    mMinFirstHingeSlope = 0;
    mMaxSecondHingeSlope = 0;
    mMinSecondHingeSlope = 0;
  }

  void SetTimeStamps(int *timestamps, int size) {
    assert(size = mFrame);
    mTimeStamps.resize(size);
    copy(&timestamps[0], &timestamps[0] + size, mTimeStamps.begin());
  }

  void SetWindowSize(int size) { mWindowSize = size; }
  void SetMaxFirstHingeSlope(float maxSlope) { mMaxFirstHingeSlope = maxSlope; }
  void SetMinFirstHingeSlope(float minSlope) { mMinFirstHingeSlope = minSlope; }
  void SetMaxSecondHingeSlope(float maxSlope) { mMaxSecondHingeSlope = maxSlope; }
  void SetMinSecondHingeSlope(float minSlope) { mMinSecondHingeSlope = minSlope; }

  void SetMask(Mask *mask) { mMask = mask; }

  /** Setup our t0 calculator. */
  void Init(size_t nRow, size_t nCol, size_t nFrame,
	    size_t nRowStep, size_t nColStep, size_t nThreads) {
    mRow = nRow;
    mCol = nCol;
    mFrame = nFrame;
    mRowStep = nRowStep;
    mColStep = nColStep;
    /* Init our grids. */
    mRegionSum.Init(mRow, mCol, mRowStep, mColStep);
    mT0.Init(mRow, mCol, mRowStep, mColStep);
    mT0Prior.resize(mRegionSum.GetNumBin());

    /* Set some default values. */
    size_t numBin = mRegionSum.GetNumBin();
    for (size_t bIx = 0; bIx < numBin; bIx++) {
      float &t0 = mT0.GetItem(bIx);
      t0 = 0;
      std::pair<size_t, std::vector<float> > &bin = mRegionSum.GetItem(bIx);
      bin.first = 0;
      bin.second.resize(mFrame);
      std::fill(bin.second.begin(), bin.second.end(), 0.0);
    }
    mQueue.Init(nThreads, numBin);
  }

  /** Currently a stub, should be checking for pinned or sd pixels */
  bool isOk(size_t wIx) {
    if(mMask == NULL || !(mMask->Match(wIx, MaskPinned) || mMask->Match(wIx, MaskExclude) || mMask->Match(wIx, MaskIgnore))) {
      return true;
    }
    return false;
  }

  /** Algorithm to fit t0 for this trace using a two piece linear model. */
  static void CalcT0(T0Finder &finder, std::vector<float> &trace, 
                     std::vector<float> &timestamps,
                     T0Prior &prior, float &t0) {
    int frameEnd = min( prior.mFrameEnd, trace.size());
    int frameStart = max( prior.mFrameStart, 0ul);
    finder.SetSearchRange(frameStart, frameEnd);
    bool ok = finder.FindT0Time(&trace[0], &timestamps[0], trace.size());
    t0 = finder.GetT0Est();
    if (ok) {
      t0 = (prior.mT0Weight * prior.mT0Prior) + t0;
      t0 = t0 / (prior.mT0Weight + 1);
    }
    else {
      t0 = -1;
    }
  }

  void CalcIndividualT0(std::vector<float> &t0, int useMeshNeighbors) {
    std::vector<double> dist(7);
    std::vector<float *> values;

    t0.resize(mRow*mCol);
    fill(t0.begin(), t0.end(), -1);
    for (size_t rowIx = 0; rowIx < mRow; rowIx++) {
      for (size_t colIx = 0; colIx < mCol; colIx++) {
        int idx = rowIx * mCol + colIx;
        mT0.GetClosestNeighbors(rowIx, colIx, useMeshNeighbors, dist, values);
        double distWeight = 0;
        double startX = 0;
        for (size_t i = 0; i < values.size(); i++) {
          if (*(values[i]) > 0) {
            double w = Traces::WeightDist(dist[i]); //1/sqrt(dist[i] + 1);
            distWeight += w;
            startX += w * (*(values[i]));
          }
        }
        if (distWeight > 0 && startX >= 0) {
          t0[idx]  = startX / distWeight;
        }
      }
    }
  }

  /** Calculate the t0 from accumulated traces. */
  void CalcT0FromSum() {
    size_t stepSize = mRegionSum.GetNumBin() / mQueue.NumThreads();
    size_t numBin = mRegionSum.GetNumBin();
    size_t numJobs = ceil((double)numBin / stepSize);
    std::vector<T0RegionJob> jobs(numJobs);
    for (size_t jobIx = 0; jobIx < numJobs; jobIx++) {
      jobs[jobIx].Init(this, jobIx * stepSize, min(numBin, (jobIx + 1) * stepSize), 
                       mWindowSize, mMinFirstHingeSlope, mMaxFirstHingeSlope,
                       mMinSecondHingeSlope, mMaxSecondHingeSlope, &mTimeStamps);
      mQueue.AddJob(jobs[jobIx]);
    }
    mQueue.WaitUntilDone();
  }
  
  /** Sum up the traces for the regions. */
  void CalcAllSumTrace(short *data) {
    size_t stepSize = mRegionSum.GetNumBin() / mQueue.NumThreads();
    size_t numBin = mRegionSum.GetNumBin();
    size_t numJobs = ceil((double)numBin / stepSize);
    std::vector<T0AccumulateJob> jobs(numJobs);
    for (size_t jobIx = 0; jobIx < numJobs; jobIx++) {
      jobs[jobIx].Init(this, data, jobIx * stepSize, min(numBin, (jobIx + 1) * stepSize));
      mQueue.AddJob(jobs[jobIx]);
    }
    mQueue.WaitUntilDone();
  }

  /** Sum up the traces for this region. */
  void CalcSumTrace(const short *data, 
		    int rowStart, int rowEnd,
		    int colStart, int colEnd) {
    int frameIx = 0;
    /* 
       Loop through first frame to count wells in each region. This
       loop is optimized to loop through the memory in order of the
       frames in memory while minimizing the number of lookups into
       the regionSum grid (once every mColStep)
    */
    for (int rowIx = rowStart; rowIx < rowEnd; rowIx++) {
      for (int colIx = colStart; colIx < (int)colEnd; colIx+= mColStep) {
        int wIx = rowIx * mCol + colIx;
	int end = min(mCol, mColStep + colIx); 
	std::pair<size_t, std::vector<float> > &bin = mRegionSum.GetItemByRowCol(rowIx, colIx);
	for (int cIx = colIx; cIx < end; cIx++) {
	  if (isOk(wIx)) {
	    bin.first++;
	    bin.second[frameIx] += (float)data[wIx] - data[wIx];
          }
	  wIx++;
	}
      }
    }

    /* Loop through rest of the frames. */
    for (int frameIx = 1; frameIx < (int)mFrame; frameIx++) {
      for (int rowIx = rowStart; rowIx < rowEnd; rowIx++) {
        for (int colIx = colStart; colIx < (int)colEnd; colIx+= mColStep) {
        int fwIx = rowIx * mCol + colIx;
        int wIx = fwIx + frameIx * mRow * mCol;
        //for (int colIx = 0; colIx < (int)mCol; colIx+= max(1ul,min(mColStep, mCol-colIx))) {
	  int end = min(mCol, mColStep + colIx); 
	  std::pair<size_t, std::vector<float> > &bin = mRegionSum.GetItemByRowCol(rowIx, colIx);
	  for (int cIx = colIx; cIx < end; cIx++) {
	    if (isOk(fwIx)) {
	      bin.second[frameIx] += (float) data[wIx] - data[fwIx];
	    }
	    wIx++;
            fwIx++;
	  }
	}
      }
    }
  }

  /** Write out our fits and average trace as a text file if requested. */
  void WriteResults(std::ostream &out) {
    int rowStart, rowEnd, colStart, colEnd;
    for (int bIx = 0; bIx < (int)mRegionSum.GetNumBin(); bIx++) {
      float t0 = mT0.GetItem(bIx);
      int t0Frame = GetFrameForTime(t0);
      mRegionSum.GetBinCoords(bIx, rowStart, rowEnd, colStart, colEnd);
      std::pair<size_t, std::vector<float> > &vec = mRegionSum.GetItem(bIx);
      out << rowStart << '\t' << rowEnd << '\t' << colStart << '\t' << colEnd << '\t' << vec.first << '\t' << t0 << '\t' << t0Frame;
      for (size_t i = 0; i < vec.second.size(); i++) {
        out << '\t' << vec.second.at(i);
      }
      out << std::endl;
    }
  }
  
  float GetT0(int bin) { return mT0.GetItem(bin); }

  float GetT0(int row, int col) { return mT0.GetItemByRowCol(row, col); }

  int GetFrameForTime(float time) {
    if (time <= 0) {
      return -1;
    }
    for (size_t i = 0; i < mTimeStamps.size() - 1; i++) {
      if (time < mTimeStamps[i+1]) {
        return i;
      }
    }
    assert(0);
    return -1;
  }


  int GetNumRegions() { return mT0.GetNumBin(); }
  
  void GetRegionCoords(int regionIdx,int &rowStart, int &rowEnd, int &colStart, int &colEnd) {
    return mT0.GetBinCoords(regionIdx, rowStart, rowEnd, colStart, colEnd);
  }

  /** Pointer to the mask characterizing wells. */
  Mask *mMask;
  /** Dimensions of chip. */ 
  size_t mRow, mCol, mFrame;
  /** Timestamps for each frame. */
  std::vector<float> mTimeStamps; 
  /** Region height and width */
  size_t mRowStep, mColStep;
  /** Actual t0 as calculated. */
  GridMesh<float> mT0;
  /** Prior for t0 */
  std::vector<T0Prior>  mT0Prior;
  /** Pair of number of wells seen and vector average of the region. */
  GridMesh<std::pair<size_t, std::vector<float> > > mRegionSum;
  /** Pthreads for multithreading. */
  PJobQueue mQueue;
  /** Window size for edges of the hinges. */
  int mWindowSize; 
  /* Constraints for slopes on first and second portion of hinge. */
  float mMaxFirstHingeSlope;
  float mMinFirstHingeSlope;
  float mMaxSecondHingeSlope;
  float mMinSecondHingeSlope;

};


#endif // T0CALCMT_H

