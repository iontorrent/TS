/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACECHUNK_H
#define TRACECHUNK_H
#include <stdint.h>
#include <ostream>
#include "TimeCompression.h"
#include "GridMesh.h"
#include "DatCompression.h"
#include "Region.h"
#include "IonErr.h"

#define TC_T0_SLOP 0

// Forward declaration of data structure for serialization
struct FlowChunk;

/**
 * Class representing the frame level well data for a particular
 * region in a particular flow. Called a chunk as it is a data cube of
 * x columns (width) by y rows (height) and z frames (depth)
 */ 
class TraceChunk {

public:

  /** Generic constructor. */
  TraceChunk() {
    Init();
  }

  /** Set everything to defaults. */
  void Init() {
    mRowStart = mColStart = mFrameStart = mFrameStep = 0;
    mChipRow = mChipCol = mChipFrame = mOrigFrames = 0;
    mHeight = mWidth = mDepth = 0;
    mStartDetailedTime = mStopDetailedTime = mLeftAvg = 0;
    mT0 = 0;
    mSigma = 0;
    mTMidNuc = 0;
    mBaseFrameRate = 0;
  }
  
  /** Set the information about the global chip data. */
  void SetChipInfo(size_t chipRow, size_t chipCol, size_t chipFrame) {
    mChipRow = chipRow;
    mChipCol = chipCol;
    mChipFrame = chipFrame;
  }
  
  /** set the dimensions of this data chunk. */
  void SetDimensions(size_t row, size_t height, size_t col, size_t width, size_t frameStart, size_t depth) {
    mRowStart = row;
    mHeight = height;
    mColStart = col;
    mWidth = width;
    mFrameStep = mWidth * mHeight;
    mFrameStart = frameStart;
    mDepth = depth;
    mData.resize(height * width * depth);
    assert(mWidth > 0 && mHeight > 0 && mDepth > 0);
  }

  /** Fill in the data with all zeros. */
  void ZeroData() {
    fill(mData.begin(), mData.end(), 0.0f);
  }

  /** Set the timestamps for the frames. */
  void SetTimeData(size_t origFrames, float t0, int startDetailed, int endDetailed, int leftAvg) {
    mOrigFrames = origFrames;
    mT0 = t0;
    mStartDetailedTime = startDetailed;
    mStopDetailedTime = endDetailed;
    mLeftAvg = leftAvg;
  }

  /** Accessor */
  inline uint16_t & At(size_t chipRow, size_t chipCol, size_t chipFrame) {
    chipFrame = std::min((int)mDepth - 1, (int)chipFrame);
    size_t idx = mFrameStep * chipFrame + (chipRow - mRowStart) * mWidth + (chipCol - mColStart);
    ION_ASSERT(idx < mData.size(), "Outside of bounds");
    return (mData[idx]);
  }

  /** Const accessor. */
  inline const uint16_t & At(size_t chipRow, size_t chipCol, size_t chipFrame) const {
    chipFrame = std::min((int)mDepth - 1, (int)chipFrame);
    size_t idx = mFrameStep * chipFrame + (chipRow - mRowStart) * mWidth + (chipCol - mColStart);
    ION_ASSERT(idx < mData.size(), "Outside of bounds");
    return (mData[idx]);
  }

  /** Const accessor. */
  inline const uint16_t & At(size_t wellIx, size_t chipFrame) const {
    chipFrame = std::min((int)mDepth - 1, (int)chipFrame);
    size_t idx = mFrameStep * chipFrame + wellIx;
    ION_ASSERT(idx < mData.size(), "Outside of bounds");
    return (mData[idx]);
  }
  
  /** accessor. */
  inline uint16_t & At(size_t wellIx, size_t chipFrame)  {
    chipFrame = std::min((int)mDepth - 1, (int)chipFrame);
    size_t idx = mFrameStep * chipFrame + wellIx;
    ION_ASSERT(idx < mData.size(), "Outside of bounds");
    return (mData[idx]);
  }

  /** Accessor by well index. */
  inline uint16_t & At(size_t idx)  {
    ION_ASSERT(idx < mData.size(), "Outside of bounds");
    return (mData[idx]);
  }

  /** Const accessor by well index. */
  inline const uint16_t & At(size_t idx) const {
    ION_ASSERT(idx < mData.size(), "Outside of bounds");
    return (mData[idx]);
  }

  inline void SubDcOffset() { 
    for (size_t row = mRowStart; row < mRowStart + mHeight; row++) {
      for (size_t col = mColStart; col < mColStart + mWidth; col++) {
        SubDcOffset(row, col);
      }
    }
  }

  inline void SubDcOffset(size_t row, size_t col) {
    float m = DcOffset(row, col);
    for (size_t i = 0; i < mTimePoints.size(); i++) {
      uint16_t &val = At(row, col, i);
      val -= m;
    }
  }

  inline float CalcDriftAdjust() {
    double frameRate = mBaseFrameRate/1000.0; 
    double avg[mDepth];
    memset(avg, 0, sizeof(avg[0]) * mDepth);
    double t0 = mT0 * frameRate; // mT0 is in frames at baseframe rate, convert to seconds
    t0 -= TC_T0_SLOP; // a frame and change at 15 frame per seconds
    size_t maxFrame = 0;
    for (size_t i = 0; i < mTimePoints.size() && t0 >= mTimePoints[i]; i++) {
      maxFrame = i;
    }

    for (size_t row = mRowStart; row < mRowStart + mHeight; row++) {
      for (size_t col = mColStart; col < mColStart + mWidth; col++) {
        for (size_t frame = 0; frame < maxFrame; frame++) {
          avg[frame] += (At(row, col, frame)- At(row, col, 0));
        }
      }
    }

    for (size_t frame = 0; frame < maxFrame; frame++) {
      avg[frame] = avg[frame] / (mWidth * mHeight);
    }
    
    double slope = 0;
    double weight = 0;
    for (size_t frame = 1; frame < maxFrame; frame++) {
      double last = (mTimePoints[frame-1] + frame == 1 ? 0 : mTimePoints[frame -2])/2.0f; 
      double current = (mTimePoints[frame] + mTimePoints[frame-1])/2.0;
      double w = (current - last);
      double s = (avg[frame] - avg[frame-1]) / w;
      slope += (s * w);
      weight += w;
    }
    if (weight > 0) {
      slope = slope / weight;
    }
    return slope;
  }

  inline void AdjustForDrift() {
    double slope = CalcDriftAdjust();
    for (size_t frame = 0; frame < mDepth; frame++) {
      float adjust = slope * (mTimePoints[frame] + frame == 0 ? 0 : mTimePoints[frame-1]) / 2.0f;
      for (size_t row = mRowStart; row < mRowStart + mHeight; row++) {
        for (size_t col = mColStart; col < mColStart + mWidth; col++) {
          uint16_t &val = At(row, col, frame);
          val -= adjust;
        }
      }
    }
  }

  inline float DcOffset(size_t row, size_t col) {
    double frameRate = mBaseFrameRate/1000.0; 
    double weight = 0.0;
    double m = 0.0;
    double t0 = mT0 * frameRate; // mT0 is in frames at baseframe rate, convert to seconds
    t0 -= TC_T0_SLOP; // a frame and change at 15 frame per seconds
    for (size_t i = 0; i < mTimePoints.size() && t0 >= mTimePoints[i]; i++) {
      double val = At(row, col, i);
      double last = i == 0 ? 0.0 : mTimePoints[i-1];
      double w = mTimePoints[i] - last;
      //      double w = log(i+2);
      m += w * val;
      weight += w;
    }
    if (weight > 0) {
      m = m / weight;
    }
    return m;
  }

  inline void InterpolatedAt(int row, int col, std::vector<float>& newTime, std::vector<float>& values){
    assert( newTime.size() == values.size() );
    values.assign(values.size(), 0);
    size_t ii = 0;
    size_t imin = (newTime.size() > mDepth ) ? mDepth : newTime.size();
    // we assume the default case is that mTimePoints matches newTime
    for (; ii < imin && newTime[ii]== mTimePoints[ii]; ++ii ) {
      // mData contains values that should be interpreted as
      // signed shorts after SubDcOffset
      values[ii] = (short)At((size_t)row, (size_t)col, ii);
    }
    if (ii == mDepth)
      for (; ii < values.size(); ++ii){
	// At handles the extrapolation beyond the last time point
	values[ii] = (short)At((size_t)row, (size_t)col, ii); 
      }

    if (ii < values.size())
    {
      // real interpolation
      std::vector<float> v(mDepth,0);
      size_t idx = mFrameStep * ii + (row - mRowStart) * mWidth + (col - mColStart);
      for (size_t i = ii; i < mDepth; i++, idx += mFrameStep){
	v[i] = (short)mData[idx];
      }
      TimeCompression::Interpolate(&mTimePoints[ii], &v[ii], (int)(mTimePoints.size()-ii), &newTime[ii], &values[ii], (int)(newTime.size()-ii));
    }
  }

  inline bool RegionMatch(Region& region){
    bool match = false;
    if ((size_t)region.row == mRowStart &&
	(size_t)region.col == mColStart &&
	(size_t)region.w == mWidth &&
	(size_t)region.h == mHeight)
    {
      match = true;
    }
    return match;
  }

  inline bool TimingMatch(std::vector<float>& otherTimePoints){
    bool match = false;
    if (otherTimePoints.size() <= mTimePoints.size()) {
      match = true;

      for (size_t i=0; i<otherTimePoints.size(); i++) {
	// floating point numbers have to be bit for bit identical
	// should we be comparing to within machine precision?
	if (mTimePoints[i] != otherTimePoints[i]) {
	  match = false;
	}
      }
    }
    return match;
  }

  inline float FramesToSeconds(float frame){
    return ( frame * mBaseFrameRate / 1000.0f );
  }

  inline float GetT0TimeVal(int row, int col, float seconds) {
    seconds = mT0 * mBaseFrameRate / 1000.0f + seconds;
    if (seconds < mTimePoints[0]) { return 0.0f; }
    std::vector<float>::iterator i = std::upper_bound(mTimePoints.begin(), mTimePoints.end(), seconds);
    assert(i != mTimePoints.end());
    assert(i != mTimePoints.begin());
    int frameAbove = i - mTimePoints.begin();
    int frameBelow = frameAbove - 1;
    assert(frameAbove < (int) mDepth && frameBelow >= 0);
    float lowerVal = At(row,col,frameBelow);
    float upperVal = At(row,col,frameAbove);
    return lowerVal + (upperVal - lowerVal) * (seconds - mTimePoints[frameBelow])/(mTimePoints[frameAbove] - mTimePoints[frameBelow]);
  }

  inline float GetTimeVal(int row, int col, float seconds) {
    if (seconds < mTimePoints.front()) { return 0.0f; } // before intial timepoint we specify 0
    if (seconds >= mTimePoints.back()) { return At(row,col,mDepth-1); } // just pad out with last value
    std::vector<float>::iterator i = std::upper_bound(mTimePoints.begin(), mTimePoints.end(), seconds);
    assert(i != mTimePoints.end());
    assert(i != mTimePoints.begin());
    int frameAbove = i - mTimePoints.begin();
    int frameBelow = frameAbove - 1;
    assert(frameAbove < (int) mDepth && frameBelow >= 0);
    float lowerVal = At(row,col,frameBelow);
    float upperVal = At(row,col,frameAbove);
    return lowerVal + (upperVal - lowerVal) * (seconds - mTimePoints[frameBelow])/(mTimePoints[frameAbove] - mTimePoints[frameBelow]);
  }

  size_t mRowStart, mColStart, mFrameStart, mFrameStep;
  size_t mChipRow, mChipCol, mChipFrame;
  size_t mOrigFrames;
  int mStartDetailedTime, mStopDetailedTime, mLeftAvg;
  float mT0; // units = base frames
  float mSigma;
  float mTMidNuc; // units = base frames
  float mBaseFrameRate; // units = (seconds / frame) * 1000
  size_t mHeight, mWidth, mDepth; // row, col, frame
  std::vector<float> mTimePoints; // units = seconds
  TimeCompression mTime;
  std::vector<uint16_t> mData;
};

#endif // TRACECHUNK_H
