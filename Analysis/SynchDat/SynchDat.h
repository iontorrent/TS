/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef SYNCHDAT_H
#define SYNCHDAT_H
#include <iostream>
#include "TraceChunk.h"
#include "TimeCompression.h"
#include "GridMesh.h"
#include "DatCompression.h"
#include "IonErr.h"
#include "Image.h"
#include "Utils.h"

class SynchDat : public AcqMovie {

 public:

  void Init(size_t numRows, size_t numCols, size_t rowStep, size_t colStep) {
    mChunks.Init(numRows, numCols, rowStep, colStep);
  }

  void Close() {
    mChunks.Clear();
  }

  size_t GetNumBin() const { return mChunks.GetNumBin(); }

  void SubDcOffset() { 
    for (size_t bIx = 0; bIx < mChunks.GetNumBin(); bIx++) {
      TraceChunk &chunk = mChunks.GetItem(bIx);
      chunk.SubDcOffset();
    }
  }

  void AdjustForDrift() { 
    for (size_t bIx = 0; bIx < mChunks.GetNumBin(); bIx++) {
      TraceChunk &chunk = mChunks.GetItem(bIx);
      chunk.AdjustForDrift();
    }
  }

  float DcOffset(size_t row, size_t col) {
    TraceChunk &chunk = mChunks.GetItemByRowCol(row, col);
    return chunk.DcOffset(row, col);
  }

  float T0RegionFrame(size_t binIx) {
    TraceChunk &chunk = mChunks.GetItem(binIx);
    float t0 = chunk.mT0 * chunk.mBaseFrameRate / 1000.0f;
    int preT0 = 0;
    for (size_t i = 0; i < chunk.mTimePoints.size() -1; i++) {
      if (t0 > chunk.mTimePoints[i]) {
        preT0++;
      }
      else {
        break;
      }
    }
    t0 = preT0;
    return t0;
  }

  float T0Frame(size_t wellIx) {
    return T0RegionFrame(mChunks.GetBin(wellIx));
  }

  int GetMaxFrames() const {
    int maxSize = -1;
    for (size_t bIx = 0; bIx < mChunks.mBins.size(); bIx++) {
      if ((int)mChunks.mBins[bIx].mDepth > maxSize) {
        maxSize = (int)mChunks.mBins[bIx].mDepth;
      }
    }
    return maxSize;
  }

  int GetMaxFrames(std::vector<int> &timeStamps) {
    std::vector<float> *timePoint = NULL;
    int maxSize = -1;
    for (size_t bIx = 0; bIx < mChunks.mBins.size(); bIx++) {
      if ((int)mChunks.mBins[bIx].mDepth > maxSize) {
        maxSize = (int)mChunks.mBins[bIx].mDepth;
        timePoint = &mChunks.mBins[bIx].mTimePoints;
      }
    }
    if (timePoint != NULL) {
      timeStamps.resize(timePoint->size());
      for (size_t i = 0; i < timeStamps.size(); i++) {
        timeStamps[i] = timePoint->at(i) * 1000;
      }
    }
    return maxSize;
  }

  /* Do the interpolation to get this t0 relative time. */
  float GetValue(float t0time, int row, int col) {
    TraceChunk &chunk = mChunks.GetItemByRowCol(row, col);
    return chunk.GetT0TimeVal(row, col, t0time);
  }

  /* Do the interpolation to get this t0 relative time. */
  float GetTimeVal(float seconds, int row, int col) {
    TraceChunk &chunk = mChunks.GetItemByRowCol(row, col);
    return chunk.GetTimeVal(row, col, seconds);
  }

  void PrintData(size_t wellIx) {
    int row = wellIx / mChunks.mBins[0].mChipCol;
    int col = wellIx % mChunks.mBins[0].mChipCol;
    TraceChunk &chunk = mChunks.GetItemByRowCol(row, col);
    std::cout << "well: " << wellIx;
    for (size_t i = 0; i < chunk.mDepth; i++) {
      std::cout << ", " << chunk.At(row, col, i);
    }
    std::cout << std::endl;
  }

  TraceChunk & GetChunk(int regionIx) { return mChunks.GetItem(regionIx); }

  int GetOrigChipFrames() {
    return mChunks.mBins[0].mChipFrame;
  }

  void GetBinCoords(size_t regionIdx, int &rowStart, int &rowEnd, int &colStart, int &colEnd) {
    mChunks.GetBinCoords(regionIdx, rowStart, rowEnd, colStart, colEnd);
  }

  inline size_t GetNumRegions() { 
    return mChunks.GetNumBin();
  }
  
  inline float GetT0(size_t chipRow, size_t chipCol) {
    return mChunks.GetItemByRowCol(chipRow, chipCol).mT0;
  }
  inline float GetT0(size_t wellIndex) {
    return mChunks.GetItem(mChunks.GetBin(wellIndex)).mT0;
  }

  inline size_t NumCol() {
    return mChunks.mBins[0].mChipCol;
  }

  inline size_t NumRow() {
    return mChunks.mBins[0].mChipRow;
  }

  inline size_t FrameRate() {
    return mChunks.mBins[0].mBaseFrameRate;
  }

  inline size_t NumFrames(size_t chipRow, size_t chipCol) {
    return mChunks.GetItemByRowCol(chipRow, chipCol).mDepth;
  }

  inline size_t NumFrames(size_t wellIndex) {
    return mChunks.GetItem(mChunks.GetBin(wellIndex)).mDepth;
  }

  inline short At(int chipRow, int chipCol, int chipFrame) const {
    return mChunks.GetItemByRowCol(chipRow, chipCol).At(chipRow, chipCol, chipFrame);
  }

  inline short &At(int chipRow, int chipCol, int chipFrame) {
    return mChunks.GetItemByRowCol(chipRow, chipCol).At(chipRow, chipCol, chipFrame);
  }


  inline short At(int wellIx, int chipFrame) {
    return At(wellIx / GetCols(), wellIx % GetCols(), chipFrame);
  }

  inline void InterpolatedAt(int chipRow, int chipCol, std::vector<float>& newTime, std::vector<float>& interpolations){
    mChunks.GetItemByRowCol(chipRow, chipCol).InterpolatedAt(chipRow, chipCol, newTime, interpolations);
  }

  static inline float InterpolateValue(float *curVals, std::vector<float> &curTimes, float desiredTime) {
    if (desiredTime < curTimes[0]) {
      return curVals[0];
    }
    int end = curTimes.size() -1;
    if (desiredTime > (curTimes[end]+curTimes[end-1])/2) {
      // extrapolate
      float t2 =(curTimes[end] + curTimes[end -1])/2;
      float t1 = (curTimes[end-1] + curTimes[end -2])/2;
      float slope = (curVals[end] - curVals[end -1])/(t2 - t1);
      float y = slope * (desiredTime - t1);
      return y;
    }
    std::vector<float>::iterator i = std::upper_bound(curTimes.begin(), curTimes.end(), desiredTime);
    assert(i != curTimes.end());
    assert(i != curTimes.begin());
    int frameAbove = i - curTimes.begin();
    int frameBelow = frameAbove - 1;
    if ((curTimes[frameAbove] + curTimes[frameBelow])/2 < desiredTime) {
      frameAbove++;
      frameBelow++;
    }
    float lowerVal = curVals[frameBelow];
    float upperVal = curVals[frameAbove];
    float aboveTime = (curTimes[frameAbove] + curTimes[frameBelow])/2;
    float belowTime = (curTimes[frameBelow] + (frameBelow == 0 ? 0 : curTimes[frameBelow-1]))/2;
    assert(desiredTime >= belowTime);
    assert(aboveTime > belowTime);
    return lowerVal + (desiredTime - belowTime) * ((upperVal - lowerVal) /(aboveTime - belowTime));
  }


  /* static inline float InterpolateValue(float *curVals, std::vector<float> &curTimes, float desiredTime) { */
  /*   if (desiredTime < curTimes[0]) { */
  /*     return curVals[0]; */
  /*   } */
  /*   if (desiredTime >= curTimes.back()) { */
  /*     // extrapolate */
  /*     int end = curTimes.size() -1; */
  /*     float slope = (curVals[end] - curVals[end -1])/(curTimes[end] - curTimes[end -1]); */
  /*     float y = slope * (desiredTime - curTimes[end-1]); // y = mx + b  */
  /*     return y; */
  /*   } */
  /*   std::vector<float>::iterator i = std::upper_bound(curTimes.begin(), curTimes.end(), desiredTime); */
  /*   assert(i != curTimes.end()); */
  /*   assert(i != curTimes.begin()); */
  /*   int frameAbove = i - curTimes.begin(); */
  /*   int frameBelow = frameAbove - 1; */
  /*   float lowerVal = curVals[frameBelow]; */
  /*   float upperVal = curVals[frameAbove]; */
  /*   return lowerVal + (upperVal - lowerVal) * (desiredTime - curTimes[frameBelow])/(curTimes[frameAbove] - curTimes[frameBelow]); */
  /* } */

  /* inline const short At(size_t chipRow, size_t chipCol, size_t chipFrame) const { */
  /*   return mChunks.GetItemByRowCol(chipRow, chipCol).At(chipRow, chipCol, chipFrame); */
  /* } */
  TraceChunk &GetItemByRowCol(int row, int col) { return mChunks.GetItemByRowCol(row, col); }

  TraceChunk &GetTraceChunk(size_t binIx) { return mChunks.GetItem(binIx); }
  const TraceChunk &GetTraceChunk(size_t binIx) const { return mChunks.GetItem(binIx); }

  GridMesh<TraceChunk> &GetMesh() { return mChunks; }

  int GetRows() const { return mChunks.mBins[0].mChipRow; }
  int GetCols() const { return mChunks.mBins[0].mChipCol; }
  int GetFrames() const { return GetMaxFrames(); }
  
  size_t GetRowStep() const { return mChunks.GetRowStep(); } 
  size_t GetColStep() const { return mChunks.GetColStep(); }
  size_t GetRowBin() const { return mChunks.GetRowBin(); }
  size_t GetColBin() const { return mChunks.GetColBin(); }

  void Clear() { 
    mChunks.Clear();
    ClearOrigTimes();
    mInfo.Clear();
  }

  int GetOrigUncompFrames() { 
    std::string val;
    GetValue("uncompressed_frames",val);
    return atoi(val.c_str()); 
  }
  int GetBaseFrameRate() { return mChunks.mBins[0].mBaseFrameRate; }

  void ClearOrigTimes() { mOrigTimes.resize(0); }

  const std::vector<int> & GetOriginalTimes() {
    if (!mOrigTimes.empty()) {
      return mOrigTimes;
    }
    std::vector<std::string> tokens;
    std::string s;
    GetValue("orig_timestamps", s);
    split(s, ',', tokens);
    mOrigTimes.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
      mOrigTimes[i] = atoi(tokens[i].c_str());
    }
    return mOrigTimes;
  }

  /* Metadata accessors. */
  /** Get the value associated with a particular key, return false if key not present. */
  bool GetValue(const std::string &key, std::string &value)  const { return mInfo.GetValue(key, value); }
  
  /** 
   * Set the value associated with a particular key. Newer values for
   * same key overwrite previos values. */
  bool SetValue(const std::string &key, const std::string &value) { return mInfo.SetValue(key, value); }

  /** Get the key and the value associated at index. */
  bool GetEntry(int index, std::string &key, std::string &value) const { return mInfo.GetEntry(index, key, value); }

  /** Get the total count of key value pairs valid for GetEntry() */
  int GetCount() const { return mInfo.GetCount(); }

  /** Entry exists. */
  bool KeyExists(const std::string &key) const { return mInfo.KeyExists(key); }

  /** Empty out the keys, value pairs. */
  void ValuesClear() { mInfo.Clear(); };

  // should be private...
  GridMesh<TraceChunk> mChunks;
  std::vector<std::string> mVersion;
  std::vector<int> mOrigTimes;
  Info mInfo;         ///< Any key,value information associated with this hdf5 file.
};

#endif // SYNCHDAT_H
