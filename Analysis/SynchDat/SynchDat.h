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
    /* std::cout << "t0: " << t0 << " binIx: " << binIx << " Num frames:" << chunk.mDeltaFrame.size() << std::endl; */
    /* for (size_t i = 0; i < chunk.mDeltaFrame.size() -1; i++) { */
    /*   std::cout << "frame: " << i << ": " << chunk.mDeltaFrame[i] << std::endl; */
    /* } */
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

  inline short At(int chipRow, int chipCol, int chipFrame) {
    return mChunks.GetItemByRowCol(chipRow, chipCol).At(chipRow, chipCol, chipFrame);
  }

  inline short At(int wellIx, int chipFrame) {
    return At(wellIx / GetCols(), wellIx % GetCols(), chipFrame);
  }

  inline void InterpolatedAt(int chipRow, int chipCol, std::vector<float>& newTime, std::vector<float>& interpolations){
    mChunks.GetItemByRowCol(chipRow, chipCol).InterpolatedAt(chipRow, chipCol, newTime, interpolations);
  }

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
  void Clear() { mInfo.Clear(); };

  // should be private...
  GridMesh<TraceChunk> mChunks;
  std::vector<std::string> mVersion;
  Info mInfo;         ///< Any key,value information associated with this hdf5 file.
};

#endif // SYNCHDAT_H
