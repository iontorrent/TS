/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <time.h>
#include <limits>

#include "Traces.h"
#include "FindSlopeChange.h"
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include "Utils.h"
#include "IonErr.h"

using namespace std;

Traces::~Traces() {
  if (mMask != NULL) {
    delete mMask;
  }
  if (mRawTraces != NULL) {
    delete [] mRawTraces;
  }
  if (mCriticalTraces != NULL) {
    delete [] mCriticalTraces;
  }
}

/* @todo - better way to set mask? This is safe as we 
   don't own the memory but could be faster as we're copying every time. */
void Traces::SetMask(Mask *mask) {
  if (mMask == NULL) {
    mMask = new Mask(mask->W(), mask->H());
  }
  int size = mMask->W() * mMask->H();
  assert(size == mask->W()* mask->H());
  for (int i = 0; i < size; i++) {
    (*mMask)[i] = (*mask)[i];
  }
}

const std::vector<std::vector<float> > &Traces::GetReportTraces() {
  return mReportTraces;
}

const std::vector<std::vector<float> > &Traces::GetReportCriticalFrames() {
  return mReportCriticalFrames;
}

void Traces::SetReportSampling(const ReportSet &set, bool keepExclude) {
  mSampleMap.resize(mRow * mCol);
  fill(mSampleMap.begin(), mSampleMap.end(), -1);
  int count = 0;
  const vector<int> &reportWells = set.GetReportIndexes();
  for (size_t i = 0; i < reportWells.size(); i++) {
    if ((mMask == NULL || keepExclude) || !((*mMask)[reportWells[i]] & MaskExclude)) {
      count++;
    }
  }
  mReportIdx.resize(count);
  mMedTraces.resize(count);
  mReportTraces.resize(count);
  mReportCriticalFrames.resize(count);
  int current = 0;
  size_t row = 0, col = 0;
  //  for (size_t idx = 0; idx < idxCount; idx += stepSize) {
  for (size_t i = 0; i < reportWells.size(); i++) {
    size_t idx = reportWells[i];
    if (mMask != NULL && ((*mMask)[idx] & MaskExclude)) {
      continue;
    }
    IndexToRowCol(idx, row, col);
    mReportIdx[current].resize(4);
    mReportIdx[current][0] = row;
    mReportIdx[current][1] = col;
    mReportIdx[current][2] = -1;
    mReportIdx[current][3] = -1;
    mSampleMap[idx] = current++;
  }
}

void Traces::MakeReportSubset(const std::vector<std::vector<float> > &source, std::vector<std::vector<float> > &fill) {
  fill.resize(mReportIdx.size());
  for (size_t i = 0; i < source.size(); i++) {
    int rIx = mSampleMap[i];
    if (rIx >= 0) {
      fill[rIx] = source[i];
    }
  }
}

void Traces::MakeReportSubset(const std::vector<std::vector<int8_t> > &source, std::vector<std::vector<float> > &fill) {
  fill.resize(mReportIdx.size());
  vector<float> buffer;
  for (size_t i = 0; i < source.size(); i++) {
    int rIx = mSampleMap[i];
    if (rIx >= 0) {
      GetTraces(i, buffer);
      fill[rIx] = buffer;
    }
  }
}

void Traces::Init(Image *img, Mask *mask) {
  Init(img, mask, -1, -1, -1, -1);
}

void Traces::Init(Image *img, Mask *mask, int startFrame, int endFrame,
                  int dcOffsetStart, int dcOffsetEnd) {
  mRefOut = NULL;
  mFlow = -1;
  mT0Step = 32;
  mUseMeshNeighbors = 1;
  mRow = (img->GetImage())->rows;
  mCol = (img->GetImage())->cols;
  startFrame = std::max(startFrame, 0);
  if( endFrame > 0 )
    endFrame = std::min(endFrame, (int)img->GetUnCompFrames());
  else
    endFrame = (int)img->GetUnCompFrames();
  
  mFrames = endFrame - startFrame;
  mTimes.resize(mFrames);
  //  copy(&raw->timestamps[0], &raw->timestamps[0] + mFrames, mTimes.begin());
  //  need to take into account variable frame compression here...  ??
  if (mIndexes.size() != mRow*mCol) {
    mIndexes.resize(mRow*mCol);
  }
  fill(mIndexes.begin(), mIndexes.end(), -1);
  vector<float> tBuff(mFrames, 0);
  SampleStats<float> traceStats;
  std::vector<float> sdTrace(mRow *mCol, 0);

  // Figure out how much memory to allocate for data pool
  int count = 0;
  for (size_t rowIx = 0; rowIx < mRow; rowIx++) {
    for (size_t colIx = 0; colIx < mCol; colIx++) {
      size_t traceIdx = RowColToIndex(rowIx, colIx);
      if (mask == NULL || !((*mask)[traceIdx] & MaskExclude)) {
        count++;
      }
    }
  }
  // Allocate memory
  if (mRawTraces != NULL) {
    delete [] mRawTraces ;
  }
  mRawTraces = new int8_t[count * mFrames];

  SampleQuantiles<float> chipTraceSdQuant(10000);
  count = 0;
  for (size_t rowIx = 0; rowIx < mRow; rowIx++) {
    for (size_t colIx = 0; colIx < mCol; colIx++) {
      size_t traceIdx = RowColToIndex(rowIx, colIx);
      if (mask == NULL || !((*mask)[traceIdx] & MaskExclude)) {
        mIndexes[traceIdx] = count++ * mFrames;
        double mean = 0;
        if (dcOffsetEnd > 0 && dcOffsetStart > 0) {
          assert(dcOffsetEnd > dcOffsetStart);
          for (int frameIx = dcOffsetStart; frameIx < dcOffsetEnd; frameIx++) {
            mean += img->GetInterpolatedValue(frameIx,colIx,rowIx);
          }
          mean = mean / (dcOffsetEnd - dcOffsetStart);
        }
				
        float val = img->GetInterpolatedValue(startFrame,colIx,rowIx) - mean;
        traceStats.Clear();
        for (int frameIx = 0; frameIx < endFrame; frameIx++) {
          val = (img->GetInterpolatedValue(frameIx,colIx,rowIx) - mean);
          tBuff[frameIx] = val;
          traceStats.AddValue(tBuff[frameIx]);
        }
        sdTrace[traceIdx] = traceStats.GetSD();
        chipTraceSdQuant.AddValue(sdTrace[traceIdx]);
        SetTraces(traceIdx, tBuff, mRawTraces);
      }
    }
  }
  mFlags.resize(mRow*mCol, 0);

  double traceSDThresh = chipTraceSdQuant.GetMedian() - 5 * (chipTraceSdQuant.GetQuantile(.5) - chipTraceSdQuant.GetQuantile(.25));
  traceSDThresh = max(0.0, traceSDThresh);
  int badCount = 0;
  for (size_t wellIx = 0; wellIx < sdTrace.size(); wellIx++) {
    if (mask != NULL && ((*mask)[wellIx] & MaskExclude)) {
      continue;
    }
    if (sdTrace[wellIx] <= traceSDThresh) {
      mFlags[wellIx] = BAD_VAR;
      badCount++;
    }
  }
  cout << "Found: " << badCount << " wells <= sd: " << traceSDThresh << endl;
  mT0.resize(0);
  if(mask != NULL) {
    if (mMask != NULL) 
      delete mMask;
    mMask = new Mask(mask);
    size_t size = mFlags.size();
    for (size_t i = 0; i < size; i++) {
      if ((*mMask)[i] & MaskExclude) {
        mFlags[i] = NOT_AVAIL;
      }
    }
  }
  else {
    mMask = NULL;
  }
  mSampleMap.resize(mRow * mCol);
  mCurrentData = mRawTraces;
  fill(mSampleMap.begin(), mSampleMap.end(), -1);
}

int Traces::DCOffset(size_t startFrame, size_t endFrame) {
  size_t size = mIndexes.size();
  vector<float> traceBuffer;
  for (size_t i = 0; i < size; i++) {
    double mean = 0;
    GetTraces(i, traceBuffer);
    for (size_t frameIx = startFrame; frameIx < endFrame; frameIx++) {
      mean += traceBuffer[frameIx];
    }
    mean /= (endFrame - startFrame);
    size_t frameSize = traceBuffer.size();
    for (size_t frameIx = 0; frameIx < frameSize; frameIx++) {
      traceBuffer[frameIx] -= mean;
    }
    SetTraces(i, traceBuffer, mCurrentData);
  }
  return 0;
}


void Traces::CalcIncorporationRegionStart(size_t rowStart, size_t rowEnd, 
					  size_t colStart, size_t colEnd,
					  SampleStats<float> &starts, MaskType maskType) {
  FindSlopeChange<float> finder;
  double xDist = mCol - ((colStart + colEnd)/2.0);
  double yDist = mRow - ((rowStart + rowEnd)/2.0);
  int offset = floor(xDist * yDist * 3.738038e-06);
  size_t startNucFrame = 0;
  size_t iGuessStart = 14+offset, iGuessEnd = 24+offset;
  //  size_t iRangeStart = 10, iRangeEnd = 60, iGuessStart = 16, iGuessEnd = 45;
  //  size_t iRangeStart = 0, iRangeEnd = 25, iGuessStart = 3, iGuessEnd = 10;
  /* 15-25 is usual start

     17-25
  */
  Mask &mask = *mMask;
  vector<SampleQuantiles<float> >frameAvgs(80);
  for (size_t i = 0; i < frameAvgs.size(); i++) {
    frameAvgs[i].Init(200);
  }
  vector<float> trace;
  vector<float> traceBuffer;
  for (size_t rowIx = rowStart; rowIx < rowEnd; rowIx++) {
    for (size_t colIx = colStart; colIx < colEnd; colIx++) {
      size_t idx = RowColToIndex(rowIx, colIx);
      if ((mask[idx] & maskType) && mFlags[idx] == OK) {
	GetTraces(idx, traceBuffer);
        for (size_t i = 0; i < frameAvgs.size(); i++) {
          frameAvgs[i].AddValue(traceBuffer[i]);
        }
      }
    }
  }
  if (frameAvgs[0].GetNumSeen() < MIN_T0_PROBES) {
    return;
  }
  trace.resize(frameAvgs.size());
  for (size_t i = 0; i < frameAvgs.size(); i++) {
    trace[i] = frameAvgs[i].GetMedian();
  }
  bool ok = true;
  //  finder.findChangeIndex(startNucFrame, startSumSeq, trace, iRangeStart, iRangeEnd, iGuessStart, iGuessEnd);
  if (startNucFrame >= iGuessEnd || startNucFrame <= iGuessStart) {
    ok = false;
    startNucFrame = -2;
  }
  if (ok && frameAvgs[0].GetNumSeen() >= MIN_T0_PROBES ) {
    starts.AddValue(startNucFrame-.3);
  }
}

void Traces::CalcIncorpBreakRegionStart(size_t rowStart, size_t rowEnd, 
					size_t colStart, size_t colEnd,
                                        SampleStats<float> &starts) { //, MaskType maskType) {
  FindSlopeChange<double> finder;
  float startNucFrame = 0, startSumSeq = 0, slope = 0, yIntercept = 0;
  int valveOpen = 6;
  int numFrames = std::min(100, (int)mFrames);
  int maxSearch = std::min(70, (int)mFrames-FRAME_ZERO_WINDOW);
  //size_t iRangeStart = 5+offset, iRangeEnd = 52+offset, iGuessStart = 10+offset, iGuessEnd = 46+offset;
  //  size_t iRangeStart = 0, iRangeEnd = 25, iGuessStart = 3, iGuessEnd = 10;
  /* 15-25 is usual start
     17-25
  */
  Mask &mask = *mMask;
  int sampleSize = 100;
  vector<SampleQuantiles<float> >frameAvgs(numFrames);
  for (size_t i = 0; i < frameAvgs.size(); i++) {
    frameAvgs[i].Init(sampleSize);
  }
  //  vector<SampleStats<float> > frameAvgs(numFrames);

  vector<double> trace;
  vector<float> traceBuffer;
  int seen = 0;
  for (size_t rowIx = rowStart; rowIx < rowEnd; rowIx++) {
    for (size_t colIx = colStart; colIx < colEnd; colIx++) {
      size_t idx = RowColToIndex(rowIx, colIx);
      //      if ((mask[idx] & maskType) && mFlags[idx] == OK) {
      if (!(mask[idx] & MaskPinned) && !(mask[idx] & MaskExclude) && mFlags[idx] == OK) {
        GetTraces(idx, traceBuffer);
        seen++;
        for (size_t i = 0; i < frameAvgs.size(); i++) {
          frameAvgs[i].AddValue(traceBuffer[i]);
        }
      }
    }
  }
  if(frameAvgs[0].GetNumSeen() <= MIN_T0_PROBES) {
    return;
  }
  trace.resize(frameAvgs.size());
  for (size_t i = 0; i < frameAvgs.size(); i++) {
    //    trace[i] = frameAvgs[i].GetMedian();
    trace[i] = frameAvgs[i].GetMedian();
  }
  double zero = 0, hinge = 0;
  bool ok = finder.findNonZeroRatioChangeIndex(startNucFrame,startSumSeq, 
					       slope, yIntercept,
                                               valveOpen, maxSearch, zero, hinge, trace, 1, 5);
  if (mRefOut != NULL) {
    std::ostream &o = *mRefOut;
    o << mFlow << "\t" << rowStart << "\t" << rowEnd << "\t" << colStart << "\t" << colEnd << "\t" 
      << ok << "\t" << startNucFrame << "\t" << frameAvgs[0].GetNumSeen();
    for (size_t i = 0; i < trace.size(); i++) {
      o << "\t" << trace[i];
    }
    o << endl;
  }
  if (ok && frameAvgs[0].GetNumSeen() >= MIN_T0_PROBES ) {
    starts.AddValue(startNucFrame);
  }
}

void Traces::CalcIncorporationStartReference(int nRowStep, int nColStep, 
                                             GridMesh<SampleStats<float> > &grid) {
  grid.Init(mRow, mCol, nRowStep, nColStep);

  int numBin = grid.GetNumBin();
  int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  for (int binIx = 0; binIx < numBin; binIx++) {
    SampleStats<float> &startStat = grid.GetItem(binIx);
    grid.GetBinCoords(binIx, rowStart, rowEnd, colStart, colEnd);
    CalcIncorpBreakRegionStart(rowStart, rowEnd,
			       colStart, colEnd,
                               startStat);
  }
}

bool Traces::CalcStartFrame(size_t row, size_t col, 
                            GridMesh<SampleStats<float> > &regionStart, float &start) {
  std::vector<double> dist(7);
  std::vector<SampleStats<float> *> values;
  bool allOk = false;
  regionStart.GetClosestNeighbors(row, col, mUseMeshNeighbors, dist, values);
  double distWeight = 0;
  double startX = 0;
  start = 0;
  for (size_t i = 0; i < values.size(); i++) {
    if (values[i]->GetCount() > 0) {
      double w = WeightDist(dist[i]); //1/sqrt(dist[i] + 1);
      distWeight += w;
      startX += w * values[i]->GetMean();
    }
  }

  if (distWeight > 0 && start >= 0) {
    start = startX / distWeight;
    if (startX >= 0 && isfinite(startX)) {
      allOk = true; 
    }
  }

  return allOk;
}

void Traces::CalcT0Reference() {
  CalcIncorporationStartReference(mT0Step, mT0Step, mT0ReferenceGrid);
}

void Traces::CalcT0(bool force) {
  size_t size = mRow * mCol;
  if (mT0.size() == size && !force) {
    return;
  }
  CalcT0Reference();
  bool allOk = true;
  mT0.resize(size, 0);
  for (size_t rowIx = 0; rowIx < mRow; rowIx++) {
    for (size_t colIx = 0; colIx < mCol; colIx++) {
      int idx = RowColToIndex(rowIx, colIx);
      if (mFlags[idx] == OK) {
        allOk = true;
        allOk = CalcStartFrame(rowIx, colIx, mT0ReferenceGrid, mT0[idx]);
        if (!allOk) {
          mFlags[idx] = BAD_T0;
        }
      }
    }
  }
}

void Traces::DCOffsetT0() {
  vector<float> traceBuffer;
  for (size_t i = 0; i < mT0.size(); i++) {
    if (mIndexes[i] >= 0 && mFlags[i] == OK) {
      GetTraces(i, traceBuffer);
      int dcStart = floor(mT0[i]) - 6;
      double mean = 0;
      int count = 0;
      for (int frameIx = dcStart; frameIx < dcStart+5; frameIx++) {
	mean = mean +traceBuffer[frameIx];
	count++;
      }
      mean = mean / count;
      for (size_t frameIx = 0; frameIx < mFrames; frameIx++) {
	traceBuffer[frameIx] = traceBuffer[frameIx] - mean;
      }
      SetTraces(i, traceBuffer, mCurrentData);
    }
  }
}

bool Traces::FillCriticalWellFrames(size_t idx, int nFrames) {
  static int numWarn = 0;
  double t0 = mT0[idx];
  double offSet = floor(t0);
  size_t frameSize = nFrames;
  mFrames = nFrames;
  vector<float> traceBuffer;
  if (mIndexes[idx] >= 0) {
    GetTraces(idx, traceBuffer);

    int reportIdx = mSampleMap[idx];
    if (reportIdx >= 0) {
      mReportTraces[reportIdx].resize(traceBuffer.size());
      copy(traceBuffer.begin(), traceBuffer.end(), mReportTraces[reportIdx].begin());
    }
    frameSize = min((size_t) (traceBuffer.size() - offSet), frameSize);
    if ((offSet+frameSize) >= traceBuffer.size() || (offSet) < 0) {
      if (numWarn++ < 20) {
        //				ION_WARN("Want to read to frame: " + ToStr(offSet+frameSize) + " but only: " + ToStr(traceBuffer.size()) + "frames for well: " + ToStr(idx));
      }
      offSet = traceBuffer.size() - (frameSize+1);
    }
    for (size_t frameIx = 0; frameIx < frameSize; frameIx++) {
			
      // @todo - replace assumption that distance is constant
      // Small interpolation assuming that the distance between the frames is 1
      double diff = traceBuffer[offSet+frameIx+1] - traceBuffer[offSet+frameIx];
      traceBuffer[frameIx] = (t0 - offSet) * diff + traceBuffer[offSet+frameIx];
    }
    traceBuffer.resize(nFrames);
    SetTraces(idx, traceBuffer, mCurrentData);
  }
  return OK;
}

void Traces::T0DcOffset(int t0Minus, int t0Plus) {
  vector<float> traceBuffer;
  SampleQuantiles<float> q(min(t0Plus - t0Minus, 10));
  for (size_t idx = 0; idx < mIndexes.size(); idx++) {
    if (mIndexes[idx] < 0) {
      continue;
    }
    q.Clear();
    int t0 = round(GetT0(idx));
    t0 = max(1,t0);
    t0 = min(t0,(int)mFrames-1);
    assert(t0 > 0);
    GetTraces(idx, traceBuffer);
    for (int i = max(t0 - t0Minus,0); i < min(t0+t0Plus,(int)mFrames); i++) {
      q.AddValue(traceBuffer[i]);
    }
    float offset = q.GetMedian();
    for (size_t i = 0; i < traceBuffer.size(); i++) {
      traceBuffer[i] -= offset;
    }
    SetTraces(idx, traceBuffer, mRawTraces);
  }
}

bool Traces::FillCriticalFrames() {
  size_t nFrames = min((size_t)50, mFrames);
  size_t size = mIndexes.size();
  vector<int64_t> idxOut(mIndexes.size(), -1);
  int count = 0;
  // Count how many wells we actually have
  for (size_t i = 0; i < mIndexes.size(); i++) {
    if (mIndexes[i] >= 0) {
      count++;
    }
  }
  int numWarn = 0;
  // Allocate memory and loop through creating smaller footprint
  mCriticalTraces = new int8_t[count * nFrames];
  vector<float> traceBuffer;
  count = 0;
  for (size_t idx = 0; idx < size; idx++) {
    if (mIndexes[idx] < 0) {
      continue;
    }
    idxOut[idx] = count++ * nFrames;
    double t0 = mT0[idx];
    double offSet = floor(t0);
    size_t frameSize = nFrames;
    GetTraces(idx, traceBuffer);
		
    int reportIdx = mSampleMap[idx];
    if (reportIdx >= 0) {
      mReportTraces[reportIdx].resize(traceBuffer.size());
      copy(traceBuffer.begin(), traceBuffer.end(), mReportTraces[reportIdx].begin());
    }
    frameSize = min((size_t) (traceBuffer.size() - offSet), frameSize);
    if ((offSet+frameSize) >= traceBuffer.size() || (offSet) < 0) {
      if (numWarn++ < 20) {
        //				ION_WARN("Want to read to frame: " + ToStr(offSet+frameSize) + " but only: " + ToStr(traceBuffer.size()) + "frames for well: " + ToStr(idx));
      }
      offSet = traceBuffer.size() - (frameSize+1);
    }
    for (size_t frameIx = 0; frameIx < frameSize; frameIx++) {
      // @todo - replace assumption that distance is constant
      // Small interpolation assuming that the distance between the frames is 1
      double diff = traceBuffer[offSet+frameIx+1] - traceBuffer[offSet+frameIx];
      traceBuffer[frameIx] = (t0 - offSet) * diff + traceBuffer[offSet+frameIx];
    }
    traceBuffer.resize(nFrames);
    mIndexes[idx] = idxOut[idx];
    SetTraces(idx, traceBuffer, mCriticalTraces);
  }
  mFrames = nFrames;
  mIndexes = idxOut;
  delete [] mRawTraces;
  mRawTraces = NULL;
  mCurrentData = mCriticalTraces;
  return true;
}

int Traces::CalcRegionReference(unsigned int type, int rowStart, int rowEnd, 
				int colStart, int colEnd,
				std::vector<float> &trace) {
  trace.resize(mFrames);
  fill(trace.begin(), trace.end(), 0.0f);
  vector<vector<float> > matrix;
  vector<float> traceBuffer;
  matrix.resize(trace.size());
  for (int rowIx = rowStart; rowIx < rowEnd; rowIx++) {
    for (int colIx = colStart; colIx < colEnd; colIx++) {
      int wellIdx = RowColToIndex(rowIx,colIx);
      if ((*mMask)[wellIdx] & type && mFlags[wellIdx] == OK) {
	GetTraces(wellIdx,traceBuffer);
        for (size_t frameIx = 0; frameIx < traceBuffer.size(); frameIx++) {
          matrix[frameIx].push_back(traceBuffer[frameIx]);
        }
      }
    }
  }
  int length = matrix[0].size();
  size_t size = matrix.size();
  if (length > MIN_REF_PROBES) {
    for (size_t i = 0; i < size; i++) {
      sort(matrix[i].begin(), matrix[i].end());
      float med = 0;
      if( matrix[i].size() % 2 == 0 ) {
        med = (matrix[i][length / 2] + matrix[i][(length / 2)-1])/2.0;
      }
      else {
        med = matrix[i][length/2];
      }
      trace[i] = med;
    }
    return OK;
  }
  else {
    trace.resize(0);
  }
  return BAD_REGION;
}

void Traces::CalcReference(int rowStep, int colStep, GridMesh<std::vector<float> > &gridReference) {
  gridReference.Init(mRow, mCol, rowStep, colStep);
  int numBin = gridReference.GetNumBin();
  int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  for (int binIx = 0; binIx < numBin; binIx++) {
    gridReference.GetBinCoords(binIx, rowStart, rowEnd, colStart, colEnd);
    vector<float> &trace = gridReference.GetItem(binIx);
    CalcRegionReference(MaskEmpty, rowStart, rowEnd, colStart, colEnd, trace);
  }
}


int Traces::CalcMedianReference(int row, int col, 
				GridMesh<std::vector<float> > &regionMed,
				std::vector<double> &dist,
				std::vector<std::vector<float> *> &values,
				std::vector<float> &reference) {
  int retVal = OK;
  // vector<float> &binVal = regionMed.GetItemByRowCol(row,col);
  // if (binVal.size() > 0) {
  //   reference.resize(binVal.size());
  //   copy(binVal.begin(), binVal.end(), reference.begin());
  //   return OK;
  // }
  regionMed.GetClosestNeighbors(row, col, mUseMeshNeighbors, dist, values);
  size_t size = 0;
  for (size_t i = 0; i < values.size(); i++) {
    size = max(values[i]->size(), size);
  }
  reference.resize(size);
  fill(reference.begin(), reference.end(), 0.0);
  double distWeight = 0;
  size_t valSize = values.size();
  for (size_t i = 0; i < valSize; i++) {
    if (values[i]->size()  == 0) {
      continue;
    }
    double w = WeightDist(dist[i]); //1/sqrt(dist[i]+1);
    distWeight += w;
    size_t vSize = values[i]->size();
    for (size_t j = 0; j < vSize; j++) {
      reference[j] += w * values[i]->at(j);
    }
  }
  // Divide by our total weight to get weighted mean
  if (distWeight > 0) {
    for (size_t i = 0; i < reference.size(); i++) {
      reference[i] /= distWeight;
    }
    retVal = OK;
  }
  else {
    retVal = BAD_DATA;
  }
  return retVal;
}

int Traces::maxWidth(const std::vector<std::vector<float> > &d) {
  size_t m = 0;
  for (size_t i = 0; i < d.size(); i++) {
    m = std::max(d[i].size(), m);
  }
  return (int) m;
}

int Traces::maxWidth(const std::vector<std::vector<int8_t> > &d) {
  size_t m = 0;
  for (size_t i = 0; i < d.size(); i++) {
    m = std::max(d[i].size(), m);
  }
  return (int) m;
}

int Traces::maxWidth(const std::vector<std::vector<double> > &d) {
  size_t m = 0;
  for (size_t i = 0; i < d.size(); i++) {
    m = std::max(d[i].size(), m);
  }
  return (int) m;
}

void Traces::SetTraces(int idx, const std::vector<float> &trace, int8_t *outData) {
  int index = mIndexes[idx];
  float sum = 0;
  
  for (size_t i = 0; i < trace.size(); i++) {
    float tmp = trace[i] - sum;
    if (tmp >= std::numeric_limits<int8_t>::max()) {
      outData[index+i] = std::numeric_limits<int8_t>::max();
    }
    if (tmp <= (-1 * std::numeric_limits<int8_t>::max())) {
      outData[index+i] = -1 * std::numeric_limits<int8_t>::max();
    }
    else {
      outData[index+i] = (int8_t)(tmp + .5);
    }
    sum += outData[index+i];
  }
}

void Traces::PrintVec(const std::vector<float> &x) {
  for (size_t i = 0; i < x.size(); i++) {
    std::cout << x[i] << '\t';
  }
  std::cout << endl;
}

void Traces::PrintTrace(int idx) {
  vector<float> x;
  GetTraces(idx,x);
  PrintVec(x);
}

// void Traces::GetTraces(int idx, std::vector<float> &trace) {
//   trace.resize(mTraces[idx].size(), 0);
//   if (trace.size() > 0) {
//     trace[0] = mTraces[idx][0];
//     for (size_t i = 1; i < trace.size(); i++) {
//       trace[i] = trace[i-1] + (mTraces[idx][i]);
//     }
//   }
// }

void Traces::PrintVec(const std::vector<int8_t> &x) {
  for (size_t i = 0; i < x.size(); i++) {
    std::cout << x[i] << '\t';
  }
  std::cout << endl;
}
