/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACESTORECOL_H
#define TRACESTORECOL_H

#include <algorithm>
#include <iostream>
#include "Mask.h"
#include "TraceStore.h"
#include "MathOptim.h"
#include "GridMesh.h"
#include "ChipReduction.h"
#define TSM_MIN_REF_PROBES 10
#define TSM_OK 1
#define MIN_REGION_REF_WELLS 50
#define MIN_REGION_REF_PERCENT 0.20f
#define REF_REDUCTION_SIZE 10
#define REF_SMOOTH_SIZE 100
#define THUMBNAIL_SIZE 100
/**
 * Abstract interface to a repository for getting traces.
 */
class TraceStoreCol : public TraceStore
  {

  public:

  void SetMinRefProbes (int n) { mMinRefProbes = n; }

  TraceStoreCol (Mask &mask, size_t frames, const char *flowOrder,
                 int numFlowsBuff, int maxFlow, int rowStep, int colStep) {
    Init(mask, frames,flowOrder, numFlowsBuff, maxFlow, rowStep, colStep);
  }
  TraceStoreCol() { 
    mUseMeshNeighbors = 0;
    mRows = mCols = mFrames = mRowRefStep = mColRefStep = 0;
    mFrames = mFrameStride = mMaxDist = mFlowFrameStride = 0;
  }
  void Init(Mask &mask, size_t frames, const char *flowOrder,
       int numFlowsBuff, int maxFlow, int rowStep, int colStep) {

    pthread_mutex_init (&mLock, NULL);
    mUseMeshNeighbors = 1;
    mRowRefStep = rowStep;
    mColRefStep = colStep;
    mMinRefProbes = floor (mRowRefStep * mColRefStep * .1);
    mRows = mask.H();
    mCols = mask.W();
    mFrames = frames;
    mFrameStride = mRows * mCols;
    mFlows = mFlowsBuf = maxFlow;
    mFlowFrameStride = mFrameStride * maxFlow;
    mMaxDist = 2 * sqrt(rowStep*rowStep + colStep+colStep);
    mFlowOrder = flowOrder;
    
    mWells = mRows * mCols;
    mUseAsReference.resize (mWells, false);
    int keep = 0;
    int empties = 0;
    mRefGridsValid.resize (mFlowsBuf, 0);
    mRefGrids.resize (mFlowsBuf);
    mData.resize (mWells * mFrames * mFlowsBuf);
    std::fill (mData.begin(), mData.end(), 0);
  }

  ~TraceStoreCol() {
    pthread_mutex_destroy (&mLock);
  }

  void SetSize(int frames) {
    mFrames = frames;
    mData.resize(mWells * mFrames * mFlowsBuf);
    std::fill (mData.begin(), mData.end(), 0);
  }

  void SplineLossyCompress(const std::string &strategy, int order, char *bad_wells, float *mad);
  void SplineLossyCompress(const std::string &strategy, int order, int flow_ix, char *bad_wells, float *mad);
  static void SplineLossyCompress(const std::string &strategy, int order, int flow_ix, char *bad_wells, 
                                  float *mad, size_t num_rows, size_t num_cols, size_t num_frames, size_t num_flows,
                                  int use_mesh_neighbors, size_t frame_stride, size_t flow_frame_stride, int16_t *data);

  void PcaLossyCompress(int row_start, int row_end,
                        int col_start, int col_end,
                        int flow_ix,
                        float *ssq, char *filters,
                        int row_step, int col_step,
                        int num_pca);
  static bool PcaLossyCompressChunk(int row_start, int row_end,
                                    int col_start, int col_end,
                                    int num_rows, int num_cols, int num_frames,
                                    int frame_stride,
                                    int flow_ix, int flow_frame_stride,
                                    short *data, bool replace,
                                    float *ssq, char *filters,
                                    int row_step, int col_step,
                                    int num_pca);

  void SetFlowIndex (size_t flowIx, size_t index) {
    // no op, we store things in a specific flow order
    assert(0);
  }

  size_t GetNumFrames() { return mFrames; }
  size_t GetNumWells() { return mRows * mCols; }
  size_t GetNumRows() { return mRows; }
  size_t GetNumCols() { return mCols; }
  size_t GetNumFlows() { return mFlows; }
  size_t GetFlowBuff() { return GetNumFlows(); }
  const std::string &GetFlowOrder() { return mFlowOrder; }
  double GetTime (size_t frame) { return mTime.at (frame); }
  void SetTime(double *time, int npts) { mTime.resize(npts); std::copy(time, time+npts, mTime.begin()); }

  bool HaveWell (size_t wellIx) {
    return true;
  }

  void SetHaveWellFalse (size_t wellIx) {
    assert(0);
  }

  bool HaveFlow (size_t flowIx) {
    assert(0);
  }

  void SetReference(size_t wellIx, bool isReference) { mUseAsReference[wellIx] = isReference; }

  bool IsReference (size_t wellIx) { return mUseAsReference[wellIx]; }

  int GetTrace (size_t wellIx, size_t flowIx,  std::vector<float>::iterator traceBegin) {
    int16_t *__restrict start = &mData[0] + flowIx * mFrameStride + wellIx;
    int16_t *__restrict end = start + mFlowFrameStride * mFrames;
    while (start != end) {
      *traceBegin++ = *start;
      start += mFlowFrameStride;
    }
    return TSM_OK;
  }

  int GetTrace (size_t wellIx, size_t flowIx, float *traceBegin) {
    int16_t *__restrict start = &mData[0] + flowIx * mFrameStride + wellIx;
    int16_t *__restrict end = start + mFlowFrameStride * mFrames;
    while (start != end) {
      *traceBegin++ = *start;
      start += mFlowFrameStride;
    }
    return TSM_OK;
  }

  int SetTrace (size_t wellIx, size_t flowIx,
                 std::vector<float>::iterator traceBegin,  std::vector<float>::iterator traceEnd)
  {
    int16_t *__restrict out = &mData[0] + flowIx * mFrameStride + wellIx;
    while (traceBegin != traceEnd) {
      *out = (int16_t) (*traceBegin + .5);
      traceBegin++;
      out += mFlowFrameStride;
    }
    return TSM_OK;
  }

  int SetTrace (size_t wellIx, size_t flowIx,
                float * traceBegin, 
                float * traceEnd) {
    int16_t *__restrict out = &mData[0] + flowIx * mFrameStride + wellIx;
    while (traceBegin != traceEnd) {
      *out = (int16_t) (*traceBegin + .5);
      traceBegin++;
      out += mFlowFrameStride;
    }
    return TSM_OK;
  }

  size_t RowColToIndex (size_t rowIx, size_t colIx) { return rowIx * mCols + colIx; }

  void IndexToRowCol (size_t index, size_t &rowIx, size_t &colIx) {
    rowIx = index / mCols;
    colIx = index % mCols;
  }

  int PrepareReference(size_t flowIx, std::vector<char> &filteredWells) {
    mRefWells.resize(mUseAsReference.size());
    for (size_t i = 0; i < mRefWells.size(); i++) {
      if (mUseAsReference[i]) {
        mRefWells[i] = 0;
      }
      else {
        mRefWells[i] = 1;
      }
    }
    assert(flowIx < mRefReduction.size());

    int x_clip = mCols;
    int y_clip = mRows;
    if (mUseMeshNeighbors == 0) {
      x_clip = THUMBNAIL_SIZE;
      y_clip = THUMBNAIL_SIZE;
    }
    mRefReduction[flowIx].Init(mRows, mCols, mFrames,
                               REF_REDUCTION_SIZE, REF_REDUCTION_SIZE, 
                               y_clip, x_clip, 1);
    for (size_t frame_ix = 0; frame_ix < mFrames; frame_ix++) {
      mRefReduction[flowIx].ReduceFrame(&mData[0] + flowIx * mFrameStride + frame_ix * mFlowFrameStride, &mRefWells[0], frame_ix);
    }
    mRefReduction[flowIx].SmoothBlocks(REF_SMOOTH_SIZE, REF_SMOOTH_SIZE);
    return TSM_OK;
  }

  

  int PrepareReferenceOld (size_t flowIx, std::vector<char> &filteredWells) {
    int fIdx = flowIx;
    CalcReference (mRowRefStep, mColRefStep, flowIx, mRefGrids[fIdx], filteredWells);
    mRefGridsValid[fIdx] = 1;
    mFineRefGrids.resize (mRefGrids.size());
    mFineRefGrids[fIdx].Init (mRows, mCols, mRowRefStep/2, mColRefStep/2);
    int numBin = mFineRefGrids[fIdx].GetNumBin();
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    for (int binIx = 0; binIx < numBin; binIx++) {
      mFineRefGrids[fIdx].GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
      vector<float> &trace = mFineRefGrids[fIdx].GetItem (binIx);
      CalcMedianReference ( (rowEnd + rowStart) /2, (colEnd + colStart) /2, mRefGrids[fIdx],
                            mDist, mValues, trace);
    }
    return TSM_OK;
  }

  virtual int GetReferenceTrace (size_t wellIx, size_t flowIx,
                                 float *traceBegin) {
    size_t row, col;
    IndexToRowCol (wellIx, row, col);
    mRefReduction[flowIx].GetSmoothEstFrames(row, col, traceBegin);
    return TSM_OK;
  }

  /* virtual int GetReferenceTrace (size_t wellIx, size_t flowIx, */
  /*                                float *traceBegin) { */
  /*   int wIdx = wellIx; */
  /*   int fIdx = flowIx; */
  /*   size_t row, col; */
  /*   IndexToRowCol (wellIx, row, col); */
  /*   vector<float> &ref = mFineRefGrids[fIdx].GetItemByRowCol (row, col); */
  /*   //vector<float> &ref = mRefGrids[fIdx].GetItemByRowCol (row, col); */
  /*   copy (ref.begin(), ref.end(), traceBegin); */
  /*   return TSM_OK; */
  /* } */

  virtual void SetT0 (std::vector<float> &t0)
  {
    mT0 = t0;
  }

  virtual float GetT0 (int idx)
  {
    return mT0[idx];
  }

  void SetMeshDist (int size) { 
    mUseMeshNeighbors = size;     
    mMaxDist = (size+1) * sqrt(mRowRefStep*mRowRefStep + mColRefStep+mColRefStep); 
  }

  int GetMeshDist() { return mUseMeshNeighbors; }
  float GetMaxDist() { return mMaxDist; }

  /** wIx is the well index from mWellIndex, not the usual global one. */
  inline size_t ToIdx (size_t wIx, size_t frameIx, size_t flowIx)
  {
    // Organize so flows for same well are near each other.
    return (wIx + flowIx * mFrameStride + frameIx * mFlowFrameStride);
  }


  int CalcMedianReference (size_t row, size_t col,
                           GridMesh<std::vector<float> > &regionMed,
                           std::vector<double> &dist,
                           std::vector<std::vector<float> *> &values,
                           std::vector<float> &reference) {
    int retVal = TraceStore::TS_OK;
    reference.resize(mFrames);
    std::fill(reference.begin(), reference.end(), 0.0);
    regionMed.GetClosestNeighbors (row, col, mUseMeshNeighbors, dist, values);
    int num_good = 0;
    size_t valSize = values.size();
    for (size_t i =0; i < valSize; i++) {
      if (values[i]->size() > 0) {
        num_good++;
      }
    }
    // try reaching a little farther if no good reference close by
    if (num_good == 0) {
      regionMed.GetClosestNeighbors (row, col, 2*mUseMeshNeighbors, dist, values);
    }
    size_t size = 0;
    double maxDist = 0;
    for (size_t i = 0; i < values.size(); i++) {
        size = max (values[i]->size(), size);
        maxDist = max(dist[i], maxDist);
    }
    reference.resize (size);
    std::fill (reference.begin(), reference.end(), 0.0);
    double distWeight = 0;
         valSize = values.size();

    for (size_t i = 0; i < valSize; i++) {
      if (values[i]->size()  == 0) {
        continue;
      }
      double w = TraceStore::WeightDist (dist[i], mRowRefStep); //1/sqrt(dist[i]+1);
      distWeight += w;
      size_t vSize = values[i]->size();
      for (size_t j = 0; j < vSize; j++) {
        reference[j] += w * values[i]->at (j);
      }
    }
    // Divide by our total weight to get weighted mean
    if (distWeight > 0)  {
      for (size_t i = 0; i < reference.size(); i++) {
        reference[i] /= distWeight;
      }
      retVal = TraceStore::TS_OK;
    }
    else {
      retVal = TraceStore::TS_BAD_DATA;
    }
    return retVal;
  }

  int CalcRegionReference (int rowStart, int rowEnd,
                           int colStart, int colEnd, size_t flowIx,
                           std::vector<float> &trace) {
    trace.resize (mFrames);
    std::fill (trace.begin(), trace.end(), 0.0f);
    vector<vector<float> > matrix;
    vector<float> traceBuffer (mFrames,0);
    matrix.resize (trace.size());
    for (int rowIx = rowStart; rowIx < rowEnd; rowIx++)
      {
        for (int colIx = colStart; colIx < colEnd; colIx++)
          {
            size_t wellIdx = RowColToIndex (rowIx,colIx);
            if (mUseAsReference[wellIdx])
              {
                GetTrace (wellIdx, flowIx, traceBuffer.begin());
                for (size_t frameIx = 0; frameIx < traceBuffer.size(); frameIx++)
                  {
                    if (isfinite (traceBuffer[frameIx]))
                      {
                        matrix[frameIx].push_back (traceBuffer[frameIx]);
                      }
                  }
              }
          }
      }
    size_t length = matrix[0].size();
    size_t size = matrix.size();
    size_t minRefProbes = max(TSM_MIN_REF_PROBES, (int)floor(1.0 * mMinRefProbes * (1.0 * (rowEnd - rowStart) * (colEnd-colStart) / (mRowRefStep * mColRefStep))));
    if (length >= minRefProbes)
      {
        for (size_t i = 0; i < size; i++)
          {
            std::sort (matrix[i].begin(), matrix[i].end());
            float med = 0;
            if (matrix[i].size() % 2 == 0)
              {
                med = (matrix[i][length / 2] + matrix[i][ (length / 2)-1]) /2.0;
              }
            else
              {
                med = matrix[i][length/2];
              }
            trace[i] = med;
          }
        return TraceStore::TS_OK;
      }
    else
      {
        trace.resize (0);
      }
    return TraceStore::TS_BAD_REGION;
  }

  int16_t *GetMemPtr() { return &mData[0]; }

  bool RegionOk(int rowStart, int rowEnd, int colStart, int colEnd, std::vector<char> &filteredWells,
                int minWells, float minPercent) {
    int count = 0;
    float totalWells = (colEnd - colStart) * (rowEnd - rowStart);
    for (int rowIx = rowStart; rowIx < rowEnd; rowIx++) {
      for (int colIx = colStart; colIx < colEnd; colIx++) {
        if (filteredWells[rowIx * mCols + colIx] == 0) {
          count++;
        }
      }
    }
    float percent = count/ totalWells;
    return( count >= minWells && percent >= minPercent);
  }

  void CalcReference (size_t rowStep, size_t colStep, size_t flowIx,
                      GridMesh<std::vector<float> > &gridReference,
                      std::vector<char> &filteredWells) {
    gridReference.Init (mRows, mCols, rowStep, colStep);
    int numBin = gridReference.GetNumBin();
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    for (int binIx = 0; binIx < numBin; binIx++) {
        gridReference.GetBinCoords (binIx, rowStart, rowEnd, colStart, colEnd);
        vector<float> &trace = gridReference.GetItem (binIx);
        if (RegionOk(rowStart,rowEnd, colStart,colEnd, filteredWells, MIN_REGION_REF_WELLS, MIN_REGION_REF_PERCENT)) {
          CalcRegionReference (rowStart, rowEnd, colStart, colEnd, flowIx, trace);
        }
        else {
          trace.resize(0);
        }
    }
  }

  void WellProj(TraceStoreCol &store,
                std::vector<KeySeq> & key_vectors,
                vector<char> &filter,
                vector<float> &mad);

  size_t mFrames;
  size_t mFrameStride;
  size_t mFlowFrameStride;
  size_t mFlows;
  size_t mRows;
  size_t mCols;
  size_t mWells;
  size_t mFlowsBuf;
  size_t mRowRefStep;
  size_t mColRefStep;
  size_t mMinRefProbes;

  std::string mFlowOrder;
  std::vector<bool> mUseAsReference;
  std::vector<char> mRefWells;
  std::vector<int16_t> mData;
  std::vector<int> mRefGridsValid;
  std::vector<GridMesh<std::vector<float> > > mRefGrids;
  std::vector<GridMesh<std::vector<float> > > mFineRefGrids;
  std::vector<double> mDist;
  std::vector<std::vector<float> *> mValues;
  std::vector<float> mReference;
  std::vector<float> mT0;
  std::vector<double> mTime;
  std::vector<ChipReduction> mRefReduction;
  float mMaxDist;
  // Cube<int8_t> mData;  // rows = frames, cols = wells,
  pthread_mutex_t mLock;
  int mUseMeshNeighbors;
  };

#endif // TRACESTORECOL_H
