/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef TRACES_H
#define TRACES_H

#include <vector>
#include <string>
#include "Image.h"
#include "Mask.h"
#include "GridMesh.h"
#include "SampleStats.h"
#include "MathOptim.h"
#include "ReportSet.h"
#include "FindSlopeChange.h"
#define OK 0
#define BAD_DATA 1
#define BAD_FIT 2
#define NOT_AVAIL 3
#define EXCLUDE 4
#define BAD_REGION 5
#define BAD_VAR 6
#define BAD_T0 7

#define FRAME_ZERO_WINDOW 20
// #define T0STEP 50
//#define STEP 50
//#define BFSTEP 50
#define MIN_T0_PROBES 50
#define MIN_TOT_PROBES 128
#define MIN_REF_PROBES 50

class Traces
{

  public:

    Traces()
    {
      mMask = NULL;
      mFrames = 0;
      mRow = 0;
      mCol = 0;
      mCurrentData = NULL;
      mRawTraces = NULL;
      mCriticalTraces = NULL;
      mT0Step = 50;
      mUseMeshNeighbors = 1;
      mFlow = 0;
    }

  static float WeightDist(float dist) {
    return ExpApprox(-1*dist/60.0);
    //		return 1/log(dist + 3);
  }

    ~Traces();

    size_t RowColToIndex (size_t rowIx, size_t colIx) { return rowIx * mCol + colIx; }

    void IndexToRowCol (size_t index, size_t &rowIx, size_t &colIx)
    {
      rowIx = index / mCol;
      colIx = index % mCol;
    }

    size_t GetNumRow() { return mRow; }
    size_t GetNumCol() { return mCol; }

    int DCOffset (size_t startFrame, size_t endFrame);
    void DCOffsetT0();

    void CalcIncorpBreakRegionStart (size_t rowStart, size_t rowEnd,
                                     size_t colStart, size_t colEnd,
                                     SampleStats<float> &starts);
    //                 SampleStats<float> &starts, MaskType maskType);

    void CalcIncorporationRegionStart (size_t rowStart, size_t rowEnd,
                                       size_t colStart, size_t colEnd,
                                       SampleStats<float> &starts,  MaskType maskType=MaskEmpty);

    void CalcIncorporationStartReference (int nRowBin, int nColBin,
                                          GridMesh<SampleStats<float> > &grid);

    bool CalcStartFrame (size_t row, size_t col,
                         GridMesh<SampleStats<float> > &regionStart, float &start);

    void CalcT0Reference();

    void SetReferenceOut (std::ostream *out)
    {
      mRefOut = out;
    }

    void SetFlow (int flow)
    {
      mFlow = flow;
    }

    void CalcT0 (bool force=false);

    const std::vector<float> &GetT0() { return mT0; }

    void SetT0 (const std::vector<float> &t0) { mT0 = t0; }

    void SetT0 (float t0, size_t index) { mT0[index] = t0; }

    float GetT0 (int idx) { return mT0[idx]; }

    bool FillCriticalWellFrames (size_t idx, int nFrames);

    bool FillCriticalFrames();

    void CalcReference (int rowStep, int colStep, GridMesh<std::vector<float> > &gridReference);

    const std::vector<std::vector<float> > &GetReportMedTraces() { return mMedTraces; }

    const std::vector<std::vector<float> > &GetReportTraces();

    const std::vector<std::vector<float> > &GetReportCriticalFrames();

    void SetReportSampling (const ReportSet &set, bool keepExclude);

    void MakeReportSubset (const std::vector<std::vector<float> > &source, std::vector<std::vector<float> > &fill);

    void MakeReportSubset (const std::vector<std::vector<int8_t> > &source, std::vector<std::vector<float> > &fill);

    void Init (Image *img, Mask *m);

    void Init (Image *img, Mask *mask, int startFrame, int endFrame,
               int dcOffsetStart, int dcOffsetEnd) ;

    void SetMask (Mask *mask);

    static int maxWidth (const std::vector<std::vector<float> > &d);

    static int maxWidth (const std::vector<std::vector<int8_t> > &d);

    static int maxWidth (const std::vector<std::vector<double> > &d);

    int CalcRegionReference (unsigned int type, int rowStart, int rowEnd,
                             int colStart, int colEnd,
                             std::vector<float> &trace);

    int CalcMedianReference (int row, int col,
                             GridMesh<std::vector<float> > &regionMed,
                             std::vector<double> &dist,
                             std::vector<std::vector<float> *> &values,
                             std::vector<float> &reference);

    int CalcMedianReference (size_t idx,
                             GridMesh<std::vector<float> > &regionMed,
                             std::vector<double> &dist,
                             std::vector<std::vector<float> *> &values,
                             std::vector<float> &reference)
    {
      size_t row, col;
      IndexToRowCol (idx, row, col);
      return CalcMedianReference (row, col, regionMed, dist, values, reference);
    }

    void T0DcOffset (int t0Minus, int t0Plus);
    void SetTraces (int idx, const std::vector<float> &trace, int8_t *outData);

    void PrintVec (const std::vector<float> &x);

    void PrintVec (const std::vector<int8_t> &x);

    void PrintTrace (int idx);

    template<typename OutIter>
    void GetTraces (int idx, OutIter out)
    {
      int index = mIndexes[idx];
      if (index >= 0)
      {
        *out = mCurrentData[index];
        for (size_t i = 1; i < mFrames; i++)
        {
          float next = (*out) + (mCurrentData[index+i]);
          * (++out) = next;
        }
      }
    }

    void GetTraces (int idx, std::vector<float> &trace)
    {
      // @todo - make sure trace is correct size already
      trace.resize (mFrames);
      GetTraces (idx, trace.begin());
    }

    int GetNumFrames()
    {
      return mFrames;
    }

    bool IsGood (int idx) { return mFlags[idx] == OK; }

    int GetFlag (int idx) { return mFlags[idx]; }

    /* template<typename OutIter> */
    /*   void GetTraces(int idx, OutIter out) { */
    /*   size_t size = mTraces[idx].size(); */
    /*   if (size > 0) { */
    /*     *out = mTraces[idx][0]; */
    /*     for (size_t i = 1; i < size; i++) { */
    /* float next =  (*out) + (mTraces[idx][i]); */
    /* *(++out) = next; */
    /*     } */
    /*   } */
    /*  }*/

    /* Set the region step size for t0 reference calc. */
    void SetT0Step (int size) { mT0Step = size; }
    int GetT0Step() { return mT0Step; }

    void SetMeshDist (int size) { mUseMeshNeighbors = size; }
    int GetMeshDist() { return mUseMeshNeighbors; }

    size_t mCol;
    size_t mRow;
    size_t mFrames;
    Mask *mMask;
    std::vector<int> mTimes;
    std::vector<float> mT0;
    //  std::vector<std::vector<int8_t> > mTraces;
    int8_t *mCurrentData;
    int8_t *mRawTraces;
    int8_t *mCriticalTraces;
    std::vector<int64_t> mIndexes;
    std::vector<int8_t> mFlags;
    GridMesh<std::vector<float> > mGridMedian;
    std::vector<int> mSampleMap;
    std::vector<std::vector<int> > mReportIdx;
    std::vector<std::vector<float> > mCriticalFrames;
    std::vector<std::vector<float> > mMedTraces;
    std::vector<std::vector<float> > mReportTraces;
    std::vector<std::vector<float> > mReportCriticalFrames;
    GridMesh<SampleStats<float> > mT0ReferenceGrid;
    FindSlopeChange<float> finder;
    vector<float> mTraceBuffer;
    int mT0Step;
    int mUseMeshNeighbors;
    int mFlow;
    std::ostream *mRefOut;
};

#endif // TRACES_H
