/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef REGIONAVGKEYREPORTER_H
#define REGIONAVGKEYREPORTER_H

#include <string>
#include <vector>
#include <pthread.h>
#include "KeyReporter.h"
#include "SampleStats.h"
#include "SampleQuantiles.h"
#include "KeyClassifier.h"
#include "GridMesh.h"

#define KEY_SAMPLE_SIZE 1000

template <class T>
class RegionAvgKeyReporter : public KeyReporter<T> {
  
public:
  void Init(const std::string &prefix, int nRows, int nCols,
            int rowStep, int colStep, const std::vector<KeySeq> &keys,
            const std::vector<float> &t0) {
    mPrefix = prefix;
    pthread_mutex_init(&mLock, NULL);
    mRegionAvgTraces.Init(nRows, nCols, rowStep, colStep);
 
    mKeys = keys;
    mT0.resize(mRegionAvgTraces.GetNumBin(), -1);
    mMinPeakSig.resize(keys.size());
    std::fill(mMinPeakSig.begin(), mMinPeakSig.end(), std::numeric_limits<double>::max() * -1);
    
    for (size_t binIx = 0; binIx < mRegionAvgTraces.GetNumBin(); binIx++) {
      int rowBin, colBin;
      mRegionAvgTraces.IndexToXY((int)binIx, rowBin, colBin); 
      int rowStart,rowEnd,colStart,colEnd; 
      mRegionAvgTraces.GetBinCoords(binIx, rowStart, rowEnd, colStart, colEnd);
      double sum = 0;
      size_t count = 0;
      for (int rIx = rowStart; rIx < rowEnd; rIx++) {
        for (int cIx = colStart; cIx < colEnd; cIx++) {
          size_t wIx = rIx * nCols + cIx;
          if (t0[wIx] > 0) {
            sum += t0[wIx];
            count++;
          }
        }
      }
      if (count > 0) {
        sum = sum / count;
      }
      mT0[binIx] = sum;
      /* int wellIndex = (rowEnd+rowStart)/2 * mRegionAvgTraces.GetCol() + (colEnd+colStart)/2; */
      /* mT0[binIx] = t0[wellIndex]; */
    }
    mMinPeakSig.resize(keys.size());
    std::fill(mMinPeakSig.begin(), mMinPeakSig.end(), std::numeric_limits<double>::max() * -1);
  }

  ~RegionAvgKeyReporter() {
    pthread_mutex_destroy(&mLock);
  }
  
  void SetMinKeyThreshold(int key, double val) {
    mMinPeakSig[key] = val;
  }

  void Report(const KeyFit &fit, 
	      const Mat<T> &wellFlows,
	      const Mat<T> &refFlows,
	      const Mat<T> &predicted) {
    if (fit.keyIndex < 0 || fit.mad > 50 || refFlows.n_cols == 0) {
      return;
    }
    if (fit.peakSig < mMinPeakSig[fit.keyIndex]) {
      return;
    }
    pthread_mutex_lock(&mLock);
    std::vector<SampleStats<float> > &regionTrace = mRegionAvgTraces.GetItem(mRegionAvgTraces.GetBin(fit.wellIdx));
    if (mGlobalAvg.size() < wellFlows.n_rows) {
      mGlobalAvg.resize(wellFlows.n_rows);
    }
    if (regionTrace.size() < wellFlows.n_rows) {
      regionTrace.resize(wellFlows.n_rows);
    }
    for (size_t flowIx = 0; flowIx < wellFlows.n_cols; flowIx++) {
      if (mKeys[fit.keyIndex].flows[flowIx] == 1) {
        for (size_t frameIx = 0; frameIx < wellFlows.n_rows; frameIx++) {
          float d = wellFlows.at(frameIx,flowIx) - predicted.at(frameIx,flowIx);
          regionTrace[frameIx].AddValue(d);
          mGlobalAvg[frameIx].AddValue(d);
        }
      }
    }
    pthread_mutex_unlock(&mLock);
  }

  void Finish() {
    std::string  s = mPrefix  + ".region-avg-traces.txt";
    mTraces.open(s.c_str());
    s = mPrefix  + ".region-sd-traces.txt";
    mSdTraces.open(s.c_str());
    mRegionTraces.resize(mRegionAvgTraces.GetNumBin());
    for (size_t binIx = 0; binIx < mRegionAvgTraces.GetNumBin(); binIx++) {
      vector<SampleStats<float> > &regionTrace = mRegionAvgTraces.GetItem(binIx);
      mRegionTraces[binIx].resize(regionTrace.size(), 0);
      int rowBin, colBin;
      mRegionAvgTraces.IndexToXY((int)binIx, rowBin, colBin); 
      if (regionTrace.size() > 0 && regionTrace[0].GetCount() > 50) {
        mTraces << binIx << "\t" << rowBin * mRegionAvgTraces.GetRowStep() << "\t" << colBin * mRegionAvgTraces.GetColStep() 
                << "\t" << regionTrace[0].GetCount() << "\t" << mT0[binIx];
        mSdTraces << binIx << "\t" << rowBin * mRegionAvgTraces.GetRowStep() << "\t" << colBin * mRegionAvgTraces.GetColStep() 
                << "\t" << regionTrace[0].GetCount() << "\t" << mT0[binIx];
        for (size_t frameIx = 0; frameIx < regionTrace.size(); frameIx++) {
          mRegionTraces[binIx][frameIx] = regionTrace[frameIx].GetMean();
          mTraces << "\t" << regionTrace[frameIx].GetMean();
          mSdTraces << "\t" << regionTrace[frameIx].GetSD();
        } 
        mTraces << endl;
        mSdTraces << endl;
      }
      else {
        mRegionTraces[binIx].resize(mGlobalAvg.size(), 0);
        for (size_t frameIx = 0; frameIx < mGlobalAvg.size(); frameIx++) {
          mRegionTraces[binIx][frameIx] = mGlobalAvg[frameIx].GetMean();
        }
      }
    }
    mTraces.close();
    mSdTraces.close();
  }

  float *GetAvgKeySig(int region, int rStart, int rEnd, int cStart, int cEnd) {
    region++;
    // Don't use region num as they are row major for some reason in analysis...
    int binIx = mRegionAvgTraces.GetBin((rEnd+rStart)/2, (cEnd+cStart)/2);
    assert(binIx >=0 && binIx < (int)mRegionTraces.size());
    return &mRegionTraces[binIx][0];
  }

  double GetAvgKeySigLen() {
    size_t maxCount = 0;
    for (size_t i = 0; i < mRegionTraces.size(); i++) {
      maxCount = max(mRegionTraces[i].size(), maxCount);
    }
    return (double) maxCount;
  }
  
  float GetStart(int regionNum, int rStart, int rEnd, int cStart, int cEnd) {
    // Don't use region num as they are row major for some reason in analysis...
    int binIx = mRegionAvgTraces.GetBin((rEnd+rStart)/2, (cEnd+cStart)/2);
    assert(binIx >=0 && binIx < (int)mRegionTraces.size());
    return(mT0[binIx]); 
  }

private:
  std::vector<std::vector<float> > mRegionTraces;
  GridMesh<std::vector<SampleStats<float> > >  mRegionAvgTraces;
  std::vector<SampleStats<float> > mGlobalAvg;
  std::vector<double> mMinPeakSig;
  std::ofstream mTraces;
  std::ofstream mSdTraces;
  std::string mPrefix;
  std::vector<KeySeq> mKeys;
  std::vector<float> mT0;
  pthread_mutex_t mLock;  
};

#endif // REGIONAVGKEYREPORTER_H
