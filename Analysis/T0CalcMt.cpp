/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "T0CalcMt.h"

void T0RegionJob::Run() {
  for (size_t bIx = mStartBin; bIx < mEndBin; bIx++) {
    std::pair<size_t, std::vector<float> > &bin = mCalc->mRegionSum.GetItem(bIx);
    if (bin.first < 20) {
      continue;
    }
    for (size_t fIx = 0; fIx < bin.second.size(); fIx++) {
      bin.second[fIx] = bin.second[fIx] / bin.first;
    }
    T0Prior &prior = mCalc->mT0Prior[bIx];
    float &t0 = mCalc->mT0.GetItem(bIx);
    float &slope =  mCalc->mSlope.GetItem(bIx);
    T0CalcMt::CalcT0(mFinder, bin.second, *mTimeStamps, prior, t0, slope);
  }
}

void T0AccumulateJob::Run() {
  int rowStart, rowEnd, colStart, colEnd;
  for (size_t bIx = mStartBin; bIx < mEndBin; bIx++) {
    mCalc->mRegionSum.GetBinCoords(bIx, rowStart, rowEnd, colStart, colEnd);
    mCalc->CalcSumTrace(mData, rowStart, rowEnd, colStart, colEnd);
  }
}
