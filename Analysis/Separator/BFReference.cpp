/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <algorithm>
#include <iostream>
#include <limits>
#include "BFReference.h"
#include "Image.h"
#include "IonErr.h"
#include "Traces.h"
#include "Stats.h"
using namespace std;
#define FRAMEZERO 0
#define FRAMELAST 150
#define FIRSTDCFRAME 3
#define LASTDCFRAME 12

BFReference::BFReference() {
  mDoRegionalBgSub = false;
  mMinQuantile = -1;
  mMaxQuantile = -1;
  mDcStart = 5;
  mDcEnd = 15;
  mRegionXSize = 50;
  mRegionYSize = 50;
  mNumEmptiesPerRegion = -1;
  mIqrOutlierMult = 2.5;
}

void BFReference::Init(int nRow, int nCol, 
		       int nRowStep, int nColStep, 
		       double minQ, double maxQ) {
  mGrid.Init(nRow, nCol, nRowStep, nColStep);
  
  mMinQuantile = minQ;
  mMaxQuantile = maxQ;
  assert(mMinQuantile >= 0);
  assert(mMinQuantile <= 1);
  assert(mMaxQuantile >= 0);
  assert(mMaxQuantile <= 1);
  
  mWells.resize(nRow * nCol);
  fill(mWells.begin(), mWells.end(), Unknown);
}

bool BFReference::InSpan(size_t rowIx, size_t colIx,
			 const std::vector<int> &rowStarts,
			 const std::vector<int> &colStarts,
			 int span)  {
  for (size_t rIx = 0; rIx < rowStarts.size(); rIx++) {
    if ((int)rowIx >= rowStarts[rIx] && (int)rowIx < (rowStarts[rIx] + span) &&
	(int)colIx >= colStarts[rIx] && (int)colIx < (colStarts[rIx] + span)) {
      return true;
    }
  }
  return false;
}


void BFReference::DebugTraces(const std::string &fileName,  Mask &mask, Image &bfImg) {
  ION_ASSERT(!fileName.empty(), "Have to have non-zero length file name");
  ofstream out(fileName.c_str());
  const RawImage *raw = bfImg.GetImage();
  vector<int> rowStarts;
  vector<int> colStarts;
  size_t nRow = raw->rows;
  size_t nCol = raw->cols;
  size_t nFrames = raw->frames;
  double percents[3] = {.2, .5, .8};
  int span = 7;
  for (size_t i = 0; i < ArraySize(percents); i++) {
    rowStarts.push_back(percents[i] * nRow);
    colStarts.push_back(percents[i] * nCol);
  }
  char d = '\t';
  for (size_t rowIx = 0; rowIx < nRow; rowIx++) {
    for (size_t colIx = 0; colIx < nCol; colIx++) {
      if (InSpan(rowIx, colIx, rowStarts, colStarts, span)) {
	out << rowIx << d << colIx;
	for (size_t frameIx = 0; frameIx < nFrames; frameIx++) {
	  out << d << '\t' << bfImg.GetInterpolatedValue(frameIx,colIx,rowIx);
	}
	out << endl;
      }
    }
  }
  out.close();
}

void BFReference::CalcReference(const std::string &datFile, Mask &mask) {
  CalcReference(datFile, mask, mBfMetric);
  for (size_t i = 0; i < mBfMetric.size(); i++) {
    if (mask[i] & MaskExclude || mask[i] & MaskPinned) {
      mWells[i] = Exclude;
    }
    else {
      mask[i] = MaskIgnore;
    }
  }
  cout << "Filling reference. " << endl;
  FillInReference(mWells, mBfMetric, mGrid, mMinQuantile, mMaxQuantile, mNumEmptiesPerRegion);
  for (size_t i = 0; i < mBfMetric.size(); i++) {
    if (mWells[i] == Reference) {
      mask[i] = MaskEmpty;
    }
  }
}

void BFReference::CalcDualReference(const std::string &datFile1, const std::string &datFile2, Mask &mask) {
  vector<float> metric1, metric2;
  CalcReference(datFile1, mask, metric1);
  CalcReference(datFile2, mask, metric2);
  mBfMetric.resize(metric1.size(), 0);
  for (size_t i = 0; i < metric1.size(); i++) {
    mBfMetric[i] = (metric1[i] + metric2[i])/2.0f;
  }
  for (size_t i = 0; i < mBfMetric.size(); i++) {
    if (mask[i] & MaskExclude || mask[i] & MaskPinned || mask[i] & MaskIgnore) {
      mWells[i] = Exclude;
    }
    else {
      mask[i] = MaskIgnore;
    }
  }
  cout << "Filling reference. " << endl;
  FillInReference(mWells, mBfMetric, mGrid, mMinQuantile, mMaxQuantile, mNumEmptiesPerRegion);
  for (size_t i = 0; i < mWells.size(); i++) {
    if (mWells[i] == Reference) {
      mask[i] = MaskEmpty;
    }
  }
  // ofstream out("bfreference.txt");
  // for (size_t i = 0; i < mWells.size(); i++) {
  //   out << i << "\t" << (int)mWells[i] << endl;
  // }
}

void BFReference::FilterRegionOutliers(Image &bfImg, Mask &mask, float iqrThreshold, 
                                       int rowStart, int rowEnd, int colStart, int colEnd) {

  const RawImage *raw = bfImg.GetImage();
  /* Figure out how many wells are not pinned/excluded right out of the gate. */
  int okCount = 0;
  for (int r = rowStart; r < rowEnd; r++) {
    for (int c = colStart; c < colEnd; c++) {
      int idx = r * raw->cols + c;
      if (!mask.Match(c, r, MaskPinned) && !mask.Match(c,r,MaskExclude) && 
          mWells[idx] != Exclude && mWells[idx] != Filtered) {
        okCount++;
      }
    }
  }
  /* If not enough, just mark them all as bad. */
  if (okCount <= MIN_OK_WELLS || (mNumEmptiesPerRegion > 0 && okCount < mNumEmptiesPerRegion)) {
    for (int r = rowStart; r < rowEnd; r++) {
      for (int c = colStart; c < colEnd; c++) {
        mWells[r * raw->cols + c] = Exclude;
      }
    }
    return;
  }

  // Make a mtrix for our region
  mTraces.set_size(okCount, raw->frames ); // wells in row major order by frames
  int count = 0;

  vector<int> mapping(mTraces.n_rows, -1);
  for (int r = rowStart; r < rowEnd; r++) {
    for (int c = colStart; c < colEnd; c++) {
      int idx = r * raw->cols + c;
      if (!mask.Match(c, r, MaskPinned) && !mask.Match(c,r,MaskExclude) && 
          mWells[idx] != Exclude && mWells[idx] != Filtered) {
        for (int f = 0; f < raw->frames; f++) {
          mTraces.at(count,f) = bfImg.At(r,c,f) -  bfImg.At(r,c,0);
        }
        mapping[count++] = r * raw->cols + c;
      }
    }
  }
  for (size_t r = 0; r < mTraces.n_rows; r++) {
    for (size_t c = 0; c < mTraces.n_cols; c++) {
      assert(isfinite(mTraces.at(r,c)));
    }
  }
  assert(mapping.size() == (size_t)count);

  /* Subtract off the median. */
  fmat colMed = median(mTraces);
  frowvec colMedV = colMed.row(0);
  for(size_t i = 0; i < mTraces.n_rows; i++) {
    mTraces.row(i) = mTraces.row(i) - colMedV;
  }

  /* Get the quantiles of the mean difference for well and exclude outliers */
  fmat mad = mean(mTraces, 1);
  fvec madV = mad.col(0);
  mMadSample.Clear();
  mMadSample.Init(1000);
  mMadSample.AddValues(madV.memptr(), madV.n_elem);
  float minVal = mMadSample.GetQuantile(.25) - iqrThreshold * mMadSample.GetIQR();
  float maxVal = mMadSample.GetQuantile(.75) + iqrThreshold * mMadSample.GetIQR();
  for (size_t i = 0; i < madV.n_rows; i++) {
    if (madV[i] <= minVal || madV[i] >= maxVal) {
      mWells[mapping[i]] = Filtered;
    }
  }
}

void BFReference::FilterForOutliers(Image &bfImg, Mask &mask, float iqrThreshold, int rowStep, int colStep) {
  const RawImage *raw = bfImg.GetImage();
  GridMesh<float> grid;
  grid.Init(raw->rows, raw->cols, rowStep, colStep);
  int numBin = grid.GetNumBin();
  int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  for (int binIx = 0; binIx < numBin; binIx++) {
    grid.GetBinCoords(binIx, rowStart, rowEnd, colStart, colEnd);
    FilterRegionOutliers(bfImg, mask, iqrThreshold, rowStart, rowEnd, colStart, colEnd);
  }
}

void BFReference::CalcReference(const std::string &datFile, Mask &mask, std::vector<float> &metric) {
  Image bfImg;
  bfImg.SetImgLoadImmediate (false);
  bool loaded = bfImg.LoadRaw(datFile.c_str());
  if (!loaded) {
    ION_ABORT("*Error* - No beadfind file found, did beadfind run? are files transferred?  (" + datFile + ")");
  }

  const RawImage *raw = bfImg.GetImage();
  
  assert(raw->cols == GetNumCol());
  assert(raw->rows == GetNumRow());
  assert(raw->cols == mask.W());
  assert(raw->rows == mask.H());
  if (!mDebugFile.empty()) {
    DebugTraces(mDebugFile, mask, bfImg);
  }
  bfImg.FilterForPinned(&mask, MaskEmpty, false);
  // int StartFrame= bfImg.GetFrame((GetDcStart()*1000/15)-1000);
  // int EndFrame = bfImg.GetFrame((GetDcEnd()*1000/15)-1000);
  int StartFrame = bfImg.GetFrame(-663); //5
  int EndFrame = bfImg.GetFrame(350); //20
  cout << "DC start frame: " << StartFrame << " end frame: " << EndFrame << endl;
  bfImg.XTChannelCorrect();
  FilterForOutliers(bfImg, mask, mIqrOutlierMult, mRegionYSize, mRegionXSize);
  bfImg.Normalize(StartFrame, EndFrame);
  // bfImg.XTChannelCorrect(&mask);

  int NNinnerx = 1, NNinnery = 1, NNouterx = 12, NNoutery = 8;
  if (mDoRegionalBgSub) {
    GridMesh<float> grid;
    grid.Init(raw->rows, raw->cols, mRegionYSize, mRegionXSize);
    int numBin = grid.GetNumBin();
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    for (int binIx = 0; binIx < numBin; binIx++) {
      grid.GetBinCoords(binIx, rowStart, rowEnd, colStart, colEnd);
      Region reg;
      reg.row = rowStart;
      reg.h = rowEnd - rowStart;
      reg.col = colStart;
      reg.w = colEnd - colStart;
      bfImg.BackgroundCorrectRegion(&mask, reg, MaskAll, MaskEmpty, NNinnerx, NNinnery, NNouterx, NNoutery, NULL);
    }
  }
  else {
    bfImg.BackgroundCorrect(&mask, MaskEmpty, MaskEmpty, NNinnerx, NNinnery, NNouterx, NNoutery, NULL);
  }
  Region region;
  region.col = 0;
  region.row = 0;
  region.w = GetNumCol(); //mGrid.GetColStep();
  region.h = GetNumRow(); // mGrid.GetRowStep();

  int startFrame = bfImg.GetFrame(12); // frame 15 on uncompressed 314
  //  int endFrame = bfImg.GetFrame(raw->timestamps[bfImg.Ge]5300); // frame 77 or so
  int endFrame = bfImg.GetFrame(5000); // frame 77 or so
  bfImg.CalcBeadfindMetric_1(&mask, region, "pre", startFrame, endFrame);
  const double *results = bfImg.GetResults();

  int length = GetNumRow() * GetNumCol();
  metric.resize(length);
  copy(&results[0], &results[0] + (length), metric.begin());
  bfImg.Close();
}

void BFReference::CalcSignalReference2(const std::string &datFile, const std::string &bgFile,
				      Mask &mask, int traceFrame) {
  Image bfImg;
  Image bfBkgImg;
  bfImg.SetImgLoadImmediate (false);
  bfBkgImg.SetImgLoadImmediate (false);
  bool loaded = bfImg.LoadRaw(datFile.c_str());
  bool bgLoaded = bfBkgImg.LoadRaw(bgFile.c_str());
  if (!loaded) {
    ION_ABORT("*Error* - No beadfind file found, did beadfind run? are files transferred?  (" + datFile + ")");
  }
  if (!bgLoaded) {
    ION_ABORT("*Error* - No beadfind background file found, did beadfind run? are files transferred?  (" + bgFile + ")");
  }
  const RawImage *raw = bfImg.GetImage();
  
  assert(raw->cols == GetNumCol());
  assert(raw->rows == GetNumRow());
  assert(raw->cols == mask.W());
  assert(raw->rows == mask.H());
  int StartFrame = bfImg.GetFrame(-663); //5
  int EndFrame = bfImg.GetFrame(350); //20
  int NNinnerx = 1, NNinnery = 1, NNouterx = 12, NNoutery = 8;
  cout << "DC start frame: " << StartFrame << " end frame: " << EndFrame << endl;
  bfImg.FilterForPinned(&mask, MaskEmpty, false);
  bfImg.XTChannelCorrect();
  // bfImg.XTChannelCorrect(&mask);
  Traces trace;  
  trace.Init(&bfImg, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME,LASTDCFRAME);
  bfImg.Normalize(StartFrame, EndFrame);
  if (mDoRegionalBgSub) {
     trace.SetMeshDist(0);
  }
  trace.CalcT0(true);
  if (mDoRegionalBgSub) {
    GridMesh<float> grid;
    grid.Init(raw->rows, raw->cols, mRegionYSize, mRegionXSize);
    int numBin = grid.GetNumBin();
    int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
    for (int binIx = 0; binIx < numBin; binIx++) {
      cout << "BG Subtract Region: " << binIx << endl;
      grid.GetBinCoords(binIx, rowStart, rowEnd, colStart, colEnd);
      Region reg;
      reg.row = rowStart;
      reg.h = rowEnd - rowStart;
      reg.col = colStart;
      reg.w = colEnd - colStart;
      bfImg.BackgroundCorrectRegion(&mask, reg, MaskAll, MaskEmpty, NNinnerx, NNinnery, NNouterx, NNoutery, NULL);
    }
  }
  else {
    bfImg.BackgroundCorrect(&mask, MaskEmpty, MaskEmpty, NNinnerx, NNinnery, NNouterx, NNoutery, NULL);
  }
  int length = GetNumRow() * GetNumCol();
  mBfMetric.resize(length, std::numeric_limits<double>::signaling_NaN());
  for (int wIx = 0; wIx < length; wIx++) {
    if (mask[wIx] & MaskExclude || mask[wIx] & MaskPinned) 
      continue;
    int t0 = (int)trace.GetT0(wIx);
    mBfMetric[wIx] = 0;
    float zSum  = 0;
    int count = 0;
    for (int fIx = min(t0-20, 0); fIx < t0-10; fIx++) {
      zSum += bfImg.At(wIx,fIx);
      count ++;
    }
    for (int fIx = t0+3; fIx < t0+15; fIx++) {
      mBfMetric[wIx] += (bfImg.At(wIx,fIx) - (zSum / count));
    }
  }
  bfImg.Close();
  for (int i = 0; i < length; i++) {
    if (mask[i] & MaskExclude || mWells[i] == Exclude) {
      mWells[i] = Exclude;
    }
    else {
      mask[i] = MaskIgnore;
    }
  }
  cout << "Filling reference. " << endl;
  FillInReference(mWells, mBfMetric, mGrid, mMinQuantile, mMaxQuantile, mNumEmptiesPerRegion);
  for (int i = 0; i < length; i++) {
    if (mWells[i] == Reference) {
      mask[i] = MaskEmpty;
    }
  }
}

void BFReference::CalcSignalReference(const std::string &datFile, const std::string &bgFile,
				      Mask &mask, int traceFrame) {
  Image bfImg;
  Image bfBkgImg;
  bfImg.SetImgLoadImmediate (false);
  bfBkgImg.SetImgLoadImmediate (false);
  bool loaded = bfImg.LoadRaw(datFile.c_str());
  bool bgLoaded = bfBkgImg.LoadRaw(bgFile.c_str());
  if (!loaded) {
    ION_ABORT("*Error* - No beadfind file found, did beadfind run? are files transferred?  (" + datFile + ")");
  }
  if (!bgLoaded) {
    ION_ABORT("*Error* - No beadfind background file found, did beadfind run? are files transferred?  (" + bgFile + ")");
  }
  const RawImage *raw = bfImg.GetImage();
  
  assert(raw->cols == GetNumCol());
  assert(raw->rows == GetNumRow());
  assert(raw->cols == mask.W());
  assert(raw->rows == mask.H());
  bfImg.FilterForPinned(&mask, MaskEmpty, false);
  bfBkgImg.FilterForPinned(&mask, MaskEmpty, false);

  // bfImg.XTChannelCorrect(&mask);
  bfImg.XTChannelCorrect();
  // bfBkgImg.XTChannelCorrect(&mask);
  bfBkgImg.XTChannelCorrect();

  Traces trace;  
  trace.Init(&bfImg, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME,LASTDCFRAME);
  bfImg.Close();
  Traces bgTrace;
  bgTrace.Init(&bfBkgImg, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME,LASTDCFRAME);
  bfBkgImg.Close();
  if (mDoRegionalBgSub) {
    trace.SetMeshDist(0);
    bgTrace.SetMeshDist(0);
  }
  trace.SetT0Step(mRegionXSize);
  bgTrace.SetT0Step(mRegionXSize);
  trace.CalcT0(true);
  size_t numWells = trace.GetNumRow() * trace.GetNumCol();
  for (size_t i = 0; i < numWells; i++) {
    trace.SetT0(max(trace.GetT0(i) - 3, 0.0f), i);
  }
  bgTrace.SetT0(trace.GetT0());
  trace.T0DcOffset(0,4);
  trace.FillCriticalFrames();
  trace.CalcReference(mRegionXSize,mRegionYSize,trace.mGridMedian);
  bgTrace.T0DcOffset(0,4);
  bgTrace.FillCriticalFrames();
  bgTrace.CalcReference(mRegionXSize,mRegionYSize,bgTrace.mGridMedian);

  int length = GetNumRow() * GetNumCol();
  mBfMetric.resize(length, std::numeric_limits<double>::signaling_NaN());
  vector<double> rawTrace(trace.GetNumFrames());
  vector<double> bgRawTrace(bgTrace.GetNumFrames());
  int pinned =0, excluded = 0;
  for (int i = 0; i < length; i++) {
    if (mask[i] & MaskExclude || mask[i] & MaskPinned) {
      continue;
      if (mask[i] & MaskExclude) {
        excluded++;
      }
      else if (mask[i] & MaskPinned) {
        pinned++;
      }
    }
    trace.GetTraces(i, rawTrace.begin());
    bgTrace.GetTraces(i, bgRawTrace.begin());
    mBfMetric[i] = 0;
    for (int s = 3; s < 15; s++) {
      mBfMetric[i] += rawTrace[s] - bgRawTrace[s];
    }
  }
  cout << "Pinned: " << pinned << " excluded: " << excluded << endl;
  for (int i = 0; i < length; i++) {
    if (mask[i] & MaskExclude || mask[i] & MaskPinned || mask[i] & MaskIgnore) {
      mWells[i] = Exclude;
    }
    else {
      mask[i] = MaskIgnore;
    }
  }
  cout << "Filling reference. " << endl;
  FillInReference(mWells, mBfMetric, mGrid, mMinQuantile, mMaxQuantile, mNumEmptiesPerRegion);
  for (int i = 0; i < length; i++) {
    if (mWells[i] == Reference) {
      mask[i] = MaskEmpty;
    }
  }
  bfImg.Close();
}

void BFReference::FillInRegionRef(int rStart, int rEnd, int cStart, int cEnd,
				  std::vector<float> &metric, 
				  double minQuantile, double maxQuantile,
                                  int numWells,
				  std::vector<char> &wells) {
  std::vector<std::pair<float,int> > wellMetric;
  for (int rIx = rStart; rIx < rEnd; rIx++) {
    for (int cIx = cStart; cIx < cEnd; cIx++) {
      int idx = RowColIdx(rIx,cIx);
      if (wells[idx] != Exclude && wells[idx] != Filtered && isfinite(metric[idx])) {
	wellMetric.push_back(std::pair<float,int>(metric[idx],idx));
      }
    }
  }

  if (numWells <= 0) {
    std::sort(wellMetric.rbegin(), wellMetric.rend());
    int minQIx = Round(minQuantile * wellMetric.size());
    int maxQIx = Round(maxQuantile * wellMetric.size());
    for (int qIx = minQIx; qIx < maxQIx; qIx++) {
      wells[wellMetric[qIx].second] = Reference;
    }
  }
  else {
    std::sort(wellMetric.begin(), wellMetric.end());
    int nWells =  min((size_t)numWells, wellMetric.size());
    for (int i = 0; i < nWells; i++) {
      wells[wellMetric[i].second] = Reference;
    }
  }
}

void BFReference::FillInReference(std::vector<char> &wells, 
				  std::vector<float> &metric,
				  GridMesh<int> &grid,
				  double minQuantile,
				  double maxQuantile,
                                  int numWells) {
  int rStart = -1, rEnd = -1, cStart = -1, cEnd = -1;
  for (size_t bIx = 0; bIx < grid.GetNumBin(); bIx++) {
    grid.GetBinCoords(bIx, rStart, rEnd, cStart, cEnd);
    FillInRegionRef(rStart, rEnd, cStart, cEnd,
		    metric, minQuantile, maxQuantile,numWells, wells);
  }
} 
 
