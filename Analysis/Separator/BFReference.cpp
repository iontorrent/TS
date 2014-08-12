/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <algorithm>
#include <iostream>
#include <limits>
#include "BFReference.h"
#include "Image.h"
#include "IonErr.h"
#include "Traces.h"
#include "SynchDatSerialize.h"
#include "ComparatorNoiseCorrector.h"
#include "BkgTrace.h"
#include "H5File.h"
#include "H5Arma.h"
#include "Stats.h"

using namespace std;
using namespace arma;

#define FRAMEZERO 0
#define FRAMELAST 150
#define FIRSTDCFRAME 3
#define LASTDCFRAME 12
#define MIN_TRACE_SD 100
#define SAMPLE_PER_REGION 1000
#define BF_INTEGRATION_WIDTH 1
#define BF_INTEGRATION_WINDOW 30
#define THUMBNAIL_STEP 100
#define NUM_EIGEN_FILTER 3
#define MIN_PERCENT_OK_BEADS .3
#define MIN_GOOD_ROWS 50
#define MAX_SD_SEARCH_WINDOW 25
BFReference::BFReference() {
  mDoRegionalBgSub = false;
  mMinQuantile = -1;
  mMaxQuantile = -1;
  mDcStart = 5;
  mDcEnd = 15;
  mRegionXSize = 50;
  mRegionYSize = 50;
  mIqrOutlierMult = 2.5;
  mNumEmptiesPerRegion = 0;
  doSdat = false;
  mIsThumbnail = false;
  mDoComparatorCorrect = false;
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
  std::fill(mWells.begin(), mWells.end(), Unknown);
  mBfMetric.resize(nRow * nCol);
  std::fill(mBfMetric.begin(), mBfMetric.end(), 1.0f);
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

  int numEmptiesPerRegion = max(mNumEmptiesPerRegion, (int)(MIN_PERCENT_OK_BEADS * (rowEnd - rowStart) * (colEnd-colStart) / (mRegionXSize * mRegionYSize)));
  /* If not enough, just mark them all as bad. */
  //  if ((mNumEmptiesPerRegion <= 0 && okCount <= MIN_OK_WELLS) || (mNumEmptiesPerRegion > 0 && okCount <= numEmptiesPerRegion)) {
  if ((mNumEmptiesPerRegion <= 0 && okCount <= MIN_OK_WELLS) || (mNumEmptiesPerRegion > 0 && okCount <= numEmptiesPerRegion)) {
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

int CountFiltered(std::vector<char> &wells) {
  int count = 0;
  for (size_t i = 0; i < wells.size(); i++) {
    if (wells[i] == BFReference::Filtered) {
      count++;
    }
  }
  return count;
}
        

void BFReference::FilterForOutliers(Image &bfImg, Mask &mask, float iqrThreshold, int rowStep, int colStep) {
  const RawImage *raw = bfImg.GetImage();
  GridMesh<float> grid;
  grid.Init(raw->rows, raw->cols, rowStep, colStep);
  int numBin = grid.GetNumBin();
  int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  //  int filteredCountBefore = 0, filteredCountAfter = 0;
  //  filteredCountBefore = CountFiltered(mWells);
  for (int binIx = 0; binIx < numBin; binIx++) {
    grid.GetBinCoords(binIx, rowStart, rowEnd, colStart, colEnd);
    FilterRegionOutliers(bfImg, mask, iqrThreshold, rowStart, rowEnd, colStart, colEnd);
  }
  //  filteredCountAfter = CountFiltered(mWells);
  //  cout << "FilterForOutliers() - Filtered: " << filteredCountAfter - filteredCountBefore << " wells." << endl;
}

void BFReference::CalcReference(Image &bfImg, Mask &mask, BufferMeasurement bf_type) {
  CalcShiftedReference(bfImg, mask, mBfMetric, bf_type);
  for (size_t i = 0; i < mBfMetric.size(); i++) {
    if (mask[i] & MaskExclude || mask[i] & MaskPinned) {
      mWells[i] = Exclude;
    }
    // else {
    //   mask[i] = MaskIgnore;
    // }
  }
  //  cout << "Filling reference. " << endl;
  FillInReference(mWells, mBfMetric, mGrid, mMinQuantile, mMaxQuantile);
  for (size_t i = 0; i < mBfMetric.size(); i++) {
    if (mWells[i] == Reference) {
      mask[i] |= MaskReference;
    }
  }
}
void BFReference::CalcReference(const std::string &datFile, Mask &mask, BufferMeasurement bf_type) {
  CalcShiftedReference(datFile, mask, mBfMetric, bf_type);
  for (size_t i = 0; i < mBfMetric.size(); i++) {
    if (mask[i] & MaskExclude || mask[i] & MaskPinned) {
      mWells[i] = Exclude;
    }
    // else {
    //   mask[i] = MaskIgnore;
    // }
  }
  //  cout << "Filling reference. " << endl;
  FillInReference(mWells, mBfMetric, mGrid, mMinQuantile, mMaxQuantile);
  for (size_t i = 0; i < mBfMetric.size(); i++) {
    if (mWells[i] == Reference) {
      mask[i] |= MaskReference;
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
    if (mask[i] & MaskExclude || mask[i] & MaskPinned) {
      mWells[i] = Exclude;
    }
    // else {
    //   mask[i] = MaskEmpty;
    // }
  }
  //  cout << "Filling reference. " << endl;
  FillInReference(mWells, mBfMetric, mGrid, mMinQuantile, mMaxQuantile);
  for (size_t i = 0; i < mWells.size(); i++) {
    if (mWells[i] == Reference) {
      mask[i] |= MaskReference;
    }
  }
}

bool BFReference::LoadImage(Image &img, const std::string &fileName) {
  bool loaded = false;
  if (doSdat) {
    SynchDat sdat;
    TraceChunkSerializer readSerializer;
    readSerializer.Read(fileName.c_str(), sdat);
    img.InitFromSdat(&sdat);
    loaded = true;
  }
  else {
    loaded = img.LoadRaw(fileName.c_str());
  }
  return loaded;
}

void BFReference::CalcReference(const std::string &datFile, Mask &mask, std::vector<float> &metric) {
  Image bfImg;
  bfImg.SetImgLoadImmediate (false);
  //  bool loaded = bfImg.LoadRaw(datFile.c_str());
  bool loaded = LoadImage(bfImg, datFile);
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
  bfImg.SetMeanOfFramesToZero(StartFrame, EndFrame);
  // bfImg.XTChannelCorrect(&mask);
  ImageTransformer::XTChannelCorrect(bfImg.raw, bfImg.results_folder);
  FilterForOutliers(bfImg, mask, mIqrOutlierMult, mRegionYSize, mRegionXSize);
  Region region;
  region.col = 0;
  region.row = 0;
  region.w = GetNumCol(); //mGrid.GetColStep();
  region.h = GetNumRow(); // mGrid.GetRowStep();

  int startFrame = bfImg.GetFrame(0); // frame 15 on uncompressed 314
  int endFrame = bfImg.GetFrame(5000); // frame 77 or so

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
    bfImg.CalcBeadfindMetricRegionMean(&mask, reg, "pre",startFrame, endFrame);
  }
  const double *results = bfImg.GetResults();

  int length = GetNumRow() * GetNumCol();
  metric.resize(length);
  copy(&results[0], &results[0] + (length), metric.begin());
  bfImg.Close();
}

// void BFReference::CalcAvgNeighborBuffering(int rowStart, int rowEnd, int colStart, int colEnd, Mask &mask, int avg_window,
//                                            std::vector<float> &metric, std::vector<float> &neighbor_avg) {

//   size_t size = (colEnd - colStart) * (rowEnd - rowStart);
//   size_t width = (colEnd - colStart);
//   vector<double> metric_sum(size);
//   vector<int> metric_count(size);
//   fill(metric_sum.begin(), metric_sum.end(), 0.0);
//   fill(metric_counts.begin(), metric_counts.end(), 0.0);
//   for (int row = rowStart; row < rowEnd; row++) {
//     for (int col = colStart; col < colEnd; col++) {
//       size_t wellIx = row * mask.W() + col;
//       int above = row > rowStart ? (row - rowStart - 1) * width + col - colStart : 0;
//       int behind = col > colStart ? (row - rowStart) * width + col - colStart - 1 : 0;
//       int corner = col > colStart && row > rowStart ?  (row - rowStart - 1) * width + col - colStart - 1 : 0;
//       int patchIx = (row - rowStart) * width + (col - colStart);
//       float current = 0.0f;
//       int current_count = 0;
//       if (IsOk(wellIx, mask)) {
//         current =  metric[wellIx];
//         current_count = 1;
//       }
//       metric_sum[patchIx] = current + metric_sum[above] + metric_sum[behind] - metric_sum[corner]; 
//       metric_count[patchIx] = current_count + metric_count[above] + metric_count[behind] - metric_count[corner]; 
//     }
//   }

//   for (int row = rowStart; row < rowEnd; row++) {
//     for (int col = colStart; col < colEnd; col++) {
//       size_t wellIx = row * mask.W() + col;
//       int lower_corner;
//       int upper_corner;
//       int top_right;
//       int lower_left;
//       int patchIx = (row - rowStart) * width + (col - colStart);
//       float current = 0.0f;
//       int current_count = 0;
//     }
//   }
// }

void BFReference::CalcAvgNeighborBuffering(int rowStart, int rowEnd, int colStart, int colEnd, Mask &mask, int avg_window,
                                           std::vector<float> &metric, std::vector<float> &neighbor_avg) {

  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      int r_avg_start = max(rowStart, row - avg_window);
      int r_avg_end = min(rowEnd - 1, row + avg_window);
      int count = 0;
      double sum = 0.0;
      for (int r = r_avg_start; r <= r_avg_end; r++) {
        int c_avg_start = max(colStart, col - avg_window);
        int c_avg_end = min(colEnd - 1, col + avg_window);
        for (int c = c_avg_start; c <= c_avg_end; c++) {
          size_t wellIx = r * mask.W() + c;
          if (IsOk(wellIx, mask, mWells)) {
            sum += metric[wellIx];
            count++;
          }
        }
      }
      if (count == 0) {
        sum = 0;
      }
      else {
        sum /= count;
      }
      neighbor_avg[row *mask.W() + col] = sum;
    }
  }
}

void BFReference::CalcNeighborDistribution(int rowStart, int rowEnd, int colStart, int colEnd, Mask &mask, int avg_window,
                                           std::vector<float> &metric, std::vector<float> &ksD, std::vector<float> &ksProb,
                                           std::map<std::pair<int,double>,double> &mZDCache) {

  int size = (rowEnd - rowStart) * (colEnd - colStart);
  float desiredSize = 1000;
  int step = max(1, (int)floor(size/desiredSize));
  vector<double> region_sample;
  region_sample.reserve(desiredSize);
  size_t minSize = 10;
  int count = 0;
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      size_t wellIx = row * mask.W() + col;
      if(count++ % step == 0 && IsOk(wellIx, mask, mWells)) {
        region_sample.push_back(metric[wellIx]);
      }
    }
  }
  std::sort(region_sample.begin(),region_sample.end());
  vector<double> local_sample;
  local_sample.reserve((2 * avg_window + 1) * (2 * avg_window + 1));
  
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      double prob = -1.0;
      double D = std::numeric_limits<double>::quiet_NaN();
      size_t wellIx = row * mask.W() + col;
      if( IsOk(wellIx, mask, mWells) ) {
        local_sample.clear();
        int r_avg_start = max(rowStart, row - avg_window);
        int r_avg_end = min(rowEnd - 1, row + avg_window);
        for (int r = r_avg_start; r <= r_avg_end; r++) {
          int c_avg_start = max(colStart, col - avg_window);
          int c_avg_end = min(colEnd - 1, col + avg_window);
          for (int c = c_avg_start; c <= c_avg_end; c++) {
            size_t wellIx = r * mask.W() + c;
            if (IsOk(wellIx, mask, mWells)) {
              local_sample.push_back(metric[wellIx]);
            }
          }
        }
        if (local_sample.size() > minSize && region_sample.size() > minSize) {
          
          std::pair<int,double> m_pair;
          std::sort(local_sample.begin(), local_sample.end());
          D =  ionStats::KolmogorovTest(local_sample.size(), &local_sample[0], 
                                        region_sample.size(), &region_sample[0], 1);
          int N = min(local_sample.size(), region_sample.size());
          //        prob = ionStats::K(N,D);
          if (isfinite(D)) {
            m_pair.first = N;
            m_pair.second = D;
            std::map<std::pair<int,double>,double>::iterator cacheP = mZDCache.find(m_pair);
            if (cacheP != mZDCache.end()) {
              prob =  cacheP->second;
            }
            else {
              // assume that N is the smaller local sample
              prob = ionStats::SmirnovK(N,D);
              mZDCache[m_pair] = prob;
            }
          }
          else {
            prob = -1.0f;
          }
        }
      }
      ksD[row *mask.W() + col] = D;
      ksProb[row *mask.W() + col] = prob;
    }
  }
}


/*
 * wells at later t0 often have less beadfind due to solution buffering and dillution,
 * adjust for this as a linear effect over the region.
 */
void BFReference::AdjustForT0(int rowStart, int rowEnd, int colStart, int colEnd,
			      int nCol, std::vector<float> &t0, std::vector<char> &filters,
			      std::vector<float> &metric) {

  /* Solve using OLS
     Good proof from wikipedia...
     \hat\beta & = \frac{ \sum_{i=1}^{n} (x_{i}-\bar{x})(y_{i}-\bar{y}) }{ \sum_{i=1}^{n} (x_{i}-\bar{x})^2 }
                 = \frac{ \sum_{i=1}^{n}{x_{i}y_{i}} - \frac1n \sum_{i=1}^{n}{x_{i}}\sum_{j=1}^{n}{y_{j}}}{ \sum_{i=1}^{n}({x_{i}^2}) - \frac1n (\sum_{i=1}^{n}{x_{i}})^2 } \\[6pt]
               & = \frac{ \overline{xy} - \bar{x}\bar{y} }{ \overline{x^2} - \bar{x}^2 }   
     
      \hat\alpha & = \bar{y} - \hat\beta\,\bar{x},
  */
  int count = 0;
  double xy = 0, x = 0, y = 0, x2 = 0;
  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      int idx = row * nCol + col;
      if (filters[idx] != Filtered && filters[idx] != Exclude && t0[idx] > 0) {
	count++;
	xy += (metric[idx] * t0[idx]);
	x += (t0[idx]);
	x2 += (t0[idx] * t0[idx]);
	y += metric[idx];
      }
    }
  }
  if (count < MIN_OK_WELLS) {
    return; // no correction
  }

  xy /= count;
  x /= count;
  x2 /= count;
  y /= count;
  if (x == 0 || x2 == 0) {
    return;
  }

  double beta =  (xy - x *y) / (x2 - x*x);
  double alpha = y - (beta * x);

  for (int row = rowStart; row < rowEnd; row++) {
    for (int col = colStart; col < colEnd; col++) {
      int idx = row * nCol + col;
      metric[idx] -= (beta * t0[idx] + alpha);
    }
  }
}

void BFReference::CalcShiftedReference(const std::string &datFile, Mask &mask, std::vector<float> &metric, BufferMeasurement bf_type) {
  Image bfImg;
  bfImg.SetImgLoadImmediate (false);
  bool loaded = LoadImage(bfImg, datFile);
  if (!loaded) { ION_ABORT("*Error* - No beadfind file found, did beadfind run? are files transferred?  (" + datFile + ")"); }
  CalcShiftedReference(bfImg, mask, metric, bf_type);
  bfImg.Close();
}

void BFReference::CalcShiftedReference(Image &bfImg, Mask &mask, std::vector<float> &metric, BufferMeasurement bf_type) {
  // cache to avoid looking up the same values over and over
  std::map<std::pair<int,double>,double> mZDCache;

  const RawImage *raw = bfImg.GetImage();
  metric.resize(raw->rows * raw->cols);  
  mTraceSd.resize(metric.size());
  std::fill(metric.begin(), metric.end(), 0.0f);

  // Sanity checks
  assert(raw->cols == GetNumCol());
  assert(raw->rows == GetNumRow());
  assert(raw->cols == mask.W());
  assert(raw->rows == mask.H());
  if (!mDebugFile.empty()) {
    DebugTraces(mDebugFile, mask, bfImg);
  }

  // // Basic image processing @todo - should we be doing comparator correction?
  // bfImg.FilterForPinned(&mask, MaskEmpty, false);
  // ImageTransformer::XTChannelCorrect(bfImg.raw, bfImg.results_folder);
  // if (ImageTransformer::gain_correction != NULL) {
  //   ImageTransformer::GainCorrectImage(bfImg.raw);
  // }
  ClockTimer timer;
  // Filter for odd wells based on response to wash
  FilterForOutliers(bfImg, mask, mIqrOutlierMult, mRegionYSize, mRegionXSize);
  GridMesh<float> grid;
  grid.Init(raw->rows, raw->cols, mRegionYSize, mRegionXSize);
  int numBin = grid.GetNumBin();
  int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  int num_wells = mRegionXSize * mRegionYSize;
  int num_frames = bfImg.raw->uncompFrames;
  float uncomp[raw->uncompFrames]; // uncompress to 15 frames per second
  float shifted[raw->uncompFrames]; // aligned/shifted to a common t0 
  double region_avg[raw->uncompFrames]; // average of unfiltered wells in region
  vector<SampleStats<double> > region_stats(raw->uncompFrames);
  Mat<float> data(num_frames, num_wells); // frames x wells for a region
  Mat<float> data_trans(num_wells, num_frames); // frames x wells for a region
  Mat<int> sample_data_dbg; // if debugging some sample data filled in here to be output at end
  Mat<float> buffer_metrics_dbg; // if debugging mode save some extra info about each well
  int sd_filtered = 0;
  vector<float> neighbor_avg_metric(metric.size(),0);
  vector<float> ks_D(metric.size(),std::numeric_limits<double>::quiet_NaN());
  std::fill(ks_D.begin(), ks_D.end(), std::numeric_limits<double>::quiet_NaN());
  vector<float> ks_prob(metric.size(),-1);
  std::fill(ks_prob.begin(), ks_prob.end(), -1.0f);
  if (!mDebugH5File.empty()) {
    //    sample_data_dbg.resize(SAMPLE_PER_REGION * numBin, num_frames + 4);
    sample_data_dbg.resize(raw->frameStride, num_frames + 4);
    sample_data_dbg.fill(0);
    buffer_metrics_dbg.resize(raw->rows * raw->cols, 10);
    // Initialize as these might not end up being calculated with sparse regions.
    for(size_t i =0; i < buffer_metrics_dbg.n_rows; i++) {
      buffer_metrics_dbg(i,7) = std::numeric_limits<double>::quiet_NaN();
      buffer_metrics_dbg(i,8) = -1;
    }
  }

  for (int binIx = 0; binIx < numBin; binIx++) {
    // cleanup buffers for this chip.
    std::fill(region_avg, region_avg+raw->uncompFrames, 0.0);
    for (size_t i = 0; i < region_stats.size(); i++) {
      region_stats[i].Clear();
    }

    grid.GetBinCoords(binIx, rowStart, rowEnd, colStart, colEnd);
    int well_count = 0;
    double avg_t0 = 0;
    size_t count_t0 = 0;
    
    // Calculate average t0 for region
    for (int row = rowStart; row < rowEnd; row++) {
      float *__restrict t0_start = &mT0[0] + row * raw->cols + colStart;
      float *__restrict t0_end = t0_start + (colEnd - colStart);
      while (t0_start != t0_end) {
        if (*t0_start > 0) {
          avg_t0 += *t0_start;
          count_t0++;
        }
        t0_start++;
      }
    }

    // skip region if no decent beads
    if (count_t0 == 0) {
      continue;
    }
    avg_t0 /= count_t0; 

    // Uncompress the data, shift to common t0 and copy into our data structure
    for (int row = rowStart; row < rowEnd; row++) {
      for (int col = colStart; col < colEnd; col++) {
        int idx = row * raw->cols + col;
        int data_idx = (row-rowStart) * mRegionXSize + (col-colStart);
        float t0_diff = mT0[idx] > 0 ? mT0[idx] - avg_t0 : 0.0f;
        bfImg.GetUncompressedTrace(uncomp, raw->uncompFrames, col, row);
        TraceHelper::ShiftTrace(uncomp, shifted, raw->uncompFrames, t0_diff);

        double dc_offset = 0;
        int dc_count = 0;
        int last_frame = floor(avg_t0);
        for (int fIx = 0; fIx < last_frame; fIx++) {
          dc_offset += shifted[fIx];
          dc_count++;
        }
        dc_count = max(dc_count,1);
        dc_offset /= dc_count;
        SampleStats<float> sd;
        for (int fIx = 0; fIx < num_frames; fIx++) {
          sd.AddValue(shifted[fIx]);
          shifted[fIx] -= dc_offset;
        }
        mTraceSd[idx] = sd.GetSD();
        copy(shifted, shifted+raw->uncompFrames, data.begin_col(data_idx));
        if (mTraceSd[idx] < MIN_TRACE_SD) {
          mWells[idx] = Filtered;
          sd_filtered++;
        }
      }
    }
   //    data_trans = data.t();
    //    FilterOutlierSignalWells(rowStart, rowEnd, colStart, colEnd, mask.W(), data_trans, mWells);

    for (int row = rowStart; row < rowEnd; row++) {
      for (int col = colStart; col < colEnd; col++) {
        int idx = row * raw->cols + col;
        int data_idx = (row-rowStart) * mRegionXSize + (col-colStart);
        if (!(mask[idx] & (MaskPinned | MaskExclude)) && mWells[idx] != Filtered) {
          for (int i = 0; i < raw->uncompFrames; i++) {
            region_avg[i] += data.at(i,data_idx);
            region_stats[i].AddValue(data.at(i,data_idx));
          }
          well_count++;
        }
      }
    }
    
    // Capture debug information if necessary
    if (sample_data_dbg.n_rows > 0) {
      int region_wells = (rowEnd - rowStart) * (colEnd - colStart);
      int step = 1; //ceil(region_wells/(SAMPLE_PER_REGION*1.0f));
      int count = 0;
      //      int sample_idx = binIx * SAMPLE_PER_REGION;
      for (int row = rowStart; row < rowEnd; row++) {
        for (int col = colStart; col < colEnd; col++) {
          int data_idx = (row-rowStart) * mRegionXSize + (col-colStart);
          int sample_idx = row * raw->cols + col;
          if (count++ % step == 0) {
            sample_data_dbg(sample_idx, 0) = rowStart;
            sample_data_dbg(sample_idx, 1) = colStart;
            sample_data_dbg(sample_idx, 2) = row;
            sample_data_dbg(sample_idx, 3) = col;
            for (int frame = 0; frame < num_frames; frame++) {
              sample_data_dbg(sample_idx,frame+4) = data.at(frame, data_idx);
            }
            //            sample_idx++;
          }
        }
      }
    }
    
    // If no good wells move on to next region
    if (well_count == 0) {
      continue;
    }

    // Calculate average
    for (int i = 0; i < raw->uncompFrames; i++) {
      region_avg[i] /= well_count;
    }
    float max_sd = 0.0f;
    int max_sd_idx = 0;
    float frame_sd[num_frames];
    int end_search_window = ceil(avg_t0) + MAX_SD_SEARCH_WINDOW;
    for (int fIx = 0; fIx < num_frames; fIx++) {
      frame_sd[fIx] = region_stats[fIx].GetSD();
      if (max_sd < frame_sd[fIx] && fIx < end_search_window) {
        max_sd_idx = fIx;
        max_sd = frame_sd[fIx];
      }
    }
        
    // Loop through the data looking for largest difference between well and average in region
    for (int row = rowStart; row < rowEnd; row++) {
      for (int col = colStart; col < colEnd; col++) {
        int idx = row * raw->cols + col;
        int data_idx = (row-rowStart) * mRegionXSize + (col-colStart);
        int start_frame = floor(avg_t0);
        int end_frame = min(start_frame + BF_INTEGRATION_WINDOW, raw->uncompFrames);
        float min_val = data.at(start_frame, data_idx) - region_avg[start_frame];
        float max_val = min_val;
        for (int frame = start_frame; frame < end_frame; frame++) {
          int fIx = frame - start_frame;
          float val = data.at(fIx, data_idx) - region_avg[fIx];
          max_val = max(max_val, val);
          min_val = min(min_val, val);
        }

        // Try an integrated value looking at the frames in traces with most variation
        start_frame = max(0, max_sd_idx - BF_INTEGRATION_WIDTH);
        //        end_frame = min((num_frames - 1), max_sd_idx + BF_INTEGRATION_WIDTH);
        end_frame = min((num_frames), max_sd_idx + BF_INTEGRATION_WIDTH);
        float integrated_val = 0;
        for (int frame = start_frame; frame < end_frame; frame++) {
          integrated_val += data.at(frame, data_idx) - region_avg[frame];
        }

        // Fill In the metric based on that requested
        if (bf_type == BFLegacy) {
          metric[idx] = max_val - abs ( min_val );
        }
        else if(bf_type == BFMaxSd) {
          metric[idx] = data.at(max_sd_idx, data_idx);
        }
        else if(bf_type == BFIntMaxSd) {
          metric[idx] = integrated_val;
        }
        else {
          ION_ABORT("Don't recognize BufferMeasurement");
        }

        // Debugging metrics
        if (buffer_metrics_dbg.n_rows > 0) {
          int c = 0;
          buffer_metrics_dbg(idx,c++) = metric[idx]; // 0 
          buffer_metrics_dbg(idx,c++) = data.at(max_sd_idx, data_idx); // 1
          buffer_metrics_dbg(idx,c++) = max_sd_idx; // 2
          buffer_metrics_dbg(idx,c++) = mWells[idx]; // 3
          buffer_metrics_dbg(idx,c++) = mTraceSd[idx]; //4
          buffer_metrics_dbg(idx,c++) = mT0[idx]; // 5
        }
      }
    }
    AdjustForT0(rowStart, rowEnd, colStart, colEnd, raw->cols, mT0, mWells, metric);
    if (buffer_metrics_dbg.n_rows > 0) {
      CalcAvgNeighborBuffering(rowStart, rowEnd, colStart, colEnd, mask, 2, metric, neighbor_avg_metric);
      CalcNeighborDistribution(rowStart, rowEnd, colStart, colEnd, mask, 3, metric, ks_D, ks_prob, mZDCache);
      int last_col = buffer_metrics_dbg.n_cols - 1;
      for (int row = rowStart; row < rowEnd; row++) {
        for (int col = colStart; col < colEnd; col++) {
          int idx = row * raw->cols + col;
          buffer_metrics_dbg(idx,last_col-3) = neighbor_avg_metric[idx]; // 6
          buffer_metrics_dbg(idx,last_col-2) = metric[idx]; // 7
          buffer_metrics_dbg(idx,last_col-1) = ks_D[idx]; // 8
          buffer_metrics_dbg(idx,last_col) = ks_prob[idx]; // 9
        }
      }
    }
  }
  if (buffer_metrics_dbg.n_rows > 0) {
    size_t diffLoaded = 0;
    size_t excluded = 0;
    size_t why = 0;
    for (size_t i = 0; i < buffer_metrics_dbg.n_rows; i++) {
      double value = buffer_metrics_dbg(i,buffer_metrics_dbg.n_cols-1);
      if ((mask[i] & MaskExclude) != 0) {
        excluded++;
      }
      if (fabs(value) < .001 && value >= 0) {
        diffLoaded++;
      }
      if (fabs(value) < .001 && value >= 0 && ((mask[i] & MaskExclude) != 0)) {
        why++;
      }
    }
    cout << "Got: " << why << " impossible results..." << endl;
    cout << "Diff loaded: " << diffLoaded << " of " << (buffer_metrics_dbg.n_rows-excluded) << " (" << 100.0f * diffLoaded/(buffer_metrics_dbg.n_rows - excluded) << "%)" << endl;
      
  }
  if (!mDebugH5File.empty()) {
    H5Arma::WriteMatrix(mDebugH5File + ":/beadfind/sample_traces", sample_data_dbg, false);
    H5Arma::WriteMatrix(mDebugH5File + ":/beadfind/buffer_metrics", buffer_metrics_dbg, false);
  }

  int filteredCount = 0;
  int excludeCount = 0;
  for (size_t i = 0; i < metric.size(); i++) {
    if ((mask[i] & (MaskPinned | MaskExclude)) > 0) { excludeCount++; }
    else { if (mWells[i] == Filtered) { filteredCount++; }
    }
  }
  char buff[32];
  snprintf(buff, sizeof(buff), "( %.2f %%)", (filteredCount*100.0)/(metric.size() - excludeCount));
  cout << "Filtered: " << filteredCount << " of " << metric.size() - excludeCount << " possible wells. " << buff << endl;
  timer.PrintSeconds(cout, "Beadfind stats took: ");
}

void BFReference::GetNEigenScatter(arma::Mat<float> &YY, arma::Mat<float> &E, int nEigen) {
    try {
      Cov = YY.t() * YY;
      eig_sym(EVal, EVec, Cov);
      E.set_size(YY.n_cols, nEigen);
      // Copy largest N eigen vectors as our basis vectors
      int count = 0;
      for(size_t v = Cov.n_rows - 1; v >= Cov.n_rows - nEigen; v--) {
	std::copy(EVec.begin_col(v), EVec.end_col(v), E.begin_col(count++));
      }
    }
    catch(std::exception &e) {
      const char *w = e.what();
      ION_ABORT(w);
    }
  }

void BFReference::GetEigenProjection(arma::Mat<float> &data, arma::Col<unsigned int> &goodRows, size_t nEigen, arma::Mat<float> &proj) {
    ION_ASSERT(nEigen > 0 && nEigen < data.n_cols, "Must specify reasonable selection of eigen values.");
    ION_ASSERT(goodRows.n_rows > 2, "Must have at least a few good columns.");
    Y = data.rows(goodRows);
    GetNEigenScatter(Y,X, nEigen);
    // Calculate our best projection of data onto eigen vectors, as vectors are already orthonomal don't need to solve, just multiply
    try {
      B = data * X;
      proj = B * X.t();
      ION_ASSERT(proj.n_rows == data.n_rows && proj.n_cols == data.n_cols,"Wrong dimensions.");
      // arma::Mat<float> D;
      // D = abs(data - proj);
      // double val = arma::mean(arma::mean(D,0));
      // std::cout << "Mean abs val diff is: " << val << std::endl;
    }
    catch(std::exception &e) {
      const char *w = e.what();
      ION_ABORT(w);
    }
  }

void BFReference::FilterOutlierSignalWells(int rowStart, int rowEnd, int colStart,  int colEnd, int chipWidth,
                                           arma::Mat<float> &data, std::vector<char> &wells) {

  // Figure out the good wells to use for our projection
  arma::Col<unsigned int> goodRows;
  vector<unsigned int> good;
  size_t goodCount = 0;
  for (int r = rowStart; r < rowEnd; r++) {
    for (int c = colStart; c < colEnd; c++) {
      int idx = r * chipWidth + c;
      int data_idx = (r-rowStart) * (colEnd - colStart) + (c-colStart);
      if (wells[idx] == Unknown) {
        goodCount++;
        good.push_back(data_idx);
      }
    }
  }
  
  goodRows.resize(good.size());
  for (size_t i = 0; i < good.size(); i++) {
    goodRows[i] = good[i];
  }
  
  // Get the smoothed/filtered version of traces
  Mat<float> proj;
  if (goodRows.n_rows < MIN_GOOD_ROWS) { return; }
  GetEigenProjection(data, goodRows, NUM_EIGEN_FILTER, proj);
  // Calculate the mean absolute difference
  Col<float> mad = mean(square(data - proj), 1);
  Col<float> mad_filt = mad.elem(goodRows); // only set threshold on the wells we know are ok
  sort(mad_filt.begin(), mad_filt.end());
  double med = ionStats::quantile_sorted(mad_filt.memptr(), mad_filt.n_rows, .5);
  double q25 = ionStats::quantile_sorted(mad_filt.memptr(), mad_filt.n_rows, .25);
  double q75 = ionStats::quantile_sorted(mad_filt.memptr(), mad_filt.n_rows, .75);
  double threshold = med + 3 * (q75 - q25);
  int filtered_count = 0;
  for (int r = rowStart; r < rowEnd; r++) {
    for (int c = colStart; c < colEnd; c++) {
      int idx = r * chipWidth + c;
      int data_idx = (r-rowStart) * (colEnd - colStart) + (c-colStart);
      if (mad[data_idx] > threshold) {
        wells[idx] = Filtered;
        filtered_count++;
      }
    }
  }
  cout << "For region: " << rowStart << "," << colStart << " filtered: " << filtered_count << " of " << mad.n_rows << " ( " << filtered_count * 100.0 / mad.n_rows << "% ) with threshold "  << threshold << endl;
  data = proj;
}

void BFReference::CalcSignalShiftedReference(const std::string &datFile, const std::string &bgFile, Mask &mask, std::vector<float> &metric, float minTraceSd, int bfIntegrationWindow, int bfIntegrationWidth, BufferMeasurement bf_type) {
  Image img;
  Image bgImg;
  img.SetImgLoadImmediate (false);
  bgImg.SetImgLoadImmediate (false);
  bool loaded = LoadImage(img, datFile);
  loaded &= LoadImage(bgImg, bgFile);
  if (!loaded) { ION_ABORT("*Error* - No beadfind file found, did beadfind run? are files transferred?  (" + datFile + ")"); }
  const RawImage *raw = img.GetImage();
  const RawImage *bgRaw = bgImg.GetImage();
  
  metric.resize(raw->rows * raw->cols);  
  mTraceSd.resize(metric.size());
  size_t chip_wells = raw->rows * raw->cols;
  std::fill(metric.begin(), metric.end(), 0.0f);
  mWells.resize(chip_wells);
  std::fill(mWells.begin(), mWells.end(), Unknown);
  // Sanity checks
  assert(raw->cols == GetNumCol());
  assert(raw->rows == GetNumRow());
  assert(raw->cols == mask.W());
  assert(raw->rows == mask.H());
  if (!mDebugFile.empty()) {
    DebugTraces(mDebugFile, mask, img);
  }

  // Basic image processing @todo - should we be doing comparator correction?
  img.FilterForPinned(&mask, MaskEmpty, false);
  bgImg.FilterForPinned(&mask, MaskEmpty, false);
  if (mDoComparatorCorrect) {
    ComparatorNoiseCorrector cnc;
    if (mIsThumbnail) {
      cnc.CorrectComparatorNoiseThumbnail(img.raw, &mask, THUMBNAIL_STEP, THUMBNAIL_STEP, false);
      cnc.CorrectComparatorNoiseThumbnail(bgImg.raw, &mask, THUMBNAIL_STEP, THUMBNAIL_STEP, false);
    }
    else {
      cnc.CorrectComparatorNoise(img.raw, &mask, false, true);
      cnc.CorrectComparatorNoise(bgImg.raw, &mask, false, true);
    }
  }
  ImageTransformer::XTChannelCorrect(img.raw, img.results_folder);
  ImageTransformer::XTChannelCorrect(bgImg.raw, img.results_folder);
  if (ImageTransformer::gain_correction != NULL) {
    ImageTransformer::GainCorrectImage(img.raw);
    ImageTransformer::GainCorrectImage(bgImg.raw);
  }
  for (size_t i = 0; i < mWells.size(); i++) {
    if (mask[i] & MaskPinned || mask[i] & MaskExclude || mask[i] & MaskIgnore) {
      mWells[i] = Filtered;
    }
  }
  // Filter for odd wells based on response to wash
  //  FilterForOutliers(img, mask, mIqrOutlierMult, mRegionYSize, mRegionXSize);
  GridMesh<float> grid;
  grid.Init(raw->rows, raw->cols, mRegionYSize, mRegionXSize);
  int numBin = grid.GetNumBin();
  int rowStart = -1, rowEnd = -1, colStart = -1, colEnd = -1;
  int num_wells = mRegionXSize * mRegionYSize;
  int num_frames = img.raw->uncompFrames;
  float uncomp[raw->uncompFrames]; // uncompress to 15 frames per second
  float shifted[raw->uncompFrames]; // aligned/shifted to a common t0 
  float bg_uncomp[raw->uncompFrames]; // uncompress to 15 frames per second
  float bg_shifted[raw->uncompFrames]; // aligned/shifted to a common t0 
  double region_avg[raw->uncompFrames]; // average of unfiltered wells in region
  vector<SampleStats<double> > region_stats(raw->uncompFrames);
  Mat<float> data(num_frames, num_wells); // frames x wells for a region
  Mat<float> raw_data(num_frames, num_wells); // frames x wells for a region
  Mat<float> bg_data(num_frames, num_wells); // frames x wells for a region
  Mat<float> data_trans(num_wells, num_frames); // frames x wells for a region
  Mat<int> sample_data_dbg; // if debugging some sample data filled in here to be output at end
  Mat<int> sample_raw_data_dbg; // if debugging some sample data filled in here to be output at end
  Mat<int> sample_bg_data_dbg; // if debugging some sample data filled in here to be output at end
  Mat<float> buffer_metrics_dbg; // if debugging mode save some extra info about each well
  if (!mDebugH5File.empty()) {
    sample_data_dbg.resize(SAMPLE_PER_REGION * numBin, num_frames + 4);
    sample_data_dbg.fill(0);
    sample_bg_data_dbg.resize(SAMPLE_PER_REGION * numBin, num_frames + 4);
    sample_bg_data_dbg.fill(0);
    sample_raw_data_dbg.resize(SAMPLE_PER_REGION * numBin, num_frames + 4);
    sample_raw_data_dbg.fill(0);
    buffer_metrics_dbg.resize(raw->rows * raw->cols, 3);
  }

  for (int binIx = 0; binIx < numBin; binIx++) {
    // cleanup buffers for this chip.
    std::fill(region_avg, region_avg+raw->uncompFrames, 0.0);
    for (size_t i = 0; i < region_stats.size(); i++) {
      region_stats[i].Clear();
    }

    grid.GetBinCoords(binIx, rowStart, rowEnd, colStart, colEnd);
    int well_count = 0;
    double avg_t0 = 0;
    size_t count_t0 = 0;
    
    // Calculate average t0 for region
    for (int row = rowStart; row < rowEnd; row++) {
      for (int col = colStart; col < colEnd; col++) {
        int idx = row * raw->cols + col;
        if (mT0[idx] > 0) {
          avg_t0 += mT0[idx];
          count_t0++;
        }
      }
    }
    // skip region if no decent beads
    if (count_t0 == 0) {
      continue;
    }
    avg_t0 /= count_t0; 

    // Uncompress the data, shift to common t0 and copy into our data structure
    for (int row = rowStart; row < rowEnd; row++) {
      for (int col = colStart; col < colEnd; col++) {
        int idx = row * raw->cols + col;
        int data_idx = (row-rowStart) * mRegionXSize + (col-colStart);
        float t0_diff = mT0[idx] > 0 ? mT0[idx] - avg_t0 : 0.0f;
        img.GetUncompressedTrace(uncomp, raw->uncompFrames, col, row);
        bgImg.GetUncompressedTrace(bg_uncomp, bgRaw->uncompFrames, col, row);
        double dc_offset = 0;
        int dc_count = 0;
        double bg_dc_offset = 0;
        int last_frame = floor(avg_t0);
        for (int fIx = 0; fIx < last_frame; fIx++) {
          dc_offset += uncomp[fIx];
          bg_dc_offset += bg_uncomp[fIx];
          dc_count++;
        }
        dc_count = max(dc_count,1);
        dc_offset /= dc_count;
        bg_dc_offset /= dc_count;

        for (int i = 0; i < raw->uncompFrames; i++) {
          uncomp[i] = uncomp[i] - dc_offset;
          bg_uncomp[i] = bg_uncomp[i] - bg_dc_offset;
        }
        TraceHelper::ShiftTrace(uncomp, shifted, raw->uncompFrames, t0_diff);
        TraceHelper::ShiftTrace(bg_uncomp, bg_shifted, raw->uncompFrames, t0_diff);
        SampleStats<float> sd;
        for (int fIx = 0; fIx < num_frames; fIx++) {
          data.at(fIx, data_idx) = shifted - bg_shifted;
          sd.AddValue(shifted[fIx]);
          raw_data.at(fIx, data_idx) = shifted[fIx];
          bg_data.at(fIx,data_idx) = bg_shifted[fIx];
        }
        mTraceSd[idx] = sd.GetSD();
      }
    }
    data_trans = data.t();
    FilterOutlierSignalWells(rowStart, rowEnd, colStart, colEnd, raw->cols, data_trans, mWells);
    //    data = data_trans.t();

    // Use the smoothed version for stats
    for (int row = rowStart; row < rowEnd; row++) {
      for (int col = colStart; col < colEnd; col++) {
        int idx = row * raw->cols + col;
        int data_idx = (row-rowStart) * mRegionXSize + (col-colStart);
        if (!(mask[idx] & (MaskPinned | MaskExclude)) && mWells[idx] != Filtered && mTraceSd[idx] > minTraceSd) {
          for (int i = 0; i < raw->uncompFrames; i++) {
            region_avg[i] += data(i, data_idx);
            region_stats[i].AddValue(data(i, data_idx));
          }
          well_count++;
        }
      }
    }

    // Capture debug information if necessary
    if (sample_data_dbg.n_rows > 0) {
      int region_wells = (rowEnd - rowStart) * (colEnd - colStart);
      int step = ceil(region_wells/(SAMPLE_PER_REGION*1.0f));
      int count = 0;
      int sample_idx = binIx * SAMPLE_PER_REGION;
      for (int row = rowStart; row < rowEnd; row++) {
        for (int col = colStart; col < colEnd; col++) {
          int data_idx = (row-rowStart) * mRegionXSize + (col-colStart);
          if (count++ % step == 0) {
            sample_data_dbg(sample_idx, 0) = rowStart;
            sample_data_dbg(sample_idx, 1) = colStart;
            sample_data_dbg(sample_idx, 2) = row;
            sample_data_dbg(sample_idx, 3) = col;
            sample_raw_data_dbg(sample_idx, 0) = rowStart;
            sample_raw_data_dbg(sample_idx, 1) = colStart;
            sample_raw_data_dbg(sample_idx, 2) = row;
            sample_raw_data_dbg(sample_idx, 3) = col;
            sample_bg_data_dbg(sample_idx, 0) = rowStart;
            sample_bg_data_dbg(sample_idx, 1) = colStart;
            sample_bg_data_dbg(sample_idx, 2) = row;
            sample_bg_data_dbg(sample_idx, 3) = col;
            for (int frame = 0; frame < num_frames; frame++) {
              sample_data_dbg(sample_idx,frame+4) = data(frame, data_idx);
              sample_raw_data_dbg(sample_idx,frame+4) = raw_data(frame, data_idx);
              sample_bg_data_dbg(sample_idx,frame+4) = bg_data(frame, data_idx);
            }
            sample_idx++;
          }
        }
      }
    }
    
    // If no good wells move on to next region
    if (well_count == 0) {
      continue;
    }

    // Calculate average
    for (int i = 0; i < raw->uncompFrames; i++) {
      region_avg[i] /= well_count;
    }
    float max_sd = 0.0f;
    int max_sd_idx = 0;
    float frame_sd[num_frames];
    for (int fIx = 0; fIx < num_frames; fIx++) {
      frame_sd[fIx] = region_stats[fIx].GetSD();
      if (max_sd < frame_sd[fIx]) {
        max_sd_idx = fIx;
        max_sd = frame_sd[fIx];
      }
    }
    size_t crazy = 0;
    // Loop through the data looking for largest difference between well and average in region
    for (int row = rowStart; row < rowEnd; row++) {
      for (int col = colStart; col < colEnd; col++) {
        int idx = row * raw->cols + col;
        int data_idx = (row-rowStart) * mRegionXSize + (col-colStart);
        int start_frame = floor(avg_t0);
        int end_frame = min(start_frame + bfIntegrationWindow, raw->uncompFrames);
        float min_val = data(start_frame, data_idx);
        float max_val = min_val;
        for (int frame = start_frame; frame < end_frame; frame++) {
          int fIx = frame - start_frame;
          float val = data(fIx, data_idx);
          max_val = max(max_val, val);
          min_val = min(min_val, val);
        }

        // Try an integrated value looking at the frames in traces with most variation
        start_frame = max(0, max_sd_idx - bfIntegrationWidth);
        end_frame = min(num_frames, max_sd_idx + bfIntegrationWidth);
        float integrated_val = 0;
        for (int frame = start_frame; frame < end_frame; frame++) {
          integrated_val += data(frame, data_idx);
        }
        if (integrated_val < -100) {
          crazy++;
        }
        // Fill in the metric based on that requested
        if (bf_type == BFLegacy) {
          metric[idx] = max_val - abs ( min_val );
        }
        else if(bf_type == BFMaxSd) {
          metric[idx] = data(max_sd_idx, data_idx);
        }
        else if(bf_type == BFIntMaxSd) {
          metric[idx] = integrated_val;
        }
        else {
          ION_ABORT("Don't recognize BufferMeasurement");
        }

        // Debugging metrics
        if (buffer_metrics_dbg.n_rows > 0) {
          int c = 0;
          buffer_metrics_dbg(idx,c++) = metric[idx]; // 0 
          buffer_metrics_dbg(idx,c++) = data(max_sd_idx, data_idx); // 1
          buffer_metrics_dbg(idx,c++) = max_sd_idx; // 2
        }
      }
    }
    cout << rowStart << "," << colStart << " crazy " << crazy << " times." << endl;
  }

  if (!mDebugH5File.empty()) {
    H5Arma::WriteMatrix(mDebugH5File + ":/beadfind/sample_traces", sample_data_dbg, false);
    H5Arma::WriteMatrix(mDebugH5File + ":/beadfind/sample_fg_traces", sample_raw_data_dbg, false);
    H5Arma::WriteMatrix(mDebugH5File + ":/beadfind/sample_bg_traces", sample_bg_data_dbg, false);
    H5Arma::WriteMatrix(mDebugH5File + ":/beadfind/buffer_metrics", buffer_metrics_dbg, false);
  }
  
  img.Close();
  bgImg.Close();
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
  ImageTransformer::XTChannelCorrect(bfImg.raw,bfImg.results_folder);
  // bfImg.XTChannelCorrect(&mask);
  Traces trace;  
  trace.Init(&bfImg, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME,LASTDCFRAME);
  bfImg.SetMeanOfFramesToZero(StartFrame, EndFrame);
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
      bfImg.SubtractLocalReferenceTraceInRegion( reg,&mask, MaskAll, MaskEmpty, NNinnerx, NNinnery, NNouterx, NNoutery);
    }
  }
  else {
    bfImg.SubtractLocalReferenceTrace(&mask, MaskEmpty, MaskEmpty, NNinnerx, NNinnery, NNouterx, NNoutery);
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
    if (mask[i] & MaskExclude || mask[i] & MaskPinned) {
      mWells[i] = Exclude;
    }
  }
  //  cout << "Filling reference. " << endl;
  FillInReference(mWells, mBfMetric, mGrid, mMinQuantile, mMaxQuantile);
  for (int i = 0; i < length; i++) {
    if (mWells[i] == Reference) {
      mask[i] |= MaskReference;
    }
  }
}

void BFReference::CalcSignalReference(const std::string &datFile, const std::string &bgFile,
				      Mask &mask, int traceFrame) {
  CalcSignalShiftedReference(datFile, bgFile, mask, mBfMetric, 0, 20,3, BFIntMaxSd);
  // Image bfImg;
  // Image bfBkgImg;
  // bfImg.SetImgLoadImmediate (false);
  // bfBkgImg.SetImgLoadImmediate (false);
  // bool loaded = bfImg.LoadRaw(datFile.c_str());
  // bool bgLoaded = bfBkgImg.LoadRaw(bgFile.c_str());
  // if (!loaded) {
  //   ION_ABORT("*Error* - No beadfind file found, did beadfind run? are files transferred?  (" + datFile + ")");
  // }
  // if (!bgLoaded) {
  //   ION_ABORT("*Error* - No beadfind background file found, did beadfind run? are files transferred?  (" + bgFile + ")");
  // }
  // const RawImage *raw = bfImg.GetImage();
  
  // assert(raw->cols == GetNumCol());
  // assert(raw->rows == GetNumRow());
  // assert(raw->cols == mask.W());
  // assert(raw->rows == mask.H());
  // bfImg.FilterForPinned(&mask, MaskEmpty, false);
  // bfBkgImg.FilterForPinned(&mask, MaskEmpty, false);

  // // bfImg.XTChannelCorrect(&mask);
  // ImageTransformer::XTChannelCorrect(bfImg.raw,bfImg.results_folder);
  // // bfBkgImg.XTChannelCorrect(&mask);
  // //bfBkgImg.XTChannelCorrect();
  // ImageTransformer::XTChannelCorrect(bfBkgImg.raw,bfImg.results_folder);

  // Traces trace;  
  // trace.Init(&bfImg, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME,LASTDCFRAME);
  // bfImg.Close();
  // Traces bgTrace;
  // bgTrace.Init(&bfBkgImg, &mask, FRAMEZERO, FRAMELAST, FIRSTDCFRAME,LASTDCFRAME);
  // bfBkgImg.Close();
  // if (mDoRegionalBgSub) {
  //   trace.SetMeshDist(0);
  //   bgTrace.SetMeshDist(0);
  // }
  // trace.SetT0Step(mRegionXSize);
  // bgTrace.SetT0Step(mRegionXSize);
  // trace.CalcT0(true);
  // size_t numWells = trace.GetNumRow() * trace.GetNumCol();
  // for (size_t i = 0; i < numWells; i++) {
  //   trace.SetT0(max(trace.GetT0(i) - 3, 0.0f), i);
  // }
  // bgTrace.SetT0(trace.GetT0());
  // trace.T0DcOffset(0,4);
  // trace.FillCriticalFrames();
  // trace.CalcReference(mRegionXSize,mRegionYSize,trace.mGridMedian);
  // bgTrace.T0DcOffset(0,4);
  // bgTrace.FillCriticalFrames();
  // bgTrace.CalcReference(mRegionXSize,mRegionYSize,bgTrace.mGridMedian);

  // int length = GetNumRow() * GetNumCol();
  // mBfMetric.resize(length, std::numeric_limits<double>::signaling_NaN());
  // vector<double> rawTrace(trace.GetNumFrames());
  // vector<double> bgRawTrace(bgTrace.GetNumFrames());
  // int pinned =0, excluded = 0;
  // for (int i = 0; i < length; i++) {
  //   if (mask[i] & MaskExclude || mask[i] & MaskPinned) {
  //     continue;
  //     if (mask[i] & MaskExclude) {
  //       excluded++;
  //     }
  //     else if (mask[i] & MaskPinned) {
  //       pinned++;
  //     }
  //   }
  //   trace.GetTraces(i, rawTrace.begin());
  //   bgTrace.GetTraces(i, bgRawTrace.begin());
  //   mBfMetric[i] = 0;
  //   for (int s = 3; s < 15; s++) {
  //     mBfMetric[i] += rawTrace[s] - bgRawTrace[s];
  //   }
  // }
  // cout << "Pinned: " << pinned << " excluded: " << excluded << endl;
  // for (int i = 0; i < length; i++) {
  //   if (mask[i] & MaskExclude || mask[i] & MaskPinned) {
  //     mWells[i] = Exclude;
  //   }
  //   // else {
  //   //   mask[i] = MaskIgnore;
  //   // }
  // }
  // cout << "Filling reference. " << endl;
  // FillInReference(mWells, mBfMetric, mGrid, mMinQuantile, mMaxQuantile);
  // for (int i = 0; i < length; i++) {
  //   if (mWells[i] == Reference) {
  //     mask[i] |= MaskReference;
  //   }
  // }
  // bfImg.Close();
}

void BFReference::FillInRegionRef(int rStart, int rEnd, int cStart, int cEnd,
				  std::vector<float> &metric, 
				  double minQuantile, double maxQuantile,
				  std::vector<char> &wells) {
  std::vector<std::pair<float,int> > wellMetric;
  for (int rIx = rStart; rIx < rEnd; rIx++) {
    for (int cIx = cStart; cIx < cEnd; cIx++) {
      int idx = RowColIdx(rIx,cIx);
      if (wells[idx] != Exclude && isfinite(metric[idx])) {
	wellMetric.push_back(std::pair<float,int>(metric[idx],idx));
      }
    }
  }
  std::sort(wellMetric.rbegin(), wellMetric.rend());
  int minQIx = Round(minQuantile * wellMetric.size());
  int maxQIx = Round(maxQuantile * wellMetric.size());
  for (int qIx = minQIx; qIx < maxQIx; qIx++) {
    wells[wellMetric[qIx].second] = Reference;
  }
}

void BFReference::FillInReference(std::vector<char> &wells, 
				  std::vector<float> &metric,
				  GridMesh<int> &grid,
				  double minQuantile,
				  double maxQuantile) {
  int rStart = -1, rEnd = -1, cStart = -1, cEnd = -1;
  for (size_t bIx = 0; bIx < grid.GetNumBin(); bIx++) {
    grid.GetBinCoords(bIx, rStart, rEnd, cStart, cEnd);
    FillInRegionRef(rStart, rEnd, cStart, cEnd,
		    metric, minQuantile, maxQuantile, wells);
  }
} 
 
